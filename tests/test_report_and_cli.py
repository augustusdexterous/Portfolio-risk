import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import portfolio_risk as pr
from portfolio_risk.cli import load_assets, main

REPO = Path(__file__).resolve().parent.parent


def test_risk_report_fields_are_consistent():
    r = np.random.default_rng(9).normal(0.0004, 0.01, 1000)
    rep = pr.risk_report(r, alpha=0.05, periods_per_year=252)
    assert rep.n_obs == 1000
    assert rep.annual_volatility == pytest.approx(r.std(ddof=1) * np.sqrt(252))
    assert rep.historical_es >= rep.historical_var
    text = rep.summary()
    assert "VaR 95% historical" in text and "max drawdown" in text


def test_risk_report_marks_undefined_ratios_nan():
    rep = pr.risk_report(np.full(40, 0.001))
    assert np.isnan(rep.sortino) and np.isnan(rep.calmar)
    assert "n/a" in rep.summary()


def _write(path, rows, header):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def test_cli_skips_prices_and_dates(tmp_path, capsys):
    rng = np.random.default_rng(10)
    p = tmp_path / "a.csv"
    rows = [
        [f"2020-01-{i % 28 + 1:02d}", f"{100 + i:.2f}"] + [f"{x:.6f}" for x in rng.normal(0, 0.01, 2)]
        for i in range(60)
    ]
    _write(p, rows, ["date", "price", "a", "b"])
    skipped = []
    names, matrix = load_assets(p, skipped)
    assert names == ["a", "b"]
    assert matrix.shape == (60, 2)
    assert any("price" in s for s in skipped)
    assert main([str(p)]) == 0
    out = capsys.readouterr().out
    assert "== a" in out and "equal risk contribution weights" in out


def test_cli_missing_value_is_error(tmp_path, capsys):
    p = tmp_path / "b.csv"
    _write(p, [["0.01", "0.02"], ["", "0.01"], ["0.00", "0.03"]], ["a", "b"])
    assert main([str(p)]) == 2
    assert "missing value at data row 2" in capsys.readouterr().err


def test_module_entry_point_runs(tmp_path):
    p = tmp_path / "c.csv"
    rng = np.random.default_rng(11)
    rows = [[f"{x:.6f}" for x in rng.normal(0, 0.01, 3)] for _ in range(120)]
    _write(p, rows, ["x", "y", "z"])
    out = subprocess.run(
        [sys.executable, "-m", "portfolio_risk", str(p)], capture_output=True, text=True, check=True
    )
    assert "== z" in out.stdout
