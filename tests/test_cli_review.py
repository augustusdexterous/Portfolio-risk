"""Regression tests for the command-line and reporting defects found in review."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

import portfolio_risk as pr
from portfolio_risk.cli import load_assets, main

ROOT = Path(__file__).resolve().parent.parent


def _write(path, rows, header):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def _clean_rows(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return [[f"{x:.6f}" for x in rng.normal(0, 0.01, 2)] for _ in range(n)]


def test_row_with_extra_fields_is_an_error_not_a_silent_truncation(tmp_path, capsys):
    p = tmp_path / "ragged.csv"
    rows = _clean_rows()
    rows[5] = rows[5] + ["0.999999"]
    _write(p, rows, ["a", "b"])
    assert main([str(p)]) == 2
    assert "data row 6 has 3 fields" in capsys.readouterr().err


def test_text_token_in_asset_column_is_an_error_not_a_silent_drop(tmp_path, capsys):
    p = tmp_path / "na.csv"
    rows = _clean_rows()
    rows[10][1] = "NA"
    _write(p, rows, ["a", "b"])
    assert main([str(p)]) == 2
    assert "missing value at data row 11" in capsys.readouterr().err


def test_nan_text_in_price_column_is_still_recognised_as_missing_not_as_an_asset(tmp_path, capsys):
    p = tmp_path / "px.csv"
    rows = [
        [f"{100 + i}.5", f"{x:.6f}", f"{y:.6f}"]
        for i, (x, y) in enumerate(np.random.default_rng(1).normal(0, 0.01, (60, 2)))
    ]
    rows[3][0] = "nan"
    _write(p, rows, ["px", "a", "b"])
    assert main([str(p)]) == 2
    err = capsys.readouterr().err
    assert "px" in err and "missing value at data row 4" in err


def test_constant_asset_fails_before_any_report_is_printed(tmp_path, capsys):
    p = tmp_path / "cash.csv"
    rows = [[f"{x:.6f}", "0.000000"] for x in np.random.default_rng(2).normal(0, 0.01, 60)]
    _write(p, rows, ["equities", "cash"])
    assert main([str(p)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "'cash' is constant" in captured.err


def test_minus_one_return_is_an_error_not_a_dropped_column(tmp_path, capsys):
    p = tmp_path / "wipeout.csv"
    rows = _clean_rows()
    rows[7][0] = "-1.0"
    _write(p, rows, ["a", "b"])
    assert main([str(p)]) == 2
    assert "-100%" in capsys.readouterr().err


def test_blank_first_line_is_skipped(tmp_path):
    p = tmp_path / "blank_first.csv"
    p.write_text("\n" + "a,b\n" + "\n".join(f"{r[0]},{r[1]}" for r in _clean_rows()) + "\n")
    names, matrix = load_assets(p)
    assert names == ["a", "b"] and matrix.shape == (60, 2)


def test_bad_arguments_are_rejected_before_any_output(tmp_path, capsys):
    p = tmp_path / "ok.csv"
    _write(p, _clean_rows(), ["a", "b"])
    assert main([str(p), "--alpha", "1.5"]) == 2
    assert capsys.readouterr().out == ""
    assert main([str(p), "--shrinkage", "7"]) == 2
    assert capsys.readouterr().out == ""


def test_risk_report_is_hashable_and_its_notes_are_immutable():
    r = np.random.default_rng(3).standard_t(3, 300) * 0.01
    rep = pr.risk_report(r)
    assert isinstance(hash(rep), int)
    assert isinstance(rep.notes, tuple)


def test_tiny_alpha_is_labelled_by_alpha_not_as_100_percent():
    r = np.random.default_rng(4).normal(0, 0.01, 300)
    text = pr.risk_report(r, alpha=1e-8).summary()
    assert "100%" not in text
    assert "alpha=1e-08" in text


def test_importing_main_module_does_not_run_the_cli():
    code = "import portfolio_risk.__main__ as m; print('after import')"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert out.returncode == 0 and "after import" in out.stdout


def test_package_declares_typing_marker():
    assert (ROOT / "portfolio_risk" / "py.typed").exists()


def test_skip_reasons_are_printed_even_when_loading_fails(tmp_path, capsys):
    p = tmp_path / "prices_only.csv"
    rows = [[f"{100 + i}.5", f"{150 + i}.25"] for i in range(40)]
    _write(p, rows, ["px_a", "px_b"])
    assert main([str(p)]) == 2
    err = capsys.readouterr().err
    assert "skipped column: px_a" in err and "skipped column: px_b" in err
    assert "no numeric return columns found" in err
