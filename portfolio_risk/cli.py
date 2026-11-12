"""Command-line report: ``portfolio-risk returns.csv``.

The CSV has a header row and one row per period, oldest first. Each numeric column is one
asset's simple returns as decimals (0.01 = 1%). Columns that contain text (such as dates
or labels) are skipped with a note on stderr, and so are columns that look like prices or
dates (average magnitude above 1, or values of 10 or more). A missing value (blank, NA,
NaN, null, None) in an asset column is an error, and so is a row with more fields than the
header.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np

from .report import risk_report
from .risk import equal_risk_contribution_weights, risk_contributions, shrink_covariance

_MISSING = {"", "na", "n/a", "nan", "null", "none"}


def _as_float(cell: str) -> float | None:
    try:
        return float(cell)
    except ValueError:
        return None


def load_assets(path: str, notes: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Read the asset columns. Skip reasons are appended to ``notes`` when given."""
    skipped = notes if notes is not None else []
    with open(path, newline="", encoding="utf-8") as fh:
        raw = list(csv.reader(fh))
    # Blank lines are allowed only before the header and after the data; in the middle they are
    # a missing row, which is an error.
    start = 0
    while start < len(raw) and not any(cell.strip() for cell in raw[start]):
        start += 1
    end = len(raw)
    while end > start and not any(cell.strip() for cell in raw[end - 1]):
        end -= 1
    rows = raw[start:end]
    for offset, row in enumerate(rows):
        if not any(cell.strip() for cell in row):
            raise ValueError(f"blank line at data row {offset} (a missing row, not a separator)")
    if len(rows) < 2:
        raise ValueError("CSV must have a header row and at least one data row")
    header, body = rows[0], rows[1:]
    for i, row in enumerate(body, start=1):
        if len(row) > len(header):
            raise ValueError(f"data row {i} has {len(row)} fields but the header has {len(header)}")
    names: list[str] = []
    data: list[np.ndarray] = []
    for j, name in enumerate(header):
        cells = [row[j].strip() if j < len(row) else "" for row in body]
        is_missing = [c.lower() in _MISSING for c in cells]
        numeric = [None if miss else _as_float(c) for c, miss in zip(cells, is_missing)]
        if all(is_missing):
            skipped.append(f"{name} (empty)")
            continue
        if any(value is None and not miss for value, miss in zip(numeric, is_missing)):
            skipped.append(f"{name} (non-numeric, e.g. dates or labels)")
            continue
        if any(is_missing):
            first = is_missing.index(True)
            raise ValueError(f"column {name!r} has a missing value at data row {first + 1}")
        values = np.asarray(numeric, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"column {name!r} contains infinite values")
        if np.max(np.abs(values)) >= 10.0 or np.mean(np.abs(values)) > 1.0:
            skipped.append(f"{name} (looks like prices, an index or dates; returns are decimals)")
            continue
        names.append(name)
        data.append(values)
    if not names:
        raise ValueError("no numeric return columns found")
    return names, np.column_stack(data)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="portfolio-risk",
        description="Portfolio risk report from a CSV of asset returns.",
    )
    parser.add_argument("csv", help="CSV of asset returns: header row, one column per asset")
    parser.add_argument("--alpha", type=float, default=0.05, help="tail probability, e.g. 0.05 for 95%% VaR")
    parser.add_argument("--periods-per-year", type=float, default=252.0)
    parser.add_argument(
        "--shrinkage", type=float, default=0.5, help="covariance shrinkage intensity in [0, 1]"
    )
    args = parser.parse_args(argv)
    if not (1e-12 <= args.alpha <= 1 - 1e-12):
        print("error: --alpha must be between 0 and 1", file=sys.stderr)
        return 2
    if not (math.isfinite(args.periods_per_year) and args.periods_per_year > 0):
        print("error: --periods-per-year must be a positive number", file=sys.stderr)
        return 2
    if not (0.0 <= args.shrinkage <= 1.0):
        print("error: --shrinkage must be between 0 and 1", file=sys.stderr)
        return 2
    notes: list[str] = []
    try:
        names, matrix = load_assets(args.csv, notes)
        for note in notes:
            print(f"skipped column: {note}", file=sys.stderr)
        zero = [names[j] for j in range(matrix.shape[1]) if np.ptp(matrix[:, j]) == 0.0]
        if zero:
            raise ValueError(f"asset {zero[0]!r} is constant; a constant return series has no risk to report")
        reports = [
            (name, risk_report(column, alpha=args.alpha, periods_per_year=args.periods_per_year))
            for name, column in zip(names, matrix.T)
        ]
        cov = shrink_covariance(matrix, delta=args.shrinkage) if matrix.shape[1] >= 2 else None
        weights = equal_risk_contribution_weights(cov) if cov is not None else None
        shares = risk_contributions(weights, cov) if cov is not None else None
    except (OSError, ValueError, RuntimeError) as exc:
        for note in notes:  # the reasons for skipped columns still help when loading fails
            print(f"skipped column: {note}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for name, rep in reports:
        print(f"== {name}")
        print(rep.summary())
        print()
    if weights is not None:
        print(f"== equal risk contribution weights (covariance shrinkage {args.shrinkage})")
        for name, w, s in zip(names, weights, shares):
            print(f"{name:<24} weight {w * 100:6.2f}%   risk share {s * 100:6.2f}%")
    return 0
