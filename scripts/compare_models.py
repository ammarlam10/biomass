# -*- coding: utf-8 -*-
"""
compare_models.py -- side-by-side test metrics across UNet, XGBoost, and SegFormer.

Usage
-----
Minimal (any subset of flags is accepted):

    python scripts/compare_models.py \\
        --unet      artifacts/checkpoints/test_metrics.json \\
        --xgb       artifacts/test_metrics_xgboost.json \\
        --segformer artifacts/segformer/checkpoints/test_metrics.json

Additional named runs (repeatable):

    python scripts/compare_models.py \\
        --unet      artifacts/checkpoints/test_metrics.json \\
        --run "SegFormer-B2" artifacts/segformer/checkpoints/test_metrics.json \\
        --run "SegFormer-B3" artifacts/segformer_b3/checkpoints/test_metrics.json

Output format:
    --format table   (default) pretty-printed fixed-width table
    --format csv     CSV suitable for pasting into a spreadsheet
    --format json    JSON list of dicts
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any


# Metrics shown in the comparison table.
# Each entry is (display_name, json_key).
# _orig keys are original-scale (after inverse transform);
# non-_orig keys are in transformed (log1p) space.
_METRICS: list[tuple[str, str]] = [
    # Original scale (most interpretable)
    ("RMSE tree_count  (orig)", "rmse_tree_count_orig"),
    ("MAE  tree_count  (orig)", "mae_tree_count_orig"),
    ("R2   tree_count  (orig)", "r2_tree_count_orig"),
    ("RMSE mean_height (orig)", "rmse_mean_height_orig"),
    ("MAE  mean_height (orig)", "mae_mean_height_orig"),
    ("R2   mean_height (orig)", "r2_mean_height_orig"),
    # Transformed (log1p) space -- used during training / val selection
    ("RMSE tree_count  (log)", "rmse_tree_count"),
    ("MAE  tree_count  (log)", "mae_tree_count"),
    ("R2   tree_count  (log)", "r2_tree_count"),
    ("RMSE mean_height (log)", "rmse_mean_height"),
    ("MAE  mean_height (log)", "mae_mean_height"),
    ("R2   mean_height (log)", "r2_mean_height"),
]

_MISSING = "N/A"


def _load_metrics(path: str) -> dict[str, Any]:
    """Load a test_metrics.json file, handling both flat and nested formats.

    Neural network runs (train.py / evaluate.py) produce a flat dict.
    XGBoost runs (evaluate_xgboost.py) nest metrics under a 'metrics' key.
    """
    data = json.loads(Path(path).read_text())
    if "metrics" in data and isinstance(data["metrics"], dict):
        return data["metrics"]
    return data


def _fmt(value: Any) -> str:
    if value is None:
        return _MISSING
    try:
        return "{:.4f}".format(float(value))
    except (TypeError, ValueError):
        return str(value)


def _build_rows(
    runs: list[tuple[str, dict[str, Any]]]
) -> tuple[list[str], list[dict[str, str]]]:
    """Return (header, rows) where each row is a dict keyed by column name."""
    model_names = [name for name, _ in runs]

    rows: list[dict[str, str]] = []
    for display_name, key in _METRICS:
        row: dict[str, str] = {"Metric": display_name}
        for model_name, metrics in runs:
            row[model_name] = _fmt(metrics.get(key))
        rows.append(row)

    header = ["Metric"] + model_names
    return header, rows


def _print_table(header: list[str], rows: list[dict[str, str]]) -> None:
    col_widths = {h: len(h) for h in header}
    for row in rows:
        for h in header:
            col_widths[h] = max(col_widths[h], len(row.get(h, _MISSING)))

    sep = "+-" + "-+-".join("-" * col_widths[h] for h in header) + "-+"

    def fmt_row(r: dict[str, str]) -> str:
        return "| " + " | ".join(r.get(h, _MISSING).ljust(col_widths[h]) for h in header) + " |"

    print(sep)
    print(fmt_row({h: h for h in header}))
    print(sep)

    for i, row in enumerate(rows):
        if i == 6:
            # Divider between orig-scale and log-space sections
            print(sep)
        print(fmt_row(row))
    print(sep)


def _print_csv(header: list[str], rows: list[dict[str, str]]) -> None:
    writer = csv.DictWriter(sys.stdout, fieldnames=header)
    writer.writeheader()
    writer.writerows(rows)


def _print_json_output(header: list[str], rows: list[dict[str, str]]) -> None:
    print(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare test metrics across model runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--unet", metavar="PATH", help="UNet test_metrics.json path")
    parser.add_argument("--xgb", metavar="PATH", help="XGBoost test_metrics.json path")
    parser.add_argument(
        "--segformer", metavar="PATH", help="SegFormer test_metrics.json path"
    )
    parser.add_argument(
        "--run",
        nargs=2,
        metavar=("NAME", "PATH"),
        action="append",
        default=[],
        help="Add an arbitrary named run (repeatable)",
    )
    parser.add_argument(
        "--format",
        choices=["table", "csv", "json"],
        default="table",
        help="Output format (default: table)",
    )
    args = parser.parse_args()

    runs: list[tuple[str, dict[str, Any]]] = []

    if args.unet:
        runs.append(("UNet+ResNet50", _load_metrics(args.unet)))
    if args.xgb:
        runs.append(("XGBoost", _load_metrics(args.xgb)))
    if args.segformer:
        runs.append(("SegFormer-B2", _load_metrics(args.segformer)))
    for name, path in args.run:
        runs.append((name, _load_metrics(path)))

    if not runs:
        parser.error(
            "Provide at least one of --unet, --xgb, --segformer, or --run NAME PATH"
        )

    header, rows = _build_rows(runs)

    if args.format == "table":
        _print_table(header, rows)
    elif args.format == "csv":
        _print_csv(header, rows)
    else:
        _print_json_output(header, rows)


if __name__ == "__main__":
    main()
