"""Command-line entry point for Stage 2 sweeps and sanity checks."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from uhbm_simulator.config import load_config
from uhbm_simulator.sanity import run_sanity_checks
from uhbm_simulator.sweep import run_sweep, write_csv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="uHBM Stage 2 simulator")
    parser.add_argument("--config", type=Path, help="chapter 9 YAML configuration")
    parser.add_argument(
        "--output", type=Path, default=Path("stage2_results.csv"), help="CSV output"
    )
    parser.add_argument(
        "--sanity", action="store_true", help="run chapter 10 validation checks"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.sanity:
        report = run_sanity_checks()
        for name, passed in report.checks.items():
            print(f"{name}: {'PASS' if passed else 'FAIL'}")
        for name, value in report.details.items():
            print(f"{name}: {value:.12g}")
        return 0 if all(report.checks.values()) else 1

    if args.config is None:
        parser.error("--config is required unless --sanity is used")
    rows = run_sweep(load_config(args.config))
    write_csv(rows, args.output)
    print(f"Simulated {len(rows)} design points; wrote {args.output}")
    return 0
