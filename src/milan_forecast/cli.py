"""Command line entry points; stages can be rerun independently."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import prepare
from .experiments import MAX_EPOCHS, run_all
from .figures import plot_eda, plot_forecasts


def main():
    parser = argparse.ArgumentParser(description="Milan Internet traffic forecasting study")
    parser.add_argument("stage", choices=["prepare", "eda", "train", "plots", "all"])
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--results", type=Path, default=Path("results/results"))
    parser.add_argument("--figures", type=Path, default=Path("figures/figures"))
    parser.add_argument("--models", type=Path, default=Path("models/models"))
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS,
                        help="neural epoch cap; 8 reproduces tuning iteration 1")
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    if args.max_epochs < 1:
        parser.error("--max-epochs must be positive")
    if args.stage in ["prepare", "all"]:
        print(json.dumps(prepare(args.raw, args.processed, args.chunk_size), indent=2))
    if args.stage in ["eda", "all"]:
        print(json.dumps(plot_eda(args.processed, args.figures, args.results), indent=2))
    if args.stage in ["train", "all"]:
        print(json.dumps(run_all(args.processed, args.results, args.models, args.max_epochs), indent=2))
    if args.stage in ["plots", "all"]:
        plot_forecasts(args.processed, args.results, args.figures)


if __name__ == "__main__":
    main()
