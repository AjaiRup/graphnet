"""Plot angular/vertex resolution and training curves for IceMix runs."""

import os
from typing import Any, Dict

import pandas as pd

from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.utilities.plotting import (
    DEFAULT_ENERGY_BINS,
    DEFAULT_LOG_ENERGY_RANGE,
    METRICS,
    filter_by_flavours,
    find_results,
    load_results,
    plot_resolution_vs_energy,
    plot_training_curve,
    resolution_statistics,
)

REQUIRED_COLUMNS = ("energy", "pid", "interaction_type")


def main(args: Any) -> None:
    """Write every requested plot into `--output-dir`."""
    logger = Logger()
    os.makedirs(args.output_dir, exist_ok=True)
    written = []

    named_results: Dict[str, pd.DataFrame] = {}
    if args.base_dir:
        for project, path in find_results(args.base_dir).items():
            logger.info(f"Found {project}: {path}")
            named_results[project] = load_results(path, REQUIRED_COLUMNS)
    for path in args.results:
        label = os.path.basename(os.path.dirname(os.path.dirname(path)))
        named_results[label or path] = load_results(path, REQUIRED_COLUMNS)
    if args.reference_csv:
        named_results[args.reference_label] = load_results(
            args.reference_csv, REQUIRED_COLUMNS
        )

    if args.flavours:
        for label in list(named_results):
            before = len(named_results[label])
            named_results[label] = filter_by_flavours(
                named_results[label], args.flavours
            )
            logger.info(
                f"{label}: kept {len(named_results[label])}/{before} events "
                f"with |pid| in {sorted(args.flavours)}"
            )
            if named_results[label].empty:
                logger.warning(
                    f"{label} has no events left after the flavour filter; "
                    "its curves and ratios will be empty."
                )

    if named_results:
        if args.reference and args.reference not in named_results:
            raise KeyError(
                f"Reference '{args.reference}' not among the loaded runs: "
                f"{sorted(named_results)}"
            )
        for metric in args.metrics:
            written += plot_resolution_vs_energy(
                named_results,
                metric=metric,
                output_dir=args.output_dir,
                reference=args.reference,
                n_bins=args.energy_bins,
                log_energy_range=tuple(args.log_energy_range),
            )
        if args.statistics_csv:
            tables = [
                resolution_statistics(
                    named_results,
                    metric=metric,
                    reference=args.reference,
                    n_bins=args.energy_bins,
                    log_energy_range=tuple(args.log_energy_range),
                )
                for metric in args.metrics
            ]
            statistics = pd.concat(
                [table for table in tables if not table.empty],
                ignore_index=True,
            )
            statistics.to_csv(args.statistics_csv, index=False)
            logger.info(f"Wrote {args.statistics_csv}")
    else:
        logger.warning("No results CSVs found; skipping resolution plots.")

    if args.metrics_csv:
        curve = plot_training_curve(args.metrics_csv, args.output_dir)
        if curve:
            written.append(curve)

    for path in written:
        logger.info(f"Wrote {path}")
    if not written:
        logger.warning("Nothing to plot. Check the input paths.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot IceMix reconstruction results.")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Scan for `*/predictions/results.csv` beneath this directory.",
    )
    parser.add_argument(
        "--results",
        nargs="*",
        default=[],
        help="Explicit results.csv paths to include.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Label of the run to draw ratio panels against.",
    )
    parser.add_argument(
        "--reference-csv",
        default=None,
        help="External baseline predictions to include as a reference curve.",
    )
    parser.add_argument("--reference-label", default="baseline")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(METRICS),
        choices=list(METRICS),
    )
    parser.add_argument(
        "--metrics-csv",
        default=None,
        help="`CSVLogger` metrics.csv to plot the training curve from.",
    )
    parser.add_argument(
        "--statistics-csv",
        default=None,
        help="Write binned metrics and reference ratios to CSV.",
    )
    parser.add_argument(
        "--flavours",
        nargs="+",
        type=int,
        default=None,
        metavar="PID",
        help="Keep selected absolute neutrino PIDs (12, 14, 16) in every run.",
    )
    parser.add_argument("--energy-bins", type=int, default=DEFAULT_ENERGY_BINS)
    parser.add_argument(
        "--log-energy-range",
        nargs=2,
        type=float,
        default=list(DEFAULT_LOG_ENERGY_RANGE),
    )
    parser.add_argument("--output-dir", default="outputs/plots")

    args = parser.parse_args()
    main(args)
