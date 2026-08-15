"""Generate resolution plots for one or more IceMix runs.

The plotting logic lives in `graphnet.utilities.plotting`; this is the
command line around it.

    # every run under outputs/, compared against a named baseline
    python examples/04_training/11_plot_icemix_results.py \
        --base-dir outputs --reference IceMix-Standard

    # a single run, plus its training curve
    python examples/04_training/11_plot_icemix_results.py \
        --results outputs/my-run/predictions/results.csv \
        --metrics-csv outputs/my-run/logs/training/version_0/metrics.csv

    # a token-drop robustness sweep
    python examples/04_training/11_plot_icemix_results.py \
        --token-drop-sweep outputs/token_drop_sweep
"""

import os
from typing import Any, Dict

import pandas as pd

from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.utilities.plotting import (
    DEFAULT_ENERGY_BINS,
    DEFAULT_LOG_ENERGY_RANGE,
    METRICS,
    find_results,
    load_results,
    plot_resolution_vs_energy,
    plot_token_drop_sweep,
    plot_training_curve,
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
    else:
        logger.warning("No results CSVs found; skipping resolution plots.")

    # These two return None when their input is absent, unlike
    # `plot_resolution_vs_energy` above, which always returns a list.
    if args.metrics_csv:
        curve = plot_training_curve(args.metrics_csv, args.output_dir)
        if curve:
            written.append(curve)

    if args.token_drop_sweep:
        for metric in args.metrics:
            sweep = plot_token_drop_sweep(
                args.token_drop_sweep, args.output_dir, metric=metric
            )
            if sweep:
                written.append(sweep)

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
        "--token-drop-sweep",
        default=None,
        help="Directory of `10_evaluate_token_drop_robustness.py` output.",
    )
    parser.add_argument("--energy-bins", type=int, default=DEFAULT_ENERGY_BINS)
    parser.add_argument(
        "--log-energy-range",
        nargs=2,
        type=float,
        default=list(DEFAULT_LOG_ENERGY_RANGE),
    )
    parser.add_argument("--output-dir", default="outputs/plots")

    args, _ = parser.parse_known_args()
    main(args)
