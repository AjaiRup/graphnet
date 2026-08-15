"""Resolution plots for joint direction and vertex reconstruction."""

import glob
import os
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from graphnet.utilities.logging import Logger

MUON_NEUTRINO_PID = 14
CHARGED_CURRENT = 1

DEFAULT_ENERGY_BINS = 20
DEFAULT_LOG_ENERGY_RANGE = (1.0, 4.0)
DEFAULT_PERCENTILES = (16, 50, 84)
MIN_EVENTS_PER_BIN = 6

EVENT_CLASSES = ("all", "tracks", "cascades")

_RUN_NAME_PATTERN = re.compile(
    r"^(?P<project>.*?)_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
    r"(?:_job-(?P<job_id>\d+))?$"
)

_CURSOR_PREDICTION_COLUMNS = {
    "pos_x_pred": "position_x_pred",
    "pos_y_pred": "position_y_pred",
    "pos_z_pred": "position_z_pred",
    "dir_kappa_pred": "direction_kappa",
}


# Physics quantities
def angular_difference(results: pd.DataFrame) -> pd.Series:
    """Return the opening angle in degrees between truth and prediction.

    Uses the haversine form, which stays well conditioned for the small
    separations that dominate a well-trained model's distribution.
    """
    true_azimuth = results["azimuth"].to_numpy()
    true_zenith = results["zenith"].to_numpy()
    pred_x = results["dir_x_pred"].to_numpy()
    pred_y = results["dir_y_pred"].to_numpy()
    pred_z = results["dir_z_pred"].to_numpy()

    pred_azimuth = np.arctan2(pred_y, pred_x)
    pred_zenith = np.arctan2(np.hypot(pred_x, pred_y), pred_z)

    delta_azimuth = pred_azimuth - true_azimuth
    delta_zenith = pred_zenith - true_zenith
    haversine = (
        np.sin(delta_zenith / 2.0) ** 2
        + np.sin(true_zenith)
        * np.sin(pred_zenith)
        * np.sin(delta_azimuth / 2.0) ** 2
    )
    separation = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return pd.Series(np.degrees(separation), index=results.index)


def vertex_distance(results: pd.DataFrame) -> pd.Series:
    """Return the distance in metres between true and predicted vertex."""
    squared = sum(
        (results[f"position_{axis}_pred"] - results[f"position_{axis}"]) ** 2
        for axis in ("x", "y", "z")
    )
    return np.sqrt(squared)


METRICS: Dict[str, Tuple[Callable[[pd.DataFrame], pd.Series], str]] = {
    "angular_resolution": (angular_difference, "Angular error [deg]"),
    "vertex_resolution": (vertex_distance, "Vertex error [m]"),
}


# Selection
def filter_by_class(results: pd.DataFrame, event_class: str) -> pd.DataFrame:
    """Return the subset of `results` belonging to `event_class`.

    `tracks` are charged-current muon-neutrino interactions, which leave an
    extended track; everything else is treated as a cascade. The two have very
    different resolution, so a combined curve hides most of what matters.
    """
    if event_class == "all":
        return results
    is_track = (np.abs(results["pid"]) == MUON_NEUTRINO_PID) & (
        results["interaction_type"] == CHARGED_CURRENT
    )
    if event_class == "tracks":
        return results[is_track]
    if event_class == "cascades":
        return results[~is_track]
    raise ValueError(
        f"Unknown event class '{event_class}'. "
        f"Expected one of {EVENT_CLASSES}."
    )


def binned_statistics(
    values: pd.Series,
    energy: pd.Series,
    n_bins: int = DEFAULT_ENERGY_BINS,
    log_energy_range: Tuple[float, float] = DEFAULT_LOG_ENERGY_RANGE,
    percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    min_events: int = MIN_EVENTS_PER_BIN,
) -> pd.DataFrame:
    """Return fixed-bin percentiles, requiring ``min_events`` per bin."""
    log_energy = np.log10(np.asarray(energy, dtype=float))
    edges = np.linspace(*log_energy_range, n_bins + 1)
    indices = np.digitize(log_energy, edges) - 1

    rows = []
    for index in range(n_bins):
        selected = np.asarray(values)[indices == index]
        row = {
            "log_energy": 0.5 * (edges[index] + edges[index + 1]),
            "count": selected.size,
        }
        if selected.size >= min_events:
            row.update(
                {
                    f"p{int(percentile)}": float(
                        np.percentile(selected, percentile)
                    )
                    for percentile in percentiles
                }
            )
        else:
            row.update({f"p{int(percentile)}": np.nan for percentile in percentiles})
        rows.append(row)
    return pd.DataFrame(rows)


# Run discovery
def find_results(base_dir: str) -> Dict[str, str]:
    """Return the newest `results.csv` per project under `base_dir`.

    Run directories are named `<project>_<date>_<time>[_job-<id>]`;
    where a project has several runs, the one with the highest job id
    wins.
    """
    logger = Logger()
    newest: Dict[str, Tuple[int, str]] = {}

    pattern = os.path.join(base_dir, "**", "predictions", "results.csv")
    for path in glob.glob(pattern, recursive=True):
        run_dir = os.path.dirname(os.path.dirname(path))
        match = _RUN_NAME_PATTERN.match(os.path.basename(run_dir))
        if match is None:
            logger.debug(f"Skipping unrecognised run directory: {run_dir}")
            continue
        project = match.group("project")
        job_id = int(match.group("job_id") or 0)
        if project not in newest or job_id > newest[project][0]:
            newest[project] = (job_id, path)

    return {project: path for project, (_, path) in sorted(newest.items())}


def load_results(path: str, required: Sequence[str] = ()) -> pd.DataFrame:
    """Load current or cursor-format results, checking required columns."""
    results = pd.read_csv(path)
    rename = {
        source: target
        for source, target in _CURSOR_PREDICTION_COLUMNS.items()
        if source in results and target not in results
    }
    if rename:
        results = results.rename(columns=rename)
    missing = [column for column in required if column not in results.columns]
    if missing:
        raise KeyError(f"{path} is missing column(s) {missing}")
    return results


# Plots
def plot_resolution_vs_energy(
    named_results: Dict[str, pd.DataFrame],
    metric: str,
    output_dir: str,
    event_classes: Sequence[str] = EVENT_CLASSES,
    reference: Optional[str] = None,
    n_bins: int = DEFAULT_ENERGY_BINS,
    log_energy_range: Tuple[float, float] = DEFAULT_LOG_ENERGY_RANGE,
) -> List[str]:
    """Plot `metric` against energy, one figure per event class.

    When `reference` names one of `named_results`, a lower panel shows each
    curve as a ratio to it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_fn, ylabel = METRICS[metric]
    os.makedirs(output_dir, exist_ok=True)
    written = []

    def statistics_for(
        results: pd.DataFrame, event_class: str
    ) -> pd.DataFrame:
        """Return the binned statistics of one run and event class."""
        subset = filter_by_class(results, event_class)
        if subset.empty:
            return pd.DataFrame()
        return binned_statistics(
            metric_fn(subset),
            subset["energy"],
            n_bins=n_bins,
            log_energy_range=log_energy_range,
        )

    for event_class in event_classes:
        if reference is not None:
            figure, (axis, ratio_axis) = plt.subplots(
                2,
                1,
                sharex=True,
                figsize=(7, 7),
                gridspec_kw={"height_ratios": [3, 1]},
            )
        else:
            figure, axis = plt.subplots(figsize=(7, 5))
            ratio_axis = None

        # Resolved up front so the ratio panel does not depend on the order
        # `named_results` happens to iterate in.
        baseline: Optional[pd.DataFrame] = None
        if reference is not None and reference in named_results:
            candidate = statistics_for(named_results[reference], event_class)
            baseline = None if candidate.empty else candidate

        for label, results in named_results.items():
            statistics = statistics_for(results, event_class)
            if statistics.empty:
                continue

            line = axis.plot(
                statistics["log_energy"], statistics["p50"], label=label
            )[0]
            axis.fill_between(
                statistics["log_energy"],
                statistics["p16"],
                statistics["p84"],
                alpha=0.15,
                color=line.get_color(),
            )
            if (
                ratio_axis is not None
                and baseline is not None
                and label != reference
            ):
                merged = statistics.merge(
                    baseline, on="log_energy", suffixes=("", "_reference")
                )
                ratio_axis.plot(
                    merged["log_energy"],
                    merged["p50"] / merged["p50_reference"],
                    color=line.get_color(),
                )

        axis.set_ylabel(ylabel)
        axis.set_title(f"{metric.replace('_', ' ').title()} - {event_class}")
        axis.legend()
        axis.grid(alpha=0.3)
        axis.set_yscale("log")
        if ratio_axis is not None:
            ratio_axis.axhline(1.0, color="black", linewidth=0.8)
            ratio_axis.set_ylabel(f"/ {reference}")
            ratio_axis.grid(alpha=0.3)
            ratio_axis.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")
        else:
            axis.set_xlabel(r"$\log_{10}(E\,/\,\mathrm{GeV})$")

        path = os.path.join(output_dir, f"{metric}_{event_class}.png")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        written.append(path)

    return written


def plot_training_curve(metrics_csv: str, output_dir: str) -> Optional[str]:
    """Plot train and validation loss from a `CSVLogger` metrics file."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(metrics_csv):
        Logger().warning(f"No metrics file at {metrics_csv}")
        return None

    metrics = pd.read_csv(metrics_csv)
    figure, axis = plt.subplots(figsize=(7, 5))
    for column, label in (("train_loss", "train"), ("val_loss", "validation")):
        if column not in metrics.columns:
            continue
        series = metrics[["epoch", column]].dropna()
        if series.empty:
            continue
        axis.plot(series["epoch"], series[column], marker="o", label=label)

    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.legend()
    axis.grid(alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "loss.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_token_drop_sweep(
    sweep_dir: str, output_dir: str, metric: str = "angular_resolution"
) -> Optional[str]:
    """Plot median resolution against the inference-time drop fraction.

    Consumes the CSVs written by `10_evaluate_token_drop_robustness.py`.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = sorted(glob.glob(os.path.join(sweep_dir, "token_drop_*.csv")))
    if not paths:
        Logger().warning(f"No sweep CSVs in {sweep_dir}")
        return None

    metric_fn, ylabel = METRICS[metric]
    figure, axis = plt.subplots(figsize=(7, 5))

    for event_class in EVENT_CLASSES:
        fractions, medians = [], []
        for path in paths:
            results = pd.read_csv(path)
            subset = filter_by_class(results, event_class)
            if subset.empty:
                continue
            fractions.append(float(results["token_drop"].iloc[0]))
            medians.append(float(np.median(metric_fn(subset))))
        if fractions:
            axis.plot(fractions, medians, marker="o", label=event_class)

    axis.set_xlabel("Fraction of pulses dropped")
    axis.set_ylabel(f"Median {ylabel.lower()}")
    axis.set_title("Robustness to missing pulses")
    axis.legend()
    axis.grid(alpha=0.3)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"token_drop_{metric}.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
