"""Tests for the reconstruction plotting interface."""

import pandas as pd

from graphnet.utilities.plotting import (
    binned_statistics,
    filter_by_flavours,
    load_results,
    resolution_statistics,
)


def test_load_results_preserves_prediction_names(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Read canonical prediction columns."""
    path = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "position_x_pred": [1.0],
            "position_y_pred": [2.0],
            "position_z_pred": [3.0],
            "direction_kappa": [4.0],
            "energy": [10.0],
            "pid": [14],
            "interaction_type": [1],
        }
    ).to_csv(path, index=False)

    results = load_results(str(path), required=("energy",))

    assert results["position_x_pred"].iloc[0] == 1.0
    assert results["position_y_pred"].iloc[0] == 2.0
    assert results["position_z_pred"].iloc[0] == 3.0
    assert results["direction_kappa"].iloc[0] == 4.0


def test_filter_by_flavours_matches_absolute_pid() -> None:
    """Keep both neutrinos and antineutrinos of the requested flavours."""
    results = pd.DataFrame({"pid": [12, -12, 14, -14, 16, -16]})

    kept = filter_by_flavours(results, [12, 14])

    assert list(kept["pid"]) == [12, -12, 14, -14]
    assert filter_by_flavours(results, [16])["pid"].abs().eq(16).all()
    # A run that never contained a flavour is unaffected by removing it.
    no_tau = results[results["pid"].abs() != 16]
    pd.testing.assert_frame_equal(filter_by_flavours(no_tau, [12, 14]), no_tau)


def test_binned_statistics_requires_minimum_count() -> None:
    """Suppress bins with fewer than six events."""
    values = pd.Series([1.0] * 5 + [2.0] * 6)
    energy = pd.Series([10.0] * 5 + [100.0] * 6)

    statistics = binned_statistics(values, energy)

    assert statistics.loc[statistics["count"] == 5, "p50"].isna().all()
    assert statistics.loc[statistics["count"] == 6, "p50"].iloc[0] == 2.0


def test_resolution_statistics_records_counts_and_ratios() -> None:
    """Expose the per-class event mix behind an all-event curve."""
    events = 6
    results = pd.DataFrame(
        {
            "azimuth": [0.0] * events,
            "zenith": [0.0] * events,
            "dir_x_pred": [1.0] * events,
            "dir_y_pred": [0.0] * events,
            "dir_z_pred": [0.0] * events,
            "energy": [10.0] * events,
            "pid": [14] * events,
            "interaction_type": [1] * events,
        }
    )
    statistics = resolution_statistics(
        {"candidate": results, "reference": results},
        metric="angular_resolution",
        reference="reference",
    )

    candidate = statistics[
        (statistics["run"] == "candidate")
        & (statistics["event_class"] == "tracks")
        & (statistics["count"] == events)
    ].iloc[0]
    assert candidate["event_fraction"] == 1.0
    assert candidate["ratio_to_reference"] == 1.0


def test_basic_plots_from_fixture(tmp_path):
    from graphnet.utilities.plotting import (
        plot_resolution_vs_energy,
        plot_training_curve,
    )

    metrics = tmp_path / "metrics.csv"
    pd.DataFrame(
        {"epoch": [0, 1], "train_loss": [3.0, 2.0], "val_loss": [3.2, 2.2]}
    ).to_csv(metrics, index=False)
    assert plot_training_curve(str(metrics), str(tmp_path))
    results = pd.DataFrame(
        {
            "azimuth": [0.0] * 6,
            "zenith": [0.0] * 6,
            "dir_x_pred": [1.0] * 6,
            "dir_y_pred": [0.0] * 6,
            "dir_z_pred": [0.0] * 6,
            "energy": [10.0] * 6,
            "pid": [14] * 6,
            "interaction_type": [1] * 6,
        }
    )
    assert plot_resolution_vs_energy(
        {"IceMix": results}, "angular_resolution", str(tmp_path)
    )
