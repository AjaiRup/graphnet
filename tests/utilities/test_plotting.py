"""Tests for the cursor-compatible reconstruction plotting interface."""

import pandas as pd

from graphnet.utilities.plotting import binned_statistics, load_results


def test_load_results_normalizes_cursor_prediction_names(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Accept the position names written by the cursor prediction script."""
    path = tmp_path / "results.csv"
    pd.DataFrame(
        {
            "pos_x_pred": [1.0],
            "pos_y_pred": [2.0],
            "pos_z_pred": [3.0],
            "dir_kappa_pred": [4.0],
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


def test_binned_statistics_matches_cursor_minimum_count() -> None:
    """Suppress bins with fewer than six events, as cursor does."""
    values = pd.Series([1.0] * 5 + [2.0] * 6)
    energy = pd.Series([10.0] * 5 + [100.0] * 6)

    statistics = binned_statistics(values, energy)

    assert statistics.loc[statistics["count"] == 5, "p50"].isna().all()
    assert statistics.loc[statistics["count"] == 6, "p50"].iloc[0] == 2.0
