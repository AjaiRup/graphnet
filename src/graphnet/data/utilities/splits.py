"""Utilities for partitioning datasets into train/validation/test splits."""

from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from .sqlite_utilities import query_database


def seeded_event_splits(
    paths: Sequence[str],
    seed: int = 42,
    ratio: Sequence[float] = (0.8, 0.1, 0.1),
    index_column: str = "event_no",
    truth_table: str = "truth",
) -> Dict[str, List[List[int]]]:
    """Split each database in `paths` into train/validation/test selections.

    Args:
        paths: Paths to SQLite databases.
        seed: Random seed of the split.
        ratio: Train, validation and test fractions. Must sum to one.
        index_column: Name of the event index column.
        truth_table: Name of the table with one row per event.

    Returns:
        For each split, one list of event indices per database.
    """
    if len(ratio) != 3 or not np.isclose(sum(ratio), 1.0):
        raise ValueError(
            f"`ratio` must be three fractions summing to 1: {ratio}"
        )
    train, validation, test = ratio
    splits: Dict[str, List[List[int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    query = f"SELECT {index_column} FROM {truth_table} ORDER BY {index_column}"
    for path in paths:
        events = query_database(path, query)[index_column].tolist()
        rest, test_events = train_test_split(
            events, test_size=test, random_state=seed
        )
        train_events, validation_events = train_test_split(
            rest,
            test_size=validation / (train + validation),
            random_state=seed,
        )
        splits["train"].append(train_events)
        splits["validation"].append(validation_events)
        splits["test"].append(test_events)
    return splits
