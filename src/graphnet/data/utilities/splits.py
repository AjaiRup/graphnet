"""Utilities for partitioning datasets into train/validation/test splits."""

import json
import os
import random
import sqlite3
from typing import Dict, List, Optional

from graphnet.utilities.logging import Logger


def seeded_event_splits(
    paths: List[str],
    seed: int = 42,
    ratio: Optional[List[float]] = None,
    index_column: str = "event_no",
    truth_table: str = "truth",
    cache_dir: Optional[str] = None,
) -> Dict[str, List[List[int]]]:
    """Split each database in `paths` into train/validation/test selections."""
    if ratio is None:
        ratio = [0.8, 0.1, 0.1]
    if len(ratio) != 3:
        raise ValueError(f"`ratio` must have three entries. Got {ratio}.")
    if any(fraction < 0 for fraction in ratio) or sum(ratio) > 1.0 + 1e-9:
        raise ValueError(
            f"`ratio` entries must be non-negative and sum to at most 1. "
            f"Got {ratio}."
        )

    logger = Logger()
    splits: Dict[str, List[List[int]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)

    for path in paths:
        key = (
            f"{os.path.basename(path)}-seed{seed}-"
            f"ratio{'_'.join(str(fraction) for fraction in ratio)}"
        )
        cache_path = (
            os.path.join(cache_dir, f"{key}.json")
            if cache_dir is not None
            else None
        )

        if cache_path is not None and os.path.exists(cache_path):
            with open(cache_path, "r") as handle:
                cached = json.load(handle)
            for name in splits:
                splits[name].append(cached[name])
            logger.info(f"Loaded cached split for {path} from {cache_path}")
            continue

        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            events = [
                row[0]
                for row in connection.execute(
                    f"SELECT {index_column} FROM {truth_table}"
                )
            ]

        random.Random(seed).shuffle(events)
        n_train = int(len(events) * ratio[0])
        n_validation = int(len(events) * ratio[1])
        resolved = {
            "train": events[:n_train],
            "validation": events[n_train : n_train + n_validation],
            "test": events[n_train + n_validation :],
        }
        logger.info(
            f"Split {os.path.basename(path)} ({len(events)} events) into "
            f"{len(resolved['train'])} train / "
            f"{len(resolved['validation'])} validation / "
            f"{len(resolved['test'])} test."
        )

        if cache_path is not None:
            with open(cache_path, "w") as handle:
                json.dump(resolved, handle)

        for name in splits:
            splits[name].append(resolved[name])

    return splits
