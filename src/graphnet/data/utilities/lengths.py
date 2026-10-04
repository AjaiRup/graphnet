"""Utilities for reading and caching the number of pulses per event."""

import os
import sqlite3
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

import numpy as np

from graphnet.utilities.logging import Logger

# `(event_no, n_pulses)`, both sorted ascending by `event_no`.
PulseCountIndex = Tuple[np.ndarray, np.ndarray]


def _cache_path(cache_dir: str, path: str, pulsemap: str) -> str:
    """Return the cache file for `path`'s counts over `pulsemap`."""
    stem = os.path.basename(path)
    if stem.endswith(".db"):
        stem = stem[: -len(".db")]
    return os.path.join(cache_dir, f"lengths_{stem}.{pulsemap}.npz")


# The cached arrays are never mutated by `event_lengths`, so they are safe
# to hand out repeatedly.
@lru_cache(maxsize=None)
def pulse_counts(
    path: str,
    pulsemap: str = "SRTInIcePulses",
    index_column: str = "event_no",
    cache_dir: Optional[str] = None,
) -> PulseCountIndex:
    """Return the number of pulses of every event in `path`."""
    logger = Logger()

    if cache_dir is not None:
        cache_path = _cache_path(cache_dir, path, pulsemap)
        if os.path.exists(cache_path):
            try:
                with np.load(cache_path) as handle:
                    cached = (handle["event_no"], handle["n_pulses"])
                logger.info(
                    f"Read {len(cached[0])} pulse counts for "
                    f"{os.path.basename(path)} from {cache_path}"
                )
                return cached
            except (OSError, ValueError, KeyError):
                logger.warning(
                    f"Ignoring unreadable length cache {cache_path}"
                )

    logger.info(
        f"No cached pulse counts for {os.path.basename(path)}; scanning "
        f"{pulsemap}. This is a full index scan - pass `cache_dir` to do it "
        "once rather than once per run."
    )
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        rows = connection.execute(
            f"SELECT {index_column}, COUNT(*) FROM {pulsemap} "
            f"GROUP BY {index_column}"
        ).fetchall()

    # `GROUP BY` over the index already emits ascending keys, but the sort is
    # cheap next to the scan and makes the postcondition unconditional -
    # `event_lengths` relies on it for `searchsorted`.
    event_no = np.fromiter(
        (row[0] for row in rows), dtype=np.int64, count=len(rows)
    )
    counts = np.fromiter(
        (row[1] for row in rows), dtype=np.int32, count=len(rows)
    )
    order = np.argsort(event_no, kind="stable")
    index = (event_no[order], counts[order])

    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = _cache_path(cache_dir, path, pulsemap)
        np.savez(cache_path, event_no=index[0], n_pulses=index[1])
        logger.info(f"Wrote {len(index[0])} pulse counts to {cache_path}")

    return index


def event_lengths(
    paths: Sequence[str],
    selections: Sequence[Sequence[int]],
    pulsemap: str = "SRTInIcePulses",
    index_column: str = "event_no",
    cache_dir: Optional[str] = None,
    max_pulses: Optional[int] = None,
) -> np.ndarray:
    """Return pulse counts positionally, in `EnsembleDataset` order."""
    if len(paths) != len(selections):
        raise ValueError(
            f"Got {len(paths)} path(s) but {len(selections)} selection(s); "
            "they must line up one-to-one."
        )

    lengths: List[np.ndarray] = []
    for path, selection in zip(paths, selections):
        wanted = np.asarray(selection, dtype=np.int64)
        event_no, counts = pulse_counts(
            path=path,
            pulsemap=pulsemap,
            index_column=index_column,
            cache_dir=cache_dir,
        )
        position = np.searchsorted(event_no, wanted)
        # An event absent from the pulsemap has no pulses. `searchsorted` puts
        # it at the insertion point, which may be one past the end, so clip
        # before indexing and zero the misses afterwards.
        position = np.clip(position, 0, len(event_no) - 1)
        found = event_no[position] == wanted
        resolved = np.where(found, counts[position], 0).astype(np.int32)

        missing = int((~found).sum())
        if missing:
            Logger().warning(
                f"{missing} of {len(wanted)} selected events have no rows in "
                f"{pulsemap} of {os.path.basename(path)}; treating them as "
                "zero-pulse events."
            )
        lengths.append(resolved)

    stacked = (
        np.concatenate(lengths) if lengths else np.empty(0, dtype=np.int32)
    )
    if max_pulses is not None:
        np.minimum(stacked, max_pulses, out=stacked)
    return stacked
