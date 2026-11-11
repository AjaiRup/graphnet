"""Unit tests for `DistributedLengthGroupedSampler`."""

from typing import Any, Dict, List

import numpy as np
import pytest

from graphnet.data.dataset import DistributedLengthGroupedSampler

LENGTHS = np.random.default_rng(0).integers(1, 200, size=1003)
BATCH_SIZE = 16


def _shards(
    num_replicas: int, epoch: int = 0, **kwargs: Any
) -> List[List[int]]:
    """Return the indices of every rank for one epoch."""
    shards = []
    for rank in range(num_replicas):
        sampler = DistributedLengthGroupedSampler(
            LENGTHS,
            batch_size=BATCH_SIZE,
            num_replicas=num_replicas,
            rank=rank,
            **kwargs,
        )
        sampler.set_epoch(epoch)
        shards.append(list(sampler))
    return shards


@pytest.mark.parametrize("num_replicas", [1, 3, 4])
def test_shards_cover_dataset_equally(num_replicas: int) -> None:
    """Test that ranks get equal-sized shards that cover every event."""
    shards = _shards(num_replicas)
    num_samples = -(-len(LENGTHS) // num_replicas)
    assert all(len(shard) == num_samples for shard in shards)
    indices = np.concatenate(shards)
    assert set(indices) == set(range(len(LENGTHS)))
    assert len(indices) - len(LENGTHS) < num_replicas


def _padding_fraction(shards: List[List[int]]) -> float:
    """Return the padded fraction when each global step pads to its longest."""
    padded = real = 0
    for start in range(0, len(shards[0]) - BATCH_SIZE + 1, BATCH_SIZE):
        step = LENGTHS[
            [i for s in shards for i in s[start : start + BATCH_SIZE]]
        ]
        padded += step.size * step.max()
        real += step.sum()
    return 1 - real / padded


def test_global_steps_are_length_matched() -> None:
    """Test that grouping removes most padding across ranks in a step."""
    num_replicas = 4
    order = np.random.default_rng(1).permutation(len(LENGTHS))
    random = [list(order[rank::num_replicas]) for rank in range(num_replicas)]
    assert _padding_fraction(_shards(num_replicas)) < 0.25 * _padding_fraction(
        random
    )


def test_order_depends_on_seed_and_epoch() -> None:
    """Test that the order is reproducible and reshuffled every epoch."""
    assert _shards(2, epoch=1) == _shards(2, epoch=1)
    assert _shards(2, epoch=1) != _shards(2, epoch=2)
    assert _shards(2, seed=1) != _shards(2, seed=2)
    assert _shards(2, epoch=1, shuffle=False) == _shards(
        2, epoch=2, shuffle=False
    )


@pytest.mark.parametrize("argument", ["batch_size", "group_factor"])
def test_rejects_non_positive_sizes(argument: str) -> None:
    """Test that `batch_size` and `group_factor` must be positive."""
    kwargs: Dict[str, Any] = {"batch_size": BATCH_SIZE, argument: 0}
    with pytest.raises(ValueError):
        DistributedLengthGroupedSampler(
            LENGTHS, num_replicas=1, rank=0, **kwargs
        )
