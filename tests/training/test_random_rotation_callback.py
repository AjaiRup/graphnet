"""Tests for coordinate-consistent random rotation augmentation."""

from types import SimpleNamespace
from typing import Tuple

import numpy as np
import torch

from graphnet.training.callbacks import RandomRotationCallback


def _rotation(angles: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cosine and sine for a callback seed's event angles."""
    return torch.cos(angles), torch.sin(angles)


def _rotate_about(
    xy: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    origin: Tuple[float, float],
) -> torch.Tensor:
    """Return `xy` rotated about an `(x, y)` origin."""
    shifted = xy - torch.tensor(origin, dtype=xy.dtype)
    return torch.stack(
        [
            shifted[:, 0] * cos - shifted[:, 1] * sin + origin[0],
            shifted[:, 0] * sin + shifted[:, 1] * cos + origin[1],
        ],
        dim=1,
    )


def test_rotation_uses_matching_node_and_label_pivots() -> None:
    """Rotate global labels and standardized nodes about one physical pivot."""
    node_origin = (46.3 / 500.0, -34.9 / 500.0)
    position_origin = (46.3, -34.9)
    batch = torch.tensor([0, 0, 1])
    node_xy = torch.tensor(
        [[0.15, -0.02], [0.11, -0.12], [-0.03, 0.04]]
    )
    positions = torch.tensor([[60.0, -20.0], [25.0, -50.0]])
    directions = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    data = SimpleNamespace(
        batch=batch,
        x=torch.cat([node_xy, torch.ones(3, 1)], dim=1),
        position_x=positions[:, 0].clone(),
        position_y=positions[:, 1].clone(),
        direction=directions.clone(),
        azimuth=torch.tensor([0.0, np.pi / 2]),
    )

    seed = 17
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    angles = torch.rand(2, generator=generator) * 2.0 * np.pi
    cos, sin = _rotation(angles)

    callback = RandomRotationCallback(
        seed=seed,
        node_xy_origin=node_origin,
        position_xy_origin=position_origin,
    )
    callback._apply_rotation(data)

    expected_nodes = _rotate_about(
        node_xy, cos[batch], sin[batch], node_origin
    )
    expected_positions = _rotate_about(positions, cos, sin, position_origin)
    expected_centered_positions = _rotate_about(
        positions - torch.tensor(position_origin), cos, sin, (0.0, 0.0)
    )
    expected_directions = _rotate_about(
        directions[:, :2], cos, sin, (0.0, 0.0)
    )

    assert torch.allclose(data.x[:, :2], expected_nodes)
    assert torch.allclose(
        torch.stack([data.position_x, data.position_y], dim=1),
        expected_positions,
    )
    assert torch.allclose(data.direction[:, :2], expected_directions)
    assert torch.allclose(
        torch.stack([data.position_x, data.position_y], dim=1)
        - torch.tensor(position_origin),
        expected_centered_positions,
    )


def test_rotation_default_remains_global_origin() -> None:
    """Preserve the legacy global-origin behaviour for generic callers."""
    data = SimpleNamespace(
        batch=torch.tensor([0]),
        x=torch.tensor([[1.0, 0.0]]),
        position_x=torch.tensor([1.0]),
        position_y=torch.tensor([0.0]),
    )
    seed = 3
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    angle = torch.rand(1, generator=generator) * 2.0 * np.pi
    cos, sin = _rotation(angle)

    RandomRotationCallback(seed=seed)._apply_rotation(data)

    expected = _rotate_about(
        torch.tensor([[1.0, 0.0]]), cos, sin, (0.0, 0.0)
    )
    assert torch.allclose(data.x[:, :2], expected)
    assert torch.allclose(
        torch.stack([data.position_x, data.position_y], dim=1), expected
    )
