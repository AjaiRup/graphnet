"""Regression tests for IceMix Stage-2 uncertainty calibration."""

from types import SimpleNamespace

import torch

from graphnet.models.task.reconstruction import PositionReconstructionWithSigma
from graphnet.training.callbacks import PhysicsMetricsCallback
from graphnet.training.loss_functions import (
    IsotropicGaussianNLLLoss,
    PerEventRecordingLoss,
    VonMisesFisher3DLoss,
)
from graphnet.utilities.config import ModelConfig


def test_sigma_task_and_loss_preserve_frozen_positions() -> None:
    """Clamp log-sigma and differentiate only its output in Stage 2."""
    task = PositionReconstructionWithSigma(
        hidden_size=4, loss_function=IsotropicGaussianNLLLoss()
    )
    output = task._forward(torch.tensor([[1.0, 2.0, 3.0, -10.0]]))
    assert torch.equal(output, torch.tensor([[1.0, 2.0, 3.0, -4.0]]))

    prediction = torch.tensor([[1.0, -1.0, 2.0, 0.0]], requires_grad=True)
    target = torch.tensor([[0.0, 1.0, 2.0]])
    loss = IsotropicGaussianNLLLoss(detach_position=True)(
        prediction, target, return_elements=True
    )
    assert torch.allclose(loss, torch.tensor([2.5]))
    loss.sum().backward()
    assert torch.equal(prediction.grad[:, :3], torch.zeros(1, 3))
    assert prediction.grad[0, 3] != 0


def test_selected_direction_metric_uses_the_energy_threshold() -> None:
    """Select the primary direction metric at energy >= 10 GeV."""
    module = torch.nn.Module()
    module.direction_loss = PerEventRecordingLoss(
        VonMisesFisher3DLoss(), metric_name="direction"
    )
    target = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    prediction = torch.tensor([[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 1.0]])
    module.direction_loss(prediction, target)

    values = PhysicsMetricsCallback(sel_energy_min=10.0)._per_event_values(
        module, SimpleNamespace(energy=torch.tensor([9.0, 10.0]))
    )
    assert torch.allclose(
        values["angular_error_deg_sel"], torch.tensor([90.0]), atol=1e-4
    )


def test_stage_two_config_declares_sigma_calibration() -> None:
    """Keep the committed Stage-2 architecture loadable and explicit."""
    config = ModelConfig.load("configs/models/icemix_joint_reconstruction_stage2.yml")
    tasks = {task.class_name: task for task in config.arguments["tasks"]}
    position = tasks["PositionReconstructionWithSigma"].arguments
    loss = position["loss_function"].arguments["loss_function"]
    assert loss.class_name == "IsotropicGaussianNLLLoss"
    assert loss.arguments["detach_position"]
