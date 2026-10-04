"""Training CLI, configuration, and checkpoint contracts."""

import os
import runpy

import pytest
import torch

from graphnet.constants import GRAPHNET_ROOT_DIR
from graphnet.models.transformer.icemix import (
    IceMixReconstruction,
    load_icemix_model,
)
from graphnet.utilities.config import ModelConfig

SCRIPT = os.path.join(
    GRAPHNET_ROOT_DIR,
    "examples/04_training/09_train_icemix_joint_reconstruction.py",
)


@pytest.fixture(scope="module")
def training():
    return runpy.run_path(SCRIPT, run_name="icemix_training_test")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--fine-tune-from", "old.ckpt"],
        ["--dual-task-queries"],
        ["--hidden-dim", "32"],
        ["--max-pulses", "32"],
        ["--distribution-strategy", "ddp"],
    ],
)
def test_removed_arguments_are_rejected(training, arguments):
    with pytest.raises(SystemExit):
        training["build_parser"]().parse_args(arguments)


def test_defaults_and_rotation(training, tmp_path):
    args = training["build_parser"]().parse_args([])
    assert args.max_epochs == 80 and args.early_stopping_patience == 0
    assert args.rotation_augmentation and args.seed == 42
    callbacks = training["build_callbacks"](args, str(tmp_path))
    rotation = callbacks[-1]
    assert rotation._position_xy_origin == (46.3, -34.9)
    assert rotation._node_xy_origin == (46.3 / 500.0, -34.9 / 500.0)
    monitors = {getattr(callback, "monitor", None) for callback in callbacks}
    assert {
        "val_loss",
        "val/position_error_m_mean",
        "val/angular_error_deg_sel_median",
    } <= monitors
    args = training["build_parser"]().parse_args(
        ["--no-rotation-augmentation"]
    )
    assert not args.rotation_augmentation


def test_config_overrides_reach_optimizer_and_loss(training, tmp_path):
    path = tmp_path / "model.yml"
    ModelConfig(
        class_name="IceMixReconstruction",
        arguments={"lr": 0.001, "vertex_weight": 1.3},
    ).dump(str(path))
    args = training["build_parser"]().parse_args(
        [
            "--model-config",
            str(path),
            "--lr",
            "0.0002",
            "--weight-decay",
            "0.02",
            "--adam-eps",
            "0.00002",
            "--direction-weight",
            "0.5",
            "--scheduler-factor",
            "0.3",
            "--scheduler-patience",
            "5",
            "--dropout",
            "0.2",
            "--attn-drop",
            "0.15",
            "--proj-drop",
            "0.25",
            "--drop-path-rate",
            "0.3",
        ]
    )
    model = training["build_model"](args)
    configured = model.configure_optimizers()
    assert configured["optimizer"].param_groups[0]["lr"] == 0.0002
    assert configured["optimizer"].param_groups[0]["weight_decay"] == 0.02
    assert configured["optimizer"].param_groups[0]["eps"] == 0.00002
    assert configured["lr_scheduler"]["scheduler"].patience == 5
    assert configured["lr_scheduler"]["scheduler"].factor == 0.3
    assert model._tasks[0]._loss_function._loss_function._factors == [1.3]
    assert model._tasks[1]._loss_function._loss_function._factors == [0.5]
    assert model.backbone.sandwich[0].attn.proj_drop.p == 0.25
    assert model.backbone.blocks[-1].drop_path.drop_prob == 0.3
    saved = tmp_path / "saved.yml"
    model.save_config(str(saved))
    restored = load_icemix_model(str(saved))
    assert restored.config.arguments == model.config.arguments
    restored.load_state_dict(model.state_dict(), strict=True)
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_canonical_config_matches_default_constructor(training):
    args = training["build_parser"]().parse_args([])
    configured = training["build_model"](args)
    default = IceMixReconstruction()
    assert configured.config.arguments == default.config.arguments
    with pytest.raises(TypeError):
        IceMixReconstruction(vertex_only_norm=False)


def test_resume_path_is_forwarded_to_fit(training, monkeypatch, tmp_path):
    from types import SimpleNamespace

    recorded = {}
    model = SimpleNamespace(
        fit=lambda *args, **kwargs: recorded.update(kwargs),
        save_config=lambda path: None,
        save_state_dict=lambda path: None,
    )
    monkeypatch.setitem(
        training["main"].__globals__, "build_model", lambda args: model
    )
    monkeypatch.setitem(
        training["main"].__globals__,
        "build_dataloaders",
        lambda **kwargs: {
            "train": SimpleNamespace(dataset=[1]),
            "validation": SimpleNamespace(dataset=[2]),
        },
    )
    args = training["build_parser"]().parse_args(
        [
            "--resume-from",
            "interrupted.ckpt",
            "--skip-prediction",
            "--output-dir",
            str(tmp_path),
        ]
    )
    args.gpus = None
    training["main"](args)
    assert recorded["ckpt_path"] == "interrupted.ckpt"
    assert recorded["max_epochs"] == 80


def test_smoke_keeps_architecture(training):
    args = training["build_parser"]().parse_args(
        ["--smoke-test", "--gpus", "0"]
    )
    training["_apply_smoke_test_overrides"](args)
    assert args.gpus == [0]
    assert args.limit_train_batches == args.limit_val_batches == 1
    assert args.lr is None and args.dropout is None
