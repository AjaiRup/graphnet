"""CLI checks for the IceMix joint-reconstruction example."""

import os
import runpy

import pytest

from graphnet.constants import GRAPHNET_ROOT_DIR


SCRIPT = os.path.join(
    GRAPHNET_ROOT_DIR,
    "examples/04_training/09_train_icemix_joint_reconstruction.py",
)


@pytest.fixture(name="parser")
def fixture_parser():  # type: ignore[no-untyped-def]
    """Load the parser without executing the training entry point."""
    namespace = runpy.run_path(SCRIPT, run_name="icemix_training_test")
    return namespace["build_parser"]()


@pytest.mark.parametrize(
    "arguments", [["--distribution-strategy", "ddp"], ["--scheduler-patience", "3"]]
)
def test_rejects_unknown_arguments(parser, arguments):  # type: ignore[no-untyped-def]
    """Reject options that would otherwise be silently ignored."""
    with pytest.raises(SystemExit):
        parser.parse_args(arguments)


def test_approved_stage_one_arguments(parser):  # type: ignore[no-untyped-def]
    """Accept the canonical Stage-1 interface."""
    args = parser.parse_args(
        ["--seed", "42", "--max-epochs", "100", "--rotation-augmentation"]
    )
    assert args.seed == 42
    assert args.max_epochs == 100
    assert args.rotation_augmentation
