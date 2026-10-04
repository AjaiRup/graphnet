"""Regression for re-saving named functions reconstructed from YAML."""

import pytest

from graphnet.utilities.config import ModelConfig


@pytest.mark.parametrize("source", [
    "def scale_position(x):\n    return x / 100.0\n",
    "def unscale_position(x):\n    return x * 100.0\n",
])
def test_named_function_can_be_serialized_repeatedly(source):
    serialized = "!function " + source
    for _ in range(3):
        function = ModelConfig._deserialise(serialized, trust=True)
        assert function(100.0) in (1.0, 10000.0)
        serialized = ModelConfig._serialise(function)
        assert serialized == "!function " + source


def test_named_function_still_requires_trust():
    with pytest.raises(ValueError, match="trust=False"):
        ModelConfig._deserialise("!function def identity(x):\n    return x\n")
