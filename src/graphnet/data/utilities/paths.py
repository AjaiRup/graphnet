"""Path resolution for dataset configuration files."""

import os
from typing import List, Union

from graphnet.constants import GRAPHNET_ROOT_DIR


def resolve_dataset_paths(
    paths: Union[str, List[str]],
) -> List[str]:
    """Expand `$GRAPHNET` and environment variables in dataset paths."""
    if isinstance(paths, str):
        paths = [paths]

    resolved = []
    for path in paths:
        for token in ("$graphnet", "$GRAPHNET", "${graphnet}", "${GRAPHNET}"):
            path = path.replace(token, GRAPHNET_ROOT_DIR)
        resolved.append(os.path.expandvars(path))
    return resolved
