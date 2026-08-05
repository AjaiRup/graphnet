"""Path resolution for dataset configuration files."""

import os
from typing import List, Union

from graphnet.constants import GRAPHNET_ROOT_DIR


def resolve_dataset_paths(
    paths: Union[str, List[str]],
) -> List[str]:
    """Expand `$GRAPHNET` and environment variables in dataset paths.

    `DatasetConfig` paths may refer to data relative to the GraphNeT checkout
    as `$GRAPHNET/...`, which `Dataset._resolve_graphnet_paths` substitutes,
    and to site-specific locations through ordinary environment variables such
    as `$DATA_ROOT`, which it does not. This applies both, in that order.

    Order matters: `os.path.expandvars` would otherwise silently blank
    `$GRAPHNET` whenever that variable is not exported - which it normally is
    not, since it is a GraphNeT convention rather than a real environment
    variable.

    Args:
        paths: A path, or list of paths, as given in a `DatasetConfig`.

    Returns:
        The paths with both kinds of reference substituted.
    """
    if isinstance(paths, str):
        paths = [paths]

    resolved = []
    for path in paths:
        for token in ("$graphnet", "$GRAPHNET", "${graphnet}", "${GRAPHNET}"):
            path = path.replace(token, GRAPHNET_ROOT_DIR)
        resolved.append(os.path.expandvars(path))
    return resolved
