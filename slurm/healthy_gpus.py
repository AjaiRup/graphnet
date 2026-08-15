"""Print comma-separated visible GPU indices that pass ECC and kernel checks."""

import re
import subprocess
import sys
from typing import List


def _has_ecc_errors(index: int) -> bool:
    """Return True if device `index` reports uncorrectable ECC errors."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "-q", "-d", "ECC"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    match = re.search(r"Volatile Uncorr\. ECC.*?:\s+(\d+)", output)
    return bool(match) and match.group(1) not in ("0", "N/A")


def _runs_a_kernel(index: int) -> bool:
    """Return True if device `index` completes a small matmul."""
    import torch

    try:
        torch.cuda.set_device(index)
        a = torch.randn(512, 512, device=f"cuda:{index}")
        b = torch.randn(512, 512, device=f"cuda:{index}")
        (a @ b).sum().item()
        torch.cuda.synchronize(index)
    except Exception:  # noqa: BLE001 - any failure means "do not use it"
        return False
    return True


def healthy_gpus() -> List[int]:
    """Return the indices of the visible GPUs that pass both checks."""
    import torch

    return [
        index
        for index in range(torch.cuda.device_count())
        if not _has_ecc_errors(index) and _runs_a_kernel(index)
    ]


if __name__ == "__main__":
    healthy = healthy_gpus()
    print(",".join(str(index) for index in healthy))
    sys.exit(0 if healthy else 42)
