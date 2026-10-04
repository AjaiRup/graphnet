"""Print the comma-separated indices of GPUs that pass ECC and kernel checks."""

import os
import re
import subprocess
import sys
from typing import List, Tuple


def _log(message: str) -> None:
    print(message, file=sys.stderr)


def _visible_gpus() -> List[Tuple[int, str]]:
    """Return (index, UUID) for each GPU this process may use."""
    try:
        listing = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return []

    gpus = []
    for line in listing.splitlines():
        index, _, uuid = line.partition(",")
        if index.strip().isdigit():
            gpus.append((int(index), uuid.strip()))

    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        return gpus

    visible = []
    for entry in (field.strip() for field in mask.split(",")):
        for index, uuid in gpus:
            if entry and (entry == str(index) or uuid.startswith(entry)):
                visible.append((index, uuid))
                break
    return visible


def _has_ecc_errors(uuid: str) -> bool:
    """Return True if GPU `uuid` reports volatile uncorrectable ECC."""
    try:
        report = subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "-q", "-d", "ECC"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False

    volatile = re.search(r"Volatile([\s\S]*?)(?:Aggregate|\Z)", report)
    if volatile is None:
        return False
    for line in volatile.group(1).splitlines():
        count = re.search(r"Uncorrectable[^:]*:\s*(\d+)\s*$", line)
        if count and int(count.group(1)) > 0:
            _log(f"{uuid}: {line.strip()}")
            return True
    return False


def _error_line(stderr: str, returncode: int) -> str:
    """Return the exception a probe died on."""
    lines = stderr.splitlines()
    header = "Traceback (most recent call last):"
    starts = [i for i, line in enumerate(lines) if line.startswith(header)]
    if starts:
        for line in lines[starts[-1] + 1 :]:
            if line and not line[0].isspace():
                return line
    filled = [line for line in lines if line.strip()]
    return filled[-1] if filled else f"exit code {returncode}"


def _runs_a_kernel(uuid: str) -> bool:
    """Return True if GPU `uuid` completes a small matmul."""
    try:
        result = subprocess.run(
            [sys.executable, __file__, "--probe", uuid],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        _log(f"{uuid}: kernel probe timed out after 120 s")
        return False
    if result.returncode:
        reason = _error_line(result.stderr, result.returncode)
        _log(f"{uuid}: kernel probe failed: {reason}")
        return False
    return True


def _cuda_matches_listed(listed: int) -> bool:
    """Return whether NVML and CUDA agree on the visible GPU count."""
    try:
        import torch

        opened = torch._C._cuda_getDeviceCount()
    except Exception as error:  # no torch, or CUDA dead outright
        _log(f"cannot read the CUDA device count: {error}")
        return False
    if opened != listed:
        _log(
            f"[FATAL] nvidia-smi lists {listed} GPU(s), CUDA opens {opened}. "
            f"A card is visible and unusable; the indices below are NVML's "
            f"and need not match CUDA's."
        )
        return False
    return True


def _probe_main(device: str) -> None:
    """Run the matmul on `device`, a UUID or an index. Raises on failure."""
    os.environ["CUDA_VISIBLE_DEVICES"] = device

    import torch

    seen = torch.cuda.device_count()
    if seen != 1:
        raise RuntimeError(f"pinned {device}, but torch sees {seen}")
    a = torch.randn(512, 512, device="cuda:0")
    b = torch.randn(512, 512, device="cuda:0")
    (a @ b).sum().item()
    torch.cuda.synchronize()


def healthy_gpus() -> List[int]:
    """Return the indices of the visible GPUs that pass both checks."""
    gpus = _visible_gpus()
    if not gpus:
        _log("nvidia-smi lists no visible GPUs")
        return []
    if not _cuda_matches_listed(len(gpus)):
        _log("GPU health check failed: NVML and CUDA visible GPU counts differ")
        return []
    return [
        index
        for index, uuid in gpus
        if not _has_ecc_errors(uuid) and _runs_a_kernel(uuid)
    ]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--probe":
        _probe_main(sys.argv[2])  # raises (non-zero exit) on failure
        sys.exit(0)
    healthy = healthy_gpus()
    print(",".join(str(index) for index in healthy))
    sys.exit(0 if healthy else 42)
