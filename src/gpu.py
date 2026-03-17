"""GPU detection, selection, and logging utility.

Provides name-based GPU resolution so configs can reference GPUs by
substring of their name (e.g. ``"RTX 4070"``) rather than fragile
positional indices that may change between reboots.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

log = logging.getLogger(__name__)


def discover_gpus() -> list[dict[str, Any]]:
    """Return a list of dicts describing each visible CUDA GPU."""
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free_mem, _ = torch.cuda.mem_get_info(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "vram_gb": round(props.total_memory / (1024**3), 1),
            "free_gb": round(free_mem / (1024**3), 1),
        })
    return gpus


def resolve_gpu(identifier: str | int) -> int:
    """Resolve a GPU name substring or index to a CUDA device index.

    Accepts either:
    - An integer device index (returned as-is after bounds check).
    - A string matched case-insensitively against GPU names.
      The first GPU whose name contains the substring wins.

    Raises ``ValueError`` if no GPU matches.
    """
    if not torch.cuda.is_available():
        log.warning("CUDA not available — cannot resolve GPU identifier %r", identifier)
        return 0

    n = torch.cuda.device_count()

    if isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        idx = int(identifier)
        if 0 <= idx < n:
            return idx
        raise ValueError(f"GPU index {idx} out of range (have {n} GPUs)")

    needle = str(identifier).lower()
    for i in range(n):
        name = torch.cuda.get_device_properties(i).name.lower()
        if needle in name:
            log.info("Resolved GPU %r → device %d (%s)", identifier, i,
                     torch.cuda.get_device_properties(i).name)
            return i

    available = [torch.cuda.get_device_properties(i).name for i in range(n)]
    raise ValueError(
        f"No GPU matching {identifier!r}. Available: {available}"
    )


def get_device(identifier: str | int | None = None) -> torch.device:
    """Return a torch device, optionally resolving by name.

    When ``CUDA_VISIBLE_DEVICES`` is already set the visible GPU is
    always ``cuda:0``.  Otherwise *identifier* is resolved via
    :func:`resolve_gpu`.
    """
    if not torch.cuda.is_available():
        log.warning("CUDA not available — falling back to CPU")
        return torch.device("cpu")
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return torch.device("cuda:0")
    if identifier is not None:
        idx = resolve_gpu(identifier)
        return torch.device(f"cuda:{idx}")
    return torch.device("cuda:0")


def log_gpu_status() -> None:
    """Log available GPU information: name, VRAM, CUDA version."""
    if not torch.cuda.is_available():
        log.warning("No CUDA GPUs detected")
        return

    cuda_vis = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    log.info("CUDA_VISIBLE_DEVICES = %s", cuda_vis)
    log.info("CUDA version: %s", torch.version.cuda)

    for gpu in discover_gpus():
        log.info(
            "GPU %d: %s — %.1f GB total, %.1f GB free",
            gpu["index"], gpu["name"], gpu["vram_gb"], gpu["free_gb"],
        )


def check_vram(batch_size: int, min_gb: float = 2.0) -> None:
    """Warn if the visible GPU has less than *min_gb* free VRAM."""
    if not torch.cuda.is_available():
        return
    free_mem, _ = torch.cuda.mem_get_info(0)
    free_gb = free_mem / (1024**3)
    if free_gb < min_gb:
        log.warning(
            "GPU has only %.1f GB free VRAM (min %.1f GB recommended for batch_size=%d). "
            "Consider reducing batch_size.",
            free_gb,
            min_gb,
            batch_size,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    log_gpu_status()
