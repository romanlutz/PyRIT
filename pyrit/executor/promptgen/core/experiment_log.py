# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared GPU-memory logging helpers for the promptgen experiment harnesses.

Both the GCG and PGD ``experiments.log`` modules re-export these so their
training loops report free GPU memory identically without duplicating the
``nvidia-smi`` parsing.
"""

import logging
import subprocess as sp

logger = logging.getLogger(__name__)


def get_gpu_memory() -> dict[str, int]:
    """
    Query free GPU memory via ``nvidia-smi``.

    Returns:
        dict[str, int]: Mapping of GPU identifiers to free memory in MiB.
    """
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    memory_free_values = {f"gpu{i + 1}_free_memory": int(val.split()[0]) for i, val in enumerate(memory_free_info)}
    memory_free_string = ", ".join(f"{val} MiB" for val in memory_free_values.values())
    logger.info("Free GPU memory:\n%s", memory_free_string)
    return memory_free_values


def log_gpu_memory(*, step: int) -> None:
    """
    Log free GPU memory, tolerating the absence of ``nvidia-smi``.

    Args:
        step (int): The current step number (for context in the log line).
    """
    try:
        memory_values = get_gpu_memory()
        logger.info("Step %d GPU memory: %s", step, memory_values)
    except Exception:
        logger.debug("Could not query GPU memory (nvidia-smi not available)")


__all__ = ["get_gpu_memory", "log_gpu_memory"]
