# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Logging helpers for the PGD experiment harness (mirrors the GCG
``experiments.log`` helpers).
"""

import logging

from pyrit.executor.promptgen.core.experiment_log import get_gpu_memory, log_gpu_memory

logger = logging.getLogger(__name__)


def log_train_behaviors(*, behaviors: list[str]) -> None:
    """
    Log the behaviors being attacked.

    Args:
        behaviors (list[str]): The behavior strings queued for the run.
    """
    logger.info("Behaviors (%d): %s", len(behaviors), behaviors)


__all__ = ["get_gpu_memory", "log_gpu_memory", "log_train_behaviors"]
