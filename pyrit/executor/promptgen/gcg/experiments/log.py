# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any

from pyrit.executor.promptgen.core.experiment_log import get_gpu_memory, log_gpu_memory

logger = logging.getLogger(__name__)

_DEFAULT_PARAM_KEYS: list[str] = [
    "model_name",
    "transfer",
    "n_train_data",
    "n_test_data",
    "n_steps",
    "batch_size",
]


def log_params(
    *,
    params: Any,
    param_keys: list[str] | None = None,
) -> None:
    """
    Log selected parameters via Python logging.

    Args:
        params (Any): A config object with a `to_dict()` method containing all parameters.
        param_keys (list[str] | None): Keys to extract and log. Defaults to standard GCG training keys.
    """
    if param_keys is None:
        param_keys = _DEFAULT_PARAM_KEYS
    logged_params = {key: params.to_dict()[key] for key in param_keys}
    logger.info(f"Training parameters: {logged_params}")


def log_train_goals(*, train_goals: list[str]) -> None:
    """
    Log training goals via Python logging.

    Args:
        train_goals (list[str]): The list of training goal strings to log.
    """
    logger.info(f"Training goals ({len(train_goals)}): {train_goals}")


def log_loss(*, step: int, loss: float) -> None:
    """
    Log training loss via Python logging.

    Args:
        step (int): The current training step number.
        loss (float): The loss value to log.
    """
    logger.info(f"Step {step} loss: {loss}")


def log_table_summary(*, losses: list[float], controls: list[str], n_steps: int) -> None:
    """
    Log a summary of losses and controls via Python logging.

    Args:
        losses (list[float]): Loss values for each step.
        controls (list[str]): Control strings for each step.
        n_steps (int): Total number of steps.
    """
    logger.info(f"Training complete ({n_steps} steps). Final loss: {losses[-1] if losses else 'N/A'}")
    if controls:
        logger.info(f"Final control: {controls[-1]}")


__all__ = [
    "get_gpu_memory",
    "log_gpu_memory",
    "log_loss",
    "log_params",
    "log_table_summary",
    "log_train_goals",
]
