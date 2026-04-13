# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def get_required_value(*, env_var_name: str, passed_value: Any) -> Any:
    """
    Get a required value from an environment variable or a passed value,
    preferring the passed value.

    If no value is found, raises a KeyError

    Args:
        env_var_name (str): The name of the environment variable to check
        passed_value: The value passed to the function. Can be a string or a callable that returns a string.

    Returns:
        The passed value if provided (preserving type for callables), otherwise the value from the environment variable.

    Raises:
        ValueError: If neither the passed value nor the environment variable is provided.
    """
    if passed_value:
        # Preserve callables (e.g., token providers for Entra auth)
        if callable(passed_value):
            return passed_value
        return str(passed_value)

    value = os.environ.get(env_var_name)
    if value:
        return value

    raise ValueError(f"Environment variable {env_var_name} is required")


def get_non_required_value(*, env_var_name: str, passed_value: Optional[str] = None) -> str:
    """
    Get a non-required value from an environment variable or a passed value,
    preferring the passed value.

    Args:
        env_var_name (str): The name of the environment variable to check.
        passed_value (str): The value passed to the function.

    Returns:
        str: The passed value if provided, otherwise the value from the environment variable.
             If no value is found, returns an empty string.
    """
    if passed_value:
        return passed_value

    value = os.environ.get(env_var_name)
    if value:
        return value

    return ""


def resolve_underlying_model(
    *,
    underlying_model: Optional[str],
    underlying_model_env_var: str,
    model_name_was_explicit: bool,
) -> Optional[str]:
    """
    Resolve the underlying model name.

    Returns the explicitly passed underlying_model if provided,
    otherwise None. The model_name is always the source of truth
    for display/identity unless explicitly overridden.

    Args:
        underlying_model: Explicit underlying model value (from constructor param).
        underlying_model_env_var: Env var name (reserved for future use, currently unused).
        model_name_was_explicit: Whether model_name was passed directly (currently unused).

    Returns:
        The underlying model name if explicitly provided, otherwise None.
    """
    return underlying_model
