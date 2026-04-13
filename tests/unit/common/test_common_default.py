# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

import pytest

from pyrit.common import default_values


def test_get_required_value_prefers_passed():
    os.environ["TEST_ENV_VAR"] = "fail"
    assert default_values.get_required_value(env_var_name="TEST_ENV_VAR", passed_value="passed") == "passed"


def test_get_required_value_uses_default():
    os.environ["TEST_ENV_VAR"] = "default"
    assert default_values.get_required_value(env_var_name="TEST_ENV_VAR", passed_value="") == "default"


def test_get_required_value_throws_if_not_set():
    os.environ["TEST_ENV_VAR"] = ""
    with pytest.raises(ValueError):
        default_values.get_required_value(env_var_name="TEST_ENV_VAR", passed_value="")


def test_resolve_underlying_model_explicit_value():
    result = default_values.resolve_underlying_model(
        underlying_model="gpt-4o",
        underlying_model_env_var="UNUSED_VAR",
        model_name_was_explicit=True,
    )
    assert result == "gpt-4o"


def test_resolve_underlying_model_returns_none_when_not_passed():
    os.environ["TEST_UNDERLYING"] = "gpt-4o-from-env"
    result = default_values.resolve_underlying_model(
        underlying_model=None,
        underlying_model_env_var="TEST_UNDERLYING",
        model_name_was_explicit=False,
    )
    # env var is NOT consulted — only explicit underlying_model matters
    assert result is None
    del os.environ["TEST_UNDERLYING"]


def test_resolve_underlying_model_returns_none_when_nothing_set():
    result = default_values.resolve_underlying_model(
        underlying_model=None,
        underlying_model_env_var="NONEXISTENT_VAR_12345",
        model_name_was_explicit=False,
    )
    assert result is None
