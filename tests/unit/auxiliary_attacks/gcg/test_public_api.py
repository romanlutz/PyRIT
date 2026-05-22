# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the top-level :mod:`pyrit.auxiliary_attacks.gcg` public API surface."""

import pyrit.auxiliary_attacks.gcg as gcg_pkg
from pyrit.auxiliary_attacks.gcg import (
    GCG,
    GCGAlgorithmConfig,
    GCGConfig,
    GCGContext,
    GCGDataConfig,
    GCGGenerator,
    GCGModelConfig,
    GCGOutputConfig,
    GCGResult,
    GCGStrategyConfig,
    load_goals_and_targets,
)


def test_gcg_alias_is_gcg_generator() -> None:
    assert GCG is GCGGenerator


def test_public_api_symbols_are_exported() -> None:
    expected = {
        "GCG",
        "GCGAlgorithmConfig",
        "GCGConfig",
        "GCGContext",
        "GCGDataConfig",
        "GCGGenerator",
        "GCGModelConfig",
        "GCGOutputConfig",
        "GCGResult",
        "GCGStrategyConfig",
        "load_goals_and_targets",
    }
    assert expected.issubset(set(gcg_pkg.__all__))


def test_public_api_symbols_are_importable_from_package() -> None:
    # Smoke-test that the imports at module top resolved to real objects so the
    # short import path (e.g. ``from pyrit.auxiliary_attacks.gcg import GCG``)
    # stays stable.
    symbols = (
        GCG,
        GCGAlgorithmConfig,
        GCGConfig,
        GCGContext,
        GCGDataConfig,
        GCGGenerator,
        GCGModelConfig,
        GCGOutputConfig,
        GCGResult,
        GCGStrategyConfig,
        load_goals_and_targets,
    )
    for sym in symbols:
        assert sym is not None
