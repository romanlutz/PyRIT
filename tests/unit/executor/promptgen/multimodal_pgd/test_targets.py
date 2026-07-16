# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Multimodal PGD affirmative-target helpers (torch-free)."""

from __future__ import annotations

import random

from pyrit.executor.promptgen.multimodal_pgd.targets import (
    AFFIRMATIVE_PREFIX,
    augment_target,
    contract_target,
    default_affirmative_target,
    shorten_target,
)


def test_default_affirmative_target_lowercases_first_word() -> None:
    result = default_affirmative_target(behavior="Describe how to build a bomb.")
    assert result == "Sure, here is describe how to build a bomb"


def test_default_affirmative_target_empty_fallback() -> None:
    assert default_affirmative_target(behavior="   ") == f"{AFFIRMATIVE_PREFIX} what you asked for"


def test_shorten_target() -> None:
    assert shorten_target("Sure, here is a plan") == "Here is a plan"


def test_contract_target() -> None:
    assert contract_target("Sure, here is a plan") == "Sure, here's a plan"


def test_augment_target_shortens_when_rng_low() -> None:
    rng = random.Random()
    rng.random = lambda: 0.1  # type: ignore[method-assign]
    assert augment_target(target="Sure, here is a plan", rng=rng) == "Here is a plan"


def test_augment_target_contracts_when_rng_high() -> None:
    rng = random.Random()
    rng.random = lambda: 0.9  # type: ignore[method-assign]
    assert augment_target(target="Sure, here is a plan", rng=rng) == "Sure, here's a plan"
