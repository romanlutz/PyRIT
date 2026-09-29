# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pinned benign Inspect/GHCP pilot settings, not a cyber benchmark profile."""

from __future__ import annotations

from typing import TYPE_CHECKING

from inspect_ai.model import GenerateConfig, Model, get_model

from pyrit.executor.attack.core.attack_config import AttackAdversarialConfig, AttackScoringConfig
from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.prompt_target import OpenAIChatTarget, PromptTarget
from pyrit.score.true_false.substring_scorer import SubStringScorer

if TYPE_CHECKING:
    from pyrit.executor.benchmark._inspect_ghcp_adversary_capture import InspectGhcpAdversarialCapture


class InspectGhcpProtocolPins:
    """Local-only identities already reviewed for the benign GHCP pilot."""

    PYTHON_GHCP_BASE_ID = "sha256:74241307a041d9370c1fd0d63b1b392e0ba0c155645dca295d654d89c839712f"
    TARGET_IMAGE_ID = "sha256:eb0733bfb68366fe045307e8631717d5228f8da196e471a39c5e120b63c2e4c7"
    GHCP_CLI_SHA256 = "0059754cf78c3f3bf2c9d4564dfa7e9e25f3a3f8f411f2f0cdad9363f5662748"
    CLI_PATH = "/opt/pyrit/copilot"
    MODEL_ENDPOINT = "http://127.0.0.1:11435/v1"
    MODEL_NAME = "qwen3:1.7b"
    CLI_MODEL_ALIAS = "qwen3-local"


def build_benign_red_teaming_attack(
    *, target: PromptTarget, capture: InspectGhcpAdversarialCapture
) -> RedTeamingAttack:
    """
    Use one real loopback adversary and PyRIT's existing red-teaming decisions.

    Returns:
        RedTeamingAttack: The same bounded two-turn attack used by the direct pilot.
    """
    adversary = OpenAIChatTarget(
        endpoint=InspectGhcpProtocolPins.MODEL_ENDPOINT,
        model_name=InspectGhcpProtocolPins.MODEL_NAME,
        api_key="ollama",
        max_completion_tokens=512,
        temperature=0.2,
        httpx_client_kwargs={"http_client": capture.client, "max_retries": 0},
    )
    return RedTeamingAttack(
        objective_target=target,
        attack_adversarial_config=AttackAdversarialConfig(target=adversary),
        attack_scoring_config=AttackScoringConfig(
            objective_scorer=SubStringScorer(substring="__impossible_scoring_canary__")
        ),
        max_turns=2,
    )


def create_benign_inspect_model() -> Model:
    """
    Build only the original pinned trusted-host loopback Inspect provider.

    Returns:
        Model: A non-memoized, capture-capable Qwen provider for this pilot.
    """
    from pyrit.executor.benchmark.inspect_ghcp_model import InspectLoopbackModelAPI  # noqa: F401

    return get_model(
        "pyrit_loopback/qwen3:1.7b",
        base_url=InspectGhcpProtocolPins.MODEL_ENDPOINT,
        api_key="ollama",
        config=GenerateConfig(max_tokens=512, timeout=60, max_retries=0, temperature=0.2),
        memoize=False,
    )
