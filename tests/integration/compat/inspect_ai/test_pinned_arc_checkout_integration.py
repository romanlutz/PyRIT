# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from pyrit.compat.inspect_ai import PINNED_INSPECT_EVALS_PROFILE, load_inspect_eval, run_inspect_eval_async
from pyrit.models import Message, TargetResponseMetadata
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration


class _ArcTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset({frozenset({"text"})}),
        )
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        response = Message.from_prompt(prompt="ANSWER: B", role="assistant")
        conversation_id = normalized_conversation[-1].conversation_id
        for piece in response.message_pieces:
            piece.conversation_id = conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="pinned-arc-integration",
                stop_reason="completed",
                provider_stop_reason="completed",
            )
        )
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        return


@pytest.mark.run_only_if_all_tests
async def test_user_supplied_pinned_checkout_runs_unchanged_arc_source(sqlite_instance) -> None:
    del sqlite_instance
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")
    records: list[dict[str, Any]] = [
        {
            "id": "integration-arc",
            "question": "Which number is even?",
            "choices": {"label": ["A", "B"], "text": ["3", "4"]},
            "answerKey": "B",
        }
    ]

    execution = await run_inspect_eval_async(
        source_root=Path(source_value),
        task_spec="arc/arc.py@arc_challenge",
        target=_ArcTarget(),
        dataset_loader=lambda *args, **kwargs: records,
    )

    assert execution.loaded.report.source_revision_verified is True
    assert execution.loaded.report.source_revision == PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision
    assert execution.loaded.suite.cases[0].scorers[0].config["expected_value"] == "B"
    task_result = execution.result.attempts[0].task_result
    assert task_result is not None
    assert task_result.scores[0].score_value == "True"


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_checkout_constructs_unchanged_in_house_ctf_source() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf",
        task_parameters={"challenges": "ssh", "epochs": 1},
    )

    assert loaded.report.source_revision_verified is True
    assert loaded.suite.cases[0].case_id == "ssh"
    assert loaded.suite.cases[0].scorers[0].kind == "inspect_check_flag"


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_checkout_constructs_unchanged_intercode_ctf_source() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    cache_value = os.getenv("PYRIT_INSPECT_EVALS_CACHE_DIR")
    if not source_value or not cache_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT and PYRIT_INSPECT_EVALS_CACHE_DIR to exact pinned inputs.")

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="gdm_intercode_ctf/gdm_intercode_ctf.py@gdm_intercode_ctf",
        task_parameters={"sample_ids": [2]},
        inspect_evals_cache_dir=Path(cache_value),
    )

    assert loaded.report.source_revision_verified is True
    assert loaded.suite.cases[0].source is not None
    assert loaded.suite.cases[0].source.source_id == "2"
    assert [tool.declaration.name for tool in loaded.suite.cases[0].tools] == ["bash", "python", "submit"]
