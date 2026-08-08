# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from pyrit.compat.inspect_ai import PINNED_INSPECT_EVALS_PROFILE, load_inspect_eval, run_inspect_eval_async
from pyrit.models import Message, MessagePiece, TargetResponseMetadata, ToolCallRequest
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration, TargetRequestOptions

if TYPE_CHECKING:
    from pyrit.executor.capability import ToolDeclaration, ToolExecutionPolicy


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "compose", "version"], check=True, capture_output=True, timeout=10)
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class _RequestOptionsFactory:
    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        del declarations, execution_policy
        return TargetRequestOptions()


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


class _SubmitTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset({frozenset({"text"}), frozenset({"function_call_output"})}),
        )
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        response = Message(
            message_pieces=[
                MessagePiece(
                    role="assistant",
                    original_value=ToolCallRequest(
                        call_id="submit-1",
                        name="submit",
                        arguments='{"answer":"not-the-flag"}',
                    ).to_json(),
                    original_value_data_type="function_call",
                    converted_value_data_type="function_call",
                )
            ]
        )
        for piece in response.message_pieces:
            piece.conversation_id = normalized_conversation[-1].conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="pinned-in-house-integration",
                stop_reason="tool_calls",
                provider_stop_reason="tool_calls",
            )
        )
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        del normalized_conversation


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


@pytest.mark.docker
@pytest.mark.run_only_if_all_tests
@pytest.mark.skipif(not _docker_available(), reason="Docker CLI, Compose v2, and a running daemon are required.")
async def test_user_supplied_pinned_in_house_ctf_runs_real_docker(sqlite_instance) -> None:
    del sqlite_instance
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    execution = await run_inspect_eval_async(
        source_root=Path(source_value),
        task_spec="gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf",
        task_parameters={"challenges": "ssh", "epochs": 1},
        target=_SubmitTarget(),
        request_options_factory=_RequestOptionsFactory(),
    )

    attempt = execution.result.attempts[0]
    assert attempt.task_result is not None, attempt.error
    assert attempt.task_result.tool_calls == 1
    assert attempt.task_result.scores[0].score_value == "False"
