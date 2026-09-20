# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pyrit.executor.benchmark import InspectBenchmark, InspectTaskBinding
from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark._inspect_response_trace import InspectResponseTrace
from pyrit.executor.benchmark.inspect_sandbox import InspectDockerProfile, InspectSandboxTools
from pyrit.memory import CentralMemory
from pyrit.models import MessagePiece, ScoringExpectation
from pyrit.prompt_target import OpenAIResponseTarget
from tests.unit.mocks import MockPromptTarget, openai_response_json_dict

pytest.importorskip("inspect_ai")


def _task() -> Any:
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import includes

    return Task(
        dataset=[Sample(id=4, input="fixture task", target="fixture-answer")], scorer=includes(), name="unit-fixture"
    )


def _binding(task: Any) -> InspectTaskBinding:
    return InspectTaskBinding(
        task=task,
        sample_id=4,
        candidate=MessagePiece(role="user", original_value="fixture task").to_message(),
        system_prompt="Inspect the task files and return the answer.",
        native_scorer="includes",
        expectation=ScoringExpectation(objective="Fixture correctness"),
    )


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("outcome", ["success", "timeout", "launch_failure"])
async def test_real_target_loop_native_eval_and_persistence_async(tmp_path: Path, outcome: str) -> None:
    from inspect_ai.util import ExecResult, SandboxEnvironment

    artifacts = InspectRunArtifacts(directory=tmp_path / "run", provenance={"fixture": True})
    trace = InspectResponseTrace(artifacts=artifacts, max_requests=9)
    tools = InspectSandboxTools(
        artifacts=artifacts, trace=trace, max_executions=8, timeout_seconds=30, output_limit_bytes=16384
    )
    profile = InspectDockerProfile(artifacts=artifacts, image="python:test@sha256:123")
    requests: list[dict[str, Any]] = []
    timeout = outcome == "timeout"

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        body = openai_response_json_dict()
        body["usage"] = {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10}
        if len(requests) == 1:
            body["output"] = [
                {
                    "type": "function_call",
                    "name": "bash",
                    "arguments": '{"command":"cat flag"}',
                    "call_id": "provider-call-1",
                    "id": "function-1",
                    "status": "completed",
                }
            ]
        else:
            body["output"][0]["content"][0]["text"] = "fixture-answer"
        return httpx.Response(200, json=body)

    def execute(*, cmd: list[str], **kwargs: Any) -> Any:
        if outcome == "launch_failure":
            return ExecResult(success=False, returncode=125, stdout="", stderr="Docker rejected exec")
        options = json.loads(kwargs["input"])
        result = {
            "stdout": "partial" if timeout else "fixture-answer\n",
            "stderr": "",
            "returncode": None if timeout else 0,
            "timed_out": timeout,
            "truncated": False,
            "error": "tool_timeout" if timeout else None,
            "execution_id": options["execution_id"],
        }
        start = {"protocol": "pyrit-inspect-tool-v1", "event": "started", "execution_id": result["execution_id"]}
        end = {**start, "event": "completed", "result": result, "termination_error": None}
        return ExecResult(success=True, returncode=0, stdout=json.dumps(start) + "\n" + json.dumps(end), stderr="")

    sandbox = MagicMock(spec=SandboxEnvironment)
    sandbox.exec = AsyncMock(side_effect=execute)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(provider),
        event_hooks={"request": [trace.request_async], "response": [trace.response_async]},
    ) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.invalid/v1",
            model_name="fixture-model",
            api_key="fixture-key",
            custom_functions={"bash": tools.bash_async, "python": tools.python_async},
            max_output_tokens=2048,
            fail_on_missing_function=True,
            extra_body_parameters={"tools": tools.schemas(), "parallel_tool_calls": False, "store": False},
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        benchmark = InspectBenchmark(
            objective_target=target,
            model_name="fixture-model",
            artifacts=artifacts,
            tools=tools,
            trace=trace,
            docker_profile=profile,
            episode_timeout_seconds=60,
        )
        with (
            patch("inspect_ai.util.sandbox", return_value=sandbox),
            patch.object(profile, "verify_running_async", new_callable=AsyncMock),
            patch.object(profile, "verify_cleanup_async", new_callable=AsyncMock) as cleanup,
            patch.dict("os.environ", {"RETRY_MAX_NUM_ATTEMPTS": "1"}),
        ):
            if outcome != "success":
                with pytest.raises(ValueError, match="No authentic final response"):
                    await benchmark.execute_async(binding=_binding(_task()))
                assert len(requests) == 1
                assert artifacts.manifest["grade_status"] == "not_available"
                assert "score" not in artifacts.manifest
            else:
                result = await benchmark.execute_async(binding=_binding(_task()))
                assert result.response.get_value() == "fixture-answer"
                assert result.score.get_value() is True
                assert len(requests) == 2
                assert trace.summary()["usage"]["total_tokens"] == 20
                messages = CentralMemory.get_memory_instance().get_conversation_messages(
                    conversation_id=result.conversation_id
                )
                pieces = [piece for message in messages for piece in message.message_pieces]
                calls = [
                    json.loads(piece.original_value)
                    for piece in pieces
                    if piece.original_value_data_type == "function_call"
                ]
                outputs = [
                    json.loads(piece.original_value)
                    for piece in pieces
                    if piece.original_value_data_type == "function_call_output"
                ]
                assert calls[0]["call_id"] == outputs[0]["call_id"] == "provider-call-1"
                assert tools.executions[0]["provider_call_id"] == "provider-call-1"
                assert tools.executions[0]["execution_id"] != "provider-call-1"
                assert all("solution" not in json.dumps(request) for request in requests)
                assert artifacts.manifest["evidence_status"] == "complete"
            cleanup.assert_awaited_once()
    assert sandbox.exec.await_count == 1
    assert "fixture-key" not in (artifacts.directory / "events.jsonl").read_text()


@pytest.mark.usefixtures("patch_central_database")
async def test_write_only_normalizer_echo_is_not_an_answer_async(tmp_path: Path) -> None:
    artifacts = InspectRunArtifacts(directory=tmp_path / "run", provenance={})
    profile = InspectDockerProfile(artifacts=artifacts, image="python:test@sha256:123")
    target = MockPromptTarget()
    trace = InspectResponseTrace(artifacts=artifacts, max_requests=1)
    tools = InspectSandboxTools(
        artifacts=artifacts, trace=trace, max_executions=1, timeout_seconds=3, output_limit_bytes=16
    )
    benchmark = InspectBenchmark(
        objective_target=target,
        model_name="fixture",
        artifacts=artifacts,
        tools=tools,
        trace=trace,
        docker_profile=profile,
        episode_timeout_seconds=60,
    )
    with (
        patch.object(target, "_send_prompt_to_target_async", new_callable=AsyncMock, return_value=[]) as send,
        patch.object(profile, "verify_running_async", new_callable=AsyncMock),
        patch.object(profile, "verify_cleanup_async", new_callable=AsyncMock) as cleanup,
    ):
        with pytest.raises(ValueError, match="No authentic final response"):
            await benchmark.execute_async(binding=_binding(_task()))
    assert artifacts.manifest["grade_status"] == "not_available"
    assert tools.executions == []
    send.assert_awaited_once()
    cleanup.assert_awaited_once()


@pytest.mark.parametrize("kind", ["user", "empty", "truncated", "error"])
def test_final_output_validation_rejects_unusable_evidence(kind: str) -> None:
    piece = MessagePiece(
        role="user" if kind == "user" else "assistant", original_value="" if kind == "empty" else "fixture"
    )
    if kind == "truncated":
        piece.mark_as_truncated()
    if kind == "error":
        piece.response_error = "processing"
    with pytest.raises(ValueError):
        InspectBenchmark._final_piece(piece.to_message())
