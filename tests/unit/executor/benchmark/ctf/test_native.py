# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import httpx
import pytest
from azure.identity.aio import DefaultAzureCredential

from pyrit.executor.benchmark.ctf.__main__ import _evaluate_async, _run_async
from pyrit.executor.benchmark.ctf.docker_environment import CommandResult, DockerCommandError, DockerCTFEnvironment
from pyrit.executor.benchmark.ctf.evidence import DockerToolHarness, RunEvidence
from pyrit.executor.benchmark.ctf.gdm_intercode import CTFTask, GDMIntercodeTask4
from pyrit.executor.benchmark.ctf.native import NativeCTFBenchmark
from pyrit.models import Message, Score, SeedPrompt
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import OpenAIResponseTarget, PromptTarget
from pyrit.score import IncludesScorer
from tests.unit.executor.benchmark.ctf.mocks import mock_docker_exec

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _provider_response(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": f"resp_{uuid4().hex}",
        "object": "response",
        "created_at": 1234567890,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "model": "fixture",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


async def test_real_responses_target_loop_persists_correlated_tools_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    run_id = str(uuid4())
    evidence = RunEvidence(directory=tmp_path, run_id=run_id)
    environment = MagicMock(spec=DockerCTFEnvironment)
    environment.container_id = "a" * 64
    envelope = {
        "stdout": "picoCTF{synthetic_fixture}\n",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "truncated": False,
        "error": None,
        "execution_id": "fixture-execution",
    }
    environment.execute_async = AsyncMock(return_value=envelope)
    harness = DockerToolHarness(environment=environment, evidence=evidence)
    bodies = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if len(bodies) == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_fixture",
                    "call_id": "call_fixture",
                    "name": "bash",
                    "arguments": '{"command":"cat flag"}',
                    "status": "completed",
                }
            ]
        else:
            output = [
                {
                    "type": "message",
                    "id": "msg_fixture",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "picoCTF{synthetic_fixture}", "annotations": []}],
                }
            ]
        return httpx.Response(200, json=_provider_response(output), headers={"x-request-id": f"request-{len(bodies)}"})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks={"request": [evidence.on_request_async], "response": [evidence.on_response_async]},
    ) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.openai.azure.com/openai/v1",
            model_name="fixture",
            api_key="fixture-not-a-secret",
            custom_functions={"bash": harness.bash_async, "python": harness.python_async},
            max_output_tokens=2048,
            fail_on_missing_function=True,
            extra_body_parameters={
                "tools": GDMIntercodeTask4.tool_definitions(),
                "parallel_tool_calls": False,
                "store": False,
            },
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        benchmark = NativeCTFBenchmark(
            objective_target=target, scorer=IncludesScorer(expected="picoCTF{synthetic_fixture}")
        )
        response = await benchmark.send_async(
            seed=SeedPrompt(value="Find the fixture flag.", data_type="text", role="user"),
            system_prompt=GDMIntercodeTask4.SYSTEM_PROMPT,
            conversation_id=run_id,
            run_id=run_id,
        )
        score = await benchmark.score_async(response)
    assert score.get_value() is True
    assert len(bodies) == 2
    assert all(body["parallel_tool_calls"] is False and body["max_output_tokens"] == 2048 for body in bodies)
    assert "picoCTF{synthetic_fixture}" not in json.dumps(bodies[0])
    assert bodies[1]["input"][-1]["type"] == "function_call_output"
    assert bodies[1]["input"][-1]["call_id"] == "call_fixture"
    environment.execute_async.assert_awaited_once()
    entry = evidence.state["tool_executions"][0]
    assert entry["call_id"] == "call_fixture" and entry["execution_id"] == "fixture-execution"
    assert evidence.state["model_requests"] == 2
    assert [item["usage"]["total_tokens"] for item in evidence.state["model_responses"]] == [15, 15]
    assert "fixture-not-a-secret" not in (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    messages = sqlite_instance.get_conversation_messages(conversation_id=run_id)
    pieces = [piece for message in messages for piece in message.message_pieces]
    assert [piece.converted_value_data_type for piece in pieces] == [
        "text",
        "text",
        "function_call",
        "function_call_output",
        "text",
    ]
    calls = [
        json.loads(piece.converted_value) for piece in pieces if piece.converted_value_data_type == "function_call"
    ]
    assert calls[0]["call_id"] == entry["call_id"]
    assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].message_piece_id == response.get_piece().id


@pytest.mark.parametrize("kind", ["request_echo", "empty", "error", "truncated", "tool", "simulated"])
async def test_no_fabricated_final_answer_or_grade_async(*, kind: str, sqlite_instance: SQLiteMemory) -> None:
    role = "user" if kind == "request_echo" else "assistant"
    response = Message.from_prompt(prompt="" if kind == "empty" else "untrusted fixture", role=role)
    if kind == "error":
        response.get_piece().response_error = "processing"
    if kind == "truncated":
        response.get_piece().mark_as_truncated()
    if kind == "tool":
        response.get_piece().converted_value_data_type = "function_call"
    if kind == "simulated":
        response.set_simulated_role()
    normalizer = MagicMock(spec=PromptNormalizer)
    normalizer.send_prompt_async = AsyncMock(return_value=response)
    target = MagicMock(spec=PromptTarget)
    benchmark = NativeCTFBenchmark(
        objective_target=target, scorer=IncludesScorer(expected="untrusted fixture"), prompt_normalizer=normalizer
    )
    with pytest.raises(ValueError):
        await benchmark.send_async(
            seed=SeedPrompt(value="Find the flag", data_type="text", role="user"),
            system_prompt="fixture",
            conversation_id=str(uuid4()),
            run_id=str(uuid4()),
        )
    assert not sqlite_instance.get_scores(score_type="true_false")


async def test_parallel_provider_calls_are_retained_then_rejected_async(tmp_path: Path) -> None:
    evidence = RunEvidence(directory=tmp_path, run_id=str(uuid4()))
    request = httpx.Request("POST", "https://fixture.invalid/responses", json={"model": "fixture"})
    await evidence.on_request_async(request)
    calls = [
        {
            "type": "function_call",
            "name": "bash",
            "call_id": f"call-{index}",
            "arguments": '{"command":"pwd"}',
        }
        for index in range(2)
    ]
    response = httpx.Response(200, request=request, json=_provider_response(calls))
    with pytest.raises(RuntimeError, match="Parallel"):
        await evidence.on_response_async(response)
    events = (tmp_path / "events.jsonl").read_text(encoding="utf-8")
    assert "call-0" in events and "call-1" in events
    assert not evidence.state["tool_executions"]


async def test_callback_without_model_call_cannot_execute_async(tmp_path: Path) -> None:
    evidence = RunEvidence(directory=tmp_path, run_id=str(uuid4()))
    environment = MagicMock(spec=DockerCTFEnvironment)
    environment.execute_async = AsyncMock()
    harness = DockerToolHarness(environment=environment, evidence=evidence)
    with pytest.raises(RuntimeError, match="provider call"):
        await harness.bash_async({"command": "pwd"})
    environment.execute_async.assert_not_called()


async def test_partial_tool_failure_preserves_provider_identity_async(tmp_path: Path) -> None:
    evidence = RunEvidence(directory=tmp_path, run_id=str(uuid4()))
    request = httpx.Request("POST", "https://fixture.invalid/responses", json={"model": "fixture"})
    await evidence.on_request_async(request)
    call = {"type": "function_call", "name": "python", "call_id": "call-partial", "arguments": '{"code":"print(1)"}'}
    await evidence.on_response_async(httpx.Response(200, request=request, json=_provider_response([call])))
    environment = MagicMock(spec=DockerCTFEnvironment)
    environment.container_id = "b" * 64
    environment.execute_async = AsyncMock(side_effect=RuntimeError("Docker unavailable"))
    harness = DockerToolHarness(environment=environment, evidence=evidence)
    with pytest.raises(RuntimeError, match="Docker unavailable"):
        await harness.python_async({"code": "print(1)"})
    state = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert state["tool_executions"][0]["call_id"] == "call-partial"
    assert state["tool_executions"][0]["status"] == "requested"
    assert "result" not in state["tool_executions"][0]
    assert state["final_output"] is None and state["raw_grade"] is None


async def test_timeout_stops_real_target_loop_without_a_second_request_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    run_id = str(uuid4())
    evidence = RunEvidence(directory=tmp_path, run_id=run_id)
    environment = MagicMock(spec=DockerCTFEnvironment)
    environment.container_id = "a" * 64
    environment.execute_async = AsyncMock(
        return_value={
            "stdout": "partial output",
            "stderr": "",
            "returncode": None,
            "timed_out": True,
            "truncated": False,
            "error": "tool_timeout",
            "execution_id": "fixture-timeout",
        }
    )
    harness = DockerToolHarness(environment=environment, evidence=evidence)
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        assert len(calls) == 1, "A terminal tool timeout must not trigger another model request."
        output = [
            {
                "type": "function_call",
                "id": "fc_timeout",
                "call_id": "call_timeout",
                "name": "bash",
                "arguments": '{"command":"sleep 99"}',
                "status": "completed",
            }
        ]
        return httpx.Response(200, json=_provider_response(output))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks={"request": [evidence.on_request_async], "response": [evidence.on_response_async]},
    ) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.openai.azure.com/openai/v1",
            model_name="fixture",
            api_key="fixture",
            custom_functions={"bash": harness.bash_async},
            fail_on_missing_function=True,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        benchmark = NativeCTFBenchmark(objective_target=target, scorer=IncludesScorer(expected="fixture"))
        with pytest.raises(Exception, match="Error sending prompt"):
            await benchmark.send_async(
                seed=SeedPrompt(value="fixture", data_type="text", role="user"),
                system_prompt="fixture",
                conversation_id=run_id,
                run_id=run_id,
            )
    assert len(calls) == 1
    environment.execute_async.assert_awaited_once()
    assert evidence.state["terminal_error"] == "tool_timeout"
    assert evidence.state["tool_executions"][0]["status"] == "timed_out"
    assert evidence.state["tool_executions"][0]["result"]["stdout"] == "partial output"
    assert not sqlite_instance.get_scores(score_type="true_false")
    assert evidence.state["raw_grade"] is None and evidence.state["final_output"] is None
    with pytest.raises(RuntimeError, match="terminal"):
        await evidence.on_request_async(httpx.Request("POST", "https://fixture.invalid/responses", json={}))
    assert evidence.state["model_requests"] == 1


@pytest.mark.parametrize("outcome", ["success", "error", "cancelled"])
async def test_credential_closes_on_every_outcome_async(*, tmp_path: Path, outcome: str) -> None:
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True, exclude_broker_credential=True)
    provider = AsyncMock(return_value="fixture-token")
    task = CTFTask(
        seed=SeedPrompt(value="fixture", role="user", data_type="text"),
        files={},
        expected="grading-only",
        provenance={},
    )
    evidence = RunEvidence(directory=tmp_path, run_id=str(uuid4()))
    evidence.state["tool_executions"] = [{"status": "completed", "call_id": "fixture"}]
    environment = MagicMock(spec=DockerCTFEnvironment)
    environment.executions = 1
    environment.timed_out = False
    environment.terminal_error = None
    response = Message.from_prompt(prompt="grading-only", role="assistant")
    score = Score(score_type="true_false", score_value="true", score_metadata={"value": "C"})
    failure = {"success": None, "error": RuntimeError("send failed"), "cancelled": asyncio.CancelledError()}[outcome]
    with (
        patch("azure.identity.aio.DefaultAzureCredential", return_value=credential) as make_credential,
        patch("pyrit.auth.get_azure_openai_auth", return_value=provider) as make_provider,
        patch.object(credential, "close", new_callable=AsyncMock, wraps=credential.close) as close,
        patch.object(
            NativeCTFBenchmark, "send_async", new_callable=AsyncMock, side_effect=failure, return_value=response
        ),
        patch.object(NativeCTFBenchmark, "score_async", new_callable=AsyncMock, return_value=score),
    ):
        if failure is None:
            await _evaluate_async(
                endpoint="https://fixture.openai.azure.com/openai/v1",
                model="fixture",
                task=task,
                environment=environment,
                evidence=evidence,
            )
            assert evidence.state["status"] == "completed"
        else:
            with pytest.raises(type(failure)):
                await _evaluate_async(
                    endpoint="https://fixture.openai.azure.com/openai/v1",
                    model="fixture",
                    task=task,
                    environment=environment,
                    evidence=evidence,
                )
    close.assert_awaited_once()
    assert make_credential.call_args.kwargs == {
        "exclude_interactive_browser_credential": True,
        "exclude_broker_credential": True,
    }
    assert make_provider.call_args.kwargs["credential"] is credential
    provider.assert_not_called()


async def test_setup_failure_retains_error_without_answer_or_grade_async(tmp_path: Path) -> None:
    run_id = str(uuid4())
    with (
        patch.object(
            GDMIntercodeTask4, "load_async", new_callable=AsyncMock, side_effect=ValueError("source mismatch")
        ),
        patch(
            "pyrit.executor.benchmark.ctf.__main__._git_commit_async", new_callable=AsyncMock, return_value="fixture"
        ),
    ):
        with pytest.raises(ValueError, match="source mismatch"):
            await _run_async(
                endpoint="https://fixture.openai.azure.com/openai/v1",
                model="fixture",
                directory=tmp_path,
                run_id=run_id,
            )
    state = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert state["status"] == "error"
    assert state["errors"] == [{"type": "ValueError", "message": "source mismatch"}]
    assert state["cleanup_status"] == "not_created"
    assert state["memory_export_status"] == "not_initialized"
    assert state["model_requests"] == 0 and state["raw_grade"] is None and state["final_output"] is None


@pytest.mark.parametrize("failure", ["stopped", "daemon_lost", "invalid_receipt", "timeout_kill_failed"])
async def test_docker_failure_terminates_real_responses_loop_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, failure: str
) -> None:
    run_id = str(uuid4())
    evidence = RunEvidence(directory=tmp_path, run_id=run_id)
    environment = DockerCTFEnvironment(image=GDMIntercodeTask4.IMAGE, run_id=run_id)
    environment.container_id = "a" * 64
    environment.cleanup_status = "pending"
    harness = DockerToolHarness(environment=environment, evidence=evidence)
    model_requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        model_requests.append(json.loads(request.content))
        assert len(model_requests) == 1, "No model continuation is allowed after Docker fails."
        return httpx.Response(
            200,
            json=_provider_response(
                [
                    {
                        "type": "function_call",
                        "id": "fc_failure",
                        "call_id": "call_failure",
                        "name": "bash",
                        "arguments": '{"command":"fixture command"}',
                        "status": "completed",
                    }
                ]
            ),
        )

    async def failed_termination_async(*, arguments: list[str], **kwargs: Any) -> CommandResult:
        assert arguments == ["kill", environment.container_id]
        journal = json.loads(await asyncio.to_thread((tmp_path / "manifest.json").read_text, encoding="utf-8"))
        entry = journal["tool_executions"][0]
        assert journal["terminal_error"] == environment.terminal_error
        if failure == "timeout_kill_failed":
            assert entry["result"]["stdout"] == "captured before timeout"
            assert entry["result"]["stderr"] == "partial stderr"
            assert entry["result"]["timed_out"] is True
        raise DockerCommandError("container exited at boundary or daemon unavailable")

    process = AsyncMock(
        return_value={
            "stopped": CommandResult(stdout="", stderr="Container is not running", returncode=1),
            "daemon_lost": CommandResult(stdout="", stderr="Cannot connect to daemon", returncode=125),
            "invalid_receipt": CommandResult(stdout="not a runner result", stderr="", returncode=0),
        }.get(failure)
    )
    if failure == "timeout_kill_failed":
        process.side_effect = mock_docker_exec(
            CommandResult(stdout="captured before timeout", stderr="partial stderr", returncode=None, timed_out=True)
        )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks={"request": [evidence.on_request_async], "response": [evidence.on_response_async]},
    ) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.openai.azure.com/openai/v1",
            model_name="fixture",
            api_key="fixture",
            custom_functions={"bash": harness.bash_async},
            fail_on_missing_function=True,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        benchmark = NativeCTFBenchmark(objective_target=target, scorer=IncludesScorer(expected="fixture"))
        with (
            patch("pyrit.executor.benchmark.ctf.docker_environment.run_process_async", process),
            patch.object(environment, "_control_async", new_callable=AsyncMock, side_effect=failed_termination_async),
        ):
            with pytest.raises(Exception, match="Error sending prompt"):
                response = await benchmark.send_async(
                    seed=SeedPrompt(value="fixture", data_type="text", role="user"),
                    system_prompt="fixture",
                    conversation_id=run_id,
                    run_id=run_id,
                )
                await benchmark.score_async(response)
    assert len(model_requests) == 1
    assert environment.execution_attempts == 1
    assert environment.executions == (1 if failure == "timeout_kill_failed" else 0)
    assert not sqlite_instance.get_scores(score_type="true_false")
    messages = sqlite_instance.get_conversation_messages(conversation_id=run_id)
    assert not any(
        piece.role == "assistant" and piece.original_value_data_type == "text"
        for message in messages
        for piece in message.message_pieces
    )
    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    entry = saved["tool_executions"][0]
    assert entry["execution_started_confirmed"] is (failure == "timeout_kill_failed")
    assert entry["termination_error"]["message"] == "container exited at boundary or daemon unavailable"
    assert saved["final_output"] is None and saved["raw_grade"] is None
    if failure == "timeout_kill_failed":
        assert entry["result"]["stdout"] == "captured before timeout"
        assert entry["result"]["stderr"] == "partial stderr"
        assert entry["status"] == "timed_out"
    else:
        assert entry["status"] == "execution_failed" and "result" not in entry
        assert entry["diagnostics"]["transport"]["returncode"] == process.return_value.returncode
    with pytest.raises(RuntimeError, match="terminal"):
        await evidence.on_request_async(httpx.Request("POST", "https://fixture.invalid/responses", json={}))
    assert evidence.state["model_requests"] == 1


async def test_ordinary_nonzero_allows_real_target_recovery_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    run_id = str(uuid4())
    evidence = RunEvidence(directory=tmp_path, run_id=run_id)
    environment = DockerCTFEnvironment(image=GDMIntercodeTask4.IMAGE, run_id=run_id)
    environment.container_id = "a" * 64
    environment.cleanup_status = "pending"
    harness = DockerToolHarness(environment=environment, evidence=evidence)
    model_requests: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        model_requests.append(body)
        if len(model_requests) == 1:
            output = [
                {
                    "type": "function_call",
                    "id": "fc_nonzero",
                    "call_id": "call_nonzero",
                    "name": "bash",
                    "arguments": '{"command":"fixture command"}',
                    "status": "completed",
                }
            ]
        else:
            result = json.loads(body["input"][-1]["output"])
            assert result["returncode"] == 127 and result["error"] == "nonzero_exit"
            output = [
                {
                    "type": "message",
                    "id": "msg_fixture",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "fixture", "annotations": []}],
                }
            ]
        return httpx.Response(200, json=_provider_response(output))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond),
        event_hooks={"request": [evidence.on_request_async], "response": [evidence.on_response_async]},
    ) as client:
        target = OpenAIResponseTarget(
            endpoint="https://fixture.openai.azure.com/openai/v1",
            model_name="fixture",
            api_key="fixture",
            custom_functions={"bash": harness.bash_async},
            fail_on_missing_function=True,
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        benchmark = NativeCTFBenchmark(objective_target=target, scorer=IncludesScorer(expected="fixture"))
        with patch(
            "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
            new_callable=AsyncMock,
            side_effect=mock_docker_exec(
                CommandResult(stdout="", stderr="Error response from daemon: Container is not running", returncode=127)
            ),
        ):
            response = await benchmark.send_async(
                seed=SeedPrompt(value="fixture", data_type="text", role="user"),
                system_prompt="fixture",
                conversation_id=run_id,
                run_id=run_id,
            )
            score = await benchmark.score_async(response)
    assert len(model_requests) == 2 and environment.executions == 1
    assert environment.terminal_error is None
    assert score.get_value() is True
    assert sqlite_instance.get_scores(score_ids=[str(score.id)])


async def test_terminal_environment_cannot_be_graded_even_if_send_returns_async(tmp_path: Path) -> None:
    evidence = RunEvidence(directory=tmp_path, run_id=str(uuid4()))
    task = CTFTask(
        seed=SeedPrompt(value="fixture", data_type="text", role="user"), files={}, expected="fixture", provenance={}
    )
    environment = MagicMock(spec=DockerCTFEnvironment)
    environment.terminal_error = "docker_execution_error"
    with (
        patch("pyrit.auth.get_azure_openai_auth", return_value=AsyncMock(return_value="fixture-token")),
        patch.object(
            NativeCTFBenchmark,
            "send_async",
            new_callable=AsyncMock,
            return_value=Message.from_prompt(prompt="fixture", role="assistant"),
        ),
        patch.object(NativeCTFBenchmark, "score_async", new_callable=AsyncMock) as score,
    ):
        with pytest.raises(RuntimeError, match="terminal Docker"):
            await _evaluate_async(
                endpoint="https://fixture.openai.azure.com/openai/v1",
                model="fixture",
                task=task,
                environment=environment,
                evidence=evidence,
            )
    score.assert_not_awaited()
    assert evidence.state["final_output"] is None and evidence.state["raw_grade"] is None


async def test_outer_cleanup_failure_preserves_timeout_evidence_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    run_id = str(uuid4())
    task = CTFTask(
        seed=SeedPrompt(value="fixture", data_type="text", role="user"), files={}, expected="fixture", provenance={}
    )
    environment = DockerCTFEnvironment(image=GDMIntercodeTask4.IMAGE, run_id=run_id)
    environment.container_id = "a" * 64
    environment.cleanup_status = "pending"
    environment._create_started = True

    async def evaluate_timeout_async(*, evidence: RunEvidence, **kwargs: Any) -> None:
        request = httpx.Request("POST", "https://fixture.invalid/responses", json={"model": "fixture"})
        await evidence.on_request_async(request)
        call = {
            "type": "function_call",
            "name": "bash",
            "call_id": "call_cleanup",
            "arguments": '{"command":"fixture"}',
        }
        await evidence.on_response_async(httpx.Response(200, request=request, json=_provider_response([call])))
        harness = DockerToolHarness(environment=environment, evidence=evidence)
        await harness.bash_async({"command": "fixture"})

    with (
        patch("pyrit.executor.benchmark.ctf.docker_environment.DockerCTFEnvironment", return_value=environment),
        patch.object(GDMIntercodeTask4, "load_async", new_callable=AsyncMock, return_value=task),
        patch.object(environment, "acquire_image_async", new_callable=AsyncMock),
        patch.object(environment, "start_async", new_callable=AsyncMock),
        patch.object(
            environment, "_control_async", new_callable=AsyncMock, side_effect=DockerCommandError("daemon unavailable")
        ),
        patch(
            "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
            new_callable=AsyncMock,
            side_effect=mock_docker_exec(
                CommandResult(
                    stdout="retained timeout stdout", stderr="retained timeout stderr", returncode=None, timed_out=True
                )
            ),
        ),
        patch(
            "pyrit.executor.benchmark.ctf.__main__._git_commit_async", new_callable=AsyncMock, return_value="fixture"
        ),
        patch("pyrit.executor.benchmark.ctf.__main__._create_memory", return_value=sqlite_instance),
        patch("pyrit.executor.benchmark.ctf.__main__._evaluate_async", side_effect=evaluate_timeout_async),
    ):
        with pytest.raises(DockerCommandError, match="daemon unavailable"):
            await _run_async(
                endpoint="https://fixture.openai.azure.com/openai/v1",
                model="fixture",
                directory=tmp_path,
                run_id=run_id,
            )
    state = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert state["status"] == "error"
    assert state["cleanup_status"] == "failed"
    assert state["cleanup_errors"][0]["message"] == "daemon unavailable"
    assert state["tool_executions"][0]["termination_error"]["message"] == "daemon unavailable"
    assert state["tool_executions"][0]["result"]["stdout"] == "retained timeout stdout"
    assert state["tool_executions"][0]["result"]["stderr"] == "retained timeout stderr"
    assert state["tool_execution_attempts"] == state["confirmed_tool_executions"] == 1
    assert state["final_output"] is None and state["raw_grade"] is None
