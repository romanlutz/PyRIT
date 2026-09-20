# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import io
import json
import selectors
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from pyrit.executor.benchmark import _inspect_tool_worker
from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark._inspect_response_trace import InspectResponseTrace
from pyrit.executor.benchmark.inspect_sandbox import InspectDockerProfile, InspectSandboxTools

pytest.importorskip("inspect_ai")


@pytest.fixture
def artifacts(tmp_path: Path) -> InspectRunArtifacts:
    return InspectRunArtifacts(directory=tmp_path / "run", provenance={"test": True})


@pytest.fixture
def tools(artifacts: InspectRunArtifacts) -> InspectSandboxTools:
    return InspectSandboxTools(
        artifacts=artifacts, trace=None, max_executions=2, timeout_seconds=3, output_limit_bytes=16
    )


def _sandbox_result(*, cmd: list[str], **kwargs: Any) -> Any:
    from inspect_ai.util import ExecResult

    options = json.loads(kwargs["input"])
    payload = {
        "stdout": "fixture",
        "stderr": "",
        "returncode": 0,
        "timed_out": False,
        "truncated": False,
        "error": None,
        "execution_id": options["execution_id"],
    }
    return ExecResult(success=True, returncode=0, stdout=_framed_result(payload), stderr="")


def _framed_result(payload: dict[str, Any], *, termination_error: str | None = None) -> str:
    start = {"protocol": "pyrit-inspect-tool-v1", "event": "started", "execution_id": payload["execution_id"]}
    end = {**start, "event": "completed", "result": payload, "termination_error": termination_error}
    return json.dumps(start) + "\n" + json.dumps(end) + "\n"


async def test_tools_use_inspect_only_with_no_timeout_retry_async(tools: InspectSandboxTools) -> None:
    from inspect_ai.util import SandboxEnvironment

    environment = MagicMock(spec=SandboxEnvironment)
    environment.exec = AsyncMock(side_effect=_sandbox_result)
    with patch("inspect_ai.util.sandbox", return_value=environment) as select_sandbox:
        result = await tools.bash_async({"command": "printf fixture"})
    assert result["stdout"] == "fixture"
    assert result["execution_id"] != ""
    assert select_sandbox.call_args.args == ("default",)
    kwargs = environment.exec.call_args.kwargs
    assert kwargs["cwd"] == "/workspace"
    assert kwargs["timeout"] == 3
    assert kwargs["timeout_retry"] is False
    assert kwargs["cmd"][:4] == ["python", "-I", "-u", "-c"]
    assert json.loads(kwargs["input"])["command"] == ["bash", "-c", "printf fixture"]


@pytest.mark.parametrize("arguments", [{}, {"command": 1}, {"command": ""}, {"command": "true", "extra": "bad"}])
async def test_invalid_arguments_never_execute_async(tools: InspectSandboxTools, arguments: dict[str, Any]) -> None:
    with patch("inspect_ai.util.sandbox") as sandbox:
        with pytest.raises(ValueError, match="exactly one nonempty string"):
            await tools.bash_async(arguments)
    sandbox.assert_not_called()
    assert tools.executions == []


async def test_execution_budget_and_concurrency_fail_closed_async(tools: InspectSandboxTools) -> None:
    from inspect_ai.util import SandboxEnvironment

    environment = MagicMock(spec=SandboxEnvironment)
    entered, release = asyncio.Event(), asyncio.Event()

    async def execute_async(**kwargs: Any) -> Any:
        entered.set()
        await release.wait()
        return _sandbox_result(**kwargs)

    environment.exec = AsyncMock(side_effect=execute_async)
    with patch("inspect_ai.util.sandbox", return_value=environment):
        pending = asyncio.create_task(tools.python_async({"code": "print('fixture')"}))
        await entered.wait()
        with pytest.raises(RuntimeError, match="Sequential"):
            await tools.python_async({"code": "print('not run')"})
        release.set()
        await pending
        await tools.python_async({"code": "print('fixture')"})
        with pytest.raises(RuntimeError, match="budget"):
            await tools.python_async({"code": "print('not run')"})
    assert environment.exec.await_count == 2


async def test_transport_timeout_retains_unknown_not_fabricated_output_async(tools: InspectSandboxTools) -> None:
    from inspect_ai.util import SandboxEnvironment

    environment = MagicMock(spec=SandboxEnvironment)
    environment.exec = AsyncMock(side_effect=TimeoutError("daemon timeout"))
    with patch("inspect_ai.util.sandbox", return_value=environment):
        with pytest.raises(RuntimeError, match="capture is incomplete"):
            await tools.bash_async({"command": "true"})
    execution = tools.executions[0]
    assert execution["status"] == "error"
    assert execution["error_type"] == "TimeoutError"
    assert "result" not in execution


async def test_inner_timeout_is_terminal_and_keeps_partial_streams_async(tools: InspectSandboxTools) -> None:
    from inspect_ai.util import ExecResult, SandboxEnvironment

    def timeout_result(**kwargs: Any) -> Any:
        result = _sandbox_result(**kwargs)
        payload = json.loads(result.stdout.splitlines()[1])["result"]
        payload.update(stdout="partial", stderr="warning", returncode=None, timed_out=True, error="tool_timeout")
        return ExecResult(success=True, returncode=0, stdout=_framed_result(payload), stderr="")

    environment = MagicMock(spec=SandboxEnvironment)
    environment.exec = AsyncMock(side_effect=timeout_result)
    with patch("inspect_ai.util.sandbox", return_value=environment):
        with pytest.raises(RuntimeError, match="do not continue or grade"):
            await tools.bash_async({"command": "sleep 10"})
    assert tools.executions[0]["result"]["stdout"] == "partial"
    assert tools.executions[0]["result"]["stderr"] == "warning"
    assert tools.executions[0]["result"]["returncode"] is None


async def test_output_capture_contract_rejects_oversized_results_async(tools: InspectSandboxTools) -> None:
    from inspect_ai.util import ExecResult, SandboxEnvironment

    def oversized_result(**kwargs: Any) -> Any:
        result = _sandbox_result(**kwargs)
        payload = json.loads(result.stdout.splitlines()[1])["result"]
        payload["stdout"] = "x" * 17
        return ExecResult(success=True, returncode=0, stdout=_framed_result(payload), stderr="")

    environment = MagicMock(spec=SandboxEnvironment)
    environment.exec = AsyncMock(side_effect=oversized_result)
    with patch("inspect_ai.util.sandbox", return_value=environment):
        with pytest.raises(ValueError, match="byte limit"):
            await tools.bash_async({"command": "fixture"})


async def test_trace_correlates_real_call_without_logging_headers_async(artifacts: InspectRunArtifacts) -> None:
    trace = InspectResponseTrace(artifacts=artifacts, max_requests=1)
    request = httpx.Request(
        "POST",
        "https://example.invalid/responses",
        headers={"Authorization": "Bearer do-not-retain"},
        json={"parallel_tool_calls": False, "store": False},
    )
    await trace.request_async(request)
    await trace.response_async(
        httpx.Response(
            200,
            json={
                "id": "response-1",
                "output": [
                    {"type": "function_call", "name": "bash", "arguments": '{"command":"true"}', "call_id": "call-1"}
                ],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            },
        )
    )
    assert trace.claim_call(name="bash", arguments={"command": "true"}) == "call-1"
    with pytest.raises(ValueError, match="pending"):
        trace.claim_call(name="bash", arguments={"command": "true"})
    with pytest.raises(RuntimeError, match="budget"):
        await trace.request_async(request)
    assert trace.summary()["usage"]["total_tokens"] == 10
    journal = (artifacts.directory / "events.jsonl").read_text()
    assert "do-not-retain" not in journal
    assert "Authorization" not in journal


async def test_trace_rejects_multiple_calls_before_any_execution_async(artifacts: InspectRunArtifacts) -> None:
    trace = InspectResponseTrace(artifacts=artifacts, max_requests=2)
    call = {"type": "function_call", "name": "bash", "arguments": '{"command":"true"}', "call_id": "call-1"}
    with pytest.raises(ValueError, match="Multiple executable calls"):
        await trace.response_async(httpx.Response(200, json={"output": [call, {**call, "call_id": "call-2"}]}))
    assert trace.summary()["provider_call_ids"] == []
    assert "call-2" in (artifacts.directory / "events.jsonl").read_text()


async def test_trace_stops_provider_followup_after_terminal_error_async(artifacts: InspectRunArtifacts) -> None:
    trace = InspectResponseTrace(artifacts=artifacts, max_requests=2)
    trace.abort("tool_timeout")
    with pytest.raises(RuntimeError, match="cannot continue"):
        await trace.request_async(
            httpx.Request(
                "POST", "https://example.invalid/responses", json={"parallel_tool_calls": False, "store": False}
            )
        )
    assert trace.requests_observed == 0


async def test_compose_has_complete_profile_without_mounts_or_fixed_name_async(artifacts: InspectRunArtifacts) -> None:
    profile = InspectDockerProfile(artifacts=artifacts, image="python:test@sha256:123")
    await profile.write_compose_async()
    config = json.loads(profile.compose_file.read_text())
    service = config["services"]["default"]
    assert service["network_mode"] == "none"
    assert service["cpus"] == 1
    assert service["mem_limit"] == service["memswap_limit"] == "512m"
    assert service["pids_limit"] == 128
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert not {"volumes", "ports", "privileged", "devices", "container_name"}.intersection(service)


async def test_cleanup_reports_only_owned_resources_without_deleting_async(artifacts: InspectRunArtifacts) -> None:
    profile = InspectDockerProfile(artifacts=artifacts, image="python:test@sha256:123")
    with patch.object(profile, "_docker_async", new_callable=AsyncMock, return_value="owned-container\n") as docker:
        with pytest.raises(RuntimeError, match="left owned"):
            await profile.verify_cleanup_async()
    assert docker.call_args.args == ("ps", "-aq", "--filter", f"label=org.pyrit.inspect.run={artifacts.run_id}")
    assert artifacts.manifest["cleanup_status"] == "failed"


async def test_cleanup_query_error_does_not_claim_resource_absence_async(artifacts: InspectRunArtifacts) -> None:
    profile = InspectDockerProfile(artifacts=artifacts, image="python:test@sha256:123")
    with patch.object(profile, "_docker_async", new_callable=AsyncMock, side_effect=RuntimeError("daemon unavailable")):
        with pytest.raises(RuntimeError, match="daemon unavailable"):
            await profile.verify_cleanup_async()
    assert artifacts.manifest["cleanup_status"] == "unknown"
    assert artifacts.manifest["cleanup_error"] == "daemon unavailable"


@pytest.mark.parametrize("failure", ["launch", "missing_completion", "invalid_frame"])
async def test_control_plane_failures_are_not_command_feedback_async(tools: InspectSandboxTools, failure: str) -> None:
    from inspect_ai.util import ExecResult, SandboxEnvironment

    def failed_result(**kwargs: Any) -> Any:
        result = _sandbox_result(**kwargs)
        if failure == "launch":
            return ExecResult(success=False, returncode=125, stdout="", stderr="Docker could not start exec")
        if failure == "missing_completion":
            return ExecResult(success=True, returncode=0, stdout=result.stdout.splitlines()[0], stderr="")
        return ExecResult(success=True, returncode=0, stdout="invalid control output", stderr="")

    environment = MagicMock(spec=SandboxEnvironment)
    environment.exec = AsyncMock(side_effect=failed_result)
    with patch("inspect_ai.util.sandbox", return_value=environment):
        with pytest.raises((RuntimeError, ValueError)):
            await tools.bash_async({"command": "true"})
    assert "result" not in tools.executions[0]
    assert tools.executions[0]["confirmed_started"] is (failure == "missing_completion")
    assert tools.executions[0]["status"] == "error"
    assert "transport" in tools.executions[0]


async def test_timeout_result_survives_termination_failure_async(tools: InspectSandboxTools) -> None:
    from inspect_ai.util import ExecResult, SandboxEnvironment

    def failed_cleanup(**kwargs: Any) -> Any:
        result = _sandbox_result(**kwargs)
        payload = json.loads(result.stdout.splitlines()[1])["result"]
        payload.update(stdout="partial", returncode=None, timed_out=True, error="tool_timeout")
        stdout = _framed_result(payload, termination_error="fixture kill failed")
        return ExecResult(success=True, returncode=0, stdout=stdout, stderr="")

    environment = MagicMock(spec=SandboxEnvironment)
    environment.exec = AsyncMock(side_effect=failed_cleanup)
    with patch("inspect_ai.util.sandbox", return_value=environment):
        with pytest.raises(RuntimeError, match="do not continue"):
            await tools.bash_async({"command": "sleep 10"})
    execution = tools.executions[0]
    assert execution["result"]["stdout"] == "partial"
    assert execution["result"]["timed_out"] is True
    assert execution["termination_error"] == "fixture kill failed"


def test_worker_keeps_captured_bytes_when_kill_and_wait_fail() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.stdout, process.stderr = io.BytesIO(), io.BytesIO()
    process.wait.side_effect = subprocess.TimeoutExpired("fixture", 0.5)
    selector = MagicMock(spec=selectors.BaseSelector)
    selector.__enter__.return_value = selector
    selector.get_map.return_value = {1: "fixture"}
    key = selectors.SelectorKey(fileobj=process.stdout, fd=1, events=selectors.EVENT_READ, data="stdout")
    selector.select.return_value = [(key, selectors.EVENT_READ)]
    with (
        patch.object(_inspect_tool_worker.subprocess, "Popen", return_value=process),
        patch.object(_inspect_tool_worker.selectors, "DefaultSelector", return_value=selector),
        patch.object(_inspect_tool_worker.time, "monotonic", side_effect=[0, 0.1, 2.1]),
        patch.object(_inspect_tool_worker.os, "read", return_value=b"partial"),
        patch.object(_inspect_tool_worker, "_kill_group", return_value="fixture kill failed"),
    ):
        result, termination_error = _inspect_tool_worker._capture(
            command=["fixture"], timeout=2, limit=16, execution_id="fixture-execution"
        )
    assert result["stdout"] == "partial"
    assert result["timed_out"] is True
    assert result["returncode"] is None
    assert termination_error == "fixture kill failed"
    assert process.stdout.closed and process.stderr.closed


@pytest.fixture
def metadata_process() -> MagicMock:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.pid = 12345
    process.returncode = 0
    process.communicate = AsyncMock(return_value=(b"owned-container\n", b""))
    process.wait = AsyncMock(return_value=-9)
    return process


async def test_metadata_query_normal_completion_needs_no_kill_async(metadata_process: MagicMock) -> None:
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=metadata_process) as spawn:
        result = await InspectDockerProfile._docker_async("ps", "-aq")
    assert result == "owned-container\n"
    assert spawn.call_args.args == ("docker", "ps", "-aq")
    metadata_process.kill.assert_not_called()
    metadata_process.wait.assert_not_awaited()


@pytest.mark.parametrize("trigger", ["cancel", "timeout"])
async def test_blocked_metadata_query_kills_and_reaps_owned_process_async(
    metadata_process: MagicMock, trigger: str
) -> None:
    started = asyncio.Event()

    async def communicate_async() -> tuple[bytes, bytes]:
        started.set()
        await asyncio.Future()
        raise AssertionError("Blocked communication unexpectedly returned.")

    metadata_process.communicate.side_effect = communicate_async
    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=metadata_process),
        patch.object(InspectDockerProfile, "_METADATA_TIMEOUT_SECONDS", 0.02 if trigger == "timeout" else 30),
    ):
        task = asyncio.create_task(InspectDockerProfile._docker_async("inspect", "owned-container"))
        await started.wait()
        if trigger == "cancel":
            task.cancel("original cancellation")
            with pytest.raises(asyncio.CancelledError, match="original cancellation"):
                await task
        else:
            with pytest.raises(TimeoutError):
                await task
    metadata_process.kill.assert_called_once()
    metadata_process.wait.assert_awaited_once()


async def test_repeated_cancellation_does_not_abandon_metadata_reaper_async(metadata_process: MagicMock) -> None:
    communicating, reaping, exited = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def communicate_async() -> tuple[bytes, bytes]:
        communicating.set()
        await asyncio.Future()
        raise AssertionError("Blocked communication unexpectedly returned.")

    async def wait_async() -> int:
        reaping.set()
        await exited.wait()
        return -9

    metadata_process.communicate.side_effect = communicate_async
    metadata_process.wait.side_effect = wait_async
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=metadata_process):
        task = asyncio.create_task(InspectDockerProfile._docker_async("ps", "-aq"))
        await communicating.wait()
        task.cancel("first cancellation")
        await reaping.wait()
        task.cancel("second cancellation")
        await asyncio.sleep(0)
        assert not task.done()
        exited.set()
        with pytest.raises(asyncio.CancelledError, match="first cancellation"):
            await task
    metadata_process.kill.assert_called_once()
    metadata_process.wait.assert_awaited_once()


@pytest.mark.parametrize("cleanup_failure", ["kill", "wait", "wait_timeout", "already_exited"])
async def test_metadata_cleanup_failure_does_not_hide_cancellation_async(
    metadata_process: MagicMock, cleanup_failure: str, caplog: pytest.LogCaptureFixture
) -> None:
    original = asyncio.CancelledError("original cancellation")
    metadata_process.communicate.side_effect = original
    if cleanup_failure == "kill":
        metadata_process.kill.side_effect = OSError("kill denied")
    elif cleanup_failure == "already_exited":
        metadata_process.kill.side_effect = ProcessLookupError("already exited")
    elif cleanup_failure == "wait":
        metadata_process.wait.side_effect = OSError("wait failed")
    else:

        async def blocked_wait_async() -> int:
            await asyncio.Future()
            raise AssertionError("Blocked reaper unexpectedly returned.")

        metadata_process.wait.side_effect = blocked_wait_async
    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=metadata_process),
        patch.object(InspectDockerProfile, "_METADATA_REAP_TIMEOUT_SECONDS", 0.02),
    ):
        with pytest.raises(asyncio.CancelledError, match="original cancellation") as caught:
            await InspectDockerProfile._docker_async("ps", "-aq")
    assert caught.value is original
    metadata_process.kill.assert_called_once()
    metadata_process.wait.assert_awaited_once()
    if cleanup_failure != "already_exited":
        assert any("12345" in note for note in caught.value.__notes__)
        assert "Docker metadata subprocess 12345 cleanup:" in caplog.text
    else:
        assert "Docker metadata subprocess" not in caplog.text


async def test_metadata_query_nonzero_exit_remains_an_explicit_error_async(metadata_process: MagicMock) -> None:
    metadata_process.returncode = 1
    metadata_process.communicate.return_value = (b"", b"daemon query failed")
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=metadata_process):
        with pytest.raises(RuntimeError, match="daemon query failed"):
            await InspectDockerProfile._docker_async("ps", "-aq")
    metadata_process.kill.assert_not_called()
    metadata_process.wait.assert_not_awaited()
