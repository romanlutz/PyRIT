# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from pyrit.executor.benchmark.ctf.docker_environment import (
    CommandResult,
    DockerCommandError,
    DockerCTFEnvironment,
    DockerExecutionError,
    run_process_async,
)
from pyrit.executor.benchmark.ctf.gdm_intercode import GDMIntercodeTask4
from tests.unit.executor.benchmark.ctf.mocks import mock_docker_exec, runner_transport


def _environment(**kwargs: Any) -> DockerCTFEnvironment:
    return DockerCTFEnvironment(image=GDMIntercodeTask4.IMAGE, run_id=str(uuid4()), **kwargs)


def _running_environment(**kwargs: Any) -> DockerCTFEnvironment:
    environment = _environment(**kwargs)
    environment.container_id = "a" * 64
    environment.cleanup_status = "pending"
    environment._create_started = True
    return environment


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cpus": 2},
        {"memory_bytes": 0},
        {"pids_limit": 129},
        {"max_tool_executions": 9},
        {"tool_timeout_seconds": 31},
        {"platform": "windows/amd64"},
    ],
)
def test_reject_unsupported_resources(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _environment(**kwargs)


def test_reject_mutable_image() -> None:
    with pytest.raises(ValueError, match="digest"):
        DockerCTFEnvironment(image="python:3.12-slim", run_id=str(uuid4()))


@pytest.mark.parametrize(("name", "key"), [("bash", "command"), ("python", "code")])
async def test_commands_are_data_in_docker_argv_async(*, name: str, key: str) -> None:
    environment = _running_environment()
    payload = 'echo "quoted"; $(host_command) &\nnext line'
    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        side_effect=mock_docker_exec(CommandResult(stdout="output", stderr="", returncode=0)),
    ) as run:
        result = await environment.execute_async(name=name, arguments={key: payload})
    argv = run.call_args.kwargs["arguments"]
    assert argv[:5] == ["docker", "exec", "--workdir", "/workspace", environment.container_id]
    assert json.loads(argv[-3])[-1] == payload
    assert argv[5:9] == ["python", "-I", "-u", "-c"]
    assert float(argv[-2]) == 29 and int(argv[-1]) == 16384
    assert result["stdout"] == "output"
    assert result["error"] is None
    assert set(result) == {"stdout", "stderr", "returncode", "timed_out", "truncated", "error", "execution_id"}


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("host_shell", {"command": "pwd"}),
        ("bash", {"code": "pwd"}),
        ("bash", {"command": 1}),
        ("bash", {"command": "pwd", "extra": True}),
        ("python", {"code": "\x00"}),
        ("python", {"code": ""}),
    ],
)
async def test_invalid_tool_never_executes_async(*, name: str, arguments: dict[str, Any]) -> None:
    environment = _running_environment()
    with patch("pyrit.executor.benchmark.ctf.docker_environment.run_process_async", new_callable=AsyncMock) as run:
        with pytest.raises(ValueError):
            await environment.execute_async(name=name, arguments=arguments)
    run.assert_not_called()


async def test_tool_execution_budget_is_hard_async() -> None:
    environment = _running_environment(max_tool_executions=1)
    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        side_effect=mock_docker_exec(CommandResult(stdout="", stderr="", returncode=0)),
    ) as run:
        await environment.execute_async(name="bash", arguments={"command": "pwd"})
        with pytest.raises(RuntimeError, match="budget"):
            await environment.execute_async(name="bash", arguments={"command": "pwd"})
    run.assert_awaited_once()
    assert environment.execution_attempts == environment.executions == 1


@pytest.mark.parametrize(
    ("process_result", "error"),
    [
        (CommandResult(stdout="partial", stderr="", returncode=None, timed_out=True), "tool_timeout"),
        (CommandResult(stdout="prefix", stderr="", returncode=0, truncated=True), "output_truncated"),
        (CommandResult(stdout="", stderr="bad command", returncode=2), "nonzero_exit"),
    ],
)
async def test_tool_errors_are_explicit_async(*, process_result: CommandResult, error: str) -> None:
    environment = _running_environment()
    with (
        patch(
            "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
            new_callable=AsyncMock,
            side_effect=mock_docker_exec(process_result),
        ),
        patch.object(environment, "_control_async", new_callable=AsyncMock) as control,
    ):
        if process_result.timed_out:
            with pytest.raises(DockerExecutionError) as caught:
                await environment.execute_async(name="bash", arguments={"command": "pwd"})
            result = caught.value.result
            assert result is not None
            control.assert_not_awaited()
            await environment.terminate_async()
        else:
            result = await environment.execute_async(name="bash", arguments={"command": "pwd"})
    assert result["error"] == error
    if process_result.timed_out:
        assert control.call_args.kwargs["arguments"] == ["kill", environment.container_id]
        assert environment.timed_out


@pytest.mark.parametrize(
    "transport",
    [
        CommandResult(stdout="", stderr="Error response from daemon: container is not running", returncode=1),
        CommandResult(stdout="", stderr="connection to daemon lost", returncode=125),
        CommandResult(stdout="not a runner receipt", stderr="", returncode=0),
        CommandResult(stdout='{"event":"started","execution_id":"wrong","pid":42}\n', stderr="", returncode=0),
    ],
)
async def test_rejected_exec_is_terminal_and_not_counted_async(transport: CommandResult) -> None:
    environment = _running_environment()
    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        return_value=transport,
    ) as run:
        with pytest.raises(DockerExecutionError) as caught:
            await environment.execute_async(name="bash", arguments={"command": "pwd"})
        with pytest.raises(RuntimeError, match="not available"):
            await environment.execute_async(name="bash", arguments={"command": "pwd"})
    assert environment.executions == 0 and environment.execution_attempts == 1
    assert environment.terminal_error == "docker_execution_error"
    assert caught.value.result is None
    assert caught.value.execution_started is False
    assert caught.value.diagnostics["transport"]["stderr"] == transport.stderr
    run.assert_awaited_once()


@pytest.mark.parametrize("returncode", [1, 125, 126, 127, 137])
async def test_command_nonzero_is_distinct_from_docker_failure_async(returncode: int) -> None:
    environment = _running_environment()
    command_result = CommandResult(
        stdout='{"event":"launch_error","message":"fake control record"}',
        stderr="Error response from daemon: container is not running",
        returncode=returncode,
    )
    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        side_effect=mock_docker_exec(command_result),
    ):
        result = await environment.execute_async(name="bash", arguments={"command": "fixture"})
    assert result["error"] == "nonzero_exit" and result["returncode"] == returncode
    assert result["stdout"] == command_result.stdout and result["stderr"] == command_result.stderr
    assert environment.executions == 1 and environment.terminal_error is None


async def test_lost_transport_retains_confirmed_partial_execution_async() -> None:
    environment = _running_environment()

    def lose_connection(*, arguments: list[str], **kwargs: Any) -> CommandResult:
        return runner_transport(
            execution_id=arguments[-4],
            result=CommandResult(stdout="partial", stderr="diagnostic", returncode=None),
            finished=False,
            transport_exit=137,
        )

    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        side_effect=lose_connection,
    ):
        with pytest.raises(DockerExecutionError) as caught:
            await environment.execute_async(name="python", arguments={"code": "fixture"})
    assert caught.value.result["stdout"] == "partial"
    assert caught.value.result["stderr"] == "diagnostic"
    assert caught.value.result["returncode"] is None
    assert caught.value.result["error"] == "docker_execution_error"
    assert environment.executions == 1
    assert caught.value.diagnostics["completion_confirmed"] is False


async def test_failed_cleanup_does_not_claim_removal_async() -> None:
    environment = _running_environment()
    with patch.object(
        environment, "_control_async", new_callable=AsyncMock, side_effect=DockerCommandError("daemon unavailable")
    ):
        with pytest.raises(DockerCommandError, match="daemon unavailable"):
            await environment.cleanup_async()
    assert environment.cleanup_status == "failed"


async def test_outer_transport_timeout_retains_received_chunks_async() -> None:
    environment = _running_environment()

    def timeout_transport(*, arguments: list[str], **kwargs: Any) -> CommandResult:
        received = runner_transport(
            execution_id=arguments[-4],
            result=CommandResult(stdout="received partial", stderr="received error", returncode=None),
            finished=False,
        )
        return CommandResult(
            stdout=received.stdout + '{"event":',
            stderr="",
            returncode=None,
            timed_out=True,
        )

    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        side_effect=timeout_transport,
    ):
        with pytest.raises(DockerExecutionError) as caught:
            await environment.execute_async(name="python", arguments={"code": "fixture"})
    assert caught.value.reason == "tool_timeout"
    assert caught.value.result["stdout"] == "received partial"
    assert caught.value.result["stderr"] == "received error"
    assert caught.value.result["returncode"] is None
    assert caught.value.result["timed_out"] is True
    assert caught.value.diagnostics["protocol_error"]
    assert caught.value.diagnostics["completion_confirmed"] is False


async def test_missing_docker_executable_does_not_demonstrate_execution_async() -> None:
    environment = _running_environment()
    with patch(
        "pyrit.executor.benchmark.ctf.docker_environment.run_process_async",
        new_callable=AsyncMock,
        side_effect=FileNotFoundError("Docker CLI unavailable"),
    ):
        with pytest.raises(DockerExecutionError) as caught:
            await environment.execute_async(name="bash", arguments={"command": "fixture"})
    assert not caught.value.execution_started and caught.value.result is None
    assert caught.value.diagnostics["transport_error"]["type"] == "FileNotFoundError"
    assert environment.execution_attempts == 1 and environment.executions == 0


async def test_process_capture_is_bounded_while_draining_async() -> None:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = 0
    process.wait = AsyncMock(return_value=0)
    for name, content in (("stdout", b"a" * 50000), ("stderr", b"b" * 50000)):
        stream = asyncio.StreamReader()
        stream.feed_data(content)
        stream.feed_eof()
        setattr(process, name, stream)
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process) as create:
        result = await run_process_async(arguments=["docker", "version"], timeout=1, output_limit=16384)
    assert len(result.stdout.encode("utf-8")) == 16384
    assert len(result.stderr.encode("utf-8")) == 16384
    assert result.truncated
    assert process.stdout.at_eof() and process.stderr.at_eof()
    assert "shell" not in create.call_args.kwargs


async def test_cleanup_refuses_wrong_owner_async() -> None:
    environment = _running_environment()
    container = {"Id": environment.container_id, "Config": {"Labels": {environment.OWNER_LABEL: "another-run"}}}
    results = [
        CommandResult(stdout=environment.container_id or "", stderr="", returncode=0),
        CommandResult(stdout=json.dumps([container]), stderr="", returncode=0),
    ]
    with patch.object(environment, "_control_async", new_callable=AsyncMock, side_effect=results) as control:
        with pytest.raises(DockerCommandError, match="ownership"):
            await environment.cleanup_async()
    assert all(call.kwargs["arguments"][0] != "rm" for call in control.call_args_list)
    assert environment.cleanup_status == "failed"


async def test_cleanup_removes_only_owned_id_async() -> None:
    environment = _running_environment()
    container = {"Id": environment.container_id, "Config": {"Labels": {environment.OWNER_LABEL: environment.run_id}}}
    results = [
        CommandResult(stdout=environment.container_id or "", stderr="", returncode=0),
        CommandResult(stdout=json.dumps([container]), stderr="", returncode=0),
        CommandResult(stdout="", stderr="", returncode=0),
        CommandResult(stdout="", stderr="", returncode=0),
    ]
    with patch.object(environment, "_control_async", new_callable=AsyncMock, side_effect=results) as control:
        await environment.cleanup_async()
        await environment.cleanup_async()
    assert control.call_args_list[2].kwargs["arguments"] == ["rm", "--force", environment.container_id]
    assert control.call_count == 4
    assert environment.cleanup_status == "removed"


async def test_start_copies_only_named_asset_and_verifies_profile_async(tmp_path: Path) -> None:
    asset = tmp_path / "flag"
    asset.write_text("synthetic", encoding="utf-8")
    environment = _environment()
    container_id = "b" * 64
    container = {
        "Id": container_id,
        "Image": "sha256:fixture",
        "Mounts": [],
        "Config": {"Labels": {environment.OWNER_LABEL: environment.run_id}, "WorkingDir": "/workspace"},
        "HostConfig": {
            "NetworkMode": "none",
            "NanoCpus": 1000000000,
            "Memory": 536870912,
            "MemorySwap": 536870912,
            "PidsLimit": 128,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "Init": True,
        },
    }
    results = [
        CommandResult(stdout=container_id, stderr="", returncode=0),
        CommandResult(stdout=container_id, stderr="", returncode=0),
        CommandResult(stdout=json.dumps([container]), stderr="", returncode=0),
        CommandResult(stdout="", stderr="", returncode=0),
    ]
    with patch.object(environment, "_control_async", new_callable=AsyncMock, side_effect=results) as control:
        await environment.start_async(files={"flag": asset})
    create = control.call_args_list[0].kwargs["arguments"]
    assert "--network" in create and "none" in create and "--cap-drop" in create
    assert not any(argument in create for argument in ("--volume", "-v", "--mount", "--publish", "--privileged"))
    assert control.call_args_list[3].kwargs["arguments"] == [
        "cp",
        "--",
        str(asset.resolve()),
        f"{container_id}:/workspace/flag",
    ]
    environment.container_identity["HostConfig"]["NetworkMode"] = "host"
    with pytest.raises(DockerCommandError, match="profile"):
        environment._verify_isolation()
