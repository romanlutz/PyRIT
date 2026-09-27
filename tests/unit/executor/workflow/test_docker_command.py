# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.workflow.docker_command import SubprocessDockerRunner

if TYPE_CHECKING:
    from pathlib import Path


def runner(tmp_path: Path) -> SubprocessDockerRunner:
    work, config = tmp_path / "work", tmp_path / "config"
    work.mkdir()
    config.mkdir()
    return SubprocessDockerRunner(
        executable=tmp_path / "docker.exe",
        working_directory=work,
        config_directory=config,
        daemon_endpoint="npipe:////./pipe/inert-docker",
        output_limit=16,
    )


def fake_process(*, returncode: int | None, output: bytes = b"ok") -> MagicMock:
    process = MagicMock(spec=asyncio.subprocess.Process)
    process.returncode = returncode
    process.pid = 12345
    process.stdin = MagicMock(spec=asyncio.StreamWriter)
    process.stdin.drain = AsyncMock()
    for stream_name, content in (("stdout", output), ("stderr", b"")):
        stream = asyncio.StreamReader()
        stream.feed_data(content)
        stream.feed_eof()
        setattr(process, stream_name, stream)
    process.wait = AsyncMock(return_value=returncode)
    return process


async def test_argv_environment_and_stdin_never_forward_ambient_credentials_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=0)
    with (
        patch.dict(
            os.environ,
            {"GH_TOKEN": "inert", "OPENAI_API_KEY": "inert", "COMPOSE_FILE": "evil.yaml", "DOCKER_HOST": "tcp://other"},
        ),
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process) as spawn,
    ):
        result = await command_runner.run_async(
            arguments=("compose", "--file", "-", "up"), input_text="{}", timeout_seconds=1
        )
    assert result.returncode == 0 and result.stdout == "ok" and not result.truncated
    assert spawn.call_args.args == (
        str(tmp_path / "docker.exe"),
        "--host",
        "npipe:////./pipe/inert-docker",
        "compose",
        "--file",
        "-",
        "up",
    )
    environment = spawn.call_args.kwargs["env"]
    assert not set(environment) & {"GH_TOKEN", "OPENAI_API_KEY", "COMPOSE_FILE", "DOCKER_HOST"}
    assert environment["DOCKER_CONFIG"] == str(tmp_path / "config")
    assert environment["COMPOSE_DISABLE_ENV_FILE"] == "1"
    assert spawn.call_args.kwargs["cwd"] == tmp_path / "work"
    process.stdin.write.assert_called_once_with(b"{}")
    process.stdin.close.assert_called_once()


async def test_output_is_drained_but_capture_is_bounded_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=0, output=b"x" * 100)
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process):
        result = await command_runner.run_async(arguments=("container", "ls"), timeout_seconds=1)
    assert result.stdout == "x" * 16 and result.truncated


@pytest.mark.parametrize("cancel", [False, True])
async def test_timeout_or_cancellation_reaps_owned_process_async(*, tmp_path: Path, cancel: bool) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=None)
    waiting = asyncio.Event()

    async def wait_async() -> None:
        waiting.set()
        await asyncio.Event().wait()

    async def terminate_async(value: asyncio.subprocess.Process) -> None:
        assert value is process
        process.returncode = -1

    process.wait.side_effect = wait_async
    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process),
        patch.object(command_runner, "_terminate_async", side_effect=terminate_async) as terminate,
    ):
        task = asyncio.create_task(
            command_runner.run_async(arguments=("compose", "up"), timeout_seconds=0.01 if not cancel else 1)
        )
        await asyncio.wait_for(waiting.wait(), timeout=1)
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.timed_out and result.returncode is None
    terminate.assert_awaited_once()


@pytest.mark.parametrize("directory", ["work", "config"])
async def test_nonempty_work_or_docker_config_is_rejected_before_spawn_async(*, tmp_path: Path, directory: str) -> None:
    command_runner = runner(tmp_path)
    (tmp_path / directory / ".env").write_text("GH_TOKEN=inert", encoding="utf-8")
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as spawn:
        with pytest.raises(ValueError, match="empty host"):
            await command_runner.run_async(arguments=("compose", "up"), timeout_seconds=1)
    spawn.assert_not_awaited()


@pytest.mark.parametrize("endpoint", ["tcp://remote:2375", "ssh://host", ""])
def test_nonlocal_daemon_requires_a_different_explicit_provider(*, tmp_path: Path, endpoint: str) -> None:
    with pytest.raises(ValueError, match="local"):
        SubprocessDockerRunner(
            executable=tmp_path / "docker",
            working_directory=tmp_path,
            config_directory=tmp_path,
            daemon_endpoint=endpoint,
        )


async def test_windows_termination_names_only_owned_pid_and_checks_exit_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=None)
    killer = fake_process(returncode=0)

    async def reaped_async() -> int:
        process.returncode = -1
        return -1

    process.wait.side_effect = reaped_async
    with (
        patch("pyrit.executor.workflow.docker_command.os", spec=os) as platform,
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=killer) as spawn,
    ):
        platform.name = "nt"
        platform.environ = {"SYSTEMROOT": str(tmp_path)}
        await command_runner._terminate_async(process)
    assert spawn.call_args.args[1:] == ("/PID", "12345", "/T", "/F")
    assert spawn.call_args.args[0] == str(tmp_path / "System32" / "taskkill.exe")
    process.wait.assert_awaited_once()


async def test_cancellation_during_spawn_still_reaps_the_created_process_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=None)
    spawning, resume = asyncio.Event(), asyncio.Event()

    async def spawn_async(*args: object, **kwargs: object) -> MagicMock:
        spawning.set()
        await resume.wait()
        return process

    with (
        patch("asyncio.create_subprocess_exec", side_effect=spawn_async),
        patch.object(command_runner, "_terminate_async", new_callable=AsyncMock) as terminate,
    ):
        task = asyncio.create_task(command_runner.run_async(arguments=("compose", "up"), timeout_seconds=1))
        await asyncio.wait_for(spawning.wait(), timeout=1)
        task.cancel()
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    terminate.assert_awaited_once_with(process)


async def test_repeated_cancellation_cannot_interrupt_process_cleanup_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=None)
    waiting, terminating, resume = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def wait_async() -> None:
        waiting.set()
        await asyncio.Event().wait()

    async def terminate_async(value: asyncio.subprocess.Process) -> None:
        terminating.set()
        await resume.wait()
        assert value is process
        process.returncode = -1

    process.wait.side_effect = wait_async
    with (
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=process),
        patch.object(command_runner, "_terminate_async", side_effect=terminate_async) as terminate,
    ):
        task = asyncio.create_task(command_runner.run_async(arguments=("compose", "up"), timeout_seconds=1))
        await asyncio.wait_for(waiting.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(terminating.wait(), timeout=1)
        task.cancel()
        resume.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    terminate.assert_awaited_once()
    assert process.returncode == -1


async def test_failed_windows_tree_termination_is_explicit_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process, killer = fake_process(returncode=None), fake_process(returncode=1)
    with (
        patch("pyrit.executor.workflow.docker_command.os", spec=os) as platform,
        patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, return_value=killer),
    ):
        platform.name = "nt"
        platform.environ = {"SYSTEMROOT": str(tmp_path)}
        with pytest.raises(OSError, match="could not be confirmed"):
            await command_runner._terminate_async(process)


async def test_posix_termination_targets_only_owned_process_group_async(tmp_path: Path) -> None:
    command_runner = runner(tmp_path)
    process = fake_process(returncode=-1)
    with (
        patch("pyrit.executor.workflow.docker_command.os", spec=os) as platform,
        patch("pyrit.executor.workflow.docker_command.signal") as signals,
    ):
        platform.name = "posix"
        platform.killpg = MagicMock()
        signals.SIGKILL = 9
        await command_runner._terminate_async(process)
    platform.killpg.assert_called_once_with(process.pid, 9)
