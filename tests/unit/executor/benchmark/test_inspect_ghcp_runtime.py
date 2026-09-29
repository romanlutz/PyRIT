# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The gateway has one READY read, no post-ready polling or duplicate kill."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from inspect_ai.util import ExecCompleted, ExecRemoteProcess, ExecStdout

from pyrit.executor.benchmark._inspect_ghcp_runtime import (
    InspectGhcpLimits,
    InspectGhcpSandboxRuntime,
    _SandboxRpc,
)


async def test_gateway_ready_does_not_leave_a_poll_task_or_kill_early() -> None:
    process = MagicMock(spec=ExecRemoteProcess)
    process.pid = 123
    process.__aiter__.return_value = iter([ExecStdout(data='{"kind":"ready","pid":123}\n')])
    process.kill = AsyncMock()
    gateway = _SandboxRpc(process=process, monitor=False)
    assert gateway._reader is None
    assert await gateway.receive_async(kind="ready", timeout=2) == {"kind": "ready", "pid": 123}
    process.kill.assert_not_awaited()


async def test_explicit_failed_run_cleanup_cannot_kill_gateway_twice() -> None:
    runtime = InspectGhcpSandboxRuntime(
        run_id="11111111-1111-1111-1111-111111111111",
        token="x" * 43,
        model_id="approved",
        wire_model="qwen3:1.7b",
        cli_path="/opt/pyrit/copilot",
        cli_sha256="a" * 64,
        gateway_image_service="model-bridge",
        limits=InspectGhcpLimits(),
        allowed_tools=("bash",),
        control_receipt_sink=AsyncMock(),
        verify_image_async=AsyncMock(return_value="a" * 64),
        approved_image_ids={"agent": "sha256:" + "c" * 64, "model-bridge": "sha256:" + "c" * 64},
    )
    gateway = MagicMock(spec=_SandboxRpc)
    gateway.kill_async = AsyncMock()
    runtime._gateway = gateway
    runtime._stopped = True
    await runtime.close_async()
    await runtime.close_async()
    assert gateway.kill_async.await_count == 1
    assert runtime.closed


def _stopped_agent(*, exit_code: int) -> MagicMock:
    process = MagicMock(spec=ExecRemoteProcess)
    process.pid = 163
    process.__aiter__.return_value = iter(
        [
            ExecStdout(data='{"kind":"stopped","cli_exited":true}\n'),
            ExecCompleted(exit_code=exit_code),
        ]
    )
    process.close_stdin = AsyncMock()
    return process


@pytest.mark.parametrize("exit_code", [0, 1])
async def test_worker_stop_requires_matching_job_completion_without_closing_stdin(exit_code: int) -> None:
    process = _stopped_agent(exit_code=exit_code)
    worker = _SandboxRpc(process=process)
    assert (await worker.receive_async(kind="stopped", timeout=2))["cli_exited"] is True
    if exit_code:
        with pytest.raises(RuntimeError, match="exited with code 1"):
            await worker.finish_async(timeout=2)
    else:
        await worker.finish_async(timeout=2)
    process.close_stdin.assert_not_awaited()


async def test_worker_stop_rejects_changed_job_id() -> None:
    process = _stopped_agent(exit_code=0)
    worker = _SandboxRpc(process=process)
    await worker.receive_async(kind="stopped", timeout=2)
    process.pid = 164
    with pytest.raises(RuntimeError, match="job identity"):
        await worker.finish_async(timeout=2)
    process.close_stdin.assert_not_awaited()


async def test_worker_stop_rejects_missing_delayed_completion() -> None:
    process = MagicMock(spec=ExecRemoteProcess)
    process.pid = 165

    async def only_stopped_async():
        yield ExecStdout(data='{"kind":"stopped","cli_exited":true}\n')
        await asyncio.Event().wait()

    process.__aiter__.side_effect = only_stopped_async
    process.close_stdin = AsyncMock()
    worker = _SandboxRpc(process=process)
    await worker.receive_async(kind="stopped", timeout=2)
    with pytest.raises(RuntimeError, match="approved stop budget"):
        await worker.finish_async(timeout=1)
    process.close_stdin.assert_not_awaited()
