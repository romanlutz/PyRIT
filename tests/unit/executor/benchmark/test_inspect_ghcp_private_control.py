# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The same real Inspect sandbox can hide secret control I/O without muting public events."""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.util import ExecResult, SandboxConnection
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment
from inspect_ai.util._sandbox.docker.util import ComposeProject
from inspect_ai.util._sandbox.events import SandboxEnvironmentProxy

from pyrit.executor.benchmark._inspect_ghcp_runtime import InspectGhcpLimits, InspectGhcpSandboxRuntime


def _sandbox_pair(*, service: str = "agent") -> tuple[SandboxEnvironmentProxy, DockerSandboxEnvironment]:
    project = ComposeProject(name="isolated-test", config=None, sample_id="one", epoch=1, env=None)
    raw = DockerSandboxEnvironment(service=service, project=project, working_dir="/workspace")
    return SandboxEnvironmentProxy(raw), raw


def _runtime(*, sink: AsyncMock) -> InspectGhcpSandboxRuntime:
    return InspectGhcpSandboxRuntime(
        run_id="11111111-1111-1111-1111-111111111111",
        token="x" * 43,
        model_id="approved",
        wire_model="qwen3:1.7b",
        cli_path="/opt/pyrit/copilot",
        cli_sha256="a" * 64,
        gateway_image_service="model-bridge",
        limits=InspectGhcpLimits(),
        allowed_tools=("bash",),
        control_receipt_sink=sink,
        verify_image_async=AsyncMock(return_value="a" * 64),
        approved_image_ids={"agent": "sha256:" + "c" * 64, "model-bridge": "sha256:" + "c" * 64},
    )


async def test_public_as_type_raw_control_does_not_suppress_concurrent_proxied_event() -> None:
    proxy, raw = _sandbox_pair()
    observed = ExecResult(success=True, returncode=0, stdout="", stderr="")
    with (
        patch.object(raw, "exec", new_callable=AsyncMock, return_value=observed),
        patch("inspect_ai.log._transcript.transcript") as transcript,
    ):
        scoped = proxy.as_type(DockerSandboxEnvironment)
        assert type(scoped) is DockerSandboxEnvironment
        assert scoped is raw
        await asyncio.gather(
            proxy.exec(["/usr/bin/true"], input="public control frame"),
            scoped.exec(["/usr/bin/true"], input="ephemeral-run-token-test-value"),
        )
        calls = transcript.return_value._event.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0].input == "public control frame"
        assert "ephemeral-run-token-test-value" not in calls[0].args[0].model_dump_json()
        assert proxy._events is True


@pytest.mark.parametrize(
    ("service", "container"),
    [("agent", "inspect-one-agent-1"), ("model-bridge", "inspect-one-model-bridge-1")],
)
async def test_exact_provider_private_write_has_bounded_token_free_receipt_and_public_events(
    service: str, container: str
) -> None:
    proxy, raw = _sandbox_pair(service=service)
    receipt_sink = AsyncMock()
    runtime = _runtime(sink=receipt_sink)
    runtime._container_ref = "inspect-one-agent-1"
    runtime._model_container_ref = "inspect-one-model-bridge-1"
    image_ids = {"agent": "a" * 64, "model-bridge": "b" * 64}
    runtime._verify_image_async = AsyncMock(
        side_effect=lambda ref, image: image_ids["agent" if ref == runtime._container_ref else "model-bridge"]
    )
    await runtime._attest_container_ids_async()
    connection = SandboxConnection(type="docker", command="docker", container=container)
    result = ExecResult(success=True, returncode=0, stdout="321\n", stderr="")
    with (
        patch.object(raw, "connection", new_callable=AsyncMock, return_value=connection),
        patch.object(raw, "exec", new_callable=AsyncMock, return_value=result) as exec_control,
        patch("inspect_ai.log._transcript.transcript") as transcript,
    ):
        scoped = await runtime._scoped_raw_sandbox_async(
            environment=proxy, service=service, expected_container_ref=container
        )
        assert scoped is raw
        await runtime._deliver_private_token_async(
            raw=scoped,
            service=service,
            helper="/tmp/pyrit-inspect/inspect_ghcp_token_file.py",
            expected_container_ref=container,
        )
        await proxy.exec(["/usr/bin/true"], input="public original task event")
        first = exec_control.call_args_list[0]
        assert first.kwargs["input"] == "x" * 43
        assert first.kwargs["timeout"] == 20
        assert first.kwargs["timeout_retry"] is False
        assert "x" * 43 not in json.dumps(first.args)
        assert len(transcript.return_value._event.call_args_list) == 1
        assert transcript.return_value._event.call_args.args[0].input == "public original task event"
        assert "x" * 43 not in transcript.return_value._event.call_args.args[0].model_dump_json()
    receipt_sink.assert_awaited_once()
    receipt = receipt_sink.await_args.args[0]
    assert receipt["container_id"] == image_ids[service]
    assert receipt["container_id"] != container
    assert receipt["observed_job_id"] == 321
    assert receipt["completed_exit_code"] == 0
    assert receipt["frame_size_bytes"] == 43
    assert receipt["frame_sha256"] == hashlib.sha256(b"x" * 43).hexdigest()
    assert "x" * 43 not in json.dumps(receipt)


@pytest.mark.parametrize(
    "failure",
    [
        ExecResult(success=False, returncode=1, stdout="", stderr="x" * 43),
        RuntimeError("x" * 43),
        asyncio.CancelledError("x" * 43),
    ],
)
async def test_private_write_failure_never_echoes_secret_or_suppresses_public_events(
    failure: ExecResult[str] | RuntimeError | asyncio.CancelledError,
) -> None:
    proxy, raw = _sandbox_pair()
    receipt_sink = AsyncMock()
    runtime = _runtime(sink=receipt_sink)
    runtime._container_ref = "a" * 64
    runtime._attested_container_ids["agent"] = "a" * 64
    with (
        patch.object(
            raw,
            "exec",
            new_callable=AsyncMock,
            side_effect=[failure, ExecResult(success=True, returncode=0, stdout="", stderr="")],
        ),
        patch("inspect_ai.log._transcript.transcript") as transcript,
    ):
        expected = asyncio.CancelledError if isinstance(failure, asyncio.CancelledError) else RuntimeError
        with pytest.raises(expected) as captured:
            await runtime._deliver_private_token_async(
                raw=raw,
                service="agent",
                helper="/tmp/pyrit-inspect/inspect_ghcp_token_file.py",
                expected_container_ref="a" * 64,
            )
        assert "x" * 43 not in str(captured.value)
        receipt_sink.assert_not_awaited()
        await proxy.exec(["/usr/bin/true"], input="visible after failure")
        assert transcript.return_value._event.call_args.args[0].input == "visible after failure"
        assert proxy._events is True


async def test_cancelled_private_writer_finishes_before_token_file_cleanup() -> None:
    proxy, raw = _sandbox_pair()
    runtime = _runtime(sink=AsyncMock())
    runtime._container_ref = "a" * 64
    runtime._attested_container_ids["agent"] = "a" * 64
    helper = "/tmp/pyrit-inspect/inspect_ghcp_token_file.py"
    runtime._token_helpers["agent"] = helper
    started = asyncio.Event()
    release = asyncio.Event()
    commands: list[str] = []

    async def execute_async(cmd: list[str], **kwargs: object) -> ExecResult[str]:
        commands.append(cmd[2])
        if cmd[2] == "write":
            started.set()
            await release.wait()
            return ExecResult(success=True, returncode=0, stdout="901\n", stderr="")
        return ExecResult(success=True, returncode=0, stdout="", stderr="")

    with (
        patch.object(raw, "exec", new_callable=AsyncMock, side_effect=execute_async),
        patch("pyrit.executor.benchmark._inspect_ghcp_runtime.sandbox", return_value=proxy),
        patch("inspect_ai.log._transcript.transcript") as transcript,
    ):
        delivery = asyncio.create_task(
            runtime._deliver_private_token_async(
                raw=raw, service="agent", helper=helper, expected_container_ref="a" * 64
            )
        )
        await started.wait()
        delivery.cancel()
        with pytest.raises(asyncio.CancelledError):
            await delivery
        assert runtime._pending_token_write is not None
        release.set()
        await runtime.close_async()
        assert commands == ["write", "clear", "absent"]
        assert runtime.closed
        assert len(transcript.return_value._event.call_args_list) == 2
        assert all(call.args[0].input is None for call in transcript.return_value._event.call_args_list)


async def test_raw_provider_must_match_exact_service_and_container() -> None:
    proxy, raw = _sandbox_pair()
    runtime = _runtime(sink=AsyncMock())
    connection = SandboxConnection(type="docker", command="docker", container="b" * 64)
    with patch.object(raw, "connection", new_callable=AsyncMock, return_value=connection):
        with pytest.raises(ValueError, match="different Inspect sample"):
            await runtime._scoped_raw_sandbox_async(environment=proxy, service="agent", expected_container_ref="a" * 64)
    with patch.object(proxy, "as_type", return_value=object()):
        with pytest.raises(TypeError, match="exact pinned Docker provider"):
            await runtime._scoped_raw_sandbox_async(environment=proxy, service="agent", expected_container_ref="a" * 64)


async def test_private_write_rejects_full_id_drift_before_and_after_raw_exec() -> None:
    proxy, raw = _sandbox_pair()
    receipt_sink = AsyncMock()
    runtime = _runtime(sink=receipt_sink)
    runtime._container_ref = "inspect-one-agent-1"
    runtime._attested_container_ids["agent"] = "a" * 64
    connection = SandboxConnection(type="docker", command="docker", container=runtime._container_ref)
    result = ExecResult(success=True, returncode=0, stdout="401\n", stderr="")
    with (
        patch.object(raw, "connection", new_callable=AsyncMock, return_value=connection),
        patch.object(raw, "exec", new_callable=AsyncMock, return_value=result) as execute,
    ):
        runtime._verify_image_async = AsyncMock(return_value="b" * 64)
        with pytest.raises(ValueError, match="before the write"):
            await runtime._deliver_private_token_async(
                raw=raw, service="agent", helper="/tmp/helper.py", expected_container_ref=runtime._container_ref
            )
        execute.assert_not_awaited()
        runtime._verify_image_async = AsyncMock(side_effect=["a" * 64, "b" * 64])
        with pytest.raises(ValueError, match="changed its verified full Docker ID"):
            await runtime._deliver_private_token_async(
                raw=raw, service="agent", helper="/tmp/helper.py", expected_container_ref=runtime._container_ref
            )
        execute.assert_awaited_once()
        receipt_sink.assert_not_awaited()


async def test_compose_names_must_be_resolved_to_distinct_full_ids_before_delivery() -> None:
    runtime = _runtime(sink=AsyncMock())
    runtime._container_ref = "inspect-one-agent-1"
    runtime._model_container_ref = "inspect-one-model-bridge-1"
    runtime._verify_image_async = AsyncMock(side_effect=["b" * 64, "a" * 64])
    await runtime._attest_container_ids_async()
    assert runtime.attested_container_ids == {"model-bridge": "b" * 64, "agent": "a" * 64}
    assert [call.args[0] for call in runtime._verify_image_async.await_args_list] == [
        "inspect-one-model-bridge-1",
        "inspect-one-agent-1",
    ]
    runtime._attested_container_ids.clear()
    runtime._verify_image_async = AsyncMock(return_value="inspect-one-model-bridge-1")
    with pytest.raises(ValueError, match="full Docker IDs"):
        await runtime._attest_container_ids_async()


async def test_unconfirmed_token_file_cleanup_cannot_be_reported_as_closed() -> None:
    proxy, raw = _sandbox_pair()
    runtime = _runtime(sink=AsyncMock())
    runtime._token_helpers["agent"] = "/tmp/pyrit-inspect/inspect_ghcp_token_file.py"
    with (
        patch.object(
            raw,
            "exec",
            new_callable=AsyncMock,
            return_value=ExecResult(success=False, returncode=1, stdout="", stderr="not removed"),
        ),
        patch("pyrit.executor.benchmark._inspect_ghcp_runtime.sandbox", return_value=proxy),
        patch("inspect_ai.log._transcript.transcript"),
    ):
        with pytest.raises(RuntimeError, match="not observed"):
            await runtime.close_async()
    assert not runtime.closed
    assert not runtime.private_tokens_consumed


@pytest.mark.parametrize("failure", [RuntimeError("raw control failed"), asyncio.CancelledError()])
async def test_failed_or_cancelled_raw_control_cannot_mute_following_proxied_task_event(
    failure: BaseException,
) -> None:
    proxy, raw = _sandbox_pair()
    observed = ExecResult(success=True, returncode=0, stdout="", stderr="")
    with (
        patch.object(raw, "exec", new_callable=AsyncMock, side_effect=[failure, observed]),
        patch("inspect_ai.log._transcript.transcript") as transcript,
    ):
        with pytest.raises(type(failure)):
            await proxy.as_type(DockerSandboxEnvironment).exec(
                ["/usr/bin/true"], input="ephemeral-run-token-test-value"
            )
        await proxy.exec(["/usr/bin/true"], input="task setup remains visible")
        calls = transcript.return_value._event.call_args_list
        assert len(calls) == 1
        assert calls[0].args[0].input == "task setup remains visible"
        assert proxy._events is True
