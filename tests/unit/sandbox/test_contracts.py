# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sandbox model and lifecycle contract tests."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from pyrit.executor.capability import (
    CapabilityToolRuntime,
    SandboxOperationEvidence,
    ToolDeclaration,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionOutput,
    ToolRegistry,
)
from pyrit.models import ToolCallRequest
from pyrit.sandbox import (
    SandboxEnvironment,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxProvider,
    SandboxSession,
    SandboxSessionSpec,
    SandboxTaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.executor.capability.evidence import CapabilityEvidenceSink
    from pyrit.models import JSONValue


class _CountingSession(SandboxSession):
    def __init__(self, *, spec: SandboxSessionSpec, fail_initialize: bool = False) -> None:
        super().__init__(provider_name="counting", spec=spec)
        self.initialize_count = 0
        self.close_count = 0
        self.fail_initialize = fail_initialize

    @property
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        return ()

    async def _initialize_async(self) -> None:
        self.initialize_count += 1
        if self.fail_initialize:
            raise RuntimeError("partial initialization")

    async def _close_async(self) -> None:
        self.close_count += 1
        await asyncio.sleep(0.01)


class _CountingProvider(SandboxProvider):
    def __init__(self, *, fail_prepare: bool = False, fail_session: bool = False) -> None:
        super().__init__()
        self.prepare_count = 0
        self.cleanup_count = 0
        self.task_prepare_count = 0
        self.task_cleanup_count = 0
        self.fail_prepare = fail_prepare
        self.fail_session = fail_session

    @property
    def name(self) -> str:
        return "counting"

    async def _prepare_async(self) -> None:
        self.prepare_count += 1
        await asyncio.sleep(0.01)
        if self.fail_prepare:
            raise RuntimeError("partial provider preparation")

    async def _prepare_task_async(self, task: SandboxTaskSpec) -> None:
        self.task_prepare_count += 1
        await asyncio.sleep(0.01)

    async def _cleanup_task_async(self, task: SandboxTaskSpec) -> None:
        self.task_cleanup_count += 1
        await asyncio.sleep(0.01)

    async def _create_session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> SandboxSession:
        return _CountingSession(spec=spec, fail_initialize=self.fail_session)

    async def _cleanup_async(self) -> None:
        self.cleanup_count += 1
        await asyncio.sleep(0.01)

    async def _cleanup_orphans_async(self) -> int:
        return 0


class _BlockingCleanupProvider(_CountingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.cleanup_started = asyncio.Event()
        self.release_cleanup = asyncio.Event()

    async def _cleanup_async(self) -> None:
        self.cleanup_count += 1
        self.cleanup_started.set()
        await self.release_cleanup.wait()


class _RetryEvidenceTool:
    def __init__(self) -> None:
        self.call_count = 0

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        self.call_count += 1
        now = datetime.now(tz=timezone.utc)
        evidence = SandboxOperationEvidence(
            provider="test",
            operation="exec",
            outcome="failed" if self.call_count == 1 else "succeeded",
            started_at=now,
            ended_at=now,
            call_id=context.call_id,
            attempt_id=context.attempt_id,
        )
        if self.call_count == 1:
            raise ToolExecutionError(
                code="transient",
                message="retry",
                retryable=True,
                evidence=(evidence,),
            )
        return ToolExecutionOutput(output={"ok": True}, evidence=(evidence,))


def test_session_spec_default_resolution_and_validation() -> None:
    spec = SandboxSessionSpec(
        environments=(
            SandboxEnvironmentSpec(name="zeta"),
            SandboxEnvironmentSpec(name="alpha"),
        )
    )
    assert spec.resolve_default_environment() == "alpha"
    with pytest.raises(ValidationError, match="names must be unique"):
        SandboxSessionSpec(
            environments=(
                SandboxEnvironmentSpec(name="same"),
                SandboxEnvironmentSpec(name="same"),
            )
        )
    with pytest.raises(ValidationError, match="not defined"):
        SandboxSessionSpec(default_environment="missing")


def test_models_are_frozen_and_exec_form_is_exclusive() -> None:
    spec = SandboxSessionSpec()
    with pytest.raises(ValidationError, match="frozen"):
        spec.session_id = "changed"
    with pytest.raises(ValidationError, match="Exactly one"):
        SandboxExecRequest()
    with pytest.raises(ValidationError, match="Exactly one"):
        SandboxExecRequest(argv=("command",), shell_script="command")


async def test_concurrent_provider_and_task_lifecycle_runs_once() -> None:
    provider = _CountingProvider()
    task = SandboxTaskSpec(task_id="task")
    await asyncio.gather(provider.prepare_async(), provider.prepare_async())
    await asyncio.gather(provider.prepare_task_async(task), provider.prepare_task_async(task))
    await asyncio.gather(provider.cleanup_task_async(task), provider.cleanup_task_async(task))
    await asyncio.gather(provider.cleanup_async(), provider.cleanup_async())
    assert provider.prepare_count == 1
    assert provider.task_prepare_count == 1
    assert provider.task_cleanup_count == 1
    assert provider.cleanup_count == 1


async def test_repeated_cancellation_does_not_interrupt_provider_cleanup() -> None:
    provider = _BlockingCleanupProvider()
    await provider.prepare_async()
    cleanup = asyncio.create_task(provider.cleanup_async())
    await provider.cleanup_started.wait()

    cleanup.cancel()
    await asyncio.sleep(0)
    assert not cleanup.done()
    cleanup.cancel()
    await asyncio.sleep(0)
    assert not cleanup.done()
    provider.release_cleanup.set()

    with pytest.raises(asyncio.CancelledError):
        await cleanup
    assert provider.cleanup_count == 1


async def test_provider_partial_prepare_failure_cleans_once() -> None:
    provider = _CountingProvider(fail_prepare=True)
    with pytest.raises(RuntimeError, match="partial provider preparation"):
        await provider.prepare_async()
    await provider.cleanup_async()
    assert provider.prepare_count == 1
    assert provider.cleanup_count == 1


async def test_provider_rejects_sessions_after_cleanup() -> None:
    provider = _CountingProvider()
    await provider.prepare_async()
    await provider.cleanup_async()
    with pytest.raises(RuntimeError, match="not cleaning up"):
        await provider.create_session_async(spec=SandboxSessionSpec())
    with pytest.raises(RuntimeError, match="not available"):
        await provider.prepare_task_async(SandboxTaskSpec(task_id="late"))


async def test_session_initialization_and_cleanup_are_exactly_once() -> None:
    session = _CountingSession(spec=SandboxSessionSpec())
    await asyncio.gather(session.initialize_async(), session.initialize_async())
    await asyncio.gather(session.close_async(), session.close_async())
    assert session.initialize_count == 1
    assert session.close_count == 1
    closed_session = _CountingSession(spec=SandboxSessionSpec())
    await closed_session.close_async()
    with pytest.raises(RuntimeError, match="already closed"):
        await closed_session.initialize_async()


async def test_session_cleanup_is_serialized_after_initialization() -> None:
    session = _CountingSession(spec=SandboxSessionSpec())
    initialization = asyncio.create_task(session.initialize_async())
    await asyncio.sleep(0)
    cleanup = asyncio.create_task(session.close_async())
    await asyncio.gather(initialization, cleanup)
    assert session.initialize_count == 1
    assert session.close_count == 1


async def test_partial_session_initialization_cleans_once() -> None:
    session = _CountingSession(spec=SandboxSessionSpec(), fail_initialize=True)
    with pytest.raises(RuntimeError, match="partial initialization"):
        await session.initialize_async()
    await session.close_async()
    assert session.initialize_count == 1
    assert session.close_count == 1


async def test_tool_runtime_retains_sandbox_evidence_from_all_retry_attempts() -> None:
    implementation = _RetryEvidenceTool()
    registry = ToolRegistry()
    registry.register(
        declaration=ToolDeclaration(
            name="retry",
            idempotent=True,
            max_retries=1,
            retryable_error_codes=("transient",),
        ),
        implementation=implementation,
    )
    runtime = CapabilityToolRuntime(registry=registry)
    request = ToolCallRequest(call_id="call", name="retry", arguments=json.dumps({}))
    case_id = uuid.uuid4()
    prepared, _ = await runtime.prepare_calls_async(
        calls=((request, uuid.uuid4()),),
        case_id=case_id,
        conversation_id="conversation",
        asset_references=(),
        environment_requirement_references=(),
        cancellation_event=None,
    )
    record = await runtime.execute_call_async(
        call=prepared[0],
        case_id=case_id,
        conversation_id="conversation",
        asset_references=(),
        environment_requirement_references=(),
        cancellation_event=None,
    )
    assert record.result.error is None
    assert implementation.call_count == 2
    assert len(record.additional_evidence) == 2
    assert [evidence.outcome for evidence in record.additional_evidence] == ["failed", "succeeded"]
