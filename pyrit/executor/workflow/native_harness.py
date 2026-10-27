# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Explicit harness configuration and per-run evidence, separate from tasks and sandbox providers."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from uuid import uuid4

from pyrit.executor.workflow.copilot_sdk_evidence import CopilotSdkDatabaseEvidenceSink
from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
from pyrit.executor.workflow.native_cli_report_adapter import build_native_cli_run_report, project_native_cli_evidence
from pyrit.models.native_cli_report import (
    NativeCliArtifactReference,
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportStatus,
    NativeCliRunReport,
)
from pyrit.models.native_cyber import NativeAgentCapabilities, NativeCyberReport, NativeCyberStatus
from pyrit.models.native_harness import (
    NativeHarnessEvidence,
    NativeHarnessIdentity,
    NativeModelCaptureIdentity,
    NativeModelResponseOrigin,
)
from pyrit.prompt_target import NativeCliTarget
from pyrit.prompt_target.native_agent_target import CopilotSdkAgentSession
from pyrit.prompt_target.native_cli_models import NativeCliProtocol

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from datetime import datetime
    from pathlib import Path
    from uuid import UUID

    from pydantic import JsonValue

    from pyrit.executor.workflow.native_model_transport import NativeModelObserver
    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore
    from pyrit.models.native_cyber import (
        NativeCyberCleanup,
        NativeCyberJudgment,
        NativeCyberReadiness,
        NativeCyberRequest,
    )
    from pyrit.models.native_cyber_evidence import NativeCyberRawStreamKey, NativeCyberResponseMode
    from pyrit.prompt_target import NativeAgentTarget, PromptTarget
    from pyrit.prompt_target.native_cli_models import NativeCliRunConfig
    from pyrit.prompt_target.native_cli_transport import SandboxProcessLauncher


@runtime_checkable
class NativeTaskRuntime(Protocol):
    """Task/provider-owned runtime. It does not choose or implement an agent harness."""

    run_id: str
    environment_id: str
    simulated: bool

    async def validate_agent_storage_async(self, *, directory: Path) -> None:
        """Verify the owned sandbox excludes host evidence and credentials."""
        ...

    async def grade_async(self, *, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        """Acquire the original judgment once after quiescence, before owned cleanup."""
        ...


@dataclass(frozen=True, kw_only=True)
class NativeHarnessReportContext:
    """Controller-owned task, lifecycle and acquired judgment, not harness-derived claims."""

    run_id: str
    binding_name: str
    binding_version: str
    request: NativeCyberRequest
    input_sha256: str
    seed_id: str | None
    status: NativeCyberStatus
    readiness: NativeCyberReadiness | None
    started_at: datetime
    expires_at: datetime
    ended_at: datetime
    conversation_id: str | None
    attack_result_id: str | None
    technique_identifier: dict[str, JsonValue] | None
    judgment: NativeCyberJudgment | None
    cleanup: NativeCyberCleanup
    errors: tuple[str, ...]


class NativeHarnessRun(ABC):
    """Per-run target and recorder; the controller still owns grading and publication."""

    target: PromptTarget

    def __init__(self, *, runtime: NativeTaskRuntime, run_id: str, harness_id: str) -> None:
        """
        Bind the exact allocated runtime without selecting a provider or task.

        Raises:
            ValueError: If run identity or simulation provenance is invalid.
        """
        if runtime.run_id != run_id or type(runtime.simulated) is not bool:
            raise ValueError("Harness runtime belongs to a different run or has unknown provenance.")
        self.runtime = runtime
        self.identity = NativeHarnessIdentity(
            run_id=run_id, environment_id=runtime.environment_id, harness_id=harness_id
        )
        self.simulated = runtime.simulated
        self._quiescent = False

    def verify_identity(self) -> None:
        """
        Reject a changed owned runtime before capture, grading or publication.

        Raises:
            ValueError: If runtime identity changed.
        """
        if (
            self.runtime.run_id != self.identity.run_id
            or self.runtime.environment_id != self.identity.environment_id
            or self.runtime.simulated != self.simulated
        ):
            raise ValueError("Owned harness runtime identity changed during the episode.")

    @property
    @abstractmethod
    def turn_open(self) -> bool:
        """Whether the current capture turn can be sealed once."""
        ...

    @abstractmethod
    async def begin_turn_async(
        self, *, turn_index: int, started_at: datetime, response_mode: NativeCyberResponseMode
    ) -> None:
        """Open declared evidence sources before sending the prepared instruction."""
        ...

    @abstractmethod
    async def finish_turn_async(
        self,
        *,
        request_piece_ids: tuple[UUID, ...],
        response_piece_ids: tuple[UUID, ...],
        tool_request_piece_ids: tuple[UUID, ...],
        tool_result_piece_ids: tuple[UUID, ...],
        interrupted: bool,
    ) -> None:
        """Seal observed sources and link authentic normalizer-persisted pieces."""
        ...

    @abstractmethod
    async def quiesce_async(self) -> None:
        """Confirm harness-owned work has settled without destroying the task environment."""
        ...

    @abstractmethod
    async def checkpoint_async(self) -> None:
        """Retain gaps and observed watermarks, including after a turn was sealed."""
        ...

    @abstractmethod
    def evidence(self) -> NativeHarnessEvidence:
        """Return neutral grading gates and unchanged harness-specific source evidence."""
        ...

    @abstractmethod
    def build_report(self, context: NativeHarnessReportContext) -> NativeCyberReport | NativeCliRunReport:
        """Project the original task judgment and lifecycle into this harness's report format."""
        ...


class NativeHarnessAdapter(ABC):
    """Reusable explicit harness selection, validated before environment allocation."""

    harness_id: str
    capabilities: NativeAgentCapabilities

    def validate_request(self, request: NativeCyberRequest) -> None:
        """
        Reject unsupported retained sessions or attack loops before opening a resource.

        Raises:
            ValueError: If the adapter cannot execute the selected lifecycle.
        """
        if request.technique != "literal" or (request.operator_steps and not self.capabilities.operator_steps):
            raise ValueError("The selected harness does not support this technique or retained operator turns.")

    @abstractmethod
    def required_raw_streams(self) -> tuple[NativeCyberRawStreamKey, ...]:
        """Source requirements declared before dispatch, owned by this harness."""
        ...

    @abstractmethod
    async def attach_async(
        self, *, runtime: NativeTaskRuntime, run_id: str, store: NativeCyberEvidenceStore
    ) -> NativeHarnessRun:
        """Attach the configured harness to this owned environment without dispatching a task."""
        ...


@dataclass(frozen=True, kw_only=True)
class CopilotSdkHarnessConnection:
    """Already-created SDK target and host transport injection, provided by the caller."""

    target: NativeAgentTarget
    bind_model_observer: Callable[[NativeModelObserver], None] = field(repr=False)
    quiesce_agent_async: Callable[[], Awaitable[None]] = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class CopilotSdkHarnessAdapter(NativeHarnessAdapter):
    """Retained GHCP SDK plus original host-model evidence; not the universal harness contract."""

    connect: Callable[[NativeTaskRuntime], CopilotSdkHarnessConnection] = field(repr=False)
    capabilities: NativeAgentCapabilities
    harness_id: str = field(default="copilot_sdk", init=False)

    def required_raw_streams(self) -> tuple[NativeCyberRawStreamKey, ...]:
        """
        Declare the observed SDK and original request/response sources.

        Returns:
            tuple[NativeCyberRawStreamKey, ...]: Required source identities.
        """
        return CopilotSdkDatabaseEvidenceSink.required_raw_streams()

    async def attach_async(
        self, *, runtime: NativeTaskRuntime, run_id: str, store: NativeCyberEvidenceStore
    ) -> NativeHarnessRun:
        """
        Attach the actual SDK observer before any events or task delivery.

        Returns:
            NativeHarnessRun: This run's SDK target and recorder.
        """
        return _CopilotRun(adapter=self, runtime=runtime, run_id=run_id, store=store)


class _CopilotRun(NativeHarnessRun):
    target: NativeAgentTarget

    def __init__(
        self,
        *,
        adapter: CopilotSdkHarnessAdapter,
        runtime: NativeTaskRuntime,
        run_id: str,
        store: NativeCyberEvidenceStore,
    ) -> None:
        super().__init__(runtime=runtime, run_id=run_id, harness_id=adapter.harness_id)
        connection = adapter.connect(runtime)
        self.target = connection.target
        self.session = self.target.session
        if not isinstance(self.session, CopilotSdkAgentSession):
            raise TypeError("The Copilot adapter requires its actual SDK session adapter.")
        if (
            self.session.environment_id != self.identity.environment_id
            or self.session.simulated != self.simulated
            or self.session.capabilities != adapter.capabilities
        ):
            raise ValueError("Copilot session identity or capabilities disagree with the allocated runtime.")
        self.identity = self.identity.model_copy(update={"source_session_id": self.session.session_id})
        self._bind_model = connection.bind_model_observer
        self._quiesce_agent = connection.quiesce_agent_async
        self.sink = CopilotSdkDatabaseEvidenceSink(
            store=store,
            run_id=run_id,
            session_id=self.session.session_id,
            environment_id=self.identity.environment_id,
            harness_id=self.identity.harness_id,
        )
        self.session.attach_event_observer(self.sink)

    @property
    def turn_open(self) -> bool:
        return self.sink.turn_open

    async def begin_turn_async(
        self, *, turn_index: int, started_at: datetime, response_mode: NativeCyberResponseMode
    ) -> None:
        self.verify_identity()
        observer = await self.sink.start_turn_async(
            turn_id=str(uuid4()), turn_index=turn_index, started_at=started_at, response_mode=response_mode
        )
        self._bind_model(observer)

    async def finish_turn_async(
        self,
        *,
        request_piece_ids: tuple[UUID, ...],
        response_piece_ids: tuple[UUID, ...],
        tool_request_piece_ids: tuple[UUID, ...],
        tool_result_piece_ids: tuple[UUID, ...],
        interrupted: bool,
    ) -> None:
        self.verify_identity()
        await self.sink.finish_turn_async(
            evidence=self.session.evidence(),
            request_piece_ids=request_piece_ids,
            response_piece_ids=response_piece_ids,
            tool_request_piece_ids=tool_request_piece_ids,
            tool_result_piece_ids=tool_result_piece_ids,
            interrupted=interrupted,
        )

    async def quiesce_async(self) -> None:
        await self.session.quiesce_async()
        await self._quiesce_agent()
        self.verify_identity()
        self._quiescent = True

    async def checkpoint_async(self) -> None:
        self.verify_identity()
        await self.sink.checkpoint_async()

    def evidence(self) -> NativeHarnessEvidence:
        source = self.sink.project_evidence(self.session.evidence())
        return NativeHarnessEvidence(
            identity=self.identity,
            simulated=self.simulated,
            quiescent=self._quiescent,
            coverage_complete=source.coverage_complete,
            gaps=source.gaps,
            sdk=source,
        )

    def build_report(self, context: NativeHarnessReportContext) -> NativeCyberReport:
        return NativeCyberReport(
            run_id=context.run_id,
            binding_name=context.binding_name,
            binding_version=context.binding_version,
            request=context.request,
            input_sha256=context.input_sha256,
            seed_id=context.seed_id,
            status=context.status,
            simulated=self.simulated,
            readiness=context.readiness,
            started_at=context.started_at,
            expires_at=context.expires_at,
            ended_at=context.ended_at,
            conversation_id=context.conversation_id,
            attack_result_id=context.attack_result_id,
            technique_identifier=context.technique_identifier,
            agent=self.evidence().sdk,
            judgment=context.judgment,
            cleanup=context.cleanup,
            errors=context.errors,
        )


class NativeCliConnector(Protocol):
    """Bind the admitted gateway/launcher to the allocated environment and exact recorder."""

    def __call__(self, *, runtime: NativeTaskRuntime, sink: NativeCliDatabaseEvidenceSink) -> SandboxProcessLauncher:
        """Supply a pinned sandbox launcher; no CLI or model dispatch occurs here."""
        ...


@dataclass(frozen=True, kw_only=True)
class NativeCliHarnessAdapter(NativeHarnessAdapter):
    """One-shot documented CLI JSONL with its unchanged provider wire evidence."""

    config: NativeCliRunConfig
    connect: NativeCliConnector = field(repr=False)
    capabilities: NativeAgentCapabilities = field(
        default_factory=lambda: NativeAgentCapabilities(evidence_scope="native_cli_jsonl"), init=False
    )

    def __post_init__(self) -> None:
        """
        Validate the explicitly selected native CLI protocol.

        Raises:
            ValueError: If the source protocol is unknown.
        """
        if not isinstance(self.config.protocol, NativeCliProtocol):
            raise ValueError("A documented CLI protocol is required.")

    @property
    def harness_id(self) -> str:
        """The explicit native CLI source protocol, not an SDK identity."""
        return self.config.protocol.value

    def required_raw_streams(self) -> tuple[NativeCyberRawStreamKey, ...]:
        """
        Require stdout/stderr and original model requests/responses, never SDK events.

        Returns:
            tuple[NativeCyberRawStreamKey, ...]: Exact CLI and provider source identities.
        """
        return NativeCliDatabaseEvidenceSink.required_raw_streams(
            protocol=self.config.protocol, include_model_gateway=True
        )

    async def attach_async(
        self, *, runtime: NativeTaskRuntime, run_id: str, store: NativeCyberEvidenceStore
    ) -> NativeHarnessRun:
        """
        Bind the native target/runner/parser and sink to this exact owned environment.

        Returns:
            NativeHarnessRun: This run's one-shot target and recorder.
        """
        return _CliRun(adapter=self, runtime=runtime, run_id=run_id, store=store)


class CodexCliHarnessAdapter(NativeCliHarnessAdapter):
    """Codex exec profile; item output alone cannot prove model-visible tool causality."""

    def __post_init__(self) -> None:
        """
        Require the Codex protocol.

        Raises:
            ValueError: If the selected protocol is not Codex.
        """
        if self.config.protocol is not NativeCliProtocol.CODEX_EXEC_JSON:
            raise ValueError("The Codex harness requires the Codex exec JSONL profile.")


class ClaudeCliHarnessAdapter(NativeCliHarnessAdapter):
    """Claude print profile with actual tool_use/tool_result source correlations."""

    def __post_init__(self) -> None:
        """
        Require the Claude Messages and verbose stream-JSON protocol.

        Raises:
            ValueError: If the selected protocol is not Claude.
        """
        if self.config.protocol is not NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE:
            raise ValueError("The Claude harness requires its verbose print stream-JSON profile.")


class _CliRun(NativeHarnessRun):
    target: NativeCliTarget

    def __init__(
        self,
        *,
        adapter: NativeCliHarnessAdapter,
        runtime: NativeTaskRuntime,
        run_id: str,
        store: NativeCyberEvidenceStore,
    ) -> None:
        super().__init__(runtime=runtime, run_id=run_id, harness_id=adapter.harness_id)
        self.config = adapter.config
        self.turn_id = str(uuid4())
        self.sink = NativeCliDatabaseEvidenceSink(
            store=store,
            run_id=run_id,
            turn_id=self.turn_id,
            turn_index=1,
            protocol=self.config.protocol,
            include_model_gateway=True,
            environment_id=self.identity.environment_id,
        )
        self.target = NativeCliTarget(
            run_config=self.config, launcher=adapter.connect(runtime=runtime, sink=self.sink), evidence_sink=self.sink
        )
        self._events = ()

    @property
    def turn_open(self) -> bool:
        return self.sink.turn_open

    async def begin_turn_async(
        self, *, turn_index: int, started_at: datetime, response_mode: NativeCyberResponseMode
    ) -> None:
        self.verify_identity()
        if turn_index != 1:
            raise ValueError("The CLI print/exec harness is one-shot and cannot retain an outer operator turn.")
        await self.sink.start_async(started_at=started_at, response_mode=response_mode)

    async def finish_turn_async(
        self,
        *,
        request_piece_ids: tuple[UUID, ...],
        response_piece_ids: tuple[UUID, ...],
        tool_request_piece_ids: tuple[UUID, ...],
        tool_result_piece_ids: tuple[UUID, ...],
        interrupted: bool,
    ) -> None:
        self.verify_identity()
        if interrupted:
            self.sink.fail("The native CLI task delivery was interrupted.")
        outcome = self.target.last_run.outcome if self.target.last_run else None
        try:
            await self.sink.finish_async(
                outcome=outcome,
                request_piece_ids=request_piece_ids,
                response_piece_ids=response_piece_ids,
                tool_request_piece_ids=tool_request_piece_ids,
                tool_result_piece_ids=tool_result_piece_ids,
            )
        finally:
            self._events = await self.sink.read_report_events_async(allow_partial=True)
        if outcome is not None:
            self.identity = self.identity.model_copy(update={"source_session_id": outcome.source_session_id})

    async def quiesce_async(self) -> None:
        self.verify_identity()
        if self.target.last_run is None or not self.target.stop_confirmed:
            raise RuntimeError("The owned CLI process stop barrier was not confirmed.")
        self._quiescent = True
        await self.sink.checkpoint_async()

    async def checkpoint_async(self) -> None:
        self.verify_identity()
        await self.sink.checkpoint_async()
        self._events = await self.sink.read_report_events_async(allow_partial=True)

    def evidence(self) -> NativeHarnessEvidence:
        source = project_native_cli_evidence(
            outcome=self.target.last_run.outcome if self.target.last_run else None,
            events=self._events,
            raw_evidence_ref=f"db-episode:{self.identity.run_id}",
        )
        identity = self.identity.model_copy(update={"source_session_id": source.source_session_id})
        return NativeHarnessEvidence(
            identity=identity,
            simulated=self.simulated,
            quiescent=self._quiescent,
            coverage_complete=source.coverage_complete and not self.sink.gaps,
            gaps=(*source.gaps, *self.sink.gaps),
            cli=source,
        )

    def build_report(self, context: NativeHarnessReportContext) -> NativeCliRunReport:
        status = (
            NativeCliReportStatus.COMPLETED
            if context.status is NativeCyberStatus.COMPLETED
            else NativeCliReportStatus.CANCELLED
            if context.status is NativeCyberStatus.CANCELLED
            else NativeCliReportStatus.ERROR
            if context.errors
            else NativeCliReportStatus.INCOMPLETE
        )
        judgment = None
        artifacts = ()
        if context.judgment is not None:
            retained = context.judgment.model_dump(mode="json")
            digest = hashlib.sha256(
                json.dumps(retained, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            judgment = NativeCliOriginalJudgment(
                grader_ref=f"{context.binding_name}@{context.binding_version}",
                grader_evidence_ref=f"inline-sha256:{digest}",
                retained_evidence=retained,
                value=context.judgment.value,
                complete=context.judgment.complete,
                rationale=context.judgment.rationale,
            )
            artifacts = tuple(
                NativeCliArtifactReference.model_validate(item.model_dump()) for item in context.judgment.artifacts
            )
        report = build_native_cli_run_report(
            config=self.config,
            outcome=self.target.last_run.outcome if self.target.last_run else None,
            events=self._events,
            task_id=context.binding_name,
            task_version=context.binding_version,
            run_id=self.identity.run_id,
            turn_id=self.turn_id,
            status=status,
            cleanup=NativeCliReportCleanup(context.cleanup.value),
            conversation_id=context.conversation_id,
            parent_run_id=context.request.parent_run_id,
            simulated=self.simulated,
            judgment=judgment,
            artifacts=artifacts,
            raw_evidence_ref=f"db-episode:{self.identity.run_id}",
            errors=context.errors,
        )
        return report.model_copy(
            update={"environment_id": self.identity.environment_id, "harness_id": self.identity.harness_id}
        )


@dataclass(frozen=True, kw_only=True)
class NativeCliModelObserver:
    """Whole Responses body observation for a bound CLI turn, not an SDK session."""

    identity: NativeModelCaptureIdentity
    sink: NativeCliDatabaseEvidenceSink = field(repr=False)
    _requests: dict[str, bool] = field(default_factory=dict, repr=False)

    @property
    def max_body_bytes(self) -> int:
        """The recorder's original-body cap."""
        return self.sink.max_body_bytes

    def ensure_active(self) -> None:
        """
        Check exact run/turn before allowing model progress.

        Raises:
            ValueError: If the observer is bound to another run, environment or turn.
        """
        if (
            self.identity.run_id != self.sink.run_id
            or self.identity.turn_id != self.sink.turn_id
            or self.identity.turn_index != 1
            or self.identity.harness_id != "codex_exec_json"
            or self.sink.environment_id is None
            or self.identity.environment_id != self.sink.environment_id
            or self.identity.session_id is not None
        ):
            self.fail("Model observer identity differs from its owned CLI turn.")
            raise ValueError("CLI model observer has a foreign identity.")
        self.sink.ensure_active()

    def fail(self, reason: str) -> None:
        """Latch a content-free failure in the existing recorder."""
        self.sink.fail(reason)

    async def record_request_async(self, *, request_id: str, body: bytes, accepted: bool) -> None:
        """
        Capture original ingress before model dispatch.

        Raises:
            ValueError: If admission is not an explicit boolean.
        """
        self.ensure_active()
        if type(accepted) is not bool:
            self.fail("Model request admission was not explicit.")
            raise ValueError("An explicit model admission decision is required.")
        await self.sink.record_model_body_async(request_id=request_id, body=body, accepted=accepted)
        self._requests[request_id] = accepted

    async def record_response_async(
        self, *, request_id: str, body: bytes, status_code: int, origin: NativeModelResponseOrigin, complete: bool
    ) -> None:
        """
        Capture original response bytes and explicit provenance before return.

        Raises:
            ValueError: If the response lacks a matching admitted request.
        """
        self.ensure_active()
        if request_id not in self._requests or (
            not self._requests[request_id] and origin is NativeModelResponseOrigin.PROVIDER
        ):
            self.fail("Model response has no matching admitted CLI request.")
            raise ValueError("CLI model response lacks a matching original request.")
        await self.sink.record_model_body_async(
            request_id=request_id, body=body, origin=origin, status_code=status_code, complete=complete
        )
        del self._requests[request_id]
