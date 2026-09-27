# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-shot native CLI evaluation with caller-owned sandbox, grader, and evidence."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
from pyrit.executor.workflow.native_cli_report_adapter import build_native_cli_run_report
from pyrit.models import Message, Score, ScoringExpectation
from pyrit.models.environment_lease import EnvironmentLeaseSnapshot, EnvironmentLeaseState
from pyrit.models.native_cli_report import (
    NativeCliArtifactReference,
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportStatus,
    NativeCliRunReport,
)
from pyrit.models.native_cyber_evidence import (
    NativeCyberCoverageAssessment,
    NativeCyberCoveragePhase,
    NativeCyberEpisodeSnapshot,
    NativeCyberEpisodeStart,
    NativeCyberResponseMode,
    NativeCyberResponsePolicy,
)
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import NativeCliTarget
from pyrit.prompt_target.native_cli_models import NativeCliRunConfig
from pyrit.prompt_target.native_cli_transport import SandboxProcessLauncher, SandboxProcessSession
from pyrit.score.float_scale.native_cli_report_scorer import build_native_cli_report_score

if TYPE_CHECKING:
    from collections.abc import Coroutine

    from pyrit.memory.memory_interface import MemoryInterface
    from pyrit.prompt_target.gateway.run_listener import GatewayBridgeBinding
ResultT = TypeVar("ResultT")


@runtime_checkable
class NativeCliAgentStopObservation(Protocol):
    """Stop-only provider receipt for one executed CLI and preserved target services."""

    container_id: str
    stopped: bool
    exec_id: str | None
    preserved_services: tuple[tuple[str, str], ...]
    error: str | None


class NativeCliStopLease(Protocol):
    """A stop-only lease whose observation is produced by the verified provider."""

    run_id: str

    @property
    def agent_stop(self) -> NativeCliAgentStopObservation | None:
        """The latched provider receipt, not a caller assertion of successful stop."""
        ...

    def snapshot(self) -> EnvironmentLeaseSnapshot:
        """Describe the owned services and whether the lease remains ready."""
        ...

    async def stop_agent_async(self) -> None:
        """Settle the provider's latched stop barrier without shutting down the target."""
        ...

    async def close_async(self) -> None:
        """Release the entire owned environment after original grading."""
        ...


@runtime_checkable
class NativeCliAttributedProcessSession(SandboxProcessSession, Protocol):
    """Provider-issued execution identity that the stop-only lease must later corroborate."""

    @property
    def exec_id(self) -> str:
        """The provider-observed CLI process identity."""
        ...

    @property
    def container_id(self) -> str:
        """The provider-observed agent resource identity."""
        ...


class _AttributedLauncher:
    """Observe which exact process the injected sandbox launcher returned."""

    def __init__(self, *, delegate: SandboxProcessLauncher) -> None:
        self._delegate = delegate
        self.process_identity: tuple[str, str] | None = None

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> NativeCliAttributedProcessSession:
        """
        Reject unattributed or repeated guest processes before accepting output.

        Returns:
            NativeCliAttributedProcessSession: The provider-attributed guest process.

        Raises:
            RuntimeError: If the launcher returned no provider-issued process identity.
        """
        if self.process_identity is not None:
            raise RuntimeError("A native CLI run cannot launch a second provider process.")
        session = await self._delegate.launch_async(config=config, prompt=prompt)
        if (
            not isinstance(session, NativeCliAttributedProcessSession)
            or not isinstance(session.exec_id, str)
            or not session.exec_id.strip()
            or not isinstance(session.container_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", session.container_id) is None
        ):
            failure = RuntimeError("The CLI launcher returned no provider-attributed guest process.")
            try:
                await session.stop_async()
            except (Exception, asyncio.CancelledError) as cleanup_error:
                failure.add_note("Stopping the unattributed CLI process was not confirmed.")
                raise failure from cleanup_error
            raise failure
        self.process_identity = session.exec_id, session.container_id
        return session


class NativeCliGatewayListener(Protocol):
    """The binding-owned, already-running, run-scoped model-only listener."""

    binding: GatewayBridgeBinding
    base_url: str

    @property
    def is_locally_running(self) -> bool:
        """Whether the listener has a live server and not just an allocated address."""
        ...

    async def close_async(self) -> None:
        """Stop this run's model listener and release its private socket."""
        ...


@dataclass(frozen=True, kw_only=True)
class NativeCliOriginalAssessment:
    """The binding's original grader output and caller-retained artifact references."""

    judgment: NativeCliOriginalJudgment
    artifacts: tuple[NativeCliArtifactReference, ...] = ()


class NativeCliEvaluationRuntime(Protocol):
    """Trusted task-owned resources, qualified before the CLI is permitted to run."""

    lease: NativeCliStopLease
    launcher: SandboxProcessLauncher
    gateway_listener: NativeCliGatewayListener

    async def verify_guest_exclusion_async(self) -> None:
        """Verify the guest cannot access the host grader, evidence DB, or credentials."""
        ...

    async def grade_async(self, *, report: NativeCliRunReport) -> NativeCliOriginalAssessment:
        """Acquire original target-side grading once, after verified agent stop."""
        ...


class NativeCliEvaluationBinding(Protocol):
    """Trusted task configuration; no model key or shell callback crosses this boundary."""

    name: str
    version: str
    task_id: str
    task_version: str
    simulated: bool
    run_config: NativeCliRunConfig
    response_policy: NativeCyberResponsePolicy
    response_mode: NativeCyberResponseMode
    request_converters: tuple[ConverterConfiguration, ...]
    runtime_timeout_seconds: float
    grader_timeout_seconds: float
    cleanup_timeout_seconds: float
    raw_byte_limit: int

    async def open_runtime_async(
        self, *, run_id: str, sink: NativeCliDatabaseEvidenceSink
    ) -> NativeCliEvaluationRuntime:
        """Acquire the stop-only lease and gateway, rolling back failed acquisition."""
        ...


@dataclass(frozen=True, kw_only=True)
class NativeCliEvaluationResult:
    """The one atomic DB publication, not the unpersisted score candidate."""

    report: NativeCliRunReport
    score: Score
    episode: NativeCyberEpisodeSnapshot
    pregrading: NativeCyberCoverageAssessment | None
    cancellation_after_commit_started: bool = False


class NativeCliEvaluation:
    """Run one prepared CLI turn, then stop, pregrade, grade, clean up, and publish."""

    def __init__(
        self,
        *,
        binding: NativeCliEvaluationBinding,
        instruction: str,
        memory: MemoryInterface,
        normalizer: PromptNormalizer | None = None,
        expectation: ScoringExpectation | None = None,
    ) -> None:
        """
        Bind a fresh caller-owned task without opening an environment or writing evidence.

        Raises:
            ValueError: If task provenance, instruction, grading bound, or memory differs.
        """
        self._validate_binding(binding=binding, instruction=instruction, expectation=expectation)
        self._binding = binding
        self._instruction = instruction
        self._memory = memory
        self._normalizer = normalizer if normalizer is not None else PromptNormalizer()
        if self._normalizer.memory is not memory:
            raise ValueError("Native CLI normalization and evidence capture must use the same memory instance.")
        self._expectation = expectation
        self._store = memory.native_cyber_evidence
        self.run_id, self.turn_id, self.conversation_id = str(uuid4()), str(uuid4()), str(uuid4())
        self._sink = NativeCliDatabaseEvidenceSink(
            store=self._store,
            run_id=self.run_id,
            turn_id=self.turn_id,
            turn_index=1,
            protocol=binding.run_config.protocol,
            include_model_gateway=True,
        )
        self._started = False
        self._sink_started = False
        self._finish_attempted = False
        self._sink_finished = False
        self._stop_attempted = False
        self._stop_confirmed = False
        self._send_returned = False
        self._grading_started = False
        self._runtime: NativeCliEvaluationRuntime | None = None
        self._attributed_launcher: _AttributedLauncher | None = None
        self._target: NativeCliTarget | None = None
        self._pregrading: NativeCyberCoverageAssessment | None = None
        self._judgment: NativeCliOriginalJudgment | None = None
        self._artifacts: tuple[NativeCliArtifactReference, ...] = ()
        self._cleanup = NativeCliReportCleanup.NOT_OPENED
        self._errors: list[str] = []
        self._fatal = False
        self._cancelled: asyncio.CancelledError | None = None
        self._publication_started = False
        self._late_cancellation = False
        self.report: NativeCliRunReport | None = None
        self.score: Score | None = None
        self.episode: NativeCyberEpisodeSnapshot | None = None

    async def run_async(self) -> NativeCliEvaluationResult:
        """
        Execute one captured CLI turn and atomically publish its verified final result.

        Returns:
            NativeCliEvaluationResult: Persisted report, actual stored score, and episode.

        Raises:
            RuntimeError: If reused or if evidence cannot be atomically retained.
            asyncio.CancelledError: After cleanup and undetermined publication on cancellation.
        """
        if self._started:
            raise RuntimeError("A native CLI evaluation cannot be run twice.")
        self._started = True
        await self._settle_async(self._start_episode_async())
        try:
            if self._cancelled is None:
                await self._capture_and_grade_async()
        except asyncio.CancelledError as error:
            self._cancelled = error
            self._record_failure(stage="Native CLI evaluation cancellation", error=error)
        except Exception as error:
            self._record_failure(stage="Native CLI evaluation", error=error)
        finally:
            try:
                await self._settle_async(self._stop_and_finish_async())
            finally:
                await self._settle_async(self._close_runtime_async())
        result = await self._settle_async(self._publish_async())
        if self._cancelled is not None:
            raise self._cancelled
        return result

    async def _start_episode_async(self) -> None:
        binding = self._binding
        start = NativeCyberEpisodeStart(
            run_id=self.run_id,
            binding_name=binding.name,
            binding_version=binding.version,
            task_id=binding.task_id,
            task_version=binding.task_version,
            simulated=binding.simulated,
            required_raw_streams=self._sink.required_raw_streams(
                protocol=binding.run_config.protocol, include_model_gateway=True
            ),
            response_policy=binding.response_policy,
            require_separate_tool_results=True,
            raw_byte_limit=binding.raw_byte_limit,
        )
        await asyncio.to_thread(self._store.create_episode, start=start)

    async def _capture_and_grade_async(self) -> None:
        self._sink_started = True
        await self._sink.start_async(started_at=datetime.now(UTC), response_mode=self._binding.response_mode)
        self._cleanup = NativeCliReportCleanup.UNKNOWN
        async with asyncio.timeout(self._binding.runtime_timeout_seconds):
            self._runtime = await self._binding.open_runtime_async(run_id=self.run_id, sink=self._sink)
            self._validate_runtime()
            await self._runtime.verify_guest_exclusion_async()
            self._validate_runtime()
        self._attributed_launcher = _AttributedLauncher(delegate=self._runtime.launcher)
        self._target = NativeCliTarget(
            run_config=self._binding.run_config,
            launcher=self._attributed_launcher,
            evidence_sink=self._sink,
        )
        try:
            await self._normalizer.send_prompt_async(
                message=Message.from_prompt(prompt=self._instruction, role="user"),
                target=self._target,
                conversation_id=self.conversation_id,
                request_converter_configurations=list(self._binding.request_converters),
            )
            self._send_returned = True
        finally:
            await self._settle_async(self._stop_and_finish_async())
        if not self._stop_confirmed or not self._sink_finished or self._fatal or self._cancelled is not None:
            return
        await self._pregrade_and_grade_async()

    async def _stop_and_finish_async(self) -> None:
        if self._runtime is not None and self._target is not None and not self._stop_attempted:
            self._stop_attempted = True
            try:
                async with asyncio.timeout(self._binding.run_config.timeout_seconds):
                    await self._runtime.lease.stop_agent_async()
                self._require_stopped()
                self._stop_confirmed = True
            except (Exception, asyncio.CancelledError) as error:
                self._record_failure(stage="Native CLI agent stop", error=error)
        if self._sink_started and not self._finish_attempted:
            self._finish_attempted = True
            await self._finish_capture_async()

    async def _finish_capture_async(self) -> None:
        try:
            request_ids, response_ids = await self._persisted_pieces_async()
        except (Exception, asyncio.CancelledError) as error:
            self._record_failure(stage="Native CLI message evidence lookup", error=error)
            request_ids, response_ids = (), ()
        try:
            await self._sink.finish_async(
                outcome=self._target.last_run.outcome if self._target and self._target.last_run else None,
                request_piece_ids=request_ids,
                response_piece_ids=response_ids,
            )
            self._sink_finished = True
        except (Exception, asyncio.CancelledError) as error:
            self._record_failure(stage="Native CLI evidence sealing", error=error)

    async def _persisted_pieces_async(self) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
        messages = await asyncio.to_thread(self._memory.get_conversation_messages, conversation_id=self.conversation_id)
        requests = [piece for message in messages if message.api_role == "user" for piece in message.message_pieces]
        responses = [
            piece
            for message in messages
            if message.api_role == "assistant"
            for piece in message.message_pieces
            if piece.response_error == "none" and not piece.is_simulated
        ]
        if len(requests) > 1:
            raise ValueError("A one-shot CLI turn cannot link more than one genuine user request.")
        if self._send_returned and self._target is not None and self._target.last_run is not None:
            sources = self._target.last_run.response_sources
            if len(responses) != len(sources) or any(
                piece.original_value != source.text
                or piece.prompt_metadata.get("native_cli_source_event_id") != source.source_event_id
                or piece.prompt_metadata.get("native_cli_source_kind") != source.kind.value
                for piece, source in zip(responses, sources, strict=True)
            ):
                self._record_failure(
                    stage="Native CLI message evidence mismatch",
                    error=ValueError("Persisted response differs from observed provider output."),
                )
                responses.clear()
        elif responses:
            self._record_failure(
                stage="Native CLI message evidence mismatch",
                error=ValueError("An unreturned CLI response cannot be linked to observed output."),
            )
            responses.clear()
        return tuple(piece.id for piece in requests), tuple(piece.id for piece in responses)

    async def _pregrade_and_grade_async(self) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Original grading requires a live, qualified runtime.")
        self._require_stopped()
        draft = await self._build_report_async(
            status=NativeCliReportStatus.INCOMPLETE, cleanup=NativeCliReportCleanup.UNKNOWN
        )
        assessment = await asyncio.to_thread(self._store.assess_cli_pregrading_coverage, report=draft, expected_turns=1)
        self._pregrading = assessment
        if assessment.phase is not NativeCyberCoveragePhase.PREGRADING:
            raise ValueError("CLI evidence store did not return a pregrading assessment.")
        if not assessment.required_complete:
            self._errors.append(f"CLI pregrading evidence has {len(assessment.required_gaps)} required gap(s).")
            return
        self._require_stopped()
        if self._grading_started:
            raise RuntimeError("An original CLI grader cannot be invoked twice for one episode.")
        self._grading_started = True
        async with asyncio.timeout(self._binding.grader_timeout_seconds):
            acquired = await runtime.grade_async(report=draft)
        if not isinstance(acquired, NativeCliOriginalAssessment):
            raise TypeError("Original CLI grader returned no typed judgment and artifact references.")
        self._judgment, self._artifacts = acquired.judgment, acquired.artifacts
        if not acquired.judgment.complete:
            self._errors.append("Original CLI grader did not acquire a complete judgment.")
        if self._binding.response_mode is NativeCyberResponseMode.ARTIFACT_ONLY and not self._artifacts:
            self._errors.append("Artifact-only CLI grading has no retained artifact reference.")

    def _require_stopped(self) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("An agent stop receipt requires the active trusted runtime.")
        snapshot = runtime.lease.snapshot()
        stop = runtime.lease.agent_stop
        launched = self._attributed_launcher.process_identity if self._attributed_launcher else None
        if not isinstance(snapshot, EnvironmentLeaseSnapshot):
            raise RuntimeError("The stop-only lease has no typed environment observation.")
        expected = {service.name for service in snapshot.services if "agent" not in service.roles}
        preserved = dict(stop.preserved_services) if isinstance(stop, NativeCliAgentStopObservation) else {}
        if (
            snapshot.run_id != self.run_id
            or snapshot.state is not EnvironmentLeaseState.READY
            or not isinstance(stop, NativeCliAgentStopObservation)
            or stop.stopped is not True
            or stop.error is not None
            or launched is None
            or (stop.exec_id, stop.container_id) != launched
            or re.fullmatch(r"[0-9a-f]{64}", stop.container_id) is None
            or not expected
            or not any("target" in service.roles or "grader" in service.roles for service in snapshot.services)
            or len(preserved) != len(stop.preserved_services)
            or set(preserved) != expected
            or stop.container_id in preserved.values()
            or any(re.fullmatch(r"[0-9a-f]{64}", value) is None for value in preserved.values())
            or not runtime.gateway_listener.is_locally_running
        ):
            raise RuntimeError("Agent stop and preserved target/grader services were not proven before grading.")

    def _validate_runtime(self) -> None:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("A trusted, acquired runtime is required.")
        route = urlsplit(self._binding.run_config.model_gateway_endpoint)
        listener = urlsplit(runtime.gateway_listener.base_url)
        if (
            runtime.lease.run_id != self.run_id
            or runtime.lease.snapshot().state is not EnvironmentLeaseState.READY
            or runtime.gateway_listener.binding.run_id != self.run_id
            or not runtime.gateway_listener.is_locally_running
            or (route.scheme, route.hostname, route.port) != (listener.scheme, listener.hostname, listener.port)
            or runtime.lease.agent_stop is not None
        ):
            raise RuntimeError("CLI lease, listener, gateway route, and one-shot state must match this run.")

    async def _close_runtime_async(self) -> None:
        runtime = self._runtime
        if runtime is None:
            return
        failed = False
        for stage, close in (
            ("Native CLI gateway shutdown", runtime.gateway_listener.close_async),
            ("Native CLI environment cleanup", runtime.lease.close_async),
        ):
            try:
                async with asyncio.timeout(self._binding.cleanup_timeout_seconds):
                    await close()
            except (Exception, asyncio.CancelledError) as error:
                failed = True
                self._record_failure(stage=stage, error=error)
        try:
            closed = (
                runtime.lease.snapshot().state is EnvironmentLeaseState.CLOSED
                and not runtime.gateway_listener.is_locally_running
            )
        except (Exception, asyncio.CancelledError) as error:
            self._record_failure(stage="Native CLI cleanup verification", error=error)
            closed = False
        if failed or not closed:
            self._cleanup = NativeCliReportCleanup.FAILED
            self._errors.append("Native CLI gateway or environment cleanup was not confirmed.")
        else:
            self._cleanup = NativeCliReportCleanup.CLOSED

    async def _build_report_async(
        self, *, status: NativeCliReportStatus, cleanup: NativeCliReportCleanup
    ) -> NativeCliRunReport:
        outcome = self._target.last_run.outcome if self._target and self._target.last_run else None
        events = ()
        if self._sink_finished:
            try:
                events = await self._sink.read_report_events_async()
            except (Exception, asyncio.CancelledError) as error:
                self._record_failure(stage="Native CLI event summary retrieval", error=error)
        if outcome is not None and (not self._sink_finished or not events):
            outcome = replace(
                outcome,
                coverage_complete=False,
                gaps=(*outcome.gaps, "CLI database event summaries were not completely available."),
            )
        return build_native_cli_run_report(
            config=self._binding.run_config,
            outcome=outcome,
            events=events,
            task_id=self._binding.task_id,
            task_version=self._binding.task_version,
            run_id=self.run_id,
            turn_id=self.turn_id,
            conversation_id=self.conversation_id if self._send_returned else None,
            status=status,
            cleanup=cleanup,
            simulated=self._binding.simulated,
            judgment=self._judgment,
            artifacts=self._artifacts,
            raw_evidence_ref=f"db-episode:{self.run_id}" if self._sink_finished else None,
            errors=tuple(self._errors),
        )

    async def _publish_async(self) -> NativeCliEvaluationResult:
        draft = await self._build_report_async(status=NativeCliReportStatus.INCOMPLETE, cleanup=self._cleanup)
        final_status = self._final_status()
        report = (
            draft
            if final_status is NativeCliReportStatus.INCOMPLETE
            else NativeCliRunReport.model_validate(
                {**draft.model_dump(mode="json"), "status": final_status.value, "errors": tuple(self._errors)}
            )
        )
        score = build_native_cli_report_score(report=report, expectation=self._expectation)
        self.report = report
        self._publication_started = True
        episode = await asyncio.to_thread(
            self._store.finalize_cli_episode_atomic,
            report=report,
            score=score,
            expected_turns=1,
        )
        if episode.score_id != score.id or episode.report_sha256 != report.sha256():
            raise RuntimeError("Atomic CLI publication returned a different score or canonical report.")
        stored = await asyncio.to_thread(self._memory.get_scores, score_ids=[str(score.id)])
        if len(stored) != 1 or stored[0].status is not episode.score_status:
            raise RuntimeError("Final CLI Score was not persisted with the episode.")
        self.episode, self.score = episode, stored[0]
        return NativeCliEvaluationResult(
            report=report,
            score=stored[0],
            episode=episode,
            pregrading=self._pregrading,
            cancellation_after_commit_started=self._late_cancellation,
        )

    def _final_status(self) -> NativeCliReportStatus:
        if self._cancelled is not None:
            return NativeCliReportStatus.CANCELLED
        if self._fatal or self._cleanup is NativeCliReportCleanup.FAILED:
            return NativeCliReportStatus.ERROR
        if (
            self._pregrading is not None
            and self._pregrading.required_complete
            and self._judgment is not None
            and self._judgment.complete
            and (self._binding.response_mode is not NativeCyberResponseMode.ARTIFACT_ONLY or self._artifacts)
            and self._cleanup is NativeCliReportCleanup.CLOSED
            and not self._errors
        ):
            return NativeCliReportStatus.COMPLETED
        return NativeCliReportStatus.INCOMPLETE

    async def _settle_async(self, operation: Coroutine[object, object, ResultT]) -> ResultT:
        task = asyncio.create_task(operation)
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as error:
                if self._publication_started:
                    self._late_cancellation = True
                    continue
                self._cancelled = self._cancelled or error
                self._record_failure(stage="Native CLI evaluation cancellation", error=error)
            except Exception:
                break
        return task.result()

    def _record_failure(self, *, stage: str, error: BaseException) -> None:
        detail = f"{stage} failed ({type(error).__name__})."
        if detail not in self._errors:
            self._errors.append(detail)
        self._fatal = True

    @staticmethod
    def _validate_binding(
        *, binding: NativeCliEvaluationBinding, instruction: str, expectation: ScoringExpectation | None
    ) -> None:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("A native CLI evaluation requires one nonempty approved instruction.")
        ScoringExpectation.validate_type(expectation)
        if (
            not isinstance(binding.name, str)
            or not binding.name.strip()
            or not isinstance(binding.version, str)
            or not binding.version.strip()
            or not isinstance(binding.task_id, str)
            or not binding.task_id.strip()
            or not isinstance(binding.task_version, str)
            or not binding.task_version.strip()
            or not isinstance(binding.run_config, NativeCliRunConfig)
            or type(binding.simulated) is not bool
            or not isinstance(binding.response_policy, NativeCyberResponsePolicy)
            or not isinstance(binding.response_mode, NativeCyberResponseMode)
            or type(binding.raw_byte_limit) is not int
            or not 1 <= binding.raw_byte_limit <= 1_099_511_627_776
            or not isinstance(binding.request_converters, tuple)
            or type(binding.runtime_timeout_seconds) not in (int, float)
            or not math.isfinite(binding.runtime_timeout_seconds)
            or binding.runtime_timeout_seconds <= 0
            or type(binding.grader_timeout_seconds) not in (int, float)
            or not math.isfinite(binding.grader_timeout_seconds)
            or binding.grader_timeout_seconds <= 0
            or type(binding.cleanup_timeout_seconds) not in (int, float)
            or not math.isfinite(binding.cleanup_timeout_seconds)
            or binding.cleanup_timeout_seconds <= 0
        ):
            raise ValueError("CLI task provenance, response mode, converters, and grading bounds must be qualified.")
        if (
            binding.response_mode is NativeCyberResponseMode.ARTIFACT_ONLY
            and not binding.response_policy.allow_artifact_only
        ):
            raise ValueError("Artifact-only grading requires a task-owned response policy.")
