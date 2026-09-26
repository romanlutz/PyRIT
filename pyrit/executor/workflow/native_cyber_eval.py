# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import stat
from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from functools import cache
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

import aiofiles

from pyrit.executor.attack import PromptSendingAttack
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.memory import CentralMemory
from pyrit.models import ComponentIdentifier, ContentEntryScorable, ContentScorable, Identifiable, Message, SeedPrompt
from pyrit.models.native_cyber import (
    NativeAgentEvidence,
    NativeCyberCleanup,
    NativeCyberJudgment,
    NativeCyberReadiness,
    NativeCyberReport,
    NativeCyberRequest,
    NativeCyberRunView,
    NativeCyberStatus,
)
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.registry import ConverterRegistry
from pyrit.registry.instance_registry import DefaultInstanceRegistry, InstanceRegistry
from pyrit.scenario.core.attack_technique_factory import AttackTechniqueFactory
from pyrit.score.float_scale.native_cyber_scorer import NativeCyberReportScorer

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    from pyrit.models import AttackResult, Score
    from pyrit.prompt_target import NativeAgentTarget

logger = logging.getLogger(__name__)


class NativeCyberRuntime(Protocol):
    """One lease-owned agent environment; original grading happens before context exit."""

    target: NativeAgentTarget

    async def validate_agent_storage_async(self, *, directory: Path) -> None:
        """Verify the agent cannot access host reports or memory, including through mounts or host tools."""
        ...

    async def grade_async(self, *, evidence: NativeAgentEvidence) -> NativeCyberJudgment:
        """Acquire original grading and immutable artifacts exactly once while the environment exists."""
        ...


class NativeCyberTaskBinding(Identifiable, ABC):
    """Caller-owned task/environment/grader packaging, without private configuration in public registries."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        description: str,
        max_ttl_seconds: int = 180,
        allowed_converter_names: tuple[str, ...] = (),
        techniques: tuple[AttackTechniqueFactory, ...] = (),
    ) -> None:
        """
        Initialize public catalog metadata and allow-listed technique choices.

        Raises:
            ValueError: If identity, limits or technique names are invalid.
        """
        if not name or not version or not 1 <= max_ttl_seconds <= 3600:
            raise ValueError("A binding requires an identity and a bounded lease TTL.")
        if any(factory.name == "literal" for factory in techniques):
            raise ValueError("The literal PromptSendingAttack baseline is reserved.")
        self.name = name
        self.version = version
        self.description = description
        self.max_ttl_seconds = max_ttl_seconds
        self.allowed_converter_names = allowed_converter_names
        self.techniques = {factory.name: factory for factory in techniques}

    @abstractmethod
    async def readiness_async(self) -> NativeCyberReadiness:
        """Describe actual transport/image/auth qualification without starting the evaluated agent."""
        ...

    async def create_host_storage_async(self, *, directory: Path) -> None:
        """
        Verify the existing private parent and create only the exact absent run directory.

        Windows bindings must override this with a guarded creator that verifies
        effective parent ACLs and protected ancestry before using inherited permissions.
        A failing override owns cleanup of any empty child it created; successful
        return transfers that ownership to the controller. Never modify existing paths.

        Raises:
            PermissionError: If private parent storage cannot be established.
            OSError: If the parent is missing or the fresh child cannot be created.
        """
        await asyncio.to_thread(self._create_posix_storage, directory)

    async def validate_host_storage_async(self, *, directory: Path) -> None:
        """
        Verify private host report storage independently of the runtime's guest-exclusion check.

        Windows bindings must override this with actual ACL verification; chmod is not sufficient.

        Raises:
            PermissionError: If owner-only storage is not established.
        """
        await asyncio.to_thread(self._validate_posix_storage, directory)

    @abstractmethod
    def open_runtime(
        self, *, run_id: str, request: NativeCyberRequest
    ) -> AbstractAsyncContextManager[NativeCyberRuntime]:
        """Create one fresh owned environment; context exit must remove only its resources."""
        ...

    def seed(self, request: NativeCyberRequest) -> SeedPrompt:
        """
        Package the exact approved instruction, not an MCQ or implicit role-play template.

        Returns:
            SeedPrompt: The literal baseline seed.
        """
        return SeedPrompt(
            value=request.instruction,
            role="user",
            data_type="text",
            metadata={"native_binding": self.name, "native_binding_version": self.version},
        )

    def _build_identifier(self) -> ComponentIdentifier:
        return ComponentIdentifier.of(
            self,
            params={
                "name": self.name,
                "version": self.version,
                "max_ttl_seconds": self.max_ttl_seconds,
                "allowed_converters": list(self.allowed_converter_names),
                "techniques": ["literal", *self.techniques],
            },
        )

    def _create_posix_storage(self, directory: Path) -> None:
        self._validate_posix_directory(directory.parent)
        directory.mkdir(mode=0o700, exist_ok=False)

    def _validate_posix_storage(self, directory: Path) -> None:
        for path in (directory.parent, directory):
            self._validate_posix_directory(path)

    @staticmethod
    def _validate_posix_directory(directory: Path) -> None:
        if os.name != "posix":
            raise PermissionError("This binding must verify host directory ACLs before retaining native evidence.")
        metadata = directory.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise PermissionError("Native evidence requires caller-owned private directories with mode 0700.")


@cache
def get_native_cyber_bindings() -> InstanceRegistry[NativeCyberTaskBinding]:
    """
    Return the standard instance-registry surface for trusted initializer-installed bindings.

    Returns:
        InstanceRegistry[NativeCyberTaskBinding]: Registered binding instances.
    """
    return DefaultInstanceRegistry(instance_type=NativeCyberTaskBinding)


class NativeCyberEvaluation:
    """Own one native attack/retained-session/grading lifecycle and immutable report."""

    _CLEANUP_SECONDS = 10
    _FINAL_STATES = frozenset(
        {
            NativeCyberStatus.COMPLETED,
            NativeCyberStatus.BLOCKED,
            NativeCyberStatus.CANCELLED,
            NativeCyberStatus.EXPIRED,
            NativeCyberStatus.ERROR,
        }
    )

    def __init__(self, *, binding: NativeCyberTaskBinding, request: NativeCyberRequest, directory: Path) -> None:
        """
        Create a fresh run without opening an agent or executing a tool.

        Raises:
            ValueError: If edits exceed the binding's allow-list or lineage is not verified by ``rerun``.
        """
        if request.parent_run_id is not None:
            raise ValueError(
                "Parent lineage requires an existing evaluation's rerun(); constructor lineage is unverified."
            )
        if request.ttl_seconds > binding.max_ttl_seconds:
            raise ValueError("Requested TTL exceeds the binding's approved maximum.")
        if request.technique != "literal" and request.technique not in binding.techniques:
            raise ValueError("This binding does not permit the requested attack technique.")
        if set(request.converter_names) - set(binding.allowed_converter_names):
            raise ValueError("This binding does not permit one or more requested converters.")
        self.binding = binding
        self.request = request
        self.run_id = str(uuid4())
        self.directory = directory / self.run_id
        self.started_at = datetime.now(UTC)
        self.expires_at = self.started_at + timedelta(seconds=request.ttl_seconds)
        self.status = NativeCyberStatus.PREPARING
        self.cancel_requested = False
        self.turn_count = 0
        self.report: NativeCyberReport | None = None
        self.score: Score | None = None
        self._readiness: NativeCyberReadiness | None = None
        self._runtime: NativeCyberRuntime | None = None
        self._context: AbstractAsyncContextManager[NativeCyberRuntime] | None = None
        self._entered = False
        self._closed = False
        self._cleanup = NativeCyberCleanup.NOT_OPENED
        self._attack_result: AttackResult | None = None
        self._technique_identifier: dict[str, Any] | None = None
        self._conversation_id: str | None = None
        self._normalizer = PromptNormalizer()
        self._memory = CentralMemory.get_memory_instance()
        self._errors: list[str] = []
        self._events_written = 0
        self._lock = asyncio.Lock()
        self._expiry: asyncio.Task[None] | None = None
        self._start_called = False
        self._seed: SeedPrompt | None = None
        self._judgment: NativeCyberJudgment | None = None
        self._grading_started = False
        self._seen_environments: set[str] = set()
        self._seen_sessions: set[str] = set()
        self._directory_created = False
        self._host_storage_verified = False
        self._agent_storage_verified = False

    async def start_async(self) -> NativeCyberRunView:
        """
        Execute one native baseline/registered attack, retaining the environment if stepping is allowed.

        Returns:
            NativeCyberRunView: Current operational state and persistent references when finalized.

        Raises:
            ValueError: If the run was already started.
            RuntimeError: If incomplete native evidence cannot be retained.
            OSError: If neither report-file nor memory retention succeeds.
            asyncio.CancelledError: If the caller cancels; owned cleanup is still attempted.
        """
        async with self._lock:
            if self._start_called:
                raise ValueError("Each native run starts only once; reruns require a new evaluation.")
            self._start_called = True
            try:
                await self._settle_async(self._create_host_storage_async())
                async with asyncio.timeout(self._remaining_seconds()):
                    await self.binding.validate_host_storage_async(directory=self.directory)
                    self._host_storage_verified = True
                    self._readiness = await self.binding.readiness_async()
                if not self._readiness.ready:
                    self.status = NativeCyberStatus.BLOCKED
                    await self._publish_async(judgment=None)
                    return self.view()
                if self.request.operator_steps and (
                    not self._readiness.capabilities.operator_steps or self.request.technique != "literal"
                ):
                    raise ValueError("Operator stepping is not qualified for this binding/backend.")
                self._context = self.binding.open_runtime(run_id=self.run_id, request=self.request)
                self._runtime = await self._context.__aenter__()
                self._entered = True
                self._cleanup = NativeCyberCleanup.UNKNOWN
                async with asyncio.timeout(self._remaining_seconds()):
                    await self._runtime.validate_agent_storage_async(directory=self.directory)
                self._agent_storage_verified = True
                if self._runtime.target.session.simulated != self._readiness.simulated:
                    raise ValueError("Runtime provenance disagrees with binding qualification.")
                if self._runtime.target.session.capabilities != self._readiness.capabilities:
                    raise ValueError("Runtime capabilities disagree with qualified binding capabilities.")
                if self._runtime.target.session.environment_id in self._seen_environments:
                    raise ValueError("A fresh rerun cannot reuse a previous task environment.")
                if self._runtime.target.session.session_id in self._seen_sessions:
                    raise ValueError("A fresh rerun cannot reuse a previous native session.")
                self.status = NativeCyberStatus.RUNNING
                factory = self.binding.techniques.get(self.request.technique) or AttackTechniqueFactory(
                    name="literal",
                    attack_class=PromptSendingAttack,
                    uses_adversarial=False,
                    supports_additional_request_converters=True,
                    attack_kwargs={"max_attempts_on_failure": 0},
                )
                technique = factory.create(
                    objective_target=self._runtime.target,
                    attack_scoring_config=AttackScoringConfig(),
                    attack_converter_config_override=AttackConverterConfig(request_converters=self._converters()),
                )
                scoring = technique.attack.get_attack_scoring_config()
                if scoring and (scoring.objective_scorer or scoring.auxiliary_scorers):
                    raise ValueError("Task grading is owned by this lease; technique-level grading would duplicate it.")
                self._technique_identifier = technique.get_identifier().model_dump(mode="json")
                seed = self.binding.seed(self.request)
                self._seed = seed
                await self._memory.add_seeds_to_memory_async(seeds=[seed], added_by="native_cyber")
                async with asyncio.timeout(self._remaining_seconds()):
                    self._attack_result = await technique.attack.execute_async(
                        objective=seed.value,
                        next_message=Message.from_prompt(prompt=seed.value, role="user"),
                        memory_labels={"native_cyber_run_id": self.run_id},
                    )
                self._conversation_id = self._attack_result.conversation_id
                self.turn_count = 1
                await self._retain_events_async()
                if self.request.operator_steps and not self.cancel_requested:
                    if not self._runtime.target.session.evidence().coverage_complete:
                        raise RuntimeError("The native turn did not produce complete retained event coverage.")
                    self.status = NativeCyberStatus.AWAITING_INSTRUCTION
                    self._expiry = asyncio.create_task(self._expire_async())
                else:
                    await self._finalize_async(cancelled=self.cancel_requested)
            except (Exception, asyncio.CancelledError) as error:
                await self._fail_async(error)
                if isinstance(error, asyncio.CancelledError):
                    raise
            return self.view()

    async def step_async(self, instruction: str) -> NativeCyberRunView:
        """
        Send the next instruction only to the same still-live session and workspace.

        Returns:
            NativeCyberRunView: State after the genuine native turn.

        Raises:
            ValueError: If the run cannot accept another instruction.
            RuntimeError: If the new turn has incomplete native evidence.
            asyncio.CancelledError: If the caller cancels; owned cleanup is still attempted.
        """
        async with self._lock:
            if not self.view().can_step or not instruction.strip() or len(instruction) > 32768:
                raise ValueError("No compatible unexpired idle native session can accept this instruction.")
            assert self._runtime is not None and self._conversation_id is not None
            self.status = NativeCyberStatus.RUNNING
            try:
                async with asyncio.timeout(min(self.request.turn_timeout_seconds, self._remaining_seconds())):
                    await self._normalizer.send_prompt_async(
                        message=Message.from_prompt(prompt=instruction, role="user"),
                        target=self._runtime.target,
                        conversation_id=self._conversation_id,
                        request_converter_configurations=self._converters(),
                    )
                self.turn_count += 1
                await self._retain_events_async()
                self.status = NativeCyberStatus.AWAITING_INSTRUCTION
                if not self._runtime.target.session.evidence().coverage_complete:
                    raise RuntimeError("The next native turn did not produce complete event coverage.")
                if self.cancel_requested:
                    await self._finalize_async(cancelled=True)
            except (Exception, asyncio.CancelledError) as error:
                await self._fail_async(error)
                if isinstance(error, asyncio.CancelledError):
                    raise
            return self.view()

    async def finish_async(self) -> NativeCyberRunView:
        """
        Acquire one original judgment before releasing the environment.

        Returns:
            NativeCyberRunView: Final persistent outcome.

        Raises:
            ValueError: If a turn is currently running or the run is not started.
            asyncio.CancelledError: If the caller cancels; owned cleanup is still attempted.
        """
        async with self._lock:
            if self.status in self._FINAL_STATES:
                return self.view()
            if self.status is not NativeCyberStatus.AWAITING_INSTRUCTION:
                raise ValueError("Finish is available only between completed native turns.")
            try:
                await self._finalize_async(cancelled=self.cancel_requested)
            except (Exception, asyncio.CancelledError) as error:
                await self._fail_async(error)
                if isinstance(error, asyncio.CancelledError):
                    raise
            return self.view()

    def rerun(self, *, request: NativeCyberRequest) -> NativeCyberEvaluation:
        """
        Create a fresh run with immutable parent lineage, never cloning a live session.

        Cross-restart reruns without this controller's verified ancestry are not supported.

        Returns:
            NativeCyberEvaluation: An unstarted child run with a fresh identity.

        Raises:
            ValueError: If the parent is unfinished or lineage points elsewhere.
        """
        if self.report is None:
            raise ValueError("A fresh rerun requires an immutable completed parent report.")
        if request.parent_run_id not in {None, self.run_id}:
            raise ValueError("Rerun lineage must reference the source run.")
        child = NativeCyberEvaluation(
            binding=self.binding,
            request=request.model_copy(update={"parent_run_id": None}),
            directory=self.directory.parent,
        )
        child.request = request.model_copy(update={"parent_run_id": self.run_id})
        child._seen_environments = self._seen_environments.copy()
        child._seen_sessions = self._seen_sessions.copy()
        if self.report.agent:
            child._seen_environments.add(self.report.agent.environment_id)
            child._seen_sessions.add(self.report.agent.session_id)
        return child

    async def cancel_async(self) -> NativeCyberRunView:
        """
        Request cancellation at the next native turn boundary, not in the middle of a tool.

        Returns:
            NativeCyberRunView: A visible cancellation request or final cancelled state.
        """
        if self.status in self._FINAL_STATES:
            return self.view()
        self.cancel_requested = True
        if self.status is NativeCyberStatus.AWAITING_INSTRUCTION:
            return await self.finish_async()
        return self.view()

    def view(self) -> NativeCyberRunView:
        """
        Describe supported actions without exposing private raw evidence.

        Returns:
            NativeCyberRunView: Capability-gated operational state.
        """
        from pyrit.models.native_cyber import NativeAgentCapabilities

        capabilities = self._readiness.capabilities if self._readiness else NativeAgentCapabilities()
        content_id = (
            str(self.score.scorable.content_id)
            if self.score and isinstance(self.score.scorable, ContentEntryScorable)
            else None
        )
        return NativeCyberRunView(
            run_id=self.run_id,
            binding_name=self.binding.name,
            status=self.status,
            expires_at=self.expires_at,
            capabilities=capabilities,
            can_step=(
                self.status is NativeCyberStatus.AWAITING_INSTRUCTION
                and not self.cancel_requested
                and capabilities.operator_steps
                and self.turn_count < capabilities.max_turns
                and datetime.now(UTC) < self.expires_at
            ),
            cancel_requested=self.cancel_requested,
            turn_count=self.turn_count,
            parent_run_id=self.request.parent_run_id,
            conversation_id=self._conversation_id,
            score_id=str(self.score.id) if self.score else None,
            content_id=content_id,
            report_sha256=self.report.sha256() if self.report else None,
            blockers=self._readiness.blockers if self._readiness else (),
        )

    def _converters(self) -> list[ConverterConfiguration]:
        converters = []
        for name in self.request.converter_names:
            converter = ConverterRegistry.get_registry_singleton().instances.get(name)
            if converter is None:
                raise ValueError(f"Converter {name!r} is no longer registered.")
            converters.append(ConverterConfiguration(converters=[converter]))
        return converters

    def _remaining_seconds(self) -> float:
        remaining = (self.expires_at - datetime.now(UTC)).total_seconds()
        if remaining <= 0:
            raise TimeoutError("The native environment lease expired.")
        return remaining

    async def _create_host_storage_async(self) -> None:
        await self.binding.create_host_storage_async(directory=self.directory)
        self._directory_created = True

    async def _finalize_async(self, *, cancelled: bool = False) -> None:
        if self.report is not None:
            return
        self.status = NativeCyberStatus.FINALIZING
        assert self._runtime is not None
        await self._runtime.target.session.quiesce_async()
        await self._retain_events_async()
        evidence = self._runtime.target.session.evidence()
        if not cancelled and evidence.coverage_complete and evidence.idle:
            if self._grading_started:
                raise RuntimeError("Original grading was already attempted; it cannot be retried implicitly.")
            self._grading_started = True
            async with asyncio.timeout(self._remaining_seconds()):
                self._judgment = await self._runtime.grade_async(evidence=evidence)
        elif not cancelled:
            self._errors.extend(evidence.gaps)
        await self._settle_async(self._close_async())
        self.status = (
            NativeCyberStatus.CANCELLED
            if cancelled
            else NativeCyberStatus.COMPLETED
            if self._judgment and self._judgment.complete and self._cleanup == "closed" and not self._errors
            else NativeCyberStatus.ERROR
        )
        await self._publish_async(judgment=self._judgment)

    async def _close_async(self) -> None:
        if self._closed or not self._entered:
            return
        self._closed = True
        assert self._context is not None
        try:
            async with asyncio.timeout(self._CLEANUP_SECONDS):
                await self._context.__aexit__(None, None, None)
            self._cleanup = NativeCyberCleanup.CLOSED
        except (Exception, asyncio.CancelledError) as error:
            self._cleanup = NativeCyberCleanup.FAILED
            self._errors.append(f"Environment cleanup failed: {type(error).__name__}: {error}")
            raise

    async def _retain_events_async(self) -> None:
        if self._runtime is None or not self._agent_storage_verified:
            return
        events = self._runtime.target.session.evidence().events
        async with aiofiles.open(self.directory / "native-events.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            for event in events[self._events_written :]:
                await stream.write(event.model_dump_json() + "\n")
            await stream.flush()
        self._events_written = len(events)

    async def _fail_async(self, error: BaseException) -> None:
        try:
            await self._settle_async(self._record_failure_async(error))
        except (Exception, asyncio.CancelledError) as retention_error:
            if isinstance(error, asyncio.CancelledError) and retention_error is not error:
                raise error from retention_error
            raise

    async def _record_failure_async(self, error: BaseException) -> None:
        self._errors.append(f"{type(error).__name__}: {error}")
        self.status = (
            NativeCyberStatus.CANCELLED
            if isinstance(error, asyncio.CancelledError)
            else NativeCyberStatus.EXPIRED
            if isinstance(error, TimeoutError)
            else NativeCyberStatus.ERROR
        )
        try:
            await self._retain_events_async()
        finally:
            try:
                await self._close_async()
            except (Exception, asyncio.CancelledError):
                logger.exception("Native evaluation cleanup was not confirmed.")
            if (
                self._directory_created
                and not self._can_write_report_file()
                and (not self._entered or self._cleanup is NativeCyberCleanup.CLOSED)
            ):
                try:
                    await asyncio.to_thread(self.directory.rmdir)
                except OSError as directory_error:
                    self._errors.append(f"Empty report directory cleanup failed: {directory_error}")
                    logger.exception("Native report directory cleanup was not confirmed.")
            await self._publish_async(judgment=self._judgment)

    async def _settle_async(self, operation: Coroutine[Any, Any, None]) -> None:
        task = asyncio.create_task(operation)
        deadline = asyncio.get_running_loop().time() + self._CLEANUP_SECONDS + 5
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                task.cancel()
                task.add_done_callback(self._settled)
                error = TimeoutError("Native cleanup/evidence retention exceeded its bounded grace.")
                if cancellation:
                    raise cancellation from error
                raise error
            try:
                await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
                self.cancel_requested = True
        try:
            task.result()
        except (Exception, asyncio.CancelledError) as error:
            if cancellation:
                raise cancellation from error
            raise
        if cancellation:
            raise cancellation

    @staticmethod
    def _settled(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception():
            logger.error("Native cleanup remained failed after the bounded wait: %s", task.exception())

    async def _publish_async(self, *, judgment: NativeCyberJudgment | None) -> None:
        if self.report is not None:
            return
        if self._entered and not self._agent_storage_verified and self._cleanup is not NativeCyberCleanup.CLOSED:
            raise PermissionError("Cannot retain raw evidence while an unverified agent storage boundary remains open.")
        report = self._build_report(judgment)
        if self._can_write_report_file():
            try:
                async with aiofiles.open(
                    self.directory / f"{report.sha256()}.json", "x", encoding="utf-8", newline="\n"
                ) as stream:
                    await stream.write(report.canonical_json())
                    await stream.flush()
            except OSError as error:
                self.status = NativeCyberStatus.ERROR
                self._errors.append(f"Report file retention failed: {type(error).__name__}: {error}")
                logger.exception("Native report file retention failed; attempting undetermined memory retention.")
                report = self._build_report(judgment)
        scores = await NativeCyberReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json())
        )
        self.score = scores[0]
        self.report = report
        if self._expiry is not None and self._expiry is not asyncio.current_task():
            self._expiry.cancel()

    def _build_report(self, judgment: NativeCyberJudgment | None) -> NativeCyberReport:
        return NativeCyberReport(
            run_id=self.run_id,
            binding_name=self.binding.name,
            binding_version=self.binding.version,
            request=self.request,
            input_sha256=hashlib.sha256(
                (self._seed.value if self._seed else self.request.instruction).encode()
            ).hexdigest(),
            seed_id=str(self._seed.id) if self._seed else None,
            status=self.status,
            simulated=self._readiness.simulated if self._readiness else None,
            readiness=self._readiness,
            started_at=self.started_at,
            expires_at=self.expires_at,
            ended_at=datetime.now(UTC),
            conversation_id=self._conversation_id,
            attack_result_id=self._attack_result.attack_result_id if self._attack_result else None,
            technique_identifier=self._technique_identifier,
            agent=self._runtime.target.session.evidence() if self._runtime else None,
            judgment=judgment,
            cleanup=self._cleanup,
            errors=tuple(self._errors),
        )

    def _can_write_report_file(self) -> bool:
        return self._host_storage_verified and (not self._entered or self._agent_storage_verified)

    async def _expire_async(self) -> None:
        await asyncio.sleep(max(0, (self.expires_at - datetime.now(UTC)).total_seconds()))
        async with self._lock:
            if self.status not in self._FINAL_STATES:
                await self._fail_async(TimeoutError("The retained native session TTL expired."))
