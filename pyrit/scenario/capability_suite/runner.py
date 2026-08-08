# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Capability-suite runner: bounded-concurrency, attempt/epoch lifecycle orchestration.

The runner owns exactly the concerns a capability suite adds on top of a single
``CapabilityTaskExecutor`` run: fresh-sandbox-per-attempt lifecycle, staging assets and
setup, binding tools, scoring while the sandbox is still alive, always cleaning up
(while distinguishing a cleanup failure from a run failure), bounded concurrency across
case x epoch units, an explicit opt-in retry policy for known-retryable failures, and
cooperative cancellation. It does not decide what a case's messages/tools/scorers mean
(that is the manifest's and compiler's job) and it does not itself evaluate correctness
(that is the scorers' job).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from contextlib import suppress
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

import aiofiles

from pyrit.executor.capability import (
    CapabilityOutcome,
    CapabilitySource,
    CapabilityTask,
    CapabilityTaskExecutor,
    ErrorEvidence,
    ToolRegistry,
)
from pyrit.models import Message, MessagePiece
from pyrit.sandbox import (
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxSessionSpec,
    SandboxSetupFile,
    SandboxSetupScript,
    SandboxTaskSpec,
    SandboxToolAdapter,
)
from pyrit.scenario.capability_suite.aggregation import aggregate_attempts
from pyrit.scenario.capability_suite.expansion import CaseRunUnit, expand_suite
from pyrit.scenario.capability_suite.manifest import validate_safe_relative_path
from pyrit.scenario.capability_suite.results import (
    AttemptOutcomeKind,
    CapabilitySuiteAttemptRecord,
    CapabilitySuiteProgress,
    CapabilitySuiteRunResult,
)
from pyrit.scenario.capability_suite.serialization import manifest_hash

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.executor.capability import CapabilityEvidenceSink, CapabilityTaskResult
    from pyrit.executor.capability.target_adapter import CapabilityRequestOptionsFactory
    from pyrit.prompt_target import PromptTarget
    from pyrit.sandbox import SandboxProvider, SandboxSession
    from pyrit.scenario.capability_suite.manifest import (
        CapabilityCaseManifest,
        CapabilitySuiteManifest,
        CaseAssetManifest,
    )
    from pyrit.scenario.capability_suite.registries import (
        CapabilitySuiteScorerFactoryRegistry,
        SandboxProviderFactoryRegistry,
        ToolImplementationFactoryRegistry,
    )


class AssetSourceResolver(Protocol):
    """Read declared asset content by ``asset_id`` (e.g. from a compiled suite's asset store)."""

    async def read_asset_async(self, *, asset: CaseAssetManifest) -> bytes:
        """Return the raw bytes for one manifest asset."""


class LocalAssetSourceResolver:
    """Resolve manifest asset sources beneath one explicit local containment root."""

    def __init__(self, *, root: Path) -> None:
        """Initialize the resolver with a canonical containment root."""
        self._root = root.resolve()

    async def read_asset_async(self, *, asset: CaseAssetManifest) -> bytes:
        """
        Read one contained local asset.

        Returns:
            bytes: The asset bytes.

        Raises:
            ValueError: If the source escapes the configured root.
        """
        validate_safe_relative_path(asset.source)
        resolved = (self._root / asset.source).resolve()
        if resolved != self._root and self._root not in resolved.parents:
            raise ValueError(f"Asset '{asset.asset_id}' source escapes the configured asset root.")
        async with aiofiles.open(resolved, "rb") as handle:
            return await handle.read()


class CapabilitySuiteProgressSink(Protocol):
    """Receive monotonic suite progress snapshots."""

    async def report_progress_async(self, *, progress: CapabilitySuiteProgress) -> None:
        """Report progress after one logical run unit completes."""


class CapabilitySuiteRunner:
    """Drive a capability suite's cases through bounded-concurrency sandboxed attempts."""

    def __init__(
        self,
        *,
        manifest: CapabilitySuiteManifest,
        target: PromptTarget,
        request_options_factory: CapabilityRequestOptionsFactory,
        sandbox_provider_registry: SandboxProviderFactoryRegistry,
        tool_implementation_registry: ToolImplementationFactoryRegistry | None = None,
        scorer_registry: CapabilitySuiteScorerFactoryRegistry | None = None,
        asset_resolver: AssetSourceResolver | None = None,
        evidence_sink: CapabilityEvidenceSink | None = None,
        progress_sink: CapabilitySuiteProgressSink | None = None,
    ) -> None:
        """
        Initialize a runner for one manifest.

        Args:
            manifest: The immutable, already-validated capability-suite manifest to run.
            target: The prompt target every case's conversation is executed against.
            request_options_factory: Injected per-call request-options seam, reused from
                ``pyrit.executor.capability``.
            sandbox_provider_registry: Explicit registry resolving the manifest's
                ``sandbox_provider.provider_type`` into a live ``SandboxProvider``.
            tool_implementation_registry: Explicit registry resolving each case's
                symbolic ``tools[].implementation.kind`` into a live implementation.
                Required only if any case declares custom tools.
            scorer_registry: Explicit registry resolving each case's symbolic
                ``scorers[].kind`` into a live ``CapabilitySuiteScorer``. Required only
                if any case declares scorers.
            asset_resolver: Reads declared asset content by ``asset_id`` for staging.
                Required only if any case declares assets.
            evidence_sink: Optional sink for sandbox-lifecycle evidence.
            progress_sink: Optional sink for monotonic logical-run progress updates.
        """
        self._manifest = manifest
        self._target = target
        self._request_options_factory = request_options_factory
        self._sandbox_provider_registry = sandbox_provider_registry
        self._tool_implementation_registry = tool_implementation_registry
        self._scorer_registry = scorer_registry
        self._asset_resolver = asset_resolver
        self._evidence_sink = evidence_sink
        self._progress_sink = progress_sink

    async def run_async(
        self,
        *,
        run_id: str | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> CapabilitySuiteRunResult:
        """
        Run every case x epoch unit under the manifest's bounded concurrency and attempts.

        Returns:
            CapabilitySuiteRunResult: Every preserved attempt record plus the computed
            aggregate, alongside the manifest's content hash for provenance.

        Raises:
            ValueError: If any case is explicitly marked non-runnable.
        """
        unsupported = tuple(case for case in self._manifest.cases if not case.runnable)
        if unsupported:
            details = "; ".join(f"{case.case_id}: {case.unsupported_reason}" for case in unsupported)
            raise ValueError(f"Capability suite contains non-runnable cases: {details}")
        content_hash = manifest_hash(self._manifest)
        run_id = run_id or f"capability-suite-{content_hash[:32]}"
        provider_configs = {
            case.case_id: case.sandbox_provider or self._manifest.sandbox_provider for case in self._manifest.cases
        }
        provider_keys = {
            case_id: json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            for case_id, config in provider_configs.items()
        }
        unique_configs = {provider_keys[case_id]: config for case_id, config in provider_configs.items()}
        providers_by_key: dict[str, SandboxProvider] = {}
        for key, config in unique_configs.items():
            providers_by_key[key] = await asyncio.to_thread(self._sandbox_provider_registry.build, config)
        providers = {case_id: providers_by_key[key] for case_id, key in provider_keys.items()}
        task_specs = {
            case.case_id: SandboxTaskSpec(task_id=f"{run_id}:{case.case_id}") for case in self._manifest.cases
        }
        prepared_tasks: list[tuple[SandboxProvider, SandboxTaskSpec]] = []
        prepared_providers: list[SandboxProvider] = []
        provider_cleanup_error: str | None = None
        try:
            for provider in providers_by_key.values():
                await provider.prepare_async()
                prepared_providers.append(provider)
            for case in self._manifest.cases:
                provider = providers[case.case_id]
                task_spec = task_specs[case.case_id]
                await provider.prepare_task_async(task_spec)
                prepared_tasks.append((provider, task_spec))
            units = expand_suite(self._manifest)
            semaphore = asyncio.Semaphore(self._manifest.run_policy.max_concurrency)
            progress_lock = asyncio.Lock()
            completed_units = 0

            async def _run_unit_bounded_async(unit: CaseRunUnit) -> list[CapabilitySuiteAttemptRecord]:
                nonlocal completed_units
                async with semaphore:
                    records = await self._run_unit_async(
                        unit=unit,
                        run_id=run_id,
                        provider=providers[unit.case.case_id],
                        task_spec=task_specs[unit.case.case_id],
                        cancellation_event=cancellation_event,
                    )
                if self._progress_sink is not None:
                    async with progress_lock:
                        completed_units += 1
                        await self._progress_sink.report_progress_async(
                            progress=CapabilitySuiteProgress(
                                completed_units=completed_units,
                                total_units=len(units),
                                latest_attempts=tuple(records),
                            )
                        )
                return records

            unit_results = await asyncio.gather(*(_run_unit_bounded_async(unit) for unit in units))
        finally:
            cleanup_errors: list[str] = []
            for provider, task_spec in reversed(prepared_tasks):
                try:
                    await provider.cleanup_task_async(task_spec)
                except Exception as error:
                    cleanup_errors.append(f"{type(error).__name__}: {error}")
            for provider in reversed(prepared_providers):
                try:
                    await provider.cleanup_async()
                except Exception as error:
                    cleanup_errors.append(f"{type(error).__name__}: {error}")
            if cleanup_errors:
                provider_cleanup_error = "; ".join(cleanup_errors)

        attempts = tuple(record for records in unit_results for record in records)
        return CapabilitySuiteRunResult(
            run_id=run_id,
            manifest_hash=content_hash,
            attempts=attempts,
            aggregate=aggregate_attempts(
                attempts,
                epoch_reducer=self._manifest.run_policy.epoch_reducer,
            ),
            provider_cleanup_error=provider_cleanup_error,
        )

    async def _run_unit_async(
        self,
        *,
        unit: CaseRunUnit,
        run_id: str,
        provider: SandboxProvider,
        task_spec: SandboxTaskSpec,
        cancellation_event: asyncio.Event | None,
    ) -> list[CapabilitySuiteAttemptRecord]:
        records: list[CapabilitySuiteAttemptRecord] = []
        max_attempts = self._manifest.run_policy.max_retries + 1
        for attempt_number in range(1, max_attempts + 1):
            if cancellation_event is not None and cancellation_event.is_set():
                records.append(self._cancelled_record(unit=unit, run_id=run_id, attempt_number=attempt_number))
                break
            record = await self._run_attempt_async(
                unit=unit,
                run_id=run_id,
                attempt_number=attempt_number,
                provider=provider,
                task_spec=task_spec,
                cancellation_event=cancellation_event,
            )
            is_terminal_kind = record.outcome_kind in (
                AttemptOutcomeKind.SUCCESS,
                AttemptOutcomeKind.CANCELLED,
                AttemptOutcomeKind.CLEANUP_FAILURE,
            )
            if is_terminal_kind or attempt_number == max_attempts or not self._is_retryable(record):
                records.append(record)
                break
            reason = self._retry_reason(record)
            records.append(
                CapabilitySuiteAttemptRecord(
                    attempt_key=record.attempt_key,
                    attempt_id=record.attempt_id,
                    case_id=record.case_id,
                    epoch=record.epoch,
                    repetition=record.repetition,
                    attempt_number=record.attempt_number,
                    outcome_kind=AttemptOutcomeKind.RETRY,
                    task_result=record.task_result,
                    error=record.error,
                    retry_reason=reason,
                    started_at=record.started_at,
                    ended_at=record.ended_at,
                )
            )
        return records

    def _cancelled_record(
        self,
        *,
        unit: CaseRunUnit,
        run_id: str,
        attempt_number: int,
    ) -> CapabilitySuiteAttemptRecord:
        now = datetime.now(tz=timezone.utc)
        attempt_id = self._attempt_id(run_id=run_id, unit=unit, attempt_number=attempt_number)
        return CapabilitySuiteAttemptRecord(
            attempt_key=f"{unit.unit_key}:try{attempt_number}",
            attempt_id=attempt_id,
            case_id=unit.case.case_id,
            epoch=unit.epoch,
            repetition=unit.repetition,
            attempt_number=attempt_number,
            outcome_kind=AttemptOutcomeKind.CANCELLED,
            task_result=None,
            error=None,
            retry_reason=None,
            started_at=now,
            ended_at=now,
        )

    def _is_retryable(self, record: CapabilitySuiteAttemptRecord) -> bool:
        allow_list = self._manifest.run_policy.retryable_error_codes
        if not allow_list:
            return False
        codes = self._failure_codes(record)
        return any(code in allow_list for code in codes)

    def _retry_reason(self, record: CapabilitySuiteAttemptRecord) -> str:
        codes = self._failure_codes(record)
        return f"retryable error code(s): {', '.join(codes) or record.error or 'unknown'}"

    @staticmethod
    def _failure_codes(record: CapabilitySuiteAttemptRecord) -> tuple[str, ...]:
        codes: list[str] = []
        if record.task_result is not None:
            for evidence in record.task_result.evidence:
                code = getattr(evidence, "code", None)
                if isinstance(code, str):
                    codes.append(code)
        if record.error is not None:
            codes.append(record.error.split(":", 1)[0])
        return tuple(codes)

    async def _run_attempt_async(
        self,
        *,
        unit: CaseRunUnit,
        run_id: str,
        attempt_number: int,
        provider: SandboxProvider,
        task_spec: SandboxTaskSpec,
        cancellation_event: asyncio.Event | None,
    ) -> CapabilitySuiteAttemptRecord:
        started_at = datetime.now(tz=timezone.utc)
        attempt_key = f"{unit.unit_key}:try{attempt_number}"
        attempt_id = self._attempt_id(run_id=run_id, unit=unit, attempt_number=attempt_number)
        case = unit.case

        try:
            session_spec = await self._build_session_spec_async(
                case=case,
                attempt_id=attempt_id,
                session_id=f"cap-{attempt_id}",
                task_spec=task_spec,
            )
            session = await provider.create_session_async(spec=session_spec, evidence_sink=self._evidence_sink)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # session creation failed before any sandbox exists to clean up
            return self._finish_record(
                unit=unit,
                attempt_key=attempt_key,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                outcome_kind=AttemptOutcomeKind.FAILURE,
                task_result=None,
                error=f"{type(error).__name__}: {error}",
                started_at=started_at,
            )

        task_result: CapabilityTaskResult | None = None
        run_error: Exception | None = None
        try:
            await session.initialize_async()
            tool_registry = ToolRegistry()
            self._bind_tools(case=case, session=session, registry=tool_registry)
            executor = CapabilityTaskExecutor(
                target=self._target,
                tool_registry=tool_registry,
                request_options_factory=self._request_options_factory,
            )
            task = self._build_task(case=case)
            capability_case = executor.create_case(task).model_copy(
                update={"case_id": uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{unit.unit_key}:case")}
            )
            task_result = await executor.execute_case_async(
                case=capability_case,
                conversation_id=str(attempt_id),
                cancellation_event=cancellation_event,
            )
            if task_result.outcome is not CapabilityOutcome.CANCELLED:
                task_result = await self._apply_scorers_async(
                    case=case,
                    result=task_result,
                    session=session,
                    cancellation_event=cancellation_event,
                )
        except asyncio.CancelledError:
            with suppress(Exception):
                await session.close_async()
            raise
        except Exception as error:
            run_error = error

        cleanup_error: Exception | None = None
        try:
            await session.close_async()
        except Exception as error:
            cleanup_error = error

        ended_at = datetime.now(tz=timezone.utc)
        if cleanup_error is not None:
            error_message = f"cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
            if run_error is not None:
                error_message = f"run failed: {type(run_error).__name__}: {run_error}; {error_message}"
            return self._finish_record(
                unit=unit,
                attempt_key=attempt_key,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                outcome_kind=AttemptOutcomeKind.CLEANUP_FAILURE,
                task_result=task_result,
                error=error_message,
                started_at=started_at,
                ended_at=ended_at,
            )
        if run_error is not None:
            return self._finish_record(
                unit=unit,
                attempt_key=attempt_key,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                outcome_kind=AttemptOutcomeKind.FAILURE,
                task_result=task_result,
                error=f"{type(run_error).__name__}: {run_error}",
                started_at=started_at,
                ended_at=ended_at,
            )
        outcome_kind = self._task_outcome_kind(task_result)
        return self._finish_record(
            unit=unit,
            attempt_key=attempt_key,
            attempt_id=attempt_id,
            attempt_number=attempt_number,
            outcome_kind=outcome_kind,
            task_result=task_result,
            error=None,
            started_at=started_at,
            ended_at=ended_at,
        )

    @staticmethod
    def _finish_record(
        *,
        unit: CaseRunUnit,
        attempt_key: str,
        attempt_id: uuid.UUID,
        attempt_number: int,
        outcome_kind: AttemptOutcomeKind,
        task_result: CapabilityTaskResult | None,
        error: str | None,
        started_at: datetime,
        ended_at: datetime | None = None,
    ) -> CapabilitySuiteAttemptRecord:
        return CapabilitySuiteAttemptRecord(
            attempt_key=attempt_key,
            attempt_id=attempt_id,
            case_id=unit.case.case_id,
            epoch=unit.epoch,
            repetition=unit.repetition,
            attempt_number=attempt_number,
            outcome_kind=outcome_kind,
            task_result=task_result,
            error=error,
            retry_reason=None,
            started_at=started_at,
            ended_at=ended_at or datetime.now(tz=timezone.utc),
        )

    async def _build_session_spec_async(
        self,
        *,
        case: CapabilityCaseManifest,
        attempt_id: uuid.UUID,
        session_id: str,
        task_spec: SandboxTaskSpec,
    ) -> SandboxSessionSpec:
        environment_names = {asset.environment or "default" for asset in case.assets}
        environment_names.update(step.environment or "default" for step in case.setup)
        environment_names.update(environment for tool in case.tools for environment in tool.required_environments)
        environment_names.update(environment for scorer in case.scorers for environment in scorer.required_environments)
        if case.sandbox_tools_default_environment is not None:
            environment_names.add(case.sandbox_tools_default_environment)
        environment_names.update(case.sandbox_tools_allowed_environments)
        environment_names.update(case.sandbox_environment_workdirs)
        if not environment_names:
            environment_names.add("default")

        setup_files: dict[str, list[SandboxSetupFile]] = {name: [] for name in environment_names}
        for asset in case.assets:
            if self._asset_resolver is None:
                raise ValueError(
                    f"Case '{case.case_id}' declares assets but no asset_resolver was provided to the runner."
                )
            content = await self._asset_resolver.read_asset_async(asset=asset)
            actual_sha256 = hashlib.sha256(content).hexdigest()
            if actual_sha256 != asset.sha256:
                raise ValueError(
                    f"Asset '{asset.asset_id}' sha256 mismatch: expected {asset.sha256}, got {actual_sha256}."
                )
            setup_files[asset.environment or "default"].append(
                SandboxSetupFile(
                    path=asset.destination,
                    content=content,
                    executable=asset.mode.value == "executable",
                )
            )

        setup_scripts: dict[str, list[SandboxSetupScript]] = {name: [] for name in environment_names}
        for step in case.setup:
            setup_scripts[step.environment or "default"].append(
                SandboxSetupScript(
                    request=SandboxExecRequest(
                        argv=step.argv,
                        shell_script=step.shell_script,
                        timeout_seconds=step.timeout_seconds,
                    )
                )
            )

        environments = tuple(
            SandboxEnvironmentSpec(
                name=name,
                default=(name == "default"),
                setup_files=tuple(setup_files[name]),
                setup_scripts=tuple(setup_scripts[name]),
                metadata=(
                    {"docker_workdir": case.sandbox_environment_workdirs[name]}
                    if name in case.sandbox_environment_workdirs
                    else {}
                ),
            )
            for name in sorted(environment_names)
        )
        return SandboxSessionSpec(
            session_id=session_id,
            attempt_id=attempt_id,
            task=task_spec,
            environments=environments,
            default_environment=(
                case.sandbox_tools_default_environment
                or ("default" if "default" in environment_names else min(environment_names))
            ),
        )

    def _bind_tools(self, *, case: CapabilityCaseManifest, session: SandboxSession, registry: ToolRegistry) -> None:
        if case.sandbox_tools_prefix is not None:
            SandboxToolAdapter(
                session=session,
                default_environment=case.sandbox_tools_default_environment,
                allowed_environments=case.sandbox_tools_allowed_environments,
                default_user=case.sandbox_tools_default_user,
                allow_user_override=case.sandbox_tools_allow_user_override,
                include_file_tools=case.sandbox_tools_include_file_tools,
            ).register(registry=registry, prefix=case.sandbox_tools_prefix)
        for tool in case.tools:
            if self._tool_implementation_registry is None:
                raise ValueError(
                    f"Case '{case.case_id}' declares custom tools but no tool_implementation_registry "
                    "was provided to the runner."
                )
            implementation = self._tool_implementation_registry.build(
                kind=tool.implementation.kind,
                config=tool.implementation.config,
                session=session,
            )
            registry.register(declaration=tool.declaration, implementation=implementation)

    @staticmethod
    def _build_task(*, case: CapabilityCaseManifest) -> CapabilityTask:
        initial_messages = tuple(
            Message(
                message_pieces=[
                    MessagePiece(
                        original_value=message.content,
                        role=message.role,
                        original_value_data_type=message.data_type,
                    )
                ]
            )
            for message in case.messages
        )
        source = case.source or CapabilitySource(
            source_type="capability_suite",
            source_id=case.case_id,
            metadata={"tags": list(case.tags), **case.metadata},
        )
        required_tools = [tool.declaration.name for tool in case.tools]
        if case.sandbox_tools_prefix is not None:
            required_tools.append(f"{case.sandbox_tools_prefix}_exec")
            if case.sandbox_tools_include_file_tools:
                required_tools.extend(
                    (
                        f"{case.sandbox_tools_prefix}_read_file",
                        f"{case.sandbox_tools_prefix}_write_file",
                    )
                )
        environment_references = {asset.environment or "default" for asset in case.assets} | {
            step.environment or "default" for step in case.setup
        }
        environment_references.update(environment for tool in case.tools for environment in tool.required_environments)
        environment_references.update(
            environment for scorer in case.scorers for environment in scorer.required_environments
        )
        return CapabilityTask(
            objective=case.objective,
            initial_messages=initial_messages,
            required_tools=tuple(required_tools),
            asset_references=tuple(asset.asset_id for asset in case.assets),
            environment_requirement_references=tuple(sorted(environment_references)),
            limits=case.limits,
            source=source,
            expected_evidence=case.expected_evidence,
            completion_tool_name=case.completion_tool_name,
            continue_prompt=case.continue_prompt,
        )

    async def _apply_scorers_async(
        self,
        *,
        case: CapabilityCaseManifest,
        result: CapabilityTaskResult,
        session: SandboxSession,
        cancellation_event: asyncio.Event | None,
    ) -> CapabilityTaskResult:
        if not case.scorers:
            return result
        if self._scorer_registry is None:
            raise ValueError(f"Case '{case.case_id}' declares scorers but no scorer_registry was provided.")
        scores = list(result.scores)
        errors: list[ErrorEvidence] = []
        for scorer_manifest in case.scorers:
            try:
                scorer = self._scorer_registry.build(kind=scorer_manifest.kind, config=scorer_manifest.config)
                scores.extend(
                    await scorer.score_async(
                        result=result,
                        objective=case.objective,
                        session=session,
                        cancellation_event=cancellation_event,
                    )
                )
            except Exception as error:
                errors.append(
                    ErrorEvidence(
                        phase="suite_scoring",
                        code=type(error).__name__,
                        message=str(error),
                    )
                )
        return result.model_copy(update={"scores": tuple(scores), "evidence": (*result.evidence, *errors)})

    @staticmethod
    def _task_outcome_kind(result: CapabilityTaskResult | None) -> AttemptOutcomeKind:
        if result is None:
            return AttemptOutcomeKind.FAILURE
        if result.outcome is CapabilityOutcome.CANCELLED:
            return AttemptOutcomeKind.CANCELLED
        if result.outcome is CapabilityOutcome.COMPLETED:
            return AttemptOutcomeKind.SUCCESS
        return AttemptOutcomeKind.FAILURE

    @staticmethod
    def _attempt_id(*, run_id: str, unit: CaseRunUnit, attempt_number: int) -> uuid.UUID:
        return uuid.uuid5(uuid.NAMESPACE_URL, f"{run_id}:{unit.unit_key}:try{attempt_number}")
