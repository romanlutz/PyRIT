# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import logging
import math
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from pyrit.executor.benchmark._inspect_native_generate import InspectNativeLimitError
from pyrit.executor.benchmark._inspect_v2_support import (
    InspectNativeGenerateV2,
    InspectProviderObserverV2,
    InspectRunArtifactsV2,
)
from pyrit.executor.benchmark.inspect_native_task import InspectNativeTaskBridge
from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.models import ContentScorable
from pyrit.models.submission import SubmissionCleanupStatus, SubmissionReportStatus, SubmissionTerminationReason
from pyrit.models.submission_v2 import RetainedSubmissionReportV2, StrictSubmissionReportV2
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from inspect_ai import Task
    from inspect_ai.log import EvalLog, EvalSample
    from inspect_ai.solver import Solver, TaskState
    from pydantic import JsonValue

    from pyrit.executor.benchmark.submission.hooks_v2 import (
        SubmissionEnvironmentHooksV2,
        SubmissionLimitsV2,
        SubmissionTargetFactoryV2,
    )
    from pyrit.models import Score, ScoringExpectation
    from pyrit.prompt_target import OpenAIResponseTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class InspectNativeTaskResultV2:
    """A v2 content score and actual native identities, including partial cancellation evidence."""

    report: RetainedSubmissionReportV2
    score: Score
    report_path: Path
    native_log: Path | None
    native_eval_id: str | None
    native_sample_uuid: str | None


class InspectNativeTaskBridgeV2:
    """Adapt a caller-owned native Inspect task with explicitly versioned provenance and lifetimes."""

    _FINALIZATION_SECONDS = 5
    _AUDIT_SECONDS = 5
    _TERMINAL = {
        SubmissionReportStatus.INCOMPLETE,
        SubmissionReportStatus.ERROR,
        SubmissionReportStatus.UNKNOWN,
        SubmissionReportStatus.CANCELLED,
    }

    def __init__(
        self,
        *,
        target_factory: SubmissionTargetFactoryV2,
        read_report: Callable[[], dict[str, Any]],
        directory: Path,
        environment: SubmissionEnvironmentHooksV2,
        limits: SubmissionLimitsV2,
        max_tool_output_bytes: int | None = None,
    ) -> None:
        """
        Bind caller infrastructure without creating a target, credential, service, or resource.

        Raises:
            TypeError: If the tested synchronous SQLite publication backend is not in use.
            ValueError: If the report provenance or tool byte bound is invalid.
        """
        memory = CentralMemory.get_memory_instance()
        if type(memory) is not SQLiteMemory:
            raise TypeError("V2 requires the unchanged synchronous SQLiteMemory publication backend.")
        tool_bound = max_tool_output_bytes if max_tool_output_bytes is not None else limits.max_response_bytes
        if type(tool_bound) is not int or tool_bound <= 0:
            raise ValueError("max_tool_output_bytes must be a positive UTF-8 byte bound.")
        self._memory = memory
        self._read_report = read_report
        self._report = StrictSubmissionReportV2.model_validate(read_report())
        self._provenance = self._report.provenance()
        self._factory = target_factory
        self._environment = environment
        self._limits = limits
        self._artifacts = InspectRunArtifactsV2(directory=directory, provenance=self._provenance)
        self._observer = InspectProviderObserverV2(artifacts=self._artifacts, limits=limits)
        self._seen_submissions: set[str] = set()
        self._generator = InspectNativeGenerateV2(
            acquire_target_async=self._acquire_target_async,
            capture_report=self._capture_report,
            after_tool_async=self._after_tool_async,
            artifacts=self._artifacts,
            limits=limits,
            observer=self._observer,
            max_tool_output_bytes=tool_bound,
        )
        self._target_stack = AsyncExitStack()
        self._target_identifier: dict[str, JsonValue] | None = None
        self._audit: JsonValue = None
        self._cleanup = SubmissionCleanupStatus.UNKNOWN
        self._lifecycle_errors: list[str] = []
        self._runner_error: str | None = None
        self._limit_error = False
        self._state: TaskState | None = None
        self._sample_id: str | int | None = None
        self._native_log: Path | None = None
        self._native_eval: EvalLog | None = None
        self._native_sample: EvalSample | None = None
        self._last_result: InspectNativeTaskResultV2 | None = None
        self._used = False

    @property
    def last_result(self) -> InspectNativeTaskResultV2 | None:
        """The published v2 result, when durable retention succeeded before return or interruption."""
        return self._last_result

    async def execute_async(
        self,
        *,
        task: Task,
        sample_id: str | int,
        native_scorer: str,
        expectation: ScoringExpectation | None = None,
    ) -> InspectNativeTaskResultV2:
        """
        Run the original task/solver with public Generate and tools, then publish acquired report evidence.

        Returns:
            InspectNativeTaskResultV2: The content-anchored result and observed native references.

        Raises:
            RuntimeError: If reused or CentralMemory changed.
            ValueError: If the native sample/report binding is not fresh and unambiguous.
            asyncio.CancelledError: The original caller or tool cancellation after bounded retention.
        """
        from inspect_ai import eval_async

        if self._used or CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("V2 requires one attempt and the unchanged CentralMemory instance.")
        self._used = True
        self._capture_report()
        if self._report.submissions or self._report.status is not SubmissionReportStatus.NO_SUBMISSION:
            raise ValueError("V2 execution requires fresh healthy binding-owned submission state.")
        if len(task.dataset) != 1 or task.dataset[0].id != sample_id:
            raise ValueError("Bind exactly one native task sample before v2 execution.")
        if task.model is not None or task.model_roles:
            raise ValueError("The v2 external-target bridge requires a model-free native Task.")
        self._sample_id = sample_id
        self._generator.configure_native_limits(task)
        await self._artifacts.writer.initialize_async()
        self._artifacts.manifest.update(
            conversation_id=self._generator.conversation_id,
            harness_status="running",
            completion_policy="strict-after-tool; native no-input continuation",
        )
        await self._artifacts.save_async()
        evaluation: asyncio.Task[list[EvalLog]] | None = None
        error: BaseException | None = None
        try:
            seconds = min(
                self._limits.episode_timeout_seconds,
                task.time_limit if task.time_limit is not None else self._limits.episode_timeout_seconds,
            )
            async with asyncio.timeout(seconds):
                audit = await self._audit_environment_async()
                json.dumps(audit, allow_nan=False)
                self._audit = audit
                evaluation = asyncio.create_task(
                    eval_async(
                        tasks=task,
                        model=None,
                        solver=self._solver(task.solver),
                        sample_id=sample_id,
                        epochs=1,
                        retry_on_error=0,
                        task_retry_attempts=0,
                        score_on_error=False,
                        fail_on_error=True,
                        sandbox_cleanup=True,
                        max_samples=1,
                        max_tasks=1,
                        log_dir=str(self._artifacts.directory / "native"),
                        log_realtime=False,
                        log_model_api=False,
                        ctl_server=False,
                        acp_server=False,
                        metadata={
                            "pyrit_run_id": self._artifacts.run_id,
                            "pyrit_attempt_id": self._artifacts.attempt_id,
                            **self._provenance.model_dump(mode="json"),
                            "evidence_label": self._provenance.evidence_label,
                            "contract_version": "strict-submission-v2",
                        },
                    )
                )
                await asyncio.shield(evaluation)
                if self._generator.callback_cancellation is not None:
                    raise self._generator.callback_cancellation
        except BaseException as caught:
            error = caught
            self._runner_error = f"{type(caught).__name__}: {caught}"
            self._generator.stop()
            if evaluation is not None and not evaluation.done():
                evaluation.cancel()
        error = await self._finish_async(evaluation=evaluation, original=error)
        if error is None:
            try:
                self._validate_native(native_scorer)
            except Exception as caught:
                error = caught
                self._runner_error = f"{type(caught).__name__}: {caught}"
        if error is None:
            try:
                result = await self._publish_async(error=None, expectation=expectation)
            except (asyncio.CancelledError, TimeoutError) as caught:
                error = caught
            else:
                return result
        try:
            result = await self._publish_interrupted_async(error=error, expectation=expectation)
        except BaseException as caught:
            if caught is error:
                raise
            error.add_note(f"V2 report publication failed: {type(caught).__name__}: {caught}")
            raise error from caught
        if isinstance(error, (asyncio.CancelledError, TimeoutError)):
            raise error
        return result

    async def _audit_environment_async(self) -> JsonValue:
        async def invoke_async() -> JsonValue:
            return await self._environment.audit_async()

        audit = asyncio.create_task(invoke_async())
        try:
            async with asyncio.timeout(self._AUDIT_SECONDS):
                result = await asyncio.shield(audit)
            self._artifacts.manifest["environment_audit_status"] = "returned"
            return result
        except BaseException as error:
            self._artifacts.manifest["environment_audit_status"] = "interrupted"
            self._lifecycle_errors.append(f"Environment audit: {type(error).__name__}: {error}")
            if not audit.done():
                audit.cancel()
                audit.add_done_callback(self._observe_late_audit)
            raise

    @staticmethod
    def _observe_late_audit(task: asyncio.Task[JsonValue]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning("Caller environment audit finished after interruption with an error: %s", error)

    async def _acquire_target_async(self, schemas: list[dict[str, Any]]) -> OpenAIResponseTarget:
        self._capture_report()
        target = await self._target_stack.enter_async_context(
            self._factory(
                tools=schemas,
                request_hook=self._observer.request_async,
                response_hook=self._observer.response_async,
            )
        )
        if target.auto_execute_tools:
            raise ValueError("The v2 target must use auto_execute_tools=False.")
        identifier = target.get_identifier()
        model_name = identifier.params.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("The caller target must identify its actual model through its public identifier.")
        self._target_identifier = identifier.model_dump(mode="json")
        self._observer.schemas = schemas
        self._observer.model_name = model_name
        await self._artifacts.append_async(event="target_bound", data={"target_identifier": self._target_identifier})
        return target

    def _solver(self, original: Solver) -> Solver:
        from inspect_ai.solver import Generate, solver

        @solver("pyrit_native_solver_bridge_v2")
        def bridge() -> Solver:
            async def solve_async(state: TaskState, generate: Generate) -> TaskState:
                self._state = state
                try:
                    result = await original(state, self._generator.generate_async)
                    if result is not state:
                        raise ValueError(
                            "Replacing native TaskState is not supported by this identity-bound v2 bridge."
                        )
                    return state
                except BaseException as error:
                    self._runner_error = f"{type(error).__name__}: {error}"
                    self._limit_error = isinstance(error, InspectNativeLimitError)
                    raise
                finally:
                    try:
                        self._capture_report()
                    except Exception as error:
                        self._lifecycle_errors.append(f"Binding report: {type(error).__name__}: {error}")
                    state.metadata.update(
                        pyrit_strict_report_v2=self._report.model_dump(mode="json"),
                        pyrit_conversation_id=self._generator.conversation_id,
                        pyrit_run_id=self._artifacts.run_id,
                        **self._provenance.model_dump(mode="json"),
                        evidence_label=self._provenance.evidence_label,
                    )

            return solve_async

        return bridge()

    def _capture_report(self) -> None:
        report = StrictSubmissionReportV2.model_validate(self._read_report())
        if report.provenance() != self._provenance:
            raise ValueError("The v2 binding changed provenance during an identity-bound run.")
        self._report = report
        newly_observed = tuple(
            item.submission_id for item in report.submissions if item.submission_id not in self._seen_submissions
        )
        if newly_observed and self._generator.tool_calls:
            call = self._generator.tool_calls[-1]
            call["submission_ids"] = list(call.get("submission_ids", [])) + list(newly_observed)
        self._seen_submissions.update(newly_observed)

    async def _after_tool_async(self) -> bool:
        self._capture_report()
        await self._artifacts.writer.snapshot_async(self._report)
        if self._report.status in self._TERMINAL:
            raise RuntimeError(f"V2 acquisition stopped as {self._report.status.value}; no further dispatch.")
        return self._report.full_success

    async def _finish_async(
        self, *, evaluation: asyncio.Task[list[EvalLog]] | None, original: BaseException | None
    ) -> BaseException | None:
        async def finalize_async() -> None:
            async with asyncio.timeout(self._FINALIZATION_SECONDS):
                if evaluation is not None:
                    try:
                        await asyncio.shield(evaluation)
                    except asyncio.CancelledError:
                        if not evaluation.done():
                            raise
                    except Exception as error:
                        self._lifecycle_errors.append(f"Native evaluation: {type(error).__name__}: {error}")
                await self._finalize_owned_async()

        finalization = asyncio.create_task(finalize_async())
        deadline = asyncio.get_running_loop().time() + self._FINALIZATION_SECONDS
        primary = original
        while not finalization.done():
            try:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise TimeoutError("V2 owned finalization exceeded its absolute deadline.")
                await asyncio.wait_for(asyncio.shield(finalization), timeout=remaining)
            except asyncio.CancelledError as error:
                primary = primary or error
                self._generator.stop()
            except BaseException as error:
                self._lifecycle_errors.append(f"Owned finalization: {type(error).__name__}: {error}")
                primary = primary or error
                if not finalization.done():
                    finalization.cancel()
                    finalization.add_done_callback(InspectNativeTaskBridge._observe_late_finalization)
                break
        if finalization.done() and not finalization.cancelled():
            try:
                finalization.result()
            except BaseException as error:
                primary = primary or error
        return primary

    async def _finalize_owned_async(self) -> None:
        try:
            self._capture_report()
        except Exception as error:
            self._lifecycle_errors.append(f"Binding finalization: {type(error).__name__}: {error}")
        try:
            cleanup = await self._environment.cleanup_async()
            if not isinstance(cleanup, SubmissionCleanupStatus):
                raise TypeError("The environment hook must return an explicit SubmissionCleanupStatus.")
            self._cleanup = cleanup
        except asyncio.CancelledError:
            self._lifecycle_errors.append("Environment cleanup was interrupted; its outcome is unknown.")
            raise
        except Exception as error:
            self._lifecycle_errors.append(f"Environment cleanup: {type(error).__name__}: {error}")
        finally:
            try:
                await self._target_stack.aclose()
            except asyncio.CancelledError:
                self._lifecycle_errors.append("Target context exit was interrupted; closure is unconfirmed.")
                raise
            except Exception as error:
                self._lifecycle_errors.append(f"Target context exit: {type(error).__name__}: {error}")
        try:
            if self._state is not None and self._generator.target is not None:
                await self._generator.retain_pending_async(self._state)
            await self._recover_native_async()
        except asyncio.CancelledError:
            self._lifecycle_errors.append("Evidence finalization was interrupted; persistence is unconfirmed.")
            raise
        except Exception as error:
            self._lifecycle_errors.append(f"Evidence finalization: {type(error).__name__}: {error}")
        self._artifacts.manifest.update(
            binding_report=self._report.model_dump(mode="json"),
            target_identifier=self._target_identifier,
            environment_audit=self._audit,
            local_cleanup=self._cleanup.value,
            lifecycle_errors=self._lifecycle_errors,
            provider_request_count=self._observer.request_count,
            generation_count=self._generator.requests,
            token_usage=self._observer.token_usage,
            tool_calls=self._generator.tool_calls,
        )
        await self._artifacts.save_async()

    async def _recover_native_async(self) -> None:
        from inspect_ai.log import list_eval_logs_async, read_eval_log_async

        directory = self._artifacts.directory / "native"
        matches: list[tuple[Path, EvalLog, EvalSample]] = []
        for info in await list_eval_logs_async(str(directory), formats=["eval"], recursive=False):
            path = await asyncio.to_thread(
                InspectNativeTaskBridge._owned_log_path, location=info.name, directory=directory
            )
            log = await read_eval_log_async(path, resolve_attachments=True)
            metadata = log.eval.metadata or {}
            if (
                log.status == "started"
                or not log.stats.completed_at
                or metadata.get("pyrit_run_id") != self._artifacts.run_id
                or metadata.get("pyrit_attempt_id") != self._artifacts.attempt_id
            ):
                continue
            sample = await asyncio.to_thread(InspectNativeTaskBridge._materialize_sample, log)
            if (
                sample.id != self._sample_id
                or sample.epoch != 1
                or not sample.uuid
                or (self._state is not None and sample.uuid != self._state.uuid)
            ):
                continue
            if (
                metadata.get("mode") != self._provenance.mode.value
                or metadata.get("simulated") is not self._provenance.simulated
                or metadata.get("evidence_label") != self._provenance.evidence_label
            ):
                raise ValueError("The retained native log changed v2 provenance.")
            matches.append((path, log, sample))
        if len(matches) > 1:
            raise ValueError("Multiple finalized native logs match one v2 attempt.")
        if matches:
            self._native_log, self._native_eval, self._native_sample = matches[0]
            self._artifacts.manifest.update(
                native_log=str(self._native_log),
                native_eval_id=self._native_eval.eval.eval_id,
                native_sample_uuid=self._native_sample.uuid,
            )

    def _validate_native(self, scorer_name: str) -> None:
        log, sample = self._native_eval, self._native_sample
        if log is None or sample is None:
            raise ValueError("No finalized native log is available to accept a clean v2 result.")
        if log.status != "success" or log.invalidated or sample.error or sample.invalidation or sample.error_retries:
            raise RuntimeError("The native evaluation did not complete cleanly.")
        if sample.limit and self._generator.termination_reason != "budget":
            raise InspectNativeLimitError("The native evaluation ended with an unresolved limit.")
        if self._report.status is not SubmissionReportStatus.COMPLETED:
            return
        if sample.metadata.get("pyrit_strict_report_v2") != self._report.model_dump(mode="json"):
            raise ValueError("The native sample report does not match retained v2 evidence.")
        grade = (sample.scores or {}).get(scorer_name)
        value = grade.value if grade is not None else None
        if type(value) not in (int, float) or not isinstance(value, (int, float)):
            raise ValueError("The native v2 grade must be an actually returned numeric value.")
        if not math.isfinite(value) or not 0 <= value <= 1 or value != self._report.last_valid_grade:
            raise ValueError("The native v2 grade differs from the latest valid acquired submission.")

    def _retained_report(self, error: BaseException | None) -> RetainedSubmissionReportV2:
        status = self._report.status
        reason = self._runner_error
        if isinstance(error, asyncio.CancelledError):
            status = SubmissionReportStatus.CANCELLED
        elif isinstance(error, TimeoutError) or self._limit_error or isinstance(error, InspectNativeLimitError):
            status = SubmissionReportStatus.INCOMPLETE
        elif error is not None or reason or self._lifecycle_errors:
            status = status if status in self._TERMINAL else SubmissionReportStatus.ERROR
        elif self._cleanup not in (SubmissionCleanupStatus.COMPLETE, SubmissionCleanupStatus.NOT_REQUIRED):
            status = SubmissionReportStatus.INCOMPLETE
        if error is not None:
            reason = "\n".join([f"{type(error).__name__}: {error}", *getattr(error, "__notes__", [])])
        elif self._lifecycle_errors:
            reason = "; ".join(self._lifecycle_errors)
        return RetainedSubmissionReportV2(
            **self._provenance.model_dump(mode="json"),
            evidence_label=self._provenance.evidence_label,
            run_id=self._artifacts.run_id,
            conversation_id=self._generator.conversation_id,
            status=status,
            report=self._report,
            calls=tuple(
                InspectNativeTaskBridge._call_evidence(call)
                for call in self._generator.tool_calls
                if call["status"] != "not_dispatched"
            ),
            final_text=(self._state.output.completion or None) if self._state else None,
            runner_error=reason,
            termination_reason=(
                SubmissionTerminationReason(self._generator.termination_reason)
                if self._generator.termination_reason
                else SubmissionTerminationReason.RUNNER_ERROR
                if error or reason
                else SubmissionTerminationReason.BINDING_STATE
            ),
            generation_count=self._generator.requests,
            provider_request_count=self._observer.request_count,
            message_count=len(self._state.messages) if self._state else 0,
            total_tokens=self._observer.total_tokens,
            token_usage=tuple(self._observer.token_usage),
            termination_limit=self._generator.termination_limit,
            limits={key: value for key, value in asdict(self._limits).items() if value is not None},
            target_identifier=self._target_identifier,
            environment_audit=self._audit,
            local_cleanup=self._cleanup,
            lifecycle_errors=tuple(self._lifecycle_errors),
        )

    async def _publish_async(
        self, *, error: BaseException | None, expectation: ScoringExpectation | None
    ) -> InspectNativeTaskResultV2:
        if CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("CentralMemory changed before v2 publication.")
        retained = self._retained_report(error)
        path = await self._artifacts.writer.retain_async(retained)
        self._artifacts.manifest.update(
            publication_state="unscored_candidate",
            publication_boundary="pyrit_score_commit",
            report_path=str(path),
            report_sha256=retained.sha256(),
            run_status=retained.status.value,
        )
        await self._artifacts.save_async()
        await asyncio.sleep(0)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError("V2 cancellation accepted before score publication.")
        if CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("CentralMemory changed at v2 publication.")
        scores = await SubmissionReportScorerV2(report_sha256=retained.sha256()).score_async(
            scorable=ContentScorable(value=retained.canonical_json(), data_type="text"), expectation=expectation
        )
        self._artifacts.writer.mark_published(path)
        self._last_result = InspectNativeTaskResultV2(
            report=retained,
            score=scores[0],
            report_path=path,
            native_log=self._native_log,
            native_eval_id=self._native_eval.eval.eval_id if self._native_eval else None,
            native_sample_uuid=self._state.uuid if self._state else None,
        )
        return self._last_result

    async def _publish_interrupted_async(
        self, *, error: BaseException, expectation: ScoringExpectation | None
    ) -> InspectNativeTaskResultV2:
        async def retain_async() -> InspectNativeTaskResultV2:
            async with asyncio.timeout(self._FINALIZATION_SECONDS):
                return await self._publish_async(error=error, expectation=expectation)

        retention = asyncio.create_task(retain_async())
        while not retention.done():
            try:
                await asyncio.shield(retention)
            except asyncio.CancelledError:
                error.add_note("Additional caller cancellation during bounded v2 report retention.")
        return retention.result()
