# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlsplit

import aiofiles

from pyrit.executor.benchmark._inspect_native_generate import InspectNativeGenerate, InspectNativeLimitError
from pyrit.models import ContentScorable
from pyrit.models.submission import (
    RetainedSubmissionReport,
    StrictSubmissionReport,
    SubmissionCallEvidence,
    SubmissionFeedbackKind,
    SubmissionReportStatus,
    SubmissionTerminationReason,
)
from pyrit.score import SubmissionReportScorer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from inspect_ai import Task
    from inspect_ai.log import EvalLog, EvalSample
    from inspect_ai.solver import Solver, TaskState

    from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
    from pyrit.executor.benchmark.submission.hooks import SubmissionLimits
    from pyrit.models import Score, ScoringExpectation
    from pyrit.prompt_target import OpenAIResponseTarget

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class InspectTaskEnvironment:
    """Caller-owned audit and cleanup checks, not a sandbox provider or remote cancel API."""

    audit_async: Callable[[TaskState], Awaitable[None]]
    cleanup_async: Callable[[], Awaitable[None]]


@dataclass(frozen=True, kw_only=True)
class InspectNativeTaskResult:
    """Acquired native and canonical report references, including undetermined outcomes."""

    report: RetainedSubmissionReport
    score: Score
    report_path: Path
    native_log: Path | None
    native_eval_id: str | None
    native_sample_uuid: str | None


class InspectNativeTaskBridge:
    """Run one native task/solver with injected offline tools and content-based grade evidence."""

    _TERMINAL_STATUSES = {
        SubmissionReportStatus.INCOMPLETE,
        SubmissionReportStatus.ERROR,
        SubmissionReportStatus.UNKNOWN,
        SubmissionReportStatus.CANCELLED,
    }
    _CANCELLATION_SETTLE_SECONDS = 5

    def __init__(
        self,
        *,
        target_factory: Callable[[list[dict[str, Any]]], OpenAIResponseTarget],
        model_name: str,
        read_report: Callable[[], dict[str, Any]],
        artifacts: InspectRunArtifacts,
        environment: InspectTaskEnvironment,
        limits: SubmissionLimits,
        max_tool_output_bytes: int | None = None,
    ) -> None:
        """
        Bind task-independent infrastructure; tools, prompts, submission policy stay caller-owned.

        Raises:
            TypeError: If the offline scorer publication path does not use the tested SQLite backend.
            ValueError: If the explicit tool-feedback byte bound is invalid.
        """
        from pyrit.memory import CentralMemory, SQLiteMemory

        memory = CentralMemory.get_memory_instance()
        if type(memory) is not SQLiteMemory:
            raise TypeError("The offline native bridge requires the unchanged synchronous SQLiteMemory backend.")
        self._memory = memory
        self._read_report = read_report
        self._artifacts = artifacts
        self._environment = environment
        self._limits = limits
        tool_bound = max_tool_output_bytes if max_tool_output_bytes is not None else limits.max_response_bytes
        if type(tool_bound) is not int or tool_bound <= 0:
            raise ValueError("max_tool_output_bytes must be a positive UTF-8 byte bound.")
        self._generator = InspectNativeGenerate(
            target_factory=target_factory,
            model_name=model_name,
            artifacts=artifacts,
            after_tool_async=self._after_tool_async,
            on_tool_return=self._capture_report,
            max_requests=limits.max_requests,
            max_tool_calls=limits.max_tool_calls,
            max_tool_output_bytes=tool_bound,
            max_response_bytes=limits.max_response_bytes,
        )
        self._report: StrictSubmissionReport | None = None
        self._seen_submissions: set[str] = set()
        self._runner_error: str | None = None
        self._limit_reached = False
        self._used = False
        self._active_state: TaskState | None = None
        self._last_result: InspectNativeTaskResult | None = None
        self._finalization_task: asyncio.Task[None] | None = None
        self._finalization_deadline: float | None = None
        self._expected_sample_id: str | int | None = None

    @property
    def last_result(self) -> InspectNativeTaskResult | None:
        """The retained result, including an undetermined report acquired before caller cancellation."""
        return self._last_result

    async def execute_async(
        self,
        *,
        task: Task,
        sample_id: str | int,
        native_scorer: str,
        expectation: ScoringExpectation | None = None,
    ) -> InspectNativeTaskResult:
        """
        Preserve the native solver and grader while adapting only its Generate callback.

        Returns:
            InspectNativeTaskResult: The checked native log and retained content-based verdict.

        Raises:
            ValueError: If the task is not a single selected sample or evidence is inconsistent.
            RuntimeError: If this bridge is reused.
        """
        from inspect_ai import eval_async

        from pyrit.memory import CentralMemory

        if self._used:
            raise RuntimeError("Each native bridge owns one independent task attempt.")
        if CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("CentralMemory changed before the offline run.")
        self._used = True
        self._expected_sample_id = sample_id
        if len(task.dataset) != 1 or task.dataset[0].id != sample_id:
            raise ValueError("Select exactly one native task sample before running the strict bridge.")
        self._generator.configure_native_limits(task)
        self._report = self._snapshot_report()
        if self._report.submissions:
            raise ValueError("A new attempt requires fresh binding-owned submission state.")
        if self._report.status is not SubmissionReportStatus.NO_SUBMISSION:
            raise ValueError("A new attempt requires a healthy no-submission report before any provider dispatch.")
        self._artifacts.manifest.update(
            mode="OFFLINE/SIMULATED",
            contract_version="strict-submission-v1",
            completion_policy="strict-after-tool; native no-input continuation",
            conversation_id=self._generator.conversation_id,
            harness_status="running",
            remote_stop_claim=False,
            max_tool_output_bytes=self._generator.max_tool_output_bytes,
            native_limits={
                "message_limit": task.message_limit,
                "turn_limit": task.turn_limit,
                "token_limit": task.token_limit,
                "token_limit_type": task.token_limit_type,
                "time_limit": task.time_limit,
            },
        )
        await self._artifacts.save_async()
        logs: list[EvalLog] = []
        interrupted: BaseException | None = None
        evaluation: asyncio.Task[list[EvalLog]] | None = None
        try:
            timeout_seconds = min(
                self._limits.episode_timeout_seconds,
                task.time_limit if task.time_limit is not None else self._limits.episode_timeout_seconds,
            )
            async with asyncio.timeout(timeout_seconds):
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
                            "evidence_label": "OFFLINE/SIMULATED",
                        },
                    )
                )
                logs = await asyncio.shield(evaluation)
                if self._generator.callback_cancellation is not None:
                    if len(logs) == 1:
                        self._artifacts.manifest.update(
                            native_log=logs[0].location or None,
                            native_eval_id=logs[0].eval.eval_id,
                        )
                    raise self._generator.callback_cancellation
        except BaseException as error:
            interrupted = error
            self._generator.stop()
            if evaluation is not None and not evaluation.done():
                evaluation.cancel()
            self._runner_error = f"{type(error).__name__}: {error}"
            self._artifacts.manifest.update(harness_status="error", runner_error=self._runner_error)
            raise
        finally:
            try:
                await self._finish_environment_async(evaluation=evaluation, original_error=interrupted)
            except BaseException as cleanup_error:
                if interrupted is not None:
                    if cleanup_error is interrupted:
                        raise
                    interrupted.add_note(f"Caller cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}")
                    raise interrupted from cleanup_error
                interrupted = cleanup_error
                raise
            finally:
                if interrupted is not None:
                    await self._retain_interrupted_result_async(error=interrupted, expectation=expectation)
        try:
            return await self._accept_async(
                logs=logs, sample_id=sample_id, native_scorer=native_scorer, expectation=expectation
            )
        except BaseException as error:
            self._last_result = None
            self._artifacts.manifest.update(
                harness_status="error",
                grade_status="undetermined",
                acceptance_error=f"{type(error).__name__}: {error}",
            )
            if isinstance(error, (asyncio.CancelledError, TimeoutError)):
                await self._retain_interrupted_result_async(error=error, expectation=expectation)
            else:
                try:
                    await self._artifacts.save_async()
                except BaseException as retention_error:
                    error.add_note(
                        f"Acceptance diagnostic retention failed: {type(retention_error).__name__}: {retention_error}"
                    )
                    logger.warning("Could not retain acceptance failure diagnostic: %s", retention_error)
            raise

    async def _settle_cancelled_evaluation_async(
        self, *, evaluation: asyncio.Task[list[EvalLog]], original_error: BaseException
    ) -> None:
        try:
            await asyncio.shield(evaluation)
        except asyncio.CancelledError:
            if not evaluation.done():
                raise
            self._artifacts.manifest["native_cancel_settled"] = evaluation.done()
        except Exception as error:
            original_error.add_note(f"Local Inspect cancellation cleanup: {type(error).__name__}: {error}")
            self._artifacts.manifest["native_cancel_settled"] = evaluation.done()
        else:
            self._artifacts.manifest["native_cancel_settled"] = True

    async def _retain_interrupted_result_async(
        self, *, error: BaseException, expectation: ScoringExpectation | None
    ) -> None:
        async def retain_async() -> None:
            async with asyncio.timeout(self._CANCELLATION_SETTLE_SECONDS):
                await self._write_interrupted_result_async(error=error, expectation=expectation)

        retention = asyncio.create_task(retain_async())
        try:
            while not retention.done():
                try:
                    await asyncio.shield(retention)
                except asyncio.CancelledError:
                    error.add_note("Additional caller cancellation arrived during bounded report retention.")
            retention.result()
        except BaseException as retention_error:
            error.add_note(f"Retaining interrupted report failed: {type(retention_error).__name__}: {retention_error}")
            logger.warning("Could not retain interrupted submission report: %s", retention_error)

    async def _write_interrupted_result_async(
        self, *, error: BaseException, expectation: ScoringExpectation | None
    ) -> None:
        status = (
            SubmissionReportStatus.CANCELLED
            if isinstance(error, asyncio.CancelledError)
            else SubmissionReportStatus.INCOMPLETE
            if isinstance(error, TimeoutError)
            else SubmissionReportStatus.ERROR
        )
        retained = RetainedSubmissionReport(
            run_id=self._artifacts.run_id,
            conversation_id=self._generator.conversation_id,
            status=status,
            report=self._report,
            calls=tuple(
                self._call_evidence(call) for call in self._generator.tool_calls if call["status"] != "not_dispatched"
            ),
            final_text=(self._active_state.output.completion or None) if self._active_state else None,
            runner_error="\n".join([f"{type(error).__name__}: {error}", *getattr(error, "__notes__", [])]),
            termination_reason=SubmissionTerminationReason.RUNNER_ERROR,
            generation_count=self._generator.requests,
            limits=asdict(self._limits),
        )
        await self._persist_result_async(
            retained=retained,
            expectation=expectation,
            native_log=Path(self._artifacts.manifest["native_log"])
            if self._artifacts.manifest.get("native_log")
            else None,
            native_eval_id=self._artifacts.manifest.get("native_eval_id"),
            sample_uuid=self._active_state.uuid if self._active_state else None,
        )

    def _solver(self, native_solver: Solver) -> Solver:
        from inspect_ai.solver import Generate, solver

        @solver("pyrit_native_solver_bridge")
        def bridge() -> Solver:
            async def solve_async(state: TaskState, generate: Generate) -> TaskState:
                self._active_state = state
                if state.epoch != 1:
                    raise ValueError("The strict bridge supports one native epoch per independent attempt.")
                try:
                    await self._environment.audit_async(state)
                    result = await native_solver(state, self._generator.generate_async)
                    if result is not state:
                        raise ValueError("Replacing TaskState is unsupported by this identity-bound bridge.")
                    return state
                except BaseException as error:
                    self._runner_error = f"{type(error).__name__}: {error}"
                    self._limit_reached = isinstance(error, InspectNativeLimitError)
                    raise
                finally:
                    try:
                        self._capture_report()
                    except (ValueError, TypeError) as error:
                        self._runner_error = self._runner_error or f"Invalid binding report: {error}"
                    state.metadata["pyrit_strict_report"] = (
                        self._report.model_dump(mode="json") if self._report else None
                    )
                    state.metadata["pyrit_conversation_id"] = self._generator.conversation_id
                    if self._generator.target is not None:
                        await self._generator.retain_pending_async(state)
                    await self._artifacts.append_async(
                        event="strict_native_state",
                        data={
                            "sample_id": state.sample_id,
                            "sample_uuid": state.uuid,
                            "report": state.metadata["pyrit_strict_report"],
                            "runner_error": self._runner_error,
                        },
                    )

            return solve_async

        return bridge()

    async def _after_tool_async(self) -> bool:
        self._capture_report()
        if self._report is None:
            raise ValueError("The binding did not provide a valid strict report.")
        await self._artifacts.append_async(event="strict_report_snapshot", data=self._report.model_dump(mode="json"))
        if self._report.status in self._TERMINAL_STATUSES:
            raise RuntimeError(f"Strict acquisition stopped as {self._report.status.value}; no further dispatch.")
        return self._report.full_success

    def _snapshot_report(self) -> StrictSubmissionReport:
        return StrictSubmissionReport.model_validate(self._read_report())

    def _capture_report(self) -> None:
        self._report = self._snapshot_report()
        newly_observed = tuple(
            item.submission_id for item in self._report.submissions if item.submission_id not in self._seen_submissions
        )
        if newly_observed and self._generator.tool_calls:
            call = self._generator.tool_calls[-1]
            call["submission_ids"] = list(call.get("submission_ids", [])) + list(newly_observed)
        self._seen_submissions.update(newly_observed)

    async def _finish_environment_async(
        self, *, evaluation: asyncio.Task[list[EvalLog]] | None = None, original_error: BaseException | None = None
    ) -> None:
        if self._finalization_task is None:

            async def finalize_async() -> None:
                async with asyncio.timeout(self._CANCELLATION_SETTLE_SECONDS):
                    if original_error is not None and evaluation is not None:
                        await self._settle_cancelled_evaluation_async(
                            evaluation=evaluation, original_error=original_error
                        )
                    await self._reconcile_cancelled_log_async(original_error)
                    await self._finish_environment_impl_async()

            self._finalization_task = asyncio.create_task(finalize_async())
            self._finalization_deadline = asyncio.get_running_loop().time() + self._CANCELLATION_SETTLE_SECONDS
        finalization = self._finalization_task
        cancellation = original_error if isinstance(original_error, asyncio.CancelledError) else None
        try:
            while not finalization.done():
                try:
                    deadline = self._finalization_deadline
                    if deadline is None:
                        raise RuntimeError("The owned finalization task has no deadline.")
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        raise TimeoutError("The owned finalization deadline expired.")
                    await asyncio.wait_for(asyncio.shield(finalization), timeout=remaining)
                except asyncio.CancelledError as error:
                    cancellation = cancellation or error
            finalization.result()
        except BaseException as error:
            if not finalization.done():
                finalization.cancel()
                finalization.add_done_callback(self._observe_late_finalization)
            self._artifacts.manifest.update(
                canonical_messages_status="unknown",
                pending_message_ids=self._generator.pending_message_ids(),
                finalization_error=f"{type(error).__name__}: {error}",
            )
            self._artifacts.manifest.setdefault("local_environment_cleanup", "unknown")
            if cancellation is not None:
                cancellation.add_note(f"Bounded local persistence finalization failed: {type(error).__name__}: {error}")
                raise cancellation from error
            raise
        if cancellation is not None and cancellation is not original_error:
            raise cancellation

    @staticmethod
    def _observe_late_finalization(task: asyncio.Task[None]) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning("Owned finalization finished after its deadline with an error: %s", error)

    async def _finish_environment_impl_async(self) -> None:
        try:
            self._capture_report()
        except (ValueError, TypeError) as error:
            self._runner_error = f"Invalid binding report: {error}"
        retention_error: BaseException | None = None
        try:
            if (
                self._active_state is not None
                and self._generator.target is not None
                and self._artifacts.manifest.get("native_cancel_settled", True)
            ):
                await self._generator.retain_pending_async(self._active_state)
                self._artifacts.manifest["canonical_messages_status"] = "complete"
        except asyncio.CancelledError:
            self._artifacts.manifest["canonical_messages_status"] = "unknown"
            raise
        except BaseException as error:
            retention_error = error
            self._runner_error = self._runner_error or f"Authentic message retention failed: {error}"
        try:
            await self._environment.cleanup_async()
            self._artifacts.manifest["local_environment_cleanup"] = "verified_by_caller"
        except BaseException as error:
            self._artifacts.manifest["local_environment_cleanup"] = "unknown"
            self._artifacts.manifest["local_cleanup_error"] = f"{type(error).__name__}: {error}"
            self._runner_error = self._runner_error or f"Environment cleanup failed: {error}"
            raise
        finally:
            self._artifacts.manifest.update(
                provider_requests=self._generator.requests,
                provider_reported_usage=self._generator.usage,
                termination_reason=self._generator.termination_reason,
                tool_calls=self._generator.tool_calls,
                binding_report=self._report.model_dump(mode="json") if self._report else None,
                runner_error=self._runner_error,
                remote_stop_claim=False,
            )
            await self._artifacts.save_async()
        if retention_error is not None:
            raise retention_error

    async def _reconcile_cancelled_log_async(self, original_error: BaseException | None) -> None:
        from inspect_ai.log import list_eval_logs_async, read_eval_log_async

        directory = self._artifacts.directory / "native"
        matches: list[tuple[Path, EvalLog, EvalSample]] = []
        try:
            for info in await list_eval_logs_async(str(directory), formats=["eval"], recursive=False):
                try:
                    path = await asyncio.to_thread(self._owned_log_path, location=info.name, directory=directory)
                    log = await read_eval_log_async(path, resolve_attachments=True)
                except Exception as error:
                    diagnostic = f"Skipped unavailable native log candidate: {type(error).__name__}: {error}"
                    if original_error is not None:
                        original_error.add_note(diagnostic)
                    logger.warning("%s", diagnostic)
                    continue
                metadata = log.eval.metadata or {}
                if (
                    log.status == "started"
                    or not log.stats.completed_at
                    or metadata.get("pyrit_run_id") != self._artifacts.run_id
                    or metadata.get("pyrit_attempt_id") != self._artifacts.attempt_id
                ):
                    continue
                sample = await asyncio.to_thread(self._materialize_sample, log)
                if (
                    sample.id != self._expected_sample_id
                    or sample.epoch != 1
                    or not sample.uuid
                    or (self._active_state is not None and sample.uuid != self._active_state.uuid)
                ):
                    continue
                matches.append((path, log, sample))
            if len(matches) > 1:
                raise ValueError("Multiple finalized native logs match this one independent attempt.")
            if not matches:
                self._artifacts.manifest.update(native_log=None, native_eval_id=None, native_log_recovery="unavailable")
                return
            path, log, sample = matches[0]
            self._artifacts.manifest.update(
                native_log=str(path),
                native_eval_id=log.eval.eval_id,
                native_sample_uuid=sample.uuid,
                native_log_recovery="identity_verified_finalized",
                native_status=log.status,
                native_completed_at=log.stats.completed_at,
            )
        except Exception as error:
            diagnostic = f"Finalized native log recovery unavailable: {type(error).__name__}: {error}"
            if original_error is not None:
                original_error.add_note(diagnostic)
            self._artifacts.manifest.update(native_log=None, native_eval_id=None, native_log_recovery=diagnostic)
            logger.warning("%s", diagnostic)

    @staticmethod
    def _owned_log_path(*, location: str, directory: Path) -> Path:
        if location.startswith("file:"):
            parsed = urlsplit(location)
            if parsed.netloc and parsed.netloc != "localhost":
                if len(parsed.netloc) != 2 or parsed.netloc[1] != ":":
                    raise ValueError("Native recovery must not access a remote file host.")
                location = parsed.netloc + unquote(parsed.path)
            else:
                location = unquote(parsed.path)
                if len(location) > 2 and location[0] == "/" and location[2] == ":":
                    location = location[1:]
        elif "://" in location:
            raise ValueError("Native recovery only reads local owned log files.")
        path = Path(location).resolve()
        if path.parent != directory.resolve():
            raise ValueError("Native log path is outside this owned run directory.")
        return path

    async def _accept_async(
        self,
        *,
        logs: list[EvalLog],
        sample_id: str | int,
        native_scorer: str,
        expectation: ScoringExpectation | None,
    ) -> InspectNativeTaskResult:
        from inspect_ai.log import read_eval_log_async

        if len(logs) != 1 or not logs[0].location:
            raise ValueError("Expected one retained native log, including error evidence.")
        log = await read_eval_log_async(logs[0].location, resolve_attachments=True)
        sample = await asyncio.to_thread(self._materialize_sample, log)
        self._artifacts.manifest.update(
            native_log=log.location,
            native_eval_id=log.eval.eval_id,
            native_sample_uuid=sample.uuid,
            native_grade={name: grade.model_dump(mode="json") for name, grade in (sample.scores or {}).items()},
        )
        await self._artifacts.append_async(event="strict_native_sample", data=sample.model_dump(mode="json"))
        if sample.id != sample_id or sample.epoch != 1 or sample.uuid is None:
            raise ValueError("Native sample identity does not match this attempt.")
        metadata = log.eval.metadata or {}
        if (
            metadata.get("pyrit_run_id") != self._artifacts.run_id
            or metadata.get("pyrit_attempt_id") != self._artifacts.attempt_id
        ):
            raise ValueError("Native log identity does not match this attempt.")
        status = self._report.status if self._report else SubmissionReportStatus.ERROR
        error = self._runner_error
        if self._limit_reached:
            status = SubmissionReportStatus.INCOMPLETE
        elif (
            log.status != "success"
            or log.invalidated
            or sample.error
            or sample.invalidation
            or sample.error_retries
            or (sample.limit is not None and self._generator.termination_reason != "budget")
            or error
        ):
            status = status if status in self._TERMINAL_STATUSES else SubmissionReportStatus.ERROR
            error = error or (sample.error.message if sample.error else f"Native evaluation status: {log.status}")
        if status is SubmissionReportStatus.COMPLETED:
            self._validate_native_grade(sample=sample, scorer_name=native_scorer)
        retained = RetainedSubmissionReport(
            run_id=self._artifacts.run_id,
            conversation_id=self._generator.conversation_id,
            status=status,
            report=self._report,
            calls=tuple(
                self._call_evidence(call) for call in self._generator.tool_calls if call["status"] != "not_dispatched"
            ),
            final_text=sample.output.completion or None,
            runner_error=error,
            termination_reason=(
                SubmissionTerminationReason(self._generator.termination_reason)
                if self._generator.termination_reason
                else SubmissionTerminationReason.BINDING_STATE
                if self._report and self._report.status in self._TERMINAL_STATUSES
                else SubmissionTerminationReason.RUNNER_ERROR
                if error
                else None
            ),
            generation_count=self._generator.requests,
            limits=asdict(self._limits),
        )
        self._artifacts.manifest.update(
            harness_status="completed" if log.status == "success" and error is None else "error",
        )
        return await self._persist_result_async(
            retained=retained,
            expectation=expectation,
            native_log=Path(log.location),
            native_eval_id=log.eval.eval_id,
            sample_uuid=sample.uuid,
        )

    async def _persist_result_async(
        self,
        *,
        retained: RetainedSubmissionReport,
        expectation: ScoringExpectation | None,
        native_log: Path | None,
        native_eval_id: str | None,
        sample_uuid: str | None,
    ) -> InspectNativeTaskResult:
        from pyrit.memory import CentralMemory

        if CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("CentralMemory changed during the identity-bound offline run.")
        path = self._artifacts.directory / f"submission-report-{retained.sha256()}.json"
        async with aiofiles.open(path, "x", encoding="utf-8", newline="\n") as stream:
            await stream.write(retained.canonical_json())
        self._artifacts.manifest.update(
            publication_state="unscored_candidate",
            publication_boundary="pyrit_score_commit",
            grade_status="not_published",
            evidence_status=retained.status.value,
            report_sha256=retained.sha256(),
            report_path=str(path),
        )
        await self._artifacts.save_async()
        await asyncio.sleep(0)
        current = asyncio.current_task()
        if current is not None and current.cancelling():
            raise asyncio.CancelledError("Caller cancellation was accepted before score publication.")
        if CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("CentralMemory changed before offline score publication.")
        # This exact loose-text scorer path commits synchronously without yielding.
        # No await is permitted between that commit, result publication, and return.
        scores = await SubmissionReportScorer(report_sha256=retained.sha256()).score_async(
            scorable=ContentScorable(value=retained.canonical_json(), data_type="text"), expectation=expectation
        )
        result = InspectNativeTaskResult(
            report=retained,
            score=scores[0],
            report_path=path,
            native_log=native_log,
            native_eval_id=native_eval_id,
            native_sample_uuid=sample_uuid,
        )
        self._artifacts.manifest.update(
            publication_state="published_in_memory",
            grade_status=scores[0].status.value,
            evidence_status=retained.status.value,
            report_sha256=retained.sha256(),
            report_path=str(path),
            score=scores[0].model_dump(mode="json"),
        )
        self._last_result = result
        return result

    def _validate_native_grade(self, *, sample: EvalSample, scorer_name: str) -> None:
        if self._report is None or sample.metadata.get("pyrit_strict_report") != self._report.model_dump(mode="json"):
            raise ValueError("The native stored report differs from the retained submission report.")
        grade = (sample.scores or {}).get(scorer_name)
        value = grade.value if grade is not None else None
        if type(value) not in (int, float) or not isinstance(value, (int, float)):
            raise ValueError("Native submission grade is missing or not numeric.")
        if not math.isfinite(value) or not 0 <= value <= 1 or value != self._report.last_valid_grade:
            raise ValueError("Native numeric grade does not equal the latest acquired valid submission grade.")

    @staticmethod
    def _call_evidence(call: dict[str, Any]) -> SubmissionCallEvidence:
        result = call.get("result")
        if isinstance(result, dict):
            error = result.get("error")
            feedback = f"Error: {error['message']}" if error else result["content"]
            kind = (
                SubmissionFeedbackKind.CANCELLED
                if error and error["type"] == "cancelled"
                else SubmissionFeedbackKind.RECOVERABLE_ERROR
                if error
                else SubmissionFeedbackKind.RETURNED
            )
            if kind is SubmissionFeedbackKind.CANCELLED:
                feedback = None
        else:
            feedback = None
            kind = (
                SubmissionFeedbackKind.CANCELLED
                if call.get("error_type") == "CancelledError"
                else SubmissionFeedbackKind.TERMINAL_ERROR
            )
        return SubmissionCallEvidence(
            call_id=call["provider_call_id"],
            tool_name=call["name"],
            arguments_json=call["arguments_json"],
            feedback=feedback,
            feedback_kind=kind,
            submission_ids=tuple(call.get("submission_ids", [])),
        )

    @staticmethod
    def _materialize_sample(log: EvalLog) -> EvalSample:
        from inspect_ai.log import EvalSample

        samples = list(log.samples or [])
        if len(samples) != 1:
            raise ValueError("Expected exactly one retained native sample.")
        return EvalSample.model_validate(samples[0].model_dump(mode="json"))
