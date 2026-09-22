# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import logging
import sys
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, TypeVar
from uuid import uuid4

import httpx

from pyrit.executor.benchmark.submission.evidence import SubmissionEvidenceWriter
from pyrit.executor.benchmark.submission.hooks import SubmissionHooks, SubmissionLimits
from pyrit.memory import CentralMemory, SQLiteMemory, set_message_piece_sha256_async
from pyrit.models import ContentScorable, Message, MessagePiece
from pyrit.models.submission import (
    RetainedSubmissionReport,
    StrictSubmissionReport,
    SubmissionCallEvidence,
    SubmissionFeedbackKind,
    SubmissionReportStatus,
    SubmissionTerminationReason,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.score import SubmissionReportScorer

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from pathlib import Path

    from pyrit.models import Score, SeedPrompt

_RetainedT = TypeVar("_RetainedT")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class SubmissionRunResult:
    """A content-anchored offline result, which need not contain final assistant text."""

    report: RetainedSubmissionReport
    report_path: Path
    score: Score


class SubmissionRunError(RuntimeError):
    """An unsuccessful runner operation with durable report and score evidence."""

    def __init__(self, *, result: SubmissionRunResult) -> None:
        """Retain the undetermined result while surfacing the operational failure."""
        super().__init__(result.report.runner_error or f"Submission run ended with {result.report.status.value}.")
        self.result = result


class NativeSubmissionRunner:
    """Drive a caller-owned strict submission binding through a mocked Responses target."""

    _RETENTION_TIMEOUT_SECONDS = 5.0

    def __init__(self, *, hooks: SubmissionHooks, directory: Path, limits: SubmissionLimits | None = None) -> None:
        """
        Initialize a single-use, offline-only executor with no evaluator or credentials of its own.

        Raises:
            TypeError: If memory is not the supported synchronous SQLite backend.
        """
        self._hooks = hooks
        self._limits = limits or SubmissionLimits()
        self._run_id = str(uuid4())
        self._writer = SubmissionEvidenceWriter(directory=directory / self._run_id)
        self._memory = CentralMemory.get_memory_instance()
        if type(self._memory) is not SQLiteMemory:
            raise TypeError("Offline submission publication supports only synchronous SQLiteMemory.")
        self._normalizer = PromptNormalizer()
        self._report: StrictSubmissionReport | None = None
        self._calls: list[SubmissionCallEvidence] = []
        self._seen_call_ids: set[str] = set()
        self._requests = 0
        self._started = False
        self._pending_tool: Message | None = None
        self._final_text: str | None = None
        self._failure: BaseException | None = None
        self._first_cancellation: asyncio.CancelledError | None = None
        self._last_projection_json: str | None = None
        self._retention_failures: list[dict[str, str]] = []
        self._owner_task: asyncio.Task[Any] | None = None
        self._initial_cancellations = 0
        self._finalizing = False
        self._publication_cancelled = False
        self._published = False
        self._termination_reason: SubmissionTerminationReason | None = None
        self.last_result: SubmissionRunResult | None = None

    async def run_offline_async(
        self, *, seed: SeedPrompt, system_prompt: str, transport: httpx.MockTransport
    ) -> SubmissionRunResult:
        """
        Run the real target and normalizer using only an explicitly supplied MockTransport.

        The injected Python hooks are trusted test code, not isolated by this factory.
        Their submission policy, artifact acquisition and evaluator belong to the caller.

        Returns:
            SubmissionRunResult: The immutable report and its persisted content score.

        Raises:
            ValueError: If transport is not mocked or the runner has already been used.
            SubmissionRunError: If generation, binding, or report acquisition fails.
            asyncio.CancelledError: If the caller cancels; ``last_result`` retains partial evidence.
        """
        if not isinstance(transport, httpx.MockTransport):
            raise ValueError("Only an explicit httpx.MockTransport is supported; live execution is not implemented.")
        if self._started:
            raise ValueError("A submission runner is single-use.")
        self._owner_task = asyncio.current_task()
        self._initial_cancellations = self._owner_task.cancelling() if self._owner_task else 0
        self._started = True
        await self._writer.initialize_async()
        try:
            async with asyncio.timeout(self._limits.episode_timeout_seconds):
                await self._snapshot_async()
                if self._report is None or self._report.submissions:
                    raise ValueError("Each offline run requires fresh binding-owned submission state.")
                await self._run_target_async(seed=seed, system_prompt=system_prompt, transport=transport)
        except (Exception, asyncio.CancelledError) as error:
            if isinstance(error, asyncio.CancelledError):
                self._failure = self._first_cancellation or error
            else:
                self._failure = error
                if isinstance(error, TimeoutError):
                    self._first_cancellation = None
        self._finalizing = True
        try:
            result = await self._retain_async(self._finish_async())
        except (Exception, asyncio.CancelledError) as retention_error:
            if isinstance(self._failure, asyncio.CancelledError) and retention_error is not self._failure:
                raise self._failure from retention_error
            raise
        if isinstance(self._failure, asyncio.CancelledError):
            raise self._failure
        if self._failure is not None:
            raise SubmissionRunError(result=result) from self._failure
        return result

    async def _run_target_async(self, *, seed: SeedPrompt, system_prompt: str, transport: httpx.MockTransport) -> None:
        async with httpx.AsyncClient(
            transport=transport,
            trust_env=False,
            event_hooks={"request": [self._request_async], "response": [self._response_async]},
        ) as client:
            target = OpenAIResponseTarget(
                endpoint="https://offline.invalid/v1",
                model_name="offline-simulated",
                api_key="offline-simulated-not-a-credential",
                headers="{}",
                auto_execute_tools=False,
                max_output_tokens=2048,
                fail_on_missing_function=True,
                extra_body_parameters={
                    "tools": [tool.response_definition() for tool in self._hooks.tools],
                    "parallel_tool_calls": False,
                    "store": False,
                },
                httpx_client_kwargs={"http_client": client, "max_retries": 0},
            )
            await self._memory.add_seeds_to_memory_async(seeds=[seed], added_by="offline_submission")
            await asyncio.to_thread(target.set_system_prompt, system_prompt=system_prompt, conversation_id=self._run_id)
            await self._generate_loop_async(
                target=target,
                first_message=Message.from_prompt(
                    prompt=seed.value, role="user", prompt_metadata={"evidence_label": "OFFLINE/SIMULATED"}
                ),
            )

    async def _generate_loop_async(self, *, target: OpenAIResponseTarget, first_message: Message) -> None:
        message: Message | None = first_message
        while True:
            if self._requests >= self._limits.max_requests or len(self._calls) >= self._limits.max_tool_calls:
                self._termination_reason = SubmissionTerminationReason.BUDGET
                return
            if message is None:
                response = await self._normalizer.continue_conversation_async(
                    target=target, conversation_id=self._run_id
                )
            else:
                response = await self._normalizer.send_prompt_async(
                    message=message, target=target, conversation_id=self._run_id
                )
            self._pending_tool = None
            if response.api_role != "assistant" or any(
                piece.has_error() or piece.is_truncated for piece in response.message_pieces
            ):
                raise RuntimeError("Provider completion was missing, truncated, or erroneous.")
            calls = response.get_pieces_by_type(data_type="function_call")
            texts = response.get_pieces_by_type(data_type="text")
            self._final_text = "\n".join(piece.converted_value for piece in texts) or None
            if not calls:
                message = None
                continue
            if len(calls) != 1:
                raise RuntimeError("This offline policy requires sequential, not parallel, tool calls.")
            message = await self._invoke_tool_async(calls[0])
            self._pending_tool = message
            if self._report is None:
                raise RuntimeError("A tool completed without an acquired binding report.")
            if self._report.full_success or self._report.status not in {
                SubmissionReportStatus.COMPLETED,
                SubmissionReportStatus.NO_SUBMISSION,
            }:
                self._termination_reason = (
                    SubmissionTerminationReason.FULL_SUCCESS
                    if self._report.full_success and self._report.status is SubmissionReportStatus.COMPLETED
                    else SubmissionTerminationReason.BINDING_STATE
                )
                self._final_text = None
                return

    async def _invoke_tool_async(self, piece: MessagePiece) -> Message:
        call = json.loads(piece.converted_value)
        if not isinstance(call, dict) or any(
            not isinstance(call.get(key), str) or not call[key] for key in ("call_id", "name", "arguments")
        ):
            raise ValueError("A tool invocation requires an actual provider call ID, name, and JSON arguments.")
        call_id, name, arguments_json = call["call_id"], call["name"], call["arguments"]
        if call_id in self._seen_call_ids or len(self._seen_call_ids) >= self._limits.max_tool_calls:
            raise RuntimeError("Tool identity was reused or the offline tool-call limit was reached.")
        self._seen_call_ids.add(call_id)
        arguments = json.loads(arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        tool = next((tool for tool in self._hooks.tools if tool.name == name), None)
        if tool is None:
            raise ValueError(f"Unregistered injected tool: {name}")
        previous_ids = {item.submission_id for item in self._report.submissions} if self._report else set()
        await self._writer.append_async(
            event="tool_requested",
            data={
                "call_id": call_id,
                "tool_name": name,
                "arguments": arguments,
                "local_invocation_id": str(uuid4()),
            },
        )
        feedback: str | None = None
        kind = SubmissionFeedbackKind.RETURNED
        try:
            feedback = await tool.callback_async(**arguments)
            if not isinstance(feedback, str):
                feedback = None
                raise TypeError("Injected tools must return strings without JSON output wrapping.")
        except self._hooks.recoverable_errors as error:
            kind = SubmissionFeedbackKind.RECOVERABLE_ERROR
            try:
                feedback = self._hooks.error_feedback(error)
            except Exception:
                kind = SubmissionFeedbackKind.TERMINAL_ERROR
                raise
            if not isinstance(feedback, str):
                kind = SubmissionFeedbackKind.TERMINAL_ERROR
                feedback = None
                raise TypeError("The native tool-error feedback mapper must return a string.") from error
        except asyncio.CancelledError as error:
            self._first_cancellation = self._first_cancellation or error
            kind = SubmissionFeedbackKind.CANCELLED
            raise
        except Exception:
            kind = SubmissionFeedbackKind.TERMINAL_ERROR
            raise
        finally:
            callback_error = sys.exception()
            snapshot_error: Exception | None = None
            try:
                self._capture_report()
            except Exception as error:
                snapshot_error = error
                if not isinstance(callback_error, asyncio.CancelledError):
                    kind = SubmissionFeedbackKind.TERMINAL_ERROR
            ids = (
                tuple(item.submission_id for item in self._report.submissions if item.submission_id not in previous_ids)
                if self._report
                else ()
            )
            evidence = SubmissionCallEvidence(
                call_id=call_id,
                tool_name=name,
                arguments_json=arguments_json,
                feedback=feedback,
                feedback_kind=kind,
                submission_ids=ids,
            )
            self._calls.append(evidence)
            self._capture_tool_message(evidence)
            # The report, correlation, and genuine output now exist before cancellation can be delivered.
            try:
                await self._retain_async(self._retain_call_async(evidence=evidence, snapshot_error=snapshot_error))
            except (Exception, asyncio.CancelledError) as retention_error:
                if isinstance(callback_error, asyncio.CancelledError) and retention_error is not callback_error:
                    raise callback_error from retention_error
                raise
            if snapshot_error is not None:
                if isinstance(callback_error, asyncio.CancelledError):
                    raise callback_error from snapshot_error
                raise snapshot_error
        if self._pending_tool is None:
            raise RuntimeError("The callback did not provide an actual string tool result.")
        return self._pending_tool

    def _capture_tool_message(self, evidence: SubmissionCallEvidence) -> None:
        if evidence.feedback is None:
            return
        self._pending_tool = MessagePiece(
            role="tool",
            conversation_id=self._run_id,
            original_value_data_type="function_call_output",
            original_value=json.dumps(
                {"type": "function_call_output", "call_id": evidence.call_id, "output": evidence.feedback},
                separators=(",", ":"),
            ),
            prompt_metadata={"evidence_label": "OFFLINE/SIMULATED", "feedback_kind": evidence.feedback_kind.value},
        ).to_message()

    async def _snapshot_async(self) -> None:
        self._capture_report()
        await self._retain_projection_async()

    def _capture_report(self) -> None:
        self._last_projection_json = None
        snapshot = self._hooks.read_report()
        # Detach all nested values from the binding before retaining or validating them.
        raw = json.dumps(snapshot, ensure_ascii=True, allow_nan=False)
        self._last_projection_json = raw
        report = StrictSubmissionReport.model_validate_json(raw)
        if self._report is not None and report.submissions[: len(self._report.submissions)] != self._report.submissions:
            raise ValueError("A binding must not replace or rewrite previously retained submission observations.")
        self._report = report

    async def _retain_projection_async(self) -> None:
        if self._last_projection_json is not None:
            await self._writer.append_async(event="binding_projection", data={"json": self._last_projection_json})
        if self._report is not None:
            await self._writer.snapshot_async(self._report)

    async def _retain_call_async(self, *, evidence: SubmissionCallEvidence, snapshot_error: Exception | None) -> None:
        await self._retain_projection_async()
        if snapshot_error is not None:
            await self._writer.append_async(
                event="report_acquisition_failed",
                data={"error_type": type(snapshot_error).__name__, "message": str(snapshot_error)},
            )
        await self._writer.append_async(event="tool_finished", data=evidence.model_dump(mode="json"))

    async def _finish_async(self) -> SubmissionRunResult:
        for failure in self._retention_failures:
            await self._writer.append_async(event="retention_failed", data=failure)
        if self._failure is not None:
            await self._writer.append_async(
                event="runner_failed",
                data={"error_type": type(self._failure).__name__, "message": str(self._failure)},
            )
        if self._pending_tool is not None:
            piece = self._pending_tool.get_piece()
            existing = await asyncio.to_thread(self._memory.get_message_pieces, prompt_ids=[piece.id])
            if not existing:
                await set_message_piece_sha256_async(piece)
                await asyncio.to_thread(self._memory.add_message_to_memory, request=self._pending_tool)
        while True:
            report = self._build_final_report()
            path = await self._writer.retain_async(report)
            if report != self._build_final_report():
                continue
            if CentralMemory.get_memory_instance() is not self._memory:
                raise RuntimeError("The SQLite memory instance changed before report publication.")
            # This fixed text-content scorer commits through synchronous SQLite memory without yielding.
            scores = await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
                scorable=ContentScorable(value=report.canonical_json(), data_type="text")
            )
            self._writer.mark_published(path)
            self._published = True
            self.last_result = SubmissionRunResult(report=report, report_path=path, score=scores[0])
            return self.last_result

    def _build_final_report(self) -> RetainedSubmissionReport:
        if self._owner_task and self._owner_task.cancelling() > self._initial_cancellations:
            self._publication_cancelled = True
        status = self._report.status if self._report else SubmissionReportStatus.ERROR
        runner_error = f"{type(self._failure).__name__}: {self._failure}" if self._failure is not None else None
        if self._failure is not None or self._publication_cancelled:
            self._termination_reason = SubmissionTerminationReason.RUNNER_ERROR
            if isinstance(self._failure, asyncio.CancelledError) or self._publication_cancelled:
                status = SubmissionReportStatus.CANCELLED
                if runner_error is None:
                    runner_error = (
                        f"CancelledError: {self._first_cancellation}"
                        if self._first_cancellation
                        else "Cancellation requested before final report publication."
                    )
            elif status in {SubmissionReportStatus.COMPLETED, SubmissionReportStatus.NO_SUBMISSION}:
                status = SubmissionReportStatus.INCOMPLETE
        return RetainedSubmissionReport(
            run_id=self._run_id,
            conversation_id=self._run_id,
            status=status,
            report=self._report,
            calls=tuple(self._calls),
            final_text=self._final_text,
            runner_error=runner_error,
            termination_reason=self._termination_reason,
            generation_count=self._requests,
            limits=asdict(self._limits),
        )

    async def _retain_async(self, operation: Coroutine[Any, Any, _RetainedT]) -> _RetainedT:
        task = asyncio.create_task(operation)
        deadline = asyncio.get_running_loop().time() + self._RETENTION_TIMEOUT_SECONDS
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                task.cancel()
                task.add_done_callback(self._retention_done)
                error = TimeoutError("Offline evidence retention exceeded its bounded deadline.")
                self._retention_failures.append({"error_type": type(error).__name__, "message": str(error)})
                logger.error("%s", error)
                if cancellation is not None:
                    raise cancellation from error
                raise error
            try:
                await asyncio.wait({task}, timeout=remaining)
            except asyncio.CancelledError as error:
                self._first_cancellation = self._first_cancellation or error
                if self._finalizing and not self._published:
                    self._publication_cancelled = True
                cancellation = self._first_cancellation
        try:
            result = task.result()
        except (Exception, asyncio.CancelledError) as error:
            self._retention_failures.append({"error_type": type(error).__name__, "message": str(error)})
            logger.error("Offline evidence retention failed: %s", error)
            if cancellation is not None:
                raise cancellation from error
            raise
        if cancellation is not None:
            raise cancellation
        return result

    @staticmethod
    def _retention_done(task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("Offline evidence retention failed after its deadline: %s", task.exception())

    async def _request_async(self, request: httpx.Request) -> None:
        if self._requests >= self._limits.max_requests:
            raise RuntimeError("The offline provider-request limit was reached.")
        self._requests += 1
        await self._writer.append_async(
            event="provider_request",
            data={
                "sequence": self._requests,
                "body": json.loads(await request.aread()),
            },
        )

    async def _response_async(self, response: httpx.Response) -> None:
        payload = await response.aread()
        if len(payload) > self._limits.max_response_bytes:
            raise RuntimeError("Mocked provider response exceeded the local evidence limit.")
        await self._writer.append_async(
            event="provider_response",
            data={
                "sequence": self._requests,
                "http_status": response.status_code,
                "body": json.loads(payload),
            },
        )
