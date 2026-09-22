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

from pydantic import JsonValue, TypeAdapter

from pyrit.executor.benchmark.submission.evidence_v2 import SubmissionEvidenceWriterV2
from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionLimitsV2
from pyrit.memory import CentralMemory, SQLiteMemory, set_message_piece_sha256_async
from pyrit.models import ContentScorable, Message, MessagePiece
from pyrit.models.submission import (
    SubmissionCallEvidence,
    SubmissionCleanupStatus,
    SubmissionFeedbackKind,
    SubmissionReportStatus,
    SubmissionTerminationReason,
)
from pyrit.models.submission_v2 import RetainedSubmissionReportV2, StrictSubmissionReportV2, SubmissionProvenanceV2
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2

if TYPE_CHECKING:
    from collections.abc import Coroutine
    from contextlib import AbstractAsyncContextManager
    from pathlib import Path

    import httpx

    from pyrit.executor.benchmark.submission.hooks import SubmissionHooks
    from pyrit.executor.benchmark.submission.hooks_v2 import (
        SubmissionEnvironmentHooksV2,
        SubmissionTargetFactoryV2,
    )
    from pyrit.models import Score, SeedPrompt

_ResultT = TypeVar("_ResultT")
logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class SubmissionRunResultV2:
    """Published v2 content and score with no inference about remote execution."""

    report: RetainedSubmissionReportV2
    report_path: Path
    score: Score


class SubmissionRunV2Error(RuntimeError):
    """An unsuccessful v2 run with retained acquisition and lifecycle evidence."""

    def __init__(self, *, result: SubmissionRunResultV2) -> None:
        """Surface failure while retaining the published undetermined outcome."""
        super().__init__(result.report.runner_error or f"V2 run ended with {result.report.status.value}.")
        self.result = result


class NativeSubmissionRunnerV2:
    """Drive injected v2 tools using a caller-owned target and resource lifecycle."""

    _RETENTION_TIMEOUT_SECONDS = 5.0
    _CLEANUP_TIMEOUT_SECONDS = 5.0

    def __init__(self, *, hooks: SubmissionHooks, directory: Path, limits: SubmissionLimitsV2 | None = None) -> None:
        """
        Initialize a version-isolated runner without constructing a provider or environment.

        Raises:
            TypeError: If the supported synchronous SQLite publication backend is unavailable.
        """
        self._hooks = hooks
        self._directory = directory
        self._limits = limits or SubmissionLimitsV2()
        self._memory = CentralMemory.get_memory_instance()
        if type(self._memory) is not SQLiteMemory:
            raise TypeError("V2 publication supports only synchronous SQLiteMemory.")
        self._normalizer = PromptNormalizer()
        self._run_id = str(uuid4())
        self._report: StrictSubmissionReportV2 | None = None
        self._provenance: SubmissionProvenanceV2 | None = None
        self._writer: SubmissionEvidenceWriterV2 | None = None
        self._calls: list[SubmissionCallEvidence] = []
        self._seen_call_ids: set[str] = set()
        self._generations = 0
        self._requests = 0
        self._messages = 0
        self._total_tokens: int | None = 0
        self._token_usage: list[dict[str, int] | None] = []
        self._termination_limit: str | None = None
        self._request_generation: int | None = None
        self._response_generation: int | None = None
        self._generation_active = False
        self._started = False
        self._owner_task: asyncio.Task[Any] | None = None
        self._initial_cancellations = 0
        self._first_cancellation: asyncio.CancelledError | None = None
        self._publication_cancelled = False
        self._finalizing = False
        self._published = False
        self._failure: BaseException | None = None
        self._retention_failures: list[dict[str, str]] = []
        self._last_projection_json: str | None = None
        self._pending_tool: Message | None = None
        self._final_text: str | None = None
        self._termination_reason: SubmissionTerminationReason | None = None
        self._environment_audit: JsonValue = None
        self._target_identifier: dict[str, JsonValue] | None = None
        self._local_cleanup = SubmissionCleanupStatus.UNKNOWN
        self._lifecycle_errors: list[str] = []
        self._target_context: AbstractAsyncContextManager[OpenAIResponseTarget] | None = None
        self._target_entered = False
        self.last_result: SubmissionRunResultV2 | None = None

    async def run_with_target_factory_async(
        self,
        *,
        seed: SeedPrompt,
        system_prompt: str,
        target_factory: SubmissionTargetFactoryV2,
        environment: SubmissionEnvironmentHooksV2,
    ) -> SubmissionRunResultV2:
        """
        Execute a fresh validated v2 binding using only caller-supplied resources.

        The factory owns endpoint, model, credentials, HTTPX hooks, retry configuration
        and context exit. Audit and cleanup are mandatory caller hooks. Their code is
        trusted, not a security sandbox. No mode is inferred from a caller enable flag.

        Returns:
            SubmissionRunResultV2: The durable content-anchored outcome.

        Raises:
            ValueError: If the runner is reused or the initial v2 binding is invalid.
            SubmissionRunV2Error: If provider, binding or resource operations fail.
            asyncio.CancelledError: If cancelled; acquired observations remain retained when possible.
        """
        if self._started:
            raise ValueError("A v2 submission runner is single-use.")
        self._capture_report()
        if (
            self._report is None
            or self._report.submissions
            or self._report.status is not SubmissionReportStatus.NO_SUBMISSION
        ):
            raise ValueError("A v2 run requires fresh no-submission binding state.")
        self._provenance = self._report.provenance()
        self._owner_task = asyncio.current_task()
        self._initial_cancellations = self._owner_task.cancelling() if self._owner_task else 0
        self._started = True
        self._writer = SubmissionEvidenceWriterV2(directory=self._directory / self._run_id, provenance=self._provenance)
        await self._writer.initialize_async()
        try:
            async with asyncio.timeout(self._limits.episode_timeout_seconds):
                await self._retain_projection_async()
                await self._retain_async(
                    self._audit_environment_async(environment), timeout=self._CLEANUP_TIMEOUT_SECONDS
                )
                self._target_context = target_factory(
                    tools=self._tool_definitions(), request_hook=self._request_async, response_hook=self._response_async
                )
                target = await self._target_context.__aenter__()
                self._target_entered = True
                self._validate_target(target)
                await self._run_target_async(target=target, seed=seed, system_prompt=system_prompt)
        except (Exception, asyncio.CancelledError) as error:
            self._record_failure(error)
        self._finalizing = True
        await self._release_resources_async(environment)
        try:
            result = await self._retain_async(self._finish_async())
        except (Exception, asyncio.CancelledError) as retention_error:
            if isinstance(self._failure, asyncio.CancelledError) and retention_error is not self._failure:
                raise self._failure from retention_error
            raise
        if isinstance(self._failure, asyncio.CancelledError):
            raise self._failure
        if self._failure is not None:
            raise SubmissionRunV2Error(result=result) from self._failure
        return result

    def _tool_definitions(self) -> list[dict[str, Any]]:
        return [tool.response_definition() for tool in self._hooks.tools]

    def _validate_target(self, target: OpenAIResponseTarget) -> None:
        if not isinstance(target, OpenAIResponseTarget) or target.auto_execute_tools:
            raise ValueError("The v2 factory must yield OpenAIResponseTarget(auto_execute_tools=False).")
        if not target.supports_conversation_continuation:
            raise ValueError("The target must support authentic no-input continuation.")
        if CentralMemory.get_memory_instance() is not self._memory:
            raise RuntimeError("CentralMemory changed while constructing the caller target.")
        self._target_identifier = target.get_identifier().model_dump(mode="json")

    def _record_failure(self, error: BaseException) -> None:
        if isinstance(error, asyncio.CancelledError):
            self._first_cancellation = self._first_cancellation or error
            self._failure = self._first_cancellation
        elif self._failure is None:
            self._failure = error
            if isinstance(error, TimeoutError):
                self._first_cancellation = None

    async def _run_target_async(self, *, target: OpenAIResponseTarget, seed: SeedPrompt, system_prompt: str) -> None:
        await self._memory.add_seeds_to_memory_async(seeds=[seed], added_by="submission_v2")
        await asyncio.to_thread(target.set_system_prompt, system_prompt=system_prompt, conversation_id=self._run_id)
        assert self._provenance is not None
        message: Message | None = Message.from_prompt(
            prompt=seed.value, role="user", prompt_metadata={"evidence_label": self._provenance.evidence_label}
        )
        while True:
            self._termination_limit = await self._budget_boundary_async(message)
            if self._termination_limit is not None:
                self._termination_reason = SubmissionTerminationReason.BUDGET
                if self._generations == 0 and message is not None:
                    message.get_piece().conversation_id = self._run_id
                    await set_message_piece_sha256_async(message.get_piece())
                    await asyncio.to_thread(self._memory.add_message_to_memory, request=message)
                return
            self._generations += 1
            self._generation_active = True
            try:
                if message is None:
                    response = await self._normalizer.continue_conversation_async(
                        target=target, conversation_id=self._run_id
                    )
                else:
                    response = await self._normalizer.send_prompt_async(
                        message=message, target=target, conversation_id=self._run_id
                    )
                if self._request_generation != self._generations or self._response_generation != self._generations:
                    raise RuntimeError("The caller factory bypassed the required provider observation hooks.")
            finally:
                self._generation_active = False
                if len(self._token_usage) < self._requests:
                    self._token_usage.extend([None] * (self._requests - len(self._token_usage)))
                    self._total_tokens = None
            self._pending_tool = None
            if response.api_role != "assistant" or any(
                piece.has_error() or piece.is_truncated for piece in response.message_pieces
            ):
                raise RuntimeError("Provider completion was missing, truncated or erroneous.")
            calls = response.get_pieces_by_type(data_type="function_call")
            self._final_text = (
                "\n".join(piece.converted_value for piece in response.get_pieces_by_type(data_type="text")) or None
            )
            if not calls:
                message = None
                continue
            if len(calls) != 1:
                raise RuntimeError("V2 submissions require sequential provider tool calls.")
            message = await self._invoke_tool_async(calls[0])
            if self._report is None:
                raise RuntimeError("A tool completed without a validated v2 report.")
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

    async def _budget_boundary_async(self, message: Message | None) -> str | None:
        messages = await asyncio.to_thread(self._memory.get_conversation_messages, conversation_id=self._run_id)
        stored_ids = {piece.id for item in messages for piece in item.message_pieces}
        self._messages = len(messages) + int(message is not None and message.get_piece().id not in stored_ids)
        if self._generations >= self._limits.max_requests:
            return "max_requests"
        if len(self._calls) >= self._limits.max_tool_calls:
            return "max_tool_calls"
        if self._limits.max_messages is not None and self._messages >= self._limits.max_messages:
            return "max_messages"
        if (
            self._limits.max_tokens is not None
            and self._total_tokens is not None
            and self._total_tokens >= self._limits.max_tokens
        ):
            return "max_tokens"
        return None

    async def _invoke_tool_async(self, piece: MessagePiece) -> Message:
        call = json.loads(piece.converted_value)
        if not isinstance(call, dict) or any(
            not isinstance(call.get(key), str) or not call[key] for key in ("call_id", "name", "arguments")
        ):
            raise ValueError("A v2 tool call requires an actual provider identity and JSON arguments.")
        call_id, name, arguments_json = call["call_id"], call["name"], call["arguments"]
        if call_id in self._seen_call_ids or len(self._seen_call_ids) >= self._limits.max_tool_calls:
            raise RuntimeError("A tool call identity was reused or its invocation budget was exhausted.")
        self._seen_call_ids.add(call_id)
        arguments = json.loads(arguments_json)
        if not isinstance(arguments, dict):
            raise ValueError("Tool arguments must be a JSON object.")
        tool = next((item for item in self._hooks.tools if item.name == name), None)
        if tool is None:
            raise ValueError(f"Unregistered caller tool: {name}")
        previous_ids = {item.submission_id for item in self._report.submissions} if self._report else set()
        assert self._writer is not None
        await self._writer.append_async(
            event="tool_requested",
            data={"call_id": call_id, "tool_name": name, "arguments": arguments, "local_invocation_id": str(uuid4())},
        )
        feedback: str | None = None
        kind = SubmissionFeedbackKind.RETURNED
        try:
            feedback = await tool.callback_async(**arguments)
            if not isinstance(feedback, str):
                feedback = None
                raise TypeError("Caller tools must return exact strings.")
        except self._hooks.recoverable_errors as error:
            kind = SubmissionFeedbackKind.RECOVERABLE_ERROR
            try:
                feedback = self._hooks.error_feedback(error)
                if not isinstance(feedback, str):
                    feedback = None
                    raise TypeError("Caller error-feedback mapping must return a string.")
            except Exception:
                kind = SubmissionFeedbackKind.TERMINAL_ERROR
                raise
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
            raise RuntimeError("The caller did not return an actual tool result.")
        return self._pending_tool

    def _capture_tool_message(self, evidence: SubmissionCallEvidence) -> None:
        if evidence.feedback is None:
            return
        assert self._provenance is not None
        self._pending_tool = MessagePiece(
            role="tool",
            conversation_id=self._run_id,
            original_value_data_type="function_call_output",
            original_value=json.dumps(
                {"type": "function_call_output", "call_id": evidence.call_id, "output": evidence.feedback},
                separators=(",", ":"),
            ),
            prompt_metadata={
                "evidence_label": self._provenance.evidence_label,
                "feedback_kind": evidence.feedback_kind.value,
                "contract_version": "strict-submission-v2",
            },
        ).to_message()

    def _capture_report(self) -> None:
        self._last_projection_json = None
        raw = json.dumps(self._hooks.read_report(), ensure_ascii=True, allow_nan=False)
        self._last_projection_json = raw
        report = StrictSubmissionReportV2.model_validate_json(raw)
        if self._provenance is not None and report.provenance() != self._provenance:
            raise ValueError("Binding provenance cannot change during a v2 run.")
        if self._report is not None and report.submissions[: len(self._report.submissions)] != self._report.submissions:
            raise ValueError("A binding cannot rewrite previously retained v2 submission observations.")
        self._report = report

    async def _retain_projection_async(self) -> None:
        assert self._writer is not None
        if self._last_projection_json is not None:
            await self._writer.append_async(event="binding_projection", data={"json": self._last_projection_json})
        if self._report is not None:
            await self._writer.snapshot_async(self._report)

    async def _retain_call_async(self, *, evidence: SubmissionCallEvidence, snapshot_error: Exception | None) -> None:
        await self._retain_projection_async()
        assert self._writer is not None
        if snapshot_error is not None:
            await self._writer.append_async(
                event="report_acquisition_failed",
                data={"error_type": type(snapshot_error).__name__, "message": str(snapshot_error)},
            )
        await self._writer.append_async(event="tool_finished", data=evidence.model_dump(mode="json"))

    async def _release_resources_async(self, environment: SubmissionEnvironmentHooksV2) -> None:
        if self._target_entered:
            try:
                await self._retain_async(self._close_target_async(), timeout=self._CLEANUP_TIMEOUT_SECONDS)
            except (Exception, asyncio.CancelledError) as error:
                self._lifecycle_errors.append(f"Target context exit: {type(error).__name__}: {error}")
                self._record_failure(error)
        try:
            await self._retain_async(
                self._cleanup_environment_async(environment), timeout=self._CLEANUP_TIMEOUT_SECONDS
            )
        except (Exception, asyncio.CancelledError) as error:
            self._lifecycle_errors.append(f"Environment cleanup: {type(error).__name__}: {error}")
            self._record_failure(error)

    async def _close_target_async(self) -> None:
        assert self._target_context is not None
        failure = self._failure
        await self._target_context.__aexit__(
            type(failure) if failure is not None else None, failure, failure.__traceback__ if failure else None
        )
        assert self._writer is not None
        await self._writer.append_async(event="target_context_exited", data={})

    async def _audit_environment_async(self, environment: SubmissionEnvironmentHooksV2) -> None:
        audit = await environment.audit_async()
        validated_audit = TypeAdapter(JsonValue).validate_python(audit)
        json.dumps(validated_audit, allow_nan=False)
        self._environment_audit = validated_audit
        assert self._writer is not None
        await self._writer.append_async(event="environment_audit", data={"audit": self._environment_audit})

    async def _cleanup_environment_async(self, environment: SubmissionEnvironmentHooksV2) -> None:
        status = await environment.cleanup_async()
        if not isinstance(status, SubmissionCleanupStatus):
            raise TypeError("Caller cleanup must return an explicit SubmissionCleanupStatus.")
        self._local_cleanup = status
        assert self._writer is not None
        await self._writer.append_async(event="environment_cleanup", data={"status": status.value})

    async def _finish_async(self) -> SubmissionRunResultV2:
        assert self._writer is not None
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
        self._messages = len(
            await asyncio.to_thread(self._memory.get_conversation_messages, conversation_id=self._run_id)
        )
        while True:
            report = self._build_final_report()
            path = await self._writer.retain_async(report)
            if report != self._build_final_report():
                continue
            if CentralMemory.get_memory_instance() is not self._memory:
                raise RuntimeError("CentralMemory changed before v2 report publication.")
            scores = await SubmissionReportScorerV2(report_sha256=report.sha256()).score_async(
                scorable=ContentScorable(value=report.canonical_json(), data_type="text")
            )
            self._writer.mark_published(path)
            self._published = True
            self.last_result = SubmissionRunResultV2(report=report, report_path=path, score=scores[0])
            return self.last_result

    def _build_final_report(self) -> RetainedSubmissionReportV2:
        if self._owner_task and self._owner_task.cancelling() > self._initial_cancellations:
            self._publication_cancelled = True
        assert self._provenance is not None
        status = self._report.status if self._report else SubmissionReportStatus.ERROR
        runner_error = f"{type(self._failure).__name__}: {self._failure}" if self._failure else None
        if self._failure is not None or self._publication_cancelled:
            self._termination_reason = SubmissionTerminationReason.RUNNER_ERROR
            if isinstance(self._failure, asyncio.CancelledError) or self._publication_cancelled:
                status = SubmissionReportStatus.CANCELLED
                if runner_error is None:
                    runner_error = (
                        f"CancelledError: {self._first_cancellation}"
                        if self._first_cancellation
                        else "Cancellation requested before v2 report publication."
                    )
            elif status in {SubmissionReportStatus.COMPLETED, SubmissionReportStatus.NO_SUBMISSION}:
                status = SubmissionReportStatus.INCOMPLETE
        elif self._local_cleanup not in {SubmissionCleanupStatus.COMPLETE, SubmissionCleanupStatus.NOT_REQUIRED}:
            status = SubmissionReportStatus.INCOMPLETE
        return RetainedSubmissionReportV2(
            mode=self._provenance.mode,
            simulated=self._provenance.simulated,
            evidence_label=self._provenance.evidence_label,
            run_id=self._run_id,
            conversation_id=self._run_id,
            status=status,
            report=self._report,
            calls=tuple(self._calls),
            final_text=self._final_text,
            runner_error=runner_error,
            termination_reason=self._termination_reason,
            generation_count=self._generations,
            provider_request_count=self._requests,
            message_count=self._messages,
            total_tokens=self._total_tokens,
            token_usage=tuple(self._token_usage),
            termination_limit=self._termination_limit,
            limits={key: value for key, value in asdict(self._limits).items() if value is not None},
            target_identifier=self._target_identifier,
            environment_audit=self._environment_audit,
            local_cleanup=self._local_cleanup,
            lifecycle_errors=tuple(self._lifecycle_errors),
        )

    async def _retain_async(
        self, operation: Coroutine[Any, Any, _ResultT], *, timeout: float | None = None
    ) -> _ResultT:
        task = asyncio.create_task(operation)
        deadline = asyncio.get_running_loop().time() + (timeout or self._RETENTION_TIMEOUT_SECONDS)
        cancellation: asyncio.CancelledError | None = None
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                task.cancel()
                task.add_done_callback(self._retention_done)
                error = TimeoutError("V2 retention or lifecycle operation exceeded its bounded deadline.")
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
            logger.error("V2 retention or lifecycle operation failed: %s", error)
            if cancellation is not None:
                raise cancellation from error
            raise
        if cancellation is not None:
            raise cancellation
        return result

    @staticmethod
    def _retention_done(task: asyncio.Task[Any]) -> None:
        if not task.cancelled() and task.exception() is not None:
            logger.error("V2 retention failed after its deadline: %s", task.exception())

    async def _request_async(self, request: httpx.Request) -> None:
        if not self._generation_active or self._request_generation == self._generations:
            raise RuntimeError(
                "V2 permits exactly one observed provider request per explicit generation, without retries."
            )
        if self._requests >= self._limits.max_requests:
            raise RuntimeError("The v2 provider-request budget was reached.")
        body = json.loads(await request.aread())
        if not isinstance(body, dict):
            raise ValueError("Observed provider request must be a JSON object.")
        if body.get("tools") != self._tool_definitions() or body.get("parallel_tool_calls") is not False:
            raise ValueError("The caller factory must send the exact sequential injected tool definitions.")
        if self._target_identifier is None or body.get("model") != self._target_identifier.get("model_name"):
            raise ValueError("The observed provider model must match the caller target's public identity.")
        self._request_generation = self._generations
        self._requests += 1
        request.extensions["pyrit_submission_v2_generation"] = self._generations
        assert self._writer is not None
        await self._writer.append_async(
            event="provider_request", data={"sequence": self._requests, "generation": self._generations, "body": body}
        )

    async def _response_async(self, response: httpx.Response) -> None:
        generation = response.request.extensions.get("pyrit_submission_v2_generation")
        if (
            not self._generation_active
            or generation != self._generations
            or self._response_generation == self._generations
        ):
            raise RuntimeError("V2 provider response correlation is missing or repeated.")
        payload = await response.aread()
        if len(payload) > self._limits.max_response_bytes:
            raise RuntimeError("Provider response exceeded the configured v2 evidence limit.")
        self._response_generation = self._generations
        assert self._writer is not None
        body = json.loads(payload)
        await self._writer.append_async(
            event="provider_response",
            data={
                "sequence": self._requests,
                "generation": self._generations,
                "http_status": response.status_code,
                "body": body,
            },
        )
        if response.is_success:
            self._capture_usage(body)
        else:
            self._token_usage.append(None)
            self._total_tokens = None

    def _capture_usage(self, body: object) -> None:
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict) or any(
            type(usage.get(key)) is not int or usage[key] < 0 for key in ("input_tokens", "output_tokens")
        ):
            self._token_usage.append(None)
            self._total_tokens = None
            if self._limits.max_tokens is not None:
                raise RuntimeError(
                    "Required actual provider token usage is missing or malformed; no zero estimate is used."
                )
            return
        captured = {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
        captured["total_tokens"] = captured["input_tokens"] + captured["output_tokens"]
        self._token_usage.append(captured)
        if self._total_tokens is not None:
            self._total_tokens += captured["total_tokens"]
