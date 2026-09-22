# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark._inspect_native_generate import (
    InspectNativeGenerate,
    InspectNativeLimitError,
    _NativeBudgetStopError,
)
from pyrit.executor.benchmark.submission.evidence_v2 import SubmissionEvidenceWriterV2

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    import httpx
    from inspect_ai.solver import TaskState

    from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionLimitsV2
    from pyrit.models import Message
    from pyrit.models.submission_v2 import SubmissionProvenanceV2
    from pyrit.prompt_target import OpenAIResponseTarget


class InspectRunArtifactsV2(InspectRunArtifacts):
    """Use the shared v2 writer with inherited atomic manifest persistence."""

    def __init__(self, *, directory: Path, provenance: SubmissionProvenanceV2) -> None:
        """Prepare identities and truthful provenance without creating a directory."""
        self.directory = directory.resolve()
        self.run_id = str(uuid4())
        self.attempt_id = str(uuid4())
        self.writer = SubmissionEvidenceWriterV2(directory=self.directory, provenance=provenance)
        self.manifest: dict[str, Any] = {
            "schema_version": 2,
            "contract_version": "strict-submission-v2",
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "created_at": datetime.now(UTC).isoformat(),
            "mode": provenance.mode.value,
            "simulated": provenance.simulated,
            "evidence_label": provenance.evidence_label,
            "harness_status": "not_started",
            "grade_status": "not_published",
            "publication_state": "not_prepared",
            "remote_stop_claim": False,
        }
        self._lock = asyncio.Lock()

    async def append_async(self, *, event: str, data: dict[str, Any]) -> None:
        """Retain events with the writer's immutable provenance and run identity."""
        await self.writer.append_async(
            event=event, data={"run_id": self.run_id, "attempt_id": self.attempt_id, "data": data}
        )


class InspectProviderObserverV2:
    """Enforce one observed HTTP request per explicit generation, without accessing an SDK client."""

    def __init__(self, *, artifacts: InspectRunArtifactsV2, limits: SubmissionLimitsV2) -> None:
        """Bind bounded, sequential public request and response hooks."""
        self.artifacts = artifacts
        self.limits = limits
        self.request_count = 0
        self.response_count = 0
        self.token_usage: list[dict[str, int] | None] = []
        self.schemas: list[dict[str, Any]] = []
        self.model_name = ""
        self._generation: int | None = None
        self._request_seen = False
        self._response_seen = False

    @property
    def total_tokens(self) -> int | None:
        """The sum of actually reported input and output tokens, or unknown."""
        if not self.token_usage or any(usage is None for usage in self.token_usage):
            return None
        return sum(usage["input_tokens"] + usage["output_tokens"] for usage in self.token_usage if usage is not None)

    def begin(self, generation: int) -> None:
        """
        Admit a single generation boundary.

        Raises:
            RuntimeError: If another generation is already active.
        """
        if self._generation is not None:
            raise RuntimeError("V2 provider generations must not overlap.")
        self._generation = generation
        self._request_seen = self._response_seen = False

    def finish(self, *, successful: bool) -> None:
        """
        Require attached hooks for successful generation, then close its admission.

        Raises:
            RuntimeError: If a successful provider call bypassed either observation hook.
        """
        observed = self._request_seen and self._response_seen
        self._generation = None
        if successful and not observed:
            raise RuntimeError("The caller target factory did not attach both required provider hooks.")

    async def request_async(self, request: httpx.Request) -> None:
        """
        Validate an actual request before transport without retaining headers or credentials.

        Raises:
            RuntimeError: If a retry, extra generation, or request budget would be violated.
            ValueError: If the target changed the agreed model/tool protocol.
        """
        if self._generation is None or self._request_seen or self.request_count >= self.limits.max_requests:
            raise RuntimeError("V2 forbids uncorrelated requests, SDK retries, and exhausted provider budgets.")
        body = json.loads(await request.aread())
        if (
            not isinstance(body, dict)
            or body.get("model") != self.model_name
            or body.get("tools") != self.schemas
            or body.get("parallel_tool_calls") is not False
            or body.get("store") is not False
            or body.get("stream") is not False
        ):
            raise ValueError(
                "The factory target changed the native schemas, model, or non-storing sequential protocol."
            )
        self._request_seen = True
        self.request_count += 1
        self.token_usage.append(None)
        await self.artifacts.append_async(
            event="provider_request",
            data={"generation": self._generation, "request_ordinal": self.request_count, "body": body},
        )

    async def response_async(self, response: httpx.Response) -> None:
        """
        Retain bounded actual response evidence and only explicitly reported usage.

        Raises:
            RuntimeError: If the response is uncorrelated or repeats within a generation.
            ValueError: If response evidence exceeds the bound or is not a JSON object.
        """
        if self._generation is None or not self._request_seen or self._response_seen:
            raise RuntimeError("The v2 provider response has no unique admitted request.")
        raw = await response.aread()
        self._response_seen = True
        self.response_count += 1
        if len(raw) > self.limits.max_response_bytes:
            await self.artifacts.append_async(
                event="provider_response_limit",
                data={"status_code": response.status_code, "bytes": len(raw), "retained_body": False},
            )
            raise ValueError("Provider response evidence exceeds the configured byte bound.")
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("The v2 provider response must be a JSON object.")
        usage = body.get("usage")
        known = (
            isinstance(usage, dict)
            and type(usage.get("input_tokens")) is int
            and type(usage.get("output_tokens")) is int
            and usage["input_tokens"] >= 0
            and usage["output_tokens"] >= 0
        )
        await self.artifacts.append_async(
            event="provider_response",
            data={"generation": self._generation, "status_code": response.status_code, "body": body},
        )
        self.token_usage[self.request_count - 1] = (
            {"input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]} if known else None
        )


class InspectNativeGenerateV2(InspectNativeGenerate):
    """Reuse native Generate semantics while acquiring a caller-owned target asynchronously."""

    def __init__(
        self,
        *,
        acquire_target_async: Callable[[list[dict[str, Any]]], Awaitable[OpenAIResponseTarget]],
        capture_report: Callable[[], None],
        after_tool_async: Callable[[], Awaitable[bool]],
        artifacts: InspectRunArtifactsV2,
        limits: SubmissionLimitsV2,
        observer: InspectProviderObserverV2,
        max_tool_output_bytes: int,
    ) -> None:
        """Keep the reviewed v1 tool execution/continuation implementation, with separate v2 bindings."""
        self._acquire_target_async = acquire_target_async
        self._provided_target: OpenAIResponseTarget | None = None
        self._capture_report_v2 = capture_report
        self._limits_v2 = limits
        self.observer = observer
        self.message_count = 0
        self.termination_limit: str | None = None
        super().__init__(
            target_factory=self._provided_factory,
            model_name="",
            artifacts=artifacts,
            after_tool_async=after_tool_async,
            on_tool_return=capture_report,
            max_requests=limits.max_requests,
            max_tool_calls=limits.max_tool_calls,
            max_tool_output_bytes=max_tool_output_bytes,
            max_response_bytes=limits.max_response_bytes,
        )

    async def generate_async(
        self, state: TaskState, tool_calls: Literal["loop", "single", "none"] = "loop", **kwargs: Any
    ) -> TaskState:
        """
        Bind the actual caller target after native setup, preserving the original solver's input.

        Returns:
            TaskState: The original state containing actual responses and tool feedback.
        """
        from inspect_ai.tool import ToolDef

        try:
            self._check_dispatch(state=state, provider=True)
        except _NativeBudgetStopError as budget:
            return await self._finish_budget_async(state, reason=budget.reason)
        if self._provided_target is None:
            schemas = [
                {
                    "type": "function",
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.parameters.model_dump(exclude_none=True),
                }
                for definition in (ToolDef(tool) for tool in state.tools)
            ]
            self._provided_target = await self._acquire_target_async(schemas)
            self.model_name = self.observer.model_name
        try:
            return await super().generate_async(state, tool_calls=tool_calls, **kwargs)
        finally:
            self.message_count = len(state.messages)

    def _provided_factory(self, schemas: list[dict[str, Any]]) -> OpenAIResponseTarget:
        if self._provided_target is None or schemas != self.observer.schemas:
            raise ValueError("Native tools do not match the caller-owned v2 target binding.")
        return self._provided_target

    def _check_dispatch(self, *, state: TaskState, provider: bool, reserved_tool: bool = False) -> None:
        self._capture_report_v2()
        self.message_count = len(state.messages)
        try:
            super()._check_dispatch(state=state, provider=provider, reserved_tool=reserved_tool)
        except _NativeBudgetStopError:
            self.termination_limit = "native_or_dispatch_budget"
            raise
        if self._limits_v2.max_messages is not None and self.message_count >= self._limits_v2.max_messages:
            self.termination_limit = "messages"
            raise _NativeBudgetStopError
        if self._limits_v2.max_tokens is not None and self.requests:
            if self.observer.total_tokens is None:
                raise InspectNativeLimitError("V2 token budgeting requires actual reported input and output usage.")
            if provider and self.observer.total_tokens >= self._limits_v2.max_tokens:
                self.termination_limit = "tokens"
                raise _NativeBudgetStopError

    async def _send_one_async(self, state: TaskState) -> Message:
        self.observer.begin(self.requests + 1)
        successful = False
        try:
            response = await super()._send_one_async(state)
            successful = True
            return response
        finally:
            self.observer.finish(successful=successful)
