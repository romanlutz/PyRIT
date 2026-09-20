# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts


class InspectResponseTrace:
    """Capture the Responses API boundary using public HTTPX hooks, not target internals."""

    def __init__(self, *, artifacts: InspectRunArtifacts, max_requests: int) -> None:
        """
        Bind an evidence journal and a finite request ceiling.

        Raises:
            ValueError: If the request ceiling is not positive.
        """
        if max_requests < 1:
            raise ValueError("max_requests must be positive.")
        self.artifacts = artifacts
        self.max_requests = max_requests
        self.requests_observed = 0
        self.responses_observed = 0
        self.usage: dict[str, int] = {}
        self._pending: dict[str, Any] | None = None
        self._call_ids: set[str] = set()
        self._terminal_error: str | None = None

    async def request_async(self, request: httpx.Request) -> None:
        """
        Record a request body and enforce the sequential, non-storing harness contract.

        Raises:
            RuntimeError: If the request budget is exhausted.
            ValueError: If the request violates the tool or storage contract.
        """
        if self._terminal_error is not None or self._pending is not None:
            raise RuntimeError("The preceding tool call did not complete; this episode cannot continue.")
        if self.requests_observed >= self.max_requests:
            raise RuntimeError("Provider request budget exhausted; the episode must not be retried.")
        body = json.loads(await request.aread())
        if not isinstance(body, dict) or body.get("parallel_tool_calls") is not False or body.get("store") is not False:
            raise ValueError("The Inspect bridge requires parallel_tool_calls=false and store=false.")
        self.requests_observed += 1
        await self.artifacts.append_async(
            event="provider_request",
            data={"ordinal": self.requests_observed, "body": body},
        )

    async def response_async(self, response: httpx.Response) -> None:
        """
        Retain each response before checking executable calls or accepting its evidence.

        Raises:
            ValueError: If the response cannot be safely correlated with one execution.
        """
        raw = await response.aread()
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("Expected a JSON object from the Responses API.")
        self.responses_observed += 1
        await self.artifacts.append_async(
            event="provider_response",
            data={"ordinal": self.responses_observed, "status_code": response.status_code, "body": body},
        )
        usage = body.get("usage")
        if isinstance(usage, dict):
            for name in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    self.usage[name] = self.usage.get(name, 0) + value
        calls = [item for item in body.get("output", []) if item.get("type") == "function_call"]
        if len(calls) > 1:
            raise ValueError("Multiple executable calls in one response are not supported; none were executed.")
        if self._pending is not None:
            raise ValueError("An earlier provider call has not been correlated with an execution.")
        if calls:
            self._pending = calls[0]

    def claim_call(self, *, name: str, arguments: dict[str, Any]) -> str:
        """
        Correlate a real model-requested call with exactly one tool execution.

        Returns:
            str: The provider's call identity, distinct from the adapter execution identity.

        Raises:
            ValueError: If the call is missing, mismatched, or already executed.
        """
        call = self._pending
        if call is None or call.get("name") != name or json.loads(call["arguments"]) != arguments:
            raise ValueError("Tool execution does not match the pending real provider request.")
        call_id = call.get("call_id")
        if not isinstance(call_id, str) or not call_id or call_id in self._call_ids:
            raise ValueError("Provider call identity is missing or has already been executed.")
        self._pending = None
        self._call_ids.add(call_id)
        return call_id

    def summary(self) -> dict[str, Any]:
        """Return observed boundary counts, without claiming Inspect metered this external model."""
        return {
            "requests_observed": self.requests_observed,
            "responses_observed": self.responses_observed,
            "usage": dict(self.usage),
            "provider_call_ids": sorted(self._call_ids),
        }

    def abort(self, reason: str) -> None:
        """Prevent subsequent provider requests after a terminal operational error."""
        self._terminal_error = reason
