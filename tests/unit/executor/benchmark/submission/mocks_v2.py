# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import httpx

from pyrit.executor.benchmark.submission.hooks_v2 import SubmissionEnvironmentHooksV2
from pyrit.models.submission import SubmissionCleanupStatus
from pyrit.prompt_target import OpenAIResponseTarget
from tests.unit.executor.benchmark.submission.mocks import InertSubmissionBinding, OfflineProvider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from pydantic import JsonValue


class InertSubmissionBindingV2(InertSubmissionBinding):
    """A public inert fixture projected as v2 offline evidence."""

    def read_report(self) -> dict[str, Any]:
        report = super().read_report()
        report.update(contract_version="strict-submission-v2", mode="offline", simulated=True)
        return report


class MockTargetFactoryV2:
    """A caller factory that only constructs an HTTPX MockTransport-backed target."""

    def __init__(
        self,
        *,
        provider: OfflineProvider,
        auto_execute_tools: bool = False,
        attach_request_hook: bool = True,
        attach_response_hook: bool = True,
        close_error: Exception | None = None,
        sdk_retries: int = 0,
    ) -> None:
        self.provider = provider
        self.auto_execute_tools = auto_execute_tools
        self.attach_request_hook = attach_request_hook
        self.attach_response_hook = attach_response_hook
        self.close_error = close_error
        self.sdk_retries = sdk_retries
        self.entered = 0
        self.exited = 0
        self.client: httpx.AsyncClient | None = None
        self.tools: list[dict[str, Any]] = []

    @asynccontextmanager
    async def __call__(
        self,
        *,
        tools: list[dict[str, Any]],
        request_hook: Callable[[httpx.Request], Awaitable[None]],
        response_hook: Callable[[httpx.Response], Awaitable[None]],
    ) -> AsyncIterator[OpenAIResponseTarget]:
        self.entered += 1
        self.tools = tools
        hooks = {
            "request": [request_hook] if self.attach_request_hook else [],
            "response": [response_hook] if self.attach_response_hook else [],
        }
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(self.provider), trust_env=False, event_hooks=hooks
        ) as client:
            self.client = client
            try:
                yield OpenAIResponseTarget(
                    endpoint="https://offline-fixture.invalid/v1",
                    model_name="offline-fixture",
                    api_key="offline-simulated-not-a-credential",
                    headers="{}",
                    auto_execute_tools=self.auto_execute_tools,
                    max_output_tokens=128,
                    extra_body_parameters={"tools": tools, "parallel_tool_calls": False, "store": False},
                    httpx_client_kwargs={"http_client": client, "max_retries": self.sdk_retries},
                )
            finally:
                self.exited += 1
                if self.close_error is not None:
                    raise self.close_error


class InertEnvironmentV2:
    """Record explicit no-resource audit and cleanup callbacks."""

    def __init__(
        self,
        *,
        cleanup: SubmissionCleanupStatus = SubmissionCleanupStatus.NOT_REQUIRED,
        audit_error: Exception | None = None,
        cleanup_error: Exception | None = None,
    ) -> None:
        self.cleanup = cleanup
        self.audit_error = audit_error
        self.cleanup_error = cleanup_error
        self.audit_calls = 0
        self.cleanup_calls = 0

    async def audit_async(self) -> JsonValue:
        self.audit_calls += 1
        if self.audit_error:
            raise self.audit_error
        return {"fixture": "OFFLINE/SIMULATED", "actual_resource_operations": 0}

    async def cleanup_async(self) -> SubmissionCleanupStatus:
        self.cleanup_calls += 1
        if self.cleanup_error:
            raise self.cleanup_error
        return self.cleanup

    def hooks(self) -> SubmissionEnvironmentHooksV2:
        return SubmissionEnvironmentHooksV2(audit_async=self.audit_async, cleanup_async=self.cleanup_async)


class UsageProviderV2(OfflineProvider):
    def __init__(self, *, turns: list[dict[str, Any]], usages: list[dict[str, Any] | None]) -> None:
        super().__init__(turns)
        self.usages = usages

    def __call__(self, request: httpx.Request) -> httpx.Response:
        response = super().__call__(request)
        body = json.loads(response.content)
        body["usage"] = self.usages[len(self.requests) - 1]
        return httpx.Response(200, json=body)
