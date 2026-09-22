# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from pyrit.executor.benchmark.submission.hooks import SubmissionLimits

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from contextlib import AbstractAsyncContextManager

    import httpx
    from pydantic import JsonValue

    from pyrit.models.submission import SubmissionCleanupStatus
    from pyrit.prompt_target import OpenAIResponseTarget


class SubmissionTargetFactoryV2(Protocol):
    """Caller-owned target/client/auth lifetime with explicit provider observation hooks."""

    def __call__(
        self,
        *,
        tools: list[dict[str, Any]],
        request_hook: Callable[[httpx.Request], Awaitable[None]],
        response_hook: Callable[[httpx.Response], Awaitable[None]],
    ) -> AbstractAsyncContextManager[OpenAIResponseTarget]:
        """
        Construct a manual-tool Responses target and release it on context exit.

        Attach the hooks to the caller's HTTPX client and disable SDK retries.
        No endpoint, credential, transport, or environment is supplied by PyRIT.
        """
        ...


@dataclass(frozen=True, kw_only=True)
class SubmissionEnvironmentHooksV2:
    """Explicit caller-owned audit and cleanup, without resource or sandbox defaults."""

    audit_async: Callable[[], Awaitable[JsonValue]]
    cleanup_async: Callable[[], Awaitable[SubmissionCleanupStatus]]


@dataclass(frozen=True, kw_only=True)
class SubmissionLimitsV2(SubmissionLimits):
    """Distinct generation, tool, conversation-message and actual-token budgets."""

    max_messages: int | None = None
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        """
        Validate v2-only optional message and token bounds without changing v1.

        Raises:
            ValueError: If a configured bound is nonpositive or not an integer.
        """
        super().__post_init__()
        for value in (
            self.max_requests,
            self.max_tool_calls,
            self.max_response_bytes,
            self.max_messages,
            self.max_tokens,
        ):
            if value is not None and (type(value) is not int or value <= 0):
                raise ValueError("V2 request, tool, response, message and token limits must be positive integers.")
        if not math.isfinite(self.episode_timeout_seconds):
            raise ValueError("The v2 episode deadline must be finite.")
