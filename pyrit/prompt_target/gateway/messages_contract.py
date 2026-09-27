# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host-owned Anthropic Messages wire contract, separate from OpenAI Responses."""

import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

from pyrit.prompt_target.gateway.responses_contract import GatewayFrameKind


class MessagesCoverage(str, Enum):
    """Anthropic Messages features observed on the original wire."""

    STREAMING = "streaming"
    TEXT = "text"
    TOOL_DEFINITION = "tool_definition"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    PROMPT_CACHING = "prompt_caching"
    PING = "ping"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class MessagesCapabilities:
    """Explicitly verified features of one pinned Anthropic-format model."""

    streaming: bool = False
    tool_use: bool = False
    thinking: bool = False
    prompt_caching: bool = False
    effort: bool = False
    allowed_beta_values: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        """
        Reject unknown, loosely typed, or OAuth-scoped capabilities.

        Raises:
            ValueError: If a capability was not explicitly configured.
        """
        for name in ("streaming", "tool_use", "thinking", "prompt_caching", "effort"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be an explicitly verified boolean")
        if type(self.allowed_beta_values) is not frozenset or len(self.allowed_beta_values) > 32:
            raise ValueError("allowed_beta_values must be an explicit bounded frozenset")
        for value in self.allowed_beta_values:
            if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,127}", value):
                raise ValueError("allowed_beta_values contain an invalid beta token")
            if "oauth" in value:
                raise ValueError("OAuth capability headers cannot be forwarded with a host API key")


@dataclass(frozen=True)
class MessagesRequest:
    """Validated original Messages body and approved provider headers."""

    run_id: str
    request_id: str
    body_bytes: bytes
    body: dict[str, Any]
    anthropic_version: str
    anthropic_beta: str | None
    query_string: bytes
    max_tokens: int
    streaming: bool
    advertised_tools: frozenset[str]


@dataclass(frozen=True)
class MessagesResponse:
    """Original provider JSON body and safe, selected provider headers."""

    status_code: int
    body: bytes
    headers: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, kw_only=True)
class MessagesStream:
    """Provider-owned SSE frames and the close action for an open response."""

    frames: AsyncGenerator[bytes, None]
    headers: tuple[tuple[str, str], ...]
    _close: Callable[[], Awaitable[None]] = field(repr=False)

    async def close_async(self) -> None:
        """Stop generation and release the provider response even if no frame was read."""
        try:
            await self.frames.aclose()
        finally:
            await self._close()


@dataclass(frozen=True)
class MessagesObservation:
    """Exact body/SSE bytes or a separately labeled host error, with safe headers."""

    run_id: str
    request_id: str
    kind: GatewayFrameKind
    frame: bytes
    coverage: frozenset[MessagesCoverage]
    headers: tuple[tuple[str, str], ...] = ()
    query_string: str = ""
    error_code: str | None = None
    status_code: int | None = None


MessagesObservationCallback = Callable[[MessagesObservation], Awaitable[None]]


class ModelOnlyMessagesBackend(Protocol):
    """Send Anthropic-format Messages to a trusted host-owned model client, never tools."""

    capabilities: MessagesCapabilities

    async def create_message_async(self, *, request: MessagesRequest) -> MessagesResponse:
        """Return original provider JSON bytes and response metadata."""
        ...

    async def open_stream_async(self, *, request: MessagesRequest) -> MessagesStream:
        """Open a provider SSE response without buffering it or executing tools."""
        ...


class MessagesBackendErrorCode(str, Enum):
    """Sanitized transport failures that are not genuine provider error bodies."""

    UPSTREAM_NETWORK_ERROR = "upstream_network_error"
    UPSTREAM_STREAM_ERROR = "upstream_stream_error"
    UPSTREAM_TIMEOUT = "upstream_timeout"


class MessagesBackendError(Exception):
    """A fixed, credential-free explanation of a host transport failure."""

    _MESSAGES = {
        MessagesBackendErrorCode.UPSTREAM_NETWORK_ERROR: "Unable to reach the upstream model",
        MessagesBackendErrorCode.UPSTREAM_STREAM_ERROR: "Upstream Anthropic Messages stream is invalid",
        MessagesBackendErrorCode.UPSTREAM_TIMEOUT: "Upstream model request timed out",
    }

    def __init__(self, *, code: MessagesBackendErrorCode) -> None:
        """
        Accept only known safe error codes.

        Raises:
            ValueError: If the backend supplied an unrecognized failure code.
        """
        if not isinstance(code, MessagesBackendErrorCode):
            raise ValueError("Unsupported Messages backend error code")
        self.code = code
        self.status_code = 504 if code is MessagesBackendErrorCode.UPSTREAM_TIMEOUT else 502
        super().__init__(self._MESSAGES[code])


class MessagesProviderError(Exception):
    """A real, bounded provider HTTP error awaiting safety validation before forwarding."""

    def __init__(self, *, response: MessagesResponse) -> None:
        """Keep provider bytes out of error strings and logs."""
        super().__init__("Upstream provider returned an HTTP error")
        self.response = response
