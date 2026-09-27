# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The host-owned contract between a Responses gateway and its model backend."""

import math
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class GatewayFrameKind(str, Enum):
    """The wire boundary at which a frame was observed."""

    REQUEST = "request"
    RESPONSE = "response"
    RESPONSE_EVENT = "response_event"
    GATEWAY_ERROR = "gateway_error"


class GatewayCoverage(str, Enum):
    """Features seen on the original wire, not inferred from model text."""

    STREAMING = "streaming"
    FUNCTION_TOOL = "function_tool"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESULT = "function_result"
    CUSTOM_TOOL = "custom_tool"
    CUSTOM_CALL = "custom_call"
    CUSTOM_RESULT = "custom_result"
    REASONING = "reasoning"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


@dataclass(frozen=True)
class GatewayRoute:
    """The one run and model a gateway instance is allowed to serve."""

    run_id: str
    model: str
    guest_token: str = field(repr=False)

    def __post_init__(self) -> None:
        """
        Reject ambiguous routes and weak or malformed guest-only credentials.

        Raises:
            ValueError: If the route or its run-scoped token is invalid.
        """
        if not isinstance(self.run_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_id):
            raise ValueError("run_id must be a nonempty, URL-safe run identifier")
        if not isinstance(self.model, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,127}", self.model):
            raise ValueError("model must be a nonempty model alias, not a URL")
        if not isinstance(self.guest_token, str) or not re.fullmatch(r"[A-Za-z0-9_-]{32,256}", self.guest_token):
            raise ValueError("guest_token must be a random, run-scoped, URL-safe token of 32-256 characters")


@dataclass(frozen=True)
class GatewayLimits:
    """Hard per-run request/token limits and per-request time/byte limits."""

    max_requests: int = 32
    max_request_bytes: int = 524_288
    max_total_request_bytes: int = 8_388_608
    max_response_bytes: int = 8_388_608
    max_output_tokens_per_request: int = 8_192
    max_total_output_tokens: int = 131_072
    timeout_seconds: float = 120.0

    def __post_init__(self) -> None:
        """
        Reject unlimited, negative, or non-finite budgets.

        Raises:
            ValueError: If any budget is not finite and positive.
        """
        for name in (
            "max_requests",
            "max_request_bytes",
            "max_total_request_bytes",
            "max_response_bytes",
            "max_output_tokens_per_request",
            "max_total_output_tokens",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.timeout_seconds, bool) or not isinstance(self.timeout_seconds, (int, float)):
            raise ValueError("timeout_seconds must be a positive finite number")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")


@dataclass(frozen=True)
class BackendCapabilities:
    """Responses features the injected backend actually supports."""

    streaming: bool = False
    function_tools: bool = False
    custom_tools: bool = False
    reasoning: bool = False


@dataclass(frozen=True)
class ModelRequest:
    """Validated Responses body, with no guest headers, URL, or credentials."""

    run_id: str
    request_id: str
    body: dict[str, Any]
    output_token_limit: int


@dataclass(frozen=True)
class GatewayObservation:
    """Original wire frame or distinct host-generated error boundary and coverage."""

    run_id: str
    request_id: str
    kind: GatewayFrameKind
    frame: bytes
    coverage: frozenset[GatewayCoverage]
    error_code: str | None = None
    status_code: int | None = None


ObservationCallback = Callable[[GatewayObservation], Awaitable[None]]


class ModelOnlyResponsesBackend(Protocol):
    """
    Supply Responses wire bytes from a trusted, host-configured MODEL-ONLY client.

    ``create_response_async`` returns one raw JSON response body;
    ``stream_response_async`` yields complete SSE frames, in provider order,
    including the final ``data: [DONE]`` frame. The backend must honor
    ``max_output_tokens``, pin its upstream endpoint and authentication on the
    host, and NEVER execute tools or use guest-supplied URLs/headers. In
    particular, ``OpenAIResponseTarget`` is not a backend for this protocol:
    it forces ``stream=False`` and can run a host-side function callback loop.
    """

    capabilities: BackendCapabilities

    async def create_response_async(self, *, request: ModelRequest) -> bytes:
        """Return the original non-streaming Responses JSON bytes."""
        ...

    def stream_response_async(self, *, request: ModelRequest) -> AsyncGenerator[bytes, None]:
        """Return an async generator of original, complete Responses SSE frames."""
        ...


class ModelBackendErrorCode(str, Enum):
    """Sanitized upstream failures the gateway may expose to a guest."""

    UPSTREAM_HTTP_ERROR = "upstream_http_error"
    UPSTREAM_NETWORK_ERROR = "upstream_network_error"
    UPSTREAM_STREAM_ERROR = "upstream_stream_error"
    UPSTREAM_TIMEOUT = "upstream_timeout"


class ModelBackendError(Exception):
    """A typed backend failure that cannot include a URL, token, or provider body."""

    _MESSAGES = {
        ModelBackendErrorCode.UPSTREAM_HTTP_ERROR: "Upstream model returned an HTTP error",
        ModelBackendErrorCode.UPSTREAM_NETWORK_ERROR: "Unable to reach the upstream model",
        ModelBackendErrorCode.UPSTREAM_STREAM_ERROR: "Upstream Responses payload or stream is invalid",
        ModelBackendErrorCode.UPSTREAM_TIMEOUT: "Upstream model request timed out",
    }

    def __init__(self, *, code: ModelBackendErrorCode) -> None:
        """
        Build an error from a fixed, guest-safe code and message.

        Raises:
            ValueError: If the backend supplied an unrecognized code.
        """
        if not isinstance(code, ModelBackendErrorCode):
            raise ValueError("Unsupported model backend error code")
        self.code = code
        self.status_code = 504 if code is ModelBackendErrorCode.UPSTREAM_TIMEOUT else 502
        super().__init__(self._MESSAGES[code])


class GatewayError(Exception):
    """A safe, explicit wire error without provider or guest credentials."""

    def __init__(self, *, status_code: int, code: str, message: str, param: str | None = None) -> None:
        """Store only guest-safe error details for serialization."""
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.param = param
