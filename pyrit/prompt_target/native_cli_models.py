# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inert, provider-independent observations from sandbox-owned coding CLIs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from pydantic import JsonValue


class NativeCliProtocol(str, Enum):
    """Documented JSONL output profiles, not executable names or auth modes."""

    CODEX_EXEC_JSON = "codex_exec_json"
    CLAUDE_PRINT_STREAM_JSON_VERBOSE = "claude_print_stream_json_verbose"


class NativeCliStream(str, Enum):
    """The sandbox process pipe from which a byte chunk was read."""

    STDOUT = "stdout"
    STDERR = "stderr"


class NativeCliEventKind(str, Enum):
    """An observed provider action or an explicit coverage boundary."""

    SESSION_STARTED = "session_started"
    TURN_STARTED = "turn_started"
    TURN_COMPLETED = "turn_completed"
    RUN_FINISHED = "run_finished"
    MODEL_MESSAGE = "model_message"
    TOOL_REQUESTED = "tool_requested"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_RESULT = "tool_result"
    PROGRESS = "progress"
    AUXILIARY = "auxiliary"
    PARTIAL = "partial"
    ERROR = "error"
    EOF = "eof"


class NativeCliEventStatus(str, Enum):
    """Observed lifecycle status, never an inferred objective score."""

    UNKNOWN = "unknown"
    REQUESTED = "requested"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, kw_only=True)
class NativeCliRunConfig:
    """One explicit sandbox run; the launcher owns the CLI, gateway, and isolation."""

    protocol: NativeCliProtocol
    cli_version: str
    cli_profile: str
    agent_workdir: PurePosixPath
    model_gateway_endpoint: str
    max_steps: int
    timeout_seconds: float
    max_frame_bytes: int = 8_388_608

    def __post_init__(self) -> None:
        """
        Reject unpinned or credential-bearing run configuration.

        Raises:
            ValueError: If the profile, path, gateway, or budgets are not explicit and safe.
        """
        if not isinstance(self.protocol, NativeCliProtocol):
            raise ValueError("A supported native CLI JSONL protocol is required.")
        if not isinstance(self.cli_version, str) or not re.fullmatch(
            r"\d+\.\d+\.\d+(?:[-+][A-Za-z0-9.-]+)?", self.cli_version
        ):
            raise ValueError("Pin a concrete native CLI version, not a floating tag.")
        if not isinstance(self.cli_profile, str) or not re.fullmatch(r"[a-z][a-z0-9._-]*", self.cli_profile):
            raise ValueError("The sandbox CLI profile must be an explicit identifier, not command arguments.")
        if (
            not isinstance(self.agent_workdir, PurePosixPath)
            or not self.agent_workdir.is_absolute()
            or ".." in self.agent_workdir.parts
            or "\\" in str(self.agent_workdir)
            or any(ord(char) < 32 or ord(char) == 127 for char in str(self.agent_workdir))
        ):
            raise ValueError("The agent workdir must be an absolute sandbox POSIX path without traversal.")
        if type(self.max_steps) is not int or self.max_steps < 1:
            raise ValueError("The native CLI step budget must be a positive integer.")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("The native CLI run timeout must be finite and positive.")
        if type(self.max_frame_bytes) is not int or self.max_frame_bytes < 1:
            raise ValueError("The JSONL frame limit must be a positive integer.")
        self._validate_gateway()

    def _validate_gateway(self) -> None:
        if not isinstance(self.model_gateway_endpoint, str):
            raise ValueError("A sandbox-reachable model gateway URL is required.")
        if any(char.isspace() or ord(char) < 32 or ord(char) == 127 for char in self.model_gateway_endpoint):
            raise ValueError("The model gateway URL must not contain whitespace or control characters.")
        parts = urlsplit(self.model_gateway_endpoint)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("The model gateway URL must have a host and contain no credentials, query, or fragment.")
        _ = parts.port


@dataclass(frozen=True, kw_only=True)
class NativeCliProcessChunk:
    """Bytes read from one sandbox process pipe, in the launcher's observed order."""

    stream: NativeCliStream
    data: bytes


@dataclass(frozen=True, kw_only=True)
class NativeCliRawChunk:
    """An unchanged process chunk with a monotonic cross-pipe recorder sequence."""

    sequence: int
    stream: NativeCliStream
    data: bytes


@dataclass(frozen=True, kw_only=True)
class NativeCliObservation:
    """Provider fields observed on one frame, without fabricated IDs or status."""

    kind: NativeCliEventKind
    status: NativeCliEventStatus = NativeCliEventStatus.UNKNOWN
    source_event_id: str | None = None
    source_message_id: str | None = None
    source_session_id: str | None = None
    source_tool_id: str | None = None
    parent_tool_use_id: str | None = None
    source_status: str | None = None
    name: str | None = None
    text: str | None = None
    arguments: JsonValue = None
    result: JsonValue = None
    exit_code: int | None = None
    detail: str | None = None


@dataclass(frozen=True, kw_only=True)
class NativeCliEvent:
    """An ordered observation paired with its exact JSONL frame when one exists."""

    sequence: int
    frame_number: int | None
    raw_frame: bytes | None
    observation: NativeCliObservation


@dataclass(frozen=True, kw_only=True)
class NativeCliRunOutcome:
    """Process and evidence coverage, not a grader verdict or tool success."""

    exit_code: int
    terminal_observed: bool
    coverage_complete: bool
    source_session_id: str | None
    observed_steps: int
    frame_count: int
    raw_chunk_count: int
    raw_stdout_bytes: int
    raw_stderr_bytes: int
    gaps: tuple[str, ...]
