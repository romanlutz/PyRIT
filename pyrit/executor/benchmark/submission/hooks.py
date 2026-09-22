# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass(frozen=True, kw_only=True)
class SubmissionTool:
    """A caller-owned tool schema and async string-returning implementation."""

    name: str
    description: str
    parameters: dict[str, Any]
    callback_async: Callable[..., Awaitable[str]]

    def __post_init__(self) -> None:
        """
        Validate the public tool boundary.

        Raises:
            ValueError: If tool identity, schema, or callback is invalid.
        """
        if not self.name or not self.description or self.parameters.get("type") != "object":
            raise ValueError("A submission tool requires a name, description, and object argument schema.")
        if not inspect.iscoroutinefunction(self.callback_async):
            raise ValueError("Tool implementations must be async keyword-argument callbacks.")

    def response_definition(self) -> dict[str, Any]:
        """
        Build the Responses function definition without rewriting its supplied schema.

        Returns:
            dict[str, Any]: The tool definition.
        """
        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


@dataclass(frozen=True, kw_only=True)
class SubmissionHooks:
    """The explicit binding-owned state and exception-to-feedback boundary."""

    tools: tuple[SubmissionTool, ...]
    read_report: Callable[[], dict[str, Any]]
    recoverable_errors: tuple[type[Exception], ...]
    terminal_errors: tuple[type[Exception], ...]
    error_feedback: Callable[[Exception], str]

    def __post_init__(self) -> None:
        """
        Require explicit, disjoint exception policies.

        Raises:
            ValueError: If tool names or exception classifiers conflict.
        """
        names = [tool.name for tool in self.tools]
        if not names or len(set(names)) != len(names):
            raise ValueError("Tool names must be nonempty and unique.")
        if any(
            issubclass(recoverable, terminal) or issubclass(terminal, recoverable)
            for recoverable in self.recoverable_errors
            for terminal in self.terminal_errors
        ):
            raise ValueError("Recoverable and terminal exception classifiers must not overlap.")


@dataclass(frozen=True, kw_only=True)
class SubmissionLimits:
    """Local offline loop limits, unrelated to a remote worker's capabilities."""

    max_requests: int = 12
    max_tool_calls: int = 12
    episode_timeout_seconds: float = 30
    max_response_bytes: int = 1_048_576

    def __post_init__(self) -> None:
        """
        Reject invalid local limits.

        Raises:
            ValueError: If any limit is nonpositive.
        """
        if min(self.max_requests, self.max_tool_calls, self.episode_timeout_seconds, self.max_response_bytes) <= 0:
            raise ValueError("All offline loop limits must be positive.")
