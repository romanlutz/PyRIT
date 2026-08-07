# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Target request-option adapters for capability execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pyrit.executor.capability.models import ToolExecutionPolicy
from pyrit.prompt_target import (
    OpenAIResponsesFunctionTool,
    OpenAIResponsesRequestOptions,
    TargetRequestOptions,
)

if TYPE_CHECKING:
    from pyrit.executor.capability.tools import ToolDeclaration


class CapabilityRequestOptionsFactory(Protocol):
    """Build target options for exactly one provider generation."""

    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        """Return immutable options that force one provider generation."""


class OpenAIResponsesCapabilityRequestOptionsFactory:
    """Map neutral declarations to OpenAI Responses single-generation options."""

    def __init__(self, *, base_options: OpenAIResponsesRequestOptions | None = None) -> None:
        """Initialize the adapter."""
        self._base_options = base_options or OpenAIResponsesRequestOptions()

    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        """
        Build one-generation OpenAI Responses options.

        Returns:
            TargetRequestOptions: Immutable options forcing one provider generation.
        """
        tools = tuple(
            OpenAIResponsesFunctionTool(
                name=declaration.name,
                description=declaration.description,
                parameters=declaration.input_schema,
            )
            for declaration in declarations
        )
        return self._base_options.model_copy(
            update={
                "tools": tools,
                "parallel_tool_calls": execution_policy is ToolExecutionPolicy.PARALLEL,
                "tool_execution_mode": "single_generation",
            }
        )
