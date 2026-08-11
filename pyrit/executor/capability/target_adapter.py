# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Target request-option adapters for capability execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from pyrit.executor.capability.models import ToolExecutionPolicy
from pyrit.prompt_target import (
    CapabilityName,
    OpenAIResponsesFunctionTool,
    OpenAIResponsesRequestOptions,
    PromptTarget,
    TargetRequestOptions,
    TargetRequirements,
)

if TYPE_CHECKING:
    from pyrit.executor.capability.tools import ToolDeclaration
    from pyrit.models import PromptDataType


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


def build_capability_request_options_factory(*, target: PromptTarget) -> CapabilityRequestOptionsFactory:
    """
    Build the supported capability adapter for a target's declared options transport.

    Returns:
        CapabilityRequestOptionsFactory: The compatible request-options adapter.

    Raises:
        ValueError: If no adapter supports the target's request-options transport.
    """
    if target.request_options_type is OpenAIResponsesRequestOptions:
        return OpenAIResponsesCapabilityRequestOptionsFactory()
    target_name = target.get_identifier().class_name
    raise ValueError(
        f"Target '{target_name}' uses request-options transport '{target.request_options_type.__name__}', "
        "which has no registered capability tool adapter. Select a target using "
        "'OpenAIResponsesRequestOptions' or supply a compatible adapter programmatically."
    )


def validate_capability_target(
    *,
    target: PromptTarget,
    request_options_factory: CapabilityRequestOptionsFactory | None,
    requires_multi_turn: bool,
    requires_tools: bool,
    requires_system_prompt: bool = False,
    required_input_modalities: frozenset[frozenset[PromptDataType]] | None = None,
    required_output_modalities: frozenset[frozenset[PromptDataType]] | None = None,
) -> None:
    """
    Validate one resolved target instance before capability execution starts.

    Raises:
        ValueError: If the target or request-options adapter cannot satisfy the task.
    """
    native_required = set[CapabilityName]()
    if requires_multi_turn:
        native_required.update((CapabilityName.MULTI_TURN, CapabilityName.EDITABLE_HISTORY))
    if requires_system_prompt:
        native_required.add(CapabilityName.SYSTEM_PROMPT)
    required_inputs: set[frozenset[PromptDataType]] = set(required_input_modalities or ())
    required_outputs: set[frozenset[PromptDataType]] = set(required_output_modalities or ())
    if not required_inputs:
        required_inputs.add(frozenset({"text"}))
    if not required_outputs:
        required_outputs.add(frozenset({"text"}))
    if requires_tools:
        native_required.add(CapabilityName.EXTERNAL_TOOL_EXECUTION)
        required_inputs.add(frozenset({"function_call_output"}))
        required_outputs.add(frozenset({"function_call"}))
    requirements = TargetRequirements(
        native_required=frozenset(native_required),
        required_input_modalities=frozenset(required_inputs),
        required_output_modalities=frozenset(required_outputs),
    )
    target_name = target.get_identifier().class_name
    try:
        requirements.validate(target=target)
    except ValueError as error:
        raise ValueError(
            f"Target '{target_name}' is incompatible with this capability task: {error} "
            "Select a target whose declared capabilities and modalities satisfy every listed requirement."
        ) from error
    if request_options_factory is None:
        return
    options = request_options_factory.build_request_options(
        declarations=(),
        execution_policy=ToolExecutionPolicy.SEQUENTIAL,
    )
    if type(options) is not target.request_options_type:
        resolution = (
            "Select a compatible target/adapter pair that supports caller-owned single-generation tool execution."
            if requires_tools
            else "Select a compatible target/adapter pair that uses the target's declared request-options transport."
        )
        raise ValueError(
            f"Target '{target_name}' requires request-options transport '{target.request_options_type.__name__}', "
            f"but the capability adapter produces '{type(options).__name__}'. {resolution}"
        )
    if (
        requires_tools
        and isinstance(options, OpenAIResponsesRequestOptions)
        and options.tool_execution_mode != "single_generation"
    ):
        raise ValueError(
            f"Target '{target_name}' requires OpenAI Responses single-generation external-tool ownership, "
            f"but the capability adapter resolved tool_execution_mode={options.tool_execution_mode!r}. "
            "Use an adapter that sets tool_execution_mode='single_generation'."
        )
