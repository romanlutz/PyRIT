# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from pyrit.models import JSONValue  # noqa: TC001 (runtime-required by Pydantic)


class UnsetValue(str, Enum):
    """Sentinel used when a per-call option should inherit its target default."""

    UNSET = "__unset__"


UNSET = UnsetValue.UNSET


class TargetRequestOptions(BaseModel):
    """Immutable per-call request options with tri-state resolution semantics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def resolve(self, *, defaults: TargetRequestOptions) -> TargetRequestOptions:
        """
        Resolve inherited values against target constructor defaults.

        Returns:
            A fully resolved immutable options object.

        Raises:
            TypeError: If the supplied defaults use a different options type.
        """
        if type(self) is not type(defaults):
            raise TypeError(f"{type(defaults).__name__} is required for this target; received {type(self).__name__}.")

        values = {}
        for name in type(self).model_fields:
            requested_value = getattr(self, name)
            value = getattr(defaults, name) if requested_value is UNSET else requested_value
            values[name] = copy.deepcopy(value)
        return type(self).model_validate(values)

    def to_effective_dict(self) -> dict[str, object]:
        """
        Return fully resolved options as JSON-compatible values.

        Returns:
            The fully resolved JSON-compatible option mapping.

        Raises:
            ValueError: If any option remains unresolved.
        """
        unresolved = [name for name in type(self).model_fields if getattr(self, name) is UNSET]
        if unresolved:
            raise ValueError(f"Request options are not fully resolved: {', '.join(unresolved)}")
        return self.model_dump(mode="json")


class TextGenerationRequestOptions(TargetRequestOptions):
    """Common sampling controls shared by text-generation targets."""

    temperature: float | None | UnsetValue = UNSET
    top_p: float | None | UnsetValue = UNSET

    @model_validator(mode="after")
    def _validate_sampling_controls(self) -> TextGenerationRequestOptions:
        if isinstance(self.temperature, float) and not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be between 0 and 2.")
        if isinstance(self.top_p, float) and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        return self


class OpenAIChatRequestOptions(TextGenerationRequestOptions):
    """Per-call request options for ``OpenAIChatTarget``."""

    max_completion_tokens: int | None | UnsetValue = UNSET
    frequency_penalty: float | None | UnsetValue = UNSET
    presence_penalty: float | None | UnsetValue = UNSET
    seed: int | None | UnsetValue = UNSET
    n: int | None | UnsetValue = UNSET
    extra_body_parameters: dict[str, object] | None | UnsetValue = UNSET


class OpenAIResponsesFunctionTool(BaseModel):
    """A function declaration for the OpenAI Responses API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: dict[str, JSONValue] | None = None
    strict: bool | None = None


class OpenAIResponsesGrammarFormat(BaseModel):
    """A grammar format for an OpenAI custom tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["grammar"] = "grammar"
    syntax: Literal["lark", "regex"]
    definition: str


class OpenAIResponsesGrammarTool(BaseModel):
    """A custom grammar tool declaration for the OpenAI Responses API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["custom"] = "custom"
    name: str
    description: str | None = None
    format: OpenAIResponsesGrammarFormat


OpenAIResponsesTool = OpenAIResponsesFunctionTool | OpenAIResponsesGrammarTool | dict[str, JSONValue]


class OpenAIResponsesNamedToolChoice(BaseModel):
    """A named function or custom tool choice for the OpenAI Responses API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["function", "custom"]
    name: str


OpenAIResponsesToolChoice = Literal["none", "auto", "required"] | OpenAIResponsesNamedToolChoice | dict[str, JSONValue]
ToolExecutionMode = Literal["single_generation", "legacy_auto"]


class OpenAIResponsesRequestOptions(TextGenerationRequestOptions):
    """Per-call request options for ``OpenAIResponseTarget``."""

    max_output_tokens: int | None | UnsetValue = UNSET
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None | UnsetValue = UNSET
    reasoning_summary: Literal["auto", "concise", "detailed"] | None | UnsetValue = UNSET
    reasoning_extra: dict[str, object] | None | UnsetValue = UNSET
    tools: tuple[OpenAIResponsesTool, ...] | None | UnsetValue = UNSET
    tool_choice: OpenAIResponsesToolChoice | None | UnsetValue = UNSET
    parallel_tool_calls: bool | None | UnsetValue = UNSET
    tool_execution_mode: ToolExecutionMode | UnsetValue = UNSET
    extra_body_parameters: dict[str, object] | None | UnsetValue = UNSET


class LiteLLMRequestOptions(TextGenerationRequestOptions):
    """Per-call request options for ``LiteLLMChatTarget``."""

    max_tokens: int | None | UnsetValue = UNSET
    frequency_penalty: float | None | UnsetValue = UNSET
    presence_penalty: float | None | UnsetValue = UNSET
    seed: int | None | UnsetValue = UNSET
    n: int | None | UnsetValue = UNSET
    stop: str | tuple[str, ...] | None | UnsetValue = UNSET
    drop_unsupported_params: bool | None | UnsetValue = UNSET
    extra_body_parameters: dict[str, object] | None | UnsetValue = UNSET


class HuggingFaceRequestOptions(TargetRequestOptions):
    """Per-call generation options for ``HuggingFaceChatTarget``."""

    max_new_tokens: int | UnsetValue = UNSET
    temperature: float | UnsetValue = UNSET
    top_p: float | UnsetValue = UNSET
    top_k: int | None | UnsetValue = UNSET
    do_sample: bool | None | UnsetValue = UNSET
    repetition_penalty: float | None | UnsetValue = UNSET
    random_seed: int | None | UnsetValue = UNSET
    skip_special_tokens: bool | UnsetValue = UNSET

    @model_validator(mode="after")
    def _validate_generation_controls(self) -> HuggingFaceRequestOptions:
        if isinstance(self.max_new_tokens, int) and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if isinstance(self.top_p, float) and not 0 <= self.top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        return self


class OpenAIImageRequestOptions(TargetRequestOptions):
    """Per-call request options for ``OpenAIImageTarget``."""

    image_size: Literal["auto", "1024x1024", "1536x1024", "1024x1536"] | UnsetValue = UNSET
    output_format: Literal["png", "jpeg", "webp"] | None | UnsetValue = UNSET
    quality: Literal["auto", "low", "medium", "high"] | None | UnsetValue = UNSET
    background: Literal["transparent", "opaque", "auto"] | None | UnsetValue = UNSET

    @model_validator(mode="after")
    def _validate_transparency(self) -> OpenAIImageRequestOptions:
        if self.background == "transparent" and self.output_format == "jpeg":
            raise ValueError("background='transparent' requires output_format='png' or 'webp'.")
        return self


class OpenAITTSRequestOptions(TargetRequestOptions):
    """Per-call request options for ``OpenAITTSTarget``."""

    voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"] | UnsetValue = UNSET
    response_format: Literal["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"] | UnsetValue = UNSET
    speed: float | None | UnsetValue = UNSET

    @model_validator(mode="after")
    def _validate_speed(self) -> OpenAITTSRequestOptions:
        if isinstance(self.speed, float) and not 0.25 <= self.speed <= 4:
            raise ValueError("speed must be between 0.25 and 4.")
        return self


class OpenAIVideoRequestOptions(TargetRequestOptions):
    """Per-call request options for ``OpenAIVideoTarget``."""

    resolution_dimensions: Literal["720x1280", "1280x720", "1024x1792", "1792x1024"] | UnsetValue = UNSET
    n_seconds: Literal["4", "8", "12"] | UnsetValue = UNSET
