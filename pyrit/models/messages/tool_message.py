# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Canonical provider-neutral tool request and result payloads."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from pyrit.models.identifiers import JSONValue  # noqa: TC001 (runtime-required by Pydantic)


class ToolCallProvenance(BaseModel):
    """Optional provider provenance retained alongside a canonical tool payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    raw: dict[str, JSONValue] | None = None


class ToolCallError(BaseModel):
    """A structured error produced while handling a tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str
    message: str
    details: JSONValue = None


class ToolCallRequest(BaseModel):
    """A canonical function-tool invocation requested by a provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["function_call"] = "function_call"
    call_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: str
    provenance: ToolCallProvenance | None = None

    def to_json(self) -> str:
        """
        Serialize this request to the stable persisted JSON schema.

        Returns:
            The canonical JSON representation.
        """
        return self.model_dump_json(exclude_none=True)

    def to_openai_chat_tool_call(self) -> dict[str, JSONValue]:
        """
        Serialize this request as an OpenAI Chat Completions tool call.

        Returns:
            The Chat Completions wire payload.
        """
        return {
            "id": self.call_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    def to_openai_responses_function_call(self) -> dict[str, JSONValue]:
        """
        Serialize this request as an OpenAI Responses function-call item.

        Returns:
            The Responses API wire payload.
        """
        return {
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
        }

    @classmethod
    def from_json(cls, value: str) -> ToolCallRequest:
        """
        Parse canonical or legacy provider-shaped persisted JSON.

        Returns:
            The normalized tool call.

        Raises:
            ValueError: If the payload is not a JSON object or contains non-string fields.
        """
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Tool call payload must be a JSON object.")

        payload_type = payload.get("type")
        function = payload.get("function")
        if isinstance(function, dict):
            if payload_type != "function":
                raise ValueError("Legacy Chat Completions tool calls must have type 'function'.")
            return cls(
                call_id=_required_string(payload=payload, key="id"),
                name=_required_string(payload=function, key="name"),
                arguments=_required_string(payload=function, key="arguments"),
            )
        if payload_type != "function_call":
            raise ValueError("Canonical tool calls must have type 'function_call'.")
        provenance = payload.get("provenance")
        return cls(
            call_id=_required_string(payload=payload, key="call_id"),
            name=_required_string(payload=payload, key="name"),
            arguments=_required_string(payload=payload, key="arguments"),
            provenance=ToolCallProvenance.model_validate(provenance) if provenance is not None else None,
        )

    @classmethod
    def from_openai_chat(
        cls,
        *,
        call_id: str,
        name: str,
        arguments: str,
        raw: dict[str, JSONValue] | None = None,
    ) -> ToolCallRequest:
        """
        Normalize an OpenAI Chat Completions tool call.

        Returns:
            The canonical tool call.
        """
        return cls(
            call_id=call_id,
            name=name,
            arguments=arguments,
            provenance=ToolCallProvenance(provider="openai_chat_completions", raw=raw),
        )

    @classmethod
    def from_openai_responses(
        cls,
        *,
        call_id: str,
        name: str,
        arguments: str,
        raw: dict[str, JSONValue] | None = None,
    ) -> ToolCallRequest:
        """
        Normalize an OpenAI Responses function call.

        Returns:
            The canonical tool call.
        """
        return cls(
            call_id=call_id,
            name=name,
            arguments=arguments,
            provenance=ToolCallProvenance(provider="openai_responses", raw=raw),
        )


class ToolCallResult(BaseModel):
    """A canonical result returned for a tool invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: Literal["function_call_output"] = "function_call_output"
    call_id: str = Field(min_length=1)
    output: str
    error: ToolCallError | None = None
    provenance: ToolCallProvenance | None = None

    def to_json(self) -> str:
        """
        Serialize this result to the stable persisted JSON schema.

        Returns:
            The canonical JSON representation.
        """
        return self.model_dump_json(exclude_none=True)

    def to_openai_chat_message(self) -> dict[str, JSONValue]:
        """
        Serialize this result as an OpenAI Chat Completions tool message.

        Returns:
            The Chat Completions wire payload.
        """
        return {
            "role": "tool",
            "tool_call_id": self.call_id,
            "content": self.output,
        }

    def to_openai_responses_function_call_output(self) -> dict[str, JSONValue]:
        """
        Serialize this result as an OpenAI Responses function-call output item.

        Returns:
            The Responses API wire payload.
        """
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": self.output,
        }

    @classmethod
    def from_json(cls, value: str) -> ToolCallResult:
        """
        Parse a canonical persisted result payload.

        Returns:
            The normalized tool result.

        Raises:
            ValueError: If the payload is not a JSON object or contains non-string fields.
        """
        payload = json.loads(value)
        if not isinstance(payload, dict):
            raise ValueError("Tool result payload must be a JSON object.")
        if payload.get("type") != "function_call_output":
            raise ValueError("Canonical tool results must have type 'function_call_output'.")
        call_id = _required_string(payload=payload, key="call_id")
        output = payload.get("output")
        if not isinstance(output, str):
            output = json.dumps(output, separators=(",", ":"))
        error = payload.get("error")
        provenance = payload.get("provenance")
        return cls(
            call_id=call_id,
            output=output,
            error=ToolCallError.model_validate(error) if error is not None else None,
            provenance=ToolCallProvenance.model_validate(provenance) if provenance is not None else None,
        )


def _required_string(*, payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Tool payload field '{key}' must be a string.")
    return value
