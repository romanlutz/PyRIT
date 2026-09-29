# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A trusted-host Inspect provider for a verified, loopback-only local model."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessageAssistant,
    ContentText,
    GenerateConfig,
    ModelAPI,
    ModelCall,
    ModelOutput,
    ModelUsage,
    modelapi,
)
from inspect_ai.tool import ToolCall, ToolFunction

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from inspect_ai.model import ChatMessage
    from inspect_ai.tool import ToolChoice, ToolInfo


class InspectLoopbackModelAPI(ModelAPI):
    """Forward only text/function calls to the original local model, never a remote URL."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        config: GenerateConfig | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Pin the local provider to host loopback and a caller-owned output budget.

        Raises:
            ValueError: If the endpoint could reach a wildcard, remote or credential URL.
        """
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config if config is not None else GenerateConfig(),
        )
        parsed = urlsplit(base_url or "")
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "::1"}
            or parsed.port is None
            or parsed.path.rstrip("/") not in {"", "/v1"}
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
        ):
            raise ValueError("Inspect host model must use a verified loopback-only /v1 endpoint.")
        self._url = (base_url or "").rstrip("/") + (
            "/chat/completions" if parsed.path.rstrip("/") == "/v1" else "/v1/chat/completions"
        )
        self._transport = transport
        self._capture: Callable[[bytes, bytes | None, int | None, str | None], Awaitable[None]] | None = None

    def set_capture_sink(
        self, *, sink: Callable[[bytes, bytes | None, int | None, str | None], Awaitable[None]]
    ) -> None:
        """
        Retain original host HTTP bytes in the run's private PyRIT evidence stream.

        Raises:
            ValueError: If another capture sink is already bound to this model.
        """
        if self._capture is not None:
            raise ValueError("The trusted host model capture sink may be bound only once.")
        self._capture = sink

    def clear_capture_sink(
        self, *, sink: Callable[[bytes, bytes | None, int | None, str | None], Awaitable[None]]
    ) -> None:
        """
        Remove only this run's capture sink after its Inspect Task has finished.

        Raises:
            ValueError: If another run owns the model capture sink.
        """
        if self._capture is not sink:
            raise ValueError("A different Inspect run owns the local model capture sink.")
        self._capture = None

    async def generate(  # pyrit-async-suffix-exempt
        self,
        input: list[ChatMessage],  # noqa: A002 (Inspect's required ModelAPI signature)
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput | tuple[ModelOutput | Exception, ModelCall]:
        """
        Produce a real local model answer and retain both actual HTTP byte bodies.

        Returns:
            ModelOutput | tuple[ModelOutput | Exception, ModelCall]: Original model call and typed output.

        Raises:
            ValueError: If a request or response is unsupported or lacks quota-relevant usage.
            httpx.HTTPError: If the approved loopback model call fails.
        """
        if config.max_tokens is None or config.max_tokens < 1 or self._capture is None:
            raise ValueError("Inspect GHCP model requires bounded output and a source-byte capture sink.")
        request = self._request(messages=input, tools=tools, tool_choice=tool_choice, config=config)
        original = json.dumps(request, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
                timeout=config.timeout or 60,
            ) as client:
                result = await client.post(
                    self._url,
                    content=original,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key or 'local'}",
                        "Accept-Encoding": "identity",
                    },
                )
        except httpx.HTTPError as error:
            await self._capture(original, None, None, type(error).__name__)
            raise
        await self._capture(
            original,
            result.content,
            result.status_code,
            "host_model_rejected" if result.status_code != 200 else None,
        )
        result.raise_for_status()
        payload = result.json()
        if not isinstance(payload, dict):
            raise ValueError("The trusted host model did not return a JSON object.")
        output = self._output(payload=payload)
        output.metadata = {
            "host_model_response_sha256": hashlib.sha256(result.content).hexdigest(),
            "host_model_http_status": result.status_code,
        }
        return output, ModelCall(request=request, response=payload)

    def _request(
        self,
        *,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> dict[str, Any]:
        prepared: list[dict[str, Any]] = []
        for message in messages:
            body: dict[str, Any] = {"role": message.role, "content": self._text(message.content)}
            if isinstance(message, ChatMessageAssistant) and message.tool_calls:
                body["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function,
                            "arguments": json.dumps(call.arguments, separators=(",", ":")),
                        },
                    }
                    for call in message.tool_calls
                ]
            if message.role == "tool":
                if not message.tool_call_id:
                    raise ValueError("A model-visible tool result needs its observed call ID.")
                body["tool_call_id"] = message.tool_call_id
            prepared.append(body)
        result: dict[str, Any] = {
            "model": self.model_name,
            "messages": prepared,
            "stream": False,
            "max_tokens": config.max_tokens,
        }
        if config.temperature is not None:
            result["temperature"] = config.temperature
        if tools:
            result["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters.model_dump(mode="json", exclude_none=True),
                    },
                }
                for tool in tools
            ]
            result["parallel_tool_calls"] = False
            result["tool_choice"] = self._tool_choice(tool_choice=tool_choice)
        elif tool_choice not in ("auto", "none"):
            raise ValueError("An explicit tool choice cannot be satisfied without approved tools.")
        return result

    @staticmethod
    def _text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if not isinstance(part, ContentText):
                    raise ValueError("Inspect GHCP model requests cannot include non-text media.")
                parts.append(part.text)
            return "".join(parts)
        raise ValueError("Inspect GHCP model requests can contain only inline text, not host resources.")

    @staticmethod
    def _tool_choice(*, tool_choice: ToolChoice) -> str | dict[str, Any]:
        if isinstance(tool_choice, ToolFunction):
            return {"type": "function", "function": {"name": tool_choice.name}}
        if tool_choice == "any":
            return "required"
        if tool_choice in {"auto", "none"}:
            return tool_choice
        raise ValueError("The local provider received an unsupported tool-choice mode.")

    def _output(self, *, payload: dict[str, Any]) -> ModelOutput:
        choices, usage = payload.get("choices"), payload.get("usage")
        if (
            not isinstance(choices, list)
            or len(choices) != 1
            or not isinstance(choices[0], dict)
            or not isinstance(usage, dict)
            or type(usage.get("prompt_tokens")) is not int
            or type(usage.get("completion_tokens")) is not int
        ):
            raise ValueError("The local model response needs one choice and real token usage.")
        choice = choices[0]
        answer = choice.get("message")
        if not isinstance(answer, dict):
            raise ValueError("The local model answer is not structured.")
        text = answer.get("content")
        if text is not None and not isinstance(text, str):
            raise ValueError("The local model returned unsupported non-text content.")
        calls = self._parse_tool_calls(raw=answer.get("tool_calls"))
        finish = choice.get("finish_reason")
        if finish not in {"stop", "tool_calls", "length", "content_filter"}:
            raise ValueError("The local model returned an unknown completion state.")
        reason = "max_tokens" if finish == "length" else finish
        return ModelOutput(
            model=self.model_name,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(content=text or "", tool_calls=calls or None, model=self.model_name),
                    stop_reason=reason,
                )
            ],
            usage=ModelUsage(
                input_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                total_tokens=usage.get("total_tokens") or (usage["prompt_tokens"] + usage["completion_tokens"]),
            ),
        )

    @staticmethod
    def _parse_tool_calls(*, raw: Any) -> list[ToolCall]:
        if raw is None:
            return []
        if not isinstance(raw, list):
            raise ValueError("The local model did not return a list of tool calls.")
        calls: list[ToolCall] = []
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("function"), dict):
                raise ValueError("The local model returned an invalid tool call.")
            function = item["function"]
            arguments = function.get("arguments")
            parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
            if (
                not isinstance(item.get("id"), str)
                or not isinstance(function.get("name"), str)
                or not isinstance(parsed, dict)
            ):
                raise ValueError("The local model tool call has no ID, name, or object arguments.")
            calls.append(ToolCall(id=item["id"], function=function["name"], arguments=parsed))
        return calls


@modelapi("pyrit_loopback")
def pyrit_loopback() -> type[ModelAPI]:
    """
    Register the trusted local model provider with Inspect.

    Returns:
        type[ModelAPI]: The real bounded loopback model implementation.
    """
    return InspectLoopbackModelAPI
