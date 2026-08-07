# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError
from unit.mocks import MockPromptTarget

from pyrit.models import (
    ComponentIdentifier,
    Message,
    MessagePiece,
    TargetInvocation,
    construct_response_from_request,
)
from pyrit.prompt_normalizer import NormalizerRequest, PromptNormalizer
from pyrit.prompt_target import (
    UNSET,
    OpenAIChatRequestOptions,
    OpenAIChatTarget,
    OpenAIResponsesFunctionTool,
    OpenAIResponsesGrammarFormat,
    OpenAIResponsesGrammarTool,
    OpenAIResponsesNamedToolChoice,
    OpenAIResponsesRequestOptions,
    OpenAIResponseTarget,
    RoundRobinTarget,
)


class RequestOptionsTarget(MockPromptTarget):
    """A target that records the resolved options visible to each coroutine."""

    def __init__(self, *, name: str, default_temperature: float | None) -> None:
        super().__init__()
        self._name = name
        self._default_temperature = default_temperature
        self.seen_options: dict[str, OpenAIChatRequestOptions] = {}

    def _get_default_request_options(self) -> OpenAIChatRequestOptions:
        return OpenAIChatRequestOptions(
            max_completion_tokens=None,
            temperature=self._default_temperature,
            top_p=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            n=None,
            extra_body_parameters=None,
        )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        await asyncio.sleep(0)
        message = normalized_conversation[-1]
        self.seen_options[message.get_value()] = self._get_request_options(OpenAIChatRequestOptions)
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(params={"name": self._name})


class FailingRequestOptionsTarget(RequestOptionsTarget):
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        self._get_request_options(OpenAIChatRequestOptions)
        raise RuntimeError("target failed")


class LeakyResponseOptionsTarget(RequestOptionsTarget):
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1].message_pieces[0]
        return [
            MessagePiece(
                role="assistant",
                original_value="response",
                conversation_id=request.conversation_id,
                prompt_metadata=dict(request.prompt_metadata),
            ).to_message()
        ]


def _message(*, prompt: str, conversation_id: str) -> Message:
    message = Message.from_prompt(prompt=prompt, role="user")
    message.message_pieces[0].conversation_id = conversation_id
    return message


def test_request_options_resolve_inherit_override_and_explicit_clear() -> None:
    defaults = OpenAIChatRequestOptions(
        max_completion_tokens=100,
        temperature=0.7,
        top_p=0.9,
        frequency_penalty=None,
        presence_penalty=0.2,
        seed=42,
        n=1,
        extra_body_parameters={"service_tier": "default"},
    )
    requested = OpenAIChatRequestOptions(
        temperature=None,
        top_p=0.4,
    )

    resolved = requested.resolve(defaults=defaults)

    assert resolved.max_completion_tokens == 100
    assert resolved.temperature is None
    assert resolved.top_p == 0.4
    assert resolved.seed == 42
    assert requested.max_completion_tokens is UNSET
    with pytest.raises(ValidationError):
        requested.temperature = 1.0  # type: ignore[misc]


def test_construct_response_does_not_copy_invocation_metadata() -> None:
    request = MessagePiece(
        role="user",
        original_value="request",
        prompt_metadata={
            "custom": "preserved",
            TargetInvocation.METADATA_KEY: {"stale": "invocation"},
        },
    )

    response = construct_response_from_request(
        request=request,
        response_text_pieces=["response"],
        prompt_metadata={TargetInvocation.METADATA_KEY: {"explicit": "invocation"}},
    )

    assert response.message_pieces[0].prompt_metadata == {"custom": "preserved"}


@pytest.mark.usefixtures("patch_central_database")
async def test_prompt_target_strips_invocation_from_custom_response_builders() -> None:
    target = LeakyResponseOptionsTarget(name="leaky", default_temperature=0.25)
    request = _message(prompt="hello", conversation_id="leaky-response")

    responses = await target.send_prompt_async(message=request)

    assert TargetInvocation.METADATA_KEY in request.message_pieces[0].prompt_metadata
    assert TargetInvocation.METADATA_KEY not in responses[0].message_pieces[0].prompt_metadata


@pytest.mark.usefixtures("patch_central_database")
async def test_prompt_target_records_fully_resolved_invocation_once() -> None:
    target = RequestOptionsTarget(name="direct", default_temperature=0.25)
    request = _message(prompt="hello", conversation_id="provenance")

    responses = await target.send_prompt_async(
        message=request,
        request_options=OpenAIChatRequestOptions(top_p=0.8),
    )

    invocation = TargetInvocation.from_metadata(metadata=request.message_pieces[0].prompt_metadata)
    assert invocation is not None
    assert invocation.target_identifier == target.get_identifier()
    assert invocation.effective_options["temperature"] == 0.25
    assert invocation.effective_options["top_p"] == 0.8
    assert TargetInvocation.METADATA_KEY not in responses[0].message_pieces[0].prompt_metadata


@pytest.mark.usefixtures("patch_central_database")
async def test_prompt_target_snapshots_nested_options_before_await() -> None:
    target = RequestOptionsTarget(name="snapshot", default_temperature=0.25)
    request = _message(prompt="snapshot", conversation_id="snapshot-options")
    options = OpenAIChatRequestOptions(extra_body_parameters={"metadata": {"source": "original"}})

    send_task = asyncio.create_task(target.send_prompt_async(message=request, request_options=options))
    await asyncio.sleep(0)
    metadata = options.extra_body_parameters
    assert isinstance(metadata, dict)
    nested = metadata["metadata"]
    assert isinstance(nested, dict)
    nested["source"] = "mutated"
    await send_task

    seen_metadata = target.seen_options["snapshot"].extra_body_parameters
    assert isinstance(seen_metadata, dict)
    assert seen_metadata["metadata"] == {"source": "original"}


@pytest.mark.usefixtures("patch_central_database")
async def test_normalizer_persists_target_invocation() -> None:
    target = RequestOptionsTarget(name="persisted", default_temperature=0.3)
    normalizer = PromptNormalizer()
    conversation_id = "persisted-invocation"

    await normalizer.send_prompt_async(
        message=Message.from_prompt(prompt="persist me", role="user"),
        target=target,
        conversation_id=conversation_id,
        request_options=OpenAIChatRequestOptions(temperature=0.6),
    )

    persisted = normalizer.memory.get_conversation_messages(conversation_id=conversation_id)
    invocation = TargetInvocation.from_metadata(metadata=persisted[0].message_pieces[0].prompt_metadata)
    assert invocation is not None
    assert invocation.target_identifier == target.get_identifier()
    assert invocation.effective_options["temperature"] == 0.6


@pytest.mark.usefixtures("patch_central_database")
async def test_normalizer_persists_invocation_when_target_fails() -> None:
    target = FailingRequestOptionsTarget(name="failure", default_temperature=0.3)
    normalizer = PromptNormalizer()
    conversation_id = "failed-invocation"

    with pytest.raises(Exception, match="Error sending prompt"):
        await normalizer.send_prompt_async(
            message=Message.from_prompt(prompt="fail", role="user"),
            target=target,
            conversation_id=conversation_id,
            request_options=OpenAIChatRequestOptions(temperature=0.9),
        )

    persisted = normalizer.memory.get_conversation_messages(conversation_id=conversation_id)
    invocation = TargetInvocation.from_metadata(metadata=persisted[0].message_pieces[0].prompt_metadata)
    assert invocation is not None
    assert invocation.target_identifier == target.get_identifier()
    assert invocation.effective_options["temperature"] == 0.9


@pytest.mark.usefixtures("patch_central_database")
async def test_normalizer_batch_resolves_each_request_independently() -> None:
    target = RequestOptionsTarget(name="batch", default_temperature=0.1)
    normalizer = PromptNormalizer()
    requests = [
        NormalizerRequest(
            message=Message.from_prompt(prompt="override", role="user"),
            request_options=OpenAIChatRequestOptions(temperature=0.8),
        ),
        NormalizerRequest(
            message=Message.from_prompt(prompt="clear", role="user"),
            request_options=OpenAIChatRequestOptions(temperature=None),
        ),
        NormalizerRequest(message=Message.from_prompt(prompt="inherit", role="user")),
    ]

    await normalizer.send_prompt_batch_to_target_async(requests=requests, target=target, batch_size=3)

    assert target.seen_options["override"].temperature == 0.8
    assert target.seen_options["clear"].temperature is None
    assert target.seen_options["inherit"].temperature == 0.1
    assert target._default_temperature == 0.1


@pytest.mark.usefixtures("patch_central_database")
async def test_round_robin_forwards_options_and_records_selected_target() -> None:
    first = RequestOptionsTarget(name="first", default_temperature=0.2)
    second = RequestOptionsTarget(name="second", default_temperature=0.9)
    target = RoundRobinTarget(targets=[first, second])
    first_request = _message(prompt="override", conversation_id="round-robin-1")
    second_request = _message(prompt="inherit", conversation_id="round-robin-2")

    await target.send_prompt_async(
        message=first_request,
        request_options=OpenAIChatRequestOptions(temperature=0.5),
    )
    await target.send_prompt_async(message=second_request)

    assert first.seen_options["override"].temperature == 0.5
    assert second.seen_options["inherit"].temperature == 0.9
    first_invocation = TargetInvocation.from_metadata(metadata=first_request.message_pieces[0].prompt_metadata)
    second_invocation = TargetInvocation.from_metadata(metadata=second_request.message_pieces[0].prompt_metadata)
    assert first_invocation is not None
    assert second_invocation is not None
    assert first_invocation.target_identifier == first.get_identifier()
    assert second_invocation.target_identifier == second.get_identifier()


@pytest.mark.usefixtures("patch_central_database")
async def test_openai_chat_request_options_override_request_body() -> None:
    target = OpenAIChatTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="key",
        temperature=0.2,
        top_p=0.7,
    )
    request = _message(prompt="chat", conversation_id="chat-options")
    create = AsyncMock(return_value=MagicMock())

    async def handle_request_async(*, api_call, request):
        await api_call()
        return MessagePiece(role="assistant", original_value="ok", conversation_id=request.conversation_id).to_message()

    with (
        patch.object(target._client.chat.completions, "create", create),
        patch.object(target, "_handle_openai_request_async", side_effect=handle_request_async),
    ):
        await target.send_prompt_async(
            message=request,
            request_options=OpenAIChatRequestOptions(temperature=None, top_p=0.4, max_completion_tokens=25),
        )

    body = create.call_args.kwargs
    assert "temperature" not in body
    assert body["top_p"] == 0.4
    assert body["max_completion_tokens"] == 25


@pytest.mark.usefixtures("patch_central_database")
async def test_openai_chat_uses_per_call_audio_format_for_response() -> None:
    target = OpenAIChatTarget(
        model_name="gpt-4o-audio-preview",
        endpoint="https://mock.azure.com/",
        api_key="key",
    )
    request = _message(prompt="audio", conversation_id="chat-audio-options")
    completion = MagicMock()
    completion.choices = [MagicMock(finish_reason="stop")]
    response_piece = MessagePiece(
        role="assistant",
        original_value="audio",
        conversation_id=request.conversation_id,
    )

    async def handle_request_async(*, api_call, request):
        return await target._construct_message_from_response_async(completion, request.message_pieces[0])

    with (
        patch.object(target, "_handle_openai_request_async", side_effect=handle_request_async),
        patch(
            "pyrit.prompt_target.openai.openai_chat_target.build_response_pieces_async",
            new_callable=AsyncMock,
            return_value=[response_piece],
        ) as build_response,
    ):
        await target.send_prompt_async(
            message=request,
            request_options=OpenAIChatRequestOptions(
                extra_body_parameters={
                    "modalities": ["text", "audio"],
                    "audio": {"voice": "alloy", "format": "mp3"},
                }
            ),
        )

    assert build_response.call_args.kwargs["audio_format"] == "mp3"


@pytest.mark.usefixtures("patch_central_database")
async def test_openai_responses_typed_options_build_request_body() -> None:
    target = OpenAIResponseTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="key",
        reasoning_effort="low",
    )
    request = _message(prompt="responses", conversation_id="responses-options")
    create = AsyncMock(return_value=MagicMock())
    function_tool = OpenAIResponsesFunctionTool(
        name="lookup",
        description="Look up an item",
        parameters={"type": "object", "properties": {"id": {"type": "string"}}},
        strict=True,
    )
    grammar_tool = OpenAIResponsesGrammarTool(
        name="answer",
        format=OpenAIResponsesGrammarFormat(syntax="lark", definition='start: "yes" | "no"'),
    )

    async def handle_request_async(*, api_call, request):
        await api_call()
        return MessagePiece(role="assistant", original_value="ok", conversation_id=request.conversation_id).to_message()

    with (
        patch.object(target._client.responses, "create", create),
        patch.object(target, "_handle_openai_request_async", side_effect=handle_request_async),
    ):
        await target.send_prompt_async(
            message=request,
            request_options=OpenAIResponsesRequestOptions(
                reasoning_effort="high",
                reasoning_summary="concise",
                reasoning_extra={"generate_summary": "auto"},
                tools=(function_tool, grammar_tool),
                tool_choice=OpenAIResponsesNamedToolChoice(type="function", name="lookup"),
                parallel_tool_calls=False,
            ),
        )

    body = create.call_args.kwargs
    assert body["reasoning"] == {
        "effort": "high",
        "summary": "concise",
        "generate_summary": "auto",
    }
    assert body["tools"][0]["name"] == "lookup"
    assert body["tools"][1]["format"]["type"] == "grammar"
    assert body["tool_choice"] == {"type": "function", "name": "lookup"}
    assert body["parallel_tool_calls"] is False


@pytest.mark.usefixtures("patch_central_database")
def test_openai_responses_preserves_supported_passthrough_defaults() -> None:
    target = OpenAIResponseTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="key",
        reasoning_effort="xhigh",
        extra_body_parameters={
            "tools": [
                {"type": "web_search_preview"},
                {
                    "type": "custom",
                    "name": "legacy_grammar",
                    "format": {"type": "grammar", "syntax": "lark", "definition": 'start: "yes"'},
                },
            ],
            "tool_choice": {"type": "hosted_tool", "name": "web_search_preview"},
            "reasoning": {"effort": "max", "generate_summary": "auto"},
        },
    )

    defaults = target._get_default_request_options()
    repeated_defaults = target._get_default_request_options()

    assert defaults.reasoning_effort == "max"
    assert defaults.reasoning_extra == {"generate_summary": "auto"}
    assert isinstance(defaults.tools, tuple)
    assert defaults.tools[0] == {"type": "web_search_preview"}
    assert defaults.tool_choice == {"type": "hosted_tool", "name": "web_search_preview"}
    assert target._get_grammar_name() == "legacy_grammar"
    assert repeated_defaults == defaults
    assert target._extra_body_parameters["reasoning"] == {
        "effort": "max",
        "generate_summary": "auto",
    }


@pytest.mark.usefixtures("patch_central_database")
def test_openai_chat_preserves_json_passthrough_with_integer_nested_keys() -> None:
    target = OpenAIChatTarget(
        model_name="gpt-4o",
        endpoint="https://mock.azure.com/",
        api_key="key",
        extra_body_parameters={"logit_bias": {123: 2}},
    )

    defaults = target._get_default_request_options()

    assert isinstance(defaults.extra_body_parameters, dict)
    assert defaults.extra_body_parameters["logit_bias"] == {123: 2}
