# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from pyrit.exceptions import InvalidJsonException, pyrit_json_retry
from pyrit.models import JSON_SCHEMA_METADATA_KEY, Message, MessagePiece

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.models import (
        ComponentIdentifier,
        JsonSchemaDefinition,
        PromptDataType,
        UnvalidatedScore,
    )
    from pyrit.prompt_target import PromptTarget
    from pyrit.score.response_handler import ResponseHandler


async def run_llm_scoring_async(
    *,
    chat_target: PromptTarget,
    system_prompt: str,
    response_handler: ResponseHandler,
    value: str,
    data_type: PromptDataType,
    scored_prompt_id: str | uuid.UUID,
    scorer_identifier: ComponentIdentifier,
    prepended_text: str | None = None,
    category: Sequence[str] | str | None = None,
    objective: str | None = None,
    response_json_schema: JsonSchemaDefinition | None = None,
    numeric_value: bool = False,
) -> UnvalidatedScore:
    """
    Perform a single scoring round-trip against an LLM target and parse the result.

    This is the shared LLM evaluation mechanism: it sets the system prompt on the target,
    sends the value to be scored, applies the standard JSON retry behavior, and delegates
    parsing to ``response_handler``. It is intentionally stateless and independent of any
    particular ``Scorer`` so that scorers can compose it without inheriting LLM machinery.

    Args:
        chat_target (PromptTarget): The target LLM to send the message to.
        system_prompt (str): The system-level prompt that guides the target LLM.
        response_handler (ResponseHandler): Parser that turns the target's raw text into an
            ``UnvalidatedScore``.
        value (str): The content to be scored (e.g. text, image path, audio path).
        data_type (PromptDataType): The data type of ``value`` (e.g. "text", "image_path").
        scored_prompt_id (str | uuid.UUID): The ID of the message piece being scored.
        scorer_identifier (ComponentIdentifier): Identifier of the calling scorer, stored on
            the resulting score.
        prepended_text (str | None): Text context to prepend before ``value`` as a separate
            piece. Useful for adding objective/context when scoring non-text content.
            Defaults to None.
        category (Sequence[str] | str | None): The category of the score. May instead be parsed
            from the response; supplying both is an error. Defaults to None.
        objective (str | None): The objective associated with the score, used for
            contextualizing the result. Defaults to None.
        response_json_schema (JsonSchemaDefinition | None): Optional JSON schema constraining the
            response. Forwarded to the request metadata; targets that natively support JSON
            schemas enforce it, others have it omitted by normalization. Defaults to None.
        numeric_value (bool): When True, the parsed ``raw_score_value`` must be parsable as a
            float; otherwise an ``InvalidJsonException`` is raised (without retrying). Defaults
            to False.

    Returns:
        UnvalidatedScore: The parsed score, whose ``raw_score_value`` still needs to be
            normalized and validated by the caller.

    Raises:
        InvalidJsonException: If the response is not valid JSON, is missing required keys, or
            (when ``numeric_value`` is True) the score value is not a float.
        Exception: For other unexpected errors during scoring.
    """
    score = await _send_and_parse_async(
        chat_target=chat_target,
        system_prompt=system_prompt,
        response_handler=response_handler,
        value=value,
        data_type=data_type,
        scored_prompt_id=scored_prompt_id,
        scorer_identifier=scorer_identifier,
        prepended_text=prepended_text,
        category=category,
        objective=objective,
        response_json_schema=response_json_schema,
    )

    if numeric_value:
        try:
            # Raise an exception if the score value is not parsable as a float. This mirrors the
            # historical float-scale behavior: the check runs outside the JSON retry, so a
            # well-formed-but-non-numeric response is not retried.
            float(score.raw_score_value)
        except ValueError:
            raise InvalidJsonException(
                message=f"Invalid JSON response, score_value should be a float not this: {score.raw_score_value}"
            ) from None

    return score


@pyrit_json_retry
async def _send_and_parse_async(
    *,
    chat_target: PromptTarget,
    system_prompt: str,
    response_handler: ResponseHandler,
    value: str,
    data_type: PromptDataType,
    scored_prompt_id: str | uuid.UUID,
    scorer_identifier: ComponentIdentifier,
    prepended_text: str | None = None,
    category: Sequence[str] | str | None = None,
    objective: str | None = None,
    response_json_schema: JsonSchemaDefinition | None = None,
) -> UnvalidatedScore:
    conversation_id = str(uuid.uuid4())

    chat_target.set_system_prompt(
        system_prompt=system_prompt,
        conversation_id=conversation_id,
    )
    prompt_metadata: dict[str, Any] = {"response_format": "json"}
    if response_json_schema is not None:
        # Always forward the schema; the target's normalization pipeline omits it
        # when the target cannot natively enforce a JSON schema.
        prompt_metadata[JSON_SCHEMA_METADATA_KEY] = response_json_schema

    # Build message pieces - prepended text context first (if provided), then the main message being scored
    message_pieces: list[MessagePiece] = []

    # Add prepended text context piece if provided (e.g., objective context for non-text scoring)
    if prepended_text:
        message_pieces.append(
            MessagePiece(
                role="user",
                original_value=prepended_text,
                original_value_data_type="text",
                converted_value_data_type="text",
                conversation_id=conversation_id,
                prompt_metadata=prompt_metadata,
            )
        )

    # Add the main message piece being scored
    message_pieces.append(
        MessagePiece(
            role="user",
            original_value=value,
            original_value_data_type=data_type,
            converted_value_data_type=data_type,
            conversation_id=conversation_id,
            prompt_metadata=prompt_metadata,
        )
    )

    scorer_llm_request = Message(message_pieces=message_pieces)
    try:
        response = await chat_target.send_prompt_async(message=scorer_llm_request)
    except Exception as ex:
        raise Exception(f"Error scoring prompt with original prompt ID: {scored_prompt_id}") from ex

    # Get the text piece which contains the JSON response containing the score_value and rationale from the LLM
    text_piece = next(piece for piece in response[0].message_pieces if piece.converted_value_data_type == "text")

    return response_handler.parse(
        response_text=text_piece.converted_value,
        scorer_identifier=scorer_identifier,
        scored_prompt_id=scored_prompt_id,
        category=category,
        objective=objective,
    )
