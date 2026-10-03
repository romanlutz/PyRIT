# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Solver-independent projection of typed Inspect log samples into PyRIT evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

from inspect_ai.event import ModelEvent, ScoreEvent, ToolEvent
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ContentText

from pyrit.models import MessagePiece
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberToolPhase,
)

if TYPE_CHECKING:
    import uuid

    from inspect_ai.event import Event
    from inspect_ai.log import EvalSample
    from inspect_ai.model import ChatMessage


class InspectProjectionVersion(IntEnum):
    """Immutable import schemas; the original text-only schema remains readable."""

    TEXT_ONLY = 2
    TOOL_CALLS = 3

    @property
    def binding_version(self) -> str:
        """The native evidence binding version for this projection schema."""
        return str(self.value - 1)

    @classmethod
    def from_binding_version(cls, binding_version: str) -> InspectProjectionVersion:
        """
        Resolve a sealed episode's schema without upgrading its projection.

        Returns:
            InspectProjectionVersion: The originally persisted message schema.

        Raises:
            ValueError: If the episode uses an unsupported projection binding.
        """
        for version in cls:
            if version.binding_version == binding_version:
                return version
        raise ValueError("Original Inspect episode has an unsupported projection binding.")


@dataclass(frozen=True, kw_only=True)
class InspectSampleProjection:
    """Observed Inspect events and a source-preserving projection of its messages."""

    events: tuple[NativeCyberCapturedEvent, ...]
    message_pieces: tuple[MessagePiece, ...]
    request_ids: tuple[uuid.UUID, ...]
    response_ids: tuple[uuid.UUID, ...]
    tool_request_ids: tuple[uuid.UUID, ...]
    tool_result_ids: tuple[uuid.UUID, ...]
    unprojected_messages: int
    original_event_count: int


def final_original_score_event(*, sample: EvalSample, scorer_name: str) -> ScoreEvent | None:
    """
    Match the acquired original Score to one final, source-identified ScoreEvent.

    Returns:
        ScoreEvent | None: The original event, if present and uniquely identified.

    Raises:
        ValueError: If the recorded original ScoreEvent contradicts the sample score.
    """
    score = (sample.scores or {}).get(scorer_name)
    if score is None:
        return None
    matches = [
        event
        for event in sample.events
        if isinstance(event, ScoreEvent) and event.scorer == scorer_name and not event.intermediate
    ]
    if len(matches) != 1 or not matches[0].uuid:
        return None
    if matches[0].score.model_dump(mode="json", exclude_none=True) != score.model_dump(mode="json", exclude_none=True):
        raise ValueError("Original Inspect ScoreEvent disagrees with its one sample Score.")
    return matches[0]


def project_inspect_sample(
    *,
    sample: EvalSample,
    log_run_id: str,
    eval_id: str,
    archive_sha256: str,
    sample_index: int,
    start_sequence: int,
    conversation_id: str,
    case_run_id: str | None = None,
    projection_version: InspectProjectionVersion = InspectProjectionVersion.TOOL_CALLS,
) -> InspectSampleProjection:
    """
    Project original attempts, text, and canonical tool calls without executing them.

    Returns:
        InspectSampleProjection: Typed events and persisted-message candidates.

    Raises:
        TypeError: If the projection schema is not an explicit supported version.
    """
    if not isinstance(projection_version, InspectProjectionVersion):
        raise TypeError("Original Inspect projection requires an InspectProjectionVersion.")
    events: list[NativeCyberCapturedEvent] = []
    for attempt, retry in enumerate(sample.error_retries or [], start=1):
        for event in retry.events or []:
            events.append(
                _capture_event(
                    event=event,
                    log_run_id=log_run_id,
                    sample=sample,
                    sample_index=sample_index,
                    attempt=attempt,
                    sequence=start_sequence + len(events),
                )
            )
    last_attempt = len(sample.error_retries or []) + 1
    for event in sample.events:
        events.append(
            _capture_event(
                event=event,
                log_run_id=log_run_id,
                sample=sample,
                sample_index=sample_index,
                attempt=last_attempt,
                sequence=start_sequence + len(events),
            )
        )
    original_count = len(events)
    events.append(
        NativeCyberCapturedEvent(
            source=NativeCyberEvidenceSource.HARNESS,
            event=NativeCyberObservedEvent(
                controller_sequence=start_sequence + len(events),
                source_session_id=log_run_id,
                event_type="inspect.projection.sample",
                payload={
                    "type": "inspect.projection.sample",
                    "eval_id": eval_id,
                    "sample_id": str(sample.id),
                    "sample_uuid": sample.uuid,
                    "sample_index": sample_index,
                    "epoch": sample.epoch,
                    "case_run_id": case_run_id,
                    "attempts": last_attempt,
                    "error": sample.error.model_dump(mode="json", exclude_none=True) if sample.error else None,
                    "scores": {
                        name: score.model_dump(mode="json", exclude_none=True)
                        for name, score in (sample.scores or {}).items()
                    },
                    "model_usage": {
                        name: usage.model_dump(mode="json", exclude_none=True)
                        for name, usage in sample.model_usage.items()
                    },
                    "turn_count": sample.turn_count,
                    **(
                        {"projection_schema": projection_version.value}
                        if projection_version is not InspectProjectionVersion.TEXT_ONLY
                        else {}
                    ),
                },
            ),
        )
    )
    pieces, unprojected = _project_messages(
        sample=sample,
        archive_sha256=archive_sha256,
        conversation_id=conversation_id,
        projection_version=projection_version,
    )
    return InspectSampleProjection(
        events=tuple(events),
        message_pieces=tuple(pieces),
        request_ids=tuple(piece.id for piece in pieces if piece.role in {"system", "developer", "user"}),
        response_ids=tuple(
            piece.id
            for piece in pieces
            if piece.role == "assistant" and piece.original_value_data_type != "function_call"
        ),
        tool_request_ids=tuple(piece.id for piece in pieces if piece.original_value_data_type == "function_call"),
        tool_result_ids=tuple(piece.id for piece in pieces if piece.role == "tool"),
        unprojected_messages=unprojected,
        original_event_count=original_count,
    )


def _capture_event(
    *, event: Event, log_run_id: str, sample: EvalSample, sample_index: int, attempt: int, sequence: int
) -> NativeCyberCapturedEvent:
    source = (
        NativeCyberEvidenceSource.TOOL
        if isinstance(event, ToolEvent)
        else NativeCyberEvidenceSource.MODEL
        if isinstance(event, ModelEvent)
        else NativeCyberEvidenceSource.HARNESS
    )
    call_id = event.id if isinstance(event, ToolEvent) else None
    return NativeCyberCapturedEvent(
        source=source,
        captured_at=event.timestamp,
        event=NativeCyberObservedEvent(
            controller_sequence=sequence,
            source_event_id=event.uuid,
            source_session_id=log_run_id,
            event_type=event.event,
            tool_call_id=call_id,
            tool_phase=NativeCyberToolPhase.COMPLETE
            if isinstance(event, ToolEvent) and event.completed is not None
            else None,
            payload={
                "type": event.event,
                "sample_id": str(sample.id),
                "sample_uuid": sample.uuid,
                "sample_index": sample_index,
                "epoch": sample.epoch,
                "attempt": attempt,
                "original": event.model_dump(mode="json", exclude_none=True),
            },
        ),
    )


def _project_messages(
    *,
    sample: EvalSample,
    archive_sha256: str,
    conversation_id: str,
    projection_version: InspectProjectionVersion,
) -> tuple[list[MessagePiece], int]:
    pieces: list[MessagePiece] = []
    unprojected = 0
    for position, message in enumerate(sample.messages):
        if projection_version is InspectProjectionVersion.TOOL_CALLS:
            projected, incomplete = _project_message(
                sample=sample,
                message=message,
                position=position,
                archive_sha256=archive_sha256,
                conversation_id=conversation_id,
            )
            pieces.extend(projected)
            unprojected += int(incomplete)
            continue
        texts = _message_text_parts(message=message)
        if texts is None:
            unprojected += 1
            continue
        for part_index, text in enumerate(texts):
            pieces.append(
                MessagePiece(
                    role=message.role,
                    original_value=text,
                    original_value_data_type="function_call_output" if isinstance(message, ChatMessageTool) else "text",
                    conversation_id=conversation_id,
                    sequence=position,
                    prompt_metadata={
                        "inspect_archive_sha256": archive_sha256,
                        "inspect_sample_id": str(sample.id),
                        "inspect_sample_uuid": sample.uuid,
                        "inspect_epoch": sample.epoch,
                        "inspect_message_id": message.id,
                        "inspect_message_source": message.source,
                        "inspect_part_index": part_index,
                        "inspect_tool_call_id": message.tool_call_id if isinstance(message, ChatMessageTool) else None,
                    },
                )
            )
    return pieces, unprojected


def _project_message(
    *,
    sample: EvalSample,
    message: ChatMessage,
    position: int,
    archive_sha256: str,
    conversation_id: str,
) -> tuple[list[MessagePiece], bool]:
    pieces: list[MessagePiece] = []
    if isinstance(message, ChatMessageTool) and not message.tool_call_id:
        return [], True
    content = message.content
    parts = (
        [(0, content)]
        if isinstance(content, str)
        else [(index, part.text) for index, part in enumerate(content) if isinstance(part, ContentText)]
    )
    incomplete = not isinstance(content, str) and (
        (not content and not (isinstance(message, ChatMessageAssistant) and message.tool_calls))
        or any(not isinstance(part, ContentText) for part in content)
    )
    metadata = {
        "inspect_archive_sha256": archive_sha256,
        "inspect_sample_id": str(sample.id),
        "inspect_sample_uuid": sample.uuid,
        "inspect_epoch": sample.epoch,
        "inspect_message_id": message.id,
        "inspect_message_source": message.source,
    }
    for part_index, text in parts:
        if not text and isinstance(message, ChatMessageAssistant) and message.tool_calls:
            continue
        tool = message if isinstance(message, ChatMessageTool) else None
        linked_tool = tool is not None and tool.tool_call_id is not None
        value = (
            json.dumps(
                {"type": "function_call_output", "call_id": tool.tool_call_id, "output": text},
                separators=(",", ":"),
            )
            if linked_tool
            else text
        )
        pieces.append(
            MessagePiece(
                role=message.role,
                original_value=value,
                original_value_data_type="function_call_output" if linked_tool else "text",
                conversation_id=conversation_id,
                sequence=position,
                response_error="unknown" if tool is not None and tool.error else "none",
                prompt_metadata={
                    **metadata,
                    "inspect_part_index": part_index,
                    "inspect_tool_call_id": tool.tool_call_id if tool is not None else None,
                    **(
                        {
                            "inspect_tool_function": tool.function,
                            "inspect_tool_error_type": tool.error.type if tool.error else None,
                            "inspect_tool_error_message": tool.error.message if tool.error else None,
                        }
                        if tool is not None
                        else {}
                    ),
                },
            )
        )
    if isinstance(message, ChatMessageAssistant):
        offset = 1 if isinstance(content, str) else len(content)
        for call_index, call in enumerate(message.tool_calls or []):
            if call.type != "function" or not call.id:
                incomplete = True
                continue
            pieces.append(
                MessagePiece(
                    role="assistant",
                    original_value=json.dumps(
                        {
                            "type": "function_call",
                            "call_id": call.id,
                            "name": call.function,
                            "arguments": json.dumps(call.arguments, separators=(",", ":")),
                        },
                        separators=(",", ":"),
                    ),
                    original_value_data_type="function_call",
                    conversation_id=conversation_id,
                    sequence=position,
                    prompt_metadata={
                        **metadata,
                        "inspect_part_index": offset + call_index,
                        "inspect_tool_call_id": call.id,
                        "inspect_tool_parse_error": call.parse_error,
                    },
                )
            )
    return pieces, incomplete


def _message_text_parts(*, message: ChatMessage) -> tuple[str, ...] | None:
    content = message.content
    if isinstance(content, str):
        if not content and isinstance(message, ChatMessageAssistant) and message.tool_calls:
            return None
        return (content,)
    if not content or any(not isinstance(part, ContentText) for part in content):
        return None
    return tuple(part.text for part in content if isinstance(part, ContentText))
