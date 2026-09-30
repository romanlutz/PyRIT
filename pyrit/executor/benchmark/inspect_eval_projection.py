# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Solver-independent projection of typed Inspect log samples into PyRIT evidence."""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True, kw_only=True)
class InspectSampleProjection:
    """Observed Inspect events and a text-only projection of its original messages."""

    events: tuple[NativeCyberCapturedEvent, ...]
    message_pieces: tuple[MessagePiece, ...]
    request_ids: tuple[uuid.UUID, ...]
    response_ids: tuple[uuid.UUID, ...]
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
) -> InspectSampleProjection:
    """
    Project the original attempt/event order and only lossless text messages.

    Returns:
        InspectSampleProjection: Typed events and persisted-message candidates.
    """
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
                },
            ),
        )
    )
    pieces, unprojected = _project_messages(
        sample=sample, archive_sha256=archive_sha256, conversation_id=conversation_id
    )
    return InspectSampleProjection(
        events=tuple(events),
        message_pieces=tuple(pieces),
        request_ids=tuple(piece.id for piece in pieces if piece.role in {"system", "developer", "user"}),
        response_ids=tuple(piece.id for piece in pieces if piece.role == "assistant"),
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
    *, sample: EvalSample, archive_sha256: str, conversation_id: str
) -> tuple[list[MessagePiece], int]:
    pieces: list[MessagePiece] = []
    unprojected = 0
    for position, message in enumerate(sample.messages):
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


def _message_text_parts(*, message: ChatMessage) -> tuple[str, ...] | None:
    content = message.content
    if isinstance(content, str):
        if not content and isinstance(message, ChatMessageAssistant) and message.tool_calls:
            return None
        return (content,)
    if not content or any(not isinstance(part, ContentText) for part in content):
        return None
    return tuple(part.text for part in content if isinstance(part, ContentText))
