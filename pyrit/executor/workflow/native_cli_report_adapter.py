# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pure projection of CLI target observations into caller-owned run evidence."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from pyrit.models.native_cli_report import (
    NativeCliArtifactReference,
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportEvent,
    NativeCliReportEventKind,
    NativeCliReportEventStatus,
    NativeCliReportEvidence,
    NativeCliReportProtocol,
    NativeCliReportStatus,
    NativeCliRunReport,
)
from pyrit.prompt_target.native_cli_models import NativeCliEvent

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from pyrit.prompt_target.native_cli_models import NativeCliRunConfig, NativeCliRunOutcome


def build_native_cli_run_report(
    *,
    config: NativeCliRunConfig,
    outcome: NativeCliRunOutcome | None,
    events: Iterable[NativeCliEvent | NativeCliReportEvent],
    task_id: str,
    task_version: str,
    run_id: str,
    turn_id: str,
    status: NativeCliReportStatus,
    cleanup: NativeCliReportCleanup,
    turn_index: int = 1,
    parent_run_id: str | None = None,
    conversation_id: str | None = None,
    simulated: bool | None = None,
    judgment: NativeCliOriginalJudgment | None = None,
    artifacts: Sequence[NativeCliArtifactReference] = (),
    raw_evidence_ref: str | None = None,
    errors: Sequence[str] = (),
) -> NativeCliRunReport:
    """
    Snapshot the actual CLI process outcome and ordered events without acquiring evidence.

    Args:
        config (NativeCliRunConfig): Pinned caller-owned CLI profile and step budget.
        outcome (NativeCliRunOutcome | None): Actual process outcome, or None if interrupted.
        events (Iterable[NativeCliEvent | NativeCliReportEvent]): Recorded raw events or persisted
            summaries with actual frame ordinals, digests, sizes, and stdout offsets.
        task_id (str): Caller-assigned task identity.
        task_version (str): Caller-owned version of the original task definition.
        run_id (str): Caller-assigned run identity.
        turn_id (str): Caller-assigned turn identity, not a provider event ID.
        status (NativeCliReportStatus): Caller-owned finalization status.
        cleanup (NativeCliReportCleanup): Caller-observed sandbox cleanup outcome.
        turn_index (int): Caller-assigned turn number. Defaults to 1.
        parent_run_id (str | None): Prior run, if continuing at a separate lease.
        conversation_id (str | None): PyRIT conversation, if one was opened.
        simulated (bool | None): Authoritatively known provenance, not inferred from CLI.
        judgment (NativeCliOriginalJudgment | None): Already acquired original grader judgment.
        artifacts (Sequence[NativeCliArtifactReference]): Caller-retained artifact references.
        raw_evidence_ref (str | None): Caller-owned raw-chunk evidence reference.
        errors (Sequence[str]): Caller-observed lifecycle failures.

    Returns:
        NativeCliRunReport: Canonical v1 CLI-only report, not a verified DB result.

    Raises:
        ValueError: If observations contradict their process outcome or final status.
    """
    evidence = _project_evidence(outcome=outcome, events=events, raw_evidence_ref=raw_evidence_ref)
    return NativeCliRunReport(
        schema_version=1,
        task_id=task_id,
        task_version=task_version,
        run_id=run_id,
        turn_id=turn_id,
        turn_index=turn_index,
        parent_run_id=parent_run_id,
        conversation_id=conversation_id,
        protocol=NativeCliReportProtocol(config.protocol.value),
        cli_version=config.cli_version,
        cli_profile=config.cli_profile,
        max_steps=config.max_steps,
        simulated=simulated,
        status=status,
        evidence=evidence,
        judgment=judgment,
        artifacts=tuple(artifacts),
        cleanup=cleanup,
        errors=tuple(errors),
    )


def _project_evidence(
    *,
    outcome: NativeCliRunOutcome | None,
    events: Iterable[NativeCliEvent | NativeCliReportEvent],
    raw_evidence_ref: str | None,
) -> NativeCliReportEvidence:
    summaries = _project_events(events=events)
    observed_ids = {item.source_session_id for item in summaries if item.source_session_id}
    source_session_id = (
        outcome.source_session_id
        if outcome is not None and outcome.source_session_id
        else next(iter(observed_ids))
        if len(observed_ids) == 1
        else None
    )
    return NativeCliReportEvidence(
        source_session_id=source_session_id,
        exit_code=outcome.exit_code if outcome is not None else None,
        terminal_observed=outcome.terminal_observed if outcome is not None else False,
        coverage_complete=outcome.coverage_complete if outcome is not None else False,
        observed_steps=outcome.observed_steps if outcome is not None else None,
        frame_count=outcome.frame_count if outcome is not None else None,
        raw_chunk_count=outcome.raw_chunk_count if outcome is not None else None,
        raw_stdout_bytes=outcome.raw_stdout_bytes if outcome is not None else None,
        raw_stderr_bytes=outcome.raw_stderr_bytes if outcome is not None else None,
        gaps=outcome.gaps if outcome is not None else ("No native CLI process outcome was acquired.",),
        events=summaries,
        raw_evidence_ref=raw_evidence_ref,
    )


def _project_events(*, events: Iterable[NativeCliEvent | NativeCliReportEvent]) -> tuple[NativeCliReportEvent, ...]:
    summaries: list[NativeCliReportEvent] = []
    last_frame = 0
    last_metadata: tuple[str | None, int | None, int | None] | None = None
    next_offset = 0
    for event in events:
        if isinstance(event, NativeCliReportEvent):
            summary = NativeCliReportEvent.model_validate(event.model_dump(mode="json"))
        elif isinstance(event, NativeCliEvent):
            frame_number = event.frame_number
            if frame_number is not None and frame_number not in {last_frame, last_frame + 1}:
                raise ValueError("Cannot infer a missing stdout frame offset; supply its persisted event summary.")
            offset = (
                last_metadata[2]
                if frame_number is not None and frame_number == last_frame and last_metadata is not None
                else next_offset
                if frame_number is not None
                else None
            )
            summary = _project_event(event=event, stdout_offset_bytes=offset)
        else:
            raise TypeError("CLI report events must be recorded native events or validated event summaries.")
        if summary.sequence != len(summaries) + 1:
            raise ValueError("CLI report event ordinals must be contiguous and preserve recorded order.")
        if summary.frame_number is not None:
            number = summary.frame_number
            metadata = (summary.raw_frame_sha256, summary.raw_frame_size_bytes, summary.stdout_offset_bytes)
            if number == last_frame and last_metadata != metadata:
                raise ValueError("CLI frame observations must agree on digest, size, offset, and order.")
            if number != last_frame:
                if number < last_frame:
                    raise ValueError("CLI stdout frames cannot be reported out of order.")
                last_frame = number
                last_metadata = metadata
                if summary.stdout_offset_bytes is None or summary.raw_frame_size_bytes is None:
                    raise ValueError("A CLI frame summary requires an actual byte offset and length.")
                next_offset = summary.stdout_offset_bytes + summary.raw_frame_size_bytes
        summaries.append(summary)
    return tuple(summaries)


def _project_event(*, event: NativeCliEvent, stdout_offset_bytes: int | None) -> NativeCliReportEvent:
    observation = event.observation
    return NativeCliReportEvent(
        sequence=event.sequence,
        frame_number=event.frame_number,
        kind=NativeCliReportEventKind(observation.kind.value),
        status=NativeCliReportEventStatus(observation.status.value),
        source_event_id=observation.source_event_id,
        source_message_id=observation.source_message_id,
        source_session_id=observation.source_session_id,
        source_tool_id=observation.source_tool_id,
        parent_tool_use_id=observation.parent_tool_use_id,
        source_status=observation.source_status,
        name=observation.name,
        exit_code=observation.exit_code,
        detail=observation.detail,
        raw_frame_sha256=hashlib.sha256(event.raw_frame).hexdigest() if event.raw_frame is not None else None,
        raw_frame_size_bytes=len(event.raw_frame) if event.raw_frame is not None else None,
        stdout_offset_bytes=stdout_offset_bytes,
    )
