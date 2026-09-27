# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Synthetic native evidence capture and existing-score linkage."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event
from sqlalchemy.exc import IntegrityError, OperationalError

from pyrit.memory.memory_models import (
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberTurnMessagePieceEntry,
    PromptMemoryEntry,
    ScorableContentEntry,
    ScoreEntry,
)
from pyrit.models import ContentEntryScorable, ContentScorable, MessagePiece, Score, ScoreStatus
from pyrit.models.native_cyber import (
    NativeAgentCapabilities,
    NativeAgentEvent,
    NativeAgentEvidence,
    NativeCyberCleanup,
    NativeCyberJudgment,
    NativeCyberReadiness,
    NativeCyberReport,
    NativeCyberRequest,
    NativeCyberStatus,
    NativeToolRequest,
    NativeToolTrace,
)
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberEpisodeStart,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberToolPhase,
    NativeCyberTurnFinish,
    NativeCyberTurnStart,
)
from pyrit.score.float_scale.native_cyber_scorer import NativeCyberReportScorer

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface
    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore


@dataclass(frozen=True, kw_only=True)
class _Case:
    memory: MemoryInterface
    store: NativeCyberEvidenceStore
    report: NativeCyberReport
    required_stream: NativeCyberRawStreamKey
    request_piece_id: UUID
    response_piece_id: UUID


def _events() -> tuple[NativeAgentEvent, ...]:
    data = [
        ("assistant.message", {"toolRequests": [{"toolCallId": "call-1", "name": "read", "arguments": {"x": 1}}]}),
        ("tool.execution_start", {"toolCallId": "call-1", "toolName": "read", "arguments": {"x": 1}}),
        (
            "tool.execution_complete",
            {"toolCallId": "call-1", "success": True, "result": {"content": "synthetic tool result"}},
        ),
        ("session.idle", {"aborted": False}),
    ]
    return tuple(
        NativeAgentEvent(
            sequence=index,
            event_id=f"source-event-{index}",
            session_id="native-session-1",
            event_type=event_type,
            payload={"id": f"source-event-{index}", "type": event_type, "sessionId": "native-session-1", "data": body},
        )
        for index, (event_type, body) in enumerate(data, start=1)
    )


def _report(*, run_id: str, started_at: datetime) -> NativeCyberReport:
    events = _events()
    return NativeCyberReport(
        run_id=run_id,
        binding_name="synthetic-binding",
        binding_version="1",
        request=NativeCyberRequest(instruction="synthetic instruction"),
        input_sha256=hashlib.sha256(b"synthetic instruction").hexdigest(),
        status=NativeCyberStatus.COMPLETED,
        simulated=True,
        readiness=NativeCyberReadiness(
            ready=True,
            simulated=True,
            capabilities=NativeAgentCapabilities(),
        ),
        started_at=started_at,
        expires_at=started_at + timedelta(minutes=1),
        ended_at=started_at + timedelta(seconds=1),
        conversation_id="native-conversation-1",
        agent=NativeAgentEvidence(
            session_id="native-session-1",
            environment_id="synthetic-environment-1",
            simulated=True,
            events=events,
            tools=(
                NativeToolTrace(
                    call_id="call-1",
                    name="read",
                    arguments={"x": 1},
                    request_sequence=1,
                    start_sequence=2,
                    completion_sequence=3,
                    success=True,
                    result={"content": "synthetic tool result"},
                    status="succeeded",
                    model_visible_output="synthetic tool result",
                ),
            ),
            tool_requests=(NativeToolRequest(call_id="call-1", name="read", arguments={"x": 1}, request_sequence=1),),
            idle=True,
            coverage_complete=True,
            gaps=(),
        ),
        judgment=NativeCyberJudgment(value=0.75, rationale="synthetic original grade", complete=True),
        cleanup=NativeCyberCleanup.CLOSED,
    )


def _prepare_case(
    *,
    memory: MemoryInterface,
    raw_byte_limit: int = 268_435_456,
    require_stream: bool = True,
) -> _Case:
    started_at = datetime.now(UTC)
    report = _report(run_id=str(uuid4()), started_at=started_at)
    request = MessagePiece(
        role="user",
        original_value="synthetic instruction",
        conversation_id=report.conversation_id,
        sequence=0,
    )
    response = MessagePiece(
        role="assistant",
        original_value="synthetic model answer",
        conversation_id=report.conversation_id,
        sequence=1,
    )
    memory.add_message_pieces_to_memory(message_pieces=[request, response])
    required_stream = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="source-jsonl-1",
    )
    store = memory.native_cyber_evidence
    store.create_episode(
        start=NativeCyberEpisodeStart(
            run_id=report.run_id,
            binding_name=report.binding_name,
            binding_version=report.binding_version,
            started_at=started_at,
            source_session_id="native-session-1",
            environment_id="synthetic-environment-1",
            simulated=True,
            required_raw_streams=(required_stream,) if require_stream else (),
            raw_byte_limit=raw_byte_limit,
        )
    )
    store.begin_turn(
        turn=NativeCyberTurnStart(
            run_id=report.run_id,
            turn_index=1,
            started_at=started_at,
            request_piece_ids=(request.id,),
        )
    )
    return _Case(
        memory=memory,
        store=store,
        report=report,
        required_stream=required_stream,
        request_piece_id=request.id,
        response_piece_id=response.id,
    )


def _append_events(*, case: _Case, count: int = 4) -> None:
    assert case.report.agent is not None
    events = case.report.agent.events[:count]
    case.store.append_events(
        run_id=case.report.run_id,
        turn_index=1,
        events=[
            NativeCyberCapturedEvent.from_native_agent_event(
                source=(
                    NativeCyberEvidenceSource.MODEL
                    if event.event_type.startswith("assistant.")
                    else NativeCyberEvidenceSource.TOOL
                    if event.event_type.startswith("tool.")
                    else NativeCyberEvidenceSource.HARNESS
                ),
                event=event,
            )
            for event in events
        ],
    )


def _finish_turn(*, case: _Case, observed_event_count: int = 4) -> None:
    case.store.finish_turn(
        finish=NativeCyberTurnFinish(
            run_id=case.report.run_id,
            turn_index=1,
            finished_at=case.report.ended_at,
            response_piece_ids=(case.response_piece_id,),
            observed_event_count=observed_event_count,
            source_complete=True,
        )
    )


def _close_stream(
    *,
    case: _Case,
    key: NativeCyberRawStreamKey,
    data: bytes,
    turn_index: int | None = None,
) -> UUID:
    stream = NativeCyberRawStreamStart(run_id=case.report.run_id, key=key, turn_index=turn_index)
    case.store.open_raw_stream(stream=stream)
    if data:
        for offset in range(0, len(data), case.store.MAX_APPEND_BYTES):
            written = case.store.append_raw(
                run_id=case.report.run_id,
                stream_id=stream.stream_id,
                data=data[offset : offset + case.store.MAX_APPEND_BYTES],
            )
            assert written.received_bytes == len(data[offset : offset + case.store.MAX_APPEND_BYTES])
    case.store.close_raw_stream(
        run_id=case.report.run_id,
        stream_id=stream.stream_id,
        source_complete=True,
        expected_bytes=len(data),
        observed_sha256=hashlib.sha256(data).hexdigest(),
    )
    return stream.stream_id


async def _persist_score_async(*, report: NativeCyberReport) -> Score:
    scores = await NativeCyberReportScorer(report_sha256=report.sha256()).score_async(
        scorable=ContentScorable(value=report.canonical_json())
    )
    assert len(scores) == 1
    return scores[0]


def _unpersisted_score(*, report: NativeCyberReport) -> Score:
    value = report.judgment.value if report.status is NativeCyberStatus.COMPLETED and report.judgment else None
    return Score(
        score_type="float_scale",
        score_value=str(value) if value is not None else None,
        status=ScoreStatus.COMPLETE if value is not None else ScoreStatus.UNDETERMINED,
        score_rationale="synthetic original native judgment",
        scorer_class_identifier=NativeCyberReportScorer(report_sha256=report.sha256()).get_identifier(),
        scorable=ContentScorable(value=report.canonical_json()),
        score_metadata={"run_id": report.run_id, "report_sha256": report.sha256()},
    )


@pytest.mark.usefixtures("patch_central_database")
class TestNativeCyberEvidence:
    def test_repeated_observed_stream_id_retains_distinct_raw_segments(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        stream_ids = [
            _close_stream(case=case, key=case.required_stream, data=payload, turn_index=1)
            for payload in (b"first segment", b"second segment")
        ]
        _finish_turn(case=case)
        snapshot = case.store.finalize_episode_atomic(
            report=case.report,
            score=_unpersisted_score(report=case.report),
            expected_turns=1,
        )

        assert snapshot.score_status is ScoreStatus.COMPLETE
        assert len(snapshot.raw_streams) == 2
        assert [stream.key.observed_source_id for stream in snapshot.raw_streams] == ["source-jsonl-1"] * 2
        assert snapshot.stored_raw_bytes == len(b"first segment") + len(b"second segment")
        assert [
            b"".join(
                chunk.data
                for chunk in case.store.read_raw_chunks(
                    run_id=case.report.run_id,
                    stream_id=stream_id,
                    allow_sensitive=True,
                )
            )
            for stream_id in stream_ids
        ] == [b"first segment", b"second segment"]

    @pytest.mark.parametrize(
        ("source", "event_types", "observed_id", "tool_phases"),
        [
            (
                NativeCyberEvidenceSource.TOOL,
                ("item.started", "item.updated", "item.completed"),
                "observed-item-1",
                (NativeCyberToolPhase.START, None, NativeCyberToolPhase.COMPLETE),
            ),
            (
                NativeCyberEvidenceSource.MODEL,
                ("assistant.block_started", "assistant.block_delta"),
                "claude-message-uuid",
                (None, None),
            ),
        ],
    )
    def test_repeated_real_source_ids_are_distinct_controller_events(
        self,
        *,
        sqlite_instance: MemoryInterface,
        source: NativeCyberEvidenceSource,
        event_types: tuple[str, ...],
        observed_id: str,
        tool_phases: tuple[NativeCyberToolPhase | None, ...],
    ) -> None:
        run_id = str(uuid4())
        store = sqlite_instance.native_cyber_evidence
        store.create_episode(
            start=NativeCyberEpisodeStart(run_id=run_id, binding_name="synthetic-cli", binding_version="1")
        )
        store.begin_turn(turn=NativeCyberTurnStart(run_id=run_id, turn_index=1))
        store.append_events(
            run_id=run_id,
            turn_index=1,
            events=[
                NativeCyberCapturedEvent(
                    source=source,
                    event=NativeCyberObservedEvent(
                        controller_sequence=index,
                        source_event_id=observed_id,
                        event_type=event_type,
                        payload={"type": event_type},
                        tool_call_id=observed_id if source is NativeCyberEvidenceSource.TOOL else None,
                        tool_phase=phase,
                    ),
                )
                for index, (event_type, phase) in enumerate(zip(event_types, tool_phases, strict=True), start=1)
            ],
        )

        snapshot = store.get_episode(run_id=run_id)
        assert [(event.sequence, event.observed_event_id) for event in snapshot.events] == [
            (index, observed_id) for index in range(1, len(event_types) + 1)
        ]
        assert [
            captured.event.source_event_id
            for captured in store.read_event_payloads(
                run_id=run_id,
                allow_sensitive=True,
            )
        ] == [observed_id] * len(event_types)
        if source is NativeCyberEvidenceSource.TOOL:
            assert [event.tool_call_id for event in snapshot.events] == [observed_id] * len(event_types)
            assert snapshot.tools[0].start_sequence == 1
            assert snapshot.tools[0].completion_sequence == 3
        else:
            assert snapshot.tools == ()

    def test_cli_frames_without_provider_event_ids_keep_controller_order_and_offsets(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        run_id = str(uuid4())
        store = sqlite_instance.native_cyber_evidence
        source_key = NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.HARNESS,
            kind=NativeCyberRawKind.JSONL,
            observed_source_id="codex-jsonl",
        )
        store.create_episode(
            start=NativeCyberEpisodeStart(
                run_id=run_id,
                binding_name="synthetic-cli",
                binding_version="1",
                required_raw_streams=(source_key,),
            )
        )
        store.begin_turn(turn=NativeCyberTurnStart(run_id=run_id, turn_index=1))
        stream = NativeCyberRawStreamStart(run_id=run_id, key=source_key, turn_index=1)
        store.open_raw_stream(stream=stream)
        raw = (
            b'{"type":"turn.started"}\n'
            b'{"type":"item.started","item":{"id":"observed-item-1"}}\n'
            b'{"type":"turn.completed"}\n'
        )
        store.append_raw(run_id=run_id, stream_id=stream.stream_id, data=raw)
        store.append_events(
            run_id=run_id,
            turn_index=1,
            events=[
                NativeCyberCapturedEvent(
                    source=NativeCyberEvidenceSource.HARNESS,
                    event=NativeCyberObservedEvent(
                        controller_sequence=1,
                        event_type="turn.started",
                        payload={"type": "turn.started"},
                        observed_stream_id="codex-jsonl",
                        stream_offset=0,
                    ),
                ),
                NativeCyberCapturedEvent(
                    source=NativeCyberEvidenceSource.TOOL,
                    event=NativeCyberObservedEvent(
                        controller_sequence=2,
                        source_event_id="observed-item-1",
                        event_type="item.started",
                        payload={"type": "item.started", "item": {"id": "observed-item-1"}},
                        observed_stream_id="codex-jsonl",
                        stream_offset=24,
                        tool_call_id="observed-item-1",
                        tool_phase=NativeCyberToolPhase.START,
                    ),
                ),
                NativeCyberCapturedEvent(
                    source=NativeCyberEvidenceSource.HARNESS,
                    event=NativeCyberObservedEvent(
                        controller_sequence=3,
                        event_type="turn.completed",
                        payload={"type": "turn.completed"},
                        observed_stream_id="codex-jsonl",
                        stream_offset=79,
                    ),
                ),
            ],
        )
        store.close_raw_stream(
            run_id=run_id,
            stream_id=stream.stream_id,
            source_complete=True,
            expected_bytes=len(raw),
            observed_sha256=hashlib.sha256(raw).hexdigest(),
        )

        snapshot = store.get_episode(run_id=run_id)
        assert snapshot.run.source_session_id is None
        assert [(event.sequence, event.observed_event_id, event.stream_offset) for event in snapshot.events] == [
            (1, None, 0),
            (2, "observed-item-1", 24),
            (3, None, 79),
        ]
        assert snapshot.tools[0].call_id == "observed-item-1"
        assert snapshot.tools[0].start_sequence == 2
        assert snapshot.tools[0].request_sequence is None
        captured = store.read_event_payloads(run_id=run_id, allow_sensitive=True)
        assert [entry.event.source_event_id for entry in captured] == [None, "observed-item-1", None]
        assert captured[1].event.observed_stream_id == "codex-jsonl"
        assert captured[1].event.stream_offset == 24
        assert captured[1].event.tool_phase is NativeCyberToolPhase.START
        assert [
            entry.observed_event_id
            for entry in sorted(
                sqlite_instance._query_entries(NativeCyberEventEntry),
                key=lambda entry: entry.sequence,
            )
        ] == [None, "observed-item-1", None]
        with pytest.raises(ValueError, match="offset requires its observed stream ID"):
            NativeCyberObservedEvent(
                controller_sequence=4,
                event_type="turn.completed",
                payload={"type": "turn.completed"},
                stream_offset=100,
            )

    def test_atomic_finalizer_retains_blocked_run_without_fake_events(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        started_at = datetime.now(UTC)
        report = NativeCyberReport(
            run_id=str(uuid4()),
            binding_name="synthetic-binding",
            binding_version="1",
            request=NativeCyberRequest(instruction="synthetic instruction"),
            input_sha256=hashlib.sha256(b"synthetic instruction").hexdigest(),
            status=NativeCyberStatus.BLOCKED,
            simulated=True,
            readiness=NativeCyberReadiness(
                ready=False,
                blockers=("synthetic policy",),
                simulated=True,
            ),
            started_at=started_at,
            expires_at=started_at + timedelta(minutes=1),
            ended_at=started_at + timedelta(seconds=1),
            cleanup=NativeCyberCleanup.NOT_OPENED,
        )
        store = sqlite_instance.native_cyber_evidence
        store.create_episode(
            start=NativeCyberEpisodeStart(
                run_id=report.run_id,
                binding_name=report.binding_name,
                binding_version=report.binding_version,
                started_at=started_at,
                simulated=True,
            )
        )
        score = _unpersisted_score(report=report)

        snapshot = store.finalize_episode_atomic(report=report, score=score, expected_turns=0)

        assert snapshot == store.get_finalized_episode(run_id=report.run_id)
        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert not snapshot.coverage_complete
        assert snapshot.turns == ()
        assert snapshot.events == ()
        assert snapshot.raw_streams == ()
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1

    def test_atomic_finalizer_commits_one_score_report_and_episode(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"synthetic raw", turn_index=1)
        _finish_turn(case=case)
        score = _unpersisted_score(report=case.report)
        assert sqlite_instance.get_scores(score_ids=[str(score.id)]) == []

        snapshot = case.store.finalize_episode_atomic(report=case.report, score=score, expected_turns=1)

        assert snapshot == case.store.get_finalized_episode(run_id=case.report.run_id)
        assert snapshot.score_id == score.id
        assert snapshot.score_status is ScoreStatus.COMPLETE
        assert snapshot.coverage_complete
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1
        stored_score = sqlite_instance.get_scores(score_ids=[str(score.id)])[0]
        assert isinstance(stored_score.scorable, ContentEntryScorable)
        assert stored_score.scorable.content_id == snapshot.report_content_id
        assert (
            sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
                snapshot.report_content_id
            ].value
            == case.report.canonical_json()
        )

    def test_atomic_finalizer_only_persists_undetermined_when_required_raw_is_missing(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _finish_turn(case=case)
        score = _unpersisted_score(report=case.report)

        snapshot = case.store.finalize_episode_atomic(report=case.report, score=score, expected_turns=1)

        assert score.status is ScoreStatus.COMPLETE
        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].is_undetermined
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert (
            NativeCyberReport.model_validate_json(
                sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
                    snapshot.report_content_id
                ].value
            ).judgment.value
            == 0.75
        )

    def test_atomic_finalizer_rolls_back_report_and_score_after_link_failure(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        _finish_turn(case=case)
        score = _unpersisted_score(report=case.report)
        updates = 0

        def _fail_first_episode_update(*args: object) -> None:
            nonlocal updates
            if "UPDATE" in str(args[2]) and "NativeCyberEpisodeEntries" in str(args[2]):
                updates += 1
                if updates == 1:
                    raise OperationalError("UPDATE", {}, Exception("synthetic atomic link failure"))

        event.listen(sqlite_instance.engine, "after_cursor_execute", _fail_first_episode_update)
        try:
            with pytest.raises(OperationalError, match="synthetic atomic link failure"):
                case.store.finalize_episode_atomic(report=case.report, score=score, expected_turns=1)
        finally:
            event.remove(sqlite_instance.engine, "after_cursor_execute", _fail_first_episode_update)

        assert updates >= 1
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert sqlite_instance._query_entries(ScorableContentEntry) == []
        pending = case.store.get_episode(run_id=case.report.run_id)
        assert pending.score_id is None
        assert pending.report_content_id is None
        assert pending.finalized_at is None
        assert any("database write failed" in gap for gap in pending.gaps)
        with pytest.raises(ValueError, match="not finalized"):
            case.store.get_finalized_episode(run_id=case.report.run_id)

        recovered = case.store.finalize_episode_atomic(report=case.report, score=score, expected_turns=1)
        assert recovered.score_id == score.id
        assert recovered.score_status is ScoreStatus.UNDETERMINED
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1

    async def test_atomic_finalizer_rejects_an_existing_score_id_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        _finish_turn(case=case)
        existing = await _persist_score_async(report=case.report)
        candidate = _unpersisted_score(report=case.report).model_copy(update={"id": existing.id})

        with pytest.raises(ValueError, match="unpersisted Score ID"):
            case.store.finalize_episode_atomic(report=case.report, score=candidate, expected_turns=1)

        assert case.store.get_episode(run_id=case.report.run_id).finalized_at is None
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1

    async def test_round_trip_links_one_real_score_and_chunked_raw_bytes_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        payload = b"synthetic raw \x00 bytes" + b"x" * (case.store.MAX_CHUNK_BYTES + 2)
        stream_id = _close_stream(case=case, key=case.required_stream, data=payload, turn_index=1)
        _finish_turn(case=case)
        coverage = case.store.assess_required_coverage(report=case.report, expected_turns=1)
        assert coverage.required_complete
        assert coverage.required_gaps == ()

        score = await _persist_score_async(report=case.report)
        assert isinstance(score.scorable, ContentEntryScorable)
        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        assert snapshot == case.store.get_finalized_episode(run_id=case.report.run_id)
        assert snapshot.score_id == score.id
        assert snapshot.score_status is ScoreStatus.COMPLETE
        assert snapshot.coverage_complete
        assert snapshot.turns[0].request_piece_ids == (case.request_piece_id,)
        assert snapshot.turns[0].response_piece_ids == (case.response_piece_id,)
        assert [(event.sequence, event.observed_event_id) for event in snapshot.events] == [
            (index, f"source-event-{index}") for index in range(1, 5)
        ]
        assert len(snapshot.tools) == 1
        assert (
            snapshot.tools[0].request_sequence,
            snapshot.tools[0].start_sequence,
            snapshot.tools[0].completion_sequence,
        ) == (1, 2, 3)
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1
        assert (
            sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
                snapshot.report_content_id
            ].value
            == case.report.canonical_json()
        )
        assert snapshot.report_sha256 == case.report.sha256()
        assert "synthetic tool result" not in snapshot.model_dump_json()
        assert "synthetic raw" not in snapshot.model_dump_json()
        with pytest.raises(PermissionError, match="authorized internal read"):
            case.store.read_event_payloads(run_id=case.report.run_id)
        with pytest.raises(PermissionError, match="authorized internal read"):
            case.store.read_raw_chunks(run_id=case.report.run_id, stream_id=stream_id)
        assert [
            (
                captured.event.controller_sequence,
                captured.event.source_event_id,
                captured.event.source_session_id,
                captured.event.event_type,
                captured.event.payload,
            )
            for captured in case.store.read_event_payloads(run_id=case.report.run_id, allow_sensitive=True)
        ] == [
            (event.sequence, event.event_id, event.session_id, event.event_type, event.payload)
            for event in case.report.agent.events
        ]
        chunks = case.store.read_raw_chunks(run_id=case.report.run_id, stream_id=stream_id, allow_sensitive=True)
        assert len(chunks) == 2
        assert all(chunk.length <= case.store.MAX_CHUNK_BYTES for chunk in chunks)
        assert b"".join(chunk.data for chunk in chunks) == payload

    async def test_linked_messages_report_and_score_cannot_change_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)
        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)
        links = sqlite_instance._query_entries(NativeCyberTurnMessagePieceEntry)
        assert len(links) == 2
        assert all(len(link.piece_sha256) == 64 for link in links)

        with pytest.raises(ValueError, match="native cyber evidence"):
            sqlite_instance.update_prompt_entries_by_conversation_id(
                conversation_id=case.report.conversation_id,
                update_fields={"original_value": "modified"},
            )
        with sqlite_instance.get_session() as session:
            piece = session.get(PromptMemoryEntry, case.request_piece_id)
            assert piece is not None
            piece.original_value = "modified"
            with pytest.raises(IntegrityError, match="native cyber evidence is immutable"):
                session.commit()
        with sqlite_instance.get_session() as session:
            stored_score = session.get(ScoreEntry, score.id)
            assert stored_score is not None
            stored_score.score_rationale = "modified"
            with pytest.raises(IntegrityError, match="native cyber evidence is immutable"):
                session.commit()
        with sqlite_instance.get_session() as session:
            content = session.get(ScorableContentEntry, snapshot.report_content_id)
            assert content is not None
            content.value = "modified"
            with pytest.raises(IntegrityError, match="native cyber evidence is immutable"):
                session.commit()
        assert case.store.get_finalized_episode(run_id=case.report.run_id) == snapshot

    async def test_missing_required_stream_downgrades_existing_score_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _finish_turn(case=case)
        assessment = case.store.assess_required_coverage(report=case.report, expected_turns=1)
        assert not assessment.required_complete
        assert any("never opened" in gap for gap in assessment.required_gaps)
        score = await _persist_score_async(report=case.report)
        assert score.status is ScoreStatus.COMPLETE

        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        assert snapshot.score_id == score.id
        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].is_undetermined
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1
        retained = sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
            snapshot.report_content_id
        ]
        assert NativeCyberReport.model_validate_json(retained.value).judgment.value == 0.75

    async def test_pre_score_coverage_gate_prevents_transient_complete_score_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _finish_turn(case=case)
        assessment = case.store.assess_required_coverage(report=case.report, expected_turns=1)
        assert not assessment.required_complete
        # The owning workflow incorporates this decision before invoking score_async.
        gated_report = NativeCyberReport.model_validate(
            {
                **case.report.model_dump(mode="json"),
                "status": NativeCyberStatus.ERROR.value,
                "errors": list(assessment.required_gaps),
            }
        )

        score = await _persist_score_async(report=gated_report)
        assert score.status is ScoreStatus.UNDETERMINED
        assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].is_undetermined
        snapshot = case.store.finalize_episode(report=gated_report, score=score, expected_turns=1)

        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        retained = sqlite_instance.get_scorable_content(content_ids=[snapshot.report_content_id])[
            snapshot.report_content_id
        ]
        assert NativeCyberReport.model_validate_json(retained.value).judgment.value == 0.75

    def test_native_run_without_a_required_raw_manifest_is_incomplete(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance, require_stream=False)
        _append_events(case=case)
        _finish_turn(case=case)
        report = NativeCyberReport.model_validate(
            {
                **case.report.model_dump(mode="json"),
                "status": NativeCyberStatus.ERROR.value,
                "errors": ["synthetic external error"],
            }
        )

        assessment = case.store.assess_required_coverage(report=report, expected_turns=1)

        assert not assessment.required_complete
        assert any("No task-required raw source streams" in gap for gap in assessment.required_gaps)

    def test_rejects_an_event_id_not_observed_in_the_source_payload(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        first_event = _events()[0].model_copy(update={"event_id": "fabricated-event-id"})

        with pytest.raises(ValueError, match="captured source payload"):
            case.store.append_events(
                run_id=case.report.run_id,
                turn_index=1,
                events=[
                    NativeCyberCapturedEvent.from_native_agent_event(
                        source=NativeCyberEvidenceSource.MODEL, event=first_event
                    )
                ],
            )

        assert case.store.get_episode(run_id=case.report.run_id).events == ()

    async def test_optional_raw_quota_gap_does_not_change_required_verdict_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance, raw_byte_limit=3)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        optional_key = NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.HARNESS,
            kind=NativeCyberRawKind.STDOUT,
            observed_source_id="optional-console",
        )
        optional_id = _close_stream(case=case, key=optional_key, data=b"12345", turn_index=1)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)

        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        assert snapshot.score_status is ScoreStatus.COMPLETE
        assert snapshot.coverage_complete
        assert snapshot.gaps == ()
        assert snapshot.optional_gaps
        optional = next(stream for stream in snapshot.raw_streams if stream.stream_id == optional_id)
        assert optional.truncated
        assert optional.received_bytes == 5
        assert optional.stored_bytes == 3
        assert optional.omitted_bytes == 2
        assert not optional.source_complete

    async def test_required_raw_quota_loss_is_explicit_and_undetermined_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance, raw_byte_limit=4)
        _append_events(case=case)
        stream_id = _close_stream(case=case, key=case.required_stream, data=b"123456", turn_index=1)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)

        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert any("quota" in gap for gap in snapshot.gaps)
        assert snapshot.raw_streams[0].received_bytes == 6
        assert snapshot.raw_streams[0].stored_bytes == 4
        assert (
            b"".join(
                chunk.data
                for chunk in case.store.read_raw_chunks(
                    run_id=case.report.run_id,
                    stream_id=stream_id,
                    allow_sensitive=True,
                )
            )
            == b"1234"
        )

    async def test_missing_native_event_fails_required_coverage_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case, count=3)
        _close_stream(case=case, key=case.required_stream, data=b"{}", turn_index=1)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)

        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert any("source count" in gap or "event count" in gap for gap in snapshot.gaps)
        assert len(sqlite_instance._query_entries(NativeCyberEventEntry)) == 3

    async def test_inconsistent_tool_arguments_cannot_claim_complete_capture_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        _finish_turn(case=case)
        report_data = case.report.model_dump(mode="json")
        report_data["agent"]["tools"][0]["arguments"] = {"x": 99}
        inconsistent = NativeCyberReport.model_validate(report_data)
        score = await _persist_score_async(report=inconsistent)

        snapshot = case.store.finalize_episode(report=inconsistent, score=score, expected_turns=1)

        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert any("tool trace" in gap for gap in snapshot.gaps)

    async def test_raw_chunk_digest_detects_later_modification_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        stream_id = _close_stream(case=case, key=case.required_stream, data=b"synthetic bytes", turn_index=1)
        with sqlite_instance.get_session() as session:
            chunk = session.get(NativeCyberRawChunkEntry, (stream_id, 1))
            assert chunk is not None
            chunk.data = b"modified bytes"
            session.commit()
        with pytest.raises(ValueError, match="corrupt byte range"):
            case.store.read_raw_chunks(
                run_id=case.report.run_id,
                stream_id=stream_id,
                allow_sensitive=True,
            )

    async def test_failed_raw_write_rolls_back_and_marks_required_gap_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        stream = NativeCyberRawStreamStart(run_id=case.report.run_id, key=case.required_stream, turn_index=1)
        case.store.open_raw_stream(stream=stream)
        failure = OperationalError("INSERT", {}, Exception("synthetic database failure"))
        with patch.object(case.store, "_open_stream", side_effect=failure):
            with pytest.raises(OperationalError, match="synthetic database failure"):
                case.store.append_raw(run_id=case.report.run_id, stream_id=stream.stream_id, data=b"synthetic raw")
        assert sqlite_instance._query_entries(NativeCyberRawChunkEntry) == []
        assert case.store.get_episode(run_id=case.report.run_id).stored_raw_bytes == 0
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)
        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)
        assert snapshot.score_status is ScoreStatus.UNDETERMINED
        assert any("database write failed" in gap for gap in snapshot.gaps)

    async def test_failure_after_chunk_insert_rolls_back_whole_append_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        stream = NativeCyberRawStreamStart(run_id=case.report.run_id, key=case.required_stream, turn_index=1)
        case.store.open_raw_stream(stream=stream)

        def _fail_after_chunk_insert(*args: object) -> None:
            if "NativeCyberRawChunkEntries" in str(args[2]) and "INSERT" in str(args[2]):
                raise OperationalError("INSERT", {}, Exception("synthetic post-insert failure"))

        event.listen(sqlite_instance.engine, "after_cursor_execute", _fail_after_chunk_insert)
        try:
            with pytest.raises(OperationalError, match="synthetic post-insert failure"):
                case.store.append_raw(
                    run_id=case.report.run_id,
                    stream_id=stream.stream_id,
                    data=b"x" * (case.store.MAX_CHUNK_BYTES + 1),
                )
        finally:
            event.remove(sqlite_instance.engine, "after_cursor_execute", _fail_after_chunk_insert)

        assert sqlite_instance._query_entries(NativeCyberRawChunkEntry) == []
        pending = case.store.get_episode(run_id=case.report.run_id)
        assert pending.stored_raw_bytes == 0
        assert pending.raw_streams[0].received_bytes == 0
        assert any("database write failed" in gap for gap in pending.gaps)

    async def test_optional_raw_write_failure_is_visible_but_not_required_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        optional = NativeCyberRawStreamStart(
            run_id=case.report.run_id,
            turn_index=1,
            key=NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS,
                kind=NativeCyberRawKind.STDERR,
                observed_source_id="optional-stderr",
            ),
        )
        case.store.open_raw_stream(stream=optional)
        failure = OperationalError("INSERT", {}, Exception("synthetic database failure"))
        with patch.object(case.store, "_open_stream", side_effect=failure):
            with pytest.raises(OperationalError):
                case.store.append_raw(run_id=case.report.run_id, stream_id=optional.stream_id, data=b"x")
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)

        snapshot = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        assert snapshot.coverage_complete
        assert snapshot.score_status is ScoreStatus.COMPLETE
        assert snapshot.optional_gaps
        assert not next(
            stream for stream in snapshot.raw_streams if stream.stream_id == optional.stream_id
        ).source_complete

    async def test_failed_finalization_leaves_orphan_score_unpublished_and_retryable_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"{}", turn_index=1)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)
        assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].status is ScoreStatus.COMPLETE
        failure = OperationalError("UPDATE", {}, Exception("synthetic finalization failure"))
        with patch.object(case.store, "_link_score", side_effect=failure):
            with pytest.raises(OperationalError):
                case.store.finalize_episode(report=case.report, score=score, expected_turns=1)

        # The existing scorer commit cannot be rolled back by this separate transaction.
        assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].status is ScoreStatus.COMPLETE
        pending = case.store.get_episode(run_id=case.report.run_id)
        assert pending.finalized_at is None
        assert pending.score_id is None
        assert pending.score_status is ScoreStatus.UNDETERMINED
        assert pending.gaps
        with pytest.raises(ValueError, match="not finalized"):
            case.store.get_finalized_episode(run_id=case.report.run_id)

        recovered = case.store.finalize_episode(report=case.report, score=score, expected_turns=1)
        assert recovered.score_id == score.id
        assert recovered.score_status is ScoreStatus.UNDETERMINED
        assert case.store.get_finalized_episode(run_id=case.report.run_id) == recovered
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert len(sqlite_instance._query_entries(ScorableContentEntry)) == 1

    async def test_episode_link_failure_rolls_back_score_downgrade_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)
        updates = 0

        def _fail_first_episode_update(*args: object) -> None:
            nonlocal updates
            if "UPDATE" in str(args[2]) and "NativeCyberEpisodeEntries" in str(args[2]):
                updates += 1
                if updates == 1:
                    raise OperationalError("UPDATE", {}, Exception("synthetic link failure"))

        event.listen(sqlite_instance.engine, "after_cursor_execute", _fail_first_episode_update)
        try:
            with pytest.raises(OperationalError, match="synthetic link failure"):
                case.store.finalize_episode(report=case.report, score=score, expected_turns=1)
        finally:
            event.remove(sqlite_instance.engine, "after_cursor_execute", _fail_first_episode_update)

        assert updates >= 1
        assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].status is ScoreStatus.COMPLETE
        assert case.store.get_episode(run_id=case.report.run_id).finalized_at is None
        with pytest.raises(ValueError, match="not finalized"):
            case.store.get_finalized_episode(run_id=case.report.run_id)

    async def test_rejects_foreign_message_piece_and_unpersisted_score_async(
        self,
        *,
        sqlite_instance: MemoryInterface,
    ) -> None:
        case = _prepare_case(memory=sqlite_instance)
        _append_events(case=case)
        _close_stream(case=case, key=case.required_stream, data=b"", turn_index=1)
        _finish_turn(case=case)
        score = await _persist_score_async(report=case.report)
        unstored = score.model_copy(update={"id": uuid4()})
        with pytest.raises(ValueError, match="already-persisted Score"):
            case.store.finalize_episode(report=case.report, score=unstored, expected_turns=1)
        assert case.store.get_episode(run_id=case.report.run_id).finalized_at is None

        foreign_piece = MessagePiece(
            role="user",
            original_value="not part of the run",
            conversation_id="other-conversation",
            sequence=2,
        )
        sqlite_instance.add_message_pieces_to_memory(message_pieces=[foreign_piece])
        with pytest.raises(ValueError, match="one persisted conversation"):
            case.store.begin_turn(
                turn=NativeCyberTurnStart(
                    run_id=case.report.run_id,
                    turn_index=2,
                    request_piece_ids=(foreign_piece.id,),
                )
            )
