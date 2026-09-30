# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Database-only capture of native cyber events and raw model/tool/harness streams."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from collections import defaultdict
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError

from pyrit.memory.memory_models import (
    NativeCyberEpisodeEntry,
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    NativeCyberToolEventEntry,
    NativeCyberTurnEntry,
    NativeCyberTurnMessagePieceEntry,
    PromptMemoryEntry,
    ScorableContentEntry,
    ScoreEntry,
)
from pyrit.memory.memory_session import _begin_sqlite_write
from pyrit.memory.native_cli_evidence_validation import _NativeCliEvidenceValidator
from pyrit.models import ContentEntryScorable, ContentScorable, Score, ScoreStatus
from pyrit.models.native_cli_report import NativeCliReportCleanup, NativeCliReportStatus, NativeCliRunReport
from pyrit.models.native_cyber import NativeCyberReport, NativeCyberStatus
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberCoverageAssessment,
    NativeCyberCoveragePhase,
    NativeCyberEpisodeSnapshot,
    NativeCyberEpisodeStart,
    NativeCyberEventSummary,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberRawChunk,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberRawStreamSummary,
    NativeCyberRawWrite,
    NativeCyberResponseMode,
    NativeCyberResponsePolicy,
    NativeCyberToolCorrelation,
    NativeCyberTurnFinish,
    NativeCyberTurnStart,
    NativeCyberTurnSummary,
)
from pyrit.models.score.observation import _message_piece_digest

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from sqlalchemy.orm import Session

    from pyrit.memory.memory_interface import MemoryInterface

logger = logging.getLogger(__name__)


class NativeCyberEvidenceStore:
    """Persist source-observed evidence through either SQLite or SQL Server memory."""

    MAX_CHUNK_BYTES = 65_536
    MAX_APPEND_BYTES = 1_048_576
    MAX_EVENT_BATCH = 100
    MAX_EVENT_BYTES = 1_048_576
    MAX_RAW_READ_CHUNKS = 16
    _WRITE_FAILURE_GAP = "A native evidence database write failed; complete capture cannot be established."

    def __init__(self, *, memory: MemoryInterface) -> None:
        """Bind the native evidence store to a configured PyRIT memory backend."""
        self._memory = memory

    def create_episode(self, *, start: NativeCyberEpisodeStart) -> None:
        """
        Fix run identity, source provenance, quota and required streams before capture.

        Raises:
            SQLAlchemyError: If the run ID already exists or the insert fails.
        """
        with closing(self._memory.get_session()) as session, session.begin():
            session.add(
                NativeCyberEpisodeEntry(
                    run_id=start.run_id,
                    binding_name=start.binding_name,
                    binding_version=start.binding_version,
                    task_id=start.task_id,
                    task_version=start.task_version,
                    started_at=start.started_at,
                    source_session_id=start.source_session_id,
                    environment_id=start.environment_id,
                    simulated=start.simulated,
                    required_raw_streams=[key.model_dump(mode="json") for key in start.required_raw_streams],
                    require_separate_tool_results=start.require_separate_tool_results,
                    response_policy_version=start.response_policy.schema_version,
                    artifact_only_allowed=start.response_policy.allow_artifact_only,
                    raw_byte_limit=start.raw_byte_limit,
                    stored_raw_bytes=0,
                    capture_gaps=[],
                    optional_gaps=[],
                    coverage_complete=False,
                )
            )

    def begin_turn(self, *, turn: NativeCyberTurnStart) -> None:
        """
        Begin the next outer turn or an explicitly ungraded Inspect sample capture.

        Raises:
            ValueError: If ordering, source identity or message-piece provenance disagrees.
        """
        with self._write_session(run_id=turn.run_id) as session:
            episode = self._lock_episode(session=session, run_id=turn.run_id)
            self._require_open(episode)
            if (
                turn.response_mode is NativeCyberResponseMode.SAMPLE_CAPTURE
                and episode.binding_name != "inspect-original"
            ):
                raise ValueError("Only an ungraded original Inspect import can store sample capture units.")
            last_index = session.scalar(
                select(func.max(NativeCyberTurnEntry.turn_index)).where(NativeCyberTurnEntry.run_id == turn.run_id)
            )
            if turn.turn_index != (last_index or 0) + 1:
                raise ValueError(f"Outer turns for run {turn.run_id} must be opened in consecutive order.")
            if last_index is not None:
                previous = session.get(NativeCyberTurnEntry, (turn.run_id, last_index))
                if previous is None or previous.finished_at is None:
                    raise ValueError("The previous native outer turn must be sealed before opening the next one.")
            if (
                turn.source_turn_id is not None
                and session.scalar(
                    select(NativeCyberTurnEntry.turn_index).where(
                        NativeCyberTurnEntry.run_id == turn.run_id,
                        NativeCyberTurnEntry.source_turn_id == turn.source_turn_id,
                    )
                )
                is not None
            ):
                raise ValueError("Observed native outer-turn identities must be unique within a run.")
            if turn.response_mode is NativeCyberResponseMode.ARTIFACT_ONLY and not episode.artifact_only_allowed:
                raise ValueError("Artifact-only completion requires the task's explicit response-policy approval.")
            entry = NativeCyberTurnEntry(
                run_id=turn.run_id,
                turn_index=turn.turn_index,
                source_turn_id=turn.source_turn_id,
                response_mode=turn.response_mode.value,
                started_at=turn.started_at,
                capture_gaps=[],
            )
            session.add(entry)
            self._link_pieces(
                session=session,
                episode=episode,
                turn_index=turn.turn_index,
                direction="request",
                piece_ids=turn.request_piece_ids,
            )

    def append_events(self, *, run_id: str, turn_index: int, events: Sequence[NativeCyberCapturedEvent]) -> None:
        """
        Atomically append source-tagged events in their actual native sequence.

        Raises:
            ValueError: If the batch, order, session, source tag or payload is invalid.
        """
        if not events or len(events) > self.MAX_EVENT_BATCH:
            raise ValueError(f"Append between 1 and {self.MAX_EVENT_BATCH} native events per call.")
        with self._write_session(run_id=run_id) as session:
            episode = self._lock_episode(session=session, run_id=run_id)
            turn = self._open_turn(session=session, episode=episode, turn_index=turn_index)
            next_sequence = (
                session.scalar(
                    select(func.max(NativeCyberEventEntry.sequence)).where(NativeCyberEventEntry.run_id == run_id)
                )
                or 0
            ) + 1
            rows: list[NativeCyberEventEntry] = []
            links: list[NativeCyberToolEventEntry] = []
            gaps = list(turn.capture_gaps)
            existing = {
                (link.call_id, link.phase)
                for link in session.scalars(
                    select(NativeCyberToolEventEntry).where(NativeCyberToolEventEntry.run_id == run_id)
                )
            }
            for captured in events:
                event = captured.event
                if event.sequence != next_sequence:
                    raise ValueError(f"Native event sequence must be {next_sequence}, got {event.sequence}.")
                self._check_event_origin(episode=episode, captured=captured)
                if episode.source_session_id is None and event.source_session_id is not None:
                    episode.source_session_id = event.source_session_id
                payload_hash = self._hash_payload(event.payload)
                rows.append(
                    NativeCyberEventEntry(
                        run_id=run_id,
                        sequence=event.sequence,
                        turn_index=turn_index,
                        source=captured.source.value,
                        observed_event_id=event.source_event_id,
                        observed_session_id=event.source_session_id,
                        observed_stream_id=event.observed_stream_id,
                        stream_offset=event.stream_offset,
                        tool_call_id=event.tool_call_id,
                        tool_phase=event.tool_phase.value if event.tool_phase else None,
                        event_type=event.event_type,
                        payload=event.payload,
                        payload_sha256=payload_hash,
                        captured_at=captured.captured_at,
                    )
                )
                tool_phases, tool_gaps = self._extract_tool_phases(event=event)
                gaps.extend(tool_gaps)
                for call_id, phase in tool_phases:
                    if (call_id, phase) in existing:
                        gaps.append(f"Native tool phase {phase} was observed more than once at event {event.sequence}.")
                        continue
                    existing.add((call_id, phase))
                    links.append(
                        NativeCyberToolEventEntry(
                            run_id=run_id, call_id=call_id, phase=phase, event_sequence=event.sequence
                        )
                    )
                next_sequence += 1
            session.add_all(rows)
            session.flush()
            session.add_all(links)
            turn.capture_gaps = gaps

    def open_raw_stream(self, *, stream: NativeCyberRawStreamStart) -> None:
        """
        Register a real byte source before any of its bytes are appended.

        Raises:
            ValueError: If the run or bound turn is already sealed.
        """
        with self._write_session(run_id=stream.run_id, raw_key=stream.key) as session:
            episode = self._lock_episode(session=session, run_id=stream.run_id)
            self._require_open(episode)
            if stream.turn_index is not None:
                self._open_turn(session=session, episode=episode, turn_index=stream.turn_index)
            key = stream.key
            session.add(
                NativeCyberRawStreamEntry(
                    stream_id=stream.stream_id,
                    run_id=stream.run_id,
                    turn_index=stream.turn_index,
                    source=key.source.value,
                    kind=key.kind.value,
                    observed_source_id=key.observed_source_id,
                    tool_call_id=stream.tool_call_id,
                    received_bytes=0,
                    stored_bytes=0,
                    truncated=False,
                    capture_gaps=[],
                )
            )

    def append_raw(self, *, run_id: str, stream_id: uuid.UUID, data: bytes) -> NativeCyberRawWrite:
        """
        Split captured bytes into bounded DB rows; report every quota-omitted byte.

        Returns:
            NativeCyberRawWrite: Committed byte counts and explicit truncation.

        Raises:
            ValueError: If the append exceeds the per-call limit or the stream is sealed.
        """
        if not isinstance(data, bytes) or not 0 < len(data) <= self.MAX_APPEND_BYTES:
            raise ValueError(f"Raw append must contain 1 to {self.MAX_APPEND_BYTES} bytes.")
        with self._write_session(run_id=run_id, stream_id=stream_id) as session:
            episode = self._lock_episode(session=session, run_id=run_id)
            stream = self._open_stream(session=session, episode=episode, stream_id=stream_id)
            available = max(0, episode.raw_byte_limit - episode.stored_raw_bytes)
            retained = data[:available]
            next_sequence = (
                session.scalar(
                    select(func.max(NativeCyberRawChunkEntry.sequence)).where(
                        NativeCyberRawChunkEntry.stream_id == stream_id
                    )
                )
                or 0
            ) + 1
            for offset in range(0, len(retained), self.MAX_CHUNK_BYTES):
                chunk = retained[offset : offset + self.MAX_CHUNK_BYTES]
                session.add(
                    NativeCyberRawChunkEntry(
                        stream_id=stream_id,
                        sequence=next_sequence,
                        byte_offset=stream.stored_bytes + offset,
                        byte_length=len(chunk),
                        sha256=hashlib.sha256(chunk).hexdigest(),
                        data=chunk,
                    )
                )
                next_sequence += 1
            stream.received_bytes += len(data)
            stream.stored_bytes += len(retained)
            episode.stored_raw_bytes += len(retained)
            stream.truncated = stream.truncated or len(retained) < len(data)
            return NativeCyberRawWrite(
                stream_id=stream_id,
                received_bytes=len(data),
                stored_bytes=len(retained),
                omitted_bytes=len(data) - len(retained),
                truncated=stream.truncated,
            )

    def close_raw_stream(
        self,
        *,
        run_id: str,
        stream_id: uuid.UUID,
        source_complete: bool,
        expected_bytes: int,
        observed_sha256: str | None,
        gaps: Sequence[str] = (),
    ) -> None:
        """
        Seal a byte stream, recording source gaps, quota loss and content mismatch.

        Raises:
            ValueError: If a digest is invalid, chunks are corrupt, or the stream is sealed.
        """
        if expected_bytes < 0:
            raise ValueError("Expected native raw byte length cannot be negative.")
        if observed_sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", observed_sha256):
            raise ValueError("Native raw digest must be lowercase SHA-256 hex.")
        with self._write_session(run_id=run_id, stream_id=stream_id) as session:
            episode = self._lock_episode(session=session, run_id=run_id)
            stream = self._open_stream(session=session, episode=episode, stream_id=stream_id)
            stored_sha256 = self._hash_stream(session=session, stream=stream)
            issues = list(gaps)
            if not source_complete:
                issues.append("Raw source did not report complete capture.")
            if stream.truncated:
                issues.append("Raw stream exceeded the run byte quota.")
            if expected_bytes != stream.received_bytes or stream.received_bytes != stream.stored_bytes:
                issues.append("Raw byte lengths do not match the source capture.")
            if observed_sha256 is None or observed_sha256 != stored_sha256:
                issues.append("Raw digest does not match the source capture.")
            stream.expected_bytes = expected_bytes
            stream.observed_sha256 = observed_sha256
            stream.stored_sha256 = stored_sha256
            stream.source_complete = not issues
            stream.capture_gaps = list(dict.fromkeys(issues))
            stream.closed_at = datetime.now(UTC)

    def finish_turn(self, *, finish: NativeCyberTurnFinish) -> None:
        """
        Seal one outer turn; missing pieces or events remain visible as gaps.

        Raises:
            ValueError: If the turn is sealed, a request was already linked, or a piece is invalid.
        """
        with self._write_session(run_id=finish.run_id) as session:
            episode = self._lock_episode(session=session, run_id=finish.run_id)
            turn = self._open_turn(session=session, episode=episode, turn_index=finish.turn_index)
            if finish.finished_at < turn.started_at:
                raise ValueError("A native outer turn cannot finish before it starts.")
            if finish.request_piece_ids:
                existing_request = session.scalar(
                    select(NativeCyberTurnMessagePieceEntry.message_piece_id)
                    .where(
                        NativeCyberTurnMessagePieceEntry.run_id == finish.run_id,
                        NativeCyberTurnMessagePieceEntry.turn_index == finish.turn_index,
                        NativeCyberTurnMessagePieceEntry.direction == "request",
                    )
                    .limit(1)
                )
                if existing_request is not None:
                    raise ValueError("Native request pieces were already linked at begin_turn.")
                self._link_pieces(
                    session=session,
                    episode=episode,
                    turn_index=finish.turn_index,
                    direction="request",
                    piece_ids=finish.request_piece_ids,
                )
            self._link_pieces(
                session=session,
                episode=episode,
                turn_index=finish.turn_index,
                direction="response",
                piece_ids=finish.response_piece_ids,
            )
            self._link_pieces(
                session=session,
                episode=episode,
                turn_index=finish.turn_index,
                direction="tool_request",
                piece_ids=finish.tool_request_piece_ids,
            )
            self._link_pieces(
                session=session,
                episode=episode,
                turn_index=finish.turn_index,
                direction="tool_result",
                piece_ids=finish.tool_result_piece_ids,
            )
            count = session.scalar(
                select(func.count())
                .select_from(NativeCyberEventEntry)
                .where(
                    NativeCyberEventEntry.run_id == finish.run_id,
                    NativeCyberEventEntry.turn_index == finish.turn_index,
                )
            )
            gaps = [*turn.capture_gaps, *finish.gaps]
            if not finish.source_complete:
                gaps.append("Native turn source did not report complete event coverage.")
            if count != finish.observed_event_count:
                gaps.append("Native turn source count differs from stored events.")
            if finish.source_complete and not count:
                gaps.append("Native outer turn has no observed source events.")
            directions = set(
                session.scalars(
                    select(NativeCyberTurnMessagePieceEntry.direction).where(
                        NativeCyberTurnMessagePieceEntry.run_id == finish.run_id,
                        NativeCyberTurnMessagePieceEntry.turn_index == finish.turn_index,
                    )
                )
            )
            if (
                finish.source_complete
                and turn.response_mode != NativeCyberResponseMode.SAMPLE_CAPTURE.value
                and "request" not in directions
            ):
                gaps.append("Native turn lacks a stored request MessagePiece.")
            if (
                finish.source_complete
                and turn.response_mode == NativeCyberResponseMode.MESSAGE_REQUIRED.value
                and "response" not in directions
            ):
                gaps.append("Native chat turn lacks a genuine assistant response MessagePiece.")
            if finish.source_complete and turn.response_mode == NativeCyberResponseMode.ARTIFACT_ONLY.value:
                events = list(
                    session.scalars(
                        select(NativeCyberEventEntry).where(
                            NativeCyberEventEntry.run_id == finish.run_id,
                            NativeCyberEventEntry.turn_index == finish.turn_index,
                        )
                    )
                )
                if not self._has_terminal_event(events=events):
                    gaps.append("Artifact-only native turn lacks an observed terminal source event.")
            turn.finished_at = finish.finished_at
            turn.observed_event_count = finish.observed_event_count
            turn.source_complete = finish.source_complete and not gaps
            turn.capture_gaps = list(dict.fromkeys(gaps))

    def mark_capture_gap(self, *, run_id: str, reason: str, required: bool = True) -> None:
        """
        Append an irreversible redacted required or optional capture gap.

        Raises:
            ValueError: If the reason is invalid or the episode is already finalized.
        """
        if not reason.strip() or len(reason) > 512:
            raise ValueError("A native capture gap requires a short, nonempty reason without raw content.")
        with self._write_session(run_id=run_id) as session:
            episode = self._lock_episode(session=session, run_id=run_id)
            self._require_open(episode)
            if required:
                episode.capture_gaps = list(dict.fromkeys([*episode.capture_gaps, reason]))
            else:
                episode.optional_gaps = list(dict.fromkeys([*episode.optional_gaps, reason]))

    def assess_required_coverage(
        self,
        *,
        report: NativeCyberReport,
        expected_turns: int,
    ) -> NativeCyberCoverageAssessment:
        """
        Assess final capture and acquired judgment before scoring.

        Returns:
            NativeCyberCoverageAssessment: Final verdict gate and optional telemetry gaps.

        Raises:
            ValueError: If the turn count or report provenance is invalid.
        """
        return self._assess_coverage(
            report=report,
            expected_turns=expected_turns,
            phase=NativeCyberCoveragePhase.FINAL,
        )

    def assess_pregrading_coverage(
        self,
        *,
        report: NativeCyberReport,
        expected_turns: int,
    ) -> NativeCyberCoverageAssessment:
        """
        Check captured source evidence before the original grader acquires a judgment.

        An artifact-only turn still needs its real request and terminal event, but
        its not-yet-acquired judgment and artifact are checked only at FINAL.

        Returns:
            NativeCyberCoverageAssessment: Pregrading gate and optional telemetry gaps.

        Raises:
            ValueError: If the report already includes a judgment or its provenance is invalid.
        """
        if report.judgment is not None:
            raise ValueError("Pregrading coverage requires a report without an acquired original judgment.")
        return self._assess_coverage(
            report=report,
            expected_turns=expected_turns,
            phase=NativeCyberCoveragePhase.PREGRADING,
        )

    def _assess_coverage(
        self,
        *,
        report: NativeCyberReport,
        expected_turns: int,
        phase: NativeCyberCoveragePhase,
    ) -> NativeCyberCoverageAssessment:
        if expected_turns < 0:
            raise ValueError("The expected outer-turn count cannot be negative.")
        with self._write_session(run_id=report.run_id) as session:
            episode = self._lock_episode(session=session, run_id=report.run_id)
            self._require_open(episode)
            self._validate_report(episode=episode, report=report)
            required, optional = self._capture_gaps(
                session=session,
                episode=episode,
                report=report,
                expected_turns=expected_turns,
                phase=phase,
            )
            return NativeCyberCoverageAssessment(
                phase=phase,
                required_complete=bool(report.agent and report.agent.coverage_complete and not required),
                required_gaps=tuple(required),
                optional_gaps=tuple(optional),
            )

    def finalize_episode(
        self,
        *,
        report: NativeCyberReport,
        score: Score,
        expected_turns: int,
    ) -> NativeCyberEpisodeSnapshot:
        """
        Link the already-persisted Score and report in one transaction.

        The scorer's earlier Score/content commit is separate. This method never
        creates another score; only task-required gaps downgrade the same score ID.

        Returns:
            NativeCyberEpisodeSnapshot: Linked report and final required-coverage verdict.

        Raises:
            ValueError: If the source, Score, content or turn count is invalid.
        """
        if expected_turns < 0:
            raise ValueError("The expected outer-turn count cannot be negative.")
        with self._write_session(run_id=report.run_id) as session:
            episode = self._lock_episode(session=session, run_id=report.run_id)
            self._finalize_in_session(
                session=session,
                episode=episode,
                report=report,
                score=score,
                expected_turns=expected_turns,
                atomic=False,
            )
        return self.get_episode(run_id=report.run_id)

    def finalize_episode_atomic(
        self,
        *,
        report: NativeCyberReport,
        score: Score,
        expected_turns: int,
    ) -> NativeCyberEpisodeSnapshot:
        """
        Commit report content, one caller-prepared Score and episode link together.

        The caller obtains an unpersisted Score from the scorer before invoking
        this method. No scorer is called from memory, and any failed insert rolls
        back all three records without leaving an orphan complete Score.

        Returns:
            NativeCyberEpisodeSnapshot: Final report and score references.

        Raises:
            ValueError: If the report, Score, capture or run identity is invalid.
        """
        if expected_turns < 0:
            raise ValueError("The expected outer-turn count cannot be negative.")
        with self._write_session(run_id=report.run_id) as session:
            episode = self._lock_episode(session=session, run_id=report.run_id)
            self._finalize_in_session(
                session=session,
                episode=episode,
                report=report,
                score=score,
                expected_turns=expected_turns,
                atomic=True,
            )
        return self.get_episode(run_id=report.run_id)

    def finalize_unscored_inspect_capture(
        self,
        *,
        run_id: str,
        expected_samples: int,
        required_gaps: Sequence[str] = (),
        optional_gaps: Sequence[str] = (),
    ) -> NativeCyberEpisodeSnapshot:
        """
        Seal original Inspect bytes and sample projections without inventing a Score.

        Returns:
            NativeCyberEpisodeSnapshot: Finalized capture; its Score and report links remain absent.

        Raises:
            ValueError: If a graded run, missing source, or damaged sample is supplied.
        """
        if expected_samples < 0 or any(not gap.strip() or len(gap) > 512 for gap in (*required_gaps, *optional_gaps)):
            raise ValueError("Inspect capture needs a valid sample count and short redacted coverage reasons.")
        with self._write_session(run_id=run_id) as session:
            episode = self._lock_episode(session=session, run_id=run_id)
            self._require_open(episode)
            if episode.binding_name != "inspect-original" or episode.score_id is not None:
                raise ValueError("Only an original unscored Inspect capture can be sealed without a Score.")
            required, optional = self._unscored_inspect_gaps(
                session=session, episode=episode, expected_samples=expected_samples
            )
            episode.capture_gaps = list(dict.fromkeys([*required, *required_gaps]))
            episode.optional_gaps = list(dict.fromkeys([*optional, *optional_gaps]))
            episode.coverage_complete = not episode.capture_gaps
            episode.finalized_at = datetime.now(UTC)
            session.flush()
        return self.get_finalized_unscored_inspect_capture(run_id=run_id)

    def get_finalized_unscored_inspect_capture(self, *, run_id: str) -> NativeCyberEpisodeSnapshot:
        """
        Read only a finalized original Inspect import, not a graded native run.

        Returns:
            NativeCyberEpisodeSnapshot: Metadata-only sample, event and stream evidence.

        Raises:
            ValueError: If the episode is pending, graded, or from another binding.
        """
        snapshot = self.get_episode(run_id=run_id)
        if (
            snapshot.run.binding_name != "inspect-original"
            or snapshot.finalized_at is None
            or snapshot.score_id is not None
            or snapshot.report_content_id is not None
        ):
            raise ValueError("An original unscored Inspect capture is not finalized for readback.")
        return snapshot

    @classmethod
    def _unscored_inspect_gaps(
        cls, *, session: Session, episode: NativeCyberEpisodeEntry, expected_samples: int
    ) -> tuple[list[str], list[str]]:
        required = list(episode.capture_gaps)
        optional = list(episode.optional_gaps)
        streams = list(
            session.scalars(select(NativeCyberRawStreamEntry).where(NativeCyberRawStreamEntry.run_id == episode.run_id))
        )
        if sum(stream.stored_bytes for stream in streams) != episode.stored_raw_bytes:
            required.append("Original Inspect raw stream lengths differ from the episode byte total.")
        for key in episode.required_raw_streams:
            matching = [
                stream
                for stream in streams
                if (stream.source, stream.kind, stream.observed_source_id)
                == (key["source"], key["kind"], key["observed_source_id"])
            ]
            if len(matching) != 1:
                required.append("An original Inspect raw source is missing or duplicated.")
                continue
            stream = matching[0]
            if (
                stream.closed_at is None
                or stream.source_complete is not True
                or stream.truncated
                or stream.expected_bytes != stream.stored_bytes
                or stream.observed_sha256 != stream.stored_sha256
                or cls._hash_stream(session=session, stream=stream) != stream.stored_sha256
            ):
                required.extend(stream.capture_gaps or ["An original Inspect raw source is incomplete."])
        turns = list(
            session.scalars(
                select(NativeCyberTurnEntry)
                .where(NativeCyberTurnEntry.run_id == episode.run_id)
                .order_by(NativeCyberTurnEntry.turn_index)
            )
        )
        if len(turns) != expected_samples or [turn.turn_index for turn in turns] != list(
            range(1, expected_samples + 1)
        ):
            required.append("Inspect sample capture count differs from the original EvalLog.")
        for turn in turns:
            event_count = session.scalar(
                select(func.count(NativeCyberEventEntry.sequence)).where(
                    NativeCyberEventEntry.run_id == episode.run_id,
                    NativeCyberEventEntry.turn_index == turn.turn_index,
                )
            )
            if (
                turn.response_mode != NativeCyberResponseMode.SAMPLE_CAPTURE.value
                or turn.source_complete is not True
                or turn.finished_at is None
                or turn.observed_event_count != event_count
            ):
                required.extend(turn.capture_gaps or ["An original Inspect sample projection is incomplete."])
        links = list(
            session.scalars(
                select(NativeCyberTurnMessagePieceEntry).where(
                    NativeCyberTurnMessagePieceEntry.run_id == episode.run_id
                )
            )
        )
        if not cls._piece_links_intact(session=session, links=links):
            required.append("An original Inspect transcript MessagePiece is missing or modified.")
        required_keys = {
            (key["source"], key["kind"], key["observed_source_id"]) for key in episode.required_raw_streams
        }
        if required_keys != {
            ("harness", "eval_log", "inspect-original-eval-archive"),
            ("harness", "jsonl", "inspect-resolved-eval-log"),
        }:
            required.append("Original Inspect import lacks its exact archive and resolved-log source contract.")
        for stream in streams:
            if (stream.source, stream.kind, stream.observed_source_id) not in required_keys and (
                stream.closed_at is None or stream.source_complete is not True
            ):
                optional.extend(stream.capture_gaps or ["Optional live Inspect hook coverage is incomplete."])
        return list(dict.fromkeys(required)), list(dict.fromkeys(optional))

    def finalize_cli_episode_atomic(
        self,
        *,
        report: NativeCliRunReport,
        score: Score,
        expected_turns: int,
    ) -> NativeCyberEpisodeSnapshot:
        """
        Commit one CLI report, its existing generic Score type, and episode link atomically.

        Validate the real parser, process, model-gateway, and raw DB rows without
        projecting CLI events into GHCP events or calling a provider or grader.

        Returns:
            NativeCyberEpisodeSnapshot: A metadata-only view of the committed CLI outcome.

        Raises:
            ValueError: If the report or unpersisted canonical Score is invalid.
            SQLAlchemyError: If no single transaction can retain the report, Score, and link.
        """
        if expected_turns < 0:
            raise ValueError("A CLI expected outer-turn count cannot be negative.")
        report = NativeCliRunReport.model_validate(report.model_dump(mode="json"))
        self._validate_cli_score(report=report, score=score)
        with self._write_session(run_id=report.run_id) as session:
            episode = self._lock_episode(session=session, run_id=report.run_id)
            self._require_open(episode)
            required, optional = self._cli_capture_gaps(
                session=session,
                episode=episode,
                report=report,
                expected_turns=expected_turns,
                phase=NativeCyberCoveragePhase.FINAL,
            )
            complete = (
                report.status is NativeCliReportStatus.COMPLETED and report.evidence.coverage_complete and not required
            )
            prepared = self._cli_score_for_capture(score=score, complete=complete, gap_count=len(required))
            stored = self._insert_prepared_score_row(session=session, score=prepared)
            self._validate_cli_stored_score(session=session, stored=stored, report=report)
            episode.report_content_id = stored.scorable_content_id
            episode.report_sha256 = report.sha256()
            episode.score_id = uuid.UUID(str(stored.id))
            episode.coverage_complete = complete
            episode.capture_gaps = required
            episode.optional_gaps = optional
            episode.finalized_at = datetime.now(UTC)
            session.flush()
        return self.get_episode(run_id=report.run_id)

    def assess_cli_pregrading_coverage(
        self, *, report: NativeCliRunReport, expected_turns: int
    ) -> NativeCyberCoverageAssessment:
        """
        Check sealed CLI source evidence before calling the original grader.

        The draft has no judgment or verified cleanup yet. This uses the same
        persisted provenance, byte, gateway and tool checks as finalization;
        only the post-grading verdict and artifact checks are deferred.

        Returns:
            NativeCyberCoverageAssessment: Typed PREGRADING verdict and explicit gaps.

        Raises:
            ValueError: If the report is not an ungraded, still-live draft or the turn count is invalid.
        """
        if expected_turns < 0:
            raise ValueError("A CLI expected outer-turn count cannot be negative.")
        report = NativeCliRunReport.model_validate(report.model_dump(mode="json"))
        if (
            report.status is not NativeCliReportStatus.INCOMPLETE
            or report.cleanup is not NativeCliReportCleanup.UNKNOWN
            or report.judgment is not None
        ):
            raise ValueError("CLI pregrading requires an incomplete live-lease report without an original judgment.")
        with self._write_session(run_id=report.run_id) as session:
            episode = self._lock_episode(session=session, run_id=report.run_id)
            self._require_open(episode)
            required, optional = self._cli_capture_gaps(
                session=session,
                episode=episode,
                report=report,
                expected_turns=expected_turns,
                phase=NativeCyberCoveragePhase.PREGRADING,
            )
            return NativeCyberCoverageAssessment(
                phase=NativeCyberCoveragePhase.PREGRADING,
                required_complete=report.evidence.coverage_complete and not required,
                required_gaps=tuple(required),
                optional_gaps=tuple(optional),
            )

    def _cli_capture_gaps(
        self,
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        report: NativeCliRunReport,
        expected_turns: int,
        phase: NativeCyberCoveragePhase,
    ) -> tuple[list[str], list[str]]:
        validator = _NativeCliEvidenceValidator(
            session=session,
            episode=episode,
            report=report,
            expected_turns=expected_turns,
            phase=phase,
            digest_stream=lambda stream: self._hash_stream(session=session, stream=stream),
            digest_event=self._hash_payload,
            verify_pieces=lambda links: self._piece_links_intact(session=session, links=links),
            is_terminal_event=self._is_terminal_event,
        )
        return validator.validate()

    @staticmethod
    def _validate_cli_score(*, report: NativeCliRunReport, score: Score) -> None:
        scorable = score.scorable
        if (
            score.score_type != "float_scale"
            or score.message_piece_id is not None
            or score.observation_ids
            or not isinstance(scorable, ContentScorable)
            or scorable.data_type != "text"
            or scorable.value != report.canonical_json()
        ):
            raise ValueError("CLI finalization requires one unpersisted canonical report ContentScorable.")
        expected = {
            "contract_version": "native-cli-run-v1",
            "task_id": report.task_id,
            "task_version": report.task_version,
            "run_id": report.run_id,
            "turn_id": report.turn_id,
            "turn_index": report.turn_index,
            "report_sha256": report.sha256(),
            "cli_protocol": report.protocol.value,
            "cli_version": report.cli_version,
            "cli_profile": report.cli_profile,
            "run_status": report.status.value,
            "publication_state": "unpersisted_candidate",
        }
        if any((score.score_metadata or {}).get(key) != value for key, value in expected.items()):
            raise ValueError("CLI Score does not identify the exact task, turn, and canonical report.")
        scorer = score.scorer_class_identifier
        if (
            scorer is None
            or scorer.class_name != "NativeCliReportScoreBuilder"
            or scorer.class_module != "pyrit.score.float_scale.native_cli_report_scorer"
            or scorer.params.get("contract") != "native-cli-run-v1"
            or scorer.params.get("report_sha256") != report.sha256()
        ):
            raise ValueError("CLI Score was not prepared by the native CLI report contract.")
        grade = report.judgment.value if report.status is NativeCliReportStatus.COMPLETED and report.judgment else None
        if grade is None:
            if score.status is not ScoreStatus.UNDETERMINED or score.score_value is not None:
                raise ValueError("An incomplete CLI report cannot claim a complete Score.")
        elif score.status is not ScoreStatus.COMPLETE or score.get_value() != grade:
            raise ValueError("A CLI Score must retain the exact acquired original grader value.")

    @staticmethod
    def _cli_score_for_capture(*, score: Score, complete: bool, gap_count: int) -> Score:
        metadata = {
            **(score.score_metadata or {}),
            "publication_state": "committed_final_result",
            "native_required_capture": "complete" if complete else "incomplete",
            "native_required_gap_count": gap_count,
        }
        rationale = (
            score.score_rationale
            if complete or score.status is ScoreStatus.UNDETERMINED
            else f"Required CLI database capture is incomplete ({gap_count} gap(s)); original judgment is retained."
        )
        return Score.model_validate(
            {
                **score.model_dump(exclude={"objective"}),
                "score_value": score.score_value if complete else None,
                "status": ScoreStatus.COMPLETE if complete else ScoreStatus.UNDETERMINED,
                "score_rationale": rationale,
                "score_metadata": metadata,
            }
        )

    @staticmethod
    def _validate_cli_stored_score(*, session: Session, stored: ScoreEntry, report: NativeCliRunReport) -> None:
        content = session.get(ScorableContentEntry, stored.scorable_content_id) if stored.scorable_content_id else None
        if (
            content is None
            or content.data_type != "text"
            or content.value != report.canonical_json()
            or content.value_sha256 != report.sha256()
            or stored.score_type != "float_scale"
            or stored.prompt_request_response_id is not None
            or stored.scorable is None
            or stored.scorable.get("scorable_type") != "content_entry"
            or stored.scorable.get("content_id") != str(content.id)
            or (stored.score_metadata or {}).get("report_sha256") != report.sha256()
            or (stored.score_metadata or {}).get("run_id") != report.run_id
        ):
            raise ValueError("Stored CLI Score is not anchored to the exact canonical report.")
        if stored.status == ScoreStatus.COMPLETE.value and (
            report.judgment is None or stored.score_value is None or float(stored.score_value) != report.judgment.value
        ):
            raise ValueError("Stored CLI Score differs from the original acquired judgment.")

    def _finalize_in_session(
        self,
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        report: NativeCyberReport,
        score: Score,
        expected_turns: int,
        atomic: bool,
    ) -> None:
        self._require_open(episode)
        self._validate_report(episode=episode, report=report)
        self._validate_score(report=report, score=score, unpersisted=atomic)
        required_gaps, optional_gaps = self._capture_gaps(
            session=session,
            episode=episode,
            report=report,
            expected_turns=expected_turns,
            phase=NativeCyberCoveragePhase.FINAL,
        )
        required_complete = bool(report.agent and report.agent.coverage_complete and not required_gaps)
        canonical_score = self._score_for_capture(
            score=score,
            report=report,
            required_complete=required_complete,
            gap_count=len(required_gaps),
        )
        if atomic:
            content_id, stored_score = self._insert_atomic_score(
                session=session,
                report=report,
                score=canonical_score,
            )
        else:
            content_id, stored_score = self._link_score(
                session=session,
                report=report,
                score=canonical_score,
            )
        episode.report_content_id = content_id
        episode.report_sha256 = report.sha256()
        episode.score_id = uuid.UUID(str(stored_score.id))
        episode.coverage_complete = required_complete
        episode.capture_gaps = required_gaps
        episode.optional_gaps = optional_gaps
        episode.finalized_at = datetime.now(UTC)
        session.flush()

    def get_episode(self, *, run_id: str) -> NativeCyberEpisodeSnapshot:
        """
        Read provenance, coverage and links without any raw payload or log bytes.

        Returns:
            NativeCyberEpisodeSnapshot: Safe status, or an undetermined pending run.

        Raises:
            KeyError: If the run does not exist.
            ValueError: If its linked Score is missing.
        """
        with closing(self._memory.get_session()) as session:
            episode = session.get(NativeCyberEpisodeEntry, run_id)
            if episode is None:
                raise KeyError(f"Native cyber episode {run_id} does not exist.")
            turns = list(
                session.scalars(
                    select(NativeCyberTurnEntry)
                    .where(NativeCyberTurnEntry.run_id == run_id)
                    .order_by(NativeCyberTurnEntry.turn_index)
                )
            )
            events = list(
                session.scalars(
                    select(NativeCyberEventEntry)
                    .where(NativeCyberEventEntry.run_id == run_id)
                    .order_by(NativeCyberEventEntry.sequence)
                )
            )
            pieces = list(
                session.scalars(
                    select(NativeCyberTurnMessagePieceEntry)
                    .where(NativeCyberTurnMessagePieceEntry.run_id == run_id)
                    .order_by(
                        NativeCyberTurnMessagePieceEntry.turn_index,
                        NativeCyberTurnMessagePieceEntry.direction,
                        NativeCyberTurnMessagePieceEntry.position,
                    )
                )
            )
            if not self._piece_links_intact(session=session, links=pieces):
                raise ValueError(f"Native episode {run_id} has a missing or modified linked MessagePiece.")
            tool_links = list(
                session.scalars(select(NativeCyberToolEventEntry).where(NativeCyberToolEventEntry.run_id == run_id))
            )
            streams = list(
                session.scalars(
                    select(NativeCyberRawStreamEntry)
                    .where(NativeCyberRawStreamEntry.run_id == run_id)
                    .order_by(
                        NativeCyberRawStreamEntry.source,
                        NativeCyberRawStreamEntry.observed_source_id,
                        NativeCyberRawStreamEntry.stream_id,
                    )
                )
            )
            stored_score = session.get(ScoreEntry, episode.score_id) if episode.score_id else None
            if episode.score_id and stored_score is None:
                raise ValueError(f"Native episode {run_id} references a missing Score.")
            return self._build_snapshot(
                episode=episode,
                turns=turns,
                events=events,
                pieces=pieces,
                tool_links=tool_links,
                streams=streams,
                stored_score=stored_score,
            )

    def get_finalized_episode(self, *, run_id: str) -> NativeCyberEpisodeSnapshot:
        """
        Return a run-facing result only after the report, score and episode link commit.

        Returns:
            NativeCyberEpisodeSnapshot: A committed publication view.

        Raises:
            ValueError: If the episode is pending or a complete Score lacks required capture.
        """
        snapshot = self.get_episode(run_id=run_id)
        if snapshot.finalized_at is None or snapshot.score_id is None or snapshot.report_content_id is None:
            raise ValueError(f"Native cyber episode {run_id} is not finalized and cannot be published.")
        if not snapshot.coverage_complete and snapshot.score_status is ScoreStatus.COMPLETE:
            raise ValueError(f"Native cyber episode {run_id} has incomplete required capture and a complete Score.")
        return snapshot

    def read_event_payloads(
        self,
        *,
        run_id: str,
        allow_sensitive: bool = False,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> tuple[NativeCyberCapturedEvent, ...]:
        """
        Page native event payloads only for an explicitly authorized internal caller.

        Returns:
            tuple[NativeCyberCapturedEvent, ...]: Verified source events in order.

        Raises:
            PermissionError: If sensitive access was not explicitly granted.
            ValueError: If the page bounds or event digests are invalid.
        """
        self._require_sensitive_access(allow_sensitive=allow_sensitive)
        if after_sequence < 0 or not 1 <= limit <= self.MAX_EVENT_BATCH:
            raise ValueError("Native event reads require a nonnegative cursor and a page of at most 100.")
        with closing(self._memory.get_session()) as session:
            self._lock_episode(session=session, run_id=run_id)
            entries = session.scalars(
                select(NativeCyberEventEntry)
                .where(
                    NativeCyberEventEntry.run_id == run_id,
                    NativeCyberEventEntry.sequence > after_sequence,
                )
                .order_by(NativeCyberEventEntry.sequence)
                .limit(limit)
            )
            result: list[NativeCyberCapturedEvent] = []
            for entry in entries:
                if entry.payload_sha256 != self._hash_payload(entry.payload):
                    raise ValueError(f"Native event {entry.sequence} payload failed integrity validation.")
                result.append(
                    NativeCyberCapturedEvent(
                        source=entry.source,
                        captured_at=entry.captured_at,
                        event=NativeCyberObservedEvent(
                            controller_sequence=entry.sequence,
                            source_event_id=entry.observed_event_id,
                            source_session_id=entry.observed_session_id,
                            event_type=entry.event_type,
                            payload=entry.payload,
                            observed_stream_id=entry.observed_stream_id,
                            stream_offset=entry.stream_offset,
                            tool_call_id=entry.tool_call_id,
                            tool_phase=entry.tool_phase,
                        ),
                    )
                )
            return tuple(result)

    def read_raw_chunks(
        self,
        *,
        run_id: str,
        stream_id: uuid.UUID,
        allow_sensitive: bool = False,
        after_sequence: int = 0,
        limit: int = 16,
    ) -> tuple[NativeCyberRawChunk, ...]:
        """
        Page verified DB-resident bytes; never include raw bytes in episode summaries.

        Returns:
            tuple[NativeCyberRawChunk, ...]: Verified bounded byte ranges in order.

        Raises:
            PermissionError: If sensitive access was not explicitly granted.
            KeyError: If the stream is missing or belongs to another run.
            ValueError: If a cursor, chunk length, offset or digest is invalid.
        """
        self._require_sensitive_access(allow_sensitive=allow_sensitive)
        if after_sequence < 0 or not 1 <= limit <= self.MAX_RAW_READ_CHUNKS:
            raise ValueError("Native raw reads require a nonnegative cursor and a page of at most 16 chunks.")
        with closing(self._memory.get_session()) as session:
            stream = session.get(NativeCyberRawStreamEntry, stream_id)
            if stream is None or stream.run_id != run_id:
                raise KeyError(f"Native raw stream {stream_id} does not exist in run {run_id}.")
            previous = session.get(NativeCyberRawChunkEntry, (stream_id, after_sequence)) if after_sequence else None
            if after_sequence and previous is None:
                raise ValueError("Native raw chunk cursor does not exist.")
            expected_offset = previous.byte_offset + previous.byte_length if previous else 0
            entries = session.scalars(
                select(NativeCyberRawChunkEntry)
                .where(
                    NativeCyberRawChunkEntry.stream_id == stream_id,
                    NativeCyberRawChunkEntry.sequence > after_sequence,
                )
                .order_by(NativeCyberRawChunkEntry.sequence)
                .limit(limit)
            )
            chunks: list[NativeCyberRawChunk] = []
            for entry in entries:
                if (
                    entry.sequence != after_sequence + len(chunks) + 1
                    or entry.byte_offset != expected_offset
                    or not 0 < entry.byte_length <= self.MAX_CHUNK_BYTES
                    or entry.byte_length != len(entry.data)
                    or entry.sha256 != hashlib.sha256(entry.data).hexdigest()
                ):
                    raise ValueError(f"Native raw stream {stream_id} has a missing or corrupt byte range.")
                chunks.append(
                    NativeCyberRawChunk(
                        sequence=entry.sequence,
                        offset=entry.byte_offset,
                        length=entry.byte_length,
                        sha256=entry.sha256,
                        data=entry.data,
                    )
                )
                expected_offset += entry.byte_length
            return tuple(chunks)

    @staticmethod
    def _require_sensitive_access(*, allow_sensitive: bool) -> None:
        if not allow_sensitive:
            raise PermissionError("Native event payloads and raw logs require an explicitly authorized internal read.")

    @classmethod
    def _build_snapshot(
        cls,
        *,
        episode: NativeCyberEpisodeEntry,
        turns: Sequence[NativeCyberTurnEntry],
        events: Sequence[NativeCyberEventEntry],
        pieces: Sequence[NativeCyberTurnMessagePieceEntry],
        tool_links: Sequence[NativeCyberToolEventEntry],
        streams: Sequence[NativeCyberRawStreamEntry],
        stored_score: ScoreEntry | None,
    ) -> NativeCyberEpisodeSnapshot:
        piece_ids: dict[tuple[int, str], list[uuid.UUID]] = defaultdict(list)
        event_counts: dict[int, int] = defaultdict(int)
        tool_sequences: dict[str, dict[str, int]] = defaultdict(dict)
        for piece in pieces:
            piece_ids[(piece.turn_index, piece.direction)].append(piece.message_piece_id)
        for event in events:
            event_counts[event.turn_index] += 1
        for link in tool_links:
            tool_sequences[link.call_id][link.phase] = link.event_sequence
        if episode.response_policy_version != 1:
            raise ValueError(f"Native episode {episode.run_id} has an unsupported response policy version.")
        run = NativeCyberEpisodeStart(
            run_id=episode.run_id,
            binding_name=episode.binding_name,
            binding_version=episode.binding_version,
            task_id=episode.task_id,
            task_version=episode.task_version,
            started_at=episode.started_at,
            source_session_id=episode.source_session_id,
            environment_id=episode.environment_id,
            simulated=episode.simulated,
            required_raw_streams=tuple(
                NativeCyberRawStreamKey.model_validate(key) for key in episode.required_raw_streams
            ),
            require_separate_tool_results=episode.require_separate_tool_results,
            response_policy=NativeCyberResponsePolicy(
                schema_version=1,
                allow_artifact_only=episode.artifact_only_allowed,
            ),
            raw_byte_limit=episode.raw_byte_limit,
        )
        return NativeCyberEpisodeSnapshot(
            run=run,
            conversation_id=episode.conversation_id,
            finalized_at=episode.finalized_at,
            coverage_complete=episode.coverage_complete,
            gaps=tuple(episode.capture_gaps),
            optional_gaps=tuple(episode.optional_gaps),
            stored_raw_bytes=episode.stored_raw_bytes,
            report_content_id=episode.report_content_id,
            report_sha256=episode.report_sha256,
            score_id=episode.score_id,
            score_status=ScoreStatus(stored_score.status) if stored_score else ScoreStatus.UNDETERMINED,
            turns=tuple(
                NativeCyberTurnSummary(
                    turn_index=turn.turn_index,
                    source_turn_id=turn.source_turn_id,
                    response_mode=turn.response_mode,
                    started_at=turn.started_at,
                    finished_at=turn.finished_at,
                    request_piece_ids=tuple(piece_ids[(turn.turn_index, "request")]),
                    response_piece_ids=tuple(piece_ids[(turn.turn_index, "response")]),
                    tool_request_piece_ids=tuple(piece_ids[(turn.turn_index, "tool_request")]),
                    tool_result_piece_ids=tuple(piece_ids[(turn.turn_index, "tool_result")]),
                    observed_event_count=turn.observed_event_count,
                    stored_event_count=event_counts[turn.turn_index],
                    source_complete=turn.source_complete is True,
                    gaps=tuple(turn.capture_gaps),
                )
                for turn in turns
            ),
            events=tuple(
                NativeCyberEventSummary(
                    sequence=event.sequence,
                    source=event.source,
                    observed_event_id=event.observed_event_id,
                    observed_session_id=event.observed_session_id,
                    observed_stream_id=event.observed_stream_id,
                    stream_offset=event.stream_offset,
                    tool_call_id=event.tool_call_id,
                    tool_phase=event.tool_phase,
                    event_type=event.event_type,
                    captured_at=event.captured_at,
                    payload_sha256=event.payload_sha256,
                )
                for event in events
            ),
            tools=tuple(
                NativeCyberToolCorrelation(
                    call_id=call_id,
                    request_sequence=phases.get("request"),
                    start_sequence=phases.get("start"),
                    completion_sequence=phases.get("complete"),
                    result_sequence=phases.get("result"),
                )
                for call_id, phases in sorted(tool_sequences.items())
            ),
            raw_streams=tuple(
                NativeCyberRawStreamSummary(
                    stream_id=stream.stream_id,
                    key=NativeCyberRawStreamKey(
                        source=stream.source,
                        kind=stream.kind,
                        observed_source_id=stream.observed_source_id,
                    ),
                    turn_index=stream.turn_index,
                    tool_call_id=stream.tool_call_id,
                    expected_bytes=stream.expected_bytes,
                    received_bytes=stream.received_bytes,
                    stored_bytes=stream.stored_bytes,
                    omitted_bytes=stream.received_bytes - stream.stored_bytes,
                    stored_sha256=stream.stored_sha256,
                    observed_sha256=stream.observed_sha256,
                    source_complete=stream.source_complete is True,
                    truncated=stream.truncated,
                    gaps=tuple(stream.capture_gaps),
                )
                for stream in streams
            ),
        )

    @staticmethod
    def _validate_report(*, episode: NativeCyberEpisodeEntry, report: NativeCyberReport) -> None:
        if (
            episode.run_id != report.run_id
            or episode.binding_name != report.binding_name
            or episode.binding_version != report.binding_version
            or episode.started_at.astimezone(UTC) != report.started_at.astimezone(UTC)
            or (episode.conversation_id is not None and episode.conversation_id != report.conversation_id)
            or (episode.simulated is not None and episode.simulated != report.simulated)
        ):
            raise ValueError("Canonical native report disagrees with the episode's immutable provenance.")
        if report.agent is not None and (
            (episode.source_session_id is not None and episode.source_session_id != report.agent.session_id)
            or (episode.environment_id is not None and episode.environment_id != report.agent.environment_id)
        ):
            raise ValueError("Canonical native report disagrees with the observed session or environment.")

    @staticmethod
    def _validate_score(*, report: NativeCyberReport, score: Score, unpersisted: bool) -> None:
        if score.score_type != "float_scale" or score.message_piece_id is not None or score.observation_ids:
            raise ValueError(
                "Native report scores must be float-scale content scores without message/observation links."
            )
        metadata = score.score_metadata or {}
        if metadata.get("run_id") != report.run_id or metadata.get("report_sha256") != report.sha256():
            raise ValueError("Native Score does not identify this canonical report and run.")
        scorable = score.scorable
        if unpersisted:
            if (
                not isinstance(scorable, ContentScorable)
                or scorable.data_type != "text"
                or scorable.value != report.canonical_json()
            ):
                raise ValueError("Unpersisted native Score must carry exactly the canonical report text.")
        elif not isinstance(scorable, ContentEntryScorable) or scorable.data_type != "text":
            raise ValueError("Native Score must already name persisted canonical report content.")
        if (
            score.status is ScoreStatus.COMPLETE
            and report.status is NativeCyberStatus.COMPLETED
            and report.judgment is not None
            and score.get_value() != report.judgment.value
        ):
            raise ValueError("Native Score value disagrees with the original retained judgment.")

    @classmethod
    def _capture_gaps(
        cls,
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        report: NativeCyberReport,
        expected_turns: int,
        phase: NativeCyberCoveragePhase,
    ) -> tuple[list[str], list[str]]:
        required = list(episode.capture_gaps)
        optional = list(episode.optional_gaps)
        if (
            phase is NativeCyberCoveragePhase.FINAL
            and report.agent is not None
            and (report.judgment is None or not report.judgment.complete)
        ):
            required.append("Original native judgment has not been completely acquired.")
        turns = list(
            session.scalars(
                select(NativeCyberTurnEntry)
                .where(NativeCyberTurnEntry.run_id == episode.run_id)
                .order_by(NativeCyberTurnEntry.turn_index)
            )
        )
        if [turn.turn_index for turn in turns] != list(range(1, expected_turns + 1)):
            required.append("Stored native outer turns differ from the declared run turn count.")
        for turn in turns:
            if turn.source_complete is not True or turn.finished_at is None:
                required.extend(turn.capture_gaps or ["A native outer turn was not completely captured."])
        pieces = list(
            session.scalars(
                select(NativeCyberTurnMessagePieceEntry).where(
                    NativeCyberTurnMessagePieceEntry.run_id == episode.run_id
                )
            )
        )
        if not cls._piece_links_intact(session=session, links=pieces):
            required.append("A linked native MessagePiece is missing or modified.")
        events = list(
            session.scalars(
                select(NativeCyberEventEntry)
                .where(NativeCyberEventEntry.run_id == episode.run_id)
                .order_by(NativeCyberEventEntry.sequence)
            )
        )
        required.extend(
            cls._turn_response_gaps(
                episode=episode,
                turns=turns,
                pieces=pieces,
                events=events,
                report=report,
                phase=phase,
            )
        )
        required.extend(cls._event_gaps(events=events, report=report))
        links = list(
            session.scalars(select(NativeCyberToolEventEntry).where(NativeCyberToolEventEntry.run_id == episode.run_id))
        )
        required.extend(cls._tool_gaps(events=events, links=links, report=report))
        required.extend(
            cls._tool_result_gaps(
                events=events,
                links=links,
                require_separate=episode.require_separate_tool_results,
            )
        )
        streams = list(
            session.scalars(select(NativeCyberRawStreamEntry).where(NativeCyberRawStreamEntry.run_id == episode.run_id))
        )
        raw_required, raw_optional = cls._raw_gaps(episode=episode, streams=streams, report=report)
        required.extend(raw_required)
        optional.extend(raw_optional)
        return list(dict.fromkeys(required)), list(dict.fromkeys(optional))

    @classmethod
    def _turn_response_gaps(
        cls,
        *,
        episode: NativeCyberEpisodeEntry,
        turns: Sequence[NativeCyberTurnEntry],
        pieces: Sequence[NativeCyberTurnMessagePieceEntry],
        events: Sequence[NativeCyberEventEntry],
        report: NativeCyberReport,
        phase: NativeCyberCoveragePhase,
    ) -> list[str]:
        directions: dict[int, set[str]] = defaultdict(set)
        events_by_turn: dict[int, list[NativeCyberEventEntry]] = defaultdict(list)
        for piece in pieces:
            directions[piece.turn_index].add(piece.direction)
        for event in events:
            events_by_turn[event.turn_index].append(event)
        gaps: list[str] = []
        for turn in turns:
            if "request" not in directions[turn.turn_index]:
                gaps.append("Native turn lacks a stored request MessagePiece.")
            if turn.response_mode == NativeCyberResponseMode.MESSAGE_REQUIRED.value:
                if "response" not in directions[turn.turn_index]:
                    gaps.append("Native chat turn lacks a genuine assistant response MessagePiece.")
            elif turn.response_mode == NativeCyberResponseMode.ARTIFACT_ONLY.value:
                if episode.response_policy_version != 1 or not episode.artifact_only_allowed:
                    gaps.append("Artifact-only native turn lacks trusted task response-policy approval.")
                if not cls._has_terminal_event(events=events_by_turn[turn.turn_index]):
                    gaps.append("Artifact-only native turn lacks an observed terminal source event.")
                if phase is NativeCyberCoveragePhase.FINAL and (
                    report.judgment is None or not report.judgment.complete or not report.judgment.artifacts
                ):
                    gaps.append("Artifact-only native turn lacks a complete judgment with retained artifact.")
            else:
                gaps.append("Native turn has an unsupported response mode.")
        return gaps

    @classmethod
    def _event_gaps(
        cls,
        *,
        events: Sequence[NativeCyberEventEntry],
        report: NativeCyberReport,
    ) -> list[str]:
        agent = report.agent
        if agent is None:
            return ["Native event rows exist without a report agent."] if events else []
        gaps = list(agent.gaps)
        if not agent.coverage_complete or not agent.idle:
            gaps.append("Native event source did not confirm complete idle coverage.")
        if len(agent.events) != len(events) or [event.sequence for event in events] != list(range(1, len(events) + 1)):
            gaps.append("Retained native events do not match the canonical report's event count or order.")
            return gaps
        for stored, observed in zip(events, agent.events, strict=True):
            digest = cls._hash_payload(observed.payload)
            if (
                stored.sequence != observed.sequence
                or stored.observed_event_id != observed.event_id
                or stored.observed_session_id != observed.session_id
                or stored.event_type != observed.event_type
                or stored.payload_sha256 != digest
                or stored.payload_sha256 != cls._hash_payload(stored.payload)
            ):
                gaps.append("Retained native event identity or payload differs from the canonical report.")
                break
        return gaps

    @staticmethod
    def _tool_gaps(
        *,
        events: Sequence[NativeCyberEventEntry],
        links: Sequence[NativeCyberToolEventEntry],
        report: NativeCyberReport,
    ) -> list[str]:
        agent = report.agent
        if agent is None:
            return ["Native tool links exist without a report agent."] if links else []
        phases = {(link.call_id, link.phase): link.event_sequence for link in links}
        request_ids = {request.call_id for request in agent.tool_requests}
        trace_ids = {tool.call_id for tool in agent.tools}
        gaps: list[str] = []
        if (
            {link.call_id for link in links} != request_ids | trace_ids
            or request_ids != trace_ids
            or len(request_ids) != len(agent.tool_requests)
            or len(trace_ids) != len(agent.tools)
        ):
            gaps.append("Observed native tool calls differ from the canonical report.")
        events_by_sequence = {event.sequence: event for event in events}
        for request in agent.tool_requests:
            event = events_by_sequence.get(request.request_sequence)
            data = event.payload.get("data") if event is not None else None
            tool_requests = data.get("toolRequests") if isinstance(data, dict) else None
            matches_request = isinstance(tool_requests, list) and any(
                isinstance(item, dict)
                and item.get("toolCallId") == request.call_id
                and item.get("name") == request.name
                and item.get("arguments") == request.arguments
                for item in tool_requests
            )
            if phases.get((request.call_id, "request")) != request.request_sequence or not matches_request:
                gaps.append("Native model tool request has no matching observed event.")
        for tool in agent.tools:
            request_seq = phases.get((tool.call_id, "request"))
            start_seq = phases.get((tool.call_id, "start"))
            complete_seq = phases.get((tool.call_id, "complete"))
            if (
                request_seq != tool.request_sequence
                or start_seq != tool.start_sequence
                or complete_seq != tool.completion_sequence
                or complete_seq is None
                or request_seq is None
                or not request_seq < start_seq < complete_seq
            ):
                gaps.append("Native tool request, start and completion are not correlated in order.")
                continue
            start = events_by_sequence.get(start_seq)
            completion = events_by_sequence.get(complete_seq)
            start_data = start.payload.get("data") if start is not None else None
            complete_data = completion.payload.get("data") if completion is not None else None
            if (
                not isinstance(start_data, dict)
                or start_data.get("toolCallId") != tool.call_id
                or start_data.get("toolName") != tool.name
                or start_data.get("arguments") != tool.arguments
                or not isinstance(complete_data, dict)
                or complete_data.get("toolCallId") != tool.call_id
                or complete_data.get("success") != tool.success
                or complete_data.get("result") != tool.result
                or complete_data.get("error") != tool.error
            ):
                gaps.append("Native tool trace disagrees with its observed execution events.")
        event_sequences = {event.sequence for event in events}
        if any(link.event_sequence not in event_sequences for link in links):
            gaps.append("A native tool link has no retained source event.")
        return gaps

    @staticmethod
    def _tool_result_gaps(
        *,
        events: Sequence[NativeCyberEventEntry],
        links: Sequence[NativeCyberToolEventEntry],
        require_separate: bool,
    ) -> list[str]:
        phases: dict[str, dict[str, int]] = defaultdict(dict)
        for link in links:
            phases[link.call_id][link.phase] = link.event_sequence
        by_sequence = {event.sequence: event for event in events}
        gaps: list[str] = []
        for call_id, observed in phases.items():
            result_sequence = observed.get("result")
            if result_sequence is None:
                if require_separate:
                    gaps.append("A task-required model-visible tool result was not captured.")
                continue
            predecessor = observed.get("complete")
            if predecessor is None:
                predecessor = observed.get("request")
            if predecessor is None or predecessor >= result_sequence:
                gaps.append("A native tool result did not follow its completion or request.")
            result = by_sequence.get(result_sequence)
            if result is None or result.tool_call_id != call_id or result.tool_phase != "result":
                gaps.append("A native tool result does not match its observed source event.")
        return gaps

    @staticmethod
    def _raw_gaps(
        *,
        episode: NativeCyberEpisodeEntry,
        streams: Sequence[NativeCyberRawStreamEntry],
        report: NativeCyberReport,
    ) -> tuple[list[str], list[str]]:
        expected = {(key["source"], key["kind"], key["observed_source_id"]) for key in episode.required_raw_streams}
        found = {(stream.source, stream.kind, stream.observed_source_id) for stream in streams}
        required: list[str] = []
        optional: list[str] = []
        if report.agent is not None and not expected:
            required.append("No task-required raw source streams were declared for a native run.")
        if expected - found:
            required.append("One or more task-required native raw streams were never opened.")
        if sum(stream.stored_bytes for stream in streams) != episode.stored_raw_bytes:
            required.append("Native raw byte totals disagree with the episode quota ledger.")
        observed_tool_ids = {tool.call_id for tool in report.agent.tools} if report.agent else set()
        for stream in streams:
            issues = list(stream.capture_gaps)
            if stream.source_complete is not True or stream.closed_at is None:
                issues.append("Native raw stream was not sealed with complete source coverage.")
            if stream.truncated or stream.received_bytes != stream.stored_bytes:
                issues.append("Native raw bytes were omitted by the run quota.")
            if stream.tool_call_id is not None and stream.tool_call_id not in observed_tool_ids:
                issues.append("Native raw tool stream has no observed matching tool execution.")
            if issues:
                target = required if (stream.source, stream.kind, stream.observed_source_id) in expected else optional
                target.extend(issues)
        return required, optional

    @staticmethod
    def _score_for_capture(
        *,
        score: Score,
        report: NativeCyberReport,
        required_complete: bool,
        gap_count: int,
    ) -> Score:
        metadata = dict(score.score_metadata or {})
        metadata["native_required_capture"] = "complete" if required_complete else "incomplete"
        metadata["native_required_gap_count"] = gap_count
        cannot_determine = (
            not required_complete
            or report.status is not NativeCyberStatus.COMPLETED
            or report.judgment is None
            or not report.judgment.complete
        )
        rationale = score.score_rationale
        if not required_complete:
            rationale = (
                f"Task-required native capture is incomplete ({gap_count} gap(s)); original judgment is retained."
            )
        return Score.model_validate(
            {
                **score.model_dump(exclude={"objective"}),
                "score_value": None if cannot_determine else score.score_value,
                "status": ScoreStatus.UNDETERMINED if cannot_determine else score.status,
                "score_rationale": rationale,
                "score_metadata": metadata,
            }
        )

    def _insert_atomic_score(
        self,
        *,
        session: Session,
        report: NativeCyberReport,
        score: Score,
    ) -> tuple[uuid.UUID, Score]:
        stored = self._insert_prepared_score_row(session=session, score=score)
        self._validate_stored_score(session=session, stored=stored, report=report)
        if stored.scorable_content_id is None:
            raise ValueError("Atomic native finalization did not retain its report content.")
        return stored.scorable_content_id, stored.get_score()

    def _insert_prepared_score_row(self, *, session: Session, score: Score) -> ScoreEntry:
        """
        Use the existing generic content/Score writers inside the caller's transaction.

        Returns:
            ScoreEntry: The stored generic Score with one report-content anchor.

        Raises:
            ValueError: If the Score ID or its canonical content is already persisted.
        """
        score_id = uuid.UUID(str(score.id))
        if session.get(ScoreEntry, score_id) is not None:
            raise ValueError("Atomic native finalization requires an unpersisted Score ID.")
        content, anchored, observations = self._memory._prepare_score_anchors(
            scores=[score],
            observations=(),
            prepared_content_hashes={},
        )
        if len(content) != 1 or len(anchored) != 1 or observations:
            raise ValueError("Atomic native finalization requires exactly one canonical report and Score.")
        session.add_all(content)
        session.flush()
        self._memory._persist_score_rows(session=session, scores=anchored, observations=[])
        session.flush()
        stored = session.get(ScoreEntry, score_id)
        if stored is None or stored.scorable_content_id is None:
            raise ValueError("Atomic native finalization did not persist its report and Score.")
        return stored

    def _link_score(
        self,
        *,
        session: Session,
        report: NativeCyberReport,
        score: Score,
    ) -> tuple[uuid.UUID, Score]:
        score_id = uuid.UUID(str(score.id))
        stored = session.get(ScoreEntry, score_id)
        if stored is None:
            raise ValueError("Native finalization requires the scorer's already-persisted Score.")
        if (
            not isinstance(score.scorable, ContentEntryScorable)
            or stored.scorable_content_id != score.scorable.content_id
        ):
            raise ValueError("Native Score anchor differs from the scorer's persisted report content.")
        self._validate_stored_score(session=session, stored=stored, report=report)
        if score.status is ScoreStatus.UNDETERMINED and stored.status == ScoreStatus.COMPLETE.value:
            stored.score_value = None
            stored.status = ScoreStatus.UNDETERMINED.value
        if score.status is ScoreStatus.UNDETERMINED:
            stored.score_rationale = score.score_rationale
            stored.score_metadata = score.score_metadata or {}
        session.flush()
        if stored.scorable_content_id is None:
            raise ValueError("Native Score has no durable report-content anchor.")
        return stored.scorable_content_id, stored.get_score()

    @staticmethod
    def _validate_stored_score(*, session: Session, stored: ScoreEntry, report: NativeCyberReport) -> None:
        content = session.get(ScorableContentEntry, stored.scorable_content_id) if stored.scorable_content_id else None
        if (
            content is None
            or content.data_type != "text"
            or content.value != report.canonical_json()
            or content.value_sha256 != report.sha256()
            or stored.score_type != "float_scale"
            or stored.prompt_request_response_id is not None
            or stored.scorable is None
            or stored.scorable.get("scorable_type") != "content_entry"
            or stored.scorable.get("content_id") != str(content.id)
            or (stored.score_metadata or {}).get("run_id") != report.run_id
            or (stored.score_metadata or {}).get("report_sha256") != report.sha256()
        ):
            raise ValueError("Stored native Score or content does not match the canonical report.")
        if (
            stored.status == ScoreStatus.COMPLETE.value
            and report.status is NativeCyberStatus.COMPLETED
            and (
                report.judgment is None
                or stored.score_value is None
                or float(stored.score_value) != report.judgment.value
            )
        ):
            raise ValueError("Stored native Score value differs from the original retained judgment.")

    @contextmanager
    def _write_session(
        self,
        *,
        run_id: str,
        raw_key: NativeCyberRawStreamKey | None = None,
        stream_id: uuid.UUID | None = None,
    ) -> Iterator[Session]:
        try:
            with closing(self._memory.get_session()) as session, session.begin():
                _begin_sqlite_write(session)
                yield session
        except SQLAlchemyError as error:
            logger.error("Native evidence database write failed for run %s (%s).", run_id, type(error).__name__)
            self._mark_failed_write(run_id=run_id, raw_key=raw_key, stream_id=stream_id)
            raise

    def _mark_failed_write(
        self,
        *,
        run_id: str,
        raw_key: NativeCyberRawStreamKey | None,
        stream_id: uuid.UUID | None,
    ) -> None:
        try:
            with closing(self._memory.get_session()) as session, session.begin():
                _begin_sqlite_write(session)
                episode = session.get(NativeCyberEpisodeEntry, run_id)
                if episode is not None and episode.finalized_at is None:
                    if stream_id is not None:
                        stream = session.get(NativeCyberRawStreamEntry, stream_id)
                        if stream is not None and stream.run_id == run_id:
                            raw_key = NativeCyberRawStreamKey(
                                source=stream.source, kind=stream.kind, observed_source_id=stream.observed_source_id
                            )
                    expected = {
                        (key["source"], key["kind"], key["observed_source_id"]) for key in episode.required_raw_streams
                    }
                    key_tuple = (
                        (raw_key.source.value, raw_key.kind.value, raw_key.observed_source_id) if raw_key else None
                    )
                    if key_tuple is not None and key_tuple not in expected:
                        episode.optional_gaps = list(dict.fromkeys([*episode.optional_gaps, self._WRITE_FAILURE_GAP]))
                    else:
                        episode.capture_gaps = list(dict.fromkeys([*episode.capture_gaps, self._WRITE_FAILURE_GAP]))
        except SQLAlchemyError as error:
            logger.error("Could not record native evidence write failure (%s).", type(error).__name__)

    @staticmethod
    def _lock_episode(*, session: Session, run_id: str) -> NativeCyberEpisodeEntry:
        statement = select(NativeCyberEpisodeEntry).where(NativeCyberEpisodeEntry.run_id == run_id)
        if session.get_bind().dialect.name == "mssql":
            statement = statement.with_hint(NativeCyberEpisodeEntry, "WITH (UPDLOCK, HOLDLOCK)", dialect_name="mssql")
        episode = session.scalar(statement)
        if episode is None:
            raise KeyError(f"Native cyber episode {run_id} does not exist.")
        return episode

    @staticmethod
    def _require_open(episode: NativeCyberEpisodeEntry) -> None:
        if episode.finalized_at is not None:
            raise ValueError(f"Native cyber episode {episode.run_id} has already been finalized.")

    @classmethod
    def _open_turn(
        cls,
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        turn_index: int,
    ) -> NativeCyberTurnEntry:
        cls._require_open(episode)
        turn = session.get(NativeCyberTurnEntry, (episode.run_id, turn_index))
        if turn is None or turn.finished_at is not None:
            raise ValueError(f"Native outer turn {turn_index} is missing or already sealed.")
        return turn

    @classmethod
    def _open_stream(
        cls,
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        stream_id: uuid.UUID,
    ) -> NativeCyberRawStreamEntry:
        cls._require_open(episode)
        stream = session.get(NativeCyberRawStreamEntry, stream_id)
        if stream is None or stream.run_id != episode.run_id or stream.closed_at is not None:
            raise ValueError(f"Native raw stream {stream_id} is missing, foreign or already sealed.")
        if stream.turn_index is not None:
            cls._open_turn(session=session, episode=episode, turn_index=stream.turn_index)
        return stream

    @staticmethod
    def _link_pieces(
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        turn_index: int,
        direction: str,
        piece_ids: Sequence[uuid.UUID],
    ) -> None:
        if len(piece_ids) != len(set(piece_ids)):
            raise ValueError("A native outer turn cannot reference a MessagePiece twice.")
        valid_roles = {
            "request": {"user", "system", "developer"} if episode.binding_name == "inspect-original" else {"user"},
            "response": {"assistant", "simulated_assistant"},
            "tool_request": {"assistant", "simulated_assistant"},
            "tool_result": {"tool"},
        }.get(direction)
        if valid_roles is None:
            raise ValueError(f"Unknown native MessagePiece direction: {direction}.")
        for position, piece_id in enumerate(piece_ids):
            piece = session.get(PromptMemoryEntry, piece_id)
            if piece is None or piece.role not in valid_roles or not piece.conversation_id:
                raise ValueError(f"Native {direction} MessagePiece {piece_id} is absent or has the wrong role.")
            tool_request_type = piece.original_value_data_type in {"function_call", "tool_call"}
            if (
                (
                    direction == "response"
                    and piece.original_value_data_type in {"function_call", "tool_call", "function_call_output"}
                )
                or (direction == "tool_request" and not tool_request_type)
                or (direction == "tool_result" and piece.original_value_data_type != "function_call_output")
            ):
                raise ValueError(f"Native {direction} MessagePiece {piece_id} has the wrong data type.")
            if episode.binding_name != "inspect-original":
                if episode.conversation_id is None:
                    episode.conversation_id = piece.conversation_id
                elif episode.conversation_id != piece.conversation_id:
                    raise ValueError("Native message pieces must belong to one persisted conversation.")
            session.add(
                NativeCyberTurnMessagePieceEntry(
                    run_id=episode.run_id,
                    turn_index=turn_index,
                    direction=direction,
                    position=position,
                    message_piece_id=piece_id,
                    piece_sha256=_message_piece_digest(piece.get_message_piece(), include_id=True),
                )
            )

    @staticmethod
    def _has_terminal_event(*, events: Sequence[NativeCyberEventEntry]) -> bool:
        last_action = max((event.sequence for event in events if event.source in {"model", "tool"}), default=0)
        terminal = max(
            (event.sequence for event in events if NativeCyberEvidenceStore._is_terminal_event(event)),
            default=None,
        )
        return terminal is not None and terminal >= last_action

    @staticmethod
    def _is_terminal_event(event: NativeCyberEventEntry) -> bool:
        if event.source != "harness":
            return False
        payload = event.payload
        if event.event_type in {"session.idle", "turn.completed"}:
            data = payload.get("data")
            return not (
                (event.event_type == "session.idle" and payload.get("agentId"))
                or (isinstance(data, dict) and data.get("aborted") is True)
            )
        if event.event_type not in {"native_cli.turn_completed", "native_cli.run_finished"}:
            return False
        frame_number = payload.get("frame_number")
        status = payload.get("status")
        if type(frame_number) is not int or not isinstance(status, str) or status != "completed":
            return False
        if event.event_type == "native_cli.run_finished":
            source_status = payload.get("source_status")
            if not isinstance(source_status, str) or source_status != "success":
                return False
        return frame_number > 0 and event.observed_stream_id is not None and event.stream_offset is not None

    @staticmethod
    def _piece_links_intact(*, session: Session, links: Sequence[NativeCyberTurnMessagePieceEntry]) -> bool:
        if not links:
            return True
        pieces: dict[uuid.UUID, PromptMemoryEntry] = {}
        ids = [link.message_piece_id for link in links]
        for start in range(0, len(ids), 500):
            rows = session.scalars(select(PromptMemoryEntry).where(PromptMemoryEntry.id.in_(ids[start : start + 500])))
            pieces.update((piece.id, piece) for piece in rows)
        return all(
            (piece := pieces.get(link.message_piece_id)) is not None
            and _message_piece_digest(piece.get_message_piece(), include_id=True) == link.piece_sha256
            for link in links
        )

    @staticmethod
    def _check_event_origin(*, episode: NativeCyberEpisodeEntry, captured: NativeCyberCapturedEvent) -> None:
        event = captured.event
        if (
            episode.source_session_id is not None
            and event.source_session_id is not None
            and event.source_session_id != episode.source_session_id
        ):
            raise ValueError("Native event belongs to a different observed session.")
        if (
            (
                event.source_event_id is not None
                and event.payload.get("id", event.source_event_id) != event.source_event_id
            )
            or event.payload.get("type", event.event_type) != event.event_type
            or (
                event.source_session_id is not None
                and event.payload.get("sessionId", event.source_session_id) != event.source_session_id
            )
        ):
            raise ValueError("Native event identifiers disagree with the captured source payload.")
        expected_source = None
        if event.event_type.startswith("assistant."):
            expected_source = NativeCyberEvidenceSource.MODEL
        elif event.event_type.startswith("tool."):
            expected_source = NativeCyberEvidenceSource.TOOL
        elif event.event_type.startswith("session."):
            expected_source = NativeCyberEvidenceSource.HARNESS
        if expected_source is not None and captured.source is not expected_source:
            raise ValueError(f"Native event {event.sequence} has a mismatched source tag.")

    @classmethod
    def _hash_payload(cls, payload: Mapping[str, object]) -> str:
        encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > cls.MAX_EVENT_BYTES:
            raise ValueError("Native event payload exceeds the bounded capture size.")
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _extract_tool_phases(*, event: NativeCyberObservedEvent) -> tuple[list[tuple[str, str]], list[str]]:
        if event.tool_call_id is not None and event.tool_phase is not None:
            return [(event.tool_call_id, event.tool_phase.value)], []
        data = event.payload.get("data")
        if event.event_type not in {"assistant.message", "tool.execution_start", "tool.execution_complete"}:
            return [], []
        if not isinstance(data, dict):
            return [], [f"Native event {event.sequence} has no structured tool data."]
        if event.event_type == "assistant.message":
            requests = data.get("toolRequests") or []
            if not isinstance(requests, list):
                return [], [f"Native event {event.sequence} has malformed tool requests."]
            ids = [request.get("toolCallId") if isinstance(request, dict) else None for request in requests]
            phase = "request"
        else:
            ids = [data.get("toolCallId")]
            phase = "start" if event.event_type == "tool.execution_start" else "complete"
        links: list[tuple[str, str]] = []
        gaps: list[str] = []
        for call_id in ids:
            if not isinstance(call_id, str) or not 0 < len(call_id) <= 128:
                gaps.append(f"Native event {event.sequence} has an invalid tool call ID.")
            else:
                links.append((call_id, phase))
        return links, gaps

    @classmethod
    def _hash_stream(cls, *, session: Session, stream: NativeCyberRawStreamEntry) -> str:
        digest = hashlib.sha256()
        expected_offset = 0
        statement = (
            select(NativeCyberRawChunkEntry)
            .where(NativeCyberRawChunkEntry.stream_id == stream.stream_id)
            .order_by(NativeCyberRawChunkEntry.sequence)
        )
        for expected_sequence, chunk in enumerate(
            session.scalars(statement).yield_per(cls.MAX_RAW_READ_CHUNKS), start=1
        ):
            if (
                chunk.sequence != expected_sequence
                or chunk.byte_offset != expected_offset
                or not 0 < chunk.byte_length <= cls.MAX_CHUNK_BYTES
                or chunk.byte_length != len(chunk.data)
                or chunk.sha256 != hashlib.sha256(chunk.data).hexdigest()
            ):
                raise ValueError(f"Native raw stream {stream.stream_id} has a missing or corrupt byte range.")
            digest.update(chunk.data)
            expected_offset += chunk.byte_length
        if expected_offset != stream.stored_bytes:
            raise ValueError(f"Native raw stream {stream.stream_id} length disagrees with stored chunks.")
        return digest.hexdigest()
