# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provider-neutral episode rows reused for an independent Inspect-owned task."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from contextlib import closing
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import select

from pyrit.memory.memory_models import (
    NativeCyberEpisodeEntry,
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    ScorableContentEntry,
    ScoreEntry,
)
from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore
from pyrit.models import ContentScorable, Score, ScoreStatus
from pyrit.models.inspect_ghcp import InspectGhcpReport, InspectGhcpStatus
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberEpisodeStart,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberTurnFinish,
    NativeCyberTurnStart,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pyrit.memory import MemoryInterface
    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot, NativeCyberToolCorrelation
    from pyrit.prompt_target.inspect_ghcp_target import InspectGhcpTurn


class InspectGhcpEvidenceStore:
    """Use existing generic event/turn/raw rows without importing the native evaluator."""

    SDK_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="inspect-ghcp-sdk-events",
    )
    MODEL_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.MODEL,
        kind=NativeCyberRawKind.MODEL,
        observed_source_id="inspect-ghcp-model-gateway",
    )
    HOST_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.MODEL,
        kind=NativeCyberRawKind.MODEL,
        observed_source_id="inspect-trusted-host-model-http",
    )
    ADVERSARIAL_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.MODEL,
        kind=NativeCyberRawKind.MODEL,
        observed_source_id="pyrit-adversarial-model-http",
    )
    CONTROL_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="inspect-private-control-receipts",
    )
    LOG_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="inspect-original-eval-log",
    )

    def __init__(
        self,
        *,
        memory: MemoryInterface,
        run_id: str,
        task_name: str,
        task_version: str,
        started_at: datetime,
        raw_byte_limit: int,
    ) -> None:
        """Create immutable episode provenance and the five pregrading source streams."""
        self._memory = memory
        self._capture: NativeCyberEvidenceStore = memory.native_cyber_evidence
        self.run_id = run_id
        self._recovery_only = False
        self._sequence = 0
        self._turn_count = 0
        self._sdk_bytes = bytearray()
        self._gateway_bytes = b""
        self._host_bytes = bytearray()
        self._host_count = 0
        self._host_success_count = 0
        self._adversarial_bytes = bytearray()
        self._adversarial_count = 0
        self._adversarial_success_count = 0
        self._adversarial_event_count = 0
        self._adversarial_pending: set[str] = set()
        self._control_bytes = bytearray()
        self._control_seen: set[str] = set()
        self._control_token_sha: str | None = None
        self._log_bytes = b""
        self._stderr_recorded = False
        self._capture.create_episode(
            start=NativeCyberEpisodeStart(
                run_id=run_id,
                binding_name="inspect-ghcp",
                binding_version="3",
                task_id=task_name,
                task_version=task_version,
                started_at=started_at,
                simulated=False,
                required_raw_streams=(
                    self.SDK_KEY,
                    self.MODEL_KEY,
                    self.HOST_KEY,
                    self.ADVERSARIAL_KEY,
                    self.CONTROL_KEY,
                ),
                raw_byte_limit=raw_byte_limit,
            )
        )
        self._sdk_stream = NativeCyberRawStreamStart(run_id=run_id, key=self.SDK_KEY)
        self._model_stream = NativeCyberRawStreamStart(run_id=run_id, key=self.MODEL_KEY)
        self._host_stream = NativeCyberRawStreamStart(run_id=run_id, key=self.HOST_KEY)
        self._adversarial_stream = NativeCyberRawStreamStart(run_id=run_id, key=self.ADVERSARIAL_KEY)
        self._control_stream = NativeCyberRawStreamStart(run_id=run_id, key=self.CONTROL_KEY)
        self._capture.open_raw_stream(stream=self._sdk_stream)
        self._capture.open_raw_stream(stream=self._model_stream)
        self._capture.open_raw_stream(stream=self._host_stream)
        self._capture.open_raw_stream(stream=self._adversarial_stream)
        self._capture.open_raw_stream(stream=self._control_stream)

    def record_control_receipt(self, *, receipt: dict[str, Any]) -> None:
        """
        Retain only SHA/size/identity of a secret-elided Inspect control frame.

        Raises:
            ValueError: If a receipt contains raw content or disagrees with its source.
        """
        self._require_live_capture()
        expected = {
            "run_id",
            "service",
            "source_id",
            "observed_job_id",
            "container_id",
            "frame_size_bytes",
            "frame_sha256",
            "completed_exit_code",
            "inspect_raw_control_elided",
            "provenance",
        }
        if set(receipt) != expected:
            raise ValueError("A private Inspect control receipt has an unsafe or missing field.")
        service, pid, digest = receipt["service"], receipt["observed_job_id"], receipt["frame_sha256"]
        if (
            receipt["run_id"] != self.run_id
            or not isinstance(service, str)
            or service not in {"agent", "model-bridge"}
            or service in self._control_seen
            or type(pid) is not int
            or pid < 1
            or receipt["source_id"] != f"{self.run_id}:{service}:{pid}"
            or not isinstance(receipt["container_id"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", receipt["container_id"])
            or type(receipt["frame_size_bytes"]) is not int
            or receipt["frame_size_bytes"] != 43
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or (self._control_token_sha is not None and digest != self._control_token_sha)
            or type(receipt["completed_exit_code"]) is not int
            or receipt["completed_exit_code"] != 0
            or receipt["inspect_raw_control_elided"] is not True
            or receipt["provenance"] != "Inspect as_type Docker provider; owner-only tmpfs token file removed on read"
        ):
            raise ValueError("Run-scoped Inspect control proof differs from its observed one-shot source.")
        encoded = self._event_bytes(frame=receipt)
        self._append_raw(stream_id=self._control_stream.stream_id, data=encoded)
        self._control_bytes.extend(encoded)
        self._control_seen.add(service)
        self._control_token_sha = digest

    def seal_control_receipts(self) -> str:
        """
        Require both independently completed private controller handoffs before grading.

        Returns:
            str: The digest of token-free, source-identified control receipts.

        Raises:
            ValueError: If either bounded one-shot handoff was not observed.
        """
        self._require_live_capture()
        if self._control_seen != {"agent", "model-bridge"}:
            raise ValueError("Inspect agent and bridge bootstrap receipts must both be observed.")
        content = bytes(self._control_bytes)
        self._seal_raw(stream_id=self._control_stream.stream_id, data=content)
        return hashlib.sha256(content).hexdigest()

    @classmethod
    def open_pending_for_recovery(cls, *, memory: MemoryInterface, run_id: str) -> InspectGhcpEvidenceStore:
        """
        Reopen source rows without creating another episode or replaying any provider work.

        Returns:
            InspectGhcpEvidenceStore: Finalization-only access to the existing run.

        Raises:
            ValueError: If the run is absent or already finalized.
        """
        capture = memory.native_cyber_evidence
        snapshot = capture.get_episode(run_id=run_id)
        if snapshot.finalized_at is not None:
            raise ValueError("This Inspect run already has an immutable published outcome.")
        recovered = object.__new__(cls)
        recovered._memory = memory
        recovered._capture = capture
        recovered.run_id = run_id
        recovered._recovery_only = True
        return recovered

    @classmethod
    def open_finalized_for_readback(cls, *, memory: MemoryInterface, run_id: str) -> InspectGhcpEvidenceStore:
        """
        Reassess sealed original sources without opening another evidence episode.

        Returns:
            InspectGhcpEvidenceStore: Read-only access to the finalized run.

        Raises:
            ValueError: If the episode is not a finalized Inspect run.
        """
        snapshot = memory.native_cyber_evidence.get_finalized_episode(run_id=run_id)
        if snapshot.run.binding_name != "inspect-ghcp":
            raise ValueError("This finalized episode does not belong to the original Inspect Task.")
        recovered = object.__new__(cls)
        recovered._memory = memory
        recovered._capture = memory.native_cyber_evidence
        recovered.run_id = run_id
        recovered._recovery_only = True
        return recovered

    def bind_agent(self, *, container_id: str, session_id: str) -> None:
        """
        Bind the actual Inspect container and GHCP session before the first turn.

        Raises:
            ValueError: If the run is missing, finalized, or already bound.
        """
        self._require_live_capture()
        if not container_id or not session_id:
            raise ValueError("Observed Inspect agent and GHCP session IDs are required.")
        with closing(self._memory.get_session()) as session, session.begin():
            episode = session.get(NativeCyberEpisodeEntry, self.run_id)
            if episode is None or episode.finalized_at is not None:
                raise ValueError("The Inspect episode is missing or finalized.")
            if episode.environment_id is not None or episode.source_session_id is not None:
                raise ValueError("Inspect agent identity may be bound only once.")
            episode.environment_id = container_id
            episode.source_session_id = session_id

    def record_turn(self, *, turn: InspectGhcpTurn) -> None:
        """
        Link real PyRIT pieces and preserve the SDK's ordered original event frames.

        Raises:
            ValueError: If a source ID, response, turn order or event stream disagrees.
        """
        self._require_live_capture()
        if turn.turn_index != self._turn_count + 1 or not turn.events:
            raise ValueError("Inspect GHCP evidence requires consecutive, nonempty source turns.")
        self._capture.begin_turn(
            turn=NativeCyberTurnStart(
                run_id=self.run_id,
                turn_index=turn.turn_index,
                source_turn_id=str(turn.turn_index),
                request_piece_ids=(turn.request_piece_id,),
            )
        )
        captured = [self._captured_event(frame=frame, session_id=turn.session_id) for frame in turn.events]
        for start in range(0, len(captured), self._capture.MAX_EVENT_BATCH):
            self._capture.append_events(
                run_id=self.run_id,
                turn_index=turn.turn_index,
                events=captured[start : start + self._capture.MAX_EVENT_BATCH],
            )
        raw = b"".join(self._event_bytes(frame=frame) for frame in turn.events)
        self._sdk_bytes.extend(raw)
        self._append_raw(stream_id=self._sdk_stream.stream_id, data=raw)
        last_action = max(
            (item.event.sequence for item in captured if item.source is not NativeCyberEvidenceSource.HARNESS),
            default=0,
        )
        last_idle = max(
            (
                item.event.sequence
                for item in captured
                if item.event.event_type == "session.idle" and not item.event.payload.get("agentId")
            ),
            default=0,
        )
        matching_response = any(
            self._is_matching_response(frame=frame, text=turn.assistant_text) for frame in turn.events
        )
        source_complete = bool(last_idle >= last_action and matching_response)
        self._capture.finish_turn(
            finish=NativeCyberTurnFinish(
                run_id=self.run_id,
                turn_index=turn.turn_index,
                response_piece_ids=(turn.response_piece_id,),
                observed_event_count=len(captured),
                source_complete=source_complete,
                gaps=() if source_complete else ("Missing observed root idle or matching assistant message.",),
            )
        )
        self._turn_count += 1

    def seal_sdk_events(self) -> None:
        """Seal all serialized SDK frames against their actual byte digest."""
        self._require_live_capture()
        self._seal_raw(stream_id=self._sdk_stream.stream_id, data=bytes(self._sdk_bytes))

    def record_gateway_audit(self, *, records: Sequence[dict[str, Any]], turns: Sequence[InspectGhcpTurn]) -> str:
        """
        Cross-check the separate bridge's actual HTTP bytes against guest SDK records.

        Returns:
            str: SHA256 of the retained original gateway audit bytes.

        Raises:
            ValueError: If source IDs, bytes, status, or response completeness disagree.
        """
        self._require_live_capture()
        exchanges = [exchange for turn in turns for exchange in turn.model_exchanges]
        if not records or len(records) != len(exchanges):
            raise ValueError("Gateway request count differs from the guest SDK event source.")
        for sequence, (record, exchange) in enumerate(zip(records, exchanges, strict=True), start=1):
            request = self._decode(record=record, name="request_base64")
            response = self._decode(record=record, name="response_base64")
            if (
                record.get("sequence") != sequence
                or record.get("run_id") != self.run_id
                or record.get("source_request_id") != exchange.get("request_id")
                or exchange.get("source_session_id") != turns[0].session_id
                or record.get("request_sha256") != hashlib.sha256(request).hexdigest()
                or record.get("response_sha256") != hashlib.sha256(response).hexdigest()
                or exchange.get("request_base64") != record["request_base64"]
                or exchange.get("response_base64") != record["response_base64"]
                or exchange.get("status") != record.get("response_status")
            ):
                raise ValueError("Guest model traffic differs from the separate authenticated bridge audit.")
            if record.get("error") or record.get("response_status") != 200 or exchange.get("error"):
                self._capture.mark_capture_gap(
                    run_id=self.run_id, reason="The model gateway rejected or failed a GHCP model request."
                )
        self._gateway_bytes = b"".join(self._event_bytes(frame=record) for record in records)
        self._append_raw(stream_id=self._model_stream.stream_id, data=self._gateway_bytes)
        self._seal_raw(stream_id=self._model_stream.stream_id, data=self._gateway_bytes)
        return hashlib.sha256(self._gateway_bytes).hexdigest()

    def record_host_model_exchange(
        self, *, request: bytes, response: bytes | None, status: int | None, error: str | None
    ) -> None:
        """
        Capture exact trusted-host provider bytes and explicit transport rejections.

        Raises:
            ValueError: If the provider request body is absent.
        """
        self._require_live_capture()
        if not request:
            raise ValueError("An Inspect model call has no original request body.")
        self._host_count += 1
        self._host_success_count += int(status == 200 and response is not None and error is None)
        row = {
            "sequence": self._host_count,
            "request_sha256": hashlib.sha256(request).hexdigest(),
            "request_base64": base64.b64encode(request).decode("ascii"),
            "response_sha256": hashlib.sha256(response).hexdigest() if response is not None else None,
            "response_base64": base64.b64encode(response).decode("ascii") if response is not None else None,
            "status": status,
            "error": error,
        }
        encoded = self._event_bytes(frame=row)
        self._host_bytes.extend(encoded)
        self._append_raw(stream_id=self._host_stream.stream_id, data=encoded)
        if status != 200 or response is None or error is not None:
            self._capture.mark_capture_gap(
                run_id=self.run_id, reason="Trusted host model request failed or lacked original response bytes."
            )

    def seal_host_model(self) -> str:
        """
        Seal the host-side provider calls after the agent stops.

        Returns:
            str: SHA256 of the complete run-scoped host model HTTP audit.
        """
        self._require_live_capture()
        content = bytes(self._host_bytes)
        self._seal_raw(stream_id=self._host_stream.stream_id, data=content)
        return hashlib.sha256(content).hexdigest()

    @property
    def host_model_counts(self) -> tuple[int, int]:
        """Original trusted-host HTTP request count and successful 200 responses."""
        return self._host_count, self._host_success_count

    def record_adversarial_model_frame(
        self, *, request_id: str, phase: str, body: bytes, status: int | None, error: str | None
    ) -> None:
        """
        Retain actual PyRIT adversarial-model request and response bytes by source ID.

        Raises:
            ValueError: If an original source frame is missing, duplicated, or out of order.
        """
        self._require_live_capture()
        if not request_id or phase not in {"request", "response", "error"}:
            raise ValueError("PyRIT adversarial source frames require an ID and observed phase.")
        if phase == "request":
            if not body or request_id in self._adversarial_pending:
                raise ValueError("PyRIT adversarial request is absent or duplicated.")
            self._adversarial_pending.add(request_id)
            self._adversarial_count += 1
        elif request_id not in self._adversarial_pending:
            raise ValueError("An adversarial response or error has no observed request.")
        else:
            self._adversarial_pending.remove(request_id)
            self._adversarial_success_count += int(phase == "response" and status == 200 and not error)
        self._adversarial_event_count += 1
        row = {
            "sequence": self._adversarial_event_count,
            "request_id": request_id,
            "phase": phase,
            "body_base64": base64.b64encode(body).decode("ascii"),
            "body_sha256": hashlib.sha256(body).hexdigest(),
            "status": status,
            "error": error,
        }
        encoded = self._event_bytes(frame=row)
        self._adversarial_bytes.extend(encoded)
        self._append_raw(stream_id=self._adversarial_stream.stream_id, data=encoded)
        if phase == "error" or (phase == "response" and status != 200):
            self._capture.mark_capture_gap(
                run_id=self.run_id, reason="PyRIT adversarial model lacked a complete original HTTP response."
            )

    def seal_adversarial_model(self) -> str:
        """
        Seal PyRIT's adversarial-model source after it makes its last decision.

        Returns:
            str: SHA256 of the retained adversarial-model HTTP source.
        """
        self._require_live_capture()
        content = bytes(self._adversarial_bytes)
        complete = bool(content) and not self._adversarial_pending
        self._capture.close_raw_stream(
            run_id=self.run_id,
            stream_id=self._adversarial_stream.stream_id,
            source_complete=complete,
            expected_bytes=len(content),
            observed_sha256=hashlib.sha256(content).hexdigest(),
            gaps=() if complete else ("PyRIT adversarial request lacked its original response.",),
        )
        return hashlib.sha256(content).hexdigest()

    @property
    def adversarial_model_counts(self) -> tuple[int, int]:
        """Original PyRIT adversarial HTTP request count and completed 200 responses."""
        return self._adversarial_count, self._adversarial_success_count

    def record_inspect_log(self, *, content: bytes) -> str:
        """
        Retain Inspect's one resolved EvalLog including its original scorer and task assets.

        Returns:
            str: SHA256 of the exact retained log bytes.

        Raises:
            ValueError: If the log is empty or a second log is supplied.
        """
        self._require_live_capture()
        if not content or self._log_bytes:
            raise ValueError("Exactly one nonempty original Inspect EvalLog may be retained.")
        self._log_bytes = content
        log_stream = NativeCyberRawStreamStart(run_id=self.run_id, key=self.LOG_KEY)
        self._capture.open_raw_stream(stream=log_stream)
        self._append_raw(stream_id=log_stream.stream_id, data=content)
        self._seal_raw(stream_id=log_stream.stream_id, data=content)
        return hashlib.sha256(content).hexdigest()

    def record_guest_stderr(self, *, data: bytes, omitted_bytes: int = 0) -> str:
        """
        Retain bounded original guest process errors with explicit truncation.

        Returns:
            str: SHA256 of the retained source bytes.

        Raises:
            ValueError: If guest stderr is empty or was already captured.
        """
        self._require_live_capture()
        if not data or self._stderr_recorded or omitted_bytes < 0:
            raise ValueError("Original guest stderr requires one nonempty bounded capture.")
        self._stderr_recorded = True
        stream = NativeCyberRawStreamStart(
            run_id=self.run_id,
            key=NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS,
                kind=NativeCyberRawKind.STDERR,
                observed_source_id="inspect-ghcp-worker-stderr",
            ),
        )
        self._capture.open_raw_stream(stream=stream)
        self._append_raw(stream_id=stream.stream_id, data=data)
        self._capture.close_raw_stream(
            run_id=self.run_id,
            stream_id=stream.stream_id,
            source_complete=omitted_bytes == 0,
            expected_bytes=len(data) + omitted_bytes,
            observed_sha256=hashlib.sha256(data).hexdigest() if omitted_bytes == 0 else None,
            gaps=() if omitted_bytes == 0 else ("Guest stderr exceeded its bounded original source capture.",),
        )
        return hashlib.sha256(data).hexdigest()

    def assess(self, *, report: InspectGhcpReport | None, expected_turns: int) -> tuple[str, ...]:
        """
        Check DB source bytes, pieces, events, tools and the original log before scoring.

        Returns:
            tuple[str, ...]: Required capture gaps; an empty tuple permits grading.

        Raises:
            ValueError: If the run identity or already-retained report disagrees.
        """
        if expected_turns < 0:
            raise ValueError("Expected Inspect GHCP turn count cannot be negative.")
        snapshot = self._capture.get_episode(run_id=self.run_id)
        with closing(self._memory.get_session()) as session:
            episode = session.get(NativeCyberEpisodeEntry, self.run_id)
            if episode is None:
                raise ValueError("Inspect GHCP episode disappeared before coverage assessment.")
            if report is not None:
                self._validate_report(episode=episode, report=report)
            gaps = list(episode.capture_gaps)
            gaps.extend(self._turn_gaps(session=session, snapshot=snapshot, expected_turns=expected_turns))
            gaps.extend(self._event_gaps(session=session, snapshot=snapshot, report=report))
            gaps.extend(self._tool_gaps(session=session, snapshot=snapshot, report=report))
            gaps.extend(
                self._raw_gaps(session=session, snapshot=snapshot, report=report, expected_turns=expected_turns)
            )
            if report is not None:
                gaps.extend(self._original_log_gaps(session=session, report=report))
                gaps.extend(self._result_gaps(report=report))
        return tuple(dict.fromkeys(gaps))

    def finalize_atomic(self, *, report: InspectGhcpReport, score: Score) -> NativeCyberEpisodeSnapshot:
        """
        Commit one immutable report, one Score, and their episode link in one transaction.

        Returns:
            NativeCyberEpisodeSnapshot: The committed metadata-only DB view.

        Raises:
            ValueError: If source coverage, report, or score does not match stored evidence.
        """
        gaps = self.assess(report=report, expected_turns=report.turn_count)
        if tuple(report.required_gaps) != gaps:
            raise ValueError("Inspect report gaps differ from the required source coverage in PyRIT DB.")
        complete = report.status is InspectGhcpStatus.COMPLETED
        if complete and gaps:
            raise ValueError("A complete original Inspect score cannot contain a capture gap.")
        self._validate_score(report=report, score=score, complete=complete)
        with closing(self._memory.get_session()) as session, session.begin():
            episode = session.get(NativeCyberEpisodeEntry, self.run_id)
            if episode is None or episode.finalized_at is not None:
                raise ValueError("Inspect GHCP run is missing or was already published.")
            self._validate_report(episode=episode, report=report)
            stored = self._capture._insert_prepared_score_row(session=session, score=score)
            self._validate_stored_score(session=session, stored=stored, report=report)
            episode.report_content_id = stored.scorable_content_id
            episode.report_sha256 = report.sha256()
            episode.score_id = uuid.UUID(str(stored.id))
            episode.conversation_id = report.conversation_id or episode.conversation_id
            episode.coverage_complete = complete and not gaps
            episode.capture_gaps = list(gaps)
            episode.optional_gaps = list(report.optional_gaps)
            episode.finalized_at = datetime.now(UTC)
        return self._capture.get_finalized_episode(run_id=self.run_id)

    def _captured_event(self, *, frame: dict[str, Any], session_id: str) -> NativeCyberCapturedEvent:
        kind, event_id = frame.get("type"), frame.get("id")
        if not isinstance(kind, str) or not isinstance(event_id, str) or not event_id:
            raise ValueError("A GHCP SDK event is missing its original type or ID.")
        self._sequence += 1
        source = (
            NativeCyberEvidenceSource.MODEL
            if kind.startswith("assistant.")
            else NativeCyberEvidenceSource.TOOL
            if kind.startswith("tool.")
            else NativeCyberEvidenceSource.HARNESS
        )
        return NativeCyberCapturedEvent(
            source=source,
            event=NativeCyberObservedEvent(
                controller_sequence=self._sequence,
                source_event_id=event_id,
                source_session_id=session_id,
                event_type=kind,
                payload=frame,
            ),
        )

    @staticmethod
    def _event_bytes(*, frame: dict[str, Any]) -> bytes:
        return (
            json.dumps(frame, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
            + b"\n"
        )

    @staticmethod
    def _is_matching_response(*, frame: dict[str, Any], text: str) -> bool:
        data = frame.get("data")
        if frame.get("type") != "assistant.message" or not isinstance(data, dict):
            return False
        return bool(data.get("content") == text)

    @staticmethod
    def _decode(*, record: dict[str, Any], name: str) -> bytes:
        value = record.get(name)
        if not isinstance(value, str):
            raise ValueError("An observed model request or response is missing raw bytes.")
        return base64.b64decode(value, validate=True)

    def _append_raw(self, *, stream_id: uuid.UUID, data: bytes) -> None:
        self._require_live_capture()
        for start in range(0, len(data), self._capture.MAX_APPEND_BYTES):
            result = self._capture.append_raw(
                run_id=self.run_id,
                stream_id=stream_id,
                data=data[start : start + self._capture.MAX_APPEND_BYTES],
            )
            if result.omitted_bytes:
                self._capture.mark_capture_gap(
                    run_id=self.run_id, reason="Inspect GHCP source bytes exceeded the run quota."
                )

    def _seal_raw(self, *, stream_id: uuid.UUID, data: bytes) -> None:
        self._require_live_capture()
        self._capture.close_raw_stream(
            run_id=self.run_id,
            stream_id=stream_id,
            source_complete=bool(data),
            expected_bytes=len(data),
            observed_sha256=hashlib.sha256(data).hexdigest(),
        )

    def _require_live_capture(self) -> None:
        if self._recovery_only:
            raise RuntimeError("A recovered Inspect episode may only publish an undetermined report.")

    @staticmethod
    def _validate_report(*, episode: NativeCyberEpisodeEntry, report: InspectGhcpReport) -> None:
        if (
            episode.run_id != report.run_id
            or (episode.binding_version == "3") != (report.schema_version == 3)
            or episode.task_id != report.task_name
            or episode.task_version != report.task_version
            or episode.started_at.astimezone(UTC) != report.started_at.astimezone(UTC)
            or (episode.source_session_id is not None and episode.source_session_id != report.ghcp_session_id)
            or (episode.environment_id is not None and episode.environment_id != report.agent_container_id)
            or (episode.conversation_id is not None and episode.conversation_id != report.conversation_id)
            or episode.simulated is not False
        ):
            raise ValueError("Inspect GHCP report differs from the immutable episode provenance.")

    @staticmethod
    def _turn_gaps(*, session: Any, snapshot: NativeCyberEpisodeSnapshot, expected_turns: int) -> list[str]:
        gaps: list[str] = []
        if [turn.turn_index for turn in snapshot.turns] != list(range(1, expected_turns + 1)):
            gaps.append("Persisted GHCP turns differ from the claimed PyRIT turn count.")
        rows = list(
            session.scalars(
                select(NativeCyberEventEntry)
                .where(NativeCyberEventEntry.run_id == snapshot.run.run_id)
                .order_by(NativeCyberEventEntry.sequence)
            )
        )
        for turn in snapshot.turns:
            gaps.extend(turn.gaps)
            if (
                not turn.source_complete
                or not turn.request_piece_ids
                or not turn.response_piece_ids
                or turn.observed_event_count != turn.stored_event_count
            ):
                gaps.append("A GHCP turn has missing SDK events or genuine linked PyRIT MessagePieces.")
            events = [event for event in rows if event.turn_index == turn.turn_index]
            last_action = max((event.sequence for event in events if event.source in {"model", "tool"}), default=0)
            if not any(
                event.event_type == "session.idle"
                and not event.payload.get("agentId")
                and event.sequence >= last_action
                for event in events
            ):
                gaps.append(f"GHCP turn {turn.turn_index} lacks a terminal root session.idle event.")
        return gaps

    @staticmethod
    def _event_gaps(
        *, session: Any, snapshot: NativeCyberEpisodeSnapshot, report: InspectGhcpReport | None
    ) -> list[str]:
        rows = list(
            session.scalars(
                select(NativeCyberEventEntry)
                .where(NativeCyberEventEntry.run_id == snapshot.run.run_id)
                .order_by(NativeCyberEventEntry.sequence)
            )
        )
        gaps: list[str] = []
        if [row.sequence for row in rows] != list(range(1, len(rows) + 1)):
            gaps.append("GHCP SDK event ordinals are not contiguous.")
        if report is not None and len(rows) != report.sdk_event_count:
            gaps.append("GHCP SDK event count disagrees with the report.")
        for row in rows:
            if row.payload_sha256 != NativeCyberEvidenceStore._hash_payload(row.payload):
                gaps.append("Stored GHCP SDK event payload digest changed.")
                break
        sdk = session.scalar(
            select(NativeCyberRawStreamEntry).where(
                NativeCyberRawStreamEntry.run_id == snapshot.run.run_id,
                NativeCyberRawStreamEntry.observed_source_id == InspectGhcpEvidenceStore.SDK_KEY.observed_source_id,
            )
        )
        if sdk is not None and sdk.source_complete:
            try:
                originals = [
                    json.loads(line)
                    for line in InspectGhcpEvidenceStore._stored_bytes(session=session, stream=sdk).splitlines()
                ]
            except (UnicodeError, json.JSONDecodeError):
                gaps.append("Retained GHCP SDK event bytes are not valid JSONL.")
            else:
                if originals != [row.payload for row in rows]:
                    gaps.append("GHCP SDK raw event bytes disagree with the structured event rows.")
        return gaps

    @staticmethod
    def _tool_gaps(
        *, session: Any, snapshot: NativeCyberEpisodeSnapshot, report: InspectGhcpReport | None
    ) -> list[str]:
        rows = {
            row.sequence: row
            for row in session.scalars(
                select(NativeCyberEventEntry).where(NativeCyberEventEntry.run_id == snapshot.run.run_id)
            )
        }
        outcomes = [
            InspectGhcpEvidenceStore._tool_outcome(tool=tool, rows=rows, session_id=snapshot.run.source_session_id)
            for tool in snapshot.tools
        ]
        gaps: list[str] = []
        if any(outcome is None for outcome in outcomes):
            gaps.append("GHCP model-proposed tool ID/name/arguments or its execution result is not correlated.")
        if report is not None and (
            report.tool_start_count != sum(tool.start_sequence is not None for tool in snapshot.tools)
            or report.tool_complete_count != sum(tool.completion_sequence is not None for tool in snapshot.tools)
            or report.successful_tool_execution_count != sum(outcome is True for outcome in outcomes)
        ):
            gaps.append("GHCP tool counts differ from their original SDK event correlations.")
        return gaps

    @staticmethod
    def _tool_outcome(
        *, tool: NativeCyberToolCorrelation, rows: dict[int, NativeCyberEventEntry], session_id: str | None
    ) -> bool | None:
        request_seq, start_seq, complete_seq = (
            tool.request_sequence,
            tool.start_sequence,
            tool.completion_sequence,
        )
        if (
            not session_id
            or request_seq is None
            or start_seq is None
            or complete_seq is None
            or not request_seq < start_seq < complete_seq
        ):
            return None
        request, start, completion = (
            rows.get(request_seq),
            rows.get(start_seq),
            rows.get(complete_seq),
        )
        if (
            request is None
            or start is None
            or completion is None
            or request.event_type != "assistant.message"
            or start.event_type != "tool.execution_start"
            or completion.event_type != "tool.execution_complete"
            or any(row.observed_session_id != session_id for row in (request, start, completion))
        ):
            return None
        proposed, started, completed = (row.payload.get("data") for row in (request, start, completion))
        if not isinstance(proposed, dict) or not isinstance(started, dict) or not isinstance(completed, dict):
            return None
        proposals = proposed.get("toolRequests")
        if not isinstance(proposals, list):
            return None
        matches = [item for item in proposals if isinstance(item, dict) and item.get("toolCallId") == tool.call_id]
        if len(matches) != 1:
            return None
        requested = matches[0]
        if (
            not isinstance(requested.get("name"), str)
            or not isinstance(requested.get("arguments"), dict)
            or started.get("toolCallId") != tool.call_id
            or started.get("toolName") != requested["name"]
            or started.get("arguments") != requested["arguments"]
            or completed.get("toolCallId") != tool.call_id
            or type(completed.get("success")) is not bool
        ):
            return None
        result = completed.get("result")
        if completed["success"] is True:
            return True if isinstance(result, dict) and isinstance(result.get("content"), str) else None
        return False if completed.get("error") is not None or isinstance(result, dict) else None

    def count_successful_tool_executions(self) -> int:
        """
        Resolve only validated, successful model-proposed tool executions.

        Returns:
            int: Real successful tool calls with complete source correlation.

        Raises:
            ValueError: If a model request, tool start, or result is missing or inconsistent.
        """
        snapshot = self._capture.get_episode(run_id=self.run_id)
        with closing(self._memory.get_session()) as session:
            gaps = self._tool_gaps(session=session, snapshot=snapshot, report=None)
            if gaps:
                raise ValueError("Inspect GHCP tool evidence is not completely correlated.")
            rows = {
                row.sequence: row
                for row in session.scalars(
                    select(NativeCyberEventEntry).where(NativeCyberEventEntry.run_id == self.run_id)
                )
            }
            return sum(
                self._tool_outcome(tool=tool, rows=rows, session_id=snapshot.run.source_session_id) is True
                for tool in snapshot.tools
            )

    def _raw_gaps(
        self,
        *,
        session: Any,
        snapshot: NativeCyberEpisodeSnapshot,
        report: InspectGhcpReport | None,
        expected_turns: int,
    ) -> list[str]:
        gaps: list[str] = []
        streams = list(
            session.scalars(select(NativeCyberRawStreamEntry).where(NativeCyberRawStreamEntry.run_id == self.run_id))
        )
        by_source = {stream.observed_source_id: stream for stream in streams}
        if len(by_source) != len(streams):
            gaps.append("Inspect GHCP raw source IDs are not unique.")
        gaps.extend(
            f"Required Inspect GHCP stream {key.observed_source_id} was not opened."
            for key in snapshot.run.required_raw_streams
            if key.observed_source_id not in by_source
        )
        if sum(stream.stored_bytes for stream in streams) != snapshot.stored_raw_bytes:
            gaps.append("Inspect GHCP raw byte totals differ from the quota ledger.")
        gaps.extend(
            f"Inspect GHCP raw stream {stream.observed_source_id} is incomplete or modified."
            for stream in streams
            if (
                stream.source_complete is not True
                or stream.truncated
                or stream.received_bytes != stream.stored_bytes
                or stream.stored_sha256 != self._capture._hash_stream(session=session, stream=stream)
            )
        )
        for stream in streams:
            gaps.extend(stream.capture_gaps)
        if report is not None:
            sdk = by_source.get(self.SDK_KEY.observed_source_id)
            if sdk and sdk.stored_sha256 != report.sdk_event_raw_sha256:
                gaps.append("Original GHCP SDK event-byte digest differs from the report.")
            model = by_source.get(self.MODEL_KEY.observed_source_id)
            if model and model.stored_sha256 != report.gateway_audit_sha256:
                gaps.append("The Inspect model audit digest differs from the report.")
            host = by_source.get(self.HOST_KEY.observed_source_id)
            if host and host.stored_sha256 != report.host_model_audit_sha256:
                gaps.append("The trusted host provider audit digest differs from the report.")
            adversarial = by_source.get(self.ADVERSARIAL_KEY.observed_source_id)
            if adversarial and adversarial.stored_sha256 != report.adversarial_audit_sha256:
                gaps.append("The PyRIT adversarial-model audit digest differs from the report.")
            if report.schema_version == 3:
                control = by_source.get(self.CONTROL_KEY.observed_source_id)
                if control is None or control.stored_sha256 != report.control_receipt_sha256:
                    gaps.append("Token-free Inspect controller receipt digest differs from the report.")
            original = by_source.get(self.LOG_KEY.observed_source_id)
            if original is None or original.stored_sha256 != report.inspect_log_sha256:
                gaps.append("The original Inspect EvalLog is missing or has changed.")
        if self.CONTROL_KEY in snapshot.run.required_raw_streams:
            gaps.extend(
                self._control_gaps(
                    session=session,
                    stream=by_source.get(self.CONTROL_KEY.observed_source_id),
                    run_id=self.run_id,
                    agent_container=snapshot.run.environment_id,
                    report=report,
                )
            )
        model = by_source.get(self.MODEL_KEY.observed_source_id)
        if model is not None and model.source_complete:
            try:
                records = [json.loads(line) for line in self._stored_bytes(session=session, stream=model).splitlines()]
            except (UnicodeError, json.JSONDecodeError):
                gaps.append("Inspect model gateway audit bytes are not valid JSONL.")
            else:
                if not all(isinstance(record, dict) for record in records):
                    gaps.append("Inspect model gateway audit contains a non-object record.")
                    return gaps
                statuses = [record.get("response_status") for record in records]
                if (
                    [record.get("sequence") for record in records] != list(range(1, len(records) + 1))
                    or statuses.count(200) < expected_turns
                    or any(
                        status != 200 or record.get("error") for status, record in zip(statuses, records, strict=True)
                    )
                ):
                    gaps.append("Inspect model gateway audit has missing, rejected or out-of-order responses.")
                if report is not None and (
                    len(records) != report.model_request_count or statuses.count(200) != report.model_http_200_count
                ):
                    gaps.append("Inspect gateway model counts differ from the GHCP report.")
        host = by_source.get(self.HOST_KEY.observed_source_id)
        if host is not None and host.source_complete:
            try:
                host_records = [
                    json.loads(line) for line in self._stored_bytes(session=session, stream=host).splitlines()
                ]
            except (UnicodeError, json.JSONDecodeError):
                gaps.append("Trusted host provider audit bytes are not valid JSONL.")
            else:
                if not all(isinstance(record, dict) for record in host_records):
                    gaps.append("Trusted host provider audit contains a non-object record.")
                    return gaps
                host_success = sum(
                    record.get("status") == 200
                    and record.get("response_base64") is not None
                    and record.get("error") is None
                    for record in host_records
                )
                if (
                    [record.get("sequence") for record in host_records] != list(range(1, len(host_records) + 1))
                    or host_success < expected_turns
                    or host_success != len(host_records)
                ):
                    gaps.append("Trusted host provider returned a missing, rejected or out-of-order response.")
                if report is not None and (
                    len(host_records) != report.host_model_request_count
                    or host_success != report.host_model_http_200_count
                    or len(host_records) != report.model_request_count
                ):
                    gaps.append("Host provider calls do not match the guest gateway and run report.")
        adversarial = by_source.get(self.ADVERSARIAL_KEY.observed_source_id)
        if adversarial is not None and adversarial.source_complete:
            try:
                frames = [
                    json.loads(line) for line in self._stored_bytes(session=session, stream=adversarial).splitlines()
                ]
            except (UnicodeError, json.JSONDecodeError):
                gaps.append("PyRIT adversarial-model source bytes are not valid JSONL.")
            else:
                if not all(isinstance(frame, dict) for frame in frames):
                    gaps.append("PyRIT adversarial-model audit contains a non-object record.")
                    return gaps
                requests = {frame.get("request_id") for frame in frames if frame.get("phase") == "request"}
                successes = {
                    frame.get("request_id")
                    for frame in frames
                    if frame.get("phase") == "response" and frame.get("status") == 200 and not frame.get("error")
                }
                if (
                    len(requests) != len(successes)
                    or requests != successes
                    or len(successes) < max(0, expected_turns - 1)
                ):
                    gaps.append("PyRIT adversarial model has missing or rejected original HTTP responses.")
                if report is not None and (
                    len(requests) != report.adversarial_request_count
                    or len(successes) != report.adversarial_http_200_count
                ):
                    gaps.append("PyRIT adversarial-model counts differ from retained source rows.")
        return gaps

    @staticmethod
    def _control_gaps(
        *,
        session: Any,
        stream: NativeCyberRawStreamEntry | None,
        run_id: str,
        agent_container: str | None,
        report: InspectGhcpReport | None,
    ) -> list[str]:
        if stream is None or stream.source_complete is not True:
            return ["Both private Inspect control handoffs were not completely observed."]
        try:
            rows = [
                json.loads(line)
                for line in InspectGhcpEvidenceStore._stored_bytes(session=session, stream=stream).splitlines()
            ]
        except (UnicodeError, json.JSONDecodeError):
            return ["Inspect control receipts are not valid retained JSONL."]
        if not all(isinstance(row, dict) for row in rows) or len(rows) != 2:
            return ["Inspect control receipts have no two distinct source records."]
        if [row.get("service") for row in rows] != ["model-bridge", "agent"]:
            return ["Inspect control handoffs occurred out of approved source order."]
        digest = rows[0].get("frame_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return ["Inspect controller did not retain the original control-frame SHA256."]
        expected_keys = {
            "run_id",
            "service",
            "source_id",
            "observed_job_id",
            "container_id",
            "frame_size_bytes",
            "frame_sha256",
            "completed_exit_code",
            "inspect_raw_control_elided",
            "provenance",
        }
        for row in rows:
            service, pid = row.get("service"), row.get("observed_job_id")
            container = row.get("container_id")
            if (
                set(row) != expected_keys
                or row.get("run_id") != run_id
                or type(pid) is not int
                or pid < 1
                or row.get("source_id") != f"{run_id}:{service}:{pid}"
                or not isinstance(container, str)
                or not re.fullmatch(r"[0-9a-f]{64}", container)
                or type(row.get("frame_size_bytes")) is not int
                or row.get("frame_size_bytes") != 43
                or row.get("frame_sha256") != digest
                or type(row.get("completed_exit_code")) is not int
                or row.get("completed_exit_code") != 0
                or row.get("inspect_raw_control_elided") is not True
                or row.get("provenance")
                != "Inspect as_type Docker provider; owner-only tmpfs token file removed on read"
                or (service == "agent" and agent_container is not None and container != agent_container)
                or (
                    report is not None
                    and container != (report.agent_container_id if service == "agent" else report.model_container_id)
                )
            ):
                return ["Inspect controller receipt source IDs, length or digest disagree."]
        return []

    @staticmethod
    def _stored_bytes(*, session: Any, stream: NativeCyberRawStreamEntry) -> bytes:
        return b"".join(
            chunk.data
            for chunk in session.scalars(
                select(NativeCyberRawChunkEntry)
                .where(NativeCyberRawChunkEntry.stream_id == stream.stream_id)
                .order_by(NativeCyberRawChunkEntry.sequence)
            )
        )

    def _original_log_gaps(self, *, session: Any, report: InspectGhcpReport) -> list[str]:
        stream = session.scalar(
            select(NativeCyberRawStreamEntry).where(
                NativeCyberRawStreamEntry.run_id == self.run_id,
                NativeCyberRawStreamEntry.observed_source_id == self.LOG_KEY.observed_source_id,
            )
        )
        if stream is None:
            return ["Original Inspect scorer result was not retained in PyRIT DB."]
        data = self._stored_bytes(session=session, stream=stream)
        try:
            log = json.loads(data)
            sample = log["samples"][0]
            score = sample["scores"][report.judgment.scorer_name] if report.judgment else None
        except (ValueError, KeyError, IndexError, TypeError):
            return ["Original Inspect scorer result is not present in the retained log."]
        if report.schema_version >= 2:
            identity = log.get("eval")
            if (
                not isinstance(identity, dict)
                or identity.get("task") != report.task_name
                or identity.get("run_id") != report.inspect_log_id
                or sample.get("epoch") != report.sample_epoch
            ):
                return ["Original Inspect task, sample epoch or eval identity differs from the report."]
        if not isinstance(score, dict):
            return ["Original Inspect Score is not a structured source judgment."]
        score_bytes = json.dumps(score, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        if report.judgment is None or hashlib.sha256(score_bytes).hexdigest() != report.judgment.raw_sha256:
            return ["Original Inspect Score differs from the retained task log."]
        score_events = [
            event
            for event in sample.get("events", [])
            if isinstance(event, dict)
            and event.get("event") == "score"
            and event.get("scorer") == report.judgment.scorer_name
            and event.get("intermediate") is not True
        ]
        if len(score_events) != 1 or score_events[0].get("score") != score:
            return ["The original Inspect scorer did not produce exactly one matching Score event."]
        if report.schema_version >= 2 and score_events[0].get("uuid") != report.judgment.source_event_id:
            return ["Original Inspect ScoreEvent source ID is missing or differs from the report."]
        if str(sample.get("id")) != report.sample_id or log.get("status") != "success" or sample.get("error"):
            return ["Original Inspect Task/sample did not finish successfully."]
        return []

    @staticmethod
    def _result_gaps(*, report: InspectGhcpReport) -> list[str]:
        gaps: list[str] = []
        if report.schema_version >= 3 and not report.token_files_absent_before_turn:
            gaps.append("Both run-scoped token files were not observed absent before GHCP inference.")
        if not report.benchmark_verified:
            gaps.append("Original cyber task/scorer/target image were not independently verified.")
        if not report.provider_verified:
            gaps.append("The trusted model provider was not verified loopback-only.")
        if report.judgment is None or report.judgment.numeric_value is None:
            gaps.append("Original Inspect scorer did not provide a numeric cyber judgment.")
        if not report.stopped_before_scoring:
            gaps.append("The GHCP process was not observed stopped before original grading.")
        if not report.gateway_alive_before_scoring or not report.gateway_alive_after_scoring:
            gaps.append("The authenticated model gateway did not remain alive through original grading.")
        if report.successful_tool_execution_count < report.required_tool_executions:
            gaps.append("Task-required GHCP tool requests and executions were not observed.")
        if not report.target_alive_before_scoring or not report.target_alive_after_scoring:
            gaps.append("The original target service was not observed alive around grading.")
        if not report.original_cleanup_called or not report.original_cleanup_succeeded:
            gaps.append("Original Inspect task cleanup did not complete.")
        if not report.sandbox_cleanup_observed:
            gaps.append("Inspect sandbox cleanup was not independently observed.")
        return gaps

    @staticmethod
    def _validate_score(*, report: InspectGhcpReport, score: Score, complete: bool) -> None:
        if complete and report.judgment is None:
            raise ValueError("An Inspect complete report has no acquired original judgment.")
        if (
            score.score_type != "float_scale"
            or score.message_piece_id is not None
            or score.observation_ids
            or not isinstance(score.scorable, ContentScorable)
            or score.scorable.value != report.canonical_json()
            or (score.score_metadata or {}).get("run_id") != report.run_id
            or (score.score_metadata or {}).get("report_sha256") != report.sha256()
            or (
                complete
                and report.judgment is not None
                and (score.status is not ScoreStatus.COMPLETE or score.get_value() != report.judgment.numeric_value)
            )
            or (not complete and (score.status is not ScoreStatus.UNDETERMINED or score.score_value is not None))
        ):
            raise ValueError("The prepared PyRIT Score differs from the one original Inspect report.")

    @staticmethod
    def _validate_stored_score(*, session: Any, stored: ScoreEntry, report: InspectGhcpReport) -> None:
        content = session.get(ScorableContentEntry, stored.scorable_content_id) if stored.scorable_content_id else None
        if (
            content is None
            or content.value != report.canonical_json()
            or content.value_sha256 != report.sha256()
            or stored.score_type != "float_scale"
            or (stored.score_metadata or {}).get("report_sha256") != report.sha256()
        ):
            raise ValueError("The stored PyRIT Score does not reference the canonical Inspect report.")
