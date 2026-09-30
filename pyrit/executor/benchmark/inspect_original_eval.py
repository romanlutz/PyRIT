# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Import original Inspect `.eval` bytes without executing a solver or creating a Score."""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol
from zipfile import BadZipFile, ZipFile

from inspect_ai.event import ScoreEvent, ToolEvent
from inspect_ai.log import EvalLog, read_eval_log

from pyrit.executor.benchmark.inspect_eval_projection import final_original_score_event, project_inspect_sample
from pyrit.models import Conversation, EvalCaseRef, EvalRunRef, ScoreStatus, config_hash
from pyrit.models.native_cyber_evidence import (
    NativeCyberEpisodeSnapshot,
    NativeCyberEpisodeStart,
    NativeCyberEvidenceSource,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberResponseMode,
    NativeCyberTurnFinish,
    NativeCyberTurnStart,
)

if TYPE_CHECKING:
    from pathlib import Path
    from threading import Event

    from inspect_ai.log import EvalSample

    from pyrit.executor.benchmark.inspect_eval_projection import InspectSampleProjection
    from pyrit.memory import MemoryInterface


class InspectLiveObserver(Protocol):
    """The optional hook source; an offline importer never registers a hook."""

    episode_id: str

    def reconcile(self, *, log: EvalLog) -> tuple[str, ...]:
        """Compare observed completed hook events with the final typed EvalLog."""
        ...


@dataclass(frozen=True, kw_only=True)
class InspectOriginalImport:
    """Safe result of an original-log import, never an inferred PyRIT benchmark verdict."""

    episode: NativeCyberEpisodeSnapshot
    inspect_run_id: str
    inspect_eval_id: str
    archive_sha256: str
    resolved_sha256: str
    log_status: str
    sample_count: int
    observed_event_count: int
    message_piece_count: int
    tool_event_count: int
    original_final_score_events: int
    case_run_ids: tuple[str, ...]
    no_grade_reasons: tuple[str, ...]


class InspectOriginalEvalImporter:
    """Bounded offline import of original Inspect logs using the existing evidence store."""

    MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
    MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024
    MAX_ARCHIVE_MEMBERS = 256
    MAX_RESOLVED_BYTES = 32 * 1024 * 1024
    MAX_LIVE_HOOK_BYTES = 2 * 1024 * 1024
    RAW_QUOTA = MAX_ARCHIVE_BYTES + MAX_RESOLVED_BYTES + MAX_LIVE_HOOK_BYTES
    ARCHIVE_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.EVAL_LOG,
        observed_source_id="inspect-original-eval-archive",
    )
    RESOLVED_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="inspect-resolved-eval-log",
    )

    def __init__(self, *, memory: MemoryInterface) -> None:
        """Bind a trusted PyRIT memory backend; the import never loads task code."""
        self._memory = memory
        self._capture = memory.native_cyber_evidence

    async def import_eval_log_async(
        self,
        *,
        path: Path,
        cases: tuple[EvalCaseRef, ...] | None = None,
        run: EvalRunRef | None = None,
    ) -> InspectOriginalImport:
        """
        Import an existing `.eval` without importing or running any authored solver.

        Returns:
            InspectOriginalImport: Verified original bytes and typed sample evidence, ungraded in PyRIT.

        Raises:
            ValueError: If the log, case bindings, or raw byte quota is unsupported.
        """
        return await self._import_async(path=path, cases=cases, run=run, live_observer=None)

    async def _import_async(
        self,
        *,
        path: Path,
        cases: tuple[EvalCaseRef, ...] | None,
        run: EvalRunRef | None,
        live_observer: InspectLiveObserver | None,
        require_no_model_calls: bool = False,
        binding_name: str = "inspect-original",
        mode2_control_ids: frozenset[str] = frozenset(),
        mode2_cancellation: Event | None = None,
    ) -> InspectOriginalImport:
        if binding_name not in {"inspect-original", "inspect-mode2"} or (
            binding_name == "inspect-mode2" and (live_observer is None or mode2_cancellation is None)
        ):
            raise ValueError("Mode 2 import requires its active, separately labeled live capture.")
        if binding_name == "inspect-original" and mode2_cancellation is not None:
            raise ValueError("Offline original Inspect import cannot accept a Mode 2 cancellation signal.")
        archive, relogged_samples = await asyncio.to_thread(self._read_archive, path=path)
        log = await asyncio.to_thread(read_eval_log, io.BytesIO(archive), resolve_attachments="full", format="eval")
        resolved = log.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
        if len(resolved) > self.MAX_RESOLVED_BYTES:
            raise ValueError("Resolved original Inspect log exceeds its bounded sensitive-memory quota.")
        archive_sha = hashlib.sha256(archive).hexdigest()
        self._validate_log(log=log, cases=cases, run=run)
        case_run_ids = self._case_run_ids(log=log, cases=cases, run=run)
        live_run_id = live_observer.episode_id if live_observer is not None else None
        episode_id = live_run_id or (
            "inspect-import-" + config_hash({"archive_sha256": archive_sha, "case_run_ids": case_run_ids, "schema": 1})
        )
        return await asyncio.to_thread(
            self._persist_import,
            log=log,
            archive=archive,
            resolved=resolved,
            archive_sha=archive_sha,
            episode_id=episode_id,
            case_run_ids=case_run_ids,
            live_run_id=live_run_id,
            live_gaps=live_observer.reconcile(log=log) if live_observer is not None else (),
            require_no_model_calls=require_no_model_calls,
            relogged_samples=relogged_samples,
            binding_name=binding_name,
            mode2_control_ids=mode2_control_ids,
            mode2_cancellation=mode2_cancellation,
        )

    @classmethod
    def _read_archive(cls, *, path: Path) -> tuple[bytes, bool]:
        if path.suffix != ".eval" or not path.is_file() or path.is_symlink():
            raise ValueError("Original Inspect import requires one explicit, regular `.eval` file.")
        with path.open("rb") as source:
            content = source.read(cls.MAX_ARCHIVE_BYTES + 1)
        if not content or len(content) > cls.MAX_ARCHIVE_BYTES:
            raise ValueError("Original Inspect archive is empty or exceeds its bounded byte quota.")
        try:
            with ZipFile(io.BytesIO(content)) as archive:
                members = archive.infolist()
        except BadZipFile as error:
            raise ValueError("Original Inspect archive is not a readable `.eval` ZIP.") from error
        if (
            not members
            or len(members) > cls.MAX_ARCHIVE_MEMBERS
            or sum(member.file_size for member in members) > cls.MAX_UNCOMPRESSED_BYTES
            or any(member.flag_bits & 1 for member in members)
        ):
            raise ValueError("Original Inspect archive exceeds its uncompressed/member quota or is encrypted.")
        sample_names = [
            member.filename
            for member in members
            if member.filename.startswith("samples/") and member.filename.endswith(".json")
        ]
        return content, len(sample_names) != len(set(sample_names))

    @staticmethod
    def _validate_log(*, log: EvalLog, cases: tuple[EvalCaseRef, ...] | None, run: EvalRunRef | None) -> None:
        if log.version != 2 or not log.eval.run_id or not log.eval.eval_id or not log.eval.task:
            raise ValueError("Original Inspect `.eval` has an unsupported or incomplete typed run identity.")
        if log.samples is not None and len(log.samples) > 32:
            raise ValueError("Original Inspect import supports at most 32 materialized samples.")
        if (cases is None) != (run is None):
            raise ValueError("Qualified original Inspect cases require their matching EvalRunRef.")
        if cases is None:
            return
        assert run is not None
        if not cases or run.spec.input_variant is not None or len({case.case_id for case in cases}) != len(cases):
            raise ValueError("Unchanged Inspect import requires unique source cases and no input overlay.")
        if any(
            case.package != run.spec.package
            or case.task_name != log.eval.task
            or case.task_version != str(log.eval.task_version)
            for case in cases
        ):
            raise ValueError("Original Inspect log Task differs from the selected source cases.")
        if log.status == "success":
            actual = {(str(sample.id), sample.epoch) for sample in log.samples or []}
            expected = {(case.sample_id, case.epoch) for case in cases}
            if actual != expected or len(log.samples or []) != len(cases):
                raise ValueError("Original Inspect log Samples/epochs differ from the approved case inventory.")
        elif any(
            (str(sample.id), sample.epoch) not in {(case.sample_id, case.epoch) for case in cases}
            for sample in log.samples or []
        ):
            raise ValueError("Partial original Inspect log contains an unapproved Sample or epoch.")

    @staticmethod
    def _case_run_ids(
        *, log: EvalLog, cases: tuple[EvalCaseRef, ...] | None, run: EvalRunRef | None
    ) -> tuple[str, ...]:
        if cases is None or run is None:
            return ()
        by_sample = {(case.sample_id, case.epoch): run.case_run_id(case=case) for case in cases}
        return tuple(by_sample[(str(sample.id), sample.epoch)] for sample in log.samples or [])

    def _persist_import(
        self,
        *,
        log: EvalLog,
        archive: bytes,
        resolved: bytes,
        archive_sha: str,
        episode_id: str,
        case_run_ids: tuple[str, ...],
        live_run_id: str | None,
        live_gaps: tuple[str, ...],
        require_no_model_calls: bool,
        relogged_samples: bool,
        binding_name: str,
        mode2_control_ids: frozenset[str],
        mode2_cancellation: Event | None,
    ) -> InspectOriginalImport:
        try:
            existing = self._capture.get_episode(run_id=episode_id)
        except KeyError:
            existing = None
        if existing is not None and live_run_id is None:
            if existing.finalized_at is None:
                raise ValueError("Prior Inspect import is partial; reconcile its evidence before importing again.")
            return self._existing_result(
                log=log, archive_sha=archive_sha, resolved=resolved, snapshot=existing, case_run_ids=case_run_ids
            )
        if live_run_id is not None:
            if existing is None or existing.finalized_at is not None or existing.run.binding_name != binding_name:
                raise ValueError("The original Inspect live capture is absent or already finalized.")
            if existing.run.raw_byte_limit - existing.stored_raw_bytes < len(archive) + len(resolved):
                self._capture.mark_capture_gap(run_id=episode_id, reason="Original Inspect archive exceeds live quota.")
                raise ValueError("Original Inspect live capture cannot retain both exact source and resolved bytes.")
        else:
            created = datetime.fromisoformat(log.eval.created)
            if created.tzinfo is None:
                raise ValueError("Original Inspect log has no timezone-aware creation timestamp.")
            self._capture.create_episode(
                start=NativeCyberEpisodeStart(
                    run_id=episode_id,
                    binding_name="inspect-original",
                    binding_version="1",
                    task_id=log.eval.task,
                    task_version=str(log.eval.task_version),
                    started_at=created,
                    source_session_id=log.eval.run_id,
                    simulated=None,
                    required_raw_streams=(self.ARCHIVE_KEY, self.RESOLVED_KEY),
                    raw_byte_limit=self.RAW_QUOTA,
                )
            )
        self._write_source(run_id=episode_id, key=self.ARCHIVE_KEY, content=archive)
        self._write_source(run_id=episode_id, key=self.RESOLVED_KEY, content=resolved)
        gaps, optional = self._project_log(
            log=log,
            episode_id=episode_id,
            archive_sha=archive_sha,
            case_run_ids=case_run_ids,
            binding_name=binding_name,
            mode2_control_ids=mode2_control_ids,
        )
        if binding_name == "inspect-mode2":
            gaps.extend(live_gaps)
            live_gaps = ()
        if relogged_samples:
            gaps.append(
                "Inspect archive re-logged a Sample; earlier ZIP member events were retained only as raw bytes."
            )
        if b"attachment://" in resolved or b"tc://" in resolved:
            gaps.append("Original Inspect EvalLog contains unresolved attachment references.")
        if require_no_model_calls and any(
            event.event_type == "model" for event in self._capture.get_episode(run_id=episode_id).events
        ):
            gaps.append("The approved inert original Task unexpectedly invoked an Inspect model.")
        if binding_name == "inspect-mode2":
            pending = self._capture.get_episode(run_id=episode_id)
            self._verify_source_readback(snapshot=pending, archive_sha=archive_sha, resolved=resolved)
            self._verify_event_readback(snapshot=pending)
        snapshot = self._capture.finalize_unscored_inspect_capture(
            run_id=episode_id,
            expected_samples=len(log.samples or []),
            required_gaps=tuple(gaps),
            optional_gaps=(*optional, *live_gaps),
            cancellation_event=mode2_cancellation,
        )
        self._verify_source_readback(snapshot=snapshot, archive_sha=archive_sha, resolved=resolved)
        self._verify_event_readback(snapshot=snapshot)
        return self._result(
            log=log,
            archive_sha=archive_sha,
            resolved=resolved,
            snapshot=snapshot,
            case_run_ids=case_run_ids,
            binding_name=binding_name,
        )

    def _write_source(self, *, run_id: str, key: NativeCyberRawStreamKey, content: bytes) -> None:
        stream = NativeCyberRawStreamStart(run_id=run_id, key=key)
        self._capture.open_raw_stream(stream=stream)
        for start in range(0, len(content), self._capture.MAX_APPEND_BYTES):
            write = self._capture.append_raw(
                run_id=run_id,
                stream_id=stream.stream_id,
                data=content[start : start + self._capture.MAX_APPEND_BYTES],
            )
            if write.omitted_bytes:
                self._capture.mark_capture_gap(run_id=run_id, reason="Original Inspect source was truncated by quota.")
        self._capture.close_raw_stream(
            run_id=run_id,
            stream_id=stream.stream_id,
            source_complete=True,
            expected_bytes=len(content),
            observed_sha256=hashlib.sha256(content).hexdigest(),
        )

    def _project_log(
        self,
        *,
        log: EvalLog,
        episode_id: str,
        archive_sha: str,
        case_run_ids: tuple[str, ...],
        binding_name: str,
        mode2_control_ids: frozenset[str],
    ) -> tuple[list[str], list[str]]:
        gaps: list[str] = []
        optional: list[str] = []
        if log.status != "success" or log.invalidated or log.error is not None:
            gaps.append("Original Inspect run did not finish successfully with an unmodified EvalLog.")
        if not log.samples:
            gaps.append("Original Inspect log has no fully retained Sample events.")
        sequence = 1
        seen_sample_uuids: set[str] = set()
        for sample_index, sample in enumerate(log.samples or [], start=1):
            sample_identity = f"{sample_index}:{sample.uuid or ''}:{sample.epoch}"
            conversation_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{binding_name}:{episode_id}:{log.eval.run_id}:{sample_identity}",
                )
            )
            projection = project_inspect_sample(
                sample=sample,
                log_run_id=log.eval.run_id,
                eval_id=log.eval.eval_id,
                archive_sha256=archive_sha,
                sample_index=sample_index,
                start_sequence=sequence,
                conversation_id=conversation_id,
                case_run_id=case_run_ids[sample_index - 1] if case_run_ids else None,
                mode2_control_ids=mode2_control_ids,
                omit_mode2_controls=binding_name == "inspect-mode2",
            )
            if projection.message_pieces:
                self._memory.add_conversation_to_memory(conversation=Conversation(conversation_id=conversation_id))
            self._memory.add_message_pieces_to_memory(message_pieces=projection.message_pieces)
            source_id = sample.uuid if sample.uuid and sample.uuid not in seen_sample_uuids else None
            if sample.uuid:
                seen_sample_uuids.add(sample.uuid)
            self._capture.begin_turn(
                turn=NativeCyberTurnStart(
                    run_id=episode_id,
                    turn_index=sample_index,
                    source_turn_id=source_id,
                    started_at=self._sample_timestamp(sample.started_at, fallback=log.eval.created),
                    request_piece_ids=projection.request_ids,
                    response_mode=NativeCyberResponseMode.SAMPLE_CAPTURE,
                )
            )
            for start in range(0, len(projection.events), self._capture.MAX_EVENT_BATCH):
                self._capture.append_events(
                    run_id=episode_id,
                    turn_index=sample_index,
                    events=projection.events[start : start + self._capture.MAX_EVENT_BATCH],
                )
            sample_gaps = self._sample_gaps(sample=sample, projection=projection)
            if source_id is None and sample.uuid:
                sample_gaps.append("Original Inspect sample UUID is duplicated across epochs or samples.")
            self._capture.finish_turn(
                finish=NativeCyberTurnFinish(
                    run_id=episode_id,
                    turn_index=sample_index,
                    finished_at=self._sample_timestamp(
                        sample.completed_at, fallback=sample.started_at or log.eval.created
                    ),
                    response_piece_ids=projection.response_ids,
                    tool_result_piece_ids=projection.tool_result_ids,
                    observed_event_count=len(projection.events),
                    source_complete=not sample_gaps,
                    gaps=tuple(sample_gaps),
                )
            )
            sequence += len(projection.events)
            optional.extend(
                ["Inspect non-text messages remain in the exact resolved EvalLog, not text MessagePieces."]
                if projection.unprojected_messages
                else []
            )
        return gaps, optional

    @staticmethod
    def _sample_timestamp(value: str | None, *, fallback: str) -> datetime:
        timestamp = datetime.fromisoformat(value or fallback)
        if timestamp.tzinfo is None:
            raise ValueError("Original Inspect Sample timestamps must be timezone-aware.")
        return timestamp.astimezone(UTC)

    @staticmethod
    def _sample_gaps(*, sample: EvalSample, projection: InspectSampleProjection) -> list[str]:
        gaps: list[str] = []
        if sample.error is not None or sample.completed_at is None:
            gaps.append("Original Inspect Sample errored or lacks a completion timestamp.")
        if not sample.uuid or not projection.original_event_count:
            gaps.append("Original Inspect Sample lacks its source UUID or typed events.")
        if any(not item.event.source_event_id for item in projection.events[:-1]):
            gaps.append("An original Inspect event lacks its observed source UUID.")
        if any(retry.events is None for retry in sample.error_retries or []):
            gaps.append("An original Inspect retry has no retained attempt events.")
        if any(
            isinstance(event, ScoreEvent) and not event.intermediate and event.scorer not in (sample.scores or {})
            for event in sample.events
        ):
            gaps.append("A final Inspect ScoreEvent has no matching acquired sample Score.")
        for name in sample.scores or {}:
            try:
                match = final_original_score_event(sample=sample, scorer_name=name)
            except ValueError:
                gaps.append("Original Inspect final ScoreEvent contradicts its acquired sample score.")
                continue
            if match is None:
                gaps.append("Original Inspect final sample score lacks one matching ScoreEvent.")
        return list(dict.fromkeys(gaps))

    def _existing_result(
        self,
        *,
        log: EvalLog,
        archive_sha: str,
        resolved: bytes,
        snapshot: NativeCyberEpisodeSnapshot,
        case_run_ids: tuple[str, ...],
    ) -> InspectOriginalImport:
        if snapshot.run.binding_name != "inspect-original" or snapshot.run.source_session_id != log.eval.run_id:
            raise ValueError("Prior Inspect import belongs to another original run.")
        sources = {stream.key.observed_source_id: stream for stream in snapshot.raw_streams}
        archive = sources.get(self.ARCHIVE_KEY.observed_source_id)
        typed = sources.get(self.RESOLVED_KEY.observed_source_id)
        if (
            archive is None
            or typed is None
            or not archive.source_complete
            or not typed.source_complete
            or archive.stored_sha256 != archive_sha
            or typed.stored_sha256 != hashlib.sha256(resolved).hexdigest()
            or len(snapshot.turns) != len(log.samples or [])
        ):
            raise ValueError("Existing Inspect import has incomplete or changed source bytes.")
        self._verify_source_readback(snapshot=snapshot, archive_sha=archive_sha, resolved=resolved)
        self._verify_event_readback(snapshot=snapshot)
        return self._result(
            log=log, archive_sha=archive_sha, resolved=resolved, snapshot=snapshot, case_run_ids=case_run_ids
        )

    def _verify_source_readback(
        self, *, snapshot: NativeCyberEpisodeSnapshot, archive_sha: str, resolved: bytes
    ) -> None:
        expected = {
            self.ARCHIVE_KEY.observed_source_id: archive_sha,
            self.RESOLVED_KEY.observed_source_id: hashlib.sha256(resolved).hexdigest(),
        }
        for key, digest in expected.items():
            matches = [item for item in snapshot.raw_streams if item.key.observed_source_id == key]
            if len(matches) != 1 or not matches[0].source_complete:
                raise ValueError("Original Inspect archive or resolved sample stream is not sealed.")
            stream = matches[0]
            checksum = hashlib.sha256()
            size = 0
            after = 0
            while True:
                chunks = self._capture.read_raw_chunks(
                    run_id=snapshot.run.run_id,
                    stream_id=stream.stream_id,
                    allow_sensitive=True,
                    after_sequence=after,
                    limit=self._capture.MAX_RAW_READ_CHUNKS,
                )
                for chunk in chunks:
                    size += chunk.length
                    checksum.update(chunk.data)
                if len(chunks) < self._capture.MAX_RAW_READ_CHUNKS:
                    break
                after = chunks[-1].sequence
            if checksum.hexdigest() != digest or stream.stored_sha256 != digest or size != stream.stored_bytes:
                raise ValueError("The retained original Inspect bytes differ from the authorized source digest.")

    def _verify_event_readback(self, *, snapshot: NativeCyberEpisodeSnapshot) -> None:
        cursor = 0
        while True:
            events = self._capture.read_event_payloads(
                run_id=snapshot.run.run_id,
                allow_sensitive=True,
                after_sequence=cursor,
                limit=self._capture.MAX_EVENT_BATCH,
            )
            for captured in events:
                cursor += 1
                if captured.event.sequence != cursor:
                    raise ValueError("Original Inspect projected event order changed in PyRIT memory.")
            if len(events) < self._capture.MAX_EVENT_BATCH:
                break
        if cursor != len(snapshot.events):
            raise ValueError("Original Inspect projected event count changed in PyRIT memory.")

    @classmethod
    def _result(
        cls,
        *,
        log: EvalLog,
        archive_sha: str,
        resolved: bytes,
        snapshot: NativeCyberEpisodeSnapshot,
        case_run_ids: tuple[str, ...],
        binding_name: str = "inspect-original",
    ) -> InspectOriginalImport:
        if snapshot.score_id is not None or snapshot.score_status is not ScoreStatus.UNDETERMINED:
            raise ValueError("Original Inspect import must never create or link a PyRIT Score.")
        samples = log.samples or []
        score_count = sum(len(sample.scores or {}) for sample in samples)
        if binding_name == "inspect-mode2":
            grade_reason = "Mode 2 is a steered variant; its original Inspect scorer is not a PyRIT grade."
        else:
            grade_reason = (
                "Mode 1 retains Inspect scores as source evidence only; no qualified PyRIT scorer or AttackResult."
            )
        return InspectOriginalImport(
            episode=snapshot,
            inspect_run_id=log.eval.run_id,
            inspect_eval_id=log.eval.eval_id,
            archive_sha256=archive_sha,
            resolved_sha256=hashlib.sha256(resolved).hexdigest(),
            log_status=log.status,
            sample_count=len(samples),
            observed_event_count=sum(
                len(sample.events) + sum(len(retry.events or []) for retry in sample.error_retries or [])
                for sample in samples
            ),
            message_piece_count=sum(
                len(turn.request_piece_ids) + len(turn.response_piece_ids) + len(turn.tool_result_piece_ids)
                for turn in snapshot.turns
            ),
            tool_event_count=sum(cls._tool_count(sample=sample) for sample in samples),
            original_final_score_events=sum(cls._final_score_count(sample=sample) for sample in samples),
            case_run_ids=case_run_ids,
            no_grade_reasons=(
                grade_reason,
                "Inspect events cannot attest external CLI, provider, or OS activity outside the original log.",
                *(() if score_count else ("The original Inspect log contains no final scorer verdict.",)),
            ),
        )

    @staticmethod
    def _tool_count(*, sample: EvalSample) -> int:
        return sum(isinstance(event, ToolEvent) for event in sample.events) + sum(
            isinstance(event, ToolEvent) for retry in sample.error_retries or [] for event in retry.events or []
        )

    @staticmethod
    def _final_score_count(*, sample: EvalSample) -> int:
        matches = 0
        for name in sample.scores or {}:
            try:
                matches += final_original_score_event(sample=sample, scorer_name=name) is not None
            except ValueError:
                continue
        return matches
