# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Import original Inspect `.eval` bytes and project source-attributed offline case results."""

from __future__ import annotations

import asyncio
import hashlib
import io
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Protocol
from zipfile import BadZipFile, ZipFile

from inspect_ai.event import ScoreEvent, ToolEvent
from inspect_ai.log import EvalLog, read_eval_log

from pyrit.executor.benchmark.inspect_eval_projection import (
    InspectProjectionVersion,
    final_original_score_event,
    project_inspect_sample,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    Conversation,
    EvalCaseRef,
    EvalRunRef,
    Score,
    ScoreStatus,
    config_hash,
)
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

    from inspect_ai.log import EvalSample
    from inspect_ai.scorer import Score as InspectScore

    from pyrit.executor.benchmark.inspect_eval_projection import InspectSampleProjection
    from pyrit.memory import MemoryInterface
    from pyrit.models.native_cyber_evidence import NativeCyberCapturedEvent, NativeCyberTurnSummary
    from pyrit.models.score.score import ScoreType


class InspectLiveObserver(Protocol):
    """The optional hook source; an offline importer never registers a hook."""

    episode_id: str

    def reconcile(self, *, log: EvalLog) -> tuple[str, ...]:
        """Compare observed completed hook events with the final typed EvalLog."""
        ...


class InspectSuccessDirection(str, Enum):
    """The explicitly reviewed meaning of a numeric original scorer value."""

    AT_LEAST = "at_least"
    AT_MOST = "at_most"


@dataclass(frozen=True, kw_only=True)
class InspectOriginalScorePolicy:
    """Select one Task's original final scorer, optionally declaring its success threshold."""

    task_name: str
    task_version: str
    primary_scorer: str
    success_direction: InspectSuccessDirection | None = None
    success_threshold: float | None = None

    def __post_init__(self) -> None:
        """
        Reject an ambiguous scorer or an incomplete success criterion.

        Raises:
            ValueError: If the policy is incomplete or its threshold is outside zero to one.
        """
        if any(not value.strip() for value in (self.task_name, self.task_version, self.primary_scorer)):
            raise ValueError("Original Inspect score policy requires a Task, version, and primary scorer.")
        if (self.success_direction is None) != (self.success_threshold is None):
            raise ValueError("Original Inspect success requires both a direction and a threshold.")
        if self.success_direction is not None and not isinstance(self.success_direction, InspectSuccessDirection):
            raise ValueError("Original Inspect success direction must be an InspectSuccessDirection.")
        threshold = self.success_threshold
        if threshold is not None and (
            type(threshold) not in (int, float) or not 0 <= threshold <= 1 or not math.isfinite(threshold)
        ):
            raise ValueError("Original Inspect success threshold must be a finite number between zero and one.")
        if threshold is not None:
            object.__setattr__(self, "success_threshold", 0.0 if threshold == 0 else float(threshold))


@dataclass(frozen=True, kw_only=True)
class InspectOriginalCaseResult:
    """One original Sample/epoch and its linked, persisted PyRIT result."""

    sample_id: str
    epoch: int
    case_run_id: str | None
    primary_scorer: str | None
    score: Score
    attack_result: AttackResult


@dataclass(frozen=True, kw_only=True)
class InspectOriginalImport:
    """Retained original evidence and offline results; the episode itself remains unscored."""

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
    case_results: tuple[InspectOriginalCaseResult, ...] = ()


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

    def __init__(
        self,
        *,
        memory: MemoryInterface,
        projection_version: InspectProjectionVersion = InspectProjectionVersion.TOOL_CALLS,
        capture_only: bool = False,
    ) -> None:
        """
        Bind memory and an explicit projection schema without loading Task code.

        Raises:
            TypeError: If the schema is not an explicit supported projection version.
        """
        if not isinstance(projection_version, InspectProjectionVersion):
            raise TypeError("Original Inspect import requires an InspectProjectionVersion.")
        if type(capture_only) is not bool:
            raise TypeError("Original Inspect capture_only must be an explicit boolean.")
        self._memory = memory
        self._capture = memory.native_cyber_evidence
        self._projection_version = projection_version
        self._capture_only = capture_only

    async def import_eval_log_async(
        self,
        *,
        path: Path,
        cases: tuple[EvalCaseRef, ...] | None = None,
        run: EvalRunRef | None = None,
        score_policy: InspectOriginalScorePolicy | None = None,
    ) -> InspectOriginalImport:
        """
        Import an existing `.eval` without importing or running any authored solver.

        Returns:
            InspectOriginalImport: Verified source bytes and linked offline case results.

        Raises:
            ValueError: If the log, case bindings, or raw byte quota is unsupported.
        """
        return await self._import_async(path=path, cases=cases, run=run, live_observer=None, score_policy=score_policy)

    async def import_eval_bytes_async(
        self,
        *,
        content: bytes,
        cases: tuple[EvalCaseRef, ...] | None = None,
        run: EvalRunRef | None = None,
        score_policy: InspectOriginalScorePolicy | None = None,
    ) -> InspectOriginalImport:
        """
        Import bounded original binary evidence without a caller-selected filesystem path.

        Returns:
            InspectOriginalImport: The same canonical import used for retained local files.

        Raises:
            TypeError: If the original binary archive is not bytes.
            ValueError: If the original archive, case or policy is invalid.
        """
        if not isinstance(content, bytes):
            raise TypeError("Original Inspect binary evidence must be bytes.")
        relogged_samples = await asyncio.to_thread(self._validate_archive_bytes, content=content)
        return await self._import_content_async(
            archive=content,
            relogged_samples=relogged_samples,
            cases=cases,
            run=run,
            live_observer=None,
            score_policy=score_policy,
        )

    async def _import_async(
        self,
        *,
        path: Path,
        cases: tuple[EvalCaseRef, ...] | None,
        run: EvalRunRef | None,
        live_observer: InspectLiveObserver | None,
        require_no_model_calls: bool = False,
        score_policy: InspectOriginalScorePolicy | None = None,
    ) -> InspectOriginalImport:
        archive, relogged_samples = await asyncio.to_thread(self._read_archive, path=path)
        return await self._import_content_async(
            archive=archive,
            relogged_samples=relogged_samples,
            cases=cases,
            run=run,
            live_observer=live_observer,
            require_no_model_calls=require_no_model_calls,
            score_policy=score_policy,
        )

    async def _import_content_async(
        self,
        *,
        archive: bytes,
        relogged_samples: bool,
        cases: tuple[EvalCaseRef, ...] | None,
        run: EvalRunRef | None,
        live_observer: InspectLiveObserver | None,
        require_no_model_calls: bool = False,
        score_policy: InspectOriginalScorePolicy | None = None,
    ) -> InspectOriginalImport:
        log = await asyncio.to_thread(read_eval_log, io.BytesIO(archive), resolve_attachments="full", format="eval")
        resolved = log.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
        if len(resolved) > self.MAX_RESOLVED_BYTES:
            raise ValueError("Resolved original Inspect log exceeds its bounded sensitive-memory quota.")
        archive_sha = hashlib.sha256(archive).hexdigest()
        self._validate_log(log=log, cases=cases, run=run)
        self._validate_score_policy(log=log, cases=cases, score_policy=score_policy)
        case_run_ids = self._case_run_ids(log=log, cases=cases, run=run)
        live_run_id = live_observer.episode_id if live_observer is not None else None
        episode_id = live_run_id or (
            "inspect-import-"
            + config_hash(
                {
                    "archive_sha256": archive_sha,
                    "case_run_ids": case_run_ids,
                    "score_policy": asdict(score_policy) if score_policy is not None else None,
                    "schema": self._projection_version.value,
                    **({"capture_only": True} if self._capture_only else {}),
                }
            )
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
            score_policy=score_policy,
        )

    @staticmethod
    def _validate_score_policy(
        *, log: EvalLog, cases: tuple[EvalCaseRef, ...] | None, score_policy: InspectOriginalScorePolicy | None
    ) -> None:
        if score_policy is None:
            return
        if not isinstance(score_policy, InspectOriginalScorePolicy):
            raise TypeError("score_policy must be an InspectOriginalScorePolicy.")
        if score_policy.task_name != log.eval.task or score_policy.task_version != str(log.eval.task_version):
            raise ValueError("Original Inspect score policy differs from the retained Task and version.")
        if score_policy.success_direction is not None and cases is None:
            raise ValueError("Success thresholds require an approved EvalCaseRef inventory and EvalRunRef.")

    @classmethod
    def _read_archive(cls, *, path: Path) -> tuple[bytes, bool]:
        if path.suffix != ".eval" or not path.is_file() or path.is_symlink():
            raise ValueError("Original Inspect import requires one explicit, regular `.eval` file.")
        with path.open("rb") as source:
            content = source.read(cls.MAX_ARCHIVE_BYTES + 1)
        return content, cls._validate_archive_bytes(content=content)

    @classmethod
    def _validate_archive_bytes(cls, *, content: bytes) -> bool:
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
        return len(sample_names) != len(set(sample_names))

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
        score_policy: InspectOriginalScorePolicy | None,
    ) -> InspectOriginalImport:
        try:
            existing = self._capture.get_episode(run_id=episode_id)
        except KeyError:
            existing = None
        if existing is not None and live_run_id is None:
            if existing.finalized_at is None:
                raise ValueError("Prior Inspect import is partial; reconcile its evidence before importing again.")
            return self._existing_result(
                log=log,
                archive_sha=archive_sha,
                resolved=resolved,
                snapshot=existing,
                case_run_ids=case_run_ids,
                score_policy=score_policy,
                relogged_samples=relogged_samples,
            )
        if live_run_id is not None:
            if (
                existing is None
                or existing.finalized_at is not None
                or existing.run.binding_name != "inspect-original"
                or existing.run.binding_version != self._projection_version.binding_version
            ):
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
                    binding_version=self._projection_version.binding_version,
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
        optional = self._project_log(log=log, episode_id=episode_id, archive_sha=archive_sha, case_run_ids=case_run_ids)
        source_gaps = self._source_log_gaps(
            log=log,
            resolved=resolved,
            relogged_samples=relogged_samples,
            require_no_model_calls=require_no_model_calls,
        )
        snapshot = self._capture.finalize_unscored_inspect_capture(
            run_id=episode_id,
            expected_samples=len(log.samples or []),
            required_gaps=tuple(source_gaps),
            optional_gaps=(*optional, *live_gaps),
        )
        self._verify_source_readback(snapshot=snapshot, archive_sha=archive_sha, resolved=resolved)
        self._verify_event_readback(
            log=log,
            snapshot=snapshot,
            archive_sha=archive_sha,
            case_run_ids=case_run_ids,
            required_source_gaps=source_gaps,
        )
        case_results = (
            self._ensure_case_results(
                log=log,
                snapshot=snapshot,
                archive_sha=archive_sha,
                case_run_ids=case_run_ids,
                score_policy=score_policy,
                required_source_gaps=source_gaps,
                persist=True,
            )
            if live_run_id is None and not self._capture_only
            else ()
        )
        return self._result(
            log=log,
            archive_sha=archive_sha,
            resolved=resolved,
            snapshot=snapshot,
            case_run_ids=case_run_ids,
            case_results=case_results,
            offline=live_run_id is None,
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
        self, *, log: EvalLog, episode_id: str, archive_sha: str, case_run_ids: tuple[str, ...]
    ) -> list[str]:
        optional: list[str] = []
        sequence = 1
        seen_sample_uuids: set[str] = set()
        for sample_index, sample in enumerate(log.samples or [], start=1):
            conversation_id = self._conversation_id(
                episode_id=episode_id, run_id=log.eval.run_id, sample=sample, sample_index=sample_index
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
                projection_version=self._projection_version,
            )
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
                    tool_request_piece_ids=projection.tool_request_ids,
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
        return optional

    @staticmethod
    def _source_log_gaps(
        *, log: EvalLog, resolved: bytes, relogged_samples: bool, require_no_model_calls: bool
    ) -> list[str]:
        """
        Recompute required run-level gaps from the original typed log and archive.

        Returns:
            list[str]: Source-required gaps that must remain in the sealed episode.
        """
        gaps: list[str] = []
        if log.status != "success" or log.invalidated or log.error is not None:
            gaps.append("Original Inspect run did not finish successfully with an unmodified EvalLog.")
        if not log.samples:
            gaps.append("Original Inspect log has no fully retained Sample events.")
        if relogged_samples:
            gaps.append(
                "Inspect archive re-logged a Sample; earlier ZIP member events were retained only as raw bytes."
            )
        if b"attachment://" in resolved or b"tc://" in resolved:
            gaps.append("Original Inspect EvalLog contains unresolved attachment references.")
        if require_no_model_calls and any(
            event.event == "model"
            for sample in log.samples or []
            for events in (sample.events, *(retry.events or [] for retry in sample.error_retries or []))
            for event in events
        ):
            gaps.append("The approved inert original Task unexpectedly invoked an Inspect model.")
        return gaps

    @staticmethod
    def _conversation_id(*, episode_id: str, run_id: str, sample: EvalSample, sample_index: int) -> str:
        identity = f"{sample_index}:{sample.uuid or ''}:{sample.epoch}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"inspect-original:{episode_id}:{run_id}:{identity}"))

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
        score_policy: InspectOriginalScorePolicy | None,
        relogged_samples: bool,
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
        source_gaps = self._source_log_gaps(
            log=log, resolved=resolved, relogged_samples=relogged_samples, require_no_model_calls=False
        )
        self._verify_event_readback(
            log=log,
            snapshot=snapshot,
            archive_sha=archive_sha,
            case_run_ids=case_run_ids,
            required_source_gaps=source_gaps,
        )
        case_results = (
            self._ensure_case_results(
                log=log,
                snapshot=snapshot,
                archive_sha=archive_sha,
                case_run_ids=case_run_ids,
                score_policy=score_policy,
                required_source_gaps=source_gaps,
                persist=False,
            )
            if not self._capture_only
            else ()
        )
        return self._result(
            log=log,
            archive_sha=archive_sha,
            resolved=resolved,
            snapshot=snapshot,
            case_run_ids=case_run_ids,
            case_results=case_results,
            offline=True,
        )

    def _ensure_case_results(
        self,
        *,
        log: EvalLog,
        snapshot: NativeCyberEpisodeSnapshot,
        archive_sha: str,
        case_run_ids: tuple[str, ...],
        score_policy: InspectOriginalScorePolicy | None,
        required_source_gaps: list[str],
        persist: bool,
    ) -> tuple[InspectOriginalCaseResult, ...]:
        samples = log.samples or []
        if len(snapshot.turns) != len(samples):
            raise ValueError("Original Inspect case projection differs from the retained Sample count.")
        turn_gaps = {gap for turn in snapshot.turns for gap in turn.gaps}
        global_issue = next(iter(required_source_gaps), None) or next(
            (gap for gap in snapshot.gaps if gap not in turn_gaps), None
        )
        expected = tuple(
            self._case_result(
                log=log,
                sample=sample,
                turn=turn,
                sample_index=index,
                episode_id=snapshot.run.run_id,
                archive_sha=archive_sha,
                case_run_id=case_run_ids[index - 1] if case_run_ids else None,
                global_issue=global_issue,
                score_policy=score_policy,
            )
            for index, (sample, turn) in enumerate(zip(samples, snapshot.turns, strict=True), start=1)
        )
        if not expected:
            return ()
        if persist:
            self._memory.add_score_attack_result_pairs_to_memory(
                pairs=[(item.score, item.attack_result) for item in expected]
            )
        return self._readback_case_results(expected=expected)

    def _readback_case_results(
        self, *, expected: tuple[InspectOriginalCaseResult, ...]
    ) -> tuple[InspectOriginalCaseResult, ...]:
        score_ids = [str(item.score.id) for item in expected]
        result_ids = [item.attack_result.attack_result_id for item in expected]
        scores = self._memory.get_scores(score_ids=score_ids)
        results = self._memory.get_attack_results(attack_result_ids=result_ids)
        if len(scores) != len(expected) or len(results) != len(expected):
            raise ValueError("Original Inspect case Score/AttackResult projection is partial or missing.")
        by_score = {str(score.id): score for score in scores}
        by_result = {result.attack_result_id: result for result in results}
        verified: list[InspectOriginalCaseResult] = []
        for item in expected:
            score = by_score.get(str(item.score.id))
            result = by_result.get(item.attack_result.attack_result_id)
            if (
                score is None
                or result is None
                or score.model_dump(mode="json") != item.score.model_dump(mode="json")
                or result.model_dump(mode="json") != item.attack_result.model_dump(mode="json")
            ):
                raise ValueError("Original Inspect case Score/AttackResult differs from its typed source.")
            verified.append(
                InspectOriginalCaseResult(
                    sample_id=item.sample_id,
                    epoch=item.epoch,
                    case_run_id=item.case_run_id,
                    primary_scorer=item.primary_scorer,
                    score=score,
                    attack_result=result,
                )
            )
        return tuple(verified)

    def _case_result(
        self,
        *,
        log: EvalLog,
        sample: EvalSample,
        turn: NativeCyberTurnSummary,
        sample_index: int,
        episode_id: str,
        archive_sha: str,
        case_run_id: str | None,
        global_issue: str | None,
        score_policy: InspectOriginalScorePolicy | None,
    ) -> InspectOriginalCaseResult:
        name, source_score, event, selection_issue = self._select_final_score(sample=sample, score_policy=score_policy)
        score_type, value, numeric = self._representable_value(value=source_score.value if source_score else None)
        if sample.error is not None or log.error is not None:
            issue = "The original Inspect execution recorded an infrastructure exception."
        elif global_issue is not None:
            issue = global_issue
        elif not turn.source_complete:
            issue = turn.gaps[0] if turn.gaps else "Original Inspect Sample coverage is incomplete."
        elif selection_issue is not None:
            issue = selection_issue
        elif source_score is not None and value is None:
            issue = "The original Inspect final score value has no supported PyRIT representation."
        else:
            issue = None
        metadata: dict[str, str | int | float] = {
            "inspect_source": "original_eval_log",
            "inspect_archive_sha256": archive_sha,
            "inspect_run_id": log.eval.run_id,
            "inspect_eval_id": log.eval.eval_id,
            "inspect_task": log.eval.task,
            "inspect_task_version": str(log.eval.task_version),
            "inspect_sample_id": str(sample.id),
            "inspect_epoch": sample.epoch,
        }
        if sample.uuid:
            metadata["inspect_sample_uuid"] = sample.uuid
        if case_run_id is not None:
            metadata["inspect_case_run_id"] = case_run_id
        if name is not None:
            metadata["inspect_primary_scorer"] = name
        if event is not None:
            assert event.uuid is not None
            metadata["inspect_final_score_event_id"] = event.uuid
            metadata["inspect_final_score_event_sha256"] = config_hash(
                {"event": event.model_dump(mode="json", exclude_none=True)}
            )
        timestamp = self._sample_timestamp(sample.completed_at, fallback=sample.started_at or log.eval.created)
        score = Score(
            id=uuid.uuid5(uuid.NAMESPACE_URL, f"inspect-original-score:{episode_id}:{sample_index}"),
            score_type=score_type,
            score_value=value if issue is None else None,
            status=ScoreStatus.UNDETERMINED if issue else ScoreStatus.COMPLETE,
            score_value_description=issue,
            score_metadata=metadata,
            timestamp=timestamp,
        )
        outcome, reason = self._case_outcome(
            log=log, sample=sample, issue=issue, numeric=numeric, score_policy=score_policy
        )
        result_metadata = dict(metadata)
        if sample.turn_count is not None:
            result_metadata["inspect_turn_count"] = sample.turn_count
        result = AttackResult(
            attack_result_id=str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"inspect-original-result:{episode_id}:{sample_index}")
            ),
            conversation_id=self._conversation_id(
                episode_id=episode_id, run_id=log.eval.run_id, sample=sample, sample_index=sample_index
            ),
            objective=f"Original Inspect task {log.eval.task} Sample {sample.id} epoch {sample.epoch} (offline import)",
            automated_score=score,
            outcome=outcome,
            outcome_reason=reason,
            metadata=result_metadata,
            error_type="InspectEvalError" if outcome is AttackOutcome.ERROR else None,
            error_message=reason if outcome is AttackOutcome.ERROR else None,
            timestamp=timestamp,
        )
        return InspectOriginalCaseResult(
            sample_id=str(sample.id),
            epoch=sample.epoch,
            case_run_id=case_run_id,
            primary_scorer=name,
            score=score,
            attack_result=result,
        )

    @staticmethod
    def _select_final_score(
        *, sample: EvalSample, score_policy: InspectOriginalScorePolicy | None
    ) -> tuple[str | None, InspectScore | None, ScoreEvent | None, str | None]:
        scores = sample.scores or {}
        if score_policy is not None:
            name = score_policy.primary_scorer
        elif len(scores) == 1:
            name = next(iter(scores))
        elif scores:
            return None, None, None, "Multiple final Inspect scorers require an explicitly reviewed primary."
        else:
            return None, None, None, "The original Inspect Sample has no final score."
        if name not in scores:
            return name, None, None, "The selected original Inspect scorer has no final sample score."
        try:
            event = final_original_score_event(sample=sample, scorer_name=name)
        except ValueError:
            return name, None, None, "The original Inspect final ScoreEvent disagrees with its sample score."
        if event is None:
            return name, None, None, "The original Inspect final sample score has no unique matching ScoreEvent."
        return name, scores[name], event, None

    @staticmethod
    def _representable_value(*, value: object) -> tuple[ScoreType, str | None, float | None]:
        if isinstance(value, bool):
            return "true_false", str(value).lower(), float(value)
        if isinstance(value, (int, float)) and 0 <= value <= 1 and math.isfinite(value):
            return "float_scale", str(value), float(value)
        return "unknown", None, None

    @staticmethod
    def _case_outcome(
        *,
        log: EvalLog,
        sample: EvalSample,
        issue: str | None,
        numeric: float | None,
        score_policy: InspectOriginalScorePolicy | None,
    ) -> tuple[AttackOutcome, str]:
        if sample.error is not None or log.error is not None:
            return AttackOutcome.ERROR, "The original Inspect execution recorded an infrastructure exception."
        if issue is not None:
            return AttackOutcome.UNDETERMINED, issue
        if score_policy is None or score_policy.success_direction is None:
            return AttackOutcome.UNDETERMINED, "No task-specific success direction and threshold were supplied."
        threshold = score_policy.success_threshold
        if threshold is None or numeric is None:
            raise ValueError("A complete original Inspect Score requires an explicit representable threshold.")
        meets = (
            numeric >= threshold
            if score_policy.success_direction is InspectSuccessDirection.AT_LEAST
            else numeric <= threshold
        )
        return (
            AttackOutcome.SUCCESS if meets else AttackOutcome.FAILURE,
            (
                f"Original Inspect {score_policy.primary_scorer} compared using the explicit "
                f"{score_policy.success_direction.value} {threshold} success criterion."
            ),
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

    def _verify_event_readback(
        self,
        *,
        log: EvalLog,
        snapshot: NativeCyberEpisodeSnapshot,
        archive_sha: str,
        case_run_ids: tuple[str, ...],
        required_source_gaps: list[str],
    ) -> None:
        if snapshot.coverage_complete != (not snapshot.gaps) or not set(required_source_gaps).issubset(snapshot.gaps):
            raise ValueError("Original Inspect stored run coverage differs from its source coverage.")
        expected: list[NativeCyberCapturedEvent] = []
        sequence = 1
        seen_sample_uuids: set[str] = set()
        observed_tool_phases: set[tuple[str, str]] = set()
        for sample_index, sample in enumerate(log.samples or [], start=1):
            projection = project_inspect_sample(
                sample=sample,
                log_run_id=log.eval.run_id,
                eval_id=log.eval.eval_id,
                archive_sha256=archive_sha,
                sample_index=sample_index,
                start_sequence=sequence,
                conversation_id=self._conversation_id(
                    episode_id=snapshot.run.run_id,
                    run_id=log.eval.run_id,
                    sample=sample,
                    sample_index=sample_index,
                ),
                case_run_id=case_run_ids[sample_index - 1] if case_run_ids else None,
                projection_version=InspectProjectionVersion.from_binding_version(snapshot.run.binding_version),
            )
            turn = snapshot.turns[sample_index - 1]
            source_id = sample.uuid if sample.uuid and sample.uuid not in seen_sample_uuids else None
            source_gaps = self._sample_gaps(sample=sample, projection=projection)
            if source_id is None and sample.uuid:
                source_gaps.append("Original Inspect sample UUID is duplicated across epochs or samples.")
            if sample.uuid:
                seen_sample_uuids.add(sample.uuid)
            expected_turn_gaps = self._source_turn_gaps(
                projection=projection, sample_gaps=source_gaps, observed_tool_phases=observed_tool_phases
            )
            if (
                turn.turn_index != sample_index
                or turn.source_turn_id != source_id
                or turn.response_mode is not NativeCyberResponseMode.SAMPLE_CAPTURE
                or turn.finished_at is None
                or turn.observed_event_count != len(projection.events)
                or turn.stored_event_count != len(projection.events)
                or turn.source_complete != (not turn.gaps)
                or not set(turn.gaps).issubset(snapshot.gaps)
                or turn.gaps != expected_turn_gaps
            ):
                raise ValueError("Original Inspect stored Sample coverage differs from its source coverage.")
            expected.extend(projection.events)
            sequence += len(projection.events)
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
                if cursor > len(expected):
                    raise ValueError("Original Inspect projected event count changed in PyRIT memory.")
                original = expected[cursor - 1]
                if captured.source != original.source or config_hash(
                    captured.event.model_dump(mode="json")
                ) != config_hash(original.event.model_dump(mode="json")):
                    raise ValueError("Original Inspect projected event differs from its typed source.")
                if (
                    original.event.event_type != "inspect.projection.sample"
                    and captured.captured_at != original.captured_at
                ):
                    raise ValueError("Original Inspect event timestamp differs from its typed source.")
            if len(events) < self._capture.MAX_EVENT_BATCH:
                break
        if cursor != len(snapshot.events) or cursor != len(expected):
            raise ValueError("Original Inspect projected event count changed in PyRIT memory.")

    @staticmethod
    def _source_turn_gaps(
        *, projection: InspectSampleProjection, sample_gaps: list[str], observed_tool_phases: set[tuple[str, str]]
    ) -> tuple[str, ...]:
        """
        Reconstruct native capture gaps from this Sample's typed events.

        Returns:
            tuple[str, ...]: The only turn gaps attributable to this source Sample and its captured tool phases.
        """
        gaps: list[str] = []
        for captured in projection.events:
            event = captured.event
            if event.tool_call_id is None or event.tool_phase is None:
                continue
            phase = event.tool_phase.value
            key = (event.tool_call_id, phase)
            if key in observed_tool_phases:
                gaps.append(f"Native tool phase {phase} was observed more than once at event {event.sequence}.")
            observed_tool_phases.add(key)
        gaps.extend(sample_gaps)
        if sample_gaps:
            gaps.append("Native turn source did not report complete event coverage.")
        return tuple(dict.fromkeys(gaps))

    @classmethod
    def _result(
        cls,
        *,
        log: EvalLog,
        archive_sha: str,
        resolved: bytes,
        snapshot: NativeCyberEpisodeSnapshot,
        case_run_ids: tuple[str, ...],
        case_results: tuple[InspectOriginalCaseResult, ...],
        offline: bool,
    ) -> InspectOriginalImport:
        if snapshot.score_id is not None or snapshot.score_status is not ScoreStatus.UNDETERMINED:
            raise ValueError("The original Inspect evidence episode must remain unscored.")
        samples = log.samples or []
        score_count = sum(len(sample.scores or {}) for sample in samples)
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
                len(turn.request_piece_ids)
                + len(turn.response_piece_ids)
                + len(turn.tool_request_piece_ids)
                + len(turn.tool_result_piece_ids)
                for turn in snapshot.turns
            ),
            tool_event_count=sum(cls._tool_count(sample=sample) for sample in samples),
            original_final_score_events=sum(cls._final_score_count(sample=sample) for sample in samples),
            case_run_ids=case_run_ids,
            no_grade_reasons=(
                "Offline results only mirror the original Inspect scorer; no independent PyRIT grading was performed."
                if offline
                else "The original Task runner retains source evidence without a PyRIT Score or AttackResult.",
                "Inspect events cannot attest external CLI, provider, or OS activity outside the original log.",
                *(() if score_count else ("The original Inspect log contains no final scorer verdict.",)),
            ),
            case_results=case_results,
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
