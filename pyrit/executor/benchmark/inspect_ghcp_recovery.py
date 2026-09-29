# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Idempotently publish an UND result after an interrupted Inspect controller."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.models import ScoreStatus
from pyrit.models.inspect_ghcp import (
    InspectGhcpJudgment,
    InspectGhcpReport,
    InspectGhcpStatus,
    InspectGhcpTaskKind,
)
from pyrit.score.float_scale.inspect_ghcp_report_scorer import InspectGhcpReportScorer

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface
    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot, NativeCyberRawStreamSummary


def _read_verified_stream(*, memory: MemoryInterface, run_id: str, stream: NativeCyberRawStreamSummary) -> bytes:
    """
    Read retained bytes explicitly and verify their source-level length and digest.

    Returns:
        bytes: Exact stored source bytes, or empty when the stream was not sealed.

    Raises:
        ValueError: If the DB byte ledger or SHA256 differs from the raw source.
    """
    if not stream.source_complete or stream.stored_sha256 is None:
        return b""
    content = bytearray()
    cursor = 0
    while True:
        page = memory.native_cyber_evidence.read_raw_chunks(
            run_id=run_id,
            stream_id=stream.stream_id,
            allow_sensitive=True,
            after_sequence=cursor,
            limit=16,
        )
        if not page:
            break
        content.extend(b"".join(chunk.data for chunk in page))
        cursor = page[-1].sequence
        if len(content) > stream.stored_bytes:
            raise ValueError("Interrupted Inspect stream exceeds its committed raw-byte ledger.")
    if len(content) != stream.stored_bytes or hashlib.sha256(content).hexdigest() != stream.stored_sha256:
        raise ValueError("Interrupted Inspect stream bytes differ from its retained SHA256 and length.")
    return bytes(content)


def _source_rows(*, content: bytes) -> list[dict[str, Any]]:
    if not content:
        return []
    try:
        rows = [json.loads(line) for line in content.splitlines()]
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Interrupted Inspect source is not valid retained JSONL.") from error
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("Interrupted Inspect source contains an unstructured raw record.")
    return rows


def _original_judgment(
    *, content: bytes, sample_id: str, task_name: str
) -> tuple[InspectGhcpJudgment | None, str | None, int]:
    if not content:
        return None, None, 1
    try:
        log = json.loads(content)
        identity = log.get("eval")
        if not isinstance(identity, dict) or identity.get("task") != task_name:
            raise ValueError("Interrupted Inspect log has a different original task identity.")
        eval_run_id = identity.get("run_id")
        if not isinstance(eval_run_id, str) or not eval_run_id:
            raise ValueError("Interrupted Inspect log has no original eval run identity.")
        if log.get("status") != "success" or len(log.get("samples") or []) != 1:
            return None, eval_run_id, 1
        sample = log["samples"][0]
        epoch = sample.get("epoch")
        if str(sample.get("id")) != sample_id or sample.get("error"):
            raise ValueError("Interrupted Inspect log belongs to a different or failed original sample.")
        if type(epoch) is not int or epoch != 1:
            raise ValueError("Interrupted Inspect score belongs to an unsupported sample epoch.")
        scores = sample.get("scores") or {}
        if not isinstance(scores, dict) or len(scores) != 1:
            return None, eval_run_id, epoch
        scorer_name, score = next(iter(scores.items()))
        events = [
            event
            for event in sample.get("events", [])
            if isinstance(event, dict)
            and event.get("event") == "score"
            and event.get("scorer") == scorer_name
            and event.get("intermediate") is not True
        ]
        if not isinstance(score, dict) or len(events) != 1 or events[0].get("score") != score:
            return None, eval_run_id, epoch
        event_id = events[0].get("uuid")
        if not isinstance(event_id, str) or not event_id:
            return None, eval_run_id, epoch
        encoded = json.dumps(score, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        raw_value = score.get("value")
        numeric = float(raw_value) if type(raw_value) in (int, float) and 0 <= raw_value <= 1 else None
        return (
            InspectGhcpJudgment(
                scorer_name=scorer_name,
                source_event_id=event_id,
                normalization_version=1,
                raw_value=raw_value,
                numeric_value=numeric,
                explanation=score.get("explanation"),
                raw_sha256=hashlib.sha256(encoded).hexdigest(),
            ),
            eval_run_id,
            epoch,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("Interrupted original Inspect task log is not valid JSON.") from error


async def recover_interrupted_inspect_run_async(
    *,
    memory: MemoryInterface,
    run_id: str,
    sample_id: str,
    cli_sha256: str,
    model_id: str,
    wire_model: str,
    cleanup_confirmed: bool,
) -> NativeCyberEpisodeSnapshot:
    """
    Publish only an undetermined Score from an existing, cleaned-up source episode.

    This never resumes the agent, calls a model, replays a task scorer, or creates a
    replacement episode. A second call returns the original linked Score ID.

    Returns:
        NativeCyberEpisodeSnapshot: The one committed report/Score and source rows.

    Raises:
        RuntimeError: If Inspect's owned containers have not been proven cleaned up.
        ValueError: If the episode is not an Inspect run or its source identity changed.
    """
    capture = memory.native_cyber_evidence
    snapshot = await asyncio.to_thread(capture.get_episode, run_id=run_id)
    if snapshot.finalized_at is not None:
        return await asyncio.to_thread(capture.get_finalized_episode, run_id=run_id)
    if not cleanup_confirmed:
        raise RuntimeError("Inspect project cleanup must be independently verified before UND recovery.")
    task_name, task_version = snapshot.run.task_id, snapshot.run.task_version
    if snapshot.run.binding_name != "inspect-ghcp" or not task_name or not task_version:
        raise ValueError("This source episode does not name one original Inspect Task.")
    if snapshot.run.binding_version not in {"1", "2", "3"}:
        raise ValueError("Interrupted Inspect evidence has an unsupported report schema.")
    schema_version: Literal[1, 2, 3] = (
        3 if snapshot.run.binding_version == "3" else 2 if snapshot.run.binding_version == "2" else 1
    )
    store = await asyncio.to_thread(
        InspectGhcpEvidenceStore.open_pending_for_recovery,
        memory=memory,
        run_id=run_id,
    )
    await asyncio.to_thread(
        capture.mark_capture_gap,
        run_id=run_id,
        reason="Inspect controller ended before a complete atomic report and Score publication.",
    )
    snapshot = await asyncio.to_thread(capture.get_episode, run_id=run_id)
    digests = {
        stream.key.observed_source_id: stream.stored_sha256
        for stream in snapshot.raw_streams
        if stream.source_complete and stream.stored_sha256 is not None
    }
    sources = {
        stream.key.observed_source_id: await asyncio.to_thread(
            _read_verified_stream, memory=memory, run_id=run_id, stream=stream
        )
        for stream in snapshot.raw_streams
        if stream.source_complete
    }
    gateway = _source_rows(content=sources.get(InspectGhcpEvidenceStore.MODEL_KEY.observed_source_id, b""))
    host_model = _source_rows(content=sources.get(InspectGhcpEvidenceStore.HOST_KEY.observed_source_id, b""))
    adversarial = _source_rows(content=sources.get(InspectGhcpEvidenceStore.ADVERSARIAL_KEY.observed_source_id, b""))
    judgment, inspect_log_id, sample_epoch = _original_judgment(
        content=sources.get(InspectGhcpEvidenceStore.LOG_KEY.observed_source_id, b""),
        sample_id=sample_id,
        task_name=task_name,
    )
    report = InspectGhcpReport(
        schema_version=schema_version,
        run_id=run_id,
        task_name=task_name,
        task_version=task_version,
        sample_id=sample_id,
        sample_epoch=sample_epoch,
        task_kind=InspectGhcpTaskKind.PROTOCOL_SMOKE,
        cli_sha256=cli_sha256,
        model_id=model_id,
        wire_model=wire_model,
        started_at=snapshot.run.started_at,
        ended_at=datetime.now(UTC),
        inspect_log_id=inspect_log_id,
        inspect_log_sha256=digests.get(InspectGhcpEvidenceStore.LOG_KEY.observed_source_id),
        sdk_event_raw_sha256=digests.get(InspectGhcpEvidenceStore.SDK_KEY.observed_source_id),
        gateway_audit_sha256=digests.get(InspectGhcpEvidenceStore.MODEL_KEY.observed_source_id),
        host_model_audit_sha256=digests.get(InspectGhcpEvidenceStore.HOST_KEY.observed_source_id),
        adversarial_audit_sha256=digests.get(InspectGhcpEvidenceStore.ADVERSARIAL_KEY.observed_source_id),
        **(
            {"control_receipt_sha256": digests.get(InspectGhcpEvidenceStore.CONTROL_KEY.observed_source_id)}
            if snapshot.run.binding_version == "3"
            else {}
        ),
        agent_container_id=snapshot.run.environment_id,
        ghcp_session_id=snapshot.run.source_session_id,
        conversation_id=snapshot.conversation_id,
        turn_count=len(snapshot.turns),
        sdk_event_count=len(snapshot.events),
        model_request_count=len(gateway),
        model_http_200_count=sum(item.get("response_status") == 200 for item in gateway),
        host_model_request_count=len(host_model),
        host_model_http_200_count=sum(
            item.get("status") == 200 and item.get("response_base64") is not None for item in host_model
        ),
        adversarial_request_count=sum(item.get("phase") == "request" for item in adversarial),
        adversarial_http_200_count=sum(
            item.get("phase") == "response" and item.get("status") == 200 for item in adversarial
        ),
        tool_start_count=sum(tool.start_sequence is not None for tool in snapshot.tools),
        tool_complete_count=sum(tool.completion_sequence is not None for tool in snapshot.tools),
        sandbox_cleanup_observed=True,
        judgment=judgment,
        status=InspectGhcpStatus.ERROR,
    )
    gaps = await asyncio.to_thread(store.assess, report=report, expected_turns=len(snapshot.turns))
    report = InspectGhcpReport.model_validate_json(report.model_copy(update={"required_gaps": gaps}).canonical_json())
    score = InspectGhcpReportScorer(report_sha256=report.sha256()).prepare_unpersisted_score(report=report)
    if score.status is not ScoreStatus.UNDETERMINED:
        raise ValueError("Recovery may not publish a complete or numeric cyber Score.")
    try:
        return await asyncio.to_thread(store.finalize_atomic, report=report, score=score)
    except ValueError:
        latest = await asyncio.to_thread(capture.get_episode, run_id=run_id)
        if latest.finalized_at is not None and latest.score_status is ScoreStatus.UNDETERMINED:
            return await asyncio.to_thread(capture.get_finalized_episode, run_id=run_id)
        raise
