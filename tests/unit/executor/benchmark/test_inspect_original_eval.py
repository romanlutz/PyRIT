# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""An unchanged public Inspect Task and solver-neutral offline `.eval` import."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from inspect_ai import eval_async
from inspect_ai.event import ModelEvent, ScoreEvent, ToolEvent
from inspect_ai.log import EvalRetryError, read_eval_log, write_eval_log
from inspect_ai.model import ContentImage, ContentText, GenerateConfig, ModelOutput
from inspect_ai.scorer import Score
from sqlalchemy import func, select

from pyrit.executor.benchmark.inspect_eval_source import (
    EvalSourceFactory,
    ResolvedOriginalInspectTask,
    _sha256_normalized_python,
)
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter
from pyrit.executor.benchmark.inspect_original_runner import _approved_log_location, run_original_inert_eval_async
from pyrit.memory.memory_models import (
    AttackResultEntry,
    NativeCyberEpisodeEntry,
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    ScoreEntry,
)
from pyrit.models import ScoreStatus

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

    from pyrit.memory import SQLiteMemory


def test_public_task_pin_matches_git_lf_and_windows_crlf_checkouts(tmp_path: Path) -> None:
    source = EvalSourceFactory.resolve_original_inert(family="inspect_original_inert")
    content = source.source_file.read_bytes().replace(b"\r\n", b"\n")
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(content)
    crlf.write_bytes(content.replace(b"\n", b"\r\n"))
    assert _sha256_normalized_python(lf) == _sha256_normalized_python(crlf) == source.source_sha256
    crlf.write_bytes(content.replace(b"\n", b"\r"))
    with pytest.raises(ValueError, match="unsupported line endings"):
        _sha256_normalized_python(crlf)


def test_original_runner_resolves_only_bare_or_local_file_uris_within_approved_dir(tmp_path: Path) -> None:
    approved = tmp_path / "approved"
    approved.mkdir()
    archive = approved / "one.eval"
    archive.write_bytes(b"fixture")
    outside = tmp_path / "outside-original.eval"
    outside.write_bytes(b"fixture")
    assert _approved_log_location(path=str(archive), log_dir=approved) == archive
    assert _approved_log_location(path=archive.as_uri(), log_dir=approved) == archive
    for path in (outside.as_uri(), "https://example.invalid/run.eval", "file://remote-host/run.eval"):
        with pytest.raises(ValueError, match="approved local directory|local file path|local, unmodified"):
            _approved_log_location(path=path, log_dir=approved)


@pytest.fixture
async def original_log_async(tmp_path: Path) -> EvalLog:
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
    logs = await eval_async(tasks=source.task, log_dir=str(tmp_path), log_format="eval", log_realtime=False)
    assert len(logs) == 1 and logs[0].status == "success" and logs[0].location
    return await asyncio.to_thread(read_eval_log, logs[0].location, resolve_attachments="full")


@pytest.mark.usefixtures("patch_central_database")
async def test_unmodified_inert_inspect_task_imports_original_archive_without_pyrit_grade(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    source = EvalSourceFactory.resolve_original_inert(family="inspect_original_inert")
    source.verify_unchanged()
    logs = await eval_async(tasks=source.task, log_dir=str(tmp_path), log_format="eval", log_realtime=False)
    assert len(logs) == 1 and logs[0].status == "success" and logs[0].location
    archive = Path(logs[0].location)
    original_bytes = archive.read_bytes()
    typed = read_eval_log(archive, resolve_attachments="full")
    assert typed.samples and typed.samples[0].store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
    assert len([event for event in typed.samples[0].events if isinstance(event, ScoreEvent)]) == 1
    assert typed.samples[0].scores and typed.samples[0].scores["original_inert_scorer"].value == 1.0
    assert typed.samples[0].model_usage == {}

    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    assert imported.archive_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert imported.inspect_run_id == typed.eval.run_id
    assert imported.sample_count == 1
    assert imported.original_final_score_events == 1
    assert imported.message_piece_count == 2
    assert imported.episode.coverage_complete
    assert imported.episode.score_id is None
    assert imported.episode.score_status is ScoreStatus.UNDETERMINED
    assert imported.episode.finalized_at is not None
    assert imported.case_run_ids == ()
    assert "no qualified PyRIT scorer" in imported.no_grade_reasons[0]
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0

    repeated = await importer.import_eval_log_async(path=archive)
    assert repeated.episode.run.run_id == imported.episode.run.run_id
    assert repeated.episode.events == imported.episode.events


@pytest.mark.usefixtures("patch_central_database")
async def test_allowlisted_original_runner_keeps_solver_setup_scorer_and_cleanup(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    completed = await run_original_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    imported = completed.imported
    assert completed.run.case_run_id(case=completed.case) == imported.case_run_ids[0]
    assert imported.episode.coverage_complete
    assert imported.episode.score_id is None
    assert imported.original_final_score_events == 1
    assert imported.log_status == "success"
    assert not [event for event in imported.episode.events if event.event_type == "model"]
    assert len(list(tmp_path.glob("*.eval"))) == 1
    typed = read_eval_log(next(tmp_path.glob("*.eval")), resolve_attachments="full")
    assert typed.samples is not None
    assert typed.samples[0].store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
    assert imported.episode.run.task_id == typed.eval.task
    assert imported.inspect_run_id == typed.eval.run_id
    assert imported.episode.run.source_session_id == typed.eval.run_id
    projected = sqlite_instance.native_cyber_evidence.read_event_payloads(
        run_id=imported.episode.run.run_id,
        allow_sensitive=True,
    )
    [sample_summary] = [event for event in projected if event.event.event_type == "inspect.projection.sample"]
    assert sample_summary.event.payload["case_run_id"] == imported.case_run_ids[0]
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0

    streams = {item.key.observed_source_id: item for item in imported.episode.raw_streams}
    archive_stream = streams["inspect-original-eval-archive"]
    live_stream = streams["inspect-original-live-hooks"]
    assert archive_stream.source_complete
    assert live_stream.stored_bytes > 0
    assert live_stream.source_complete
    hook_bytes = b"".join(
        chunk.data
        for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
            run_id=imported.episode.run.run_id,
            stream_id=live_stream.stream_id,
            allow_sensitive=True,
        )
    )
    hook_frames = [json.loads(line) for line in hook_bytes.splitlines()]
    assert {frame["run_id"] for frame in hook_frames} == {typed.eval.run_id}
    hooked = [
        (frame["event"].get("uuid"), frame["event"]["event"]) for frame in hook_frames if frame["kind"] == "event"
    ]
    final = [(event.uuid, event.event) for event in typed.samples[0].events]
    assert set(hooked).issubset(set(final))
    assert len(hooked) < len(final)
    assert imported.episode.optional_gaps == ("Inspect live event source differs from the finalized sample.",)
    raw = b"".join(
        chunk.data
        for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
            run_id=imported.episode.run.run_id,
            stream_id=archive_stream.stream_id,
            allow_sensitive=True,
        )
    )
    assert raw == next(tmp_path.glob("*.eval")).read_bytes()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("authored_solver", ["inspect_ai/react", "custom/agent_solver"])
async def test_offline_import_retains_retry_intermediate_score_and_tool_without_duplicate_assistant(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, authored_solver: str
) -> None:
    assert original_log_async.samples
    original = original_log_async.samples[0]
    assert original.scores
    model_event = ModelEvent(
        uuid=f"model-{uuid.uuid4().hex}",
        model="recorded/original",
        input=[],
        tools=[],
        tool_choice="none",
        config=GenerateConfig(),
        output=ModelOutput.from_content(model="recorded/original", content="inert response"),
        completed=datetime.now(UTC),
    )
    tool_event = ToolEvent(
        uuid=f"tool-{uuid.uuid4().hex}",
        id="call-original-1",
        function="inert_tool",
        arguments={"value": "harmless"},
        result="harmless tool result",
        completed=datetime.now(UTC),
    )
    intermediate = ScoreEvent(
        uuid=f"intermediate-{uuid.uuid4().hex}",
        scorer="original_inert_scorer",
        intermediate=True,
        score=Score(value=0.25),
    )
    retry = EvalRetryError(
        message="First authored attempt failed",
        traceback="fixture",
        traceback_ansi="fixture",
        events=[intermediate],
    )
    sample = original.model_copy(
        update={
            "epoch": 2,
            "events": [model_event, tool_event, *original.events],
            "error_retries": [retry],
        }
    )
    typed = original_log_async.model_copy(
        update={
            "eval": original_log_async.eval.model_copy(update={"solver": authored_solver}),
            "samples": [sample],
        }
    )
    archive = tmp_path / f"{authored_solver.replace('/', '_')}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")

    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    with patch.object(EvalSourceFactory, "resolve_original_inert", side_effect=AssertionError("no task import")):
        imported = await importer.import_eval_log_async(path=archive)
    assert imported.episode.coverage_complete
    assert imported.episode.score_id is None
    assert imported.original_final_score_events == 1
    assert imported.tool_event_count == 1
    assert imported.sample_count == 1 and imported.observed_event_count == len(sample.events) + 1
    assert imported.message_piece_count == len(original.messages) == 2
    assert imported.episode.turns[0].source_turn_id == sample.uuid
    assert [event.observed_event_id for event in imported.episode.events[:3]] == [
        intermediate.uuid,
        model_event.uuid,
        tool_event.uuid,
    ]
    assert imported.episode.events[0].event_type == "score"
    assert imported.episode.events[1].source.value == "model"
    assert imported.episode.events[2].source.value == "tool"
    assert imported.episode.tools[0].call_id == tool_event.id
    assert imported.episode.tools[0].completion_sequence == 3
    assert imported.episode.tools[0].start_sequence is None
    assert imported.episode.tools[0].result_sequence is None
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("corruption", ["error_status", "mismatched_final_score"])
async def test_offline_import_seals_original_error_or_score_disagreement_as_ungraded(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, corruption: str
) -> None:
    assert original_log_async.samples
    original = original_log_async.samples[0]
    if corruption == "error_status":
        typed = original_log_async.model_copy(update={"status": "error"})
    else:
        final = next(event for event in original.events if isinstance(event, ScoreEvent))
        modified = final.model_copy(update={"score": Score(value=0.0)})
        sample = original.model_copy(
            update={"events": [modified if event is final else event for event in original.events]}
        )
        typed = original_log_async.model_copy(update={"samples": [sample]})
    archive = tmp_path / f"{corruption}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    assert not imported.episode.coverage_complete
    assert imported.episode.gaps
    assert imported.episode.score_id is None
    assert imported.original_final_score_events == (1 if corruption == "error_status" else 0)
    original_stream = next(
        stream
        for stream in imported.episode.raw_streams
        if stream.key.observed_source_id == "inspect-original-eval-archive"
    )
    assert original_stream.source_complete
    assert original_stream.stored_sha256 == hashlib.sha256(await asyncio.to_thread(archive.read_bytes)).hexdigest()
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_two_original_samples_with_same_text_and_overlapping_sequences_have_distinct_conversations(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog
) -> None:
    assert original_log_async.samples
    first = original_log_async.samples[0]
    second = first.model_copy(
        update={
            "id": "original-inert-2",
            "epoch": 2,
            "uuid": uuid.uuid4().hex,
            "messages": [message.model_copy(update={"id": f"second-{uuid.uuid4().hex}"}) for message in first.messages],
            "events": [event.model_copy(update={"uuid": f"second-{uuid.uuid4().hex}"}) for event in first.events],
        }
    )
    dataset = original_log_async.eval.dataset.model_copy(
        update={"samples": 2, "sample_ids": [str(first.id), str(second.id)]}
    )
    typed = original_log_async.model_copy(
        update={
            "eval": original_log_async.eval.model_copy(update={"dataset": dataset}),
            "samples": [first, second],
        }
    )
    archive = tmp_path / "two-samples.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    imported = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_log_async(path=archive)

    assert imported.episode.coverage_complete
    assert imported.episode.conversation_id is None
    assert imported.sample_count == 2 and imported.message_piece_count == 4
    assert [(turn.source_turn_id, turn.turn_index) for turn in imported.episode.turns] == [
        (first.uuid, 1),
        (second.uuid, 2),
    ]
    [first_turn, second_turn] = imported.episode.turns
    first_piece = sqlite_instance.get_message_pieces(prompt_ids=list(first_turn.request_piece_ids))[0]
    second_piece = sqlite_instance.get_message_pieces(prompt_ids=list(second_turn.request_piece_ids))[0]
    assert first_piece.conversation_id != second_piece.conversation_id
    assert first_piece.sequence == second_piece.sequence == 0
    assert first_piece.original_value == second_piece.original_value == "harmless fixture"
    assert first_piece.prompt_metadata["inspect_sample_uuid"] == first.uuid
    assert second_piece.prompt_metadata["inspect_sample_uuid"] == second.uuid
    for piece in (first_piece, second_piece):
        assert piece.conversation_id is not None
        messages = sqlite_instance.get_conversation_messages(conversation_id=piece.conversation_id)
        assert len(messages) == 2
        assert [message.get_piece().role for message in messages] == ["user", "assistant"]
    assert set(first_turn.request_piece_ids + first_turn.response_piece_ids).isdisjoint(
        second_turn.request_piece_ids + second_turn.response_piece_ids
    )
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("corrupt", ["archive", "event"])
async def test_reimport_rejects_tampered_original_archive_or_event_payload(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, corrupt: str
) -> None:
    archive = tmp_path / "readback.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session, session.begin():
        if corrupt == "archive":
            stream = next(
                item
                for item in imported.episode.raw_streams
                if item.key.observed_source_id == "inspect-original-eval-archive"
            )
            chunk = session.get(NativeCyberRawChunkEntry, (stream.stream_id, 1))
            assert chunk is not None
            chunk.data = b"x" * len(chunk.data)
        else:
            event = session.get(NativeCyberEventEntry, (imported.episode.run.run_id, 1))
            assert event is not None
            event.payload = {"type": "tampered"}
    with pytest.raises(ValueError, match="corrupt byte range|failed integrity validation"):
        await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


def _write_compressed_archive(*, path: Path) -> None:
    with ZipFile(path, mode="w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("samples/unbounded.json", b"x" * 512)


@pytest.mark.usefixtures("patch_central_database")
async def test_offline_import_rejects_compressed_archive_over_uncompressed_quota_before_parsing(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    archive = tmp_path / "oversized.eval"
    await asyncio.to_thread(_write_compressed_archive, path=archive)
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    with patch.object(InspectOriginalEvalImporter, "MAX_UNCOMPRESSED_BYTES", 128):
        with pytest.raises(ValueError, match="uncompressed/member quota"):
            await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(NativeCyberEpisodeEntry.run_id))) == 0


def _relog_sample_member(*, path: Path) -> None:
    with ZipFile(path, mode="a") as archive:
        name = next(name for name in archive.namelist() if name.startswith("samples/") and name.endswith(".json"))
        content = archive.read(name)
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr(name, content)


@pytest.mark.usefixtures("patch_central_database")
async def test_relogged_sample_keeps_original_zip_bytes_but_blocks_complete_projection(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog
) -> None:
    archive = tmp_path / "relogged.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    await asyncio.to_thread(_relog_sample_member, path=archive)
    imported = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_log_async(path=archive)
    assert not imported.episode.coverage_complete
    assert "re-logged a Sample" in " ".join(imported.episode.gaps)
    assert imported.episode.score_id is None
    original = next(
        stream
        for stream in imported.episode.raw_streams
        if stream.key.observed_source_id == "inspect-original-eval-archive"
    )
    assert original.stored_sha256 == hashlib.sha256(await asyncio.to_thread(archive.read_bytes)).hexdigest()
    assert original.source_complete
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("include_attachment", [True, False])
async def test_import_resolves_embedded_inspect_attachments_and_flags_missing_references(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, include_attachment: bool
) -> None:
    assert original_log_async.samples
    original = original_log_async.samples[0]
    request = original.messages[0].model_copy(
        update={"content": [ContentText(text="harmless"), ContentImage(image="attachment://fixture-image")]}
    )
    sample = original.model_copy(
        update={
            "messages": [request, *original.messages[1:]],
            "attachments": {"fixture-image": "data:image/png;base64,SU5FUlQ="} if include_attachment else {},
        }
    )
    typed = original_log_async.model_copy(update={"samples": [sample]})
    archive = tmp_path / f"attachment-{include_attachment}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    imported = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_log_async(path=archive)
    assert imported.episode.score_id is None
    assert imported.archive_sha256 == hashlib.sha256(await asyncio.to_thread(archive.read_bytes)).hexdigest()
    if include_attachment:
        assert imported.episode.coverage_complete
        assert "non-text messages" in " ".join(imported.episode.optional_gaps)
    else:
        assert not imported.episode.coverage_complete
        assert "unresolved attachment" in " ".join(imported.episode.gaps)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_process_wide_hook_is_default_off_for_a_separate_unobserved_eval(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    tracked_dir = tmp_path / "tracked"
    other_dir = tmp_path / "untracked"
    await asyncio.to_thread(tracked_dir.mkdir)
    await asyncio.to_thread(other_dir.mkdir)
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")

    tracked = await run_original_inert_eval_async(memory=sqlite_instance, log_dir=tracked_dir)
    unrelated = await eval_async(tasks=source.task, log_dir=str(other_dir), log_format="eval", log_realtime=False)
    assert unrelated[0].status == "success"
    assert tracked.imported.episode.coverage_complete
    stream = next(
        item
        for item in tracked.imported.episode.raw_streams
        if item.key.observed_source_id == "inspect-original-live-hooks"
    )
    content = b"".join(
        chunk.data
        for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
            run_id=tracked.imported.episode.run.run_id,
            stream_id=stream.stream_id,
            allow_sensitive=True,
        )
    )
    assert {json.loads(line)["run_id"] for line in content.splitlines()} == {tracked.imported.inspect_run_id}
    assert unrelated[0].eval.run_id != tracked.imported.inspect_run_id
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_unapproved_original_profile_fails_before_inspect_launch(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    with (
        patch.object(EvalSourceFactory, "ORIGINAL_INERT_SHA256", "f" * 64),
        patch("pyrit.executor.benchmark.inspect_original_runner.eval_async") as launched,
        pytest.raises(ValueError, match="pinned SHA256"),
    ):
        await run_original_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    launched.assert_not_called()
    assert not list(tmp_path.glob("*.eval"))


@pytest.mark.usefixtures("patch_central_database")
async def test_source_drift_after_original_run_retains_archive_without_case_grade(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    with patch.object(
        ResolvedOriginalInspectTask, "verify_unchanged", side_effect=[None, ValueError("source drifted")]
    ):
        with pytest.raises(RuntimeError, match=r"ungraded archive (inspect-run-[0-9a-f]{32}) was retained") as failed:
            await run_original_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    assert len(list(tmp_path.glob("*.eval"))) == 1
    episode_id = str(failed.value).split("ungraded archive ", 1)[1].split(" ", 1)[0]
    episode = sqlite_instance.native_cyber_evidence.get_finalized_unscored_inspect_capture(run_id=episode_id)
    assert not episode.coverage_complete
    assert "source drifted" in " ".join(episode.gaps)
    assert episode.score_id is None and episode.report_content_id is None
    assert any(stream.key.observed_source_id == "inspect-original-eval-archive" for stream in episode.raw_streams)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0
