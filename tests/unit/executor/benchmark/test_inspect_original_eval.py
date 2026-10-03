# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""An unchanged public Inspect Task and solver-neutral offline `.eval` import."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from inspect_ai import eval_async
from inspect_ai.event import ModelEvent, ScoreEvent, ToolEvent
from inspect_ai.log import (
    EvalConfig,
    EvalDataset,
    EvalError,
    EvalLog,
    EvalRetryError,
    EvalSample,
    EvalSpec,
    EvalStats,
    read_eval_log,
    write_eval_log,
)
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
    GenerateConfig,
    ModelOutput,
)
from inspect_ai.scorer import Score
from inspect_ai.tool import ToolCall, ToolCallError
from sqlalchemy import func, select

from pyrit.executor.benchmark.inspect_eval_projection import InspectProjectionVersion, project_inspect_sample
from pyrit.executor.benchmark.inspect_eval_source import (
    EvalSourceFactory,
    ResolvedOriginalInspectTask,
    _sha256_normalized_python,
)
from pyrit.executor.benchmark.inspect_original_eval import (
    InspectOriginalEvalImporter,
    InspectOriginalScorePolicy,
    InspectSuccessDirection,
)
from pyrit.executor.benchmark.inspect_original_runner import _approved_log_location, run_original_inert_eval_async
from pyrit.memory.memory_models import (
    AttackResultEntry,
    ConversationEntry,
    NativeCyberEpisodeEntry,
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberTurnEntry,
    ScoreEntry,
)
from pyrit.models import (
    AttackOutcome,
    EvalCaseRef,
    EvalPackageRef,
    EvalRunRef,
    EvalSourceKind,
    EvalSpecRef,
    HarnessProfileRef,
    ModelRouteRef,
    ScoreStatus,
    config_hash,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session

    from pyrit.memory import SQLiteMemory
    from pyrit.models import AttackResult


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


def _case_inventory(
    *, source: ResolvedOriginalInspectTask, samples: list[EvalSample]
) -> tuple[tuple[EvalCaseRef, ...], EvalRunRef]:
    cases = tuple(
        source.case.model_copy(update={"sample_id": str(sample.id), "epoch": sample.epoch}) for sample in samples
    )
    return cases, EvalRunRef(spec=source.spec, run_instance_id=uuid.uuid4())


@pytest.fixture
def cancelled_offline_eval() -> tuple[EvalLog, tuple[EvalCaseRef, ...], EvalRunRef, InspectOriginalScorePolicy]:
    """Build a two-Sample original log from typed data without executing a Task."""
    created = datetime.now(UTC).isoformat()
    task_name = "inert_offline_import"
    scorer_name = "recorded_scorer"
    sample_ids = ("offline-one", "offline-two")
    samples = [
        EvalSample(
            id=sample_id,
            epoch=1,
            input=f"inert input {sample_id}",
            target="inert target",
            uuid=f"sample-{uuid.uuid4().hex}",
            started_at=created,
            completed_at=created,
            messages=[ChatMessageUser(content=f"inert input {sample_id}")],
            events=[ScoreEvent(uuid=f"score-{uuid.uuid4().hex}", scorer=scorer_name, score=Score(value=1.0))],
            scores={scorer_name: Score(value=1.0)},
            model_usage={},
        )
        for sample_id in sample_ids
    ]
    log = EvalLog(
        status="cancelled",
        eval=EvalSpec(
            eval_id=f"eval-{uuid.uuid4().hex}",
            run_id=f"run-{uuid.uuid4().hex}",
            created=created,
            task=task_name,
            task_version=1,
            dataset=EvalDataset(samples=2, sample_ids=list(sample_ids)),
            model="mockllm/model",
            config=EvalConfig(),
        ),
        samples=samples,
        stats=EvalStats(),
    )
    package = EvalPackageRef(kind=EvalSourceKind.NAMED, name="offline_fixture", source_sha256="a" * 64)
    run = EvalRunRef(
        spec=EvalSpecRef(
            package=package,
            harness=HarnessProfileRef(name="offline", config_sha256="b" * 64),
            model_route=ModelRouteRef(name="none", config_sha256="c" * 64),
        ),
        run_instance_id=uuid.uuid4(),
    )
    cases = tuple(
        EvalCaseRef(
            package=package,
            task_name=task_name,
            task_version="1",
            sample_id=str(sample.id),
            epoch=sample.epoch,
        )
        for sample in samples
    )
    policy = InspectOriginalScorePolicy(
        task_name=task_name,
        task_version="1",
        primary_scorer=scorer_name,
        success_direction=InspectSuccessDirection.AT_LEAST,
        success_threshold=0.5,
    )
    return log, cases, run, policy


@pytest.fixture
def recorded_tool_log(
    cancelled_offline_eval: tuple[EvalLog, tuple[EvalCaseRef, ...], EvalRunRef, InspectOriginalScorePolicy],
) -> EvalLog:
    log, _, _, _ = cancelled_offline_eval
    assert log.samples
    sample = log.samples[0].model_copy(
        update={
            "messages": [
                ChatMessageSystem(id="system-source", content="Only inspect recorded harmless values."),
                ChatMessageUser(id="user-source", content="Read the recorded tool evidence."),
                ChatMessageAssistant(
                    id="calls-source",
                    content="",
                    tool_calls=[
                        ToolCall(id="call-one", function="echo", arguments={"value": "one"}),
                        ToolCall(id="call-two", function="echo", arguments={"value": {"number": 2}}),
                    ],
                ),
                ChatMessageTool(
                    id="reply-two-source", tool_call_id="call-two", function="echo", content='{"number":2}'
                ),
                ChatMessageTool(
                    id="reply-one-source",
                    tool_call_id="call-one",
                    function="echo",
                    content="",
                    error=ToolCallError(type="permission", message="This harmless fixture denies the request."),
                ),
                ChatMessageAssistant(
                    id="mixed-source",
                    content="The original calls are recorded.",
                    tool_calls=[ToolCall(id="call-three", function="echo", arguments={"value": "three"})],
                ),
                ChatMessageTool(
                    id="reply-three-source",
                    tool_call_id="call-three",
                    function="echo",
                    content=[ContentText(text="first"), ContentText(text="second")],
                ),
            ]
        }
    )
    return log.model_copy(
        update={
            "status": "success",
            "samples": [sample],
            "eval": log.eval.model_copy(update={"dataset": EvalDataset(samples=1, sample_ids=[str(sample.id)])}),
        }
    )


@pytest.mark.usefixtures("patch_central_database")
def test_tool_projection_preserves_calls_mixed_parts_ids_results_and_errors(recorded_tool_log: EvalLog) -> None:
    assert recorded_tool_log.samples
    sample = recorded_tool_log.samples[0]
    original = sample.model_dump(mode="json")
    projected = project_inspect_sample(
        sample=sample,
        log_run_id=recorded_tool_log.eval.run_id,
        eval_id=recorded_tool_log.eval.eval_id,
        archive_sha256="a" * 64,
        sample_index=1,
        start_sequence=1,
        conversation_id=str(uuid.uuid4()),
    )
    pieces = projected.message_pieces
    assert len(pieces) == 10 and projected.unprojected_messages == 0
    calls = [piece for piece in pieces if piece.original_value_data_type == "function_call"]
    replies = [piece for piece in pieces if piece.original_value_data_type == "function_call_output"]
    assert [json.loads(piece.original_value)["call_id"] for piece in calls] == ["call-one", "call-two", "call-three"]
    assert [json.loads(piece.original_value)["call_id"] for piece in replies] == [
        "call-two",
        "call-one",
        "call-three",
        "call-three",
    ]
    assert json.loads(json.loads(calls[1].original_value)["arguments"]) == {"value": {"number": 2}}
    assert [piece.sequence for piece in pieces] == [0, 1, 2, 2, 3, 4, 5, 5, 6, 6]
    assert [piece.prompt_metadata["inspect_part_index"] for piece in pieces] == [0, 0, 1, 2, 0, 0, 0, 1, 0, 1]
    assert calls[0].prompt_metadata["inspect_message_id"] == "calls-source"
    assert replies[1].response_error == "unknown"
    assert replies[1].prompt_metadata["inspect_tool_error_message"] == "This harmless fixture denies the request."
    assert json.loads(replies[1].original_value)["output"] == ""
    assert len(projected.request_ids) == 2 and len(projected.response_ids) == 1
    assert len(projected.tool_request_ids) == 3
    assert len(projected.tool_result_ids) == 4
    assert sample.model_dump(mode="json") == original


@pytest.mark.usefixtures("patch_central_database")
async def test_binary_import_matches_file_identity_without_loading_or_rerunning_the_task(
    tmp_path: Path, sqlite_instance: SQLiteMemory, recorded_tool_log: EvalLog
) -> None:
    archive = tmp_path / "public-binary.eval"
    await asyncio.to_thread(write_eval_log, recorded_tool_log, location=archive, format="eval")
    content = await asyncio.to_thread(archive.read_bytes)
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    binary = await importer.import_eval_bytes_async(content=content)
    file = await importer.import_eval_log_async(path=archive)
    assert binary.episode.model_dump(mode="json") == file.episode.model_dump(mode="json")
    assert binary.case_results == file.case_results
    assert binary.archive_sha256 == hashlib.sha256(content).hexdigest()
    assert binary.message_piece_count == 10
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count()).select_from(ScoreEntry)) == 1
        assert session.scalar(select(func.count()).select_from(AttackResultEntry)) == 1
    assert await asyncio.to_thread(archive.read_bytes) == content


@pytest.mark.usefixtures("patch_central_database")
async def test_capture_only_retains_original_calls_and_grade_but_persists_no_result_pair(
    tmp_path: Path, sqlite_instance: SQLiteMemory, recorded_tool_log: EvalLog
) -> None:
    archive = tmp_path / "public-capture-only.eval"
    await asyncio.to_thread(write_eval_log, recorded_tool_log, location=archive, format="eval")
    content = await asyncio.to_thread(archive.read_bytes)
    importer = InspectOriginalEvalImporter(memory=sqlite_instance, capture_only=True)
    captured = await importer.import_eval_bytes_async(content=content)
    assert captured.case_results == () and captured.message_piece_count == 10
    assert captured.archive_sha256 == hashlib.sha256(content).hexdigest()
    assert (await importer.import_eval_bytes_async(content=content)).episode == captured.episode
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count()).select_from(ScoreEntry)) == 0
        assert session.scalar(select(func.count()).select_from(AttackResultEntry)) == 0
    projected = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_bytes_async(content=content)
    assert projected.episode.run.run_id != captured.episode.run.run_id
    assert projected.case_results[0].score.status is ScoreStatus.COMPLETE
    assert projected.case_results[0].attack_result.outcome is AttackOutcome.UNDETERMINED
    assert (await importer.import_eval_bytes_async(content=content)).case_results == ()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("capture_only", [1, "true", None])
def test_capture_only_requires_an_exact_boolean(sqlite_instance: SQLiteMemory, capture_only: object) -> None:
    with pytest.raises(TypeError, match="explicit boolean"):
        InspectOriginalEvalImporter(memory=sqlite_instance, capture_only=capture_only)  # type: ignore[arg-type]


@pytest.mark.usefixtures("patch_central_database")
async def test_new_tool_schema_does_not_mutate_sealed_text_only_import(
    tmp_path: Path, sqlite_instance: SQLiteMemory, recorded_tool_log: EvalLog
) -> None:
    archive = tmp_path / "public-recorded-tools.eval"
    await asyncio.to_thread(write_eval_log, recorded_tool_log, location=archive, format="eval")
    before = archive.read_bytes()
    legacy_importer = InspectOriginalEvalImporter(
        memory=sqlite_instance, projection_version=InspectProjectionVersion.TEXT_ONLY
    )
    legacy = await legacy_importer.import_eval_log_async(path=archive)
    legacy_dump = legacy.episode.model_dump(mode="json")
    legacy_conversation_id = legacy.case_results[0].attack_result.conversation_id
    legacy_pieces = sqlite_instance.get_message_pieces(conversation_id=legacy_conversation_id)
    legacy_piece_dump = [piece.model_dump(mode="json") for piece in legacy_pieces]
    assert legacy.message_piece_count == 7 and legacy.episode.run.binding_version == "1"
    assert legacy.episode.run.run_id == "inspect-import-" + config_hash(
        {
            "archive_sha256": legacy.archive_sha256,
            "case_run_ids": (),
            "score_policy": None,
            "schema": 2,
        }
    )
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    current = await importer.import_eval_log_async(path=archive)
    assert current.message_piece_count == 10 and current.episode.run.binding_version == "2"
    assert current.episode.run.run_id != legacy.episode.run.run_id
    assert current.case_results[0].score.id != legacy.case_results[0].score.id
    assert (
        current.case_results[0].attack_result.attack_result_id != legacy.case_results[0].attack_result.attack_result_id
    )
    assert current.case_results[0].score.status is ScoreStatus.COMPLETE
    assert current.case_results[0].score.score_value == legacy.case_results[0].score.score_value == "1.0"
    assert current.case_results[0].attack_result.outcome is AttackOutcome.UNDETERMINED
    assert (await importer.import_eval_log_async(path=archive)).case_results == current.case_results
    assert (await legacy_importer.import_eval_log_async(path=archive)).episode.model_dump(mode="json") == legacy_dump
    assert [
        piece.model_dump(mode="json")
        for piece in sqlite_instance.get_message_pieces(conversation_id=legacy_conversation_id)
    ] == legacy_piece_dump
    assert archive.read_bytes() == before


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("version", [True, 1, 2, 3, "3"])
def test_importer_requires_an_explicit_projection_enum(sqlite_instance: SQLiteMemory, version: object) -> None:
    with pytest.raises(TypeError, match="InspectProjectionVersion"):
        InspectOriginalEvalImporter(memory=sqlite_instance, projection_version=version)  # type: ignore[arg-type]


@pytest.mark.usefixtures("patch_central_database")
def test_projection_retains_empty_content_calls_and_does_not_fabricate_missing_call_ids(
    recorded_tool_log: EvalLog,
) -> None:
    assert recorded_tool_log.samples
    sample = recorded_tool_log.samples[0].model_copy(
        update={
            "messages": [
                ChatMessageAssistant(
                    content=[], tool_calls=[ToolCall(id="recorded-id", function="echo", arguments={})]
                ),
                ChatMessageTool(content="Original reply without an ID."),
                ChatMessageAssistant(
                    content="", tool_calls=[ToolCall(id="custom-id", function="custom", arguments={}, type="custom")]
                ),
            ]
        }
    )
    projection = project_inspect_sample(
        sample=sample,
        log_run_id=recorded_tool_log.eval.run_id,
        eval_id=recorded_tool_log.eval.eval_id,
        archive_sha256="a" * 64,
        sample_index=1,
        start_sequence=1,
        conversation_id=str(uuid.uuid4()),
    )
    assert len(projection.message_pieces) == 1
    assert json.loads(projection.message_pieces[0].original_value)["call_id"] == "recorded-id"
    assert projection.tool_result_ids == ()
    assert projection.unprojected_messages == 2


@pytest.mark.usefixtures("patch_central_database")
async def test_sealed_tool_projection_cannot_be_relabelled_as_legacy(
    tmp_path: Path, sqlite_instance: SQLiteMemory, recorded_tool_log: EvalLog
) -> None:
    archive = tmp_path / "public-tools-version.eval"
    await asyncio.to_thread(write_eval_log, recorded_tool_log, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        episode = session.get(NativeCyberEpisodeEntry, imported.episode.run.run_id)
        assert episode is not None
        episode.binding_version = "1"
        session.commit()
    with pytest.raises(ValueError, match="differs from its typed source"):
        await importer.import_eval_log_async(path=archive)


@pytest.mark.usefixtures("patch_central_database")
async def test_unmodified_inert_inspect_task_projects_source_score_without_inferring_success(
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
    [case] = imported.case_results
    assert case.sample_id == str(typed.samples[0].id) and case.epoch == typed.samples[0].epoch
    assert case.score.score_type == "float_scale" and case.score.score_value == "1.0"
    assert case.score.status is ScoreStatus.COMPLETE
    assert case.score.score_metadata["inspect_archive_sha256"] == imported.archive_sha256
    assert case.score.score_metadata["inspect_final_score_event_id"] == next(
        event.uuid for event in typed.samples[0].events if isinstance(event, ScoreEvent)
    )
    assert case.attack_result.automated_score == case.score
    assert case.attack_result.outcome is AttackOutcome.UNDETERMINED
    assert "No task-specific success" in case.attack_result.outcome_reason
    assert "no independent PyRIT grading" in imported.no_grade_reasons[0]
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 1
        result = session.get(AttackResultEntry, uuid.UUID(case.attack_result.attack_result_id))
        assert result is not None and result.automated_score_id == case.score.id

    repeated = await importer.import_eval_log_async(path=archive)
    assert repeated.episode.run.run_id == imported.episode.run.run_id
    assert repeated.episode.events == imported.episode.events
    assert repeated.case_results == imported.case_results
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 1


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
    [case] = imported.case_results
    assert case.score.status is ScoreStatus.COMPLETE
    assert case.attack_result.outcome is AttackOutcome.UNDETERMINED
    assert case.score.score_metadata["inspect_final_score_event_id"] == next(
        event.uuid for event in sample.events if isinstance(event, ScoreEvent) and not event.intermediate
    )
    retained = sqlite_instance.native_cyber_evidence.read_event_payloads(
        run_id=imported.episode.run.run_id, allow_sensitive=True
    )
    assert retained[0].event.payload["original"]["intermediate"] is True
    [final_capture] = [
        item
        for item in retained
        if item.event.source_event_id == case.score.score_metadata["inspect_final_score_event_id"]
    ]
    assert final_capture.event.payload["original"]["score"] == original.scores["original_inert_scorer"].model_dump(
        mode="json", exclude_none=True
    )
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 1


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("corruption", ["error_status", "sample_exception", "run_exception", "mismatched_final_score"])
async def test_offline_import_seals_original_error_or_score_disagreement_without_false_failure(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, corruption: str
) -> None:
    assert original_log_async.samples
    original = original_log_async.samples[0]
    if corruption == "error_status":
        typed = original_log_async.model_copy(update={"status": "error"})
    elif corruption in {"sample_exception", "run_exception"}:
        error = EvalError(message="fixture infrastructure exception", traceback="fixture", traceback_ansi="fixture")
        typed = original_log_async.model_copy(
            update={
                "status": "error",
                "samples": [original.model_copy(update={"error": error})]
                if corruption == "sample_exception"
                else [original],
                "error": error if corruption == "run_exception" else None,
            }
        )
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
    if corruption in {"sample_exception", "run_exception"}:
        source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
        cases, run = _case_inventory(source=source, samples=typed.samples or [])
        imported = await importer.import_eval_log_async(
            path=archive,
            cases=cases,
            run=run,
            score_policy=InspectOriginalScorePolicy(
                task_name=typed.eval.task,
                task_version=str(typed.eval.task_version),
                primary_scorer="original_inert_scorer",
                success_direction=InspectSuccessDirection.AT_LEAST,
                success_threshold=0.5,
            ),
        )
    else:
        imported = await importer.import_eval_log_async(path=archive)
    assert not imported.episode.coverage_complete
    assert imported.episode.gaps
    assert imported.episode.score_id is None
    assert imported.original_final_score_events == (0 if corruption == "mismatched_final_score" else 1)
    [case] = imported.case_results
    assert case.score.status is ScoreStatus.UNDETERMINED and case.score.score_value is None
    assert case.attack_result.outcome is (
        AttackOutcome.ERROR if corruption in {"sample_exception", "run_exception"} else AttackOutcome.UNDETERMINED
    )
    assert case.attack_result.outcome is not AttackOutcome.FAILURE
    if corruption in {"sample_exception", "run_exception"}:
        assert case.attack_result.error_type == "InspectEvalError"
        assert case.attack_result.error_message
    else:
        assert case.attack_result.error_type is None
    original_stream = next(
        stream
        for stream in imported.episode.raw_streams
        if stream.key.observed_source_id == "inspect-original-eval-archive"
    )
    assert original_stream.source_complete
    assert original_stream.stored_sha256 == hashlib.sha256(await asyncio.to_thread(archive.read_bytes)).hexdigest()
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 1


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
    assert len(imported.case_results) == 2
    assert all(case.score.status is ScoreStatus.COMPLETE for case in imported.case_results)
    assert all(case.attack_result.outcome is AttackOutcome.UNDETERMINED for case in imported.case_results)
    assert (
        imported.case_results[0].attack_result.conversation_id != imported.case_results[1].attack_result.conversation_id
    )
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
        assert session.scalar(select(func.count(ScoreEntry.id))) == 2
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 2


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("direction", "threshold", "outcomes"),
    [
        (InspectSuccessDirection.AT_LEAST, 0.5, (AttackOutcome.SUCCESS, AttackOutcome.FAILURE)),
        (InspectSuccessDirection.AT_MOST, 0.5, (AttackOutcome.FAILURE, AttackOutcome.SUCCESS)),
        (InspectSuccessDirection.AT_LEAST, 0.25, (AttackOutcome.SUCCESS, AttackOutcome.SUCCESS)),
    ],
)
async def test_offline_import_matches_final_events_across_retries_and_epochs_with_explicit_success_rule(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    original_log_async: EvalLog,
    direction: InspectSuccessDirection,
    threshold: float,
    outcomes: tuple[AttackOutcome, AttackOutcome],
) -> None:
    assert original_log_async.samples
    first = original_log_async.samples[0]
    assert first.scores
    scorer = "original_inert_scorer"
    first_final = next(event for event in first.events if isinstance(event, ScoreEvent))
    second_score = Score(value=0.25)
    retry_intermediate = ScoreEvent(
        uuid=f"retry-intermediate-{uuid.uuid4().hex}", scorer=scorer, intermediate=True, score=Score(value=0.5)
    )
    retry_final = ScoreEvent(uuid=f"retry-final-{uuid.uuid4().hex}", scorer=scorer, score=Score(value=0.75))
    second_events = [
        event.model_copy(
            update={
                "uuid": f"epoch2-{uuid.uuid4().hex}",
                **({"score": second_score} if isinstance(event, ScoreEvent) else {}),
            }
        )
        for event in first.events
    ]
    second = first.model_copy(
        update={
            "epoch": first.epoch + 1,
            "uuid": uuid.uuid4().hex,
            "scores": {scorer: second_score},
            "messages": [item.model_copy(update={"id": f"epoch2-{uuid.uuid4().hex}"}) for item in first.messages],
            "events": second_events,
            "error_retries": [
                EvalRetryError(
                    message="Inert first attempt failed",
                    traceback="fixture",
                    traceback_ansi="fixture",
                    events=[retry_intermediate, retry_final],
                )
            ],
        }
    )
    typed = original_log_async.model_copy(update={"samples": [first, second]})
    archive = tmp_path / f"two-epochs-{direction.value}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
    cases, run = _case_inventory(source=source, samples=[first, second])
    policy = InspectOriginalScorePolicy(
        task_name=typed.eval.task,
        task_version=str(typed.eval.task_version),
        primary_scorer=scorer,
        success_direction=direction,
        success_threshold=threshold,
    )
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    with patch.object(EvalSourceFactory, "resolve_original_inert", side_effect=AssertionError("no task import")):
        imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
        repeated = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    assert imported.episode.coverage_complete
    assert imported.episode.score_id is None
    assert imported.case_run_ids == tuple(run.case_run_id(case=case) for case in cases)
    assert len(set(imported.case_run_ids)) == 2
    assert [(item.sample_id, item.epoch) for item in imported.case_results] == [
        (str(first.id), first.epoch),
        (str(second.id), second.epoch),
    ]
    assert [item.score.score_value for item in imported.case_results] == ["1.0", "0.25"]
    assert [item.attack_result.outcome for item in imported.case_results] == list(outcomes)
    assert all(item.score.status is ScoreStatus.COMPLETE for item in imported.case_results)
    assert all(item.score.score_metadata["inspect_case_run_id"] == item.case_run_id for item in imported.case_results)
    assert all("pyrit_eval_role" not in item.score.score_metadata for item in imported.case_results)
    assert (
        imported.case_results[0].attack_result.conversation_id != imported.case_results[1].attack_result.conversation_id
    )
    assert [item.score.score_metadata["inspect_final_score_event_id"] for item in imported.case_results] == [
        first_final.uuid,
        next(event.uuid for event in second_events if isinstance(event, ScoreEvent)),
    ]
    events = sqlite_instance.native_cyber_evidence.read_event_payloads(
        run_id=imported.episode.run.run_id, allow_sensitive=True
    )
    assert [item.event.source_event_id for item in events if item.event.event_type == "score"] == [
        first_final.uuid,
        retry_intermediate.uuid,
        retry_final.uuid,
        next(event.uuid for event in second_events if isinstance(event, ScoreEvent)),
    ]
    archive_stream = next(
        item for item in imported.episode.raw_streams if item.key.observed_source_id == "inspect-original-eval-archive"
    )
    assert b"".join(
        chunk.data
        for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
            run_id=imported.episode.run.run_id, stream_id=archive_stream.stream_id, allow_sensitive=True
        )
    ) == await asyncio.to_thread(archive.read_bytes)
    assert repeated.case_results == imported.case_results
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 2
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 2


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("secondary_matches", [True, False])
async def test_multiple_final_scorers_need_reviewed_primary_and_all_declared_events_must_match(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, secondary_matches: bool
) -> None:
    assert original_log_async.samples
    original = original_log_async.samples[0]
    assert original.scores
    secondary_score = Score(value=0.3)
    secondary = ScoreEvent(
        uuid=f"secondary-{uuid.uuid4().hex}",
        scorer="secondary_scorer",
        score=secondary_score if secondary_matches else Score(value=0.9),
    )
    sample = original.model_copy(
        update={
            "scores": {**original.scores, "secondary_scorer": secondary_score},
            "events": [*original.events, secondary],
        }
    )
    typed = original_log_async.model_copy(update={"samples": [sample]})
    archive = tmp_path / f"multiple-final-{secondary_matches}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    unselected = await importer.import_eval_log_async(path=archive)
    assert unselected.case_results[0].score.status is ScoreStatus.UNDETERMINED
    assert unselected.case_results[0].attack_result.outcome is AttackOutcome.UNDETERMINED
    assert unselected.original_final_score_events == (2 if secondary_matches else 1)
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
    cases, run = _case_inventory(source=source, samples=[sample])
    policy = InspectOriginalScorePolicy(
        task_name=typed.eval.task,
        task_version=str(typed.eval.task_version),
        primary_scorer="secondary_scorer" if secondary_matches else "original_inert_scorer",
    )
    with patch.object(EvalSourceFactory, "resolve_original_inert", side_effect=AssertionError("no task import")):
        imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    [case] = imported.case_results
    assert case.primary_scorer == policy.primary_scorer
    assert case.score.status is (ScoreStatus.COMPLETE if secondary_matches else ScoreStatus.UNDETERMINED)
    assert case.score.score_value == ("0.3" if secondary_matches else None)
    assert case.attack_result.outcome is AttackOutcome.UNDETERMINED
    if secondary_matches:
        assert case.score.score_metadata["inspect_final_score_event_id"] == secondary.uuid
    else:
        assert "contradicts" in " ".join(imported.episode.gaps)
    source_events = sqlite_instance.native_cyber_evidence.read_event_payloads(
        run_id=imported.episode.run.run_id, allow_sensitive=True
    )
    assert secondary.uuid in [event.event.source_event_id for event in source_events]
    assert imported.archive_sha256 == hashlib.sha256(await asyncio.to_thread(archive.read_bytes)).hexdigest()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("variant", "expected_type", "expected_value", "expected_outcome"),
    [
        ("boolean", "true_false", "true", AttackOutcome.SUCCESS),
        ("boolean_false", "true_false", "false", AttackOutcome.FAILURE),
        ("string", "unknown", None, AttackOutcome.UNDETERMINED),
        ("out_of_range", "unknown", None, AttackOutcome.UNDETERMINED),
        ("large_integer", "unknown", None, AttackOutcome.UNDETERMINED),
        ("mapping", "unknown", None, AttackOutcome.UNDETERMINED),
        ("missing_event", "unknown", None, AttackOutcome.UNDETERMINED),
        ("missing_score", "unknown", None, AttackOutcome.UNDETERMINED),
    ],
)
async def test_original_scalar_values_project_only_when_supported_and_source_matched(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    original_log_async: EvalLog,
    variant: str,
    expected_type: str,
    expected_value: str | None,
    expected_outcome: AttackOutcome,
) -> None:
    assert original_log_async.samples
    original = original_log_async.samples[0]
    source_value: bool | str | int | float | dict[str, str] | None = {
        "boolean": True,
        "boolean_false": False,
        "string": "PASS",
        "out_of_range": 2.0,
        "large_integer": 10**400,
        "mapping": {"result": "fixture"},
        "missing_event": None,
        "missing_score": None,
    }[variant]
    score = Score(value=source_value) if source_value is not None else None
    events = [
        event.model_copy(update={"score": score}) if isinstance(event, ScoreEvent) and score else event
        for event in original.events
        if variant not in {"missing_event", "missing_score"} or not isinstance(event, ScoreEvent)
    ]
    sample = original.model_copy(
        update={
            "events": events,
            "scores": {}
            if variant == "missing_score"
            else ({"original_inert_scorer": score} if score else original.scores),
        }
    )
    typed = original_log_async.model_copy(update={"samples": [sample]})
    archive = tmp_path / f"source-value-{variant}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
    cases, run = _case_inventory(source=source, samples=[sample])
    policy = InspectOriginalScorePolicy(
        task_name=typed.eval.task,
        task_version=str(typed.eval.task_version),
        primary_scorer="original_inert_scorer",
        success_direction=InspectSuccessDirection.AT_LEAST,
        success_threshold=0.5,
    )
    imported = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_log_async(
        path=archive, cases=cases, run=run, score_policy=policy
    )
    [case] = imported.case_results
    assert case.score.score_type == expected_type
    assert case.score.score_value == expected_value
    assert case.score.status is (
        ScoreStatus.COMPLETE if variant in {"boolean", "boolean_false"} else ScoreStatus.UNDETERMINED
    )
    assert case.attack_result.outcome is expected_outcome
    assert case.attack_result.automated_score == case.score
    if variant in {"string", "out_of_range", "large_integer", "mapping"}:
        assert imported.original_final_score_events == 1
        assert case.score.score_metadata["inspect_final_score_event_id"] is not None
    else:
        assert imported.original_final_score_events == (1 if variant in {"boolean", "boolean_false"} else 0)


@pytest.mark.usefixtures("patch_central_database")
async def test_task_specific_success_policy_must_match_retained_task_and_bound_cases(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog
) -> None:
    archive = tmp_path / "policy-validation.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    with pytest.raises(ValueError, match="both a direction and a threshold"):
        InspectOriginalScorePolicy(
            task_name=original_log_async.eval.task,
            task_version=str(original_log_async.eval.task_version),
            primary_scorer="original_inert_scorer",
            success_direction=InspectSuccessDirection.AT_LEAST,
        )
    with pytest.raises(ValueError, match="between zero and one"):
        InspectOriginalScorePolicy(
            task_name=original_log_async.eval.task,
            task_version=str(original_log_async.eval.task_version),
            primary_scorer="original_inert_scorer",
            success_direction=InspectSuccessDirection.AT_LEAST,
            success_threshold=float("nan"),
        )
    policy = InspectOriginalScorePolicy(
        task_name=original_log_async.eval.task,
        task_version=str(original_log_async.eval.task_version),
        primary_scorer="original_inert_scorer",
        success_direction=InspectSuccessDirection.AT_LEAST,
        success_threshold=0.5,
    )
    with pytest.raises(ValueError, match="approved EvalCaseRef"):
        await importer.import_eval_log_async(path=archive, score_policy=policy)
    with pytest.raises(ValueError, match="differs from the retained Task"):
        await importer.import_eval_log_async(
            path=archive,
            score_policy=InspectOriginalScorePolicy(
                task_name="unapproved_task", task_version=policy.task_version, primary_scorer=policy.primary_scorer
            ),
        )
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(NativeCyberEpisodeEntry.run_id))) == 0


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("source_gap", ["missing_completion", "cancelled_run"])
async def test_reimport_rejects_forged_source_coverage_and_case_verdict(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, source_gap: str
) -> None:
    assert original_log_async.samples
    sample = original_log_async.samples[0]
    typed = (
        original_log_async.model_copy(update={"samples": [sample.model_copy(update={"completed_at": None})]})
        if source_gap == "missing_completion"
        else original_log_async.model_copy(update={"status": "cancelled"})
    )
    archive = tmp_path / f"source-gap-{source_gap}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
    cases, run = _case_inventory(source=source, samples=typed.samples or [])
    policy = InspectOriginalScorePolicy(
        task_name=typed.eval.task,
        task_version=str(typed.eval.task_version),
        primary_scorer="original_inert_scorer",
        success_direction=InspectSuccessDirection.AT_LEAST,
        success_threshold=0.5,
    )
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    [case] = imported.case_results
    assert not imported.episode.coverage_complete
    assert case.score.status is ScoreStatus.UNDETERMINED
    assert case.attack_result.outcome is AttackOutcome.UNDETERMINED
    with sqlite_instance.get_session() as session, session.begin():
        episode = session.get(NativeCyberEpisodeEntry, imported.episode.run.run_id)
        turn = session.get(NativeCyberTurnEntry, (imported.episode.run.run_id, 1))
        score = session.get(ScoreEntry, case.score.id)
        result = session.get(AttackResultEntry, uuid.UUID(case.attack_result.attack_result_id))
        assert episode is not None and turn is not None and score is not None and result is not None
        episode.capture_gaps = []
        episode.coverage_complete = True
        turn.capture_gaps = []
        turn.source_complete = True
        score.status = ScoreStatus.COMPLETE.value
        score.score_value = "1.0"
        score.score_value_description = None
        result.outcome = AttackOutcome.SUCCESS.value
        result.outcome_reason = (
            "Original Inspect original_inert_scorer compared using the explicit at_least 0.5 success criterion."
        )
    with pytest.raises(ValueError, match="source coverage"):
        await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)


@pytest.mark.usefixtures("patch_central_database")
async def test_cancelled_two_sample_reread_rejects_run_gap_laundered_into_first_turn(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    cancelled_offline_eval: tuple[EvalLog, tuple[EvalCaseRef, ...], EvalRunRef, InspectOriginalScorePolicy],
) -> None:
    log, cases, run, policy = cancelled_offline_eval
    archive = tmp_path / "cancelled-two-samples.eval"
    await asyncio.to_thread(write_eval_log, log, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    with patch.object(EvalSourceFactory, "resolve_original_inert", side_effect=AssertionError("no Task source")):
        imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    assert imported.sample_count == 2
    assert [case.score.status for case in imported.case_results] == [ScoreStatus.UNDETERMINED] * 2
    assert [case.attack_result.outcome for case in imported.case_results] == [AttackOutcome.UNDETERMINED] * 2
    assert all(turn.source_complete for turn in imported.episode.turns)
    assert len(imported.episode.gaps) == 1
    [run_gap] = imported.episode.gaps
    with sqlite_instance.get_session() as session, session.begin():
        turn = session.get(NativeCyberTurnEntry, (imported.episode.run.run_id, 1))
        second = imported.case_results[1]
        score = session.get(ScoreEntry, second.score.id)
        result = session.get(AttackResultEntry, uuid.UUID(second.attack_result.attack_result_id))
        assert turn is not None and score is not None and result is not None
        turn.capture_gaps = [run_gap]
        turn.source_complete = False
        score.status = ScoreStatus.COMPLETE.value
        score.score_value = "1.0"
        score.score_value_description = None
        result.outcome = AttackOutcome.SUCCESS.value
        result.outcome_reason = (
            "Original Inspect recorded_scorer compared using the explicit at_least 0.5 success criterion."
        )
    with (
        patch.object(EvalSourceFactory, "resolve_original_inert", side_effect=AssertionError("no Task source")),
        pytest.raises(ValueError, match="Sample coverage"),
    ):
        await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)


@pytest.mark.usefixtures("patch_central_database")
async def test_cancelled_two_sample_reread_rejects_unattributed_turn_gap(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    cancelled_offline_eval: tuple[EvalLog, tuple[EvalCaseRef, ...], EvalRunRef, InspectOriginalScorePolicy],
) -> None:
    log, cases, run, policy = cancelled_offline_eval
    archive = tmp_path / "cancelled-unattributed-gap.eval"
    await asyncio.to_thread(write_eval_log, log, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    with sqlite_instance.get_session() as session, session.begin():
        episode = session.get(NativeCyberEpisodeEntry, imported.episode.run.run_id)
        turn = session.get(NativeCyberTurnEntry, (imported.episode.run.run_id, 1))
        assert episode is not None and turn is not None
        turn.capture_gaps = ["unattributed turn gap"]
        turn.source_complete = False
        episode.capture_gaps = [*episode.capture_gaps, "unattributed turn gap"]
    with pytest.raises(ValueError, match="Sample coverage"):
        await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("capture_kind", ["missing_completion", "duplicate_tool_phase"])
async def test_cancelled_two_sample_reread_keeps_genuine_sample_and_capture_gaps(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    cancelled_offline_eval: tuple[EvalLog, tuple[EvalCaseRef, ...], EvalRunRef, InspectOriginalScorePolicy],
    capture_kind: str,
) -> None:
    log, cases, run, policy = cancelled_offline_eval
    assert log.samples
    if capture_kind == "missing_completion":
        samples = [log.samples[0].model_copy(update={"completed_at": None}), log.samples[1]]
        affected_turn = 0
        expected_gap = "Original Inspect Sample errored or lacks a completion timestamp."
    else:
        samples = [
            sample.model_copy(
                update={
                    "events": [
                        ToolEvent(
                            uuid=f"tool-{uuid.uuid4().hex}",
                            id="same-tool-call",
                            function="inert_tool",
                            arguments={"value": "fixture"},
                            result="fixture",
                            completed=datetime.now(UTC),
                        ),
                        *sample.events,
                    ]
                }
            )
            for sample in log.samples
        ]
        affected_turn = 1
        expected_gap = "Native tool phase complete was observed more than once"
    log = log.model_copy(update={"samples": samples})
    archive = tmp_path / f"genuine-gap-{capture_kind}.eval"
    await asyncio.to_thread(write_eval_log, log, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    assert expected_gap in " ".join(imported.episode.turns[affected_turn].gaps)
    assert all(case.attack_result.outcome is AttackOutcome.UNDETERMINED for case in imported.case_results)
    repeated = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    assert repeated.case_results == imported.case_results
    assert repeated.episode.turns == imported.episode.turns


@pytest.mark.usefixtures("patch_central_database")
async def test_two_sample_offline_success_keeps_distinct_source_verdicts_without_task_execution(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    cancelled_offline_eval: tuple[EvalLog, tuple[EvalCaseRef, ...], EvalRunRef, InspectOriginalScorePolicy],
) -> None:
    log, cases, run, policy = cancelled_offline_eval
    assert log.samples
    second_score = Score(value=0.0)
    second = log.samples[1].model_copy(
        update={
            "scores": {"recorded_scorer": second_score},
            "events": [
                event.model_copy(update={"score": second_score}) if isinstance(event, ScoreEvent) else event
                for event in log.samples[1].events
            ],
        }
    )
    log = log.model_copy(update={"status": "success", "samples": [log.samples[0], second]})
    archive = tmp_path / "two-sample-offline-success.eval"
    await asyncio.to_thread(write_eval_log, log, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    assert imported.episode.coverage_complete
    assert [case.score.score_value for case in imported.case_results] == ["1.0", "0.0"]
    assert [case.attack_result.outcome for case in imported.case_results] == [
        AttackOutcome.SUCCESS,
        AttackOutcome.FAILURE,
    ]
    assert len({case.attack_result.conversation_id for case in imported.case_results}) == 2
    reread = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policy)
    assert reread.case_results == imported.case_results
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 2
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 2


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(("first_threshold", "same_threshold"), [(1, 1.0), (-0.0, 0.0)])
async def test_equivalent_thresholds_reuse_one_import_and_case_projection(
    tmp_path: Path,
    sqlite_instance: SQLiteMemory,
    original_log_async: EvalLog,
    first_threshold: float,
    same_threshold: float,
) -> None:
    assert original_log_async.samples
    archive = tmp_path / "equivalent-policy.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
    cases, run = _case_inventory(source=source, samples=original_log_async.samples)
    policies = [
        InspectOriginalScorePolicy(
            task_name=original_log_async.eval.task,
            task_version=str(original_log_async.eval.task_version),
            primary_scorer="original_inert_scorer",
            success_direction=InspectSuccessDirection.AT_LEAST,
            success_threshold=threshold,
        )
        for threshold in (first_threshold, same_threshold)
    ]
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    first = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policies[0])
    same = await importer.import_eval_log_async(path=archive, cases=cases, run=run, score_policy=policies[1])
    assert first.episode.run.run_id == same.episode.run.run_id
    assert first.case_results == same.case_results
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(NativeCyberEpisodeEntry.run_id))) == 1
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 1


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("corrupt", ["archive", "event", "event_rehashed"])
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
            if corrupt == "event_rehashed":
                event.payload_sha256 = sqlite_instance.native_cyber_evidence._hash_payload(event.payload)
    with pytest.raises(
        ValueError, match="corrupt byte range|failed integrity validation|differs from its typed source"
    ):
        await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 1


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("source_event", ["first", "final_score", "retry_intermediate"])
async def test_reimport_rejects_tampered_original_event_timestamp_without_payload_change(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, source_event: str
) -> None:
    assert original_log_async.samples
    sample = original_log_async.samples[0]
    retry_event: ScoreEvent | None = None
    if source_event == "retry_intermediate":
        retry_event = ScoreEvent(
            uuid=f"retry-{uuid.uuid4().hex}",
            scorer="original_inert_scorer",
            score=Score(value=0.25),
            intermediate=True,
        )
        retry = EvalRetryError(
            message="Inert earlier attempt failed",
            traceback="fixture",
            traceback_ansi="fixture",
            events=[retry_event],
        )
        sample = sample.model_copy(update={"error_retries": [retry]})
    typed = original_log_async.model_copy(update={"samples": [sample]})
    archive = tmp_path / f"event-time-{source_event}.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    if source_event == "retry_intermediate":
        assert retry_event is not None
        source_id = retry_event.uuid
    elif source_event == "final_score":
        source_id = next(event.uuid for event in sample.events if isinstance(event, ScoreEvent))
    else:
        source_id = imported.episode.events[0].observed_event_id
    event_summary = next(event for event in imported.episode.events if event.observed_event_id == source_id)
    with sqlite_instance.get_session() as session, session.begin():
        event = session.get(NativeCyberEventEntry, (imported.episode.run.run_id, event_summary.sequence))
        assert event is not None and event.event_type != "inspect.projection.sample"
        original_payload_digest = event.payload_sha256
        event.captured_at += timedelta(days=1)
        assert event.payload_sha256 == original_payload_digest
    with pytest.raises(ValueError, match="timestamp differs from its typed source"):
        await importer.import_eval_log_async(path=archive)


@pytest.mark.usefixtures("patch_central_database")
async def test_reimport_does_not_require_a_source_timestamp_for_synthetic_sample_summary(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog
) -> None:
    archive = tmp_path / "summary-time.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    summary = next(event for event in imported.episode.events if event.event_type == "inspect.projection.sample")
    with sqlite_instance.get_session() as session, session.begin():
        event = session.get(NativeCyberEventEntry, (imported.episode.run.run_id, summary.sequence))
        assert event is not None
        event.captured_at += timedelta(seconds=10)
    repeated = await importer.import_eval_log_async(path=archive)
    assert repeated.case_results == imported.case_results


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "corrupt", ["score_value", "score_metadata", "result_outcome", "result_score_link", "both_deleted"]
)
async def test_reimport_rejects_tampered_projected_score_or_result(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog, corrupt: str
) -> None:
    archive = tmp_path / f"projected-{corrupt}.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)
    imported = await importer.import_eval_log_async(path=archive)
    [case] = imported.case_results
    with sqlite_instance.get_session() as session, session.begin():
        score = session.get(ScoreEntry, case.score.id)
        result = session.get(AttackResultEntry, uuid.UUID(case.attack_result.attack_result_id))
        assert score is not None and result is not None
        if corrupt == "score_value":
            score.score_value = "0.0"
        elif corrupt == "score_metadata":
            score.score_metadata = {**score.score_metadata, "inspect_archive_sha256": "f" * 64}
        elif corrupt == "result_outcome":
            result.outcome = AttackOutcome.FAILURE.value
        elif corrupt == "both_deleted":
            session.delete(result)
            session.delete(score)
        else:
            result.automated_score_id = None
    with pytest.raises(ValueError, match="Score/AttackResult (differs from its typed source|projection is partial)"):
        await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        expected_count = 0 if corrupt == "both_deleted" else 1
        assert session.scalar(select(func.count(ScoreEntry.id))) == expected_count
        assert session.scalar(select(func.count(AttackResultEntry.id))) == expected_count


@pytest.mark.usefixtures("patch_central_database")
async def test_score_and_attack_result_write_is_atomic_and_incomplete_projection_fails_closed(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog
) -> None:
    archive = tmp_path / "interrupted-projection.eval"
    await asyncio.to_thread(write_eval_log, original_log_async, location=archive, format="eval")
    importer = InspectOriginalEvalImporter(memory=sqlite_instance)

    def _interrupt_after_score_flush(*, session: Session, attack_results: Sequence[AttackResult]) -> None:
        assert len(attack_results) == 1
        session.flush()
        raise ValueError("interrupted after score flush")

    with patch.object(sqlite_instance, "_persist_attack_result_rows", side_effect=_interrupt_after_score_flush):
        with pytest.raises(ValueError, match="interrupted after score flush"):
            await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0
        episode = session.scalar(select(NativeCyberEpisodeEntry))
        assert episode is not None and episode.finalized_at is not None
    with pytest.raises(ValueError, match="projection is partial or missing"):
        await importer.import_eval_log_async(path=archive)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(NativeCyberEpisodeEntry.run_id))) == 1
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_non_text_original_sample_still_has_its_own_persisted_case_conversation(
    tmp_path: Path, sqlite_instance: SQLiteMemory, original_log_async: EvalLog
) -> None:
    assert original_log_async.samples
    sample = original_log_async.samples[0].model_copy(update={"messages": []})
    typed = original_log_async.model_copy(update={"samples": [sample]})
    archive = tmp_path / "nontext-case.eval"
    await asyncio.to_thread(write_eval_log, typed, location=archive, format="eval")
    imported = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_log_async(path=archive)
    assert imported.message_piece_count == 0
    [case] = imported.case_results
    assert case.score.status is ScoreStatus.COMPLETE
    with sqlite_instance.get_session() as session:
        assert session.get(ConversationEntry, case.attack_result.conversation_id) is not None
    assert not sqlite_instance.get_conversation_messages(conversation_id=case.attack_result.conversation_id)


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
    assert imported.case_results[0].score.status is ScoreStatus.UNDETERMINED
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1


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
        assert imported.case_results[0].score.status is ScoreStatus.COMPLETE
    else:
        assert not imported.episode.coverage_complete
        assert "unresolved attachment" in " ".join(imported.episode.gaps)
        assert imported.case_results[0].score.status is ScoreStatus.UNDETERMINED
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 1


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
