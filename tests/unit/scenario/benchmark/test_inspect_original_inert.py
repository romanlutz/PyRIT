# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real public original Inspect Task, SQLite and one-click Scenario contract."""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from inspect_ai.event import ScoreEvent
from inspect_ai.log import read_eval_log
from sqlalchemy import func, select

from pyrit.backend.main import app
from pyrit.backend.services.attack_service import AttackService
from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.backend.services.scenario_service import ScenarioService
from pyrit.common.utils import to_sha256
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter
from pyrit.memory.memory_models import (
    AttackResultEntry,
    NativeCyberEpisodeEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    NativeCyberTurnMessagePieceEntry,
    PromptMemoryEntry,
    ScoreEntry,
)
from pyrit.models import (
    AtomicAttackIdentifier,
    AttackIdentifier,
    AttackOutcome,
    AttackResult,
    MessagePiece,
    ScenarioRunPlan,
    ScenarioRunState,
    ScoreStatus,
    config_hash,
)
from pyrit.models.catalog.scenario import OriginalInspectImportSummary, OriginalInspectTaskId, RunScenarioRequest
from pyrit.models.native_cyber_evidence import NativeCyberEvidenceSource, NativeCyberRawKind
from pyrit.models.score.observation import _message_piece_digest
from pyrit.registry import ScenarioRegistry
from pyrit.scenario.scenarios.benchmark.inspect_original_inert import (
    InspectOriginalInertScenario,
    _allocate_log_dir,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory


@pytest.mark.usefixtures("patch_central_database")
async def test_registry_discovers_only_pinned_public_task_without_executing_it() -> None:
    with patch.object(EvalSourceFactory, "resolve_original_inert", side_effect=AssertionError("Task was loaded")):
        registry = ScenarioRegistry()
        assert registry.get_class("benchmark.inspect_original_inert") is InspectOriginalInertScenario
        metadata = registry.get_class_metadata(InspectOriginalInertScenario)
        assert metadata.default_techniques == ("original_task",)
        assert metadata.baseline_policy == "forbidden"
        parameters = {parameter.name: parameter for parameter in metadata.supported_parameters}
        assert "objective_target" not in parameters
        assert "trusted_eval_dir" not in parameters
        assert "memory_labels" not in parameters
        assert parameters["eval_family"].choices == [OriginalInspectTaskId.INERT.value]
        assert parameters["max_concurrency"].default == 1

        service = ScenarioService()
        service._registry = registry
        scenario = await service.get_scenario_async(scenario_name="benchmark.inspect_original_inert")

    assert scenario is not None
    assert scenario.default_run_size.estimated_attack_count == 1
    assert scenario.all_techniques == ["original_task"]


@pytest.mark.usefixtures("patch_central_database")
async def test_one_click_runs_unchanged_inspect_twice_with_distinct_offline_sqlite_results(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    registry = ScenarioRegistry()
    run_references: list[OriginalInspectImportSummary] = []
    run_ids: list[str] = []
    original_dirs: list[Path] = []

    def allocate_test_dir(*, run_instance_id: uuid.UUID) -> Path:
        directory = tmp_path / f"inspect-original-{run_instance_id.hex}"
        directory.mkdir()
        original_dirs.append(directory)
        return directory

    with patch("pyrit.scenario.scenarios.benchmark.inspect_original_inert._allocate_log_dir", new=allocate_test_dir):
        for _ in range(2):
            scenario = await registry.create_and_initialize_async(
                "benchmark.inspect_original_inert",
                scenario_params={"eval_family": OriginalInspectTaskId.INERT.value},
            )
            assert scenario.atomic_attack_count == 1
            [work] = scenario._atomic_attacks
            result = await scenario.run_async()
            run_ids.append(str(result.id))
            assert result.scenario_run_state == ScenarioRunState.COMPLETED
            assert result.attack_results == {}
            with pytest.raises(RuntimeError, match="replay is disabled"):
                await scenario.run_async()

            reference = OriginalInspectImportSummary.model_validate(
                result.metadata[OriginalInspectImportSummary.METADATA_KEY]
            )
            run_references.append(reference)
            assert reference.task_id is OriginalInspectTaskId.INERT
            assert reference.case_run_id == work.case_run_id
            assert reference.source_sha256 == EvalSourceFactory.ORIGINAL_INERT_SHA256
            assert reference.score_status is ScoreStatus.COMPLETE
            assert reference.score_type == "float_scale"
            assert reference.score_value == "1.0"
            assert reference.outcome is AttackOutcome.UNDETERMINED
            assert reference.primary_scorer == "original_inert_scorer"
            assert reference.episode_id == f"inspect-run-{work.run.run_instance_id.hex}"
            assert reference.episode_id != reference.projection_episode_id

            snapshot = sqlite_instance.native_cyber_evidence.get_finalized_unscored_inspect_capture(
                run_id=reference.episode_id
            )
            assert snapshot.coverage_complete
            assert snapshot.score_id is None and snapshot.score_status is ScoreStatus.UNDETERMINED
            projected_snapshot = sqlite_instance.native_cyber_evidence.get_finalized_unscored_inspect_capture(
                run_id=reference.projection_episode_id
            )
            assert projected_snapshot.coverage_complete
            assert projected_snapshot.score_id is None
            assert len(projected_snapshot.turns) == 1
            archive_stream = next(
                stream
                for stream in snapshot.raw_streams
                if stream.key.observed_source_id == "inspect-original-eval-archive"
            )
            archive = b"".join(
                chunk.data
                for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
                    run_id=reference.episode_id,
                    stream_id=archive_stream.stream_id,
                    allow_sensitive=True,
                )
            )
            assert hashlib.sha256(archive).hexdigest() == reference.archive_sha256
            projected_stream = next(
                stream
                for stream in projected_snapshot.raw_streams
                if stream.key.observed_source_id == "inspect-original-eval-archive"
            )
            assert archive == b"".join(
                chunk.data
                for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
                    run_id=reference.projection_episode_id,
                    stream_id=projected_stream.stream_id,
                    allow_sensitive=True,
                )
            )
            typed = await asyncio.to_thread(
                read_eval_log, io.BytesIO(archive), resolve_attachments="full", format="eval"
            )
            assert typed.eval.run_id == reference.inspect_run_id
            assert typed.eval.eval_id == reference.inspect_eval_id
            assert typed.samples is not None and len(typed.samples) == 1
            assert typed.samples[0].store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
            assert len([event for event in typed.samples[0].events if isinstance(event, ScoreEvent)]) == 1
            assert typed.samples[0].scores and typed.samples[0].scores["original_inert_scorer"].value == 1.0
            assert typed.samples[0].model_usage == {}
            [score] = sqlite_instance.get_scores(score_ids=[str(reference.score_id)])
            [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(reference.attack_result_id)])
            assert score.status is ScoreStatus.COMPLETE and score.score_value == "1.0"
            assert score.score_metadata["inspect_archive_sha256"] == reference.archive_sha256
            assert score.score_metadata["inspect_case_run_id"] == reference.case_run_id
            final_event = next(event for event in typed.samples[0].events if isinstance(event, ScoreEvent))
            assert score.score_metadata["inspect_final_score_event_id"] == final_event.uuid
            assert score.score_metadata["inspect_final_score_event_sha256"] == config_hash(
                {"event": final_event.model_dump(mode="json", exclude_none=True)}
            )
            assert attack.automated_score == score
            assert attack.outcome is AttackOutcome.UNDETERMINED
            assert attack.attribution_parent_id is None
            plan = ScenarioRunPlan.model_validate(result.metadata["run_plan"])
            assert plan.run_instance_id == work.run.run_instance_id
            assert plan.seed_groups[0].id == reference.case_run_id
            assert plan.seed_groups[0].source_sha256 == reference.source_sha256

    assert len({reference.episode_id for reference in run_references}) == 2
    assert len({reference.projection_episode_id for reference in run_references}) == 2
    assert len({reference.case_run_id for reference in run_references}) == 2
    assert len({reference.inspect_run_id for reference in run_references}) == 2
    assert len({reference.archive_sha256 for reference in run_references}) == 2
    assert len({reference.score_id for reference in run_references}) == 2
    assert len({reference.attack_result_id for reference in run_references}) == 2
    assert all(not directory.exists() for directory in original_dirs)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 2
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 2

    first_id = run_ids[0]
    original = run_references[0].model_dump(mode="json")
    other = run_references[1]
    swaps = (
        ("score_id", str(uuid.uuid4())),
        ("score_id", str(other.score_id)),
        ("attack_result_id", str(uuid.uuid4())),
        ("attack_result_id", str(other.attack_result_id)),
        ("projection_episode_id", other.projection_episode_id),
    )
    [original_score] = sqlite_instance.get_scores(score_ids=[original["score_id"]])
    [original_attack] = sqlite_instance.get_attack_results(attack_result_ids=[original["attack_result_id"]])
    assert original_score.score_metadata is not None
    original_score_metadata = dict(original_score.score_metadata)
    original_attack_metadata = dict(original_attack.metadata)

    def set_stored_corruption(*, corruption: str | None) -> None:
        score_id = uuid.UUID(str(original_score.id))
        with sqlite_instance.get_session() as session, session.begin():
            score_row = session.get(ScoreEntry, score_id)
            attack_row = session.get(AttackResultEntry, uuid.UUID(original_attack.attack_result_id))
            assert score_row is not None and attack_row is not None
            score_row.score_metadata = original_score_metadata
            attack_row.attack_metadata = original_attack_metadata
            attack_row.automated_score_id = score_id
            attack_row.outcome = AttackOutcome.UNDETERMINED.value
            if corruption == "score_case":
                score_row.score_metadata = {**original_score_metadata, "inspect_case_run_id": other.case_run_id}
            elif corruption == "score_archive":
                score_row.score_metadata = {**original_score_metadata, "inspect_archive_sha256": other.archive_sha256}
            elif corruption == "attack_archive":
                attack_row.attack_metadata = {
                    **original_attack_metadata,
                    "inspect_archive_sha256": other.archive_sha256,
                }
            elif corruption == "attack_score_fk":
                attack_row.automated_score_id = other.score_id
            elif corruption == "attack_outcome":
                attack_row.outcome = AttackOutcome.SUCCESS.value
            elif corruption is not None:
                raise AssertionError(f"Unknown stored corruption: {corruption}")

    service = ScenarioRunService()
    try:
        with patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                endpoints = (
                    f"/api/scenarios/runs/{first_id}",
                    f"/api/scenarios/runs/{first_id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                )
                for field, swapped_id in swaps:
                    await asyncio.to_thread(
                        sqlite_instance.update_scenario_metadata_fields,
                        scenario_result_id=first_id,
                        fields={OriginalInspectImportSummary.METADATA_KEY: {**original, field: swapped_id}},
                    )
                    for endpoint in endpoints:
                        response = await client.get(endpoint)
                        assert response.status_code >= 400, f"{field}={swapped_id} was accepted by {endpoint}"
                        assert "harmless fixture" not in response.text
                        assert "inert response" not in response.text

                await asyncio.to_thread(
                    sqlite_instance.update_scenario_metadata_fields,
                    scenario_result_id=first_id,
                    fields={OriginalInspectImportSummary.METADATA_KEY: original},
                )
                for corruption in (
                    "score_case",
                    "score_archive",
                    "attack_archive",
                    "attack_score_fk",
                    "attack_outcome",
                ):
                    await asyncio.to_thread(set_stored_corruption, corruption=corruption)
                    for endpoint in endpoints:
                        response = await client.get(endpoint)
                        assert response.status_code >= 400, f"{corruption} was accepted by {endpoint}"
                        assert "harmless fixture" not in response.text
                    await asyncio.to_thread(set_stored_corruption, corruption=None)

                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "tamper",
    [
        "score_event",
        "archive_chunk",
        "archive_chunk_rehashed",
        "archive_truncated",
        "archive_source",
        "archive_kind",
        "archive_not_required",
        "resolved_chunk",
        "resolved_chunk_rehashed",
        "resolved_missing",
        "resolved_kind",
        "resolved_not_required",
        "extra_required_missing",
    ],
)
async def test_one_click_readback_rejects_tampered_final_event_or_archive_evidence(
    sqlite_instance: SQLiteMemory, tamper: str
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    reference = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    episode = sqlite_instance.native_cyber_evidence.get_finalized_unscored_inspect_capture(
        run_id=reference.projection_episode_id
    )
    archive = next(
        stream for stream in episode.raw_streams if stream.key.observed_source_id == "inspect-original-eval-archive"
    )
    resolved_stream = next(
        stream for stream in episode.raw_streams if stream.key.observed_source_id == "inspect-resolved-eval-log"
    )
    assert InspectOriginalEvalImporter.ARCHIVE_KEY in episode.run.required_raw_streams
    assert InspectOriginalEvalImporter.RESOLVED_KEY in episode.run.required_raw_streams
    with sqlite_instance.get_session() as session:
        score = session.get(ScoreEntry, reference.score_id)
        attack = session.get(AttackResultEntry, reference.attack_result_id)
        chunk = session.get(NativeCyberRawChunkEntry, (archive.stream_id, 1))
        stream = session.get(NativeCyberRawStreamEntry, archive.stream_id)
        resolved_chunk = session.get(NativeCyberRawChunkEntry, (resolved_stream.stream_id, 1))
        resolved_row = session.get(NativeCyberRawStreamEntry, resolved_stream.stream_id)
        episode_row = session.get(NativeCyberEpisodeEntry, reference.projection_episode_id)
        assert score is not None and score.score_metadata is not None
        assert attack is not None and attack.attack_metadata is not None
        assert chunk is not None and len(chunk.data) > 1
        assert resolved_chunk is not None and len(resolved_chunk.data) > 1
        assert stream is not None and resolved_row is not None and episode_row is not None
        original_score_metadata = dict(score.score_metadata)
        original_attack_metadata = dict(attack.attack_metadata)
        original_chunk = chunk.data
        original_chunk_sha256 = chunk.sha256
        original_chunk_length = chunk.byte_length
        original_source = stream.source
        original_kind = stream.kind
        original_resolved_chunk = resolved_chunk.data
        original_resolved_chunk_sha256 = resolved_chunk.sha256
        original_resolved_stored_sha256 = resolved_row.stored_sha256
        original_resolved_observed_sha256 = resolved_row.observed_sha256
        original_resolved_id = resolved_row.observed_source_id
        original_resolved_kind = resolved_row.kind
        original_required = [dict(key) for key in episode_row.required_raw_streams]

    def set_tamper(*, enabled: bool) -> None:
        with sqlite_instance.get_session() as session, session.begin():
            score = session.get(ScoreEntry, reference.score_id)
            attack = session.get(AttackResultEntry, reference.attack_result_id)
            chunk = session.get(NativeCyberRawChunkEntry, (archive.stream_id, 1))
            stream = session.get(NativeCyberRawStreamEntry, archive.stream_id)
            resolved_chunk = session.get(NativeCyberRawChunkEntry, (resolved_stream.stream_id, 1))
            resolved_row = session.get(NativeCyberRawStreamEntry, resolved_stream.stream_id)
            episode_row = session.get(NativeCyberEpisodeEntry, reference.projection_episode_id)
            assert score is not None and attack is not None and chunk is not None and resolved_chunk is not None
            assert stream is not None and resolved_row is not None and episode_row is not None
            event_fields = (
                {"inspect_final_score_event_id": "forged-final-event", "inspect_final_score_event_sha256": "f" * 64}
                if enabled and tamper == "score_event"
                else {}
            )
            score.score_metadata = {**original_score_metadata, **event_fields}
            attack.attack_metadata = {**original_attack_metadata, **event_fields}
            if enabled and tamper == "archive_truncated":
                chunk.data = original_chunk[:-1]
                chunk.byte_length = len(chunk.data)
            elif enabled and tamper in {"archive_chunk", "archive_chunk_rehashed"}:
                chunk.data = bytes([original_chunk[0] ^ 1]) + original_chunk[1:]
                chunk.byte_length = original_chunk_length
            else:
                chunk.data = original_chunk
                chunk.byte_length = original_chunk_length
            chunk.sha256 = (
                hashlib.sha256(chunk.data).hexdigest()
                if enabled and tamper in {"archive_chunk_rehashed", "archive_truncated"}
                else original_chunk_sha256
            )
            stream.source = (
                NativeCyberEvidenceSource.TOOL.value if enabled and tamper == "archive_source" else original_source
            )
            stream.kind = NativeCyberRawKind.JSONL.value if enabled and tamper == "archive_kind" else original_kind
            resolved_chunk.data = (
                bytes([original_resolved_chunk[0] ^ 1]) + original_resolved_chunk[1:]
                if enabled and tamper in {"resolved_chunk", "resolved_chunk_rehashed"}
                else original_resolved_chunk
            )
            resolved_chunk.sha256 = (
                hashlib.sha256(resolved_chunk.data).hexdigest()
                if enabled and tamper == "resolved_chunk_rehashed"
                else original_resolved_chunk_sha256
            )
            resolved_row.stored_sha256 = (
                hashlib.sha256(resolved_chunk.data).hexdigest()
                if enabled and tamper == "resolved_chunk_rehashed"
                else original_resolved_stored_sha256
            )
            resolved_row.observed_sha256 = (
                hashlib.sha256(resolved_chunk.data).hexdigest()
                if enabled and tamper == "resolved_chunk_rehashed"
                else original_resolved_observed_sha256
            )
            resolved_row.observed_source_id = (
                "missing-resolved-eval-log" if enabled and tamper == "resolved_missing" else original_resolved_id
            )
            resolved_row.kind = (
                NativeCyberRawKind.TOOL.value if enabled and tamper == "resolved_kind" else original_resolved_kind
            )
            required = [dict(key) for key in original_required]
            if enabled and tamper in {"archive_not_required", "resolved_not_required"}:
                removed = (
                    InspectOriginalEvalImporter.ARCHIVE_KEY.observed_source_id
                    if tamper == "archive_not_required"
                    else InspectOriginalEvalImporter.RESOLVED_KEY.observed_source_id
                )
                required = [key for key in required if key["observed_source_id"] != removed]
            elif enabled and tamper == "extra_required_missing":
                required.append({**original_required[1], "observed_source_id": "missing-required-stream"})
            episode_row.required_raw_streams = required

    service = ScenarioRunService()
    try:
        with patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                endpoints = (
                    f"/api/scenarios/runs/{result.id}",
                    f"/api/scenarios/runs/{result.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                )
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200

                await asyncio.to_thread(set_tamper, enabled=True)
                for endpoint in endpoints:
                    response = await client.get(endpoint)
                    assert response.status_code >= 400, f"{tamper} was accepted by {endpoint}"
                    assert "harmless fixture" not in response.text
                    assert "inert response" not in response.text
                    assert "forged-final-event" not in response.text

                await asyncio.to_thread(set_tamper, enabled=False)
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("tamper", ["text", "role", "source_metadata"])
async def test_one_click_readback_rejects_rehashed_source_message_piece(
    sqlite_instance: SQLiteMemory, tamper: str
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
    user_piece = next(
        piece
        for piece in sqlite_instance.get_message_pieces(conversation_id=attack.conversation_id)
        if piece.role == "user"
    )
    with sqlite_instance.get_session() as session:
        row = session.get(PromptMemoryEntry, user_piece.id)
        link = session.execute(
            select(NativeCyberTurnMessagePieceEntry).where(
                NativeCyberTurnMessagePieceEntry.run_id == imported.projection_episode_id,
                NativeCyberTurnMessagePieceEntry.message_piece_id == user_piece.id,
            )
        ).scalar_one()
        assert row is not None and row.prompt_metadata is not None
        original = (
            row.role,
            row.original_value,
            row.original_value_sha256,
            dict(row.prompt_metadata),
            link.piece_sha256,
        )

    def set_tamper(*, enabled: bool) -> None:
        with sqlite_instance.get_session() as session, session.begin():
            row = session.get(PromptMemoryEntry, user_piece.id)
            link = session.execute(
                select(NativeCyberTurnMessagePieceEntry).where(
                    NativeCyberTurnMessagePieceEntry.run_id == imported.projection_episode_id,
                    NativeCyberTurnMessagePieceEntry.message_piece_id == user_piece.id,
                )
            ).scalar_one()
            assert row is not None
            role, value, value_sha256, metadata, link_sha256 = original
            link_identity = (link.run_id, link.turn_index, link.direction, link.position, link.message_piece_id)
            session.delete(link)
            session.flush()
            row.role = "system" if enabled and tamper == "role" else role
            row.original_value = "forged source text" if enabled and tamper == "text" else value
            row.original_value_sha256 = to_sha256(row.original_value) if enabled and tamper == "text" else value_sha256
            row.prompt_metadata = (
                {**metadata, "inspect_message_source": "forged"}
                if enabled and tamper == "source_metadata"
                else dict(metadata)
            )
            session.flush()
            session.add(
                NativeCyberTurnMessagePieceEntry(
                    run_id=link_identity[0],
                    turn_index=link_identity[1],
                    direction=link_identity[2],
                    position=link_identity[3],
                    message_piece_id=link_identity[4],
                    piece_sha256=(
                        _message_piece_digest(row.get_message_piece(), include_id=True) if enabled else link_sha256
                    ),
                )
            )

    service = ScenarioRunService()
    attack_service = AttackService()
    try:
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service),
            patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                endpoints = (
                    f"/api/scenarios/runs/{result.id}",
                    f"/api/scenarios/runs/{result.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                    f"/api/attacks/{imported.attack_result_id}",
                    f"/api/attacks/{imported.attack_result_id}/messages?conversation_id={attack.conversation_id}",
                )
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200

                await asyncio.to_thread(set_tamper, enabled=True)
                snapshot = sqlite_instance.native_cyber_evidence.get_finalized_unscored_inspect_capture(
                    run_id=imported.projection_episode_id
                )
                assert snapshot.coverage_complete
                statuses = {endpoint: await client.get(endpoint) for endpoint in endpoints}
                assert all(response.status_code >= 400 for response in statuses.values()), {
                    endpoint: response.status_code for endpoint, response in statuses.items()
                }
                for response in statuses.values():
                    assert "forged source text" not in response.text
                    assert "harmless fixture" not in response.text

                await asyncio.to_thread(set_tamper, enabled=False)
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "tamper",
    ["sample_uuid", "attack_objective", "turn_count_changed", "turn_count_presence", "timestamp"],
)
async def test_one_click_readback_binds_typed_sample_and_imported_result_fields(
    sqlite_instance: SQLiteMemory, tamper: str
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
    with sqlite_instance.get_session() as session:
        score_row = session.get(ScoreEntry, imported.score_id)
        attack_row = session.get(AttackResultEntry, imported.attack_result_id)
        assert score_row is not None and score_row.score_metadata is not None
        assert attack_row is not None and attack_row.attack_metadata is not None
        original_score = dict(score_row.score_metadata)
        original_attack = dict(attack_row.attack_metadata)
        original_objective = attack_row.objective
        original_score_timestamp = score_row.timestamp
        original_attack_timestamp = attack_row.timestamp
    forged_uuid = str(uuid.uuid4())
    assert original_score["inspect_sample_uuid"] != forged_uuid

    def set_tamper(*, enabled: bool) -> None:
        with sqlite_instance.get_session() as session, session.begin():
            score_row = session.get(ScoreEntry, imported.score_id)
            attack_row = session.get(AttackResultEntry, imported.attack_result_id)
            assert score_row is not None and attack_row is not None
            changed = {"inspect_sample_uuid": forged_uuid} if enabled and tamper == "sample_uuid" else {}
            score_row.score_metadata = {**original_score, **changed}
            attack_metadata = {**original_attack, **changed}
            if enabled and tamper == "turn_count_changed":
                attack_metadata["inspect_turn_count"] = int(original_attack.get("inspect_turn_count") or 0) + 7
            elif enabled and tamper == "turn_count_presence":
                if "inspect_turn_count" in attack_metadata:
                    attack_metadata.pop("inspect_turn_count")
                else:
                    attack_metadata["inspect_turn_count"] = 7
            attack_row.attack_metadata = attack_metadata
            attack_row.objective = (
                "forged original Inspect objective" if enabled and tamper == "attack_objective" else original_objective
            )
            delta = timedelta(days=1) if enabled and tamper == "timestamp" else timedelta()
            score_row.timestamp = original_score_timestamp + delta
            attack_row.timestamp = original_attack_timestamp + delta

    service = ScenarioRunService()
    attack_service = AttackService()
    try:
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service),
            patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                endpoints = (
                    f"/api/scenarios/runs/{result.id}",
                    f"/api/scenarios/runs/{result.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                    f"/api/attacks/{imported.attack_result_id}",
                    f"/api/attacks/{imported.attack_result_id}/messages?conversation_id={attack.conversation_id}",
                )
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200

                await asyncio.to_thread(set_tamper, enabled=True)
                statuses = {endpoint: await client.get(endpoint) for endpoint in endpoints}
                assert all(response.status_code >= 400 for response in statuses.values()), {
                    endpoint: response.status_code for endpoint, response in statuses.items()
                }
                for response in statuses.values():
                    assert "forged original Inspect objective" not in response.text
                    assert "harmless fixture" not in response.text

                await asyncio.to_thread(set_tamper, enabled=False)
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("tamper", ["case_id", "source_pin", "objective_text", "objective_hash"])
async def test_one_click_readback_rejects_tampered_plan_case_source_or_objective(
    sqlite_instance: SQLiteMemory, tamper: str
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    plan = result.metadata["run_plan"]
    reference = dict(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    assert len(plan["seed_groups"]) == 1
    assert plan["seed_groups"][0]["case_id"] != "f" * 64
    source = EvalSourceFactory.resolve_original_inert(family=OriginalInspectTaskId.INERT.value)
    foreign_package = source.case.package.model_copy(update={"source_sha256": "f" * 64})
    foreign_case = source.case.model_copy(update={"package": foreign_package})
    changed_seed = dict(plan["seed_groups"][0])
    if tamper == "case_id":
        changed_seed["case_id"] = "f" * 64
    elif tamper == "source_pin":
        changed_seed.update(case_id=foreign_case.case_id, source_sha256="f" * 64)
    elif tamper == "objective_text":
        changed_seed["objective"] = "a different harmless fixture"
    else:
        changed_seed["objective_sha256"] = "f" * 64
    changed_plan = {**plan, "seed_groups": [changed_seed]}
    changed_reference = {**reference, "source_sha256": "f" * 64} if tamper == "source_pin" else reference
    service = ScenarioRunService()
    try:
        with patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                endpoints = (
                    f"/api/scenarios/runs/{result.id}",
                    f"/api/scenarios/runs/{result.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                )
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200
                await asyncio.to_thread(
                    sqlite_instance.update_scenario_metadata_fields,
                    scenario_result_id=str(result.id),
                    fields={
                        "run_plan": changed_plan,
                        OriginalInspectImportSummary.METADATA_KEY: changed_reference,
                    },
                )
                for endpoint in endpoints:
                    response = await client.get(endpoint)
                    assert response.status_code >= 400, f"Foreign {tamper} was accepted by {endpoint}"
                    assert "harmless fixture" not in response.text
                await asyncio.to_thread(
                    sqlite_instance.update_scenario_metadata_fields,
                    scenario_result_id=str(result.id),
                    fields={"run_plan": plan, OriginalInspectImportSummary.METADATA_KEY: reference},
                )
                for endpoint in endpoints:
                    assert (await client.get(endpoint)).status_code == 200
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "corruption",
    ["conversation", "last_response", "related_pruned", "related_preparation", "related_adversarial"],
)
async def test_one_click_attack_conversation_links_reject_unrelated_real_run(
    sqlite_instance: SQLiteMemory, corruption: str
) -> None:
    registry = ScenarioRegistry()
    runs = []
    for _ in range(2):
        scenario = await registry.create_and_initialize_async("benchmark.inspect_original_inert")
        run = await scenario.run_async()
        imported = OriginalInspectImportSummary.model_validate(run.metadata[OriginalInspectImportSummary.METADATA_KEY])
        [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
        runs.append((run, imported, attack))
    first_run, first_import, first_attack = runs[0]
    _, _, other_attack = runs[1]
    assert first_attack.conversation_id != other_attack.conversation_id
    other_response = next(
        piece
        for piece in sqlite_instance.get_message_pieces(conversation_id=other_attack.conversation_id)
        if piece.role == "assistant"
    )

    def set_corruption(*, enabled: bool) -> None:
        with sqlite_instance.get_session() as session, session.begin():
            row = session.get(AttackResultEntry, first_import.attack_result_id)
            assert row is not None
            row.conversation_id = (
                other_attack.conversation_id
                if enabled and corruption == "conversation"
                else first_attack.conversation_id
            )
            row.last_response_id = (
                uuid.UUID(str(other_response.id)) if enabled and corruption == "last_response" else None
            )
            row.pruned_conversation_ids = (
                [other_attack.conversation_id] if enabled and corruption == "related_pruned" else None
            )
            row.preparation_conversation_ids = (
                [other_attack.conversation_id] if enabled and corruption == "related_preparation" else None
            )
            row.adversarial_chat_conversation_ids = (
                [other_attack.conversation_id] if enabled and corruption == "related_adversarial" else None
            )

    service = ScenarioRunService()
    attack_service = AttackService()
    try:
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service),
            patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                scenario_endpoints = (
                    f"/api/scenarios/runs/{first_run.id}",
                    f"/api/scenarios/runs/{first_run.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                )
                attack_id = str(first_import.attack_result_id)
                attack_endpoints = (
                    f"/api/attacks/{attack_id}",
                    f"/api/attacks/{attack_id}/conversations",
                )
                list_endpoint = "/api/attacks?limit=20"
                for endpoint in (*scenario_endpoints, *attack_endpoints):
                    assert (await client.get(endpoint)).status_code == 200
                baseline_list = await client.get(list_endpoint)
                assert baseline_list.status_code == 200
                assert any(item["attack_result_id"] == attack_id for item in baseline_list.json()["items"])
                original_messages = f"/api/attacks/{attack_id}/messages?conversation_id={first_attack.conversation_id}"
                assert (await client.get(original_messages)).status_code == 200

                await asyncio.to_thread(set_corruption, enabled=True)
                for endpoint in (*attack_endpoints, *scenario_endpoints):
                    response = await client.get(endpoint)
                    assert response.status_code >= 400, f"{corruption} was accepted by {endpoint}"
                    assert other_attack.conversation_id not in response.text
                attack_list = await client.get(list_endpoint)
                if attack_list.status_code == 200:
                    assert all(item["attack_result_id"] != attack_id for item in attack_list.json()["items"])
                else:
                    assert attack_list.status_code >= 400
                    assert other_attack.conversation_id not in attack_list.text
                active_conversation_id = (
                    other_attack.conversation_id if corruption == "conversation" else first_attack.conversation_id
                )
                messages = await client.get(
                    f"/api/attacks/{attack_id}/messages?conversation_id={active_conversation_id}"
                )
                assert messages.status_code >= 400
                assert "inert response" not in messages.text

                await asyncio.to_thread(set_corruption, enabled=False)
                for endpoint in (*scenario_endpoints, *attack_endpoints):
                    assert (await client.get(endpoint)).status_code == 200
                assert (await client.get(list_endpoint)).status_code == 200
                assert (await client.get(original_messages)).status_code == 200
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("tamper", ["colluded_metadata", "source_markers_removed"])
async def test_one_click_direct_attack_readback_rejects_independent_source_spoof(
    sqlite_instance: SQLiteMemory, tamper: str
) -> None:
    runs = []
    for _ in range(2):
        scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
        run = await scenario.run_async()
        imported = OriginalInspectImportSummary.model_validate(run.metadata[OriginalInspectImportSummary.METADATA_KEY])
        [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
        [score] = sqlite_instance.get_scores(score_ids=[str(imported.score_id)])
        assert score.score_metadata is not None
        runs.append((run, imported, attack, score.score_metadata))
    first_run, first_import, first_attack, _ = runs[0]
    _, _, other_attack, other_score_metadata = runs[1]
    assert first_attack.conversation_id != other_attack.conversation_id

    with sqlite_instance.get_session() as session:
        score_row = session.get(ScoreEntry, first_import.score_id)
        attack_row = session.get(AttackResultEntry, first_import.attack_result_id)
        assert score_row is not None and score_row.score_metadata is not None
        assert attack_row is not None and attack_row.attack_metadata is not None
        original_score_metadata = dict(score_row.score_metadata)
        original_attack_metadata = dict(attack_row.attack_metadata)

    def set_tamper(*, enabled: bool) -> None:
        with sqlite_instance.get_session() as session, session.begin():
            score_row = session.get(ScoreEntry, first_import.score_id)
            attack_row = session.get(AttackResultEntry, first_import.attack_result_id)
            assert score_row is not None and attack_row is not None
            score_metadata = dict(original_score_metadata)
            attack_metadata = dict(original_attack_metadata)
            if enabled and tamper == "colluded_metadata":
                for key in ("inspect_archive_sha256", "inspect_sample_uuid"):
                    score_metadata[key] = other_score_metadata[key]
                    attack_metadata[key] = other_score_metadata[key]
                attack_row.conversation_id = other_attack.conversation_id
            elif enabled:
                score_metadata.pop("inspect_source")
                attack_metadata.pop("inspect_source")
            score_row.score_metadata = score_metadata
            attack_row.attack_metadata = attack_metadata
            if not enabled:
                attack_row.conversation_id = first_attack.conversation_id

    scenario_service = ScenarioRunService()
    attack_service = AttackService()
    try:
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=scenario_service),
            patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                attack_id = str(first_import.attack_result_id)
                detail = f"/api/attacks/{attack_id}"
                conversation = (
                    other_attack.conversation_id if tamper == "colluded_metadata" else first_attack.conversation_id
                )
                conversation_path = f"/api/attacks/{attack_id}/messages?conversation_id={conversation}"
                assert (await client.get(detail)).status_code == 200
                assert (
                    await client.get(
                        f"/api/attacks/{attack_id}/messages?conversation_id={first_attack.conversation_id}"
                    )
                ).status_code == 200

                await asyncio.to_thread(set_tamper, enabled=True)
                for endpoint in (detail, f"/api/attacks/{attack_id}/conversations", conversation_path):
                    response = await client.get(endpoint)
                    assert response.status_code >= 400, f"{tamper} was accepted by {endpoint}"
                    assert other_attack.conversation_id not in response.text
                    assert "harmless fixture" not in response.text
                    assert "inert response" not in response.text
                history = await client.get("/api/attacks?limit=20")
                if history.status_code == 200:
                    assert all(item["attack_result_id"] != attack_id for item in history.json()["items"])
                else:
                    assert history.status_code >= 400
                    assert "inert response" not in history.text
                for endpoint in (
                    f"/api/scenarios/runs/{first_run.id}",
                    f"/api/scenarios/runs/{first_run.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                ):
                    assert (await client.get(endpoint)).status_code >= 400

                await asyncio.to_thread(set_tamper, enabled=False)
                assert (await client.get(detail)).status_code == 200
                assert (await client.get(f"/api/scenarios/runs/{first_run.id}")).status_code == 200
    finally:
        await scenario_service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
async def test_one_click_direct_read_rejects_extra_message_with_copied_metadata(sqlite_instance: SQLiteMemory) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
    original_pieces = sqlite_instance.get_message_pieces(conversation_id=attack.conversation_id)
    assert original_pieces and original_pieces[0].prompt_metadata is not None
    sqlite_instance.add_message_pieces_to_memory(
        message_pieces=[
            MessagePiece(
                role="assistant",
                original_value="forged unrelated transcript",
                original_value_data_type="text",
                conversation_id=attack.conversation_id,
                sequence=max(piece.sequence for piece in original_pieces) + 1,
                prompt_metadata=dict(original_pieces[0].prompt_metadata),
            )
        ]
    )

    scenario_service = ScenarioRunService()
    attack_service = AttackService()
    try:
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=scenario_service),
            patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
            ) as client:
                for endpoint in (
                    f"/api/attacks/{imported.attack_result_id}",
                    f"/api/attacks/{imported.attack_result_id}/messages?conversation_id={attack.conversation_id}",
                    f"/api/scenarios/runs/{result.id}",
                    f"/api/scenarios/runs/{result.id}/progress",
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert",
                ):
                    response = await client.get(endpoint)
                    assert response.status_code >= 400, f"Forged message was accepted by {endpoint}"
                    assert "forged unrelated transcript" not in response.text
    finally:
        await scenario_service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
async def test_one_click_direct_attack_read_rejects_ambiguous_persisted_import_links(
    sqlite_instance: SQLiteMemory,
) -> None:
    registry = ScenarioRegistry()
    runs = []
    for _ in range(2):
        scenario = await registry.create_and_initialize_async("benchmark.inspect_original_inert")
        result = await scenario.run_async()
        reference = dict(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
        runs.append((result, reference))
    first, second = runs
    attack_id = first[1]["attack_result_id"]
    await asyncio.to_thread(
        sqlite_instance.update_scenario_metadata_fields,
        scenario_result_id=str(second[0].id),
        fields={OriginalInspectImportSummary.METADATA_KEY: {**second[1], "attack_result_id": attack_id}},
    )
    attack_service = AttackService()
    with patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            response = await client.get(f"/api/attacks/{attack_id}")
    assert response.status_code >= 400
    assert "harmless fixture" not in response.text


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("operation", ["patch", "delete"])
@pytest.mark.parametrize("remove_markers", [False, True])
async def test_one_click_direct_attack_writes_preserve_original_source_outcome(
    sqlite_instance: SQLiteMemory, operation: str, remove_markers: bool
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    attack_id = str(imported.attack_result_id)
    attack_service = AttackService()
    with patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            assert (await client.get(f"/api/attacks/{attack_id}")).status_code == 200
            with sqlite_instance.get_session() as session:
                attack_row = session.get(AttackResultEntry, imported.attack_result_id)
                assert attack_row is not None
                original_timestamp = attack_row.timestamp
            if remove_markers:
                await asyncio.to_thread(
                    _remove_original_import_markers, sqlite_instance=sqlite_instance, imported=imported
                )

            response = (
                await client.patch(f"/api/attacks/{attack_id}", json={"outcome": "success"})
                if operation == "patch"
                else await client.delete(f"/api/attacks/{attack_id}/human-score")
            )
            assert response.status_code >= 400, f"{operation} changed a source-attributed original result"
            assert "harmless fixture" not in response.text
            with sqlite_instance.get_session() as session:
                attack_row = session.get(AttackResultEntry, imported.attack_result_id)
                assert attack_row is not None
                assert attack_row.outcome == AttackOutcome.UNDETERMINED.value
                assert attack_row.timestamp == original_timestamp
                assert attack_row.human_score_id is None


def _remove_original_import_markers(*, sqlite_instance: SQLiteMemory, imported: OriginalInspectImportSummary) -> None:
    with sqlite_instance.get_session() as session, session.begin():
        attack = session.get(AttackResultEntry, imported.attack_result_id)
        score = session.get(ScoreEntry, imported.score_id)
        assert attack is not None and attack.attack_metadata is not None
        assert score is not None and score.score_metadata is not None
        attack.attack_metadata = {
            key: value for key, value in attack.attack_metadata.items() if key != "inspect_source"
        }
        score.score_metadata = {key: value for key, value in score.score_metadata.items() if key != "inspect_source"}


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("remove_markers", [False, True])
async def test_one_click_manual_score_rejected_before_scorer_or_attack_update(
    sqlite_instance: SQLiteMemory, remove_markers: bool
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
    piece = next(
        piece
        for piece in sqlite_instance.get_message_pieces(conversation_id=attack.conversation_id)
        if piece.role == "assistant"
    )

    def count_scores() -> int:
        with sqlite_instance.get_session() as session:
            return session.execute(select(func.count()).select_from(ScoreEntry)).scalar_one()

    score_count = await asyncio.to_thread(count_scores)
    with sqlite_instance.get_session() as session:
        row = session.get(AttackResultEntry, imported.attack_result_id)
        assert row is not None
        original_timestamp = row.timestamp
    if remove_markers:
        await asyncio.to_thread(_remove_original_import_markers, sqlite_instance=sqlite_instance, imported=imported)

    async with AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/scores/manual",
            json={
                "attack_result_id": str(imported.attack_result_id),
                "message_id": str(piece.id),
                "value": True,
                "rationale": "unapproved judgment",
                "update_attack": True,
            },
        )
    assert response.status_code >= 400
    assert "harmless fixture" not in response.text
    assert await asyncio.to_thread(count_scores) == score_count
    with sqlite_instance.get_session() as session:
        row = session.get(AttackResultEntry, imported.attack_result_id)
        assert row is not None
        assert row.outcome == AttackOutcome.UNDETERMINED.value
        assert row.timestamp == original_timestamp
        assert row.human_score_id is None


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("send", [False, True])
@pytest.mark.parametrize("remove_markers", [False, True])
async def test_one_click_add_message_rejects_before_storage_or_target_dispatch(
    sqlite_instance: SQLiteMemory, send: bool, remove_markers: bool
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    [attack] = sqlite_instance.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
    pieces = sqlite_instance.get_message_pieces(conversation_id=attack.conversation_id)
    count = len(pieces)
    assert count > 0 and pieces[0].prompt_metadata is not None
    with sqlite_instance.get_session() as session:
        row = session.get(AttackResultEntry, imported.attack_result_id)
        assert row is not None
        original_timestamp = row.timestamp
    if remove_markers:
        await asyncio.to_thread(_remove_original_import_markers, sqlite_instance=sqlite_instance, imported=imported)
    attack_service = AttackService()
    with (
        patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service),
        patch.object(attack_service, "_send_and_store_message_async", new_callable=AsyncMock) as dispatch,
    ):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            response = await client.post(
                f"/api/attacks/{imported.attack_result_id}/messages",
                json={
                    "role": "user",
                    "pieces": [
                        {
                            "original_value": "forged unrelated transcript",
                            "prompt_metadata": pieces[0].prompt_metadata,
                        }
                    ],
                    "send": send,
                    "target_registry_name": "offline-test-target" if send else None,
                    "target_conversation_id": attack.conversation_id,
                },
            )
        assert response.status_code >= 400
        assert "forged unrelated transcript" not in response.text
        dispatch.assert_not_awaited()
    assert len(sqlite_instance.get_message_pieces(conversation_id=attack.conversation_id)) == count
    with sqlite_instance.get_session() as session:
        row = session.get(AttackResultEntry, imported.attack_result_id)
        assert row is not None and row.timestamp == original_timestamp


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("operation", ["create", "promote"])
@pytest.mark.parametrize("remove_markers", [False, True])
async def test_one_click_conversation_mutations_are_rejected_before_writes(
    sqlite_instance: SQLiteMemory, operation: str, remove_markers: bool
) -> None:
    scenario = await ScenarioRegistry().create_and_initialize_async("benchmark.inspect_original_inert")
    result = await scenario.run_async()
    imported = OriginalInspectImportSummary.model_validate(result.metadata[OriginalInspectImportSummary.METADATA_KEY])
    branch_id = str(uuid.uuid4())
    if operation == "promote":
        with sqlite_instance.get_session() as session, session.begin():
            row = session.get(AttackResultEntry, imported.attack_result_id)
            assert row is not None
            row.pruned_conversation_ids = [branch_id]
    with sqlite_instance.get_session() as session:
        row = session.get(AttackResultEntry, imported.attack_result_id)
        assert row is not None
        original_conversation = row.conversation_id
        original_related = list(row.pruned_conversation_ids or [])
        original_timestamp = row.timestamp
    if remove_markers:
        await asyncio.to_thread(_remove_original_import_markers, sqlite_instance=sqlite_instance, imported=imported)
    attack_service = AttackService()
    with patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            response = (
                await client.post(f"/api/attacks/{imported.attack_result_id}/conversations", json={})
                if operation == "create"
                else await client.post(
                    f"/api/attacks/{imported.attack_result_id}/update-main-conversation",
                    json={"conversation_id": branch_id},
                )
            )
    assert response.status_code >= 400
    assert branch_id not in response.text
    with sqlite_instance.get_session() as session:
        row = session.get(AttackResultEntry, imported.attack_result_id)
        assert row is not None
        assert row.conversation_id == original_conversation
        assert (row.pruned_conversation_ids or []) == original_related
        assert row.timestamp == original_timestamp


@pytest.mark.usefixtures("patch_central_database")
async def test_ordinary_attack_http_patch_and_human_score_removal_still_work(sqlite_instance: SQLiteMemory) -> None:
    now = datetime.now(UTC)
    attack = AttackResult(
        attack_result_id=str(uuid.uuid4()),
        conversation_id=str(uuid.uuid4()),
        objective="ordinary fixture",
        atomic_attack_identifier=AtomicAttackIdentifier.build(
            attack_identifier=AttackIdentifier(class_name="ManualAttack", class_module="pyrit.backend")
        ),
        outcome=AttackOutcome.UNDETERMINED,
        timestamp=now,
    )
    sqlite_instance.add_attack_results_to_memory(attack_results=[attack])
    attack_service = AttackService()
    with patch("pyrit.backend.routes.attacks.get_attack_service", return_value=attack_service):
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
        ) as client:
            patched = await client.patch(f"/api/attacks/{attack.attack_result_id}", json={"outcome": "success"})
            assert patched.status_code == 200
            assert patched.json()["outcome"] == AttackOutcome.SUCCESS.value
            created = await client.post(f"/api/attacks/{attack.attack_result_id}/conversations", json={})
            assert created.status_code == 201
            branch_id = created.json()["conversation_id"]
            promoted = await client.post(
                f"/api/attacks/{attack.attack_result_id}/update-main-conversation",
                json={"conversation_id": branch_id},
            )
            assert promoted.status_code == 200
            assert promoted.json()["conversation_id"] == branch_id
            message = await client.post(
                f"/api/attacks/{attack.attack_result_id}/messages",
                json={
                    "role": "assistant",
                    "pieces": [{"original_value": "ordinary response"}],
                    "send": False,
                    "target_conversation_id": branch_id,
                },
            )
            assert message.status_code == 200
            [piece] = sqlite_instance.get_message_pieces(conversation_id=branch_id)
            human = await client.post(
                "/api/scores/manual",
                json={
                    "attack_result_id": attack.attack_result_id,
                    "message_id": str(piece.id),
                    "value": True,
                    "update_attack": True,
                },
            )
            assert human.status_code == 201
            assert human.json()["score_value"] == "True"
            removed = await client.delete(f"/api/attacks/{attack.attack_result_id}/human-score")
            assert removed.status_code == 200
            assert removed.json()["outcome"] == AttackOutcome.UNDETERMINED.value


@pytest.mark.usefixtures("patch_central_database")
async def test_one_click_backend_returns_original_score_with_undetermined_outcome(
    sqlite_instance: SQLiteMemory,
) -> None:
    service = ScenarioRunService()
    try:
        scenario = await service._prepare_run_async(
            request=RunScenarioRequest(
                scenario_name="benchmark.inspect_original_inert",
                scenario_params={"eval_family": "inspect_original_inert"},
            )
        )
        assert isinstance(scenario, InspectOriginalInertScenario)
        result = await scenario.run_async()
        summary = service.get_run(scenario_result_id=str(result.id))
        progress = service.get_run_progress(scenario_result_id=str(result.id), since=None, limit=10)
        assert summary is not None and progress is not None
        assert summary.status == ScenarioRunState.COMPLETED
        assert summary.target is None
        assert (summary.completed_attacks, summary.total_attacks, summary.successful_attacks) == (1, 1, 0)
        assert summary.objective_achieved_rate is None
        assert summary.original_inspect_import is not None
        assert progress.run.original_inspect_import == summary.original_inspect_import
        assert progress.results == []
        assert (progress.summary.overall.completed, progress.summary.overall.planned) == (1, 1)
        assert progress.summary.overall.succeeded == 0
        assert progress.summary.overall.success_percentage is None
        assert len(progress.summary.atomic_groups) == 1
        assert progress.summary.atomic_groups[0].status == "COMPLETED"
        assert progress.summary.atomic_groups[0].completed == 1
        assert progress.summary.display_groups[0].completed == 1
        assert progress.summary.techniques[0].completed == 1
        assert progress.summary.seed_groups[0].id == summary.original_inspect_import.case_run_id
        assert progress.summary.seed_groups[0].completed == 1
        assert summary.successful_attacks == 0
        assert summary.original_inspect_import.score_status is ScoreStatus.COMPLETE
        assert summary.original_inspect_import.outcome is AttackOutcome.UNDETERMINED
        assert summary.original_inspect_import.score_value == "1.0"
        assert sqlite_instance.get_scores(score_ids=[str(summary.original_inspect_import.score_id)])
        assert sqlite_instance.get_attack_results(
            attack_result_ids=[str(summary.original_inspect_import.attack_result_id)]
        )
        mismatch_pattern = "Original Inspect import (does not match|references missing or mismatched)"
        for mismatch in (
            {"case_run_id": "f" * 64},
            {"source_sha256": "f" * 64},
            {"episode_id": f"inspect-run-{'f' * 32}"},
        ):
            sqlite_instance.update_scenario_metadata_fields(
                scenario_result_id=str(result.id),
                fields={
                    OriginalInspectImportSummary.METADATA_KEY: {
                        **summary.original_inspect_import.model_dump(mode="json"),
                        **mismatch,
                    }
                },
            )
            with pytest.raises(ValueError, match=mismatch_pattern):
                service.get_run(scenario_result_id=str(result.id))
            with pytest.raises(ValueError, match=mismatch_pattern):
                service.get_run_progress(scenario_result_id=str(result.id), since=None, limit=10)
            with pytest.raises(ValueError, match=mismatch_pattern):
                service.list_runs(scenario_names=["benchmark.inspect_original_inert"])
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
async def test_http_catalog_and_one_click_run_without_target_or_credentials(sqlite_instance: SQLiteMemory) -> None:
    service = ScenarioRunService()
    catalog = ScenarioService()
    try:
        with (
            patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service),
            patch("pyrit.backend.routes.scenarios.get_scenario_service", return_value=catalog),
            patch.dict("os.environ", {"OPENAI_CHAT_MODEL": ""}),
        ):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                details = await client.get("/api/scenarios/catalog/benchmark.inspect_original_inert")
                assert details.status_code == 200
                assert details.json()["scenario_name"] == "benchmark.inspect_original_inert"
                missing_target = await client.post(
                    "/api/scenarios/runs",
                    json={"scenario_name": "foundry.red_team_agent"},
                )
                assert missing_target.status_code == 400
                assert "target_name is required" in missing_target.json()["detail"]
                rejected = await client.post(
                    "/api/scenarios/runs",
                    json={
                        "scenario_name": "benchmark.inspect_original_inert",
                        "task_url": "https://example.invalid/private",
                    },
                )
                assert rejected.status_code == 400
                assert "task_url" in rejected.json()["detail"]

                started = await client.post(
                    "/api/scenarios/runs",
                    json={
                        "scenario_name": "benchmark.inspect_original_inert",
                        "scenario_params": {"eval_family": "inspect_original_inert"},
                    },
                )
                assert started.status_code == 202
                run_id = started.json()["scenario_result_id"]
                active = service._active_tasks[run_id]
                assert active.task is not None
                await asyncio.wait_for(active.task, timeout=45)

                response = await client.get(f"/api/scenarios/runs/{run_id}")
                assert response.status_code == 200
                assert response.json()["status"] == "COMPLETED"
                assert response.json()["total_attacks"] == 1
                assert response.json()["completed_attacks"] == 1
                assert response.json()["successful_attacks"] == 0
                reference = response.json()["original_inspect_import"]
                assert reference["score_status"] == "complete"
                assert reference["score_value"] == "1.0"
                assert reference["outcome"] == "undetermined"
                assert sqlite_instance.get_scores(score_ids=[reference["score_id"]])
                assert sqlite_instance.get_attack_results(attack_result_ids=[reference["attack_result_id"]])

                progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
                assert progress.status_code == 200
                assert progress.json()["run"]["original_inspect_import"] == reference
                assert progress.json()["summary"]["overall"]["completed"] == 1
                assert progress.json()["summary"]["overall"]["succeeded"] == 0
                assert progress.json()["summary"]["overall"]["success_percentage"] is None
                assert progress.json()["summary"]["atomic_groups"][0]["status"] == "COMPLETED"
                assert progress.json()["results"] == []
                history = await client.get("/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert")
                assert history.status_code == 200
                assert len(history.json()["items"]) == 1
                [item] = history.json()["items"]
                assert item["scenario_result_id"] == run_id
                assert (item["completed_attacks"], item["total_attacks"], item["successful_attacks"]) == (1, 1, 0)
                assert item["objective_achieved_rate"] is None
                assert response.json()["objective_achieved_rate"] is None
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
async def test_repeated_one_click_http_runs_then_ordinary_offline_scenario_leave_no_tasks(
    sqlite_instance: SQLiteMemory,
) -> None:
    from pyrit.models import SeedPrompt
    from pyrit.scenario.scenarios.garak.encoding import Encoding, EncodingDatasetConfiguration, EncodingTechnique
    from pyrit.score import SubStringScorer
    from tests.unit.mocks import MockPromptTarget

    def worker_pending_tasks() -> tuple[str, ...]:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            return ()
        if loop.is_closed():
            return ()
        return tuple(sorted(task.get_name() for task in asyncio.all_tasks(loop) if not task.done()))

    target_instances: list[MockPromptTarget] = []

    async def prepare_ordinary_async(*, request: RunScenarioRequest) -> Encoding:
        _ = request
        target = MockPromptTarget()
        target_instances.append(target)
        ordinary = Encoding(objective_scorer=SubStringScorer(substring="default"), encoding_templates=[])
        ordinary.set_params_from_args(
            args={
                "objective_target": target,
                "scenario_techniques": [EncodingTechnique.ROT13],
                "dataset_config": EncodingDatasetConfiguration(seeds=[SeedPrompt(value="harmless fixture")]),
                "include_baseline": False,
                "max_retries": 0,
            }
        )
        await ordinary.initialize_async()
        return ordinary

    current = asyncio.current_task()
    existing_tasks = {task for task in asyncio.all_tasks() if task is not current and not task.done()}
    service = ScenarioRunService()
    try:
        with patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                run_ids: list[str] = []
                for _ in range(2):
                    started = await client.post(
                        "/api/scenarios/runs",
                        json={
                            "scenario_name": "benchmark.inspect_original_inert",
                            "scenario_params": {"eval_family": "inspect_original_inert"},
                        },
                    )
                    assert started.status_code == 202
                    run_id = started.json()["scenario_result_id"]
                    run_ids.append(run_id)
                    active = service._active_tasks[run_id]
                    assert active.task is not None
                    await asyncio.wait_for(active.task, timeout=45)
                    detail = await client.get(f"/api/scenarios/runs/{run_id}")
                    progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
                    assert detail.status_code == 200 and progress.status_code == 200
                    assert detail.json()["status"] == "COMPLETED"
                    assert (detail.json()["completed_attacks"], detail.json()["successful_attacks"]) == (1, 0)
                    assert progress.json()["summary"]["overall"]["completed"] == 1
                    assert progress.json()["run"]["original_inspect_import"]["outcome"] == "undetermined"
                    assert not await asyncio.wrap_future(service._prepare_executor.submit(worker_pending_tasks))
                assert len(set(run_ids)) == 2

        with patch.object(service, "_prepare_run_async", new=prepare_ordinary_async):
            ordinary = await asyncio.wrap_future(
                service._prepare_executor.submit(
                    service._prepare_run_blocking,
                    request=RunScenarioRequest(scenario_name="garak.encoding", target_name="offline-mock"),
                )
            )
        ordinary_result = await ordinary.run_async()
        assert ordinary_result.scenario_run_state is ScenarioRunState.COMPLETED
        assert ordinary_result.attack_results
        assert len(target_instances) == 1 and target_instances[0].prompt_sent
        assert not await asyncio.wrap_future(service._prepare_executor.submit(worker_pending_tasks))
    finally:
        await service.shutdown_async()
    await asyncio.sleep(0)
    assert not [
        task for task in asyncio.all_tasks() if task is not current and task not in existing_tasks and not task.done()
    ]


@pytest.mark.usefixtures("patch_central_database")
async def test_offline_projection_failure_keeps_evidence_but_never_completes_scenario(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    registry = ScenarioRegistry()
    scenario = await registry.create_and_initialize_async("benchmark.inspect_original_inert")
    log_dir = tmp_path / "failed-projection"
    log_dir.mkdir()
    with (
        patch("pyrit.scenario.scenarios.benchmark.inspect_original_inert._allocate_log_dir", return_value=log_dir),
        patch.object(
            InspectOriginalEvalImporter,
            "import_eval_log_async",
            new_callable=AsyncMock,
            side_effect=ValueError(
                "Original Inspect archive is not a readable `.eval` ZIP. "
                "api_key=not-for-ui https://example.invalid/private"
            ),
        ) as projection,
        pytest.raises(ValueError, match="Original Inspect archive is not a readable"),
    ):
        await scenario.run_async()
    projection.assert_awaited_once()
    [stored] = sqlite_instance.get_scenario_results(scenario_result_ids=[scenario._scenario_result_id])
    assert stored.scenario_run_state is ScenarioRunState.FAILED
    assert OriginalInspectImportSummary.METADATA_KEY not in stored.metadata
    assert stored.error_message is not None
    assert "api_key=not-for-ui" in stored.error_message
    assert list(log_dir.glob("*.eval"))
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0
    service = ScenarioRunService()
    try:
        with patch("pyrit.backend.routes.scenarios.get_scenario_run_service", return_value=service):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                run_id = str(stored.id)
                summary = await client.get(f"/api/scenarios/runs/{run_id}")
                progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
                assert summary.status_code == progress.status_code == 200
                expected = "Original Inspect archive is not a readable `.eval` ZIP."
                assert summary.json()["error"] == expected
                assert progress.json()["run"]["failure_reason"] == expected
                assert progress.json()["summary"]["overall"]["completed"] == 0
                assert progress.json()["summary"]["atomic_groups"][0]["status"] == "INCOMPLETE"
                assert "api_key=not-for-ui" not in progress.text
                assert "example.invalid" not in summary.text
                history = await client.get("/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert")
                assert history.status_code == 200
                [item] = history.json()["items"]
                assert item["scenario_result_id"] == run_id
                assert item["error"] == expected
                assert item["completed_attacks"] == 0
                assert item["objective_achieved_rate"] is None
                assert summary.json()["objective_achieved_rate"] is None
                assert "api_key=not-for-ui" not in history.text
                assert "example.invalid" not in history.text
                await asyncio.to_thread(
                    sqlite_instance.update_scenario_run_state,
                    scenario_result_id=run_id,
                    scenario_run_state=ScenarioRunState.FAILED,
                    error_message="Parser failed at C:\\private\\credentials.txt api_key=not-for-ui",
                    error_type="ValueError",
                )
                fallback_summary = await client.get(f"/api/scenarios/runs/{run_id}")
                fallback_progress = await client.get(f"/api/scenarios/runs/{run_id}/progress")
                assert fallback_summary.status_code == fallback_progress.status_code == 200
                assert fallback_summary.json()["error"] == (
                    "Original Inspect Task or offline projection failed; reconcile its retained log before retrying."
                )
                assert fallback_progress.json()["run"]["failure_reason"] == fallback_summary.json()["error"]
                assert "credentials.txt" not in fallback_progress.text
                assert "api_key=not-for-ui" not in fallback_summary.text
                fallback_history = await client.get(
                    "/api/scenarios/runs?scenario_names=benchmark.inspect_original_inert"
                )
                assert fallback_history.status_code == 200
                [item] = fallback_history.json()["items"]
                assert item["error"] == fallback_summary.json()["error"]
                assert "credentials.txt" not in fallback_history.text
                assert "api_key=not-for-ui" not in fallback_history.text
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("input_fields", "expected"),
    [
        ({"scenario_params": {"eval_family": "private_task"}}, "Only the named"),
        ({"scenario_params": {"eval_family": "https://example.invalid/task"}}, "Only the named"),
        ({"scenario_params": {"eval_family": "inspect_original_inert", "model_route": "external"}}, "Only the named"),
        ({"target_name": "external_model"}, "target_name"),
        ({"initializers": ["custom_python"]}, "initializers"),
        ({"initializer_args": {"custom_python": {"api_key": "secret"}}}, "initializer_args"),
        ({"dataset_names": ["private_data"]}, "dataset_names"),
        ({"dataset_filters": {"data_types": ["text"]}}, "dataset_filters"),
        ({"techniques": ["original_task:converter.unsafe"]}, "techniques"),
        ({"labels": {"api_key": "secret"}}, "labels"),
        ({"scenario_result_id": str(uuid.uuid4())}, "scenario_result_id"),
        ({"task_url": "https://example.invalid/task"}, "task_url"),
        ({"python_source": "print('unsafe')"}, "python_source"),
        ({"sandbox_profile": "docker"}, "sandbox_profile"),
        ({"max_concurrency": 2}, "one case"),
        ({"max_retries": 1}, "no retries"),
        ({"include_baseline": True}, "no baseline"),
    ],
)
async def test_backend_rejects_unsupported_fields_before_initializers_or_task(
    sqlite_instance: SQLiteMemory, input_fields: dict[str, object], expected: str
) -> None:
    service = ScenarioRunService()
    request = RunScenarioRequest.model_validate({"scenario_name": "benchmark.inspect_original_inert", **input_fields})
    try:
        with (
            patch.object(service, "_run_initializers_async", new_callable=AsyncMock) as initializers,
            patch.object(EvalSourceFactory, "resolve_original_inert") as source,
        ):
            with pytest.raises(ValueError, match=expected):
                await service._prepare_run_async(request=request)
        initializers.assert_not_awaited()
        source.assert_not_called()
    finally:
        await service.shutdown_async()


@pytest.mark.usefixtures("patch_central_database")
async def test_source_pin_drift_rejects_before_creating_a_run(sqlite_instance: SQLiteMemory) -> None:
    scenario = InspectOriginalInertScenario()
    scenario.set_params_from_args(args={"eval_family": OriginalInspectTaskId.INERT.value})
    with (
        patch.object(EvalSourceFactory, "ORIGINAL_INERT_SHA256", "f" * 64),
        patch("pyrit.executor.benchmark.inspect_original_runner.eval_async", new_callable=AsyncMock) as launched,
        pytest.raises(ValueError, match="pinned SHA256"),
    ):
        await scenario.initialize_async()
    launched.assert_not_awaited()
    assert sqlite_instance.get_scenario_results() == []


@pytest.mark.parametrize("temp_directory", [r"\\remote\share", "//remote/share"])
def test_log_allocation_rejects_network_shares_before_creation(temp_directory: str) -> None:
    with (
        patch(
            "pyrit.scenario.scenarios.benchmark.inspect_original_inert.tempfile.gettempdir",
            return_value=temp_directory,
        ),
        patch("pyrit.scenario.scenarios.benchmark.inspect_original_inert.tempfile.mkdtemp") as create,
        pytest.raises(ValueError, match="local temporary directory"),
    ):
        _allocate_log_dir(run_instance_id=uuid.uuid4())
    create.assert_not_called()
