# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real public original Inspect Task, SQLite and one-click Scenario contract."""

from __future__ import annotations

import asyncio
import hashlib
import io
import uuid
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
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter
from pyrit.memory.memory_models import (
    AttackResultEntry,
    NativeCyberEpisodeEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    ScoreEntry,
)
from pyrit.models import AttackOutcome, ScenarioRunPlan, ScenarioRunState, ScoreStatus, config_hash
from pyrit.models.catalog.scenario import OriginalInspectImportSummary, OriginalInspectTaskId, RunScenarioRequest
from pyrit.models.native_cyber_evidence import NativeCyberEvidenceSource, NativeCyberRawKind
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
            episode_row.required_raw_streams = (
                [
                    key
                    for key in original_required
                    if key["observed_source_id"]
                    != (
                        InspectOriginalEvalImporter.ARCHIVE_KEY.observed_source_id
                        if tamper == "archive_not_required"
                        else InspectOriginalEvalImporter.RESOLVED_KEY.observed_source_id
                    )
                ]
                if enabled and tamper in {"archive_not_required", "resolved_not_required"}
                else [dict(key) for key in original_required]
            )

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
@pytest.mark.parametrize("tamper", ["case_id", "source_pin"])
async def test_one_click_readback_rejects_foreign_plan_case_or_source_pin(
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
    changed_plan = {
        **plan,
        "seed_groups": [
            {
                **plan["seed_groups"][0],
                "case_id": "f" * 64 if tamper == "case_id" else foreign_case.case_id,
                "source_sha256": "f" * 64 if tamper == "source_pin" else plan["seed_groups"][0]["source_sha256"],
            }
        ],
    }
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
