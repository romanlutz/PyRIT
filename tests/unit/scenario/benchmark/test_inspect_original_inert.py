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
from inspect_ai.event import ScoreEvent
from inspect_ai.log import read_eval_log
from sqlalchemy import func, select

from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.backend.services.scenario_service import ScenarioService
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.memory.memory_models import AttackResultEntry, ScoreEntry
from pyrit.models import ScenarioRunPlan, ScenarioRunState, ScoreStatus
from pyrit.models.catalog.scenario import OriginalInspectImportSummary, OriginalInspectTaskId, RunScenarioRequest
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
async def test_one_click_runs_unchanged_inspect_twice_with_distinct_unscored_sqlite_evidence(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    registry = ScenarioRegistry()
    run_references: list[OriginalInspectImportSummary] = []
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
            assert reference.score_status == "unscored"
            assert reference.episode_id == f"inspect-run-{work.run.run_instance_id.hex}"

            snapshot = sqlite_instance.native_cyber_evidence.get_finalized_unscored_inspect_capture(
                run_id=reference.episode_id
            )
            assert snapshot.coverage_complete
            assert snapshot.score_id is None and snapshot.score_status is ScoreStatus.UNDETERMINED
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
            plan = ScenarioRunPlan.model_validate(result.metadata["run_plan"])
            assert plan.run_instance_id == work.run.run_instance_id
            assert plan.seed_groups[0].id == reference.case_run_id
            assert plan.seed_groups[0].source_sha256 == reference.source_sha256

    assert len({reference.episode_id for reference in run_references}) == 2
    assert len({reference.case_run_id for reference in run_references}) == 2
    assert len({reference.inspect_run_id for reference in run_references}) == 2
    assert len({reference.archive_sha256 for reference in run_references}) == 2
    assert all(not directory.exists() for directory in original_dirs)
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_one_click_backend_returns_unscored_summary_and_progress(sqlite_instance: SQLiteMemory) -> None:
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
        assert summary.original_inspect_import is not None
        assert progress.run.original_inspect_import == summary.original_inspect_import
        assert progress.results == []
        assert summary.successful_attacks == 0
        assert summary.original_inspect_import.score_status == "unscored"
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
