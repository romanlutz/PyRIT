# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Ordinary app lifespan/API to a genuine public original child and fresh canonical SQLite."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from inspect_ai.log import read_eval_log
from pydantic import TypeAdapter
from starlette.datastructures import State

from pyrit._compatibility import COMPATIBILITY_HEADER, get_compatibility_id
from pyrit.backend.main import app
from pyrit.backend.middleware.auth import AuthenticatedUser, EntraAuthMiddleware
from pyrit.backend.services.original_evidence_service import (
    OriginalEvidenceRecord,
    OriginalEvidenceService,
    get_original_evidence_service,
)
from pyrit.backend.services.original_run_admission import APPROVED_ORIGINAL_SCENARIO
from pyrit.backend.services.original_worker_artifacts import OriginalWorkerArtifacts
from pyrit.backend.services.original_worker_runtime import OriginalWorkerRuntime
from pyrit.common.singleton import Singleton
from pyrit.executor.benchmark.inspect_eval_projection import project_inspect_sample
from pyrit.memory import CentralMemory, SQLiteMemory
from pyrit.models import AttackOutcome, ScoreStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pyrit.backend.models.original_worker import CohostBackendConfig

_ACTOR = AuthenticatedUser(
    oid="00000000-0000-4000-8000-000000000003",
    name="Public fixture",
    email="fixture@example.test",
    groups=["00000000-0000-4000-8000-000000000004"],
)


@pytest.fixture
def compatibility_id() -> str:
    """Use the same actual source stamp in the API and genuine installed child."""
    return get_compatibility_id()


@pytest.fixture(name="configured_environment")
async def configured_environment_async(
    *, tmp_path: Path, worker_config: CohostBackendConfig
) -> AsyncIterator[dict[str, str]]:
    config_path = tmp_path / "cohost.json"
    await asyncio.to_thread(config_path.write_text, worker_config.model_dump_json(), encoding="utf-8")
    normal = tmp_path / "backend.yaml"
    await asyncio.to_thread(
        normal.write_text,
        "memory_db_type: sqlite\nenv_files: []\ninitialization_scripts: []\ninitializers: []\n"
        "silent: true\nmax_concurrent_scenario_runs: 1\n",
        encoding="utf-8",
    )
    database_root = tmp_path / "canonical"
    await asyncio.to_thread(database_root.mkdir)
    environment = {
        "PYRIT_DEV_MODE": "true",
        "PYRIT_CONFIG_FILE": str(normal),
        "PYRIT_ORIGINAL_WORKER_CONFIG": str(config_path),
        "PYRIT_ORIGINAL_WORKER_CONFIG_SHA256": hashlib.sha256(
            await asyncio.to_thread(config_path.read_bytes)
        ).hexdigest(),
        "ENTRA_TENANT_ID": "00000000-0000-4000-8000-000000000006",
        "ENTRA_CLIENT_ID": "00000000-0000-4000-8000-000000000007",
        "ENTRA_ALLOWED_GROUP_IDS": _ACTOR.groups[0],
        "PYRIT_ALLOW_UNAUTHENTICATED_ADMIN": "false",
    }
    from pyrit.memory import sqlite_memory

    memory = SQLiteMemory.__new__(SQLiteMemory)
    with patch.object(memory, "cleanup"):
        memory.__init__(db_path=database_root / "backend.sqlite", silent=True, _defer_initialization=True)
    memory.results_path = str(database_root)
    memory.disable_embedding()
    await memory.initialize_async()
    with (
        patch.dict(os.environ, environment),
        patch.object(sqlite_memory, "DB_DATA_PATH", database_root),
        patch.object(CentralMemory, "_memory_instance", memory),
        patch.object(CentralMemory, "get_memory_instance", return_value=memory),
        patch.dict(Singleton._instances, {SQLiteMemory: memory}),
        patch.object(app, "state", State()),
        patch.object(app, "middleware_stack", None),
        patch.object(EntraAuthMiddleware, "_authenticate_with_graph_async", AsyncMock(return_value=_ACTOR)),
        patch("pyrit.setup.configuration_loader.DEFAULT_CONFIG_PATH", tmp_path / "no-default-config"),
    ):
        get_original_evidence_service.cache_clear()
        yield environment
        get_original_evidence_service.cache_clear()
    await memory.dispose_engine_async()


async def _terminal_async(*, client: AsyncClient, run_id: str, timeout_seconds: float) -> dict[str, object]:
    async with asyncio.timeout(timeout_seconds):
        while True:
            response = await client.get(f"/api/scenarios/runs/{run_id}")
            assert response.status_code == 200, response.text
            result = TypeAdapter(dict[str, object]).validate_json(response.content)
            if result["status"] not in ("CREATED", "IN_PROGRESS", "QUEUED"):
                return result
            await asyncio.sleep(0.02)


@pytest.mark.filterwarnings("ignore:MemoryInterface:DeprecationWarning")
@pytest.mark.parametrize("completion_limit", [4096, 8192])
async def test_normal_lifespan_api_child_original_archive_canonical_sqlite_and_view_async(
    *, configured_environment: dict[str, str], worker_config: CohostBackendConfig, completion_limit: int
) -> None:
    from pyrit.backend import main

    config = worker_config.model_copy(
        update={
            "relay": worker_config.relay.model_copy(
                update={
                    "max_completion_tokens": completion_limit,
                    "request_timeout_seconds": 60 if completion_limit == 8192 else 180,
                }
            )
        }
    )
    config_path = Path(configured_environment["PYRIT_ORIGINAL_WORKER_CONFIG"])
    await asyncio.to_thread(config_path.write_text, config.model_dump_json(), encoding="utf-8")
    config_sha256 = hashlib.sha256(await asyncio.to_thread(config_path.read_bytes)).hexdigest()
    pending, resume = Event(), Event()
    publish = OriginalEvidenceService._publish

    def gated_publish(self: OriginalEvidenceService, *, record: OriginalEvidenceRecord, terminal: bool) -> None:
        publish(self, record=record, terminal=terminal)
        if not terminal:
            pending.set()
            if not resume.wait(30):
                raise TimeoutError("Public fixture pending-publication barrier expired.")

    headers = {COMPATIBILITY_HEADER: get_compatibility_id(), "Authorization": "Bearer public-fixture"}
    with (
        patch.object(main, "setup_frontend"),
        patch.dict(os.environ, {"PYRIT_ORIGINAL_WORKER_CONFIG_SHA256": config_sha256}),
        patch.object(OriginalEvidenceService, "_publish", new=gated_publish),
    ):
        async with main.app.router.lifespan_context(main.app):
            assert main.app.state.runtime_lifecycle.state == "ready"
            owned = main.app.state.original_worker_runtime
            assert isinstance(owned, OriginalWorkerRuntime)
            assert owned.relay.config.max_completion_tokens == completion_limit
            assert owned.relay.config.request_timeout_seconds == (60 if completion_limit == 8192 else 180)
            async with AsyncClient(
                transport=ASGITransport(app=main.app), base_url="http://test", headers=headers
            ) as client:
                catalog = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
                assert catalog.status_code == 200, catalog.text
                admission = catalog.json()["original_run_admission"]
                assert admission["status"] == "ready"
                blocked = await client.post(
                    "/api/scenarios/runs", json={"scenario_name": "benchmark.inspect_original_inert"}
                )
                assert blocked.status_code == 400
                started = await client.post(
                    "/api/scenarios/runs",
                    json={
                        "scenario_name": APPROVED_ORIGINAL_SCENARIO,
                        "original_admission_ref": admission["admission_ref"],
                    },
                )
                assert started.status_code == 202, started.text
                run_id = started.json()["scenario_result_id"]
                try:
                    assert await asyncio.to_thread(pending.wait, worker_config.active_timeout_seconds + 5)
                    publishing = await client.get(f"/api/scenarios/runs/{run_id}")
                    assert publishing.status_code == 200, publishing.text
                    assert publishing.json()["status"] == "IN_PROGRESS"
                    assert publishing.json()["original_source_result"] is None
                    publishing_history = await client.get("/api/scenarios/runs")
                    assert publishing_history.status_code == 200, publishing_history.text
                    late_cancel = await client.post(f"/api/scenarios/runs/{run_id}/cancel")
                    assert late_cancel.status_code == 409, late_cancel.text
                    assert "publication" in late_cancel.json()["detail"]
                    memory = CentralMemory.get_memory_instance()
                    pending_header = await asyncio.to_thread(
                        memory.get_scenario_result_header, scenario_result_id=run_id
                    )
                    assert pending_header is not None
                    pending_fields = {
                        key: copy.deepcopy(pending_header.metadata[key])
                        for key in (OriginalEvidenceRecord.METADATA_KEY, OriginalEvidenceRecord.PENDING_KEY)
                    }
                    forged = copy.deepcopy(pending_fields)
                    forged[OriginalEvidenceRecord.METADATA_KEY]["envelope_sha256"] = "0" * 64
                    forged[OriginalEvidenceRecord.PENDING_KEY] = "0" * 64
                    await asyncio.to_thread(
                        memory.update_scenario_metadata_fields, scenario_result_id=run_id, fields=forged
                    )
                    try:
                        rejected = await client.get(f"/api/scenarios/runs/{run_id}")
                        assert rejected.status_code == 409, rejected.text
                    finally:
                        await asyncio.to_thread(
                            memory.update_scenario_metadata_fields,
                            scenario_result_id=run_id,
                            fields=pending_fields,
                        )
                finally:
                    resume.set()
                result = await _terminal_async(
                    client=client,
                    run_id=run_id,
                    timeout_seconds=worker_config.active_timeout_seconds + worker_config.cleanup_timeout_seconds + 5,
                )
                [record] = owned.supervisor._jobs.values()
                assert record.request is not None and record.terminal is not None and record.process is not None
                independently_read = await asyncio.to_thread(
                    OriginalWorkerArtifacts.read,
                    config=owned.preflight.config,
                    request=record.request,
                    terminal=record.terminal,
                    process_id=record.process.pid,
                )
                assert independently_read.provenance.framework_run_instance_id != record.control_id
                assert result["status"] == "COMPLETED", {
                    "response": result,
                    "worker_failure": record.failure_code,
                    "worker_cancel_requested": record.cancel_requested,
                }
                assert result["original_source_result"]["original_score"] == "1.0"
                assert result["original_source_result"]["pyrit_score_status"] == ScoreStatus.COMPLETE.value
                assert result["original_source_result"]["pyrit_outcome"] == AttackOutcome.UNDETERMINED.value
                assert record.process is not None and record.process.returncode == 0
                assert record.receipt is not None and record.artifacts is not None
                envelope = record.artifacts.admission.envelope
                assert record.request is not None
                assert envelope.run_instance_id != record.request.run_instance_id
                assert envelope.run_instance_id != envelope.app_run_id
                assert envelope.run_instance_id != envelope.job_ref
                assert hashlib.sha256(record.artifacts.archive).hexdigest() == envelope.archive_sha256
                assert record.artifacts.provenance.process_identity["pid"] == record.process.pid
                log = await asyncio.to_thread(read_eval_log, str(record.root / "source.eval"))
                assert log.samples is not None
                [sample] = log.samples
                assert sample.scores is not None
                [original] = sample.scores.values()
                [score] = await CentralMemory.get_memory_instance().get_scores_async(
                    score_ids=[str(record.receipt.score_id)]
                )
                assert float(score.score_value) == original.value
                assert score.score_metadata["inspect_final_score_event_id"] == envelope.final_score_event_id
                assert record.receipt.attack_result_id != envelope.worker_attack_result_id
                assert record.receipt.score_id != envelope.worker_score_id
                attack_id = str(record.receipt.attack_result_id)
                conversations = await client.get(f"/api/attacks/{attack_id}/conversations")
                assert conversations.status_code == 200, conversations.text
                history = await client.get("/api/scenarios/runs")
                assert history.status_code == 200, history.text
                detail = await client.get(f"/api/attacks/{attack_id}")
                assert detail.status_code == 200, detail.text
                assert detail.json()["source_read_only"] is True
                conversation_id = detail.json()["conversation_id"]
                messages = await client.get(
                    f"/api/attacks/{attack_id}/messages", params={"conversation_id": conversation_id}
                )
                assert messages.status_code == 200, messages.text
                pieces = [piece for message in messages.json()["messages"] for piece in message["message_pieces"]]
                projection = project_inspect_sample(
                    sample=sample,
                    log_run_id=log.eval.run_id,
                    eval_id=log.eval.eval_id,
                    archive_sha256=envelope.archive_sha256,
                    sample_index=1,
                    start_sequence=1,
                    conversation_id=conversation_id,
                    case_run_id=envelope.case_run_id,
                )
                assert projection.unprojected_messages == 0
                assert len(pieces) == len(projection.message_pieces)
                assert [
                    (piece["role"], piece["converted_value_data_type"], piece["converted_value"]) for piece in pieces
                ] == [
                    (piece.role, piece.converted_value_data_type, piece.converted_value)
                    for piece in projection.message_pieces
                ]
                assert [hashlib.sha256(piece["converted_value"].encode()).hexdigest() for piece in pieces] == [
                    hashlib.sha256(piece.converted_value.encode()).hexdigest() for piece in projection.message_pieces
                ]
                assert sum(piece["converted_value_data_type"] == "function_call" for piece in pieces) == len(
                    projection.tool_request_ids
                )
                assert sum(piece["converted_value_data_type"] == "function_call_output" for piece in pieces) == len(
                    projection.tool_result_ids
                )
                edited = await client.patch(f"/api/attacks/{attack_id}", json={"outcome": "success"})
                assert edited.status_code == 409
                assert await asyncio.to_thread(worker_config.jobs_root.exists)
                assert not owned.has_active_work()


@pytest.mark.parametrize("mode,expected", [("slow", "CANCELLED"), ("error", "FAILED"), ("invalid-terminal", "FAILED")])
@pytest.mark.filterwarnings("ignore:MemoryInterface:DeprecationWarning")
async def test_normal_api_preserves_cancel_error_and_invalid_terminal_without_grade_async(
    *, configured_environment: dict[str, str], worker_config: CohostBackendConfig, mode: str, expected: str
) -> None:
    from pyrit.backend import main

    config = worker_config.model_copy(
        update={"worker_environment": {**worker_config.worker_environment, "PUBLIC_COHOST_FIXTURE_MODE": mode}}
    )
    config_path = Path(configured_environment["PYRIT_ORIGINAL_WORKER_CONFIG"])
    await asyncio.to_thread(config_path.write_text, config.model_dump_json(), encoding="utf-8")
    config_sha256 = hashlib.sha256(await asyncio.to_thread(config_path.read_bytes)).hexdigest()
    headers = {COMPATIBILITY_HEADER: get_compatibility_id(), "Authorization": "Bearer public-fixture"}
    with (
        patch.dict(os.environ, {"PYRIT_ORIGINAL_WORKER_CONFIG_SHA256": config_sha256}),
        patch.object(main, "setup_frontend"),
    ):
        async with main.app.router.lifespan_context(main.app):
            assert main.app.state.runtime_lifecycle.state == "ready"
            owned = main.app.state.original_worker_runtime
            async with AsyncClient(
                transport=ASGITransport(app=main.app), base_url="http://test", headers=headers
            ) as client:
                admitted = (await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")).json()
                started = await client.post(
                    "/api/scenarios/runs",
                    json={
                        "scenario_name": APPROVED_ORIGINAL_SCENARIO,
                        "original_admission_ref": admitted["original_run_admission"]["admission_ref"],
                    },
                )
                assert started.status_code == 202, started.text
                run_id = started.json()["scenario_result_id"]
                if mode == "slow":
                    for _ in range(1000):
                        if owned.supervisor._jobs and next(iter(owned.supervisor._jobs.values())).process is not None:
                            break
                        await asyncio.sleep(0.01)
                    cancelled = await client.post(f"/api/scenarios/runs/{run_id}/cancel")
                    assert cancelled.status_code in (200, 202), cancelled.text
                terminal = await _terminal_async(
                    client=client,
                    run_id=run_id,
                    timeout_seconds=config.active_timeout_seconds + config.cleanup_timeout_seconds + 5,
                )
                assert terminal["status"] == expected, terminal
                assert terminal["original_source_result"]["original_score"] is None
                assert terminal["original_source_result"]["pyrit_score_status"] is None
                assert terminal["original_source_result"]["pyrit_outcome"] is None
                assert await CentralMemory.get_memory_instance().get_scores_async() == []
                if mode == "invalid-terminal":
                    assert owned.supervisor._quarantined


@pytest.mark.filterwarnings("ignore:MemoryInterface:DeprecationWarning")
async def test_publication_failure_is_readable_failed_without_a_published_grade_async(
    *, configured_environment: dict[str, str], worker_config: CohostBackendConfig
) -> None:
    from pyrit.backend import main

    publish = OriginalEvidenceService._publish

    def fail_terminal(self: OriginalEvidenceService, *, record: OriginalEvidenceRecord, terminal: bool) -> None:
        if terminal:
            raise ValueError("Public fixture canonical terminal publication failed.")
        publish(self, record=record, terminal=False)

    with (
        patch.object(main, "setup_frontend"),
        patch.object(OriginalEvidenceService, "_publish", new=fail_terminal),
    ):
        async with main.app.router.lifespan_context(main.app):
            async with AsyncClient(
                transport=ASGITransport(app=main.app),
                base_url="http://test",
                headers={COMPATIBILITY_HEADER: get_compatibility_id(), "Authorization": "Bearer public-fixture"},
            ) as client:
                catalog = await client.get(f"/api/scenarios/catalog/{APPROVED_ORIGINAL_SCENARIO}")
                assert catalog.status_code == 200
                started = await client.post(
                    "/api/scenarios/runs",
                    json={
                        "scenario_name": APPROVED_ORIGINAL_SCENARIO,
                        "original_admission_ref": catalog.json()["original_run_admission"]["admission_ref"],
                    },
                )
                assert started.status_code == 202
                run_id = started.json()["scenario_result_id"]
                terminal = await _terminal_async(
                    client=client,
                    run_id=run_id,
                    timeout_seconds=worker_config.active_timeout_seconds + worker_config.cleanup_timeout_seconds + 5,
                )
                assert terminal["status"] == "FAILED"
                assert terminal["original_source_result"] is None
                history = await client.get("/api/scenarios/runs")
                assert history.status_code == 200
                [record] = main.app.state.original_worker_runtime.supervisor._jobs.values()
                assert record.receipt is None
                assert record.failure_code == "worker_canonical_intake_unverified"
                memory = CentralMemory.get_memory_instance()
                pending_header = await asyncio.to_thread(memory.get_scenario_result_header, scenario_result_id=run_id)
                assert pending_header is not None
                stored = OriginalEvidenceRecord.from_metadata(
                    pending_header.metadata[OriginalEvidenceRecord.METADATA_KEY]
                )
                assert stored.persistence_verified is False and stored.attack_result_id is not None
                detail = await client.get(f"/api/attacks/{stored.attack_result_id}")
                assert detail.status_code == 403


async def test_normal_startup_rejects_changed_worker_before_registering_or_spawning_async(
    *, configured_environment: dict[str, str], worker_config: CohostBackendConfig
) -> None:
    from pyrit.backend import main
    from pyrit.backend.services.original_run_admission import get_original_run_gateway

    config = worker_config.model_copy(update={"worker_entrypoint_sha256": "0" * 64})
    config_path = Path(configured_environment["PYRIT_ORIGINAL_WORKER_CONFIG"])
    await asyncio.to_thread(config_path.write_text, config.model_dump_json(), encoding="utf-8")
    digest = hashlib.sha256(await asyncio.to_thread(config_path.read_bytes)).hexdigest()
    with (
        patch.dict(os.environ, {"PYRIT_ORIGINAL_WORKER_CONFIG_SHA256": digest}),
        patch.object(main, "setup_frontend"),
        patch("pyrit.backend.services.original_worker_preflight.asyncio.create_subprocess_exec") as spawn,
    ):
        async with main.app.router.lifespan_context(main.app):
            assert main.app.state.runtime_lifecycle.state == "restart-required"
            assert main.app.state.original_worker_runtime is None
            assert get_original_run_gateway() is None
            spawn.assert_not_called()
