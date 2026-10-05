# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test-only cohost ABI child using the unchanged public, no-model original Inspect Task."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stdout
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    from pathlib import Path

    from inspect_ai.event import ScoreEvent
    from inspect_ai.log import EvalLog

    from pyrit.backend.models.original_worker import CohostWorkerRequest
    from pyrit.models import EvalRunRef, ScenarioResult


def _write(*, root: Path, name: str, value: dict[str, JsonValue]) -> bytes:
    content = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    (root / name).write_bytes(content)
    return content


async def _run_async(request: dict[str, JsonValue]) -> dict[str, JsonValue]:
    from pyrit.backend.models.original_worker import CohostWorkerRequest
    from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
    from pyrit.memory import CentralMemory, SQLiteMemory
    from pyrit.models import EvalRunRef, ScenarioIdentifier, ScenarioResult, ScenarioRunState, config_hash

    message = CohostWorkerRequest.model_validate_json(json.dumps(request))
    root = message.run_root
    memory = SQLiteMemory(db_path=root / "worker-scratch.sqlite", silent=True, _defer_initialization=True)
    memory.results_path = str(root / "results")
    memory.disable_embedding()
    await memory.initialize_async()
    CentralMemory.set_memory_instance(memory)
    try:
        source = EvalSourceFactory.resolve_original_inert(family="inspect_original_inert")
        assert source.spec.package.source_sha256 == message.source_sha256
        state: Literal["success", "error", "cancelled"] = "success"
        run = EvalRunRef(spec=source.spec, run_instance_id=uuid4())
        archive: bytes | None = None
        inspected: EvalLog | None = None
        event: ScoreEvent | None = None
        mode = os.getenv("PUBLIC_COHOST_FIXTURE_MODE", "")
        if mode == "invalid-terminal":
            return {"unexpected": True}
        if mode == "slow":
            while not await asyncio.to_thread((root / "cancel.request.json").exists):
                await asyncio.sleep(0.02)
            state = "cancelled"
        elif mode == "error":
            state = "error"
        else:
            from inspect_ai.event import ScoreEvent
            from inspect_ai.log import read_eval_log

            from pyrit.executor.benchmark.inspect_original_runner import run_original_inert_eval_async

            logs = root / "tmp" / "original-logs"
            await asyncio.to_thread(logs.mkdir)
            completed = await run_original_inert_eval_async(memory=memory, log_dir=logs)
            run = completed.run
            archive = await asyncio.to_thread(completed.archive_path.read_bytes)
            await asyncio.to_thread((root / "source.eval").write_bytes, archive)
            inspected = await asyncio.to_thread(read_eval_log, io.BytesIO(archive), format="eval")
            assert inspected.samples is not None
            [sample] = inspected.samples
            [event] = [item for item in sample.events if isinstance(item, ScoreEvent)]
        scenario_result = ScenarioResult(
            scenario_identifier=ScenarioIdentifier(
                class_name="PublicCohostOriginalWorker",
                version=1,
                params={
                    "execution_owner": "task_owned",
                    "eval_spec_sha256": run.spec.spec_sha256,
                    "source_kind": run.spec.package.kind.value,
                    "source_name": run.spec.package.name,
                    "source_sha256": run.spec.package.source_sha256,
                    "harness_name": run.spec.harness.name,
                    "harness_sha256": run.spec.harness.config_sha256,
                    "model_route_name": run.spec.model_route.name,
                    "model_route_sha256": run.spec.model_route.config_sha256,
                    "case_set_sha256": config_hash({"case_ids": [source.case.case_id]}),
                },
            ),
            attack_results={},
            scenario_run_state=(
                ScenarioRunState.COMPLETED
                if state == "success"
                else ScenarioRunState.CANCELLED
                if state == "cancelled"
                else ScenarioRunState.FAILED
            ),
            metadata={"run_instance_id": str(run.run_instance_id), "eval_spec_sha256": run.spec.spec_sha256},
        )
        await memory.add_scenario_results_to_memory_async(scenario_results=[scenario_result])
        await asyncio.to_thread(
            _write_evidence,
            message=message,
            run=run,
            scenario_result=scenario_result,
            state=state,
            archive=archive,
            inspected=inspected,
            event=event,
        )
        return {
            "abi_version": 1,
            "job_ref": str(message.job_ref),
            "run_instance_id": str(message.run_instance_id),
            "state": state,
        }
    finally:
        await memory.dispose_engine_async()


def _write_evidence(
    *,
    message: CohostWorkerRequest,
    run: EvalRunRef,
    scenario_result: ScenarioResult,
    state: Literal["success", "error", "cancelled"],
    archive: bytes | None,
    inspected: EvalLog | None,
    event: ScoreEvent | None,
) -> None:
    from pyrit._compatibility import get_compatibility_id
    from pyrit.backend.models.original_worker import CohostRelayConfig
    from pyrit.backend.services.original_model_relay import OriginalModelRelay
    from pyrit.backend.services.original_sandbox_observer import OriginalSandboxObserver
    from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
    from pyrit.models import config_hash

    root = message.run_root
    source = EvalSourceFactory.resolve_original_inert(family="inspect_original_inert")
    route = CohostRelayConfig(endpoint="https://pyrit-github-pipeline.openai.azure.com/").route_metadata
    manifest: dict[str, JsonValue] = {
        "app_run_id": str(message.app_run_id),
        "job_ref": str(message.job_ref),
        "operator_oid": message.operator_oid,
        "profile_ref": message.profile_ref,
        "model_route": route,
    }
    assert config_hash(manifest) == message.manifest_sha256
    scenario_id = scenario_result.id
    scenario = _write(
        root=root,
        name="worker-scenario.json",
        value={
            "id": str(scenario_id),
            "authority": "transient-worker-scenario-provenance-only",
            "metadata": scenario_result.metadata,
        },
    )
    _write(
        root=root,
        name="source-manifest.json",
        value={"synthetic_public_fixture_only": True, "source_sha256": message.source_sha256},
    )
    lease: dict[str, JsonValue] = {
        "schema": "cohost-owned-sandbox-cleanup/v1",
        "run_id": str(message.run_instance_id),
        "create_attempted": False,
        "create_outcome_known": True,
        "owned_sandbox_ids": [],
        "owned_snapshot_ids": [],
        "errors": [],
        "unresolved_operation_ids": [],
        **dict.fromkeys(OriginalSandboxObserver.REQUIRED_TRUE, True),
    }
    operation: dict[str, JsonValue] = {
        "schema": "cohost-original-operation-terminal/v1",
        "run_id": str(message.run_instance_id),
        "requested": [],
        "terminal": [],
    }
    _write(
        root=root,
        name="runner-closure.json",
        value={
            "lease_closure": lease,
            "operation_terminal": operation,
            "cleanup_errors": [],
            "sdk_thread_drain": {"sdk_calls_drained": True, "active_sdk_calls": 0},
        },
    )
    role = OriginalModelRelay.make_close_receipt(
        run_id=message.run_instance_id,
        active_requests=0,
        request_count=0,
        observed_tokens=0,
        upstream_drained=True,
        usage_complete=True,
        evidence=b"",
    )
    envelope: dict[str, JsonValue] = {
        "schema_version": 1,
        "app_run_id": str(message.app_run_id),
        "job_ref": str(message.job_ref),
        "profile_ref": message.profile_ref,
        "operator_oid": message.operator_oid,
        "run_instance_id": str(run.run_instance_id),
        "worker_scenario_id": str(scenario_id),
        "worker_scenario_sha256": hashlib.sha256(scenario).hexdigest(),
        "manifest_sha256": message.manifest_sha256,
        "source_sha256": message.source_sha256,
        "case_run_id": run.case_run_id(case=source.case),
        "model_role": "evaluated",
        "model_route_sha256": source.spec.model_route.config_sha256,
        "model_role_receipt_id": role["receipt_id"],
        "model_role_sha256": role["receipt_sha256"],
        "source_state": state,
        "source_complete": state == "success",
        "operation_terminal_receipt_id": f"cohost-terminal-{message.run_instance_id}",
        "operation_terminal_sha256": config_hash(operation),
        "cleanup": {"state": "proved", "receipt_id": f"cohost-cleanup-{message.run_instance_id}"},
        "cleanup_sha256": config_hash(lease),
    }
    if archive is not None:
        assert inspected is not None and event is not None
        envelope.update(
            archive_sha256=hashlib.sha256(archive).hexdigest(),
            archive_bytes=len(archive),
            inspect_run_id=inspected.eval.run_id,
            inspect_eval_id=inspected.eval.eval_id,
            final_score_event_id=event.uuid,
            final_score_event_sha256=config_hash({"event": event.model_dump(mode="json", exclude_none=True)}),
        )
    _write(root=root, name="backend-intake-envelope.json", value=envelope)
    names = ["backend-intake-envelope.json", "worker-scenario.json", "source-manifest.json", "runner-closure.json"]
    if archive is not None:
        names.append("source.eval")
    _write(
        root=root,
        name="worker-provenance.json",
        value={
            "abi_version": 1,
            "execution_profile": "COHOST-TRUSTED/v1",
            "app_run_id": str(message.app_run_id),
            "job_ref": str(message.job_ref),
            "control_run_instance_id": str(message.run_instance_id),
            "framework_run_instance_id": str(run.run_instance_id),
            "operator_oid": message.operator_oid,
            "profile_ref": message.profile_ref,
            "source_alias": message.source_alias,
            "source_sha256": message.source_sha256,
            "manifest_sha256": message.manifest_sha256,
            "model_route": route,
            "public_commit": get_compatibility_id().rsplit("+g", 1)[-1],
            "state": state,
            "failure_type": None,
            "process_identity": {"pid": os.getpid()},
            "private_imports_after_memory": True,
            "authority": "public-harmless-process-fixture-only",
            "guest_receives_preview_credentials": False,
            "scratch_memory_authority": "transient-worker-staging",
            "durable_memory_authority": "authenticated-backend-only",
            "scratch_grade_counts": {},
            "artifacts": {
                name: {
                    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                    "bytes": (root / name).stat().st_size,
                }
                for name in names
            },
        },
    )


def main() -> None:
    """Use supervisor-provided roots before any PyRIT or original-source import."""
    request = TypeAdapter(dict[str, JsonValue]).validate_json(sys.stdin.buffer.readline(65_537), strict=True)
    sys.path.insert(0, os.environ["PUBLIC_COHOST_FIXTURE_REPO"])
    tempfile.tempdir = None
    if sys.platform == "win32":
        from platformdirs import windows as platformdirs_windows

        platformdirs_windows.get_win_folder = platformdirs_windows.get_win_folder_from_env_vars
    with redirect_stdout(io.StringIO()):
        terminal = asyncio.run(_run_async(request))
    sys.stdout.write(json.dumps(terminal, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
