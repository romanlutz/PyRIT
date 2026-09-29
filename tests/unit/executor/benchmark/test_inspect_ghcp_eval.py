# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real Inspect Task/scorer lifecycle and fail-closed Compose validation."""

from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai import Task, eval_async
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import ChatMessageUser, Model, ModelOutput
from inspect_ai.scorer import Target, includes
from inspect_ai.solver import TaskState, solver
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec

import examples.inspect_ghcp_protocol_smoke as smoke
from examples.inspect_ghcp_protocol_smoke import (
    _exact_missing_container,
    _require_protocol_proof,
    original_target_marker_scorer,
)
from pyrit.executor.benchmark._inspect_ghcp_runtime import InspectGhcpLimits
from pyrit.executor.benchmark.inspect_ghcp_eval import InspectGhcpEvaluation, InspectGhcpOutcome, InspectGhcpTaskBinding
from pyrit.executor.benchmark.inspect_ghcp_model import InspectLoopbackModelAPI
from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.models.inspect_ghcp import (
    InspectGhcpJudgment,
    InspectGhcpReport,
    InspectGhcpStatus,
    InspectGhcpTaskKind,
)
from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot
from pyrit.score.float_scale.inspect_ghcp_report_scorer import InspectGhcpReportScorer

if TYPE_CHECKING:
    from inspect_ai.solver import Generate, Solver

    from pyrit.memory import SQLiteMemory


def _sandbox() -> SandboxEnvironmentSpec:
    agent_image = "approved/agent@sha256:" + "a" * 64
    target_image = "approved/target@sha256:" + "b" * 64
    contained = {
        "image": agent_image,
        "user": "10001:10001",
        "read_only": True,
        "pull_policy": "never",
        "cap_drop": ["ALL"],
        "security_opt": ["no-new-privileges:true"],
        "pids_limit": 64,
        "mem_limit": "1.25gb",
        "cpus": 1.0,
        "networks": ["only_internal"],
        "tmpfs": [
            "/home/bridge:rw,exec,size=256m",
            "/workspace:rw,exec,size=64m",
            "/tmp:rw,exec,size=512m",
        ],
    }
    return SandboxEnvironmentSpec(
        "docker",
        ComposeConfig(
            services={
                "agent": ComposeService.model_validate(
                    {
                        **contained,
                        "x-default": True,
                        "tmpfs": [
                            *contained["tmpfs"],
                            "/var/tmp:rw,exec,nosuid,nodev,size=128m,mode=1777",
                        ],
                    }
                ),
                "model-bridge": ComposeService(
                    **{
                        **contained,
                        "tmpfs": [
                            *contained["tmpfs"],
                            "/var/tmp:rw,exec,nosuid,nodev,size=128m,mode=1777",
                        ],
                    }
                ),
                "target": ComposeService(**{**contained, "image": target_image, "mem_limit": "384m"}),
            },
            networks={"only_internal": {"internal": True}},
        ),
    )


def _task() -> Task:
    return Task(
        dataset=[Sample(input="original cyber task prompt", target="answer", id="sample-one")],
        scorer=includes(),
        sandbox=_sandbox(),
        name="original_task",
        version="1",
    )


@pytest.mark.parametrize(
    ("exit_code", "output", "error", "expected"),
    [
        (1, "[]", "Error response from daemon: No such container: {id}", True),
        (1, "", "Error response from daemon: No such object: {id}", True),
        (1, "[]", "Error: No such object: {id}", True),
        (1, "[]", "Error response from daemon: No such container: {other_id}", False),
        (1, "", "permission denied", False),
        (255, "", "Error response from daemon: No such container: {id}", False),
        (1, "unexpected", "Error response from daemon: No such container: {id}", False),
    ],
)
def test_cleanup_accepts_only_exact_container_absence(exit_code: int, output: str, error: str, expected: bool) -> None:
    container_id = "a" * 64
    rendered = error.replace("{id}", container_id).replace("{other_id}", "b" * 64)
    assert (
        _exact_missing_container(code=exit_code, output=output, error=rendered, container_ref=container_id) is expected
    )


def test_original_task_keeps_scorer_and_rejects_egress_or_host_mounts() -> None:
    task = _task()
    binding = InspectGhcpTaskBinding(
        task=task,
        sample_id="sample-one",
        scorer_name="includes",
        provider_endpoint="http://127.0.0.1:11435",
        verify_provider_async=AsyncMock(return_value=True),
    )
    assert binding.validate() is task.sandbox.config
    copied = copy.copy(task)
    assert copied.setup is task.setup
    assert copied.scorer[0] is task.scorer[0]
    assert copied.dataset[0].input == task.dataset[0].input
    bad = _sandbox()
    bad.config.services["agent"].networks = None
    task.sandbox = bad
    with pytest.raises(ValueError, match="internal network"):
        binding.validate()
    task.sandbox = _sandbox()
    task.sandbox.config.services["agent"].volumes = ["/var/run/docker.sock:/var/run/docker.sock"]
    with pytest.raises(ValueError, match="daemon socket"):
        binding.validate()
    with pytest.raises(ValueError, match="loopback"):
        replace(binding, provider_endpoint="http://0.0.0.0:11435").validate()
    with pytest.raises(ValueError, match="separate"):
        replace(binding, target_service="agent").validate()


def test_pinned_multi_sample_binding_selects_exact_id_from_original_dataset() -> None:
    task = _task()
    task.dataset = MemoryDataset(
        samples=[
            Sample(input="identical benign prompt", target="answer", id="sample-one"),
            Sample(input="identical benign prompt", target="answer", id="sample-two"),
        ]
    )
    original_dataset = task.dataset
    binding = InspectGhcpTaskBinding(
        task=task,
        sample_id="sample-two",
        scorer_name="includes",
        approved_sample_ids=("sample-one", "sample-two"),
        provider_endpoint="http://127.0.0.1:11435",
        verify_provider_async=AsyncMock(return_value=True),
    )
    assert binding.validate() is task.sandbox.config
    assert binding.selected_sample() is original_dataset[1]
    assert task.dataset is original_dataset
    with pytest.raises(ValueError, match="exactly one selected original Inspect sample"):
        replace(binding, approved_sample_ids=None).validate()
    with pytest.raises(ValueError, match="pinned original dataset inventory"):
        replace(binding, approved_sample_ids=("sample-one", "unknown")).validate()
    with pytest.raises(ValueError, match="pinned original dataset inventory"):
        replace(binding, sample_id="unknown").validate()
    task.dataset = MemoryDataset(
        samples=[
            original_dataset[0],
            original_dataset[1].model_copy(update={"sandbox": _sandbox()}),
        ]
    )
    with pytest.raises(ValueError, match="Per-sample sandbox overrides"):
        binding.validate()


def test_unqualified_benign_task_cannot_claim_a_cyber_benchmark_score() -> None:
    task = _task()
    binding = InspectGhcpTaskBinding(
        task=task,
        sample_id="sample-one",
        scorer_name="includes",
        kind=InspectGhcpTaskKind.CYBER_BENCHMARK,
        expected_target_image="approved/target@sha256:" + "b" * 64,
        approved_assets={Path(__file__): "a" * 64},
        provider_endpoint="http://127.0.0.1:11435",
        verify_provider_async=AsyncMock(return_value=True),
        verify_image_async=AsyncMock(return_value="b" * 64),
        verify_removed_async=AsyncMock(return_value=True),
        verify_project_cleanup_async=AsyncMock(return_value=True),
    )
    with pytest.raises(ValueError, match="not qualified"):
        binding.validate()


@pytest.mark.usefixtures("patch_central_database")
async def test_task_asset_report_uses_source_relative_labels_not_local_paths(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    task = _task()
    source_path = tmp_path / "private_task.py"
    binding = InspectGhcpTaskBinding(
        task=task,
        sample_id="sample-one",
        scorer_name="includes",
        approved_assets={source_path: "a" * 64},
        approved_asset_labels={source_path: "task.py"},
        provider_endpoint="http://127.0.0.1:11435/v1",
        verify_provider_async=AsyncMock(return_value=True),
    )
    model = MagicMock(spec=Model)
    model.api = InspectLoopbackModelAPI(model_name="qwen3:1.7b", base_url="http://127.0.0.1:11435/v1")
    evaluation = InspectGhcpEvaluation(
        binding=binding,
        attack_factory=MagicMock(),
        model=model,
        model_id="qwen3-local",
        wire_model="qwen3:1.7b",
        cli_path="/opt/pyrit/copilot",
        cli_sha256="b" * 64,
        limits=InspectGhcpLimits(),
    )
    evaluation._store = InspectGhcpEvidenceStore(
        memory=sqlite_instance,
        run_id=evaluation.run_id,
        task_name=task.name,
        task_version=str(task.version),
        started_at=evaluation._started_at,
        raw_byte_limit=10_000,
    )
    report = await evaluation._build_report_async(log=None, log_bytes=None, error="test_error")
    assert report.task_assets_sha256 == {"task.py": "a" * 64}
    assert str(tmp_path) not in report.canonical_json()
    with pytest.raises(ValueError, match="source-relative"):
        replace(binding, approved_asset_labels={source_path: str(source_path)}).validate()


async def test_inspect_original_scorer_and_cleanup_run_once_with_solver_override(tmp_path: Path) -> None:
    scorer_calls: list[str] = []
    cleanup_calls: list[str] = []
    cleanup_times: list[datetime] = []
    original = includes()

    async def cleanup_async(state: TaskState) -> None:
        cleanup_calls.append(str(state.sample_id))
        cleanup_times.append(datetime.now(UTC))
        assert state.scores is not None

    @solver
    def original_override() -> Solver:
        async def solve_async(state: TaskState, generate: Generate) -> TaskState:
            state.output = ModelOutput.from_content(model="mockllm/mock", content="answer")
            scorer_calls.append("solver-called")
            return state

        return solve_async

    task = Task(
        dataset=[Sample(input="original task", target="answer", id="sample-one")],
        scorer=original,
        cleanup=cleanup_async,
        name="original_inspect_task",
    )
    logs = await eval_async(
        tasks=copy.copy(task),
        solver=original_override(),
        model="mockllm/mock",
        log_dir=str(tmp_path),
        max_samples=1,
        retry_on_error=0,
        score_on_error=False,
    )
    assert len(logs) == 1
    assert logs[0].status == "success"
    assert logs[0].samples is not None
    assert not InspectGhcpEvaluation._original_task_failed(log=logs[0])
    assert InspectGhcpEvaluation._original_task_failed(log=logs[0].model_copy(update={"status": "error"}))
    unscored = logs[0].samples[0].model_copy(update={"scores": None})
    assert InspectGhcpEvaluation._original_task_failed(log=logs[0].model_copy(update={"samples": [unscored]}))
    sample = logs[0].samples[0]
    assert len(sample.scores or {}) == 1
    assert len([event for event in sample.events if event.event == "score"]) == 1
    assert sample.events[[event.event for event in sample.events].index("score")].timestamp <= cleanup_times[0]
    resolved = await asyncio.to_thread(read_eval_log, logs[0].location, resolve_attachments="full")
    retained = json.loads(resolved.model_dump_json(exclude_none=True))
    retained_sample = retained["samples"][0]
    score_name = next(iter(retained_sample["scores"]))
    score_events = [
        event for event in retained_sample["events"] if event["event"] == "score" and event.get("scorer") == score_name
    ]
    assert len(score_events) == 1
    assert score_events[0]["score"] == retained_sample["scores"][score_name]
    canonical_live = json.dumps(
        resolved.samples[0].scores[score_name].model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    canonical_retained = json.dumps(retained_sample["scores"][score_name], sort_keys=True, separators=(",", ":"))
    assert canonical_live == canonical_retained
    assert scorer_calls == ["solver-called"]
    assert cleanup_calls == ["sample-one"]


async def test_original_benign_scorer_refuses_to_grade_without_source_pregrading() -> None:
    state = TaskState(
        model="mockllm/mock",
        sample_id="benign-ghcp-protocol-1",
        epoch=1,
        input="benign task",
        messages=[ChatMessageUser(content="benign task")],
    )
    with pytest.raises(RuntimeError, match="pregrading source coverage"):
        await original_target_marker_scorer()(state, Target("marker"))


def test_inspect_adapter_never_imports_native_evaluator_or_environment_lease() -> None:
    root = Path(__file__).parents[4]
    files = (
        root / "pyrit" / "executor" / "benchmark" / "inspect_ghcp_eval.py",
        root / "pyrit" / "executor" / "benchmark" / "_inspect_ghcp_runtime.py",
        root / "pyrit" / "executor" / "benchmark" / "inspect_ghcp_recovery.py",
        root / "pyrit" / "memory" / "inspect_ghcp_evidence.py",
        root / "pyrit" / "prompt_target" / "inspect_ghcp_target.py",
        root / "pyrit" / "score" / "float_scale" / "inspect_ghcp_report_scorer.py",
    )
    forbidden_modules = (
        "pyrit.executor.workflow.native_cyber_eval",
        "pyrit.executor.workflow.native_cli_evaluation",
        "pyrit.prompt_target.native_agent_target",
        "pyrit.prompt_target.native_cli_target",
    )
    forbidden_classes = {"NativeCyberEvaluation", "NativeCyberTaskBinding", "DockerStopOnlyAgentLease"}
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = {
            node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not any(module.startswith(forbidden_modules) for module in imported)
        assert not forbidden_classes & names


async def test_private_stage_receipt_retains_token_fingerprint_not_value(tmp_path: Path) -> None:
    evaluation = object.__new__(InspectGhcpEvaluation)
    evaluation._run_id = "11111111-1111-1111-1111-111111111111"
    evaluation._control_token = "x" * 43
    evaluation._receipt_path = tmp_path / "controller-stage.jsonl"
    evaluation._start_monotonic = time.monotonic()
    await evaluation._checkpoint_async(stage="episode_created")
    await evaluation._checkpoint_async(stage="inspect_eval_started")
    content = evaluation._receipt_path.read_bytes()
    rows = [json.loads(line) for line in content.splitlines()]
    assert "x" * 43 not in content.decode("utf-8")
    assert rows[0]["control_token_sha256"] == hashlib.sha256(b"x" * 43).hexdigest()
    assert "control_token_sha256" not in rows[1]


@pytest.mark.usefixtures("patch_central_database")
async def test_benign_example_rejects_incomplete_evaluation_result() -> None:
    now = datetime.now(UTC)
    report = InspectGhcpReport(
        run_id="11111111-1111-1111-1111-111111111111",
        task_name="inspect_ghcp_benign_protocol_smoke",
        task_version="1",
        sample_id="benign-ghcp-protocol-1",
        cli_sha256="a" * 64,
        model_id="qwen3-local",
        wire_model="qwen3:1.7b",
        started_at=now,
        ended_at=now,
        turn_count=2,
        sdk_event_count=20,
        sdk_event_raw_sha256="b" * 64,
        model_request_count=2,
        model_http_200_count=2,
        host_model_request_count=2,
        host_model_http_200_count=2,
        adversarial_request_count=1,
        adversarial_http_200_count=1,
        tool_start_count=1,
        tool_complete_count=1,
        successful_tool_execution_count=1,
        gateway_audit_sha256="c" * 64,
        host_model_audit_sha256="d" * 64,
        adversarial_audit_sha256="e" * 64,
        control_receipt_sha256="f" * 64,
        token_files_absent_before_turn=True,
        provider_verified=True,
        agent_container_id="1" * 64,
        model_container_id="2" * 64,
        target_container_id="3" * 64,
        stopped_before_scoring=True,
        gateway_alive_before_scoring=True,
        gateway_alive_after_scoring=True,
        target_alive_before_scoring=True,
        target_alive_after_scoring=True,
        original_cleanup_called=True,
        original_cleanup_succeeded=True,
        sandbox_cleanup_observed=True,
        judgment=InspectGhcpJudgment(
            scorer_name="original_target_marker_scorer",
            source_event_id="score-event",
            raw_value=1.0,
            numeric_value=1.0,
            raw_sha256="0" * 64,
        ),
        required_gaps=("Original cyber task/scorer/target image were not independently verified.",),
        status=InspectGhcpStatus.INCOMPLETE,
    )
    score = InspectGhcpReportScorer(report_sha256=report.sha256()).prepare_unpersisted_score(report=report)
    _require_protocol_proof(report=report, score=score)
    for unsafe in (
        report.model_copy(update={"turn_count": 0}),
        report.model_copy(update={"control_receipt_sha256": None}),
        report.model_copy(update={"sandbox_cleanup_observed": False}),
        report.model_copy(update={"required_gaps": (*report.required_gaps, "Missing source bytes.")}),
    ):
        with pytest.raises(RuntimeError, match="did not prove the benign protocol"):
            _require_protocol_proof(report=unsafe, score=score)
    failed = report.model_copy(update={"control_receipt_sha256": None})
    outcome = InspectGhcpOutcome(
        report=failed,
        score=score,
        episode=MagicMock(spec=NativeCyberEpisodeSnapshot),
        log_location=None,
    )
    with (
        patch.dict(
            "os.environ",
            {
                "PYRIT_INSPECT_AGENT_IMAGE": "approved-agent",
                "PYRIT_INSPECT_AGENT_IMAGE_ID": "sha256:" + "a" * 64,
            },
        ),
        patch.object(smoke, "_docker_host", return_value="ssh://approved-child"),
        patch.object(smoke, "initialize_pyrit_async", new_callable=AsyncMock),
        patch.object(smoke, "original_benign_task", return_value=_task()),
        patch.object(smoke, "create_benign_inspect_model", return_value=MagicMock(spec=Model)),
        patch.object(smoke, "_hash_file", return_value="a" * 64),
        patch.object(smoke, "InspectGhcpEvaluation", autospec=True) as evaluation_type,
    ):
        evaluation_type.return_value.run_async.return_value = outcome
        with pytest.raises(RuntimeError, match="did not prove the benign protocol"):
            await smoke.run_async()
        evaluation_type.return_value.run_async.assert_awaited_once()
