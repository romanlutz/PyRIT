# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit and parity tests for bounded Scenario result hierarchy projections."""

import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from pyrit.analytics import compute_outcome_statistics
from pyrit.backend.main import app
from pyrit.backend.services.scenario_progress_read_model import (
    ScenarioProgressReadModel,
    ScenarioProgressSnapshot,
)
from pyrit.backend.services.scenario_run_service import ScenarioRunService
from pyrit.memory import MemoryInterface, PromptMemoryEntry, SQLiteMemory
from pyrit.memory.storage.serializers import set_message_piece_sha256_async
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    AttackResultRole,
    ConversationReference,
    ConversationType,
    MessagePiece,
    RetryEvent,
    ScenarioAtomicGroupProgress,
    ScenarioAttackResultConversations,
    ScenarioAttackResultDelta,
    ScenarioAttackTechniqueDetails,
    ScenarioConversationMessageProjection,
    ScenarioConversationProjection,
    ScenarioHierarchyGroup,
    ScenarioHierarchyNode,
    ScenarioHierarchyProjection,
    ScenarioProgressCounts,
    ScenarioProgressResult,
    ScenarioProgressSummary,
    ScenarioRunPlan,
    ScenarioRunPlanAtomicGroup,
    ScenarioRunPlanSeedGroup,
    ScenarioRunState,
)
from unit.mocks import get_mock_target_identifier, make_scenario_result


@pytest.fixture
def client(compatibility_headers: dict[str, str]) -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app, headers=compatibility_headers)


def _make_test_delta(
    *,
    attack_result_id: str,
    outcome: AttackOutcome = AttackOutcome.SUCCESS,
    role: AttackResultRole = AttackResultRole.TARGET_FACING,
    child_ids: list[str] | None = None,
    parent_collection: str = "attack_group",
    seed_group_id: str = "seed_1",
    objective: str = "Test objective",
    retries: list[RetryEvent] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    attempt_index: int | None = None,
    conversation_id: str = "conv_1",
    seconds_offset: int = 0,
) -> ScenarioAttackResultDelta:
    return ScenarioAttackResultDelta(
        attack_result_id=attack_result_id,
        conversation_id=conversation_id,
        objective=objective,
        objective_sha256="hash123",
        outcome=outcome,
        execution_time_ms=100,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds_offset),
        retry_events=retries or [],
        total_retries=len(retries or []),
        error_type=error_type,
        error_message=error_message,
        attack_metadata={
            "child_attack_result_ids": child_ids or [],
            "completion_policy": "ALL_CHILDREN",
            "executed_turns": 1,
        },
        attribution_data={
            "parent_collection": parent_collection,
            "seed_group_id": seed_group_id,
            "result_role": role.value,
            "attempt_index": attempt_index,
        },
    )


def _make_test_result(
    *,
    attack_result_id: str,
    atomic_group_id: str = "group-1",
    atomic_attack_name: str = "attack",
    seed_group_id: str = "seed-1",
    outcome: AttackOutcome = AttackOutcome.SUCCESS,
    conversation_id: str = "conv_1",
    total_retries: int = 0,
    retries: list[RetryEvent] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    child_attack_result_ids: list[str] | None = None,
    related_conversation_ids: list[str] | None = None,
    role: AttackResultRole = AttackResultRole.TARGET_FACING,
) -> ScenarioProgressResult:
    return ScenarioProgressResult(
        attack_result_id=attack_result_id,
        conversation_id=conversation_id,
        atomic_group_id=atomic_group_id,
        atomic_attack_name=atomic_attack_name,
        seed_group_id=seed_group_id,
        outcome=outcome,
        execution_time_ms=100,
        timestamp=datetime(2025, 1, 1, tzinfo=UTC),
        total_retries=total_retries,
        retries=retries or [],
        error_type=error_type,
        error_message=error_message,
        result_role=role,
        child_attack_result_ids=child_attack_result_ids or [],
        related_conversation_ids=related_conversation_ids or [],
    )


def _make_test_snapshot(
    *,
    deltas: list[ScenarioAttackResultDelta],
    plan: ScenarioRunPlan | None = None,
    atomic_groups: list[ScenarioAtomicGroupProgress] | None = None,
    summary: ScenarioProgressSummary | None = None,
) -> ScenarioProgressSnapshot:
    results: list[ScenarioProgressResult] = []
    for d in deltas:
        role_str = d.attribution_data.get("result_role", AttackResultRole.UNKNOWN.value)
        role = (
            AttackResultRole(role_str) if role_str in AttackResultRole._value2member_map_ else AttackResultRole.UNKNOWN
        )
        results.append(
            ScenarioProgressResult(
                attack_result_id=d.attack_result_id,
                conversation_id=d.conversation_id,
                atomic_group_id=str(d.attribution_data.get("parent_collection") or ""),
                atomic_attack_name=str(d.attribution_data.get("parent_collection") or ""),
                seed_group_id=str(d.attribution_data.get("seed_group_id") or ""),
                outcome=d.outcome,
                execution_time_ms=d.execution_time_ms,
                timestamp=d.timestamp,
                total_retries=d.total_retries,
                retries=d.retry_events,
                error_type=d.error_type,
                error_message=d.error_message,
                score=d.score,
                result_role=role,
                child_attack_result_ids=list(d.attack_metadata.get("child_attack_result_ids") or []),
                attempt_index=d.attribution_data.get("attempt_index"),
            )
        )
    effective_plan = plan or ScenarioRunPlan(
        atomic_groups=[
            ScenarioRunPlanAtomicGroup(
                id="attack_group",
                atomic_attack_name="attack_group",
                display_group="attack_group",
                technique_eval_hash="",
                seed_group_ids=["seed_1"],
            )
        ],
        seed_groups=[ScenarioRunPlanSeedGroup(id="seed_1", objective_sha256="hash123", objective="Test objective")],
    )
    if summary is None:
        attempts_by_unit: dict[tuple[str, str], list[ScenarioProgressResult]] = {}
        for r in results:
            unit_key = (r.atomic_group_id, r.seed_group_id)
            attempts_by_unit.setdefault(unit_key, []).append(r)

        calculated_retries = 0
        for attempts in attempts_by_unit.values():
            calculated_retries += max(len(attempts) - 1, 0) + sum(a.total_retries for a in attempts)

        effective_summary = ScenarioProgressSummary(
            overall=ScenarioProgressCounts(
                completed=len([r for r in results if r.outcome != AttackOutcome.ERROR]),
                planned=1,
                succeeded=len([r for r in results if r.outcome == AttackOutcome.SUCCESS]),
                errors=len([r for r in results if r.outcome == AttackOutcome.ERROR]),
                retries=calculated_retries,
            ),
            atomic_groups=atomic_groups or [],
        )
    else:
        effective_summary = summary
    return ScenarioProgressSnapshot(
        deltas=tuple(deltas),
        results=tuple(results),
        summary=effective_summary,
        plan=effective_plan,
    )


def test_hierarchy_ordinary_flat_results() -> None:
    delta1 = _make_test_delta(attack_result_id="attack-1", seconds_offset=1)
    delta2 = _make_test_delta(attack_result_id="attack-2", seconds_offset=2)
    snapshot = _make_test_snapshot(deltas=[delta1, delta2])

    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    assert projection.scenario_result_id == "scenario-run-1"
    assert len(projection.groups) == 1
    group = projection.groups[0]
    assert len(group.roots) == 2
    assert [r.attack_result_id for r in group.roots] == ["attack-1", "attack-2"]
    for root in group.roots:
        assert root.children == []
        assert root.unresolved_child_attack_result_ids == []
        assert root.parent_attack_result_id is None
        assert root.child_index is None
        assert root.orchestration is None


def test_hierarchy_nested_sequential_attack_with_relative_indexes() -> None:
    child1 = _make_test_delta(
        attack_result_id="child-1",
        role=AttackResultRole.TARGET_FACING,
        attempt_index=1,
        seconds_offset=1,
    )
    child2 = _make_test_delta(
        attack_result_id="child-2",
        role=AttackResultRole.TARGET_FACING,
        attempt_index=2,
        seconds_offset=2,
    )
    parent = _make_test_delta(
        attack_result_id="parent-1",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["child-1", "child-2"],
        seconds_offset=3,
    )

    snapshot = _make_test_snapshot(deltas=[child1, child2, parent])
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    assert len(projection.groups) == 1
    group = projection.groups[0]
    # Only parent is root; children are not roots
    assert len(group.roots) == 1
    root = group.roots[0]
    assert root.attack_result_id == "parent-1"
    assert root.result_role == AttackResultRole.ORCHESTRATION
    assert root.orchestration is not None
    assert root.orchestration.child_attack_result_ids == ["child-1", "child-2"]
    assert len(root.children) == 2
    assert root.children[0].attack_result_id == "child-1"
    assert root.children[0].child_index == 0
    assert root.children[0].parent_attack_result_id == "parent-1"
    assert root.children[1].attack_result_id == "child-2"
    assert root.children[1].child_index == 1
    assert root.children[1].parent_attack_result_id == "parent-1"
    assert root.unresolved_child_attack_result_ids == []


def test_hierarchy_cyclic_references_guarded() -> None:
    # A -> B and B -> A cycle
    delta_a = _make_test_delta(
        attack_result_id="node-a",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["node-b"],
        seconds_offset=1,
    )
    delta_b = _make_test_delta(
        attack_result_id="node-b",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["node-a"],
        seconds_offset=2,
    )

    snapshot = _make_test_snapshot(deltas=[delta_a, delta_b])
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    # Traversal must terminate safely without recursion error
    assert projection is not None
    assert len(projection.groups) == 1
    # Check that cycle link is placed into unresolved_child_attack_result_ids
    all_unresolved = [
        uid
        for group in projection.groups
        for root in group.roots
        for uid in (
            root.unresolved_child_attack_result_ids
            + [u for c in root.children for u in c.unresolved_child_attack_result_ids]
        )
    ]
    assert any("node-a" in u or "node-b" in u for u in all_unresolved)


def test_hierarchy_retained_failure_link_issue_3048() -> None:
    # Issue #3048: An orchestration parent encounters an error, but retains completed + failed children
    child1 = _make_test_delta(
        attack_result_id="child-1",
        outcome=AttackOutcome.SUCCESS,
        role=AttackResultRole.TARGET_FACING,
        attempt_index=1,
        seconds_offset=1,
    )
    child2 = _make_test_delta(
        attack_result_id="child-2",
        outcome=AttackOutcome.ERROR,
        role=AttackResultRole.TARGET_FACING,
        error_type="TargetError",
        error_message="Endpoint failure",
        attempt_index=2,
        seconds_offset=2,
    )
    parent = _make_test_delta(
        attack_result_id="parent-error",
        outcome=AttackOutcome.ERROR,
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["child-1", "child-2"],
        error_type="ScenarioPartialFailureException",
        error_message="Aborted after child 2 failed",
        seconds_offset=3,
    )

    snapshot = _make_test_snapshot(deltas=[child1, child2, parent])
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.FAILED,
        snapshot=snapshot,
    )

    group = projection.groups[0]
    assert len(group.roots) == 1
    root = group.roots[0]
    assert root.attack_result_id == "parent-error"
    assert root.outcome == AttackOutcome.ERROR
    assert root.error_type == "ScenarioPartialFailureException"
    assert len(root.children) == 2
    assert root.children[0].attack_result_id == "child-1"
    assert root.children[0].outcome == AttackOutcome.SUCCESS
    assert root.children[1].attack_result_id == "child-2"
    assert root.children[1].outcome == AttackOutcome.ERROR
    assert root.unresolved_child_attack_result_ids == []


def test_hierarchy_missing_and_cross_run_child_references() -> None:
    # Parent references a child ID that does not exist in deltas (or was from another run)
    parent = _make_test_delta(
        attack_result_id="parent-1",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["missing-child-99", "cross-run-child-42"],
    )
    snapshot = _make_test_snapshot(deltas=[parent])

    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    root = projection.groups[0].roots[0]
    assert root.children == []
    assert root.unresolved_child_attack_result_ids == ["missing-child-99", "cross-run-child-42"]


def test_hierarchy_diagnostics_bounding_at_100() -> None:
    # Create 105 errors and 120 retry events
    deltas: list[ScenarioAttackResultDelta] = []
    for i in range(105):
        retry = RetryEvent(exception_type="RateLimit", attempt_number=1) if i < 120 else None
        retries = [retry] if retry else []
        deltas.append(
            _make_test_delta(
                attack_result_id=f"err-{i}",
                outcome=AttackOutcome.ERROR,
                error_type="TestError",
                error_message=f"Error {i}",
                retries=retries,
                seed_group_id=f"seed_{i}",
                seconds_offset=i,
            )
        )
    deltas.extend(
        _make_test_delta(
            attack_result_id=f"retry-{i}",
            outcome=AttackOutcome.SUCCESS,
            retries=[RetryEvent(exception_type="RateLimit", attempt_number=1)],
            seed_group_id=f"seed_{i}",
            seconds_offset=i,
        )
        for i in range(105, 120)
    )

    snapshot = _make_test_snapshot(deltas=deltas)
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    diag = projection.diagnostics
    assert len(diag.failed_attacks) == 100
    assert diag.failed_attacks_count == 105
    assert diag.failed_attacks_truncated is True

    assert len(diag.attack_retries) == 100
    assert diag.attack_retries_count == 120
    assert diag.attack_retries_truncated is True
    assert diag.total_retries == 120


@pytest.mark.asyncio
async def test_service_get_run_hierarchy_node_async_bounded_read() -> None:
    memory = MagicMock(spec=MemoryInterface)
    parent_delta = _make_test_delta(
        attack_result_id="parent-1",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["child-1"],
    )
    child_delta = _make_test_delta(
        attack_result_id="child-1",
        role=AttackResultRole.TARGET_FACING,
    )

    async def mock_get_deltas(
        *, scenario_result_id: str, attack_result_ids: list[str]
    ) -> list[ScenarioAttackResultDelta]:
        res = []
        for aid in attack_result_ids:
            if aid == "parent-1":
                res.append(parent_delta)
            elif aid == "child-1":
                res.append(child_delta)
        return res

    memory.get_scenario_attack_result_deltas_by_ids_async = AsyncMock(side_effect=mock_get_deltas)
    memory.get_scenario_result_header_async = AsyncMock(return_value=None)
    memory.get_attack_result_conversation_projections_async = AsyncMock(
        return_value=ScenarioAttackResultConversations(
            conversations=[
                ScenarioConversationProjection(
                    conversation_id="conv_1",
                    conversation_type="main",
                    message_count=1,
                    messages=[
                        ScenarioConversationMessageProjection(
                            role="user",
                            content="Test prompt",
                            sequence=1,
                        )
                    ],
                )
            ],
            total_count=1,
            truncated=False,
        )
    )

    read_model = ScenarioProgressReadModel(memory=memory)
    service = ScenarioRunService(memory=memory, progress_read_model=read_model)

    node = await service.get_run_hierarchy_node_async(
        scenario_result_id="scenario-run-1",
        attack_result_id="parent-1",
    )

    assert node is not None
    assert node.attack_result_id == "parent-1"
    assert len(node.children) == 1
    assert node.children[0].attack_result_id == "child-1"
    assert len(node.conversations) == 1
    assert len(node.conversations[0].messages) == 1
    assert node.conversations[0].messages[0].content == "Test prompt"

    # Querying non-existent node returns None
    missing = await service.get_run_hierarchy_node_async(
        scenario_result_id="scenario-run-1",
        attack_result_id="non-existent",
    )
    assert missing is None


def test_rest_routes_hierarchy_and_node(client: TestClient) -> None:
    now = datetime.now(UTC)
    node = ScenarioHierarchyNode(
        attack_result_id="node-1",
        result_role=AttackResultRole.TARGET_FACING,
        atomic_attack_name="attack",
        atomic_group_id="group-1",
        seed_group_id="seed-1",
        objective="obj",
        outcome=AttackOutcome.SUCCESS,
        timestamp=now,
    )
    group = ScenarioHierarchyGroup(
        atomic_group_id="group-1",
        atomic_attack_name="attack",
        display_group="display",
        technique=ScenarioAttackTechniqueDetails(component_name="Technique"),
        roots=[node],
        counts=ScenarioProgressCounts(completed=1, planned=1, succeeded=1, success_percentage=100, errors=0, retries=0),
    )
    projection = ScenarioHierarchyProjection(
        scenario_result_id="run-123",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        groups=[group],
        overall_counts=ScenarioProgressCounts(
            completed=1, planned=1, succeeded=1, success_percentage=100, errors=0, retries=0
        ),
    )

    with patch("pyrit.backend.routes.scenarios.get_scenario_run_service") as mock_get_svc:
        mock_svc = MagicMock()
        mock_svc.get_run_hierarchy_async = AsyncMock(return_value=projection)
        mock_svc.get_run_hierarchy_node_async = AsyncMock(return_value=node)
        mock_get_svc.return_value = mock_svc

        # 1. Hierarchy projection endpoint
        resp = client.get("/api/scenarios/runs/run-123/hierarchy")
        assert resp.status_code == status.HTTP_200_OK
        data = resp.json()
        assert data["scenario_result_id"] == "run-123"
        assert len(data["groups"]) == 1
        assert data["groups"][0]["roots"][0]["attack_result_id"] == "node-1"

        # 2. Node detail endpoint
        resp_node = client.get("/api/scenarios/runs/run-123/hierarchy/nodes/node-1")
        assert resp_node.status_code == status.HTTP_200_OK
        node_data = resp_node.json()
        assert node_data["attack_result_id"] == "node-1"

        # 3. 404 on unknown run
        mock_svc.get_run_hierarchy_async.return_value = None
        resp_404 = client.get("/api/scenarios/runs/unknown-run/hierarchy")
        assert resp_404.status_code == status.HTTP_404_NOT_FOUND

        # 4. 404 on unknown node
        mock_svc.get_run_hierarchy_node_async.return_value = None
        resp_node_404 = client.get("/api/scenarios/runs/run-123/hierarchy/nodes/unknown-node")
        assert resp_node_404.status_code == status.HTTP_404_NOT_FOUND


@pytest.mark.asyncio
async def test_sqlite_memory_batched_reads_and_projections_parity(sqlite_instance: SQLiteMemory) -> None:
    """Test batched memory delta reads, conversation projections, and cross-run isolation on SQLite."""
    scenario_uuid_1 = uuid.uuid4()
    scenario_uuid_2 = uuid.uuid4()
    scenario_id_1 = str(scenario_uuid_1)
    scenario_id_2 = str(scenario_uuid_2)

    scen_1 = make_scenario_result(
        id=scenario_uuid_1,
        scenario_name="Scenario1",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    scen_2 = make_scenario_result(
        id=scenario_uuid_2,
        scenario_name="Scenario2",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen_1, scen_2])

    parent_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    cross_run_id = str(uuid.uuid4())
    child_conv_id = str(uuid.uuid4())

    parent_result = AttackResult(
        attack_result_id=parent_id,
        conversation_id="parent_conv",
        objective="Orchestration objective",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id_1,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.ORCHESTRATION.value},
        metadata={"child_attack_result_ids": [child_id, cross_run_id]},
    )
    child_result = AttackResult(
        attack_result_id=child_id,
        conversation_id=child_conv_id,
        objective="Child target-facing objective",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id_1,
        attribution_data={
            "parent_collection": "group_1",
            "result_role": AttackResultRole.TARGET_FACING.value,
            "attempt_index": 1,
        },
    )
    cross_run_result = AttackResult(
        attack_result_id=cross_run_id,
        conversation_id="cross_conv",
        objective="Cross-run child objective",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id_2,
        attribution_data={"parent_collection": "group_2", "result_role": AttackResultRole.TARGET_FACING.value},
    )

    await sqlite_instance.add_attack_results_to_memory_async(
        attack_results=[parent_result, child_result, cross_run_result]
    )

    # Insert messages into child conversation
    piece_user = MessagePiece(
        id=uuid.uuid4(),
        conversation_id=child_conv_id,
        role="user",
        original_value_data_type="text",
        original_value="Prompt to target",
        converted_value="Prompt to target",
        sequence=0,
    )
    piece_assistant = MessagePiece(
        id=uuid.uuid4(),
        conversation_id=child_conv_id,
        role="assistant",
        original_value_data_type="text",
        original_value="Response from target",
        converted_value="Response from target",
        sequence=1,
    )
    await set_message_piece_sha256_async(piece_user)
    await set_message_piece_sha256_async(piece_assistant)
    await sqlite_instance._run_database_operation_async(
        sqlite_instance._insert_entry, PromptMemoryEntry(entry=piece_user)
    )
    await sqlite_instance._run_database_operation_async(
        sqlite_instance._insert_entry, PromptMemoryEntry(entry=piece_assistant)
    )

    # 1. Test batched delta reads
    deltas = await sqlite_instance.get_scenario_attack_result_deltas_by_ids_async(
        scenario_result_id=scenario_id_1,
        attack_result_ids=[parent_id, child_id, cross_run_id],
    )
    delta_ids = [d.attack_result_id for d in deltas]
    assert parent_id in delta_ids
    assert child_id in delta_ids
    # Cross-run ID belongs to scenario_id_2, so it MUST NOT be returned!
    assert cross_run_id not in delta_ids

    # 2. Test conversation projections
    conv_projections = await sqlite_instance.get_attack_result_conversation_projections_async(
        attack_result_id=child_id,
        include_messages=True,
    )
    assert conv_projections.total_count == 1
    assert conv_projections.truncated is False
    assert len(conv_projections.conversations) == 1
    proj = conv_projections.conversations[0]
    assert proj.conversation_id == child_conv_id
    assert proj.message_count == 2
    assert len(proj.messages) == 2
    assert proj.messages[0].role == "user"
    assert proj.messages[0].content == "Prompt to target"
    assert proj.messages[1].role == "assistant"
    assert proj.messages[1].content == "Response from target"

    # 3. Test service single-node read
    service = ScenarioRunService(memory=sqlite_instance)
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id_1,
        attack_result_id=parent_id,
    )
    assert node is not None
    assert node.attack_result_id == parent_id
    assert len(node.children) == 1
    assert node.children[0].attack_result_id == child_id
    # Cross-run child is safely recognized as unresolved without failing the run
    assert node.unresolved_child_attack_result_ids == [cross_run_id]


@pytest.mark.asyncio
async def test_sqlite_canonical_columns_round_trip(sqlite_instance: SQLiteMemory) -> None:
    """Test real SQLite round-trip of canonical executed_turns and outcome_reason columns."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="CanonicalScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id="conv_canonical",
        objective="Probe objective",
        outcome=AttackOutcome.FAILURE,
        executed_turns=7,
        outcome_reason="Target actively refused harmful request",
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        metadata={},  # Explicitly empty metadata to ensure canonical DB columns are read
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    # 1. Delta query directly from SQLite selects canonical columns
    deltas = await sqlite_instance.get_scenario_attack_result_deltas_by_ids_async(
        scenario_result_id=scenario_id,
        attack_result_ids=[ar_id],
    )
    assert len(deltas) == 1
    delta = deltas[0]
    assert delta.executed_turns == 7
    assert delta.outcome_reason == "Target actively refused harmful request"
    assert "executed_turns" not in delta.attack_metadata
    assert "outcome_reason" not in delta.attack_metadata

    # 2. Hierarchy node query through service projects canonical values
    service = ScenarioRunService(memory=sqlite_instance)
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=ar_id,
    )
    assert node is not None
    assert node.executed_turns == 7
    assert node.outcome_reason == "Target actively refused harmful request"


def test_hierarchy_disconnected_cyclic_component_mixed() -> None:
    """Test that disconnected cyclic components (e.g. A <-> B) are not dropped when standard roots exist."""
    # 1. Ordinary tree: root-1 -> child-1
    root1 = _make_test_delta(
        attack_result_id="root-1",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["child-1"],
        seconds_offset=1,
    )
    child1 = _make_test_delta(
        attack_result_id="child-1",
        role=AttackResultRole.TARGET_FACING,
        seconds_offset=2,
    )

    # 2. Detached cyclic component: cycle-a -> cycle-b and cycle-b -> cycle-a
    # Neither cycle-a nor cycle-b is reachable from root-1, and both are in each other's child_ids
    cycle_a = _make_test_delta(
        attack_result_id="cycle-a",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["cycle-b"],
        seconds_offset=3,
    )
    cycle_b = _make_test_delta(
        attack_result_id="cycle-b",
        role=AttackResultRole.ORCHESTRATION,
        child_ids=["cycle-a"],
        seconds_offset=4,
    )

    snapshot = _make_test_snapshot(deltas=[root1, child1, cycle_a, cycle_b])
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    # All 4 nodes must be accounted for in the hierarchy!
    all_node_ids = set()

    def _collect_ids(nodes: list[ScenarioHierarchyNode]) -> None:
        for n in nodes:
            all_node_ids.add(n.attack_result_id)
            _collect_ids(n.children)

    for g in projection.groups:
        _collect_ids(g.roots)

    assert "root-1" in all_node_ids
    assert "child-1" in all_node_ids
    assert "cycle-a" in all_node_ids
    assert "cycle-b" in all_node_ids
    assert len(all_node_ids) == 4

    # Verify root-1 has child-1
    group = projection.groups[0]
    root_node = next(r for r in group.roots if r.attack_result_id == "root-1")
    assert len(root_node.children) == 1
    assert root_node.children[0].attack_result_id == "child-1"

    # Verify cycle was guarded and placed in unresolved_child_attack_result_ids without infinite loop
    cycle_root = next(r for r in group.roots if r.attack_result_id in ("cycle-a", "cycle-b"))
    assert len(cycle_root.children) == 1
    cycle_child = cycle_root.children[0]
    assert cycle_root.attack_result_id in cycle_child.unresolved_child_attack_result_ids


def test_hierarchy_bounded_node_budget() -> None:
    """Test that max_nodes bounds tree traversal and sets truncated=True while preserving exact overall counts."""
    # Create 6 deltas in a chain: node-0 -> node-1 -> node-2 -> node-3 -> node-4 -> node-5
    deltas = [
        _make_test_delta(
            attack_result_id=f"node-{i}",
            role=AttackResultRole.ORCHESTRATION if i < 5 else AttackResultRole.TARGET_FACING,
            child_ids=[f"node-{i + 1}"] if i < 5 else [],
            seconds_offset=i,
        )
        for i in range(6)
    ]

    snapshot = _make_test_snapshot(deltas=deltas)
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
        max_nodes=3,
    )

    assert projection.truncated is True

    # Count nodes projected in hierarchy
    node_count = 0

    def _count_nodes(nodes: list[ScenarioHierarchyNode]) -> None:
        nonlocal node_count
        for n in nodes:
            node_count += 1
            _count_nodes(n.children)

    for g in projection.groups:
        _count_nodes(g.roots)

    assert node_count == 3
    # Top-level aggregate totals must remain exact (not clipped)
    assert projection.overall_counts.completed == 6


def test_hierarchy_group_progress_outcomes_parity() -> None:
    """Test that group.counts.outcomes is preserved from atomic group progress."""
    outcomes = compute_outcome_statistics({AttackOutcome.SUCCESS: 1, AttackOutcome.UNDETERMINED: 1})
    ag_progress = ScenarioAtomicGroupProgress(
        id="attack_group",
        atomic_attack_name="attack_group",
        display_group="attack_group",
        status="COMPLETED",
        completed=2,
        planned=2,
        succeeded=1,
        success_percentage=50,
        errors=0,
        retries=0,
        outcomes=outcomes,
    )

    delta1 = _make_test_delta(attack_result_id="a-1", outcome=AttackOutcome.SUCCESS, seconds_offset=1)
    delta2 = _make_test_delta(attack_result_id="a-2", outcome=AttackOutcome.UNDETERMINED, seconds_offset=2)

    plan = ScenarioRunPlan(
        atomic_groups=[
            ScenarioRunPlanAtomicGroup(
                id="attack_group",
                atomic_attack_name="attack_group",
                display_group="attack_group",
                technique_eval_hash="",
                seed_group_ids=["seed_1"],
            )
        ],
        seed_groups=[ScenarioRunPlanSeedGroup(id="seed_1", objective_sha256="hash123", objective="Test objective")],
    )

    summary = ScenarioProgressSummary(
        overall=ScenarioProgressCounts(
            completed=2,
            planned=2,
            succeeded=1,
            success_percentage=50,
            errors=0,
            retries=0,
            outcomes=outcomes,
        ),
        atomic_groups=[ag_progress],
    )

    snapshot = ScenarioProgressSnapshot(
        deltas=(delta1, delta2),
        results=tuple(
            ScenarioProgressResult(
                attack_result_id=d.attack_result_id,
                conversation_id=d.conversation_id,
                atomic_group_id="attack_group",
                atomic_attack_name="attack_group",
                seed_group_id="seed_1",
                outcome=d.outcome,
                execution_time_ms=d.execution_time_ms,
                timestamp=d.timestamp,
                total_retries=d.total_retries,
                result_role=AttackResultRole.TARGET_FACING,
            )
            for d in [delta1, delta2]
        ),
        summary=summary,
        plan=plan,
    )

    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    assert len(projection.groups) == 1
    group = projection.groups[0]
    assert group.counts.outcomes is not None
    assert group.counts.outcomes.success_rate_decided == 1.0
    assert group.counts.outcomes.success_rate_all == 0.5


def test_hierarchy_unplanned_groups_shared_counting() -> None:
    """Test that unplanned groups use shared counting policy (latest attempt resolution + valid producer counts)."""
    # 2 attempts on the same unit in an unplanned group:
    # Attempt 1 failed, Attempt 2 succeeded
    delta1 = _make_test_delta(
        attack_result_id="unplanned-1",
        outcome=AttackOutcome.ERROR,
        role=AttackResultRole.TARGET_FACING,
        parent_collection="unplanned_group",
        seed_group_id="seed_u",
        attempt_index=1,
        seconds_offset=1,
    )
    delta2 = _make_test_delta(
        attack_result_id="unplanned-2",
        outcome=AttackOutcome.SUCCESS,
        role=AttackResultRole.TARGET_FACING,
        parent_collection="unplanned_group",
        seed_group_id="seed_u",
        attempt_index=2,
        seconds_offset=2,
    )

    # Empty plan (no planned atomic groups)
    plan = ScenarioRunPlan(atomic_groups=[], seed_groups=[])
    snapshot = _make_test_snapshot(deltas=[delta1, delta2], plan=plan)

    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    assert len(projection.groups) == 1
    group = projection.groups[0]
    assert group.atomic_group_id == "unplanned_group"
    # Latest attempt succeeded, so 1 completed unit, 1 success, 1 historical error
    assert group.counts.completed == 1
    assert group.counts.succeeded == 1
    assert group.counts.errors == 1
    # Producer counts include both historical attempts
    assert group.counts.producer_counts.target_facing.attempts == 2
    assert group.counts.producer_counts.target_facing.errors == 1
    assert group.counts.producer_counts.target_facing.retries == 0


def test_hierarchy_diagnostics_cap_retries_per_item() -> None:
    """Test that retries list inside individual AttackRetrySummary is bounded to 100 with retries_truncated=True."""
    retries = [RetryEvent(exception_type="RateLimitError", attempt_number=i) for i in range(250)]
    delta = _make_test_delta(
        attack_result_id="heavy-retry",
        outcome=AttackOutcome.SUCCESS,
        retries=retries,
    )

    snapshot = _make_test_snapshot(deltas=[delta])
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    diag = projection.diagnostics
    assert len(diag.attack_retries) == 1
    item = diag.attack_retries[0]
    assert item.attack_result_id == "heavy-retry"
    assert item.total_retries == 250
    assert len(item.retries) == 100
    assert item.retries_truncated is True


@pytest.mark.asyncio
async def test_sqlite_bounded_multimodal_conversation_projection(sqlite_instance: SQLiteMemory, tmp_path: Path) -> None:
    """Test multimodal piece projection, raw role preservation, and database-level SQL LIMIT bounding."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="MultimodalScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    conv_id = str(uuid.uuid4())
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=conv_id,
        objective="Multimodal test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    # Create real image on disk
    img_file = tmp_path / "diagram.png"
    img_file.write_bytes(b"dummy_png_bytes")
    img_path = str(img_file)

    # Sequence 0: 2 pieces (text + image)
    p0_text = MessagePiece(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        role="user",
        original_value_data_type="text",
        original_value="Analyze blueprint image",
        converted_value="Analyze blueprint image",
        sequence=0,
    )
    p0_img = MessagePiece(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        role="user",
        original_value_data_type="image_path",
        original_value=img_path,
        converted_value=img_path,
        sequence=0,
    )
    # Sequence 1: simulated_assistant role
    p1 = MessagePiece(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        role="simulated_assistant",
        original_value_data_type="text",
        original_value="I detect architectural blueprints.",
        converted_value="I detect architectural blueprints.",
        sequence=1,
    )
    # Sequence 2: 3rd message
    p2 = MessagePiece(
        id=uuid.uuid4(),
        conversation_id=conv_id,
        role="user",
        original_value_data_type="text",
        original_value="Proceed with analysis.",
        converted_value="Proceed with analysis.",
        sequence=2,
    )

    for p in [p0_text, p0_img, p1, p2]:
        await set_message_piece_sha256_async(p)
        await sqlite_instance._run_database_operation_async(sqlite_instance._insert_entry, PromptMemoryEntry(entry=p))

    # Read with limit=2
    projs = await sqlite_instance.get_attack_result_conversation_projections_async(
        attack_result_id=ar_id,
        include_messages=True,
        limit_messages_per_conversation=2,
    )

    assert projs.total_count == 1
    assert projs.truncated is False
    assert len(projs.conversations) == 1
    proj = projs.conversations[0]
    assert proj.conversation_id == conv_id
    assert proj.message_count == 3  # Exact count from COUNT query
    assert proj.messages_truncated is True
    assert len(proj.messages) == 2  # Bounded by SQL LIMIT

    # Verify multimodal pieces in message 0
    msg0 = proj.messages[0]
    assert msg0.role == "user"
    assert len(msg0.pieces) == 2
    assert msg0.pieces[0].data_type == "text"
    assert msg0.pieces[0].content == "Analyze blueprint image"
    assert msg0.pieces[1].data_type == "image_path"
    assert msg0.pieces[1].content == img_path
    assert msg0.pieces[1].converted_value == img_path

    # Verify raw custom role in message 1
    msg1 = proj.messages[1]
    assert msg1.role == "simulated_assistant"
    assert msg1.content == "I detect architectural blueprints."


def test_hierarchy_repeated_attempt_diagnostic_retries_consistency() -> None:
    """Test that units with repeated attempts without row retries match overall_counts and diagnostics retries."""
    # 2 attempts on the same unit: attempt 1 error, attempt 2 success, both with total_retries=0
    delta1 = _make_test_delta(
        attack_result_id="attempt-1",
        outcome=AttackOutcome.ERROR,
        role=AttackResultRole.TARGET_FACING,
        parent_collection="group_1",
        seed_group_id="seed_1",
        attempt_index=1,
        seconds_offset=1,
    )
    delta2 = _make_test_delta(
        attack_result_id="attempt-2",
        outcome=AttackOutcome.SUCCESS,
        role=AttackResultRole.TARGET_FACING,
        parent_collection="group_1",
        seed_group_id="seed_1",
        attempt_index=2,
        seconds_offset=2,
    )

    snapshot = _make_test_snapshot(deltas=[delta1, delta2])
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    # Re-attempt of same unit causes retry pressure of 1
    assert projection.overall_counts.retries == 1
    # Diagnostics total_retries must be consistent with overall_counts.retries
    assert projection.diagnostics.total_retries == 1
    assert projection.diagnostics.total_retries == projection.overall_counts.retries


def test_hierarchy_traversal_halts_immediately_on_budget_exhaustion() -> None:
    """Test that with 1,000 root results and max_nodes=1, traversal stops immediately without quadratic loop."""
    deltas = [
        _make_test_delta(
            attack_result_id=f"root-{i}",
            role=AttackResultRole.TARGET_FACING,
            parent_collection="group_1",
            seed_group_id=f"seed_{i}",
            seconds_offset=i,
        )
        for i in range(1000)
    ]

    snapshot = _make_test_snapshot(deltas=deltas)

    import time

    start_time = time.perf_counter()
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
        max_nodes=1,
    )
    elapsed = time.perf_counter() - start_time

    assert elapsed < 0.05, f"Expected <0.05s, got {elapsed:.4f}s"
    assert projection.truncated is True
    # Count total nodes in projection
    node_count = sum(len(g.roots) for g in projection.groups)
    assert node_count == 1
    # Aggregate total remains exact
    assert projection.overall_counts.completed == 1000


@pytest.mark.asyncio
async def test_sqlite_conversation_list_clipping_explicit(sqlite_instance: SQLiteMemory) -> None:
    """Test explicit conversation-list clipping flags and exact count with 55 conversations."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="ConversationClippingScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    main_conv_id = str(uuid.uuid4())
    # 54 preparation conversations + 1 main conversation = 55 total conversations
    prep_conv_ids = [str(uuid.uuid4()) for _ in range(54)]

    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=main_conv_id,
        related_conversations={
            ConversationReference(conversation_id=cid, conversation_type=ConversationType.PREPARATION)
            for cid in prep_conv_ids
        },
        objective="Conversation clipping test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    # 1. Direct memory read returns ScenarioAttackResultConversations with total_count=55 and truncated=True
    conv_result = await sqlite_instance.get_attack_result_conversation_projections_async(
        attack_result_id=ar_id,
        max_conversations=50,
    )
    assert conv_result.total_count == 55
    assert conv_result.truncated is True
    assert len(conv_result.conversations) == 50

    # 2. Service single-node read projects exact count and truncation flag on node
    service = ScenarioRunService(memory=sqlite_instance)
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=ar_id,
    )
    assert node is not None
    assert node.conversations_count == 55
    assert node.conversations_truncated is True
    assert len(node.conversations) == 50


async def test_executed_turns_zero_vs_invalid_metadata(sqlite_instance: SQLiteMemory) -> None:
    """Test that executed_turns=0 with invalid string in metadata safely projects 0 without ValueError."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="ExecutedTurnsZeroScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=str(uuid.uuid4()),
        objective="Executed turns zero test",
        outcome=AttackOutcome.SUCCESS,
        executed_turns=0,
        metadata={"executed_turns": "invalid"},
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    service = ScenarioRunService(memory=sqlite_instance)
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=ar_id,
    )
    assert node is not None
    assert node.executed_turns == 0

    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    assert len(hierarchy.groups) == 1
    assert hierarchy.groups[0].roots[0].executed_turns == 0


def test_eliminate_tuple_duck_typing_raw_tuples() -> None:
    """Test passing 2, 3, and 4 conversation projections as a raw Python tuple preserves all items."""
    delta = _make_test_delta(
        attack_result_id="test-node",
        outcome=AttackOutcome.SUCCESS,
    )
    result = _make_test_result(
        attack_result_id="test-node",
        atomic_group_id="group-1",
        atomic_attack_name="attack",
        seed_group_id="seed-1",
        outcome=AttackOutcome.SUCCESS,
    )
    conv1 = ScenarioConversationProjection(conversation_id="conv-1")
    conv2 = ScenarioConversationProjection(conversation_id="conv-2")
    conv3 = ScenarioConversationProjection(conversation_id="conv-3")
    conv4 = ScenarioConversationProjection(conversation_id="conv-4")

    # 2-tuple
    node2 = ScenarioProgressReadModel.build_hierarchy_node(
        delta=delta,
        result=result,
        conversations=(conv1, conv2),
    )
    assert len(node2.conversations) == 2
    assert node2.conversations == [conv1, conv2]
    assert node2.conversations_count == 2
    assert node2.conversations_truncated is False

    # 3-tuple (specifically targeted in Roman's review comment 14)
    node3 = ScenarioProgressReadModel.build_hierarchy_node(
        delta=delta,
        result=result,
        conversations=(conv1, conv2, conv3),
    )
    assert len(node3.conversations) == 3
    assert node3.conversations == [conv1, conv2, conv3]
    assert node3.conversations_count == 3
    assert node3.conversations_truncated is False

    # 4-tuple
    node4 = ScenarioProgressReadModel.build_hierarchy_node(
        delta=delta,
        result=result,
        conversations=(conv1, conv2, conv3, conv4),
    )
    assert len(node4.conversations) == 4
    assert node4.conversations == [conv1, conv2, conv3, conv4]
    assert node4.conversations_count == 4
    assert node4.conversations_truncated is False


async def test_sqlite_message_pieces_bounding(sqlite_instance: SQLiteMemory) -> None:
    """Test persisting 1 message sequence with 150 pieces and verifying piece truncation at max_pieces_per_message."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="MessagePiecesBoundingScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    conv_id = str(uuid.uuid4())

    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=conv_id,
        objective="Message pieces bounding test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    # Insert 150 pieces for sequence 0
    now = datetime.now(UTC)
    for i in range(150):
        piece = MessagePiece(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            role="user",
            original_value_data_type="text",
            original_value=f"Piece content {i}",
            converted_value=f"Piece content {i}",
            sequence=0,
            timestamp=now + timedelta(milliseconds=i),
        )
        await set_message_piece_sha256_async(piece)
        await sqlite_instance._run_database_operation_async(
            sqlite_instance._insert_entry, PromptMemoryEntry(entry=piece)
        )

    # 1. Fetch with default max_pieces_per_message=100
    projs = await sqlite_instance.get_attack_result_conversation_projections_async(
        attack_result_id=ar_id,
        include_messages=True,
        max_pieces_per_message=100,
    )
    assert len(projs.conversations) == 1
    assert len(projs.conversations[0].messages) == 1
    msg = projs.conversations[0].messages[0]
    assert len(msg.pieces) == 100
    assert msg.pieces_truncated is True

    # 2. Fetch with higher limit max_pieces_per_message=200
    projs_untruncated = await sqlite_instance.get_attack_result_conversation_projections_async(
        attack_result_id=ar_id,
        include_messages=True,
        max_pieces_per_message=200,
    )
    assert len(projs_untruncated.conversations[0].messages) == 1
    msg_untruncated = projs_untruncated.conversations[0].messages[0]
    assert len(msg_untruncated.pieces) == 150
    assert msg_untruncated.pieces_truncated is False


async def test_unloaded_conversations_in_full_hierarchy(sqlite_instance: SQLiteMemory) -> None:
    """Test that full hierarchy marks nodes with unhydrated related conversations as truncated with count=None."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="UnloadedConversationsScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    main_conv_id = str(uuid.uuid4())
    prep_conv_ids = [str(uuid.uuid4()) for _ in range(55)]

    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=main_conv_id,
        related_conversations={
            ConversationReference(conversation_id=cid, conversation_type=ConversationType.PREPARATION)
            for cid in prep_conv_ids
        },
        objective="Unloaded conversations test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    service = ScenarioRunService(memory=sqlite_instance)
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    assert len(hierarchy.groups) == 1
    root_node = hierarchy.groups[0].roots[0]
    assert root_node.attack_result_id == ar_id
    assert root_node.conversations_count is None
    assert root_node.conversations_truncated is True
    assert len(root_node.conversations) == 1


def test_diagnostic_total_retries_scope_unmatched_attempt() -> None:
    """Test that diagnostics.total_retries reconciles with unmatched attempts having retry events."""
    plan = ScenarioRunPlan(
        atomic_groups=[
            ScenarioRunPlanAtomicGroup(
                id="group-1",
                atomic_attack_name="planned_attack",
                display_group="Group 1",
                technique_eval_hash="",
                seed_group_ids=["seed-1"],
            )
        ],
        seed_groups=[ScenarioRunPlanSeedGroup(id="seed-1", objective_sha256="hash123", objective="Planned objective")],
    )

    planned_delta = _make_test_delta(
        attack_result_id="att-planned",
        outcome=AttackOutcome.SUCCESS,
        parent_collection="planned_attack",
        seed_group_id="seed-1",
    )
    planned_result = _make_test_result(
        attack_result_id="att-planned",
        atomic_group_id="group-1",
        atomic_attack_name="planned_attack",
        seed_group_id="seed-1",
        outcome=AttackOutcome.SUCCESS,
        total_retries=0,
        retries=[],
    )

    # Unmatched error attempt with 3 retry events
    retry_events = [
        RetryEvent(exception_type="RateLimitError", attempt_number=1),
        RetryEvent(exception_type="RateLimitError", attempt_number=2),
        RetryEvent(exception_type="TimeoutError", attempt_number=3),
    ]
    unmatched_delta = _make_test_delta(
        attack_result_id="att-unmatched",
        outcome=AttackOutcome.ERROR,
        retries=retry_events,
        error_type="TimeoutError",
        error_message="Timed out",
        parent_collection="unmatched_attack",
        seed_group_id="unmatched-seed",
    )
    unmatched_result = _make_test_result(
        attack_result_id="att-unmatched",
        atomic_group_id="unmatched-group",
        atomic_attack_name="unmatched_attack",
        seed_group_id="unmatched-seed",
        outcome=AttackOutcome.ERROR,
        total_retries=3,
        retries=retry_events,
    )

    summary = ScenarioProgressSummary(
        overall=ScenarioProgressCounts(
            completed=1,
            planned=1,
            succeeded=1,
            success_percentage=100,
            errors=0,
            retries=0,  # Canonical planned summary had 0 retries
        ),
        atomic_groups=[],
    )

    snapshot = ScenarioProgressSnapshot(
        deltas=(planned_delta, unmatched_delta),
        results=(planned_result, unmatched_result),
        summary=summary,
        plan=plan,
    )

    hierarchy = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scen-1",
        scenario_name="RetryScopeScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
    )

    assert hierarchy.diagnostics.total_retries == 3
    assert hierarchy.diagnostics.attack_retries_count == 1
    assert len(hierarchy.diagnostics.attack_retries) == 1
    assert hierarchy.diagnostics.attack_retries[0].attack_result_id == "att-unmatched"
    assert hierarchy.diagnostics.attack_retries[0].total_retries == 3


async def test_immediate_children_bounding_in_run_service(sqlite_instance: SQLiteMemory) -> None:
    """
    Test that get_run_hierarchy_node_async bounds immediate children to
    max_immediate_children with children_truncated=True.
    """
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="ImmediateChildrenBoundingScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    # 250 child attack results
    child_ids = [str(uuid.uuid4()) for _ in range(250)]
    parent_id = str(uuid.uuid4())

    parent_ar = AttackResult(
        attack_result_id=parent_id,
        conversation_id=str(uuid.uuid4()),
        objective="Orchestration parent with 250 children",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.ORCHESTRATION.value},
        metadata={"child_attack_result_ids": child_ids},
    )
    child_ars = [
        AttackResult(
            attack_result_id=cid,
            conversation_id=str(uuid.uuid4()),
            objective=f"Child {i}",
            outcome=AttackOutcome.SUCCESS,
            attribution_parent_id=scenario_id,
            attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        )
        for i, cid in enumerate(child_ids)
    ]

    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[parent_ar] + child_ars)

    service = ScenarioRunService(memory=sqlite_instance)
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=parent_id,
        max_immediate_children=200,
    )

    assert node is not None
    assert node.children_truncated is True
    assert len(node.children) == 200


async def test_persisted_result_with_malformed_related_conversations_metadata(
    sqlite_instance: SQLiteMemory,
) -> None:
    """Test that malformed related_conversation_ids metadata does not cause TypeError in progress or node reader."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="MalformedRelatedConvScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=str(uuid.uuid4()),
        objective="Malformed related_conversation_ids test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        metadata={"related_conversation_ids": 7},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    service = ScenarioRunService(memory=sqlite_instance)

    # 1. get_run_progress_async should succeed without raising TypeError
    progress = await service.get_run_progress_async(scenario_result_id=scenario_id, since=None, limit=25)
    assert progress is not None
    assert len(progress.results) == 1
    assert progress.results[0].related_conversation_ids == []

    # 2. get_run_hierarchy_node_async should succeed without raising TypeError
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=ar_id,
    )
    assert node is not None
    assert node.conversations_count == 1
    assert node.conversations_truncated is False
    assert len(node.conversations) == 1

    # 3. get_run_hierarchy_async should also succeed without raising TypeError
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    assert len(hierarchy.groups) == 1
    root_node = hierarchy.groups[0].roots[0]
    assert root_node.conversations_count == 1
    assert root_node.conversations_truncated is False


def test_read_related_conversation_ids_shapes() -> None:
    """Test _read_related_conversation_ids with typed list, legacy valid list, and malformed fallbacks."""
    # Typed list
    delta1 = _make_test_delta(attack_result_id="d1")
    delta1.related_conversation_ids = ["conv-1", "conv-2"]
    assert ScenarioProgressReadModel._read_related_conversation_ids(delta=delta1) == ["conv-1", "conv-2"]

    # Legacy valid metadata list
    delta2 = _make_test_delta(attack_result_id="d2")
    delta2.related_conversation_ids = []
    delta2.attack_metadata = {"related_conversation_ids": ["conv-3", "conv-4"]}
    assert ScenarioProgressReadModel._read_related_conversation_ids(delta=delta2) == ["conv-3", "conv-4"]

    # Legacy metadata with non-string elements filtered out
    delta3 = _make_test_delta(attack_result_id="d3")
    delta3.related_conversation_ids = []
    delta3.attack_metadata = {"related_conversation_ids": ["conv-5", 42, None, "conv-6"]}
    assert ScenarioProgressReadModel._read_related_conversation_ids(delta=delta3) == ["conv-5", "conv-6"]

    # Typed list with non-string elements filtered
    delta4 = _make_test_delta(attack_result_id="d4")
    delta4.related_conversation_ids = ["conv-7", 100, None]  # type: ignore[list-item]
    assert ScenarioProgressReadModel._read_related_conversation_ids(delta=delta4) == ["conv-7"]

    # Malformed typed non-list fallbacks safely
    delta5 = _make_test_delta(attack_result_id="d5")
    delta5.related_conversation_ids = 123  # type: ignore[assignment]
    delta5.attack_metadata = {"related_conversation_ids": ["conv-fallback"]}
    assert ScenarioProgressReadModel._read_related_conversation_ids(delta=delta5) == ["conv-fallback"]

    # Malformed metadata shapes (int, str, dict) return empty list
    for malformed in [7, "conv-single", {"id": "conv-1"}, None, True]:
        delta_bad = _make_test_delta(attack_result_id="dbad")
        delta_bad.related_conversation_ids = []
        delta_bad.attack_metadata = {"related_conversation_ids": malformed}
        assert ScenarioProgressReadModel._read_related_conversation_ids(delta=delta_bad) == []


async def test_invalid_scenario_id_returns_empty_deltas(sqlite_instance: SQLiteMemory) -> None:
    """Test that non-UUID scenario_result_id returns empty list without raising ValueError."""
    deltas = await sqlite_instance.get_scenario_attack_result_deltas_by_ids_async(
        scenario_result_id="not-a-valid-uuid",
        attack_result_ids=["bad-id"],
    )
    assert deltas == []


def test_build_hierarchy_node_deduplicates_duplicate_child_ids() -> None:
    """Test that duplicate child IDs in stored order are deduplicated and tracked as unresolved."""
    delta = _make_test_delta(attack_result_id="parent")
    result = _make_test_result(
        attack_result_id="parent",
        child_attack_result_ids=["child-1", "child-1"],
    )
    c1_delta = _make_test_delta(attack_result_id="child-1")
    c1_result = _make_test_result(attack_result_id="child-1")

    node = ScenarioProgressReadModel.build_hierarchy_node(
        delta=delta,
        result=result,
        children_deltas=[c1_delta],
        children_results=[c1_result],
    )
    assert len(node.children) == 1
    assert node.children[0].attack_result_id == "child-1"
    assert node.unresolved_child_attack_result_ids == ["child-1"]


async def test_persisted_conversation_statistics_parity_between_hierarchy_and_node(
    sqlite_instance: SQLiteMemory,
) -> None:
    """Test that real conversation message counts are reported in both full hierarchy and node detail."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="RealStatsScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    conv_id = str(uuid.uuid4())
    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=conv_id,
        objective="Real conversation stats test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        metadata={},  # No count metadata!
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    # Insert 3 saved messages into SQLite
    for seq in range(3):
        piece = MessagePiece(
            id=uuid.uuid4(),
            conversation_id=conv_id,
            role="user" if seq % 2 == 0 else "assistant",
            original_value_data_type="text",
            original_value=f"Message {seq}",
            converted_value=f"Message {seq}",
            sequence=seq,
        )
        await set_message_piece_sha256_async(piece)
        await sqlite_instance._run_database_operation_async(
            sqlite_instance._insert_entry, PromptMemoryEntry(entry=piece)
        )

    service = ScenarioRunService(memory=sqlite_instance)

    # 1. Node detail endpoint: full messages loaded, exact count=3, messages_truncated=False
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=ar_id,
    )
    assert node is not None
    assert len(node.conversations) == 1
    assert node.conversations[0].message_count == 3
    assert len(node.conversations[0].messages) == 3
    assert node.conversations[0].messages_truncated is False

    # 2. Full hierarchy endpoint: real count=3 via bounded stats read, empty messages list, messages_truncated=True
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    root_node = hierarchy.groups[0].roots[0]
    assert len(root_node.conversations) == 1
    assert root_node.conversations[0].message_count == 3
    assert len(root_node.conversations[0].messages) == 0
    assert root_node.conversations[0].messages_truncated is True

    # 3. Setting metadata={"conversation_message_count": 99} does not override real database stats
    ar2_uuid = uuid.uuid4()
    ar2_id = str(ar2_uuid)
    ar2 = AttackResult(
        attack_result_id=ar2_id,
        conversation_id=conv_id,
        objective="Metadata precedence test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        metadata={"conversation_message_count": 99},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar2])
    hierarchy2 = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy2 is not None
    root_node2 = [r for r in hierarchy2.groups[0].roots if r.attack_result_id == ar2_id][0]
    assert root_node2.conversations[0].message_count == 3  # Still real count 3, not 99!


async def test_persisted_result_with_non_numeric_conversation_message_count_metadata(
    sqlite_instance: SQLiteMemory,
) -> None:
    """Test that metadata={'conversation_message_count': 'unknown'} does not raise ValueError in hierarchy."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="NonNumericMetadataScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    ar_uuid = uuid.uuid4()
    ar_id = str(ar_uuid)
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=str(uuid.uuid4()),
        objective="Non-numeric metadata test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        metadata={"conversation_message_count": "unknown"},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    service = ScenarioRunService(memory=sqlite_instance)

    # Must succeed without raising ValueError
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    root_node = hierarchy.groups[0].roots[0]
    assert len(root_node.conversations) == 1
    # Real SQLite has 0 messages for this new conversation, so count is 0 and not truncated
    assert root_node.conversations[0].message_count == 0
    assert root_node.conversations[0].messages_truncated is False


async def test_bounded_conversation_messages_piece_budget_applied_at_sql_layer(
    sqlite_instance: SQLiteMemory,
) -> None:
    """Test that piece budget is applied at SQL layer, preventing hydration of unbounded pieces (e.g. 1,001 rows)."""
    ar_id = str(uuid.uuid4())
    conv_id = str(uuid.uuid4())
    now = datetime.now(UTC)

    # Insert 1,001 pieces for sequence 0
    pieces_seq0 = [
        MessagePiece(
            role="user",
            original_value=f"piece-{i}",
            converted_value=f"piece-{i}",
            conversation_id=conv_id,
            sequence=0,
            timestamp=now + timedelta(milliseconds=i),
        )
        for i in range(1001)
    ]
    # Insert 5 pieces for sequence 1
    pieces_seq1 = [
        MessagePiece(
            role="assistant",
            original_value=f"response-{i}",
            converted_value=f"response-{i}",
            conversation_id=conv_id,
            sequence=1,
            timestamp=now + timedelta(seconds=1, milliseconds=i),
        )
        for i in range(5)
    ]
    for p in pieces_seq0 + pieces_seq1:
        await set_message_piece_sha256_async(p)
    await sqlite_instance.add_message_pieces_to_memory_async(message_pieces=pieces_seq0 + pieces_seq1)

    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=conv_id,
        objective="Piece bounding test",
        outcome=AttackOutcome.SUCCESS,
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    # 1. Fetch via get_attack_result_conversation_projections_async with max_pieces_per_message=100
    projs = await sqlite_instance.get_attack_result_conversation_projections_async(
        attack_result_id=ar_id,
        include_messages=True,
        max_pieces_per_message=100,
    )
    assert len(projs.conversations) == 1
    messages = projs.conversations[0].messages
    assert len(messages) == 2

    # Sequence 0 (1,001 pieces) is capped at 100 with pieces_truncated=True
    msg0 = messages[0]
    assert msg0.sequence == 0
    assert len(msg0.pieces) == 100
    assert msg0.pieces_truncated is True

    # Sequence 1 (5 pieces) is fully returned with pieces_truncated=False
    msg1 = messages[1]
    assert msg1.sequence == 1
    assert len(msg1.pieces) == 5
    assert msg1.pieces_truncated is False

    # 2. Verify SQL-level row bounding: _fetch_bounded_conversation_messages only fetches
    # at most (max_pieces_per_message + 1) rows per sequence directly from the DB.
    # Testing directly with max_pieces_per_message=10:
    messages_capped_10, _ = sqlite_instance._fetch_bounded_conversation_messages(
        conversation_id=conv_id,
        limit=10,
        max_pieces_per_message=10,
    )
    assert len(messages_capped_10[0].pieces) == 10
    assert messages_capped_10[0].pieces_truncated is True
    assert len(messages_capped_10[1].pieces) == 5
    assert messages_capped_10[1].pieces_truncated is False


async def test_diagnostics_total_retries_count_event_mismatch_parity(
    sqlite_instance: SQLiteMemory,
) -> None:
    """
    Test that when total_retries=0 but retry events exist, run detail,
    producer counts, diagnostic item, and diagnostics.total_retries all agree on 0.
    """
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="MismatchRetryScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    retry_events = [
        RetryEvent(exception_type="RateLimitError", attempt_number=1),
        RetryEvent(exception_type="RateLimitError", attempt_number=2),
        RetryEvent(exception_type="TimeoutError", attempt_number=3),
    ]
    ar_id = str(uuid.uuid4())
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=str(uuid.uuid4()),
        objective="Retry mismatch test",
        outcome=AttackOutcome.ERROR,
        total_retries=0,  # Persisted count is 0
        retry_events=retry_events,  # 3 retry events recorded
        error_type="TimeoutError",
        error_message="Timed out",
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    service = ScenarioRunService(memory=sqlite_instance)

    # 1. Run summary / detail reports total_retries = 0
    run_summary = await service.get_run_async(scenario_result_id=scenario_id)
    assert run_summary is not None
    assert run_summary.total_retries == 0

    # 2. Hierarchy diagnostics and overall counts report total_retries = 0
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    assert hierarchy.overall_counts.retries == 0
    assert hierarchy.diagnostics.total_retries == 0
    assert hierarchy.diagnostics.attack_retries_count == 1
    assert len(hierarchy.diagnostics.attack_retries) == 1

    # The individual diagnostic item reports total_retries = 0 while preserving the 3 events
    diag_item = hierarchy.diagnostics.attack_retries[0]
    assert diag_item.total_retries == 0
    assert len(diag_item.retries) == 3


@pytest.mark.asyncio
async def test_clipped_tree_queries_actual_preview_node_conversation_stats(sqlite_instance: SQLiteMemory) -> None:
    """Test that when child is saved before parent and max_nodes=1, the actual returned parent gets real stats."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="ClippedTreeStatsScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    # 1. Parent with its own 3-message conversation
    parent_conv_id = str(uuid.uuid4())
    for seq in range(3):
        piece = MessagePiece(
            id=uuid.uuid4(),
            conversation_id=parent_conv_id,
            role="user" if seq % 2 == 0 else "assistant",
            original_value_data_type="text",
            original_value=f"Parent message {seq}",
            converted_value=f"Parent message {seq}",
            sequence=seq,
        )
        await set_message_piece_sha256_async(piece)
        await sqlite_instance._run_database_operation_async(
            sqlite_instance._insert_entry, PromptMemoryEntry(entry=piece)
        )

    child_conv_id = str(uuid.uuid4())
    child_id = str(uuid.uuid4())
    parent_id = str(uuid.uuid4())

    # Child is saved FIRST in memory
    child_ar = AttackResult(
        attack_result_id=child_id,
        conversation_id=child_conv_id,
        objective="Child attack",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    # Parent references child
    parent_ar = AttackResult(
        attack_result_id=parent_id,
        conversation_id=parent_conv_id,
        objective="Parent attack",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.ORCHESTRATION.value},
        metadata={"child_attack_result_ids": [child_id]},
    )
    # Add child first, then parent
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[child_ar, parent_ar])

    service = ScenarioRunService(memory=sqlite_instance)

    # With max_nodes=1, parent root is returned. It must report real count=3, not 0!
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id, max_nodes=1)
    assert hierarchy is not None
    assert hierarchy.truncated is True
    assert len(hierarchy.groups[0].roots) == 1
    root = hierarchy.groups[0].roots[0]
    assert root.attack_result_id == parent_id
    assert len(root.conversations) == 1
    assert root.conversations[0].conversation_id == parent_conv_id
    assert root.conversations[0].message_count == 3
    assert root.conversations[0].messages_truncated is True


@pytest.mark.asyncio
async def test_conversation_stats_query_failure_does_not_become_false_zero(sqlite_instance: SQLiteMemory) -> None:
    """Test that a failed stats query does not turn uncounted conversations into message_count=0."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="FailedStatsScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    conv_id = str(uuid.uuid4())
    ar_id = str(uuid.uuid4())
    ar = AttackResult(
        attack_result_id=ar_id,
        conversation_id=conv_id,
        objective="Failed stats test",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[ar])

    service = ScenarioRunService(memory=sqlite_instance)
    # Mock get_conversation_stats_async to fail with an exception
    sqlite_instance.get_conversation_stats_async = AsyncMock(side_effect=RuntimeError("Database error"))

    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id)
    assert hierarchy is not None
    root = hierarchy.groups[0].roots[0]
    assert len(root.conversations) == 1
    # Must NOT become 0! Must be None and marked truncated/incomplete
    assert root.conversations[0].message_count is None
    assert root.conversations[0].messages_truncated is True


@pytest.mark.asyncio
async def test_parent_with_2500_children_bounds_unresolved_references(sqlite_instance: SQLiteMemory) -> None:
    """Test that a parent with 2,500 children bounds unresolved references to 500 in both detail and full hierarchy."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="MassiveChildrenScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    parent_id = str(uuid.uuid4())
    child_ids = [str(uuid.uuid4()) for _ in range(2500)]

    # Add parent and the first 200 children into memory
    children = [
        AttackResult(
            attack_result_id=cid,
            conversation_id=str(uuid.uuid4()),
            objective=f"Child {i}",
            outcome=AttackOutcome.SUCCESS,
            attribution_parent_id=scenario_id,
            attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        )
        for i, cid in enumerate(child_ids[:200])
    ]
    parent = AttackResult(
        attack_result_id=parent_id,
        conversation_id=str(uuid.uuid4()),
        objective="Parent with 2500 children",
        outcome=AttackOutcome.SUCCESS,
        attribution_parent_id=scenario_id,
        attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.ORCHESTRATION.value},
        metadata={"child_attack_result_ids": child_ids},
    )
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=[parent, *children])

    service = ScenarioRunService(memory=sqlite_instance)

    # 1. Single node detail: with 200 immediate children loaded, unresolved references are capped at 500 (not 2,300)
    node = await service.get_run_hierarchy_node_async(
        scenario_result_id=scenario_id,
        attack_result_id=parent_id,
        max_immediate_children=200,
    )
    assert node is not None
    assert len(node.children) == 200
    assert len(node.unresolved_child_attack_result_ids) == 500
    assert node.unresolved_child_references_truncated is True

    # 2. Full hierarchy with max_nodes=1: parent has 2,500 unresolved children capped at 500 (not 2,500)
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id, max_nodes=1)
    assert hierarchy is not None
    root = hierarchy.groups[0].roots[0]
    assert len(root.unresolved_child_attack_result_ids) == 500
    assert root.unresolved_child_references_truncated is True


@pytest.mark.asyncio
async def test_cold_hierarchy_hydration_bounded_by_max_nodes(sqlite_instance: SQLiteMemory) -> None:
    """Test that a cold max_nodes=1 request only fetches 1 delta and does not hydrate 1,001 results across 3 pages."""
    scenario_uuid = uuid.uuid4()
    scenario_id = str(scenario_uuid)
    scen = make_scenario_result(
        id=scenario_uuid,
        scenario_name="ColdHydrationScenario",
        scenario_version=1,
        objective_target_identifier=get_mock_target_identifier(),
        attack_results={},
    )
    await sqlite_instance.add_scenario_results_to_memory_async(scenario_results=[scen])

    # Insert 1,001 attack results
    results = [
        AttackResult(
            attack_result_id=str(uuid.uuid4()),
            conversation_id=str(uuid.uuid4()),
            objective=f"Attack {i}",
            outcome=AttackOutcome.SUCCESS,
            attribution_parent_id=scenario_id,
            attribution_data={"parent_collection": "group_1", "result_role": AttackResultRole.TARGET_FACING.value},
        )
        for i in range(1001)
    ]
    await sqlite_instance.add_attack_results_to_memory_async(attack_results=results)

    # Track delta queries
    original_get_deltas = sqlite_instance.get_scenario_attack_result_deltas_async
    query_limits: list[int] = []

    async def tracking_get_deltas(*args: Any, **kwargs: Any) -> Any:
        query_limits.append(kwargs.get("limit", 0))
        return await original_get_deltas(*args, **kwargs)

    sqlite_instance.get_scenario_attack_result_deltas_async = tracking_get_deltas  # type: ignore[assignment]

    service = ScenarioRunService(memory=sqlite_instance)

    # Cold max_nodes=1 hierarchy call
    hierarchy = await service.get_run_hierarchy_async(scenario_result_id=scenario_id, max_nodes=1)
    assert hierarchy is not None
    assert hierarchy.truncated is True
    # Must only have executed 1 page query (capped at storage page size), NOT 3 page queries (500, 500, 1)!
    assert len(query_limits) == 1
    assert query_limits[0] == 500


def test_build_scenario_hierarchy_unqueried_conversation_not_false_zero() -> None:
    """Test that missing-key branch in build_scenario_hierarchy produces None and truncated=True, never false zero."""
    delta = _make_test_delta(
        attack_result_id="root-1",
        role=AttackResultRole.TARGET_FACING,
        parent_collection="group_1",
        seed_group_id="seed_1",
        conversation_id="conv-unqueried",
    )
    snapshot = _make_test_snapshot(deltas=[delta])
    # Empty stats passed (e.g. stats read failure or unqueried)
    projection = ScenarioProgressReadModel.build_scenario_hierarchy(
        scenario_result_id="scenario-run-1",
        scenario_name="TestScenario",
        scenario_version=1,
        status=ScenarioRunState.COMPLETED,
        snapshot=snapshot,
        conversation_stats={},
    )
    conv = projection.groups[0].roots[0].conversations[0]
    # Must NOT report 0 and False!
    assert conv.message_count is None
    assert conv.messages_truncated is True
