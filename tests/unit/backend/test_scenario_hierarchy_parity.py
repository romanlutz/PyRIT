# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit and parity tests for bounded Scenario result hierarchy projections."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import status
from fastapi.testclient import TestClient

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
    MessagePiece,
    RetryEvent,
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


def _make_test_snapshot(
    *,
    deltas: list[ScenarioAttackResultDelta],
    plan: ScenarioRunPlan | None = None,
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
    summary = ScenarioProgressSummary(
        overall=ScenarioProgressCounts(
            completed=len([r for r in results if r.outcome != AttackOutcome.ERROR]),
            planned=1,
            succeeded=len([r for r in results if r.outcome == AttackOutcome.SUCCESS]),
            errors=len([r for r in results if r.outcome == AttackOutcome.ERROR]),
            retries=sum(r.total_retries for r in results),
        ),
        atomic_groups=[],
    )
    return ScenarioProgressSnapshot(
        deltas=tuple(deltas),
        results=tuple(results),
        summary=summary,
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
                seconds_offset=i,
            )
        )
    deltas.extend(
        _make_test_delta(
            attack_result_id=f"retry-{i}",
            outcome=AttackOutcome.SUCCESS,
            retries=[RetryEvent(exception_type="RateLimit", attempt_number=1)],
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
        return_value=[
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
        ]
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
    assert len(conv_projections) == 1
    proj = conv_projections[0]
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
