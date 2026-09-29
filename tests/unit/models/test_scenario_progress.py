# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for scenario progress plan validation."""

import pytest
from pydantic import ValidationError

from pyrit.models import ScenarioRunPlan, ScenarioRunPlanAtomicGroup, ScenarioRunPlanSeedGroup


def _seed(*, seed_id: str = "seed-1") -> ScenarioRunPlanSeedGroup:
    return ScenarioRunPlanSeedGroup(id=seed_id, objective_sha256=f"sha-{seed_id}", objective=seed_id)


def _group(*, group_id: str = "group-1", seed_group_ids: list[str] | None = None) -> ScenarioRunPlanAtomicGroup:
    return ScenarioRunPlanAtomicGroup(
        id=group_id,
        atomic_attack_name=group_id,
        display_group=group_id,
        technique_eval_hash=f"eval-{group_id}",
        seed_group_ids=seed_group_ids or ["seed-1"],
    )


@pytest.mark.parametrize(
    ("atomic_groups", "seed_groups", "match"),
    [
        ([_group(), _group()], [_seed()], "duplicate atomic group IDs"),
        ([_group()], [_seed(), _seed()], "duplicate seed group IDs"),
        ([_group(seed_group_ids=["seed-1", "seed-1"])], [_seed()], "duplicate seed group IDs"),
        ([_group(seed_group_ids=["missing"])], [_seed()], "unknown seed group IDs"),
    ],
)
def test_run_plan_rejects_ambiguous_or_invalid_normalized_ids(
    atomic_groups: list[ScenarioRunPlanAtomicGroup],
    seed_groups: list[ScenarioRunPlanSeedGroup],
    match: str,
) -> None:
    with pytest.raises(ValidationError, match=match):
        ScenarioRunPlan(atomic_groups=atomic_groups, seed_groups=seed_groups)


def test_legacy_run_plan_serializes_byte_identically_after_reload() -> None:
    stored = (
        '{"version":1,"scenario_registry_name":"legacy","atomic_groups":'
        '[{"id":"group-1","atomic_attack_name":"group-1","display_group":"group-1",'
        '"technique_eval_hash":"eval-group-1","seed_group_ids":["seed-1"],"tags":[]}],'
        '"seed_groups":[{"id":"seed-1","objective_sha256":"sha-seed-1",'
        '"objective":"seed-1","prompts":[]}]}'
    )
    plan = ScenarioRunPlan.model_validate_json(stored)

    assert plan.model_dump_json(exclude_none=True) == stored
    assert '"run_instance_id"' not in plan.model_dump_json()
    assert '"case_id"' not in plan.model_dump_json()
    assert (
        ScenarioRunPlan.model_validate(plan.model_dump(mode="json", exclude_none=True)).model_dump_json(
            exclude_none=True
        )
        == stored
    )
