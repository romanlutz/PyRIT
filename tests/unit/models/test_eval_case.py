# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for source-only case identity and fresh task-owned run identity."""

import uuid

import pytest
from pydantic import ValidationError

from pyrit.models import (
    EvalCaseRef,
    EvalPackageRef,
    EvalRunRef,
    EvalScoreProvenance,
    EvalScoreRole,
    EvalSourceKind,
    EvalSpecRef,
    HarnessProfileRef,
    InputVariantRef,
    ModelRouteRef,
    ScenarioExecutionOwner,
    ScenarioIdentifier,
)


def _package() -> EvalPackageRef:
    return EvalPackageRef(kind=EvalSourceKind.NAMED, name="example_evals", source_sha256="a" * 64)


def _case(*, task: str = "first", sample: str = "1") -> EvalCaseRef:
    return EvalCaseRef(package=_package(), task_name=task, task_version="v1", sample_id=sample, epoch=0)


def _spec(*, variant: InputVariantRef | None = None) -> EvalSpecRef:
    return EvalSpecRef(
        package=_package(),
        harness=HarnessProfileRef(name="harness", config_sha256="b" * 64),
        model_route=ModelRouteRef(name="attacker", config_sha256="c" * 64),
        input_variant=variant,
    )


def test_cases_with_same_objective_text_keep_distinct_source_identity() -> None:
    first = _case(task="first", sample="1")
    second = _case(task="second", sample="1")
    third = _case(task="first", sample="2")

    assert len({first.case_id, second.case_id, third.case_id}) == 3
    assert first.case_id == _case(task="first", sample="1").case_id


def test_task_version_and_epoch_are_part_of_canonical_case_identity() -> None:
    original = _case()
    newer = original.model_copy(update={"task_version": "v2"})
    next_epoch = original.model_copy(update={"epoch": 1})

    assert len({original.case_id, newer.case_id, next_epoch.case_id}) == 3


def test_source_fingerprint_is_independent_of_harness_route_and_overlay() -> None:
    case = _case()
    default = _spec()
    overlay = _spec(variant=InputVariantRef(case_id=case.case_id, surface_id="user_input", content_sha256="d" * 64))
    different_harness = default.model_copy(
        update={"harness": HarnessProfileRef(name="another_harness", config_sha256="e" * 64)}
    )
    different_model = default.model_copy(
        update={"model_route": ModelRouteRef(name="different_model", config_sha256="f" * 64)}
    )

    assert default.package.source_sha256 == overlay.package.source_sha256
    assert default.package.source_fingerprint == different_harness.package.source_fingerprint
    assert case.case_id == _case().case_id
    assert default.spec_sha256 != overlay.spec_sha256
    assert default.spec_sha256 != different_harness.spec_sha256
    assert default.spec_sha256 != different_model.spec_sha256


def test_fresh_identical_runs_have_distinct_case_run_ids() -> None:
    spec = _spec()
    first = EvalRunRef(spec=spec, run_instance_id=uuid.uuid4())
    second = EvalRunRef(spec=spec, run_instance_id=uuid.uuid4())

    assert first.spec.spec_sha256 == second.spec.spec_sha256
    assert first.case_run_id(case=_case()) != second.case_run_id(case=_case())


def test_single_case_overlay_changes_run_identity_not_source_case_id() -> None:
    case = _case()
    variant = InputVariantRef(case_id=case.case_id, surface_id="system_prompt", content_sha256="d" * 64)
    run_id = uuid.uuid4()
    original = EvalRunRef(spec=_spec(), run_instance_id=run_id)
    edited = EvalRunRef(spec=_spec(variant=variant), run_instance_id=run_id)

    assert original.spec.package.source_sha256 == edited.spec.package.source_sha256
    assert case.case_id == variant.case_id
    assert original.case_run_id(case=case) != edited.case_run_id(case=case)


def test_variant_for_another_case_and_package_mismatch_fail_explicitly() -> None:
    case = _case()
    other = _case(sample="2")
    variant = InputVariantRef(case_id=other.case_id, surface_id="user_input", content_sha256="d" * 64)
    run = EvalRunRef(spec=_spec(variant=variant), run_instance_id=uuid.uuid4())
    with pytest.raises(ValueError, match="targets a different Eval case"):
        run.case_run_id(case=case)

    different_package = _package().model_copy(update={"source_sha256": "f" * 64})
    mismatched = case.model_copy(update={"package": different_package})
    with pytest.raises(ValueError, match="different package"):
        run.case_run_id(case=mismatched)


def test_invalid_alias_digest_and_missing_case_coordinates_fail() -> None:
    with pytest.raises(ValidationError, match="name"):
        EvalPackageRef(kind=EvalSourceKind.TRUSTED_LOCAL, name="C:\\private\\source", source_sha256="a" * 64)
    with pytest.raises(ValidationError, match="config_sha256"):
        ModelRouteRef(name="attacker", config_sha256="not-a-sha256")
    with pytest.raises(ValidationError, match="sample_id"):
        EvalCaseRef(package=_package(), task_name="first", task_version="v1", sample_id="", epoch=0)


def test_refs_are_immutable_and_round_trip_from_json() -> None:
    case = _case()
    with pytest.raises(ValidationError, match="frozen"):
        case.sample_id = "2"
    assert EvalCaseRef.model_validate_json(case.model_dump_json()).case_id == case.case_id


def test_original_score_provenance_round_trips_as_named_score_metadata() -> None:
    provenance = EvalScoreProvenance(
        role=EvalScoreRole.BENCHMARK_ORIGINAL,
        case_run_id="a" * 64,
        pyrit_scorer_hash="b" * 64,
    )

    assert EvalScoreProvenance.from_metadata(metadata=provenance.to_metadata()) == provenance
    with pytest.raises(ValueError, match="missing task-owned evaluation provenance"):
        EvalScoreProvenance.from_metadata(metadata={"pyrit_eval_role": "benchmark_original"})


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"source_name": "C:\\private\\suite"}, "public source_name alias"),
        ({"source_kind": "unknown"}, "known source_kind"),
        ({"run_instance_id": str(uuid.uuid4())}, "unsupported parameters"),
        ({"input_variant_sha256": "c" * 64}, "digest and a surface"),
    ],
)
def test_task_owned_identifier_rejects_private_or_unvalidated_params(overrides: dict[str, str], error: str) -> None:
    params = {
        "execution_owner": ScenarioExecutionOwner.TASK_OWNED.value,
        "eval_spec_sha256": "a" * 64,
        "source_kind": EvalSourceKind.NAMED.value,
        "source_name": "example_evals",
        "source_sha256": "b" * 64,
        "harness_name": "harness",
        "harness_sha256": "c" * 64,
        "model_route_name": "attacker",
        "model_route_sha256": "d" * 64,
        "case_set_sha256": "e" * 64,
        **overrides,
    }
    with pytest.raises(ValidationError, match=error):
        ScenarioIdentifier(class_name="TaskEval", class_module="pyrit.scenario", params=params)
