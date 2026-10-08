# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Execution-only wire validation; these fixtures claim no actual source execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pyrit.models import EvalPackageRef, EvalSourceKind
from pyrit.models.evaluation_job import (
    EvaluationCleanupState,
    EvaluationJobEventKind,
    EvaluationJobRequest,
    EvaluationRuntimeKind,
)
from pyrit.models.evaluation_worker import (
    EvaluationGatewaySettlement,
    EvaluationGatewaySettlementDisposition,
    EvaluationWorkerAdmission,
    EvaluationWorkerBinding,
    EvaluationWorkerEvent,
    EvaluationWorkerEvidence,
    EvaluationWorkerSnapshot,
    EvaluationWorkerState,
    EvaluationWorkerTerminal,
    evaluation_worker_schema_sha256,
)


@pytest.fixture
def worker_admission() -> EvaluationWorkerAdmission:
    request = EvaluationJobRequest(
        job_id=uuid4(),
        run_id=uuid4(),
        attempt_id=uuid4(),
        runtime=EvaluationRuntimeKind.ORIGINAL_INSPECT,
        source=EvalPackageRef(kind=EvalSourceKind.NAMED, name="public_fixture", source_sha256="a" * 64),
        case_id="b" * 64,
        execution_profile_sha256="c" * 64,
    )
    return EvaluationWorkerAdmission(
        request=request, request_sha256=request.request_sha256, gateway_fence_id=uuid4(), actor_id="operator"
    )


@pytest.fixture
def worker_binding(worker_admission: EvaluationWorkerAdmission) -> EvaluationWorkerBinding:
    admission = worker_admission
    return EvaluationWorkerBinding(
        service_id="public_fixture",
        worker_incarnation_id=uuid4(),
        worker_fence_id=uuid4(),
        gateway_fence_id=admission.gateway_fence_id,
        job_id=admission.request.job_id,
        run_id=admission.request.run_id,
        attempt_id=admission.request.attempt_id,
        request_sha256=admission.request_sha256,
        admission_sha256=admission.admission_sha256,
        actor_id=admission.actor_id,
    )


def test_binding_preserves_independent_fences_and_full_admission_identity(
    *, worker_admission: EvaluationWorkerAdmission, worker_binding: EvaluationWorkerBinding
) -> None:
    assert worker_binding.accepts(worker_admission)
    assert worker_binding.worker_fence_id != worker_binding.gateway_fence_id
    assert not worker_binding.accepts(worker_admission.model_copy(update={"actor_id": "foreign"}))
    assert not worker_binding.accepts(worker_admission.model_copy(update={"gateway_fence_id": uuid4()}))
    assert EvaluationWorkerBinding.model_validate_json(worker_binding.model_dump_json()) == worker_binding
    assert len(evaluation_worker_schema_sha256()) == 64


def test_worker_cannot_copy_the_gateway_dispatch_fence(worker_binding: EvaluationWorkerBinding) -> None:
    with pytest.raises(ValidationError, match="independent"):
        EvaluationWorkerBinding.model_validate(
            worker_binding.model_copy(update={"worker_fence_id": worker_binding.gateway_fence_id})
        )


@pytest.mark.parametrize("value", [True, "1", 1.0, 2])
def test_worker_version_rejects_raw_and_unchecked_typed_instances(
    *, worker_admission: EvaluationWorkerAdmission, value: object
) -> None:
    content = worker_admission.model_dump(mode="json")
    content["schema_version"] = value
    with pytest.raises(ValidationError):
        EvaluationWorkerAdmission.model_validate_json(json.dumps(content))
    with pytest.raises(ValidationError):
        EvaluationWorkerAdmission.model_validate(worker_admission.model_copy(update={"schema_version": value}))


@pytest.mark.parametrize("field", ["canonical", "projection_id", "score_ids", "attack_result_ids", "worker_database"])
def test_terminal_rejects_canonical_database_fields(*, worker_binding: EvaluationWorkerBinding, field: str) -> None:
    terminal = EvaluationWorkerTerminal(
        binding=worker_binding,
        state=EvaluationWorkerState.CANCELLED,
        evidence=EvaluationWorkerEvidence.ABSENT,
        cleanup=EvaluationCleanupState.VERIFIED,
        last_sequence=2,
    )
    content = terminal.model_dump(mode="json")
    content[field] = "untrusted worker data"
    with pytest.raises(ValidationError):
        EvaluationWorkerTerminal.model_validate_json(json.dumps(content))


def test_completed_is_source_retention_and_closure_not_a_grade(worker_binding: EvaluationWorkerBinding) -> None:
    for evidence, cleanup, manifest in (
        (EvaluationWorkerEvidence.ABSENT, EvaluationCleanupState.VERIFIED, None),
        (EvaluationWorkerEvidence.SOURCE_RETAINED, EvaluationCleanupState.UNKNOWN, "d" * 64),
        (EvaluationWorkerEvidence.UNKNOWN, EvaluationCleanupState.VERIFIED, None),
    ):
        with pytest.raises(ValidationError):
            EvaluationWorkerTerminal(
                binding=worker_binding,
                state=EvaluationWorkerState.COMPLETED,
                evidence=evidence,
                cleanup=cleanup,
                last_sequence=2,
                manifest_sha256=manifest,
            )


def test_closure_settlement_does_not_invent_artifact_hashes(worker_binding: EvaluationWorkerBinding) -> None:
    closure = EvaluationGatewaySettlement(
        binding=worker_binding, disposition=EvaluationGatewaySettlementDisposition.CLOSURE_OBSERVED
    )
    assert closure.artifact_sha256 is None and closure.worker_manifest_sha256 is None
    for disposition in (
        EvaluationGatewaySettlementDisposition.CANONICAL_IMPORTED,
        EvaluationGatewaySettlementDisposition.RETAINED_ONLY,
    ):
        with pytest.raises(ValidationError):
            EvaluationGatewaySettlement(binding=worker_binding, disposition=disposition)
        retained = EvaluationGatewaySettlement(
            binding=worker_binding,
            disposition=disposition,
            worker_manifest_sha256="d" * 64,
            gateway_manifest_sha256="e" * 64,
            artifact_sha256="f" * 64,
        )
        assert retained.settlement_sha256 != closure.settlement_sha256
    with pytest.raises(ValidationError):
        EvaluationGatewaySettlement.model_validate(closure.model_copy(update={"artifact_sha256": "f" * 64}))


def test_worker_snapshot_rejects_event_gaps_foreign_terminal_and_canonical_fields(
    *, worker_admission: EvaluationWorkerAdmission, worker_binding: EvaluationWorkerBinding
) -> None:
    terminal = EvaluationWorkerTerminal(
        binding=worker_binding,
        state=EvaluationWorkerState.CANCELLED,
        evidence=EvaluationWorkerEvidence.ABSENT,
        cleanup=EvaluationCleanupState.VERIFIED,
        last_sequence=2,
    )
    first = EvaluationWorkerEvent(
        sequence=1,
        kind=EvaluationJobEventKind.SUBMITTED,
        state=EvaluationWorkerState.QUEUED,
        occurred_at=datetime.now(UTC),
    )
    last = EvaluationWorkerEvent(
        sequence=2,
        kind=EvaluationJobEventKind.TERMINAL,
        state=EvaluationWorkerState.CANCELLED,
        occurred_at=datetime.now(UTC),
    )
    snapshot = EvaluationWorkerSnapshot(
        request=worker_admission.request,
        request_sha256=worker_admission.request_sha256,
        binding=worker_binding,
        state=terminal.state,
        evidence=terminal.evidence,
        cleanup=terminal.cleanup,
        last_sequence=2,
        events=(first, last),
        terminal=terminal,
    )
    for update in (
        {"events": (first, last.model_copy(update={"sequence": 3}))},
        {
            "terminal": terminal.model_copy(
                update={"binding": worker_binding.model_copy(update={"worker_fence_id": uuid4()})}
            )
        },
        {"canonical": {"score_ids": [str(uuid4())]}},
    ):
        with pytest.raises(ValidationError):
            EvaluationWorkerSnapshot.model_validate(snapshot.model_copy(update=update))
    with pytest.raises(ValidationError):
        EvaluationWorkerEvent(
            sequence=1,
            kind=EvaluationJobEventKind.FINALIZING,
            state=EvaluationWorkerState.RUNNING,
            occurred_at=datetime.now(UTC),
        )
