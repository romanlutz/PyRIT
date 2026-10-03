# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public, content-free evidence admission DTO tests."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pyrit.backend.services.original_evidence_admission import OriginalEvidenceEnvelope
from pyrit.backend.services.original_run_admission import OriginalCleanupReceipt
from pyrit.models import config_hash

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def evidence_envelope() -> OriginalEvidenceEnvelope:
    return OriginalEvidenceEnvelope(
        app_run_id=uuid4(),
        job_ref=uuid4(),
        profile_ref="public-fixture",
        operator_oid="approved-operator",
        run_instance_id=uuid4(),
        worker_scenario_id=uuid4(),
        worker_scenario_sha256="a" * 64,
        manifest_sha256="b" * 64,
        source_sha256="c" * 64,
        case_run_id="d" * 64,
        model_role="evaluated",
        model_route_sha256="e" * 64,
        model_role_receipt_id="owned-role.1",
        model_role_sha256="f" * 64,
        source_state="success",
        source_complete=True,
        archive_sha256="1" * 64,
        archive_bytes=123,
        inspect_run_id="source-run",
        inspect_eval_id="source-eval",
        final_score_event_id="original-final-event",
        final_score_event_sha256="2" * 64,
        operation_terminal_receipt_id="owned-operation.1",
        operation_terminal_sha256="3" * 64,
        cleanup=OriginalCleanupReceipt(state="proved", receipt_id="owned-physical.1"),
        cleanup_sha256="4" * 64,
    )


def test_evidence_envelope_is_immutable_and_has_canonical_actor_job_digest(
    evidence_envelope: OriginalEvidenceEnvelope,
) -> None:
    assert OriginalEvidenceEnvelope.model_validate_json(evidence_envelope.model_dump_json()) == evidence_envelope
    assert evidence_envelope.sha256 == config_hash(
        {"original_evidence": evidence_envelope.model_dump(mode="json", exclude_none=True)}
    )
    assert evidence_envelope.job.app_run_id == evidence_envelope.app_run_id
    assert evidence_envelope.binding.operator_oid == evidence_envelope.operator_oid
    with pytest.raises(ValidationError):
        evidence_envelope.source_complete = False


@pytest.mark.parametrize(
    "override",
    [
        {"schema_version": True},
        {"schema_version": "1"},
        {"source_complete": 1},
        {"source_complete": "true"},
        {"model_role": "scorer"},
        {"archive_bytes": True},
        {"archive_bytes": 16 * 1024 * 1024 + 1},
        {"task_path": "arbitrary.py"},
        {"model_url": "https://example.invalid"},
        {"grade": 1.0},
        {"final_score_event_sha256": None},
        {"operation_terminal_sha256": None},
        {"cleanup_sha256": None},
    ],
)
def test_evidence_envelope_rejects_aliases_arbitrary_configuration_and_half_receipts(
    evidence_envelope: OriginalEvidenceEnvelope, override: dict[str, Any]
) -> None:
    with pytest.raises(ValidationError):
        OriginalEvidenceEnvelope.model_validate({**evidence_envelope.model_dump(), **override})


def test_startup_failure_keeps_physical_cleanup_separate_from_absent_grade(
    evidence_envelope: OriginalEvidenceEnvelope,
) -> None:
    failure = OriginalEvidenceEnvelope.model_validate(
        {
            **evidence_envelope.model_dump(),
            "source_state": "error",
            "source_complete": False,
            "archive_sha256": None,
            "archive_bytes": None,
            "inspect_run_id": None,
            "inspect_eval_id": None,
            "final_score_event_id": None,
            "final_score_event_sha256": None,
            "operation_terminal_receipt_id": None,
            "operation_terminal_sha256": None,
        }
    )
    assert failure.cleanup.proved and failure.archive_sha256 is None
    assert failure.operation_terminal_receipt_id is None
    with pytest.raises(ValidationError):
        OriginalEvidenceEnvelope.model_validate({**failure.model_dump(), "worker_score_id": uuid4()})
