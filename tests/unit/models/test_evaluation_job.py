# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Engine-neutral immutable wire contracts and ordered source/terminal evidence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from pyrit.models import EvalPackageRef, EvalSourceKind
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactKind,
    EvaluationArtifactManifest,
    EvaluationArtifactMediaType,
    EvaluationCleanupState,
    EvaluationControlKind,
    EvaluationControlRequest,
    EvaluationEvidenceState,
    EvaluationJobEvent,
    EvaluationJobEventKind,
    EvaluationJobRequest,
    EvaluationJobSnapshot,
    EvaluationJobState,
    EvaluationRuntimeKind,
    EvaluationWaitBoundary,
)


@pytest.fixture
def job_request() -> EvaluationJobRequest:
    return EvaluationJobRequest(
        job_id=uuid4(),
        run_id=uuid4(),
        attempt_id=uuid4(),
        runtime=EvaluationRuntimeKind.ORIGINAL_INSPECT,
        source=EvalPackageRef(kind=EvalSourceKind.NAMED, name="public_fixture", source_sha256="a" * 64),
        case_id="b" * 64,
        execution_profile_sha256="c" * 64,
    )


def test_job_request_roundtrip_and_case_run_alias_dedup(job_request: EvaluationJobRequest) -> None:
    request = job_request
    assert EvaluationJobRequest.model_validate_json(request.model_dump_json()) == request
    alias = request.model_copy(update={"job_id": uuid4(), "attempt_id": uuid4()})
    assert alias.request_sha256 != request.request_sha256
    assert alias.case_run_sha256 == request.case_run_sha256
    assert request.model_copy(update={"run_id": uuid4()}).case_run_sha256 != request.case_run_sha256
    with pytest.raises(ValidationError, match="frozen"):
        request.job_id = uuid4()


@pytest.mark.parametrize("field", ["input", "task_code", "source_url", "model", "sandbox", "secrets"])
def test_job_request_rejects_producer_executable_or_input_fields(
    *, job_request: EvaluationJobRequest, field: str
) -> None:
    content = job_request.model_dump(mode="json")
    content[field] = "not admitted"
    with pytest.raises(ValidationError):
        EvaluationJobRequest.model_validate_json(json.dumps(content))


def test_original_rejects_controls_and_unknown_schema(job_request: EvaluationJobRequest) -> None:
    for update in (
        {"controls": (EvaluationControlKind.SEND_MESSAGE,)},
        *({"schema_version": value} for value in (2, True, 1.0, "1")),
    ):
        content = job_request.model_dump(mode="json")
        content.update(update)
        with pytest.raises(ValidationError):
            EvaluationJobRequest.model_validate_json(json.dumps(content))
        with pytest.raises(ValidationError):
            EvaluationJobRequest.model_validate(job_request.model_copy(update=update))


@pytest.mark.parametrize(
    ("kind", "message", "valid"),
    [
        (EvaluationControlKind.SEND_MESSAGE, "harmless follow-up", True),
        (EvaluationControlKind.NUDGE, "harmless reminder", True),
        (EvaluationControlKind.ADVANCE, None, True),
        (EvaluationControlKind.STOP, None, True),
        (EvaluationControlKind.STOP, "not shell text", False),
        (EvaluationControlKind.NUDGE, None, False),
        (EvaluationControlKind.SEND_MESSAGE, "x" * 8193, False),
        (EvaluationControlKind.SEND_MESSAGE, "\u00e9" * 4097, False),
    ],
)
def test_control_text_and_utf8_bounds(*, kind: EvaluationControlKind, message: str | None, valid: bool) -> None:
    values = {"command_id": uuid4(), "boundary_id": uuid4(), "kind": kind, "message": message}
    if valid:
        command = EvaluationControlRequest.model_validate(values)
        assert EvaluationControlRequest.model_validate_json(command.model_dump_json()) == command
    else:
        with pytest.raises(ValidationError):
            EvaluationControlRequest.model_validate(values)


@pytest.mark.parametrize("name", ["../outside.eval", "CON.eval", "manifest.json", "ambiguous."])
def test_artifact_rejects_paths_and_reserved_names(name: str) -> None:
    with pytest.raises(ValidationError):
        EvaluationArtifact(
            name=name,
            kind=EvaluationArtifactKind.INSPECT_EVAL,
            media_type=EvaluationArtifactMediaType.INSPECT_EVAL,
            sha256="d" * 64,
            bytes=1,
        )


def test_artifacts_reject_foreign_request_and_case_insensitive_collision(job_request: EvaluationJobRequest) -> None:
    request = job_request
    artifact = EvaluationArtifact(
        name="source.eval",
        kind=EvaluationArtifactKind.INSPECT_EVAL,
        media_type=EvaluationArtifactMediaType.INSPECT_EVAL,
        sha256="d" * 64,
        bytes=1,
    )
    for digest, inventory in (
        ("e" * 64, (artifact,)),
        (request.request_sha256, (artifact, artifact.model_copy(update={"name": "SOURCE.eval"}))),
    ):
        with pytest.raises(ValidationError):
            EvaluationArtifactManifest(request=request, request_sha256=digest, fence_id=uuid4(), artifacts=inventory)
    with pytest.raises(ValidationError):
        EvaluationArtifact.model_validate_json(
            artifact.model_copy(update={"kind": EvaluationArtifactKind.NATIVE_EVIDENCE}).model_dump_json()
        )


def test_snapshot_rejects_gaps_terminal_reordering_and_success_without_canonical_evidence(
    job_request: EvaluationJobRequest,
) -> None:
    request = job_request
    first = EvaluationJobEvent(
        sequence=1,
        kind=EvaluationJobEventKind.SUBMITTED,
        state=EvaluationJobState.QUEUED,
        occurred_at=datetime.now(UTC),
    )
    terminal = EvaluationJobEvent(
        sequence=3,
        kind=EvaluationJobEventKind.TERMINAL,
        state=EvaluationJobState.CANCELLED,
        occurred_at=datetime.now(UTC),
    )
    for state, events in (
        (EvaluationJobState.CANCELLED, (first, terminal)),
        (EvaluationJobState.QUEUED, (terminal, first)),
        (EvaluationJobState.SUCCEEDED, ()),
    ):
        with pytest.raises(ValidationError):
            EvaluationJobSnapshot(
                request=request,
                request_sha256=request.request_sha256,
                state=state,
                evidence=EvaluationEvidenceState.ABSENT,
                cleanup=EvaluationCleanupState.UNKNOWN,
                last_sequence=3,
                events=events,
            )


def test_boundary_rejects_duplicate_capabilities() -> None:
    with pytest.raises(ValidationError):
        EvaluationWaitBoundary(
            boundary_id=uuid4(), name="reviewed_wait", controls=(EvaluationControlKind.STOP, EvaluationControlKind.STOP)
        )
