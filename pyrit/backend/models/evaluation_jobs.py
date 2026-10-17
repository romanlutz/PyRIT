# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Opaque original-case launch and read-only result views, not worker protocol messages."""

from __future__ import annotations

from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field

from pyrit.models import AttackOutcome, ScoreStatus  # noqa: TC001 (Pydantic resolves field types)
from pyrit.models.evaluation_job import (  # noqa: TC001 (Pydantic resolves field types)
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationEvidenceState,
    EvaluationJobState,
)


class EvaluationCaseLaunch(BaseModel):
    """An actor-bound retry key; the server selects all source and execution references."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)

    request_id: UUID


class EvaluationOriginalResult(BaseModel):
    """Authorized original grade and retained provenance, without private Task or scorer configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
    run_id: UUID
    case_ref: str
    state: EvaluationJobState
    evidence: EvaluationEvidenceState
    cleanup: EvaluationCleanupState
    canonical: EvaluationCanonicalReceipt
    archive_bytes: int = Field(ge=1, le=16 * 1024 * 1024)
    original_score: str | None
    pyrit_score_status: ScoreStatus
    pyrit_attack_outcome: AttackOutcome
    final_score_event_id: str | None
    final_score_event_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
