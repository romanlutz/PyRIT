# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Explicit v2 submission provenance; the offline-only v1 contract remains unchanged."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator

from pyrit.models.submission import (
    StrictSubmissionRecord,
    SubmissionBehaviorOutcome,
    SubmissionCallEvidence,
    SubmissionCapabilities,
    SubmissionCleanupStatus,
    SubmissionDispatchState,
    SubmissionEvidenceCompleteness,
    SubmissionFeedbackKind,
    SubmissionRemoteDisposition,
    SubmissionReportStatus,
    SubmissionStatus,
    SubmissionTerminationReason,
)


class SubmissionModeV2(str, Enum):
    """The binding-declared evaluation mode, not proof that a dispatch happened."""

    OFFLINE = "offline"
    REAL = "real"


class SubmissionProvenanceV2(BaseModel):
    """A validated mode pair carried from the binding into every retained v2 surface."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: SubmissionModeV2
    simulated: bool = Field(strict=True)

    @model_validator(mode="after")
    def _validate_pair(self) -> SubmissionProvenanceV2:
        if self.simulated != (self.mode is SubmissionModeV2.OFFLINE):
            raise ValueError("V2 provenance requires offline/simulated or real/not-simulated.")
        return self

    @property
    def evidence_label(self) -> Literal["OFFLINE/SIMULATED", "REAL/UNSIMULATED"]:
        """The explicit evidence-mode label, independent of result or dispatch status."""
        return "OFFLINE/SIMULATED" if self.simulated else "REAL/UNSIMULATED"


class StrictSubmissionReportV2(SubmissionProvenanceV2):
    """The strict submission ledger with explicit v2 provenance."""

    contract_version: Literal["strict-submission-v2"]
    status: SubmissionReportStatus
    last_valid_grade: float | None
    selected_submission_id: str | None
    full_success: bool = Field(strict=True)
    submissions: tuple[StrictSubmissionRecord, ...]
    capabilities: SubmissionCapabilities

    @field_validator("last_valid_grade", mode="before")
    @classmethod
    def _validate_grade(cls, value: object) -> float | None:
        if value is None:
            return None
        if type(value) not in (int, float) or not isinstance(value, (int, float)):
            raise ValueError("A submission grade must be numeric, not a boolean or coerced string.")
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("A submission grade must be finite and between zero and one.")
        return float(value)

    @model_validator(mode="after")
    def _validate_selection(self) -> StrictSubmissionReportV2:
        identifiers = [item.submission_id for item in self.submissions]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Every submission invocation requires a distinct local identity.")
        if [item.sequence for item in self.submissions] != list(range(1, len(self.submissions) + 1)):
            raise ValueError("Submission sequence must preserve contiguous invocation order.")
        completed = [item for item in self.submissions if item.status is SubmissionStatus.COMPLETED]
        latest = completed[-1] if completed else None
        if self.selected_submission_id != (latest.submission_id if latest else None):
            raise ValueError("Selection must name the latest valid completed submission, not the best score.")
        if self.last_valid_grade != (latest.grade if latest else None):
            raise ValueError("The retained grade must match the selected submission.")
        success = latest is not None and latest.grade == 1 and latest.behavior_outcome is SubmissionBehaviorOutcome.PASS
        if self.full_success != success:
            raise ValueError("Full success requires the selected returned PASS with grade one.")
        uncertain = any(
            item.status in {SubmissionStatus.ERROR, SubmissionStatus.UNKNOWN, SubmissionStatus.CANCELLED}
            or (
                item.dispatch_state is not SubmissionDispatchState.NOT_DISPATCHED
                and (
                    item.evidence_completeness is not SubmissionEvidenceCompleteness.COMPLETE
                    or item.cleanup_status in {SubmissionCleanupStatus.UNKNOWN, SubmissionCleanupStatus.FAILED}
                    or item.remote_disposition is SubmissionRemoteDisposition.UNKNOWN
                )
            )
            for item in self.submissions
        )
        if self.status is SubmissionReportStatus.COMPLETED and (latest is None or uncertain):
            raise ValueError("A clean completed report requires a valid selection and complete acquisition.")
        if self.status is SubmissionReportStatus.NO_SUBMISSION and (latest is not None or uncertain):
            raise ValueError("No-submission cannot hide a valid observation or terminal acquisition failure.")
        return self

    def provenance(self) -> SubmissionProvenanceV2:
        """
        Copy only the validated mode pair.

        Returns:
            SubmissionProvenanceV2: The binding's explicit provenance.
        """
        return SubmissionProvenanceV2(mode=self.mode, simulated=self.simulated)


class RetainedSubmissionReportV2(BaseModel):
    """Content-anchored v2 outcome and caller-owned lifecycle observations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2] = 2
    mode: SubmissionModeV2
    simulated: bool = Field(strict=True)
    evidence_label: Literal["OFFLINE/SIMULATED", "REAL/UNSIMULATED"]
    run_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    status: SubmissionReportStatus
    report: StrictSubmissionReportV2 | None
    calls: tuple[SubmissionCallEvidence, ...]
    final_text: str | None
    runner_error: str | None
    termination_reason: SubmissionTerminationReason | None = None
    generation_count: int = Field(default=0, ge=0, strict=True)
    provider_request_count: int = Field(default=0, ge=0, strict=True)
    message_count: int = Field(default=0, ge=0, strict=True)
    total_tokens: int | None = Field(default=None, ge=0, strict=True)
    token_usage: tuple[dict[str, int] | None, ...] = ()
    termination_limit: str | None = None
    limits: dict[str, int | float] = Field(default_factory=dict)
    target_identifier: dict[str, JsonValue] | None = None
    environment_audit: JsonValue = None
    local_cleanup: SubmissionCleanupStatus
    lifecycle_errors: tuple[str, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _default_evidence_label(cls, value: Any) -> Any:
        if isinstance(value, dict) and "evidence_label" not in value:
            provenance = SubmissionProvenanceV2(mode=value.get("mode"), simulated=value.get("simulated"))
            value = {**value, "evidence_label": provenance.evidence_label}
        return value

    @model_validator(mode="after")
    def _validate_run(self) -> RetainedSubmissionReportV2:
        provenance = SubmissionProvenanceV2(mode=self.mode, simulated=self.simulated)
        if self.evidence_label != provenance.evidence_label:
            raise ValueError("The v2 evidence label must match its explicit provenance.")
        if self.report is not None and self.report.provenance() != provenance:
            raise ValueError("Outer and binding v2 provenance must match.")
        call_ids = [call.call_id for call in self.calls]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Provider tool call IDs must not be reused.")
        if self.status in {SubmissionReportStatus.COMPLETED, SubmissionReportStatus.NO_SUBMISSION}:
            if self.report is None or self.status != self.report.status or self.runner_error is not None:
                raise ValueError("A clean run requires a matching valid binding report.")
            if self.local_cleanup not in {SubmissionCleanupStatus.COMPLETE, SubmissionCleanupStatus.NOT_REQUIRED}:
                raise ValueError("A clean run requires known caller-owned cleanup.")
            if self.lifecycle_errors:
                raise ValueError("Lifecycle errors prevent a clean run.")
        if self.status is SubmissionReportStatus.COMPLETED and any(
            call.feedback_kind in {SubmissionFeedbackKind.TERMINAL_ERROR, SubmissionFeedbackKind.CANCELLED}
            for call in self.calls
        ):
            raise ValueError("Terminal tool evidence prevents a clean completed run.")
        known = {item.submission_id for item in self.report.submissions} if self.report else set()
        if any(set(call.submission_ids) - known for call in self.calls):
            raise ValueError("Call correlation references an unknown submission.")
        json.dumps(self.environment_audit, allow_nan=False)
        return self

    def canonical_json(self) -> str:
        """
        Serialize exact v2 evidence deterministically.

        Returns:
            str: Canonical JSON without nonfinite values.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sha256(self) -> str:
        """
        Hash the exact canonical v2 report.

        Returns:
            str: Lowercase SHA256.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
