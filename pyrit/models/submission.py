# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Versioned offline submission reports, not a remote-job or observation schema."""

from __future__ import annotations

import hashlib
import json
import math
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


class SubmissionReportStatus(str, Enum):
    """Overall acquisition status, independent of a retained numeric observation."""

    NO_SUBMISSION = "no_submission"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    ERROR = "error"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"


class SubmissionStatus(str, Enum):
    """One binding-owned submission invocation."""

    REJECTED = "rejected"
    COMPLETED = "completed"
    ERROR = "error"
    UNKNOWN = "unknown"
    CANCELLED = "cancelled"
    GUARDED = "guarded"


class SubmissionDispatchState(str, Enum):
    """What was observed about dispatch, not a guarantee about a remote worker."""

    NOT_DISPATCHED = "not_dispatched"
    DISPATCHED = "dispatched"
    RETURNED = "returned"


class SubmissionBehaviorOutcome(str, Enum):
    """The binding's acquired behavior judgment."""

    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"


class SubmissionEvidenceCompleteness(str, Enum):
    """Whether acquired evidence is complete."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class SubmissionAcceptance(str, Enum):
    """Observed acceptance, without inventing remote receipt identifiers."""

    NOT_DISPATCHED = "not_dispatched"
    UNKNOWN = "unknown"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class SubmissionRemoteDisposition(str, Enum):
    """Known remote disposition; cancelling an await alone does not imply cancellation."""

    NOT_DISPATCHED = "not_dispatched"
    UNKNOWN = "unknown"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class SubmissionCleanupStatus(str, Enum):
    """Cleanup observation, independent of grade."""

    NOT_REQUIRED = "not_required"
    UNKNOWN = "unknown"
    COMPLETE = "complete"
    FAILED = "failed"


class SubmissionCapabilities(BaseModel):
    """The deliberately absent remote-control capabilities of this contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    remote_query: Literal[False] = False
    remote_cancel: Literal[False] = False
    idempotency: Literal[False] = False


def _finite_grade(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not isinstance(value, (int, float)):
        raise ValueError("A submission grade must be a number, not a boolean or coerced string.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("A submission grade must be finite and between zero and one.")
    return float(value)


class StrictSubmissionRecord(BaseModel):
    """One immutable projection of binding-owned submission evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    submission_id: str = Field(min_length=1)
    sequence: int = Field(ge=1, strict=True)
    artifact_sha256: str | None
    artifact_size_bytes: int | None = Field(ge=0, strict=True)
    dispatch_state: SubmissionDispatchState
    status: SubmissionStatus
    grade: float | None
    behavior_outcome: SubmissionBehaviorOutcome | None
    feedback: str
    error_code: str | None
    raw_evidence: JsonValue
    evidence_completeness: SubmissionEvidenceCompleteness
    acceptance: SubmissionAcceptance
    observed_request_id: str | None
    observed_receipt_id: str | None
    remote_disposition: SubmissionRemoteDisposition
    cleanup_status: SubmissionCleanupStatus

    @field_validator("grade", mode="before")
    @classmethod
    def _validate_grade(cls, value: object) -> float | None:
        return _finite_grade(value)

    @model_validator(mode="after")
    def _validate_evidence(self) -> StrictSubmissionRecord:
        self._validate_dispatch_evidence()
        if (self.artifact_sha256 is None) != (self.artifact_size_bytes is None):
            raise ValueError("Artifact digest and byte count must be present together.")
        if self.artifact_sha256 is not None and (
            len(self.artifact_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.artifact_sha256)
        ):
            raise ValueError("Artifact digest must be a lowercase SHA256.")
        if self.status is SubmissionStatus.COMPLETED:
            if (
                self.dispatch_state is not SubmissionDispatchState.RETURNED
                or self.grade is None
                or self.behavior_outcome is None
                or self.artifact_sha256 is None
            ):
                raise ValueError("A completed file submission requires acquired grade and artifact evidence.")
            if self.acceptance not in {SubmissionAcceptance.ACCEPTED, SubmissionAcceptance.UNKNOWN}:
                raise ValueError("A completed behavior observation cannot have rejected or absent acceptance.")
            if self.remote_disposition not in {
                SubmissionRemoteDisposition.COMPLETED,
                SubmissionRemoteDisposition.UNKNOWN,
            }:
                raise ValueError("A completed behavior observation cannot be undispatched or remotely cancelled.")
        elif self.grade is not None or self.behavior_outcome is not None:
            raise ValueError("An uncompleted submission cannot fabricate a behavior grade.")
        if (
            self.status in {SubmissionStatus.REJECTED, SubmissionStatus.GUARDED}
            and self.dispatch_state is not SubmissionDispatchState.NOT_DISPATCHED
        ):
            raise ValueError("Rejected or guarded calls must not dispatch an evaluator.")
        json.dumps(self.raw_evidence, allow_nan=False)
        return self

    def _validate_dispatch_evidence(self) -> None:
        if self.dispatch_state is SubmissionDispatchState.NOT_DISPATCHED:
            if (
                self.acceptance is not SubmissionAcceptance.NOT_DISPATCHED
                or self.remote_disposition is not SubmissionRemoteDisposition.NOT_DISPATCHED
                or self.observed_request_id is not None
                or self.observed_receipt_id is not None
            ):
                raise ValueError("An undispatched submission cannot claim remote acceptance, disposition, or IDs.")
        elif (
            self.acceptance is SubmissionAcceptance.NOT_DISPATCHED
            or self.remote_disposition is SubmissionRemoteDisposition.NOT_DISPATCHED
        ):
            raise ValueError("Dispatched or returned evidence cannot carry a not_dispatched remote state.")


class StrictSubmissionReport(BaseModel):
    """Validate the shared strict-submission-v1 projection without owning its state machine."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    contract_version: Literal["strict-submission-v1"]
    mode: Literal["offline"]
    simulated: Literal[True]
    status: SubmissionReportStatus
    last_valid_grade: float | None
    selected_submission_id: str | None
    full_success: bool = Field(strict=True)
    submissions: tuple[StrictSubmissionRecord, ...]
    capabilities: SubmissionCapabilities

    @field_validator("last_valid_grade", mode="before")
    @classmethod
    def _validate_grade(cls, value: object) -> float | None:
        return _finite_grade(value)

    @model_validator(mode="after")
    def _validate_selection(self) -> StrictSubmissionReport:
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
        terminal = any(
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
        if self.status is SubmissionReportStatus.COMPLETED and (latest is None or terminal):
            raise ValueError("A clean completed report requires a valid selection and complete acquisition.")
        if self.status is SubmissionReportStatus.NO_SUBMISSION and (latest is not None or terminal):
            raise ValueError("No-submission cannot hide a valid observation or terminal acquisition failure.")
        return self


class SubmissionFeedbackKind(str, Enum):
    """How a real tool call ended."""

    RETURNED = "returned"
    RECOVERABLE_ERROR = "recoverable_error"
    TERMINAL_ERROR = "terminal_error"
    CANCELLED = "cancelled"


class SubmissionTerminationReason(str, Enum):
    """Why the local task loop stopped, separate from the binding report."""

    FULL_SUCCESS = "full_success"
    BUDGET = "budget"
    BINDING_STATE = "binding_state"
    RUNNER_ERROR = "runner_error"


class SubmissionCallEvidence(BaseModel):
    """Actual provider call correlation, separate from binding submission identities."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    arguments_json: str
    feedback: str | None
    feedback_kind: SubmissionFeedbackKind
    submission_ids: tuple[str, ...]


class RetainedSubmissionReport(BaseModel):
    """A versioned offline runner report suitable for content-anchored scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    evidence_label: Literal["OFFLINE/SIMULATED"] = "OFFLINE/SIMULATED"
    run_id: str = Field(min_length=1)
    conversation_id: str = Field(min_length=1)
    status: SubmissionReportStatus
    report: StrictSubmissionReport | None
    calls: tuple[SubmissionCallEvidence, ...]
    final_text: str | None
    runner_error: str | None
    termination_reason: SubmissionTerminationReason | None = None
    generation_count: int = Field(default=0, ge=0, strict=True)
    limits: dict[str, int | float] = Field(default_factory=dict)
    local_cleanup: Literal["not_required"] = "not_required"

    @model_validator(mode="after")
    def _validate_run(self) -> RetainedSubmissionReport:
        call_ids = [call.call_id for call in self.calls]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError("Provider tool call IDs must not be reused.")
        if self.status in {SubmissionReportStatus.COMPLETED, SubmissionReportStatus.NO_SUBMISSION} and (
            self.report is None or self.status != self.report.status or self.runner_error is not None
        ):
            raise ValueError("A clean run requires a matching valid binding report.")
        if self.status is SubmissionReportStatus.COMPLETED and any(
            call.feedback_kind in {SubmissionFeedbackKind.TERMINAL_ERROR, SubmissionFeedbackKind.CANCELLED}
            for call in self.calls
        ):
            raise ValueError("Terminal tool evidence prevents a clean completed run.")
        known = {item.submission_id for item in self.report.submissions} if self.report else set()
        if any(set(call.submission_ids) - known for call in self.calls):
            raise ValueError("Call correlation references an unknown submission.")
        return self

    def canonical_json(self) -> str:
        """
        Serialize the report deterministically for immutable retention.

        Returns:
            str: Canonical JSON, without nonfinite values.
        """
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), allow_nan=False)

    def sha256(self) -> str:
        """
        Hash the exact canonical report contents.

        Returns:
            str: Lowercase SHA256.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
