# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public, harness-neutral identities for task-owned evaluation cases."""

from __future__ import annotations

import uuid  # noqa: TC003  (runtime-required by Pydantic field annotations)
from enum import Enum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pyrit.models.identifiers.component_identifier import config_hash
from pyrit.models.results.attack_result import AttackOutcome  # noqa: TC001  (runtime-required by Pydantic)

if TYPE_CHECKING:
    from collections.abc import Mapping

_PublicName = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")]
_Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class EvalSourceKind(str, Enum):
    """How a trusted loader selected the evaluation package."""

    NAMED = "named"
    TRUSTED_LOCAL = "trusted_local"


class _FrozenRef(BaseModel):
    """Validation shared by public evaluation references."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class EvalPackageRef(_FrozenRef):
    """Public alias and content digest of a trusted evaluation source."""

    kind: EvalSourceKind
    name: _PublicName
    source_sha256: _Sha256

    @property
    def source_fingerprint(self) -> str:
        """The source identity, independent of execution and input variants."""
        return config_hash({"kind": self.kind.value, "name": self.name, "source_sha256": self.source_sha256})


class HarnessProfileRef(_FrozenRef):
    """Public harness profile alias and configuration digest."""

    name: _PublicName
    config_sha256: _Sha256


class ModelRouteRef(_FrozenRef):
    """Public model route alias and configuration digest, never credentials."""

    name: _PublicName
    config_sha256: _Sha256


class InputVariantRef(_FrozenRef):
    """Digest of one approved replacement for one case's input surface."""

    case_id: _Sha256
    surface_id: _PublicName
    content_sha256: _Sha256


class EvalCaseRef(_FrozenRef):
    """Canonical source Task/Sample identity, independent of execution profile."""

    package: EvalPackageRef
    task_name: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    epoch: int = Field(ge=0)

    @field_validator("task_name", "task_version", "sample_id")
    @classmethod
    def _validate_nonblank(cls, value: str) -> str:
        """
        Reject empty identifiers without silently normalizing their identity.

        Returns:
            str: The unchanged identifier.

        Raises:
            ValueError: If the identifier is blank.
        """
        if not value.strip():
            raise ValueError("Task, version, and sample identifiers cannot be blank")
        return value

    @property
    def case_id(self) -> str:
        """The source-only identity of this Task/Sample/epoch."""
        return config_hash(
            {
                "schema": 1,
                "source": self.package.source_fingerprint,
                "task": self.task_name,
                "version": self.task_version,
                "sample": self.sample_id,
                "epoch": self.epoch,
            }
        )


class EvalSpecRef(_FrozenRef):
    """Stable public configuration of an evaluation, not an execution instance."""

    package: EvalPackageRef
    harness: HarnessProfileRef
    model_route: ModelRouteRef
    input_variant: InputVariantRef | None = None

    @property
    def spec_sha256(self) -> str:
        """The configuration fingerprint; identical fresh runs share it."""
        return config_hash({"schema": 1, "spec": self.model_dump(mode="json")})


class EvalRunRef(_FrozenRef):
    """One fresh invocation of a stable evaluation specification."""

    spec: EvalSpecRef
    run_instance_id: uuid.UUID

    def case_run_id(self, *, case: EvalCaseRef) -> str:
        """
        Derive a distinct work-unit identity for this case in this run.

        Returns:
            str: The run-specific case fingerprint.

        Raises:
            ValueError: If the case belongs to another source or input variant.
        """
        if case.package != self.spec.package:
            raise ValueError("Eval case belongs to a different package than this run")
        variant = self.spec.input_variant
        if variant is not None and variant.case_id != case.case_id:
            raise ValueError("Input variant targets a different Eval case")
        return config_hash(
            {
                "schema": 1,
                "run_instance_id": str(self.run_instance_id),
                "spec_sha256": self.spec.spec_sha256,
                "case_id": case.case_id,
                "input_variant_sha256": variant.content_sha256 if variant else None,
                "input_surface_id": variant.surface_id if variant else None,
            }
        )


class EvalScoreRole(str, Enum):
    """Named purpose of a task-owned score, distinct from its numeric verdict."""

    BENCHMARK_ORIGINAL = "benchmark_original"
    PROGRESS_SIGNAL = "progress_signal"
    SUPPLEMENTAL_FINAL = "supplemental_final"


class EvalScoreProvenance(_FrozenRef):
    """PyRIT Score creator and case label, not external scorer attestation."""

    role: EvalScoreRole
    case_run_id: _Sha256
    pyrit_scorer_hash: _Sha256

    def to_metadata(self) -> dict[str, str]:
        """
        Serialize the Score metadata fields owned by task-owned evaluation.

        Returns:
            dict[str, str]: Role, case-run ID, and PyRIT Score creator hash.
        """
        return {
            "pyrit_eval_role": self.role.value,
            "pyrit_eval_case_run_id": self.case_run_id,
            "pyrit_eval_pyrit_scorer_hash": self.pyrit_scorer_hash,
        }

    @classmethod
    def from_metadata(cls, *, metadata: Mapping[str, str | int | float] | None) -> EvalScoreProvenance:
        """
        Recover typed provenance, rejecting unlabeled or incomplete scores.

        Returns:
            EvalScoreProvenance: The committed score's provenance.

        Raises:
            ValueError: If required metadata is missing or invalid.
        """
        if metadata is None:
            raise ValueError("Score is missing task-owned evaluation provenance")
        fields = {
            "role": metadata.get("pyrit_eval_role"),
            "case_run_id": metadata.get("pyrit_eval_case_run_id"),
            "pyrit_scorer_hash": metadata.get("pyrit_eval_pyrit_scorer_hash"),
        }
        if any(not isinstance(value, str) for value in fields.values()):
            raise ValueError("Score is missing task-owned evaluation provenance")
        return cls.model_validate(fields)


class CommittedCaseExecution(_FrozenRef):
    """Trusted adapter outcome referencing an already committed original Score."""

    original_score_id: uuid.UUID
    original_score_provenance: EvalScoreProvenance
    conversation_id: uuid.UUID
    outcome: AttackOutcome
    executed_turns: int = Field(default=0, ge=0)
    execution_time_ms: int = Field(default=0, ge=0)
    outcome_reason: str | None = None
