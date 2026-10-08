# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public producer, consumer, runtime, and canonical-writer ports."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pyrit.models.evaluation_job import EvaluationArtifactManifest, EvaluationCleanupState, EvaluationJobRegistration

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from uuid import UUID

    from pyrit.models.evaluation_job import (
        EvaluationCanonicalReceipt,
        EvaluationControlReceipt,
        EvaluationControlRequest,
        EvaluationDeliveryReceipt,
        EvaluationJobDelivery,
        EvaluationJobRequest,
        EvaluationJobSnapshot,
        EvaluationJobSubmission,
        EvaluationWaitBoundary,
    )


class EvaluationJobErrorCode(str, Enum):
    """Safe finite errors, without private payloads or source exceptions."""

    NOT_AUTHORIZED = "not_authorized"
    NOT_FOUND = "not_found"
    REQUEST_CONFLICT = "request_conflict"
    UNSUPPORTED_RUNTIME = "unsupported_runtime"
    UNSUPPORTED_CONTROL = "unsupported_control"
    STALE_BOUNDARY = "stale_boundary"
    ORDER_CONFLICT = "order_conflict"
    CANCEL_TOO_LATE = "cancel_too_late"
    DISPATCH_UNCERTAIN = "dispatch_uncertain"
    ARTIFACT_MISMATCH = "artifact_mismatch"
    RUNTIME_FAILED = "runtime_failed"
    WRITER_FAILED = "writer_failed"
    CLOSED = "closed"


class EvaluationJobError(ValueError):
    """A visible refusal or runtime failure using only a public error code."""

    def __init__(self, code: EvaluationJobErrorCode) -> None:
        """Keep the original failure private while exposing its finite classification."""
        super().__init__(code.value)
        self.code = code


class EvaluationCanonicalSettlementError(EvaluationJobError):
    """API-owned canonical persistence succeeded but remote settlement remains uncertain."""

    def __init__(self, canonical: EvaluationCanonicalReceipt) -> None:
        """Preserve real canonical references without falsely declaring settled closure."""
        super().__init__(EvaluationJobErrorCode.DISPATCH_UNCERTAIN)
        self.canonical = canonical


@dataclass(frozen=True, kw_only=True)
class EvaluationRuntimeArtifacts:
    """Exact source artifacts after runtime closure, never a worker database."""

    manifest: EvaluationArtifactManifest
    payloads: tuple[tuple[str, bytes], ...]
    cleanup: EvaluationCleanupState

    def verify(self) -> None:
        """
        Verify the full named inventory, immutable byte bindings, and aggregate bounds.

        Raises:
            EvaluationJobError: If source bytes or artifact identity differ.
        """
        manifest = EvaluationArtifactManifest.model_validate(self.manifest)
        if len({name for name, _ in self.payloads}) != len(self.payloads):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        payloads = dict(self.payloads)
        if set(payloads) != {item.name for item in manifest.artifacts}:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        for item in manifest.artifacts:
            content = payloads[item.name]
            if (
                not isinstance(content, bytes)
                or len(content) != item.bytes
                or hashlib.sha256(content).hexdigest() != (item.sha256)
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)


class EvaluationRuntimeCancelled(asyncio.CancelledError):
    """A runtime explicitly observed closure before propagating cancellation."""

    def __init__(self, cleanup: EvaluationCleanupState) -> None:
        """Keep cancellation distinct from source completeness or a grade."""
        super().__init__("evaluation_runtime_cancelled")
        self.cleanup = cleanup


class EvaluationRuntimeError(EvaluationJobError):
    """A finite failure with separately observed runtime-owned closure."""

    def __init__(self, *, code: EvaluationJobErrorCode, cleanup: EvaluationCleanupState) -> None:
        """Do not turn observed closure into a source grade or expose exception payloads."""
        super().__init__(code)
        self.cleanup = cleanup


class EvaluationRuntimeContext(Protocol):
    """Fenced execution context provided by an orchestrator, not by a producer."""

    @property
    def request(self) -> EvaluationJobRequest:
        """The admitted request, never an execution-time input override."""
        ...

    @property
    def fence_id(self) -> UUID:
        """The independently committed local dispatch incarnation."""
        ...

    @property
    def actor_id(self) -> str:
        """The internally admitted actor, never a browser-supplied runtime identity."""
        ...

    @property
    def run_root(self) -> Path:
        """The owner-selected local artifact directory."""
        ...

    async def wait_for_control_async(self, *, boundary: EvaluationWaitBoundary) -> EvaluationControlRequest:
        """Deliver at most one admitted command at this reviewed wait boundary."""
        ...


class EvaluationJobRuntime(Protocol):
    """Runtime registration owns source execution and actual closure."""

    registration: EvaluationJobRegistration

    async def execute_async(self, *, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        """Execute once, returning original evidence only after owned work has drained."""
        ...


class EvaluationJobRuntimeRegistry:
    """Explicit server installation, not automatic engine or private-source discovery."""

    def __init__(self, runtimes: Sequence[EvaluationJobRuntime]) -> None:
        """
        Freeze a finite, unambiguous set of runtime bindings.

        Raises:
            EvaluationJobError: If two implementations claim the same execution identity.
        """
        self._runtimes = tuple(runtimes)
        self._registrations = tuple(
            EvaluationJobRegistration.model_validate(runtime.registration) for runtime in self._runtimes
        )
        keys: set[tuple[str, str, str, str]] = set()
        for registration in self._registrations:
            key = (
                registration.runtime.value,
                registration.source.source_fingerprint,
                registration.case_id,
                registration.execution_profile_sha256,
            )
            if key in keys:
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            keys.add(key)

    @property
    def registrations(self) -> tuple[EvaluationJobRegistration, ...]:
        """Only installed source/profile capabilities, never private Task configuration."""
        return self._registrations

    def resolve(self, request: EvaluationJobRequest) -> EvaluationJobRuntime:
        """
        Refuse kinds or sources without a reviewed installed implementation.

        Returns:
            EvaluationJobRuntime: The exact server-owned runtime binding.

        Raises:
            EvaluationJobError: If this source, profile, or capability is not installed.
        """
        for registration, runtime in zip(self._registrations, self._runtimes, strict=True):
            if registration.accepts(request):
                return runtime
        raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_RUNTIME)


class EvaluationArtifactWriter(Protocol):
    """The canonical API writer imports exact evidence, never worker SQLite rows."""

    async def import_async(
        self, *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        """Validate source artifacts and return actual canonical references without rescoring."""
        ...


class EvaluationJobPort(Protocol):
    """Producer-facing job API usable from CLI, framework, and an opt-in backend."""

    async def submit_async(self, *, request: EvaluationJobRequest, actor_id: str) -> EvaluationJobSubmission:
        """Accept immutable work without interpreting acceptance as a grade."""
        ...

    async def status_async(self, *, job_id: UUID, actor_id: str, after_sequence: int = 0) -> EvaluationJobSnapshot:
        """Read actor-bound state and the next ordered event page."""
        ...

    async def cancel_async(self, *, job_id: UUID, actor_id: str) -> EvaluationJobSnapshot:
        """Request cancellation without treating uncertain closure as completion."""
        ...

    async def control_async(
        self,
        *,
        job_id: UUID,
        actor_id: str,
        capability: str,
        command: EvaluationControlRequest,
    ) -> EvaluationControlReceipt:
        """Admit a bounded structured command only at a current reviewed boundary."""
        ...


class EvaluationJobReceiver(Protocol):
    """Broker delivery is separate from producer admission and canonical publication."""

    async def receive_async(self, *, delivery: EvaluationJobDelivery) -> EvaluationDeliveryReceipt:
        """Claim previously admitted work once; duplicate delivery never executes again."""
        ...
