# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Trusted-host installation of one opaque original case on the existing application job path."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from pyrit.backend.models.evaluation_jobs import EvaluationCaseLaunch, EvaluationOriginalResult
from pyrit.executor.jobs.inspect import (
    ApprovedOriginalInspectBinding,
    OriginalInspectArtifactReadback,
    OriginalInspectArtifactWriter,
)
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.executor.jobs.remote import create_remote_job_port_async
from pyrit.models.evaluation_job import EvaluationJobRequest

if TYPE_CHECKING:
    from pathlib import Path

    import httpx
    from fastapi import FastAPI

    from pyrit.executor.jobs.worker_auth import EvaluationWorkerCredentialProvider
    from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpSettings
    from pyrit.memory import MemoryInterface
    from pyrit.models.evaluation_job import EvaluationJobSnapshot, EvaluationJobSubmission


def _validate_case_admission(*, case_ref: str, allowed_actor_ids: frozenset[str]) -> None:
    if (
        re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", case_ref) is None
        or not allowed_actor_ids
        or len(allowed_actor_ids) > 16
        or any(not 1 <= len(actor) <= 128 for actor in allowed_actor_ids)
    ):
        raise ValueError("An application job requires a bounded opaque case and explicit actor ACL.")


class EvaluationJobApplication:
    """An explicit actor/case ACL over one installed port, not a private catalog."""

    def __init__(
        self,
        *,
        port: LocalEvaluationJobPort,
        writer: OriginalInspectArtifactWriter,
        case_ref: str,
        allowed_actor_ids: frozenset[str],
    ) -> None:
        """Bind one opaque alias and the exact original canonical writer owned by trusted startup."""
        _validate_case_admission(case_ref=case_ref, allowed_actor_ids=allowed_actor_ids)
        if not isinstance(port, LocalEvaluationJobPort) or port.registry.registrations != (writer.registration,):
            raise ValueError("An application job requires one installed original binding and bounded actor/case ACL.")
        self.port = port
        self.writer = writer
        self.case_ref = case_ref
        self._allowed_actors = allowed_actor_ids

    def authorize_actor(self, actor_id: str) -> None:
        """Recheck private-case admission on every launch, status, cancel and retained read."""
        if actor_id not in self._allowed_actors:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)
        self.port.authorize_actor(actor_id)

    async def submit_async(
        self, *, case_ref: str, launch: EvaluationCaseLaunch, actor_id: str
    ) -> EvaluationJobSubmission:
        """
        Resolve immutable execution references without accepting producer-selected profiles or code.

        Returns:
            EvaluationJobSubmission: Existing durable, idempotent queue acknowledgment.
        """
        self.authorize_actor(actor_id)
        if case_ref != self.case_ref:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_FOUND)
        launch = EvaluationCaseLaunch.model_validate(launch)
        job_id = uuid5(NAMESPACE_URL, f"pyrit:approved-original-job:{actor_id}:{case_ref}:{launch.request_id}")
        registration = self.writer.registration
        request = EvaluationJobRequest(
            job_id=job_id,
            run_id=uuid5(job_id, "run"),
            attempt_id=uuid5(job_id, "attempt"),
            runtime=registration.runtime,
            source=registration.source,
            case_id=registration.case_id,
            execution_profile_sha256=registration.execution_profile_sha256,
        )
        return await self.port.submit_async(request=request, actor_id=actor_id)

    async def readback_async(
        self, *, job_id: UUID, actor_id: str
    ) -> tuple[EvaluationJobSnapshot, OriginalInspectArtifactReadback]:
        """
        Reauthorize and verify only this job's already-persisted original evidence.

        Returns:
            tuple[EvaluationJobSnapshot, OriginalInspectArtifactReadback]: Separate job state and canonical source.
        """
        self.authorize_actor(actor_id)
        snapshot = await self.port.status_async(job_id=job_id, actor_id=actor_id)
        if snapshot.canonical is None or snapshot.manifest is None:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        retained = await self.writer.readback_async(
            request=snapshot.request, manifest=snapshot.manifest, canonical=snapshot.canonical
        )
        return snapshot, retained

    async def result_async(self, *, job_id: UUID, actor_id: str) -> EvaluationOriginalResult:
        """
        Return a read-only grade view without exposing private Task/scorer implementation metadata.

        Returns:
            EvaluationOriginalResult: Original scalar/status, distinct attack outcome and exact source provenance.
        """
        snapshot, retained = await self.readback_async(job_id=job_id, actor_id=actor_id)
        if len(retained.imported.case_results) != 1:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        result = retained.imported.case_results[0]
        metadata = result.score.score_metadata or {}
        event_id = metadata.get("inspect_final_score_event_id")
        event_sha = metadata.get("inspect_final_score_event_sha256")
        assert snapshot.canonical is not None
        return EvaluationOriginalResult(
            job_id=job_id,
            run_id=snapshot.request.run_id,
            case_ref=self.case_ref,
            state=snapshot.state,
            evidence=snapshot.evidence,
            cleanup=snapshot.cleanup,
            canonical=snapshot.canonical,
            archive_bytes=len(retained.archive),
            original_score=result.score.score_value,
            pyrit_score_status=result.score.status,
            pyrit_attack_outcome=result.attack_result.outcome,
            final_score_event_id=event_id if isinstance(event_id, str) else None,
            final_score_event_sha256=event_sha if isinstance(event_sha, str) else None,
        )


class EvaluationJobApplicationFactory(Protocol):
    """One host-installed composition callable, never selected from a request or remote catalog."""

    async def __call__(self, *, memory: MemoryInterface) -> EvaluationJobApplication:
        """Construct the unstarted application facade using the API's canonical memory."""
        ...


def install_evaluation_job_application(*, app: FastAPI, factory: EvaluationJobApplicationFactory) -> None:
    """Install trusted composition before application lifespan; no string import or environment discovery."""
    if (
        not callable(factory)
        or getattr(app.state, "runtime_lifecycle", None) is not None
        or getattr(app.state, "evaluation_job_application_factory", None) is not None
    ):
        raise ValueError("Install one trusted evaluation job application factory before lifespan startup.")
    app.state.evaluation_job_application_factory = factory


async def create_remote_original_evaluation_application_async(
    *,
    memory: MemoryInterface,
    root: Path,
    case_ref: str,
    binding: ApprovedOriginalInspectBinding,
    allowed_actor_ids: frozenset[str],
    settings: EvaluationWorkerHttpSettings,
    credentials: EvaluationWorkerCredentialProvider,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EvaluationJobApplication:
    """
    Reuse the generic gateway with a metadata-bound original writer, never a native grader.

    Returns:
        EvaluationJobApplication: One unstarted, opaque original-case application installation.
    """
    _validate_case_admission(case_ref=case_ref, allowed_actor_ids=allowed_actor_ids)
    writer = OriginalInspectArtifactWriter(memory=memory, binding=binding)
    port = await create_remote_job_port_async(
        root=root,
        registration=binding.registration,
        writer=writer,
        allowed_actor_ids=allowed_actor_ids,
        settings=settings,
        credentials=credentials,
        transport=transport,
    )
    return EvaluationJobApplication(port=port, writer=writer, case_ref=case_ref, allowed_actor_ids=allowed_actor_ids)
