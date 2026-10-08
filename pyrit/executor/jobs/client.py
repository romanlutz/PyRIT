# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Framework/CoPyRIT HTTP job client; credentials and transport remain caller-configured."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.models.evaluation_job import (
    EvaluationControlReceipt,
    EvaluationControlRequest,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationJobSnapshot,
    EvaluationJobSubmission,
)

if TYPE_CHECKING:
    from uuid import UUID

    import httpx


class EvaluationJobHttpClient:
    """Use an explicitly supplied authenticated client, never discover or provision a platform."""

    def __init__(self, *, client: httpx.AsyncClient, actor_id: str) -> None:
        """
        Preserve caller ownership of authentication, compatibility headers, and client closure.

        Raises:
            ValueError: If transport is not HTTPS or explicitly local.
        """
        endpoint = client.base_url
        if (
            not actor_id
            or endpoint.scheme not in {"https", "http"}
            or (endpoint.scheme == "http" and endpoint.host not in {"localhost", "127.0.0.1", "::1"})
        ):
            raise ValueError("Job HTTP clients require an explicit authenticated HTTPS or loopback endpoint.")
        self._client = client
        self._actor_id = actor_id

    async def catalog_async(self) -> tuple[EvaluationJobRegistration, ...]:
        """
        Read only server-installed source/runtime capabilities.

        Returns:
            tuple[EvaluationJobRegistration, ...]: Finite reviewed registrations.
        """
        response = await self._client.get("/api/evaluation-jobs/catalog")
        response.raise_for_status()
        from pydantic import TypeAdapter

        return TypeAdapter(tuple[EvaluationJobRegistration, ...]).validate_json(response.content)

    async def submit_async(self, *, request: EvaluationJobRequest, actor_id: str) -> EvaluationJobSubmission:
        """
        Submit immutable references; server authentication, not this actor string, grants authority.

        Returns:
            EvaluationJobSubmission: ACK only, with exact returned request identity.

        Raises:
            EvaluationJobError: If a caller or returned immutable identity differs.
        """
        self._authorize(actor_id)
        request = EvaluationJobRequest.model_validate(request)
        response = await self._client.post(
            "/api/evaluation-jobs", content=request.model_dump_json(), headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        result = EvaluationJobSubmission.model_validate_json(response.content)
        if result.job_id != request.job_id or result.request_sha256 != request.request_sha256:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return result

    async def status_async(self, *, job_id: UUID, actor_id: str, after_sequence: int = 0) -> EvaluationJobSnapshot:
        """
        Read ordered evidence state independently of any broker settlement.

        Returns:
            EvaluationJobSnapshot: A strictly validated actor-bound status page.
        """
        self._authorize(actor_id)
        response = await self._client.get(f"/api/evaluation-jobs/{job_id}", params={"after_sequence": after_sequence})
        response.raise_for_status()
        return self._snapshot(response=response, job_id=job_id)

    async def cancel_async(self, *, job_id: UUID, actor_id: str) -> EvaluationJobSnapshot:
        """
        Request cancellation without treating local HTTP interruption as runtime closure.

        Returns:
            EvaluationJobSnapshot: Server-observed cancellation or terminal state.
        """
        self._authorize(actor_id)
        response = await self._client.post(f"/api/evaluation-jobs/{job_id}/cancel")
        response.raise_for_status()
        return self._snapshot(response=response, job_id=job_id)

    async def control_async(
        self, *, job_id: UUID, actor_id: str, capability: str, command: EvaluationControlRequest
    ) -> EvaluationControlReceipt:
        """
        Send bounded data to a reviewed wait boundary, never shell or Task source.

        Returns:
            EvaluationControlReceipt: Command acceptance, not proof of agent action.

        Raises:
            EvaluationJobError: If the server returns a foreign command receipt.
        """
        self._authorize(actor_id)
        command = EvaluationControlRequest.model_validate(command)
        response = await self._client.post(
            f"/api/evaluation-jobs/{job_id}/control",
            content=command.model_dump_json(),
            headers={"Content-Type": "application/json", "X-PyRIT-Job-Control": capability},
        )
        response.raise_for_status()
        result = EvaluationControlReceipt.model_validate_json(response.content)
        if result.job_id != job_id or result.command_id != command.command_id:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return result

    @staticmethod
    def _snapshot(*, response: httpx.Response, job_id: UUID) -> EvaluationJobSnapshot:
        result = EvaluationJobSnapshot.model_validate_json(response.content)
        if result.request.job_id != job_id:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return result

    def _authorize(self, actor_id: str) -> None:
        if actor_id != self._actor_id:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)
