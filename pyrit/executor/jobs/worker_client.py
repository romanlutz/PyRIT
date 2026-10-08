# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded, audience-authenticated execution transport with no remote canonical authority."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.executor.jobs.worker_auth import EvaluationWorkerAuthContext, EvaluationWorkerCredentialProvider
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactManifest,
    EvaluationControlReceipt,
    EvaluationControlRequest,
)
from pyrit.models.evaluation_worker import (
    EvaluationGatewaySettlement,
    EvaluationGatewaySettlementReceipt,
    EvaluationWorkerAdmission,
    EvaluationWorkerBinding,
    EvaluationWorkerCatalog,
    EvaluationWorkerProtocol,
    EvaluationWorkerSnapshot,
    EvaluationWorkerSubmission,
    evaluation_worker_schema_sha256,
)

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True, kw_only=True)
class EvaluationWorkerHttpSettings:
    """Explicit authority, compatibility and finite transport/settlement limits."""

    base_url: str
    audience: str
    service_id: str
    schema_sha256: str
    protocol_version: int = 1
    request_timeout_seconds: float = 10
    artifact_timeout_seconds: float = 30
    poll_interval_seconds: float = 0.25
    poll_deadline_seconds: float = 60
    settlement_timeout_seconds: float = 10
    allow_loopback_http: bool = False

    def __post_init__(self) -> None:
        """
        Validate the sole authority and finite limits before creating any transport.

        Raises:
            ValueError: If endpoint, protocol or timeout settings are unsupported.
        """
        endpoint = httpx.URL(self.base_url)
        local = endpoint.host in {"localhost", "127.0.0.1", "::1"}
        if (
            endpoint.scheme not in {"https", "http"}
            or (endpoint.scheme == "http" and not (local and self.allow_loopback_http))
            or endpoint.userinfo
            or endpoint.query
            or endpoint.fragment
            or endpoint.path != "/"
            or not endpoint.host
            or type(self.allow_loopback_http) is not bool
            or not 1 <= len(self.audience) <= 512
            or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", self.service_id)
            or self.schema_sha256 != evaluation_worker_schema_sha256()
            or type(self.protocol_version) is not int
            or self.protocol_version != 1
        ):
            raise ValueError("Remote evaluation jobs require an explicit authenticated compatible authority.")
        limits = (
            (self.request_timeout_seconds, 0.05, 30),
            (self.artifact_timeout_seconds, 0.05, 120),
            (self.poll_interval_seconds, 0.01, 5),
            (self.poll_deadline_seconds, 0.05, 600),
            (self.settlement_timeout_seconds, 0.05, 60),
        )
        if any(
            isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high
            for value, low, high in limits
        ):
            raise ValueError("Remote job request, polling and settlement limits must be finite and bounded.")


class EvaluationWorkerHttpClient:
    """Own one protected same-authority client, with no redirects, proxies or token discovery."""

    MAX_JSON_BYTES = 256 * 1024
    PREFIX = "/api/evaluation-worker/v1"

    def __init__(
        self,
        *,
        settings: EvaluationWorkerHttpSettings,
        credentials: EvaluationWorkerCredentialProvider,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """
        Bind a host-installed provider without requesting credentials at construction.

        Raises:
            ValueError: If audience or explicit fixture-only transport binding differs.
        """
        local = httpx.URL(settings.base_url).host in {"localhost", "127.0.0.1", "::1"}
        if credentials.audience != settings.audience or (
            credentials.fixture_only and not (local and settings.allow_loopback_http)
        ):
            raise ValueError("Worker credentials require the configured audience and explicit fixture authority.")
        self.settings = settings
        self.credentials = credentials
        self._client = httpx.AsyncClient(
            base_url=settings.base_url,
            timeout=settings.request_timeout_seconds,
            follow_redirects=False,
            trust_env=False,
            transport=transport,
        )

    async def protocol_async(self, *, actor_id: str) -> EvaluationWorkerProtocol:
        """
        Verify authenticated service/protocol identity without admitting execution.

        Returns:
            EvaluationWorkerProtocol: The matching installed protocol.

        Raises:
            EvaluationJobError: If service or schema identity differs.
        """
        result = await self._json_async(
            model=EvaluationWorkerProtocol, method="GET", path=f"{self.PREFIX}/protocol", actor_id=actor_id
        )
        if result.service_id != self.settings.service_id or result.schema_sha256 != self.settings.schema_sha256:
            raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_RUNTIME)
        return result

    async def catalog_async(self, *, actor_id: str) -> EvaluationWorkerCatalog:
        """
        Retrieve grants for the exact delegated actor, never a universal source catalog.

        Returns:
            EvaluationWorkerCatalog: The service-authorized grants.

        Raises:
            EvaluationJobError: If the actor or service differs.
        """
        result = await self._json_async(
            model=EvaluationWorkerCatalog, method="GET", path=f"{self.PREFIX}/catalog", actor_id=actor_id
        )
        if result.actor_id != actor_id or result.service_id != self.settings.service_id:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)
        return result

    async def submit_async(self, *, admission: EvaluationWorkerAdmission) -> EvaluationWorkerSubmission:
        """
        Send one exact durable dispatch intent without creating a retry identity.

        Returns:
            EvaluationWorkerSubmission: The bound execution admission, not a canonical result.
        """
        admission = EvaluationWorkerAdmission.model_validate(admission)
        result = await self._json_async(
            model=EvaluationWorkerSubmission,
            method="POST",
            path=f"{self.PREFIX}/jobs",
            actor_id=admission.actor_id,
            admission=admission,
            content=admission.model_dump_json().encode("utf-8"),
        )
        self._verify_binding(admission=admission, binding=result.binding)
        return result

    async def status_async(
        self,
        *,
        admission: EvaluationWorkerAdmission,
        binding: EvaluationWorkerBinding | None,
        after_sequence: int,
    ) -> EvaluationWorkerSnapshot:
        """
        Read only the admitted worker binding and its independent cursor.

        Returns:
            EvaluationWorkerSnapshot: Strict execution-only status.
        """
        return await self._snapshot_async(
            method="GET",
            path=f"{self.PREFIX}/jobs/{admission.request.job_id}?after_sequence={after_sequence}",
            admission=admission,
            binding=binding,
        )

    async def cancel_async(
        self, *, admission: EvaluationWorkerAdmission, binding: EvaluationWorkerBinding | None
    ) -> EvaluationWorkerSnapshot:
        """
        Request cancellation; the HTTP response alone never proves guest closure.

        Returns:
            EvaluationWorkerSnapshot: The observed execution state.
        """
        return await self._snapshot_async(
            method="POST",
            path=f"{self.PREFIX}/jobs/{admission.request.job_id}/cancel",
            admission=admission,
            binding=binding,
        )

    async def control_async(
        self,
        *,
        admission: EvaluationWorkerAdmission,
        binding: EvaluationWorkerBinding,
        command: EvaluationControlRequest,
    ) -> EvaluationControlReceipt:
        """
        Forward only an admitted structured command, not a shell or replacement Task.

        Returns:
            EvaluationControlReceipt: Acceptance, not agent application.

        Raises:
            EvaluationJobError: If capability or returned command identity differs.
        """
        command = EvaluationControlRequest.model_validate(command)
        if command.kind not in admission.request.controls:
            raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_CONTROL)
        result = await self._json_async(
            model=EvaluationControlReceipt,
            method="POST",
            path=f"{self.PREFIX}/jobs/{admission.request.job_id}/control",
            actor_id=admission.actor_id,
            admission=admission,
            binding=binding,
            content=command.model_dump_json().encode("utf-8"),
        )
        if result.job_id != admission.request.job_id or result.command_id != command.command_id:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return result

    async def manifest_async(
        self, *, admission: EvaluationWorkerAdmission, binding: EvaluationWorkerBinding, manifest_sha256: str
    ) -> tuple[EvaluationArtifactManifest, bytes]:
        """
        Fetch and preserve the original worker manifest verbatim under its own fence.

        Returns:
            tuple[EvaluationArtifactManifest, bytes]: Validated identity and exact original JSON bytes.

        Raises:
            EvaluationJobError: If request, worker fence or manifest identity differs.
        """
        content = await self._read_async(
            method="GET",
            path=f"{self.PREFIX}/jobs/{binding.job_id}/artifacts/{manifest_sha256}/manifest",
            actor_id=admission.actor_id,
            admission=admission,
            binding=binding,
            maximum=self.MAX_JSON_BYTES,
            media_type="application/json",
            timeout=self.settings.artifact_timeout_seconds,
        )
        try:
            manifest = EvaluationArtifactManifest.model_validate_json(content)
        except ValidationError as error:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH) from error
        if (
            manifest.request != admission.request
            or manifest.fence_id != binding.worker_fence_id
            or manifest.manifest_sha256 != manifest_sha256
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        return manifest, content

    async def artifact_async(
        self,
        *,
        admission: EvaluationWorkerAdmission,
        binding: EvaluationWorkerBinding,
        manifest_sha256: str,
        artifact: EvaluationArtifact,
    ) -> bytes:
        """
        Fetch only a validated flat inventory name and verify its complete source bytes.

        Returns:
            bytes: Exact declared artifact bytes.

        Raises:
            EvaluationJobError: If the content length or SHA differs.
        """
        artifact = EvaluationArtifact.model_validate(artifact)
        content = await self._read_async(
            method="GET",
            path=f"{self.PREFIX}/jobs/{binding.job_id}/artifacts/{manifest_sha256}/{artifact.name}",
            actor_id=admission.actor_id,
            admission=admission,
            binding=binding,
            maximum=artifact.bytes,
            media_type=artifact.media_type.value,
            timeout=self.settings.artifact_timeout_seconds,
        )
        if len(content) != artifact.bytes or hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        return content

    async def settle_async(
        self, *, admission: EvaluationWorkerAdmission, settlement: EvaluationGatewaySettlement
    ) -> EvaluationGatewaySettlementReceipt:
        """
        Acknowledge gateway retention/import without exporting any database IDs.

        Returns:
            EvaluationGatewaySettlementReceipt: Exact durable acknowledgment.

        Raises:
            EvaluationJobError: If the acknowledgment binds another handoff.
        """
        settlement = EvaluationGatewaySettlement.model_validate(settlement)
        result = await self._json_async(
            model=EvaluationGatewaySettlementReceipt,
            method="POST",
            path=f"{self.PREFIX}/jobs/{settlement.binding.job_id}/settlement",
            actor_id=admission.actor_id,
            admission=admission,
            binding=settlement.binding,
            content=settlement.model_dump_json().encode("utf-8"),
            timeout=self.settings.settlement_timeout_seconds,
        )
        if (
            result.job_id != admission.request.job_id
            or result.binding_sha256 != settlement.binding.binding_sha256
            or result.settlement_sha256 != settlement.settlement_sha256
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return result

    async def close_async(self) -> None:
        """Close the owned HTTP client before its explicitly installed credential provider."""
        try:
            await self._client.aclose()
        finally:
            await self.credentials.close_async()

    async def _snapshot_async(
        self,
        *,
        method: str,
        path: str,
        admission: EvaluationWorkerAdmission,
        binding: EvaluationWorkerBinding | None,
    ) -> EvaluationWorkerSnapshot:
        result = await self._json_async(
            model=EvaluationWorkerSnapshot,
            method=method,
            path=path,
            actor_id=admission.actor_id,
            admission=admission,
            binding=binding,
        )
        self._verify_binding(admission=admission, binding=result.binding)
        if result.request != admission.request or (binding is not None and result.binding != binding):
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return result

    def _verify_binding(self, *, admission: EvaluationWorkerAdmission, binding: EvaluationWorkerBinding) -> None:
        if binding.service_id != self.settings.service_id or not binding.accepts(admission):
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)

    async def _json_async(
        self,
        *,
        model: type[T],
        method: str,
        path: str,
        actor_id: str,
        admission: EvaluationWorkerAdmission | None = None,
        binding: EvaluationWorkerBinding | None = None,
        content: bytes = b"",
        timeout: float | None = None,
    ) -> T:
        content = await self._read_async(
            method=method,
            path=path,
            actor_id=actor_id,
            admission=admission,
            binding=binding,
            content=content,
            maximum=self.MAX_JSON_BYTES,
            media_type="application/json",
            timeout=timeout or self.settings.request_timeout_seconds,
        )
        try:
            return model.model_validate_json(content)
        except ValidationError as error:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH) from error

    async def _read_async(
        self,
        *,
        method: str,
        path: str,
        actor_id: str,
        maximum: int,
        media_type: str,
        timeout: float,
        admission: EvaluationWorkerAdmission | None = None,
        binding: EvaluationWorkerBinding | None = None,
        content: bytes = b"",
    ) -> bytes:
        context = EvaluationWorkerAuthContext(
            audience=self.settings.audience,
            actor_id=actor_id,
            method=method,
            path=path,
            body_sha256=hashlib.sha256(content).hexdigest(),
            job_id=admission.request.job_id if admission else None,
            request_sha256=admission.request_sha256 if admission else None,
            gateway_fence_id=admission.gateway_fence_id if admission else None,
            binding_sha256=binding.binding_sha256 if binding else None,
        )
        try:
            async with asyncio.timeout(timeout):
                credentials = await self.credentials.credentials_async(context=context)
                headers = credentials.headers()
                headers.update({"Accept-Encoding": "identity", "Content-Type": "application/json"})
                async with self._client.stream(method, path, content=content, headers=headers) as response:
                    response.raise_for_status()
                    if (
                        response.headers.get("content-encoding", "identity") != "identity"
                        or response.headers.get("content-type", "").split(";")[0] != media_type
                    ):
                        raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
                    received = bytearray()
                    async for chunk in response.aiter_bytes():
                        if len(received) + len(chunk) > maximum:
                            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
                        received.extend(chunk)
                    return bytes(received)
        except httpx.HTTPStatusError as error:
            code = (
                EvaluationJobErrorCode.NOT_AUTHORIZED
                if error.response.status_code in {401, 403}
                else EvaluationJobErrorCode.REQUEST_CONFLICT
            )
            raise EvaluationJobError(code) from error
        except (httpx.TransportError, TimeoutError) as error:
            raise EvaluationJobError(EvaluationJobErrorCode.DISPATCH_UNCERTAIN) from error
