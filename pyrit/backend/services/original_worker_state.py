# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Owned, leased Blob commit barriers for the opt-in two-job validation instance."""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import hashlib
import hmac
import json
import logging
import os
import secrets
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING
from uuid import UUID

from azure.core import MatchConditions
from azure.core.exceptions import AzureError
from pydantic import JsonValue, TypeAdapter

from pyrit.backend.models.original_worker import CohostBackendConfig, CohostModelBudget
from pyrit.backend.services.original_worker_preflight import CohostPreflightError
from pyrit.models import config_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from azure.identity.aio import DefaultAzureCredential
    from azure.storage.blob.aio import BlobClient, BlobLeaseClient

    from pyrit.backend.services.original_evidence_admission import OriginalEvidenceEnvelope
    from pyrit.backend.services.original_worker_preflight import CohostPreflight

logger = logging.getLogger(__name__)


class CohostValidationState:
    """Consume admissions/model intent remotely before use; runtime never initializes lost state."""

    BYTE_LIMIT = 131_072
    LEASE_SECONDS = 60
    IO_SECONDS = 30

    def __init__(self, *, preflight: CohostPreflight) -> None:
        """Bind one startup-owned private container/instance, not a shared application database."""
        self.preflight = preflight
        self.config = preflight.config
        scope = self.config.validation_scope
        if scope is None or scope.authority_key_sha256 is None:
            raise CohostPreflightError("Hosted validation state must be explicitly initialized before startup.")
        self.scope = scope
        self._blob: BlobClient | None = None
        self._credential: DefaultAzureCredential | None = None
        self._lease: BlobLeaseClient | None = None
        self._renew_task: asyncio.Task[None] | None = None
        self._lease_deadline = 0.0
        self._etag: str | None = None
        self._value: dict[str, JsonValue] = {}
        self._key = b""
        self._available = False
        self._lock = asyncio.Lock()

    @property
    def authority_key(self) -> bytes:
        """The frozen backend-only retained/pilot authority, never returned by HTTP."""
        return self._key

    @property
    def budget(self) -> CohostModelBudget:
        """The exact restored aggregate model intent/count/usage."""
        value = self._value.get("model")
        if not isinstance(value, dict):
            raise CohostPreflightError("Restored validation model state is unavailable.")
        return CohostModelBudget.model_validate(value)

    @property
    def retained(self) -> list[dict[str, JsonValue]]:
        """The same backend-authenticated envelopes, not worker SQLite rows."""
        value = self._value.get("retained")
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            raise CohostPreflightError("Restored validation retained authority is unavailable.")
        return [copy.deepcopy(item) for item in value if isinstance(item, dict)]

    def can_reserve(self) -> bool:
        """
        Enforce the exact validation lifetime and conservative restart boundary.

        Returns:
            bool: Whether a new original job may consume one remaining validation admission.
        """
        jobs = self._value.get("jobs")
        return (
            self.is_available()
            and datetime.now(UTC)
            + timedelta(
                seconds=self.config.active_timeout_seconds + self.config.cleanup_timeout_seconds + self.IO_SECONDS
            )
            < self.scope.expires_at
            and isinstance(jobs, list)
            and len(jobs) < self.scope.max_original_jobs
            and self._value.get("uncontained") is False
            and not any(isinstance(job, dict) and job.get("phase") != "closed" for job in jobs)
        )

    def is_available(self) -> bool:
        """
        Refuse use after lease uncertainty, lifetime expiry or state loss.

        Returns:
            bool: Whether this process still owns the exact remote state lease.
        """
        return self._available and monotonic() < self._lease_deadline and datetime.now(UTC) < self.scope.expires_at

    def has_unknown_job(self) -> bool:
        """
        Treat incomplete prior bindings as consumed, not as reusable admissions.

        Returns:
            bool: Whether a prior job lacks authenticated terminal containment.
        """
        jobs = self._value.get("jobs")
        return self._value.get("uncontained") is not False or (
            isinstance(jobs, list) and any(isinstance(job, dict) and job.get("phase") != "closed" for job in jobs)
        )

    async def startup_async(self) -> None:
        """Acquire existing exact state and its lease; never create a missing state blob."""
        from azure.identity.aio import DefaultAzureCredential
        from azure.storage.blob.aio import BlobClient, BlobLeaseClient

        self.preflight.verify_identity_environment()
        self._credential = DefaultAzureCredential(
            require_envvar=True, managed_identity_client_id=str(self.config.managed_identity_client_id)
        )
        url = self.scope.state_container_url.rstrip("/") + f"/instances/{self.scope.instance_id}/state.json"
        self._blob = BlobClient.from_blob_url(
            url, credential=self._credential, retry_total=0, connection_timeout=10, read_timeout=15
        )
        self._lease = BlobLeaseClient(self._blob)
        try:
            async with asyncio.timeout(self.IO_SECONDS):
                await self._lease.acquire(lease_duration=self.LEASE_SECONDS)
                self._lease_deadline = monotonic() + self.LEASE_SECONDS
                stream = await self._blob.download_blob(lease=self._lease, max_concurrency=1)
                content = bytearray()
                async for chunk in stream.chunks():
                    if len(chunk) > self.BYTE_LIMIT - len(content):
                        raise CohostPreflightError("Owned validation state exceeds its fixed byte bound.")
                    content.extend(chunk)
                self._etag = stream.properties.etag
                if not isinstance(self._etag, str) or not self._etag:
                    raise CohostPreflightError("Owned validation state lacks its exact conditional ETag.")
            value = TypeAdapter(dict[str, JsonValue]).validate_json(bytes(content), strict=True)
            self._key = self._validate_value(value)
            self._value = value
            self._available = True
            self._renew_task = asyncio.create_task(self._renew_async())
        except (AzureError, TimeoutError, ValueError) as error:
            self._available = False
            await self.shutdown_async()
            raise CohostPreflightError("Owned validation Blob state/lease is unavailable or unverified.") from error

    async def reserve_async(self, *, reservation_id: UUID, operator_oid: str, profile_ref: str) -> None:
        """Consume a distinct original admission remotely BEFORE prepare, spawn or RP allocation."""

        def change(value: dict[str, JsonValue]) -> None:
            jobs = value["jobs"]
            if (
                not self.can_reserve()
                or not isinstance(jobs, list)
                or operator_oid not in self.config.allowed_operator_oids
                or profile_ref != self.config.profile_ref
            ):
                raise CohostPreflightError("The owned validation scope cannot admit another original job.")
            jobs.append(
                {
                    "reservation_id": str(reservation_id),
                    "operator_oid": operator_oid,
                    "profile_ref": profile_ref,
                    "phase": "reserved",
                }
            )

        await self._commit_async(change)

    async def bind_job_async(self, *, reservation_id: UUID, app_run_id: UUID, job_ref: UUID, control_id: UUID) -> None:
        """Commit exact app/job/control binding BEFORE any child or physical source allocation."""

        def change(value: dict[str, JsonValue]) -> None:
            job = self._job(value=value, reservation_id=reservation_id)
            if job.get("phase") != "reserved":
                raise CohostPreflightError("A consumed validation reservation cannot be rebound.")
            job.update(app_run_id=str(app_run_id), job_ref=str(job_ref), control_id=str(control_id), phase="bound")

        await self._commit_async(change)

    async def update_budget_async(self, *, requests: int, observed_tokens: int, unresolved: bool) -> None:
        """Commit dispatch intent and actual metering; never restore a missing counter to zero."""

        def change(value: dict[str, JsonValue]) -> None:
            budget = value["model"]
            if not isinstance(budget, dict):
                raise CohostPreflightError("Validation model state is invalid.")
            previous = CohostModelBudget.model_validate(budget)
            if requests < previous.requests or observed_tokens < previous.observed_tokens:
                raise CohostPreflightError("Validation model counters cannot move backwards.")
            value["model"] = {"requests": requests, "observed_tokens": observed_tokens, "unresolved": unresolved}

        await self._commit_async(change)

    async def finish_job_async(
        self, *, reservation_id: UUID, proved: bool, envelope: OriginalEvidenceEnvelope | None = None
    ) -> None:
        """Release only authenticated physical closure; consumed admissions never return to the pool."""

        def change(value: dict[str, JsonValue]) -> None:
            job = self._job(value=value, reservation_id=reservation_id)
            job["phase"] = "closed" if proved else "uncontained"
            value["uncontained"] = value.get("uncontained") is True or not proved
            if envelope is not None:
                retained = value["retained"]
                if not isinstance(retained, list) or len(retained) >= self.scope.max_original_jobs:
                    raise CohostPreflightError("Validation retained authority exceeds the closed job policy.")
                if job.get("job_ref") != str(envelope.job_ref) or job.get("app_run_id") != str(envelope.app_run_id):
                    raise CohostPreflightError("Retained envelope is not its consumed exact validation job.")
                if any(
                    isinstance(item, dict)
                    and isinstance(item["envelope"], dict)
                    and item["envelope"].get("job_ref") == str(envelope.job_ref)
                    for item in retained
                ):
                    raise CohostPreflightError("Retained validation authority cannot be appended twice.")
                retained.append(
                    {
                        "created_at": datetime.now(UTC).isoformat(),
                        "expires_at": self.scope.expires_at.isoformat(),
                        "envelope": envelope.model_dump(mode="json", exclude_none=True),
                    }
                )

        await self._commit_async(change)

    async def shutdown_async(self) -> None:
        """Stop lease renewal only after the runtime has drained its child and model work."""
        self._available = False
        if self._renew_task is not None:
            self._renew_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._renew_task
            self._renew_task = None
        try:
            if self._lease is not None:
                async with asyncio.timeout(self.IO_SECONDS):
                    await self._lease.release()
        except (AzureError, TimeoutError) as error:
            logger.error("Owned validation lease release remains unverified (%s).", type(error).__name__)
        finally:
            if self._blob is not None:
                await self._blob.close()
            if self._credential is not None:
                await self._credential.close()

    async def _renew_async(self) -> None:
        try:
            while True:
                await asyncio.sleep(20)
                self.preflight.verify_identity_environment()
                if not self.is_available():
                    raise CohostPreflightError("Validation lease authority expired before renewal.")
                assert self._lease is not None
                async with asyncio.timeout(15):
                    await self._lease.renew()
                if not self.is_available():
                    raise CohostPreflightError("Validation lease authority expired during renewal.")
                self._lease_deadline = monotonic() + self.LEASE_SECONDS
        except (AzureError, TimeoutError, ValueError) as error:
            self._available = False
            logger.error("Owned validation lease renewal failed closed (%s).", type(error).__name__)

    async def _commit_async(self, change: Callable[[dict[str, JsonValue]], None]) -> None:
        async with self._lock:
            if not self.is_available():
                raise CohostPreflightError("Validation remote state/lease is unavailable; use is refused.")
            self.preflight.verify_identity_environment()
            value = copy.deepcopy(self._value)
            change(value)
            value.pop("signature", None)
            value["signature"] = hmac.new(self._key, self._content(value), hashlib.sha256).hexdigest()
            self._validate_value(value)
            content = self._content(value)
            if len(content) > self.BYTE_LIMIT:
                raise CohostPreflightError("Owned validation state exceeded its fixed byte bound.")
            assert self._blob is not None and self._lease is not None and self._etag is not None
            try:
                async with asyncio.timeout(self.IO_SECONDS):
                    result = await self._blob.upload_blob(
                        content,
                        overwrite=True,
                        lease=self._lease,
                        etag=self._etag,
                        match_condition=MatchConditions.IfNotModified,
                        max_concurrency=1,
                    )
                etag = result.get("etag")
                if not isinstance(etag, str) or not etag or not self.is_available():
                    raise CohostPreflightError("Validation commit lost its exact ETag or lease authority.")
                self._etag = etag
                self._value = value
            except (AzureError, TimeoutError, ValueError) as error:
                self._available = False
                raise CohostPreflightError("Validation commit is uncertain; no subsequent use is admitted.") from error

    def _validate_value(self, value: dict[str, JsonValue]) -> bytes:
        encoded = value.get("authority_key_b64")
        if not isinstance(encoded, str) or len(encoded) > 64:
            raise CohostPreflightError("Validation retained authority is unavailable.")
        key = base64.b64decode(encoded, validate=True)
        if len(key) != 32 or hashlib.sha256(key).hexdigest() != self.scope.authority_key_sha256:
            raise CohostPreflightError("Validation authority key differs from the frozen instance.")
        signature = value.get("signature")
        unsigned = {name: item for name, item in value.items() if name != "signature"}
        if not isinstance(signature, str) or not hmac.compare_digest(
            signature, hmac.new(key, self._content(unsigned), hashlib.sha256).hexdigest()
        ):
            raise CohostPreflightError("Owned validation state was modified.")
        created = value.get("created_at")
        if not isinstance(created, str):
            raise CohostPreflightError("Validation lifetime is unavailable.")
        started = datetime.fromisoformat(created)
        budget, jobs, retained = value.get("model"), value.get("jobs"), value.get("retained")
        if not isinstance(budget, dict):
            raise CohostPreflightError("Validation model budget is invalid.")
        CohostModelBudget.model_validate(budget)
        if (
            type(value.get("schema_version")) is not int
            or value["schema_version"] != 1
            or value.get("instance_id") != str(self.scope.instance_id)
            or value.get("policy_sha256") != self.policy_sha256(self.config)
            or value.get("expires_at") != self.scope.expires_at.isoformat()
            or set(value)
            != {
                "schema_version",
                "instance_id",
                "created_at",
                "expires_at",
                "policy_sha256",
                "authority_key_b64",
                "jobs",
                "retained",
                "uncontained",
                "model",
                "signature",
            }
            or started.utcoffset() != timedelta(0)
            or not started < self.scope.expires_at <= started + timedelta(hours=24)
            or datetime.now(UTC) >= self.scope.expires_at
            or type(value.get("uncontained")) is not bool
            or not isinstance(jobs, list)
            or len(jobs) > self.scope.max_original_jobs
            or not isinstance(retained, list)
            or len(retained) > self.scope.max_original_jobs
        ):
            raise CohostPreflightError("Validation identity/lifetime/counter state is inconsistent.")
        ids: set[str] = set()
        bound_ids: set[str] = set()
        for job in jobs:
            if not isinstance(job, dict):
                raise CohostPreflightError("Validation original job bindings are inconsistent.")
            identifier = job.get("reservation_id")
            if (
                not isinstance(identifier, str)
                or identifier in ids
                or job.get("operator_oid") not in self.config.allowed_operator_oids
                or job.get("profile_ref") != self.config.profile_ref
                or job.get("phase") not in ("reserved", "bound", "closed", "uncontained")
            ):
                raise CohostPreflightError("Validation original job bindings are inconsistent.")
            if str(UUID(identifier)) != identifier:
                raise CohostPreflightError("Validation original reservation is not a canonical UUID.")
            names = {"app_run_id", "job_ref", "control_id"}
            present = names.intersection(job)
            if (
                set(job) != {"reservation_id", "operator_oid", "profile_ref", "phase"} | present
                or (present and present != names)
                or (job["phase"] == "bound" and present != names)
                or (job["phase"] == "reserved" and present)
                or (job["phase"] == "uncontained" and value["uncontained"] is not True)
            ):
                raise CohostPreflightError("Validation app/job/control binding is incomplete or inconsistent.")
            for name in present:
                bound_id = job[name]
                if not isinstance(bound_id, str) or str(UUID(bound_id)) != bound_id or bound_id in bound_ids:
                    raise CohostPreflightError("Validation app/job/control identities are reused or invalid.")
                bound_ids.add(bound_id)
            ids.add(identifier)
        self._validate_retained(jobs=jobs, retained=retained)
        return key

    def _validate_retained(self, *, jobs: list[JsonValue], retained: list[JsonValue]) -> None:
        from pyrit.backend.services.original_evidence_admission import OriginalEvidenceEnvelope

        seen: set[str] = set()
        for packet in retained:
            if not isinstance(packet, dict) or set(packet) != {"created_at", "expires_at", "envelope"}:
                raise CohostPreflightError("Validation retained packet is incomplete.")
            envelope = OriginalEvidenceEnvelope.model_validate_json(json.dumps(packet["envelope"]))
            created = packet["created_at"]
            if not isinstance(created, str):
                raise CohostPreflightError("Validation retained creation time is invalid.")
            created_at = datetime.fromisoformat(created)
            matching = [
                job
                for job in jobs
                if isinstance(job, dict)
                and job.get("phase") == "closed"
                and job.get("job_ref") == str(envelope.job_ref)
                and job.get("app_run_id") == str(envelope.app_run_id)
                and job.get("operator_oid") == envelope.operator_oid
                and job.get("profile_ref") == envelope.profile_ref
            ]
            if (
                len(matching) != 1
                or str(envelope.job_ref) in seen
                or envelope.source_sha256 != self.config.source.spec.package.source_sha256
                or created_at.utcoffset() != timedelta(0)
                or created_at >= self.scope.expires_at
                or packet["expires_at"] != self.scope.expires_at.isoformat()
            ):
                raise CohostPreflightError("Validation retained authority is not its closed exact source job.")
            seen.add(str(envelope.job_ref))

    @staticmethod
    def _job(*, value: dict[str, JsonValue], reservation_id: UUID) -> dict[str, JsonValue]:
        jobs = value.get("jobs")
        if isinstance(jobs, list):
            matches = [
                job for job in jobs if isinstance(job, dict) and job.get("reservation_id") == str(reservation_id)
            ]
            if len(matches) == 1:
                return matches[0]
        raise CohostPreflightError("Validation admission identity is not its exact consumed reservation.")

    @staticmethod
    def _content(value: dict[str, JsonValue]) -> bytes:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()

    @staticmethod
    def policy_sha256(config: CohostBackendConfig) -> str:
        """
        Bind state to source/actor, native identity, owned durable targets and the validation scope.

        Returns:
            str: A public-safe policy fingerprint, not a secret or model replay capability.
        """
        assert config.validation_scope is not None
        return config_hash(
            {
                "public_commit": config.public_commit,
                "profile_ref": config.profile_ref,
                "source_alias": config.source_alias,
                "source": config.source.model_dump(mode="json"),
                "allowed_operator_oids": sorted(config.allowed_operator_oids),
                "allowed_group_ids": sorted(config.allowed_group_ids),
                "managed_identity_client_id": str(config.managed_identity_client_id),
                "expected_database_name": config.expected_database_name,
                "result_container_url": config.result_container_url,
                "validation_scope": config.validation_scope.model_dump(mode="json", exclude={"authority_key_sha256"}),
                "relay": config.relay.model_dump(mode="json"),
            }
        )


def bootstrap_validation_state(
    *, config: CohostBackendConfig, created_at: datetime | None = None
) -> tuple[CohostBackendConfig, bytes]:
    """
    Generate an initial private state packet without a credential, Azure call or original execution.

    Returns:
        tuple[CohostBackendConfig, bytes]: Bound server policy and secret-containing initial state.
    """
    scope = config.validation_scope
    now = created_at or datetime.now(UTC)
    if (
        scope is None
        or scope.authority_key_sha256 is not None
        or now.utcoffset() != timedelta(0)
        or not now < scope.expires_at <= now + timedelta(hours=24)
    ):
        raise CohostPreflightError("Bootstrap requires a NEW unbound validation scope of at most24 hours.")
    key = secrets.token_bytes(32)
    bound = config.model_copy(
        update={"validation_scope": scope.model_copy(update={"authority_key_sha256": hashlib.sha256(key).hexdigest()})}
    )
    value: dict[str, JsonValue] = {
        "schema_version": 1,
        "instance_id": str(scope.instance_id),
        "created_at": now.isoformat(),
        "expires_at": scope.expires_at.isoformat(),
        "policy_sha256": CohostValidationState.policy_sha256(bound),
        "authority_key_b64": base64.b64encode(key).decode(),
        "jobs": [],
        "retained": [],
        "uncontained": False,
        "model": {"requests": 0, "observed_tokens": 0, "unresolved": False},
    }
    value["signature"] = hmac.new(key, CohostValidationState._content(value), hashlib.sha256).hexdigest()
    return bound, CohostValidationState._content(value)


def main() -> None:
    """Explicit one-time no-cloud preparation; never a runtime state-loss recovery command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bound-config-output", type=Path, required=True)
    parser.add_argument("--private-state-output", type=Path, required=True)
    args = parser.parse_args()
    content = args.config.read_bytes()
    config = CohostBackendConfig.model_validate_json(content)
    bound, state = bootstrap_validation_state(config=config)
    assert bound.validation_scope is not None
    for path, data in (
        (args.bound_config_output, bound.model_dump_json().encode()),
        (args.private_state_output, state),
    ):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "bound_config_sha256": hashlib.sha256(bound.model_dump_json().encode()).hexdigest(),
                "private_state_sha256": hashlib.sha256(state).hexdigest(),
                "instance_id": str(bound.validation_scope.instance_id),
                "max_original_jobs": 2,
            }
        )
    )


if __name__ == "__main__":
    main()
