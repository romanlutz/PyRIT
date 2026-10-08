# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Startup-only local harmless job configuration; no platform provisioning or live broker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from pyrit.executor.jobs.worker_auth import EvaluationWorkerCredentialRegistry
from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpSettings
from pyrit.memory import SQLiteMemory

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.executor.jobs.local import LocalEvaluationJobPort
    from pyrit.memory import MemoryInterface


@dataclass(frozen=True, kw_only=True)
class LocalEvaluationJobSettings:
    """Explicit existing-dependency, model-free opt-in for one local backend owner."""

    root: Path
    allowed_actor_ids: frozenset[str]

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> LocalEvaluationJobSettings | None:
        """
        Refuse partial configuration, unsupported transports, or absent actor admission.

        Returns:
            LocalEvaluationJobSettings | None: None only when the job port is entirely unconfigured.

        Raises:
            ValueError: If explicitly supplied local job settings are invalid.
        """
        backend = environment.get("PYRIT_EVALUATION_JOB_BACKEND", "")
        root = environment.get("PYRIT_EVALUATION_JOB_ROOT", "")
        actors = environment.get("PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS", "")
        if not any((backend, root, actors)):
            return None
        if backend != "local" or not root or not actors:
            raise ValueError("Evaluation jobs require explicit local backend, root, and authenticated actor settings.")
        try:
            actor_ids = frozenset(str(UUID(value.strip())) for value in actors.split(","))
        except ValueError as error:
            raise ValueError("Evaluation job actor admission requires UUID object identifiers.") from error
        if not 1 <= len(actor_ids) <= 16:
            raise ValueError("Evaluation job actor admission is limited to sixteen explicit operators.")
        path = Path(root)
        if not path.is_absolute() or path.resolve() != path or path.is_symlink() or root.startswith(("\\\\", "//")):
            raise ValueError("Evaluation job storage must be an explicit local absolute non-symlink root.")
        return cls(root=path, allowed_actor_ids=actor_ids)

    async def create_port_async(self, *, memory: MemoryInterface) -> LocalEvaluationJobPort:
        """
        Bind only canonical SQLite and the public harmless original source.

        Returns:
            LocalEvaluationJobPort: A startup-owned port, not a cloud or private runtime.

        Raises:
            ValueError: If the backend is not the explicitly supported local SQLite PoC.
        """
        if not isinstance(memory, SQLiteMemory):
            raise ValueError("The local evaluation job PoC requires canonical SQLite memory.")
        from pyrit.executor.jobs.inspect import create_public_original_job_port_async

        return await create_public_original_job_port_async(
            root=self.root, memory=memory, allowed_actor_ids=self.allowed_actor_ids
        )


@dataclass(frozen=True, kw_only=True)
class RemoteEvaluationJobSettings:
    """Startup-only gateway settings, not private platform configuration or provisioning."""

    root: Path
    allowed_actor_ids: frozenset[str]
    identity_name: str
    transport: EvaluationWorkerHttpSettings

    @classmethod
    def from_environment(cls, environment: Mapping[str, str]) -> RemoteEvaluationJobSettings:
        """
        Require explicit audience, installed identity and exact execution protocol.

        Returns:
            RemoteEvaluationJobSettings: Validated configuration with no token contents.

        Raises:
            ValueError: If authority, identity, shared admission or bounded limits are incomplete.
        """
        if environment.get("PYRIT_EVALUATION_JOB_BACKEND") != "remote":
            raise ValueError("Remote evaluation jobs require explicit remote backend selection.")
        local = LocalEvaluationJobSettings.from_environment({**environment, "PYRIT_EVALUATION_JOB_BACKEND": "local"})
        if local is None:
            raise ValueError("Remote evaluation jobs require an explicit gateway root and operators.")
        prefix = "PYRIT_EVALUATION_JOB_REMOTE_"
        required = ("URL", "AUDIENCE", "IDENTITY", "SERVICE_ID", "PROTOCOL_VERSION", "PROTOCOL_SHA256")
        if any(not environment.get(prefix + key) for key in required):
            raise ValueError("Remote evaluation jobs require explicit authority, identity, audience and protocol.")
        if environment[prefix + "PROTOCOL_VERSION"] != "1":
            raise ValueError("Only the explicitly configured execution-only worker protocol version 1 is supported.")
        loopback = environment.get(prefix + "ALLOW_LOOPBACK_HTTP", "false")
        if loopback not in {"true", "false"}:
            raise ValueError("Remote fixture HTTP opt-in must be explicitly true or false.")
        defaults = {
            "REQUEST_TIMEOUT_SECONDS": 10,
            "ARTIFACT_TIMEOUT_SECONDS": 30,
            "POLL_INTERVAL_SECONDS": 0.25,
            "POLL_DEADLINE_SECONDS": 60,
            "SETTLEMENT_TIMEOUT_SECONDS": 10,
        }
        limits = {key: float(environment.get(prefix + key, str(value))) for key, value in defaults.items()}
        return cls(
            root=local.root,
            allowed_actor_ids=local.allowed_actor_ids,
            identity_name=environment[prefix + "IDENTITY"],
            transport=EvaluationWorkerHttpSettings(
                base_url=environment[prefix + "URL"],
                audience=environment[prefix + "AUDIENCE"],
                service_id=environment[prefix + "SERVICE_ID"],
                schema_sha256=environment[prefix + "PROTOCOL_SHA256"],
                allow_loopback_http=loopback == "true",
                request_timeout_seconds=limits["REQUEST_TIMEOUT_SECONDS"],
                artifact_timeout_seconds=limits["ARTIFACT_TIMEOUT_SECONDS"],
                poll_interval_seconds=limits["POLL_INTERVAL_SECONDS"],
                poll_deadline_seconds=limits["POLL_DEADLINE_SECONDS"],
                settlement_timeout_seconds=limits["SETTLEMENT_TIMEOUT_SECONDS"],
            ),
        )

    async def create_port_async(self, *, memory: MemoryInterface) -> LocalEvaluationJobPort:
        """
        Bind an installed credential provider and the sole approved API-side original writer.

        Returns:
            LocalEvaluationJobPort: The deliberately installed remote gateway facade.

        Raises:
            ValueError: If canonical SQLite or production identity/delegation is unsupported.
        """
        if not isinstance(memory, SQLiteMemory):
            raise ValueError("The remote evaluation job PoC requires API-owned canonical SQLite.")
        from pyrit.executor.jobs.remote import create_remote_original_job_port_async

        credentials = EvaluationWorkerCredentialRegistry.create(
            name=self.identity_name, audience=self.transport.audience
        )
        try:
            return await create_remote_original_job_port_async(
                root=self.root,
                memory=memory,
                allowed_actor_ids=self.allowed_actor_ids,
                settings=self.transport,
                credentials=credentials,
            )
        except BaseException:
            await credentials.close_async()
            raise


def evaluation_job_settings_from_environment(
    environment: Mapping[str, str],
) -> LocalEvaluationJobSettings | RemoteEvaluationJobSettings | None:
    """
    Select one explicit startup backend; remote fragments never silently select local/off.

    Returns:
        LocalEvaluationJobSettings | RemoteEvaluationJobSettings | None: The startup-owned configuration.

    Raises:
        ValueError: If remote settings are partial or paired with another backend.
    """
    if environment.get("PYRIT_EVALUATION_JOB_BACKEND") == "remote":
        return RemoteEvaluationJobSettings.from_environment(environment)
    if any(key.startswith("PYRIT_EVALUATION_JOB_REMOTE_") and value for key, value in environment.items()):
        raise ValueError("Remote job settings require the explicit remote backend.")
    return LocalEvaluationJobSettings.from_environment(environment)
