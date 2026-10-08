# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Startup-only local harmless job configuration; no platform provisioning or live broker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

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
