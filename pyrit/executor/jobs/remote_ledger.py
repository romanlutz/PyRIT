# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Gateway-owned execution bindings, worker cursor and original/derived manifest provenance."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING

from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.models.evaluation_job import EvaluationArtifactManifest, EvaluationCanonicalReceipt, EvaluationCleanupState
from pyrit.models.evaluation_worker import (
    EvaluationGatewaySettlement,
    EvaluationGatewaySettlementDisposition,
    EvaluationGatewaySettlementReceipt,
    EvaluationWorkerAdmission,
    EvaluationWorkerBinding,
    EvaluationWorkerEvidence,
    EvaluationWorkerSnapshot,
    EvaluationWorkerTerminal,
)

if TYPE_CHECKING:
    from pathlib import Path
    from uuid import UUID


class EvaluationRemoteJournal:
    """A companion to the exclusive local job owner, never a worker database."""

    def __init__(self, root: Path) -> None:
        """Configure a journal under the local gateway's process-owned root."""
        self.path = root / "remote.sqlite"

    def initialize(self) -> None:
        """Create bounded generic gateway metadata, without modifying canonical memory."""
        with closing(self._connect()) as connection, connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS remote_jobs (
                    job_id VARCHAR(36) PRIMARY KEY,
                    admission_json TEXT NOT NULL,
                    binding_json TEXT,
                    worker_cursor INTEGER NOT NULL DEFAULT 0,
                    worker_last_sequence INTEGER NOT NULL DEFAULT 0,
                    terminal_json TEXT,
                    worker_manifest_json BLOB,
                    worker_manifest_sha256 VARCHAR(64),
                    worker_manifest_bytes_sha256 VARCHAR(64),
                    gateway_manifest_json TEXT,
                    gateway_manifest_sha256 VARCHAR(64),
                    canonical_json TEXT,
                    settlement_request_json TEXT,
                    settlement_json TEXT,
                    uncertain INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS remote_events (
                    job_id VARCHAR(36) NOT NULL REFERENCES remote_jobs(job_id),
                    sequence INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY(job_id, sequence)
                );
                """
            )

    def admit(self, admission: EvaluationWorkerAdmission) -> None:
        """
        Retain exact dispatch intent before any remote request.

        Raises:
            EvaluationJobError: If a job was rebound to another gateway dispatch.
        """
        admission = EvaluationWorkerAdmission.model_validate(admission)
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT admission_json FROM remote_jobs WHERE job_id=?", (str(admission.request.job_id),)
            ).fetchone()
            if row is not None and row[0] != admission.model_dump_json():
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            if row is None:
                connection.execute(
                    "INSERT INTO remote_jobs(job_id,admission_json) VALUES(?,?)",
                    (str(admission.request.job_id), admission.model_dump_json()),
                )

    def bind(self, *, admission: EvaluationWorkerAdmission, binding: EvaluationWorkerBinding) -> None:
        """
        Refuse incarnation/fence replacement even after an uncertain remote response.

        Raises:
            EvaluationJobError: If the complete binding differs.
        """
        binding = EvaluationWorkerBinding.model_validate(binding)
        if not binding.accepts(admission):
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        with closing(self._connect()) as connection, connection:
            row = self._row(connection=connection, job_id=binding.job_id)
            if row["admission_json"] != admission.model_dump_json() or (
                row["binding_json"] is not None and row["binding_json"] != binding.model_dump_json()
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            connection.execute(
                "UPDATE remote_jobs SET binding_json=? WHERE job_id=?",
                (binding.model_dump_json(), str(binding.job_id)),
            )

    def observe(self, snapshot: EvaluationWorkerSnapshot) -> int:
        """
        Persist an independent worker cursor and reject gaps, changed events or terminal receipts.

        Returns:
            int: The highest durably observed worker event sequence.

        Raises:
            EvaluationJobError: If ordered execution evidence contradicts prior observations.
        """
        snapshot = EvaluationWorkerSnapshot.model_validate(snapshot)
        with closing(self._connect()) as connection, connection:
            row = self._row(connection=connection, job_id=snapshot.request.job_id)
            if (
                row["binding_json"] != snapshot.binding.model_dump_json()
                or snapshot.last_sequence < row["worker_last_sequence"]
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            cursor = row["worker_cursor"]
            for event in snapshot.events:
                previous = connection.execute(
                    "SELECT event_json FROM remote_events WHERE job_id=? AND sequence=?",
                    (str(snapshot.request.job_id), event.sequence),
                ).fetchone()
                if previous is not None:
                    if previous[0] != event.model_dump_json():
                        raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
                elif event.sequence != cursor + 1:
                    raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
                else:
                    connection.execute(
                        "INSERT INTO remote_events VALUES(?,?,?)",
                        (str(snapshot.request.job_id), event.sequence, event.model_dump_json()),
                    )
                    cursor = event.sequence
            if snapshot.last_sequence > cursor and not snapshot.events:
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            terminal = snapshot.terminal.model_dump_json() if snapshot.terminal is not None else None
            if row["terminal_json"] is not None and terminal != row["terminal_json"]:
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            connection.execute(
                "UPDATE remote_jobs SET worker_cursor=?,worker_last_sequence=?,terminal_json=? WHERE job_id=?",
                (cursor, snapshot.last_sequence, terminal, str(snapshot.request.job_id)),
            )
            return int(cursor)

    def retain(
        self, *, job_id: UUID, original: bytes, worker: EvaluationArtifactManifest, gateway: EvaluationArtifactManifest
    ) -> None:
        """
        Preserve original bytes and explicitly link a differently fenced derived inventory.

        Raises:
            EvaluationJobError: If either manifest loses its admitted provenance or exact inventory.
        """
        with closing(self._connect()) as connection, connection:
            row = self._row(connection=connection, job_id=job_id)
            binding = EvaluationWorkerBinding.model_validate_json(row["binding_json"])
            admission = EvaluationWorkerAdmission.model_validate_json(row["admission_json"])
            terminal = EvaluationWorkerTerminal.model_validate_json(row["terminal_json"])
            if (
                EvaluationArtifactManifest.model_validate_json(original) != worker
                or worker.request != admission.request
                or gateway.request != worker.request
                or gateway.artifacts != worker.artifacts
                or worker.fence_id != binding.worker_fence_id
                or gateway.fence_id != binding.gateway_fence_id
                or terminal.manifest_sha256 != worker.manifest_sha256
                or row["worker_cursor"] != terminal.last_sequence
                or (row["worker_manifest_json"] is not None and row["worker_manifest_json"] != original)
                or (
                    row["gateway_manifest_json"] is not None
                    and row["gateway_manifest_json"] != gateway.model_dump_json()
                )
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            connection.execute(
                "UPDATE remote_jobs SET worker_manifest_json=?,worker_manifest_sha256=?,"
                "worker_manifest_bytes_sha256=?,gateway_manifest_json=?,gateway_manifest_sha256=? WHERE job_id=?",
                (
                    original,
                    worker.manifest_sha256,
                    hashlib.sha256(original).hexdigest(),
                    gateway.model_dump_json(),
                    gateway.manifest_sha256,
                    str(job_id),
                ),
            )

    def canonical(self, receipt: EvaluationCanonicalReceipt) -> None:
        """
        Persist only a verified receipt returned by this API's canonical writer.

        Raises:
            EvaluationJobError: If the receipt binds another local publication.
        """
        receipt = EvaluationCanonicalReceipt.model_validate(receipt)
        with closing(self._connect()) as connection, connection:
            row = self._row(connection=connection, job_id=receipt.job_id)
            admission = EvaluationWorkerAdmission.model_validate_json(row["admission_json"])
            gateway = EvaluationArtifactManifest.model_validate_json(row["gateway_manifest_json"])
            if (
                receipt.request_sha256 != admission.request_sha256
                or receipt.manifest_sha256 != gateway.manifest_sha256
                or receipt.artifact_sha256 not in {item.sha256 for item in gateway.artifacts}
                or (row["canonical_json"] is not None and row["canonical_json"] != receipt.model_dump_json())
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            connection.execute(
                "UPDATE remote_jobs SET canonical_json=? WHERE job_id=?",
                (receipt.model_dump_json(), str(receipt.job_id)),
            )

    def settlement_intent(self, settlement: EvaluationGatewaySettlement) -> None:
        """
        Retain exact settlement intent before sending a mutable network acknowledgment.

        Raises:
            EvaluationJobError: If closure, manifests or actual canonical import are not established.
        """
        settlement = EvaluationGatewaySettlement.model_validate(settlement)
        with closing(self._connect()) as connection, connection:
            row = self._row(connection=connection, job_id=settlement.binding.job_id)
            terminal = EvaluationWorkerTerminal.model_validate_json(row["terminal_json"])
            if (
                row["binding_json"] != settlement.binding.model_dump_json()
                or row["worker_cursor"] != terminal.last_sequence
                or terminal.cleanup is not EvaluationCleanupState.VERIFIED
                or (
                    row["settlement_request_json"] is not None
                    and row["settlement_request_json"] != settlement.model_dump_json()
                )
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            if settlement.disposition is EvaluationGatewaySettlementDisposition.CLOSURE_OBSERVED:
                if terminal.evidence is not EvaluationWorkerEvidence.ABSENT:
                    raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            else:
                gateway = EvaluationArtifactManifest.model_validate_json(row["gateway_manifest_json"])
                if (
                    settlement.worker_manifest_sha256 != row["worker_manifest_sha256"]
                    or settlement.gateway_manifest_sha256 != gateway.manifest_sha256
                    or settlement.artifact_sha256 not in {item.sha256 for item in gateway.artifacts}
                    or (
                        settlement.disposition is EvaluationGatewaySettlementDisposition.CANONICAL_IMPORTED
                        and row["canonical_json"] is None
                    )
                ):
                    raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            connection.execute(
                "UPDATE remote_jobs SET settlement_request_json=? WHERE job_id=?",
                (settlement.model_dump_json(), str(settlement.binding.job_id)),
            )

    def settle(self, receipt: EvaluationGatewaySettlementReceipt) -> None:
        """
        Record an authenticated worker acknowledgment separately from canonical results.

        Raises:
            EvaluationJobError: If the acknowledgment is foreign.
        """
        with closing(self._connect()) as connection, connection:
            row = self._row(connection=connection, job_id=receipt.job_id)
            binding = EvaluationWorkerBinding.model_validate_json(row["binding_json"])
            intent = EvaluationGatewaySettlement.model_validate_json(row["settlement_request_json"])
            if (
                receipt.binding_sha256 != binding.binding_sha256
                or receipt.settlement_sha256 != intent.settlement_sha256
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            connection.execute(
                "UPDATE remote_jobs SET settlement_json=? WHERE job_id=?",
                (receipt.model_dump_json(), str(receipt.job_id)),
            )

    def uncertain(self, job_id: UUID) -> None:
        """Retain ambiguity without resetting dispatch intent or deleting worker evidence."""
        with closing(self._connect()) as connection, connection:
            self._row(connection=connection, job_id=job_id)
            connection.execute("UPDATE remote_jobs SET uncertain=1 WHERE job_id=?", (str(job_id),))

    def read(self, job_id: UUID) -> dict[str, str | bytes | int | None]:
        """
        Read protected local reconciliation metadata, never expose it as a browser worker database.

        Returns:
            dict[str, str | bytes | int | None]: The durable gateway-owned row.
        """
        with closing(self._connect()) as connection:
            return dict(self._row(connection=connection, job_id=job_id))

    def handoff(
        self, job_id: UUID
    ) -> tuple[
        EvaluationWorkerAdmission, EvaluationWorkerBinding, EvaluationArtifactManifest, EvaluationArtifactManifest
    ]:
        """
        Resolve only this gateway's retained two-fence publication identity.

        Returns:
            tuple: Authenticated admission, execution binding, original and derived manifests.

        Raises:
            EvaluationJobError: If no complete handoff was retained.
        """
        row = self.read(job_id)
        names = ("admission_json", "binding_json", "worker_manifest_json", "gateway_manifest_json")
        values = [row[name] for name in names]
        if not all(isinstance(value, (str, bytes)) for value in values):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        return (
            EvaluationWorkerAdmission.model_validate_json(self._json_value(row["admission_json"])),
            EvaluationWorkerBinding.model_validate_json(self._json_value(row["binding_json"])),
            EvaluationArtifactManifest.model_validate_json(self._json_value(row["worker_manifest_json"])),
            EvaluationArtifactManifest.model_validate_json(self._json_value(row["gateway_manifest_json"])),
        )

    @staticmethod
    def _json_value(value: str | bytes | int | None) -> str | bytes:
        if not isinstance(value, (str, bytes)):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        return value

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        return connection

    @staticmethod
    def _row(*, connection: sqlite3.Connection, job_id: UUID) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM remote_jobs WHERE job_id=?", (str(job_id),)).fetchone()
        if row is None:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_FOUND)
        if not isinstance(row, sqlite3.Row):
            raise TypeError("Gateway journal queries must return typed SQLite rows.")
        return row
