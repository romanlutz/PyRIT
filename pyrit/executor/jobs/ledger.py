# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local single-owner durable queue ledger, separate from canonical PyRIT memory."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import sqlite3
from contextlib import closing, contextmanager
from datetime import UTC, datetime
from typing import TYPE_CHECKING, BinaryIO
from uuid import UUID, uuid4

from pyrit.executor.jobs.port import EvaluationJobError, EvaluationJobErrorCode
from pyrit.models.evaluation_job import (
    EvaluationArtifactManifest,
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationControlReceipt,
    EvaluationControlRequest,
    EvaluationDeliveryState,
    EvaluationEvidenceState,
    EvaluationJobDelivery,
    EvaluationJobEvent,
    EvaluationJobEventKind,
    EvaluationJobRequest,
    EvaluationJobSnapshot,
    EvaluationJobState,
    EvaluationJobSubmission,
    EvaluationWaitBoundary,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class LocalEvaluationJobLedger:
    """Persist dispatch intent before execution and never replay an interrupted attempt."""

    MAX_QUEUED = 64
    MAX_CONTROLS = 32
    ACTIVE_STATES = (
        EvaluationJobState.RUNNING,
        EvaluationJobState.WAITING,
        EvaluationJobState.CANCEL_REQUESTED,
        EvaluationJobState.FINALIZING,
    )

    def __init__(self, root: Path) -> None:
        """
        Use only an explicit local owner root, never a producer path.

        Raises:
            ValueError: If storage can redirect or is not explicitly local.
        """
        if not root.is_absolute() or str(root).startswith(("\\\\", "//")) or root.is_symlink():
            raise ValueError("The local job ledger requires an absolute, non-symlink local root.")
        root.mkdir(parents=True, exist_ok=True)
        if root.resolve() != root:
            raise ValueError("The local job ledger root cannot redirect through a symlink.")
        self.root = root
        self.path = root / "queue.sqlite"
        if self.path.is_symlink() or (root / "queue.owner").is_symlink():
            raise ValueError("The local job ledger files cannot be symlinks.")
        self._owner: BinaryIO | None = None

    def startup(self) -> None:
        """
        Acquire exclusive process ownership, then quarantine interrupted dispatches.

        Raises:
            ValueError: If the durable schema is not supported.
        """
        self._acquire_owner()
        try:
            with self._connect() as connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS job_metadata (
                        schema_version INTEGER NOT NULL CHECK(schema_version = 1)
                    );
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id VARCHAR(36) PRIMARY KEY,
                        case_run_sha256 VARCHAR(64) UNIQUE NOT NULL,
                        actor_id VARCHAR(128) NOT NULL,
                        request_sha256 VARCHAR(64) NOT NULL,
                        request_json TEXT NOT NULL,
                        capability_sha256 VARCHAR(64),
                        fence_id VARCHAR(36),
                        state VARCHAR(32) NOT NULL,
                        evidence VARCHAR(32) NOT NULL,
                        cleanup VARCHAR(32) NOT NULL,
                        last_sequence INTEGER NOT NULL,
                        boundary_json TEXT,
                        manifest_json TEXT,
                        canonical_json TEXT,
                        reason VARCHAR(128)
                    );
                    CREATE TABLE IF NOT EXISTS job_events (
                        job_id VARCHAR(36) NOT NULL REFERENCES jobs(job_id),
                        sequence INTEGER NOT NULL,
                        event_json TEXT NOT NULL,
                        PRIMARY KEY(job_id, sequence)
                    );
                    CREATE TABLE IF NOT EXISTS job_controls (
                        job_id VARCHAR(36) NOT NULL REFERENCES jobs(job_id),
                        command_id VARCHAR(36) NOT NULL,
                        command_sha256 VARCHAR(64) NOT NULL,
                        command_json TEXT NOT NULL,
                        accepted_sequence INTEGER NOT NULL,
                        delivered INTEGER NOT NULL DEFAULT 0,
                        PRIMARY KEY(job_id, command_id)
                    );
                    """
                )
                versions = connection.execute("SELECT schema_version FROM job_metadata").fetchall()
                if not versions:
                    connection.execute("INSERT INTO job_metadata VALUES (1)")
                elif len(versions) != 1 or versions[0][0] != 1:
                    raise ValueError("Unsupported local job ledger schema.")
            with self._transaction() as connection:
                for row in connection.execute("SELECT * FROM jobs").fetchall():
                    if EvaluationJobState(row["state"]) in self.ACTIVE_STATES:
                        self._terminal(
                            connection=connection,
                            row=row,
                            state=EvaluationJobState.INTERRUPTED,
                            cleanup=EvaluationCleanupState.UNKNOWN,
                            evidence=EvaluationEvidenceState.UNKNOWN,
                            reason="restart_dispatch_uncertain",
                        )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release only this ledger's held process lock."""
        owner, self._owner = self._owner, None
        if owner is not None:
            try:
                self._lock_file(owner=owner, release=True)
            finally:
                owner.close()

    def submit(self, *, request: EvaluationJobRequest, actor_id: str) -> EvaluationJobSubmission:
        """
        Persist immutable actor/source/case/run admission exactly once.

        Returns:
            EvaluationJobSubmission: Acceptance, never execution or grading.

        Raises:
            EvaluationJobError: If actor, immutable admission, capacity, or dispatch certainty differs.
        """
        with self._transaction() as connection:
            previous = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (str(request.job_id),)).fetchone()
            if previous is not None:
                self._authorize(row=previous, actor_id=actor_id)
                if previous["request_sha256"] != request.request_sha256 or (
                    previous["request_json"] != request.model_dump_json()
                ):
                    raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
                return EvaluationJobSubmission(
                    job_id=request.job_id, request_sha256=request.request_sha256, duplicate=True
                )
            self._require_certain_dispatch(connection)
            if connection.execute(
                "SELECT 1 FROM jobs WHERE case_run_sha256 = ?", (request.case_run_sha256,)
            ).fetchone():
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            queued = connection.execute("SELECT count(*) FROM jobs WHERE state = 'queued'").fetchone()[0]
            if queued >= self.MAX_QUEUED:
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            capability = secrets.token_urlsafe(32) if request.controls else None
            connection.execute(
                "INSERT INTO jobs (job_id,case_run_sha256,actor_id,request_sha256,request_json,"
                "capability_sha256,state,evidence,cleanup,last_sequence) VALUES (?,?,?,?,?,?,?,?,?,0)",
                (
                    str(request.job_id),
                    request.case_run_sha256,
                    actor_id,
                    request.request_sha256,
                    request.model_dump_json(),
                    hashlib.sha256(capability.encode()).hexdigest() if capability is not None else None,
                    EvaluationJobState.QUEUED.value,
                    EvaluationEvidenceState.ABSENT.value,
                    EvaluationCleanupState.NOT_STARTED.value,
                ),
            )
            self._append(
                connection=connection,
                job_id=request.job_id,
                kind=EvaluationJobEventKind.SUBMITTED,
                state=EvaluationJobState.QUEUED,
            )
            return EvaluationJobSubmission(
                job_id=request.job_id, request_sha256=request.request_sha256, control_capability=capability
            )

    def snapshot(self, *, job_id: UUID, actor_id: str | None, after_sequence: int = 0) -> EvaluationJobSnapshot:
        """
        Read a coherent ordered page; None is reserved for the trusted consumer.

        Returns:
            EvaluationJobSnapshot: Exact durable state.

        Raises:
            EvaluationJobError: If actor or requested event cursor is invalid.
        """
        with self._transaction() as connection:
            row = self._row(connection=connection, job_id=job_id)
            if actor_id is not None:
                self._authorize(row=row, actor_id=actor_id)
            if after_sequence < 0 or after_sequence > row["last_sequence"]:
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            events = connection.execute(
                "SELECT event_json FROM job_events WHERE job_id = ? AND sequence > ? ORDER BY sequence LIMIT 256",
                (str(job_id), after_sequence),
            ).fetchall()
            return EvaluationJobSnapshot(
                request=EvaluationJobRequest.model_validate_json(row["request_json"]),
                request_sha256=row["request_sha256"],
                state=EvaluationJobState(row["state"]),
                evidence=EvaluationEvidenceState(row["evidence"]),
                cleanup=EvaluationCleanupState(row["cleanup"]),
                last_sequence=row["last_sequence"],
                events=tuple(EvaluationJobEvent.model_validate_json(item[0]) for item in events),
                boundary=EvaluationWaitBoundary.model_validate_json(row["boundary_json"])
                if row["boundary_json"] is not None
                else None,
                manifest=EvaluationArtifactManifest.model_validate_json(row["manifest_json"])
                if row["manifest_json"] is not None
                else None,
                canonical=EvaluationCanonicalReceipt.model_validate_json(row["canonical_json"])
                if row["canonical_json"] is not None
                else None,
                reason=row["reason"],
            )

    def queued(self) -> tuple[EvaluationJobDelivery, ...]:
        """
        Read durable queue entries without reconstructing or importing a Task.

        Returns:
            tuple[EvaluationJobDelivery, ...]: Previously admitted work.
        """
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id,request_sha256 FROM jobs WHERE state = 'queued' ORDER BY rowid LIMIT 64"
            ).fetchall()
        return tuple(EvaluationJobDelivery(job_id=UUID(row[0]), request_sha256=row[1]) for row in rows)

    def claim(
        self, *, delivery: EvaluationJobDelivery, allowed_actor_ids: frozenset[str]
    ) -> tuple[EvaluationDeliveryState, UUID | None]:
        """
        Commit one dispatch fence before any runtime invocation.

        Returns:
            tuple[EvaluationDeliveryState, UUID | None]: Busy is not an ACK; duplicates never rerun.

        Raises:
            EvaluationJobError: If actor, delivery identity, or dispatch certainty differs.
        """
        with self._transaction() as connection:
            row = self._row(connection=connection, job_id=delivery.job_id)
            if row["actor_id"] not in allowed_actor_ids:
                raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)
            if row["request_sha256"] != delivery.request_sha256:
                raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
            if EvaluationJobState(row["state"]) is not EvaluationJobState.QUEUED:
                return EvaluationDeliveryState.DUPLICATE, None
            self._require_certain_dispatch(connection)
            if connection.execute(
                "SELECT 1 FROM jobs WHERE state IN ('running','waiting','cancel_requested','finalizing')"
            ).fetchone():
                return EvaluationDeliveryState.BUSY, None
            fence_id = uuid4()
            connection.execute(
                "UPDATE jobs SET fence_id = ?, cleanup = ?, evidence = ? WHERE job_id = ?",
                (
                    str(fence_id),
                    EvaluationCleanupState.UNKNOWN.value,
                    EvaluationEvidenceState.UNKNOWN.value,
                    str(delivery.job_id),
                ),
            )
            self._append(
                connection=connection,
                job_id=delivery.job_id,
                kind=EvaluationJobEventKind.STARTED,
                state=EvaluationJobState.RUNNING,
            )
            return EvaluationDeliveryState.STARTED, fence_id

    def cancel(self, *, job_id: UUID, actor_id: str) -> bool:
        """
        Reject cancellation once canonical publication has become irreversible.

        Returns:
            bool: Whether an active runtime needs a cooperative cancellation signal.

        Raises:
            EvaluationJobError: If canonical publication already crossed its irreversible boundary.
        """
        with self._transaction() as connection:
            row = self._row(connection=connection, job_id=job_id)
            self._authorize(row=row, actor_id=actor_id)
            state = EvaluationJobState(row["state"])
            if state is EvaluationJobState.FINALIZING:
                raise EvaluationJobError(EvaluationJobErrorCode.CANCEL_TOO_LATE)
            if state.terminal:
                return False
            if state is EvaluationJobState.QUEUED:
                self._terminal(
                    connection=connection,
                    row=row,
                    state=EvaluationJobState.CANCELLED,
                    cleanup=EvaluationCleanupState.NOT_STARTED,
                    evidence=EvaluationEvidenceState.ABSENT,
                    reason="cancelled_before_dispatch",
                )
                return False
            if state is not EvaluationJobState.CANCEL_REQUESTED:
                self._append(
                    connection=connection,
                    job_id=job_id,
                    kind=EvaluationJobEventKind.CANCEL_REQUESTED,
                    state=EvaluationJobState.CANCEL_REQUESTED,
                )
                connection.execute("UPDATE jobs SET boundary_json = NULL WHERE job_id = ?", (str(job_id),))
            return True

    def open_boundary(self, *, job_id: UUID, fence_id: UUID, boundary: EvaluationWaitBoundary) -> None:
        """
        Open only a current runtime-owned boundary with admitted capabilities.

        Raises:
            EvaluationJobError: If the current fence, state, or capabilities differ.
        """
        with self._transaction() as connection:
            row = self._fenced_row(connection=connection, job_id=job_id, fence_id=fence_id)
            request = EvaluationJobRequest.model_validate_json(row["request_json"])
            if row["state"] != EvaluationJobState.RUNNING.value or not set(boundary.controls) <= set(request.controls):
                raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_CONTROL)
            connection.execute(
                "UPDATE jobs SET boundary_json = ? WHERE job_id = ?", (boundary.model_dump_json(), str(job_id))
            )
            self._append(
                connection=connection,
                job_id=job_id,
                kind=EvaluationJobEventKind.WAITING,
                state=EvaluationJobState.WAITING,
            )

    def control(
        self, *, job_id: UUID, actor_id: str, capability: str, command: EvaluationControlRequest
    ) -> EvaluationControlReceipt:
        """
        Admit one bounded command; no producer can create a runtime boundary.

        Returns:
            EvaluationControlReceipt: Durable command receipt, not an applied action.

        Raises:
            EvaluationJobError: If actor, capability, boundary, or immutable command differs.
        """
        with self._transaction() as connection:
            row = self._row(connection=connection, job_id=job_id)
            self._authorize(row=row, actor_id=actor_id)
            expected = row["capability_sha256"]
            if expected is None or not hmac.compare_digest(expected, hashlib.sha256(capability.encode()).hexdigest()):
                raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)
            previous = connection.execute(
                "SELECT * FROM job_controls WHERE job_id = ? AND command_id = ?",
                (str(job_id), str(command.command_id)),
            ).fetchone()
            if previous is not None:
                if previous["command_sha256"] != command.command_sha256:
                    raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
                return EvaluationControlReceipt(
                    job_id=job_id,
                    command_id=command.command_id,
                    accepted_sequence=previous["accepted_sequence"],
                    duplicate=True,
                )
            if row["boundary_json"] is None or row["state"] != EvaluationJobState.WAITING.value:
                raise EvaluationJobError(EvaluationJobErrorCode.STALE_BOUNDARY)
            boundary = EvaluationWaitBoundary.model_validate_json(row["boundary_json"])
            if command.boundary_id != boundary.boundary_id:
                raise EvaluationJobError(EvaluationJobErrorCode.STALE_BOUNDARY)
            if command.kind not in boundary.controls:
                raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_CONTROL)
            if (
                connection.execute("SELECT count(*) FROM job_controls WHERE job_id = ?", (str(job_id),)).fetchone()[0]
                >= self.MAX_CONTROLS
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_CONTROL)
            if connection.execute(
                "SELECT 1 FROM job_controls WHERE job_id = ? AND delivered = 0", (str(job_id),)
            ).fetchone():
                raise EvaluationJobError(EvaluationJobErrorCode.STALE_BOUNDARY)
            sequence = self._append(
                connection=connection,
                job_id=job_id,
                kind=EvaluationJobEventKind.CONTROL_ACCEPTED,
                state=EvaluationJobState.WAITING,
                command_id=command.command_id,
            )
            connection.execute(
                "INSERT INTO job_controls (job_id,command_id,command_sha256,command_json,accepted_sequence) "
                "VALUES (?,?,?,?,?)",
                (str(job_id), str(command.command_id), command.command_sha256, command.model_dump_json(), sequence),
            )
            return EvaluationControlReceipt(job_id=job_id, command_id=command.command_id, accepted_sequence=sequence)

    def take_control(self, *, job_id: UUID, fence_id: UUID) -> EvaluationControlRequest | None:
        """
        Consume exactly one command at the existing boundary, not a new Task.

        Returns:
            EvaluationControlRequest | None: A structured runtime command.

        Raises:
            EvaluationJobError: If the active fence or reviewed boundary is stale.
        """
        with self._transaction() as connection:
            row = self._fenced_row(connection=connection, job_id=job_id, fence_id=fence_id)
            if row["state"] == EvaluationJobState.CANCEL_REQUESTED.value:
                return None
            if row["state"] != EvaluationJobState.WAITING.value or row["boundary_json"] is None:
                raise EvaluationJobError(EvaluationJobErrorCode.STALE_BOUNDARY)
            pending = connection.execute(
                "SELECT * FROM job_controls WHERE job_id = ? AND delivered = 0 ORDER BY accepted_sequence LIMIT 1",
                (str(job_id),),
            ).fetchone()
            if pending is None:
                return None
            command = EvaluationControlRequest.model_validate_json(pending["command_json"])
            boundary = EvaluationWaitBoundary.model_validate_json(row["boundary_json"])
            if command.boundary_id != boundary.boundary_id:
                raise EvaluationJobError(EvaluationJobErrorCode.STALE_BOUNDARY)
            connection.execute(
                "UPDATE job_controls SET delivered = 1 WHERE job_id = ? AND command_id = ?",
                (str(job_id), str(command.command_id)),
            )
            connection.execute("UPDATE jobs SET boundary_json = NULL WHERE job_id = ?", (str(job_id),))
            self._append(
                connection=connection,
                job_id=job_id,
                kind=EvaluationJobEventKind.CONTROL_DELIVERED,
                state=EvaluationJobState.RUNNING,
                command_id=command.command_id,
            )
            return command

    def begin_finalizing(self, *, job_id: UUID, fence_id: UUID, manifest: EvaluationArtifactManifest) -> None:
        """
        Commit artifact retention and the cancellation/publication boundary atomically.

        Raises:
            EvaluationJobError: If cancellation, fence, or source artifact binding differs.
        """
        with self._transaction() as connection:
            row = self._fenced_row(connection=connection, job_id=job_id, fence_id=fence_id)
            if row["state"] == EvaluationJobState.CANCEL_REQUESTED.value:
                raise EvaluationJobError(EvaluationJobErrorCode.CANCEL_TOO_LATE)
            if row["state"] != EvaluationJobState.RUNNING.value or (
                manifest.request_sha256 != row["request_sha256"] or manifest.fence_id != fence_id
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            connection.execute(
                "UPDATE jobs SET manifest_json = ?, evidence = ?, cleanup = ? WHERE job_id = ?",
                (
                    manifest.model_dump_json(),
                    EvaluationEvidenceState.SOURCE_RETAINED.value,
                    EvaluationCleanupState.VERIFIED.value,
                    str(job_id),
                ),
            )
            self._append(
                connection=connection,
                job_id=job_id,
                kind=EvaluationJobEventKind.ARTIFACTS_RETAINED,
                state=EvaluationJobState.RUNNING,
            )
            self._append(
                connection=connection,
                job_id=job_id,
                kind=EvaluationJobEventKind.FINALIZING,
                state=EvaluationJobState.FINALIZING,
            )

    def finish(
        self,
        *,
        job_id: UUID,
        fence_id: UUID,
        state: EvaluationJobState,
        cleanup: EvaluationCleanupState,
        evidence: EvaluationEvidenceState,
        reason: str | None = None,
        canonical: EvaluationCanonicalReceipt | None = None,
        manifest: EvaluationArtifactManifest | None = None,
    ) -> None:
        """
        Write exactly one terminal event under the dispatch fence.

        Raises:
            EvaluationJobError: If transition, fence, artifact, or canonical receipt is inconsistent.
        """
        with self._transaction() as connection:
            row = self._fenced_row(connection=connection, job_id=job_id, fence_id=fence_id)
            if not state.terminal or EvaluationJobState(row["state"]).terminal:
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            finalizing = row["state"] == EvaluationJobState.FINALIZING.value
            if state is EvaluationJobState.SUCCEEDED and (
                not finalizing
                or canonical is None
                or not canonical.source_complete
                or cleanup is not EvaluationCleanupState.VERIFIED
                or evidence is not EvaluationEvidenceState.CANONICAL
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            if canonical is not None and not finalizing:
                raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
            if finalizing and state is EvaluationJobState.CANCELLED:
                raise EvaluationJobError(EvaluationJobErrorCode.CANCEL_TOO_LATE)
            if manifest is not None:
                if manifest.request_sha256 != row["request_sha256"] or manifest.fence_id != fence_id:
                    raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
                if row["manifest_json"] is not None and row["manifest_json"] != manifest.model_dump_json():
                    raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
                connection.execute(
                    "UPDATE jobs SET manifest_json = ? WHERE job_id = ?", (manifest.model_dump_json(), str(job_id))
                )
                row = self._row(connection=connection, job_id=job_id)
            if canonical is not None:
                if row["manifest_json"] is None:
                    raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
                manifest = EvaluationArtifactManifest.model_validate_json(row["manifest_json"])
                if (
                    canonical.job_id != job_id
                    or canonical.request_sha256 != row["request_sha256"]
                    or canonical.manifest_sha256 != manifest.manifest_sha256
                    or canonical.artifact_sha256 not in {artifact.sha256 for artifact in manifest.artifacts}
                ):
                    raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
                connection.execute(
                    "UPDATE jobs SET canonical_json = ? WHERE job_id = ?", (canonical.model_dump_json(), str(job_id))
                )
            self._terminal(
                connection=connection, row=row, state=state, cleanup=cleanup, evidence=evidence, reason=reason
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        with closing(sqlite3.connect(self.path, timeout=10)) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        if self._owner is None:
            raise EvaluationJobError(EvaluationJobErrorCode.CLOSED)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            yield connection

    def _acquire_owner(self) -> None:
        if self._owner is not None:
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        owner = (self.root / "queue.owner").open("a+b")
        try:
            if owner.tell() == 0:
                owner.write(b"1")
                owner.flush()
            self._lock_file(owner=owner, release=False)
        except OSError as error:
            owner.close()
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT) from error
        self._owner = owner

    @staticmethod
    def _lock_file(*, owner: BinaryIO, release: bool) -> None:
        import sys

        owner.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(owner.fileno(), msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(owner.fileno(), fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB)

    @staticmethod
    def _authorize(*, row: sqlite3.Row, actor_id: str) -> None:
        if row["actor_id"] != actor_id:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_AUTHORIZED)

    @staticmethod
    def _row(*, connection: sqlite3.Connection, job_id: UUID) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
        if row is None:
            raise EvaluationJobError(EvaluationJobErrorCode.NOT_FOUND)
        if not isinstance(row, sqlite3.Row):
            raise TypeError("Local job query must return a typed SQLite row.")
        return row

    @classmethod
    def _fenced_row(cls, *, connection: sqlite3.Connection, job_id: UUID, fence_id: UUID) -> sqlite3.Row:
        row = cls._row(connection=connection, job_id=job_id)
        if row["fence_id"] != str(fence_id):
            raise EvaluationJobError(EvaluationJobErrorCode.REQUEST_CONFLICT)
        return row

    @staticmethod
    def _require_certain_dispatch(connection: sqlite3.Connection) -> None:
        if connection.execute(
            "SELECT 1 FROM jobs WHERE state = 'interrupted' OR "
            "(state IN ('failed','cancelled') AND cleanup IN ('unknown','failed'))"
        ).fetchone():
            raise EvaluationJobError(EvaluationJobErrorCode.DISPATCH_UNCERTAIN)

    @classmethod
    def _append(
        cls,
        *,
        connection: sqlite3.Connection,
        job_id: UUID,
        kind: EvaluationJobEventKind,
        state: EvaluationJobState,
        command_id: UUID | None = None,
        reason: str | None = None,
    ) -> int:
        row = cls._row(connection=connection, job_id=job_id)
        if EvaluationJobState(row["state"]).terminal:
            raise EvaluationJobError(EvaluationJobErrorCode.ORDER_CONFLICT)
        previous_sequence = row["last_sequence"]
        if type(previous_sequence) is not int:
            raise ValueError("Local job sequence must be an integer.")
        sequence = previous_sequence + 1
        event = EvaluationJobEvent(
            sequence=sequence,
            kind=kind,
            state=state,
            occurred_at=datetime.now(UTC),
            command_id=command_id,
            reason=reason,
        )
        connection.execute(
            "INSERT INTO job_events (job_id,sequence,event_json) VALUES (?,?,?)",
            (str(job_id), sequence, event.model_dump_json()),
        )
        connection.execute(
            "UPDATE jobs SET last_sequence = ?, state = ? WHERE job_id = ?", (sequence, state.value, str(job_id))
        )
        return sequence

    @classmethod
    def _terminal(
        cls,
        *,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        state: EvaluationJobState,
        cleanup: EvaluationCleanupState,
        evidence: EvaluationEvidenceState,
        reason: str | None,
    ) -> None:
        job_id = UUID(row["job_id"])
        cls._append(
            connection=connection,
            job_id=job_id,
            kind=EvaluationJobEventKind.TERMINAL,
            state=state,
            reason=reason,
        )
        connection.execute(
            "UPDATE jobs SET cleanup = ?, evidence = ?, reason = ?, boundary_json = NULL WHERE job_id = ?",
            (cleanup.value, evidence.value, reason, str(job_id)),
        )
