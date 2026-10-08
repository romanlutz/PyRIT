# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local source mappings and commit watermarks, never a remote canonical database."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pyrit.common.local_file_lock import lock_local_file
from pyrit.models.evaluation_feedback import (
    EvaluationFeedbackArchive,
    EvaluationFeedbackCommit,
    EvaluationFeedbackControlReceipt,
    EvaluationFeedbackControlRequest,
    EvaluationFeedbackEvent,
    EvaluationFeedbackSession,
    EvaluationFeedbackTurn,
    EvaluationObservationCommit,
    EvaluationReadySnapshot,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import BinaryIO
    from uuid import UUID


class EvaluationFeedbackErrorCode(str, Enum):
    """Finite refusal reasons; raw source bodies and model content are never exception text."""

    CONFLICT = "feedback_conflict"
    GAP = "feedback_source_gap"
    STALE = "feedback_stale_snapshot"
    MEMORY = "feedback_memory_not_committed"
    SCORE = "feedback_score_not_committed"
    UNCERTAIN = "feedback_reconciliation_required"
    UNSUPPORTED = "feedback_unsupported"
    CLOSED = "feedback_closed"


class EvaluationFeedbackError(RuntimeError):
    """A source, persistence or readiness refusal that must stop the next input."""

    def __init__(self, code: EvaluationFeedbackErrorCode) -> None:
        """Expose only the finite refusal code, not retained observation or credential content."""
        super().__init__(code.value)
        self.code = code


class EvaluationFeedbackWitness(Protocol):
    """An optional independent committed-row witness, invoked before publishing readiness."""

    async def verify_writes_async(
        self,
        *,
        observation: EvaluationObservationCommit,
        feedback: EvaluationFeedbackCommit,
        candidate: EvaluationReadySnapshot,
    ) -> None:
        """Reject incomplete row visibility without treating the candidate as published readiness."""
        ...


class EvaluationWorkingMemoryJournal:
    """One process-owned durable source map; actual messages and scores stay in PyRIT memory."""

    MAX_BYTES = 16 * 1024 * 1024
    MAX_TURNS = 100

    def __init__(self, *, root: Path, session: EvaluationFeedbackSession) -> None:
        """
        Configure protected local custody without creating or replacing CentralMemory.

        Raises:
            ValueError: If the host-selected root is not explicitly local.
        """
        if not root.is_absolute() or str(root).startswith(("\\\\", "//")) or root.is_symlink():
            raise ValueError("Working-memory custody requires an absolute local non-symlink root.")
        self.root = root
        self.path = root / "feedback.sqlite"
        self.session = EvaluationFeedbackSession.model_validate(session)
        self._owner: BinaryIO | None = None

    def startup(self) -> None:
        """
        Bind one descriptor, then refuse automatic live continuation after any process restart.

        Raises:
            EvaluationFeedbackError: If custody or the installed descriptor differs.
        """
        if self._owner is not None:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.resolve() != self.root or self.path.is_symlink() or (self.root / "feedback.owner").is_symlink():
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        owner = (self.root / "feedback.owner").open("a+b")
        try:
            if owner.tell() == 0:
                owner.write(b"1")
                owner.flush()
            lock_local_file(owner=owner, release=False)
        except OSError as error:
            owner.close()
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT) from error
        self._owner = owner
        try:
            with closing(self._connect()) as connection, connection:
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS feedback_session (
                        singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                        schema_version INTEGER NOT NULL CHECK(schema_version=1),
                        descriptor_json TEXT NOT NULL,
                        conversation_id VARCHAR(36),
                        turn_index INTEGER NOT NULL DEFAULT 0,
                        source_cursor INTEGER NOT NULL DEFAULT 0,
                        stage VARCHAR(32) NOT NULL DEFAULT 'idle',
                        blocked_reason VARCHAR(64),
                        snapshot_json TEXT,
                        archive_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS feedback_turns (
                        turn_index INTEGER PRIMARY KEY,
                        stage VARCHAR(32) NOT NULL,
                        input_sha256 VARCHAR(64),
                        source_json TEXT,
                        observation_json TEXT,
                        feedback_json TEXT,
                        snapshot_json TEXT
                    );
                    CREATE TABLE IF NOT EXISTS feedback_controls (
                        command_id VARCHAR(36) PRIMARY KEY,
                        control_json TEXT NOT NULL,
                        receipt_json TEXT NOT NULL,
                        original_input_sha256 VARCHAR(64) NOT NULL,
                        turn_index INTEGER NOT NULL UNIQUE
                    );
                    """
                )
                row = connection.execute("SELECT * FROM feedback_session WHERE singleton=1").fetchone()
                if row is None:
                    connection.execute(
                        "INSERT INTO feedback_session(singleton,schema_version,descriptor_json) VALUES(1,1,?)",
                        (self.session.model_dump_json(),),
                    )
                elif row["schema_version"] != 1 or row["descriptor_json"] != self.session.model_dump_json():
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
                elif row["archive_json"] is None:
                    connection.execute(
                        "UPDATE feedback_session SET blocked_reason=?,stage='blocked' WHERE singleton=1",
                        (EvaluationFeedbackErrorCode.UNCERTAIN.value,),
                    )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release only the owned journal lock; source custody and working rows remain retained."""
        owner, self._owner = self._owner, None
        if owner is not None:
            try:
                lock_local_file(owner=owner, release=True)
            finally:
                owner.close()

    def begin_turn(self, *, conversation_id: UUID, turn_index: int, expected: EvaluationReadySnapshot | None) -> None:
        """
        Reserve next generation only after the exact previous committed snapshot.

        Raises:
            EvaluationFeedbackError: If a caller is stale, the source is blocked, or an input is already pending.
        """
        with closing(self._connect()) as connection, connection:
            row = self._session_row(connection)
            self._require_live(row)
            if row["stage"] == "reserved" and row["turn_index"] == turn_index:
                control = connection.execute(
                    "SELECT 1 FROM feedback_controls WHERE turn_index=?", (turn_index,)
                ).fetchone()
                if (
                    control is not None
                    and expected is not None
                    and row["snapshot_json"] == expected.model_dump_json()
                    and row["conversation_id"] == str(conversation_id)
                ):
                    return
            if (
                turn_index != row["turn_index"] + 1
                or turn_index > self.MAX_TURNS
                or row["stage"] not in {"idle", "ready"}
                or (row["conversation_id"] is not None and row["conversation_id"] != str(conversation_id))
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            if row["turn_index"] == 0:
                if expected is not None:
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            elif expected is None or row["snapshot_json"] != expected.model_dump_json():
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            connection.execute("INSERT INTO feedback_turns(turn_index,stage) VALUES(?,'reserved')", (turn_index,))
            connection.execute(
                "UPDATE feedback_session SET conversation_id=?,turn_index=?,stage='reserved' WHERE singleton=1",
                (str(conversation_id), turn_index),
            )

    def send_intent(self, *, conversation_id: UUID, input_sha256: str, original_input_sha256: str) -> int:
        """
        Persist one prepared-input intent before the reviewed transport is invoked.

        Returns:
            int: The reserved source turn, not proof of delivery or application.

        Raises:
            EvaluationFeedbackError: If generation/readiness was not reserved by its owner.
        """
        with closing(self._connect()) as connection, connection:
            row = self._session_row(connection)
            self._require_live(row)
            if row["stage"] != "reserved" or row["conversation_id"] != str(conversation_id):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            control = connection.execute(
                "SELECT original_input_sha256 FROM feedback_controls WHERE turn_index=?", (row["turn_index"],)
            ).fetchone()
            if control is not None and control["original_input_sha256"] != original_input_sha256:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            connection.execute(
                "UPDATE feedback_turns SET stage='sending',input_sha256=? WHERE turn_index=?",
                (input_sha256, row["turn_index"]),
            )
            connection.execute("UPDATE feedback_session SET stage='sending' WHERE singleton=1")
            return int(row["turn_index"])

    def reserve_control(
        self,
        *,
        control: EvaluationFeedbackControlRequest,
        ready: EvaluationReadySnapshot,
        original_input_sha256: str,
    ) -> EvaluationFeedbackControlReceipt:
        """
        Reserve one exact reviewed command under the actual ready memory snapshot.

        Returns:
            EvaluationFeedbackControlReceipt: Durable reservation, not source delivery or application.

        Raises:
            EvaluationFeedbackError: If the command replay or current readiness differs.
        """
        with closing(self._connect()) as connection, connection:
            previous = connection.execute(
                "SELECT * FROM feedback_controls WHERE command_id=?", (str(control.command.command_id),)
            ).fetchone()
            if previous is not None:
                if previous["control_json"] != control.model_dump_json():
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
                receipt = EvaluationFeedbackControlReceipt.model_validate_json(previous["receipt_json"])
                return receipt.model_copy(update={"duplicate": True})
            row = self._session_row(connection)
            self._require_live(row)
            if (
                row["stage"] != "ready"
                or row["snapshot_json"] != ready.model_dump_json()
                or control.session_sha256 != self.session.session_sha256
                or control.snapshot_sha256 != ready.snapshot_sha256
                or ready.generation + 1 > self.MAX_TURNS
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            receipt = EvaluationFeedbackControlReceipt(
                session_sha256=control.session_sha256,
                command_id=control.command.command_id,
                control_sha256=control.control_sha256,
                snapshot_sha256=control.snapshot_sha256,
                reserved_turn=ready.generation + 1,
            )
            connection.execute(
                "INSERT INTO feedback_controls(command_id,control_json,receipt_json,original_input_sha256,turn_index) "
                "VALUES(?,?,?,?,?)",
                (
                    str(control.command.command_id),
                    control.model_dump_json(),
                    receipt.model_dump_json(),
                    original_input_sha256,
                    receipt.reserved_turn,
                ),
            )
            connection.execute(
                "INSERT INTO feedback_turns(turn_index,stage) VALUES(?,'reserved')", (receipt.reserved_turn,)
            )
            connection.execute(
                "UPDATE feedback_session SET turn_index=?,stage='reserved' WHERE singleton=1", (receipt.reserved_turn,)
            )
            return receipt

    def previous_control(self, *, command_id: UUID) -> tuple[str, EvaluationFeedbackControlReceipt] | None:
        """
        Read immutable reservation deduplication without executing or re-reserving anything.

        Returns:
            tuple[str, EvaluationFeedbackControlReceipt] | None: Original request JSON and receipt, if retained.

        Raises:
            EvaluationFeedbackError: If the retained command has an invalid storage representation.
        """
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT control_json,receipt_json FROM feedback_controls WHERE command_id=?", (str(command_id),)
            ).fetchone()
            if row is None:
                return None
            serialized = row["control_json"]
            if not isinstance(serialized, str):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            return serialized, EvaluationFeedbackControlReceipt.model_validate_json(row["receipt_json"])

    def capture(self, turn: EvaluationFeedbackTurn) -> None:
        """
        Retain source coverage and the stable actual-row map before normalizer persistence.

        Raises:
            EvaluationFeedbackError: If raw/normalized coverage, source order or a replay differs.
        """
        turn = EvaluationFeedbackTurn.model_validate(turn)
        with closing(self._connect()) as connection, connection:
            row = self._session_row(connection)
            self._require_live(row)
            previous = self._turn_row(connection=connection, turn_index=turn.turn_index)
            if previous["source_json"] is not None:
                if previous["source_json"] != turn.model_dump_json():
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
                return
            if (
                row["stage"] != "sending"
                or turn.session_sha256 != self.session.session_sha256
                or row["turn_index"] != turn.turn_index
                or row["conversation_id"] != str(turn.conversation_id)
                or turn.events[0].source_sequence != row["source_cursor"] + 1
                or not turn.raw_complete
                or not turn.normalized_complete
                or turn.gaps
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            old = self._events(connection)
            if {item.source_event_id for item in old} & {item.source_event_id for item in turn.events}:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            stored_bytes = connection.execute(
                "SELECT COALESCE(SUM(length(source_json)),0) FROM feedback_turns"
            ).fetchone()[0]
            if stored_bytes + len(turn.model_dump_json().encode()) > self.MAX_BYTES:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            connection.execute(
                "UPDATE feedback_turns SET source_json=?,stage='captured' WHERE turn_index=?",
                (turn.model_dump_json(), turn.turn_index),
            )
            connection.execute("UPDATE feedback_session SET stage='captured' WHERE singleton=1")

    def commit_observation(self, commit: EvaluationObservationCommit) -> None:
        """
        Link only a verified readback of the captured turn's actual message rows.

        Raises:
            EvaluationFeedbackError: If a captured source or prior commit is changed.
        """
        commit = EvaluationObservationCommit.model_validate(commit)
        with closing(self._connect()) as connection, connection:
            row = self._session_row(connection)
            self._require_live(row)
            stored = self._turn_row(connection=connection, turn_index=commit.turn_index)
            turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
            if (
                commit.session_sha256 != self.session.session_sha256
                or commit.turn_sha256 != turn.turn_sha256
                or commit.conversation_id != turn.conversation_id
                or commit.source_through != turn.events[-1].source_sequence
                or commit.piece_ids != tuple(piece.piece_id for piece in turn.pieces)
                or commit.response_piece_ids != turn.response_piece_ids
                or (stored["observation_json"] is not None and stored["observation_json"] != commit.model_dump_json())
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            if stored["observation_json"] is not None:
                return
            if row["stage"] != "captured" or row["turn_index"] != commit.turn_index:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            connection.execute(
                "UPDATE feedback_turns SET observation_json=?,stage='observed' WHERE turn_index=?",
                (commit.model_dump_json(), commit.turn_index),
            )
            connection.execute(
                "UPDATE feedback_session SET source_cursor=?,stage='observed' WHERE singleton=1",
                (commit.source_through,),
            )

    def commit_feedback(self, *, commit: EvaluationFeedbackCommit, ready: EvaluationReadySnapshot) -> None:
        """
        Publish readiness only for the required feedback on the exact committed observation.

        Raises:
            EvaluationFeedbackError: If observation, feedback policy or generation identity differs.
        """
        commit = EvaluationFeedbackCommit.model_validate(commit)
        ready = EvaluationReadySnapshot.model_validate(ready)
        with closing(self._connect()) as connection, connection:
            row = self._session_row(connection)
            self._require_live(row)
            stored = self._turn_row(connection=connection, turn_index=ready.generation)
            observation = EvaluationObservationCommit.model_validate_json(stored["observation_json"])
            if (
                ready.session_sha256 != self.session.session_sha256
                or ready.memory_owner_id != self.session.memory_owner_id
                or ready.conversation_id != observation.conversation_id
                or ready.source_through != observation.source_through
                or ready.observation_sha256 != observation.observation_sha256
                or ready.feedback_sha256 != commit.feedback_sha256
                or commit.session_sha256 != self.session.session_sha256
                or commit.observation_sha256 != observation.observation_sha256
                or commit.scorer_sha256 != self.session.required_scorer_sha256
                or commit.expectation_sha256 != self.session.expectation_sha256
                or (stored["feedback_json"] is not None and stored["feedback_json"] != commit.model_dump_json())
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            if stored["feedback_json"] is not None:
                return
            if row["stage"] != "observed" or row["turn_index"] != ready.generation:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            connection.execute(
                "UPDATE feedback_turns SET feedback_json=?,snapshot_json=?,stage='ready' WHERE turn_index=?",
                (commit.model_dump_json(), ready.model_dump_json(), ready.generation),
            )
            connection.execute(
                "UPDATE feedback_session SET snapshot_json=?,stage='ready' WHERE singleton=1",
                (ready.model_dump_json(),),
            )

    def read_turn(self, turn_index: int) -> sqlite3.Row:
        """
        Read retained source and commit receipts without exposing them through job status.

        Returns:
            sqlite3.Row: One bounded local reconciliation record.
        """
        with closing(self._connect()) as connection:
            return self._turn_row(connection=connection, turn_index=turn_index)

    def read_session(self) -> sqlite3.Row:
        """
        Read the exact local owner/cursor state, never a canonical result.

        Returns:
            sqlite3.Row: The single descriptor-bound owner record.
        """
        with closing(self._connect()) as connection:
            return self._session_row(connection)

    def events(self) -> tuple[EvaluationFeedbackEvent, ...]:
        """
        Read the ordered trusted source object inventory for explicit final reconciliation.

        Returns:
            tuple[EvaluationFeedbackEvent, ...]: Exactly the retained source, not generated messages.
        """
        with closing(self._connect()) as connection:
            return self._events(connection)

    def final_archive(self) -> EvaluationFeedbackArchive:
        """
        Read immutable sealed custody after execution, without reacquiring a live owner.

        Returns:
            EvaluationFeedbackArchive: The exact final source receipt, not caller-supplied metadata.

        Raises:
            EvaluationFeedbackError: If execution is unsealed, blocked, substituted or unresolved.
        """
        if self.path.is_symlink() or self.root.resolve() != self.root:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        with closing(sqlite3.connect(f"{self.path.as_uri()}?mode=ro", uri=True)) as connection:
            connection.row_factory = sqlite3.Row
            row = self._session_row(connection)
            if (
                row["stage"] != "sealed"
                or row["archive_json"] is None
                or row["blocked_reason"] is not None
                or row["descriptor_json"] != self.session.model_dump_json()
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNCERTAIN)
            return EvaluationFeedbackArchive.model_validate_json(row["archive_json"])

    def seal_archive(self, receipt: EvaluationFeedbackArchive) -> None:
        """
        Close live continuation while linking the exact final artifact to existing working rows.

        Raises:
            EvaluationFeedbackError: If the ready/source identity or a previous final archive differs.
        """
        receipt = EvaluationFeedbackArchive.model_validate(receipt)
        with closing(self._connect()) as connection, connection:
            row = self._session_row(connection)
            if row["archive_json"] is not None:
                if row["archive_json"] != receipt.model_dump_json():
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
                return
            self._require_live(row)
            ready = EvaluationReadySnapshot.model_validate_json(row["snapshot_json"])
            if (
                row["stage"] != "ready"
                or receipt.session_sha256 != self.session.session_sha256
                or receipt.memory_owner_id != self.session.memory_owner_id
                or receipt.conversation_id != ready.conversation_id
                or receipt.source_through != row["source_cursor"]
                or receipt.last_snapshot_sha256 != ready.snapshot_sha256
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
            connection.execute(
                "UPDATE feedback_session SET archive_json=?,stage='sealed' WHERE singleton=1",
                (receipt.model_dump_json(),),
            )

    def block(self, code: EvaluationFeedbackErrorCode) -> None:
        """Durably revoke readiness without deleting source or pretending an input was cancelled."""
        with closing(self._connect()) as connection, connection:
            self._session_row(connection)
            connection.execute(
                "UPDATE feedback_session SET blocked_reason=COALESCE(blocked_reason,?),stage='blocked' "
                "WHERE singleton=1",
                (code.value,),
            )

    def _connect(self) -> sqlite3.Connection:
        if self._owner is None:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CLOSED)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN IMMEDIATE")
        return connection

    @staticmethod
    def _require_live(row: sqlite3.Row) -> None:
        if row["blocked_reason"] is not None or row["archive_json"] is not None:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNCERTAIN)

    @staticmethod
    def _session_row(connection: sqlite3.Connection) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM feedback_session WHERE singleton=1").fetchone()
        if not isinstance(row, sqlite3.Row):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        return row

    @staticmethod
    def _turn_row(*, connection: sqlite3.Connection, turn_index: int) -> sqlite3.Row:
        row = connection.execute("SELECT * FROM feedback_turns WHERE turn_index=?", (turn_index,)).fetchone()
        if not isinstance(row, sqlite3.Row):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
        return row

    @staticmethod
    def _events(connection: sqlite3.Connection) -> tuple[EvaluationFeedbackEvent, ...]:
        rows = connection.execute(
            "SELECT source_json FROM feedback_turns WHERE source_json IS NOT NULL ORDER BY turn_index"
        ).fetchall()
        return tuple(event for row in rows for event in EvaluationFeedbackTurn.model_validate_json(row[0]).events)
