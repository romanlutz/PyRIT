# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Runtime-owned ordered observation/feedback barriers, separate from final API archive import."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Never, ParamSpec, TypeVar
from uuid import NAMESPACE_URL, UUID, uuid5

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.memory.evaluation_working_memory import (
    EvaluationFeedbackError,
    EvaluationFeedbackErrorCode,
    EvaluationWorkingMemoryJournal,
)
from pyrit.models.evaluation_feedback import (
    EvaluationFeedbackArchive,
    EvaluationFeedbackCommit,
    EvaluationFeedbackControlReceipt,
    EvaluationFeedbackControlRequest,
    EvaluationFeedbackEvent,
    EvaluationFeedbackPiece,
    EvaluationFeedbackSession,
    EvaluationFeedbackSource,
    EvaluationFeedbackTurn,
    EvaluationObservationCommit,
    EvaluationReadySnapshot,
)
from pyrit.models.evaluation_job import EvaluationControlKind, EvaluationWaitBoundary
from pyrit.models.identifiers.component_identifier import config_hash
from pyrit.models.native_cyber import NativeAgentEvent, NativeAgentEvidence
from pyrit.models.score.score import ScoreStatus

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from pydantic import JsonValue

    from pyrit.memory.evaluation_working_memory import EvaluationFeedbackWitness
    from pyrit.models import Message, MessagePiece, Score, ScoringExpectation
    from pyrit.prompt_target import PromptTarget
    from pyrit.prompt_target.inspect_ghcp_target import InspectGhcpTurn
    from pyrit.prompt_target.native_agent_target import NativeAgentSession
    from pyrit.score import Scorer

logger = logging.getLogger(__name__)
_P = ParamSpec("_P")
_T = TypeVar("_T")


class EvaluationLiveFeedback:
    """Opt-in runtime coordinator; normalizers/scorers remain the actual PyRIT row writers."""

    INSPECT_LINEAGE_KEY = "pyrit_live_feedback_v1"
    MAX_ARCHIVE_BYTES = 16 * 1024 * 1024

    def __init__(
        self,
        *,
        root: Path,
        session: EvaluationFeedbackSession,
        memory: MemoryInterface,
        commit_witness: EvaluationFeedbackWitness | None = None,
    ) -> None:
        """Bind explicit runtime custody without changing the global memory owner or remote protocol."""
        self.session = EvaluationFeedbackSession.model_validate(session).model_copy(deep=True)
        self.memory = memory
        self.journal = EvaluationWorkingMemoryJournal(root=root, session=self.session)
        self._source_probe: Callable[[], NativeAgentEvidence] | None = None
        self._native_header_sha256: str | None = None
        self._target: PromptTarget | None = None
        self._scorer: Scorer | None = None
        self._lock = asyncio.Lock()
        self._quarantined = False
        self._commit_witness = commit_witness
        self._session_sha256 = self.session.session_sha256
        self._operator_steps = False
        self._source_max_turns = self.journal.MAX_TURNS

    @property
    def policy_sha256(self) -> str:
        """The behavioral source/feedback policy, excluding fresh dispatch and memory-owner IDs."""
        return config_hash(
            {
                "source": self.session.source.value,
                "scorer": self.session.required_scorer_sha256,
                "expectation": self.session.expectation_sha256,
                "required_rationale": True,
                "complete_normalized_coverage": True,
            }
        )

    async def startup_async(self) -> None:
        """Acquire local custody; restart is reconciliation-only, never automatic native replay."""
        await self._io_async(self.journal.startup)

    async def close_async(self) -> None:
        """Join journal operations and release only the runtime's local custody lock."""
        async with self._lock:
            await self._io_async(self.journal.close)

    async def execution_failed_async(self) -> None:
        """Revoke continuation after an actual attack/write failure; this is not physical source closure."""
        async with self._lock:
            self._quarantined = True
            logger.error("Local live feedback execution failed: %s", EvaluationFeedbackErrorCode.UNCERTAIN.value)
            await self._io_async(self.journal.block, EvaluationFeedbackErrorCode.UNCERTAIN)

    def bind_native(self, *, session: NativeAgentSession) -> None:
        """
        Bind the already installed SDK evidence probe; never create a provider or credentials.

        Raises:
            EvaluationFeedbackError: If the declared retained source differs.
        """
        if (
            self.session.source is not EvaluationFeedbackSource.NATIVE_SESSION
            or session.session_id != self.session.source_session_id
            or not session.capabilities.retained_session
            or self._source_probe is not None
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
        self._source_probe = session.evidence
        self._native_header_sha256 = self._native_header_sha(session.evidence())
        self._operator_steps = session.capabilities.operator_steps
        self._source_max_turns = min(session.capabilities.max_turns, self.journal.MAX_TURNS)

    def bind_inspect(
        self, *, source_probe: Callable[[], NativeAgentEvidence], max_turns: int, operator_steps: bool = False
    ) -> None:
        """
        Bind reviewed harness-owned live capture without changing its transport or metadata policy.

        Raises:
            EvaluationFeedbackError: If capture identity, installation or bounds are unsupported.
        """
        evidence = source_probe()
        if (
            self.session.source is not EvaluationFeedbackSource.INSPECT_GHCP
            or self._source_probe is not None
            or evidence.session_id != self.session.source_session_id
            or not 1 <= max_turns <= self.journal.MAX_TURNS
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
        self._source_probe = source_probe
        self._native_header_sha256 = self._native_header_sha(evidence)
        self._operator_steps = operator_steps
        self._source_max_turns = max_turns

    def validate_setup(
        self,
        *,
        memory: MemoryInterface,
        normalizer_memory: MemoryInterface,
        objective_target: PromptTarget,
        objective_scorer: Scorer,
    ) -> None:
        """
        Require actual shared attack/target/normalizer/scorer memory and the installed policy.

        Raises:
            EvaluationFeedbackError: If a component owner, capture seam or scorer differs.
        """
        if (
            memory is not self.memory
            or normalizer_memory is not self.memory
            or objective_target._memory is not self.memory
            or objective_scorer._memory is not self.memory
            or CentralMemory.get_memory_instance() is not self.memory
            or getattr(objective_target, "evaluation_feedback", None) is not self
            or not objective_target.configuration.capabilities.supports_multi_turn
            or objective_scorer.get_identifier().hash != self.session.required_scorer_sha256
            or self._target is not None
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
        self._target, self._scorer = objective_target, objective_scorer

    async def before_turn_async(
        self, *, conversation_id: str, turn_index: int, expectation: ScoringExpectation | None
    ) -> None:
        """
        Verify source and prior committed feedback before the next attack generates any input.

        Raises:
            EvaluationFeedbackError: If ownership, expectation, source or memory readiness differs.
        """
        async with self._guard_async():
            self._require_owner()
            if self.expectation_sha256(expectation) != self.session.expectation_sha256:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
            if turn_index > self._source_max_turns:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
            row = await self._io_async(self.journal.read_session)
            if row["turn_index"] == 0:
                if await self.memory.get_message_pieces_async(conversation_id=conversation_id):
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
                expected = None
            else:
                expected = await self._verify_ready_async(row["snapshot_json"])
            await self._verify_source_async()
            await self._io_async(
                self.journal.begin_turn,
                conversation_id=UUID(conversation_id),
                turn_index=turn_index,
                expected=expected,
            )

    async def before_send_async(self, *, request: Message) -> int:
        """
        Recheck source/readback immediately before the transport, then retain one input intent.

        Returns:
            int: The source watermark checked again by the installed native dispatch guard.

        Raises:
            EvaluationFeedbackError: If the prepared input or latest source/feedback snapshot differs.
        """
        async with self._guard_async():
            self._require_owner()
            if request.api_role != "user" or any(
                piece.converted_value_data_type != "text" for piece in request.message_pieces
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
            row = await self._io_async(self.journal.read_session)
            if row["snapshot_json"] is not None:
                await self._verify_ready_async(row["snapshot_json"])
            await self._verify_source_async()
            await self._io_async(
                self.journal.send_intent,
                conversation_id=UUID(request.conversation_id),
                input_sha256=self._text_sha256("\n".join(request.get_values())),
                original_input_sha256=self._text_sha256(
                    "\n".join(piece.original_value for piece in request.message_pieces)
                ),
            )
            return int(row["source_cursor"])

    async def capture_native_async(
        self, *, request: Message, responses: Sequence[Message], evidence: NativeAgentEvidence
    ) -> None:
        """
        Stage honest typed-event coverage and stable source/tool-to-row identities before writes.

        Raises:
            EvaluationFeedbackError: If source coverage, replay or the normalized projection is incomplete.
        """
        async with self._guard_async():
            if (
                self.session.source is not EvaluationFeedbackSource.NATIVE_SESSION
                or evidence.session_id != self.session.source_session_id
                or not evidence.coverage_complete
                or not evidence.idle
                or evidence.gaps
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            if self._native_header_sha(evidence) != self._native_header_sha256 or any(
                event.session_id != self.session.source_session_id
                or event.payload.get("sessionId", self.session.source_session_id) != self.session.source_session_id
                for event in evidence.events
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            row = await self._io_async(self.journal.read_session)
            retained = await self._io_async(self.journal.events)
            events = tuple(self._native_event(event) for event in evidence.events)
            if events[: len(retained)] != retained:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            new = events[len(retained) :]
            if not new or new[-1].event_type != "session.idle":
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            turn = self._native_turn(
                request=request, responses=responses, events=new, turn_index=int(row["turn_index"])
            )
            await self._io_async(self.journal.capture, turn)

    async def capture_inspect_async(self, *, request: Message, response: Message, turn: InspectGhcpTurn) -> None:
        """
        Stage only a reviewed retained frame with real event IDs and an explicit idle boundary.

        Raises:
            EvaluationFeedbackError: If the frame is unbound, lacks a boundary or has unsupported projection.
        """
        async with self._guard_async():
            if (
                self.session.source is not EvaluationFeedbackSource.INSPECT_GHCP
                or turn.session_id != self.session.source_session_id
                or not turn.events
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            retained = await self._io_async(self.journal.events)
            events = tuple(
                self._inspect_event(payload=payload, sequence=len(retained) + index + 1)
                for index, payload in enumerate(turn.events)
            )
            user = [event for event in events if event.event_type == "user.message"]
            if (
                self._source_probe is None
                or len(user) != 1
                or self._event_data(user[0]).get("content") != turn.instruction
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            observed = self._source_probe()
            actual = tuple(self._native_event(event) for event in observed.events)
            if (
                actual != (*retained, *events)
                or self._native_header_sha(observed) != self._native_header_sha256
                or not observed.idle
                or not observed.coverage_complete
                or observed.gaps
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            last = events[-1]
            data = last.payload.get("data")
            if (
                last.event_type != "session.idle"
                or last.payload.get("agentId")
                or not isinstance(data, dict)
                or data.get("aborted") is True
                or any(event.event_type == "session.error" for event in events)
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            assistant = [
                event
                for event in events
                if event.event_type == "assistant.message"
                and self._event_data(event).get("content") == turn.assistant_text
            ]
            if len(assistant) != 1 or any(
                event.event_type.startswith("tool.")
                or self._event_data(event).get("toolRequests")
                or (
                    event.event_type == "assistant.message"
                    and self._event_data(event).get("content")
                    and event.source_event_id != assistant[0].source_event_id
                )
                or event.payload.get("sessionId", turn.session_id) != turn.session_id
                for event in events
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            user_message_id = self._event_data(user[0]).get("messageId")
            response.get_piece().prompt_metadata["evaluation_source_event_id"] = assistant[0].source_event_id
            response.get_piece().prompt_metadata["evaluation_source_session_id"] = turn.session_id
            source = EvaluationFeedbackTurn(
                session_sha256=self.session.session_sha256,
                conversation_id=UUID(request.conversation_id),
                turn_index=turn.turn_index,
                events=events,
                pieces=(
                    self._piece_map(
                        piece=request.get_piece(),
                        event_id=user[0].source_event_id,
                        part_index=0,
                        message_id=user_message_id if isinstance(user_message_id, str) else None,
                    ),
                    self._piece_map(piece=response.get_piece(), event_id=assistant[0].source_event_id, part_index=0),
                ),
                response_piece_ids=(response.get_piece().id,),
                boundary_source_event_id=last.source_event_id,
                raw_only_event_ids=tuple(
                    event.source_event_id
                    for event in events
                    if event.source_event_id not in {user[0].source_event_id, assistant[0].source_event_id}
                ),
                raw_complete=True,
                normalized_complete=True,
            )
            await self._io_async(self.journal.capture, source)

    async def response_committed_async(self, *, response: Message) -> None:
        """
        Verify actual normalizer writes through the memory API before required scoring starts.

        Raises:
            EvaluationFeedbackError: If a mapped message is absent, changed or out of order.
        """
        async with self._guard_async():
            self._require_owner()
            row = await self._io_async(self.journal.read_session)
            stored = await self._io_async(self.journal.read_turn, int(row["turn_index"]))
            turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
            if tuple(piece.id for piece in response.message_pieces) != turn.response_piece_ids or response.is_error():
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
            pieces = await self._read_turn_pieces_async(turn)
            await self._verify_conversation_async(turn_index=turn.turn_index, conversation_id=turn.conversation_id)
            request = [piece for piece in pieces if piece.role == "user"]
            if self._text_sha256("\n".join(piece.converted_value for piece in request)) != stored["input_sha256"]:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
            commit = EvaluationObservationCommit(
                session_sha256=self.session.session_sha256,
                conversation_id=turn.conversation_id,
                turn_index=turn.turn_index,
                source_through=turn.events[-1].source_sequence,
                turn_sha256=turn.turn_sha256,
                normalized_sha256=self._normalized_sha256(pieces),
                piece_ids=tuple(piece.piece_id for piece in turn.pieces),
                response_piece_ids=turn.response_piece_ids,
            )
            await self._io_async(self.journal.commit_observation, commit)

    async def feedback_committed_async(
        self,
        *,
        response: Message,
        scores: Sequence[Score],
        expectation: ScoringExpectation | None,
    ) -> None:
        """
        Read required COMPLETE scores/rationales back before publishing one ready watermark.

        Raises:
            EvaluationFeedbackError: If feedback is absent, incomplete, stale or under another policy.
        """
        async with self._guard_async():
            self._require_owner()
            row = await self._io_async(self.journal.read_session)
            stored = await self._io_async(self.journal.read_turn, int(row["turn_index"]))
            observation = EvaluationObservationCommit.model_validate_json(stored["observation_json"])
            if tuple(piece.id for piece in response.message_pieces) != observation.response_piece_ids:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
            if self.expectation_sha256(expectation) != self.session.expectation_sha256:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
            await self._verify_observation_async(observation)
            persisted = await self._read_scores_async(
                score_ids=tuple(UUID(str(score.id)) for score in scores), response_ids=observation.response_piece_ids
            )
            if self._scores_sha256(persisted) != self._scores_sha256(scores):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
            commit = EvaluationFeedbackCommit(
                session_sha256=self.session.session_sha256,
                observation_sha256=observation.observation_sha256,
                scorer_sha256=self.session.required_scorer_sha256,
                expectation_sha256=self.session.expectation_sha256,
                score_ids=tuple(UUID(str(score.id)) for score in persisted),
                scores_sha256=self._scores_sha256(persisted),
            )
            ready = EvaluationReadySnapshot(
                session_sha256=self.session.session_sha256,
                memory_owner_id=self.session.memory_owner_id,
                conversation_id=observation.conversation_id,
                generation=observation.turn_index,
                source_through=observation.source_through,
                observation_sha256=observation.observation_sha256,
                feedback_sha256=commit.feedback_sha256,
            )
            if self._commit_witness is not None:
                await self._commit_witness.verify_writes_async(
                    observation=observation, feedback=commit, candidate=ready
                )
            self._require_owner()
            await self._verify_observation_async(observation)
            await self._verify_conversation_async(
                turn_index=observation.turn_index, conversation_id=observation.conversation_id
            )
            await self._verify_source_async()
            current_scores = await self._read_scores_async(
                score_ids=commit.score_ids, response_ids=observation.response_piece_ids
            )
            if self._scores_sha256(current_scores) != commit.scores_sha256:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
            await self._io_async(self.journal.commit_feedback, commit=commit, ready=ready)

    async def ready_snapshot_async(self) -> EvaluationReadySnapshot:
        """
        Revalidate the exact current memory/scorer/source watermark, not only journal state.

        Returns:
            EvaluationReadySnapshot: The verified local continuation watermark.

        Raises:
            EvaluationFeedbackError: If source or committed row readiness has changed.
        """
        async with self._lock:
            row = await self._io_async(self.journal.read_session)
            if row["stage"] != "ready":
                self._refuse_control(EvaluationFeedbackErrorCode.STALE)
            async with self._quarantine_errors_async():
                ready = await self._verify_ready_async(row["snapshot_json"])
                await self._verify_source_async()
                return ready

    async def control_boundary_async(self) -> EvaluationWaitBoundary:
        """
        Name a reviewed next-input boundary using the exact memory/scorer watermark.

        Returns:
            EvaluationWaitBoundary: Only installed send-message capability, not a lifecycle transition.

        Raises:
            EvaluationFeedbackError: If operator continuation or current readiness is not installed.
        """
        self._require_control_capability()
        ready = await self.ready_snapshot_async()
        if ready.generation >= self._source_max_turns:
            self._refuse_control(EvaluationFeedbackErrorCode.UNSUPPORTED)
        return EvaluationWaitBoundary(
            boundary_id=self._boundary_id(ready),
            name="working_memory_ready",
            controls=(EvaluationControlKind.SEND_MESSAGE,),
        )

    async def reserve_control_async(
        self, *, control: EvaluationFeedbackControlRequest
    ) -> EvaluationFeedbackControlReceipt:
        """
        Validate and reserve a command before any source input; do not deliver or claim application.

        Returns:
            EvaluationFeedbackControlReceipt: Idempotent local input reservation only.

        Raises:
            EvaluationFeedbackError: If capability, command replay, boundary or current memory differs.
        """
        control = EvaluationFeedbackControlRequest.model_validate(control).model_copy(deep=True)
        async with self._lock:
            self._require_control_capability()
            self._require_owner()
            if (
                control.session_sha256 != self.session.session_sha256
                or control.command.kind is not EvaluationControlKind.SEND_MESSAGE
                or control.command.message is None
                or not control.command.message.strip()
            ):
                self._refuse_control(EvaluationFeedbackErrorCode.UNSUPPORTED)
            previous = await self._io_async(self.journal.previous_control, command_id=control.command.command_id)
            if previous is not None:
                if previous[0] != control.model_dump_json():
                    self._refuse_control(EvaluationFeedbackErrorCode.CONFLICT)
                return previous[1].model_copy(update={"duplicate": True})
            row = await self._io_async(self.journal.read_session)
            serialized = row["snapshot_json"]
            if row["stage"] != "ready" or serialized is None:
                self._refuse_control(EvaluationFeedbackErrorCode.STALE)
            ready = EvaluationReadySnapshot.model_validate_json(serialized)
            if ready.generation >= self._source_max_turns:
                self._refuse_control(EvaluationFeedbackErrorCode.UNSUPPORTED)
            if control.snapshot_sha256 != ready.snapshot_sha256 or control.command.boundary_id != self._boundary_id(
                ready
            ):
                self._refuse_control(EvaluationFeedbackErrorCode.STALE)
            async with self._quarantine_errors_async():
                await self._verify_ready_async(serialized)
                await self._verify_source_async()
                return await self._io_async(
                    self.journal.reserve_control,
                    control=control,
                    ready=ready,
                    original_input_sha256=self._text_sha256(control.command.message),
                )

    async def reconcile_native_archive_async(self, *, content: bytes) -> EvaluationFeedbackArchive:
        """
        Link exact final native source serialization without inserting messages or scores again.

        Returns:
            EvaluationFeedbackArchive: A local lineage receipt, not an API canonical import.

        Raises:
            EvaluationFeedbackError: If the final archive is not the retained complete source.
        """
        async with self._guard_async():
            if not 0 < len(content) <= self.MAX_ARCHIVE_BYTES:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
            evidence = NativeAgentEvidence.model_validate_json(content)
            if (
                evidence.session_id != self.session.source_session_id
                or not evidence.coverage_complete
                or not evidence.idle
                or evidence.gaps
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
            if self._source_probe is None or config_hash(evidence.model_dump(mode="json")) != config_hash(
                self._source_probe().model_dump(mode="json")
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            events = tuple(self._native_event(event) for event in evidence.events)
            return await self._seal_archive_async(content=content, events=events, media_type="application/x-ndjson")

    async def reconcile_inspect_archive_async(self, *, content: bytes, task_name: str) -> EvaluationFeedbackArchive:
        """
        Reconcile the explicit controlled-Task lineage, not arbitrary Inspect logs or live imports.

        Returns:
            EvaluationFeedbackArchive: Exact final artifact custody linked to existing working rows.

        Raises:
            EvaluationFeedbackError: If the controlled Task's final typed source lineage differs.
        """
        async with self._guard_async():
            if not 0 < len(content) <= self.MAX_ARCHIVE_BYTES:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
            events = await self._io_async(self._inspect_archive_events, content=content, task_name=task_name)
            return await self._seal_archive_async(content=content, events=events, media_type="application/octet-stream")

    async def _seal_archive_async(
        self, *, content: bytes, events: tuple[EvaluationFeedbackEvent, ...], media_type: str
    ) -> EvaluationFeedbackArchive:
        retained = await self._io_async(self.journal.events)
        if retained != events:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        await self._verify_source_async()
        row = await self._io_async(self.journal.read_session)
        ready = await self._verify_ready_async(row["snapshot_json"])
        receipt = EvaluationFeedbackArchive(
            session_sha256=self.session.session_sha256,
            memory_owner_id=self.session.memory_owner_id,
            conversation_id=ready.conversation_id,
            source_through=ready.source_through,
            last_snapshot_sha256=ready.snapshot_sha256,
            archive_sha256=hashlib.sha256(content).hexdigest(),
            archive_bytes=len(content),
            media_type=media_type,
            source_sha256=config_hash({"events": [event.model_dump(mode="json") for event in retained]}),
        )
        await self._io_async(self._retain_archive, receipt=receipt, content=content)
        await self._io_async(self.journal.seal_archive, receipt)
        return receipt

    async def _verify_ready_async(self, serialized: str | None) -> EvaluationReadySnapshot:
        self._require_owner()
        if serialized is None:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
        ready = EvaluationReadySnapshot.model_validate_json(serialized)
        stored = await self._io_async(self.journal.read_turn, ready.generation)
        observation = EvaluationObservationCommit.model_validate_json(stored["observation_json"])
        feedback = EvaluationFeedbackCommit.model_validate_json(stored["feedback_json"])
        if (
            ready.session_sha256 != self.session.session_sha256
            or ready.memory_owner_id != self.session.memory_owner_id
            or ready.observation_sha256 != observation.observation_sha256
            or ready.feedback_sha256 != feedback.feedback_sha256
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.STALE)
        await self._verify_observation_async(observation)
        await self._verify_conversation_async(turn_index=ready.generation, conversation_id=ready.conversation_id)
        scores = await self._read_scores_async(
            score_ids=feedback.score_ids, response_ids=observation.response_piece_ids
        )
        if self._scores_sha256(scores) != feedback.scores_sha256:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
        return ready

    async def _verify_observation_async(self, observation: EvaluationObservationCommit) -> None:
        stored = await self._io_async(self.journal.read_turn, observation.turn_index)
        turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
        pieces = await self._read_turn_pieces_async(turn)
        if (
            observation.session_sha256 != self.session.session_sha256
            or turn.turn_sha256 != observation.turn_sha256
            or self._normalized_sha256(pieces) != observation.normalized_sha256
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)

    async def _read_turn_pieces_async(self, turn: EvaluationFeedbackTurn) -> tuple[MessagePiece, ...]:
        persisted = await self.memory.get_message_pieces_async(
            conversation_id=turn.conversation_id, prompt_ids=tuple(piece.piece_id for piece in turn.pieces)
        )
        by_id = {piece.id: piece for piece in persisted}
        if len(by_id) != len(turn.pieces):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
        ordered = tuple(by_id[piece.piece_id] for piece in turn.pieces)
        for source, piece in zip(turn.pieces, ordered, strict=True):
            if (
                piece.not_in_memory
                or piece.original_prompt_id != source.original_prompt_id
                or piece.role != source.role
                or piece.original_value_data_type != source.data_type
                or self._text_sha256(piece.original_value) != source.original_value_sha256
                or piece.prompt_metadata.get("evaluation_source_event_id") != source.source_event_id
                or piece.prompt_metadata.get("evaluation_source_session_id") != self.session.source_session_id
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
        return ordered

    async def _verify_conversation_async(self, *, turn_index: int, conversation_id: UUID) -> None:
        expected: list[UUID] = []
        for index in range(1, turn_index + 1):
            stored = await self._io_async(self.journal.read_turn, index)
            turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
            expected.extend(piece.piece_id for piece in turn.pieces)
        pieces = await self.memory.get_message_pieces_async(conversation_id=conversation_id)
        if len(pieces) != len(expected) or {piece.id for piece in pieces} != set(expected):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
        positions = {piece.id: piece.sequence for piece in pieces}
        sequences = [positions[piece_id] for piece_id in expected]
        unique = sorted(set(sequences))
        if sequences != sorted(sequences) or unique != list(range(len(unique))):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)

    async def _read_scores_async(
        self, *, score_ids: tuple[UUID, ...], response_ids: tuple[UUID, ...]
    ) -> tuple[Score, ...]:
        if not score_ids or len(set(score_ids)) != len(score_ids):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
        persisted = await self.memory.get_scores_async(score_ids=tuple(str(score_id) for score_id in score_ids))
        by_id = {UUID(str(score.id)): score for score in persisted}
        if set(by_id) != set(score_ids):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
        ordered = tuple(by_id[score_id] for score_id in score_ids)
        for score in ordered:
            if (
                score.status is not ScoreStatus.COMPLETE
                or score.score_value is None
                or not score.score_rationale
                or not score.score_rationale.strip()
                or score.scorer_class_identifier is None
                or score.scorer_class_identifier.hash != self.session.required_scorer_sha256
                or score.message_piece_id is None
                or UUID(str(score.message_piece_id)) not in response_ids
                or self.expectation_sha256(score.scored_expectation) != self.session.expectation_sha256
            ):
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.SCORE)
        return ordered

    async def _verify_source_async(self) -> None:
        if self._source_probe is None:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNSUPPORTED)
        evidence = self._source_probe()
        retained = await self._io_async(self.journal.events)
        events = tuple(self._native_event(event) for event in evidence.events)
        if (
            events != retained
            or evidence.session_id != self.session.source_session_id
            or self._native_header_sha(evidence) != self._native_header_sha256
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        if retained and (not evidence.coverage_complete or not evidence.idle or evidence.gaps):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)

    def _require_owner(self) -> None:
        if (
            self._quarantined
            or CentralMemory.get_memory_instance() is not self.memory
            or self._target is None
            or self._target._memory is not self.memory
            or self._scorer is None
            or self._scorer._memory is not self.memory
            or self._scorer.get_identifier().hash != self.session.required_scorer_sha256
            or self.session.session_sha256 != self._session_sha256
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNCERTAIN)

    @asynccontextmanager
    async def _guard_async(self) -> AsyncIterator[None]:
        async with self._lock:
            async with self._quarantine_errors_async():
                yield

    @asynccontextmanager
    async def _quarantine_errors_async(self) -> AsyncIterator[None]:
        if self._quarantined:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.UNCERTAIN)
        try:
            yield
        except BaseException as error:
            self._quarantined = True
            code = error.code if isinstance(error, EvaluationFeedbackError) else EvaluationFeedbackErrorCode.UNCERTAIN
            logger.error("Local live feedback refused: %s", code.value)
            await self._io_async(self.journal.block, code)
            raise

    def _require_control_capability(self) -> None:
        if not self._operator_steps or EvaluationControlKind.SEND_MESSAGE not in self.session.request.controls:
            self._refuse_control(EvaluationFeedbackErrorCode.UNSUPPORTED)

    @staticmethod
    def _refuse_control(code: EvaluationFeedbackErrorCode) -> Never:
        logger.error("Local live feedback control refused: %s", code.value)
        raise EvaluationFeedbackError(code)

    @staticmethod
    def _boundary_id(ready: EvaluationReadySnapshot) -> UUID:
        return uuid5(NAMESPACE_URL, f"pyrit-live-ready:{ready.session_sha256}:{ready.snapshot_sha256}")

    @staticmethod
    async def _io_async(operation: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
        task = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
        cancelled = False
        while True:
            try:
                value = await asyncio.shield(task)
                if cancelled:
                    raise asyncio.CancelledError
                return value
            except asyncio.CancelledError:
                if task.done():
                    task.result()
                    raise
                cancelled = True

    def _native_turn(
        self,
        *,
        request: Message,
        responses: Sequence[Message],
        events: tuple[EvaluationFeedbackEvent, ...],
        turn_index: int,
    ) -> EvaluationFeedbackTurn:
        by_id = {event.source_event_id: event for event in events}
        user_events = [event for event in events if event.event_type == "user.message"]
        if (
            len(request.message_pieces) != 1
            or len(user_events) != 1
            or self._event_data(user_events[0]).get("content") != "\n".join(request.get_values())
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
        user = user_events[0]
        user_message_id = self._event_data(user).get("messageId")
        pieces = [
            self._piece_map(
                piece=piece,
                event_id=user.source_event_id,
                part_index=index,
                message_id=user_message_id if isinstance(user_message_id, str) else None,
            )
            for index, piece in enumerate(request.message_pieces)
        ]
        projected: set[str] = {user.source_event_id}
        for response in responses:
            for index, piece in enumerate(response.message_pieces):
                event_id = piece.prompt_metadata.get("native_event_id")
                if not isinstance(event_id, str) or event_id not in by_id:
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
                event = by_id[event_id]
                data = event.payload.get("data")
                if not isinstance(data, dict):
                    raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
                self._verify_native_projection(piece=piece, event=event)
                piece.prompt_metadata["evaluation_source_event_id"] = event_id
                piece.prompt_metadata["evaluation_source_session_id"] = self.session.source_session_id
                call_id = data.get("toolCallId")
                message_id = data.get("messageId")
                pieces.append(
                    self._piece_map(
                        piece=piece,
                        event_id=event_id,
                        part_index=index,
                        tool_call_id=call_id if isinstance(call_id, str) else None,
                        message_id=message_id if isinstance(message_id, str) else None,
                    )
                )
                projected.add(event_id)
        required = {
            event.source_event_id
            for event in events
            if event.event_type in {"user.message", "tool.execution_start", "tool.execution_complete"}
            or (event.event_type == "assistant.message" and self._event_data(event).get("content"))
        }
        if projected != required or not responses or responses[-1].api_role != "assistant":
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
        return EvaluationFeedbackTurn(
            session_sha256=self.session.session_sha256,
            conversation_id=UUID(request.conversation_id),
            turn_index=turn_index,
            events=events,
            pieces=tuple(pieces),
            response_piece_ids=tuple(piece.id for piece in responses[-1].message_pieces),
            boundary_source_event_id=events[-1].source_event_id,
            raw_only_event_ids=tuple(
                event.source_event_id for event in events if event.source_event_id not in projected
            ),
            raw_complete=True,
            normalized_complete=True,
        )

    @staticmethod
    def _verify_native_projection(*, piece: MessagePiece, event: EvaluationFeedbackEvent) -> None:
        data = EvaluationLiveFeedback._event_data(event)
        if event.event_type == "assistant.message":
            valid = (
                piece.role == "assistant"
                and piece.original_value_data_type == "text"
                and piece.original_value == data.get("content")
            )
        elif event.event_type in {"tool.execution_start", "tool.execution_complete"}:
            expected_role = "assistant" if event.event_type == "tool.execution_start" else "tool"
            expected_type = "function_call" if expected_role == "assistant" else "function_call_output"
            retained = json.dumps(
                json.loads(piece.original_value),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
            valid = (
                piece.role == expected_role
                and piece.original_value_data_type == expected_type
                and retained == event.payload_bytes()
            )
        else:
            valid = False
        if not valid:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)

    def _piece_map(
        self,
        *,
        piece: MessagePiece,
        event_id: str,
        part_index: int,
        tool_call_id: str | None = None,
        message_id: str | None = None,
    ) -> EvaluationFeedbackPiece:
        if piece.not_in_memory:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
        if piece.original_prompt_id is None:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY)
        piece.prompt_metadata["evaluation_source_event_id"] = event_id
        piece.prompt_metadata["evaluation_source_session_id"] = self.session.source_session_id
        if piece.role != "user":
            identity = uuid5(
                NAMESPACE_URL,
                f"pyrit-live:{self.session.session_sha256}:{piece.conversation_id}:{event_id}:{part_index}",
            )
            piece.id = identity
            piece.original_prompt_id = identity
        return EvaluationFeedbackPiece(
            source_event_id=event_id,
            source_part_index=part_index,
            source_message_id=message_id,
            tool_call_id=tool_call_id,
            piece_id=piece.id,
            original_prompt_id=piece.original_prompt_id,
            role=piece.role,
            data_type=piece.original_value_data_type,
            original_value_sha256=EvaluationLiveFeedback._text_sha256(piece.original_value),
        )

    @staticmethod
    def _native_event(event: NativeAgentEvent) -> EvaluationFeedbackEvent:
        return EvaluationFeedbackEvent(
            source_sequence=event.sequence,
            source_event_id=event.event_id,
            event_type=event.event_type,
            payload=event.payload,
        )

    @staticmethod
    def _native_header_sha(evidence: NativeAgentEvidence) -> str:
        return config_hash(
            evidence.model_dump(
                mode="json", include={"session_id", "environment_id", "simulated", "scope", "provenance"}
            )
        )

    @staticmethod
    def _event_data(event: EvaluationFeedbackEvent) -> dict[str, JsonValue]:
        data = event.payload.get("data")
        if not isinstance(data, dict):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
        return data

    @staticmethod
    def _inspect_event(*, payload: dict[str, JsonValue], sequence: int) -> EvaluationFeedbackEvent:
        event_id, kind = payload.get("id"), payload.get("type")
        if not isinstance(event_id, str) or not event_id or not isinstance(kind, str) or not kind:
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
        return EvaluationFeedbackEvent.model_validate(
            {"source_sequence": sequence, "source_event_id": event_id, "event_type": kind, "payload": payload}
        )

    @staticmethod
    def _normalized_sha256(pieces: Sequence[MessagePiece]) -> str:
        return config_hash(
            {
                "pieces": [
                    piece.model_dump(
                        mode="json",
                        exclude={"timestamp", "prompt_metadata", "response_error_reason", "original_value_sha256"},
                    )
                    for piece in pieces
                ]
            }
        )

    @staticmethod
    def _scores_sha256(scores: Sequence[Score]) -> str:
        return config_hash({"scores": [score.model_dump(mode="json", exclude={"timestamp"}) for score in scores]})

    @staticmethod
    def _text_sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def expectation_sha256(expectation: ScoringExpectation | None) -> str:
        """
        Fingerprint the exact required expectation, including its absent/optional distinction.

        Returns:
            str: An immutable objective/criterion identity, not a declared achieved grade.
        """
        return config_hash({"expectation": expectation.model_dump(mode="json") if expectation is not None else None})

    def _retain_archive(self, *, receipt: EvaluationFeedbackArchive, content: bytes) -> None:
        extension = ".eval" if receipt.media_type == "application/octet-stream" else ".ndjson"
        destination = self.journal.root / f"{receipt.archive_sha256}{extension}"
        if destination.is_symlink():
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        if destination.exists():
            if destination.read_bytes() != content:
                raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
            return
        with destination.open("xb") as archive:
            archive.write(content)
            archive.flush()
            os.fsync(archive.fileno())

    def _inspect_archive_events(self, *, content: bytes, task_name: str) -> tuple[EvaluationFeedbackEvent, ...]:
        import tempfile

        from inspect_ai.log import read_eval_log

        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        InspectOriginalEvalImporter._validate_archive_bytes(content=content)
        with tempfile.TemporaryDirectory(dir=self.journal.root, prefix="lineage-") as directory:
            path = Path(directory) / "source.eval"
            path.write_bytes(content)
            log = read_eval_log(str(path))
        if (
            log.status != "success"
            or log.eval.task != task_name
            or not log.samples
            or len(log.samples) != 1
            or log.samples[0].error is not None
            or log.samples[0].store is None
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        lineage = log.samples[0].store.get(self.INSPECT_LINEAGE_KEY)
        if not isinstance(lineage, dict) or (
            lineage.get("session_sha256") != self.session.session_sha256
            or lineage.get("request_sha256") != self.session.request.request_sha256
            or lineage.get("run_id") != str(self.session.request.run_id)
        ):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.CONFLICT)
        events = lineage.get("events")
        if not isinstance(events, list):
            raise EvaluationFeedbackError(EvaluationFeedbackErrorCode.GAP)
        return tuple(EvaluationFeedbackEvent.model_validate(event) for event in events)
