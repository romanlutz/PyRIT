# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real framework ordering with explicit CPU-only source/model fixtures."""

from __future__ import annotations

import asyncio
import io
import threading
import zipfile
from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from examples.inspect_live_feedback import HarmlessLiveFeedbackCase
from pyrit.converter import FlipConverter
from pyrit.executor.jobs.live_feedback import EvaluationLiveFeedback
from pyrit.memory import CentralMemory, MemoryInterface, SQLiteMemory
from pyrit.memory.evaluation_working_memory import (
    EvaluationFeedbackError,
    EvaluationFeedbackErrorCode,
    EvaluationFeedbackWitness,
    EvaluationWorkingMemoryJournal,
)
from pyrit.models import AttackOutcome, Message, MessagePiece, Observation, Score, ScoringExpectation
from pyrit.models.evaluation_feedback import (
    EvaluationFeedbackControlRequest,
    EvaluationFeedbackSource,
    EvaluationFeedbackTurn,
    EvaluationReadySnapshot,
)
from pyrit.models.evaluation_job import EvaluationControlKind, EvaluationControlRequest
from pyrit.models.score.scorable import MessageScorable
from pyrit.models.score.score import ScoreStatus
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target.inspect_ghcp_target import (
    InspectGhcpGuardedTransport,
    InspectGhcpTarget,
    InspectGhcpTransport,
)
from pyrit.score import IncludesScorer

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from pathlib import Path

    from pydantic import JsonValue


def _expansion_archive() -> bytes:
    content = io.BytesIO()
    with zipfile.ZipFile(content, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("eval.json", b"x" * (64 * 1024 * 1024 + 1))
    return content.getvalue()


@asynccontextmanager
async def _started_case_async(
    *,
    root: Path,
    memory: MemoryInterface,
    operator_steps: bool = False,
    commit_witness: EvaluationFeedbackWitness | None = None,
) -> AsyncIterator[HarmlessLiveFeedbackCase]:
    case = HarmlessLiveFeedbackCase(
        root=root, memory=memory, operator_steps=operator_steps, commit_witness=commit_witness
    )
    await case.feedback.startup_async()
    try:
        yield case
    finally:
        if case.native_session.evidence().idle:
            await case.native_session.quiesce_async()
        await case.feedback.close_async()


async def _first_turn_async(case: HarmlessLiveFeedbackCase) -> EvaluationReadySnapshot:
    case.attack._max_turns = 1
    await case.run_attack_async()
    return await case.feedback.ready_snapshot_async()


async def _control_async(case: HarmlessLiveFeedbackCase) -> EvaluationFeedbackControlRequest:
    ready = await case.feedback.ready_snapshot_async()
    boundary = await case.feedback.control_boundary_async()
    return EvaluationFeedbackControlRequest(
        session_sha256=case.descriptor.session_sha256,
        snapshot_sha256=ready.snapshot_sha256,
        command=EvaluationControlRequest(
            command_id=uuid4(),
            boundary_id=boundary.boundary_id,
            kind=EvaluationControlKind.SEND_MESSAGE,
            message="Finish the harmless counter.",
        ),
    )


@pytest.mark.usefixtures("patch_central_database")
class TestLiveFeedback:
    async def test_actual_attack_normalizer_memory_score_rationale_barrier_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        case = HarmlessLiveFeedbackCase(root=tmp_path, memory=sqlite_instance)
        await case.feedback.startup_async()
        try:
            result = await case.run_attack_async()
            ready = await case.feedback.ready_snapshot_async()
            assert result.outcome is AttackOutcome.SUCCESS
            assert result.executed_turns == case.sdk.counter == case.adversarial.calls == 2
            assert ready.generation == 2
            assert ready.source_through == 14
            first = await asyncio.to_thread(case.feedback.journal.read_turn, 1)
            turn = EvaluationFeedbackTurn.model_validate_json(first["source_json"])
            first_ready = EvaluationReadySnapshot.model_validate_json(first["snapshot_json"])
            assert turn.raw_complete and turn.normalized_complete and not turn.gaps
            assert len(turn.raw_only_event_ids) == 3
            assert {piece.tool_call_id for piece in turn.pieces if piece.tool_call_id} == {"fixture-counter-1"}
            assert {piece.source_message_id for piece in turn.pieces if piece.source_message_id} == {
                "source-input-1",
                "source-response-1",
            }
            assert case.adversarial.boundaries[0]["snapshot_sha256"] == first_ready.snapshot_sha256
            assert case.sdk.boundaries[0]["snapshot_sha256"] == first_ready.snapshot_sha256
            pieces = await sqlite_instance.get_message_pieces_async(conversation_id=result.conversation_id)
            assert len(pieces) == 8
            assert sorted(piece.sequence for piece in pieces) == list(range(8))
            scores = await sqlite_instance.get_scores_async(score_type="true_false")
            assert len(scores) == 2
            assert {score.score_value for score in scores} == {"false", "true"}
            assert all(score.status is ScoreStatus.COMPLETE and score.score_rationale for score in scores)
            assert all(score.message_piece_id in {piece.id for piece in pieces} for score in scores)
        finally:
            if case.native_session.evidence().idle:
                await case.native_session.quiesce_async()
            await case.feedback.close_async()

    async def test_real_controlled_inspect_task_has_distinct_original_grade_and_no_reimported_working_rows_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        case = HarmlessLiveFeedbackCase(root=tmp_path, memory=sqlite_instance)
        result = await case.run_async()
        sample = result.source_log.samples[0]
        assert sample.store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
        assert sample.scores[case.SCORE_NAME].value == 1.0
        assert result.source_log.stats.model_usage == {}
        assert case.sdk.disconnected
        assert (
            len(await sqlite_instance.get_message_pieces_async(conversation_id=result.attack_result.conversation_id))
            == 8
        )
        assert len(await sqlite_instance.get_scores_async(score_type="true_false")) == 2
        assert result.archive.source_through == 14
        assert result.archive.archive_sha256 != result.archive.last_snapshot_sha256

    @pytest.mark.parametrize("writer", ["message", "score"])
    async def test_real_writers_are_awaited_and_premature_control_does_not_publish_or_apply_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, writer: str
    ) -> None:
        entered, release = asyncio.Event(), asyncio.Event()
        original_message = sqlite_instance.add_message_to_memory_async
        original_score = sqlite_instance.add_scores_to_memory_async

        async def delay_message_async(*, request: Message) -> None:
            if request.api_role == "assistant" and request.get_values() == ["progress"]:
                entered.set()
                await release.wait()
            await original_message(request=request)

        async def delay_score_async(
            *,
            scores: Sequence[Score],
            observations: Sequence[Observation] = (),
            intermediate_scores: Sequence[Score] = (),
        ) -> None:
            if not release.is_set():
                entered.set()
                await release.wait()
            await original_score(scores=scores, observations=observations, intermediate_scores=intermediate_scores)

        async with _started_case_async(root=tmp_path, memory=sqlite_instance, operator_steps=True) as case:
            method = "add_message_to_memory_async" if writer == "message" else "add_scores_to_memory_async"
            delayed = delay_message_async if writer == "message" else delay_score_async
            with patch.object(sqlite_instance, method, new=AsyncMock(side_effect=delayed)):
                execution = asyncio.create_task(case.run_attack_async())
                try:
                    await asyncio.wait_for(entered.wait(), timeout=10)
                    row = await asyncio.to_thread(case.feedback.journal.read_session)
                    assert row["stage"] == ("captured" if writer == "message" else "observed")
                    assert row["snapshot_json"] is None
                    assert case.adversarial.calls == case.sdk.counter == 1
                    assert await sqlite_instance.get_scores_async(score_type="true_false") == []
                    premature = EvaluationFeedbackControlRequest(
                        session_sha256=case.descriptor.session_sha256,
                        snapshot_sha256="0" * 64,
                        command=EvaluationControlRequest(
                            command_id=uuid4(),
                            boundary_id=uuid4(),
                            kind=EvaluationControlKind.SEND_MESSAGE,
                            message="Must not be delivered.",
                        ),
                    )
                    with pytest.raises(EvaluationFeedbackError, match="feedback_stale_snapshot"):
                        await case.feedback.reserve_control_async(control=premature)
                    with pytest.raises(EvaluationFeedbackError, match="feedback_stale_snapshot"):
                        await case.feedback.control_boundary_async()
                    assert case.sdk.counter == 1
                    assert not case.adversarial.boundaries and not case.sdk.boundaries
                finally:
                    release.set()
                    result = await execution
                assert result.outcome is AttackOutcome.SUCCESS
                assert case.sdk.counter == case.adversarial.calls == 2

    @pytest.mark.parametrize(
        "writer,raises", [("message", False), ("score", False), ("message", True), ("score", True)]
    )
    async def test_missing_or_failed_actual_write_blocks_next_attack_and_delivery_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, writer: str, raises: bool
    ) -> None:
        original_message = sqlite_instance.add_message_to_memory_async

        async def missing_message_async(*, request: Message) -> None:
            if request.api_role == "assistant" and request.get_values() == ["progress"]:
                if raises:
                    raise OSError("fixture_storage_unavailable")
                return
            await original_message(request=request)

        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            method = "add_message_to_memory_async" if writer == "message" else "add_scores_to_memory_async"
            injected = (
                AsyncMock(side_effect=missing_message_async)
                if writer == "message"
                else AsyncMock(side_effect=OSError("fixture_storage_unavailable"))
                if raises
                else AsyncMock()
            )
            with (
                patch.object(sqlite_instance, method, new=injected),
                pytest.raises(RuntimeError, match="feedback_.*not_committed|fixture_storage_unavailable"),
            ):
                await case.run_attack_async()
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "blocked" and row["snapshot_json"] is None
            assert case.sdk.counter == case.adversarial.calls == 1
            assert not case.adversarial.boundaries and not case.sdk.boundaries

    @pytest.mark.parametrize("fault", ["undetermined", "rationale", "scorer", "expectation", "response_anchor"])
    async def test_persisted_required_feedback_must_be_complete_bound_and_explained_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, fault: str
    ) -> None:
        original = sqlite_instance.add_scores_to_memory_async

        async def corrupt_score_async(
            *,
            scores: Sequence[Score],
            observations: Sequence[Observation] = (),
            intermediate_scores: Sequence[Score] = (),
        ) -> None:
            updates: dict[str, object] = {}
            if fault == "undetermined":
                updates = {"status": ScoreStatus.UNDETERMINED, "score_value": None}
            elif fault == "rationale":
                updates = {"score_rationale": " "}
            elif fault == "scorer":
                updates = {"scorer_class_identifier": IncludesScorer(expected="different").get_identifier()}
            elif fault == "expectation":
                updates = {
                    "scored_expectation": ScoringExpectation(objective="Different criterion."),
                    "objective": "Different criterion.",
                }
            else:
                assert scores[0].message_piece_id is not None
                rows = await sqlite_instance.get_message_pieces_async(prompt_ids=[str(scores[0].message_piece_id)])
                all_rows = await sqlite_instance.get_message_pieces_async(conversation_id=rows[0].conversation_id)
                anchor = next(row.id for row in all_rows if row.role == "tool")
                updates = {"message_piece_id": anchor, "scorable": MessageScorable(message_piece_ids=(anchor,))}
            changed = [score.model_copy(update=updates) for score in scores]
            await original(scores=changed, observations=observations, intermediate_scores=intermediate_scores)

        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            with patch.object(
                sqlite_instance, "add_scores_to_memory_async", new=AsyncMock(side_effect=corrupt_score_async)
            ):
                with pytest.raises(RuntimeError, match="feedback_score_not_committed"):
                    await case.run_attack_async()
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "blocked" and row["snapshot_json"] is None
            assert case.sdk.counter == case.adversarial.calls == 1
            assert len(await sqlite_instance.get_scores_async(score_type="true_false")) == 1

    async def test_independent_witness_refusal_precedes_ready_publication_and_next_input_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        witness = MagicMock(spec=EvaluationFeedbackWitness)
        witness.verify_writes_async = AsyncMock(side_effect=EvaluationFeedbackError(EvaluationFeedbackErrorCode.MEMORY))
        async with _started_case_async(root=tmp_path, memory=sqlite_instance, commit_witness=witness) as case:
            with pytest.raises(RuntimeError, match="feedback_memory_not_committed"):
                await case.run_attack_async()
            witness.verify_writes_async.assert_called_once()
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "blocked" and row["snapshot_json"] is None
            assert len(await sqlite_instance.get_scores_async(score_type="true_false")) == 1
            assert case.sdk.counter == case.adversarial.calls == 1

    async def test_actual_second_delivery_reads_exact_reserved_operator_input_and_scores_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance, operator_steps=True) as case:
            ready = await _first_turn_async(case)
            control = await _control_async(case)
            receipt = await case.feedback.reserve_control_async(control=control)
            duplicate = await case.feedback.reserve_control_async(control=control)
            assert receipt.reserved_turn == 2 and not receipt.duplicate and duplicate.duplicate
            assert receipt.control_sha256 == duplicate.control_sha256 == control.control_sha256
            assert case.sdk.counter == 1
            await case.feedback.before_turn_async(
                conversation_id=str(ready.conversation_id), turn_index=2, expectation=case.expectation
            )
            assert control.command.message is not None
            message = Message(
                message_pieces=[
                    MessagePiece(
                        role="user", original_value=control.command.message, conversation_id=str(ready.conversation_id)
                    )
                ]
            )
            response = await case.normalizer.send_prompt_async(
                message=message, target=case.target, conversation_id=str(ready.conversation_id)
            )
            await case.feedback.response_committed_async(response=response)
            scores = await case.scorer.score_async(
                scorable=MessageScorable(message_piece_ids=(response.get_piece().id,)), expectation=case.expectation
            )
            await case.feedback.feedback_committed_async(response=response, scores=scores, expectation=case.expectation)
            current = await case.feedback.ready_snapshot_async()
            assert current.generation == 2 and current.source_through == 14
            assert case.sdk.counter == 2 and case.adversarial.calls == 1
            assert case.sdk.prompts[-1] == control.command.message
            assert case.sdk.boundaries[-1]["snapshot_sha256"] == ready.snapshot_sha256
            with pytest.raises(EvaluationFeedbackError, match="feedback_unsupported"):
                await case.feedback.control_boundary_async()

    @pytest.mark.parametrize("fault", ["snapshot", "boundary", "session", "action", "changed_replay", "changed_input"])
    async def test_stale_foreign_uninstalled_and_changed_control_refused_without_source_application_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, fault: str
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance, operator_steps=True) as case:
            ready = await _first_turn_async(case)
            control = await _control_async(case)
            expected = "feedback_stale_snapshot"
            if fault == "snapshot":
                control = control.model_copy(update={"snapshot_sha256": "0" * 64})
            elif fault == "boundary":
                control = control.model_copy(
                    update={"command": control.command.model_copy(update={"boundary_id": uuid4()})}
                )
            elif fault == "session":
                control = control.model_copy(update={"session_sha256": "0" * 64})
                expected = "feedback_unsupported"
            elif fault == "action":
                control = control.model_copy(
                    update={"command": control.command.model_copy(update={"kind": EvaluationControlKind.NUDGE})}
                )
                expected = "feedback_unsupported"
            elif fault == "changed_replay":
                await case.feedback.reserve_control_async(control=control)
                control = control.model_copy(
                    update={"command": control.command.model_copy(update={"message": "Changed."})}
                )
                expected = "feedback_conflict"
            else:
                await case.feedback.reserve_control_async(control=control)
                changed = Message(
                    message_pieces=[
                        MessagePiece(
                            role="user",
                            original_value="Not the reserved input.",
                            conversation_id=str(ready.conversation_id),
                        )
                    ]
                )
                with pytest.raises(EvaluationFeedbackError, match="feedback_conflict"):
                    await case.feedback.before_send_async(request=changed)
                assert case.sdk.counter == 1
                return
            with pytest.raises(EvaluationFeedbackError, match=expected):
                await case.feedback.reserve_control_async(control=control)
            assert case.sdk.counter == 1 and not case.sdk.boundaries
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["blocked_reason"] is None

    async def test_operator_control_is_default_off_async(self, tmp_path: Path, sqlite_instance: SQLiteMemory) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            await _first_turn_async(case)
            with pytest.raises(EvaluationFeedbackError, match="feedback_unsupported"):
                await case.feedback.control_boundary_async()
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "ready" and row["blocked_reason"] is None

    @pytest.mark.parametrize("fault", ["extra_message", "owner", "source_after_idle", "source_header"])
    async def test_previously_ready_snapshot_rechecks_real_rows_owner_and_latest_source_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, fault: str
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            ready = await _first_turn_async(case)
            if fault == "extra_message":
                await sqlite_instance.add_message_to_memory_async(
                    request=Message(
                        message_pieces=[
                            MessagePiece(
                                role="assistant",
                                original_value="Unmapped extra row.",
                                conversation_id=str(ready.conversation_id),
                            )
                        ]
                    )
                )
            elif fault == "source_after_idle":
                case.sdk.emit(
                    kind="assistant.message", data={"content": "Uncommitted late source.", "messageId": "late"}
                )
            elif fault == "source_header":
                case.native_session._provenance["source"] = "changed"
            if fault == "owner":
                with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock(spec=MemoryInterface)):
                    with pytest.raises(EvaluationFeedbackError, match="feedback_reconciliation_required"):
                        await case.feedback.before_turn_async(
                            conversation_id=str(ready.conversation_id), turn_index=2, expectation=case.expectation
                        )
            else:
                with pytest.raises(EvaluationFeedbackError):
                    await case.feedback.before_turn_async(
                        conversation_id=str(ready.conversation_id), turn_index=2, expectation=case.expectation
                    )
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "blocked"
            assert case.sdk.counter == case.adversarial.calls == 1

    async def test_exact_source_map_replay_noops_and_changed_map_conflicts_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            await _first_turn_async(case)
            before = await asyncio.to_thread(case.feedback.journal.read_session)
            stored = await asyncio.to_thread(case.feedback.journal.read_turn, 1)
            turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
            await asyncio.to_thread(case.feedback.journal.capture, turn)
            after = await asyncio.to_thread(case.feedback.journal.read_session)
            assert dict(before) == dict(after)
            changed = turn.model_copy(
                update={"pieces": (turn.pieces[0].model_copy(update={"piece_id": uuid4()}), *turn.pieces[1:])}
            )
            with pytest.raises(EvaluationFeedbackError, match="feedback_conflict"):
                await asyncio.to_thread(case.feedback.journal.capture, changed)
            assert len(await sqlite_instance.get_message_pieces_async(conversation_id=turn.conversation_id)) == 4

    async def test_same_root_owner_and_unsealed_restart_cannot_resume_execution_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            await _first_turn_async(case)
            second = EvaluationWorkingMemoryJournal(root=case.feedback.journal.root, session=case.descriptor)
            with pytest.raises(EvaluationFeedbackError, match="feedback_conflict"):
                await asyncio.to_thread(second.startup)
            await case.feedback.close_async()
            await asyncio.to_thread(second.startup)
            try:
                row = await asyncio.to_thread(second.read_session)
                assert row["stage"] == "blocked"
                assert row["blocked_reason"] == "feedback_reconciliation_required"
            finally:
                await asyncio.to_thread(second.close)

    async def test_final_native_archive_is_exact_idempotent_and_does_not_reproject_rows_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            result = await case.run_attack_async()
            content = case.native_session.evidence().model_dump_json().encode()
            first = await case.feedback.reconcile_native_archive_async(content=content)
            second = await case.feedback.reconcile_native_archive_async(content=content)
            assert first == second
            retained = await asyncio.to_thread(
                (case.feedback.journal.root / f"{first.archive_sha256}.ndjson").read_bytes
            )
            assert retained == content
            assert len(await sqlite_instance.get_message_pieces_async(conversation_id=result.conversation_id)) == 8
            assert len(await sqlite_instance.get_scores_async(score_type="true_false")) == 2
            different = case.native_session.evidence().model_copy(update={"environment_id": "foreign"})
            with pytest.raises(EvaluationFeedbackError, match="feedback_conflict"):
                await case.feedback.reconcile_native_archive_async(content=different.model_dump_json().encode())

    @pytest.mark.filterwarnings(r"ignore:MemoryInterface\.:DeprecationWarning")
    async def test_final_inspect_original_grade_uses_separate_canonical_owner_and_repeat_is_idempotent_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        case = HarmlessLiveFeedbackCase(root=tmp_path / "runtime", memory=sqlite_instance)
        result = await case.run_async()
        with pytest.raises(ValueError, match="separate canonical owner"):
            await case.import_final_to_canonical_async(canonical_memory=sqlite_instance, result=result)
        canonical = SQLiteMemory.__new__(SQLiteMemory)
        canonical.__init__(db_path=tmp_path / "canonical.sqlite", silent=True, _defer_initialization=True)
        canonical.results_path = str(tmp_path / "canonical-artifacts")
        await canonical.initialize_async()
        try:
            imported = await case.import_final_to_canonical_async(canonical_memory=canonical, result=result)
            repeated = await case.import_final_to_canonical_async(canonical_memory=canonical, result=result)
            assert imported.archive_sha256 == repeated.archive_sha256 == result.archive.archive_sha256
            assert len(imported.case_results) == len(repeated.case_results) == 1
            original = imported.case_results[0]
            repeat = repeated.case_results[0]
            assert original.score.score_value == "1.0" and original.score.status is ScoreStatus.COMPLETE
            assert original.attack_result.outcome is AttackOutcome.UNDETERMINED
            assert original.score.id == repeat.score.id
            assert original.attack_result.attack_result_id == repeat.attack_result.attack_result_id
            scores = await canonical.get_scores_async(score_ids=[str(original.score.id)])
            attacks = await canonical.get_attack_results_async(
                attack_result_ids=[original.attack_result.attack_result_id]
            )
            assert scores[0].score_metadata["inspect_archive_sha256"] == result.archive.archive_sha256
            assert scores[0].score_metadata["inspect_case_run_id"] == original.case_run_id
            assert scores[0].score_metadata["inspect_primary_scorer"] == case.SCORE_NAME
            assert attacks[0].automated_score.id == scores[0].id
            assert len(await canonical.get_scores_async(score_type="float_scale")) == 1
            working = await sqlite_instance.get_message_pieces_async(
                conversation_id=result.attack_result.conversation_id
            )
            source = await canonical.get_message_pieces_async(conversation_id=imported.episode.conversation_id)
            assert {piece.id for piece in working}.isdisjoint(piece.id for piece in source)
            assert len(working) == 8 and len(await sqlite_instance.get_scores_async(score_type="true_false")) == 2
        finally:
            await canonical.dispose_engine_async()

    async def test_cancelled_journal_io_joins_underlying_operation_before_owner_close_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            original = case.feedback.journal.begin_turn

            def held_begin(*, conversation_id: UUID, turn_index: int, expected: EvaluationReadySnapshot | None) -> None:
                entered.set()
                if not release.wait(timeout=10):
                    raise TimeoutError("fixture_release_timeout")
                original(conversation_id=conversation_id, turn_index=turn_index, expected=expected)
                done.set()

            with patch.object(case.feedback.journal, "begin_turn", side_effect=held_begin):
                pending = asyncio.create_task(
                    case.feedback.before_turn_async(
                        conversation_id=str(uuid4()), turn_index=1, expectation=case.expectation
                    )
                )
                try:
                    assert await asyncio.to_thread(entered.wait, 10)
                    pending.cancel()
                    await asyncio.sleep(0)
                    pending.cancel()
                    closing = asyncio.create_task(case.feedback.close_async())
                    await asyncio.sleep(0)
                    assert not pending.done() and not closing.done() and not done.is_set()
                finally:
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await pending
                    await closing
                assert done.is_set()

    @pytest.mark.parametrize("fault", ["raw_incomplete", "normalized_incomplete", "declared_gap", "shifted_sequence"])
    async def test_source_gap_and_incomplete_projection_cannot_commit_in_a_fresh_journal_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, fault: str
    ) -> None:
        async with _started_case_async(root=tmp_path / "actual", memory=sqlite_instance) as case:
            await _first_turn_async(case)
            stored = await asyncio.to_thread(case.feedback.journal.read_turn, 1)
            turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
            if fault == "shifted_sequence":
                turn = turn.model_copy(
                    update={
                        "events": tuple(
                            event.model_copy(update={"source_sequence": event.source_sequence + 1})
                            for event in turn.events
                        )
                    }
                )
            else:
                field = {
                    "raw_incomplete": "raw_complete",
                    "normalized_incomplete": "normalized_complete",
                    "declared_gap": "gaps",
                }[fault]
                turn = turn.model_copy(update={field: ("dropped_source",) if fault == "declared_gap" else False})
            fresh = EvaluationWorkingMemoryJournal(root=tmp_path / "validator", session=case.descriptor)
            await asyncio.to_thread(fresh.startup)
            try:
                await asyncio.to_thread(
                    fresh.begin_turn, conversation_id=turn.conversation_id, turn_index=1, expected=None
                )
                await asyncio.to_thread(
                    fresh.send_intent,
                    conversation_id=turn.conversation_id,
                    input_sha256=stored["input_sha256"],
                    original_input_sha256=stored["input_sha256"],
                )
                with pytest.raises(EvaluationFeedbackError, match="feedback_source_gap"):
                    await asyncio.to_thread(fresh.capture, turn)
                row = await asyncio.to_thread(fresh.read_session)
                assert row["source_cursor"] == 0 and row["snapshot_json"] is None
            finally:
                await asyncio.to_thread(fresh.close)
            assert case.sdk.counter == 1

    async def test_unpartitioned_raw_event_and_forged_event_identity_are_not_valid_source_models_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            await _first_turn_async(case)
            stored = await asyncio.to_thread(case.feedback.journal.read_turn, 1)
            turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
            with pytest.raises(ValidationError, match="partition"):
                EvaluationFeedbackTurn.model_validate(turn.model_copy(update={"raw_only_event_ids": ()}))
            event = turn.events[0]
            with pytest.raises(ValidationError, match="stable event identity"):
                type(event).model_validate(event.model_copy(update={"source_event_id": "forged-id"}))

    async def test_exact_sdk_event_replay_noops_but_changed_same_id_revokes_readiness_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        from examples.inspect_live_feedback import _SdkEventFixture

        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            ready = await _first_turn_async(case)
            event = case.native_session.evidence().events[0]
            case.native_session._receive(_SdkEventFixture(value=event.payload))
            assert (await case.feedback.ready_snapshot_async()).snapshot_sha256 == ready.snapshot_sha256
            changed = {**event.payload, "data": {"content": "Changed replay."}}
            case.native_session._receive(_SdkEventFixture(value=changed))
            with pytest.raises(EvaluationFeedbackError, match="feedback_source_gap"):
                await case.feedback.before_turn_async(
                    conversation_id=str(ready.conversation_id), turn_index=2, expectation=case.expectation
                )
            assert case.sdk.counter == case.adversarial.calls == 1

    async def test_source_event_after_admission_is_rejected_by_native_dispatch_before_second_delivery_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            original = case.feedback.journal.send_intent

            def late_event(*, conversation_id: UUID, input_sha256: str, original_input_sha256: str) -> int:
                turn = original(
                    conversation_id=conversation_id,
                    input_sha256=input_sha256,
                    original_input_sha256=original_input_sha256,
                )
                if turn == 2:
                    case.sdk.emit(kind="assistant.usage", data={"inputTokens": 0, "outputTokens": 0}, ephemeral=True)
                return turn

            with patch.object(case.feedback.journal, "send_intent", side_effect=late_event):
                with pytest.raises(RuntimeError, match="source changed after working-memory input admission"):
                    await case.run_attack_async()
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "blocked"
            assert case.sdk.counter == 1 and case.adversarial.calls == 2
            assert not case.sdk.boundaries

    async def test_source_change_during_independent_score_witness_blocks_readiness_publication_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        witness = MagicMock(spec=EvaluationFeedbackWitness)

        async def add_late_source_async(**kwargs: object) -> None:
            case.sdk.emit(kind="assistant.usage", data={"inputTokens": 0, "outputTokens": 0}, ephemeral=True)

        witness.verify_writes_async = AsyncMock(side_effect=add_late_source_async)
        async with _started_case_async(root=tmp_path, memory=sqlite_instance, commit_witness=witness) as case:
            with pytest.raises(RuntimeError, match="feedback_conflict"):
                await case.run_attack_async()
            row = await asyncio.to_thread(case.feedback.journal.read_session)
            assert row["stage"] == "blocked" and row["snapshot_json"] is None
            assert case.sdk.counter == case.adversarial.calls == 1

    async def test_actual_converter_delivery_retains_original_prepared_and_source_mapping_separately_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            case.attack._request_converters = ConverterConfiguration.from_converters(converters=[FlipConverter()])
            result = await case.run_attack_async()
            pieces = await sqlite_instance.get_message_pieces_async(conversation_id=result.conversation_id)
            users = sorted((piece for piece in pieces if piece.role == "user"), key=lambda piece: piece.sequence)
            assert [piece.converted_value for piece in users] == case.sdk.prompts
            assert all(piece.converted_value == piece.original_value[::-1] for piece in users)
            assert all(
                piece.prompt_metadata["evaluation_source_session_id"] == case.descriptor.source_session_id
                for piece in users
            )
            assert result.outcome is AttackOutcome.SUCCESS

    async def test_inspect_zip_expansion_is_bounded_before_typed_parse_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        content = await asyncio.to_thread(_expansion_archive)
        assert len(content) < 16 * 1024 * 1024
        async with _started_case_async(root=tmp_path, memory=sqlite_instance) as case:
            await _first_turn_async(case)
            with patch("inspect_ai.log.read_eval_log") as parser:
                with pytest.raises(ValueError, match="expansion|uncompressed|large|limit"):
                    await case.feedback.reconcile_inspect_archive_async(content=content, task_name=case.TASK_NAME)
                parser.assert_not_called()

    async def test_canonical_helper_rejects_substituted_receipt_before_any_canonical_import_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        case = HarmlessLiveFeedbackCase(root=tmp_path / "runtime", memory=sqlite_instance)
        result = await case.run_async()
        forged = replace(result, archive=result.archive.model_copy(update={"archive_sha256": "0" * 64}))
        canonical = SQLiteMemory.__new__(SQLiteMemory)
        canonical.__init__(db_path=tmp_path / "canonical.sqlite", silent=True, _defer_initialization=True)
        canonical.results_path = str(tmp_path / "canonical-artifacts")
        await canonical.initialize_async()
        try:
            with pytest.raises(ValueError, match="immutable sealed receipt"):
                await case.import_final_to_canonical_async(canonical_memory=canonical, result=forged)
            assert await canonical.get_attack_results_async() == []
            assert await canonical.get_scores_async(score_type="float_scale") == []
        finally:
            await canonical.dispose_engine_async()

    @pytest.mark.parametrize("fault", ["none", "missing_user", "tool_frame", "missing_id"])
    async def test_reviewed_inspect_frame_requires_live_probe_guard_and_real_source_ids_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory, fault: str
    ) -> None:
        case = HarmlessLiveFeedbackCase(root=tmp_path, memory=sqlite_instance)
        descriptor = case.descriptor.model_copy(update={"source": EvaluationFeedbackSource.INSPECT_GHCP})
        bridge = EvaluationLiveFeedback(root=tmp_path / "inspect-capture", session=descriptor, memory=sqlite_instance)
        bridge.bind_inspect(source_probe=case.native_session.evidence, max_turns=2)
        transport = MagicMock(spec=InspectGhcpGuardedTransport)

        async def text_source_async(prompt: str, *, timeout: float) -> None:
            if fault != "missing_user":
                case.sdk.emit(kind="user.message", data={"content": prompt, "messageId": "inspect-user"})
            if fault == "tool_frame":
                call = "inspect-tool"
                case.sdk.emit(
                    kind="assistant.message",
                    data={
                        "content": "",
                        "toolRequests": [{"toolCallId": call, "name": "fixture_tool", "arguments": {}}],
                    },
                )
                case.sdk.emit(
                    kind="tool.execution_start", data={"toolCallId": call, "toolName": "fixture_tool", "arguments": {}}
                )
                case.sdk.emit(
                    kind="tool.execution_complete",
                    data={"toolCallId": call, "success": True, "result": {"content": "fixture"}},
                )
            case.sdk.emit(kind="assistant.message", data={"content": "complete", "messageId": "inspect-answer"})
            case.sdk.emit(kind="session.idle", data={})

        async def guarded_frame_async(
            *, instruction: str, turn_index: int, expected_source_cursor: int
        ) -> dict[str, JsonValue]:
            events = await case.native_session.send_async(
                prompt=instruction, timeout_seconds=10, expected_source_cursor=expected_source_cursor
            )
            values: list[JsonValue] = [
                {key: value for key, value in event.payload.items() if key != "id"}
                if fault == "missing_id"
                else event.payload
                for event in events
            ]
            return {
                "kind": "turn",
                "run_id": str(case.request.run_id),
                "turn_index": turn_index,
                "session_id": descriptor.source_session_id,
                "identity": {"scope": "explicit_text_frame_fixture"},
                "assistant_text": "complete",
                "events": values,
                "model_exchanges": [],
            }

        transport.send_turn_guarded_async = AsyncMock(side_effect=guarded_frame_async)
        target = InspectGhcpTarget(
            transport=transport,
            run_id=str(case.request.run_id),
            model_name="scripted-fixture",
            evaluation_feedback=bridge,
        )
        normalizer = PromptNormalizer()
        bridge.validate_setup(
            memory=sqlite_instance,
            normalizer_memory=normalizer.memory,
            objective_target=target,
            objective_scorer=case.scorer,
        )
        await bridge.startup_async()
        try:
            conversation_id = str(uuid4())
            await bridge.before_turn_async(conversation_id=conversation_id, turn_index=1, expectation=case.expectation)
            message = Message.from_prompt(prompt="Harmless text-only source.", role="user")
            with patch.object(case.sdk, "send_and_wait", new=AsyncMock(side_effect=text_source_async)):
                if fault != "none":
                    with pytest.raises(Exception, match="Error sending prompt"):
                        await normalizer.send_prompt_async(
                            message=message, target=target, conversation_id=conversation_id
                        )
                    row = await asyncio.to_thread(bridge.journal.read_session)
                    assert row["snapshot_json"] is None
                else:
                    response = await normalizer.send_prompt_async(
                        message=message, target=target, conversation_id=conversation_id
                    )
                    await bridge.response_committed_async(response=response)
                    scores = await case.scorer.score_async(
                        scorable=MessageScorable(message_piece_ids=(response.get_piece().id,)),
                        expectation=case.expectation,
                    )
                    await bridge.feedback_committed_async(
                        response=response, scores=scores, expectation=case.expectation
                    )
                    ready = await bridge.ready_snapshot_async()
                    assert ready.source_through == 3
                    stored = await asyncio.to_thread(bridge.journal.read_turn, 1)
                    turn = EvaluationFeedbackTurn.model_validate_json(stored["source_json"])
                    assert turn.pieces[0].source_event_id == turn.events[0].source_event_id
                    assert target.turns[0].response_piece_id == response.get_piece().id
                    assert len(turn.raw_only_event_ids) == 1
        finally:
            if case.native_session.evidence().idle:
                await case.native_session.quiesce_async()
            await bridge.close_async()

    async def test_ordinary_inspect_transport_cannot_claim_installed_live_readiness_async(
        self, tmp_path: Path, sqlite_instance: SQLiteMemory
    ) -> None:
        case = HarmlessLiveFeedbackCase(root=tmp_path, memory=sqlite_instance)
        transport = MagicMock(spec=InspectGhcpTransport)
        transport.send_turn_async = AsyncMock()
        with pytest.raises(ValueError, match="source-watermark dispatch guard"):
            InspectGhcpTarget(
                transport=transport,
                run_id=str(case.request.run_id),
                model_name="fixture",
                evaluation_feedback=case.feedback,
            )
        transport.send_turn_async.assert_not_called()
