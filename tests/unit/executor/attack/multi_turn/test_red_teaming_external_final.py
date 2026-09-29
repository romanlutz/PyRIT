# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackParameters,
    AttackScoringConfig,
    MultiTurnAttackContext,
    RedTeamingAttack,
    RedTeamingPendingExternalResult,
    RedTeamingTerminalScoring,
)
from pyrit.memory import SQLiteMemory
from pyrit.memory.memory_models import ScoreEntry
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ComponentIdentifier,
    Message,
    MessagePiece,
    PromptResponseError,
    Score,
)
from pyrit.score import TrueFalseScorer
from tests.unit.mocks import MockPromptTarget


class _AdversarialTarget(MockPromptTarget):
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1]
        self.prompt_sent.append(request.get_value())
        response = json.dumps(
            {
                "next_message": "second attack prompt",
                "rationale": "continue",
                "last_response_summary": "first target response",
            }
        )
        return [
            MessagePiece(
                role="assistant",
                original_value=response,
                converted_value=response,
                conversation_id=request.get_piece().conversation_id,
            ).to_message()
        ]


class _WorkspaceTarget(MockPromptTarget):
    def __init__(self, *, events: list[str]) -> None:
        super().__init__()
        self.workspace_alive = True
        self.events = events
        self.reset_ids: list[str] = []
        self.fail_reset = False
        self.response_error: PromptResponseError = "none"

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1]
        self.prompt_sent.append(request.get_value())
        response = "default" if self.response_error == "none" else "Target error"
        return [
            MessagePiece(
                role="assistant",
                original_value=response,
                converted_value=response,
                response_error=self.response_error,
                conversation_id=request.get_piece().conversation_id,
            ).to_message()
        ]

    async def reset_conversation_async(self, *, conversation_id: str) -> None:
        if not self.workspace_alive:
            raise RuntimeError("Workspace was destroyed before target reset.")
        self.events.append("target_reset")
        self.reset_ids.append(conversation_id)
        if self.fail_reset:
            raise RuntimeError("target reset failed")


class _OuterTask:
    def __init__(
        self,
        *,
        target: _WorkspaceTarget,
        memory: SQLiteMemory,
        events: list[str],
        fail_at: str | None = None,
    ) -> None:
        self.target = target
        self.memory = memory
        self.events = events
        self.fail_at = fail_at
        self.stop_requests = 0
        self.exit_observed = False
        self.grade_calls = 0

    async def request_stop_once_async(self) -> None:
        if self.stop_requests == 0:
            self.stop_requests = 1
            self.events.append("stop_request")

    async def observe_agent_exit_async(self) -> None:
        if self.stop_requests != 1:
            raise RuntimeError("Stop was not requested before observing exit.")
        self.events.append("observe_exit")
        if self.fail_at == "observe":
            raise RuntimeError("agent exit was not observed")
        self.exit_observed = True

    async def score_original_async(self, *, pending: RedTeamingPendingExternalResult) -> bool:
        if not self.exit_observed or not self.target.workspace_alive or pending.last_response is None:
            raise RuntimeError("Original scorer ran before verified stop or after sandbox teardown.")
        assert self.memory._query_entries(ScoreEntry) == []
        assert len(self.memory.get_attack_results()) == 0
        self.grade_calls += 1
        self.events.append("original_scorer")
        if self.fail_at == "score":
            raise RuntimeError("original scorer failed")
        return True

    async def ensure_quiesced_async(self) -> None:
        await self.request_stop_once_async()
        self.events.append("quiesce")
        if self.fail_at == "quiesce":
            raise RuntimeError("agent quiescence failed")

    async def teardown_sandbox_async(self) -> None:
        self.target.workspace_alive = False
        self.events.append("sandbox_teardown")
        if self.fail_at == "teardown":
            raise RuntimeError("sandbox teardown failed")


def _make_context() -> MultiTurnAttackContext:
    return MultiTurnAttackContext(
        params=AttackParameters(
            objective="Test objective",
            next_message=Message.from_prompt(prompt="first attack prompt", role="user"),
        )
    )


def _make_attack(
    *,
    events: list[str],
    scoring_config: AttackScoringConfig | None = None,
    score_last_turn_only: bool = False,
) -> tuple[RedTeamingAttack, _WorkspaceTarget, _AdversarialTarget]:
    target = _WorkspaceTarget(events=events)
    adversarial = _AdversarialTarget()
    attack = RedTeamingAttack(
        objective_target=target,
        attack_adversarial_config=AttackAdversarialConfig(target=adversarial),
        attack_scoring_config=scoring_config,
        max_turns=2,
        score_last_turn_only=score_last_turn_only,
        terminal_scoring=RedTeamingTerminalScoring.EXTERNAL_FINAL,
    )
    return attack, target, adversarial


async def _run_outer_async(
    *,
    attack: RedTeamingAttack,
    context: MultiTurnAttackContext,
    owner: _OuterTask,
) -> tuple[RedTeamingPendingExternalResult, bool]:
    try:
        async with attack.external_final_scoring_session() as session:
            try:
                pending = await session.execute_with_context_async(context=context)
                await owner.request_stop_once_async()
                await owner.observe_agent_exit_async()
                original_verdict = await owner.score_original_async(pending=pending)
            finally:
                await owner.ensure_quiesced_async()
    finally:
        await owner.teardown_sandbox_async()
    return pending, original_verdict


@pytest.mark.usefixtures("patch_central_database")
class TestExternalFinalScoring:
    async def test_two_turns_defer_reset_and_publish_only_original_score(
        self, *, sqlite_instance: SQLiteMemory
    ) -> None:
        events: list[str] = []
        attack, target, adversarial = _make_attack(events=events)
        context = _make_context()
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events)

        pending, verdict = await _run_outer_async(attack=attack, context=context, owner=owner)

        assert verdict is True
        assert isinstance(pending, RedTeamingPendingExternalResult)
        assert pending.outcome is AttackOutcome.UNDETERMINED
        assert pending.automated_score is None
        assert pending.human_score is None
        assert pending.executed_turns == 2
        assert pending.last_response is not None
        assert len(target.prompt_sent) == 2
        assert len(adversarial.prompt_sent) == 1
        assert target.reset_ids == [pending.conversation_id]
        assert owner.stop_requests == owner.grade_calls == 1
        assert events.index("stop_request") < events.index("observe_exit") < events.index("original_scorer")
        assert events.index("original_scorer") < events.index("quiesce") < events.index("target_reset")
        assert events.index("target_reset") < events.index("sandbox_teardown")
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

        original_score = Score(
            score_type="true_false",
            score_value="true",
            score_category=["benchmark_original"],
            scorer_class_identifier=ComponentIdentifier(
                class_name="OriginalTaskScorer",
                class_module="test_red_teaming_external_final",
            ),
            message_piece_id=pending.last_response.id,
        )
        sqlite_instance.add_scores_to_memory(scores=[original_score])
        completed = AttackResult(
            conversation_id=pending.conversation_id,
            objective=pending.objective,
            outcome=AttackOutcome.SUCCESS,
            executed_turns=pending.executed_turns,
            last_response=pending.last_response,
            automated_score=original_score,
        )
        sqlite_instance.add_attack_results_to_memory(attack_results=[completed])

        stored_scores = sqlite_instance.get_scores(score_ids=[str(original_score.id)])
        stored_results = sqlite_instance.get_attack_results()
        assert len(stored_scores) == len(stored_results) == 1
        assert len(sqlite_instance._query_entries(ScoreEntry)) == 1
        assert stored_scores[0].scorer_class_identifier.class_name == "OriginalTaskScorer"
        assert stored_results[0].automated_score.id == original_score.id

    async def test_external_mode_requires_live_owner_session(self, *, sqlite_instance: SQLiteMemory) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)

        with pytest.raises(RuntimeError, match="requires an active external_final_scoring_session"):
            await attack.execute_async(objective="Test objective")
        with pytest.raises(RuntimeError, match="requires an active external_final_scoring_session"):
            await attack.execute_with_context_async(context=_make_context())

        assert target.prompt_sent == []
        assert target.reset_ids == []
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

    async def test_external_setup_discards_unrelated_prepended_score(self) -> None:
        attack, _, _ = _make_attack(events=[])
        context = _make_context()

        async def initialize_context_async(*, context: MultiTurnAttackContext, **kwargs: Any) -> None:
            context.last_score = Score(score_type="true_false", score_value="true")

        with patch.object(
            attack._conversation_manager,
            "initialize_context_async",
            new_callable=AsyncMock,
            side_effect=initialize_context_async,
        ):
            await attack._setup_async(context=context)

        assert context.last_score is None

    @pytest.mark.parametrize("kind", ["objective", "auxiliary", "refusal"])
    def test_external_mode_rejects_internal_scorers(self, *, kind: str) -> None:
        scorer = MagicMock(spec=TrueFalseScorer)
        scorer_config = AttackScoringConfig(
            objective_scorer=scorer if kind == "objective" else None,
            auxiliary_scorers=[scorer] if kind == "auxiliary" else [],
            refusal_scorer=scorer if kind == "refusal" else None,
        )

        with pytest.raises(ValueError, match="cannot use PyRIT objective, auxiliary, or progress scorers"):
            _make_attack(events=[], scoring_config=scorer_config)

    def test_external_mode_rejects_last_turn_only_scoring(self) -> None:
        with pytest.raises(ValueError, match="score_last_turn_only is incompatible"):
            _make_attack(events=[], score_last_turn_only=True)

    def test_pending_evidence_rejects_error_or_premature_verdict(self) -> None:
        blocked_piece = MessagePiece(
            role="assistant",
            original_value="blocked",
            converted_value="blocked",
            response_error="blocked",
        )
        with pytest.raises(ValueError, match="non-error response and no final verdict"):
            RedTeamingPendingExternalResult(
                conversation_id="blocked-conversation",
                objective="Test objective",
                last_response=blocked_piece,
            )

        valid_piece = MessagePiece(role="assistant", original_value="answer", converted_value="answer")
        with pytest.raises(ValueError, match="non-error response and no final verdict"):
            RedTeamingPendingExternalResult(
                conversation_id="scored-conversation",
                objective="Test objective",
                last_response=valid_piece,
                automated_score=Score(score_type="true_false", score_value="true"),
            )

    @pytest.mark.parametrize("failure", ["observe", "score", "quiesce", "teardown"])
    async def test_outer_failure_never_publishes_final_score(
        self, *, sqlite_instance: SQLiteMemory, failure: str
    ) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events, fail_at=failure)

        with pytest.raises(RuntimeError, match="failed|not observed"):
            await _run_outer_async(attack=attack, context=_make_context(), owner=owner)

        assert owner.stop_requests == 1
        assert owner.grade_calls == (0 if failure == "observe" else 1)
        assert target.reset_ids
        assert events.index("quiesce") < events.index("target_reset")
        assert events.index("target_reset") < events.index("sandbox_teardown")
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

    async def test_reset_failure_blocks_publication_and_still_tears_down(
        self, *, sqlite_instance: SQLiteMemory
    ) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)
        target.fail_reset = True
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events)

        with pytest.raises(ExceptionGroup, match="Objective-target conversation reset failed"):
            await _run_outer_async(attack=attack, context=_make_context(), owner=owner)

        assert owner.stop_requests == owner.grade_calls == 1
        assert len(target.reset_ids) == 1
        assert events.index("target_reset") < events.index("sandbox_teardown")
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

    async def test_cancellation_during_original_grading_still_cleans_up(self, *, sqlite_instance: SQLiteMemory) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events)
        grading_started = asyncio.Event()
        grading_release = asyncio.Event()

        async def wait_for_grade_async(*, pending: RedTeamingPendingExternalResult) -> bool:
            assert pending.last_response is not None
            assert owner.exit_observed and target.workspace_alive
            owner.grade_calls += 1
            events.append("original_scorer")
            grading_started.set()
            await grading_release.wait()
            return True

        with patch.object(owner, "score_original_async", new_callable=AsyncMock, side_effect=wait_for_grade_async):
            run_task = asyncio.create_task(_run_outer_async(attack=attack, context=_make_context(), owner=owner))
            await asyncio.wait_for(grading_started.wait(), timeout=10)
            run_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run_task

        assert owner.stop_requests == owner.grade_calls == 1
        assert len(target.reset_ids) == 1
        assert events.index("quiesce") < events.index("target_reset") < events.index("sandbox_teardown")
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

    @pytest.mark.parametrize("failure", [RuntimeError("attack failed"), asyncio.CancelledError()])
    async def test_attack_failure_or_cancellation_cleans_up_without_grading(
        self, *, sqlite_instance: SQLiteMemory, failure: BaseException
    ) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events)
        messages = [Message.from_prompt(prompt="first attack prompt", role="user"), failure]

        with (
            patch.object(attack, "_generate_next_prompt_async", new_callable=AsyncMock, side_effect=messages),
            pytest.raises(type(failure)),
        ):
            await _run_outer_async(attack=attack, context=_make_context(), owner=owner)

        assert len(target.prompt_sent) == 1
        assert owner.grade_calls == 0
        assert owner.stop_requests == 1
        assert events.index("quiesce") < events.index("target_reset") < events.index("sandbox_teardown")
        assert len(target.reset_ids) == 1
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

    @pytest.mark.parametrize("response_error", ["blocked", "unknown"])
    async def test_error_response_is_not_gradeable(
        self, *, sqlite_instance: SQLiteMemory, response_error: PromptResponseError
    ) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events)
        target.response_error = response_error

        with pytest.raises(RuntimeError, match="errored or blocked target response"):
            await _run_outer_async(attack=attack, context=_make_context(), owner=owner)

        assert owner.grade_calls == 0
        assert len(target.prompt_sent) == len(target.reset_ids) == 1
        assert sqlite_instance._query_entries(ScoreEntry) == []
        assert len(sqlite_instance.get_attack_results()) == 0

    async def test_session_can_only_run_once_and_reset_cannot_repeat(self, *, sqlite_instance: SQLiteMemory) -> None:
        events: list[str] = []
        attack, target, _ = _make_attack(events=events)
        owner = _OuterTask(target=target, memory=sqlite_instance, events=events)
        session = attack.external_final_scoring_session()

        await session.__aenter__()
        try:
            pending = await session.execute_with_context_async(context=_make_context())
            assert pending.executed_turns == 2
            with pytest.raises(RuntimeError, match="active, unused session"):
                await session.execute_with_context_async(context=_make_context())
            await owner.ensure_quiesced_async()
        finally:
            await session.__aexit__(None, None, None)
            await owner.teardown_sandbox_async()

        await session.__aexit__(None, None, None)
        assert len(target.reset_ids) == owner.stop_requests == 1
