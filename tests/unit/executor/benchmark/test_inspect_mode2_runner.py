# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A local Inspect ReAct turn-control variant with original Task scoring and cleanup."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.event import ModelEvent, ScoreEvent, ToolEvent
from inspect_ai.log import read_eval_log
from sqlalchemy import func, select

from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory, ResolvedMode2InspectTask
from pyrit.executor.benchmark.inspect_mode2_runner import (
    InertReactAttackPolicy,
    InspectContinueAction,
    InspectContinueObservation,
    InspectMode2Controller,
    active_mode2_controller,
    run_mode2_inert_eval_async,
)
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter
from pyrit.memory.memory_models import AttackResultEntry, NativeCyberEpisodeEntry, ScoreEntry
from pyrit.models import ScoreStatus

if TYPE_CHECKING:
    from pathlib import Path

    from inspect_ai.log import EvalLog

    from pyrit.executor.benchmark.inspect_mode2_runner import InspectContinueDecision
    from pyrit.memory import SQLiteMemory


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_waits_for_pyrit_at_completed_turn_and_keeps_original_task_lifecycle(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    observed: list[InspectContinueObservation] = []
    original_decide = InertReactAttackPolicy.decide_async

    async def gated_decision_async(
        self: InertReactAttackPolicy, *, observation: InspectContinueObservation
    ) -> InspectContinueDecision:
        observed.append(observation)
        if observation.turn_index == 1:
            entered.set()
            await release.wait()
        return await original_decide(self, observation=observation)

    with patch.object(InertReactAttackPolicy, "decide_async", gated_decision_async):
        running = asyncio.create_task(run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path))
        try:
            await asyncio.wait_for(entered.wait(), timeout=30)
            assert not running.done()
            assert observed[0].completed_lookups == 1
            assert observed[0].lifecycle == ("setup", "tool")
            assert observed[0].sample_id == "mode2-inert-1"
        finally:
            release.set()
        completed = await running

    imported = completed.imported
    assert completed.run.case_run_id(case=completed.case) == imported.case_run_ids[0]
    assert completed.run.spec.harness.name == "inspect_mode2_react_variant"
    assert completed.run.spec.input_variant is None
    assert imported.episode.run.binding_name == InspectMode2Controller.BINDING
    assert imported.episode.run.binding_version == InspectMode2Controller.VARIANT
    assert imported.episode.coverage_complete
    assert imported.episode.score_id is None and imported.episode.score_status is ScoreStatus.UNDETERMINED
    assert "steered variant" in imported.no_grade_reasons[0]
    assert [record.action for record in completed.decisions] == [
        InspectContinueAction.NUDGE,
        InspectContinueAction.CONTINUE,
        InspectContinueAction.STOP,
    ]
    assert [item.turn_index for item in observed] == [1, 2, 3]
    assert [item.lifecycle for item in observed] == [
        ("setup", "tool"),
        ("setup", "tool", "attempt_score"),
        ("setup", "tool", "attempt_score"),
    ]
    assert len({record.observation.output_message_id for record in completed.decisions}) == 3

    [archive] = list(tmp_path.glob("*.eval"))
    log = read_eval_log(archive, resolve_attachments="full")
    assert log.status == "success"
    assert log.eval.task == completed.case.task_name
    assert log.samples and len(log.samples) == 1
    [sample] = log.samples
    assert sample.id == completed.case.sample_id and sample.uuid == imported.episode.turns[0].source_turn_id
    assert sample.store["lifecycle"] == ["setup", "tool", "attempt_score", "final_score", "cleanup"]
    assert sample.store["mode2_sample_id"] == completed.case.sample_id
    assert sample.scores and sample.scores["original_mode2_scorer"].value == 1.0
    assert len([event for event in sample.events if isinstance(event, ModelEvent)]) == 3
    assert len([event for event in sample.events if isinstance(event, ToolEvent)]) == 2
    score_events = [event for event in sample.events if isinstance(event, ScoreEvent)]
    assert len(score_events) == 2 and sum(event.intermediate for event in score_events) == 1
    assert imported.original_final_score_events == 1

    [control] = [message for message in sample.messages if (message.metadata or {}).get("pyrit_mode2_control")]
    assert control.id == completed.decisions[0].control_message_id
    assert control.text == InertReactAttackPolicy.NUDGE
    assert len(control.text.encode("utf-8")) <= 256
    turn = imported.episode.turns[0]
    pieces = sqlite_instance.get_message_pieces(
        prompt_ids=list(turn.request_piece_ids + turn.response_piece_ids + turn.tool_result_piece_ids)
    )
    assert control.id not in {piece.prompt_metadata["inspect_message_id"] for piece in pieces}
    assert InertReactAttackPolicy.NUDGE not in [piece.original_value for piece in pieces]
    assert any(piece.original_value == "harmless local fixture" for piece in pieces)
    assert imported.episode.events[-1].event_type == "inspect.projection.sample"

    [decisions] = [
        stream for stream in imported.episode.raw_streams if stream.key == InspectMode2Controller.DECISION_KEY
    ]
    assert decisions.source_complete
    assert decisions.stored_bytes <= InspectMode2Controller.MAX_RECORD_BYTES
    decision_bytes = b"".join(
        chunk.data
        for chunk in sqlite_instance.native_cyber_evidence.read_raw_chunks(
            run_id=imported.episode.run.run_id, stream_id=decisions.stream_id, allow_sensitive=True
        )
    )
    assert [json.loads(line)["action"] for line in decision_bytes.splitlines()] == ["nudge", "continue", "stop"]
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0

    with patch.object(EvalSourceFactory, "resolve_mode2_inert", side_effect=AssertionError("offline task import")):
        offline = await InspectOriginalEvalImporter(memory=sqlite_instance).import_eval_log_async(path=archive)
    assert offline.episode.run.binding_name == "inspect-original"
    assert offline.message_piece_count == imported.message_piece_count + 1
    assert offline.archive_sha256 == imported.archive_sha256
    assert offline.episode.score_id is None


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_source_pin_rejects_drift_before_launch(tmp_path: Path, sqlite_instance: SQLiteMemory) -> None:
    with (
        patch.object(EvalSourceFactory, "MODE2_INERT_SHA256", "f" * 64),
        patch("pyrit.executor.benchmark.inspect_mode2_runner.eval_async") as launched,
        pytest.raises(ValueError, match="pinned SHA256"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    launched.assert_not_called()
    with sqlite_instance.get_session() as session:
        assert session.scalar(select(func.count(NativeCyberEpisodeEntry.run_id))) == 0


def test_mode2_callback_is_default_off_outside_approved_run() -> None:
    with pytest.raises(RuntimeError, match="no approved active PyRIT attack"):
        active_mode2_controller()


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_timeout_records_required_gap_without_grade(tmp_path: Path, sqlite_instance: SQLiteMemory) -> None:
    async def stalled_decision_async(
        self: InertReactAttackPolicy, *, observation: InspectContinueObservation
    ) -> InspectContinueDecision:
        await asyncio.Event().wait()
        raise AssertionError("The timed-out decision must never be delivered.")

    with (
        patch.object(InertReactAttackPolicy, "decide_async", stalled_decision_async),
        patch.object(InspectMode2Controller, "DECISION_TIMEOUT_SECONDS", 0.01),
        pytest.raises(RuntimeError, match="incomplete evidence|pending capture"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    with sqlite_instance.get_session() as session:
        [episode] = session.scalars(select(NativeCyberEpisodeEntry)).all()
        assert not episode.coverage_complete
        assert "timed out" in " ".join(episode.capture_gaps)
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_run_deadline_fails_closed_without_a_control_server(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    async def stalled_eval_async(**kwargs: object) -> list[EvalLog]:
        await asyncio.Event().wait()
        return []

    with (
        patch("pyrit.executor.benchmark.inspect_mode2_runner.eval_async", side_effect=stalled_eval_async) as launched,
        patch.object(InspectMode2Controller, "RUN_TIMEOUT_SECONDS", 0.01),
        pytest.raises(RuntimeError, match="run timed out; pending capture"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    launched.assert_awaited_once()
    assert launched.call_args.kwargs["ctl_server"] is False
    assert launched.call_args.kwargs["acp_server"] is False
    _assert_mode2_ungraded_gap(memory=sqlite_instance, fragment="local time limit")


def _assert_mode2_ungraded_gap(*, memory: SQLiteMemory, fragment: str) -> None:
    with memory.get_session() as session:
        [episode] = session.scalars(select(NativeCyberEpisodeEntry)).all()
        assert episode.binding_name == InspectMode2Controller.BINDING
        assert not episode.coverage_complete
        assert fragment in " ".join(episode.capture_gaps)
        assert session.scalar(select(func.count(ScoreEntry.id))) == 0
        assert session.scalar(select(func.count(AttackResultEntry.id))) == 0


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_policy_error_fails_closed_without_delivering_a_nudge(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    with (
        patch.object(
            InertReactAttackPolicy,
            "decide_async",
            new_callable=AsyncMock,
            side_effect=ValueError("fixture policy error"),
        ) as decision,
        pytest.raises(RuntimeError, match="incomplete evidence|pending capture"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    decision.assert_awaited_once()
    _assert_mode2_ungraded_gap(memory=sqlite_instance, fragment="failed before delivery (ValueError)")


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_rejects_a_257_byte_nudge_before_delivery(tmp_path: Path, sqlite_instance: SQLiteMemory) -> None:
    with (
        patch.object(InertReactAttackPolicy, "NUDGE", "x" * 257),
        pytest.raises(RuntimeError, match="incomplete evidence|pending capture"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    _assert_mode2_ungraded_gap(memory=sqlite_instance, fragment="failed before delivery (ValueError)")


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_cancelled_while_waiting_for_pyrit_retains_an_incomplete_episode(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    entered = asyncio.Event()

    async def blocked_decision_async(
        self: InertReactAttackPolicy, *, observation: InspectContinueObservation
    ) -> InspectContinueDecision:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("A cancelled decision must never resume the agent.")

    with patch.object(InertReactAttackPolicy, "decide_async", blocked_decision_async):
        running = asyncio.create_task(run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path))
        await asyncio.wait_for(entered.wait(), timeout=30)
        running.cancel()
        with pytest.raises((asyncio.CancelledError, RuntimeError)):
            await running
    _assert_mode2_ungraded_gap(memory=sqlite_instance, fragment="cancelled")


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_post_run_source_drift_retains_only_ungraded_variant_evidence(
    tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    with (
        patch.object(
            ResolvedMode2InspectTask, "verify_unchanged", side_effect=[None, ValueError("fixture source drift")]
        ) as verify,
        pytest.raises(RuntimeError, match="source drifted; ungraded variant"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    assert verify.call_count == 2
    _assert_mode2_ungraded_gap(memory=sqlite_instance, fragment="source drifted")
    [archive] = list(tmp_path.glob("*.eval"))
    assert archive.stat().st_size > 0


@pytest.mark.usefixtures("patch_central_database")
async def test_mode2_foreign_log_keeps_decisions_ungraded(tmp_path: Path, sqlite_instance: SQLiteMemory) -> None:
    with (
        patch(
            "pyrit.executor.benchmark.inspect_mode2_runner._approved_log_location",
            side_effect=ValueError("foreign"),
        ),
        pytest.raises(RuntimeError, match="returned an unapproved log; pending capture"),
    ):
        await run_mode2_inert_eval_async(memory=sqlite_instance, log_dir=tmp_path)
    _assert_mode2_ungraded_gap(memory=sqlite_instance, fragment="foreign or missing local")
