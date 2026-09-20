# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from pyrit.memory import CentralMemory
from pyrit.models import Conversation, MessagePiece, MessageScorable, ScoringExpectation
from pyrit.score import InspectEvalScorer

pytest.importorskip("inspect_ai")


async def _native_fixture_async(
    *, tmp_path: Path, answer: str, expected: str, changes: dict[str, Any] | None = None
) -> tuple[InspectEvalScorer, MessageScorable]:
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log_async
    from inspect_ai.model import ModelName, ModelOutput
    from inspect_ai.scorer import Target, includes
    from inspect_ai.solver import TaskState

    piece = MessagePiece(role="assistant", original_value=answer, conversation_id=str(uuid4()))
    memory = CentralMemory.get_memory_instance()
    memory.add_conversation_to_memory(conversation=Conversation(conversation_id=piece.conversation_id))
    memory.add_message_to_memory(request=piece.to_message())
    state = TaskState(
        model=ModelName("external/pyrit"),
        sample_id=4,
        epoch=1,
        input="fixture task",
        messages=[],
        output=ModelOutput.from_content(model="fixture-model", content=answer),
    )
    grade = await includes()(state, Target(expected))
    sample_uuid, eval_id = str(uuid4()), str(uuid4())
    sample = EvalSample(
        id=4,
        epoch=1,
        input="fixture task",
        target=expected,
        uuid=sample_uuid,
        output=state.output,
        scores={"includes": grade},
        metadata={"pyrit_message_piece_id": str(piece.id), "pyrit_conversation_id": piece.conversation_id},
    )
    if changes:
        sample = sample.model_copy(update=changes)
    log = EvalLog(
        status="success",
        eval=EvalSpec(
            eval_id=eval_id,
            created="2026-09-20T00:00:00Z",
            task="fixture",
            dataset=EvalDataset(samples=1),
            model="none/none",
            config=EvalConfig(epochs=1, retry_on_error=0),
            metadata={"pyrit_run_id": "run", "pyrit_attempt_id": "attempt"},
        ),
        samples=[sample],
    )
    path = tmp_path / "native.eval"
    await write_eval_log_async(log, path)
    scorer = InspectEvalScorer(
        log_path=path,
        eval_id=eval_id,
        sample_id=4,
        sample_uuid=sample_uuid,
        run_id="run",
        attempt_id="attempt",
        message_piece_id=piece.id,
        input_sha256=hashlib.sha256(b"fixture task").hexdigest(),
        native_scorer="includes",
    )
    return scorer, MessageScorable.from_message(piece.to_message())


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("answer", "expected", "value"),
    [
        ("FLAG{fixture}", "flag{fixture}", True),
        ("not the answer", "flag{fixture}", False),
        ("x fixture y", "fixture", True),
        ("fixture", " fixture ", False),
        ("STRASSE", "stra\u00dfe", True),
    ],
)
async def test_native_includes_grade_round_trips_to_real_memory_async(
    tmp_path: Path, *, answer: str, expected: str, value: bool
) -> None:
    scorer, scorable = await _native_fixture_async(tmp_path=tmp_path, answer=answer, expected=expected)
    expectation = ScoringExpectation(objective="Native task correctness")
    scores = await scorer.score_async(scorable=scorable, expectation=expectation)
    assert len(scores) == 1
    assert scores[0].get_value() is value
    retained = CentralMemory.get_memory_instance().get_scores(score_ids=[str(scores[0].id)])
    assert retained[0].scorable == scorable
    assert retained[0].scored_expectation == expectation
    assert retained[0].score_metadata["native_grade"] == ("C" if value else "I")


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    "changes",
    [
        {"id": 5},
        {"epoch": 2},
        {"uuid": "another-sample"},
        {"input": "changed input"},
        {"scores": {}},
        {"metadata": {}},
    ],
)
async def test_projection_rejects_missing_or_mismatched_evidence_async(tmp_path: Path, changes: dict[str, Any]) -> None:
    scorer, scorable = await _native_fixture_async(
        tmp_path=tmp_path, answer="fixture", expected="fixture", changes=changes
    )
    with pytest.raises(RuntimeError):
        await scorer.score_async(scorable=scorable)
    assert CentralMemory.get_memory_instance().get_prompt_scores(prompt_ids=list(scorable.message_piece_ids)) == []


@pytest.mark.usefixtures("patch_central_database")
async def test_projection_rejects_infra_error_even_with_correct_grade_async(tmp_path: Path) -> None:
    from inspect_ai.log import EvalError

    scorer, scorable = await _native_fixture_async(
        tmp_path=tmp_path,
        answer="fixture",
        expected="fixture",
        changes={"error": EvalError(message="container failed", traceback="fixture", traceback_ansi="fixture")},
    )
    with pytest.raises(RuntimeError, match="errored"):
        await scorer.score_async(scorable=scorable)


@pytest.mark.usefixtures("patch_central_database")
async def test_projection_rejects_native_answer_mismatch_async(tmp_path: Path) -> None:
    from inspect_ai.model import ModelOutput

    scorer, scorable = await _native_fixture_async(
        tmp_path=tmp_path,
        answer="fixture",
        expected="fixture",
        changes={"output": ModelOutput.from_content(model="fixture-model", content="different")},
    )
    with pytest.raises(RuntimeError, match="completion does not match"):
        await scorer.score_async(scorable=scorable)
