# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pyrit.executor.capability import CapabilityOutcome, CapabilityTaskResult, CapabilityTerminationReason
from pyrit.models import Score, TargetIdentifier
from pyrit.scenario.capability_suite.aggregation import aggregate_attempts
from pyrit.scenario.capability_suite.results import AttemptOutcomeKind, CapabilitySuiteAttemptRecord


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _target_identifier() -> TargetIdentifier:
    return TargetIdentifier(class_name="FakeTarget", class_module="tests.unit.scenario.capability_suite")


def _score(*, value: str, score_type: str = "true_false") -> Score:
    return Score(
        score_value=value,
        score_type=score_type,
        message_piece_id=uuid.uuid4(),
        objective="finish",
    )


def _task_result(*, outcome: CapabilityOutcome, scores: tuple[Score, ...] = ()) -> CapabilityTaskResult:
    return CapabilityTaskResult(
        case_id=uuid.uuid4(),
        conversation_id=str(uuid.uuid4()),
        target_identifier=_target_identifier(),
        outcome=outcome,
        termination_reason=CapabilityTerminationReason.COMPLETION,
        scores=scores,
        started_at=_now(),
        ended_at=_now(),
    )


def _attempt(
    *,
    outcome_kind: AttemptOutcomeKind,
    task_result: CapabilityTaskResult | None = None,
    case_id: str = "case-1",
    attempt_number: int = 1,
) -> CapabilitySuiteAttemptRecord:
    return CapabilitySuiteAttemptRecord(
        attempt_key=f"{case_id}:epoch1:run1:try{attempt_number}",
        attempt_id=uuid.uuid4(),
        case_id=case_id,
        epoch=1,
        repetition=1,
        attempt_number=attempt_number,
        outcome_kind=outcome_kind,
        task_result=task_result,
        error=None,
        retry_reason=None,
        started_at=_now(),
        ended_at=_now(),
    )


def test_aggregate_attempts_empty() -> None:
    aggregate = aggregate_attempts(())
    assert aggregate.total_attempts == 0
    assert aggregate.total_runs == 0
    assert aggregate.success_rate == 0.0
    assert aggregate.score_mean is None
    assert aggregate.outcome_counts == {}


def test_aggregate_attempts_counts_outcome_kinds() -> None:
    attempts = (
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(outcome=CapabilityOutcome.COMPLETED),
        ),
        _attempt(outcome_kind=AttemptOutcomeKind.RETRY, attempt_number=1, case_id="case-2"),
        _attempt(outcome_kind=AttemptOutcomeKind.FAILURE, attempt_number=2, case_id="case-2"),
        _attempt(outcome_kind=AttemptOutcomeKind.CANCELLED, case_id="case-3"),
        _attempt(outcome_kind=AttemptOutcomeKind.CLEANUP_FAILURE, case_id="case-4"),
    )
    aggregate = aggregate_attempts(attempts)
    assert aggregate.total_attempts == 5
    assert aggregate.outcome_counts == {
        "success": 1,
        "retry": 1,
        "failure": 1,
        "cancelled": 1,
        "cleanup_failure": 1,
    }
    assert aggregate.total_runs == 4
    assert aggregate.success_rate == 1 / 4


def test_aggregate_attempts_success_rate_uses_final_run_outcomes_not_retries() -> None:
    attempts = (
        _attempt(outcome_kind=AttemptOutcomeKind.RETRY, attempt_number=1),
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            attempt_number=2,
            task_result=_task_result(outcome=CapabilityOutcome.COMPLETED),
        ),
    )
    aggregate = aggregate_attempts(attempts)
    assert aggregate.total_attempts == 2
    assert aggregate.total_runs == 1
    assert aggregate.success_rate == 1.0


def test_aggregate_attempts_task_outcome_counts_only_cover_attempts_with_results() -> None:
    attempts = (
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(outcome=CapabilityOutcome.COMPLETED),
        ),
        _attempt(outcome_kind=AttemptOutcomeKind.CANCELLED, task_result=None, case_id="case-2"),
    )
    aggregate = aggregate_attempts(attempts)
    assert aggregate.task_outcome_counts == {"completed": 1}


def test_aggregate_attempts_score_mean_and_distribution_true_false() -> None:
    result_true = _task_result(outcome=CapabilityOutcome.COMPLETED, scores=(_score(value="true"),))
    result_false = _task_result(outcome=CapabilityOutcome.COMPLETED, scores=(_score(value="false"),))
    attempts = (
        _attempt(outcome_kind=AttemptOutcomeKind.SUCCESS, task_result=result_true, case_id="case-1"),
        _attempt(outcome_kind=AttemptOutcomeKind.SUCCESS, task_result=result_false, case_id="case-2"),
    )
    aggregate = aggregate_attempts(attempts)
    assert aggregate.score_count == 2
    assert aggregate.score_mean == 0.5
    assert aggregate.score_distribution == {"true": 1, "false": 1}


def test_aggregate_attempts_score_mean_and_distribution_float_scale() -> None:
    result = _task_result(
        outcome=CapabilityOutcome.COMPLETED,
        scores=(_score(value="0.25", score_type="float_scale"), _score(value="0.75", score_type="float_scale")),
    )
    attempts = (_attempt(outcome_kind=AttemptOutcomeKind.SUCCESS, task_result=result),)
    aggregate = aggregate_attempts(attempts)
    assert aggregate.score_count == 2
    assert aggregate.score_mean == 0.5
    assert aggregate.score_distribution == {"0.2": 1, "0.8": 1}
