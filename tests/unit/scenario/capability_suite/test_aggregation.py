# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from pyrit.executor.capability import CapabilityOutcome, CapabilityTaskResult, CapabilityTerminationReason
from pyrit.models import Score, TargetIdentifier
from pyrit.scenario.capability_suite.aggregation import aggregate_attempts
from pyrit.scenario.capability_suite.manifest import ScoreMetricManifest, ScoreReducerManifest
from pyrit.scenario.capability_suite.results import AttemptOutcomeKind, CapabilitySuiteAttemptRecord


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _target_identifier() -> TargetIdentifier:
    return TargetIdentifier(class_name="FakeTarget", class_module="tests.unit.scenario.capability_suite")


def _score(
    *,
    value: str,
    score_type: str = "true_false",
    metadata: dict[str, str | int | float] | None = None,
) -> Score:
    return Score(
        score_value=value,
        score_type=score_type,
        score_metadata=metadata,
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
    epoch: int = 1,
) -> CapabilitySuiteAttemptRecord:
    return CapabilitySuiteAttemptRecord(
        attempt_key=f"{case_id}:epoch1:run1:try{attempt_number}",
        attempt_id=uuid.uuid4(),
        case_id=case_id,
        epoch=epoch,
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


def test_aggregate_attempts_at_least_one_reduces_epochs_per_case() -> None:
    attempts = (
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(outcome=CapabilityOutcome.COMPLETED, scores=(_score(value="False"),)),
            epoch=1,
        ),
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(outcome=CapabilityOutcome.COMPLETED, scores=(_score(value="True"),)),
            epoch=2,
        ),
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(outcome=CapabilityOutcome.COMPLETED, scores=(_score(value="False"),)),
            epoch=3,
        ),
    )

    aggregate = aggregate_attempts(attempts, epoch_reducer="at_least_1")

    assert aggregate.score_count == 1
    assert aggregate.score_mean == 1.0
    assert aggregate.score_distribution == {"true": 1}


def test_aggregate_attempts_computes_named_grouped_metrics() -> None:
    attempts = (
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(
                    _score(
                        value="True",
                        metadata={"capability_scorer_id": "primary", "subject": "science"},
                    ),
                ),
            ),
            case_id="case-1",
        ),
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(
                    _score(
                        value="False",
                        metadata={"capability_scorer_id": "primary", "subject": "history"},
                    ),
                ),
            ),
            case_id="case-2",
        ),
    )

    aggregate = aggregate_attempts(
        attempts,
        metrics=(
            ScoreMetricManifest(
                name="subject-accuracy",
                kind="accuracy",
                scorer_id="primary",
                group_by=("metadata.subject",),
            ),
        ),
    )

    assert aggregate.metric_values == {"subject-accuracy": 0.5}
    assert aggregate.grouped_metric_values == {
        "subject-accuracy": {
            "metadata.subject=history": 0.0,
            "metadata.subject=science": 1.0,
        }
    }


def test_aggregate_attempts_grouped_metric_can_aggregate_samples_instead_of_groups() -> None:
    attempts = tuple(
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(
                    _score(
                        value=value,
                        metadata={"capability_scorer_id": "primary", "subject": subject},
                    ),
                ),
            ),
            case_id=f"case-{index}",
        )
        for index, (subject, value) in enumerate(
            (("science", "True"), ("science", "True"), ("science", "False"), ("history", "False")),
            start=1,
        )
    )

    aggregate = aggregate_attempts(
        attempts,
        metrics=(
            ScoreMetricManifest(
                name="subject-accuracy",
                kind="accuracy",
                scorer_id="primary",
                group_by=("metadata.subject",),
                group_aggregate="samples",
            ),
        ),
    )

    assert aggregate.metric_values == {"subject-accuracy": 0.5}
    assert aggregate.grouped_metric_values["subject-accuracy"] == {
        "metadata.subject=history": 0.0,
        "metadata.subject=science": 2 / 3,
    }


def test_aggregate_attempts_computes_pass_at_and_reliability_reducers() -> None:
    attempts = tuple(
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(
                    _score(
                        value=value,
                        metadata={"capability_scorer_id": "primary"},
                    ),
                ),
            ),
            epoch=epoch,
        )
        for epoch, value in enumerate(("True", "False", "True"), start=1)
    )

    aggregate = aggregate_attempts(
        attempts,
        reducers=(
            ScoreReducerManifest(name="pass-at-2", kind="pass_at", scorer_id="primary", k=2),
            ScoreReducerManifest(name="pass-2", kind="pass_k", scorer_id="primary", k=2),
            ScoreReducerManifest(name="reliable", kind="reliability", scorer_id="primary"),
        ),
    )

    assert aggregate.reducer_values == {
        "pass-at-2": 1.0,
        "pass-2": 1 / 3,
        "reliable": 0.0,
    }


def test_aggregate_attempts_applies_reducer_before_metric() -> None:
    attempts = tuple(
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(
                    _score(
                        value=value,
                        metadata={"capability_scorer_id": "primary"},
                    ),
                ),
            ),
            case_id=case_id,
            epoch=epoch,
        )
        for case_id, epoch, value in (
            ("case-1", 1, "True"),
            ("case-1", 2, "False"),
            ("case-2", 1, "False"),
            ("case-2", 2, "False"),
        )
    )
    reducer = ScoreReducerManifest(name="primary-pass-at-2", kind="pass_at", scorer_id="primary", k=2)

    aggregate = aggregate_attempts(
        attempts,
        metrics=(
            ScoreMetricManifest(
                name="accuracy-after-pass-at-2",
                kind="accuracy",
                scorer_id="primary",
                reducer_name=reducer.name,
            ),
        ),
        reducers=(reducer,),
    )

    assert aggregate.metric_values == {"accuracy-after-pass-at-2": 0.5}


def test_aggregate_attempts_excludes_superseded_retry_scores() -> None:
    attempts = (
        _attempt(
            outcome_kind=AttemptOutcomeKind.RETRY,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(_score(value="True", metadata={"capability_scorer_id": "primary"}),),
            ),
        ),
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(_score(value="False", metadata={"capability_scorer_id": "primary"}),),
            ),
        ),
    )

    aggregate = aggregate_attempts(
        attempts,
        metrics=(ScoreMetricManifest(name="accuracy", kind="accuracy", scorer_id="primary"),),
    )

    assert aggregate.score_count == 1
    assert aggregate.score_mean == 0.0
    assert aggregate.metric_values == {"accuracy": 0.0}


def test_aggregate_attempts_reduces_dict_score_keys_independently() -> None:
    attempts = tuple(
        _attempt(
            outcome_kind=AttemptOutcomeKind.SUCCESS,
            task_result=_task_result(
                outcome=CapabilityOutcome.COMPLETED,
                scores=(
                    _score(
                        value="True",
                        metadata={"capability_scorer_id": "primary", "score_key": "accuracy"},
                    ),
                    _score(
                        value="False",
                        metadata={"capability_scorer_id": "primary", "score_key": "fluency"},
                    ),
                ),
            ),
            epoch=epoch,
        )
        for epoch in (1, 2)
    )

    aggregate = aggregate_attempts(
        attempts,
        reducers=(
            ScoreReducerManifest(
                name="reliability",
                kind="reliability",
                scorer_id="primary",
            ),
        ),
    )

    assert aggregate.reducer_values == {"reliability": 0.5}


def test_aggregate_attempts_excludes_unscored_non_finite_values() -> None:
    unscored = _attempt(
        outcome_kind=AttemptOutcomeKind.SUCCESS,
        task_result=_task_result(
            outcome=CapabilityOutcome.COMPLETED,
            scores=(_score(value="nan", score_type="unknown"),),
        ),
        epoch=1,
    )
    scored = _attempt(
        outcome_kind=AttemptOutcomeKind.SUCCESS,
        task_result=_task_result(
            outcome=CapabilityOutcome.COMPLETED,
            scores=(_score(value="0.5", score_type="float_scale"),),
        ),
        epoch=2,
    )
    metric = ScoreMetricManifest(name="mean", kind="mean")
    reducer = ScoreReducerManifest(name="mean-reducer", kind="mean")

    mixed = aggregate_attempts((unscored, scored), metrics=(metric,), reducers=(reducer,))
    all_unscored = aggregate_attempts((unscored,), metrics=(metric,), reducers=(reducer,))

    assert mixed.score_count == 1
    assert mixed.score_mean == 0.5
    assert mixed.metric_values == {"mean": 0.5}
    assert mixed.reducer_values == {"mean-reducer": 0.5}
    assert all_unscored.score_count == 0
    assert all_unscored.score_mean is None
    assert all_unscored.metric_values == {}
    assert all_unscored.reducer_values == {}
