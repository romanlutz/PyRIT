# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Result aggregation for capability-suite runs."""

from __future__ import annotations

from dataclasses import dataclass, field

from pyrit.scenario.capability_suite.results import AttemptOutcomeKind, CapabilitySuiteAttemptRecord


@dataclass(frozen=True)
class CapabilitySuiteAggregate:
    """Counts, rates, and score statistics computed over a suite's attempt records."""

    total_attempts: int
    total_runs: int
    outcome_counts: dict[str, int] = field(default_factory=dict)
    task_outcome_counts: dict[str, int] = field(default_factory=dict)
    success_rate: float = 0.0
    score_count: int = 0
    score_mean: float | None = None
    score_distribution: dict[str, int] = field(default_factory=dict)


def aggregate_attempts(attempts: tuple[CapabilitySuiteAttemptRecord, ...]) -> CapabilitySuiteAggregate:
    """
    Compute aggregate counts, rates, and score statistics over every preserved attempt.

    ``outcome_counts`` covers every ``AttemptOutcomeKind`` (including retried and
    cancelled attempts); ``task_outcome_counts`` covers only the ``CapabilityOutcome``
    of attempts that produced a terminal ``CapabilityTaskResult``. ``score_mean`` and
    ``score_distribution`` are computed over every ``Score`` on every attempt's result
    (``true_false`` scores contribute 1.0/0.0 and bucket by raw value; ``float_scale``
    scores contribute their numeric value and bucket to one decimal place).

    Returns:
        CapabilitySuiteAggregate: The computed aggregate.
    """
    outcome_counts: dict[str, int] = {}
    task_outcome_counts: dict[str, int] = {}
    score_values: list[float] = []
    score_distribution: dict[str, int] = {}

    for attempt in attempts:
        outcome_counts[attempt.outcome_kind.value] = outcome_counts.get(attempt.outcome_kind.value, 0) + 1
        if attempt.task_result is None:
            continue
        task_outcome_counts[attempt.task_result.outcome.value] = (
            task_outcome_counts.get(attempt.task_result.outcome.value, 0) + 1
        )
        for score in attempt.task_result.scores:
            if score.score_type == "true_false":
                value = 1.0 if score.score_value.lower() == "true" else 0.0
                bucket = score.score_value.lower()
            elif score.score_type == "float_scale":
                value = float(score.score_value)
                bucket = f"{round(value, 1):.1f}"
            else:
                continue
            score_values.append(value)
            score_distribution[bucket] = score_distribution.get(bucket, 0) + 1

    total_attempts = len(attempts)
    final_attempts = [attempt for attempt in attempts if attempt.outcome_kind is not AttemptOutcomeKind.RETRY]
    total_runs = len(final_attempts)
    successes = sum(attempt.outcome_kind is AttemptOutcomeKind.SUCCESS for attempt in final_attempts)
    success_rate = successes / total_runs if total_runs else 0.0
    score_mean = sum(score_values) / len(score_values) if score_values else None

    return CapabilitySuiteAggregate(
        total_attempts=total_attempts,
        total_runs=total_runs,
        outcome_counts=outcome_counts,
        task_outcome_counts=task_outcome_counts,
        success_rate=success_rate,
        score_count=len(score_values),
        score_mean=score_mean,
        score_distribution=score_distribution,
    )
