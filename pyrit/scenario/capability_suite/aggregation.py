# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Result aggregation for capability-suite runs."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field, replace
from statistics import fmean
from typing import TYPE_CHECKING

from pyrit.scenario.capability_suite.results import AttemptOutcomeKind, CapabilitySuiteAttemptRecord

if TYPE_CHECKING:
    from .manifest import ScoreMetricManifest, ScoreReducerManifest


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
    metric_values: dict[str, float] = field(default_factory=dict)
    grouped_metric_values: dict[str, dict[str, float]] = field(default_factory=dict)
    reducer_values: dict[str, float] = field(default_factory=dict)


def aggregate_attempts(
    attempts: tuple[CapabilitySuiteAttemptRecord, ...],
    *,
    epoch_reducer: str = "mean",
    metrics: tuple[ScoreMetricManifest, ...] = (),
    reducers: tuple[ScoreReducerManifest, ...] = (),
) -> CapabilitySuiteAggregate:
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

    Raises:
        ValueError: If ``epoch_reducer`` is unknown.
    """
    outcome_counts: dict[str, int] = {}
    task_outcome_counts: dict[str, int] = {}
    score_values: list[float] = []
    score_distribution: dict[str, int] = {}
    observations: list[_ScoreObservation] = []

    grouped_score_values: dict[tuple[str, int, int], list[tuple[float, str]]] = {}
    for attempt in attempts:
        outcome_counts[attempt.outcome_kind.value] = outcome_counts.get(attempt.outcome_kind.value, 0) + 1
        if attempt.task_result is None:
            continue
        task_outcome_counts[attempt.task_result.outcome.value] = (
            task_outcome_counts.get(attempt.task_result.outcome.value, 0) + 1
        )
        if attempt.outcome_kind is AttemptOutcomeKind.RETRY:
            continue
        for score_index, score in enumerate(attempt.task_result.scores):
            normalized = _normalized_score(score_type=score.score_type, score_value=score.score_value)
            if normalized is None:
                continue
            value, bucket = normalized
            metadata = score.score_metadata or {}
            observations.append(
                _ScoreObservation(
                    value=value,
                    bucket=bucket,
                    case_id=attempt.case_id,
                    epoch=attempt.epoch,
                    repetition=attempt.repetition,
                    score_index=score_index,
                    scorer_id=str(metadata.get("capability_scorer_id", score_index)),
                    score_key=str(metadata["score_key"]) if "score_key" in metadata else None,
                    metadata=metadata,
                )
            )
            if epoch_reducer == "at_least_1" and attempt.outcome_kind is not AttemptOutcomeKind.RETRY:
                key = (attempt.case_id, attempt.repetition, score_index)
                grouped_score_values.setdefault(key, []).append((value, bucket))
            else:
                score_values.append(value)
                score_distribution[bucket] = score_distribution.get(bucket, 0) + 1

    if epoch_reducer == "at_least_1":
        for values in grouped_score_values.values():
            value, bucket = max(values, key=lambda item: item[0])
            score_values.append(value)
            score_distribution[bucket] = score_distribution.get(bucket, 0) + 1
    elif epoch_reducer != "mean":
        raise ValueError(f"Unsupported epoch reducer '{epoch_reducer}'.")

    total_attempts = len(attempts)
    final_attempts = [attempt for attempt in attempts if attempt.outcome_kind is not AttemptOutcomeKind.RETRY]
    total_runs = len(final_attempts)
    successes = sum(attempt.outcome_kind is AttemptOutcomeKind.SUCCESS for attempt in final_attempts)
    success_rate = successes / total_runs if total_runs else 0.0
    score_mean = sum(score_values) / len(score_values) if score_values else None
    metric_values, grouped_metric_values = _compute_metrics(
        observations=observations,
        metrics=metrics,
        reducers=reducers,
    )
    reducer_values = _compute_reducers(observations=observations, reducers=reducers)

    return CapabilitySuiteAggregate(
        total_attempts=total_attempts,
        total_runs=total_runs,
        outcome_counts=outcome_counts,
        task_outcome_counts=task_outcome_counts,
        success_rate=success_rate,
        score_count=len(score_values),
        score_mean=score_mean,
        score_distribution=score_distribution,
        metric_values=metric_values,
        grouped_metric_values=grouped_metric_values,
        reducer_values=reducer_values,
    )


@dataclass(frozen=True)
class _ScoreObservation:
    value: float
    bucket: str
    case_id: str
    epoch: int
    repetition: int
    score_index: int
    scorer_id: str
    score_key: str | None
    metadata: dict[str, str | int | float]


def _normalized_score(*, score_type: str, score_value: str) -> tuple[float, str] | None:
    if score_type == "true_false":
        return (1.0 if score_value.lower() == "true" else 0.0, score_value.lower())
    if score_type == "float_scale":
        value = float(score_value)
        if not math.isfinite(value):
            return None
        return value, f"{round(value, 1):.1f}"
    if score_type == "unknown":
        try:
            value = float(score_value)
        except ValueError:
            return None
        if not math.isfinite(value):
            return None
        return value, score_value
    return None


def _compute_metrics(
    *,
    observations: list[_ScoreObservation],
    metrics: tuple[ScoreMetricManifest, ...],
    reducers: tuple[ScoreReducerManifest, ...],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    values: dict[str, float] = {}
    grouped_values: dict[str, dict[str, float]] = {}
    reducers_by_name = {reducer.name: reducer for reducer in reducers}
    for metric in metrics:
        selected = _select_observations(
            observations=observations,
            scorer_id=metric.scorer_id,
            score_key=metric.score_key,
        )
        if metric.reducer_name is not None:
            reducer = reducers_by_name.get(metric.reducer_name)
            if reducer is None:
                raise ValueError(f"Metric '{metric.name}' references unknown reducer '{metric.reducer_name}'.")
            selected = _reduce_observations(observations=selected, reducer=reducer)
        if metric.group_by:
            groups = _group_observations(observations=selected, group_by=metric.group_by)
            grouped_values[metric.name] = {
                key: _metric_value(
                    kind=metric.kind,
                    observations=group,
                    cluster_by=metric.cluster_by,
                )
                for key, group in sorted(groups.items())
                if group
            }
            if metric.group_aggregate == "samples" and selected:
                values[metric.name] = _metric_value(
                    kind=metric.kind,
                    observations=selected,
                    cluster_by=metric.cluster_by,
                )
            elif metric.group_aggregate == "groups" and grouped_values[metric.name]:
                values[metric.name] = fmean(grouped_values[metric.name].values())
        elif selected:
            values[metric.name] = _metric_value(
                kind=metric.kind,
                observations=selected,
                cluster_by=metric.cluster_by,
            )
    return values, grouped_values


def _reduce_observations(
    *,
    observations: list[_ScoreObservation],
    reducer: ScoreReducerManifest,
) -> list[_ScoreObservation]:
    groups = _group_observations(
        observations=observations,
        group_by=_reducer_group_by(observations=observations, reducer=reducer),
    )
    return [
        replace(
            group[0],
            value=_reducer_value(
                kind=reducer.kind,
                values=[item.value for item in group],
                k=reducer.k,
                threshold=reducer.threshold,
            ),
            bucket=reducer.name,
            epoch=0,
            repetition=0,
        )
        for group in groups.values()
        if group
    ]


def _compute_reducers(
    *,
    observations: list[_ScoreObservation],
    reducers: tuple[ScoreReducerManifest, ...],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for reducer in reducers:
        selected = _select_observations(
            observations=observations,
            scorer_id=reducer.scorer_id,
            score_key=reducer.score_key,
        )
        groups = _group_observations(
            observations=selected,
            group_by=_reducer_group_by(observations=selected, reducer=reducer),
        )
        reduced = [
            _reducer_value(
                kind=reducer.kind,
                values=[item.value for item in group],
                k=reducer.k,
                threshold=reducer.threshold,
            )
            for group in groups.values()
            if group
        ]
        if reduced:
            values[reducer.name] = fmean(reduced)
    return values


def _reducer_group_by(
    *,
    observations: list[_ScoreObservation],
    reducer: ScoreReducerManifest,
) -> tuple[str, ...]:
    if reducer.score_key is None and any(observation.score_key is not None for observation in observations):
        return (*reducer.group_by, "score_key")
    return reducer.group_by


def _select_observations(
    *,
    observations: list[_ScoreObservation],
    scorer_id: str | None,
    score_key: str | None,
) -> list[_ScoreObservation]:
    return [
        observation
        for observation in observations
        if (scorer_id is None or observation.scorer_id == scorer_id)
        and (score_key is None or observation.score_key == score_key)
    ]


def _group_observations(
    *,
    observations: list[_ScoreObservation],
    group_by: tuple[str, ...],
) -> dict[str, list[_ScoreObservation]]:
    groups: dict[str, list[_ScoreObservation]] = defaultdict(list)
    for observation in observations:
        parts = [_group_value(observation=observation, field=field) for field in group_by]
        groups["|".join(parts)].append(observation)
    return dict(groups)


def _group_value(*, observation: _ScoreObservation, field: str) -> str:
    values: dict[str, object] = {
        "case_id": observation.case_id,
        "epoch": observation.epoch,
        "repetition": observation.repetition,
        "scorer_id": observation.scorer_id,
        "score_key": observation.score_key or "",
    }
    if field in values:
        return f"{field}={values[field]}"
    if field.startswith("metadata."):
        key = field.removeprefix("metadata.")
        return f"{field}={observation.metadata.get(key, '')}"
    raise ValueError(f"Unsupported score grouping field '{field}'.")


def _metric_value(
    *,
    kind: str,
    observations: list[_ScoreObservation],
    cluster_by: str | None,
) -> float:
    values = [observation.value for observation in observations]
    if kind in {"mean", "accuracy"}:
        return fmean(values)
    if kind == "stderr":
        if cluster_by is not None:
            clusters = _group_observations(observations=observations, group_by=(f"metadata.{cluster_by}",))
            if len(clusters) < 2:
                return 0.0
            mean = fmean(values)
            squared_cluster_residuals = sum(
                sum(item.value - mean for item in cluster) ** 2 for cluster in clusters.values()
            )
            correction = len(clusters) / (len(clusters) - 1)
            return math.sqrt(correction * squared_cluster_residuals / (len(values) ** 2))
        if len(values) < 2:
            return 0.0
        mean = fmean(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        return math.sqrt(variance / len(values))
    raise ValueError(f"Unsupported score metric '{kind}'.")


def _reducer_value(
    *,
    kind: str,
    values: list[float],
    k: int | None,
    threshold: float,
) -> float:
    if kind == "mean":
        return fmean(values)
    successes = sum(value >= threshold for value in values)
    if kind == "at_least_1":
        return float(successes > 0)
    if kind == "at_least":
        if k is None:
            raise ValueError("Reducer 'at_least' requires k.")
        return float(successes >= k)
    if kind == "reliability":
        return float(successes == len(values))
    if kind in {"pass_at", "pass_at_k"}:
        if k is None:
            raise ValueError(f"Reducer '{kind}' requires k.")
        trials = len(values)
        if trials < k:
            raise ValueError(f"Reducer '{kind}' requires at least {k} observations per group, got {trials}.")
        failures = trials - successes
        return 1.0 if failures < k else 1.0 - math.comb(failures, k) / math.comb(trials, k)
    if kind == "pass_k":
        if k is None:
            raise ValueError("Reducer 'pass_k' requires k.")
        trials = len(values)
        if trials < k:
            raise ValueError(f"Reducer 'pass_k' requires at least {k} observations per group, got {trials}.")
        return 0.0 if successes < k else math.comb(successes, k) / math.comb(trials, k)
    raise ValueError(f"Unsupported score reducer '{kind}'.")
