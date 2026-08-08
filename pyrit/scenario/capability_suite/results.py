# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared result and attempt-record models for capability-suite runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import uuid
    from datetime import datetime

    from pyrit.executor.capability import CapabilityTaskResult
    from pyrit.scenario.capability_suite.aggregation import CapabilitySuiteAggregate


class AttemptOutcomeKind(str, Enum):
    """The disposition of one physical attempt (a single fresh-sandbox run)."""

    #: The attempt completed and no further attempt was needed.
    SUCCESS = "success"
    #: The attempt failed in a known-retryable way and a subsequent attempt was launched.
    RETRY = "retry"
    #: The attempt failed and was not retried (not retryable, or attempts exhausted).
    FAILURE = "failure"
    #: The attempt was skipped or interrupted by external cancellation.
    CANCELLED = "cancelled"
    #: The task result is preserved, but sandbox ``close_async()`` itself raised.
    CLEANUP_FAILURE = "cleanup_failure"


@dataclass(frozen=True)
class CapabilitySuiteAttemptRecord:
    """One preserved, immutable record of a single physical attempt."""

    attempt_key: str
    attempt_id: uuid.UUID
    case_id: str
    epoch: int
    repetition: int
    attempt_number: int
    outcome_kind: AttemptOutcomeKind
    task_result: CapabilityTaskResult | None
    error: str | None
    retry_reason: str | None
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True)
class CapabilitySuiteRunResult:
    """The complete, preserved output of one capability-suite run."""

    run_id: str
    manifest_hash: str
    attempts: tuple[CapabilitySuiteAttemptRecord, ...]
    aggregate: CapabilitySuiteAggregate
    provider_cleanup_error: str | None = None


@dataclass(frozen=True)
class CapabilitySuiteProgress:
    """A monotonic progress snapshot emitted after one logical run unit finishes."""

    completed_units: int
    total_units: int
    latest_attempts: tuple[CapabilitySuiteAttemptRecord, ...]
