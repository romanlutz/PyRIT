# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest

from pyrit.models import ContentEntryScorable, ContentScorable, ScoreStatus
from pyrit.models.native_cyber import (
    NativeAgentEvidence,
    NativeCyberCleanup,
    NativeCyberJudgment,
    NativeCyberReadiness,
    NativeCyberReport,
    NativeCyberRequest,
    NativeCyberStatus,
)
from pyrit.score.float_scale.native_cyber_scorer import NativeCyberReportScorer

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _report(*, status: NativeCyberStatus = NativeCyberStatus.COMPLETED) -> NativeCyberReport:
    now = datetime.now(UTC)
    return NativeCyberReport(
        run_id="synthetic-run",
        binding_name="synthetic-task",
        binding_version="1",
        request=NativeCyberRequest(instruction="Synthetic task input"),
        input_sha256=hashlib.sha256(b"Synthetic task input").hexdigest(),
        status=status,
        simulated=True,
        readiness=NativeCyberReadiness(ready=True, simulated=True),
        started_at=now,
        expires_at=now + timedelta(minutes=1),
        ended_at=now,
        agent=NativeAgentEvidence(
            session_id="synthetic-session",
            environment_id="synthetic-environment",
            simulated=True,
            events=(),
            tools=(),
            idle=True,
            coverage_complete=True,
            gaps=(),
        ),
        judgment=NativeCyberJudgment(value=0.75, complete=True, rationale="Synthetic original judgment"),
        cleanup=NativeCyberCleanup.CLOSED if status is NativeCyberStatus.COMPLETED else NativeCyberCleanup.FAILED,
        errors=() if status is NativeCyberStatus.COMPLETED else ("Synthetic cleanup failure",),
    )


async def test_prepared_native_score_is_unpersisted_but_matches_normal_scoring_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    report = _report()
    scorer = NativeCyberReportScorer(report_sha256=report.sha256())

    prepared = scorer.prepare_unpersisted_score(report=report)

    assert prepared.get_value() == 0.75
    assert isinstance(prepared.scorable, ContentScorable)
    assert prepared.scorable.value == report.canonical_json()
    assert not sqlite_instance.get_scores(score_type="float_scale")

    persisted = (await scorer.score_async(scorable=ContentScorable(value=report.canonical_json())))[0]
    assert persisted.score_value == prepared.score_value
    assert persisted.status is prepared.status is ScoreStatus.COMPLETE
    assert persisted.score_metadata == prepared.score_metadata
    assert persisted.scorer_class_identifier == prepared.scorer_class_identifier
    assert isinstance(persisted.scorable, ContentEntryScorable)
    assert len(sqlite_instance.get_scores(score_ids=[str(persisted.id)])) == 1


def test_prepared_native_score_rejects_wrong_report_digest(sqlite_instance: SQLiteMemory) -> None:
    report = _report()
    scorer = NativeCyberReportScorer(report_sha256="0" * 64)

    with pytest.raises(ValueError, match="not canonical retained content"):
        scorer.prepare_unpersisted_score(report=report)

    assert not sqlite_instance.get_scores(score_type="float_scale")


def test_prepared_native_score_keeps_original_judgment_but_is_undetermined_on_failure(
    sqlite_instance: SQLiteMemory,
) -> None:
    report = _report(status=NativeCyberStatus.ERROR)
    scorer = NativeCyberReportScorer(report_sha256=report.sha256())

    prepared = scorer.prepare_unpersisted_score(report=report)

    assert report.judgment.value == 0.75
    assert prepared.score_value is None
    assert prepared.status is ScoreStatus.UNDETERMINED
    assert prepared.score_rationale == "Synthetic original judgment"
    assert not sqlite_instance.get_scores(score_type="float_scale")
