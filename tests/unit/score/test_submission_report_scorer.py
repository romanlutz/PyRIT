# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pyrit.models import ContentEntryScorable, ContentScorable, ScoreStatus
from pyrit.models.submission import RetainedSubmissionReport, StrictSubmissionReport, SubmissionReportStatus
from pyrit.score import SubmissionReportScorer

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _report() -> RetainedSubmissionReport:
    submissions = [
        {
            "submission_id": f"OFFLINE-{index}",
            "sequence": index,
            "artifact_sha256": hashlib.sha256(b"OFFLINE/SIMULATED inert bytes").hexdigest(),
            "artifact_size_bytes": len(b"OFFLINE/SIMULATED inert bytes"),
            "dispatch_state": "returned",
            "status": "completed",
            "grade": grade,
            "behavior_outcome": "partial",
            "feedback": f"OFFLINE/SIMULATED observation {index}",
            "error_code": None,
            "raw_evidence": {"fixture_ordinal": index},
            "evidence_completeness": "complete",
            "acceptance": "accepted",
            "observed_request_id": None,
            "observed_receipt_id": None,
            "remote_disposition": "completed",
            "cleanup_status": "not_required",
        }
        for index, grade in enumerate((0.75, 0.25), start=1)
    ]
    projection = StrictSubmissionReport.model_validate(
        {
            "contract_version": "strict-submission-v1",
            "mode": "offline",
            "simulated": True,
            "status": "completed",
            "last_valid_grade": 0.25,
            "selected_submission_id": "OFFLINE-2",
            "full_success": False,
            "submissions": submissions,
            "capabilities": {"remote_query": False, "remote_cancel": False, "idempotency": False},
        }
    )
    return RetainedSubmissionReport(
        run_id="OFFLINE-SIMULATED-run",
        conversation_id="OFFLINE-SIMULATED-conversation",
        status=SubmissionReportStatus.COMPLETED,
        report=projection,
        calls=(),
        final_text=None,
        runner_error=None,
    )


async def test_numeric_projection_and_replay_have_no_message_or_evaluator_async(sqlite_instance: SQLiteMemory) -> None:
    report = _report()
    scorer = SubmissionReportScorer(report_sha256=report.sha256())
    score = (await scorer.score_async(scorable=ContentScorable(value=report.canonical_json())))[0]
    assert score.get_value() == 0.25
    assert isinstance(score.scorable, ContentEntryScorable)
    assert score.message_piece_id is None and score.observation_ids == []
    assert score.score_metadata["selected_submission_id"] == "OFFLINE-2"
    assert score.score_metadata["publication_role"] == "final_run_result"
    assert score.score_metadata["publication_boundary"] == "pyrit_score_commit"
    stored = sqlite_instance.get_scores(score_ids=[str(score.id)])[0]
    replay = (await scorer.score_async(scorable=stored.scorable))[0]
    assert replay.get_value() == 0.25
    assert replay.scorable == stored.scorable
    assert not sqlite_instance.get_message_pieces(conversation_id=report.conversation_id)


@pytest.mark.parametrize("status", ["incomplete", "error", "unknown", "cancelled"])
async def test_noncompleted_status_retains_grade_but_projects_undetermined_async(status: str) -> None:
    original = _report().model_dump(mode="json")
    original.update(status=status, runner_error="OFFLINE/SIMULATED interruption")
    report = RetainedSubmissionReport.model_validate(original)
    score = (
        await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json())
        )
    )[0]
    assert score.status is ScoreStatus.UNDETERMINED and score.score_value is None
    assert score.score_metadata["retained_last_valid_grade"] == 0.25
    assert score.score_metadata["run_status"] == status


@pytest.mark.parametrize("field", ["selected_submission_id", "last_valid_grade", "artifact_sha256"])
async def test_tampering_is_rejected_even_if_reserialized_async(field: str) -> None:
    report = _report()
    changed = copy.deepcopy(report.model_dump(mode="json"))
    if field == "artifact_sha256":
        changed["report"]["submissions"][-1][field] = "0" * 64
    else:
        changed["report"][field] = "OFFLINE-1" if field == "selected_submission_id" else 0.75
    with pytest.raises(RuntimeError, match="modified"):
        await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=json.dumps(changed))
        )


async def test_stored_content_change_is_rejected_on_replay_async(sqlite_instance: SQLiteMemory) -> None:
    report = _report()
    scorer = SubmissionReportScorer(report_sha256=report.sha256())
    score = (await scorer.score_async(scorable=ContentScorable(value=report.canonical_json())))[0]
    assert isinstance(score.scorable, ContentEntryScorable)
    with patch.object(
        sqlite_instance,
        "get_scorable_content",
        return_value={score.scorable.content_id: ContentScorable(value='{"tampered":true}')},
    ):
        with pytest.raises(RuntimeError, match="modified"):
            await scorer.score_async(scorable=score.scorable)


@pytest.mark.parametrize("grade", [None, True, "0.25", float("nan"), float("inf"), 1.5, -0.1])
def test_binding_grade_requires_finite_native_number(grade: object) -> None:
    data = _report().report.model_dump(mode="json")
    data["submissions"][-1]["grade"] = grade
    with pytest.raises(ValueError):
        StrictSubmissionReport.model_validate(data)


def test_best_of_or_fake_capabilities_are_not_valid_projections() -> None:
    data = _report().report.model_dump(mode="json")
    data["last_valid_grade"] = 0.75
    data["selected_submission_id"] = "OFFLINE-1"
    with pytest.raises(ValueError, match="latest valid"):
        StrictSubmissionReport.model_validate(data)
    data = _report().report.model_dump(mode="json")
    data["capabilities"]["remote_cancel"] = True
    with pytest.raises(ValueError):
        StrictSubmissionReport.model_validate(data)


@pytest.mark.parametrize(
    ("key", "value"),
    [("evidence_completeness", "partial"), ("cleanup_status", "unknown"), ("remote_disposition", "unknown")],
)
def test_uncertain_submission_cannot_project_clean_completion(*, key: str, value: str) -> None:
    data = _report().report.model_dump(mode="json")
    data["submissions"][-1][key] = value
    with pytest.raises(ValueError, match="complete acquisition"):
        StrictSubmissionReport.model_validate(data)


@pytest.mark.parametrize(
    ("acceptance", "disposition"),
    [
        ("not_dispatched", "not_dispatched"),
        ("not_dispatched", "completed"),
        ("accepted", "not_dispatched"),
        ("rejected", "completed"),
        ("accepted", "cancelled"),
    ],
)
async def test_contradictory_completed_lifecycle_rejected_at_dto_and_scorer_async(
    *, acceptance: str, disposition: str
) -> None:
    data = _report().model_dump(mode="json")
    data["report"]["submissions"][-1].update(acceptance=acceptance, remote_disposition=disposition)
    with pytest.raises(ValueError):
        StrictSubmissionReport.model_validate(data["report"])
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    scorer = SubmissionReportScorer(report_sha256=hashlib.sha256(raw.encode()).hexdigest())
    with pytest.raises(RuntimeError, match="submission"):
        await scorer.score_async(scorable=ContentScorable(value=raw))


async def test_unknown_completed_observation_is_retained_only_as_incomplete_async() -> None:
    data = _report().model_dump(mode="json")
    data["report"]["submissions"][-1].update(acceptance="unknown", remote_disposition="unknown")
    data["report"]["status"] = data["status"] = "incomplete"
    report = RetainedSubmissionReport.model_validate(data)
    score = (
        await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json())
        )
    )[0]
    assert score.is_undetermined
    assert score.score_metadata["retained_last_valid_grade"] == 0.25
    assert report.report.submissions[-1].observed_receipt_id is None
    assert report.report.submissions[-1].grade == 0.25


async def test_known_completed_outcome_does_not_require_an_ack_or_receipt_id_async() -> None:
    data = _report().model_dump(mode="json")
    data["report"]["submissions"][-1].update(
        acceptance="unknown", remote_disposition="completed", observed_request_id=None, observed_receipt_id=None
    )
    report = RetainedSubmissionReport.model_validate(data)
    score = (
        await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json())
        )
    )[0]
    assert score.get_value() == 0.25
    assert score.status is ScoreStatus.COMPLETE
    assert report.report.submissions[-1].observed_receipt_id is None


def test_predispatch_failure_may_have_frozen_bytes_but_not_remote_evidence() -> None:
    original = _report().report.submissions[-1].model_dump(mode="json")
    original.update(
        status="error",
        dispatch_state="not_dispatched",
        acceptance="not_dispatched",
        remote_disposition="not_dispatched",
        grade=None,
        behavior_outcome=None,
    )
    from pyrit.models.submission import StrictSubmissionRecord

    record = StrictSubmissionRecord.model_validate(original)
    assert record.artifact_sha256 is not None and record.observed_receipt_id is None
    for key, value in (
        ("acceptance", "accepted"),
        ("remote_disposition", "completed"),
        ("observed_receipt_id", "receipt"),
    ):
        with pytest.raises(ValueError, match="undispatched"):
            StrictSubmissionRecord.model_validate({**original, key: value})


async def test_text_report_sqlite_score_commit_does_not_yield_async(sqlite_instance: SQLiteMemory) -> None:
    report = _report()
    checkpoints: list[str] = []
    asyncio.get_running_loop().call_soon(checkpoints.append, "event-loop-yield")
    score = (
        await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json(), data_type="text")
        )
    )[0]
    assert not checkpoints
    assert sqlite_instance.get_scores(score_ids=[str(score.id)])[0].get_value() == 0.25
    await asyncio.sleep(0)
    assert checkpoints == ["event-loop-yield"]
