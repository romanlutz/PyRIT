# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from pyrit.models import ContentEntryScorable, ContentScorable
from pyrit.models.submission import RetainedSubmissionReport, StrictSubmissionReport
from pyrit.models.submission_v2 import RetainedSubmissionReportV2, StrictSubmissionReportV2, SubmissionProvenanceV2
from pyrit.score import SubmissionReportScorer
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _projection(*, mode: str = "offline", simulated: bool = True) -> dict:
    payload = b"OFFLINE/SIMULATED inert v2 fixture"
    return {
        "contract_version": "strict-submission-v2",
        "mode": mode,
        "simulated": simulated,
        "status": "completed",
        "last_valid_grade": 0.25,
        "selected_submission_id": "fixture-submission",
        "full_success": False,
        "submissions": [
            {
                "submission_id": "fixture-submission",
                "sequence": 1,
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_size_bytes": len(payload),
                "dispatch_state": "returned",
                "status": "completed",
                "grade": 0.25,
                "behavior_outcome": "partial",
                "feedback": "Exact fixture feedback",
                "error_code": None,
                "raw_evidence": {"fixture": True},
                "evidence_completeness": "complete",
                "acceptance": "unknown",
                "observed_request_id": None,
                "observed_receipt_id": None,
                "remote_disposition": "completed",
                "cleanup_status": "not_required",
            }
        ],
        "capabilities": {"remote_query": False, "remote_cancel": False, "idempotency": False},
    }


def _retained() -> RetainedSubmissionReportV2:
    return RetainedSubmissionReportV2(
        mode="offline",
        simulated=True,
        run_id="OFFLINE-v2-run",
        conversation_id="OFFLINE-v2-conversation",
        status="completed",
        report=StrictSubmissionReportV2.model_validate(_projection()),
        calls=(),
        final_text=None,
        runner_error=None,
        local_cleanup="not_required",
    )


@pytest.mark.parametrize(("mode", "simulated"), [("real", True), ("offline", False), ("real", "false"), ("offline", 1)])
def test_v2_rejects_inconsistent_or_coerced_provenance(*, mode: str, simulated: object) -> None:
    with pytest.raises(ValueError):
        SubmissionProvenanceV2(mode=mode, simulated=simulated)


def test_v1_literals_remain_offline_only_and_reject_v2() -> None:
    with pytest.raises(ValueError):
        StrictSubmissionReport.model_validate(_projection())
    with pytest.raises(ValueError):
        RetainedSubmissionReport.model_validate(_retained().model_dump(mode="json"))
    legacy = _projection(mode="real", simulated=False)
    legacy["contract_version"] = "strict-submission-v1"
    with pytest.raises(ValueError):
        StrictSubmissionReport.model_validate(legacy)


def test_real_provenance_is_validated_without_invoking_or_persisting_any_provider() -> None:
    projection = StrictSubmissionReportV2.model_validate(_projection(mode="real", simulated=False))
    report = RetainedSubmissionReportV2(
        mode="real",
        simulated=False,
        run_id="schema-only-fixture",
        conversation_id="schema-only-fixture",
        status="completed",
        report=projection,
        calls=(),
        final_text=None,
        runner_error=None,
        local_cleanup="complete",
    )
    assert report.evidence_label == "REAL/UNSIMULATED"
    assert report.report.mode.value == "real"
    assert report.simulated is False
    assert "OFFLINE/SIMULATED" not in report.canonical_json()
    with pytest.raises(ValueError, match="label"):
        RetainedSubmissionReportV2.model_validate(
            {**report.model_dump(mode="json"), "evidence_label": "OFFLINE/SIMULATED"}
        )
    with pytest.raises(ValueError, match="provenance"):
        RetainedSubmissionReportV2.model_validate(
            {
                **report.model_dump(mode="json"),
                "mode": "offline",
                "simulated": True,
                "evidence_label": "OFFLINE/SIMULATED",
            }
        )


async def test_v2_content_score_preserves_provenance_and_replay_is_read_only_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    report = _retained()
    scorer = SubmissionReportScorerV2(report_sha256=report.sha256())
    checkpoint = []
    asyncio.get_running_loop().call_soon(checkpoint.append, "yielded")
    score = (await scorer.score_async(scorable=ContentScorable(value=report.canonical_json())))[0]
    assert not checkpoint
    assert score.get_value() == 0.25 and isinstance(score.scorable, ContentEntryScorable)
    assert score.score_metadata["contract_version"] == "strict-submission-v2"
    assert score.score_metadata["mode"] == "offline" and score.score_metadata["simulated"] == 1
    assert score.score_metadata["publication_boundary"] == "pyrit_score_commit"
    assert score.message_piece_id is None and score.observation_ids == []
    stored = sqlite_instance.get_scores(score_ids=[str(score.id)])[0]
    replay = (await scorer.score_async(scorable=stored.scorable))[0]
    assert replay.get_value() == 0.25
    assert scorer.get_identifier().params["contract"] == "strict-submission-v2"


async def test_v2_scorer_rejects_tampering_and_v1_does_not_accept_v2_async() -> None:
    report = _retained()
    altered = copy.deepcopy(report.model_dump(mode="json"))
    altered["report"]["submissions"][0]["artifact_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="modified"):
        await SubmissionReportScorerV2(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=json.dumps(altered))
        )
    with pytest.raises(RuntimeError):
        await SubmissionReportScorer(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json())
        )


@pytest.mark.parametrize(
    ("acceptance", "disposition"),
    [
        ("not_dispatched", "not_dispatched"),
        ("rejected", "completed"),
        ("accepted", "cancelled"),
    ],
)
def test_v2_uses_existing_coherent_attempt_validation(*, acceptance: str, disposition: str) -> None:
    raw = _projection()
    raw["submissions"][0].update(acceptance=acceptance, remote_disposition=disposition)
    with pytest.raises(ValueError):
        StrictSubmissionReportV2.model_validate(raw)


async def test_v2_incomplete_cleanup_or_outcome_does_not_project_prior_grade_async() -> None:
    raw = _retained().model_dump(mode="json")
    raw.update(status="incomplete", local_cleanup="unknown")
    report = RetainedSubmissionReportV2.model_validate(raw)
    score = (
        await SubmissionReportScorerV2(report_sha256=report.sha256()).score_async(
            scorable=ContentScorable(value=report.canonical_json())
        )
    )[0]
    assert score.is_undetermined and score.score_metadata["retained_last_valid_grade"] == 0.25
    with pytest.raises(ValueError, match="cleanup"):
        RetainedSubmissionReportV2.model_validate({**raw, "status": "completed"})
