# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pure, unpersisted float Score projection from a real CLI report instance."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pyrit.memory import CentralMemory
from pyrit.models import ContentScorable, Score, ScoreStatus, ScoringExpectation, UndeterminedScoreError
from pyrit.models.native_cli_report import (
    NativeCliOriginalJudgment,
    NativeCliReportCleanup,
    NativeCliReportEvent,
    NativeCliReportEventKind,
    NativeCliReportEventStatus,
    NativeCliReportEvidence,
    NativeCliReportProtocol,
    NativeCliReportStatus,
    NativeCliRunReport,
)
from pyrit.score.float_scale.native_cli_report_scorer import (
    NativeCliReportScoreBuilder,
    build_native_cli_report_score,
)

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _report(*, value: float = 0.75) -> NativeCliRunReport:
    return NativeCliRunReport(
        schema_version=1,
        task_id="OFFLINE-task",
        run_id="OFFLINE-run",
        turn_id="OFFLINE-turn",
        turn_index=1,
        protocol=NativeCliReportProtocol.CODEX_EXEC_JSON,
        cli_version="0.115.0",
        cli_profile="sandbox-locked",
        max_steps=3,
        simulated=True,
        status=NativeCliReportStatus.COMPLETED,
        evidence=NativeCliReportEvidence(
            source_session_id="thread-observed",
            exit_code=0,
            terminal_observed=True,
            coverage_complete=True,
            observed_steps=1,
            frame_count=3,
            raw_chunk_count=1,
            raw_stdout_bytes=300,
            raw_stderr_bytes=0,
            raw_evidence_ref="OFFLINE raw evidence",
            events=(
                NativeCliReportEvent(
                    sequence=1,
                    kind=NativeCliReportEventKind.SESSION_STARTED,
                    status=NativeCliReportEventStatus.UNKNOWN,
                    source_event_id="thread-observed",
                    source_session_id="thread-observed",
                    frame_number=1,
                    raw_frame_sha256="a" * 64,
                    raw_frame_size_bytes=100,
                    stdout_offset_bytes=0,
                ),
                NativeCliReportEvent(
                    sequence=2,
                    kind=NativeCliReportEventKind.TURN_STARTED,
                    status=NativeCliReportEventStatus.RUNNING,
                    frame_number=2,
                    raw_frame_sha256="b" * 64,
                    raw_frame_size_bytes=100,
                    stdout_offset_bytes=100,
                ),
                NativeCliReportEvent(
                    sequence=3,
                    kind=NativeCliReportEventKind.TURN_COMPLETED,
                    status=NativeCliReportEventStatus.COMPLETED,
                    frame_number=3,
                    raw_frame_sha256="c" * 64,
                    raw_frame_size_bytes=100,
                    stdout_offset_bytes=200,
                ),
                NativeCliReportEvent(
                    sequence=4,
                    kind=NativeCliReportEventKind.EOF,
                    status=NativeCliReportEventStatus.UNKNOWN,
                    detail="Stdout reached EOF.",
                ),
            ),
        ),
        judgment=NativeCliOriginalJudgment(
            grader_ref="OFFLINE original grader",
            grader_evidence_ref="OFFLINE original feedback",
            complete=True,
            value=value,
            rationale="Original grader judged the retained fixture.",
        ),
        cleanup=NativeCliReportCleanup.CLOSED,
    )


@pytest.mark.parametrize("value", [0.0, 0.75, 1.0])
def test_complete_report_produces_unpersisted_content_score_from_original_grader(
    value: float,
    sqlite_instance: SQLiteMemory,
) -> None:
    report = _report(value=value)
    expectation = ScoringExpectation(objective="Grade the fixture task")
    with patch.object(CentralMemory, "get_memory_instance", side_effect=AssertionError("memory access")) as memory:
        score = build_native_cli_report_score(report=report, expectation=expectation)
    memory.assert_not_called()
    assert score.status is ScoreStatus.COMPLETE and score.get_value() == value
    assert score.score_value == str(value)
    assert score.score_rationale == "Original grader judged the retained fixture."
    assert isinstance(score.scorable, ContentScorable)
    assert score.scorable.value == report.canonical_json()
    assert score.scorable.data_type == "text"
    assert score.scored_expectation == expectation and score.objective == expectation.objective
    assert score.message_piece_id is None and score.observation_ids == []
    assert score.score_metadata["report_sha256"] == report.sha256()
    assert score.score_metadata["publication_state"] == "unpersisted_candidate"
    assert score.score_metadata["run_id"] == "OFFLINE-run"
    assert score.score_metadata["turn_id"] == "OFFLINE-turn"
    assert score.score_metadata["cli_protocol"] == "codex_exec_json"
    assert score.scorer_class_identifier is not None
    assert score.scorer_class_identifier.class_name == "NativeCliReportScoreBuilder"
    assert score.scorer_class_identifier.params["report_sha256"] == report.sha256()
    assert score.scorer_class_identifier.params["contract"] == "native-cli-run-v1"
    restored = Score.model_validate(score.model_dump(mode="json"))
    assert restored.scorable == score.scorable
    assert restored.scorer_class_identifier == score.scorer_class_identifier
    assert restored.get_value() == value
    assert sqlite_instance.get_scores(score_type="float_scale") == []


@pytest.mark.parametrize("status", list(NativeCliReportStatus)[1:])
def test_noncompleted_runs_cannot_project_retained_numeric_grade(status: NativeCliReportStatus) -> None:
    report = _report().model_dump(mode="json")
    report["status"] = status.value
    report["cleanup"] = NativeCliReportCleanup.UNKNOWN.value
    if status in {NativeCliReportStatus.ERROR, NativeCliReportStatus.CANCELLED}:
        report["errors"] = ["Original run did not finalize."]
    retained = NativeCliRunReport.model_validate(report)
    score = build_native_cli_report_score(report=retained)
    assert retained.judgment is not None and retained.judgment.value == 0.75
    assert score.status is ScoreStatus.UNDETERMINED and score.score_value is None
    assert score.score_metadata["run_status"] == status.value
    assert "0.75" not in str(score.score_metadata)
    with pytest.raises(UndeterminedScoreError):
        score.get_value()


def test_missing_cli_outcome_or_judgment_produces_undetermined_content_score() -> None:
    raw = _report().model_dump(mode="json")
    raw.update(
        status=NativeCliReportStatus.INCOMPLETE.value, judgment=None, cleanup=NativeCliReportCleanup.UNKNOWN.value
    )
    raw["evidence"].update(
        source_session_id=None,
        exit_code=None,
        terminal_observed=False,
        coverage_complete=False,
        observed_steps=None,
        frame_count=None,
        raw_chunk_count=None,
        raw_stdout_bytes=None,
        raw_stderr_bytes=None,
        raw_evidence_ref=None,
        gaps=["No process outcome was acquired."],
        events=[],
    )
    report = NativeCliRunReport.model_validate(raw)
    score = build_native_cli_report_score(report=report)
    assert score.is_undetermined
    assert score.score_value is None and score.score_metadata["exit_code"] == "unobserved"
    assert score.score_metadata["observed_steps"] == "unobserved"
    assert score.scorable == ContentScorable(value=report.canonical_json())


def test_builder_revalidates_copied_or_tampered_report_before_projecting_value() -> None:
    report = _report()
    invalid = report.model_copy(
        update={"status": NativeCliReportStatus.COMPLETED, "cleanup": NativeCliReportCleanup.UNKNOWN}
    )
    with pytest.raises(ValueError, match="Completed CLI results"):
        build_native_cli_report_score(report=invalid)
    invalid = report.model_copy(update={"schema_version": 2})
    with pytest.raises(ValueError, match="schema_version"):
        build_native_cli_report_score(report=invalid)
    with pytest.raises(TypeError, match="CLI report"):
        build_native_cli_report_score(report="assistant text")
    with pytest.raises(TypeError, match="ScoringExpectation"):
        build_native_cli_report_score(report=report, expectation="not a validated expectation")


def test_canonical_snapshot_and_identifier_are_stable_without_persisting() -> None:
    report = _report()
    first = NativeCliReportScoreBuilder().build_score(report=report)
    loaded = NativeCliRunReport.model_validate_json(report.canonical_json())
    second = build_native_cli_report_score(report=loaded)
    assert first.scorable == second.scorable
    assert first.scorer_class_identifier == second.scorer_class_identifier
    assert first.score_metadata["report_sha256"] == second.score_metadata["report_sha256"]
    assert first.score_metadata["report_sha256"] == hashlib.sha256(report.canonical_json().encode()).hexdigest()
