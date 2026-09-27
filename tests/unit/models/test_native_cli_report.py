# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CLI-only report validation without a provider, database, or GHCP event shape."""

from __future__ import annotations

import hashlib
import json
import math

import pytest

from pyrit.models.native_cli_report import (
    NativeCliArtifactReference,
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


def _complete_report() -> NativeCliRunReport:
    events = (
        NativeCliReportEvent(
            sequence=1,
            frame_number=1,
            raw_frame_sha256="a" * 64,
            raw_frame_size_bytes=50,
            stdout_offset_bytes=0,
            kind=NativeCliReportEventKind.SESSION_STARTED,
            status=NativeCliReportEventStatus.UNKNOWN,
            source_session_id="codex-thread",
            source_event_id="codex-thread",
        ),
        NativeCliReportEvent(
            sequence=2,
            frame_number=2,
            raw_frame_sha256="b" * 64,
            raw_frame_size_bytes=70,
            stdout_offset_bytes=50,
            kind=NativeCliReportEventKind.TURN_STARTED,
            status=NativeCliReportEventStatus.RUNNING,
        ),
        NativeCliReportEvent(
            sequence=3,
            frame_number=3,
            raw_frame_sha256="c" * 64,
            raw_frame_size_bytes=80,
            stdout_offset_bytes=120,
            kind=NativeCliReportEventKind.MODEL_MESSAGE,
            status=NativeCliReportEventStatus.UNKNOWN,
            source_event_id="msg-3",
        ),
        NativeCliReportEvent(
            sequence=4,
            frame_number=4,
            raw_frame_sha256="d" * 64,
            raw_frame_size_bytes=90,
            stdout_offset_bytes=200,
            kind=NativeCliReportEventKind.TURN_COMPLETED,
            status=NativeCliReportEventStatus.COMPLETED,
        ),
        NativeCliReportEvent(
            sequence=5,
            kind=NativeCliReportEventKind.EOF,
            status=NativeCliReportEventStatus.UNKNOWN,
            detail="Stdout reached EOF.",
        ),
    )
    return NativeCliRunReport(
        schema_version=1,
        task_id="fixture-task",
        task_version="benchmark-v1",
        run_id="fixture-run",
        turn_id="fixture-turn",
        turn_index=1,
        protocol=NativeCliReportProtocol.CODEX_EXEC_JSON,
        cli_version="0.115.0",
        cli_profile="locked-workspace",
        max_steps=4,
        simulated=True,
        status=NativeCliReportStatus.COMPLETED,
        evidence=NativeCliReportEvidence(
            source_session_id="codex-thread",
            exit_code=0,
            terminal_observed=True,
            coverage_complete=True,
            observed_steps=1,
            frame_count=4,
            raw_chunk_count=2,
            raw_stdout_bytes=290,
            raw_stderr_bytes=14,
            gaps=(),
            events=events,
            raw_evidence_ref="fixture-raw-chunks",
        ),
        judgment=NativeCliOriginalJudgment(
            grader_ref="fixture-original-grader",
            grader_evidence_ref="fixture-original-feedback",
            value=0.75,
            complete=True,
            rationale="Original grader evaluated the fixture artifact.",
        ),
        artifacts=(
            NativeCliArtifactReference(
                name="fixture.txt",
                sha256=hashlib.sha256(b"OFFLINE/SIMULATED artifact").hexdigest(),
                size_bytes=len(b"OFFLINE/SIMULATED artifact"),
                evidence_ref="fixture-retained-artifact",
            ),
        ),
        cleanup=NativeCliReportCleanup.CLOSED,
    )


def test_canonical_json_and_sha256_are_deterministic_on_real_model_round_trip() -> None:
    report = _complete_report()
    canonical = report.canonical_json()
    assert canonical == json.dumps(report.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    assert report.sha256() == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert NativeCliRunReport.model_validate_json(canonical).canonical_json() == canonical
    reversed_fields = dict(reversed(list(report.model_dump(mode="json").items())))
    assert NativeCliRunReport.model_validate(reversed_fields).sha256() == report.sha256()
    assert report.evidence.events[-1].source_event_id is None
    assert report.evidence.events[2].source_event_id == "msg-3"
    changed = report.model_dump(mode="json")
    changed["judgment"]["rationale"] = "Different original feedback"
    assert NativeCliRunReport.model_validate(changed).sha256() != report.sha256()


@pytest.mark.parametrize("version", [None, "", "  ", 123, True])
def test_task_version_is_required_for_every_report_status(version: object) -> None:
    for status in (NativeCliReportStatus.COMPLETED, NativeCliReportStatus.ERROR):
        data = _complete_report().model_dump(mode="json")
        data["status"] = status.value
        if status is NativeCliReportStatus.ERROR:
            data["errors"] = ["Original grader error."]
        if version is None:
            del data["task_version"]
        else:
            data["task_version"] = version
        with pytest.raises(ValueError):
            NativeCliRunReport.model_validate(data)


def test_task_revision_alone_changes_canonical_identity_not_cli_version() -> None:
    report = _complete_report()
    data = report.model_dump(mode="json")
    data["task_version"] = "benchmark-v2"
    revised = NativeCliRunReport.model_validate(data)
    assert revised.task_id == report.task_id == "fixture-task"
    assert revised.cli_version == report.cli_version == "0.115.0"
    assert revised.task_version == "benchmark-v2"
    assert revised.sha256() != report.sha256()
    assert '"task_version":"benchmark-v2"' in revised.canonical_json()


@pytest.mark.parametrize("version", [None, True, 0, 2, "1", 1.0])
def test_schema_version_requires_explicit_exact_integer_one(version: object) -> None:
    data = _complete_report().model_dump(mode="json")
    if version is None:
        del data["schema_version"]
    else:
        data["schema_version"] = version
    with pytest.raises(ValueError, match="schema_version|Field required"):
        NativeCliRunReport.model_validate(data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol",), "copilot_sdk"),
        (("evidence", "events", 2, "kind"), "tool.execution_start"),
        (("evidence", "events", 2, "status"), "succeeded"),
        (("evidence", "events", 2, "source_event_id"), True),
    ],
)
def test_unknown_provider_shapes_or_source_ids_cannot_be_silently_reinterpreted(
    path: tuple[str | int, ...], value: object
) -> None:
    data = _complete_report().model_dump(mode="json")
    if path == ("protocol",):
        data["protocol"] = value
    else:
        data["evidence"]["events"][path[2]][path[3]] = value
    with pytest.raises(ValueError):
        NativeCliRunReport.model_validate(data)


@pytest.mark.parametrize("value", [True, "0.75", math.nan, math.inf, -0.1, 1.1])
def test_original_grader_rejects_untrusted_or_out_of_range_numeric_values(value: object) -> None:
    with pytest.raises(ValueError, match="original grader value|Original grader value"):
        NativeCliOriginalJudgment(
            grader_ref="fixture-grader",
            grader_evidence_ref="fixture-feedback",
            complete=True,
            rationale="Original fixture rationale",
            value=value,
        )


def test_original_grader_distinguishes_zero_from_missing_verdict() -> None:
    complete = NativeCliOriginalJudgment(
        grader_ref="fixture-grader",
        grader_evidence_ref="fixture-feedback",
        complete=True,
        rationale="Original grader measured zero.",
        value=0,
    )
    assert complete.value == 0.0
    unknown = NativeCliOriginalJudgment(grader_ref="fixture-grader", complete=False, rationale="No original verdict.")
    assert unknown.value is None
    with pytest.raises(ValueError, match="Only a complete"):
        NativeCliOriginalJudgment(grader_ref="fixture-grader", complete=False, rationale="Not complete", value=0)
    with pytest.raises(ValueError, match="grader identity and rationale"):
        NativeCliOriginalJudgment(grader_ref="fixture-grader", complete=True, rationale="", value=0.5)
    with pytest.raises(ValueError, match="retained grader evidence"):
        NativeCliOriginalJudgment(grader_ref="fixture-grader", complete=True, rationale="Observed", value=0.5)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evidence", "raw_evidence_ref", " "),
        ("evidence", "source_session_id", " "),
        ("report", "parent_run_id", " "),
        ("report", "conversation_id", " "),
        ("judgment", "grader_ref", " "),
        ("judgment", "grader_evidence_ref", " "),
        ("artifact", "evidence_ref", " "),
    ],
)
def test_blank_lineage_and_evidence_references_are_not_valid_completed_evidence(
    section: str, field: str, value: object
) -> None:
    data = _complete_report().model_dump(mode="json")
    if section == "evidence":
        data["evidence"][field] = value
    elif section == "judgment":
        data["judgment"][field] = value
    elif section == "artifact":
        data["artifacts"][0][field] = value
    else:
        data[field] = value
    with pytest.raises(ValueError):
        NativeCliRunReport.model_validate(data)


def test_blank_source_identifier_or_partial_detail_cannot_be_claimed_as_observed() -> None:
    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][2]["source_event_id"] = " "
    with pytest.raises(ValueError, match="source identifiers"):
        NativeCliRunReport.model_validate(data)
    with pytest.raises(ValueError, match="coverage gap"):
        NativeCliReportEvent(
            sequence=1,
            kind=NativeCliReportEventKind.PARTIAL,
            status=NativeCliReportEventStatus.UNKNOWN,
            detail="  ",
        )


def test_model_and_tool_provenance_cannot_be_synthesized_from_missing_source_ids() -> None:
    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][2]["source_event_id"] = None
    with pytest.raises(ValueError, match="observed source event ID"):
        NativeCliRunReport.model_validate(data)
    with pytest.raises(ValueError, match="observed source session ID"):
        NativeCliReportEvent(
            sequence=1,
            kind=NativeCliReportEventKind.SESSION_STARTED,
            status=NativeCliReportEventStatus.UNKNOWN,
        )
    with pytest.raises(ValueError, match="observed source tool ID"):
        NativeCliReportEvent(
            sequence=1,
            kind=NativeCliReportEventKind.TOOL_RESULT,
            status=NativeCliReportEventStatus.COMPLETED,
            source_event_id="user-1",
        )


def test_provider_event_cannot_drop_all_frame_evidence_or_label_a_frame_as_eof() -> None:
    with pytest.raises(ValueError, match="requires its actual JSONL frame"):
        NativeCliReportEvent(
            sequence=1,
            kind=NativeCliReportEventKind.MODEL_MESSAGE,
            status=NativeCliReportEventStatus.UNKNOWN,
            source_event_id="item-1",
        )
    with pytest.raises(ValueError, match="synthetic boundary"):
        NativeCliReportEvent(
            sequence=1,
            kind=NativeCliReportEventKind.EOF,
            status=NativeCliReportEventStatus.UNKNOWN,
            frame_number=1,
            raw_frame_sha256="a" * 64,
            raw_frame_size_bytes=12,
            stdout_offset_bytes=0,
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evidence", "coverage_complete", False),
        ("evidence", "exit_code", 9),
        ("evidence", "source_session_id", None),
        ("evidence", "raw_evidence_ref", None),
        ("report", "cleanup", "unknown"),
        ("report", "simulated", None),
        ("report", "judgment", None),
        ("report", "errors", ["Original grader uncertainty."]),
    ],
)
def test_completed_report_requires_all_independent_boundaries(section: str, field: str, value: object) -> None:
    data = _complete_report().model_dump(mode="json")
    if section == "report":
        data[field] = value
    else:
        data["evidence"][field] = value
    with pytest.raises(ValueError):
        NativeCliRunReport.model_validate(data)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("evidence", "events", []),
        ("evidence", "gaps", ["missing chunk"]),
        ("evidence", "raw_stdout_bytes", -1),
        ("evidence", "raw_stdout_bytes", 0),
        ("evidence", "raw_chunk_count", 0),
        ("evidence", "frame_count", True),
        ("report", "turn_index", 0),
        ("report", "max_steps", 0),
        ("report", "cli_version", "latest"),
    ],
)
def test_invalid_event_and_counter_shapes_cannot_be_completed(section: str, field: str, value: object) -> None:
    data = _complete_report().model_dump(mode="json")
    if section == "report":
        data[field] = value
    else:
        data["evidence"][field] = value
    with pytest.raises(ValueError):
        NativeCliRunReport.model_validate(data)


def test_source_session_and_event_order_are_owned_by_observed_stream() -> None:
    data = _complete_report().model_dump(mode="json")
    data["evidence"]["source_session_id"] = "made-up-session"
    with pytest.raises(ValueError, match="source session ID"):
        NativeCliRunReport.model_validate(data)

    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][2]["sequence"] = 42
    with pytest.raises(ValueError, match="sequence"):
        NativeCliRunReport.model_validate(data)

    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][3]["frame_number"] = 8
    with pytest.raises(ValueError, match="unobserved stdout frame"):
        NativeCliRunReport.model_validate(data)

    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][2]["raw_frame_sha256"] = None
    with pytest.raises(ValueError, match="frame number, digest, size, and stdout offset"):
        NativeCliRunReport.model_validate(data)

    data = _complete_report().model_dump(mode="json")
    data["evidence"]["frame_count"] = 5
    with pytest.raises(ValueError, match="every stdout frame"):
        NativeCliRunReport.model_validate(data)

    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][2]["frame_number"] = 4
    with pytest.raises(ValueError, match="same CLI stdout frame"):
        NativeCliRunReport.model_validate(data)


def test_complete_evidence_enforces_protocol_specific_terminal_and_step_count() -> None:
    data = _complete_report().model_dump(mode="json")
    data["evidence"]["observed_steps"] = 0
    with pytest.raises(ValueError, match="Complete Codex evidence"):
        NativeCliRunReport.model_validate(data)
    data = _complete_report().model_dump(mode="json")
    data["protocol"] = NativeCliReportProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE.value
    with pytest.raises(ValueError, match="Complete Claude"):
        NativeCliRunReport.model_validate(data)
    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][3]["status"] = NativeCliReportEventStatus.UNKNOWN.value
    with pytest.raises(ValueError, match="Complete Codex"):
        NativeCliRunReport.model_validate(data)
    data = _complete_report().model_dump(mode="json")
    data["evidence"]["events"][2]["kind"] = NativeCliReportEventKind.TURN_COMPLETED.value
    data["evidence"]["events"][2]["status"] = NativeCliReportEventStatus.COMPLETED.value
    data["evidence"]["events"][3]["kind"] = NativeCliReportEventKind.AUXILIARY.value
    with pytest.raises(ValueError, match="Complete Codex"):
        NativeCliRunReport.model_validate(data)


@pytest.mark.parametrize("status", [NativeCliReportStatus.ERROR, NativeCliReportStatus.CANCELLED])
def test_uncertain_run_retains_real_grader_value_but_requires_error_reason(status: NativeCliReportStatus) -> None:
    data = _complete_report().model_dump(mode="json")
    data["status"] = status.value
    data["cleanup"] = NativeCliReportCleanup.UNKNOWN.value
    data["errors"] = ["No verified environment cleanup."]
    report = NativeCliRunReport.model_validate(data)
    assert report.judgment is not None and report.judgment.value == 0.75
    assert report.status is status
    data["errors"] = []
    with pytest.raises(ValueError, match="reason"):
        NativeCliRunReport.model_validate(data)


def test_missing_process_outcome_and_cancellation_preserve_known_frames_but_not_counts() -> None:
    data = _complete_report().model_dump(mode="json")
    data["status"] = NativeCliReportStatus.CANCELLED.value
    data["cleanup"] = NativeCliReportCleanup.UNKNOWN.value
    data["errors"] = ["Caller cancelled the sandbox run."]
    evidence = data["evidence"]
    evidence.update(
        exit_code=None,
        terminal_observed=False,
        coverage_complete=False,
        observed_steps=None,
        frame_count=None,
        raw_chunk_count=None,
        raw_stdout_bytes=None,
        raw_stderr_bytes=None,
        gaps=["No native CLI process outcome was acquired."],
    )
    report = NativeCliRunReport.model_validate(data)
    assert report.evidence.exit_code is None
    assert report.evidence.events[0].frame_number == 1
    assert report.evidence.events[0].source_session_id == "codex-thread"
    assert report.evidence.observed_steps is None


def test_prelaunch_error_is_not_a_fabricated_cli_session() -> None:
    data = _complete_report().model_dump(mode="json")
    data["status"] = NativeCliReportStatus.ERROR.value
    data["cleanup"] = NativeCliReportCleanup.NOT_OPENED.value
    data["errors"] = ["CLI launcher rejected the request."]
    data["evidence"].update(
        source_session_id=None,
        exit_code=None,
        terminal_observed=False,
        coverage_complete=False,
        observed_steps=None,
        frame_count=None,
        raw_chunk_count=None,
        raw_stdout_bytes=None,
        raw_stderr_bytes=None,
        gaps=["No native CLI process outcome was acquired."],
        raw_evidence_ref=None,
        events=[
            NativeCliReportEvent(
                sequence=1,
                kind=NativeCliReportEventKind.ERROR,
                status=NativeCliReportEventStatus.FAILED,
                detail="Launcher rejected.",
            ).model_dump(mode="json")
        ],
    )
    report = NativeCliRunReport.model_validate(data)
    assert report.evidence.source_session_id is None and report.evidence.exit_code is None
    assert report.cleanup is NativeCliReportCleanup.NOT_OPENED


def test_turn_parent_and_artifact_references_cannot_be_ambiguous() -> None:
    data = _complete_report().model_dump(mode="json")
    data["turn_index"] = 2
    with pytest.raises(ValueError, match="parent run ID"):
        NativeCliRunReport.model_validate(data)
    data["parent_run_id"] = "fixture-prior-run"
    assert NativeCliRunReport.model_validate(data).turn_index == 2
    data["parent_run_id"] = data["run_id"]
    with pytest.raises(ValueError, match="parent run ID"):
        NativeCliRunReport.model_validate(data)

    data = _complete_report().model_dump(mode="json")
    data["artifacts"].append(data["artifacts"][0])
    with pytest.raises(ValueError, match="cannot be duplicated"):
        NativeCliRunReport.model_validate(data)
    data["artifacts"][1]["sha256"] = "not a digest"
    with pytest.raises(ValueError):
        NativeCliRunReport.model_validate(data)
