# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Schema1 content replay stays exact as schema2 adds source-event provenance."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest

from pyrit.models.inspect_ghcp import InspectGhcpReport, InspectGhcpStatus


@pytest.mark.parametrize("schema_version", [1, 2])
def test_legacy_canonical_report_replays_without_added_fields(schema_version: int) -> None:
    now = datetime.now(UTC)
    initial = InspectGhcpReport(
        schema_version=schema_version,
        run_id="historical-run",
        task_name="original_task",
        task_version="1",
        sample_id="sample-one",
        cli_sha256="a" * 64,
        model_id="qwen3-local",
        wire_model="qwen3:1.7b",
        started_at=now,
        ended_at=now + timedelta(seconds=1),
        turn_count=0,
        sdk_event_count=0,
        model_request_count=0,
        model_http_200_count=0,
        host_model_request_count=0,
        host_model_http_200_count=0,
        adversarial_request_count=0,
        adversarial_http_200_count=0,
        tool_start_count=0,
        tool_complete_count=0,
        status=InspectGhcpStatus.INCOMPLETE,
    )
    historical = initial.model_dump(mode="json")
    historical.pop("control_receipt_sha256")
    historical.pop("token_files_absent_before_turn")
    if schema_version == 1:
        historical.pop("sample_epoch")
        historical.pop("successful_tool_execution_count")
        historical.pop("phase_durations_seconds")
        historical.pop("gateway_alive_before_scoring")
        historical.pop("gateway_alive_after_scoring")
    text = json.dumps(historical, sort_keys=True, separators=(",", ":"), allow_nan=False)

    replay = InspectGhcpReport.model_validate_json(text)
    assert replay.schema_version == schema_version
    assert replay.canonical_json() == text
    assert replay.sha256() == hashlib.sha256(text.encode("utf-8")).hexdigest()
