# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Native typed original ModelCall correlation, including authentic cached-prefix usage."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import pytest
from inspect_ai.event import ModelEvent, SubtaskEvent
from inspect_ai.model import GenerateConfig, ModelCall, ModelOutput, ModelUsage

from pyrit.backend.services.original_model_receipt import OriginalModelReceipt
from pyrit.backend.services.original_worker_preflight import CohostPreflightError
from pyrit.models import config_hash

if TYPE_CHECKING:
    from pydantic import JsonValue


def _model_event(*, cached: int = 0) -> tuple[ModelEvent, dict[str, JsonValue]]:
    request = {"model": "gpt-4-32", "messages": [{"role": "user", "content": "Public typed synthetic"}]}
    response = {
        "id": "public-response",
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 2,
            "total_tokens": 10,
            "prompt_tokens_details": {"cached_tokens": cached},
        },
    }
    event = ModelEvent(
        model="gpt-4o",
        input=[],
        tools=[],
        tool_choice="auto",
        config=GenerateConfig(),
        output=ModelOutput(
            model="gpt-4o",
            usage=ModelUsage(
                input_tokens=8 - cached,
                output_tokens=2,
                total_tokens=10,
                input_tokens_cache_read=cached or None,
            ),
        ),
        call=ModelCall(
            request={**request, "extra_headers": {"x-irid": "public-inspect-one"}, "tools": None, "tool_choice": None},
            response=response,
        ),
    )
    row: dict[str, JsonValue] = {
        "inspect_request_id": "public-inspect-one",
        "response_id": "public-response",
        "request_payload_sha256": config_hash({"request": request}),
        "prompt_tokens": 8,
        "completion_tokens": 2,
        "total_tokens": 10,
        "upstream_usage_total_tokens": 10,
        "cached_prompt_tokens": cached,
    }
    return event, row


@pytest.mark.parametrize("cached", [0, 3, 8])
def test_typed_call_matches_exact_incoming_request_ids_and_authentic_usage(cached: int) -> None:
    event, row = _model_event(cached=cached)
    OriginalModelReceipt.match_events(events=[event], rows=[row])


@pytest.mark.parametrize(
    "changed",
    [
        "pending",
        "error",
        "cache_read",
        "call_missing",
        "request_hash",
        "request_id",
        "response_id",
        "raw_usage",
        "framework_usage",
        "cache_write",
        "cached_mismatch",
        "duplicate_event",
        "duplicate_row",
        "missing_event",
        "missing_row",
    ],
)
def test_typed_model_match_rejects_nonbijective_pending_or_inconsistent_source(changed: str) -> None:
    event, row = _model_event(cached=3)
    events, rows = [event], [row]
    assert event.call is not None and event.output.usage is not None
    if changed == "pending":
        event.pending = True
    elif changed == "error":
        event.call.error = "Public synthetic error"
    elif changed == "cache_read":
        event.cache = "read"
    elif changed == "call_missing":
        event.call = None
    elif changed == "request_hash":
        row["request_payload_sha256"] = "0" * 64
    elif changed == "request_id":
        row["inspect_request_id"] = "other-request"
    elif changed == "response_id":
        row["response_id"] = "other-response"
    elif changed == "raw_usage":
        row["prompt_tokens"] = 9
    elif changed == "framework_usage":
        event.output.usage.input_tokens = 8
    elif changed == "cache_write":
        event.output.usage.input_tokens_cache_write = 1
    elif changed == "cached_mismatch":
        row["cached_prompt_tokens"] = 1
    elif changed == "duplicate_event":
        events.append(event.model_copy(deep=True))
    elif changed == "duplicate_row":
        rows.append(copy.deepcopy(row))
    elif changed == "missing_event":
        events.clear()
    elif changed == "missing_row":
        rows.clear()
    with pytest.raises(CohostPreflightError):
        OriginalModelReceipt.match_events(events=events, rows=rows)


def test_nested_legacy_subtask_events_are_not_omitted() -> None:
    event, row = _model_event()
    nested = SubtaskEvent(name="public-solver", type="task", input={}, events=[event.model_dump(mode="json")])
    from unittest.mock import MagicMock

    from inspect_ai.log import EvalLog, EvalSample

    log = MagicMock(spec=EvalLog)
    sample = MagicMock(spec=EvalSample)
    sample.events = [nested]
    log.samples = [sample]
    events = OriginalModelReceipt._model_events(log)
    assert len(events) == 1 and isinstance(events[0], ModelEvent)
    OriginalModelReceipt.match_events(events=events, rows=[row])
