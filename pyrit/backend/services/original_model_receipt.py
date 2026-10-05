# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Independent typed original ModelCall/relay bijection without another scorer or SDK hook."""

from __future__ import annotations

import base64
import hashlib
import io
from typing import TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter

from pyrit.backend.services.original_worker_preflight import CohostPreflightError
from pyrit.models import config_hash

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from inspect_ai.event import ModelEvent
    from inspect_ai.log import EvalLog


class OriginalModelReceipt:
    """Bind exact native role evidence to unmodified framework-generated x-irid and original usage."""

    @classmethod
    def verify(cls, *, archive: bytes, role: dict[str, JsonValue], control_id: UUID, require_success: bool) -> None:
        """Verify source-derived counts/hashes, never a retained-fixture message count or grade."""
        if (
            role.get("run_id") != str(control_id)
            or role.get("role") != "evaluated"
            or role.get("capability_revoked") is not True
            or role.get("upstream_drained") is not True
            or role.get("active_requests") != 0
            or (require_success and role.get("usage_complete") is not True)
        ):
            raise CohostPreflightError("Original model closure is not its exact drained evaluated role.")
        encoded = role.get("model_request_evidence_jsonl_b64")
        if not isinstance(encoded, str) or len(encoded) > 262_144:
            raise CohostPreflightError("Original model receipt evidence exceeds its fixed bound.")
        content = base64.b64decode(encoded, validate=True)
        if hashlib.sha256(content).hexdigest() != role.get("model_request_evidence_sha256") or len(content) != role.get(
            "model_request_evidence_bytes"
        ):
            raise CohostPreflightError("Original native model receipt bytes differ.")
        rows = [TypeAdapter(dict[str, JsonValue]).validate_json(line, strict=True) for line in content.splitlines()]
        if len(rows) > 50:
            raise CohostPreflightError("Original model receipt exceeds the per-job request bound.")
        dispatched = [row for row in rows if row.get("dispatch_state") != "not_dispatched"]
        counts = [row.get("upstream_usage_total_tokens") for row in dispatched]
        if len(dispatched) != role.get("request_count") or (
            require_success
            and any(
                row.get("dispatch_state") != "settled"
                or row.get("http_status") != 200
                or row.get("run_id") != str(control_id)
                or row.get("role") != "evaluated"
                or type(row.get("upstream_usage_total_tokens")) is not int
                for row in rows
            )
        ):
            raise CohostPreflightError("Original native model dispatch/usage coverage differs.")
        if require_success:
            if sum(count for count in counts if type(count) is int) != role.get("upstream_usage_total_tokens"):
                raise CohostPreflightError("Original native model usage differs from actual observed rows.")
            original = [row for row in rows if row.get("request_kind") == "original"]
            qualifiers = [row for row in rows if row.get("request_kind") == "qualification"]
            if (
                len(original) + len(qualifiers) != len(rows)
                or len(qualifiers) > 1
                or (qualifiers and rows[0] is not qualifiers[0])
                or any(not isinstance(row.get("response_id"), str) for row in rows)
                or len({row.get("response_id") for row in rows}) != len(rows)
            ):
                raise CohostPreflightError("Qualification is not the bounded first dispatch of this same job.")
            from inspect_ai.log import read_eval_log

            log = read_eval_log(io.BytesIO(archive), format="eval")
            events = cls._model_events(log)
            cls.match_events(events=events, rows=original)

    @staticmethod
    def _model_events(log: EvalLog) -> list[ModelEvent]:
        from inspect_ai.event import Event, ModelEvent, SubtaskEvent

        result: list[ModelEvent] = []
        remaining = [event for sample in log.samples or [] for event in reversed(sample.events)]
        count = 0
        while remaining:
            event = remaining.pop()
            count += 1
            if count > 10_000:
                raise CohostPreflightError("Original model event traversal exceeds its bounded source policy.")
            if isinstance(event, ModelEvent):
                result.append(event)
            elif isinstance(event, SubtaskEvent):
                remaining.extend(reversed(TypeAdapter(list[Event]).validate_python(event.events, strict=True)))
        return result

    @staticmethod
    def match_events(*, events: Sequence[ModelEvent], rows: Sequence[dict[str, JsonValue]]) -> None:
        """Match authentic source IDs/requests/raw usage, including legitimate cached-prefix tokens."""
        by_id: dict[str, dict[str, JsonValue]] = {}
        response_ids: set[str] = set()
        for row in rows:
            identifier = row.get("inspect_request_id")
            response_id = row.get("response_id")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in by_id
                or not isinstance(response_id, str)
                or not response_id
                or response_id in response_ids
            ):
                raise CohostPreflightError("Native original model request identity is missing or duplicated.")
            by_id[identifier] = row
            response_ids.add(response_id)
        seen: set[str] = set()
        for event in events:
            call = event.call
            if (
                call is None
                or call.error
                or event.error
                or event.pending
                or event.cache == "read"
                or event.output.error
            ):
                raise CohostPreflightError("Successful original evidence lacks its typed non-error ModelCall.")
            request = TypeAdapter(dict[str, JsonValue]).validate_python(call.request, strict=True)
            headers = request.pop("extra_headers", None)
            identifier = headers.get("x-irid") if isinstance(headers, dict) else None
            if not isinstance(identifier, str) or identifier in seen or identifier not in by_id:
                raise CohostPreflightError("Original typed ModelCall and native request IDs are not bijective.")
            seen.add(identifier)
            row = by_id[identifier]
            for key in ("tools", "tool_choice"):
                if request.get(key) is None:
                    request.pop(key, None)
            if config_hash({"request": request}) != row.get("request_payload_sha256"):
                raise CohostPreflightError("Original typed request differs from exact incoming native API JSON.")
            response = TypeAdapter(dict[str, JsonValue]).validate_python(call.response, strict=True)
            usage = response.get("usage")
            if not isinstance(usage, dict) or response.get("id") != row.get("response_id"):
                raise CohostPreflightError("Original typed response ID/usage differs from native receipt.")
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                count = usage.get(key)
                if type(count) is not int or count < 0 or count != row.get(key):
                    raise CohostPreflightError("Original typed raw token usage differs from native metering.")
            prompt, completion, total = (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
            assert type(prompt) is int and type(completion) is int and type(total) is int
            output_usage = event.output.usage
            details = usage.get("prompt_tokens_details")
            cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
            if (
                total != prompt + completion
                or total != row.get("upstream_usage_total_tokens")
                or output_usage is None
                or type(cached) is not int
                or not 0 <= cached <= prompt
                or cached != row.get("cached_prompt_tokens", 0)
                or cached != (output_usage.input_tokens_cache_read or 0)
                or output_usage.input_tokens_cache_write not in (None, 0)
                or output_usage.input_tokens + (output_usage.input_tokens_cache_read or 0) != prompt
                or output_usage.output_tokens != completion
                or output_usage.total_tokens != total
            ):
                raise CohostPreflightError("Original framework/raw usage does not preserve authentic cached tokens.")
        if seen != set(by_id):
            raise CohostPreflightError("Original typed ModelEvent coverage differs from all native original posts.")
