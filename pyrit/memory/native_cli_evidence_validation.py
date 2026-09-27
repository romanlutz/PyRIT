# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Verify CLI report provenance against exact database-resident source evidence."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from typing import TYPE_CHECKING

from pydantic import ValidationError
from sqlalchemy import func, select

from pyrit.memory.memory_models import (
    NativeCyberEventEntry,
    NativeCyberRawChunkEntry,
    NativeCyberRawStreamEntry,
    NativeCyberToolEventEntry,
    NativeCyberTurnEntry,
    NativeCyberTurnMessagePieceEntry,
)
from pyrit.models.native_cli_report import (
    NativeCliReportEvent,
    NativeCliReportEventKind,
    NativeCliReportEventStatus,
    NativeCliReportProtocol,
    NativeCliReportStatus,
)
from pyrit.models.native_cyber_evidence import NativeCyberCoveragePhase, NativeCyberResponseMode

if TYPE_CHECKING:
    import uuid
    from collections.abc import Callable, Mapping, Sequence

    from sqlalchemy.orm import Session

    from pyrit.memory.memory_models import NativeCyberEpisodeEntry
    from pyrit.models.native_cli_report import NativeCliRunReport


class _NativeCliEvidenceValidator:
    """A CLI-only completeness check, independent of GHCP's event and tool shapes."""

    _EVENT_PAGE_SIZE = 16
    _RESPONSES_COVERAGE = frozenset(
        {
            "streaming",
            "function_tool",
            "function_call",
            "function_result",
            "custom_tool",
            "custom_call",
            "custom_result",
            "reasoning",
            "completed",
            "incomplete",
            "failed",
        }
    )
    _MESSAGES_COVERAGE = frozenset(
        {
            "streaming",
            "text",
            "tool_definition",
            "tool_use",
            "tool_result",
            "thinking",
            "prompt_caching",
            "ping",
            "completed",
            "failed",
        }
    )
    _TOOL_PHASES = {
        NativeCliReportEventKind.TOOL_REQUESTED: "request",
        NativeCliReportEventKind.TOOL_STARTED: "start",
        NativeCliReportEventKind.TOOL_COMPLETED: "complete",
        NativeCliReportEventKind.TOOL_RESULT: "result",
    }

    def __init__(
        self,
        *,
        session: Session,
        episode: NativeCyberEpisodeEntry,
        report: NativeCliRunReport,
        expected_turns: int,
        phase: NativeCyberCoveragePhase,
        digest_stream: Callable[[NativeCyberRawStreamEntry], str],
        digest_event: Callable[[Mapping[str, object]], str],
        verify_pieces: Callable[[Sequence[NativeCyberTurnMessagePieceEntry]], bool],
        is_terminal_event: Callable[[NativeCyberEventEntry], bool],
    ) -> None:
        """Bind one transaction to the caller-supplied report and integrity helpers."""
        self._session = session
        self._episode = episode
        self._report = report
        self._expected_turns = expected_turns
        self._phase = phase
        self._digest_stream = digest_stream
        self._digest_event = digest_event
        self._verify_pieces = verify_pieces
        self._is_terminal_event = is_terminal_event
        self._required: list[str] = list(episode.capture_gaps)
        self._optional: list[str] = list(episode.optional_gaps)
        if any("native evidence database write failed" in gap.lower() for gap in episode.optional_gaps):
            self._required.append("A CLI evidence database write failed for an undeclared source.")
        self._streams: dict[str, NativeCyberRawStreamEntry] = {}
        self._stream_rows: list[NativeCyberRawStreamEntry] = []
        self._range_cache: dict[tuple[uuid.UUID, int, int], tuple[str, bytes | None] | None] = {}
        self._pipe_offsets = {"stdout": 0, "stderr": 0}
        self._gateway_offsets: dict[uuid.UUID, int] = defaultdict(int)
        self._gateway_requests: dict[str, int] = {}
        self._gateway_responses: dict[str, int] = {}
        self._gateway_finished: set[str] = set()
        self._parser_index = 0
        self._raw_chunk_count = 0
        self._expected_tool_links: dict[tuple[str, str], int] = {}

    def validate(self) -> tuple[list[str], list[str]]:
        """
        Assess the report without modifying source bytes or inferring missing events.

        Returns:
            tuple[list[str], list[str]]: Required and optional gaps, both de-duplicated.
        """
        self._validate_identity()
        self._validate_turns()
        self._validate_streams()
        self._validate_events()
        self._validate_tool_links()
        self._validate_tool_causality()
        return list(dict.fromkeys(self._required)), list(dict.fromkeys(self._optional))

    def _validate_identity(self) -> None:
        report, episode = self._report, self._episode
        if episode.task_id is None or episode.task_version is None:
            self._required.append("CLI task ID and version were not declared before capture.")
        elif (episode.task_id, episode.task_version) != (report.task_id, report.task_version):
            self._required.append("CLI report task identity differs from the episode's pinned task.")
        if self._expected_turns != 1 or report.turn_index != 1:
            self._required.append("The CLI v1 report cannot prove a complete multi-turn episode.")
        if episode.conversation_id is None or episode.conversation_id != report.conversation_id:
            self._required.append("CLI report conversation does not match linked message pieces.")
        if episode.simulated is None or report.simulated is None or episode.simulated != report.simulated:
            self._required.append("CLI report simulation provenance does not match the episode.")
        if (
            episode.source_session_id is None
            or report.evidence.source_session_id is None
            or episode.source_session_id != report.evidence.source_session_id
        ):
            self._required.append("CLI report source session does not match the observed episode.")
        if (
            self._phase is NativeCyberCoveragePhase.FINAL and report.status is not NativeCliReportStatus.COMPLETED
        ) or not report.evidence.coverage_complete:
            self._required.append("CLI process or report did not declare complete source coverage.")
        if self._phase is NativeCyberCoveragePhase.FINAL and (report.judgment is None or not report.judgment.complete):
            self._required.append("The original CLI grader judgment was not completely acquired.")
        if report.evidence.gaps:
            self._required.append(f"CLI parser reported {len(report.evidence.gaps)} source coverage gap(s).")
        if report.errors:
            self._required.append(f"CLI lifecycle reported {len(report.errors)} caller-observed error(s).")

    def _validate_turns(self) -> None:
        report = self._report
        turns = list(
            self._session.scalars(
                select(NativeCyberTurnEntry)
                .where(NativeCyberTurnEntry.run_id == report.run_id)
                .order_by(NativeCyberTurnEntry.turn_index)
            )
        )
        if [turn.turn_index for turn in turns] != list(range(1, self._expected_turns + 1)):
            self._required.append("CLI outer turns differ from the declared turn count.")
        turn = next((item for item in turns if item.turn_index == report.turn_index), None)
        if turn is None or not turn.source_turn_id or turn.source_turn_id != report.turn_id:
            self._required.append("CLI report turn identity differs from the observed outer turn.")
        links = list(
            self._session.scalars(
                select(NativeCyberTurnMessagePieceEntry).where(NativeCyberTurnMessagePieceEntry.run_id == report.run_id)
            )
        )
        if not self._verify_pieces(links):
            self._required.append("A linked CLI MessagePiece is missing or modified.")
        directions: dict[int, set[str]] = defaultdict(set)
        for link in links:
            directions[link.turn_index].add(link.direction)
        for item in turns:
            if item.finished_at is None or item.source_complete is not True:
                self._required.extend(item.capture_gaps or ["A CLI outer turn was not completely captured."])
            if "request" not in directions[item.turn_index]:
                self._required.append("CLI outer turn has no genuine persisted request MessagePiece.")
            if item.response_mode == NativeCyberResponseMode.MESSAGE_REQUIRED.value:
                if "response" not in directions[item.turn_index]:
                    self._required.append("CLI chat turn has no genuine assistant response MessagePiece.")
            elif item.response_mode == NativeCyberResponseMode.ARTIFACT_ONLY.value:
                if self._episode.response_policy_version != 1 or not self._episode.artifact_only_allowed:
                    self._required.append("CLI artifact-only turn lacks task-approved response policy.")
                terminal = self._session.scalar(
                    select(NativeCyberEventEntry)
                    .where(
                        NativeCyberEventEntry.run_id == report.run_id,
                        NativeCyberEventEntry.turn_index == item.turn_index,
                        NativeCyberEventEntry.event_type.in_(("native_cli.turn_completed", "native_cli.run_finished")),
                    )
                    .order_by(NativeCyberEventEntry.sequence.desc())
                    .limit(1)
                )
                latest_action = self._session.scalar(
                    select(func.max(NativeCyberEventEntry.sequence)).where(
                        NativeCyberEventEntry.run_id == report.run_id,
                        NativeCyberEventEntry.turn_index == item.turn_index,
                        NativeCyberEventEntry.source.in_(("model", "tool")),
                    )
                )
                if (
                    not report.evidence.terminal_observed
                    or terminal is None
                    or not self._is_terminal_event(terminal)
                    or terminal.sequence < (latest_action or 0)
                ):
                    self._required.append("CLI artifact-only turn lacks an observed root terminal source event.")
                if self._phase is NativeCyberCoveragePhase.FINAL and (
                    not report.artifacts or report.judgment is None or not report.judgment.complete
                ):
                    self._required.append("CLI artifact-only turn lacks original grading and retained artifact.")
            else:
                self._required.append("CLI outer turn has an unsupported response mode.")

    def _validate_streams(self) -> None:
        report = self._report
        self._stream_rows = list(
            self._session.scalars(
                select(NativeCyberRawStreamEntry).where(NativeCyberRawStreamEntry.run_id == report.run_id)
            )
        )
        if sum(stream.stored_bytes for stream in self._stream_rows) != self._episode.stored_raw_bytes:
            self._required.append("CLI raw stream bytes differ from the episode quota ledger.")
        if self._episode.stored_raw_bytes > self._episode.raw_byte_limit:
            self._required.append("CLI raw stream bytes exceed the episode's declared quota.")
        for stream in self._stream_rows:
            self._required.extend(stream.capture_gaps)
            if stream.turn_index != report.turn_index:
                self._required.append("CLI raw stream is not bound to the reported outer turn.")
            if (
                stream.closed_at is None
                or stream.source_complete is not True
                or stream.truncated
                or stream.received_bytes != stream.stored_bytes
                or stream.expected_bytes != stream.stored_bytes
            ):
                self._required.append("CLI raw bytes were missing, capped, or not sealed.")
            try:
                digest = self._digest_stream(stream)
            except ValueError:
                self._required.append("A CLI raw byte stream has a missing or corrupt database chunk.")
                continue
            if stream.stored_sha256 != digest or stream.observed_sha256 != digest:
                self._required.append("A CLI raw stream digest differs from stored source bytes.")
        protocol = report.protocol.value
        wanted = {
            "stdout": ("harness", "stdout", f"{protocol}.stdout"),
            "stderr": ("harness", "stderr", f"{protocol}.stderr"),
            "request": ("model", "model", f"{protocol}.gateway.requests"),
            "response": ("model", "model", f"{protocol}.gateway.responses"),
        }
        for name, (source, kind, observed_id) in wanted.items():
            matches = [
                stream
                for stream in self._stream_rows
                if (stream.turn_index, stream.source, stream.kind, stream.observed_source_id)
                == (report.turn_index, source, kind, observed_id)
            ]
            if len(matches) != 1:
                self._required.append(f"CLI {name} source stream is missing or ambiguous in the database.")
            else:
                self._streams[name] = matches[0]
        for name, count in (("stdout", report.evidence.raw_stdout_bytes), ("stderr", report.evidence.raw_stderr_bytes)):
            stream = self._streams.get(name)
            if count is None or stream is None or stream.stored_bytes != count:
                self._required.append(f"CLI {name} bytes differ from the reported process total.")
        for name in ("request", "response"):
            stream = self._streams.get(name)
            if stream is None or stream.stored_bytes <= 0:
                self._required.append(f"CLI model gateway {name} bytes were not retained in the database.")
        if any(
            stream.observed_source_id == f"{protocol}.gateway.errors" and stream.stored_bytes > 0
            for stream in self._stream_rows
        ):
            self._required.append("CLI model gateway stored host-generated error bytes.")

    def _range_digest(
        self, *, stream: NativeCyberRawStreamEntry, offset: int, length: int
    ) -> tuple[str, bytes | None] | None:
        if offset < 0 or length <= 0 or offset + length > stream.stored_bytes:
            return None
        key = (stream.stream_id, offset, length)
        if key in self._range_cache:
            return self._range_cache[key]
        digest = hashlib.sha256()
        sample = bytearray() if length <= 8_192 else None
        position = offset
        statement = (
            select(NativeCyberRawChunkEntry)
            .where(
                NativeCyberRawChunkEntry.stream_id == stream.stream_id,
                NativeCyberRawChunkEntry.byte_offset >= max(0, offset - 65_536 + 1),
                NativeCyberRawChunkEntry.byte_offset < offset + length,
                NativeCyberRawChunkEntry.byte_offset + NativeCyberRawChunkEntry.byte_length > offset,
            )
            .order_by(NativeCyberRawChunkEntry.byte_offset)
        )
        for chunk in self._session.scalars(statement).yield_per(16):
            start = max(offset, chunk.byte_offset)
            end = min(offset + length, chunk.byte_offset + chunk.byte_length)
            if (
                start != position
                or chunk.byte_length != len(chunk.data)
                or not 0 < chunk.byte_length <= 65_536
                or chunk.sha256 != hashlib.sha256(chunk.data).hexdigest()
            ):
                self._range_cache[key] = None
                return None
            part = chunk.data[start - chunk.byte_offset : end - chunk.byte_offset]
            digest.update(part)
            if sample is not None:
                sample.extend(part)
            position = end
        result = (
            (digest.hexdigest(), bytes(sample) if sample is not None else None) if position == offset + length else None
        )
        self._range_cache[key] = result
        return result

    def _validate_events(self) -> None:
        cursor = 0
        while True:
            page = list(
                self._session.scalars(
                    select(NativeCyberEventEntry)
                    .where(
                        NativeCyberEventEntry.run_id == self._report.run_id,
                        NativeCyberEventEntry.sequence > cursor,
                    )
                    .order_by(NativeCyberEventEntry.sequence)
                    .limit(self._EVENT_PAGE_SIZE)
                )
            )
            if not page:
                break
            for event in page:
                if event.sequence != cursor + 1 or event.turn_index != self._report.turn_index:
                    self._required.append("CLI controller event order or outer-turn identity was modified.")
                cursor = event.sequence
                if not self._has_structured_payload(value=event.payload):
                    self._required.append("A CLI database event has no structured source payload.")
                    continue
                try:
                    digest = self._digest_event(event.payload)
                except ValueError:
                    self._required.append("A CLI event exceeds the bounded payload contract.")
                    continue
                if digest != event.payload_sha256:
                    self._required.append("A retained CLI event payload failed its database digest.")
                    continue
                if event.event_type == "native_cli.raw_chunk":
                    self._validate_raw_receipt(event)
                elif event.event_type.startswith("native_cli."):
                    self._validate_parser_event(event)
                elif event.event_type.startswith(("gateway.", "messages_gateway.")):
                    self._validate_gateway_event(event)
                else:
                    self._required.append("A foreign event was mixed into the CLI episode.")
        evidence = self._report.evidence
        if self._parser_index != len(evidence.events):
            self._required.append("CLI parser rows differ from the canonical report's event count.")
        if evidence.raw_chunk_count is None or self._raw_chunk_count != evidence.raw_chunk_count:
            self._required.append("CLI cross-pipe chunk receipts differ from the reported process count.")
        for name in ("stdout", "stderr"):
            stream = self._streams.get(name)
            if stream is not None and self._pipe_offsets[name] != stream.stored_bytes:
                self._required.append(f"CLI {name} byte ranges are not fully accounted for by process receipts.")
        for name in ("request", "response"):
            stream = self._streams.get(name)
            if stream is not None and self._gateway_offsets[stream.stream_id] != stream.stored_bytes:
                self._required.append(f"CLI gateway {name} bytes are not fully accounted for by source events.")
        if not self._gateway_requests or set(self._gateway_requests) != set(self._gateway_responses):
            self._required.append("CLI model gateway requests lack correlated original model responses.")
        if set(self._gateway_requests) != self._gateway_finished:
            self._required.append("One or more CLI model gateway responses lack a confirmed terminal frame.")

    @staticmethod
    def _has_structured_payload(*, value: object) -> bool:
        return isinstance(value, dict)

    def _validate_raw_receipt(self, event: NativeCyberEventEntry) -> None:
        payload = event.payload
        self._raw_chunk_count += 1
        pipe = payload.get("stream")
        ordinal = payload.get("raw_chunk_sequence")
        length = payload.get("length")
        digest = payload.get("sha256")
        if (
            event.source != "harness"
            or event.tool_call_id is not None
            or type(ordinal) is not int
            or ordinal != self._raw_chunk_count
            or not isinstance(pipe, str)
            or pipe not in {"stdout", "stderr"}
            or type(length) is not int
            or length <= 0
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            self._required.append("CLI cross-pipe raw chunk order or source metadata is invalid.")
            return
        stream = self._streams.get(pipe)
        if stream is None or event.observed_stream_id != str(stream.stream_id):
            self._required.append("CLI cross-pipe raw chunk names no retained process pipe.")
            return
        offset = self._pipe_offsets[pipe]
        if event.stream_offset != offset:
            self._required.append("CLI cross-pipe raw chunk offsets are not contiguous.")
        region = self._range_digest(stream=stream, offset=offset, length=length)
        if region is None or region[0] != digest:
            self._required.append("CLI cross-pipe raw chunk differs from its retained source bytes.")
        self._pipe_offsets[pipe] += length

    def _validate_parser_event(self, event: NativeCyberEventEntry) -> None:
        index = self._parser_index
        self._parser_index += 1
        if index >= len(self._report.evidence.events):
            self._required.append("CLI database has a parser observation absent from the canonical report.")
            return
        expected = self._report.evidence.events[index]
        payload = event.payload
        try:
            actual = NativeCliReportEvent.model_validate(
                {
                    "sequence": payload.get("parser_sequence"),
                    "frame_number": payload.get("frame_number"),
                    "kind": event.event_type.removeprefix("native_cli."),
                    "status": payload.get("status"),
                    "source_event_id": event.observed_event_id,
                    "source_message_id": payload.get("source_message_id"),
                    "source_session_id": event.observed_session_id,
                    "source_tool_id": event.tool_call_id,
                    "parent_tool_use_id": payload.get("parent_tool_use_id"),
                    "source_status": payload.get("source_status"),
                    "name": payload.get("name"),
                    "exit_code": payload.get("exit_code"),
                    "detail": payload.get("detail"),
                    "raw_frame_sha256": payload.get("raw_frame_sha256"),
                    "raw_frame_size_bytes": payload.get("raw_frame_size_bytes"),
                    "stdout_offset_bytes": event.stream_offset,
                }
            )
        except ValidationError:
            self._required.append("A retained CLI parser row is not a valid report observation.")
            return
        if actual != expected or actual.sequence != index + 1:
            self._required.append("CLI parser rows differ from the canonical report's ordered event summaries.")
        source = (
            "model"
            if actual.kind in {NativeCliReportEventKind.MODEL_MESSAGE, NativeCliReportEventKind.TOOL_REQUESTED}
            else "tool"
            if actual.kind
            in {
                NativeCliReportEventKind.TOOL_STARTED,
                NativeCliReportEventKind.TOOL_COMPLETED,
                NativeCliReportEventKind.TOOL_RESULT,
            }
            else "harness"
        )
        if event.source != source or event.tool_phase != self._TOOL_PHASES.get(actual.kind):
            self._required.append("CLI parser row has a mismatched source tag or observed tool phase.")
        if phase := self._TOOL_PHASES.get(actual.kind):
            if not actual.source_tool_id:
                self._required.append("CLI tool phase lacks its genuine source tool ID.")
            else:
                key = (actual.source_tool_id, phase)
                if key in self._expected_tool_links:
                    self._required.append("A CLI tool phase was repeated without a distinct observed call.")
                self._expected_tool_links[key] = event.sequence
        if actual.frame_number is None:
            if event.observed_stream_id is not None or event.stream_offset is not None:
                self._required.append("CLI non-frame observation claims a source byte offset.")
            return
        stdout = self._streams.get("stdout")
        if stdout is None or event.observed_stream_id != str(stdout.stream_id):
            self._required.append("CLI provider frame does not name the retained stdout stream.")
            return
        if actual.stdout_offset_bytes is None or actual.raw_frame_size_bytes is None:
            self._required.append("CLI provider frame lacks a verified source range.")
            return
        region = self._range_digest(
            stream=stdout,
            offset=actual.stdout_offset_bytes,
            length=actual.raw_frame_size_bytes,
        )
        if region is None or region[0] != actual.raw_frame_sha256:
            self._required.append("CLI provider frame digest differs from retained stdout bytes.")

    def _validate_gateway_event(self, event: NativeCyberEventEntry) -> None:
        payload = event.payload
        is_anthropic = self._report.protocol is NativeCliReportProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE
        prefix = "messages_gateway." if is_anthropic else "gateway."
        if not event.event_type.startswith(prefix):
            self._required.append("CLI model gateway wire protocol differs from the selected CLI profile.")
            return
        kind = event.event_type.removeprefix(prefix)
        if kind not in {"request", "response", "response_event", "gateway_error"}:
            self._required.append("CLI episode has an unknown model gateway event.")
            return
        if is_anthropic:
            self._check_anthropic_metadata(payload=payload)
        elif payload.get("wire_protocol") not in (None, "openai_responses"):
            self._required.append("Codex model gateway event claims an unrelated provider wire contract.")
        stream_name = "request" if kind == "request" else "error" if kind == "gateway_error" else "response"
        stream = self._gateway_stream(stream_name)
        source = "harness" if kind == "gateway_error" else "model"
        request_id = payload.get("gateway_request_id")
        frame_digest = payload.get("frame_sha256")
        frame_size = payload.get("frame_size_bytes")
        coverage = payload.get("coverage")
        if (
            event.source != source
            or event.observed_event_id is not None
            or event.tool_call_id is not None
            or not isinstance(request_id, str)
            or not 0 < len(request_id) <= 128
            or not isinstance(frame_digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", frame_digest) is None
            or type(frame_size) is not int
            or frame_size <= 0
            or not isinstance(coverage, list)
            or any(not isinstance(item, str) for item in coverage)
        ):
            self._required.append("CLI model gateway event lacks valid host-observed correlation metadata.")
            return
        flags = set(coverage)
        allowed_flags = self._MESSAGES_COVERAGE if is_anthropic else self._RESPONSES_COVERAGE
        if not flags <= allowed_flags:
            self._required.append("CLI model gateway reported unrecognized wire coverage flags.")
        if {"failed", "incomplete"} & flags or payload.get("error_code") is not None:
            self._required.append("CLI model gateway reported a failure or incomplete original wire response.")
        status_code = payload.get("status_code")
        if is_anthropic and kind == "response" and status_code != 200:
            self._required.append("Original Anthropic provider response was not HTTP 200.")
        if status_code is not None and (type(status_code) is not int or not 200 <= status_code < 300):
            if is_anthropic and kind == "response" and "failed" in flags:
                self._required.append("Original Anthropic provider returned a failed response.")
            else:
                self._required.append("CLI model gateway recorded an unsuccessful HTTP status.")
        if stream is None or event.observed_stream_id != str(stream.stream_id):
            self._required.append("CLI model gateway event has no matching database raw stream.")
            return
        offset = self._gateway_offsets[stream.stream_id]
        if event.stream_offset != offset:
            self._required.append("CLI model gateway source frames have noncontiguous byte offsets.")
        region = self._range_digest(stream=stream, offset=offset, length=frame_size)
        if region is None or region[0] != frame_digest:
            self._required.append("CLI model gateway frame differs from its retained raw bytes.")
        self._gateway_offsets[stream.stream_id] += frame_size
        if kind == "gateway_error":
            self._required.append("CLI model gateway emitted a host-generated error, not a provider response.")
            return
        if kind == "request":
            if request_id in self._gateway_requests:
                self._required.append("CLI model gateway request identity was reused.")
            self._gateway_requests[request_id] = event.sequence
            return
        if request_id not in self._gateway_requests or self._gateway_requests[request_id] >= event.sequence:
            self._required.append("CLI model gateway response has no preceding real request ID.")
        if request_id in self._gateway_finished:
            self._required.append("CLI model gateway sent another frame after its completed response.")
        self._gateway_responses[request_id] = event.sequence
        terminal = kind == "response" or (
            kind == "response_event"
            and region is not None
            and region[1] is not None
            and (
                self._is_anthropic_message_stop(frame=region[1])
                if is_anthropic
                else region[1].replace(b"\r\n", b"\n") == b"data: [DONE]\n\n"
            )
        )
        if terminal and "completed" not in flags:
            self._required.append("CLI model gateway terminal response lacks observed COMPLETED coverage.")
        if terminal and "completed" in flags and not {"failed", "incomplete"} & flags:
            self._gateway_finished.add(request_id)

    def _check_anthropic_metadata(self, *, payload: dict[str, object]) -> None:
        if payload.get("wire_protocol") != "anthropic_messages":
            self._required.append("Claude model gateway event lacks its observed Anthropic wire protocol.")
        headers = payload.get("headers")
        query = payload.get("query_string")
        if (
            not isinstance(headers, list)
            or len(headers) > 32
            or not isinstance(query, str)
            or query not in {"", "beta=true"}
        ):
            self._required.append("Claude model gateway headers or query are not a bounded, approved snapshot.")
            return
        for pair in headers:
            allowed_name = (
                isinstance(pair, (list, tuple))
                and len(pair) == 2
                and isinstance(pair[0], str)
                and (
                    pair[0].lower()
                    in {"anthropic-version", "anthropic-beta", "content-type", "retry-after", "x-should-retry"}
                    or pair[0].lower().startswith("anthropic-ratelimit-unified-")
                )
            )
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or not all(isinstance(part, str) for part in pair)
                or not 0 < len(pair[0]) <= 128
                or not allowed_name
                or len(pair[1]) > 512
                or not pair[1].isascii()
                or any(ord(char) < 32 or ord(char) > 126 for char in pair[1])
            ):
                self._required.append("Claude model gateway contains an unapproved selected header.")
                return

    @staticmethod
    def _is_anthropic_message_stop(*, frame: bytes) -> bool:
        try:
            text = frame.decode("utf-8").replace("\r\n", "\n")
            if not text.endswith("\n\n"):
                return False
            lines = text[:-2].split("\n")
            names = [line[6:].strip() for line in lines if line.startswith("event:")]
            data = [line[5:].lstrip(" ") for line in lines if line.startswith("data:")]
            if (
                names != ["message_stop"]
                or not data
                or any(not line.startswith(("event:", "data:", ":")) for line in lines)
            ):
                return False
            parsed = json.loads("\n".join(data))
            if not isinstance(parsed, dict):
                return False
            kind = parsed.get("type")
            return isinstance(kind, str) and kind == "message_stop"
        except (UnicodeDecodeError, ValueError):
            return False

    def _gateway_stream(self, name: str) -> NativeCyberRawStreamEntry | None:
        if name != "error":
            return self._streams.get(name)
        matches = [
            stream
            for stream in self._stream_rows
            if (
                stream.turn_index,
                stream.source,
                stream.kind,
                stream.observed_source_id,
            )
            == (
                self._report.turn_index,
                "harness",
                "model",
                f"{self._report.protocol.value}.gateway.errors",
            )
        ]
        if len(matches) != 1:
            self._required.append("CLI model gateway error event has no single retained error stream.")
            return None
        return matches[0]

    def _validate_tool_links(self) -> None:
        stored: dict[tuple[str, str], int] = {}
        for link in self._session.scalars(
            select(NativeCyberToolEventEntry).where(NativeCyberToolEventEntry.run_id == self._report.run_id)
        ):
            key = (link.call_id, link.phase)
            if key in stored:
                self._required.append("CLI tool event links contain a repeated observed phase.")
            stored[key] = link.event_sequence
        if stored != self._expected_tool_links:
            self._required.append("CLI tool request/start/completion/result links differ from source observations.")

    def _validate_tool_causality(self) -> None:
        calls: dict[str, dict[NativeCliReportEventKind, NativeCliReportEvent]] = defaultdict(dict)
        for event in self._report.evidence.events:
            if event.kind not in self._TOOL_PHASES:
                continue
            if event.source_tool_id is None:
                self._required.append("A CLI tool action lacks a genuine source call ID.")
                continue
            phases = calls[event.source_tool_id]
            if event.kind in phases:
                self._required.append("A CLI tool call repeated a lifecycle phase without a distinct request.")
            phases[event.kind] = event
        if not calls:
            return
        if self._report.protocol is NativeCliReportProtocol.CODEX_EXEC_JSON:
            self._required.append("Codex tool execution has no provable model-visible request for its item ID.")
            for phases in calls.values():
                start = phases.get(NativeCliReportEventKind.TOOL_STARTED)
                complete = phases.get(NativeCliReportEventKind.TOOL_COMPLETED)
                result = phases.get(NativeCliReportEventKind.TOOL_RESULT)
                if (
                    start is None
                    or complete is None
                    or result is None
                    or not start.sequence < complete.sequence < result.sequence
                ):
                    self._required.append("Codex tool start, completion, and result were not all observed in order.")
            return
        for phases in calls.values():
            request = phases.get(NativeCliReportEventKind.TOOL_REQUESTED)
            result = phases.get(NativeCliReportEventKind.TOOL_RESULT)
            if (
                set(phases) != {NativeCliReportEventKind.TOOL_REQUESTED, NativeCliReportEventKind.TOOL_RESULT}
                or request is None
                or result is None
                or request.source_message_id is None
                or request.status is not NativeCliReportEventStatus.REQUESTED
                or result.status not in {NativeCliReportEventStatus.COMPLETED, NativeCliReportEventStatus.FAILED}
                or request.source_session_id != result.source_session_id
                or request.parent_tool_use_id != result.parent_tool_use_id
                or request.sequence >= result.sequence
            ):
                self._required.append("Claude tool request and model-visible result lack provable source causality.")
