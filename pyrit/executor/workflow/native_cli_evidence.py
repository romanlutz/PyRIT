# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Durably record a sandboxed coding CLI without assigning invented provider IDs."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

from pyrit.models.native_cli_report import NativeCliReportEvent
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberResponseMode,
    NativeCyberToolPhase,
    NativeCyberTurnFinish,
    NativeCyberTurnStart,
)
from pyrit.prompt_target.gateway.messages_contract import MessagesCoverage, MessagesObservation
from pyrit.prompt_target.gateway.responses_contract import GatewayCoverage, GatewayFrameKind, GatewayObservation
from pyrit.prompt_target.native_cli_models import NativeCliEventKind, NativeCliProtocol, NativeCliStream

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from pydantic import JsonValue

    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore
    from pyrit.prompt_target.native_cli_models import NativeCliEvent, NativeCliRawChunk, NativeCliRunOutcome


class NativeCliDatabaseEvidenceSink:
    """Record exact stdout/stderr and ordered CLI observations in an existing DB episode."""

    MAX_OBSERVATIONS = 10_000
    MAX_RAW_CHUNKS = 10_000
    MAX_GATEWAY_OBSERVATIONS = 10_000

    _TOOL_PHASES = {
        NativeCliEventKind.TOOL_REQUESTED: NativeCyberToolPhase.REQUEST,
        NativeCliEventKind.TOOL_STARTED: NativeCyberToolPhase.START,
        NativeCliEventKind.TOOL_COMPLETED: NativeCyberToolPhase.COMPLETE,
        NativeCliEventKind.TOOL_RESULT: NativeCyberToolPhase.RESULT,
    }

    def __init__(
        self,
        *,
        store: NativeCyberEvidenceStore,
        run_id: str,
        turn_id: str,
        turn_index: int,
        protocol: NativeCliProtocol,
        include_model_gateway: bool = False,
    ) -> None:
        """
        Bind a sink to one caller-created episode and one outer turn.

        Raises:
            ValueError: If the protocol, run ID or outer-turn index is invalid.
        """
        if not isinstance(protocol, NativeCliProtocol) or type(include_model_gateway) is not bool:
            raise ValueError("A documented CLI protocol and explicit model-gateway selection are required.")
        if (
            not isinstance(run_id, str)
            or not 0 < len(run_id) <= 128
            or not isinstance(turn_id, str)
            or not 0 < len(turn_id) <= 128
            or type(turn_index) is not int
            or turn_index < 1
        ):
            raise ValueError("Native CLI capture needs existing run/turn identities and a positive turn index.")
        self._store = store
        self._run_id = run_id
        self._turn_id = turn_id
        self._turn_index = turn_index
        self._protocol = protocol
        self._streams = {
            stream: NativeCyberRawStreamStart(run_id=run_id, turn_index=turn_index, key=key)
            for stream, key in zip(
                NativeCliStream, self.required_raw_streams(protocol=protocol)[: len(NativeCliStream)], strict=True
            )
        }
        self._include_model_gateway = include_model_gateway
        self._gateway_streams = (
            {
                name: NativeCyberRawStreamStart(run_id=run_id, turn_index=turn_index, key=key)
                for name, key in self._gateway_keys(protocol=protocol).items()
            }
            if include_model_gateway
            else {}
        )
        self._received = dict.fromkeys(NativeCliStream, 0)
        self._hashes = {stream: hashlib.sha256() for stream in NativeCliStream}
        self._gateway_received = dict.fromkeys(self._gateway_streams, 0)
        self._gateway_hashes = {name: hashlib.sha256() for name in self._gateway_streams}
        self._gateway_requests: dict[str, bool] = {}
        self._gateway_terminal_requests: set[str] = set()
        self._gateway_failed = False
        self._gateway_observation_count = 0
        self._raw_chunk_sequence = 0
        self._parser_event_sequence = 0
        self._controller_sequence = 0
        self._last_frame_number = 0
        self._last_frame_digest: str | None = None
        self._last_frame_size = 0
        self._last_frame_offset = 0
        self._next_frame_offset = 0
        self._started = False
        self._finished = False
        self._failed = False
        self._lock = asyncio.Lock()

    @property
    def run_id(self) -> str:
        """The existing episode receiving this CLI evidence."""
        return self._run_id

    @property
    def turn_id(self) -> str:
        """The controller-assigned outer turn, never a provider event ID."""
        return self._turn_id

    @staticmethod
    def required_raw_streams(
        *, protocol: NativeCliProtocol, include_model_gateway: bool = False
    ) -> tuple[NativeCyberRawStreamKey, ...]:
        """
        Declare both CLI pipes in the episode manifest before the process starts.

        Returns:
            tuple[NativeCyberRawStreamKey, ...]: Source-tagged stdout and stderr identities.

        Raises:
            ValueError: If the CLI protocol is not supported.
        """
        if not isinstance(protocol, NativeCliProtocol) or type(include_model_gateway) is not bool:
            raise ValueError("A documented CLI protocol and explicit model-gateway selection are required.")
        pipes = tuple(
            NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS,
                kind=NativeCyberRawKind.STDOUT if stream is NativeCliStream.STDOUT else NativeCyberRawKind.STDERR,
                observed_source_id=f"{protocol.value}.{stream.value}",
            )
            for stream in NativeCliStream
        )
        if not include_model_gateway:
            return pipes
        gateway = NativeCliDatabaseEvidenceSink._gateway_keys(protocol=protocol)
        return (*pipes, gateway["request"], gateway["response"])

    @staticmethod
    def _gateway_keys(*, protocol: NativeCliProtocol) -> dict[str, NativeCyberRawStreamKey]:
        return {
            "request": NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.MODEL,
                kind=NativeCyberRawKind.MODEL,
                observed_source_id=f"{protocol.value}.gateway.requests",
            ),
            "response": NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.MODEL,
                kind=NativeCyberRawKind.MODEL,
                observed_source_id=f"{protocol.value}.gateway.responses",
            ),
            "error": NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS,
                kind=NativeCyberRawKind.MODEL,
                observed_source_id=f"{protocol.value}.gateway.errors",
            ),
        }

    async def start_async(
        self, *, started_at: datetime, response_mode: NativeCyberResponseMode = NativeCyberResponseMode.MESSAGE_REQUIRED
    ) -> None:
        """
        Open a real outer turn and its two raw sources before launching the CLI.

        Raises:
            ValueError: If the sink has already started or task approval is invalid.
            asyncio.CancelledError: If the caller interrupts capture preparation.
        """
        async with self._lock:
            if self._started:
                raise ValueError("A CLI evidence sink can start only once.")
            self._started = True
            try:
                await asyncio.to_thread(
                    self._store.begin_turn,
                    turn=NativeCyberTurnStart(
                        run_id=self._run_id,
                        turn_index=self._turn_index,
                        source_turn_id=self._turn_id,
                        started_at=started_at,
                        response_mode=response_mode,
                    ),
                )
                for stream in NativeCliStream:
                    await asyncio.to_thread(self._store.open_raw_stream, stream=self._streams[stream])
                for stream in self._gateway_streams.values():
                    await asyncio.to_thread(self._store.open_raw_stream, stream=stream)
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def record_gateway_observation_async(self, observation: GatewayObservation) -> None:
        """
        Retain OpenAI Responses wire bytes and their host-generated failure boundaries.

        Raises:
            ValueError: If the frame has no matching run, request, or supported source kind.
            asyncio.CancelledError: If model capture is interrupted.
        """
        async with self._lock:
            self._require_capture()
            try:
                if (
                    self._protocol is not NativeCliProtocol.CODEX_EXEC_JSON
                    or not self._include_model_gateway
                    or not isinstance(observation, GatewayObservation)
                    or observation.run_id != self._run_id
                    or not isinstance(observation.coverage, frozenset)
                    or not all(isinstance(flag, GatewayCoverage) for flag in observation.coverage)
                ):
                    raise ValueError("OpenAI Responses observations require an approved Codex gateway route.")
                terminal = observation.kind in {GatewayFrameKind.RESPONSE, GatewayFrameKind.GATEWAY_ERROR} or (
                    observation.kind is GatewayFrameKind.RESPONSE_EVENT
                    and observation.frame.replace(b"\r\n", b"\n") == b"data: [DONE]\n\n"
                )
                await self._record_model_frame_async(
                    request_id=observation.request_id,
                    kind=observation.kind,
                    frame=observation.frame,
                    coverage=sorted(flag.value for flag in observation.coverage),
                    event_namespace="gateway",
                    wire_protocol="openai_responses",
                    terminal=terminal,
                    completed=GatewayCoverage.COMPLETED in observation.coverage
                    and GatewayCoverage.INCOMPLETE not in observation.coverage
                    and GatewayCoverage.FAILED not in observation.coverage,
                    error_code=observation.error_code,
                    status_code=observation.status_code,
                )
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def record_messages_observation_async(self, observation: MessagesObservation) -> None:
        """
        Retain Anthropic Messages wire bytes without translating them to Responses.

        Raises:
            ValueError: If the source, headers, query, or terminal event is unsupported.
            asyncio.CancelledError: If model capture is interrupted.
        """
        async with self._lock:
            self._require_capture()
            try:
                if (
                    self._protocol is not NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE
                    or not self._include_model_gateway
                    or not isinstance(observation, MessagesObservation)
                    or observation.run_id != self._run_id
                    or not isinstance(observation.coverage, frozenset)
                    or not all(isinstance(flag, MessagesCoverage) for flag in observation.coverage)
                    or not isinstance(observation.query_string, str)
                    or observation.query_string not in {"", "beta=true"}
                    or not isinstance(observation.headers, tuple)
                    or any(
                        len(pair) != 2
                        or not all(isinstance(item, str) for item in pair)
                        or (
                            pair[0].lower()
                            not in {
                                "anthropic-version",
                                "anthropic-beta",
                                "content-type",
                                "retry-after",
                                "x-should-retry",
                            }
                            and not pair[0].lower().startswith("anthropic-ratelimit-unified-")
                        )
                        or len(pair[1]) > 512
                        or not pair[1].isascii()
                        or any(ord(char) < 32 or ord(char) > 126 for char in pair[1])
                        for pair in observation.headers
                    )
                ):
                    raise ValueError("Anthropic Messages observation has an unapproved source, header, or query.")
                message_stop = observation.kind is GatewayFrameKind.RESPONSE_EVENT and observation.frame.replace(
                    b"\r\n", b"\n"
                ).startswith(b"event: message_stop\n")
                terminal = (
                    observation.kind in {GatewayFrameKind.RESPONSE, GatewayFrameKind.GATEWAY_ERROR} or message_stop
                )
                await self._record_model_frame_async(
                    request_id=observation.request_id,
                    kind=observation.kind,
                    frame=observation.frame,
                    coverage=sorted(flag.value for flag in observation.coverage),
                    event_namespace="messages_gateway",
                    wire_protocol="anthropic_messages",
                    terminal=terminal,
                    completed=MessagesCoverage.COMPLETED in observation.coverage
                    and MessagesCoverage.FAILED not in observation.coverage
                    and (observation.kind is not GatewayFrameKind.RESPONSE or observation.status_code == 200),
                    error_code=observation.error_code,
                    status_code=observation.status_code,
                    metadata={
                        "headers": [[name, value] for name, value in observation.headers],
                        "query_string": observation.query_string,
                    },
                )
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def _record_model_frame_async(
        self,
        *,
        request_id: str,
        kind: GatewayFrameKind,
        frame: bytes,
        coverage: list[str],
        event_namespace: str,
        wire_protocol: str,
        terminal: bool,
        completed: bool,
        error_code: str | None,
        status_code: int | None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        if (
            not isinstance(request_id, str)
            or not 0 < len(request_id) <= 128
            or not isinstance(kind, GatewayFrameKind)
            or not isinstance(frame, bytes)
            or not frame
            or (error_code is not None and (not isinstance(error_code, str) or len(error_code) > 128))
            or (status_code is not None and (type(status_code) is not int or not 100 <= status_code <= 599))
        ):
            raise ValueError("Model gateway observation requires a bounded source frame and request identity.")
        if self._gateway_observation_count >= self.MAX_GATEWAY_OBSERVATIONS:
            raise ValueError("Native CLI model gateway observation limit exceeded.")
        if kind is GatewayFrameKind.REQUEST:
            if request_id in self._gateway_requests:
                raise ValueError("A model gateway request identity was repeated.")
            self._gateway_requests[request_id] = False
        elif request_id not in self._gateway_requests or request_id in self._gateway_terminal_requests:
            raise ValueError("Model gateway output has no open observed model request.")
        name = (
            "request"
            if kind is GatewayFrameKind.REQUEST
            else "error"
            if kind is GatewayFrameKind.GATEWAY_ERROR
            else "response"
        )
        raw_stream = self._gateway_streams[name]
        offset = self._gateway_received[name]
        for start in range(0, len(frame), self._store.MAX_APPEND_BYTES):
            data = frame[start : start + self._store.MAX_APPEND_BYTES]
            await asyncio.to_thread(
                self._store.append_raw, run_id=self._run_id, stream_id=raw_stream.stream_id, data=data
            )
            self._gateway_hashes[name].update(data)
            self._gateway_received[name] += len(data)
        coverage_values: list[JsonValue] = list(coverage)
        await self._append_event_async(
            source=NativeCyberEvidenceSource.HARNESS
            if kind is GatewayFrameKind.GATEWAY_ERROR
            else NativeCyberEvidenceSource.MODEL,
            event_type=f"{event_namespace}.{kind.value}",
            payload={
                "gateway_request_id": request_id,
                "frame_sha256": hashlib.sha256(frame).hexdigest(),
                "frame_size_bytes": len(frame),
                "coverage": coverage_values,
                "wire_protocol": wire_protocol,
                "error_code": error_code,
                "status_code": status_code,
                **(metadata or {}),
            },
            observed_stream_id=str(raw_stream.stream_id),
            stream_offset=offset,
        )
        self._gateway_observation_count += 1
        if kind is GatewayFrameKind.GATEWAY_ERROR:
            self._gateway_terminal_requests.add(request_id)
            self._gateway_failed = True
            await asyncio.to_thread(
                self._store.mark_capture_gap,
                run_id=self._run_id,
                reason="Model gateway produced a host-generated failure.",
            )
        elif terminal:
            self._gateway_terminal_requests.add(request_id)
            self._gateway_requests[request_id] = completed

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        """
        Commit unchanged pipe bytes and a cross-pipe ordering receipt before returning.

        Raises:
            TypeError: If the process supplies a non-byte chunk or unknown pipe.
            ValueError: If the source's cross-pipe sequence is not consecutive.
            asyncio.CancelledError: If the caller interrupts a database write.
        """
        async with self._lock:
            self._require_capture()
            try:
                stream = chunk.stream
                if not isinstance(stream, NativeCliStream) or not isinstance(chunk.data, bytes):
                    raise TypeError("Native CLI raw chunks require a real pipe and bytes.")
                if chunk.sequence != self._raw_chunk_sequence + 1:
                    raise ValueError("Native CLI raw chunk sequence must be consecutive across both pipes.")
                if chunk.sequence > self.MAX_RAW_CHUNKS:
                    raise ValueError("Native CLI cross-pipe raw chunk limit exceeded.")
                raw_stream = self._streams[stream]
                offset = self._received[stream]
                for start in range(0, len(chunk.data), self._store.MAX_APPEND_BYTES):
                    data = chunk.data[start : start + self._store.MAX_APPEND_BYTES]
                    await asyncio.to_thread(
                        self._store.append_raw, run_id=self._run_id, stream_id=raw_stream.stream_id, data=data
                    )
                    self._hashes[stream].update(data)
                    self._received[stream] += len(data)
                await self._append_event_async(
                    source=NativeCyberEvidenceSource.HARNESS,
                    event_type="native_cli.raw_chunk",
                    payload={
                        "raw_chunk_sequence": chunk.sequence,
                        "stream": stream.value,
                        "length": len(chunk.data),
                        "sha256": hashlib.sha256(chunk.data).hexdigest(),
                    },
                    observed_stream_id=str(raw_stream.stream_id),
                    stream_offset=offset,
                )
                self._raw_chunk_sequence += 1
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        """
        Commit an actual parser observation with its source IDs and frame digest.

        Raises:
            ValueError: If the parser frame, ordinal, or source bytes disagree.
            asyncio.CancelledError: If the caller interrupts a database write.
        """
        async with self._lock:
            self._require_capture()
            try:
                if event.sequence != self._parser_event_sequence + 1:
                    raise ValueError("Native CLI parser event sequence must be consecutive.")
                if event.sequence > self.MAX_OBSERVATIONS:
                    raise ValueError("Native CLI parser observation limit exceeded.")
                observation = event.observation
                frame_offset = self._frame_offset(event=event)
                source = (
                    NativeCyberEvidenceSource.MODEL
                    if observation.kind in {NativeCliEventKind.MODEL_MESSAGE, NativeCliEventKind.TOOL_REQUESTED}
                    else NativeCyberEvidenceSource.TOOL
                    if observation.kind
                    in {
                        NativeCliEventKind.TOOL_STARTED,
                        NativeCliEventKind.TOOL_COMPLETED,
                        NativeCliEventKind.TOOL_RESULT,
                    }
                    else NativeCyberEvidenceSource.HARNESS
                )
                await self._append_event_async(
                    source=source,
                    event_type=f"native_cli.{observation.kind.value}",
                    payload={
                        "parser_sequence": event.sequence,
                        "frame_number": event.frame_number,
                        "raw_frame_sha256": self._last_frame_digest if frame_offset is not None else None,
                        "raw_frame_size_bytes": self._last_frame_size if frame_offset is not None else None,
                        "source_message_id": observation.source_message_id,
                        "parent_tool_use_id": observation.parent_tool_use_id,
                        "source_status": observation.source_status,
                        "status": observation.status.value,
                        "name": observation.name,
                        "text": observation.text,
                        "arguments": observation.arguments,
                        "result": observation.result,
                        "exit_code": observation.exit_code,
                        "detail": observation.detail,
                    },
                    source_event_id=observation.source_event_id,
                    source_session_id=observation.source_session_id,
                    observed_stream_id=str(self._streams[NativeCliStream.STDOUT].stream_id)
                    if frame_offset is not None
                    else None,
                    stream_offset=frame_offset,
                    tool_call_id=observation.source_tool_id,
                    tool_phase=self._TOOL_PHASES.get(observation.kind) if observation.source_tool_id else None,
                )
                self._parser_event_sequence += 1
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def read_report_events_async(self) -> tuple[NativeCliReportEvent, ...]:
        """
        Read verified stored observation summaries without reloading raw process frames.

        Returns:
            tuple[NativeCliReportEvent, ...]: Source-ordered metadata for the CLI report adapter.

        Raises:
            ValueError: If stored event order, source or digest metadata was modified.
            RuntimeError: If the turn has not been sealed or a page of DB events is missing.
        """
        async with self._lock:
            if not self._finished:
                raise RuntimeError("CLI observations are not available before the turn is sealed.")
            cursor = 0
            summaries: list[NativeCliReportEvent] = []
            stdout_id = str(self._streams[NativeCliStream.STDOUT].stream_id)
            while cursor < self._controller_sequence:
                page = await asyncio.to_thread(
                    self._store.read_event_payloads,
                    run_id=self._run_id,
                    allow_sensitive=True,
                    after_sequence=cursor,
                    limit=self._store.MAX_EVENT_BATCH,
                )
                if not page:
                    raise RuntimeError("A persisted native CLI event page is missing.")
                for captured in page:
                    event = captured.event
                    if event.sequence != cursor + 1:
                        raise ValueError("Persisted native CLI controller events are not consecutive.")
                    cursor = event.sequence
                    if event.event_type == "native_cli.raw_chunk":
                        continue
                    if event.event_type.startswith(("gateway.", "messages_gateway.")):
                        continue
                    if not event.event_type.startswith("native_cli."):
                        raise ValueError("Foreign events cannot be projected into a native CLI report.")
                    payload = event.payload
                    if (
                        type(payload.get("parser_sequence")) is not int
                        or payload["parser_sequence"] != len(summaries) + 1
                    ):
                        raise ValueError("Persisted native CLI parser event ordinals are not consecutive.")
                    has_frame = payload.get("frame_number") is not None
                    if (has_frame and (event.observed_stream_id != stdout_id or event.stream_offset is None)) or (
                        not has_frame and (event.observed_stream_id is not None or event.stream_offset is not None)
                    ):
                        raise ValueError("CLI frame metadata does not refer to its recorded stdout stream.")
                    summaries.append(
                        NativeCliReportEvent.model_validate(
                            {
                                "sequence": len(summaries) + 1,
                                "frame_number": payload.get("frame_number"),
                                "kind": event.event_type.removeprefix("native_cli."),
                                "status": payload["status"],
                                "source_event_id": event.source_event_id,
                                "source_message_id": payload.get("source_message_id"),
                                "source_session_id": event.source_session_id,
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
                    )
            if len(summaries) != self._parser_event_sequence:
                raise ValueError("Persisted native CLI observations differ from the recorder's count.")
            return tuple(summaries)

    async def finish_async(
        self,
        *,
        outcome: NativeCliRunOutcome | None,
        request_piece_ids: tuple[UUID, ...] = (),
        response_piece_ids: tuple[UUID, ...] = (),
        tool_request_piece_ids: tuple[UUID, ...] = (),
        tool_result_piece_ids: tuple[UUID, ...] = (),
    ) -> None:
        """
        Seal actual sources and link only caller-provided persisted conversation pieces.

        Raises:
            ValueError: If the sink is not open or piece links lack genuine provenance.
            asyncio.CancelledError: If the caller interrupts final capture.
        """
        async with self._lock:
            if not self._started or self._finished:
                raise ValueError("CLI evidence must be started and may finish only once.")
            gaps: list[str] = []
            if outcome is None:
                gaps.append("No native CLI outcome was observed.")
            else:
                if outcome.raw_chunk_count != self._raw_chunk_sequence:
                    gaps.append("Native CLI raw chunk count differs from the recorder.")
                if (
                    outcome.raw_stdout_bytes != self._received[NativeCliStream.STDOUT]
                    or outcome.raw_stderr_bytes != self._received[NativeCliStream.STDERR]
                ):
                    gaps.append("Native CLI byte counts differ from the recorder.")
                if outcome.frame_count != self._last_frame_number:
                    gaps.append("Native CLI frame count differs from the recorder.")
                gaps.extend(outcome.gaps)
            if self._failed:
                gaps.append("Native CLI evidence recording failed.")
            if self._include_model_gateway and (
                not self._gateway_requests or not all(self._gateway_requests.values()) or self._gateway_failed
            ):
                gaps.append("Model gateway request/response coverage is incomplete.")
            source_complete = bool(outcome and outcome.coverage_complete and not gaps)
            try:
                sources = (
                    (self._streams[stream], self._received[stream], self._hashes[stream]) for stream in NativeCliStream
                )
                gateway_sources = (
                    (raw_stream, self._gateway_received[name], self._gateway_hashes[name])
                    for name, raw_stream in self._gateway_streams.items()
                )
                for raw_stream, count, digest in (*sources, *gateway_sources):
                    await asyncio.to_thread(
                        self._store.close_raw_stream,
                        run_id=self._run_id,
                        stream_id=raw_stream.stream_id,
                        source_complete=source_complete,
                        expected_bytes=count,
                        observed_sha256=digest.hexdigest(),
                    )
                await asyncio.to_thread(
                    self._store.finish_turn,
                    finish=NativeCyberTurnFinish(
                        run_id=self._run_id,
                        turn_index=self._turn_index,
                        request_piece_ids=request_piece_ids,
                        response_piece_ids=response_piece_ids,
                        tool_request_piece_ids=tool_request_piece_ids,
                        tool_result_piece_ids=tool_result_piece_ids,
                        observed_event_count=self._controller_sequence,
                        source_complete=source_complete,
                        gaps=tuple(gaps),
                    ),
                )
                self._finished = True
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def _append_event_async(
        self,
        *,
        source: NativeCyberEvidenceSource,
        event_type: str,
        payload: dict[str, JsonValue],
        source_event_id: str | None = None,
        source_session_id: str | None = None,
        observed_stream_id: str | None = None,
        stream_offset: int | None = None,
        tool_call_id: str | None = None,
        tool_phase: NativeCyberToolPhase | None = None,
    ) -> None:
        observed = NativeCyberObservedEvent(
            controller_sequence=self._controller_sequence + 1,
            source_event_id=source_event_id,
            source_session_id=source_session_id,
            event_type=event_type,
            payload=payload,
            observed_stream_id=observed_stream_id,
            stream_offset=stream_offset,
            tool_call_id=tool_call_id,
            tool_phase=tool_phase,
        )
        await asyncio.to_thread(
            self._store.append_events,
            run_id=self._run_id,
            turn_index=self._turn_index,
            events=(NativeCyberCapturedEvent(source=source, event=observed),),
        )
        self._controller_sequence += 1

    def _frame_offset(self, *, event: NativeCliEvent) -> int | None:
        if event.raw_frame is None:
            if event.frame_number is not None:
                raise ValueError("A native CLI frame number requires actual raw frame bytes.")
            return None
        if not event.raw_frame or event.frame_number is None:
            raise ValueError("A native CLI raw frame requires a real frame number and bytes.")
        digest = hashlib.sha256(event.raw_frame).hexdigest()
        if event.frame_number == self._last_frame_number:
            if digest != self._last_frame_digest or len(event.raw_frame) != self._last_frame_size:
                raise ValueError("Repeated native CLI frame observations disagree on their source bytes.")
            return self._last_frame_offset
        if event.frame_number != self._last_frame_number + 1:
            raise ValueError("Native CLI stdout frame numbers must be consecutive.")
        offset = self._next_frame_offset
        if offset + len(event.raw_frame) > self._received[NativeCliStream.STDOUT]:
            raise ValueError("A native CLI frame has no corresponding recorded stdout bytes.")
        self._last_frame_number = event.frame_number
        self._last_frame_digest = digest
        self._last_frame_size = len(event.raw_frame)
        self._last_frame_offset = offset
        self._next_frame_offset += len(event.raw_frame)
        return offset

    def _require_capture(self) -> None:
        if not self._started or self._finished or self._failed:
            raise RuntimeError("Native CLI evidence is not open for capture.")
