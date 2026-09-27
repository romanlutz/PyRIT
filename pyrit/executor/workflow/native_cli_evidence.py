# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Durably record a sandboxed coding CLI without assigning invented provider IDs."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING

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
from pyrit.prompt_target.native_cli_models import NativeCliEventKind, NativeCliProtocol, NativeCliStream

if TYPE_CHECKING:
    from datetime import datetime
    from uuid import UUID

    from pydantic import JsonValue

    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore
    from pyrit.prompt_target.native_cli_models import NativeCliEvent, NativeCliRawChunk, NativeCliRunOutcome


class NativeCliDatabaseEvidenceSink:
    """Record exact stdout/stderr and ordered CLI observations in an existing DB episode."""

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
        turn_index: int,
        protocol: NativeCliProtocol,
    ) -> None:
        """
        Bind a sink to one caller-created episode and one outer turn.

        Raises:
            ValueError: If the protocol, run ID or outer-turn index is invalid.
        """
        if not isinstance(protocol, NativeCliProtocol):
            raise ValueError("A documented native coding CLI protocol is required.")
        if not isinstance(run_id, str) or not 0 < len(run_id) <= 128 or type(turn_index) is not int or turn_index < 1:
            raise ValueError("Native CLI capture needs an existing run and positive turn index.")
        self._store = store
        self._run_id = run_id
        self._turn_index = turn_index
        self._streams = {
            stream: NativeCyberRawStreamStart(run_id=run_id, turn_index=turn_index, key=key)
            for stream, key in zip(NativeCliStream, self.required_raw_streams(protocol=protocol), strict=True)
        }
        self._received = dict.fromkeys(NativeCliStream, 0)
        self._hashes = {stream: hashlib.sha256() for stream in NativeCliStream}
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

    @staticmethod
    def required_raw_streams(*, protocol: NativeCliProtocol) -> tuple[NativeCyberRawStreamKey, ...]:
        """
        Declare both CLI pipes in the episode manifest before the process starts.

        Returns:
            tuple[NativeCyberRawStreamKey, ...]: Source-tagged stdout and stderr identities.

        Raises:
            ValueError: If the CLI protocol is not supported.
        """
        if not isinstance(protocol, NativeCliProtocol):
            raise ValueError("A documented native coding CLI protocol is required.")
        return tuple(
            NativeCyberRawStreamKey(
                source=NativeCyberEvidenceSource.HARNESS,
                kind=NativeCyberRawKind.STDOUT if stream is NativeCliStream.STDOUT else NativeCyberRawKind.STDERR,
                observed_source_id=f"{protocol.value}.{stream.value}",
            )
            for stream in NativeCliStream
        )

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
                        started_at=started_at,
                        response_mode=response_mode,
                    ),
                )
                for stream in NativeCliStream:
                    await asyncio.to_thread(self._store.open_raw_stream, stream=self._streams[stream])
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

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
            source_complete = bool(outcome and outcome.coverage_complete and not gaps)
            try:
                for stream in NativeCliStream:
                    raw_stream = self._streams[stream]
                    await asyncio.to_thread(
                        self._store.close_raw_stream,
                        run_id=self._run_id,
                        stream_id=raw_stream.stream_id,
                        source_complete=source_complete,
                        expected_bytes=self._received[stream],
                        observed_sha256=self._hashes[stream].hexdigest(),
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
