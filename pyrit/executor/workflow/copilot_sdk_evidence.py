# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One recorder for controller-ordered SDK events and trusted host model observations."""

from __future__ import annotations

import asyncio
import hashlib
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from pyrit.models.native_cyber import NativeAgentEvent, NativeAgentEvidence
from pyrit.models.native_cyber_evidence import (
    NativeCyberCapturedEvent,
    NativeCyberEvidenceSource,
    NativeCyberObservedEvent,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
    NativeCyberResponseMode,
    NativeCyberTurnFinish,
    NativeCyberTurnStart,
)
from pyrit.models.native_harness import NativeModelCaptureIdentity, NativeModelResponseOrigin

if TYPE_CHECKING:
    from collections.abc import Callable
    from datetime import datetime
    from uuid import UUID

    from pydantic import JsonValue

    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore

ResultT = TypeVar("ResultT")


CopilotModelResponseOrigin = NativeModelResponseOrigin
CopilotSdkCaptureIdentity = NativeModelCaptureIdentity


@dataclass(frozen=True, kw_only=True)
class CopilotSdkModelObserver:
    """A turn-bound seam for a trusted host transport; it contains no credentials or client."""

    identity: CopilotSdkCaptureIdentity
    _sink: CopilotSdkDatabaseEvidenceSink = field(repr=False)

    @property
    def max_body_bytes(self) -> int:
        """The existing store's maximum original-body capture size."""
        return self._sink._store.MAX_APPEND_BYTES

    def ensure_active(self) -> None:
        """
        Check the bound turn again immediately before an irreversible transport boundary.

        Raises:
            RuntimeError: If capture failed or the turn has closed.
            ValueError: If this observer belongs to a different active turn.
        """
        with self._sink._state_lock:
            self._sink._require_open(self.identity)

    def fail(self, reason: str) -> None:
        """Latch a content-free transport failure for the controller's next durable checkpoint."""
        self._sink.fail(reason)

    async def record_request_async(self, *, request_id: str, body: bytes, accepted: bool) -> None:
        """Await durable original request capture BEFORE upstream dispatch; rejected ingress is not model traffic."""
        await self._sink.record_model_request_async(
            identity=self.identity, request_id=request_id, request_bytes=body, accepted=accepted
        )

    async def record_response_async(
        self,
        *,
        request_id: str,
        body: bytes,
        status_code: int,
        origin: CopilotModelResponseOrigin,
        complete: bool,
    ) -> None:
        """Await durable original response capture BEFORE returning it to the SDK, with explicit origin/completeness."""
        await self._sink.record_model_response_async(
            identity=self.identity,
            request_id=request_id,
            response_bytes=body,
            status_code=status_code,
            origin=origin,
            complete=complete,
        )


@dataclass(frozen=True, kw_only=True)
class _Observation:
    stream: str
    source: NativeCyberEvidenceSource
    event_type: str
    event_id: str
    payload: dict[str, JsonValue] = field(repr=False)
    raw: bytes = field(repr=False)
    sdk_event: NativeAgentEvent | None = field(default=None, repr=False)


class _Turn:
    def __init__(
        self,
        *,
        identity: CopilotSdkCaptureIdentity,
        keys: dict[str, NativeCyberRawStreamKey],
        event_start: int,
        sdk_start: int,
    ) -> None:
        self.identity = identity
        self.event_start, self.sdk_start = event_start, sdk_start
        self.streams = {
            name: NativeCyberRawStreamStart(run_id=identity.run_id, turn_index=identity.turn_index, key=key)
            for name, key in keys.items()
        }
        self.opened: set[str] = set()
        self.received = dict.fromkeys(keys, 0)
        self.digests = {name: hashlib.sha256() for name in keys}
        self.requests: dict[str, bool] = {}
        self.completed_requests = 0
        self.idle_observed = False
        self.sealing = False


class CopilotSdkDatabaseEvidenceSink:
    """Persist all required sources to the existing store; never dispatch, grade, or publish."""

    _KEYS = {
        "sdk": NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.HARNESS,
            kind=NativeCyberRawKind.JSONL,
            observed_source_id="ghcp.sdk.events",
        ),
        "request": NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.MODEL,
            kind=NativeCyberRawKind.MODEL,
            observed_source_id="ghcp.model.requests",
        ),
        "response": NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.MODEL,
            kind=NativeCyberRawKind.MODEL,
            observed_source_id="ghcp.model.responses",
        ),
        "rejected": NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.HARNESS,
            kind=NativeCyberRawKind.MODEL,
            observed_source_id="ghcp.model.rejected_requests",
        ),
        "host_response": NativeCyberRawStreamKey(
            source=NativeCyberEvidenceSource.HARNESS,
            kind=NativeCyberRawKind.MODEL,
            observed_source_id="ghcp.model.host_responses",
        ),
    }
    MAX_SDK_EVENTS = 10_000
    MAX_MODEL_REQUESTS = 32
    MAX_PENDING_BYTES = 8_388_608

    def __init__(
        self,
        *,
        store: NativeCyberEvidenceStore,
        run_id: str,
        session_id: str,
        environment_id: str | None = None,
        harness_id: str | None = None,
    ) -> None:
        """
        Bind a fresh run recorder to an already-created episode and actual SDK session.

        Raises:
            ValueError: If identities are not explicit and bounded.
        """
        if any(not 0 < len(value) <= 128 for value in (run_id, session_id)):
            raise ValueError("Unified capture requires bounded run and observed session identities.")
        self.run_id, self.session_id = run_id, session_id
        self._environment_id, self._harness_id = environment_id, harness_id
        self._store = store
        self._turn: _Turn | None = None
        self._turn_count = 0
        self._turn_ids: set[str] = set()
        self._event_ids: set[str] = set()
        self._request_ids: set[str] = set()
        self._sdk_events: dict[int, NativeAgentEvent] = {}
        self._sdk_to_controller: dict[int, int] = {}
        self._events: list[NativeAgentEvent] = []
        self._pending: deque[_Observation] = deque()
        self._pending_bytes = 0
        self._failed = False
        self._write_failed = False
        self._gaps: list[str] = []
        self._marked_gaps = 0
        self._state_lock = threading.RLock()
        self._write_lock = asyncio.Lock()

    @classmethod
    def required_raw_streams(cls) -> tuple[NativeCyberRawStreamKey, ...]:
        """Return the three source identities the controller must declare before dispatch."""
        return tuple(cls._KEYS[name] for name in ("sdk", "request", "response"))

    @property
    def turn_open(self) -> bool:
        """Whether an outer turn can still be sealed once, including after capture failure."""
        return self._turn is not None and not self._turn.sealing

    async def start_turn_async(
        self, *, turn_id: str, turn_index: int, started_at: datetime, response_mode: NativeCyberResponseMode
    ) -> CopilotSdkModelObserver:
        """
        Open exactly one next turn and every raw source before task delivery.

        Returns:
            CopilotSdkModelObserver: A trusted transport seam bound to this exact turn.

        Raises:
            ValueError: If turn identity/order or previous capture state is invalid.
            asyncio.CancelledError: If source creation is interrupted.
        """
        with self._state_lock:
            if (
                self._turn is not None
                or self._failed
                or self._gaps
                or not turn_id
                or len(turn_id) > 128
                or turn_id in self._turn_ids
                or type(turn_index) is not int
                or turn_index != self._turn_count + 1
            ):
                self.fail("Unified capture cannot open a reused, unordered or unhealthy turn.")
                raise ValueError("Unified capture requires a fresh consecutive turn after complete capture.")
            identity = CopilotSdkCaptureIdentity(
                run_id=self.run_id,
                session_id=self.session_id,
                turn_id=turn_id,
                turn_index=turn_index,
                environment_id=self._environment_id,
                harness_id=self._harness_id,
            )
            turn = _Turn(
                identity=identity, keys=self._KEYS, event_start=len(self._events), sdk_start=len(self._sdk_events)
            )
            self._turn = turn
            self._turn_ids.add(turn_id)
            self._turn_count = turn_index
        try:
            await self._store_write_async(
                self._store.begin_turn,
                turn=NativeCyberTurnStart(
                    run_id=self.run_id,
                    turn_index=turn_index,
                    source_turn_id=turn_id,
                    started_at=started_at,
                    response_mode=response_mode,
                ),
            )
            for name, stream in turn.streams.items():
                await self._store_write_async(self._store.open_raw_stream, stream=stream)
                turn.opened.add(name)
        except (Exception, asyncio.CancelledError):
            self.fail("Opening unified evidence sources failed.")
            raise
        return CopilotSdkModelObserver(identity=identity, _sink=self)

    def record_sdk_event(self, event: NativeAgentEvent) -> None:
        """
        Queue the detached SDK envelope synchronously in observed callback order.

        The next model boundary or session flush durably drains this same queue.

        Raises:
            ValueError: If the SDK ID/session/sequence or event shape is invalid.
            RuntimeError: If the turn is not open or capture already failed.
        """
        with self._state_lock:
            turn = self._require_open()
            if (
                event.session_id != self.session_id
                or event.event_id in self._event_ids
                or event.sequence != len(self._event_ids) + 1
                or len(self._event_ids) >= self.MAX_SDK_EVENTS
                or event.payload.get("id") != event.event_id
                or event.payload.get("type") != event.event_type
                or event.payload.get("sessionId", self.session_id) != self.session_id
                or not isinstance(event.payload.get("data"), dict)
            ):
                self.fail("SDK event identity, order or shape disagrees with unified capture.")
                raise ValueError("SDK event lacks a unique matching identity and sequence.")
            raw = (json.dumps(event.payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
            self._enqueue(
                _Observation(
                    stream="sdk",
                    event_type=event.event_type,
                    event_id=event.event_id,
                    source=NativeCyberEvidenceSource.MODEL
                    if event.event_type.startswith(("assistant.", "model."))
                    else NativeCyberEvidenceSource.TOOL
                    if event.event_type.startswith("tool.")
                    else NativeCyberEvidenceSource.HARNESS,
                    payload=json.loads(raw),
                    raw=raw,
                    sdk_event=event.model_copy(deep=True),
                )
            )
            self._event_ids.add(event.event_id)
            if event.event_type == "session.idle" and not event.payload.get("agentId"):
                data = event.payload["data"]
                turn.idle_observed = isinstance(data, dict) and data.get("aborted") is not True
            elif event.event_type.startswith(("tool.", "assistant.message", "session.error")):
                turn.idle_observed = False
            if event.event_type == "session.error":
                self._gaps.append("The SDK reported a session error.")

    async def record_model_request_async(
        self, *, identity: CopilotSdkCaptureIdentity, request_id: str, request_bytes: bytes, accepted: bool
    ) -> None:
        """
        Commit original request bytes before the trusted transport may dispatch.

        Raises:
            ValueError: If identity, request ID, origin or bytes are invalid.
        """
        with self._state_lock:
            turn = self._require_open(identity)
            if (
                not isinstance(request_id, str)
                or not 0 < len(request_id) <= 128
                or request_id in self._request_ids
                or len(self._request_ids) >= self.MAX_MODEL_REQUESTS
                or type(accepted) is not bool
            ):
                self.fail("Model request identity or admission is invalid.")
                raise ValueError("Model requests require a unique observed ID and explicit admission.")
            name = "request" if accepted else "rejected"
            self._enqueue_model(
                name=name,
                request_id=request_id,
                raw=request_bytes,
                event_type="ghcp.model.request" if accepted else "ghcp.model.rejected_request",
                metadata={"accepted": accepted},
            )
            self._request_ids.add(request_id)
            turn.requests[request_id] = accepted
            turn.idle_observed = False
            if not accepted:
                self._gaps.append("A host rejected a model request before dispatch.")
        await self.flush_async()

    async def record_model_response_async(
        self,
        *,
        identity: CopilotSdkCaptureIdentity,
        request_id: str,
        response_bytes: bytes,
        status_code: int,
        origin: CopilotModelResponseOrigin,
        complete: bool,
    ) -> None:
        """
        Commit original provider bytes, or separately tagged host/synthetic bytes, before return.

        Raises:
            ValueError: If a response is unpaired, misidentified or has an invalid origin/status.
        """
        with self._state_lock:
            turn = self._require_open(identity)
            if (
                not isinstance(request_id, str)
                or request_id not in turn.requests
                or not isinstance(origin, CopilotModelResponseOrigin)
                or type(status_code) is not int
                or not 200 <= status_code <= 599
                or type(complete) is not bool
                or (not turn.requests[request_id] and origin is CopilotModelResponseOrigin.PROVIDER)
            ):
                self.fail("Model response identity, pairing or origin is invalid.")
                raise ValueError("Model response requires a matching request and explicit valid origin.")
            provider = origin is CopilotModelResponseOrigin.PROVIDER
            self._enqueue_model(
                name="response" if provider else "host_response",
                request_id=request_id,
                raw=response_bytes,
                event_type=f"ghcp.model.{origin.value}_response" if not provider else "ghcp.model.response",
                metadata={"origin": origin.value, "status_code": status_code, "complete": complete},
            )
            del turn.requests[request_id]
            turn.idle_observed = False
            if provider and complete and status_code == 200:
                turn.completed_requests += 1
            else:
                self._gaps.append("Model response was non-original, unsuccessful or incomplete.")
        await self.flush_async()

    async def flush_async(self) -> None:
        """
        Drain observed sources before a transport proceeds; any capture failure blocks progress.

        Raises:
            RuntimeError: If durable capture previously failed.
        """
        await self._drain_async(tolerate_failure=False)
        await self._persist_gaps_async()
        if self._failed:
            raise RuntimeError("Unified source capture failed; model or SDK progress is blocked.")

    def fail(self, reason: str) -> None:
        """Latch a bounded content-free failure for the next durable checkpoint."""
        with self._state_lock:
            self._failed = True
            self._gaps.append(reason[:512])

    async def checkpoint_async(self) -> None:
        """Persist irreversible capture gaps even after a turn was sealed; no failed write is retried."""
        await self._drain_async(tolerate_failure=True)
        await self._persist_gaps_async()

    async def finish_turn_async(
        self,
        *,
        evidence: NativeAgentEvidence,
        request_piece_ids: tuple[UUID, ...],
        response_piece_ids: tuple[UUID, ...],
        tool_request_piece_ids: tuple[UUID, ...],
        tool_result_piece_ids: tuple[UUID, ...],
        interrupted: bool = False,
    ) -> None:
        """
        Seal sources once and link authentic persisted PyRIT pieces to the already-open turn.

        Raises:
            RuntimeError: If there is no turn to finish or it was already sealed.
            asyncio.CancelledError: If sealing is interrupted; complete capture is not claimed.
        """
        with self._state_lock:
            turn = self._turn
            if turn is None or turn.sealing:
                raise RuntimeError("Unified capture can finish an opened turn only once.")
            turn.sealing = True
        await self.checkpoint_async()
        projected = self.project_evidence(evidence)
        gaps = list(projected.gaps)
        if interrupted:
            gaps.append("Unified task delivery was interrupted.")
        if not evidence.idle or not turn.idle_observed or len(self._sdk_events) <= turn.sdk_start:
            gaps.append("The unified turn has no verified SDK root idle and event count.")
        if turn.requests or not turn.completed_requests:
            gaps.append("The unified turn lacks a complete paired original model exchange.")
        self._gaps.extend(gaps)
        complete = not self._failed and not gaps
        try:
            for name in turn.opened:
                await self._store_write_async(
                    self._store.close_raw_stream,
                    run_id=self.run_id,
                    stream_id=turn.streams[name].stream_id,
                    source_complete=complete,
                    expected_bytes=turn.received[name],
                    observed_sha256=turn.digests[name].hexdigest(),
                )
            await self._store_write_async(
                self._store.finish_turn,
                finish=NativeCyberTurnFinish(
                    run_id=self.run_id,
                    turn_index=turn.identity.turn_index,
                    request_piece_ids=request_piece_ids,
                    response_piece_ids=response_piece_ids,
                    tool_request_piece_ids=tool_request_piece_ids,
                    tool_result_piece_ids=tool_result_piece_ids,
                    observed_event_count=len(self._events) - turn.event_start,
                    source_complete=complete,
                    gaps=tuple(dict.fromkeys(gaps)),
                ),
            )
        except (Exception, asyncio.CancelledError):
            self.fail("Sealing unified evidence failed.")
            raise
        finally:
            self._turn = None
        await self._persist_gaps_async()

    def project_evidence(self, evidence: NativeAgentEvidence) -> NativeAgentEvidence:
        """
        Map observed SDK ordinals to the unified controller order without changing SDK payloads.

        Returns:
            NativeAgentEvidence: Canonical report evidence matching the one DB event ledger.
        """
        with self._state_lock:
            gaps = [*evidence.gaps, *self._gaps]
            if (
                evidence.session_id != self.session_id
                or len(evidence.events) != len(self._sdk_events)
                or any(self._sdk_events.get(event.sequence) != event for event in evidence.events)
            ):
                gaps.append("SDK evidence differs from the unified recorder's persisted watermark.")
            if self._failed:
                gaps.append("Unified durable evidence capture failed.")
            mapping = self._sdk_to_controller
            requests = tuple(
                request.model_copy(
                    update={"request_sequence": mapping.get(request.request_sequence, request.request_sequence)}
                )
                for request in evidence.tool_requests
            )
            tools = tuple(
                tool.model_copy(
                    update={
                        "request_sequence": mapping.get(tool.request_sequence, tool.request_sequence),
                        "start_sequence": mapping.get(tool.start_sequence, tool.start_sequence),
                        "completion_sequence": mapping.get(tool.completion_sequence, tool.completion_sequence),
                    }
                )
                for tool in evidence.tools
            )
            return evidence.model_copy(
                update={
                    "events": tuple(self._events),
                    "tools": tools,
                    "tool_requests": requests,
                    "scope": "native_sdk_and_model",
                    "gaps": tuple(dict.fromkeys(gaps)),
                    "coverage_complete": evidence.coverage_complete and not gaps,
                    "provenance": {**evidence.provenance, "evidence_contract": "copilot_sdk_and_model"},
                },
                deep=True,
            )

    def _require_open(self, identity: CopilotSdkCaptureIdentity | None = None) -> _Turn:
        turn = self._turn
        if self._failed or turn is None or turn.sealing:
            self.fail("A source observation arrived outside an open healthy evidence turn.")
            raise RuntimeError("Unified recorder is not open for source capture.")
        if identity is not None and identity != turn.identity:
            self.fail("A model observation belongs to a different run, session or turn.")
            raise ValueError("Model observation identity does not match the active evidence turn.")
        return turn

    def _enqueue_model(
        self, *, name: str, request_id: str, raw: bytes, event_type: str, metadata: dict[str, JsonValue]
    ) -> None:
        if not isinstance(raw, bytes) or not 0 < len(raw) <= self._store.MAX_APPEND_BYTES:
            self.fail("Model source bytes are missing, invalid or over the capture limit.")
            raise ValueError("Model source capture requires bounded original bytes.")
        self._enqueue(
            _Observation(
                stream=name,
                source=self._KEYS[name].source,
                event_id=request_id,
                event_type=event_type,
                raw=raw,
                payload={
                    "sdk_request_id": request_id,
                    "frame_size_bytes": len(raw),
                    "frame_sha256": hashlib.sha256(raw).hexdigest(),
                    **metadata,
                },
            )
        )

    def _enqueue(self, observation: _Observation) -> None:
        if (
            not isinstance(observation.raw, bytes)
            or not 0 < len(observation.raw) <= self._store.MAX_APPEND_BYTES
            or self._pending_bytes + len(observation.raw) > self.MAX_PENDING_BYTES
        ):
            self.fail("Source bytes exceed the bounded unified capture queue.")
            raise ValueError("Unified source capture requires bounded original bytes.")
        self._pending.append(observation)
        self._pending_bytes += len(observation.raw)

    async def _drain_async(self, *, tolerate_failure: bool) -> None:
        async with self._write_lock:
            if self._write_failed:
                if tolerate_failure:
                    return
                raise RuntimeError("Unified source capture has an earlier write failure.")
            while True:
                with self._state_lock:
                    if not self._pending:
                        break
                    observation = self._pending.popleft()
                    self._pending_bytes -= len(observation.raw)
                    turn = self._turn
                assert turn is not None
                try:
                    await self._write_observation_async(turn=turn, observation=observation)
                except (Exception, asyncio.CancelledError) as error:
                    self._write_failed = True
                    self.fail("A unified source database write failed or lost bytes.")
                    if not tolerate_failure or isinstance(error, asyncio.CancelledError):
                        raise
                    return

    async def _write_observation_async(self, *, turn: _Turn, observation: _Observation) -> None:
        stream = turn.streams[observation.stream]
        offset = turn.received[observation.stream]
        write = await self._store_write_async(
            self._store.append_raw, run_id=self.run_id, stream_id=stream.stream_id, data=observation.raw
        )
        turn.digests[observation.stream].update(observation.raw)
        turn.received[observation.stream] += len(observation.raw)
        if (
            write.stream_id != stream.stream_id
            or write.received_bytes != len(observation.raw)
            or write.stored_bytes != len(observation.raw)
            or write.omitted_bytes
            or write.truncated
        ):
            raise RuntimeError("Durable unified source write was truncated, omitted or misattributed.")
        event = NativeAgentEvent(
            sequence=len(self._events) + 1,
            event_id=observation.event_id,
            session_id=self.session_id,
            event_type=observation.event_type,
            payload=observation.payload,
        )
        await self._store_write_async(
            self._store.append_events,
            run_id=self.run_id,
            turn_index=turn.identity.turn_index,
            events=(
                NativeCyberCapturedEvent(
                    source=observation.source,
                    event=NativeCyberObservedEvent(
                        controller_sequence=event.sequence,
                        source_event_id=event.event_id,
                        source_session_id=self.session_id,
                        event_type=event.event_type,
                        payload=event.payload,
                        observed_stream_id=str(stream.stream_id),
                        stream_offset=offset,
                    ),
                ),
            ),
        )
        with self._state_lock:
            self._events.append(event)
            if observation.sdk_event is not None:
                self._sdk_events[observation.sdk_event.sequence] = observation.sdk_event
                self._sdk_to_controller[observation.sdk_event.sequence] = event.sequence

    async def _persist_gaps_async(self) -> None:
        async with self._write_lock:
            while self._marked_gaps < len(self._gaps):
                await self._store_write_async(
                    self._store.mark_capture_gap, run_id=self.run_id, reason=self._gaps[self._marked_gaps]
                )
                self._marked_gaps += 1

    async def _store_write_async(self, operation: Callable[..., ResultT], **kwargs: Any) -> ResultT:
        task = asyncio.create_task(asyncio.to_thread(operation, **kwargs))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancellation:
            self.fail("A unified evidence write was interrupted.")
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    continue
                except Exception as error:
                    raise cancellation from error
            try:
                task.result()
            except Exception as error:
                raise cancellation from error
            raise
