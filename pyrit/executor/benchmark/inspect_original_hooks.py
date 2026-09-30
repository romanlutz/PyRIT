# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Default-off, run-scoped Inspect completion hooks for the unchanged runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

from inspect_ai.hooks import Hooks, hooks

from pyrit.models.native_cyber_evidence import (
    NativeCyberEvidenceSource,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from inspect_ai.hooks import (
        RunEnd,
        RunStart,
        SampleAttemptEnd,
        SampleAttemptStart,
        SampleEnd,
        SampleEvent,
        SampleStart,
    )
    from inspect_ai.log import EvalLog

    from pyrit.memory import MemoryInterface
    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore


class InspectOriginalLiveCapture:
    """Retain bounded hook frames; final typed EvalLog remains the authority."""

    KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="inspect-original-live-hooks",
    )
    MAX_BYTES = 2 * 1024 * 1024

    def __init__(self, *, memory: MemoryInterface, episode_id: str) -> None:
        """Set up the owner-scoped buffer without starting a Task or contacting Inspect."""
        self.episode_id = episode_id
        self._capture: NativeCyberEvidenceStore = memory.native_cyber_evidence
        self._stream = NativeCyberRawStreamStart(run_id=episode_id, key=self.KEY)
        self._lock = asyncio.Lock()
        self._inspect_run_id: str | None = None
        self._ended = False
        self._closed = False
        self._stored_bytes = bytearray()
        self._omitted_bytes = 0
        self._gaps: list[str] = []
        self._sample_state: dict[tuple[str, str], tuple[int, int]] = {}
        self._events: dict[tuple[str, str, int, int], list[tuple[str | None, str]]] = {}
        self._sample_ends: set[tuple[str, str, int]] = set()

    @classmethod
    async def begin_async(cls, *, memory: MemoryInterface, episode_id: str) -> InspectOriginalLiveCapture:
        """
        Open a separate optional raw source before Inspect invokes the original Task.

        Returns:
            InspectOriginalLiveCapture: A bounded live source associated with one PyRIT episode.
        """
        capture = cls(memory=memory, episode_id=episode_id)
        await asyncio.to_thread(capture._capture.open_raw_stream, stream=capture._stream)
        return capture

    async def observe_run_start_async(self, *, data: RunStart) -> None:
        """Bind only the first run scheduled in this owner's execution context."""
        async with self._lock:
            if self._inspect_run_id is not None or self._closed:
                return
            self._inspect_run_id = data.run_id
            await self._write_frame_async(
                frame={"kind": "run_start", "run_id": data.run_id, "task_names": data.task_names}
            )

    async def observe_sample_start_async(self, *, data: SampleStart) -> None:
        """Remember the original sample epoch for subsequent completion events."""
        async with self._lock:
            if not self._owns_run(run_id=data.run_id):
                return
            key = (data.eval_id, data.sample_id)
            self._sample_state[key] = (data.summary.epoch, 1)
            await self._write_frame_async(
                frame={
                    "kind": "sample_start",
                    "run_id": data.run_id,
                    "eval_id": data.eval_id,
                    "sample_id": data.sample_id,
                    "epoch": data.summary.epoch,
                }
            )

    async def observe_attempt_start_async(self, *, data: SampleAttemptStart) -> None:
        """Associate a retry attempt with this sample's original event sequence."""
        async with self._lock:
            if not self._owns_run(run_id=data.run_id):
                return
            self._sample_state[(data.eval_id, data.sample_id)] = (data.summary.epoch, data.attempt)
            await self._write_frame_async(
                frame={
                    "kind": "attempt_start",
                    "run_id": data.run_id,
                    "eval_id": data.eval_id,
                    "sample_id": data.sample_id,
                    "epoch": data.summary.epoch,
                    "attempt": data.attempt,
                }
            )

    async def observe_attempt_end_async(self, *, data: SampleAttemptEnd) -> None:
        """Record retry/error status without turning a hook into a policy decision."""
        async with self._lock:
            if not self._owns_run(run_id=data.run_id):
                return
            await self._write_frame_async(
                frame={
                    "kind": "attempt_end",
                    "run_id": data.run_id,
                    "eval_id": data.eval_id,
                    "sample_id": data.sample_id,
                    "epoch": data.summary.epoch,
                    "attempt": data.attempt,
                    "error": data.error is not None,
                    "will_retry": data.will_retry,
                }
            )

    async def observe_sample_event_async(self, *, data: SampleEvent) -> None:
        """Persist only Inspect-completed events, retaining their exact typed fields."""
        async with self._lock:
            if not self._owns_run(run_id=data.run_id):
                return
            state = self._sample_state.get((data.eval_id, data.sample_id))
            epoch, attempt = state if state is not None else (None, None)
            stored = await self._write_frame_async(
                frame={
                    "kind": "event",
                    "run_id": data.run_id,
                    "eval_id": data.eval_id,
                    "sample_id": data.sample_id,
                    "epoch": epoch,
                    "attempt": attempt,
                    "event": data.event.model_dump(mode="json", exclude_none=True),
                }
            )
            if stored and epoch is not None and attempt is not None:
                key = (data.eval_id, data.sample_id, epoch, attempt)
                self._events.setdefault(key, []).append((data.event.uuid, data.event.event))
            elif state is None:
                self._add_gap("Inspect hook event arrived before its sample epoch/attempt was observed.")

    async def observe_sample_end_async(self, *, data: SampleEnd) -> None:
        """Retain completion metadata, never mutate the framework's sample object."""
        async with self._lock:
            if not self._owns_run(run_id=data.run_id):
                return
            self._sample_ends.add((data.eval_id, data.sample_id, data.sample.epoch))
            await self._write_frame_async(
                frame={
                    "kind": "sample_end",
                    "run_id": data.run_id,
                    "eval_id": data.eval_id,
                    "sample_id": data.sample_id,
                    "epoch": data.sample.epoch,
                    "sample_uuid": data.sample.uuid,
                    "event_count": len(data.sample.events),
                    "retry_count": len(data.sample.error_retries or []),
                    "score_count": len(data.sample.scores or {}),
                    "error": data.sample.error is not None,
                }
            )
            self._sample_state.pop((data.eval_id, data.sample_id), None)

    async def observe_run_end_async(self, *, data: RunEnd) -> None:
        """Seal this run's hook source and discard mutable per-sample hook state."""
        async with self._lock:
            if not self._owns_run(run_id=data.run_id):
                return
            await self._write_frame_async(
                frame={"kind": "run_end", "run_id": data.run_id, "error": data.exception is not None}
            )
            self._ended = True
            self._sample_state.clear()
            await self._close_async()

    async def close_async(self) -> None:
        """Seal a partial source too when Inspect never emits its run-end callback."""
        async with self._lock:
            if not self._ended:
                self._add_gap("Inspect live hook did not observe the original run end.")
            await self._close_async()

    def reconcile(self, *, log: EvalLog) -> tuple[str, ...]:
        """
        Compare completed hook source IDs/attempts against the final typed EvalLog.

        Returns:
            tuple[str, ...]: Optional live-observer gaps; the final log still governs import.
        """
        gaps = list(self._gaps)
        if not self._ended or not self._closed or self._inspect_run_id != log.eval.run_id:
            gaps.append("Inspect live hooks did not identify and finish the finalized original run.")
        for sample in log.samples or []:
            if not sample.uuid or (log.eval.eval_id, sample.uuid, sample.epoch) not in self._sample_ends:
                gaps.append("Inspect live sample completion cannot be matched to its final sample UUID.")
            for attempt, retry in enumerate(sample.error_retries or [], start=1):
                key = (log.eval.eval_id, sample.uuid or "", sample.epoch, attempt)
                if self._events.get(key, []) != [(event.uuid, event.event) for event in retry.events or []]:
                    gaps.append("Inspect live retry event source differs from the finalized attempt.")
            final_attempt = len(sample.error_retries or []) + 1
            key = (log.eval.eval_id, sample.uuid or "", sample.epoch, final_attempt)
            if self._events.get(key, []) != [(event.uuid, event.event) for event in sample.events]:
                gaps.append("Inspect live event source differs from the finalized sample.")
        return tuple(dict.fromkeys(gaps))

    def _owns_run(self, *, run_id: str) -> bool:
        return not self._closed and self._inspect_run_id is not None and run_id == self._inspect_run_id

    def _add_gap(self, reason: str) -> None:
        if reason not in self._gaps:
            self._gaps.append(reason)

    async def _write_frame_async(self, *, frame: dict[str, object]) -> bool:
        encoded = json.dumps(frame, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        encoded += b"\n"
        if len(encoded) > self._capture.MAX_APPEND_BYTES or len(self._stored_bytes) + len(encoded) > self.MAX_BYTES:
            self._omitted_bytes += len(encoded)
            self._add_gap("Inspect live hook frames exceeded their bounded optional source quota.")
            return False
        try:
            write = await asyncio.to_thread(
                self._capture.append_raw, run_id=self.episode_id, stream_id=self._stream.stream_id, data=encoded
            )
        except Exception:  # noqa: BLE001 - Inspect downgrades hook exceptions; record the gap before re-raising
            self._add_gap("Inspect live hook could not persist an observed source event.")
            raise
        if write.omitted_bytes:
            self._add_gap("Inspect live hook source was truncated in PyRIT memory.")
            self._omitted_bytes += write.omitted_bytes
            return False
        self._stored_bytes.extend(encoded)
        return True

    async def _close_async(self) -> None:
        if self._closed:
            return
        data = bytes(self._stored_bytes)
        await asyncio.to_thread(
            self._capture.close_raw_stream,
            run_id=self.episode_id,
            stream_id=self._stream.stream_id,
            source_complete=self._ended and not self._gaps,
            expected_bytes=len(data) + self._omitted_bytes,
            observed_sha256=hashlib.sha256(data).hexdigest() if not self._omitted_bytes else None,
            gaps=self._gaps,
        )
        self._closed = True


_ACTIVE_CAPTURE: ContextVar[InspectOriginalLiveCapture | None] = ContextVar(
    "pyrit_inspect_original_live_capture", default=None
)


@contextmanager
def active_original_capture(*, capture: InspectOriginalLiveCapture) -> Iterator[None]:
    """Confine this hook source to the unchanged eval's asyncio execution context."""
    token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(token)


@hooks(name="pyrit_original_inspect_memory", description="Optional run-scoped original Inspect event capture")
class _PyritOriginalInspectHooks(Hooks):
    """Inspect's process-wide singleton delegates only to an explicitly active run."""

    async def on_run_start(self, data: RunStart) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_run_start_async(data=data)

    async def on_sample_start(self, data: SampleStart) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_sample_start_async(data=data)

    async def on_sample_attempt_start(self, data: SampleAttemptStart) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_attempt_start_async(data=data)

    async def on_sample_attempt_end(self, data: SampleAttemptEnd) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_attempt_end_async(data=data)

    async def on_sample_event(self, data: SampleEvent) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_sample_event_async(data=data)

    async def on_sample_end(self, data: SampleEnd) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_sample_end_async(data=data)

    async def on_run_end(self, data: RunEnd) -> None:  # pyrit-async-suffix-exempt
        if capture := _ACTIVE_CAPTURE.get():
            await capture.observe_run_end_async(data=data)
