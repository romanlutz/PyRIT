# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Record sandbox process bytes before decoding provider JSONL; never run host tools."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Protocol, cast

from pyrit.prompt_target.native_cli_adapters import ClaudePrintStreamJsonAdapter, CodexExecJsonAdapter, NativeCliAdapter
from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliEventKind,
    NativeCliEventStatus,
    NativeCliProcessChunk,
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliRunConfig,
    NativeCliRunOutcome,
    NativeCliStream,
)
from pyrit.prompt_target.native_cli_models import NativeCliObservation as Observation

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pydantic import JsonValue


class SandboxProcessSession(Protocol):
    """A process already confined to a caller-owned sandbox, never a host subprocess."""

    def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        """Yield stdout and stderr byte chunks in observed read order."""
        ...

    async def wait_async(self) -> int:
        """Wait for the actual sandbox process exit status."""
        ...

    async def stop_async(self) -> None:
        """Stop and reap the sandbox process, including on cancellation."""
        ...


class SandboxProcessLauncher(Protocol):
    """Launch only the pinned CLI/profile in the owned sandbox, using argv, not a shell."""

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> SandboxProcessSession:
        """Launch a sandbox process with caller-controlled isolation and model routing."""
        ...


class NativeCliEvidenceSink(Protocol):
    """Caller-owned durable recorder; these methods must not execute provider tools."""

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        """Commit the complete byte chunk before returning, without truncation."""
        ...

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        """Commit an ordered observation or coverage gap before returning."""
        ...


class NativeCliStreamLimitError(RuntimeError):
    """The observed step or JSONL frame budget was exceeded after raw recording."""


class NativeCliJsonlParser:
    """Frame JSONL without losing the source bytes; correlate only observed lifecycle events."""

    _CODEX_ITEM_KINDS = frozenset(
        {
            NativeCliEventKind.MODEL_MESSAGE,
            NativeCliEventKind.TOOL_STARTED,
            NativeCliEventKind.TOOL_COMPLETED,
            NativeCliEventKind.TOOL_RESULT,
            NativeCliEventKind.PROGRESS,
            NativeCliEventKind.AUXILIARY,
        }
    )
    _PROVIDER_WORK_KINDS = frozenset(
        {
            NativeCliEventKind.TURN_STARTED,
            NativeCliEventKind.TURN_COMPLETED,
            NativeCliEventKind.RUN_FINISHED,
            NativeCliEventKind.MODEL_MESSAGE,
            NativeCliEventKind.TOOL_REQUESTED,
            NativeCliEventKind.TOOL_STARTED,
            NativeCliEventKind.TOOL_COMPLETED,
            NativeCliEventKind.TOOL_RESULT,
            NativeCliEventKind.PROGRESS,
        }
    )

    def __init__(self, *, config: NativeCliRunConfig) -> None:
        """Bind one documented profile and its budgets to a fresh parser."""
        self._config = config
        self._adapter: NativeCliAdapter = (
            CodexExecJsonAdapter()
            if config.protocol is NativeCliProtocol.CODEX_EXEC_JSON
            else ClaudePrintStreamJsonAdapter()
        )
        self._buffer = bytearray()
        self._frame_count = 0
        self._event_count = 0
        self._source_session_id: str | None = None
        self._turn_open = False
        self._terminal = False
        self._finished = False
        self._steps = 0
        self._message_ids: set[str] = set()
        self._open_tools: dict[tuple[str | None, str], NativeCliEventKind] = {}
        self._completed_tools: set[tuple[str | None, str]] = set()
        self._gaps: list[str] = []
        self._fatal_reason: str | None = None

    @property
    def frame_count(self) -> int:
        """The number of stdout frames examined."""
        return self._frame_count

    @property
    def event_count(self) -> int:
        """The number of provider observations and explicit diagnostics emitted."""
        return self._event_count

    @property
    def source_session_id(self) -> str | None:
        """The observed Codex thread ID or Claude session ID."""
        return self._source_session_id

    @property
    def observed_steps(self) -> int:
        """The number of observed Codex turns or distinct Claude assistant messages."""
        return self._steps

    @property
    def terminal_observed(self) -> bool:
        """Whether the last observed provider turn had an explicit terminal event."""
        return self._terminal

    @property
    def coverage_complete(self) -> bool:
        """Whether documented boundaries and tool lifecycles were observed without gaps."""
        return self._finished and self._terminal and not self._gaps

    @property
    def gaps(self) -> tuple[str, ...]:
        """The reasons normalized evidence is not fully covered."""
        return tuple(self._gaps)

    @property
    def fatal_reason(self) -> str | None:
        """The explicit limit error requiring the sandbox process to stop."""
        return self._fatal_reason

    def feed(self, *, data: bytes) -> tuple[NativeCliEvent, ...]:
        """
        Decode complete stdout frames without trimming raw bytes.

        Returns:
            tuple[NativeCliEvent, ...]: All observations and coverage gaps from complete frames.

        Raises:
            RuntimeError: If the parser was finished or exceeded a configured limit.
        """
        if self._finished or self._fatal_reason:
            raise RuntimeError("Cannot feed a completed or aborted native CLI JSONL parser.")
        events: list[NativeCliEvent] = []
        offset = 0
        while offset < len(data):
            newline = data.find(b"\n", offset)
            end = len(data) if newline < 0 else newline + 1
            if len(self._buffer) + end - offset > self._config.max_frame_bytes:
                reason = (
                    "Unterminated native CLI JSONL frame exceeds max_frame_bytes."
                    if newline < 0
                    else "Native CLI JSONL frame exceeds max_frame_bytes."
                )
                events.append(self._limit(reason=reason))
                self._buffer.clear()
                break
            self._buffer.extend(data[offset:end])
            offset = end
            if newline < 0:
                break
            frame = bytes(self._buffer)
            self._buffer.clear()
            events.extend(self._decode_frame(raw_frame=frame))
            if self._fatal_reason:
                break
        return tuple(events)

    def finish(self) -> tuple[NativeCliEvent, ...]:
        """
        Parse the final unterminated line, report missing boundaries, and mark actual EOF.

        Returns:
            tuple[NativeCliEvent, ...]: Last observations, explicit coverage gaps, and EOF.

        Raises:
            RuntimeError: If the parser was already finished or exceeded a limit.
        """
        if self._finished or self._fatal_reason:
            raise RuntimeError("Cannot finish a completed or aborted native CLI JSONL parser.")
        events = list(self._decode_frame(raw_frame=bytes(self._buffer))) if self._buffer else []
        self._buffer.clear()
        if self._turn_open:
            events.append(self._diagnostic(detail="Codex turn.started has no terminal turn event at EOF."))
        for (parent_id, tool_id), phase in self._open_tools.items():
            events.append(
                self._diagnostic(
                    detail=f"Observed {phase.value} for tool {tool_id} has no completion/result at EOF.",
                    tool_id=tool_id,
                    parent_id=parent_id,
                )
            )
        if self._source_session_id is None:
            events.append(self._diagnostic(detail="No documented native CLI session start was observed."))
        if not self._terminal:
            events.append(self._diagnostic(detail="No successful provider terminal event was observed before EOF."))
        events.append(self._event(observation=Observation(kind=NativeCliEventKind.EOF, detail="Stdout reached EOF.")))
        self._finished = True
        return tuple(events)

    def abort(self, *, detail: str) -> NativeCliEvent:
        """
        Record an explicit transport/process failure without a synthetic EOF.

        Returns:
            NativeCliEvent: A failed event with the supplied reason.
        """
        return self._event(
            observation=Observation(kind=NativeCliEventKind.ERROR, status=NativeCliEventStatus.FAILED, detail=detail)
        )

    def _decode_frame(self, *, raw_frame: bytes) -> tuple[NativeCliEvent, ...]:
        self._frame_count += 1
        try:
            decoded = json.loads(raw_frame.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            detail = f"Invalid native CLI JSONL frame {self._frame_count}: {error.__class__.__name__}."
            return (self._event(observation=Observation(kind=NativeCliEventKind.ERROR, detail=detail), raw=raw_frame),)
        if not isinstance(decoded, dict):
            return (self._diagnostic(detail="Native CLI JSONL frame is not an object.", raw=raw_frame),)
        payload = cast("dict[str, JsonValue]", decoded)
        events: list[NativeCliEvent] = []
        for observation in self._adapter.decode(payload=payload):
            events.extend(self._observe(observation=observation, raw=raw_frame))
        return tuple(events)

    def _observe(self, *, observation: Observation, raw: bytes) -> tuple[NativeCliEvent, ...]:
        session_id = observation.source_session_id
        if session_id is not None and self._source_session_id is not None and session_id != self._source_session_id:
            return (self._diagnostic(detail="Native CLI frame belongs to a different session.", raw=raw),)
        events = [self._event(observation=observation, raw=raw)]
        events.extend(self._order_gaps(observation=observation, raw=raw))
        kind = observation.kind
        if kind is NativeCliEventKind.SESSION_STARTED:
            if self._source_session_id is not None:
                events.append(self._diagnostic(detail="Native CLI session start was repeated.", raw=raw))
            else:
                self._source_session_id = session_id
        elif kind is NativeCliEventKind.TURN_STARTED:
            if self._turn_open:
                events.append(self._diagnostic(detail="Codex turn started before the previous turn ended.", raw=raw))
            self._turn_open, self._terminal = True, False
            self._steps += 1
        elif kind is NativeCliEventKind.TURN_COMPLETED:
            if not self._turn_open:
                events.append(self._diagnostic(detail="Codex turn completed without a start.", raw=raw))
            self._turn_open, self._terminal = False, True
        elif kind is NativeCliEventKind.RUN_FINISHED:
            if self._terminal:
                events.append(self._diagnostic(detail="Native CLI reported more than one final result.", raw=raw))
            self._terminal = True
        elif kind in {NativeCliEventKind.TOOL_REQUESTED, NativeCliEventKind.TOOL_STARTED}:
            events.extend(self._tool_open(observation=observation, raw=raw))
        elif kind in {NativeCliEventKind.TOOL_COMPLETED, NativeCliEventKind.TOOL_RESULT}:
            events.extend(self._tool_close(observation=observation, raw=raw))
        if self._config.protocol is NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE:
            self._count_claude_message(observation=observation)
        if self._steps > self._config.max_steps and not self._fatal_reason:
            events.append(self._limit(reason="Native CLI observed step budget exceeded.", raw=raw))
        return tuple(events)

    def _order_gaps(self, *, observation: Observation, raw: bytes) -> tuple[NativeCliEvent, ...]:
        kind = observation.kind
        events: list[NativeCliEvent] = []
        if self._source_session_id is None and (
            kind in self._PROVIDER_WORK_KINDS
            or (kind is NativeCliEventKind.AUXILIARY and observation.source_message_id is not None)
        ):
            events.append(self._diagnostic(detail="Provider activity preceded its session start.", raw=raw))
        if (
            self._terminal
            and kind in self._PROVIDER_WORK_KINDS
            and kind
            not in {
                NativeCliEventKind.TURN_STARTED,
                NativeCliEventKind.TURN_COMPLETED,
                NativeCliEventKind.RUN_FINISHED,
            }
        ):
            events.append(self._diagnostic(detail="Provider activity followed its terminal event.", raw=raw))
        elif (
            self._config.protocol is NativeCliProtocol.CODEX_EXEC_JSON
            and kind in self._CODEX_ITEM_KINDS
            and not self._turn_open
        ):
            events.append(self._diagnostic(detail="Codex item activity was observed outside a started turn.", raw=raw))
        return tuple(events)

    def _tool_open(self, *, observation: Observation, raw: bytes) -> tuple[NativeCliEvent, ...]:
        tool_id = observation.source_tool_id
        if tool_id is None:
            return (self._diagnostic(detail="Tool start/request lacks a source ID.", raw=raw),)
        key = (observation.parent_tool_use_id, tool_id)
        if key in self._open_tools or key in self._completed_tools:
            return (self._diagnostic(detail=f"Tool {tool_id} has a repeated source ID.", raw=raw),)
        self._open_tools[key] = observation.kind
        return ()

    def _tool_close(self, *, observation: Observation, raw: bytes) -> tuple[NativeCliEvent, ...]:
        tool_id = observation.source_tool_id
        if tool_id is None:
            return (self._diagnostic(detail="Tool completion/result lacks a source ID.", raw=raw),)
        key = (observation.parent_tool_use_id, tool_id)
        if observation.kind is NativeCliEventKind.TOOL_COMPLETED:
            prior = self._open_tools.pop(key, None)
            self._completed_tools.add(key)
            if prior is not NativeCliEventKind.TOOL_STARTED:
                return (self._diagnostic(detail=f"Tool {tool_id} completed without an observed start.", raw=raw),)
        elif self._config.protocol is NativeCliProtocol.CODEX_EXEC_JSON:
            if key not in self._completed_tools:
                return (self._diagnostic(detail=f"Tool {tool_id} has a result without completion.", raw=raw),)
            self._completed_tools.remove(key)
        elif self._open_tools.pop(key, None) is not NativeCliEventKind.TOOL_REQUESTED:
            return (self._diagnostic(detail=f"Tool {tool_id} has a result without a model request.", raw=raw),)
        return ()

    def _count_claude_message(self, *, observation: Observation) -> None:
        message_id = observation.source_message_id
        if observation.kind not in {NativeCliEventKind.MODEL_MESSAGE, NativeCliEventKind.TOOL_REQUESTED}:
            return
        if message_id is not None and message_id not in self._message_ids:
            self._message_ids.add(message_id)
            self._steps += 1

    def _diagnostic(
        self, *, detail: str, raw: bytes | None = None, tool_id: str | None = None, parent_id: str | None = None
    ) -> NativeCliEvent:
        return self._event(
            observation=Observation(
                kind=NativeCliEventKind.PARTIAL,
                detail=detail,
                source_tool_id=tool_id,
                parent_tool_use_id=parent_id,
            ),
            raw=raw,
        )

    def _limit(self, *, reason: str, raw: bytes | None = None) -> NativeCliEvent:
        self._fatal_reason = reason
        return (
            self.abort(detail=reason)
            if raw is None
            else self._event(
                observation=Observation(
                    kind=NativeCliEventKind.ERROR, status=NativeCliEventStatus.FAILED, detail=reason
                ),
                raw=raw,
            )
        )

    def _event(self, *, observation: Observation, raw: bytes | None = None) -> NativeCliEvent:
        self._event_count += 1
        if observation.kind in {NativeCliEventKind.PARTIAL, NativeCliEventKind.ERROR}:
            self._gaps.append(observation.detail or "Native CLI reported an unclassified coverage gap.")
        return NativeCliEvent(
            sequence=self._event_count,
            frame_number=self._frame_count if raw is not None else None,
            raw_frame=raw,
            observation=observation,
        )


class NativeCliRunner:
    """Run one injected sandbox process and stream raw evidence before normalization."""

    def __init__(self, *, launcher: SandboxProcessLauncher, sink: NativeCliEvidenceSink) -> None:
        """Attach an injected sandbox launcher and durable evidence recorder."""
        self._launcher = launcher
        self._sink = sink

    async def run_async(self, *, config: NativeCliRunConfig, prompt: str) -> NativeCliRunOutcome:
        """
        Launch, record, decode, and stop one pinned sandbox CLI process.

        Returns:
            NativeCliRunOutcome: Actual process exit status and evidence coverage.

        Raises:
            TimeoutError: If the run exceeds its deadline.
            NativeCliStreamLimitError: If a frame or observed step exceeds its budget.
            asyncio.CancelledError: If the caller cancels the run.
            OSError: If the sandbox or recorder fails with an operating-system error.
            RuntimeError: If the sandbox session or parser state is invalid.
            TypeError: If the sandbox returns an invalid process chunk or exit code.
            ValueError: If the prompt or sandbox output is invalid.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A native CLI run requires a nonempty prepared prompt.")
        parser = NativeCliJsonlParser(config=config)
        process: SandboxProcessSession | None = None
        chunks = stdout_bytes = stderr_bytes = 0
        try:
            async with asyncio.timeout(config.timeout_seconds):
                process = await self._launcher.launch_async(config=config, prompt=prompt)
                async for part in process.read_chunks_async():
                    if not isinstance(part, NativeCliProcessChunk):
                        raise TypeError("Sandbox process yielded an invalid output chunk.")
                    if not isinstance(part.stream, NativeCliStream) or not isinstance(part.data, bytes):
                        raise TypeError("Sandbox process chunks require a stream and unchanged bytes.")
                    chunks += 1
                    await self._sink.record_raw_async(
                        chunk=NativeCliRawChunk(sequence=chunks, stream=part.stream, data=part.data)
                    )
                    await asyncio.sleep(0)
                    if part.stream is NativeCliStream.STDERR:
                        stderr_bytes += len(part.data)
                        continue
                    stdout_bytes += len(part.data)
                    await self._record_events_async(events=parser.feed(data=part.data))
                    if parser.fatal_reason:
                        raise NativeCliStreamLimitError(parser.fatal_reason)
                await self._record_events_async(events=parser.finish())
                if parser.fatal_reason:
                    raise NativeCliStreamLimitError(parser.fatal_reason)
                exit_code = await process.wait_async()
                if type(exit_code) is not int:
                    raise TypeError("Sandbox process returned no integer exit code.")
                if exit_code:
                    await self._sink.record_event_async(
                        event=parser.abort(detail=f"Native CLI process exited with code {exit_code}.")
                    )
                return NativeCliRunOutcome(
                    exit_code=exit_code,
                    terminal_observed=parser.terminal_observed,
                    coverage_complete=exit_code == 0 and parser.coverage_complete,
                    source_session_id=parser.source_session_id,
                    observed_steps=parser.observed_steps,
                    frame_count=parser.frame_count,
                    raw_chunk_count=chunks,
                    raw_stdout_bytes=stdout_bytes,
                    raw_stderr_bytes=stderr_bytes,
                    gaps=parser.gaps,
                )
        except TimeoutError:
            await self._sink.record_event_async(event=parser.abort(detail="Native CLI run exceeded timeout_seconds."))
            raise
        except asyncio.CancelledError:
            await self._sink.record_event_async(event=parser.abort(detail="Native CLI run was cancelled."))
            raise
        except NativeCliStreamLimitError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            await self._sink.record_event_async(
                event=parser.abort(detail=f"Native CLI transport failed: {error.__class__.__name__}.")
            )
            raise
        finally:
            if process is not None:
                await asyncio.wait_for(process.stop_async(), timeout=config.timeout_seconds)

    async def _record_events_async(self, *, events: tuple[NativeCliEvent, ...]) -> None:
        for event in events:
            await self._sink.record_event_async(event=event)
