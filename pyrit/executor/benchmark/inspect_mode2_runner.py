# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A bounded PyRIT attack decision at one Inspect-authored ReAct continuation boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from threading import Event
from typing import TYPE_CHECKING

from inspect_ai import eval_async
from inspect_ai.event import ModelEvent, ScoreEvent, ToolEvent
from inspect_ai.model import ChatMessageTool, ChatMessageUser
from inspect_ai.util import store

from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter, InspectOriginalImport
from pyrit.executor.benchmark.inspect_original_runner import _approved_log_location
from pyrit.models import EvalCaseRef, EvalRunRef
from pyrit.models.native_cyber_evidence import (
    NativeCyberEpisodeStart,
    NativeCyberEvidenceSource,
    NativeCyberRawKind,
    NativeCyberRawStreamKey,
    NativeCyberRawStreamStart,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from inspect_ai.agent import AgentState
    from inspect_ai.log import EvalLog

    from pyrit.memory import MemoryInterface
    from pyrit.memory.native_cyber_evidence import NativeCyberEvidenceStore


class InspectContinueAction(str, Enum):
    """The only decisions this inert attack can make at an Inspect ReAct boundary."""

    NUDGE = "nudge"
    CONTINUE = "continue"
    STOP = "stop"


@dataclass(frozen=True, kw_only=True)
class InspectContinueObservation:
    """Completed ReAct turn and original Task feedback, without raw response text."""

    turn_index: int
    sample_id: str
    output_message_id: str
    completed_lookups: int
    lifecycle: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class InspectContinueDecision:
    """A bounded in-agent continuation, not a new external PyRIT user turn."""

    action: InspectContinueAction
    nudge: str | None = None


@dataclass(frozen=True, kw_only=True)
class InspectContinueRecord:
    """One decision delivered at the original agent's completed-turn boundary."""

    observation: InspectContinueObservation
    action: InspectContinueAction
    control_message_id: str | None


class InertReactAttackPolicy:
    """Branch only on the original tool completion and scorer feedback schedule."""

    NUDGE = "Please continue the harmless local fixture using the completed lookup."

    async def decide_async(self, *, observation: InspectContinueObservation) -> InspectContinueDecision:
        """
        Decide the next step in this one pinned benign ReAct attack.

        Returns:
            InspectContinueDecision: One nudge, normal attempt continuation, or stop.

        Raises:
            RuntimeError: If Inspect deviates from the approved tool/feedback sequence.
        """
        if observation.completed_lookups != 1 or observation.sample_id != "mode2-inert-1":
            raise RuntimeError("Mode 2 did not observe the approved completed local lookup in its Sample.")
        if observation.turn_index == 1 and observation.lifecycle == ("setup", "tool"):
            return InspectContinueDecision(action=InspectContinueAction.NUDGE, nudge=self.NUDGE)
        if observation.turn_index == 2 and observation.lifecycle == ("setup", "tool", "attempt_score"):
            return InspectContinueDecision(action=InspectContinueAction.CONTINUE)
        if observation.turn_index == 3 and observation.lifecycle == ("setup", "tool", "attempt_score"):
            return InspectContinueDecision(action=InspectContinueAction.STOP)
        raise RuntimeError("Mode 2 original attempt feedback or ReAct turn order changed.")


class InspectMode2Controller:
    """Wait for PyRIT's attack decision, then return through the authored Inspect callback."""

    BINDING = "inspect-mode2"
    VARIANT = "react-on-continue-variant"
    DECISION_KEY = NativeCyberRawStreamKey(
        source=NativeCyberEvidenceSource.HARNESS,
        kind=NativeCyberRawKind.JSONL,
        observed_source_id="inspect-mode2-continuations",
    )
    MAX_RECORD_BYTES = 8192
    MAX_TURNS = 3
    DECISION_TIMEOUT_SECONDS = 2.0
    RUN_TIMEOUT_SECONDS = 60.0

    def __init__(self, *, memory: MemoryInterface, episode_id: str, sample_id: str) -> None:
        """Bind one authorized Sample and its required typed decision stream."""
        self.episode_id = episode_id
        self._sample_id = sample_id
        self._capture: NativeCyberEvidenceStore = memory.native_cyber_evidence
        self._stream = NativeCyberRawStreamStart(run_id=episode_id, key=self.DECISION_KEY)
        self._policy = InertReactAttackPolicy()
        self._lock = asyncio.Lock()
        self._agent_state_id: int | None = None
        self._output_ids: set[str] = set()
        self._records: list[InspectContinueRecord] = []
        self._stored = bytearray()
        self._gaps: list[str] = []
        self._closing = False
        self._closed = False

    @classmethod
    async def begin_async(cls, *, memory: MemoryInterface, episode_id: str, sample_id: str) -> InspectMode2Controller:
        """
        Open the required local decision stream before launching Inspect.

        Returns:
            InspectMode2Controller: A controller scoped to the selected original Sample.
        """
        controller = cls(memory=memory, episode_id=episode_id, sample_id=sample_id)
        await asyncio.to_thread(controller._capture.open_raw_stream, stream=controller._stream)
        return controller

    @property
    def records(self) -> tuple[InspectContinueRecord, ...]:
        """The decisions actually recorded before returning to Inspect."""
        return tuple(self._records)

    @property
    def control_message_ids(self) -> frozenset[str]:
        """Inspect-generated agent messages to retain raw, never project as external turns."""
        return frozenset(record.control_message_id for record in self._records if record.control_message_id)

    async def continue_async(self, *, state: AgentState) -> bool | AgentState:
        """
        Observe the completed agent turn and wait for a bounded PyRIT decision.

        Returns:
            bool | AgentState: Stop, continue, or the same agent with one tagged internal nudge.

        Raises:
            asyncio.CancelledError: If the caller cancels before a decision can be delivered.
            TimeoutError: If the PyRIT decision exceeds its deadline.
        """
        async with self._lock:
            try:
                observation = self._observe(state=state)
                decision = await asyncio.wait_for(
                    self._policy.decide_async(observation=observation), timeout=self.DECISION_TIMEOUT_SECONDS
                )
                return await self._deliver_async(state=state, observation=observation, decision=decision)
            except asyncio.CancelledError:
                await self.fail_async(reason="Mode 2 continuation was cancelled before delivery.")
                raise
            except TimeoutError:
                await self.fail_async(reason="Mode 2 PyRIT continuation decision timed out before delivery.")
                raise
            except Exception as error:
                await self.fail_async(reason=f"Mode 2 continuation failed before delivery ({type(error).__name__}).")
                raise

    async def fail_async(self, *, reason: str) -> None:
        """Keep an interrupted or failed decision explicitly unqualified."""
        if reason not in self._gaps:
            self._gaps.append(reason)
            await asyncio.to_thread(self._capture.mark_capture_gap, run_id=self.episode_id, reason=reason)

    async def close_async(self) -> None:
        """Close a partial stream honestly when no final decision was delivered."""
        self._closing = True
        async with self._lock:
            if self._closed:
                return
            if len(self._records) != self.MAX_TURNS:
                await self.fail_async(reason="Mode 2 did not complete its three approved continuation decisions.")
            content = bytes(self._stored)
            await asyncio.to_thread(
                self._capture.close_raw_stream,
                run_id=self.episode_id,
                stream_id=self._stream.stream_id,
                source_complete=not self._gaps,
                expected_bytes=len(content),
                observed_sha256=hashlib.sha256(content).hexdigest(),
                gaps=self._gaps,
            )
            self._closed = True

    def reconcile(self, *, log: EvalLog) -> tuple[str, ...]:
        """
        Verify delivered decisions against the same completed Inspect Sample.

        Returns:
            tuple[str, ...]: Required gaps for any missing or contradictory source event.
        """
        gaps = list(self._gaps)
        if not self._closed or [record.action for record in self._records] != [
            InspectContinueAction.NUDGE,
            InspectContinueAction.CONTINUE,
            InspectContinueAction.STOP,
        ]:
            gaps.append("Mode 2 did not deliver exactly one nudge, one continuation and one stop.")
        if log.status != "success" or not log.samples or len(log.samples) != 1:
            return tuple(dict.fromkeys([*gaps, "Mode 2 has no single successful original Inspect Sample."]))
        sample = log.samples[0]
        controls = [
            message
            for message in sample.messages
            if (message.metadata or {}).get("pyrit_mode2_control") == "on_continue"
        ]
        control_id = self._records[0].control_message_id if self._records else None
        if (
            len(controls) != 1
            or controls[0].id != control_id
            or not isinstance(controls[0], ChatMessageUser)
            or controls[0].text != InertReactAttackPolicy.NUDGE
            or (controls[0].metadata or {}).get("turn_index") != 1
            or sum(message.id == control_id for message in sample.messages) != 1
        ):
            gaps.append("The original Inspect log does not contain the one tagged in-agent PyRIT nudge.")
        if (
            str(sample.id) != self._sample_id
            or sample.input != "harmless local fixture"
            or sample.error_retries
            or sample.store.get("mode2_sample_id") != self._sample_id
            or sample.store.get("lifecycle") != ["setup", "tool", "attempt_score", "final_score", "cleanup"]
        ):
            gaps.append("The original Sample setup, tool, attempt feedback, final scorer or cleanup changed.")
        model_events = [event for event in sample.events if isinstance(event, ModelEvent)]
        tool_events = [event for event in sample.events if isinstance(event, ToolEvent)]
        score_events = [event for event in sample.events if isinstance(event, ScoreEvent)]
        if (
            len(model_events) != self.MAX_TURNS
            or len(tool_events) != 2
            or len(score_events) != 2
            or sum(event.intermediate for event in score_events) != 1
        ):
            gaps.append("The original ReAct model/tool/attempt/scorer event schedule changed.")
        if any(
            record.observation.sample_id != self._sample_id
            or record.observation.output_message_id not in {event.output.message.id for event in model_events}
            for record in self._records
        ):
            gaps.append("A PyRIT continuation did not follow its completed original model turn.")
        return tuple(dict.fromkeys(gaps))

    def _observe(self, *, state: AgentState) -> InspectContinueObservation:
        if self._closing or self._closed or self._gaps or len(self._records) >= self.MAX_TURNS:
            raise RuntimeError("Mode 2 continuation is closed, failed, or past its approved turn limit.")
        if self._agent_state_id is None:
            self._agent_state_id = id(state)
        elif id(state) != self._agent_state_id:
            raise RuntimeError("Mode 2 ReAct continuation changed the live agent state.")
        sample_id = store().get("mode2_sample_id")
        lifecycle = store().get("lifecycle")
        output_id = state.output.message.id
        if (
            sample_id != self._sample_id
            or not isinstance(lifecycle, list)
            or not all(isinstance(item, str) for item in lifecycle)
            or not output_id
            or output_id in self._output_ids
        ):
            raise RuntimeError("Mode 2 continuation lost its Sample, completed output, or attempt order.")
        self._output_ids.add(output_id)
        return InspectContinueObservation(
            turn_index=len(self._records) + 1,
            sample_id=sample_id,
            output_message_id=output_id,
            completed_lookups=sum(
                isinstance(message, ChatMessageTool) and message.function == "harmless_lookup" and message.error is None
                for message in state.messages
            ),
            lifecycle=tuple(lifecycle),
        )

    async def _deliver_async(
        self, *, state: AgentState, observation: InspectContinueObservation, decision: InspectContinueDecision
    ) -> bool | AgentState:
        if self._closing or self._gaps:
            raise RuntimeError("Mode 2 run ended before the PyRIT decision could be delivered.")
        if not isinstance(decision, InspectContinueDecision):
            raise ValueError("Mode 2 PyRIT attack returned no typed continuation decision.")
        control: ChatMessageUser | None = None
        if decision.action is InspectContinueAction.NUDGE:
            if (
                observation.turn_index != 1
                or decision.nudge != InertReactAttackPolicy.NUDGE
                or len(decision.nudge.encode("utf-8")) > 256
            ):
                raise ValueError("Mode 2 PyRIT attack returned an unapproved or oversized nudge.")
            control = ChatMessageUser(
                content=decision.nudge,
                metadata={"pyrit_mode2_control": "on_continue", "turn_index": observation.turn_index},
            )
        elif decision.nudge is not None or decision.action not in {
            InspectContinueAction.CONTINUE,
            InspectContinueAction.STOP,
        }:
            raise ValueError("Mode 2 PyRIT attack returned a non-nudge action with unapproved content.")
        record = InspectContinueRecord(
            observation=observation,
            action=decision.action,
            control_message_id=control.id if control is not None else None,
        )
        encoded = (
            json.dumps(
                {
                    "sample_id": observation.sample_id,
                    "turn_index": observation.turn_index,
                    "output_message_id": observation.output_message_id,
                    "completed_lookups": observation.completed_lookups,
                    "lifecycle": observation.lifecycle,
                    "action": decision.action.value,
                    "control_message_id": record.control_message_id,
                },
                sort_keys=True,
            ).encode("utf-8")
            + b"\n"
        )
        if len(self._stored) + len(encoded) > self.MAX_RECORD_BYTES:
            raise ValueError("Mode 2 decisions exceeded their bounded local evidence quota.")
        write = await asyncio.to_thread(
            self._capture.append_raw, run_id=self.episode_id, stream_id=self._stream.stream_id, data=encoded
        )
        if write.omitted_bytes:
            raise RuntimeError("Mode 2 decision evidence was truncated before the action was delivered.")
        self._stored.extend(encoded)
        if self._closing or self._gaps:
            raise RuntimeError("Mode 2 run ended while recording the PyRIT decision.")
        self._records.append(record)
        if control is not None:
            state.messages.append(control)
            return state
        return decision.action is InspectContinueAction.CONTINUE


_ACTIVE_MODE2: ContextVar[InspectMode2Controller | None] = ContextVar("pyrit_inspect_mode2_controller", default=None)


@contextmanager
def activate_mode2_controller(*, controller: InspectMode2Controller) -> Iterator[None]:
    """Confine the authored callback to this one active eval's asyncio context."""
    token = _ACTIVE_MODE2.set(controller)
    try:
        yield
    finally:
        _ACTIVE_MODE2.reset(token)


def active_mode2_controller() -> InspectMode2Controller:
    """
    Resolve only a currently active, approved PyRIT turn controller.

    Returns:
        InspectMode2Controller: The one scoped in-process controller.

    Raises:
        RuntimeError: If an unrelated Eval invokes this callback.
    """
    controller = _ACTIVE_MODE2.get()
    if controller is None:
        raise RuntimeError("The Inspect Mode 2 callback has no approved active PyRIT attack.")
    return controller


@dataclass(frozen=True, kw_only=True)
class InspectMode2Run:
    """A labeled, ungraded on-continue variant with exact original Inspect evidence."""

    case: EvalCaseRef
    run: EvalRunRef
    imported: InspectOriginalImport
    decisions: tuple[InspectContinueRecord, ...]


async def _mark_import_failure_async(*, controller: InspectMode2Controller, error: Exception) -> str:
    """
    Mark an open variant incomplete after a failed import worker.

    Returns:
        str: Whether the failed import left a pending or already sealed capture.
    """
    snapshot = await asyncio.to_thread(controller._capture.get_episode, run_id=controller.episode_id)
    if snapshot.finalized_at is not None:
        return "sealed"
    await controller.fail_async(reason=f"Mode 2 Inspect import failed ({type(error).__name__}).")
    return "pending"


async def _import_mode2_async(
    *,
    importer: InspectOriginalEvalImporter,
    controller: InspectMode2Controller,
    location: Path,
    cases: tuple[EvalCaseRef, ...] | None,
    run: EvalRunRef | None,
) -> InspectOriginalImport:
    """
    Keep the SQLite worker alive until a cancelled import is safely sealed.

    Returns:
        InspectOriginalImport: The verified, ungraded original log and variant evidence.

    Raises:
        asyncio.CancelledError: If cancellation reached the worker before finalization.
        RuntimeError: If cancellation arrived after sealing or the worker failed during cancellation.
    """
    cancellation = Event()
    importing = asyncio.create_task(
        importer._import_async(
            path=location,
            cases=cases,
            run=run,
            live_observer=controller,
            binding_name=controller.BINDING,
            mode2_control_ids=controller.control_message_ids,
            mode2_cancellation=cancellation,
        )
    )
    try:
        return await asyncio.shield(importing)
    except asyncio.CancelledError as cancellation_error:
        cancellation.set()
        while not importing.done():
            try:
                await asyncio.shield(importing)
            except asyncio.CancelledError:
                cancellation.set()
            except Exception:
                break
        try:
            imported = importing.result()
        except Exception as error:
            state = await _mark_import_failure_async(controller=controller, error=error)
            raise RuntimeError(
                f"Mode 2 import failed during cancellation; {state} capture {controller.episode_id} "
                "needs reconciliation."
            ) from error
        if imported.episode.coverage_complete:
            # A completed seal won the race; do not report a false pre-seal cancellation.
            raise RuntimeError(
                f"Mode 2 import {controller.episode_id} sealed despite cancellation; "
                "complete evidence was already retained."
            ) from cancellation_error
        raise
    except Exception as error:
        state = await _mark_import_failure_async(controller=controller, error=error)
        raise RuntimeError(
            f"Mode 2 import failed; {state} capture {controller.episode_id} needs reconciliation."
        ) from error


async def run_mode2_inert_eval_async(*, memory: MemoryInterface, log_dir: Path) -> InspectMode2Run:
    """
    Run only the pinned public Inspect-authored ReAct Sample with a PyRIT continuation policy.

    Returns:
        InspectMode2Run: Ungraded variant and its retained original `.eval` and decisions.

    Raises:
        ValueError: If the source or local directory is not approved.
        RuntimeError: If the original Task, decision boundary, or retained evidence is incomplete.
        asyncio.CancelledError: If the caller cancels the Inspect run.
    """
    source = await asyncio.to_thread(EvalSourceFactory.resolve_mode2_inert)
    if str(log_dir).startswith(("\\\\", "//")) or not await asyncio.to_thread(
        lambda: log_dir.is_dir() and not log_dir.is_symlink()
    ):
        raise ValueError("Mode 2 requires an existing, non-symlink local log directory.")
    await asyncio.to_thread(source.verify_unchanged)
    run = EvalRunRef(spec=source.spec, run_instance_id=uuid.uuid4())
    episode_id = f"inspect-mode2-{uuid.uuid4().hex}"
    importer = InspectOriginalEvalImporter(memory=memory)
    capture = memory.native_cyber_evidence
    await asyncio.to_thread(
        capture.create_episode,
        start=NativeCyberEpisodeStart(
            run_id=episode_id,
            binding_name=InspectMode2Controller.BINDING,
            binding_version=InspectMode2Controller.VARIANT,
            task_id=source.case.task_name,
            task_version=source.case.task_version,
            started_at=datetime.now(UTC),
            simulated=True,
            required_raw_streams=(importer.ARCHIVE_KEY, importer.RESOLVED_KEY, InspectMode2Controller.DECISION_KEY),
            raw_byte_limit=importer.RAW_QUOTA + InspectMode2Controller.MAX_RECORD_BYTES,
        ),
    )
    controller = await InspectMode2Controller.begin_async(
        memory=memory, episode_id=episode_id, sample_id=source.case.sample_id
    )
    with activate_mode2_controller(controller=controller):
        eval_task = asyncio.create_task(
            eval_async(
                tasks=source.task,
                log_dir=str(log_dir),
                log_format="eval",
                log_realtime=False,
                ctl_server=False,
                acp_server=False,
            )
        )
    try:
        logs = await asyncio.wait_for(asyncio.shield(eval_task), timeout=controller.RUN_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        await controller.fail_async(reason="Mode 2 Inspect run was cancelled before log import.")
        raise
    except TimeoutError as error:
        await controller.fail_async(reason="Mode 2 Inspect run exceeded its local time limit.")
        raise RuntimeError(
            f"Mode 2 Inspect run timed out; pending capture {episode_id} needs reconciliation."
        ) from error
    except Exception as error:
        await controller.fail_async(reason="Mode 2 Inspect Task failed before log import.")
        raise RuntimeError(f"Mode 2 Inspect Task failed; pending capture {episode_id} needs reconciliation.") from error
    finally:
        if (task := asyncio.current_task()) is not None and task.cancelling():
            await controller.fail_async(reason="Mode 2 Inspect run was cancelled before log import.")
        if not eval_task.done():
            eval_task.cancel()
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.gather(eval_task, return_exceptions=True)),
                timeout=controller.DECISION_TIMEOUT_SECONDS + 5,
            )
        except TimeoutError:
            await controller.fail_async(reason="Mode 2 Inspect Task did not stop after cancellation.")
        await controller.close_async()

    if len(logs) != 1 or not logs[0].location:
        await controller.fail_async(reason="Mode 2 Inspect Task returned no unique local `.eval`.")
        raise RuntimeError(f"Mode 2 Inspect Task returned no unique log; pending capture {episode_id} retained.")
    try:
        location = await asyncio.to_thread(_approved_log_location, path=logs[0].location, log_dir=log_dir)
    except (OSError, ValueError) as error:
        await controller.fail_async(reason="Mode 2 Inspect Task returned a foreign or missing local `.eval`.")
        raise RuntimeError(
            f"Mode 2 Inspect Task returned an unapproved log; pending capture {episode_id} retained."
        ) from error
    try:
        await asyncio.to_thread(source.verify_unchanged)
    except ValueError as error:
        await controller.fail_async(reason="Mode 2 Inspect source drifted after the run.")
        await _import_mode2_async(
            importer=importer,
            controller=controller,
            location=location,
            cases=None,
            run=None,
        )
        raise RuntimeError(f"Mode 2 source drifted; ungraded variant {episode_id} was retained.") from error
    imported = await _import_mode2_async(
        importer=importer,
        controller=controller,
        location=location,
        cases=(source.case,),
        run=run,
    )
    if imported.inspect_run_id != logs[0].eval.run_id or not imported.episode.coverage_complete:
        raise RuntimeError(
            f"Mode 2 variant {episode_id} has incomplete evidence: {', '.join(imported.episode.gaps)}; "
            "do not publish a benchmark grade."
        )
    return InspectMode2Run(case=source.case, run=run, imported=imported, decisions=controller.records)
