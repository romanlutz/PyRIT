# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario run service for executing scenarios as background tasks.

Manages the lifecycle of scenario runs: starting, tracking status,
retrieving results, and cancellation.
"""

import asyncio
import base64
import binascii
import contextlib
import hashlib
import json
import logging
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

try:
    from builtins import ExceptionGroup  # type: ignore[attr-defined,ty:unresolved-import]
except ImportError:  # pragma: no cover - exercised only on 3.10
    from exceptiongroup import ExceptionGroup  # type: ignore[no-redef,ty:unresolved-import]

from pydantic import TypeAdapter, ValidationError

from pyrit.backend.models import PaginationInfo, ScenarioRunListResponse, filter_sensitive_fields
from pyrit.common.utils import to_sha256
from pyrit.memory import (
    CentralMemory,
    ScenarioHistoryKeysetCursor,
    ScenarioHistoryRunRecord,
    ScenarioHistoryUnitRecord,
    ScenarioProgressKeysetCursor,
)
from pyrit.models import (
    ADAPTIVE_ATTEMPT_LABEL,
    ADAPTIVE_TECHNIQUE_NAME_LABEL,
    SCENARIO_RUN_PLAN_METADATA_KEY,
    AtomicAttackIdentifier,
    AttackOutcome,
    ComponentIdentifier,
    ScenarioAttackResultDelta,
    ScenarioIdentifier,
    ScenarioProgressHeader,
    ScenarioProgressResult,
    ScenarioProgressResultKind,
    ScenarioQueueEntry,
    ScenarioQueueSnapshot,
    ScenarioResult,
    ScenarioRunPlan,
    ScenarioRunPlanAtomicGroup,
    ScenarioRunPlanGroupKind,
    ScenarioRunPlanSeedGroup,
    ScenarioRunProgress,
    ScenarioRunState,
    TargetIdentifier,
    config_hash,
)
from pyrit.models.catalog import (
    AttackErrorSummary,
    AttackRetrySummary,
    RunScenarioRequest,
    ScenarioOverloadSummary,
    ScenarioRunHeader,
    ScenarioRunSummary,
    ScenarioTargetSummary,
)
from pyrit.registry import (
    ConverterRegistry,
    InitializerRegistry,
    ScenarioRegistry,
    TargetRegistry,
)
from pyrit.scenario import CompoundDatasetAttackConfiguration, Scenario

if TYPE_CHECKING:
    from pyrit.converter import Converter
    from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENT_RUNS = 1
_MAX_OVERLOAD_EVENTS = 500
_MAX_OVERLOAD_ROLES = 16
_MAX_TERMINAL_ERRORS = 100
_SCHEDULER_RETRY_INITIAL_SECONDS = 0.05
_SCHEDULER_RETRY_MAX_SECONDS = 1.0
_INTERRUPTED_ERROR_TYPE = "ScenarioInterruptedError"
_RESTART_INTERRUPTION_REASON = (
    "The backend process restarted before this scenario run completed; "
    "its executable scenario objects could not be recovered safely."
)
_SHUTDOWN_INTERRUPTION_REASON = "The backend process shut down before this scenario run completed."
_USER_CANCELLATION_REASON = "Run was cancelled by user"

_CONVERTER_MODIFIER_PREFIX = "converter."
_SAFE_SCENARIO_PARAMETER_NAMES = frozenset(
    {
        "adversarial_targets",
        "jailbreak_names",
        "max_attempts_per_objective",
        "max_turns",
        "num_jailbreak_attempts",
        "num_jailbreaks",
        "sub_harm",
        "version",
    }
)
_HISTORY_ATOMIC_GROUPS_ADAPTER = TypeAdapter(list[ScenarioRunPlanAtomicGroup])
_HISTORY_SEED_ID_MAP_ADAPTER = TypeAdapter(list[dict[str, str]])


@dataclass
class _ActiveTask:
    """Tracks an in-flight scenario run's asyncio task."""

    scenario_result_id: str
    task: asyncio.Task[None] | None = None
    scenario: Scenario | None = None
    error: str | None = None
    scenario_name: str = ""
    scenario_registry_name: str = ""
    created_at: datetime | None = None
    enqueued_at: datetime | None = None
    started_at: datetime | None = None
    cancellation_state: ScenarioRunState = ScenarioRunState.CANCELLED
    cancellation_reason: str = _USER_CANCELLATION_REASON
    cancellation_error_type: str = "CancelledError"
    retain_error_on_terminalization: bool = False


@dataclass(frozen=True, slots=True)
class _ActiveRunSnapshot:
    """Event-loop-owned state copied before database work moves to a worker thread."""

    error: str | None = None
    active_group_ids: tuple[str, ...] = ()
    queue_position: int | None = None
    active_scenario_result_id: str | None = None


@dataclass(frozen=True, slots=True)
class _RunDiagnostics:
    """Per-attempt diagnostics summarized for a scenario run."""

    failed_attacks: list[AttackErrorSummary]
    attack_retries: list[AttackRetrySummary]
    total_retries: int
    overload_summaries: list[ScenarioOverloadSummary]


class ScenarioRunService:
    """
    Service for managing scenario run lifecycle.

    Uses CentralMemory (database) as the source of truth for run state.
    Keeps executable objects in a process-local single-active FIFO scheduler.
    """

    def __init__(self, *, max_concurrent_runs: int = _DEFAULT_MAX_CONCURRENT_RUNS) -> None:
        """
        Initialize the scenario run service.

        ``max_concurrent_runs`` remains accepted for configuration compatibility;
        scenario execution is always serialized to one active run.
        """
        if max_concurrent_runs < 1:
            raise ValueError("max_concurrent_runs must be at least 1.")
        self._memory = CentralMemory.get_memory_instance()
        self._active_tasks: dict[str, _ActiveTask] = {}
        self._terminal_errors: OrderedDict[str, str] = OrderedDict()
        self._active_scenario_result_id: str | None = None
        self._queued_runs: deque[_ActiveTask] = deque()
        self._handoff_retry_tasks: set[asyncio.Task[None]] = set()
        self._scheduler_lock = asyncio.Lock()
        self._launch_lock = asyncio.Lock()
        self._queue_revision = 0
        self._stopping = False

    async def start_run_async(self, *, request: RunScenarioRequest) -> ScenarioRunSummary:
        """
        Initialize and schedule a scenario run.

        Performs all validation and initialization eagerly (initializers, target
        resolution, technique validation, scenario.initialize_async) so errors are
        returned immediately. On success, starts execution when idle or appends
        the initialized run to the FIFO waiting queue.

        Args:
            request: The run request with scenario name, target, and options.

        Returns:
            ScenarioRunSummary with a stable ID and current active or queued state.

        Raises:
            ValueError: If scenario, target, initializer, or technique cannot be found.
        """
        async with self._launch_lock:
            if self._stopping:
                raise RuntimeError("Scenario run scheduling is stopping.")
            scenario_class = self._resolve_scenario_class(request=request)
            await self._run_initializers_async(request=request)
            objective_target = self._resolve_target(request=request)
            init_kwargs = self._build_init_kwargs(
                request=request, scenario_class=scenario_class, objective_target=objective_target
            )
            scenario = await self._initialize_scenario_async(request=request, init_kwargs=init_kwargs)
            scenario_result_id = scenario._scenario_result_id
            if scenario_result_id is None:
                raise ValueError("Scenario did not produce a scenario_result_id during initialization.")
            persisted = await asyncio.to_thread(
                self._memory.get_scenario_results,
                scenario_result_ids=[scenario_result_id],
            )
            if not persisted:
                raise RuntimeError(f"Scenario run {scenario_result_id} was not persisted during initialization.")
            scheduled = _ActiveTask(
                scenario_result_id=scenario_result_id,
                scenario=scenario,
                scenario_name=persisted[0].scenario_name,
                scenario_registry_name=request.scenario_name,
                created_at=persisted[0].creation_time,
                enqueued_at=datetime.now(timezone.utc),
            )
            await self._enqueue_run_async(scheduled=scheduled)

        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id)
        response = await asyncio.to_thread(
            self.get_run_from_storage,
            scenario_result_id=scenario_result_id,
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
        )
        if response is None:
            raise RuntimeError(f"Scenario run {scenario_result_id} was not found in the database after initialization.")
        return response

    def get_run(self, *, scenario_result_id: str) -> ScenarioRunSummary | None:
        """
        Get the current status of a scenario run by querying the database.

        Args:
            scenario_result_id: The scenario result ID.

        Returns:
            ScenarioRunSummary if found, None otherwise.
        """
        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id)
        return self.get_run_from_storage(
            scenario_result_id=scenario_result_id,
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
        )

    def get_run_from_storage(
        self,
        *,
        scenario_result_id: str,
        active_error: str | None,
        queue_position: int | None = None,
        active_scenario_result_id: str | None = None,
    ) -> ScenarioRunSummary | None:
        """
        Build a run summary using database state plus an event-loop snapshot.

        Args:
            scenario_result_id: The scenario result ID.
            active_error: Error copied from the active asyncio task, if any.
            queue_position: Current 1-based waiting position, if queued.
            active_scenario_result_id: Currently executing scenario result ID.

        Returns:
            ScenarioRunSummary | None: The run summary when found.
        """
        return self._build_response(
            scenario_result_id=scenario_result_id,
            active_error=active_error,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
        )

    def list_runs(
        self,
        *,
        scenario_names: Sequence[str] | None = None,
        statuses: Sequence[ScenarioRunState | str] | None = None,
        labels: Mapping[str, str | Sequence[str]] | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> ScenarioRunListResponse:
        """
        List scenario runs by querying the database (most recent first).

        Args:
            scenario_names: Registered or persisted scenario names to match.
            statuses: Run states to match.
            labels: Labels with OR-within-key and AND-across-key semantics.
            limit: Maximum number of runs to return.
            cursor: Opaque cursor from the previous page.

        Returns:
            ScenarioRunListResponse with runs.
        """
        normalized_names = sorted({name.strip() for name in scenario_names or [] if name.strip()})
        normalized_statuses = sorted(
            {
                status.value if isinstance(status, ScenarioRunState) else str(status).strip().upper()
                for status in statuses or []
                if str(status).strip()
            }
        )
        normalized_labels = self._normalize_history_labels(labels=labels)
        fingerprint = self._history_filter_fingerprint(
            scenario_names=normalized_names,
            statuses=normalized_statuses,
            labels=normalized_labels,
        )
        after = self._decode_history_cursor(cursor=cursor, fingerprint=fingerprint)
        records, units_by_run, has_more = self._memory.get_scenario_run_history_page(
            scenario_names=normalized_names,
            statuses=normalized_statuses,
            labels=normalized_labels,
            cursor=after,
            limit=limit,
        )
        items = [
            self._build_history_summary(
                record=record,
                units=units_by_run.get(record.scenario_result_id, []),
            )
            for record in records
        ]
        next_cursor = (
            self._encode_history_cursor(
                cursor=ScenarioHistoryKeysetCursor(
                    timestamp=records[-1].created_at,
                    scenario_result_id=records[-1].scenario_result_id,
                ),
                fingerprint=fingerprint,
            )
            if has_more and records
            else None
        )
        return ScenarioRunListResponse(
            items=items,
            pagination=PaginationInfo(
                limit=limit,
                has_more=has_more,
                next_cursor=next_cursor,
                prev_cursor=cursor,
            ),
        )

    async def cancel_run_async(self, *, scenario_result_id: str) -> ScenarioRunSummary | None:
        """
        Cancel a running scenario.

        Args:
            scenario_result_id: The scenario result ID.

        Returns:
            Updated ScenarioRunSummary if found, None if not found.

        Raises:
            ValueError: If the run is already in a terminal state or not active.
        """
        results = await asyncio.to_thread(
            self._memory.get_scenario_results,
            scenario_result_ids=[scenario_result_id],
        )
        if not results:
            return None

        db_status = results[0].scenario_run_state
        if self._is_terminal_state(db_status):
            raise ValueError(f"Cannot cancel run in '{db_status}' state.")

        task: asyncio.Task[None] | None = None
        async with self._scheduler_lock:
            queued = next(
                (run for run in self._queued_runs if run.scenario_result_id == scenario_result_id),
                None,
            )
            if queued is not None:
                await asyncio.to_thread(
                    self._memory.update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    scenario_run_state=ScenarioRunState.CANCELLED,
                    error_message=_USER_CANCELLATION_REASON,
                    error_type="CancelledError",
                )
                self._queued_runs.remove(queued)
                self._queue_revision += 1
            elif self._active_scenario_result_id == scenario_result_id:
                active = self._active_tasks[scenario_result_id]
                active.cancellation_state = ScenarioRunState.CANCELLED
                active.cancellation_reason = _USER_CANCELLATION_REASON
                active.cancellation_error_type = "CancelledError"
                task = active.task
            else:
                latest = await asyncio.to_thread(
                    self._memory.get_scenario_results,
                    scenario_result_ids=[scenario_result_id],
                )
                if latest and self._is_terminal_state(latest[0].scenario_run_state):
                    raise ValueError(f"Cannot cancel run in '{latest[0].scenario_run_state}' state.")
                await asyncio.to_thread(
                    self._memory.update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    scenario_run_state=ScenarioRunState.CANCELLED,
                    error_message=_USER_CANCELLATION_REASON,
                    error_type="CancelledError",
                )

        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id)
        result = await asyncio.to_thread(
            self.get_run_from_storage,
            scenario_result_id=scenario_result_id,
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
        )
        if result is not None and result.status != ScenarioRunState.CANCELLED:
            raise ValueError(f"Cannot cancel run in '{result.status}' state.")
        return result

    def get_queue_snapshot(self) -> ScenarioQueueSnapshot:
        """
        Return the current in-process FIFO scheduler state.

        Returns:
            ScenarioQueueSnapshot: Active run and ordered waiting runs.
        """
        snapshot_at = datetime.now(timezone.utc)
        active = None
        if self._active_scenario_result_id is not None:
            active_run = self._active_tasks.get(self._active_scenario_result_id)
            if active_run is not None:
                active = self._build_queue_entry(run=active_run, state=ScenarioRunState.IN_PROGRESS)
        queued = [
            self._build_queue_entry(run=run, state=ScenarioRunState.QUEUED, position=position)
            for position, run in enumerate(self._queued_runs, start=1)
        ]
        return ScenarioQueueSnapshot(
            revision=self._queue_revision,
            snapshot_at=snapshot_at,
            active=active,
            queued=queued,
        )

    async def reconcile_interrupted_runs_async(self) -> int:
        """
        Mark non-terminal persisted rows failed when executable objects were lost.

        Returns:
            int: Number of reconciled rows.
        """
        results = await asyncio.to_thread(self._memory.get_scenario_results)
        interrupted = [
            result
            for result in results
            if result.scenario_run_state
            in (ScenarioRunState.CREATED, ScenarioRunState.QUEUED, ScenarioRunState.IN_PROGRESS)
        ]
        for result in interrupted:
            await asyncio.to_thread(
                self._memory.update_scenario_run_state,
                scenario_result_id=str(result.id),
                scenario_run_state=ScenarioRunState.FAILED,
                error_message=_RESTART_INTERRUPTION_REASON,
                error_type=_INTERRUPTED_ERROR_TYPE,
            )
        return len(interrupted)

    async def shutdown_async(self) -> None:
        """Stop scheduling and terminalize active and queued runs for process shutdown."""
        task: asyncio.Task[None] | None = None
        retry_tasks: list[asyncio.Task[None]] = []
        errors: list[Exception] = []
        async with self._scheduler_lock:
            self._stopping = True
            retry_tasks = list(self._handoff_retry_tasks)
            queued = list(self._queued_runs)
            self._queued_runs.clear()
            if queued:
                self._queue_revision += 1
            for run in queued:
                try:
                    await asyncio.to_thread(
                        self._memory.update_scenario_run_state,
                        scenario_result_id=run.scenario_result_id,
                        scenario_run_state=ScenarioRunState.FAILED,
                        error_message=_SHUTDOWN_INTERRUPTION_REASON,
                        error_type=_INTERRUPTED_ERROR_TYPE,
                    )
                except Exception as exc:
                    errors.append(exc)
            if self._active_scenario_result_id is not None:
                active = self._active_tasks[self._active_scenario_result_id]
                active.cancellation_state = ScenarioRunState.FAILED
                active.cancellation_reason = _SHUTDOWN_INTERRUPTION_REASON
                active.cancellation_error_type = _INTERRUPTED_ERROR_TYPE
                task = active.task
                if task is None or task.done():
                    try:
                        await asyncio.to_thread(
                            self._memory.update_scenario_run_state,
                            scenario_result_id=active.scenario_result_id,
                            scenario_run_state=ScenarioRunState.FAILED,
                            error_message=_SHUTDOWN_INTERRUPTION_REASON,
                            error_type=_INTERRUPTED_ERROR_TYPE,
                        )
                    except Exception as exc:
                        errors.append(exc)
                    self._active_scenario_result_id = None
                    self._release_completed_task(scenario_result_id=active.scenario_result_id)
                    self._queue_revision += 1
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                errors.append(exc)
        for retry_task in retry_tasks:
            retry_task.cancel()
        if retry_tasks:
            await asyncio.gather(*retry_tasks, return_exceptions=True)
        if errors:
            raise ExceptionGroup("Failed to persist one or more scenario shutdown transitions.", errors)

    async def _enqueue_run_async(self, *, scheduled: _ActiveTask) -> None:
        """Atomically enqueue a persisted initialized run or start it immediately."""
        async with self._scheduler_lock:
            if self._stopping:
                raise RuntimeError("Scenario run scheduling is stopping.")
            scheduled_ids = {
                *(run.scenario_result_id for run in self._queued_runs),
                *self._active_tasks.keys(),
            }
            if scheduled.scenario_result_id in scheduled_ids:
                raise ValueError(f"Scenario run '{scheduled.scenario_result_id}' is already scheduled.")
            self._terminal_errors.pop(scheduled.scenario_result_id, None)
            if self._active_scenario_result_id is None:
                await self._start_scheduled_run_locked_async(scheduled=scheduled)
                return
            await asyncio.to_thread(
                self._memory.update_scenario_run_state,
                scenario_result_id=scheduled.scenario_result_id,
                scenario_run_state=ScenarioRunState.QUEUED,
            )
            self._queued_runs.append(scheduled)
            self._queue_revision += 1

    async def _start_scheduled_run_locked_async(self, *, scheduled: _ActiveTask) -> None:
        """Start one run while the scheduler lock guarantees exclusive ownership."""
        scheduled.started_at = datetime.now(timezone.utc)
        await asyncio.to_thread(
            self._memory.update_scenario_run_state,
            scenario_result_id=scheduled.scenario_result_id,
            scenario_run_state=ScenarioRunState.IN_PROGRESS,
        )
        self._active_scenario_result_id = scheduled.scenario_result_id
        self._active_tasks[scheduled.scenario_result_id] = scheduled
        scheduled.task = asyncio.create_task(self._execute_run_async(scenario_result_id=scheduled.scenario_result_id))
        self._queue_revision += 1

    async def _handoff_scheduler_async(self, *, scenario_result_id: str) -> None:
        """Release one terminal active run and start the next valid queued run once."""
        async with self._scheduler_lock:
            if self._active_scenario_result_id != scenario_result_id:
                return
            if self._stopping:
                self._active_scenario_result_id = None
                self._release_completed_task(scenario_result_id=scenario_result_id)
                self._queue_revision += 1
                return
            while self._queued_runs:
                next_run = self._queued_runs[0]
                persisted = await asyncio.to_thread(
                    self._memory.get_scenario_results,
                    scenario_result_ids=[next_run.scenario_result_id],
                )
                if not persisted or persisted[0].scenario_run_state != ScenarioRunState.QUEUED:
                    self._queued_runs.popleft()
                    self._queue_revision += 1
                    continue
                await self._start_scheduled_run_locked_async(scheduled=next_run)
                self._queued_runs.popleft()
                self._release_completed_task(scenario_result_id=scenario_result_id)
                return
            self._active_scenario_result_id = None
            self._release_completed_task(scenario_result_id=scenario_result_id)
            self._queue_revision += 1

    def _release_completed_task(self, *, scenario_result_id: str) -> None:
        """Release executable state while retaining bounded terminal error evidence."""
        completed = self._active_tasks.pop(scenario_result_id, None)
        if completed is None or completed.error is None:
            return
        self._terminal_errors[scenario_result_id] = completed.error
        self._terminal_errors.move_to_end(scenario_result_id)
        while len(self._terminal_errors) > _MAX_TERMINAL_ERRORS:
            self._terminal_errors.popitem(last=False)

    def _schedule_handoff_retry(self, *, scenario_result_id: str) -> None:
        """Retry a failed terminal handoff without permitting another active run."""
        retry_task = asyncio.create_task(self._retry_handoff_async(scenario_result_id=scenario_result_id))
        self._handoff_retry_tasks.add(retry_task)
        retry_task.add_done_callback(self._handoff_retry_tasks.discard)

    def _schedule_terminalization_retry(self, *, active: _ActiveTask) -> None:
        """Retry cancellation persistence before releasing the active slot."""
        retry_task = asyncio.create_task(self._retry_terminalization_async(active=active))
        self._handoff_retry_tasks.add(retry_task)
        retry_task.add_done_callback(self._handoff_retry_tasks.discard)

    async def _retry_handoff_async(self, *, scenario_result_id: str) -> None:
        """Retry scheduler handoff with bounded exponential delay until it succeeds or shutdown begins."""
        delay = _SCHEDULER_RETRY_INITIAL_SECONDS
        while not self._stopping and self._active_scenario_result_id == scenario_result_id:
            await asyncio.sleep(delay)
            try:
                await self._handoff_scheduler_async(scenario_result_id=scenario_result_id)
            except Exception:
                logger.exception("Scenario scheduler handoff retry failed for %s.", scenario_result_id)
                delay = min(delay * 2, _SCHEDULER_RETRY_MAX_SECONDS)
            else:
                return

    async def _retry_terminalization_async(self, *, active: _ActiveTask) -> None:
        """Retry a failed cancellation transition, then perform the terminal handoff."""
        delay = _SCHEDULER_RETRY_INITIAL_SECONDS
        while not self._stopping and self._active_scenario_result_id == active.scenario_result_id:
            await asyncio.sleep(delay)
            try:
                async with self._scheduler_lock:
                    if self._stopping or self._active_scenario_result_id != active.scenario_result_id:
                        return
                    await asyncio.to_thread(
                        self._memory.update_scenario_run_state,
                        scenario_result_id=active.scenario_result_id,
                        scenario_run_state=active.cancellation_state,
                        error_message=active.cancellation_reason,
                        error_type=active.cancellation_error_type,
                    )
                    if not active.retain_error_on_terminalization:
                        active.error = None
                await self._handoff_scheduler_async(scenario_result_id=active.scenario_result_id)
            except Exception:
                logger.exception(
                    "Scenario terminal transition retry failed for %s.",
                    active.scenario_result_id,
                )
                delay = min(delay * 2, _SCHEDULER_RETRY_MAX_SECONDS)
            else:
                return

    async def _complete_handoff_async(self, *, scenario_result_id: str) -> None:
        """Complete terminal handoff even if the execution task is cancelled while waiting for the scheduler lock."""
        handoff_task = asyncio.create_task(self._handoff_scheduler_async(scenario_result_id=scenario_result_id))
        self._handoff_retry_tasks.add(handoff_task)
        handoff_task.add_done_callback(self._handoff_retry_tasks.discard)
        try:
            await asyncio.shield(handoff_task)
        except asyncio.CancelledError:
            try:
                await handoff_task
            except asyncio.CancelledError:
                return
            except Exception:
                logger.exception("Scenario scheduler handoff failed for %s; retrying.", scenario_result_id)
                if not self._stopping:
                    self._schedule_handoff_retry(scenario_result_id=scenario_result_id)
        except Exception:
            logger.exception("Scenario scheduler handoff failed for %s; retrying.", scenario_result_id)
            if not self._stopping:
                self._schedule_handoff_retry(scenario_result_id=scenario_result_id)

    @staticmethod
    def _build_queue_entry(
        *,
        run: _ActiveTask,
        state: ScenarioRunState,
        position: int | None = None,
    ) -> ScenarioQueueEntry:
        """
        Map event-loop scheduler state to the canonical queue DTO.

        Returns:
            ScenarioQueueEntry: Canonical active or queued entry.
        """
        if run.created_at is None or run.enqueued_at is None:
            raise RuntimeError(f"Scenario run '{run.scenario_result_id}' has incomplete queue timestamps.")
        return ScenarioQueueEntry(
            scenario_result_id=run.scenario_result_id,
            scenario_name=run.scenario_name,
            scenario_registry_name=run.scenario_registry_name,
            created_at=run.created_at,
            enqueued_at=run.enqueued_at,
            started_at=run.started_at,
            state=state,
            position=position,
        )

    @staticmethod
    def _is_terminal_state(state: ScenarioRunState) -> bool:
        """Return whether a scenario state is terminal."""
        return state in (ScenarioRunState.COMPLETED, ScenarioRunState.FAILED, ScenarioRunState.CANCELLED)

    def _resolve_scenario_class(self, *, request: RunScenarioRequest) -> type[Scenario]:
        """
        Validate and resolve the scenario class from the registry.

        Args:
            request: The run request containing the scenario name.

        Returns:
            The scenario class.

        Raises:
            ValueError: If the scenario name is not found in the registry.
        """
        scenario_registry = ScenarioRegistry.get_registry_singleton()
        try:
            return scenario_registry.get_class(request.scenario_name)
        except KeyError as e:
            raise ValueError(str(e)) from None

    async def _run_initializers_async(self, *, request: RunScenarioRequest) -> None:
        """
        Validate and execute initializers specified in the request.

        Args:
            request: The run request containing initializer names and args.

        Raises:
            ValueError: If an initializer name is not found in the registry.
        """
        if not request.initializers:
            return

        initializer_registry = InitializerRegistry.get_registry_singleton()
        for initializer_name in request.initializers:
            initializer_params = (request.initializer_args or {}).get(initializer_name)
            try:
                instance = initializer_registry.create_and_configure(
                    initializer_name, initializer_params=initializer_params
                )
            except KeyError as e:
                raise ValueError(f"Initializer not found: {e}") from None
            await instance.initialize_async()

    def _resolve_target(self, *, request: RunScenarioRequest) -> "PromptTarget":
        """
        Resolve the objective target from the target registry.

        Args:
            request: The run request containing the target name.

        Returns:
            The resolved PromptTarget instance.

        Raises:
            ValueError: If the target is not found in the registry.
        """
        return self.resolve_target_name(target_name=request.target_name)

    @staticmethod
    def resolve_target_name(*, target_name: str) -> "PromptTarget":
        """
        Resolve one registered target name for launch or configured estimation.

        Args:
            target_name: Registered target instance name.

        Returns:
            PromptTarget: The resolved target.

        Raises:
            ValueError: If the target is not registered.
        """
        target_registry = TargetRegistry.get_registry_singleton()
        objective_target = target_registry.instances.get(target_name)
        if objective_target is None:
            available_names = target_registry.instances.get_names()
            if not available_names:
                raise ValueError(
                    f"Target '{target_name}' not found. The target registry is empty. "
                    "Make sure to include an initializer that registers targets "
                    "(e.g., initializers: ['target'])."
                )
            raise ValueError(
                f"Target '{target_name}' not found in registry. Available targets: {', '.join(available_names)}"
            )
        return objective_target

    def _build_init_kwargs(
        self, *, request: RunScenarioRequest, scenario_class: type[Scenario], objective_target: Any
    ) -> dict[str, Any]:
        """
        Build the kwargs dict for scenario.initialize_async.

        Resolves techniques and dataset configuration from the request.

        Dataset configuration is built so that the scenario's default
        ``DatasetAttackConfiguration`` *subclass* (e.g. ``EncodingDatasetConfiguration``)
        is preserved when the caller overrides ``dataset_names`` or
        ``max_dataset_size``. Subclasses commonly override
        ``_build_attack_groups()`` to shape seeds into scenario-appropriate
        ``AttackSeedGroup`` objects.

        Args:
            request: The run request.
            scenario_class: The resolved scenario class.
            objective_target: The resolved target instance.

        Returns:
            Dict of kwargs to pass to scenario.initialize_async.

        Raises:
            ValueError: If a technique name is invalid for the scenario, or the
                scenario class cannot be instantiated with no arguments when
                introspection is required to resolve techniques or dataset
                configuration.
        """
        return self.resolve_scenario_configuration(
            scenario_name=request.scenario_name,
            scenario_class=scenario_class,
            objective_target=objective_target,
            techniques=request.techniques,
            dataset_names=request.dataset_names,
            max_dataset_size=request.max_dataset_size,
            dataset_filters=request.dataset_filters,
            include_baseline=request.include_baseline,
            max_concurrency=request.max_concurrency,
            max_retries=request.max_retries,
            memory_labels=request.labels,
        )

    @classmethod
    def resolve_scenario_configuration(
        cls,
        *,
        scenario_name: str,
        scenario_class: type[Scenario],
        objective_target: Any | None = None,
        techniques: list[str] | None = None,
        dataset_names: list[str] | None = None,
        max_dataset_size: int | None = None,
        dataset_filters: dict[str, list[str]] | None = None,
        include_baseline: bool | None = None,
        max_concurrency: int | None = None,
        max_retries: int | None = None,
        memory_labels: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Resolve shared launch/estimate request fields into scenario parameters.

        Args:
            scenario_name: Registered scenario name used in validation errors.
            scenario_class: Scenario class used for technique and dataset introspection.
            objective_target: Optional resolved objective target.
            techniques: Requested technique tokens.
            dataset_names: Requested dataset names.
            max_dataset_size: Requested logical-group selection cap.
            dataset_filters: Validated dataset seed filters.
            include_baseline: Optional baseline policy override.
            max_concurrency: Optional launch concurrency.
            max_retries: Optional launch retry count.
            memory_labels: Optional launch memory labels.

        Returns:
            dict[str, Any]: Values accepted by ``Scenario.set_params_from_args``.

        Raises:
            ValueError: If techniques or dataset overrides are invalid.
        """
        resolved: dict[str, Any] = {}
        if objective_target is not None:
            resolved["objective_target"] = objective_target
        if max_concurrency is not None:
            resolved["max_concurrency"] = max_concurrency
        if max_retries is not None:
            resolved["max_retries"] = max_retries
        if include_baseline is not None:
            resolved["include_baseline"] = include_baseline
        if memory_labels:
            resolved["memory_labels"] = memory_labels

        filters = dataset_filters or {}
        needs_introspection = bool(techniques) or bool(dataset_names) or max_dataset_size is not None or bool(filters)
        if not needs_introspection:
            return resolved

        try:
            introspection_instance = scenario_class()  # type: ignore[ty:missing-argument]
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve runtime configuration for scenario '{scenario_name}': "
                f"scenario class is not instantiable without arguments ({exc})."
            ) from exc

        if techniques:
            technique_class = introspection_instance._technique_class
            technique_enums, technique_converters = cls._resolve_techniques_and_converters(
                tokens=techniques,
                technique_class=technique_class,
                scenario_name=scenario_name,
            )
            resolved["scenario_techniques"] = technique_enums
            if technique_converters:
                resolved["technique_converters"] = technique_converters

        if dataset_names or max_dataset_size is not None or filters:
            default_config = introspection_instance._default_dataset_config

            if isinstance(default_config, CompoundDatasetAttackConfiguration):
                names_changed = dataset_names is not None and dataset_names != default_config.dataset_names
                if names_changed:
                    try:
                        resolved["dataset_config"] = default_config.with_dataset_names(
                            dataset_names=dataset_names,
                            max_dataset_size=max_dataset_size,
                            filters=filters or None,
                        )
                    except TypeError as exc:
                        raise ValueError(
                            f"Scenario '{scenario_name}' does not support overriding datasets through "
                            f"its {type(default_config).__name__} configuration: {exc}"
                        ) from exc
                else:
                    if max_dataset_size is not None:
                        default_config.update_child_max_dataset_size(max_dataset_size=max_dataset_size)
                    if filters:
                        default_config.update_filters(filters=filters)
                    resolved["dataset_config"] = default_config
            elif dataset_names:
                # Construct a fresh instance of the scenario's own dataset-config
                # class so subclass-specific behavior is preserved.
                default_config_class = type(default_config)
                try:
                    resolved["dataset_config"] = default_config_class(
                        dataset_names=dataset_names,
                        max_dataset_size=max_dataset_size,
                        filters=filters or None,
                    )
                except TypeError as exc:
                    raise ValueError(
                        f"Scenario '{scenario_name}' does not support overriding dataset names through "
                        f"its {default_config_class.__name__} configuration: {exc}"
                    ) from exc
            else:
                # Reuse the scenario's default dataset config (preserves subtype +
                # the scenario's own default dataset names) and override only the
                # sample cap and/or filters. Safe because the introspection instance
                # is throwaway.
                if max_dataset_size is not None:
                    default_config.max_dataset_size = max_dataset_size
                if filters:
                    default_config.update_filters(filters=filters)
                resolved["dataset_config"] = default_config

        return resolved

    @classmethod
    def _resolve_techniques_and_converters(
        cls,
        *,
        tokens: list[str],
        technique_class: type[Any],
        scenario_name: str,
    ) -> tuple[list[Any], dict[str, list["Converter"]]]:
        """
        Resolve ``--techniques`` tokens into technique enums and per-technique converters.

        Each token has the form ``<technique>[:converter.<name>[:converter.<name>...]]``.
        The base ``<technique>`` is resolved to a ``ScenarioTechnique`` enum member (which may
        be an aggregate). Each ``converter.<name>`` modifier is resolved to a registered
        converter instance and appended (in token order) to every concrete technique that the
        base technique expands to.

        Args:
            tokens: The raw technique tokens from the request.
            technique_class: The scenario's ``ScenarioTechnique`` subclass.
            scenario_name: The scenario name, used for error messages.

        Returns:
            A tuple of (technique enums to pass as ``scenario_techniques``, mapping from concrete
            technique name to the list of converters to append for that technique).

        Raises:
            ValueError: If a base technique name is unknown, a modifier is malformed, or a
                converter name is not registered.
        """
        technique_enums: list[Any] = []
        technique_converters: dict[str, list[Converter]] = {}

        for token in tokens:
            base_name, _, remainder = token.partition(":")
            modifiers = [m for m in remainder.split(":") if m] if remainder else []

            try:
                technique_enum = technique_class(base_name)
            except ValueError:
                available_techniques = [s.value for s in technique_class]
                raise ValueError(
                    f"Technique '{base_name}' not found for scenario '{scenario_name}'. "
                    f"Available: {', '.join(available_techniques)}"
                ) from None
            technique_enums.append(technique_enum)

            converters = cls._resolve_converter_modifiers(modifiers=modifiers, token=token)
            if not converters:
                continue

            for concrete in technique_class.expand({technique_enum}):
                technique_converters.setdefault(concrete.value, []).extend(converters)

        return technique_enums, technique_converters

    @staticmethod
    def _resolve_converter_modifiers(*, modifiers: list[str], token: str) -> list["Converter"]:
        """
        Resolve the converter modifiers of a single technique token to converter instances.

        Args:
            modifiers: The modifier segments of the token (everything after the base technique).
            token: The full original token, used for error messages.

        Returns:
            The resolved converter instances in token order.

        Raises:
            ValueError: If a modifier does not use the ``converter.`` prefix or names a
                converter that is not registered.
        """
        if not modifiers:
            return []

        instances = ConverterRegistry.get_registry_singleton().instances
        converters: list[Converter] = []
        for modifier in modifiers:
            if not modifier.startswith(_CONVERTER_MODIFIER_PREFIX):
                raise ValueError(
                    f"Unknown technique modifier '{modifier}' in '{token}'. "
                    f"Supported modifiers must use the '{_CONVERTER_MODIFIER_PREFIX}' prefix "
                    f"(e.g. '{_CONVERTER_MODIFIER_PREFIX}translation_spanish')."
                )
            converter_name = modifier[len(_CONVERTER_MODIFIER_PREFIX) :]
            converter = instances.get(converter_name)
            if converter is None:
                available = instances.get_names()
                available_text = ", ".join(available) if available else "(none registered)"
                raise ValueError(
                    f"Converter '{converter_name}' in '{token}' is not a registered converter "
                    f"instance. Available converters: {available_text}"
                )
            converters.append(converter)
        return converters

    async def _initialize_scenario_async(self, *, request: RunScenarioRequest, init_kwargs: dict[str, Any]) -> Scenario:
        """
        Build and initialize the scenario via the registry.

        Delegates the full create + set-parameters + initialize lifecycle to
        ``ScenarioRegistry.create_and_initialize_async`` so the registry owns
        scenario creation and initialization. The run-specific common parameters
        (target, techniques, dataset config, concurrency) are resolved by
        ``_build_init_kwargs`` and forwarded as ``init_kwargs``.

        Args:
            request: The run request (for scenario_name, scenario_params, and
                scenario_result_id).
            init_kwargs: The resolved common parameters to pass to
                scenario.initialize_async.

        Returns:
            The fully initialized Scenario instance ready for run_async.
        """
        scenario_registry = ScenarioRegistry.get_registry_singleton()
        return await scenario_registry.create_and_initialize_async(
            request.scenario_name,
            scenario_params=request.scenario_params or {},
            scenario_result_id=request.scenario_result_id or None,
            **init_kwargs,
        )

    async def _execute_run_async(self, *, scenario_result_id: str) -> None:
        """
        Execute a scenario run (background task entry point).

        Only calls scenario.run_async on the already-initialized scenario.

        Terminal handoff releases executable objects. Bounded error evidence is
        retained separately for later status polling.

        Args:
            scenario_result_id: The scenario result ID for this run.
        """
        active = self._active_tasks[scenario_result_id]
        assert active.scenario is not None
        handoff_ready = True

        try:
            await active.scenario.run_async()

        except asyncio.CancelledError:
            try:
                await asyncio.to_thread(
                    self._memory.update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    scenario_run_state=active.cancellation_state,
                    error_message=active.cancellation_reason,
                    error_type=active.cancellation_error_type,
                )
            except Exception as exc:
                handoff_ready = False
                active.error = str(exc)
                if not self._stopping:
                    self._schedule_terminalization_retry(active=active)
                raise
            logger.info("Scenario run %s stopped in state %s.", scenario_result_id, active.cancellation_state.value)

        except Exception as e:
            active.error = str(e)
            active.cancellation_state = ScenarioRunState.FAILED
            active.cancellation_reason = str(e)
            active.cancellation_error_type = type(e).__name__
            active.retain_error_on_terminalization = True
            try:
                await asyncio.to_thread(
                    self._memory.update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    scenario_run_state=ScenarioRunState.FAILED,
                    error_message=str(e),
                    error_type=type(e).__name__,
                )
            except Exception:
                handoff_ready = False
                if not self._stopping:
                    self._schedule_terminalization_retry(active=active)
                logger.exception("Failed to persist terminal state for scenario run %s.", scenario_result_id)
            logger.exception(f"Scenario run {scenario_result_id} failed: {e}")

        finally:
            if handoff_ready:
                await self._complete_handoff_async(scenario_result_id=scenario_result_id)

    def _build_response(
        self,
        *,
        scenario_result_id: str,
        active_error: str | None,
        queue_position: int | None,
        active_scenario_result_id: str | None,
    ) -> ScenarioRunSummary | None:
        """
        Build a ScenarioRunResponse by querying the database and merging active task state.

        Args:
            scenario_result_id: The scenario result ID.
            active_error: Error copied from the active asyncio task, if any.
            queue_position: Current 1-based waiting position, if queued.
            active_scenario_result_id: Currently executing scenario result ID.

        Returns:
            ScenarioRunResponse if found in the database, None otherwise.
        """
        results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
        if not results:
            return None
        return self._build_response_from_db(
            scenario_result=results[0],
            active_error=active_error,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
        )

    def _build_response_from_db(
        self,
        *,
        scenario_result: ScenarioResult,
        active_error: str | None = None,
        queue_position: int | None = None,
        active_scenario_result_id: str | None = None,
    ) -> ScenarioRunSummary:
        """
        Build a ScenarioRunResponse from a database ScenarioResult, merged with active task info.

        Args:
            scenario_result: A ScenarioResult retrieved from CentralMemory.
            active_error: Error copied from the active asyncio task, if any.
            queue_position: Current 1-based waiting position, if queued.
            active_scenario_result_id: Currently executing scenario result ID.

        Returns:
            The API response model.
        """
        scenario_result_id = str(scenario_result.id)

        # Primary source: DB-persisted error fields
        error = scenario_result.error_message
        error_type = scenario_result.error_type

        # Fallback: look up error from any persisted error AttackResults linked
        # to this scenario via the new attribution_parent_id foreign key.
        if not error:
            error_ars = self._memory.get_attack_results(
                scenario_result_id=scenario_result_id,
                outcome=AttackOutcome.ERROR,
            )
            if error_ars:
                error = error_ars[0].error_message
                error_type = error_ars[0].error_type

        # Fallback: in-memory error for in-flight tasks where DB hasn't been updated yet
        if not error:
            error = active_error

        status = scenario_result.scenario_run_state
        terminal = self._is_terminal_state(status)
        plan = self._load_run_plan(scenario_result=scenario_result)

        # Build result fields from DB (always computed so in-progress runs show progress)
        total_attacks, completed_attacks, objective_achieved_rate, successful_attacks = self._calculate_progress_counts(
            scenario_result=scenario_result,
            plan=plan,
        )
        techniques_used = self._resolve_techniques_used(
            scenario_identifier=scenario_result.scenario_identifier,
            atomic_groups=plan.atomic_groups if plan is not None else None,
            fallback_names=scenario_result.get_techniques_used(),
        )
        target, datasets_used, scenario_parameters = self._safe_run_metadata(
            scenario_identifier=getattr(scenario_result, "scenario_identifier", None)
        )
        diagnostics = self._collect_run_diagnostics(scenario_result=scenario_result)
        header = self._build_run_header(
            scenario_result=scenario_result,
            scenario_registry_name=plan.scenario_registry_name if plan else None,
            techniques_used=techniques_used,
            target=target,
            datasets_used=datasets_used,
            scenario_parameters=scenario_parameters,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
            overload_summaries=diagnostics.overload_summaries,
        )
        updated_at = scenario_result.creation_time
        if terminal and scenario_result.completion_time is not None:
            updated_at = scenario_result.completion_time

        return ScenarioRunSummary(
            **header.model_dump(),
            updated_at=updated_at,
            error=error,
            error_type=error_type,
            total_attacks=total_attacks,
            completed_attacks=completed_attacks,
            objective_achieved_rate=objective_achieved_rate,
            failed_attacks=diagnostics.failed_attacks,
            attack_retries=diagnostics.attack_retries,
            total_retries=diagnostics.total_retries,
            planned_total_available=plan is not None,
            successful_attacks=successful_attacks,
            error_attacks=len(diagnostics.failed_attacks),
        )

    def _collect_run_diagnostics(self, *, scenario_result: ScenarioResult) -> _RunDiagnostics:
        """
        Summarize persisted errors, retries, and overload evidence.

        Returns:
            _RunDiagnostics: Aggregated diagnostics for the run.
        """
        failed_attacks: list[AttackErrorSummary] = []
        attack_retries: list[AttackRetrySummary] = []
        overload_events: deque[Any] = deque(maxlen=_MAX_OVERLOAD_EVENTS)
        total_retries = 0
        for atomic_attack_name, results in scenario_result.attack_results.items():
            for attack_result in results:
                retries = attack_result.total_retries
                total_retries += retries
                if attack_result.retry_events:
                    overload_events.extend(attack_result.retry_events)
                    attack_retries.append(
                        AttackRetrySummary(
                            attack_result_id=str(attack_result.attack_result_id),
                            atomic_attack_name=atomic_attack_name,
                            retries=attack_result.retry_events,
                        )
                    )
                if attack_result.outcome == AttackOutcome.ERROR:
                    failed_attacks.append(
                        AttackErrorSummary(
                            atomic_attack_name=atomic_attack_name,
                            objective=attack_result.objective,
                            error_type=attack_result.error_type,
                            error_message=attack_result.error_message,
                            total_retries=retries,
                        )
                    )
        return _RunDiagnostics(
            failed_attacks=failed_attacks,
            attack_retries=attack_retries,
            total_retries=total_retries,
            overload_summaries=self._build_overload_summaries(retry_events=overload_events),
        )

    def _build_run_header(
        self,
        *,
        scenario_result: ScenarioResult,
        scenario_registry_name: str | None,
        techniques_used: Sequence[str],
        target: ScenarioTargetSummary | None,
        datasets_used: Sequence[str],
        scenario_parameters: Mapping[str, Any],
        queue_position: int | None = None,
        active_scenario_result_id: str | None = None,
        overload_summaries: Sequence[ScenarioOverloadSummary] = (),
    ) -> ScenarioRunHeader:
        """
        Build fields shared by full summaries and progress responses.

        Returns:
            ScenarioRunHeader: Canonical shared run fields.
        """
        status = scenario_result.scenario_run_state
        return ScenarioRunHeader(
            scenario_result_id=str(scenario_result.id),
            scenario_name=scenario_result.scenario_name,
            scenario_registry_name=scenario_registry_name,
            scenario_version=scenario_result.scenario_version,
            status=status,
            created_at=scenario_result.creation_time,
            completed_at=scenario_result.completion_time if self._is_terminal_state(status) else None,
            pyrit_version=scenario_result.pyrit_version if isinstance(scenario_result.pyrit_version, str) else None,
            target=target,
            techniques_used=list(techniques_used),
            datasets_used=list(datasets_used),
            scenario_parameters=dict(scenario_parameters),
            labels=scenario_result.labels,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
            overload_summaries=list(overload_summaries),
        )

    @staticmethod
    def _resolve_techniques_used(
        *,
        scenario_identifier: ScenarioIdentifier | None,
        atomic_groups: Sequence[ScenarioRunPlanAtomicGroup] | None,
        fallback_names: Sequence[str],
    ) -> list[str]:
        """
        Resolve configured, planned, or legacy technique display names.

        Returns:
            list[str]: De-duplicated technique display names.
        """
        configured = list(dict.fromkeys(scenario_identifier.techniques or [])) if scenario_identifier else []
        if configured:
            return configured
        if atomic_groups is not None:
            return list(dict.fromkeys(group.display_group for group in atomic_groups))
        return list(dict.fromkeys(fallback_names))

    @staticmethod
    def _parse_history_plan(
        *,
        record: ScenarioHistoryRunRecord,
    ) -> tuple[list[ScenarioRunPlanAtomicGroup] | None, dict[str, str]]:
        """
        Parse the compact run-plan projection used by history.

        Returns:
            tuple[list[ScenarioRunPlanAtomicGroup] | None, dict[str, str]]:
                Parsed atomic groups and objective-hash-to-seed-ID mapping.
        """
        if record.plan_atomic_groups is None:
            return None, {}
        try:
            raw_atomic_groups = (
                json.loads(record.plan_atomic_groups)
                if isinstance(record.plan_atomic_groups, str)
                else record.plan_atomic_groups
            )
            atomic_groups = _HISTORY_ATOMIC_GROUPS_ADAPTER.validate_python(raw_atomic_groups)
            group_ids = [group.id for group in atomic_groups]
            if len(group_ids) != len(set(group_ids)):
                raise ValueError("duplicate atomic group IDs")

            raw_seed_map = (
                json.loads(record.plan_seed_id_map)
                if isinstance(record.plan_seed_id_map, str)
                else record.plan_seed_id_map
            )
            seed_map = _HISTORY_SEED_ID_MAP_ADAPTER.validate_python(raw_seed_map)
            seed_id_by_objective_hash: dict[str, str] = {}
            for seed in seed_map:
                objective_sha256 = seed["objective_sha256"]
                seed_id = seed["id"]
                previous_seed_id = seed_id_by_objective_hash.get(objective_sha256)
                if previous_seed_id is not None and previous_seed_id != seed_id:
                    raise ValueError("ambiguous objective hash in run plan")
                seed_id_by_objective_hash[objective_sha256] = seed_id
            return atomic_groups, seed_id_by_objective_hash
        except (json.JSONDecodeError, ValidationError, ValueError):
            logger.warning(
                "Scenario run %s has an incomplete persisted plan; using legacy history totals.",
                record.scenario_result_id,
            )
            return None, {}

    def _build_history_summary(
        self,
        *,
        record: ScenarioHistoryRunRecord,
        units: list[ScenarioHistoryUnitRecord],
    ) -> ScenarioRunSummary:
        """
        Map lightweight persisted history projections to the public summary DTO.

        Returns:
            ScenarioRunSummary: Safe, aggregated history summary.
        """
        scenario_identifier = None
        try:
            scenario_identifier = ScenarioIdentifier.from_component_identifier(
                ComponentIdentifier.model_validate(
                    {**record.scenario_identifier, "pyrit_version": record.pyrit_version}
                )
            )
        except (ValidationError, ValueError):
            logger.warning(
                "Scenario run %s has invalid persisted identifier metadata; using legacy history fields.",
                record.scenario_result_id,
            )
        atomic_groups, seed_id_by_objective_hash = self._parse_history_plan(record=record)
        target, datasets_used, scenario_parameters = self._safe_run_metadata(scenario_identifier=scenario_identifier)
        if target is None and record.objective_target_identifier:
            try:
                target = self._safe_target_metadata(
                    target_identifier=TargetIdentifier.from_component_identifier(
                        ComponentIdentifier.model_validate(record.objective_target_identifier)
                    )
                )
            except ValidationError:
                logger.warning(
                    "Scenario run %s has invalid persisted target metadata; omitting the target summary.",
                    record.scenario_result_id,
                )

        units_by_key: dict[tuple[str, str], ScenarioHistoryUnitRecord] = {}
        for unit in units:
            unit_key = self._history_unit_key(
                unit=unit,
                atomic_groups=atomic_groups,
                seed_id_by_objective_hash=seed_id_by_objective_hash,
            )
            existing = units_by_key.get(unit_key)
            units_by_key[unit_key] = self._merge_history_units(existing=existing, incoming=unit) if existing else unit
        planned_units = (
            {(group.id, seed_group_id) for group in atomic_groups for seed_group_id in group.seed_group_ids}
            if atomic_groups is not None
            else set(units_by_key)
        )
        included_units = [unit for key, unit in units_by_key.items() if key in planned_units]
        completed_units = [unit for unit in included_units if unit.latest_outcome != AttackOutcome.ERROR.value]
        successful = sum(unit.latest_outcome == AttackOutcome.SUCCESS.value for unit in completed_units)
        error_count = sum(unit.error_count for unit in included_units)
        retry_count = sum(max(0, unit.total_retries) for unit in included_units)
        status = ScenarioRunState(record.status)
        terminal = self._is_terminal_state(status)
        timestamps = [record.created_at, *(unit.latest_timestamp for unit in units)]
        if terminal and record.completed_at is not None:
            timestamps.append(record.completed_at)
        updated_at = max(timestamps)
        techniques = self._resolve_techniques_used(
            scenario_identifier=scenario_identifier,
            atomic_groups=atomic_groups,
            fallback_names=sorted({unit.atomic_attack_name for unit in units if unit.atomic_attack_name}),
        )
        header = ScenarioRunHeader(
            scenario_result_id=record.scenario_result_id,
            scenario_name=record.scenario_name,
            scenario_registry_name=record.scenario_registry_name,
            scenario_version=record.scenario_version,
            status=status,
            created_at=record.created_at,
            completed_at=record.completed_at if terminal else None,
            pyrit_version=record.pyrit_version,
            target=target,
            techniques_used=techniques,
            datasets_used=datasets_used,
            scenario_parameters=scenario_parameters,
            labels=record.labels,
        )
        completed = len(completed_units)
        return ScenarioRunSummary(
            **header.model_dump(),
            updated_at=updated_at,
            error=record.error_message,
            error_type=record.error_type,
            total_attacks=len(planned_units),
            completed_attacks=completed,
            objective_achieved_rate=int((successful / completed) * 100) if completed else 0,
            failed_attacks=[],
            total_retries=retry_count,
            planned_total_available=atomic_groups is not None,
            successful_attacks=successful,
            error_attacks=error_count,
            attack_details_available=False,
        )

    @staticmethod
    def _history_unit_key(
        *,
        unit: ScenarioHistoryUnitRecord,
        atomic_groups: list[ScenarioRunPlanAtomicGroup] | None,
        seed_id_by_objective_hash: dict[str, str],
    ) -> tuple[str, str]:
        """
        Resolve a projected history attempt to its logical planned unit.

        Returns:
            tuple[str, str]: Atomic-group and logical seed-group IDs.
        """
        atomic_group_id = unit.atomic_attack_name
        if atomic_groups is not None:
            for group in atomic_groups:
                if group.atomic_attack_name == unit.atomic_attack_name and (
                    not unit.technique_eval_hash or group.technique_eval_hash == unit.technique_eval_hash
                ):
                    atomic_group_id = group.id
                    break
        seed_group_id = seed_id_by_objective_hash.get(unit.seed_group_id, unit.seed_group_id)
        return atomic_group_id, seed_group_id

    @staticmethod
    def _merge_history_units(
        *,
        existing: ScenarioHistoryUnitRecord,
        incoming: ScenarioHistoryUnitRecord,
    ) -> ScenarioHistoryUnitRecord:
        """
        Merge attempt partitions that resolve to the same persisted logical unit.

        Returns:
            ScenarioHistoryUnitRecord: Combined counters and preferred latest outcome.
        """
        existing_completed = existing.latest_outcome != AttackOutcome.ERROR.value
        incoming_completed = incoming.latest_outcome != AttackOutcome.ERROR.value
        if incoming_completed != existing_completed:
            preferred = incoming if incoming_completed else existing
        else:
            preferred = incoming if incoming.latest_timestamp > existing.latest_timestamp else existing
        return ScenarioHistoryUnitRecord(
            scenario_result_id=preferred.scenario_result_id,
            atomic_attack_name=preferred.atomic_attack_name,
            technique_eval_hash=preferred.technique_eval_hash,
            seed_group_id=preferred.seed_group_id,
            objective_sha256=preferred.objective_sha256 or existing.objective_sha256 or incoming.objective_sha256,
            latest_outcome=preferred.latest_outcome,
            latest_timestamp=max(existing.latest_timestamp, incoming.latest_timestamp),
            total_retries=max(0, existing.total_retries) + max(0, incoming.total_retries),
            error_count=max(0, existing.error_count) + max(0, incoming.error_count),
        )

    @staticmethod
    def _safe_run_metadata(
        *,
        scenario_identifier: ScenarioIdentifier | None,
    ) -> tuple[ScenarioTargetSummary | None, list[str], dict[str, Any]]:
        """
        Project canonical identifiers to an allow-listed, secret-free API shape.

        Returns:
            tuple[ScenarioTargetSummary | None, list[str], dict[str, Any]]:
                Safe target, datasets, and scenario parameters.
        """
        if scenario_identifier is None:
            return None, [], {}

        target = ScenarioRunService._safe_target_metadata(target_identifier=scenario_identifier.objective_target)
        return (
            target,
            list(scenario_identifier.datasets or []),
            ScenarioRunService._safe_scenario_parameters(parameters=dict(scenario_identifier.params)),
        )

    @staticmethod
    def _build_overload_summaries(*, retry_events: Sequence[Any]) -> list[ScenarioOverloadSummary]:
        """
        Aggregate bounded HTTP overload evidence by component role.

        Returns:
            list[ScenarioOverloadSummary]: Most recently affected roles first.
        """
        aggregates: dict[str, dict[str, Any]] = {}
        for event in retry_events:
            status_code = getattr(event, "status_code", None)
            if not isinstance(status_code, int) or (status_code != 429 and not 500 <= status_code <= 599):
                continue
            role = str(getattr(event, "component_role", "") or "unknown")
            timestamp = getattr(event, "timestamp", None)
            if not isinstance(timestamp, datetime):
                continue
            aggregate = aggregates.setdefault(
                role,
                {
                    "count": 0,
                    "rate_limit_count": 0,
                    "server_error_count": 0,
                    "status_codes": set(),
                    "latest_timestamp": timestamp,
                },
            )
            aggregate["count"] += 1
            aggregate["rate_limit_count"] += status_code == 429
            aggregate["server_error_count"] += 500 <= status_code <= 599
            aggregate["status_codes"].add(status_code)
            aggregate["latest_timestamp"] = max(aggregate["latest_timestamp"], timestamp)
        ordered = sorted(
            aggregates.items(),
            key=lambda item: item[1]["latest_timestamp"],
            reverse=True,
        )[:_MAX_OVERLOAD_ROLES]
        return [
            ScenarioOverloadSummary(
                component_role=role,
                count=aggregate["count"],
                rate_limit_count=aggregate["rate_limit_count"],
                server_error_count=aggregate["server_error_count"],
                status_codes=sorted(aggregate["status_codes"]),
                latest_timestamp=aggregate["latest_timestamp"],
            )
            for role, aggregate in ordered
        ]

    @staticmethod
    def _safe_target_metadata(*, target_identifier: TargetIdentifier | None) -> ScenarioTargetSummary | None:
        """
        Project a target identifier to the secret-free public shape.

        Returns:
            ScenarioTargetSummary | None: Safe target metadata when available.
        """
        if target_identifier is None:
            return None
        return ScenarioTargetSummary(
            target_type=target_identifier.class_name,
            endpoint=ScenarioRunService._safe_endpoint(target_identifier.endpoint),
            model_name=target_identifier.model_name or target_identifier.underlying_model_name,
            identifier_hash=target_identifier.hash,
        )

    @staticmethod
    def _safe_scenario_parameters(*, parameters: dict[str, Any]) -> dict[str, Any]:
        """
        Return only explicitly approved, JSON-safe scenario configuration fields.

        Returns:
            dict[str, Any]: Allow-listed scenario parameters with sensitive keys removed.
        """
        filtered = filter_sensitive_fields(parameters)
        return {
            key: value
            for key, value in filtered.items()
            if key in _SAFE_SCENARIO_PARAMETER_NAMES
            and (
                value is None
                or isinstance(value, (bool, int, float, str))
                or (
                    isinstance(value, list)
                    and all(item is None or isinstance(item, (bool, int, float, str)) for item in value)
                )
            )
        }

    @staticmethod
    def _safe_endpoint(endpoint: str | None) -> str | None:
        """
        Remove endpoint credentials, query parameters, and fragments.

        Returns:
            str | None: Sanitized endpoint.
        """
        if not endpoint:
            return None
        parsed = urlsplit(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return None
        host = parsed.hostname or ""
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            host = f"{host}:{port}"
        return urlunsplit((parsed.scheme, host, "", "", ""))

    @staticmethod
    def _normalize_history_labels(
        *,
        labels: Mapping[str, str | Sequence[str]] | None,
    ) -> dict[str, str | list[str]] | None:
        """
        Normalize history labels for filtering and cursor binding.

        Returns:
            dict[str, str | list[str]] | None: Canonical effective labels.
        """
        normalized: dict[str, str | list[str]] = {}
        for key in sorted(labels or {}):
            raw_value = (labels or {})[key]
            if isinstance(raw_value, str):
                if raw_value:
                    normalized[key] = raw_value
                continue
            values = sorted({str(value) for value in raw_value if str(value)})
            if values:
                normalized[key] = values
        return normalized or None

    @staticmethod
    def _history_filter_fingerprint(
        *,
        scenario_names: Sequence[str],
        statuses: Sequence[str],
        labels: Mapping[str, str | Sequence[str]] | None,
    ) -> str:
        """Return a stable fingerprint binding a cursor to normalized filters."""
        payload = {
            "scenario_names": sorted(scenario_names),
            "statuses": sorted(statuses),
            "labels": labels,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _encode_history_cursor(*, cursor: ScenarioHistoryKeysetCursor, fingerprint: str) -> str:
        """
        Encode a descending scenario-history keyset anchor.

        Returns:
            str: Opaque cursor.
        """
        payload = {
            "v": 1,
            "f": fingerprint,
            "t": cursor.timestamp.isoformat(),
            "i": cursor.scenario_result_id,
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")

    @staticmethod
    def _decode_history_cursor(
        *,
        cursor: str | None,
        fingerprint: str,
    ) -> ScenarioHistoryKeysetCursor | None:
        """
        Decode and validate a filter-bound scenario-history cursor.

        Returns:
            ScenarioHistoryKeysetCursor | None: Validated keyset anchor.

        Raises:
            ValueError: If the cursor is malformed or belongs to different filters.
        """
        if cursor is None:
            return None
        try:
            padded = cursor + "=" * (-len(cursor) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        except (binascii.Error, UnicodeDecodeError, ValueError, TypeError) as exc:
            raise ValueError("Malformed scenario history cursor.") from exc
        if not isinstance(payload, dict) or payload.get("v") != 1:
            raise ValueError("Malformed scenario history cursor.")
        if payload.get("f") != fingerprint:
            raise ValueError("Scenario history cursor does not match the requested filters.")
        try:
            timestamp = datetime.fromisoformat(payload["t"])
            scenario_result_id = str(uuid.UUID(payload["i"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed scenario history cursor.") from exc
        if timestamp.tzinfo is None:
            raise ValueError("Scenario history cursor timestamp must include a timezone.")
        try:
            timestamp = timestamp.astimezone(timezone.utc)
        except (OverflowError, OSError) as exc:
            raise ValueError("Malformed scenario history cursor.") from exc
        return ScenarioHistoryKeysetCursor(
            timestamp=timestamp,
            scenario_result_id=scenario_result_id,
        )

    def _get_active_task(self, *, scenario_result_id: str) -> _ActiveTask | None:
        """Return executable state for an active run."""
        active = self._active_tasks.get(scenario_result_id)
        if (
            active is not None
            and active.task is not None
            and active.task.done()
            and self._active_scenario_result_id != scenario_result_id
        ):
            self._release_completed_task(scenario_result_id=scenario_result_id)
            return None
        return active

    def snapshot_active_run(self, *, scenario_result_id: str) -> _ActiveRunSnapshot:
        """
        Copy asyncio-owned run state for use by database-only worker-thread methods.

        Returns:
            _ActiveRunSnapshot: An immutable copy of the active state.
        """
        active_scenario_result_id = self._active_scenario_result_id
        queue_position = next(
            (
                position
                for position, queued in enumerate(self._queued_runs, start=1)
                if queued.scenario_result_id == scenario_result_id
            ),
            None,
        )
        active = self._get_active_task(scenario_result_id=scenario_result_id)
        if active is None:
            return _ActiveRunSnapshot(
                error=self._terminal_errors.get(scenario_result_id),
                queue_position=queue_position,
                active_scenario_result_id=active_scenario_result_id,
            )
        active_group_ids = tuple(sorted(active.scenario.active_atomic_group_ids)) if active.scenario is not None else ()
        return _ActiveRunSnapshot(
            error=active.error,
            active_group_ids=active_group_ids,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
        )

    @staticmethod
    def _load_run_plan(*, scenario_result: ScenarioResult) -> ScenarioRunPlan | None:
        """
        Load a validated plan from scenario metadata.

        Returns:
            ScenarioRunPlan | None: The stored plan, or None for a legacy row.
        """
        metadata = getattr(scenario_result, "metadata", None)
        raw_plan = (metadata or {}).get(SCENARIO_RUN_PLAN_METADATA_KEY)
        return ScenarioRunPlan.model_validate(raw_plan) if raw_plan is not None else None

    @staticmethod
    def _result_unit_key(
        *,
        atomic_attack_name: str,
        attack_result: Any,
        plan: ScenarioRunPlan | None,
    ) -> tuple[str, str]:
        """
        Resolve one attack attempt to its stable planned-unit key.

        Returns:
            tuple[str, str]: The atomic-group and seed-group IDs.
        """
        atomic_identifier = getattr(attack_result, "atomic_attack_identifier", None)
        typed_identifier = (
            AtomicAttackIdentifier.from_component_identifier(atomic_identifier)
            if isinstance(atomic_identifier, ComponentIdentifier)
            else None
        )
        objective = str(getattr(attack_result, "objective", ""))
        attribution_data = getattr(attack_result, "attribution_data", None)
        attributed_seed_group_id = attribution_data.get("seed_group_id") if isinstance(attribution_data, dict) else None
        seed_group_id = str(attributed_seed_group_id) if attributed_seed_group_id else ""
        if not seed_group_id and typed_identifier is not None and typed_identifier.seed_identifiers:
            seed_group_id = typed_identifier.logical_seed_group_id
        atomic_group_id = atomic_attack_name
        planned_group: ScenarioRunPlanAtomicGroup | None = None
        if plan is not None:
            eval_hash = attribution_data.get("parent_eval_hash") if isinstance(attribution_data, dict) else None
            for group in plan.atomic_groups:
                if group.atomic_attack_name == atomic_attack_name and (
                    eval_hash is None or group.technique_eval_hash == eval_hash
                ):
                    atomic_group_id = group.id
                    planned_group = group
                    break
        if not seed_group_id and plan is not None and planned_group is not None:
            objective_sha256 = str(getattr(attack_result, "objective_sha256", "") or to_sha256(objective))
            matching_seed_ids = [
                seed.id
                for seed in plan.seed_groups
                if seed.id in planned_group.seed_group_ids and seed.objective_sha256 == objective_sha256
            ]
            if len(matching_seed_ids) == 1:
                seed_group_id = matching_seed_ids[0]
        if not seed_group_id:
            seed_group_id = config_hash({"objective": objective})
        return atomic_group_id, seed_group_id

    def _calculate_progress_counts(
        self,
        *,
        scenario_result: ScenarioResult,
        plan: ScenarioRunPlan | None,
    ) -> tuple[int, int, int, int]:
        """
        Calculate planned-unit totals without inflating retries or error attempts.

        Returns:
            tuple[int, int, int, int]: Total, completed, success-rate percentage,
                and successful-unit count.
        """
        attempted_units: set[tuple[str, str]] = set()
        latest_non_error_by_unit: dict[tuple[str, str], Any] = {}
        for atomic_attack_name, results in scenario_result.attack_results.items():
            for attack_result in results:
                unit_key = self._result_unit_key(
                    atomic_attack_name=atomic_attack_name,
                    attack_result=attack_result,
                    plan=plan,
                )
                attempted_units.add(unit_key)
                if attack_result.outcome == AttackOutcome.ERROR:
                    continue
                previous = latest_non_error_by_unit.get(unit_key)
                if previous is None or self._result_order_key(attack_result) > self._result_order_key(previous):
                    latest_non_error_by_unit[unit_key] = attack_result

        planned_units = (
            {(group.id, seed_group_id) for group in plan.atomic_groups for seed_group_id in group.seed_group_ids}
            if plan is not None
            else attempted_units
        )
        total = len(planned_units)
        completed_results = [
            result for unit_key, result in latest_non_error_by_unit.items() if unit_key in planned_units
        ]
        completed = len(completed_results)
        succeeded = sum(result.outcome == AttackOutcome.SUCCESS for result in completed_results)
        rate = int((succeeded / completed) * 100) if completed else 0
        return total, completed, rate, succeeded

    @staticmethod
    def _result_order_key(attack_result: Any) -> tuple[datetime, str]:
        """Return a deterministic chronological key for one hydrated result attempt."""
        timestamp = getattr(attack_result, "timestamp", None)
        if not isinstance(timestamp, datetime):
            timestamp = datetime.min.replace(tzinfo=timezone.utc)
        return timestamp, str(getattr(attack_result, "attack_result_id", ""))

    def get_run_progress(
        self,
        *,
        scenario_result_id: str,
        since: str | None,
        limit: int,
    ) -> ScenarioRunProgress | None:
        """
        Snapshot live state and return compact incremental progress.

        Returns:
            ScenarioRunProgress | None: Compact progress when the run exists.
        """
        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id)
        return self.get_run_progress_from_storage(
            scenario_result_id=scenario_result_id,
            since=since,
            limit=limit,
            active_group_ids=snapshot.active_group_ids,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
        )

    def get_run_progress_from_storage(
        self,
        *,
        scenario_result_id: str,
        since: str | None,
        limit: int,
        active_group_ids: Sequence[str],
        queue_position: int | None = None,
        active_scenario_result_id: str | None = None,
    ) -> ScenarioRunProgress | None:
        """Return compact database progress using a previously captured live-state snapshot."""
        header_result = self._memory.get_scenario_result_header(scenario_result_id=scenario_result_id)
        if header_result is None:
            return None

        cursor = self._decode_progress_cursor(since=since, scenario_result_id=scenario_result_id)
        deltas, has_more = self._memory.get_scenario_attack_result_deltas(
            scenario_result_id=scenario_result_id,
            cursor=cursor,
            limit=limit,
        )
        plan = self._load_run_plan(scenario_result=header_result)
        plan_complete = plan is not None
        response_plan = plan if since is None else None
        if plan is None and since is None:
            response_plan = self._synthesize_legacy_plan(deltas=deltas)

        results = [self._map_progress_delta(delta=delta, plan=plan or response_plan) for delta in deltas]
        overload_events: deque[Any] = deque(maxlen=_MAX_OVERLOAD_EVENTS)
        for delta in deltas:
            overload_events.extend(delta.retry_events)
        next_cursor = (
            self._encode_progress_cursor(scenario_result_id=scenario_result_id, delta=deltas[-1]) if deltas else since
        )
        scenario_identifier = header_result.scenario_identifier
        target, datasets_used, scenario_parameters = self._safe_run_metadata(scenario_identifier=scenario_identifier)
        techniques_used = self._resolve_techniques_used(
            scenario_identifier=scenario_identifier,
            atomic_groups=plan.atomic_groups if plan is not None else None,
            fallback_names=(),
        )
        header = self._build_run_header(
            scenario_result=header_result,
            scenario_registry_name=plan.scenario_registry_name if plan else None,
            techniques_used=techniques_used,
            target=target,
            datasets_used=datasets_used,
            scenario_parameters=scenario_parameters,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
            overload_summaries=self._build_overload_summaries(retry_events=overload_events),
        )
        return ScenarioRunProgress(
            run=ScenarioProgressHeader(**header.model_dump()),
            plan=response_plan,
            reset=False,
            active_atomic_group_ids=list(active_group_ids),
            results=results,
            next_cursor=next_cursor,
            has_more=has_more,
            plan_complete=plan_complete,
        )

    @staticmethod
    def _map_progress_delta(
        *,
        delta: ScenarioAttackResultDelta,
        plan: ScenarioRunPlan | None,
    ) -> ScenarioProgressResult:
        """
        Map a lightweight memory row to its REST progress representation.

        Returns:
            ScenarioProgressResult: The mapped progress delta.
        """
        atomic_attack_name = str(delta.attribution_data.get("parent_collection") or "")
        eval_hash = delta.attribution_data.get("parent_eval_hash")
        atomic_group_id = config_hash(
            {"atomic_attack_name": atomic_attack_name, "technique_eval_hash": eval_hash or ""}
        )
        if plan is not None:
            for group in plan.atomic_groups:
                if group.atomic_attack_name == atomic_attack_name and (
                    eval_hash is None or group.technique_eval_hash == eval_hash
                ):
                    atomic_group_id = group.id
                    break
        attributed_seed_group_id = delta.attribution_data.get("seed_group_id")
        seed_group_id = str(attributed_seed_group_id) if attributed_seed_group_id else ""
        if (
            not seed_group_id
            and delta.atomic_attack_identifier is not None
            and delta.atomic_attack_identifier.seed_identifiers
        ):
            seed_group_id = delta.atomic_attack_identifier.logical_seed_group_id
        if not seed_group_id and plan is not None and delta.objective_sha256:
            matching_seed_ids = [
                seed.id
                for seed in plan.seed_groups
                if seed.objective_sha256 == delta.objective_sha256
                and any(seed.id in group.seed_group_ids for group in plan.atomic_groups if group.id == atomic_group_id)
            ]
            if len(matching_seed_ids) == 1:
                seed_group_id = matching_seed_ids[0]
        if not seed_group_id:
            seed_group_id = config_hash({"objective": delta.objective})
        result_kind, technique_name, attempt_index = ScenarioRunService._progress_result_semantics(
            delta=delta,
            plan=plan,
            atomic_group_id=atomic_group_id,
            atomic_attack_name=atomic_attack_name,
        )
        return ScenarioProgressResult(
            attack_result_id=delta.attack_result_id,
            atomic_group_id=atomic_group_id,
            atomic_attack_name=atomic_attack_name,
            seed_group_id=seed_group_id,
            outcome=delta.outcome,
            execution_time_ms=delta.execution_time_ms,
            timestamp=delta.timestamp,
            total_retries=delta.total_retries,
            retries=delta.retry_events,
            error_type=delta.error_type,
            error_message=delta.error_message,
            result_kind=result_kind,
            technique_name=technique_name,
            attempt_index=attempt_index,
        )

    @staticmethod
    def _progress_result_semantics(
        *,
        delta: ScenarioAttackResultDelta,
        plan: ScenarioRunPlan | None,
        atomic_group_id: str,
        atomic_attack_name: str,
    ) -> tuple[ScenarioProgressResultKind, str | None, int | None]:
        """
        Resolve typed progress semantics from persisted plan and child labels.

        Returns:
            tuple[ScenarioProgressResultKind, str | None, int | None]:
                Result role, registered technique name, and 1-based Adaptive attempt index.
        """
        technique_name = delta.labels.get(ADAPTIVE_TECHNIQUE_NAME_LABEL) or None
        raw_attempt_index = delta.labels.get(ADAPTIVE_ATTEMPT_LABEL)
        parsed_attempt_index = int(raw_attempt_index) if raw_attempt_index and raw_attempt_index.isdigit() else None
        attempt_index = parsed_attempt_index if parsed_attempt_index and parsed_attempt_index >= 1 else None
        if technique_name:
            return ScenarioProgressResultKind.ADAPTIVE_TECHNIQUE, technique_name, attempt_index

        attack_technique = (
            delta.atomic_attack_identifier.attack_technique if delta.atomic_attack_identifier is not None else None
        )
        attack_identifier = attack_technique.attack if attack_technique is not None else None
        is_sequential_envelope = attack_identifier is not None and attack_identifier.class_name == "SequentialAttack"
        matching_group = (
            next((group for group in plan.atomic_groups if group.id == atomic_group_id), None) if plan else None
        )
        if matching_group is not None:
            if matching_group.group_kind is ScenarioRunPlanGroupKind.DIRECT_BASELINE:
                return ScenarioProgressResultKind.DIRECT_BASELINE, None, None
            if matching_group.group_kind is ScenarioRunPlanGroupKind.ADAPTIVE:
                return ScenarioProgressResultKind.ADAPTIVE_ORCHESTRATION, None, None
            if is_sequential_envelope:
                return ScenarioProgressResultKind.AGGREGATE_PARENT, None, None
            if matching_group.group_kind is ScenarioRunPlanGroupKind.ATTACK:
                return ScenarioProgressResultKind.ATTACK, None, None

        if atomic_attack_name == "baseline":
            return ScenarioProgressResultKind.DIRECT_BASELINE, None, None
        if is_sequential_envelope:
            return ScenarioProgressResultKind.AGGREGATE_PARENT, None, None
        if delta.conversation_id.strip():
            return ScenarioProgressResultKind.ATTACK, None, None
        return ScenarioProgressResultKind.UNKNOWN, None, None

    @staticmethod
    def _synthesize_legacy_plan(*, deltas: list[ScenarioAttackResultDelta]) -> ScenarioRunPlan:
        """
        Synthesize only known completed legacy units without claiming pending totals.

        Returns:
            ScenarioRunPlan: An incomplete plan containing only known units.
        """
        seeds: dict[str, ScenarioRunPlanSeedGroup] = {}
        groups: dict[str, ScenarioRunPlanAtomicGroup] = {}
        for delta in deltas:
            mapped = ScenarioRunService._map_progress_delta(delta=delta, plan=None)
            seeds.setdefault(
                mapped.seed_group_id,
                ScenarioRunPlanSeedGroup(
                    id=mapped.seed_group_id,
                    objective_sha256=delta.objective_sha256 or to_sha256(delta.objective),
                    objective=delta.objective,
                ),
            )
            group = groups.setdefault(
                mapped.atomic_group_id,
                ScenarioRunPlanAtomicGroup(
                    id=mapped.atomic_group_id,
                    atomic_attack_name=mapped.atomic_attack_name,
                    display_group=mapped.atomic_attack_name,
                    technique_eval_hash=str(delta.attribution_data.get("parent_eval_hash") or ""),
                    seed_group_ids=[],
                ),
            )
            if mapped.seed_group_id not in group.seed_group_ids:
                group.seed_group_ids.append(mapped.seed_group_id)
        return ScenarioRunPlan(atomic_groups=list(groups.values()), seed_groups=list(seeds.values()))

    @staticmethod
    def _encode_progress_cursor(*, scenario_result_id: str, delta: ScenarioAttackResultDelta) -> str:
        payload = {
            "v": 1,
            "run": scenario_result_id,
            "timestamp": delta.timestamp.isoformat(),
            "attack_result_id": delta.attack_result_id,
        }
        return base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")

    @staticmethod
    def _decode_progress_cursor(
        *,
        since: str | None,
        scenario_result_id: str,
    ) -> ScenarioProgressKeysetCursor | None:
        if since is None:
            return None
        try:
            padded = since + "=" * (-len(since) % 4)
            payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        except Exception as exc:
            raise ValueError("Malformed scenario progress cursor.") from exc
        if not isinstance(payload, dict):
            raise ValueError("Malformed scenario progress cursor.")
        if payload.get("v") != 1 or payload.get("run") != scenario_result_id:
            raise ValueError("Cursor does not belong to this scenario run.")
        try:
            timestamp = datetime.fromisoformat(payload["timestamp"])
            attack_result_id = str(uuid.UUID(payload["attack_result_id"]))
        except Exception as exc:
            raise ValueError("Malformed scenario progress cursor.") from exc
        if timestamp.tzinfo is None:
            raise ValueError("Cursor timestamp must include a timezone.")
        return ScenarioProgressKeysetCursor(timestamp=timestamp, attack_result_id=attack_result_id)

    def get_run_results(self, *, scenario_result_id: str) -> ScenarioResult | None:
        """
        Get the ScenarioResult for a completed scenario run.

        Args:
            scenario_result_id: The scenario result ID.

        Returns:
            ScenarioResult if the run is completed and results exist, None if not found.

        Raises:
            ValueError: If the run is not in a completed state.
        """
        results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
        if not results:
            return None

        scenario_result = results[0]
        run_response = self._build_response_from_db(scenario_result=scenario_result)

        if run_response.status != ScenarioRunState.COMPLETED:
            raise ValueError(f"Results are only available for completed runs. Current status: '{run_response.status}'.")

        return scenario_result


_service_instance: ScenarioRunService | None = None


def get_scenario_run_service() -> ScenarioRunService:
    """
    Get the global scenario run service instance.

    On first call, reads ``max_concurrent_scenario_runs`` from ``app.state``
    (set by ``pyrit_backend`` CLI) if available, otherwise uses the default.

    Returns:
        The singleton ScenarioRunService instance.
    """
    global _service_instance
    if _service_instance is not None:
        return _service_instance

    max_runs = _DEFAULT_MAX_CONCURRENT_RUNS
    try:
        from pyrit.backend.main import app

        max_runs = getattr(app.state, "max_concurrent_scenario_runs", _DEFAULT_MAX_CONCURRENT_RUNS)
    except Exception:
        pass

    _service_instance = ScenarioRunService(max_concurrent_runs=max_runs)
    return _service_instance
