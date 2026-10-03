# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Scenario run service for executing scenarios as background tasks.

Manages the lifecycle of scenario runs: starting, tracking status,
retrieving results, and cancellation.
"""

import asyncio
import base64
import contextlib
import functools
import hashlib
import io
import json
import logging
import uuid
from collections import OrderedDict, deque
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from zipfile import BadZipFile

from pydantic import TypeAdapter, ValidationError

from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.models.common import PaginationInfo, filter_sensitive_fields
from pyrit.backend.models.scenarios import ScenarioRunListResponse
from pyrit.backend.services.original_run_admission import (
    APPROVED_ORIGINAL_SCENARIO,
    OriginalAdmissionError,
    OriginalCleanupReceipt,
    OriginalRunBinding,
    OriginalRunGrant,
    OriginalWorkerJob,
    get_original_run_gateway,
)
from pyrit.backend.services.pagination import (
    decode_keyset_cursor,
    encode_keyset_cursor,
    fingerprint_filters,
    normalize_label_filters,
)
from pyrit.backend.services.scenario_configuration_resolver import ScenarioConfigurationResolver
from pyrit.backend.services.scenario_progress_read_model import ResultUnitIdentity, ScenarioProgressReadModel
from pyrit.common.utils import to_sha256
from pyrit.memory import AttackResultKeysetCursor, CentralMemory, MemoryInterface, SQLiteMemory
from pyrit.memory.memory_interface import (
    ScenarioHistoryAggregate,
    ScenarioHistoryKeysetCursor,
    ScenarioHistoryRunRecord,
)
from pyrit.models import (
    SCENARIO_RUN_PLAN_METADATA_KEY,
    SCENARIO_RUN_STARTED_AT_METADATA_KEY,
    AttackOutcome,
    AttackResult,
    ComponentIdentifier,
    EvalCaseRef,
    EvalPackageRef,
    EvalSourceKind,
    MessagePiece,
    ScenarioAtomicGroupProgress,
    ScenarioAttackResultDelta,
    ScenarioDisplayGroupProgress,
    ScenarioExecutionOwner,
    ScenarioIdentifier,
    ScenarioProgressCounts,
    ScenarioProgressHeader,
    ScenarioProgressSummary,
    ScenarioQueueEntry,
    ScenarioQueueSnapshot,
    ScenarioResult,
    ScenarioRunPlan,
    ScenarioRunPlanAtomicGroup,
    ScenarioRunPlanSeedGroup,
    ScenarioRunProgress,
    ScenarioRunState,
    ScenarioSeedGroupProgress,
    ScenarioTechniqueProgress,
    ScoreStatus,
    TargetIdentifier,
    config_hash,
)
from pyrit.models.catalog.scenario import (
    AttackErrorSummary,
    AttackRetrySummary,
    OriginalInspectImportSummary,
    OriginalRunAdmission,
    OriginalRunEvidenceLink,
    OriginalRunReason,
    OriginalRunStatus,
    OriginalSourceResult,
    RunScenarioRequest,
    ScenarioOverloadSummary,
    ScenarioRunListItem,
    ScenarioRunSummary,
    ScenarioTargetSummary,
    ScenarioTechniqueSummary,
)
from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot
from pyrit.registry import InitializerRegistry, ScenarioRegistry
from pyrit.scenario import Scenario

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONCURRENT_RUNS = 1
_MAX_OVERLOAD_EVENTS = 500
_MAX_OVERLOAD_ROLES = 16
_MAX_TERMINAL_ERRORS = 100
_SCHEDULER_RETRY_INITIAL_SECONDS = 0.05
_SCHEDULER_RETRY_MAX_SECONDS = 1.0
_SCHEDULER_METADATA_KEY = "scheduler_managed_by"
_SCHEDULER_METADATA_VALUE = "ScenarioRunService.process_local_fifo"
_INTERRUPTED_ERROR_TYPE = "ScenarioInterruptedError"
_RESTART_INTERRUPTION_REASON = (
    "The backend process restarted before this scenario run completed; "
    "its executable scenario objects could not be recovered safely."
)
_SHUTDOWN_INTERRUPTION_REASON = "The backend process shut down before this scenario run completed."
_USER_CANCELLATION_REASON = "Run was cancelled by user"
_ORIGINAL_INSPECT_SCENARIO_NAME = "InspectOriginalInertScenario"
_ORIGINAL_INSPECT_REGISTRY_NAME = "benchmark.inspect_original_inert"
_INVALID_ORIGINAL_INSPECT_PROJECTION = (
    "Original Inspect import references missing or mismatched persisted Score/AttackResult evidence."
)
_SAFE_ORIGINAL_INSPECT_FAILURE_PREFIXES = (
    "Original Inspect archive is not a readable `.eval` ZIP.",
    "Original Inspect archive is empty or exceeds its bounded byte quota.",
    "Original Inspect log Samples/epochs differ from the approved case inventory.",
    "Original Inspect case Score/AttackResult projection is partial or missing.",
    "Original Inspect case Score/AttackResult differs from its typed source.",
    "Offline Inspect score projection differs from the approved live archive or case.",
    "Offline Inspect Score/AttackResult has unapproved source or success attribution.",
)

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
_STARTED_AT_ADAPTER = TypeAdapter(datetime)


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
    original_grant: OriginalRunGrant | None = None
    original_job: OriginalWorkerJob | None = None


@dataclass(frozen=True, slots=True)
class _ActiveRunSnapshot:
    """Event-loop-owned state copied before database work moves to a worker thread."""

    error: str | None = None
    active_group_ids: tuple[str, ...] = ()
    queue_position: int | None = None
    active_scenario_result_id: str | None = None


class ScenarioRunService:
    """
    Service for managing scenario run lifecycle.

    Uses CentralMemory (database) as the source of truth for run state.
    Keeps executable objects in a process-local single-active FIFO scheduler.
    FIFO ordering therefore spans only runs submitted to the same backend
    process. Deploy one backend replica to preserve a global admission order;
    multiple replicas require a shared database-backed scheduler or lease.
    """

    #: Seconds to let initialization's own background tasks (for example HTTP client teardown
    #: scheduled from ``__del__``) finish before the initialization loop is torn down. This is
    #: headroom for incidental teardown, not a waiter for real long-running work.
    _INITIALIZATION_DRAIN_TIMEOUT = 5.0

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
        self._configuration_resolver = ScenarioConfigurationResolver()
        self._progress_read_model = ScenarioProgressReadModel(memory=self._memory)
        self._technique_metadata_cache: dict[str, dict[str, ScenarioTechniqueSummary]] = {}
        self._technique_metadata_lock = Lock()

        # Initialization writes to CentralMemory, and the in-memory SQLite backend shares one
        # DBAPI connection across every thread (StaticPool, sqlite_memory.py). Two preparations
        # running at once would use that connection concurrently and lose or corrupt writes, so
        # they are serialized onto a single worker. The event loop is still free while they run,
        # which is the point of the offload.
        self._prepare_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pyrit-scenario-prep")
        self._terminal_errors: OrderedDict[str, str] = OrderedDict()
        self._active_scenario_result_id: str | None = None
        self._queued_runs: deque[_ActiveTask] = deque()
        self._handoff_retry_tasks: set[asyncio.Task[None]] = set()
        self._original_cleanup_tasks: set[asyncio.Task[None]] = set()
        self._scheduler_lock = asyncio.Lock()
        self._launch_lock = asyncio.Lock()
        self._queue_revision = 0
        self._stopping = False

    async def start_run_async(
        self, *, request: RunScenarioRequest, operator: AuthenticatedUser | None = None
    ) -> ScenarioRunSummary:
        """
        Initialize and schedule a scenario run.

        Performs all validation and initialization eagerly (initializers, target
        resolution, technique validation, scenario.initialize_async) so errors are
        returned immediately. On success, starts execution when idle or appends
        the initialized run to the FIFO waiting queue.

        Args:
            request: The run request with scenario name, target, and options.
            operator: Authenticated operator required for a protected original runner.

        Returns:
            ScenarioRunSummary with a stable ID and current active or queued state.

        Raises:
            ValueError: If scenario, target, initializer, or technique cannot be found.
        """
        async with self._launch_lock:
            if self._stopping:
                raise RuntimeError("Scenario run scheduling is stopping.")
            if request.scenario_name == APPROVED_ORIGINAL_SCENARIO:
                return await self._start_approved_original_run_async(request=request, operator=operator)
            if request.original_admission_ref is not None:
                raise OriginalAdmissionError(reason=OriginalRunReason.PROFILE_NOT_ADMITTED)
            resumed_from_cancelled = self._is_run_cancelled(scenario_result_id=request.scenario_result_id)
            prepare_task = asyncio.get_running_loop().run_in_executor(
                self._prepare_executor,
                functools.partial(self._prepare_run_blocking, request=request),
            )
            try:
                scenario = await asyncio.shield(prepare_task)
            except asyncio.CancelledError:
                if prepare_task.done():
                    try:
                        self._release_abandoned_prepare(prepare_task)
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up after a cancelled scenario preparation: {cleanup_error}")
                else:
                    prepare_task.add_done_callback(self._release_abandoned_prepare)
                raise

            scenario_result_id = scenario._scenario_result_id
            if scenario_result_id is None:
                raise ValueError("Scenario did not produce a scenario_result_id during initialization.")
            persisted = await asyncio.to_thread(
                self._memory.get_scenario_results,
                scenario_result_ids=[scenario_result_id],
            )
            if not persisted:
                raise RuntimeError(f"Scenario run {scenario_result_id} was not persisted during initialization.")
            if not resumed_from_cancelled and persisted[0].scenario_run_state == ScenarioRunState.CANCELLED:
                response = self._build_response(
                    scenario_result_id=scenario_result_id,
                    active_error=None,
                    queue_position=None,
                    active_scenario_result_id=self._active_scenario_result_id,
                    operator=operator,
                )
                if response is None:
                    raise RuntimeError(
                        f"Scenario run {scenario_result_id} was not found in the database after initialization."
                    )
                return response
            if (
                self._build_response(
                    scenario_result_id=scenario_result_id,
                    active_error=None,
                    queue_position=None,
                    active_scenario_result_id=self._active_scenario_result_id,
                    operator=operator,
                )
                is None
            ):
                raise RuntimeError(
                    f"Scenario run {scenario_result_id} was not found in the database after initialization."
                )
            scheduled = _ActiveTask(
                scenario_result_id=scenario_result_id,
                scenario=scenario,
                scenario_name=persisted[0].scenario_name,
                scenario_registry_name=request.scenario_name,
                created_at=persisted[0].creation_time,
                enqueued_at=datetime.now(UTC),
            )
            await self._enqueue_run_async(scheduled=scheduled)

        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id, operator=operator)
        response = await asyncio.to_thread(
            self.get_run_from_storage,
            scenario_result_id=scenario_result_id,
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
            operator=operator,
        )
        if response is None:
            raise RuntimeError(f"Scenario run {scenario_result_id} was not found in the database after initialization.")
        return response

    async def _start_approved_original_run_async(
        self, *, request: RunScenarioRequest, operator: AuthenticatedUser | None
    ) -> ScenarioRunSummary:
        """
        Reserve an opaque job and schedule it without resolving or importing a private Scenario.

        Returns:
            ScenarioRunSummary: The projected server-owned job, not a worker Scenario result.
        """
        if request.model_extra or request.model_fields_set != {"scenario_name", "original_admission_ref"}:
            raise OriginalAdmissionError(reason=OriginalRunReason.PROFILE_NOT_ADMITTED)
        if request.original_admission_ref is None:
            raise OriginalAdmissionError(reason=OriginalRunReason.ADMISSION_EXPIRED)
        gateway = get_original_run_gateway()
        if gateway is None:
            raise OriginalAdmissionError(reason=OriginalRunReason.RUNNER_NOT_CONFIGURED)
        grant = await gateway.claim_async(operator=operator, admission_ref=request.original_admission_ref)
        job: OriginalWorkerJob | None = None
        try:
            binding = OriginalRunBinding(profile_ref=grant.profile_ref, operator_oid=grant.operator_oid)
            job = OriginalWorkerJob(app_run_id=uuid.uuid4(), job_ref=uuid.uuid4())
            envelope = ScenarioResult(
                id=job.app_run_id,
                scenario_identifier=ScenarioIdentifier(
                    class_name="ServerApprovedOriginalScenario",
                    class_module="pyrit.backend.services.original_run_admission",
                    params={"execution_owner": ScenarioExecutionOwner.APPROVED_ORIGINAL.value},
                    version=1,
                    techniques=[],
                    datasets=[],
                ),
                scenario_description="Server-approved original Task",
                attack_results={},
                metadata={
                    _SCHEDULER_METADATA_KEY: _SCHEDULER_METADATA_VALUE,
                    OriginalRunBinding.METADATA_KEY: binding.model_dump(mode="json"),
                    OriginalWorkerJob.METADATA_KEY: job.model_dump(mode="json"),
                },
            )
        except Exception as error:
            logger.error("Approved original job envelope is invalid (%s).", type(error).__name__)
            await self._release_original_grant_async(grant=grant, job=job, cancelled=True)
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error

        async def prepare_async() -> None:
            await asyncio.to_thread(self._memory.add_scenario_results_to_memory, scenario_results=[envelope])
            await gateway.runner.prepare_worker_async(grant=grant, job=job, binding=binding)

        preparation = asyncio.create_task(prepare_async())
        try:
            await asyncio.shield(preparation)
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                self._release_abandoned_original_job_async(grant=grant, job=job, preparation=preparation)
            )
            self._original_cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._original_cleanup_tasks.discard)
            raise
        except Exception as error:
            logger.error("Original worker preparation failed (%s).", type(error).__name__)
            cleanup = await self._release_original_grant_async(grant=grant, job=job, cancelled=True)
            await self._terminalize_original_job_async(
                job=job, binding=binding, cleanup=cleanup, state=ScenarioRunState.FAILED
            )
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error

        scheduled = _ActiveTask(
            scenario_result_id=str(job.app_run_id),
            scenario_name="ServerApprovedOriginalScenario",
            scenario_registry_name=APPROVED_ORIGINAL_SCENARIO,
            created_at=envelope.creation_time,
            enqueued_at=datetime.now(UTC),
            original_grant=grant,
            original_job=job,
        )
        enqueue_task = asyncio.create_task(self._enqueue_run_async(scheduled=scheduled))
        try:
            await asyncio.shield(enqueue_task)
        except asyncio.CancelledError:
            cleanup_task = asyncio.create_task(
                self._cancel_abandoned_original_enqueue_async(
                    grant=grant, job=job, binding=binding, enqueue_task=enqueue_task, operator=operator
                )
            )
            self._original_cleanup_tasks.add(cleanup_task)
            cleanup_task.add_done_callback(self._original_cleanup_tasks.discard)
            raise
        except Exception as error:
            cleanup = await self._release_original_grant_async(grant=grant, job=job, cancelled=True)
            await self._terminalize_original_job_async(
                job=job, binding=binding, cleanup=cleanup, state=ScenarioRunState.FAILED
            )
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error
        snapshot = self.snapshot_active_run(scenario_result_id=str(job.app_run_id), operator=operator)
        response = await asyncio.to_thread(
            self.get_run_from_storage,
            scenario_result_id=str(job.app_run_id),
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
            operator=operator,
        )
        if response is None:
            raise RuntimeError("Persisted approved original job could not be read back.")
        return response

    async def _cancel_abandoned_original_enqueue_async(
        self,
        *,
        grant: OriginalRunGrant,
        job: OriginalWorkerJob,
        binding: OriginalRunBinding,
        enqueue_task: asyncio.Task[None],
        operator: AuthenticatedUser | None,
    ) -> None:
        """Finish an in-flight scheduler admission before aborting the exact job."""
        try:
            await enqueue_task
        except Exception as error:
            logger.error("Original worker enqueue failed after cancellation (%s).", type(error).__name__)
            cleanup = await self._release_original_grant_async(grant=grant, job=job, cancelled=True)
            await self._terminalize_original_job_async(
                job=job,
                binding=binding,
                cleanup=cleanup,
                state=ScenarioRunState.CANCELLED if cleanup.proved else ScenarioRunState.FAILED,
            )
        else:
            try:
                await self.cancel_run_async(scenario_result_id=str(job.app_run_id), operator=operator)
            except ValueError:
                header = await asyncio.to_thread(
                    self._memory.get_scenario_result_header, scenario_result_id=str(job.app_run_id)
                )
                if header is None or not self._is_terminal_state(header.scenario_run_state):
                    raise

    def _is_run_cancelled(self, *, scenario_result_id: str | None) -> bool:
        """
        Report whether a stored run is already in the CANCELLED state.

        Reads the header only. A resumed run can have thousands of linked attack results and
        this runs on the event loop, which the rest of this path works to keep free.

        Args:
            scenario_result_id: The run being resumed, or None for a fresh run.

        Returns:
            bool: True when a stored run with this id is CANCELLED.
        """
        if not scenario_result_id:
            return False
        stored = self._memory.get_scenario_result_header(scenario_result_id=scenario_result_id)
        return stored is not None and stored.scenario_run_state == ScenarioRunState.CANCELLED

    def _release_abandoned_prepare(self, prepare_task: "asyncio.Future[Scenario]") -> None:
        """
        Clean up after an abandoned preparation thread has finished.

        A preparation that succeeds after its caller is cancelled leaves behind a
        scenario result nobody will run. Mark it cancelled rather than leaving it
        in ``CREATED``.

        Args:
            prepare_task: The future wrapping the abandoned ``_prepare_run_blocking`` call.
        """
        if prepare_task.cancelled():
            return
        error = prepare_task.exception()
        if error is not None:
            logger.warning(f"Abandoned scenario preparation failed after the request was cancelled: {error}")
            return

        # Initialization already stored a CREATED scenario result, and nothing is going to run
        # it now, so terminalize it rather than leaving a run that never starts. A run that
        # already reached a terminal state keeps it, so a real failure is not relabelled.
        scenario_result_id = prepare_task.result()._scenario_result_id
        if scenario_result_id:
            try:
                self._memory.try_update_scenario_run_state(
                    scenario_result_id=scenario_result_id,
                    expected_states={ScenarioRunState.CREATED, ScenarioRunState.IN_PROGRESS},
                    scenario_run_state=ScenarioRunState.CANCELLED,
                    error_message="The start request was cancelled while the scenario was being initialized.",
                )
            except Exception as update_error:
                logger.warning(
                    f"Could not mark abandoned scenario run {scenario_result_id} as cancelled: {update_error}"
                )
        logger.warning("Abandoned scenario preparation completed after the request was cancelled.")

    def _prepare_run_blocking(self, *, request: RunScenarioRequest) -> Scenario:
        """
        Run the eager initialization for a scenario run on the calling thread.

        Exists so ``start_run_async`` can offload initialization onto a worker thread.
        The scenario is executed later on the caller's event loop, so initialization must not
        leave anything bound to the throwaway loop used here. Clients that schedule their own
        teardown are given a moment to finish; anything still running after that would be
        cancelled when the loop closes, so the start fails rather than handing back a scenario
        that holds dead async resources.

        Args:
            request: The run request with scenario name, target, and options.

        Returns:
            Scenario: The initialized scenario.

        Raises:
            RuntimeError: If tasks are still running on the initialization loop after the drain.
        """

        async def prepare_async() -> Scenario:
            scenario = await self._prepare_run_async(request=request)
            try:
                await self._drain_initialization_tasks_async()
            except RuntimeError as drain_error:
                # Initialization already stored a CREATED row and this start is over, so
                # terminalize it here rather than leaving a run that never begins. A cancel
                # can land while the drain is running, so keep whatever terminal state won.
                scenario_result_id = scenario._scenario_result_id
                if scenario_result_id:
                    try:
                        self._memory.try_update_scenario_run_state(
                            scenario_result_id=scenario_result_id,
                            expected_states={ScenarioRunState.CREATED, ScenarioRunState.IN_PROGRESS},
                            scenario_run_state=ScenarioRunState.FAILED,
                            error_message=str(drain_error),
                            error_type=type(drain_error).__name__,
                        )
                    except Exception as update_error:
                        logger.warning(f"Could not mark scenario run {scenario_result_id} as failed: {update_error}")
                raise
            return scenario

        return asyncio.run(prepare_async())

    async def _drain_initialization_tasks_async(self) -> None:
        """
        Let initialization's background tasks finish before the initialization loop closes.

        Initialization builds throwaway async clients, and some of them schedule their own
        teardown from ``__del__``, so a task can appear purely because a garbage collection
        landed late. Waiting for those is the difference between a scenario that starts and
        one that fails at random. A draining task can also start another one, so the set is
        rebuilt after every wait and the whole drain shares a single deadline.

        Raises:
            RuntimeError: If any task is still running after the drain timeout.
        """
        loop = asyncio.get_running_loop()
        current_task = asyncio.current_task()
        deadline = loop.time() + self._INITIALIZATION_DRAIN_TIMEOUT

        while True:
            pending = [task for task in asyncio.all_tasks() if task is not current_task]
            if not pending:
                return

            remaining = deadline - loop.time()
            if remaining <= 0:
                raise RuntimeError(
                    "Scenario initialization left background tasks on the initialization loop, which is "
                    "about to close. They would be cancelled and the scenario would hold dead async "
                    f"resources: {', '.join(sorted(task.get_name() for task in pending))}"
                )

            done, _ = await asyncio.wait(pending, timeout=remaining)
            for task in done:
                # Retrieve outcomes so a failed teardown task does not log "never retrieved" noise.
                if not task.cancelled() and task.exception() is not None:
                    logger.debug(f"A scenario initialization task failed during teardown: {task.exception()}")

    async def _prepare_run_async(self, *, request: RunScenarioRequest) -> Scenario:
        """
        Resolve and initialize the scenario for a run request.

        Args:
            request: The run request with scenario name, target, and options.

        Returns:
            Scenario: The initialized scenario.

        Raises:
            ValueError: If scenario, target, initializer, or technique cannot be found.
        """
        scenario_class = self._configuration_resolver.resolve_scenario_class(scenario_name=request.scenario_name)
        if isinstance(scenario_class, type) and getattr(scenario_class, "SERVER_ADMISSION_REQUIRED", False):
            raise OriginalAdmissionError(reason=OriginalRunReason.PROFILE_NOT_ADMITTED)
        scenario_class.validate_run_request(request=request)
        task_owned = getattr(scenario_class, "TASK_OWNED", False) is True
        if task_owned and request.target_name is not None:
            raise ValueError("Task-owned scenarios cannot use a global objective target.")
        await self._run_initializers_async(request=request)
        if task_owned:
            objective_target = None
        else:
            if not request.target_name:
                raise ValueError("target_name is required for a target-owned Scenario.")
            objective_target = self._configuration_resolver.resolve_target(target_name=request.target_name)
        init_kwargs = self._configuration_resolver.resolve_configuration(
            scenario_name=request.scenario_name,
            scenario_class=scenario_class,
            objective_target=objective_target,
            techniques=request.techniques,
            dataset_names=request.dataset_names,
            max_dataset_size=request.max_dataset_size,
            dataset_filters=request.dataset_filters,
            include_baseline=request.include_baseline,
            max_concurrency=(
                request.max_concurrency if request.max_concurrency is not None else (None if task_owned else 10)
            ),
            max_retries=request.max_retries,
            memory_labels=request.labels,
        )
        return await self._initialize_scenario_async(request=request, init_kwargs=init_kwargs)

    async def _release_original_grant_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob | None, cancelled: bool
    ) -> OriginalCleanupReceipt:
        """
        Release a reservation and treat absent owned cleanup proof as explicitly uncontained.

        Returns:
            OriginalCleanupReceipt: Exact proof or explicit uncontained cleanup.
        """
        gateway = get_original_run_gateway()
        if gateway is None:
            logger.error("Original runner disappeared before releasing its reservation.")
            return OriginalCleanupReceipt(state="uncontained")
        try:
            return await gateway.runner.release_async(grant=grant, job=job, cancelled=cancelled)
        except Exception as error:
            logger.error("Original run cleanup could not be verified (%s).", type(error).__name__)
            return OriginalCleanupReceipt(state="uncontained")

    async def _release_abandoned_original_job_async(
        self, *, grant: OriginalRunGrant, job: OriginalWorkerJob, preparation: asyncio.Task[None]
    ) -> None:
        """Wait for an abandoned worker preparation, then close only its owned reservation."""
        try:
            await preparation
        except Exception as error:
            logger.error("Abandoned original worker preparation failed (%s).", type(error).__name__)
        cleanup = await self._release_original_grant_async(grant=grant, job=job, cancelled=True)
        await self._terminalize_original_job_async(
            job=job,
            binding=OriginalRunBinding(profile_ref=grant.profile_ref, operator_oid=grant.operator_oid),
            cleanup=cleanup,
            state=ScenarioRunState.CANCELLED if cleanup.proved else ScenarioRunState.FAILED,
        )

    async def _terminalize_original_job_async(
        self,
        *,
        job: OriginalWorkerJob,
        binding: OriginalRunBinding,
        cleanup: OriginalCleanupReceipt,
        state: ScenarioRunState,
    ) -> None:
        """Persist an ungraded, physically proved or uncertain job without private evidence."""
        result = OriginalSourceResult(
            profile_ref=binding.profile_ref,
            status=OriginalRunStatus.FAILED_UNGRADED if cleanup.proved else OriginalRunStatus.CLEANUP_UNCERTAIN,
            source_state="cancelled" if state is ScenarioRunState.CANCELLED else "error",
            source_coverage_complete=False,
            cleanup_state=cleanup.state,
            reason=OriginalRunReason.SOURCE_UNVERIFIED if cleanup.proved else OriginalRunReason.CLEANUP_PENDING,
        )
        header = await asyncio.to_thread(
            self._memory.get_scenario_result_header, scenario_result_id=str(job.app_run_id)
        )
        if header is None:
            logger.error("Approved original job was not persisted before its reservation ended.")
            return
        await asyncio.to_thread(
            self._memory.update_scenario_metadata_fields,
            scenario_result_id=str(job.app_run_id),
            fields={
                OriginalCleanupReceipt.METADATA_KEY: cleanup.model_dump(mode="json"),
                OriginalSourceResult.METADATA_KEY: result.model_dump(mode="json"),
            },
        )
        await asyncio.to_thread(
            self._memory.try_update_scenario_run_state,
            scenario_result_id=str(job.app_run_id),
            expected_states={ScenarioRunState.CREATED, ScenarioRunState.QUEUED, ScenarioRunState.IN_PROGRESS},
            scenario_run_state=state,
            error_message=(
                None
                if state is ScenarioRunState.CANCELLED
                else OriginalRunReason.SOURCE_UNVERIFIED.value
                if cleanup.proved
                else OriginalRunReason.CLEANUP_PENDING.value
            ),
            error_type=None if state is ScenarioRunState.CANCELLED else "OriginalAdmissionError",
        )

    def get_run(
        self, *, scenario_result_id: str, operator: AuthenticatedUser | None = None
    ) -> ScenarioRunSummary | None:
        """
        Get the current status of a scenario run by querying the database.

        Args:
            scenario_result_id: The scenario result ID.
            operator: Authenticated operator when reading a protected original result.

        Returns:
            ScenarioRunSummary if found, None otherwise.
        """
        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id, operator=operator)
        return self.get_run_from_storage(
            scenario_result_id=scenario_result_id,
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
            operator=operator,
        )

    def get_run_from_storage(
        self,
        *,
        scenario_result_id: str,
        active_error: str | None,
        queue_position: int | None = None,
        active_scenario_result_id: str | None = None,
        operator: AuthenticatedUser | None = None,
    ) -> ScenarioRunSummary | None:
        """
        Build a run summary using database state plus an event-loop snapshot.

        Args:
            scenario_result_id: The scenario result ID.
            active_error: Error copied from the active asyncio task, if any.
            queue_position: Current 1-based waiting position, if queued.
            active_scenario_result_id: Currently executing scenario result ID.
            operator: Authenticated operator when reading a protected original result.

        Returns:
            ScenarioRunSummary | None: The run summary when found.
        """
        return self._build_response(
            scenario_result_id=scenario_result_id,
            active_error=active_error,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
            operator=operator,
        )

    def list_runs(
        self,
        *,
        scenario_names: Sequence[str] | None = None,
        statuses: Sequence[ScenarioRunState | str] | None = None,
        labels: Mapping[str, str | Sequence[str]] | None = None,
        limit: int = 100,
        cursor: str | None = None,
        operator: AuthenticatedUser | None = None,
    ) -> ScenarioRunListResponse:
        """
        List scenario runs by querying the database (most recent first).

        Args:
            scenario_names: Registered or persisted scenario names to match.
            statuses: Run states to match.
            labels: Labels with OR-within-key and AND-across-key semantics.
            limit: Maximum number of runs to return.
            cursor: Opaque cursor from the previous page.
            operator: Authenticated operator needed to read protected original history.

        Returns:
            ScenarioRunListResponse with runs.
        """
        normalized_names = sorted({name.strip() for name in scenario_names or [] if name.strip()})
        query_names = list(normalized_names)
        original_aliases = {APPROVED_ORIGINAL_SCENARIO, "ServerApprovedOriginalScenario"}
        if original_aliases.intersection(normalized_names):
            gateway = get_original_run_gateway()
            if gateway is None or not gateway.authorized(operator=operator):
                raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
            query_names = [
                "ServerApprovedOriginalScenario" if name in original_aliases else name for name in normalized_names
            ]
        normalized_statuses = sorted(
            {
                status.value if isinstance(status, ScenarioRunState) else str(status).strip().upper()
                for status in statuses or []
                if str(status).strip()
            }
        )
        normalized_labels = normalize_label_filters(labels=labels)
        fingerprint = fingerprint_filters(
            filters={
                "scenario_names": normalized_names,
                "statuses": normalized_statuses,
                "labels": normalized_labels,
            }
        )
        decoded_cursor = decode_keyset_cursor(cursor=cursor, fingerprint=fingerprint)
        after = (
            ScenarioHistoryKeysetCursor(
                timestamp=decoded_cursor.timestamp,
                scenario_result_id=decoded_cursor.identifier,
            )
            if decoded_cursor is not None
            else None
        )
        records, aggregates, has_more = self._memory.get_scenario_run_history_page(
            scenario_names=query_names,
            statuses=normalized_statuses,
            labels=normalized_labels,
            cursor=after,
            limit=limit,
        )
        headers = self._memory.get_scenario_result_headers(
            scenario_result_ids=[record.scenario_result_id for record in records]
        )
        for header in headers.values():
            self._reject_unbound_original_run(scenario_result=header)
            if OriginalRunBinding.METADATA_KEY in (getattr(header, "metadata", None) or {}):
                self._original_run_binding(scenario_result=header, operator=operator)
        plans = {record.scenario_result_id: self._parse_history_plan(record=record) for record in records}
        # Memory resolves units against every persisted plan. Runs whose plan this service
        # rejects must fall back to legacy unit identity, which needs a plan-free aggregate.
        unusable_plan_ids = [
            record.scenario_result_id
            for record in records
            if record.plan_atomic_groups is not None and plans[record.scenario_result_id] is None
        ]
        if unusable_plan_ids:
            aggregates = {
                **aggregates,
                **self._memory.get_scenario_history_aggregates(scenario_result_ids=unusable_plan_ids),
            }
        original_ids = [
            record.scenario_result_id for record in records if record.scenario_name == _ORIGINAL_INSPECT_SCENARIO_NAME
        ]
        original_results: dict[str, ScenarioResult] = (
            {str(result.id): result for result in self._memory.get_scenario_results(scenario_result_ids=original_ids)}
            if original_ids
            else {}
        )
        items: list[ScenarioRunListItem] = []
        for record in records:
            header = headers.get(record.scenario_result_id)
            if header is not None and OriginalRunBinding.METADATA_KEY in (getattr(header, "metadata", None) or {}):
                safe_summary = self._original_run_summary(
                    scenario_result=header,
                    operator=operator,
                    queue_position=None,
                    active_scenario_result_id=None,
                )
                items.append(ScenarioRunListItem.model_validate(safe_summary.model_dump(mode="json")))
            else:
                items.append(
                    self._build_history_summary(
                        record=record,
                        atomic_groups=plans[record.scenario_result_id],
                        aggregate=aggregates.get(record.scenario_result_id)
                        or ScenarioHistoryAggregate.empty(scenario_result_id=record.scenario_result_id),
                        original_result=original_results.get(record.scenario_result_id),
                    )
                )
        next_cursor = (
            encode_keyset_cursor(
                timestamp=records[-1].created_at,
                identifier=records[-1].scenario_result_id,
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

    async def cancel_run_async(
        self, *, scenario_result_id: str, operator: AuthenticatedUser | None = None
    ) -> ScenarioRunSummary | None:
        """
        Cancel a running scenario.

        Args:
            scenario_result_id: The scenario result ID.
            operator: Authenticated operator needed before cancelling a protected original run.

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

        self._reject_unbound_original_run(scenario_result=results[0])
        if OriginalRunBinding.METADATA_KEY in (getattr(results[0], "metadata", None) or {}):
            await asyncio.to_thread(self._original_run_binding, scenario_result=results[0], operator=operator)
        db_status = results[0].scenario_run_state
        if self._is_terminal_state(db_status):
            raise ValueError(f"Cannot cancel run in '{db_status}' state.")

        task: asyncio.Task[None] | None = None
        queued_original: _ActiveTask | None = None
        async with self._scheduler_lock:
            queued = next(
                (run for run in self._queued_runs if run.scenario_result_id == scenario_result_id),
                None,
            )
            if queued is not None:
                queued_original = queued if queued.original_grant is not None else None
                if queued_original is None:
                    await asyncio.to_thread(
                        self._memory.try_update_scenario_run_state,
                        scenario_result_id=scenario_result_id,
                        expected_states={ScenarioRunState.CREATED, ScenarioRunState.QUEUED},
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
                if OriginalRunBinding.METADATA_KEY in (getattr(results[0], "metadata", None) or {}):
                    raise OriginalAdmissionError(reason=OriginalRunReason.CLEANUP_PENDING)
                latest = await asyncio.to_thread(
                    self._memory.get_scenario_results,
                    scenario_result_ids=[scenario_result_id],
                )
                if latest and self._is_terminal_state(latest[0].scenario_run_state):
                    raise ValueError(f"Cannot cancel run in '{latest[0].scenario_run_state}' state.")
                await asyncio.to_thread(
                    self._memory.try_update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    expected_states={
                        ScenarioRunState.CREATED,
                        ScenarioRunState.QUEUED,
                        ScenarioRunState.IN_PROGRESS,
                    },
                    scenario_run_state=ScenarioRunState.CANCELLED,
                    error_message=_USER_CANCELLATION_REASON,
                    error_type="CancelledError",
                )

        if queued_original is not None:
            assert queued_original.original_grant is not None and queued_original.original_job is not None
            cleanup_task = asyncio.create_task(
                self._release_original_grant_async(
                    grant=queued_original.original_grant, job=queued_original.original_job, cancelled=True
                )
            )
            cancelled_during_cleanup = False
            try:
                cleanup = await asyncio.shield(cleanup_task)
            except asyncio.CancelledError:
                cleanup = await cleanup_task
                cancelled_during_cleanup = True
            await self._terminalize_original_job_async(
                job=queued_original.original_job,
                binding=OriginalRunBinding(
                    profile_ref=queued_original.original_grant.profile_ref,
                    operator_oid=queued_original.original_grant.operator_oid,
                ),
                cleanup=cleanup,
                state=ScenarioRunState.CANCELLED if cleanup.proved else ScenarioRunState.FAILED,
            )
            if cancelled_during_cleanup:
                raise asyncio.CancelledError

        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        snapshot = self.snapshot_active_run(scenario_result_id=scenario_result_id, operator=operator)
        result = await asyncio.to_thread(
            self.get_run_from_storage,
            scenario_result_id=scenario_result_id,
            active_error=snapshot.error,
            queue_position=snapshot.queue_position,
            active_scenario_result_id=snapshot.active_scenario_result_id,
            operator=operator,
        )
        if (
            result is not None
            and result.status != ScenarioRunState.CANCELLED
            and not (result.original_run_admission is not None and result.status is ScenarioRunState.FAILED)
        ):
            raise ValueError(f"Cannot cancel run in '{result.status}' state.")
        return result

    def get_queue_snapshot(self, *, operator: AuthenticatedUser | None = None) -> ScenarioQueueSnapshot:
        """
        Return the current in-process FIFO scheduler state.

        Returns:
            ScenarioQueueSnapshot: Active run and ordered waiting runs.
        """
        snapshot_at = datetime.now(UTC)
        gateway = get_original_run_gateway()

        def visible(run: _ActiveTask) -> bool:
            """
            Hide other operators' private Scenario reservations from the queue.

            Returns:
                bool: Whether this operator may see the queued or active entry.
            """
            grant = run.original_grant
            return grant is None or (
                gateway is not None
                and gateway.authorized(operator=operator)
                and operator is not None
                and operator.oid == grant.operator_oid
            )

        active = None
        if self._active_scenario_result_id is not None:
            active_run = self._active_tasks.get(self._active_scenario_result_id)
            if active_run is not None and visible(active_run):
                active = self._build_queue_entry(run=active_run, state=ScenarioRunState.IN_PROGRESS)
        queued = [
            self._build_queue_entry(run=run, state=ScenarioRunState.QUEUED, position=position)
            for position, run in enumerate(self._queued_runs, start=1)
            if visible(run)
        ]
        return ScenarioQueueSnapshot(
            revision=self._queue_revision,
            snapshot_at=snapshot_at,
            active=active,
            queued=queued,
        )

    async def reconcile_interrupted_runs_async(self) -> int:
        """
        Mark scheduler-managed local rows failed when executable objects were lost.

        Shared and unknown memory backends are intentionally non-destructive because
        another process may still own their runs. File-backed SQLite assumes one
        scheduler process has exclusive ownership of that database file.

        Returns:
            int: Number of reconciled rows.
        """
        if not isinstance(self._memory, SQLiteMemory):
            logger.info(
                "Skipping interrupted Scenario run reconciliation for shared or unsupported %s memory.",
                type(self._memory).__name__,
            )
            return 0

        states = (ScenarioRunState.CREATED, ScenarioRunState.QUEUED, ScenarioRunState.IN_PROGRESS)
        after_id = None
        reconciled = 0
        while True:
            interrupted, has_more = await asyncio.to_thread(
                self._memory.get_scenario_run_state_page,
                states=states,
                after_id=after_id,
                limit=500,
            )
            for result in interrupted:
                header = await asyncio.to_thread(
                    self._memory.get_scenario_result_header,
                    scenario_result_id=result.scenario_result_id,
                )
                if header is None or header.metadata.get(_SCHEDULER_METADATA_KEY) != _SCHEDULER_METADATA_VALUE:
                    continue
                if self._requires_original_run_binding(scenario_result=header):
                    try:
                        if header.scenario_identifier.params.get("execution_owner") != (
                            ScenarioExecutionOwner.APPROVED_ORIGINAL.value
                        ):
                            raise ValueError("A protected private Scenario cannot be reconciled in the web process.")
                        binding = OriginalRunBinding.model_validate(header.metadata[OriginalRunBinding.METADATA_KEY])
                        job = OriginalWorkerJob.model_validate(header.metadata[OriginalWorkerJob.METADATA_KEY])
                        if job.app_run_id != header.id:
                            raise ValueError("Original job identity differs from the stored run.")
                    except (KeyError, ValidationError, ValueError) as error:
                        logger.error("Interrupted protected job has no valid app binding (%s).", type(error).__name__)
                        await asyncio.to_thread(
                            self._memory.update_scenario_run_state,
                            scenario_result_id=result.scenario_result_id,
                            scenario_run_state=ScenarioRunState.FAILED,
                            error_message=OriginalRunReason.SOURCE_UNVERIFIED.value,
                            error_type=_INTERRUPTED_ERROR_TYPE,
                        )
                        reconciled += 1
                        continue
                    await asyncio.to_thread(
                        self._memory.update_scenario_run_state_and_metadata_fields,
                        scenario_result_id=result.scenario_result_id,
                        scenario_run_state=ScenarioRunState.FAILED,
                        error_message=OriginalRunReason.CLEANUP_PENDING.value,
                        error_type=_INTERRUPTED_ERROR_TYPE,
                        metadata_fields={
                            OriginalSourceResult.METADATA_KEY: OriginalSourceResult(
                                profile_ref=binding.profile_ref,
                                status=OriginalRunStatus.CLEANUP_UNCERTAIN,
                                source_state="error",
                                source_coverage_complete=False,
                                cleanup_state="uncontained",
                                reason=OriginalRunReason.CLEANUP_PENDING,
                            ).model_dump(mode="json")
                        },
                    )
                    reconciled += 1
                    continue
                await asyncio.to_thread(
                    self._memory.update_scenario_run_state,
                    scenario_result_id=result.scenario_result_id,
                    scenario_run_state=ScenarioRunState.FAILED,
                    error_message=_RESTART_INTERRUPTION_REASON,
                    error_type=_INTERRUPTED_ERROR_TYPE,
                )
                reconciled += 1
            if not has_more:
                return reconciled
            if not interrupted:
                raise RuntimeError(
                    "Scenario run state projection reported another page without returning a cursor row."
                )
            after_id = interrupted[-1].scenario_result_id

    async def shutdown_async(self) -> None:
        """Stop scheduling and terminalize active and queued runs for process shutdown."""
        task: asyncio.Task[None] | None = None
        retry_tasks: list[asyncio.Task[None]] = []
        errors: list[Exception] = []
        async with self._launch_lock:
            async with self._scheduler_lock:
                self._stopping = True
                retry_tasks = list(self._handoff_retry_tasks)
                queued = list(self._queued_runs)
                self._queued_runs.clear()
                if queued:
                    self._queue_revision += 1
                for run in queued:
                    if run.original_grant is not None:
                        continue
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
        for run in queued:
            if run.original_grant is None:
                continue
            assert run.original_job is not None
            cleanup = await self._release_original_grant_async(
                grant=run.original_grant, job=run.original_job, cancelled=True
            )
            try:
                await self._terminalize_original_job_async(
                    job=run.original_job,
                    binding=OriginalRunBinding(
                        profile_ref=run.original_grant.profile_ref, operator_oid=run.original_grant.operator_oid
                    ),
                    cleanup=cleanup,
                    state=ScenarioRunState.FAILED,
                )
            except Exception as error:
                errors.append(error)
        await asyncio.to_thread(self._prepare_executor.shutdown, wait=True)
        if self._original_cleanup_tasks:
            cleanup_outcomes = await asyncio.gather(*self._original_cleanup_tasks, return_exceptions=True)
            errors.extend(error for error in cleanup_outcomes if isinstance(error, Exception))
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
                self._memory.update_scenario_run_state_and_metadata_fields,
                scenario_result_id=scheduled.scenario_result_id,
                scenario_run_state=ScenarioRunState.QUEUED,
                metadata_fields={_SCHEDULER_METADATA_KEY: _SCHEDULER_METADATA_VALUE},
            )
            self._queued_runs.append(scheduled)
            self._queue_revision += 1

    async def _start_scheduled_run_locked_async(self, *, scheduled: _ActiveTask) -> None:
        """Start one run while the scheduler lock guarantees exclusive ownership."""
        scheduled.started_at = datetime.now(UTC)
        await asyncio.to_thread(
            self._memory.update_scenario_run_state_and_metadata_fields,
            scenario_result_id=scheduled.scenario_result_id,
            scenario_run_state=ScenarioRunState.IN_PROGRESS,
            metadata_fields={
                _SCHEDULER_METADATA_KEY: _SCHEDULER_METADATA_VALUE,
                SCENARIO_RUN_STARTED_AT_METADATA_KEY: scheduled.started_at.isoformat(),
            },
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

    def _can_retry_active_run(self, *, scenario_result_id: str) -> bool:
        """Return whether retry work may continue for the active run."""
        return not self._stopping and self._active_scenario_result_id == scenario_result_id

    async def _retry_handoff_async(self, *, scenario_result_id: str) -> None:
        """Retry scheduler handoff with bounded exponential delay until it succeeds or shutdown begins."""
        delay = _SCHEDULER_RETRY_INITIAL_SECONDS
        while self._can_retry_active_run(scenario_result_id=scenario_result_id):
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
        while self._can_retry_active_run(scenario_result_id=active.scenario_result_id):
            await asyncio.sleep(delay)
            try:
                async with self._scheduler_lock:
                    if not self._can_retry_active_run(scenario_result_id=active.scenario_result_id):
                        return
                    await asyncio.to_thread(
                        self._memory.try_update_scenario_run_state,
                        scenario_result_id=active.scenario_result_id,
                        expected_states={ScenarioRunState.CREATED, ScenarioRunState.IN_PROGRESS},
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
            scenario_name="ServerApprovedOriginalScenario" if run.original_grant is not None else run.scenario_name,
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

    async def _initialize_scenario_async(self, *, request: RunScenarioRequest, init_kwargs: dict[str, Any]) -> Scenario:
        """
        Build and initialize the scenario via the registry.

        Delegates the full create + set-parameters + initialize lifecycle to
        ``ScenarioRegistry.create_and_initialize_async`` so the registry owns
        scenario creation and initialization. The run-specific common parameters
        are resolved before this method and forwarded as ``init_kwargs``.

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
            initial_metadata={_SCHEDULER_METADATA_KEY: _SCHEDULER_METADATA_VALUE},
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
        if active.original_grant is not None:
            await self._execute_original_run_async(active=active)
            return
        assert active.scenario is not None
        handoff_ready = True

        try:
            await active.scenario.run_async()

        except asyncio.CancelledError:
            try:
                await asyncio.to_thread(
                    self._memory.try_update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    expected_states={ScenarioRunState.CREATED, ScenarioRunState.IN_PROGRESS},
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
                    self._memory.try_update_scenario_run_state,
                    scenario_result_id=scenario_result_id,
                    expected_states={ScenarioRunState.CREATED, ScenarioRunState.IN_PROGRESS},
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

    async def _execute_original_run_async(self, *, active: _ActiveTask) -> None:
        """Wait on one isolated worker job, then publish only broker-verified source and cleanup."""
        assert active.original_grant is not None and active.original_job is not None
        gateway = get_original_run_gateway()
        grant = active.original_grant
        job = active.original_job
        succeeded = False
        cancelled = False
        if gateway is None:
            logger.error("Original runner disappeared before its admitted worker started.")
        else:
            try:
                await gateway.runner.start_worker_async(grant=grant, job=job)
                await gateway.runner.wait_worker_async(grant=grant, job=job)
                succeeded = True
            except asyncio.CancelledError:
                cancelled = True
                try:
                    await gateway.runner.abort_worker_async(grant=grant, job=job)
                except Exception as error:
                    logger.error("Original worker abort could not be verified (%s).", type(error).__name__)
            except Exception as error:
                logger.error("An isolated original worker failed (%s).", type(error).__name__)

        cleanup = await self._release_original_grant_async(grant=grant, job=job, cancelled=cancelled or not succeeded)
        reason = OriginalRunReason.CLEANUP_PENDING if not cleanup.proved else OriginalRunReason.SOURCE_UNVERIFIED
        source_result: OriginalSourceResult | None = None
        evidence_link: OriginalRunEvidenceLink | None = None
        try:
            header = await asyncio.to_thread(
                self._memory.get_scenario_result_header, scenario_result_id=active.scenario_result_id
            )
            if header is None or gateway is None:
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            source_result, evidence_link = await asyncio.to_thread(
                gateway.verify_result,
                scenario_result=header,
                job=job,
                binding=OriginalRunBinding(profile_ref=grant.profile_ref, operator_oid=grant.operator_oid),
                cleanup=cleanup,
            )
            if not succeeded and source_result.status is OriginalRunStatus.COMPLETED:
                source_result = OriginalSourceResult.model_validate(
                    {
                        **source_result.model_dump(mode="json"),
                        "status": OriginalRunStatus.FAILED_SOURCE_VERIFIED.value,
                        "reason": OriginalRunReason.SOURCE_UNVERIFIED.value,
                    }
                )
            if cancelled and evidence_link is None:
                source_result = source_result.model_copy(update={"source_state": "cancelled"})
            fields = {
                OriginalCleanupReceipt.METADATA_KEY: cleanup.model_dump(mode="json"),
                OriginalSourceResult.METADATA_KEY: source_result.model_dump(mode="json"),
            }
            if evidence_link is not None:
                fields[OriginalRunEvidenceLink.METADATA_KEY] = evidence_link.model_dump(mode="json")
            await asyncio.to_thread(
                self._memory.update_scenario_metadata_fields,
                scenario_result_id=active.scenario_result_id,
                fields=fields,
            )
        except OriginalAdmissionError as error:
            reason = error.reason
            logger.error("Admitted original run lacks a verified source result (%s).", reason.value)
        except Exception as error:
            reason = OriginalRunReason.SOURCE_UNVERIFIED
            logger.error("Original source proof could not be retained (%s).", type(error).__name__)

        if source_result is None:
            active.error = reason.value
            active.retain_error_on_terminalization = True
            try:
                await self._terminalize_original_job_async(
                    job=job,
                    binding=OriginalRunBinding(profile_ref=grant.profile_ref, operator_oid=grant.operator_oid),
                    cleanup=cleanup,
                    state=ScenarioRunState.FAILED,
                )
            finally:
                await self._complete_handoff_async(scenario_result_id=active.scenario_result_id)
            return

        completed = (
            succeeded
            and cleanup.proved
            and source_result is not None
            and source_result.status is OriginalRunStatus.COMPLETED
        )
        state = (
            ScenarioRunState.COMPLETED
            if completed
            else ScenarioRunState.CANCELLED
            if cancelled and cleanup.proved and active.cancellation_state is not ScenarioRunState.FAILED
            else ScenarioRunState.FAILED
        )
        if state is ScenarioRunState.COMPLETED:
            active.error = None
        elif state is ScenarioRunState.FAILED:
            active.error = reason.value
            active.retain_error_on_terminalization = True
        try:
            updated = await asyncio.to_thread(
                self._memory.try_update_scenario_run_state,
                scenario_result_id=active.scenario_result_id,
                expected_states={
                    ScenarioRunState.CREATED,
                    ScenarioRunState.IN_PROGRESS,
                    ScenarioRunState.COMPLETED,
                    ScenarioRunState.FAILED,
                    ScenarioRunState.CANCELLED,
                },
                scenario_run_state=state,
                error_message=reason.value if state is ScenarioRunState.FAILED else None,
                error_type="OriginalAdmissionError" if state is ScenarioRunState.FAILED else None,
            )
            if not updated:
                logger.error("Original Scenario %s could not publish a terminal state.", active.scenario_result_id)
        finally:
            await self._complete_handoff_async(scenario_result_id=active.scenario_result_id)

    def _build_response(
        self,
        *,
        scenario_result_id: str,
        active_error: str | None,
        queue_position: int | None,
        active_scenario_result_id: str | None,
        operator: AuthenticatedUser | None = None,
    ) -> ScenarioRunSummary | None:
        """
        Build a ScenarioRunResponse by querying the database and merging active task state.

        Args:
            scenario_result_id: The scenario result ID.
            active_error: Error copied from the active asyncio task, if any.
            queue_position: Current 1-based waiting position, if queued.
            active_scenario_result_id: Currently executing scenario result ID.
            operator: Authenticated operator when building a protected original response.

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
            operator=operator,
        )

    def _build_response_from_db(
        self,
        *,
        scenario_result: ScenarioResult,
        active_error: str | None = None,
        queue_position: int | None = None,
        active_scenario_result_id: str | None = None,
        operator: AuthenticatedUser | None = None,
    ) -> ScenarioRunSummary:
        """
        Build a ScenarioRunResponse from a database ScenarioResult, merged with active task info.

        Args:
            scenario_result: A ScenarioResult retrieved from CentralMemory.
            active_error: Error copied from the active asyncio task, if any.
            queue_position: Current 1-based waiting position, if queued.
            active_scenario_result_id: Currently executing scenario result ID.
            operator: Authenticated operator when building a protected original response.

        Returns:
            The API response model.
        """
        self._reject_unbound_original_run(scenario_result=scenario_result)
        if OriginalRunBinding.METADATA_KEY in (getattr(scenario_result, "metadata", None) or {}):
            return self._original_run_summary(
                scenario_result=scenario_result,
                operator=operator,
                queue_position=queue_position,
                active_scenario_result_id=active_scenario_result_id,
            )
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
        safe_failure = self._original_inspect_failure_reason(
            scenario_name=scenario_result.scenario_name,
            scenario_run_state=scenario_result.scenario_run_state,
            error_message=scenario_result.error_message,
        )
        if safe_failure is not None:
            error = safe_failure

        status = scenario_result.scenario_run_state
        terminal = status in (
            ScenarioRunState.COMPLETED,
            ScenarioRunState.FAILED,
            ScenarioRunState.CANCELLED,
        )
        try:
            plan = self._load_run_plan(scenario_result=scenario_result)
        except (ValidationError, ValueError):
            logger.warning(
                "Scenario run %s has invalid persisted plan metadata; using legacy run detail fields.",
                scenario_result_id,
            )
            plan = None
        plan_lookup = self._progress_read_model.build_plan_lookup(plan=plan)
        original_inspect_import = self.verify_original_inspect_import(
            memory=self._memory, scenario_result=scenario_result, plan=plan
        )
        completed_import_units = self._original_inspect_completed_units(plan=plan, imported=original_inspect_import)

        # Build result fields from DB (always computed so in-progress runs show progress)
        total_attacks, completed_attacks, objective_achieved_rate, successful_attacks = (
            self._progress_read_model.calculate_progress_counts(
                scenario_result=scenario_result,
                plan=plan,
                plan_lookup=plan_lookup,
                completed_without_attack_result=completed_import_units,
            )
        )
        techniques_used = (
            list(dict.fromkeys(group.display_group for group in plan.atomic_groups))
            if plan is not None
            else scenario_result.get_techniques_used()
        )
        target, datasets_used, scenario_parameters = self._safe_run_metadata(
            scenario_identifier=getattr(scenario_result, "scenario_identifier", None)
        )

        # Surface per-attack errors and retry pressure regardless of overall run status:
        # a COMPLETED scenario can still hide errored objectives or rate-limit retries.
        failed_attacks: list[AttackErrorSummary] = []
        attack_retries: list[AttackRetrySummary] = []
        persisted_retries: list[int] = []
        overload_events: deque[Any] = deque(maxlen=_MAX_OVERLOAD_EVENTS)
        attempts_by_unit: dict[ResultUnitIdentity, int] = {}
        for atomic_attack_name, results in scenario_result.attack_results.items():
            for attack_result in results:
                unit_identity = self._progress_read_model.resolve_result_unit_identity(
                    atomic_attack_name=atomic_attack_name,
                    attack_result=attack_result,
                    plan_lookup=plan_lookup,
                )
                attempts_by_unit[unit_identity] = attempts_by_unit.get(unit_identity, 0) + 1
                retries = getattr(attack_result, "total_retries", 0)
                if isinstance(retries, int):
                    persisted_retries.append(retries)

                retry_events = getattr(attack_result, "retry_events", None)
                if isinstance(retry_events, list) and retry_events:
                    overload_events.extend(retry_events)
                    attack_retries.append(
                        AttackRetrySummary(
                            attack_result_id=str(attack_result.attack_result_id),
                            atomic_attack_name=atomic_attack_name,
                            retries=retry_events,
                        )
                    )

                if attack_result.outcome == AttackOutcome.ERROR:
                    failed_attacks.append(
                        AttackErrorSummary(
                            atomic_attack_name=atomic_attack_name,
                            objective=attack_result.objective,
                            error_type=attack_result.error_type,
                            error_message=attack_result.error_message,
                            total_retries=max(0, retries) if isinstance(retries, int) else 0,
                        )
                    )
        total_retries = self._progress_read_model.total_retry_pressure(
            attempts_per_unit=attempts_by_unit.values(),
            persisted_retries=persisted_retries,
        )

        updated_at = scenario_result.creation_time
        if terminal and scenario_result.completion_time is not None:
            updated_at = scenario_result.completion_time

        return ScenarioRunSummary(
            scenario_result_id=scenario_result_id,
            scenario_name=scenario_result.scenario_name,
            scenario_registry_name=plan.scenario_registry_name if plan else None,
            scenario_version=scenario_result.scenario_version,
            status=status,
            created_at=scenario_result.creation_time,
            started_at=self._load_started_at(scenario_result=scenario_result),
            updated_at=updated_at,
            error=error,
            error_type=error_type,
            techniques_used=techniques_used,
            total_attacks=total_attacks,
            completed_attacks=completed_attacks,
            objective_achieved_rate=(
                None if scenario_result.scenario_name == _ORIGINAL_INSPECT_SCENARIO_NAME else objective_achieved_rate
            ),
            failed_attacks=failed_attacks,
            attack_retries=attack_retries,
            total_retries=total_retries,
            labels=scenario_result.labels,
            completed_at=scenario_result.completion_time if terminal else None,
            pyrit_version=(
                scenario_result.pyrit_version
                if isinstance(getattr(scenario_result, "pyrit_version", None), str)
                else None
            ),
            target=target,
            datasets_used=datasets_used,
            scenario_parameters=scenario_parameters,
            planned_total_available=plan is not None,
            successful_attacks=successful_attacks,
            error_attacks=len(failed_attacks),
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
            overload_summaries=self._build_overload_summaries(retry_events=overload_events),
            original_inspect_import=original_inspect_import,
        )

    def _original_run_binding(
        self, *, scenario_result: ScenarioResult, operator: AuthenticatedUser | None
    ) -> tuple[OriginalRunBinding, OriginalSourceResult | None]:
        """
        Recheck the host ACL and typed source/cleanup receipts without exposing private Scenario metadata.

        Returns:
            tuple[OriginalRunBinding, OriginalSourceResult | None]: The authorized binding and safe result.

        Raises:
            OriginalAdmissionError: If authorization or persisted source proof is missing.
        """
        gateway = get_original_run_gateway()
        if gateway is None:
            raise OriginalAdmissionError(reason=OriginalRunReason.RUNNER_NOT_CONFIGURED)
        metadata = scenario_result.metadata or {}
        try:
            binding = OriginalRunBinding.model_validate(metadata.get(OriginalRunBinding.METADATA_KEY))
            job = OriginalWorkerJob.model_validate(metadata.get(OriginalWorkerJob.METADATA_KEY))
        except ValidationError as error:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error
        if (
            not gateway.authorized(operator=operator)
            or operator is None
            or operator.oid != binding.operator_oid
            or binding.profile_ref != gateway.runner.profile_ref
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        if job.app_run_id != scenario_result.id:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        raw_cleanup = metadata.get(OriginalCleanupReceipt.METADATA_KEY)
        try:
            cleanup = OriginalCleanupReceipt.model_validate(raw_cleanup) if raw_cleanup is not None else None
            raw_published = metadata.get(OriginalSourceResult.METADATA_KEY)
            published = OriginalSourceResult.model_validate(raw_published) if raw_published is not None else None
            raw_link = metadata.get(OriginalRunEvidenceLink.METADATA_KEY)
            retained_link = OriginalRunEvidenceLink.model_validate(raw_link) if raw_link is not None else None
        except ValidationError as error:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error
        if scenario_result.scenario_run_state in (ScenarioRunState.FAILED, ScenarioRunState.CANCELLED) and (
            published is None or (cleanup is None and published.status is not OriginalRunStatus.CLEANUP_UNCERTAIN)
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        if (
            cleanup is None
            and published is None
            and retained_link is None
            and scenario_result.scenario_run_state
            not in (
                ScenarioRunState.COMPLETED,
                ScenarioRunState.FAILED,
                ScenarioRunState.CANCELLED,
            )
        ):
            return binding, None
        verified, verified_link = gateway.verify_result(
            scenario_result=scenario_result, job=job, binding=binding, cleanup=cleanup
        )
        if (
            published != verified
            or retained_link != verified_link
            or (
                scenario_result.scenario_run_state is ScenarioRunState.COMPLETED
                and verified.status is not OriginalRunStatus.COMPLETED
            )
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        return binding, verified

    @staticmethod
    def _requires_original_run_binding(*, scenario_result: ScenarioResult) -> bool:
        """
        Detect protected TaskOwnedScenario identity even if its metadata binding was removed.

        Returns:
            bool: True only for Scenario identities that opted into server admission.
        """
        identifier = getattr(scenario_result, "scenario_identifier", None)
        return isinstance(identifier, ScenarioIdentifier) and (
            identifier.class_name == "ServerApprovedOriginalScenario"
            or identifier.params.get("execution_owner") == ScenarioExecutionOwner.APPROVED_ORIGINAL.value
            or identifier.params.get("server_admission_required") is True
        )

    @classmethod
    def _reject_unbound_original_run(cls, *, scenario_result: ScenarioResult) -> None:
        """
        Refuse a protected result whose independent admission binding has vanished.

        Raises:
            OriginalAdmissionError: If a protected run could otherwise fall through to generic output.
        """
        protected = cls._requires_original_run_binding(scenario_result=scenario_result)
        if protected and OriginalRunBinding.METADATA_KEY not in (getattr(scenario_result, "metadata", None) or {}):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)

    def _original_run_summary(
        self,
        *,
        scenario_result: ScenarioResult,
        operator: AuthenticatedUser | None,
        queue_position: int | None,
        active_scenario_result_id: str | None,
    ) -> ScenarioRunSummary:
        """
        Present only approved source grade, completion and cleanup; never a private source identifier.

        Returns:
            ScenarioRunSummary: The sanitized operator-facing run state.
        """
        binding, result = self._original_run_binding(scenario_result=scenario_result, operator=operator)
        status = scenario_result.scenario_run_state
        if result is not None and result.status is OriginalRunStatus.CLEANUP_UNCERTAIN:
            readiness = OriginalRunStatus.CLEANUP_UNCERTAIN
        elif status is ScenarioRunState.COMPLETED:
            readiness = OriginalRunStatus.COMPLETED
        elif status is ScenarioRunState.IN_PROGRESS:
            readiness = OriginalRunStatus.RUNNING
        elif status in (ScenarioRunState.CREATED, ScenarioRunState.QUEUED):
            readiness = OriginalRunStatus.ADMISSION_PENDING
        else:
            readiness = OriginalRunStatus.FAILED_UNGRADED
        safe_error = scenario_result.error_message
        stored_reason = next((reason for reason in OriginalRunReason if reason.value == safe_error), None)
        failure_reason = (
            result.reason
            if result is not None and result.reason is not None
            else (stored_reason or OriginalRunReason.SOURCE_UNVERIFIED if status is ScenarioRunState.FAILED else None)
        )
        admission = OriginalRunAdmission(
            profile_ref=binding.profile_ref,
            status=readiness,
            unmet_conditions=[failure_reason] if failure_reason is not None else [],
        )
        terminal = status in (ScenarioRunState.COMPLETED, ScenarioRunState.FAILED, ScenarioRunState.CANCELLED)
        return ScenarioRunSummary(
            scenario_result_id=str(scenario_result.id),
            scenario_name="ServerApprovedOriginalScenario",
            scenario_registry_name=APPROVED_ORIGINAL_SCENARIO,
            scenario_version=1,
            status=status,
            created_at=scenario_result.creation_time,
            started_at=self._load_started_at(scenario_result=scenario_result),
            updated_at=(
                scenario_result.completion_time
                if terminal and scenario_result.completion_time is not None
                else scenario_result.creation_time
            ),
            error=failure_reason.value if failure_reason is not None else None,
            error_type=None,
            techniques_used=["original_task"],
            total_attacks=1,
            completed_attacks=int(status is ScenarioRunState.COMPLETED),
            objective_achieved_rate=None,
            failed_attacks=[],
            attack_retries=[],
            total_retries=0,
            labels={},
            completed_at=scenario_result.completion_time if terminal else None,
            pyrit_version=scenario_result.pyrit_version,
            target=None,
            datasets_used=[],
            scenario_parameters={"profile_ref": binding.profile_ref, "model_role": "evaluated"},
            planned_total_available=True,
            successful_attacks=0,
            error_attacks=0,
            attack_details_available=False,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
            original_run_admission=admission,
            original_source_result=result,
        )

    @staticmethod
    def _parse_history_plan(*, record: ScenarioHistoryRunRecord) -> list[ScenarioRunPlanAtomicGroup] | None:
        """
        Validate the compact persisted run plan projected onto one history row.

        Returns:
            list[ScenarioRunPlanAtomicGroup] | None: Planned atomic groups, or None when the
                run has no plan or the persisted plan cannot identify units unambiguously.
        """
        if record.plan_atomic_groups is None:
            return None
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
                else record.plan_seed_id_map or []
            )
            seed_hash_by_id: dict[str, str] = {}
            for seed in _HISTORY_SEED_ID_MAP_ADAPTER.validate_python(raw_seed_map):
                seed_id = seed.get("id")
                objective_sha256 = seed.get("objective_sha256")
                if not seed_id or not objective_sha256:
                    raise ValueError("seed projection is missing required identity fields")
                previous_hash = seed_hash_by_id.get(seed_id)
                if previous_hash is not None and previous_hash != objective_sha256:
                    raise ValueError("conflicting objective hashes for seed group")
                seed_hash_by_id[seed_id] = objective_sha256
            for group in atomic_groups:
                objective_hashes = [
                    seed_hash_by_id[seed_id] for seed_id in group.seed_group_ids if seed_id in seed_hash_by_id
                ]
                if len(objective_hashes) != len(set(objective_hashes)):
                    raise ValueError("ambiguous objective hash within atomic group")
            return atomic_groups
        except (json.JSONDecodeError, ValidationError, ValueError):
            logger.warning(
                "Scenario run %s has an incomplete persisted plan; using legacy history totals.",
                record.scenario_result_id,
            )
            return None

    def _build_history_summary(
        self,
        *,
        record: ScenarioHistoryRunRecord,
        atomic_groups: list[ScenarioRunPlanAtomicGroup] | None,
        aggregate: ScenarioHistoryAggregate,
        original_result: ScenarioResult | None = None,
    ) -> ScenarioRunListItem:
        """
        Map lightweight persisted history projections to the public summary DTO.

        Returns:
            ScenarioRunListItem: Safe, aggregated history summary.
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

        planned_total = (
            len({(group.id, seed_group_id) for group in atomic_groups for seed_group_id in group.seed_group_ids})
            if atomic_groups is not None
            else aggregate.unit_count
        )
        completed = aggregate.completed_units
        successful = aggregate.successful_units
        status = ScenarioRunState(record.status)
        is_original_inspect = record.scenario_name == _ORIGINAL_INSPECT_SCENARIO_NAME
        if is_original_inspect:
            if original_result is None:
                raise ValueError("Original Inspect history run is missing its persisted Scenario result.")
            plan = self._load_run_plan(scenario_result=original_result)
            if plan is None:
                raise ValueError("Original Inspect history run is missing its persisted case plan.")
            imported = self.verify_original_inspect_import(
                memory=self._memory, scenario_result=original_result, plan=plan
            )
            completed_units = self._original_inspect_completed_units(plan=plan, imported=imported)
            planned_total, completed, _, successful = self._progress_read_model.calculate_progress_counts(
                scenario_result=original_result,
                plan=plan,
                plan_lookup=self._progress_read_model.build_plan_lookup(plan=plan),
                completed_without_attack_result=completed_units,
            )
            atomic_groups = plan.atomic_groups
        terminal = status in (
            ScenarioRunState.COMPLETED,
            ScenarioRunState.FAILED,
            ScenarioRunState.CANCELLED,
        )
        timestamps = [record.created_at]
        if aggregate.latest_attempt_timestamp is not None:
            timestamps.append(aggregate.latest_attempt_timestamp)
        if terminal and record.completed_at is not None:
            timestamps.append(record.completed_at)
        techniques = (
            list(dict.fromkeys(group.display_group for group in atomic_groups))
            if atomic_groups is not None
            else list(aggregate.atomic_attack_names)
        )
        return ScenarioRunListItem(
            scenario_result_id=record.scenario_result_id,
            scenario_name=record.scenario_name,
            scenario_registry_name=record.scenario_registry_name,
            scenario_version=record.scenario_version,
            status=status,
            created_at=record.created_at,
            started_at=record.started_at,
            updated_at=max(timestamps),
            error=self._original_inspect_failure_reason(
                scenario_name=record.scenario_name,
                scenario_run_state=status,
                error_message=record.error_message,
            )
            if is_original_inspect
            else record.error_message,
            error_type=record.error_type,
            techniques_used=techniques,
            total_attacks=planned_total if atomic_groups is not None or planned_total else None,
            completed_attacks=completed,
            objective_achieved_rate=(
                None if is_original_inspect else (int((successful / completed) * 100) if completed else 0)
            ),
            total_retries=aggregate.total_retries,
            labels=record.labels,
            completed_at=record.completed_at if terminal else None,
            pyrit_version=record.pyrit_version,
            target=target,
            datasets_used=datasets_used,
            scenario_parameters=scenario_parameters,
            planned_total_available=atomic_groups is not None,
            successful_attacks=successful,
            error_attacks=aggregate.error_attempts,
            attack_details_available=False,
        )

    @staticmethod
    def _load_started_at(*, scenario_result: ScenarioResult) -> datetime | None:
        """
        Load the persisted aware execution start timestamp from scenario metadata.

        Returns:
            datetime | None: The execution start, or None for legacy or invalid metadata.
        """
        raw_value = (getattr(scenario_result, "metadata", None) or {}).get(SCENARIO_RUN_STARTED_AT_METADATA_KEY)
        if raw_value is None:
            return None
        try:
            started_at = _STARTED_AT_ADAPTER.validate_python(raw_value)
        except ValidationError:
            return None
        return started_at if started_at.tzinfo is not None else None

    @staticmethod
    def _identifier_techniques(scenario_identifier: ScenarioIdentifier | None) -> list[str]:
        """
        Read techniques when legacy persisted metadata has an identifier.

        Returns:
            list[str]: Stored techniques or an empty list.
        """
        return list(scenario_identifier.techniques or []) if scenario_identifier is not None else []

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

    @classmethod
    def verify_original_inspect_import(
        cls, *, memory: MemoryInterface, scenario_result: ScenarioResult, plan: ScenarioRunPlan | None
    ) -> OriginalInspectImportSummary | None:
        """
        Read the original archive reference and verify its persisted offline result.

        Returns:
            OriginalInspectImportSummary | None: The verified import, or none while it is pending.

        Raises:
            ValueError: If a completed run lacks its import or linked Score/AttackResult.
        """
        if scenario_result.scenario_name != _ORIGINAL_INSPECT_SCENARIO_NAME:
            return None
        raw = (scenario_result.metadata or {}).get(OriginalInspectImportSummary.METADATA_KEY)
        if raw is None:
            if scenario_result.scenario_run_state == ScenarioRunState.COMPLETED:
                raise ValueError("Completed original Inspect Scenario has no persisted import reference.")
            return None
        imported = OriginalInspectImportSummary.model_validate(raw)
        derived_case_id, objective = cls._validate_original_inspect_projection(memory=memory, imported=imported)
        if plan is None or len(plan.seed_groups) != 1:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        seed = plan.seed_groups[0]
        if (
            seed.case_id != derived_case_id
            or seed.objective != objective
            or seed.objective_sha256 != to_sha256(objective)
            or seed.prompts
            or seed.input_variant_sha256 is not None
        ):
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        return imported

    @classmethod
    def _validate_original_inspect_projection(
        cls, *, memory: MemoryInterface, imported: OriginalInspectImportSummary
    ) -> tuple[str, str]:
        """
        Recheck the source-attributed Score, AttackResult and original archive on readback.

        Returns:
            tuple[str, str]: The source-derived case ID and original Sample input.

        Raises:
            ValueError: If referenced rows, their FK, or their source provenance differ.
        """
        from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory

        if imported.source_sha256 != EvalSourceFactory.ORIGINAL_INERT_SHA256:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        scores = memory.get_scores(score_ids=[str(imported.score_id)])
        attacks = memory.get_attack_results(attack_result_ids=[str(imported.attack_result_id)])
        if len(scores) != 1 or len(attacks) != 1:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        score, attack = scores[0], attacks[0]
        score_metadata = score.score_metadata
        if score_metadata is None:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        expected = {
            "inspect_source": "original_eval_log",
            "inspect_archive_sha256": imported.archive_sha256,
            "inspect_run_id": imported.inspect_run_id,
            "inspect_eval_id": imported.inspect_eval_id,
            "inspect_task": imported.task_id.value,
            "inspect_case_run_id": imported.case_run_id,
            "inspect_primary_scorer": imported.primary_scorer,
        }
        if (
            str(score.id) != str(imported.score_id)
            or attack.attack_result_id != str(imported.attack_result_id)
            or score.status is not ScoreStatus.COMPLETE
            or score.score_type != imported.score_type
            or score.score_value != imported.score_value
            or attack.automated_score is None
            or str(attack.automated_score.id) != str(score.id)
            or attack.outcome is not AttackOutcome.UNDETERMINED
            or attack.attribution_parent_id is not None
            or attack.timestamp != score.timestamp
            or not score_metadata.get("inspect_final_score_event_id")
            or not score_metadata.get("inspect_final_score_event_sha256")
            or any(score_metadata.get(key) != value for key, value in expected.items())
            or any(attack.metadata.get(key) != value for key, value in score_metadata.items())
        ):
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        return cls._verify_original_inspect_archive(
            memory=memory,
            imported=imported,
            task_version=score_metadata.get("inspect_task_version"),
            score_metadata=score_metadata,
            score_timestamp=score.timestamp,
            attack=attack,
        )

    @classmethod
    def _verify_original_inspect_archive(
        cls,
        *,
        memory: MemoryInterface,
        imported: OriginalInspectImportSummary,
        task_version: str | int | float | None,
        score_metadata: Mapping[str, str | int | float],
        score_timestamp: datetime,
        attack: AttackResult,
    ) -> tuple[str, str]:
        """
        Match the referenced result to the verified bytes of its finalized `.eval`.

        Returns:
            tuple[str, str]: The source-derived case ID and original Sample input.

        Raises:
            ValueError: If the offline episode, archive or original final event is inconsistent.
        """
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        try:
            episode = memory.native_cyber_evidence.get_finalized_unscored_inspect_capture(
                run_id=imported.projection_episode_id
            )
        except (KeyError, ValueError) as error:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION) from error
        archive_key = InspectOriginalEvalImporter.ARCHIVE_KEY
        archive_streams = [
            stream for stream in episode.raw_streams if stream.key.observed_source_id == archive_key.observed_source_id
        ]
        if (
            not episode.coverage_complete
            or episode.run.task_id != imported.task_id.value
            or episode.run.task_version != task_version
            or episode.run.source_session_id != imported.inspect_run_id
            or len(episode.turns) != 1
            or len(episode.run.required_raw_streams) != 2
            or set(episode.run.required_raw_streams) != {archive_key, InspectOriginalEvalImporter.RESOLVED_KEY}
            or len(archive_streams) != 1
            or archive_streams[0].key != archive_key
            or not archive_streams[0].source_complete
            or archive_streams[0].stored_sha256 != imported.archive_sha256
            or archive_streams[0].observed_sha256 != imported.archive_sha256
            or archive_streams[0].expected_bytes != archive_streams[0].stored_bytes
            or archive_streams[0].truncated
            or archive_streams[0].omitted_bytes
            or not 0 < archive_streams[0].stored_bytes <= InspectOriginalEvalImporter.MAX_ARCHIVE_BYTES
        ):
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        archive = cls._read_original_inspect_stream(
            memory=memory,
            run_id=imported.projection_episode_id,
            stream_id=archive_streams[0].stream_id,
            expected_bytes=archive_streams[0].stored_bytes,
            expected_sha256=imported.archive_sha256,
        )
        return cls._verify_final_original_event(
            memory=memory,
            imported=imported,
            episode=episode,
            archive=archive,
            score_metadata=score_metadata,
            score_timestamp=score_timestamp,
            attack=attack,
        )

    @classmethod
    def _verify_original_inspect_resolved(
        cls,
        *,
        memory: MemoryInterface,
        imported: OriginalInspectImportSummary,
        episode: NativeCyberEpisodeSnapshot,
        resolved: bytes,
    ) -> None:
        """Verify the required resolved stream against the typed original log."""
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        key = InspectOriginalEvalImporter.RESOLVED_KEY
        streams = [stream for stream in episode.raw_streams if stream.key.observed_source_id == key.observed_source_id]
        digest = hashlib.sha256(resolved).hexdigest()
        if (
            key not in episode.run.required_raw_streams
            or len(streams) != 1
            or streams[0].key != key
            or not streams[0].source_complete
            or streams[0].expected_bytes != streams[0].stored_bytes
            or streams[0].truncated
            or streams[0].omitted_bytes
            or not 0 < streams[0].stored_bytes <= InspectOriginalEvalImporter.MAX_RESOLVED_BYTES
            or streams[0].stored_bytes != len(resolved)
            or streams[0].stored_sha256 != digest
            or streams[0].observed_sha256 != digest
        ):
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        if (
            cls._read_original_inspect_stream(
                memory=memory,
                run_id=imported.projection_episode_id,
                stream_id=streams[0].stream_id,
                expected_bytes=streams[0].stored_bytes,
                expected_sha256=digest,
            )
            != resolved
        ):
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)

    @staticmethod
    def _read_original_inspect_stream(
        *, memory: MemoryInterface, run_id: str, stream_id: uuid.UUID, expected_bytes: int, expected_sha256: str
    ) -> bytes:
        """
        Read bounded private chunks through their integrity-checking memory API.

        Returns:
            bytes: A private stream for internal typed-log verification only.

        Raises:
            ValueError: If a stored chunk, length or digest differs from its sealed stream.
        """
        capture = memory.native_cyber_evidence
        checksum = hashlib.sha256()
        archive = bytearray()
        after_sequence = 0
        while True:
            try:
                chunks = capture.read_raw_chunks(
                    run_id=run_id,
                    stream_id=stream_id,
                    allow_sensitive=True,
                    after_sequence=after_sequence,
                    limit=capture.MAX_RAW_READ_CHUNKS,
                )
            except (KeyError, ValueError) as error:
                raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION) from error
            for chunk in chunks:
                archive.extend(chunk.data)
                checksum.update(chunk.data)
                if len(archive) > expected_bytes:
                    raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
            if len(chunks) < capture.MAX_RAW_READ_CHUNKS:
                break
            after_sequence = chunks[-1].sequence
        if len(archive) != expected_bytes or checksum.hexdigest() != expected_sha256:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
        return bytes(archive)

    @classmethod
    def _verify_final_original_event(
        cls,
        *,
        memory: MemoryInterface,
        imported: OriginalInspectImportSummary,
        episode: NativeCyberEpisodeSnapshot,
        archive: bytes,
        score_metadata: Mapping[str, str | int | float],
        score_timestamp: datetime,
        attack: AttackResult,
    ) -> tuple[str, str]:
        """
        Compare the persisted ScoreEvent, planned case, and conversation to the typed `.eval`.

        Returns:
            tuple[str, str]: The source-derived case ID and original Sample input.

        Raises:
            ValueError: If the original event, typed sample or projected events have changed.
        """
        from inspect_ai.log import read_eval_log

        from pyrit.executor.benchmark.inspect_eval_projection import (
            InspectProjectionVersion,
            final_original_score_event,
            project_inspect_sample,
        )
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        try:
            log = read_eval_log(io.BytesIO(archive), resolve_attachments="full", format="eval")
            InspectOriginalEvalImporter._validate_log(log=log, cases=None, run=None)
            samples = log.samples or []
            if (
                log.status != "success"
                or log.eval.run_id != imported.inspect_run_id
                or log.eval.eval_id != imported.inspect_eval_id
                or log.eval.task != imported.task_id.value
                or len(samples) != 1
            ):
                raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
            sample = samples[0]
            if not isinstance(sample.input, str):
                raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
            resolved = log.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
            cls._verify_original_inspect_resolved(memory=memory, imported=imported, episode=episode, resolved=resolved)
            event = final_original_score_event(sample=sample, scorer_name=imported.primary_scorer)
            score_type, score_value, _ = InspectOriginalEvalImporter._representable_value(
                value=event.score.value if event is not None else None
            )
            source_timestamp = InspectOriginalEvalImporter._sample_timestamp(
                sample.completed_at, fallback=sample.started_at or log.eval.created
            )
            if (
                event is None
                or event.uuid != score_metadata.get("inspect_final_score_event_id")
                or config_hash({"event": event.model_dump(mode="json", exclude_none=True)})
                != score_metadata.get("inspect_final_score_event_sha256")
                or str(sample.id) != score_metadata.get("inspect_sample_id")
                or sample.epoch != score_metadata.get("inspect_epoch")
                or not sample.uuid
                or score_metadata.get("inspect_sample_uuid") != sample.uuid
                or str(log.eval.task_version) != score_metadata.get("inspect_task_version")
                or score_type != imported.score_type
                or score_value != imported.score_value
                or score_timestamp != source_timestamp
                or attack.timestamp != source_timestamp
                or attack.objective
                != f"Original Inspect task {log.eval.task} Sample {sample.id} epoch {sample.epoch} (offline import)"
                or ("inspect_turn_count" in attack.metadata) != (sample.turn_count is not None)
                or (
                    sample.turn_count is not None
                    and (
                        type(attack.metadata.get("inspect_turn_count")) is not int
                        or attack.metadata["inspect_turn_count"] != sample.turn_count
                    )
                )
            ):
                raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
            if (
                attack.conversation_id
                != InspectOriginalEvalImporter._conversation_id(
                    episode_id=imported.projection_episode_id,
                    run_id=imported.inspect_run_id,
                    sample=sample,
                    sample_index=1,
                )
                or attack.last_response is not None
                or attack.related_conversations
            ):
                raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)
            projection = project_inspect_sample(
                sample=sample,
                log_run_id=imported.inspect_run_id,
                eval_id=imported.inspect_eval_id,
                archive_sha256=imported.archive_sha256,
                sample_index=1,
                start_sequence=1,
                conversation_id=attack.conversation_id,
                case_run_id=imported.case_run_id,
                projection_version=InspectProjectionVersion.from_binding_version(episode.run.binding_version),
            )
            cls._verify_original_inspect_message_pieces(
                memory=memory,
                episode=episode,
                conversation_id=attack.conversation_id,
                expected_pieces=projection.message_pieces,
            )
            InspectOriginalEvalImporter(memory=memory)._verify_event_readback(
                log=log,
                snapshot=episode,
                archive_sha=imported.archive_sha256,
                case_run_ids=(imported.case_run_id,),
                required_source_gaps=[],
            )
            case_id = EvalCaseRef(
                package=EvalPackageRef(
                    kind=EvalSourceKind.NAMED,
                    name=imported.task_id.value,
                    source_sha256=imported.source_sha256,
                ),
                task_name=log.eval.task,
                task_version=str(log.eval.task_version),
                sample_id=str(sample.id),
                epoch=sample.epoch,
            ).case_id
            return case_id, sample.input
        except (BadZipFile, KeyError, ValueError) as error:
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION) from error

    @staticmethod
    def _verify_original_inspect_message_pieces(
        *,
        memory: MemoryInterface,
        episode: NativeCyberEpisodeSnapshot,
        conversation_id: str,
        expected_pieces: Sequence[MessagePiece],
    ) -> None:
        """Match the retained conversation to the text and provenance projected from the typed Sample."""
        turn = episode.turns[0]
        retained_ids = (
            *turn.request_piece_ids,
            *turn.response_piece_ids,
            *turn.tool_request_piece_ids,
            *turn.tool_result_piece_ids,
        )
        stored = memory.get_message_pieces(conversation_id=conversation_id)
        if (
            not retained_ids
            or len(retained_ids) != len(expected_pieces)
            or len(stored) != len(expected_pieces)
            or {piece.id for piece in stored} != set(retained_ids)
        ):
            raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)

        def part_index(piece: MessagePiece) -> int:
            """
            Return the original text-part position or an invalid sentinel.

            Returns:
                int: The part index, or -1 if missing or malformed.
            """
            value = (piece.prompt_metadata or {}).get("inspect_part_index")
            return value if type(value) is int else -1

        ordered = sorted(stored, key=lambda piece: (piece.sequence, part_index(piece)))
        for actual, expected in zip(ordered, expected_pieces, strict=True):
            if (
                actual.role != expected.role
                or actual.sequence != expected.sequence
                or actual.original_value != expected.original_value
                or actual.original_value_sha256 != expected.original_value_sha256
                or actual.original_value_data_type != expected.original_value_data_type
                or actual.converted_value != expected.converted_value
                or actual.converted_value_sha256 != expected.converted_value_sha256
                or actual.converted_value_data_type != expected.converted_value_data_type
                or actual.response_error != expected.response_error
                or actual.prompt_metadata != expected.prompt_metadata
                or actual.converter_identifiers != expected.converter_identifiers
                or actual.original_prompt_id != actual.id
            ):
                raise ValueError(_INVALID_ORIGINAL_INSPECT_PROJECTION)

    @staticmethod
    def _original_inspect_completed_units(
        *, plan: ScenarioRunPlan | None, imported: OriginalInspectImportSummary | None
    ) -> frozenset[ResultUnitIdentity]:
        """
        Match a verified import's durable case-run ID to its one planned unit.

        Returns:
            frozenset[ResultUnitIdentity]: The completed case, or none before import.

        Raises:
            ValueError: If source, run, or case identity differs from the persisted plan.
        """
        if imported is None:
            return frozenset[ResultUnitIdentity]()
        if (
            plan is None
            or plan.scenario_registry_name != _ORIGINAL_INSPECT_REGISTRY_NAME
            or plan.run_instance_id is None
            or plan.eval_spec_sha256 is None
            or len(plan.atomic_groups) != 1
            or len(plan.seed_groups) != 1
        ):
            raise ValueError("Original Inspect import does not match its planned case or run.")
        group, seed = plan.atomic_groups[0], plan.seed_groups[0]
        if (
            imported.episode_id != f"inspect-run-{plan.run_instance_id.hex}"
            or seed.id != imported.case_run_id
            or seed.source_sha256 != imported.source_sha256
            or group.seed_group_ids != [imported.case_run_id]
            or group.atomic_attack_name != f"eval_case_{imported.case_run_id}"
            or group.technique_eval_hash != plan.eval_spec_sha256
            or group.id
            != config_hash(
                {"atomic_attack_name": group.atomic_attack_name, "technique_eval_hash": group.technique_eval_hash}
            )
        ):
            raise ValueError("Original Inspect import does not match its planned case or run.")
        return frozenset({ResultUnitIdentity(atomic_group_id=group.id, seed_group_id=seed.id)})

    @staticmethod
    def _original_inspect_failure_reason(
        *, scenario_name: str, scenario_run_state: ScenarioRunState, error_message: str | None
    ) -> str | None:
        """
        Show only vetted diagnoses from persisted original-Task errors.

        Returns:
            str | None: A safe failure reason, or none for unrelated/unfinished runs.
        """
        if scenario_name != _ORIGINAL_INSPECT_SCENARIO_NAME:
            return None
        if scenario_run_state == ScenarioRunState.CANCELLED:
            return "Original Inspect run was cancelled before a qualified result was published."
        if scenario_run_state != ScenarioRunState.FAILED:
            return None
        persisted = error_message or ""
        for prefix in _SAFE_ORIGINAL_INSPECT_FAILURE_PREFIXES:
            if persisted.startswith(prefix):
                return prefix
        return "Original Inspect Task or offline projection failed; reconcile its retained log before retrying."

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

    def snapshot_active_run(
        self, *, scenario_result_id: str, operator: AuthenticatedUser | None = None
    ) -> _ActiveRunSnapshot:
        """
        Copy asyncio-owned run state for use by database-only worker-thread methods.

        Returns:
            _ActiveRunSnapshot: An immutable copy of the active state.
        """
        active_scenario_result_id = self._active_scenario_result_id
        current = self._active_tasks.get(active_scenario_result_id) if active_scenario_result_id is not None else None
        if current is not None and current.original_grant is not None:
            gateway = get_original_run_gateway()
            if (
                gateway is None
                or not gateway.authorized(operator=operator)
                or operator is None
                or operator.oid != current.original_grant.operator_oid
            ):
                active_scenario_result_id = None
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

    def _load_run_plan(self, *, scenario_result: ScenarioResult) -> ScenarioRunPlan | None:
        """
        Load a validated plan from scenario metadata.

        Returns:
            ScenarioRunPlan | None: The stored plan, or None for a legacy row.
        """
        metadata = getattr(scenario_result, "metadata", None)
        raw_plan = (metadata or {}).get(SCENARIO_RUN_PLAN_METADATA_KEY)
        if raw_plan is None:
            return None
        plan = ScenarioRunPlan.model_validate(raw_plan)
        return self._enrich_legacy_plan_techniques(plan=plan)

    def _enrich_legacy_plan_techniques(self, *, plan: ScenarioRunPlan) -> ScenarioRunPlan:
        """
        Add technique identity and metadata to plans stored before those fields existed.

        Returns:
            ScenarioRunPlan: The original plan or a copy with recovered technique metadata.
        """
        scenario_name = plan.scenario_registry_name
        if scenario_name is None or all(group.technique_name for group in plan.atomic_groups):
            return plan

        technique_summaries = self._get_scenario_technique_summaries(scenario_name=scenario_name)
        if not technique_summaries:
            return plan

        candidate_names = sorted(technique_summaries, key=len, reverse=True)
        enriched_groups: list[ScenarioRunPlanAtomicGroup] = []
        for group in plan.atomic_groups:
            technique_name = group.technique_name
            if technique_name is None:
                technique_name = next(
                    (
                        candidate
                        for candidate in candidate_names
                        if group.display_group == candidate
                        or group.atomic_attack_name == candidate
                        or group.atomic_attack_name.startswith(f"{candidate}_")
                        or group.atomic_attack_name.startswith(f"{candidate}__")
                    ),
                    None,
                )
            summary = technique_summaries.get(technique_name) if technique_name else None
            enriched_groups.append(
                group.model_copy(
                    update={
                        "technique_name": technique_name,
                        "description": group.description or (summary.description if summary else None),
                        "tags": group.tags or (list(summary.tags) if summary else []),
                    }
                )
            )
        return plan.model_copy(update={"atomic_groups": enriched_groups})

    def _get_scenario_technique_summaries(
        self,
        *,
        scenario_name: str,
    ) -> dict[str, ScenarioTechniqueSummary]:
        """
        Get cached technique metadata without materializing unrelated scenarios.

        Returns:
            dict[str, ScenarioTechniqueSummary]: Technique metadata keyed by name.
        """
        with self._technique_metadata_lock:
            cached = self._technique_metadata_cache.get(scenario_name)
            if cached is not None:
                return cached

            registry = ScenarioRegistry.get_registry_singleton()
            if scenario_name not in registry:
                summaries: dict[str, ScenarioTechniqueSummary] = {}
            else:
                scenario_class = registry.get_class(scenario_name)
                metadata = registry.get_class_metadata(scenario_class)
                summaries = {summary.name: summary for summary in metadata.technique_summaries}
            self._technique_metadata_cache[scenario_name] = summaries
            return summaries

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
        operator: AuthenticatedUser | None = None,
    ) -> ScenarioRunProgress | None:
        """Return compact database progress using a previously captured live-state snapshot."""
        header_result = self._memory.get_scenario_result_header(scenario_result_id=scenario_result_id)
        if header_result is None:
            return None
        self._reject_unbound_original_run(scenario_result=header_result)
        if OriginalRunBinding.METADATA_KEY in (getattr(header_result, "metadata", None) or {}):
            if since is not None:
                raise ValueError("Approved original runs have no PyRIT attack-attempt cursor.")
            return self._original_run_progress(
                scenario_result=header_result,
                operator=operator,
                queue_position=queue_position,
                active_scenario_result_id=active_scenario_result_id,
            )

        try:
            plan = self._load_run_plan(scenario_result=header_result)
        except (ValidationError, ValueError):
            logger.warning(
                "Scenario run %s has invalid persisted plan metadata; treating the plan as unavailable.",
                scenario_result_id,
            )
            plan = None
        plan_complete = plan is not None
        original_inspect_import = self.verify_original_inspect_import(
            memory=self._memory, scenario_result=header_result, plan=plan
        )
        completed_import_units = self._original_inspect_completed_units(plan=plan, imported=original_inspect_import)
        cursor = self._decode_progress_cursor(since=since, scenario_result_id=scenario_result_id)
        terminal = header_result.scenario_run_state in (
            ScenarioRunState.COMPLETED,
            ScenarioRunState.FAILED,
            ScenarioRunState.CANCELLED,
        )
        objective_scorer_identifier = header_result.objective_scorer_identifier
        if not isinstance(objective_scorer_identifier, ComponentIdentifier):
            objective_scorer_identifier = None
        progress_snapshot = self._progress_read_model.get_snapshot(
            scenario_result_id=scenario_result_id,
            plan=plan,
            plan_complete=plan_complete,
            active_group_ids=active_group_ids,
            terminal=terminal,
            objective_scorer_identifier=objective_scorer_identifier,
            completed_without_attack_result=completed_import_units,
        )
        overload_events: deque[Any] = deque(maxlen=_MAX_OVERLOAD_EVENTS)
        for delta in progress_snapshot.deltas:
            overload_events.extend(delta.retry_events)
        available = [
            (delta, result)
            for delta, result in zip(progress_snapshot.deltas, progress_snapshot.results, strict=True)
            if cursor is None
            or (delta.timestamp, uuid.UUID(delta.attack_result_id))
            > (cursor.timestamp, uuid.UUID(cursor.attack_result_id))
        ]
        page = available[:limit]
        deltas = [delta for delta, _ in page]
        results = [result for _, result in page]
        has_more = len(available) > limit
        response_plan = progress_snapshot.plan if since is None else None
        next_cursor = (
            self._encode_progress_cursor(scenario_result_id=scenario_result_id, delta=deltas[-1]) if deltas else since
        )
        scenario_identifier = header_result.scenario_identifier
        target, datasets_used, scenario_parameters = self._safe_run_metadata(scenario_identifier=scenario_identifier)
        if plan is not None:
            techniques_used = list(dict.fromkeys(group.display_group for group in plan.atomic_groups))
        else:
            techniques_used = self._identifier_techniques(scenario_identifier)
        return ScenarioRunProgress(
            run=ScenarioProgressHeader(
                scenario_result_id=scenario_result_id,
                scenario_name=header_result.scenario_name,
                scenario_registry_name=plan.scenario_registry_name if plan else None,
                scenario_version=header_result.scenario_version,
                status=header_result.scenario_run_state,
                created_at=header_result.creation_time,
                started_at=self._load_started_at(scenario_result=header_result),
                completed_at=header_result.completion_time if terminal else None,
                pyrit_version=header_result.pyrit_version,
                target=target,
                techniques_used=techniques_used,
                datasets_used=datasets_used,
                scenario_parameters=scenario_parameters,
                labels=header_result.labels,
                queue_position=queue_position,
                active_scenario_result_id=active_scenario_result_id,
                overload_summaries=self._build_overload_summaries(retry_events=overload_events),
                original_inspect_import=original_inspect_import,
                failure_reason=self._original_inspect_failure_reason(
                    scenario_name=header_result.scenario_name,
                    scenario_run_state=header_result.scenario_run_state,
                    error_message=header_result.error_message,
                ),
            ),
            plan=response_plan,
            results=results,
            summary=progress_snapshot.summary,
            next_cursor=next_cursor,
            has_more=has_more,
            plan_complete=plan_complete,
        )

    def _original_run_progress(
        self,
        *,
        scenario_result: ScenarioResult,
        operator: AuthenticatedUser | None,
        queue_position: int | None,
        active_scenario_result_id: str | None,
    ) -> ScenarioRunProgress:
        """
        Return a redacted one-case plan with source completion and no invented attack attempt.

        Returns:
            ScenarioRunProgress: Safe original-run coverage, grade and cleanup status.
        """
        summary = self._original_run_summary(
            scenario_result=scenario_result,
            operator=operator,
            queue_position=queue_position,
            active_scenario_result_id=active_scenario_result_id,
        )
        assert summary.original_run_admission is not None
        case_id = f"approved-case-{scenario_result.id.hex}"
        group_id = config_hash({"original_run": str(scenario_result.id), "schema": 1})
        objective = "Approved original case"
        plan = ScenarioRunPlan(
            scenario_registry_name=APPROVED_ORIGINAL_SCENARIO,
            atomic_groups=[
                ScenarioRunPlanAtomicGroup(
                    id=group_id,
                    atomic_attack_name="original_task",
                    display_group="Original Task",
                    technique_name="original_task",
                    technique_eval_hash=config_hash({"profile_ref": summary.original_run_admission.profile_ref}),
                    seed_group_ids=[case_id],
                )
            ],
            seed_groups=[
                ScenarioRunPlanSeedGroup(
                    id=case_id,
                    objective=objective,
                    objective_sha256=to_sha256(objective),
                )
            ],
        )
        counts = ScenarioProgressCounts(
            completed=summary.completed_attacks,
            planned=1,
            succeeded=0,
            success_percentage=None,
            errors=int(summary.status is ScenarioRunState.FAILED),
            retries=0,
        )
        group_status = (
            "COMPLETED"
            if summary.status is ScenarioRunState.COMPLETED
            else "RUNNING"
            if summary.status is ScenarioRunState.IN_PROGRESS
            else "INCOMPLETE"
            if summary.status in (ScenarioRunState.FAILED, ScenarioRunState.CANCELLED)
            else "PENDING"
        )
        return ScenarioRunProgress(
            run=ScenarioProgressHeader(
                scenario_result_id=summary.scenario_result_id,
                scenario_name=summary.scenario_name,
                scenario_registry_name=summary.scenario_registry_name,
                scenario_version=summary.scenario_version,
                status=summary.status,
                created_at=summary.created_at,
                started_at=summary.started_at,
                completed_at=summary.completed_at,
                pyrit_version=summary.pyrit_version,
                techniques_used=summary.techniques_used,
                scenario_parameters=summary.scenario_parameters,
                queue_position=summary.queue_position,
                active_scenario_result_id=summary.active_scenario_result_id,
                original_run_admission=summary.original_run_admission,
                original_source_result=summary.original_source_result,
                failure_reason=summary.error,
            ),
            plan=plan,
            results=[],
            summary=ScenarioProgressSummary(
                overall=counts,
                display_groups=[
                    ScenarioDisplayGroupProgress(
                        **counts.model_dump(),
                        id="original_task",
                        display_group="Original Task",
                        atomic_attack_names=["original_task"],
                        atomic_group_ids=[group_id],
                    )
                ],
                techniques=[
                    ScenarioTechniqueProgress(
                        **counts.model_dump(),
                        id="original_task",
                        display_group="Original Task",
                        atomic_attack_names=["original_task"],
                        atomic_group_ids=[group_id],
                    )
                ],
                seed_groups=[ScenarioSeedGroupProgress(**counts.model_dump(), id=case_id, objective=None)],
                atomic_groups=[
                    ScenarioAtomicGroupProgress(
                        **counts.model_dump(),
                        id=group_id,
                        atomic_attack_name="original_task",
                        display_group="Original Task",
                        status=group_status,
                    )
                ],
            ),
            plan_complete=True,
        )

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
    ) -> AttackResultKeysetCursor | None:
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
        return AttackResultKeysetCursor(timestamp=timestamp, attack_result_id=attack_result_id)

    def get_run_results(
        self, *, scenario_result_id: str, operator: AuthenticatedUser | None = None
    ) -> ScenarioResult | None:
        """
        Get the ScenarioResult for a completed scenario run.

        Args:
            scenario_result_id: The scenario result ID.
            operator: Authenticated operator for protected original runs.

        Returns:
            ScenarioResult if the run is completed and results exist, None if not found.

        Raises:
            ValueError: If the run is not in a completed state.
        """
        results = self._memory.get_scenario_results(scenario_result_ids=[scenario_result_id])
        if not results:
            return None

        scenario_result = results[0]
        self._reject_unbound_original_run(scenario_result=scenario_result)
        if OriginalRunBinding.METADATA_KEY in (getattr(scenario_result, "metadata", None) or {}):
            self._original_run_binding(scenario_result=scenario_result, operator=operator)
            raise ValueError("Raw original evidence is not available from the Scenario results API.")
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
