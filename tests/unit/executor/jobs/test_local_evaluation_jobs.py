# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real local queue/source/archive/canonical SQLite proof and refusal boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import io
import threading
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from inspect_ai import Task, eval_async
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.solver import solver

from pyrit.executor.benchmark import inspect_original_runner
from pyrit.executor.jobs import inspect as job_inspect
from pyrit.executor.jobs.inspect import (
    OriginalInspectArtifactWriter,
    PublicOriginalInspectJobRuntime,
)
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import (
    EvaluationJobRuntimeRegistry,
    EvaluationRuntimeArtifacts,
    EvaluationRuntimeCancelled,
)
from pyrit.memory import SQLiteMemory
from pyrit.models import AttackOutcome, ScoreStatus
from pyrit.models.evaluation_job import (
    EvaluationCleanupState,
    EvaluationDeliveryState,
    EvaluationJobDelivery,
    EvaluationJobEventKind,
    EvaluationJobState,
    EvaluationRuntimeKind,
)

pytestmark = pytest.mark.filterwarnings(r"ignore:MemoryInterface\.:DeprecationWarning")

if TYPE_CHECKING:
    from pathlib import Path

    from inspect_ai.solver import Generate, Solver, TaskState

    from pyrit.executor.jobs.port import EvaluationRuntimeContext


class _TrackedOriginalRuntime:
    def __init__(self, runtime: PublicOriginalInspectJobRuntime) -> None:
        self.runtime = runtime
        self.registration = runtime.registration
        self.executions = 0
        self.failure: Exception | None = None

    async def execute_async(self, *, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        self.executions += 1
        try:
            return await self.runtime.execute_async(context=context)
        except Exception as error:
            self.failure = error
            raise


async def test_original_queue_imports_exact_source_archive_and_canonical_linked_results_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    runtime = await PublicOriginalInspectJobRuntime.create_async()
    tracked = _TrackedOriginalRuntime(runtime)
    port = LocalEvaluationJobPort(
        root=tmp_path / "queue",
        registry=EvaluationJobRuntimeRegistry((tracked,)),
        writers={
            EvaluationRuntimeKind.ORIGINAL_INSPECT: OriginalInspectArtifactWriter(
                memory=sqlite_instance, runtime=runtime
            )
        },
        allowed_actor_ids=frozenset({"operator"}),
    )
    request = runtime.request()
    await port.startup_async()
    try:
        submitted = await port.submit_async(request=request, actor_id="operator")
        assert not submitted.duplicate and submitted.control_capability is None
        queued = await port.status_async(job_id=request.job_id, actor_id="operator")
        assert queued.state is EvaluationJobState.QUEUED and queued.canonical is None
        assert (await port.submit_async(request=request, actor_id="operator")).duplicate
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id="operator", timeout_seconds=30)
        if tracked.failure is not None:
            raise tracked.failure
        assert terminal.state is EvaluationJobState.SUCCEEDED, terminal.reason
        assert terminal.canonical is not None and terminal.manifest is not None
        receipt = terminal.canonical
        assert receipt.source_complete
        scores = await sqlite_instance.get_scores_async(score_ids=[str(item) for item in receipt.score_ids])
        attacks = await sqlite_instance.get_attack_results_async(
            attack_result_ids=[str(item) for item in receipt.attack_result_ids]
        )
        assert len(scores) == len(attacks) == 1
        assert scores[0].score_value == "1.0" and scores[0].status is ScoreStatus.COMPLETE
        assert attacks[0].automated_score and attacks[0].automated_score.id == scores[0].id
        assert attacks[0].outcome is AttackOutcome.UNDETERMINED
        episode = await asyncio.to_thread(
            sqlite_instance.native_cyber_evidence.get_episode, run_id=receipt.projection_id
        )
        stream = next(
            item for item in episode.raw_streams if item.key.observed_source_id == "inspect-original-eval-archive"
        )
        chunks = await asyncio.to_thread(
            sqlite_instance.native_cyber_evidence.read_raw_chunks,
            run_id=receipt.projection_id,
            stream_id=stream.stream_id,
            allow_sensitive=True,
        )
        archive = b"".join(chunk.data for chunk in chunks)
        assert hashlib.sha256(archive).hexdigest() == receipt.artifact_sha256
        typed = await asyncio.to_thread(read_eval_log, io.BytesIO(archive), format="eval")
        assert typed.samples and typed.samples[0].store["lifecycle"] == ["setup", "solver", "score", "cleanup"]
        assert typed.samples[0].input == "harmless fixture" and not typed.samples[0].model_usage
        assert not typed.samples[0].error and typed.status == "success"
        assert terminal.events[-1].kind is EvaluationJobEventKind.TERMINAL
        assert [event.sequence for event in terminal.events] == list(range(1, terminal.last_sequence + 1))
        assert [event.kind for event in terminal.events].count(EvaluationJobEventKind.STARTED) == 1
        duplicate = await port.receive_async(
            delivery=EvaluationJobDelivery(job_id=request.job_id, request_sha256=request.request_sha256)
        )
        assert duplicate.state is EvaluationDeliveryState.DUPLICATE
        assert (await port.submit_async(request=request, actor_id="operator")).duplicate
        assert len(await sqlite_instance.get_scores_async(score_ids=[str(item) for item in receipt.score_ids])) == 1
        assert (
            len(
                await sqlite_instance.get_attack_results_async(
                    attack_result_ids=[str(item) for item in receipt.attack_result_ids]
                )
            )
            == 1
        )
        assert tracked.executions == 1
        empty = await port.status_async(
            job_id=request.job_id, actor_id="operator", after_sequence=terminal.last_sequence
        )
        assert empty.events == ()
        fresh = runtime.request(job_id=uuid4(), run_id=uuid4())
        assert fresh.case_run_sha256 != request.case_run_sha256
    finally:
        await port.shutdown_async()


async def test_cancelled_scratch_construction_disposes_the_eventually_created_memory_async(tmp_path: Path) -> None:
    entered, release = threading.Event(), threading.Event()
    memory = MagicMock(spec=SQLiteMemory)

    def held_constructor(root: Path) -> SQLiteMemory:
        entered.set()
        assert release.wait(5)
        assert isinstance(memory, SQLiteMemory)
        return memory

    with patch.object(job_inspect, "_scratch_memory", side_effect=held_constructor):
        caller = asyncio.create_task(job_inspect._create_memory_async(tmp_path / "scratch"))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            caller.cancel()
            await asyncio.sleep(0.02)
            caller.cancel()
            await asyncio.sleep(0.02)
            assert not caller.done()
            memory.dispose_engine_async.assert_not_awaited()
            release.set()
            with pytest.raises(EvaluationRuntimeCancelled) as cancelled:
                await caller
            assert cancelled.value.cleanup is EvaluationCleanupState.VERIFIED
            memory.dispose_engine_async.assert_awaited_once()
        finally:
            release.set()
            await asyncio.gather(caller, return_exceptions=True)


async def test_cancel_at_instrumented_real_harmless_solver_joins_then_fresh_original_job_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    runtime = await PublicOriginalInspectJobRuntime.create_async()
    port = LocalEvaluationJobPort(
        root=tmp_path / "queue",
        registry=EvaluationJobRuntimeRegistry((runtime,)),
        writers={
            EvaluationRuntimeKind.ORIGINAL_INSPECT: OriginalInspectArtifactWriter(
                memory=sqlite_instance, runtime=runtime
            )
        },
        allowed_actor_ids=frozenset({"operator"}),
    )
    started = asyncio.Event()
    stopped = asyncio.Event()

    @solver
    def public_queue_cancel_hold() -> Solver:
        async def held_solver_async(state: TaskState, generate: Generate) -> TaskState:
            assert state.store.get("lifecycle") == ["setup"]
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("The cancellation-only instrumented solver must not produce a grade.")

        return held_solver_async

    async def instrumented_eval_async(*, tasks: Task, log_dir: str, log_format: str) -> list[EvalLog]:
        original_solver = tasks.solver
        tasks.solver = public_queue_cancel_hold()
        try:
            return await eval_async(tasks=tasks, log_dir=log_dir, log_format="eval", log_realtime=False)
        finally:
            tasks.solver = original_solver

    await port.startup_async()
    try:
        cancelled_request = runtime.request()
        with patch.object(inspect_original_runner, "eval_async", side_effect=instrumented_eval_async):
            await port.submit_async(request=cancelled_request, actor_id="operator")
            port.start_consumer()
            await asyncio.wait_for(started.wait(), timeout=15)
            await port.cancel_async(job_id=cancelled_request.job_id, actor_id="operator")
            cancelled = await port.wait_async(job_id=cancelled_request.job_id, actor_id="operator", timeout_seconds=15)
        assert stopped.is_set()
        assert cancelled.state is EvaluationJobState.CANCELLED
        assert cancelled.canonical is None and cancelled.cleanup.value == "verified"
        assert [event.kind for event in cancelled.events].count(EvaluationJobEventKind.TERMINAL) == 1
        fresh = runtime.request()
        await port.submit_async(request=fresh, actor_id="operator")
        terminal = await port.wait_async(job_id=fresh.job_id, actor_id="operator", timeout_seconds=30)
        assert terminal.state is EvaluationJobState.SUCCEEDED and terminal.canonical
        assert terminal.request.run_id != cancelled.request.run_id
        scores = await sqlite_instance.get_scores_async(score_ids=[str(item) for item in terminal.canonical.score_ids])
        assert len(scores) == 1 and scores[0].status is ScoreStatus.COMPLETE
    finally:
        await port.shutdown_async()
