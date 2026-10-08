# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One harmless original Inspect binding and a separate exact-artifact canonical writer."""

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from pyrit.common.async_compatibility import run_legacy_sync_async
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter
from pyrit.executor.benchmark.inspect_original_runner import run_original_inert_eval_async
from pyrit.executor.jobs.local import LocalEvaluationJobPort
from pyrit.executor.jobs.port import (
    EvaluationJobError,
    EvaluationJobErrorCode,
    EvaluationJobRuntimeRegistry,
    EvaluationRuntimeArtifacts,
    EvaluationRuntimeCancelled,
    EvaluationRuntimeError,
)
from pyrit.memory import SQLiteMemory
from pyrit.models import EvalRunRef, ScoreStatus
from pyrit.models.evaluation_job import (
    EvaluationArtifact,
    EvaluationArtifactKind,
    EvaluationArtifactManifest,
    EvaluationArtifactMediaType,
    EvaluationCanonicalReceipt,
    EvaluationCleanupState,
    EvaluationJobRegistration,
    EvaluationJobRequest,
    EvaluationRuntimeKind,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.executor.benchmark.inspect_eval_source import ResolvedOriginalInspectTask
    from pyrit.executor.jobs.port import EvaluationRuntimeContext
    from pyrit.memory import MemoryInterface


def _scratch_memory(root: Path) -> SQLiteMemory:
    root.mkdir()
    # SQLiteMemory's singleton constructor must not return or replace canonical memory.
    memory = SQLiteMemory.__new__(SQLiteMemory)
    memory.__init__(db_path=str(root / "worker.sqlite"), _defer_initialization=True)
    memory.results_path = str(root)
    memory.disable_embedding()
    return memory


async def _dispose_memory_async(memory: MemoryInterface) -> None:
    task = asyncio.create_task(memory.dispose_engine_async())
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    task.result()


async def _create_memory_async(root: Path) -> SQLiteMemory:
    task = asyncio.create_task(asyncio.to_thread(_scratch_memory, root))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError as error:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
        memory = task.result()
        await _dispose_memory_async(memory)
        raise EvaluationRuntimeCancelled(EvaluationCleanupState.VERIFIED) from error


class PublicOriginalInspectJobRuntime:
    """Only the pinned public inert Task, without an external model or steering."""

    def __init__(self, source: ResolvedOriginalInspectTask) -> None:
        """Retain the server-resolved source; producers cannot install another Task."""
        self.source = source
        self.registration = EvaluationJobRegistration(
            runtime=EvaluationRuntimeKind.ORIGINAL_INSPECT,
            source=source.case.package,
            case_id=source.case.case_id,
            execution_profile_sha256=source.spec.spec_sha256,
            artifact_kind=EvaluationArtifactKind.INSPECT_EVAL,
        )

    @classmethod
    async def create_async(cls) -> PublicOriginalInspectJobRuntime:
        """
        Validate the existing public pin without executing its authored lifecycle.

        Returns:
            PublicOriginalInspectJobRuntime: The sole public model-free binding.
        """
        source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family="inspect_original_inert")
        return cls(source)

    def request(
        self, *, job_id: UUID | None = None, run_id: UUID | None = None, attempt_id: UUID | None = None
    ) -> EvaluationJobRequest:
        """
        Allocate a fresh invocation; initial input stays in the original source.

        Returns:
            EvaluationJobRequest: The exact registered identity with no controls.
        """
        return EvaluationJobRequest(
            job_id=job_id or uuid4(),
            run_id=run_id or uuid4(),
            attempt_id=attempt_id or uuid4(),
            runtime=self.registration.runtime,
            source=self.registration.source,
            case_id=self.registration.case_id,
            execution_profile_sha256=self.registration.execution_profile_sha256,
        )

    async def execute_async(self, *, context: EvaluationRuntimeContext) -> EvaluationRuntimeArtifacts:
        """
        Execute authored setup/solver/scorer/cleanup once in noncanonical scratch memory.

        Returns:
            EvaluationRuntimeArtifacts: Exact binary evidence after local runner/writer closure.

        Raises:
            EvaluationJobError: If the admitted source or archive identity differs.
            EvaluationRuntimeCancelled: If the owned harmless execution was joined after cancellation.
            EvaluationRuntimeError: If source execution failed after owned local closure.
        """
        if not self.registration.accepts(context.request):
            raise EvaluationJobError(EvaluationJobErrorCode.UNSUPPORTED_RUNTIME)
        await asyncio.to_thread(self.source.verify_unchanged)
        memory = await _create_memory_async(context.run_root / "scratch")
        try:
            await memory.initialize_async()
            log_dir = context.run_root / "logs"
            await run_legacy_sync_async(log_dir.mkdir)
            original = await run_original_inert_eval_async(
                memory=memory, log_dir=log_dir, run_instance_id=context.request.run_id
            )
            if original.case != self.source.case or original.run.spec != self.source.spec:
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            content = await asyncio.to_thread(original.archive_path.read_bytes)
            artifact = EvaluationArtifact(
                name="original.eval",
                kind=EvaluationArtifactKind.INSPECT_EVAL,
                media_type=EvaluationArtifactMediaType.INSPECT_EVAL,
                sha256=hashlib.sha256(content).hexdigest(),
                bytes=len(content),
            )
            if artifact.sha256 != original.imported.archive_sha256:
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            manifest = EvaluationArtifactManifest(
                request=context.request,
                request_sha256=context.request.request_sha256,
                fence_id=context.fence_id,
                artifacts=(artifact,),
            )
        except asyncio.CancelledError as error:
            await _dispose_memory_async(memory)
            # This proves owned local execution/storage drained, not a completed source scorer.
            raise EvaluationRuntimeCancelled(EvaluationCleanupState.VERIFIED) from error
        except Exception as error:
            await _dispose_memory_async(memory)
            code = error.code if isinstance(error, EvaluationJobError) else EvaluationJobErrorCode.RUNTIME_FAILED
            raise EvaluationRuntimeError(code=code, cleanup=EvaluationCleanupState.VERIFIED) from error
        except BaseException:
            await _dispose_memory_async(memory)
            raise
        await _dispose_memory_async(memory)
        return EvaluationRuntimeArtifacts(
            manifest=manifest,
            payloads=((artifact.name, content),),
            cleanup=EvaluationCleanupState.VERIFIED,
        )


class OriginalInspectArtifactWriter:
    """Canonical API-side typed import, never a worker database merge or scorer rerun."""

    def __init__(self, *, memory: MemoryInterface, runtime: PublicOriginalInspectJobRuntime) -> None:
        """Bind canonical memory and the same server-approved source identity."""
        self._memory = memory
        self._runtime = runtime

    async def import_async(
        self, *, request: EvaluationJobRequest, artifacts: EvaluationRuntimeArtifacts
    ) -> EvaluationCanonicalReceipt:
        """
        Import exact source bytes and verify linked canonical Score/AttackResult readback.

        Returns:
            EvaluationCanonicalReceipt: Actual source-attributed canonical references.

        Raises:
            EvaluationJobError: If evidence is foreign, incomplete, or canonical readback differs.
        """
        request = EvaluationJobRequest.model_validate(request)
        artifacts.verify()
        if (
            not self._runtime.registration.accepts(request)
            or artifacts.manifest.request != request
            or len(artifacts.manifest.artifacts) != 1
            or artifacts.manifest.artifacts[0].kind is not EvaluationArtifactKind.INSPECT_EVAL
            or artifacts.cleanup is not EvaluationCleanupState.VERIFIED
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        await asyncio.to_thread(self._runtime.source.verify_unchanged)
        artifact = artifacts.manifest.artifacts[0]
        imported = await InspectOriginalEvalImporter(memory=self._memory).import_eval_bytes_async(
            content=dict(artifacts.payloads)[artifact.name],
            cases=(self._runtime.source.case,),
            run=EvalRunRef(spec=self._runtime.source.spec, run_instance_id=request.run_id),
        )
        if imported.archive_sha256 != artifact.sha256:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        score_ids = tuple(UUID(str(result.score.id)) for result in imported.case_results)
        attack_ids = tuple(UUID(result.attack_result.attack_result_id) for result in imported.case_results)
        scores = await self._memory.get_scores_async(score_ids=[str(identifier) for identifier in score_ids])
        results = await self._memory.get_attack_results_async(
            attack_result_ids=[str(identifier) for identifier in attack_ids]
        )
        episode = await asyncio.to_thread(
            self._memory.native_cyber_evidence.get_episode, run_id=imported.episode.run.run_id
        )
        streams = tuple(
            stream for stream in episode.raw_streams if stream.key == InspectOriginalEvalImporter.ARCHIVE_KEY
        )
        if (
            not score_ids
            or {UUID(str(score.id)) for score in scores} != set(score_ids)
            or {UUID(result.attack_result_id) for result in results} != set(attack_ids)
            or len(streams) != 1
            or streams[0].observed_sha256 != artifact.sha256
            or streams[0].stored_sha256 != artifact.sha256
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        return EvaluationCanonicalReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            manifest_sha256=artifacts.manifest.manifest_sha256,
            artifact_sha256=artifact.sha256,
            projection_id=episode.run.run_id,
            source_complete=(
                episode.coverage_complete
                and imported.log_status == "success"
                and all(score.status is ScoreStatus.COMPLETE for score in scores)
            ),
            score_ids=score_ids,
            attack_result_ids=attack_ids,
        )


async def create_public_original_job_port_async(
    *, root: Path, memory: MemoryInterface, allowed_actor_ids: frozenset[str]
) -> LocalEvaluationJobPort:
    """
    Install the sole harmless binding without starting a consumer or any job.

    Returns:
        LocalEvaluationJobPort: An explicitly configured local port awaiting startup.
    """
    runtime = await PublicOriginalInspectJobRuntime.create_async()
    return LocalEvaluationJobPort(
        root=root,
        registry=EvaluationJobRuntimeRegistry((runtime,)),
        writers={EvaluationRuntimeKind.ORIGINAL_INSPECT: OriginalInspectArtifactWriter(memory=memory, runtime=runtime)},
        allowed_actor_ids=allowed_actor_ids,
    )
