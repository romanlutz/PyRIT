# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One harmless original Inspect binding and a separate exact-artifact canonical writer."""

from __future__ import annotations

import asyncio
import hashlib
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from inspect_ai.log import read_eval_log
from sqlalchemy.exc import SQLAlchemyError

from pyrit.common.async_compatibility import run_legacy_sync_async
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter, InspectOriginalScorePolicy
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
from pyrit.models import EvalCaseRef, EvalRunRef, EvalSpecRef, ScoreStatus
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
    from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalImport
    from pyrit.executor.jobs.port import EvaluationRuntimeContext
    from pyrit.memory import MemoryInterface


@dataclass(frozen=True, kw_only=True)
class ApprovedOriginalInspectBinding:
    """Server-reviewed metadata only, not a Task loader or executable source."""

    case: EvalCaseRef
    spec: EvalSpecRef
    score_policy: InspectOriginalScorePolicy

    def __post_init__(self) -> None:
        """
        Require one unchanged case and an explicit original scorer.

        Raises:
            ValueError: If source, case, profile or original-scorer policy is inconsistent.
        """
        case = EvalCaseRef.model_validate_json(self.case.model_dump_json())
        spec = EvalSpecRef.model_validate_json(self.spec.model_dump_json())
        if (
            case.package != spec.package
            or spec.input_variant is not None
            or not isinstance(self.score_policy, InspectOriginalScorePolicy)
            or self.score_policy.task_name != case.task_name
            or self.score_policy.task_version != case.task_version
            or self.score_policy.success_direction is not None
            or self.score_policy.success_threshold is not None
        ):
            raise ValueError("Approved original metadata requires one matching unchanged source and scorer policy.")
        object.__setattr__(self, "case", case)
        object.__setattr__(self, "spec", spec)

    @property
    def registration(self) -> EvaluationJobRegistration:
        """The sole original source/case/profile authorized by this server installation."""
        return EvaluationJobRegistration(
            runtime=EvaluationRuntimeKind.ORIGINAL_INSPECT,
            source=self.case.package,
            case_id=self.case.case_id,
            execution_profile_sha256=self.spec.spec_sha256,
            artifact_kind=EvaluationArtifactKind.INSPECT_EVAL,
        )


@dataclass(frozen=True, kw_only=True)
class OriginalInspectArtifactReadback:
    """Verified API-owned exact archive and its existing source-attributed canonical rows."""

    archive: bytes
    imported: InspectOriginalImport


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

    def __init__(
        self,
        *,
        memory: MemoryInterface,
        runtime: PublicOriginalInspectJobRuntime | None = None,
        binding: ApprovedOriginalInspectBinding | None = None,
    ) -> None:
        """
        Bind either the existing public runtime or explicit server-reviewed metadata.

        Raises:
            ValueError: If neither or both installation paths are supplied.
        """
        if (runtime is None) == (binding is None):
            raise ValueError("Select exactly one public original runtime or approved metadata binding.")
        self._memory = memory
        self._runtime = runtime
        if runtime is not None:
            self._case = runtime.source.case
            self._spec = runtime.source.spec
            self._score_policy: InspectOriginalScorePolicy | None = None
            self._registration = runtime.registration
        else:
            assert binding is not None
            self._case = binding.case
            self._spec = binding.spec
            self._score_policy = binding.score_policy
            self._registration = binding.registration

    @property
    def registration(self) -> EvaluationJobRegistration:
        """The source registration installed with this API-owned writer."""
        return self._registration

    @property
    def canonical_memory(self) -> MemoryInterface:
        """The API-owned canonical memory handle, never a worker-selected database."""
        return self._memory

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
            not self.registration.accepts(request)
            or artifacts.manifest.request != request
            or len(artifacts.manifest.artifacts) != 1
            or artifacts.manifest.artifacts[0].kind is not EvaluationArtifactKind.INSPECT_EVAL
            or artifacts.cleanup is not EvaluationCleanupState.VERIFIED
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        if self._runtime is not None:
            await asyncio.to_thread(self._runtime.source.verify_unchanged)
        artifact = artifacts.manifest.artifacts[0]
        imported = await InspectOriginalEvalImporter(memory=self._memory).import_eval_bytes_async(
            content=dict(artifacts.payloads)[artifact.name],
            cases=(self._case,),
            run=EvalRunRef(spec=self._spec, run_instance_id=request.run_id),
            score_policy=self._score_policy,
        )
        if imported.archive_sha256 != artifact.sha256:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        canonical = self._canonical_receipt(request=request, manifest=artifacts.manifest, imported=imported)
        await self.readback_async(request=request, manifest=artifacts.manifest, canonical=canonical)
        return canonical

    async def readback_async(
        self,
        *,
        request: EvaluationJobRequest,
        manifest: EvaluationArtifactManifest,
        canonical: EvaluationCanonicalReceipt,
    ) -> OriginalInspectArtifactReadback:
        """
        Verify retained archive, events and linked rows without importing or writing on a read.

        Returns:
            OriginalInspectArtifactReadback: Exact API-retained source and verified original results.

        Raises:
            EvaluationJobError: If any source, run, manifest, archive or canonical row differs.
        """
        try:
            request = EvaluationJobRequest.model_validate(request)
            manifest = EvaluationArtifactManifest.model_validate(manifest)
            canonical = EvaluationCanonicalReceipt.model_validate(canonical)
            if (
                not self.registration.accepts(request)
                or manifest.request != request
                or len(manifest.artifacts) != 1
                or manifest.artifacts[0].kind is not EvaluationArtifactKind.INSPECT_EVAL
                or canonical.job_id != request.job_id
                or canonical.request_sha256 != request.request_sha256
                or canonical.manifest_sha256 != manifest.manifest_sha256
                or canonical.artifact_sha256 != manifest.artifacts[0].sha256
            ):
                raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
            return await run_legacy_sync_async(self._readback, request=request, manifest=manifest, canonical=canonical)
        except (KeyError, TypeError, ValueError, SQLAlchemyError) as error:
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH) from error

    def _readback(
        self,
        *,
        request: EvaluationJobRequest,
        manifest: EvaluationArtifactManifest,
        canonical: EvaluationCanonicalReceipt,
    ) -> OriginalInspectArtifactReadback:
        importer = InspectOriginalEvalImporter(memory=self._memory)
        episode = self._memory.native_cyber_evidence.get_finalized_unscored_inspect_capture(
            run_id=canonical.projection_id
        )
        streams = tuple(stream for stream in episode.raw_streams if stream.key == importer.ARCHIVE_KEY)
        if len(streams) != 1:
            raise ValueError("Original canonical archive stream is absent or ambiguous.")
        artifact = manifest.artifacts[0]
        archive = self._read_archive(
            run_id=canonical.projection_id, stream_id=streams[0].stream_id, expected_bytes=artifact.bytes
        )
        if hashlib.sha256(archive).hexdigest() != artifact.sha256:
            raise ValueError("Original canonical archive differs from its admitted hash.")
        relogged = importer._validate_archive_bytes(content=archive)
        log = read_eval_log(io.BytesIO(archive), resolve_attachments="full", format="eval")
        run = EvalRunRef(spec=self._spec, run_instance_id=request.run_id)
        importer._validate_log(log=log, cases=(self._case,), run=run)
        importer._validate_score_policy(log=log, cases=(self._case,), score_policy=self._score_policy)
        resolved = log.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
        if len(resolved) > importer.MAX_RESOLVED_BYTES:
            raise ValueError("Resolved original archive exceeds the existing import bound.")
        imported = importer._existing_result(
            log=log,
            archive_sha=artifact.sha256,
            resolved=resolved,
            snapshot=episode,
            case_run_ids=importer._case_run_ids(log=log, cases=(self._case,), run=run),
            score_policy=self._score_policy,
            relogged_samples=relogged,
        )
        if canonical != self._canonical_receipt(request=request, manifest=manifest, imported=imported):
            raise ValueError("Original canonical receipt differs from retained source and linked rows.")
        return OriginalInspectArtifactReadback(archive=archive, imported=imported)

    def _read_archive(self, *, run_id: str, stream_id: UUID, expected_bytes: int) -> bytes:
        content = bytearray()
        after = 0
        store = self._memory.native_cyber_evidence
        while True:
            chunks = store.read_raw_chunks(
                run_id=run_id, stream_id=stream_id, allow_sensitive=True, after_sequence=after
            )
            for chunk in chunks:
                content.extend(chunk.data)
                if len(content) > expected_bytes:
                    raise ValueError("Original canonical archive exceeds its admitted byte bound.")
                after = chunk.sequence
            if not chunks:
                break
        if len(content) != expected_bytes:
            raise ValueError("Original canonical archive is truncated.")
        return bytes(content)

    @staticmethod
    def _canonical_receipt(
        *,
        request: EvaluationJobRequest,
        manifest: EvaluationArtifactManifest,
        imported: InspectOriginalImport,
    ) -> EvaluationCanonicalReceipt:
        score_ids = tuple(UUID(str(result.score.id)) for result in imported.case_results)
        attack_ids = tuple(UUID(result.attack_result.attack_result_id) for result in imported.case_results)
        episode = imported.episode
        streams = tuple(
            stream for stream in episode.raw_streams if stream.key == InspectOriginalEvalImporter.ARCHIVE_KEY
        )
        artifact = manifest.artifacts[0]
        if (
            not score_ids
            or len(streams) != 1
            or streams[0].observed_sha256 != artifact.sha256
            or streams[0].stored_sha256 != artifact.sha256
        ):
            raise EvaluationJobError(EvaluationJobErrorCode.ARTIFACT_MISMATCH)
        return EvaluationCanonicalReceipt(
            job_id=request.job_id,
            request_sha256=request.request_sha256,
            manifest_sha256=manifest.manifest_sha256,
            artifact_sha256=artifact.sha256,
            projection_id=episode.run.run_id,
            source_complete=(
                episode.coverage_complete
                and imported.log_status == "success"
                and all(result.score.status is ScoreStatus.COMPLETE for result in imported.case_results)
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
