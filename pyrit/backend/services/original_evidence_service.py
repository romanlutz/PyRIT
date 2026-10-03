# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Canonical backend-owned import of one authenticated original Inspect case."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import logging
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, ClassVar
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator
from sqlalchemy.exc import SQLAlchemyError

from pyrit.backend.services.original_evidence_admission import (
    OriginalEvidenceAdmission,
    OriginalEvidenceEnvelope,
    get_original_evidence_provider,
)
from pyrit.backend.services.original_run_admission import (
    OriginalAdmissionError,
    OriginalCleanupReceipt,
    OriginalRunBinding,
    OriginalWorkerJob,
)
from pyrit.models import (
    AttackOutcome,
    ScenarioExecutionOwner,
    ScenarioIdentifier,
    ScenarioResult,
    ScenarioRunState,
    config_hash,
)
from pyrit.models.catalog.scenario import OriginalRunReason, OriginalRunStatus, OriginalSourceResult

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

    from pyrit.backend.middleware.auth import AuthenticatedUser
    from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalImport
    from pyrit.memory import MemoryInterface

logger = logging.getLogger(__name__)


class OriginalEvidenceRecord(BaseModel):
    """Backend-only source references, separate from any scratch worker database."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    METADATA_KEY: ClassVar[str] = "approved_original_durable_import"
    PENDING_KEY: ClassVar[str] = "approved_original_intake_pending"

    envelope: OriginalEvidenceEnvelope
    envelope_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_episode_id: str | None = Field(default=None, pattern=r"^inspect-import-[0-9a-f]{64}$")
    score_id: UUID | None = None
    attack_result_id: UUID | None = None
    source_result: OriginalSourceResult
    persistence_verified: StrictBool

    @model_validator(mode="after")
    def _validate_references(self) -> OriginalEvidenceRecord:
        """
        Require a complete source pair without inventing rows for an ungraded failure.

        Returns:
            OriginalEvidenceRecord: Paired original-source references.
        """
        if (self.score_id is None) != (self.attack_result_id is None):
            raise ValueError("Original evidence requires paired canonical result references.")
        if (self.projection_episode_id is None) != (self.envelope.archive_sha256 is None):
            raise ValueError("An original archive requires its exact projection reference.")
        if self.score_id is not None and self.projection_episode_id is None:
            raise ValueError("A canonical original grade requires retained evidence.")
        return self

    @classmethod
    def from_metadata(cls, metadata: object) -> OriginalEvidenceRecord:
        """
        Decode persisted JSON using strict JSON UUID semantics, not Python coercion.

        Returns:
            OriginalEvidenceRecord: The validated internal source record.

        Raises:
            ValueError: If the stored record is not an object or contains invalid source fields.
        """
        if not isinstance(metadata, dict):
            raise ValueError("Original evidence metadata requires one JSON object.")
        return cls.model_validate_json(json.dumps(metadata))


class OriginalEvidenceReceipt(BaseModel):
    """Authenticated intake acknowledgment; no Task, scorer, credential or worker path."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    app_run_id: UUID
    job_ref: UUID
    envelope_sha256: str
    archive_sha256: str | None
    scenario_state: ScenarioRunState
    score_id: UUID | None
    attack_result_id: UUID | None
    source_result: OriginalSourceResult
    persistence_verified: StrictBool


@dataclass(frozen=True, kw_only=True)
class OriginalEvidenceReadback:
    """Exactly source-verified canonical backend rows and their authorized policy."""

    admission: OriginalEvidenceAdmission
    record: OriginalEvidenceRecord
    imported: InspectOriginalImport | None


class OriginalEvidenceService:
    """Serialize one-case intake into the configured backend-owned MemoryInterface."""

    def __init__(self, *, memory: MemoryInterface) -> None:
        """Use backend memory only; never open a worker SQLite or private runtime root."""
        self._memory = memory
        self._write_lock = asyncio.Lock()

    async def authorize_intake_async(
        self, *, capability: str, envelope: OriginalEvidenceEnvelope
    ) -> OriginalEvidenceAdmission:
        """
        Authenticate a short-lived server-issued upload capability before reading its body.

        Returns:
            OriginalEvidenceAdmission: A fixed source policy from the installed host adapter.

        Raises:
            OriginalAdmissionError: If the sender, source, profile or capability is unapproved.
        """
        provider = get_original_evidence_provider()
        if provider is None:
            raise OriginalAdmissionError(reason=OriginalRunReason.RUNNER_NOT_CONFIGURED)
        try:
            admission = await provider.authorize_intake_async(capability=capability, envelope=envelope)
        except OriginalAdmissionError:
            raise
        except Exception as error:
            logger.error("Original evidence authority failed (%s).", type(error).__name__)
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error
        try:
            self._validate_admission(admission=admission, envelope=envelope)
        except (AttributeError, TypeError, ValueError) as error:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error
        return admission

    async def intake_async(
        self, *, capability: str, envelope: OriginalEvidenceEnvelope, archive: bytes
    ) -> OriginalEvidenceReceipt:
        """
        Authenticate and import original bytes without trusting a worker grade or copied rows.

        Returns:
            OriginalEvidenceReceipt: Verified canonical IDs or a physically distinct ungraded failure.
        """
        admission = await self.authorize_intake_async(capability=capability, envelope=envelope)
        return await self.persist_async(admission=admission, archive=archive)

    async def persist_async(self, *, admission: OriginalEvidenceAdmission, archive: bytes) -> OriginalEvidenceReceipt:
        """
        Persist only the exact original evidence admitted by the host provider.

        Returns:
            OriginalEvidenceReceipt: A readback-verified durable association.

        Raises:
            OriginalAdmissionError: If closure, original bytes, source identity or persistence differ.
        """
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        envelope = admission.envelope
        self._validate_admission(admission=admission, envelope=envelope)
        provider = get_original_evidence_provider()
        if provider is None:
            raise OriginalAdmissionError(reason=OriginalRunReason.RUNNER_NOT_CONFIGURED)
        try:
            cleanup = await self._verify_cleanup_async(envelope=envelope)
            if cleanup != envelope.cleanup:
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            log = await asyncio.to_thread(self._validate_archive, admission=admission, archive=archive)
            graded = self._can_import_grade(envelope=envelope, log=log)
            async with self._write_lock:
                prior = await asyncio.to_thread(self._reserve, admission=admission)
                if prior is not None:
                    readback = await asyncio.to_thread(self._verify_record, admission=admission, scenario_result=prior)
                    return self._receipt(readback=readback)
                imported = (
                    await InspectOriginalEvalImporter(
                        memory=self._memory, capture_only=not graded
                    ).import_eval_bytes_async(
                        content=archive,
                        cases=admission.cases,
                        run=admission.run,
                        score_policy=admission.score_policy,
                    )
                    if archive
                    else None
                )
                source = self._source_result(admission=admission, imported=imported, graded=graded, log=log)
                record = OriginalEvidenceRecord(
                    envelope=envelope,
                    envelope_sha256=envelope.sha256,
                    projection_episode_id=imported.episode.run.run_id if imported is not None else None,
                    score_id=imported.case_results[0].score.id if graded and imported is not None else None,
                    attack_result_id=(
                        UUID(imported.case_results[0].attack_result.attack_result_id)
                        if graded and imported is not None
                        else None
                    ),
                    source_result=source,
                    persistence_verified=False,
                )
                await asyncio.to_thread(self._publish, record=record, terminal=False)
                header = await asyncio.to_thread(
                    self._memory.get_scenario_result_header, scenario_result_id=str(envelope.app_run_id)
                )
                if header is None:
                    raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
                await asyncio.to_thread(self._verify_record, admission=admission, scenario_result=header, pending=True)
                record = record.model_copy(update={"persistence_verified": True})
                await asyncio.to_thread(self._publish, record=record, terminal=True)
                header = await asyncio.to_thread(
                    self._memory.get_scenario_result_header, scenario_result_id=str(envelope.app_run_id)
                )
                if header is None:
                    raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
                readback = await asyncio.to_thread(self._verify_record, admission=admission, scenario_result=header)
                return self._receipt(readback=readback)
        except OriginalAdmissionError:
            raise
        except (KeyError, TypeError, ValueError, SQLAlchemyError) as error:
            logger.error("Original evidence intake failed (%s).", type(error).__name__)
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error

    def read(self, *, scenario_result: ScenarioResult, operator: AuthenticatedUser | None) -> OriginalEvidenceReadback:
        """
        Reauthorize one source-bound result before reading private evidence or returning messages.

        Returns:
            OriginalEvidenceReadback: The approved case, never a private source/runtime handle.

        Raises:
            OriginalAdmissionError: If the actor, source manifest or persisted evidence is invalid.
        """
        from pyrit.backend.middleware.auth import AuthenticatedUser

        provider = get_original_evidence_provider()
        if provider is None or not isinstance(operator, AuthenticatedUser):
            raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
        try:
            record = OriginalEvidenceRecord.from_metadata(
                scenario_result.metadata.get(OriginalEvidenceRecord.METADATA_KEY)
            )
            try:
                admission = provider.resolve_read(
                    operator=operator, job=record.envelope.job, envelope_sha256=record.envelope_sha256
                )
            except OriginalAdmissionError:
                raise
            except Exception as error:
                logger.error("Original evidence read authority failed (%s).", type(error).__name__)
                raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error
            if admission is None or operator.oid != record.envelope.operator_oid:
                raise OriginalAdmissionError(reason=OriginalRunReason.OPERATOR_NOT_AUTHORIZED)
            self._validate_admission(admission=admission, envelope=record.envelope)
            return self._verify_record(admission=admission, scenario_result=scenario_result)
        except OriginalAdmissionError:
            raise
        except (AttributeError, KeyError, TypeError, ValueError, SQLAlchemyError) as error:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED) from error

    @staticmethod
    async def _verify_cleanup_async(*, envelope: OriginalEvidenceEnvelope) -> OriginalCleanupReceipt | None:
        """
        Obtain independently authenticated physical closure without invoking destructive cleanup.

        Returns:
            OriginalCleanupReceipt | None: The observed physical receipt, if authenticated.
        """
        provider = get_original_evidence_provider()
        if provider is None:
            raise OriginalAdmissionError(reason=OriginalRunReason.RUNNER_NOT_CONFIGURED)
        try:
            return await provider.verify_cleanup_async(job=envelope.job, envelope_sha256=envelope.sha256)
        except OriginalAdmissionError:
            raise
        except Exception as error:
            logger.error("Original evidence closure authority failed (%s).", type(error).__name__)
            raise OriginalAdmissionError(reason=OriginalRunReason.PROVIDER_UNQUALIFIED) from error

    @staticmethod
    def _validate_admission(*, admission: OriginalEvidenceAdmission, envelope: OriginalEvidenceEnvelope) -> None:
        if (
            not isinstance(admission, OriginalEvidenceAdmission)
            or admission.envelope != envelope
            or len(admission.cases) != 1
            or admission.run.run_instance_id != envelope.run_instance_id
            or admission.run.spec.package.source_sha256 != envelope.source_sha256
            or admission.run.spec.input_variant is not None
            or admission.run.spec.model_route.config_sha256 != envelope.model_route_sha256
            or admission.cases[0].package != admission.run.spec.package
            or admission.run.case_run_id(case=admission.cases[0]) != envelope.case_run_id
            or admission.score_policy.task_name != admission.cases[0].task_name
            or admission.score_policy.task_version != admission.cases[0].task_version
            or admission.score_policy.success_threshold is not None
            or admission.score_policy.success_direction is not None
            or type(admission.display_values) is not frozenset
            or any(type(value) is not str or len(value) > 64 for value in admission.display_values)
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.PROFILE_NOT_ADMITTED)

    @staticmethod
    def _validate_archive(*, admission: OriginalEvidenceAdmission, archive: bytes) -> EvalLog | None:
        from inspect_ai.log import read_eval_log

        from pyrit.executor.benchmark.inspect_eval_projection import final_original_score_event
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        envelope = admission.envelope
        if (
            not isinstance(archive, bytes)
            or (envelope.archive_sha256 is None and archive)
            or (
                envelope.archive_sha256 is not None
                and (
                    len(archive) != envelope.archive_bytes
                    or hashlib.sha256(archive).hexdigest() != envelope.archive_sha256
                )
            )
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        if not archive:
            if envelope.source_state == "success" or envelope.source_complete:
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            return None
        if InspectOriginalEvalImporter._validate_archive_bytes(content=archive):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        log = read_eval_log(io.BytesIO(archive), resolve_attachments="full", format="eval")
        InspectOriginalEvalImporter._validate_log(log=log, cases=admission.cases, run=admission.run)
        InspectOriginalEvalImporter._validate_score_policy(
            log=log, cases=admission.cases, score_policy=admission.score_policy
        )
        if log.eval.run_id != envelope.inspect_run_id or log.eval.eval_id != envelope.inspect_eval_id:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        case = admission.cases[0]
        if len(log.samples or []) > 1 or any(
            str(sample.id) != case.sample_id or sample.epoch != case.epoch for sample in log.samples or []
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        if envelope.source_state == "success":
            samples = log.samples or []
            if log.status != "success" or log.error or log.invalidated or len(samples) != 1 or samples[0].error:
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            event = final_original_score_event(sample=samples[0], scorer_name=admission.score_policy.primary_scorer)
            if (
                event is None
                or event.uuid != envelope.final_score_event_id
                or config_hash({"event": event.model_dump(mode="json", exclude_none=True)})
                != envelope.final_score_event_sha256
            ):
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        return log

    @staticmethod
    def _can_import_grade(*, envelope: OriginalEvidenceEnvelope, log: EvalLog | None) -> bool:
        return (
            log is not None
            and envelope.source_state == "success"
            and envelope.source_complete
            and envelope.cleanup.proved
            and envelope.operation_terminal_receipt_id is not None
            and envelope.operation_terminal_sha256 is not None
        )

    def _reserve(self, *, admission: OriginalEvidenceAdmission) -> ScenarioResult | None:
        envelope = admission.envelope
        header = self._memory.get_scenario_result_header(scenario_result_id=str(envelope.app_run_id))
        if header is not None:
            if (
                header.scenario_name != "ServerApprovedOriginalScenario"
                or header.metadata.get(OriginalRunBinding.METADATA_KEY) != envelope.binding.model_dump(mode="json")
                or header.metadata.get(OriginalWorkerJob.METADATA_KEY) != envelope.job.model_dump(mode="json")
            ):
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            if OriginalEvidenceRecord.METADATA_KEY in header.metadata:
                return header
            if OriginalEvidenceRecord.PENDING_KEY in header.metadata or header.scenario_run_state in (
                ScenarioRunState.COMPLETED,
                ScenarioRunState.FAILED,
                ScenarioRunState.CANCELLED,
            ):
                raise OriginalAdmissionError(reason=OriginalRunReason.CAPACITY_BUSY)
            self._memory.update_scenario_metadata_fields(
                scenario_result_id=str(envelope.app_run_id),
                fields={OriginalEvidenceRecord.PENDING_KEY: envelope.sha256},
            )
        else:
            self._memory.add_scenario_results_to_memory(
                scenario_results=[
                    ScenarioResult(
                        id=envelope.app_run_id,
                        scenario_identifier=ScenarioIdentifier(
                            class_name="ServerApprovedOriginalScenario",
                            class_module="pyrit.backend.services.original_run_admission",
                            version=1,
                            techniques=[],
                            datasets=[],
                            params={
                                "execution_owner": ScenarioExecutionOwner.APPROVED_ORIGINAL.value,
                            },
                        ),
                        scenario_description="Approved original case evidence",
                        attack_results={},
                        scenario_run_state=ScenarioRunState.IN_PROGRESS,
                        metadata={
                            OriginalRunBinding.METADATA_KEY: envelope.binding.model_dump(mode="json"),
                            OriginalWorkerJob.METADATA_KEY: envelope.job.model_dump(mode="json"),
                            OriginalEvidenceRecord.PENDING_KEY: envelope.sha256,
                        },
                    )
                ]
            )
        return None

    @staticmethod
    def _source_result(
        *,
        admission: OriginalEvidenceAdmission,
        imported: InspectOriginalImport | None,
        graded: bool,
        log: EvalLog | None,
    ) -> OriginalSourceResult:
        from pyrit.executor.benchmark.inspect_eval_projection import final_original_score_event

        envelope = admission.envelope
        if graded and (imported is None or not imported.episode.coverage_complete or len(imported.case_results) != 1):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        score = imported.case_results[0].score if graded and imported is not None else None
        display_value = None
        if graded and log is not None and log.samples:
            event = final_original_score_event(sample=log.samples[0], scorer_name=admission.score_policy.primary_scorer)
            if event is not None and type(event.score.value) in (str, float):
                value = str(event.score.value)
                if value in admission.display_values:
                    display_value = value
        return OriginalSourceResult(
            profile_ref=envelope.profile_ref,
            status=(
                OriginalRunStatus.COMPLETED
                if graded
                else OriginalRunStatus.FAILED_UNGRADED
                if envelope.cleanup.proved
                else OriginalRunStatus.CLEANUP_UNCERTAIN
            ),
            source_state=envelope.source_state,
            source_coverage_complete=imported.episode.coverage_complete if imported is not None else False,
            original_score=display_value,
            original_score_available=graded,
            pyrit_score_status=score.status if score is not None else None,
            pyrit_outcome=AttackOutcome.UNDETERMINED if score is not None else None,
            cleanup_state=envelope.cleanup.state,
            reason=(
                None
                if graded
                else OriginalRunReason.SOURCE_UNVERIFIED
                if envelope.cleanup.proved
                else OriginalRunReason.CLEANUP_PENDING
            ),
        )

    def _publish(self, *, record: OriginalEvidenceRecord, terminal: bool) -> None:
        source = record.source_result
        updated = self._memory.try_update_scenario_run_state(
            scenario_result_id=str(record.envelope.app_run_id),
            expected_states={ScenarioRunState.CREATED, ScenarioRunState.QUEUED, ScenarioRunState.IN_PROGRESS},
            scenario_run_state=(
                ScenarioRunState.IN_PROGRESS
                if not terminal
                else ScenarioRunState.COMPLETED
                if source.status is OriginalRunStatus.COMPLETED
                else ScenarioRunState.CANCELLED
                if source.source_state == "cancelled" and record.envelope.cleanup.proved
                else ScenarioRunState.FAILED
            ),
            metadata_fields={
                OriginalEvidenceRecord.METADATA_KEY: record.model_dump(mode="json"),
                OriginalCleanupReceipt.METADATA_KEY: record.envelope.cleanup.model_dump(mode="json"),
                OriginalSourceResult.METADATA_KEY: source.model_dump(mode="json"),
            },
            error_message=source.reason.value if source.reason is not None else None,
        )
        if not updated:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)

    def _verify_record(
        self, *, admission: OriginalEvidenceAdmission, scenario_result: ScenarioResult, pending: bool = False
    ) -> OriginalEvidenceReadback:
        from pyrit.backend.services.scenario_run_service import ScenarioRunService
        from pyrit.executor.benchmark.inspect_eval_projection import InspectProjectionVersion, project_inspect_sample
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter

        record = OriginalEvidenceRecord.from_metadata(scenario_result.metadata.get(OriginalEvidenceRecord.METADATA_KEY))
        envelope = admission.envelope
        if (
            record.envelope != envelope
            or record.envelope_sha256 != envelope.sha256
            or scenario_result.id != envelope.app_run_id
            or scenario_result.scenario_name != "ServerApprovedOriginalScenario"
            or scenario_result.metadata.get(OriginalRunBinding.METADATA_KEY) != envelope.binding.model_dump(mode="json")
            or scenario_result.metadata.get(OriginalWorkerJob.METADATA_KEY) != envelope.job.model_dump(mode="json")
            or scenario_result.metadata.get(OriginalSourceResult.METADATA_KEY)
            != record.source_result.model_dump(mode="json")
            or scenario_result.metadata.get(OriginalCleanupReceipt.METADATA_KEY)
            != envelope.cleanup.model_dump(mode="json")
            or record.persistence_verified is not (not pending)
            or scenario_result.metadata.get(OriginalEvidenceRecord.PENDING_KEY) != envelope.sha256
        ):
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        imported = None
        log = None
        graded = record.score_id is not None
        if record.projection_episode_id is not None:
            episode = self._memory.native_cyber_evidence.get_finalized_unscored_inspect_capture(
                run_id=record.projection_episode_id
            )
            streams = [
                stream for stream in episode.raw_streams if stream.key == InspectOriginalEvalImporter.ARCHIVE_KEY
            ]
            if len(streams) != 1 or envelope.archive_bytes is None or envelope.archive_sha256 is None:
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            archive = ScenarioRunService._read_original_inspect_stream(
                memory=self._memory,
                run_id=record.projection_episode_id,
                stream_id=streams[0].stream_id,
                expected_bytes=envelope.archive_bytes,
                expected_sha256=envelope.archive_sha256,
            )
            log = self._validate_archive(admission=admission, archive=archive)
            assert log is not None
            if graded != self._can_import_grade(envelope=envelope, log=log):
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            case_run_ids = InspectOriginalEvalImporter._case_run_ids(log=log, cases=admission.cases, run=admission.run)
            identity = {
                "archive_sha256": envelope.archive_sha256,
                "case_run_ids": case_run_ids,
                "score_policy": asdict(admission.score_policy),
                "schema": InspectProjectionVersion.TOOL_CALLS.value,
                **({"capture_only": True} if not graded else {}),
            }
            if record.projection_episode_id != "inspect-import-" + config_hash(identity):
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            importer = InspectOriginalEvalImporter(memory=self._memory, capture_only=not graded)
            imported = importer._existing_result(
                log=log,
                archive_sha=envelope.archive_sha256,
                resolved=log.model_dump_json(exclude_none=True).encode("utf-8") + b"\n",
                snapshot=episode,
                case_run_ids=case_run_ids,
                score_policy=admission.score_policy,
                relogged_samples=False,
            )
            if graded and (
                len(imported.case_results) != 1
                or imported.case_results[0].score.id != record.score_id
                or UUID(imported.case_results[0].attack_result.attack_result_id) != record.attack_result_id
            ):
                raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            if record.attack_result_id is not None:
                links = self._memory.get_scenario_import_result_links(
                    scenario_name="ServerApprovedOriginalScenario",
                    metadata_key=OriginalEvidenceRecord.METADATA_KEY,
                    attack_result_ids=[str(record.attack_result_id)],
                )
                if (
                    set(links) != {str(record.attack_result_id)}
                    or links[str(record.attack_result_id)].id != envelope.app_run_id
                ):
                    raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
            for index, sample in enumerate(log.samples or [], start=1):
                conversation_id = importer._conversation_id(
                    episode_id=record.projection_episode_id,
                    run_id=log.eval.run_id,
                    sample=sample,
                    sample_index=index,
                )
                projection = project_inspect_sample(
                    sample=sample,
                    log_run_id=log.eval.run_id,
                    eval_id=log.eval.eval_id,
                    archive_sha256=envelope.archive_sha256,
                    sample_index=index,
                    start_sequence=1,
                    conversation_id=conversation_id,
                    case_run_id=envelope.case_run_id,
                )
                ScenarioRunService._verify_original_inspect_message_pieces(
                    memory=self._memory,
                    episode=episode,
                    conversation_id=conversation_id,
                    expected_pieces=projection.message_pieces,
                )
        elif envelope.archive_sha256 is not None or graded or record.attack_result_id is not None:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        expected = self._source_result(admission=admission, imported=imported, graded=graded, log=log)
        if record.source_result != expected:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        expected_state = (
            ScenarioRunState.IN_PROGRESS
            if pending
            else ScenarioRunState.COMPLETED
            if expected.status is OriginalRunStatus.COMPLETED
            else ScenarioRunState.CANCELLED
            if expected.source_state == "cancelled" and envelope.cleanup.proved
            else ScenarioRunState.FAILED
        )
        if scenario_result.scenario_run_state is not expected_state:
            raise OriginalAdmissionError(reason=OriginalRunReason.SOURCE_UNVERIFIED)
        return OriginalEvidenceReadback(admission=admission, record=record, imported=imported)

    @staticmethod
    def _receipt(*, readback: OriginalEvidenceReadback) -> OriginalEvidenceReceipt:
        record = readback.record
        source = record.source_result
        state = (
            ScenarioRunState.COMPLETED
            if source.status is OriginalRunStatus.COMPLETED
            else ScenarioRunState.CANCELLED
            if source.source_state == "cancelled" and record.envelope.cleanup.proved
            else ScenarioRunState.FAILED
        )
        return OriginalEvidenceReceipt(
            app_run_id=record.envelope.app_run_id,
            job_ref=record.envelope.job_ref,
            envelope_sha256=record.envelope_sha256,
            archive_sha256=record.envelope.archive_sha256,
            scenario_state=state,
            score_id=record.score_id,
            attack_result_id=record.attack_result_id,
            source_result=source,
            persistence_verified=record.persistence_verified,
        )


@lru_cache(maxsize=1)
def get_original_evidence_service() -> OriginalEvidenceService:
    """
    Resolve only the configured backend memory, never an upload-selected database.

    Returns:
        OriginalEvidenceService: The backend's original source writer and read verifier.
    """
    from pyrit.memory import CentralMemory

    return OriginalEvidenceService(memory=CentralMemory.get_memory_instance())
