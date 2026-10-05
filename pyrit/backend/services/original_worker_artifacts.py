# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded validation of fixed source artifacts, without reading worker database rows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import JsonValue, TypeAdapter

from pyrit.backend.models.original_worker import (
    ORIGINAL_WORKER_ARTIFACTS,
    CohostWorkerProvenance,
)
from pyrit.backend.services.original_evidence_admission import OriginalEvidenceAdmission, OriginalEvidenceEnvelope
from pyrit.backend.services.original_worker_preflight import CohostPreflight, CohostPreflightError
from pyrit.models import EvalRunRef, EvalSpecRef, HarnessProfileRef, config_hash

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.backend.models.original_worker import CohostBackendConfig, CohostWorkerRequest, CohostWorkerTerminal


@dataclass(frozen=True, kw_only=True)
class OriginalWorkerArtifacts:
    """One exact archive and its authenticated source/policy, never copied worker rows."""

    admission: OriginalEvidenceAdmission
    archive: bytes
    provenance: CohostWorkerProvenance
    closure: dict[str, JsonValue]

    @classmethod
    def read(
        cls,
        *,
        config: CohostBackendConfig,
        request: CohostWorkerRequest,
        terminal: CohostWorkerTerminal,
        process_id: int,
    ) -> OriginalWorkerArtifacts:
        """
        Validate exact fixed filenames after actual child exit.

        Returns:
            OriginalWorkerArtifacts: A source-bound original archive and server-owned score policy.
        """
        root = request.run_root
        provenance = CohostWorkerProvenance.model_validate_json(
            cls._read_file(root=root, name="worker-provenance.json", limit=65_536)
        )
        expected = {
            "app_run_id": request.app_run_id,
            "job_ref": request.job_ref,
            "control_run_instance_id": request.run_instance_id,
            "operator_oid": request.operator_oid,
            "profile_ref": request.profile_ref,
            "source_alias": request.source_alias,
            "source_sha256": request.source_sha256,
            "manifest_sha256": request.manifest_sha256,
            "public_commit": config.public_commit,
            "state": terminal.state,
        }
        if (
            any(getattr(provenance, key) != value for key, value in expected.items())
            or provenance.process_identity.get("pid") != process_id
            or type(provenance.process_identity.get("pid")) is not int
            or provenance.framework_run_instance_id in (request.run_instance_id, request.app_run_id, request.job_ref)
            or provenance.model_route != config.relay.route_metadata
        ):
            raise CohostPreflightError("Original worker provenance does not bind its owned process/run/source.")
        blobs: dict[str, bytes] = {}
        for name, digest in provenance.artifacts.items():
            blobs[name] = cls._read_file(root=root, name=name, limit=digest.bytes)
            if len(blobs[name]) != digest.bytes or hashlib.sha256(blobs[name]).hexdigest() != digest.sha256:
                raise CohostPreflightError("Original fixed artifact bytes differ from worker provenance.")
        envelope = OriginalEvidenceEnvelope.model_validate_json(blobs["backend-intake-envelope.json"])
        if (
            envelope.app_run_id != request.app_run_id
            or envelope.job_ref != request.job_ref
            or envelope.operator_oid != request.operator_oid
            or envelope.profile_ref != request.profile_ref
            or envelope.source_sha256 != request.source_sha256
            or envelope.manifest_sha256 != request.manifest_sha256
            or envelope.run_instance_id != provenance.framework_run_instance_id
            or envelope.source_state != terminal.state
            or envelope.model_route_sha256 != config.source.spec.model_route.config_sha256
        ):
            raise CohostPreflightError("Original envelope is not the exact owned actor/framework/source binding.")
        snapshot = cls._object(blobs["worker-scenario.json"])
        metadata = snapshot.get("metadata")
        if (
            snapshot.get("id") != str(envelope.worker_scenario_id)
            or snapshot.get("authority") != "transient-worker-scenario-provenance-only"
            or not isinstance(metadata, dict)
            or metadata.get("run_instance_id") != str(provenance.framework_run_instance_id)
            or hashlib.sha256(blobs["worker-scenario.json"]).hexdigest() != envelope.worker_scenario_sha256
        ):
            raise CohostPreflightError("Original source Scenario snapshot differs from the framework run.")
        manifest = cls._object(blobs["source-manifest.json"])
        if (
            not config.local_test
            and config_hash({"source": manifest.get("private_source_pins"), "assets": manifest.get("selected_assets")})
            != request.source_sha256
        ):
            raise CohostPreflightError("Original unchanged source/assets differ from the approved source digest.")
        closure = cls._object(blobs["runner-closure.json"])
        cls.validate_terminal(config=config, request=request, envelope=envelope, closure=closure)
        from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalScorePolicy

        spec = cls.run_spec(config=config, envelope=envelope)
        run = EvalRunRef(spec=spec, run_instance_id=provenance.framework_run_instance_id)
        case = config.source.cases[0]
        if envelope.case_run_id != run.case_run_id(case=case):
            raise CohostPreflightError("Original case-run identity differs from the exact admitted source.")
        return cls(
            admission=OriginalEvidenceAdmission(
                envelope=envelope,
                run=run,
                cases=config.source.cases,
                score_policy=InspectOriginalScorePolicy(
                    task_name=case.task_name,
                    task_version=case.task_version,
                    primary_scorer=config.source.primary_scorer,
                ),
                display_values=config.source.display_values,
            ),
            archive=blobs.get("source.eval", b""),
            provenance=provenance,
            closure=closure,
        )

    @staticmethod
    def server_manifest(
        *,
        app_run_id: str,
        job_ref: str,
        operator_oid: str,
        profile_ref: str,
        config: CohostBackendConfig,
    ) -> dict[str, JsonValue]:
        """
        Construct the exact bare five-field immutable child manifest.

        Returns:
            dict[str, JsonValue]: No capability, source path, Python or credential material.
        """
        return {
            "app_run_id": app_run_id,
            "job_ref": job_ref,
            "operator_oid": operator_oid,
            "profile_ref": profile_ref,
            "model_route": config.relay.route_metadata,
        }

    @classmethod
    def run_spec(cls, *, config: CohostBackendConfig, envelope: OriginalEvidenceEnvelope) -> EvalSpecRef:
        """
        Resolve only a source/profile template with this backend-issued job binding.

        Returns:
            EvalSpecRef: Original unchanged case policy with its concrete manifest identity.
        """
        manifest = cls.server_manifest(
            app_run_id=str(envelope.app_run_id),
            job_ref=str(envelope.job_ref),
            operator_oid=envelope.operator_oid,
            profile_ref=envelope.profile_ref,
            config=config,
        )
        digest = config_hash(manifest)
        if digest != envelope.manifest_sha256:
            raise CohostPreflightError("Original manifest is not the exact server-issued actor/job/model binding.")
        if config.local_test:
            return config.source.spec
        return config.source.spec.model_copy(
            update={
                "harness": HarnessProfileRef(
                    name=config.source.spec.harness.name, config_sha256=config_hash(config.harness_metadata_for(digest))
                )
            }
        )

    @classmethod
    def read_closure(cls, root: Path) -> dict[str, JsonValue]:
        """
        Read only the bounded fixed cleanup packet.

        Returns:
            dict[str, JsonValue]: Worker claims that still require independent native observation.
        """
        return cls._object(cls._read_file(root=root, name="runner-closure.json", limit=1024 * 1024))

    @staticmethod
    def validate_terminal(
        *,
        config: CohostBackendConfig,
        request: CohostWorkerRequest,
        envelope: OriginalEvidenceEnvelope,
        closure: dict[str, JsonValue],
    ) -> None:
        """Verify separate scoring/operation completion and SDK drain, never infer them from404."""
        operation = closure.get("operation_terminal")
        lease = closure.get("lease_closure")
        drain = closure.get("sdk_thread_drain")
        if (
            not isinstance(operation, dict)
            or not isinstance(lease, dict)
            or not isinstance(drain, dict)
            or closure.get("cleanup_errors") != []
            or drain.get("sdk_calls_drained") is not True
            or drain.get("active_sdk_calls") != 0
            or type(drain.get("active_sdk_calls")) is not int
            or operation.get("schema") != "cohost-original-operation-terminal/v1"
            or operation.get("run_id") != str(request.run_instance_id)
            or config_hash(operation) != envelope.operation_terminal_sha256
            or envelope.operation_terminal_receipt_id != f"cohost-terminal-{request.run_instance_id}"
            or (envelope.cleanup.proved and config_hash(lease) != envelope.cleanup_sha256)
        ):
            raise CohostPreflightError("Original operation/SDK terminal receipt is incomplete or mismatched.")
        requested, terminal = operation.get("requested"), operation.get("terminal")
        if not isinstance(requested, list) or not isinstance(terminal, list):
            raise CohostPreflightError("Original operation terminal coverage is missing.")
        expected: dict[str, str] = {}
        observed: dict[str, str] = {}
        for rows, destination, is_terminal in ((requested, expected, False), (terminal, observed, True)):
            for row in rows:
                if not isinstance(row, dict):
                    raise CohostPreflightError("Original operation terminal row is invalid.")
                identifier, action = row.get("operation_id"), row.get("action")
                if (
                    not isinstance(identifier, str)
                    or not identifier
                    or identifier in destination
                    or not isinstance(action, str)
                    or not action
                    or (is_terminal and (row.get("terminal") is not True or not isinstance(row.get("status"), str)))
                    or (
                        is_terminal
                        and row.get("status") in ("aborted", "cancelled")
                        and row.get("abort_proof") is not True
                    )
                ):
                    raise CohostPreflightError("Original operation terminal coverage is partial or conflicting.")
                destination[identifier] = action
        if expected != observed:
            raise CohostPreflightError("Original operations are not exactly terminal before cleanup.")
        if not config.local_test and envelope.source_state == "success":
            admission = operation.get("original_admission")
            remaining = admission.get("active_remaining_seconds") if isinstance(admission, dict) else None
            if (
                not isinstance(admission, dict)
                or not isinstance(remaining, (int, float))
                or isinstance(remaining, bool)
                or remaining < 900
                or operation.get("allocation_count") != 1
                or operation.get("max_inflight") != 1
                or operation.get("no_create_retry") is not True
            ):
                raise CohostPreflightError("Original source was not admitted on the one-sandbox shared active clock.")

    @staticmethod
    def _read_file(*, root: Path, name: str, limit: int) -> bytes:
        if name not in ORIGINAL_WORKER_ARTIFACTS:
            raise CohostPreflightError("Original artifacts cannot choose arbitrary files.")
        path = root / name
        if path.is_symlink() or path.resolve() != path:
            raise CohostPreflightError("Original evidence artifacts cannot redirect outside their owned root.")
        return CohostPreflight._read_bounded(path=path, limit=limit)

    @staticmethod
    def _object(content: bytes) -> dict[str, JsonValue]:
        return TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
