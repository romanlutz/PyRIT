# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run one trusted benign Inspect Task and link its existing original Score."""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from pyrit.executor.benchmark._inspect_ghcp_runtime import InspectGhcpLimits
from pyrit.executor.benchmark.inspect_ghcp_eval import InspectGhcpEvaluation, InspectGhcpOutcome, InspectGhcpTaskBinding
from pyrit.executor.benchmark.inspect_ghcp_protocol import (
    InspectGhcpProtocolPins,
    build_benign_red_teaming_attack,
    create_benign_inspect_model,
)
from pyrit.memory import CentralMemory
from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.models import (
    AttackOutcome,
    CommittedCaseExecution,
    EvalCaseRef,
    EvalRunRef,
    EvalScoreProvenance,
    EvalScoreRole,
    HarnessProfileRef,
    ModelRouteRef,
    ScoreStatus,
    config_hash,
)
from pyrit.models.inspect_ghcp import InspectGhcpStatus, InspectGhcpTaskKind

if TYPE_CHECKING:
    from pyrit.executor.benchmark.inspect_eval_source import ResolvedInspectEvalCase
    from pyrit.memory import MemoryInterface


@dataclass(frozen=True, kw_only=True)
class InspectGhcpPilotEnvironment:
    """Public benign harness settings; no model credential or Docker identity is copied to a guest."""

    agent_image: str
    agent_image_id: str
    target_image: str
    target_image_id: str

    @classmethod
    def from_environment(cls) -> InspectGhcpPilotEnvironment:
        """
        Resolve only the prebuilt images reviewed for this benign pilot.

        Returns:
            InspectGhcpPilotEnvironment: Pinned image tags and full local IDs.

        Raises:
            ValueError: If an image or its immutable ID differs from the reviewed pilot.
        """
        config = cls(
            agent_image=os.environ.get("PYRIT_INSPECT_AGENT_IMAGE", ""),
            agent_image_id=os.environ.get("PYRIT_INSPECT_AGENT_IMAGE_ID", ""),
            target_image=os.environ.get("PYRIT_INSPECT_TARGET_IMAGE", "pyrit-ghcp-agent:1.0.88-ca"),
            target_image_id=os.environ.get("PYRIT_INSPECT_TARGET_IMAGE_ID", InspectGhcpProtocolPins.TARGET_IMAGE_ID),
        )
        if (
            config.agent_image != "pyrit-inspect-ghcp-guest:1.0.88-sdk1.0.14"
            or config.target_image != "pyrit-ghcp-agent:1.0.88-ca"
            or re.fullmatch(r"sha256:[0-9a-f]{64}", config.agent_image_id) is None
            or config.target_image_id != InspectGhcpProtocolPins.TARGET_IMAGE_ID
        ):
            raise ValueError("The selected benign GHCP pilot images or full IDs are not approved.")
        return config

    @property
    def image_ids(self) -> dict[str, str]:
        """Immutable local Docker IDs for all three original Inspect services."""
        return {
            "agent": self.agent_image_id,
            "model-bridge": self.agent_image_id,
            "target": self.target_image_id,
        }

    def harness_ref(self, *, sandbox_sha256: str) -> HarnessProfileRef:
        """
        Bind the reviewed GHCP profile and effective Compose config, independently of Eval source.

        Returns:
            HarnessProfileRef: No credentials or private source path.
        """
        limits = InspectGhcpLimits(
            min_turns=2, min_tool_executions=1, max_turns=2, max_model_requests=12, run_timeout_seconds=240
        )
        digest = config_hash(
            {
                "profile": "ghcp_protocol_v1",
                "images": self.image_ids,
                "agent_image": self.agent_image,
                "target_image": self.target_image,
                "cli_sha256": InspectGhcpProtocolPins.GHCP_CLI_SHA256,
                "limits": asdict(limits),
                "allowed_tools": ["bash"],
                "cache_policy": "omit_after_capture",
                "effective_compose": sandbox_sha256,
            }
        )
        return HarnessProfileRef(name="ghcp_protocol_v1", config_sha256=digest)

    @staticmethod
    def model_route_ref() -> ModelRouteRef:
        """
        Bind the host-only model route separately from package source and harness settings.

        Returns:
            ModelRouteRef: Immutable public route descriptor without authentication material.
        """
        digest = config_hash(
            {
                "route": "qwen3_loopback_v1",
                "endpoint": InspectGhcpProtocolPins.MODEL_ENDPOINT,
                "model": InspectGhcpProtocolPins.MODEL_NAME,
                "alias": InspectGhcpProtocolPins.CLI_MODEL_ALIAS,
                "max_tokens": 512,
                "max_retries": 0,
            }
        )
        return ModelRouteRef(name="qwen3_loopback_v1", config_sha256=digest)

    async def verify_host_async(self) -> None:
        """
        Prove prebuilt local images and a loopback-only provider before starting an Inspect Task.

        Raises:
            RuntimeError: If Docker SSH, exact image IDs, or the host model cannot be verified.
        """
        from examples.inspect_ghcp_protocol_smoke import _docker_async, _docker_host, verify_provider_async

        _docker_host()
        for image, expected in (
            (self.agent_image, self.agent_image_id),
            (self.target_image, self.target_image_id),
        ):
            code, stdout, _ = await _docker_async(command=["image", "inspect", "--format", "{{.Id}}", image])
            if code != 0 or stdout.strip() != expected:
                raise RuntimeError("The reviewed benign Inspect image ID changed before Task startup.")
        if not await verify_provider_async(InspectGhcpProtocolPins.MODEL_ENDPOINT):
            raise RuntimeError("The trusted Qwen provider is not verified on host loopback.")


class InspectGhcpCaseExecutor:
    """One-shot bridge from a typed Eval case to the existing Inspect-owned evaluator."""

    def __init__(self, *, selected: ResolvedInspectEvalCase, environment: InspectGhcpPilotEnvironment) -> None:
        """Bind one reviewed benign Task, scoped execution profile and PyRIT memory."""
        self._selected = selected
        self._environment = environment
        self._memory: MemoryInterface = CentralMemory.get_memory_instance()
        self._attempted = False

    async def execute_case_async(self, *, case: EvalCaseRef, run: EvalRunRef) -> CommittedCaseExecution:
        """
        Publish one original Inspect Task result, then reread its already committed Score.

        Returns:
            CommittedCaseExecution: The original Score ID, never a second Score.

        Raises:
            ValueError: If source, profile, original scorer, or linked Score evidence differs.
            RuntimeError: If the case was already attempted or trusted preflight fails.
        """
        if self._attempted:
            raise RuntimeError("An Inspect GHCP Eval case may be launched only once.")
        if (
            case != self._selected.case
            or run.spec.package != case.package
            or run.spec.harness != self._environment.harness_ref(sandbox_sha256=self._selected.sandbox_sha256)
            or run.spec.model_route != self._environment.model_route_ref()
            or (run.spec.input_variant.content_sha256 if run.spec.input_variant else None)
            != self._selected.input_override_sha256
        ):
            raise ValueError("The one-click Eval case, input overlay, harness, or model route changed.")
        run.case_run_id(case=case)
        self._attempted = True
        await asyncio.to_thread(self._selected.verify_unchanged)
        from examples.inspect_ghcp_protocol_smoke import (
            verify_image_async,
            verify_project_cleanup_async,
            verify_provider_async,
            verify_removed_async,
        )

        files = self._selected.source_files
        binding = InspectGhcpTaskBinding(
            task=self._selected.task,
            sample_id=case.sample_id,
            scorer_name=self._selected.scorer_name,
            target_service="target",
            health_command=self._selected.health_command,
            kind=InspectGhcpTaskKind.PROTOCOL_SMOKE,
            approved_assets={path: sha for path, _, sha in files},
            approved_asset_labels={path: relative for path, relative, _ in files},
            approved_image_ids=self._environment.image_ids,
            provider_endpoint=InspectGhcpProtocolPins.MODEL_ENDPOINT,
            verify_provider_async=verify_provider_async,
            verify_image_async=verify_image_async,
            verify_removed_async=verify_removed_async,
            verify_project_cleanup_async=verify_project_cleanup_async,
            prompt_cache_key_policy="omit_after_capture",
        )
        binding.validate()
        await self._environment.verify_host_async()
        evaluator = InspectGhcpEvaluation(
            binding=binding,
            attack_factory=lambda target, capture: build_benign_red_teaming_attack(target=target, capture=capture),
            model=create_benign_inspect_model(),
            model_id=InspectGhcpProtocolPins.CLI_MODEL_ALIAS,
            wire_model=InspectGhcpProtocolPins.MODEL_NAME,
            cli_path=InspectGhcpProtocolPins.CLI_PATH,
            cli_sha256=InspectGhcpProtocolPins.GHCP_CLI_SHA256,
            limits=InspectGhcpLimits(
                min_turns=2, min_tool_executions=1, max_turns=2, max_model_requests=12, run_timeout_seconds=240
            ),
            allowed_tools=("bash",),
            case=case,
            run=run,
            original_input_sha256=self._selected.original_input_sha256,
        )
        outcome = await evaluator.run_async()
        if outcome.report.run_id != evaluator.run_id:
            raise ValueError("Inspect GHCP evaluator returned a foreign source episode.")
        return await asyncio.to_thread(self._verify_original_score, case=case, run=run, outcome=outcome)

    def _verify_original_score(
        self, *, case: EvalCaseRef, run: EvalRunRef, outcome: InspectGhcpOutcome
    ) -> CommittedCaseExecution:
        self._selected.verify_unchanged()
        report = outcome.report
        episode = self._memory.native_cyber_evidence.get_finalized_episode(run_id=report.run_id)
        if (
            report.task_kind is not InspectGhcpTaskKind.PROTOCOL_SMOKE
            or report.status is not InspectGhcpStatus.INCOMPLETE
            or report.benchmark_verified
            or report.required_gaps != ("Original cyber task/scorer/target image were not independently verified.",)
            or report.task_name != case.task_name
            or report.task_version != case.task_version
            or report.sample_id != case.sample_id
            or report.sample_epoch != case.epoch
            or report.task_assets_sha256 != {relative: sha for _, relative, sha in self._selected.source_files}
            or report.image_ids != self._environment.image_ids
            or report.cli_sha256 != InspectGhcpProtocolPins.GHCP_CLI_SHA256
            or report.model_id != InspectGhcpProtocolPins.CLI_MODEL_ALIAS
            or report.wire_model != InspectGhcpProtocolPins.MODEL_NAME
            or not report.provider_verified
            or not report.token_files_absent_before_turn
            or report.turn_count != 2
            or report.model_http_200_count < 2
            or report.host_model_http_200_count < 2
            or report.adversarial_http_200_count < 1
            or report.required_tool_executions < 1
            or report.successful_tool_execution_count < 1
            or not report.stopped_before_scoring
            or not all(
                (
                    report.gateway_alive_before_scoring,
                    report.gateway_alive_after_scoring,
                    report.target_alive_before_scoring,
                    report.target_alive_after_scoring,
                    report.original_cleanup_called,
                    report.original_cleanup_succeeded,
                    report.sandbox_cleanup_observed,
                )
            )
            or report.judgment is None
            or report.judgment.scorer_name != self._selected.scorer_name
            or not report.judgment.source_event_id
            or report.judgment.numeric_value is None
            or not report.inspect_log_sha256
            or outcome.score.status is not ScoreStatus.UNDETERMINED
            or outcome.score.score_value is not None
            or episode.score_id != outcome.score.id
            or episode.report_sha256 != report.sha256()
        ):
            raise ValueError("The finalized Inspect Task lacks one matching original ScoreEvent or UND report.")
        readback = InspectGhcpEvidenceStore.open_finalized_for_readback(memory=self._memory, run_id=report.run_id)
        if readback.assess(report=report, expected_turns=report.turn_count) != report.required_gaps:
            raise ValueError("Sealed Inspect ScoreEvent or raw source coverage changed after Score commit.")
        scores = self._memory.get_scores(score_ids=[str(episode.score_id)])
        score = scores[0] if len(scores) == 1 else None
        if score is None or score.score_metadata is None:
            raise ValueError("The original Inspect report did not commit exactly one linked PyRIT Score.")
        provenance = EvalScoreProvenance.from_metadata(metadata=score.score_metadata)
        if (
            provenance.role is not EvalScoreRole.BENCHMARK_ORIGINAL
            or provenance.case_run_id != run.case_run_id(case=case)
            or score.scorer_class_identifier is None
            or provenance.pyrit_scorer_hash != score.scorer_class_identifier.hash
            or score.id != outcome.score.id
            or score.score_metadata.get("report_sha256") != report.sha256()
            or score.score_metadata.get("pyrit_eval_source_sha256") != case.package.source_sha256
            or score.score_metadata.get("pyrit_eval_original_input_sha256") != self._selected.original_input_sha256
            or score.score_metadata.get("pyrit_eval_input_variant_sha256")
            != (run.spec.input_variant.content_sha256 if run.spec.input_variant else None)
        ):
            raise ValueError("The committed Score lacks source-verified original-case provenance.")
        if report.conversation_id is None:
            raise ValueError("The retained Inspect Task did not link a genuine PyRIT conversation.")
        return CommittedCaseExecution(
            original_score_id=uuid.UUID(str(score.id)),
            original_score_provenance=provenance,
            conversation_id=uuid.UUID(report.conversation_id),
            outcome=AttackOutcome.UNDETERMINED,
            executed_turns=report.turn_count,
            execution_time_ms=max(0, int((report.ended_at - report.started_at).total_seconds() * 1000)),
            outcome_reason=(
                "Benign GHCP input variant only; original cyber benchmark task/image/scorer unqualified."
                if run.spec.input_variant
                else "Benign GHCP protocol only; original cyber benchmark task/image/scorer unqualified."
            ),
        )
