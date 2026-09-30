# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run a genuine Inspect Task with a contained GHCP solver and PyRIT attack."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import uuid4

import aiofiles
from inspect_ai import eval_async
from inspect_ai.agent import AgentState, sandbox_agent_bridge
from inspect_ai.log import EvalLog, read_eval_log
from inspect_ai.model import ModelOutput
from inspect_ai.solver import solver
from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec, sandbox

from pyrit.executor.attack.multi_turn.red_teaming import RedTeamingAttack
from pyrit.executor.benchmark._inspect_ghcp_adversary_capture import InspectGhcpAdversarialCapture
from pyrit.executor.benchmark._inspect_ghcp_runtime import InspectGhcpLimits, InspectGhcpSandboxRuntime
from pyrit.executor.benchmark.inspect_eval_projection import final_original_score_event
from pyrit.executor.benchmark.inspect_ghcp_model import InspectLoopbackModelAPI
from pyrit.memory import CentralMemory
from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.models import Message, Score, ScoreStatus
from pyrit.models.eval_case import EvalCaseRef, EvalRunRef, EvalScoreProvenance, EvalScoreRole
from pyrit.models.inspect_ghcp import (
    InspectGhcpJudgment,
    InspectGhcpReport,
    InspectGhcpStatus,
    InspectGhcpTaskKind,
)
from pyrit.prompt_target import InspectGhcpTarget, PromptTarget
from pyrit.score.float_scale.inspect_ghcp_report_scorer import InspectGhcpReportScorer

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.model import Model
    from inspect_ai.solver import Generate, Solver, TaskState

    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class InspectGhcpTaskBinding:
    """The selected original Inspect sample, scorer, services and approved assets."""

    task: Task
    sample_id: str
    scorer_name: str
    approved_sample_ids: tuple[str, ...] | None = None
    target_service: str = "target"
    health_command: tuple[str, ...] = ("/bin/true",)
    kind: InspectGhcpTaskKind = InspectGhcpTaskKind.PROTOCOL_SMOKE
    approved_assets: dict[Path, str] = field(default_factory=dict)
    approved_asset_labels: dict[Path, str] = field(default_factory=dict)
    approved_image_ids: dict[str, str] = field(default_factory=dict)
    expected_target_image: str | None = None
    verify_image_async: Callable[[str, str], Awaitable[str | None]] | None = None
    verify_removed_async: Callable[[str], Awaitable[bool]] | None = None
    verify_project_cleanup_async: Callable[[], Awaitable[bool]] | None = None
    provider_endpoint: str | None = None
    verify_provider_async: Callable[[str], Awaitable[bool]] | None = None
    prompt_cache_key_policy: str = "reject"

    def validate(self) -> ComposeConfig:
        """
        Reject tasks whose agent/model services could reach host or Internet resources.

        Returns:
            ComposeConfig: The original task's exact checked Docker Compose configuration.

        Raises:
            ValueError: If sample, scorer, image, network, or sandbox ownership is ambiguous.
        """
        self.selected_sample()
        if len(self.task.scorer or []) != 1 or not self.scorer_name:
            raise ValueError("Inspect GHCP requires one identified original Inspect scorer.")
        if not self.health_command or any(not command for command in self.health_command):
            raise ValueError("A fixed target-service health command is required.")
        if self.prompt_cache_key_policy not in {"reject", "omit_after_capture"}:
            raise ValueError("Inspect GHCP cache-key handling must be explicit and auditable.")
        if self.target_service in {"agent", "model-bridge"}:
            raise ValueError("Original target must be separate from GHCP and the model bridge.")
        endpoint = urlsplit(self.provider_endpoint or "")
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "::1"}
            or endpoint.port is None
            or endpoint.username is not None
            or self.verify_provider_async is None
        ):
            raise ValueError("Model provider must be verified on trusted host loopback before any GHCP send.")
        if self.kind is not InspectGhcpTaskKind.PROTOCOL_SMOKE:
            raise ValueError(
                "Original cyber task/data/image/scorer parity is not qualified by this GHCP harness; "
                "only a non-scored protocol smoke may run."
            )
        if self.approved_asset_labels:
            labels = list(self.approved_asset_labels.values())
            if set(self.approved_asset_labels) != set(self.approved_assets) or len(labels) != len(set(labels)):
                raise ValueError("Every approved asset requires a distinct source-relative public label.")
            if any(
                "\\" in label
                or label.startswith("/")
                or ":" in label
                or any(part in {"", ".", ".."} for part in label.split("/"))
                for label in labels
            ):
                raise ValueError("Approved asset labels must be source-relative and cannot expose host paths.")
        spec = self.task.sandbox
        if (
            not isinstance(spec, SandboxEnvironmentSpec)
            or spec.type != "docker"
            or not isinstance(spec.config, ComposeConfig)
        ):
            raise ValueError("The original Inspect Task must declare a reviewed Docker ComposeConfig.")
        self._validate_compose(config=spec.config)
        return spec.config

    def selected_sample(self) -> Sample:
        """
        Resolve the exact authored Sample, retaining its Task's original dataset.

        Returns:
            Sample: The sole case selected by Inspect's sample_id filter.

        Raises:
            ValueError: If the dataset or a per-Sample override is unqualified.
        """
        samples = tuple(self.task.dataset)
        ids = tuple(str(sample.id) for sample in samples)
        if self.approved_sample_ids is None:
            if len(ids) != 1 or ids[0] != self.sample_id:
                raise ValueError("Inspect GHCP requires exactly one selected original Inspect sample.")
        elif (
            ids != self.approved_sample_ids
            or len(set(ids)) != len(ids)
            or self.sample_id not in self.approved_sample_ids
        ):
            raise ValueError("Inspect GHCP sample selection differs from its pinned original dataset inventory.")
        for sample in samples:
            if sample.sandbox is not None:
                raise ValueError("Per-sample sandbox overrides require separate qualification.")
            if self.approved_sample_ids is not None and (sample.files or sample.setup or sample.checkpoint is not None):
                raise ValueError("Per-sample files, setup or checkpoint overrides require separate qualification.")
        return next(sample for sample in samples if str(sample.id) == self.sample_id)

    def _validate_compose(self, *, config: ComposeConfig) -> None:
        services = config.services
        if set(services) != {"agent", "model-bridge", self.target_service}:
            raise ValueError("Inspect Compose needs exactly one agent, one model bridge and one original target.")
        if (
            services["agent"].x_default is not True
            or services["model-bridge"].x_default is True
            or services[self.target_service].x_default is True
        ):
            raise ValueError("Inspect must designate only the isolated agent as the default sandbox service.")
        temporary_paths = [
            "/home/bridge:rw,exec,size=256m",
            "/workspace:rw,exec,size=64m",
            "/tmp:rw,exec,size=512m",
        ]
        framework_tmpfs = "/var/tmp:rw,exec,nosuid,nodev,size=128m,mode=1777"
        for service_name in ("agent", "model-bridge", self.target_service):
            paths = services[service_name].tmpfs or []
            approved = service_name in {"agent", "model-bridge"}
            if paths != temporary_paths + ([framework_tmpfs] if approved else []):
                raise ValueError(f"Inspect {service_name} tmpfs profile differs from the reviewed contract.")
        internal = {
            name
            for name, network in (config.networks or {}).items()
            if isinstance(network, dict) and network.get("internal") is True
        }
        if len(internal) != 1 or set(config.networks or {}) != internal:
            raise ValueError("Inspect Compose must declare exactly one no-egress internal network.")
        network_name = next(iter(internal))
        for name, service in services.items():
            attached = (
                set(service.networks or []) if isinstance(service.networks, list) else set(service.networks or {})
            )
            if attached != {network_name} or service.network_mode or service.ports:
                raise ValueError(f"Inspect service {name} may attach only to its internal network without host ports.")
            if any("docker.sock" in str(volume) for volume in (service.volumes or [])):
                raise ValueError("No Inspect service may mount the Docker daemon socket.")
            if (
                service.user != "10001:10001"
                or service.read_only is not True
                or service.cap_drop != ["ALL"]
                or service.cap_add
                or service.security_opt != ["no-new-privileges:true"]
                or service.privileged is True
                or service.devices
                or service.volumes
                or service.extra_hosts
                or service.env_file
                or service.build is not None
                or type(service.pids_limit) is not int
                or not 1 <= service.pids_limit <= 64
                or service.mem_limit not in {"384m", "512m", "1.25gb"}
                or service.cpus is None
                or not 0 < service.cpus <= 1.0
            ):
                raise ValueError(f"Inspect service {name} is not within the approved least-privilege profile.")
            pinned_digest = bool(service.image and re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", service.image))
            pinned_local = bool(
                service.image
                and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/\-]*:[a-zA-Z0-9][a-zA-Z0-9._\-]*", service.image)
                and re.fullmatch(r"sha256:[0-9a-f]{64}", self.approved_image_ids.get(name, ""))
                and self.verify_image_async is not None
            )
            if service.pull_policy != "never" or not (pinned_digest or pinned_local):
                raise ValueError(f"Inspect service {name} needs a prebuilt image and a verified immutable image ID.")
        for name in ("agent", "model-bridge"):
            service = services[name]
            if not service.image:
                raise ValueError(f"Inspect {name} service is not pinned, nonroot, read-only and least-privilege.")
            if service.environment:
                values = (
                    service.environment.keys()
                    if isinstance(service.environment, dict)
                    else (entry.split("=", 1)[0] for entry in service.environment)
                )
                if any(
                    any(secret in key.upper() for secret in ("KEY", "TOKEN", "SECRET", "PASSWORD")) for key in values
                ):
                    raise ValueError(f"Inspect {name} service must not contain permanent credentials.")


@dataclass(frozen=True, kw_only=True)
class InspectGhcpOutcome:
    """One PyRIT report/Score linked to the original Inspect EvalLog and source episode."""

    report: InspectGhcpReport
    score: Score
    episode: NativeCyberEpisodeSnapshot
    log_location: str | None


class InspectGhcpEvaluation:
    """Select PyRIT's attack while Inspect runs original setup/scorer/cleanup once."""

    def __init__(
        self,
        *,
        binding: InspectGhcpTaskBinding,
        attack_factory: Callable[[PromptTarget, InspectGhcpAdversarialCapture], RedTeamingAttack],
        model: Model,
        model_id: str,
        wire_model: str,
        cli_path: str,
        cli_sha256: str,
        limits: InspectGhcpLimits,
        allowed_tools: tuple[str, ...] = ("bash",),
        case: EvalCaseRef | None = None,
        run: EvalRunRef | None = None,
        original_input_sha256: str | None = None,
    ) -> None:
        """
        Bind the chosen PyRIT technique and original Inspect task without starting either.

        Raises:
            ValueError: If the Task or pinned CLI does not meet the run contract.
        """
        binding.validate()
        if not re.fullmatch(r"[0-9a-f]{64}", cli_sha256):
            raise ValueError("The approved GHCP CLI requires its exact SHA256.")
        if not isinstance(model.api, InspectLoopbackModelAPI):
            raise ValueError("GHCP Inspect provider must capture real host model HTTP bytes without OpenAI 3.")
        if (case is None) != (run is None) or (case is None) != (original_input_sha256 is None):
            raise ValueError("A task-owned Inspect case requires source, run, and original input identities.")
        if case is not None and run is not None:
            sample_input = binding.selected_sample().input
            expected_input_sha256 = (
                run.spec.input_variant.content_sha256 if run.spec.input_variant else original_input_sha256
            )
            if (
                case.task_name != binding.task.name
                or case.task_version != str(binding.task.version)
                or case.sample_id != binding.sample_id
                or case.epoch != 1
                or original_input_sha256 is None
                or re.fullmatch(r"[0-9a-f]{64}", original_input_sha256) is None
                or not isinstance(sample_input, str)
                or hashlib.sha256(sample_input.encode("utf-8")).hexdigest() != expected_input_sha256
            ):
                raise ValueError("Task-owned Inspect case differs from its selected Task/Sample/input identity.")
            run.case_run_id(case=case)
        self.binding = binding
        self._case = case
        self._run = run
        self._original_input_sha256 = original_input_sha256
        self._attack_factory = attack_factory
        self._model = model
        self._model_api = model.api
        self._model_id = model_id
        self._wire_model = wire_model
        self._cli_path = cli_path
        self._cli_sha256 = cli_sha256
        self._limits = limits
        self._allowed_tools = allowed_tools
        self._memory = CentralMemory.get_memory_instance()
        self._run_id = str(uuid4())
        self._control_token = secrets.token_urlsafe(32)
        self._started_at = datetime.now(UTC)
        self._start_monotonic = time.monotonic()
        self._receipt_path = Path.cwd() / ".venv" / "inspect-ghcp" / self._run_id / "controller-stage.jsonl"
        self._bridge_seconds: float | None = None
        self._store: InspectGhcpEvidenceStore | None = None
        self._runtime: InspectGhcpSandboxRuntime | None = None
        self._target: InspectGhcpTarget | None = None
        self._attack_identifier: dict[str, Any] | None = None
        self._conversation_id: str | None = None
        self._pregrade_gaps: tuple[str, ...] = ()
        self._gateway_sha256: str | None = None
        self._host_sha256: str | None = None
        self._adversarial_sha256: str | None = None
        self._control_sha256: str | None = None
        self._adversarial_capture: InspectGhcpAdversarialCapture | None = None
        self._host_capture_lock = asyncio.Lock()
        self._adversarial_capture_lock = asyncio.Lock()
        self._capture_sink = self._capture_host_model_async
        self._target_before = False
        self._target_after = False
        self._cleanup_called = False
        self._cleanup_succeeded = False
        self._cleanup_attestation_error: str | None = None
        self._agent_stopped = False
        self._gateway_before = False
        self._gateway_after = False
        self._successful_tools = 0
        self._phase = "creating_source_episode"
        self._target_container_ref: str | None = None
        self._container_ids: dict[str, str] = {}
        self._provider_verified = False
        self._run_called = False

    @property
    def run_id(self) -> str:
        """The scoped Inspect source episode created by this evaluator."""
        return self._run_id

    async def run_async(self) -> InspectGhcpOutcome:
        """
        Run the original Task and map its one acquired judgment after Inspect cleanup.

        Returns:
            InspectGhcpOutcome: An atomic PyRIT Score or explicit undetermined report.

        Raises:
            RuntimeError: If atomic publication fails or this instance is run twice.
            asyncio.CancelledError: If the run is cancelled; owned processes are stopped.
        """
        if self._run_called:
            raise RuntimeError("Each Inspect GHCP evaluation can run only once.")
        self._run_called = True
        self._start_monotonic = time.monotonic()
        self._store = await asyncio.to_thread(
            InspectGhcpEvidenceStore,
            memory=self._memory,
            run_id=self._run_id,
            task_name=self.binding.task.name,
            task_version=str(self.binding.task.version),
            started_at=self._started_at,
            raw_byte_limit=self._limits.max_raw_bytes,
        )
        await self._checkpoint_async(stage="episode_created")
        self._model_api.set_capture_sink(sink=self._capture_sink)
        log: EvalLog | None = None
        log_bytes: bytes | None = None
        error_name: str | None = None
        task_copy = copy.copy(self.binding.task)
        task_copy.cleanup = self._cleanup_async
        try:
            self._phase = "verifying_task_assets"
            await self._verify_assets_async()
            provider_check = self.binding.verify_provider_async
            assert provider_check is not None
            self._provider_verified = await provider_check(self.binding.provider_endpoint or "")
            if not self._provider_verified:
                raise RuntimeError("Trusted provider is not verified on a loopback-only listener.")
            self._phase = "running_original_inspect_task"
            await self._checkpoint_async(stage="inspect_eval_started")
            location = Path.cwd() / ".venv" / "inspect-ghcp" / self._run_id
            await asyncio.to_thread(location.mkdir, parents=True, exist_ok=True)
            logs = await eval_async(
                tasks=task_copy,
                solver=self._registered_solver(),
                model=self._model,
                sample_id=self.binding.sample_id,
                max_samples=1,
                max_tasks=1,
                epochs=1,
                retry_on_error=0,
                score_on_error=False,
                sandbox_prebuilt=True,
                sandbox_cleanup=True,
                log_dir=str(location),
                log_model_api=True,
                log_realtime=False,
                token_limit=self._limits.max_prompt_tokens * self._limits.max_model_requests,
                time_limit=self._limits.run_timeout_seconds,
                max_retries=0,
                max_connections=1,
                attempt_timeout=self._limits.turn_timeout_seconds,
            )
            if len(logs) != 1 or not logs[0].location:
                raise RuntimeError("Inspect did not return one complete original Task log.")
            await self._checkpoint_async(stage="inspect_eval_returned")
            log = await asyncio.to_thread(read_eval_log, logs[0].location, resolve_attachments="full")
            self._phase = "retaining_original_inspect_log"
            log_bytes = log.model_dump_json(exclude_none=True).encode("utf-8")
            await asyncio.to_thread(self._store.record_inspect_log, content=log_bytes)
            await self._checkpoint_async(stage="original_log_retained")
            if self._original_task_failed(log=log):
                phase = self._runtime.phase if self._runtime is not None else self._phase
                error_name = f"OriginalInspectTaskError at {phase}"
                logger.error("Inspect GHCP run %s ended with an original Task error at %s.", self._run_id, phase)
        except (Exception, asyncio.CancelledError) as error:
            phase = self._runtime.phase if self._runtime is not None else self._phase
            error_name = f"{type(error).__name__} at {phase}"
            logger.error("Inspect GHCP run %s could not finish: %s", self._run_id, error_name)
            if isinstance(error, asyncio.CancelledError):
                raise
        finally:
            self._model_api.clear_capture_sink(sink=self._capture_sink)
            await self._checkpoint_async(stage="post_eval_cleanup_started")
            if not self._agent_stopped:
                await asyncio.to_thread(
                    self._store._capture.mark_capture_gap,
                    run_id=self._run_id,
                    reason=(
                        "Contained GHCP process stop was not observed before original grading "
                        f"(stage: {self._runtime.phase if self._runtime else self._phase})."
                    ),
                )
            await self._checkpoint_async(stage="post_eval_cleanup_finished")
        report = await self._build_report_async(log=log, log_bytes=log_bytes, error=error_name)
        await self._checkpoint_async(stage="report_built")
        assert self._store is not None
        gaps = await asyncio.to_thread(self._store.assess, report=report, expected_turns=report.turn_count)
        await self._checkpoint_async(stage="coverage_assessed")
        report = report.model_copy(
            update={
                "required_gaps": gaps,
                "status": InspectGhcpStatus.COMPLETED
                if report.benchmark_verified and not gaps
                else InspectGhcpStatus.ERROR
                if error_name
                else InspectGhcpStatus.INCOMPLETE,
            }
        )
        report = InspectGhcpReport.model_validate(report.model_dump(mode="json"))
        scorer = InspectGhcpReportScorer(report_sha256=report.sha256())
        score = scorer.prepare_unpersisted_score(report=report)
        if (
            self._case is not None
            and self._run is not None
            and report.status is InspectGhcpStatus.INCOMPLETE
            and report.required_gaps == ("Original cyber task/scorer/target image were not independently verified.",)
        ):
            self._attach_original_case_provenance(report=report, score=score)
        await self._checkpoint_async(stage="score_prepared")
        episode = await asyncio.to_thread(self._store.finalize_atomic, report=report, score=score)
        await self._checkpoint_async(stage="score_committed")
        await self._checkpoint_async(stage="controller_result_ready")
        return InspectGhcpOutcome(
            report=report,
            score=score,
            episode=episode,
            log_location=log.location if log is not None else None,
        )

    def _attach_original_case_provenance(self, *, report: InspectGhcpReport, score: Score) -> None:
        """
        Label only a source-verified original Score before its atomic publication.

        Raises:
            ValueError: If the original Task/ScoreEvent or required capture differs.
        """
        assert self._case is not None and self._run is not None
        judgment = report.judgment
        scorer = score.scorer_class_identifier
        if (
            report.task_kind is not InspectGhcpTaskKind.PROTOCOL_SMOKE
            or report.status is not InspectGhcpStatus.INCOMPLETE
            or report.benchmark_verified
            or report.required_gaps != ("Original cyber task/scorer/target image were not independently verified.",)
            or report.task_name != self._case.task_name
            or report.task_version != self._case.task_version
            or report.sample_id != self._case.sample_id
            or report.sample_epoch != self._case.epoch
            or not report.inspect_log_sha256
            or judgment is None
            or not judgment.source_event_id
            or judgment.scorer_name != self.binding.scorer_name
            or judgment.numeric_value is None
            or scorer is None
            or score.status is not ScoreStatus.UNDETERMINED
            or score.score_value is not None
        ):
            raise ValueError("The original Inspect Task/ScoreEvent is not qualified for case Score linkage.")
        provenance = EvalScoreProvenance(
            role=EvalScoreRole.BENCHMARK_ORIGINAL,
            case_run_id=self._run.case_run_id(case=self._case),
            pyrit_scorer_hash=scorer.hash,
        )
        assert self._original_input_sha256 is not None
        source_metadata = {
            "pyrit_eval_source_sha256": self._case.package.source_sha256,
            "pyrit_eval_original_input_sha256": self._original_input_sha256,
        }
        if self._run.spec.input_variant is not None:
            source_metadata["pyrit_eval_input_variant_sha256"] = self._run.spec.input_variant.content_sha256
        score.score_metadata = {**(score.score_metadata or {}), **provenance.to_metadata(), **source_metadata}

    def _registered_solver(self) -> Solver:
        @solver
        def inspect_ghcp() -> Solver:
            async def solve_async(state: TaskState, generate: Generate) -> TaskState:
                return await self._solve_async(state=state, generate=generate)

            return solve_async

        return inspect_ghcp()

    @staticmethod
    def _original_task_failed(*, log: EvalLog) -> bool:
        samples = log.samples
        if log.status != "success" or samples is None or len(samples) != 1:
            return True
        return samples[0].error is not None or not samples[0].scores

    async def _solve_async(self, *, state: TaskState, generate: Generate) -> TaskState:
        if self._store is None:
            raise RuntimeError("The Inspect GHCP run has no source episode.")
        if not isinstance(state.input, str) or not state.input.strip():
            raise ValueError("This GHCP solver accepts original Inspect text samples only.")
        image_check = self.binding.verify_image_async
        if image_check is None:
            raise RuntimeError("An Inspect GHCP sample cannot receive a token without trusted Docker ID checks.")
        runtime = InspectGhcpSandboxRuntime(
            run_id=self._run_id,
            token=self._control_token,
            model_id=self._model_id,
            wire_model=self._wire_model,
            cli_path=self._cli_path,
            cli_sha256=self._cli_sha256,
            gateway_image_service="model-bridge",
            limits=self._limits,
            allowed_tools=self._allowed_tools,
            control_receipt_sink=self._capture_control_receipt_async,
            verify_image_async=image_check,
            approved_image_ids=self.binding.approved_image_ids,
            prompt_cache_key_policy=self.binding.prompt_cache_key_policy,
        )
        self._runtime = runtime
        async with sandbox_agent_bridge(
            AgentState(messages=state.messages),
            sandbox="model-bridge",
            model_aliases={self._wire_model: self._model},
            web_search=False,
            code_execution=False,
            client_mcp_servers=False,
        ) as bridge:
            self._bridge_seconds = round(time.monotonic() - self._start_monotonic, 3)
            try:
                await runtime.start_async(inspect_proxy_port=bridge.port)
                if runtime.container_ref is None or runtime.session_id is None:
                    raise RuntimeError("Inspect GHCP runtime did not expose its observed agent/session identity.")
                if not runtime.private_tokens_consumed:
                    raise RuntimeError("Inspect did not verify both one-time token files absent before inference.")
                self._control_sha256 = await asyncio.to_thread(self._store.seal_control_receipts)
                target_connection = await sandbox(self.binding.target_service).connection()
                if not target_connection.container or target_connection.container in {
                    runtime.container_ref,
                    runtime.model_container_ref,
                }:
                    raise ValueError("Original target, model bridge, and agent must be distinct containers.")
                self._target_container_ref = target_connection.container
                identities = await self._verify_images_async(runtime=runtime)
                if identities is None:
                    raise RuntimeError("All three Inspect containers require trusted image/identity attestation.")
                self._container_ids = identities
                await asyncio.to_thread(
                    self._store.bind_agent,
                    container_id=identities["agent"],
                    session_id=runtime.session_id,
                )
                target = InspectGhcpTarget(transport=runtime, run_id=self._run_id, model_name=self._model_id)
                self._target = target
                provider = self.binding.provider_endpoint
                if provider is None:
                    raise RuntimeError("PyRIT adversarial model has no approved loopback endpoint.")
                self._adversarial_capture = InspectGhcpAdversarialCapture(
                    endpoint=provider.rstrip("/") + "/chat/completions",
                    max_body_bytes=self._limits.max_model_bytes,
                    sink=self._capture_adversarial_model_async,
                )
                attack = self._attack_factory(target, self._adversarial_capture)
                if not isinstance(attack, RedTeamingAttack):
                    raise TypeError("The Inspect Task requires a PyRIT RedTeamingAttack for its turn decisions.")
                self._attack_identifier = attack.get_identifier().model_dump(mode="json")
                result = await attack.execute_async(
                    objective=state.input,
                    next_message=Message.from_prompt(prompt=state.input, role="user"),
                    memory_labels={"inspect_ghcp_run_id": self._run_id},
                )
                await self._adversarial_capture.close_async()
                self._adversarial_sha256 = await asyncio.to_thread(self._store.seal_adversarial_model)
                self._conversation_id = result.conversation_id
                for turn in target.turns:
                    await asyncio.to_thread(self._store.record_turn, turn=turn)
                await runtime.stop_agent_async()
                self._agent_stopped = True
                await asyncio.to_thread(self._store.seal_sdk_events)
                self._host_sha256 = await asyncio.to_thread(self._store.seal_host_model)
                audit = await runtime.read_gateway_audit_async()
                self._gateway_sha256 = await asyncio.to_thread(
                    self._store.record_gateway_audit, records=audit, turns=target.turns
                )
                self._pregrade_gaps = await asyncio.to_thread(
                    self._store.assess, report=None, expected_turns=len(target.turns)
                )
                if self._pregrade_gaps:
                    raise RuntimeError(f"Pregrading Inspect GHCP coverage has {len(self._pregrade_gaps)} gap(s).")
                self._successful_tools = await asyncio.to_thread(self._store.count_successful_tool_executions)
                if self._successful_tools < self._limits.min_tool_executions:
                    await asyncio.to_thread(
                        self._store._capture.mark_capture_gap,
                        run_id=self._run_id,
                        reason="Task-required real GHCP tool request/start/completion was not observed.",
                    )
                    raise RuntimeError("A required GHCP tool was not executed in the contained agent.")
                if len(target.turns) < self._limits.min_turns:
                    await asyncio.to_thread(
                        self._store._capture.mark_capture_gap,
                        run_id=self._run_id,
                        reason="PyRIT did not make the required retained-session adversarial follow-up.",
                    )
                    raise RuntimeError("A retained Inspect GHCP run needs its PyRIT-chosen follow-up turn.")
                self._gateway_before = await runtime.gateway_alive_async()
                if not self._gateway_before:
                    raise RuntimeError("Authenticated model gateway died before original Inspect scoring.")
                self._target_before = await self._probe_target_async()
                if not self._target_before:
                    raise RuntimeError("Original Inspect target service was not alive before scoring.")
                if not target.turns:
                    raise RuntimeError("PyRIT did not send a genuine GHCP turn.")
                state.output = ModelOutput.from_content(model=self._model.name, content=target.turns[-1].assistant_text)
                state.messages = bridge.state.messages
                state.store.set("inspect_ghcp_run_id", self._run_id)
                state.store.set("inspect_ghcp_session_id", runtime.session_id)
                state.store.set("inspect_ghcp_pregraded", True)
                return state
            finally:
                if self._adversarial_capture is not None:
                    await self._adversarial_capture.close_async()
                    if self._adversarial_sha256 is None:
                        self._adversarial_sha256 = await asyncio.to_thread(self._store.seal_adversarial_model)
                stderr, omitted = runtime.agent_stderr_capture
                if stderr:
                    await asyncio.to_thread(self._store.record_guest_stderr, data=stderr, omitted_bytes=omitted)
                if not self._agent_stopped:
                    try:
                        await runtime.close_async()
                    except TimeoutError as error:
                        raise RuntimeError(
                            f"Inspect GHCP sandbox cleanup timed out before grading at {runtime.phase}."
                        ) from error

    async def _capture_host_model_async(
        self, request: bytes, response: bytes | None, status: int | None, error: str | None
    ) -> None:
        if self._store is None:
            raise RuntimeError("The original host model cannot run without a bound PyRIT evidence episode.")
        async with self._host_capture_lock:
            await asyncio.to_thread(
                self._store.record_host_model_exchange,
                request=request,
                response=response,
                status=status,
                error=error,
            )

    async def _capture_adversarial_model_async(
        self, request_id: str, phase: str, body: bytes, status: int | None, error: str | None
    ) -> None:
        if self._store is None:
            raise RuntimeError("PyRIT adversarial model cannot run without a source episode.")
        async with self._adversarial_capture_lock:
            await asyncio.to_thread(
                self._store.record_adversarial_model_frame,
                request_id=request_id,
                phase=phase,
                body=body,
                status=status,
                error=error,
            )

    async def _capture_control_receipt_async(self, receipt: dict[str, Any]) -> None:
        if self._store is None:
            raise RuntimeError("An Inspect private control handoff has no source episode.")
        await asyncio.to_thread(self._store.record_control_receipt, receipt=receipt)

    async def _checkpoint_async(self, *, stage: str) -> None:
        if not re.fullmatch(r"[a-z_]+", stage):
            raise ValueError("Controller stage names must not contain task content.")
        await asyncio.to_thread(self._receipt_path.parent.mkdir, parents=True, exist_ok=True)
        fields = {
            "run_id": self._run_id,
            "stage": stage,
            "controller_pid": os.getpid(),
            "parent_pid": os.getppid(),
            "python_executable": sys.executable,
            "elapsed_seconds": round(time.monotonic() - self._start_monotonic, 3),
            "at": datetime.now(UTC).isoformat(),
        }
        if stage == "episode_created":
            fields["control_token_sha256"] = hashlib.sha256(self._control_token.encode("ascii")).hexdigest()
        record = json.dumps(fields, separators=(",", ":"))
        async with aiofiles.open(self._receipt_path, "a", encoding="utf-8") as stream:
            await stream.write(record + "\n")

    async def _cleanup_async(self, state: TaskState) -> None:
        self._cleanup_called = True
        original = self.binding.task.cleanup
        try:
            self._target_after = await self._probe_target_async()
            self._gateway_after = bool(self._runtime and await self._runtime.gateway_alive_async())
            if not self._gateway_after:
                raise RuntimeError("The original Inspect scorer outlived its authenticated model gateway.")
        finally:
            if original is not None:
                await original(state)
            self._cleanup_succeeded = True

    async def _probe_target_async(self) -> bool:
        result = await sandbox(self.binding.target_service).exec(
            list(self.binding.health_command), timeout=5, timeout_retry=False
        )
        return result.success

    async def _verify_assets_async(self) -> None:
        for path, expected in self.binding.approved_assets.items():
            if not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise ValueError("An approved original task asset lacks a SHA256.")
            async with aiofiles.open(path, "rb") as source:
                digest = hashlib.sha256()
                while chunk := await source.read(65_536):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"An original Inspect task/scorer asset changed: {path.name}.")

    async def _verify_images_async(self, *, runtime: InspectGhcpSandboxRuntime) -> dict[str, str] | None:
        check = self.binding.verify_image_async
        if check is None:
            return None
        config = self.binding.validate()
        services = config.services
        references = (
            (
                "agent",
                runtime.container_ref,
                self.binding.approved_image_ids.get("agent", services["agent"].image),
            ),
            (
                "model-bridge",
                runtime.model_container_ref,
                self.binding.approved_image_ids.get("model-bridge", services["model-bridge"].image),
            ),
            (
                "target",
                self._target_container_ref,
                self.binding.approved_image_ids.get(
                    self.binding.target_service, services[self.binding.target_service].image
                ),
            ),
        )
        identities: dict[str, str] = {}
        for role, container, image in references:
            if not container or not image:
                return None
            observed_id = await check(container, image)
            if (
                not isinstance(observed_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", observed_id)
                or observed_id in identities.values()
                or (role in {"agent", "model-bridge"} and observed_id != runtime.attested_container_ids.get(role))
            ):
                return None
            identities[role] = observed_id
        return identities

    async def _build_report_async(
        self, *, log: EvalLog | None, log_bytes: bytes | None, error: str | None
    ) -> InspectGhcpReport:
        sample = log.samples[0] if log is not None and log.samples and len(log.samples) == 1 else None
        judgment = self._judgment(sample=sample) if sample is not None else None
        runtime = self._runtime
        if self._store is None:
            raise RuntimeError("The Inspect report has no source evidence episode.")
        snapshot = await asyncio.to_thread(self._store._capture.get_episode, run_id=self._run_id)
        sandbox_removed = False
        try:
            if self._container_ids and self.binding.verify_removed_async:
                containers = (
                    self._container_ids.get("agent"),
                    self._container_ids.get("model-bridge"),
                    self._container_ids.get("target"),
                )
                sandbox_removed = bool(all(containers))
                for container in containers:
                    if container is not None and sandbox_removed:
                        sandbox_removed = await self.binding.verify_removed_async(container)
            if sandbox_removed:
                check_project = self.binding.verify_project_cleanup_async
                sandbox_removed = bool(check_project and await check_project())
        except (OSError, RuntimeError, TimeoutError) as cleanup_error:
            self._cleanup_attestation_error = type(cleanup_error).__name__
            await asyncio.to_thread(
                self._store._capture.mark_capture_gap,
                run_id=self._run_id,
                reason=f"Inspect cleanup attestation failed: {self._cleanup_attestation_error}.",
            )
            sandbox_removed = False
        status = InspectGhcpStatus.ERROR if error or self._cleanup_attestation_error else InspectGhcpStatus.INCOMPLETE
        counts = [event.event_type for event in snapshot.events]
        tool_starts = counts.count("tool.execution_start")
        tool_completes = counts.count("tool.execution_complete")
        host_count, host_success = self._store.host_model_counts
        adversarial_count, adversarial_success = self._store.adversarial_model_counts
        sdk_stream = next(
            (
                stream
                for stream in snapshot.raw_streams
                if stream.key.observed_source_id == self._store.SDK_KEY.observed_source_id
            ),
            None,
        )
        return InspectGhcpReport(
            run_id=self._run_id,
            task_name=self.binding.task.name,
            task_version=str(self.binding.task.version),
            sample_id=self.binding.sample_id,
            sample_epoch=sample.epoch if sample is not None else 1,
            task_kind=self.binding.kind,
            benchmark_verified=False,
            provider_verified=self._provider_verified,
            task_assets_sha256={
                self.binding.approved_asset_labels.get(path, str(path)): value
                for path, value in self.binding.approved_assets.items()
            },
            image_ids=self.binding.approved_image_ids,
            target_image=self.binding.expected_target_image,
            cli_sha256=self._cli_sha256,
            model_id=self._model_id,
            wire_model=self._wire_model,
            prompt_cache_key_policy=self.binding.prompt_cache_key_policy,
            started_at=self._started_at,
            ended_at=datetime.now(UTC),
            inspect_log_id=log.eval.run_id if log is not None else None,
            inspect_log_sha256=hashlib.sha256(log_bytes).hexdigest() if log_bytes else None,
            inspect_sample_uuid=sample.uuid if sample else None,
            agent_container_id=self._container_ids.get("agent"),
            model_container_id=self._container_ids.get("model-bridge"),
            target_container_id=self._container_ids.get("target"),
            ghcp_session_id=runtime.session_id if runtime else None,
            agent_process=runtime.agent_identity if runtime else None,
            conversation_id=self._conversation_id,
            attack_identifier=self._attack_identifier,
            turn_count=len(snapshot.turns),
            sdk_event_count=len(snapshot.events),
            sdk_event_raw_sha256=sdk_stream.stored_sha256 if sdk_stream else None,
            model_request_count=sum(
                1 for turn in (self._target.turns if self._target is not None else ()) for _ in turn.model_exchanges
            ),
            model_http_200_count=sum(
                exchange.get("status") == 200
                for turn in (self._target.turns if self._target is not None else ())
                for exchange in turn.model_exchanges
            ),
            host_model_request_count=host_count,
            host_model_http_200_count=host_success,
            adversarial_request_count=adversarial_count,
            adversarial_http_200_count=adversarial_success,
            tool_start_count=tool_starts,
            tool_complete_count=tool_completes,
            successful_tool_execution_count=self._successful_tools,
            required_tool_executions=self._limits.min_tool_executions,
            gateway_audit_sha256=self._gateway_sha256,
            host_model_audit_sha256=self._host_sha256,
            adversarial_audit_sha256=self._adversarial_sha256,
            control_receipt_sha256=self._control_sha256,
            token_files_absent_before_turn=bool(runtime and runtime.private_tokens_consumed),
            phase_durations_seconds={
                "compose_and_bridge_setup": self._bridge_seconds or 0.0,
                **(runtime.phase_snapshot() if runtime is not None else {}),
            },
            stopped_before_scoring=self._agent_stopped,
            gateway_alive_before_scoring=self._gateway_before,
            gateway_alive_after_scoring=self._gateway_after,
            target_alive_before_scoring=self._target_before,
            target_alive_after_scoring=self._target_after,
            original_cleanup_called=self._cleanup_called,
            original_cleanup_succeeded=self._cleanup_succeeded,
            sandbox_cleanup_observed=sandbox_removed,
            judgment=judgment,
            required_gaps=self._pregrade_gaps,
            status=status,
        )

    def _judgment(self, *, sample: Any) -> InspectGhcpJudgment | None:
        score = (sample.scores or {}).get(self.binding.scorer_name)
        if score is None:
            return None
        event = final_original_score_event(sample=sample, scorer_name=self.binding.scorer_name)
        if event is None:
            return None
        raw = score.model_dump(mode="json", exclude_none=True)
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        value = score.value
        numeric = float(value) if type(value) in (int, float) and 0 <= value <= 1 else None
        return InspectGhcpJudgment(
            scorer_name=self.binding.scorer_name,
            source_event_id=event.uuid,
            normalization_version=1,
            raw_value=value,
            numeric_value=numeric,
            explanation=score.explanation,
            raw_sha256=hashlib.sha256(encoded).hexdigest(),
        )
