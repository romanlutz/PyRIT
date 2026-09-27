# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One controller turn through real provider classes and wholly fake Engine I/O."""

from __future__ import annotations

import asyncio
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import pytest

from pyrit.executor.workflow.docker_agent import DockerSandboxLauncher
from pyrit.executor.workflow.native_cli_evaluation import (
    NativeCliEvaluation,
    NativeCliOriginalAssessment,
)
from pyrit.models import ScoreStatus
from pyrit.models.environment_lease import EnvironmentLeaseState
from pyrit.models.native_cli_report import NativeCliOriginalJudgment, NativeCliReportStatus, NativeCliRunReport
from pyrit.prompt_target.native_cli_models import NativeCliProtocol, NativeCliRunConfig
from tests.unit.executor.workflow.test_docker_agent import make_agent
from tests.unit.executor.workflow.test_docker_engine import wire_frame
from tests.unit.executor.workflow.test_native_cli_evaluation import _FakeBinding, _FakeGateway, _stdout

if TYPE_CHECKING:
    from pyrit.executor.workflow.docker_agent import DockerStopOnlyAgentLease
    from pyrit.executor.workflow.docker_engine import DockerEngineClient
    from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
    from pyrit.memory import SQLiteMemory
    from tests.unit.executor.workflow.test_docker_engine import FakeEngine

pytestmark = pytest.mark.usefixtures("patch_central_database")


class _DockerRuntime:
    def __init__(
        self,
        *,
        lease: DockerStopOnlyAgentLease,
        gateway: _FakeGateway,
        fake_engine: FakeEngine,
        memory: SQLiteMemory,
        trace: list[str],
    ) -> None:
        self.lease = lease
        self.launcher = DockerSandboxLauncher(lease=lease)
        self.gateway_listener = gateway
        self.fake_engine = fake_engine
        self.memory = memory
        self.trace = trace
        self.grading_calls = 0

    async def verify_guest_exclusion_async(self) -> None:
        assert self.lease.snapshot().state is EnvironmentLeaseState.READY
        self.trace.append("guest_exclusion_verified")

    async def grade_async(self, *, report: NativeCliRunReport) -> NativeCliOriginalAssessment:
        self.grading_calls += 1
        self.trace.append("original_grader")
        stop = self.lease.agent_stop
        assert stop is not None and stop.stopped
        assert self.gateway_listener.is_locally_running
        assert report.status is NativeCliReportStatus.INCOMPLETE and report.judgment is None
        assert not await asyncio.to_thread(self.memory.get_scores, score_type="float_scale")
        assert all(
            self.fake_engine.containers[container_id]["State"]["Running"] for _, container_id in stop.preserved_services
        )
        return NativeCliOriginalAssessment(
            judgment=NativeCliOriginalJudgment(
                grader_ref="inert-target-side-grader",
                grader_evidence_ref=f"inert-target-observation:{report.run_id}",
                value=0.75,
                complete=True,
                rationale="Inert original criterion from preserved target state.",
            )
        )


class _DockerBinding(_FakeBinding):
    def __init__(self, *, memory: SQLiteMemory, send_gateway: bool = True) -> None:
        super().__init__(memory=memory)
        self.run_config = NativeCliRunConfig(
            protocol=NativeCliProtocol.CODEX_EXEC_JSON,
            cli_version="0.115.0",
            cli_profile="inert-stop-only",
            agent_workdir=PurePosixPath("/tmp/work"),
            model_gateway_endpoint="http://172.18.0.1:8733/v1",
            max_steps=2,
            timeout_seconds=3,
            max_frame_bytes=4096,
        )
        self.send_gateway = send_gateway
        self.runtime: _DockerRuntime | None = None
        self.engine: DockerEngineClient | None = None
        self.fake_engine: FakeEngine | None = None

    async def open_runtime_async(self, *, run_id: str, sink: NativeCliDatabaseEvidenceSink) -> _DockerRuntime:
        self.gateway = _FakeGateway(run_id=run_id, observer=sink, trace=self.trace)
        lease, _runner, fake, engine, approved = make_agent(run_config=self.run_config, route=self.gateway.route)
        assert approved == self.run_config
        fake.wire = (wire_frame(1, _stdout()), wire_frame(2, b"OFFLINE progress\n"))
        fake.on_start = self.gateway.send_inert_request_async if self.send_gateway else None
        self.engine, self.fake_engine = engine, fake
        await lease.acquire_async()
        self.runtime = _DockerRuntime(
            lease=lease,
            gateway=self.gateway,
            fake_engine=fake,
            memory=self.memory,
            trace=self.trace,
        )
        return self.runtime


async def test_controller_uses_real_stop_only_lease_and_one_atomic_score_async(sqlite_instance: SQLiteMemory) -> None:
    binding = _DockerBinding(memory=sqlite_instance)
    evaluation = NativeCliEvaluation(binding=binding, instruction="Inert task", memory=sqlite_instance)
    try:
        result = await evaluation.run_async()
        assert result.pregrading is not None and result.pregrading.required_complete
        assert result.report.status is NativeCliReportStatus.COMPLETED
        assert result.score.status is ScoreStatus.COMPLETE and result.score.get_value() == 0.75
        assert result.episode.score_id == result.score.id and result.episode.coverage_complete
        assert binding.runtime is not None and binding.fake_engine is not None
        stop = binding.runtime.lease.agent_stop
        assert stop is not None and stop.stopped and stop.exec_id == binding.fake_engine.exec_id
        assert binding.runtime.grading_calls == 1
        assert binding.trace.index("gateway_response") < binding.trace.index("original_grader")
        assert binding.runtime.lease.snapshot().state is EnvironmentLeaseState.CLOSED
        assert not binding.runtime.gateway_listener.is_locally_running
        assert len(await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")) == 1
    finally:
        if binding.engine is not None:
            await binding.engine.close_async()


async def test_controller_skips_grader_when_real_stop_lease_has_no_model_capture_async(
    sqlite_instance: SQLiteMemory,
) -> None:
    binding = _DockerBinding(memory=sqlite_instance, send_gateway=False)
    evaluation = NativeCliEvaluation(binding=binding, instruction="Inert task", memory=sqlite_instance)
    try:
        result = await evaluation.run_async()
        assert result.report.status is not NativeCliReportStatus.COMPLETED
        assert result.score.is_undetermined and not result.episode.coverage_complete
        assert result.pregrading is not None and not result.pregrading.required_complete
        assert binding.runtime is not None and binding.runtime.grading_calls == 0
        assert binding.runtime.lease.agent_stop is not None and binding.runtime.lease.agent_stop.stopped
        assert binding.runtime.lease.snapshot().state is EnvironmentLeaseState.CLOSED
        assert len(await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")) == 1
    finally:
        if binding.engine is not None:
            await binding.engine.close_async()
