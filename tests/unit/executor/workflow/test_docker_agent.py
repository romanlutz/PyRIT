# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from pathlib import PurePosixPath
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from pyrit.executor.workflow.docker_agent import (
    AgentArtifactPolicy,
    DockerAgentCliProfile,
    DockerSandboxLauncher,
    DockerStopOnlyAgentLease,
)
from pyrit.executor.workflow.docker_compose import ComposeEnvironmentSpec, ComposeServiceSpec
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.executor.workflow.docker_guest_auth import (
    DockerGuestAuth,
    codex_gateway_config,
    codex_gateway_template_sha256,
)
from pyrit.prompt_target.gateway.responses_contract import GatewayRoute
from pyrit.prompt_target.native_cli_models import (
    NativeCliEvent,
    NativeCliProtocol,
    NativeCliRawChunk,
    NativeCliRunConfig,
    NativeCliStream,
)
from pyrit.prompt_target.native_cli_transport import NativeCliRunner
from tests.unit.executor.workflow.test_docker_compose import FakeDockerRunner, service_spec
from tests.unit.executor.workflow.test_docker_engine import FakeEngine, wire_frame


def config(
    *, timeout: float = 2, protocol: NativeCliProtocol = NativeCliProtocol.CODEX_EXEC_JSON
) -> NativeCliRunConfig:
    return NativeCliRunConfig(
        protocol=protocol,
        cli_version="0.115.0",
        cli_profile="inert-stop-only",
        agent_workdir=PurePosixPath("/tmp/work"),
        model_gateway_endpoint="http://gateway.sandbox/v1",
        max_steps=2,
        timeout_seconds=timeout,
        max_frame_bytes=4096,
    )


def codex_output() -> bytes:
    return b"".join(
        json.dumps(value, separators=(",", ":")).encode() + b"\n"
        for value in (
            {"type": "thread.started", "thread_id": "inert-thread"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "inert result"}},
            {"type": "turn.completed"},
        )
    )


class Recorder:
    def __init__(self, *, fail_raw: bool = False) -> None:
        self.raw: list[NativeCliRawChunk] = []
        self.events: list[NativeCliEvent] = []
        self.fail_raw = fail_raw

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        if self.fail_raw:
            raise OSError("Inert raw sink failed.")
        self.raw.append(chunk)

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        self.events.append(event)


def make_agent(
    *,
    count: int = 2,
    run_config: NativeCliRunConfig | None = None,
    stop_timeout: float = 0.5,
    route: GatewayRoute | None = None,
) -> tuple[DockerStopOnlyAgentLease, FakeDockerRunner, FakeEngine, DockerEngineClient, NativeCliRunConfig]:
    approved = run_config or config()
    route = route or GatewayRoute(run_id="inert-run", model="inert-model", guest_token="guest-only-" + "g" * 40)
    auth = DockerGuestAuth.from_route(
        route=route, protocol=approved.protocol, gateway_endpoint=approved.model_gateway_endpoint
    )
    agent = service_spec("agent", role="agent")
    services = (
        agent,
        *(service_spec(f"target-{index}", role="target" if index == 1 else "grader") for index in range(1, count)),
    )
    spec = ComposeEnvironmentSpec(services=services)
    command = FakeDockerRunner()
    fake = FakeEngine()
    fake.wire = (wire_frame(1, codex_output()), wire_frame(2, b"diagnostic\xff\r\n"))
    engine = fake.client(control_timeout_seconds=0.25)
    environment = ["PATH=/usr/bin", "HOME=/tmp/home", "TMPDIR=/tmp"]
    config_digest = (
        hashlib.sha256(
            codex_gateway_config(model=route.model, base_url=approved.model_gateway_endpoint).encode()
        ).hexdigest()
        if approved.protocol is NativeCliProtocol.CODEX_EXEC_JSON
        else None
    )
    labels = (
        {
            "org.pyrit.native.codex-config-template-sha256": codex_gateway_template_sha256(),
            "org.pyrit.native.codex-user-config-path": "/tmp/home/.codex/config.toml",
        }
        if config_digest
        else {}
    )
    command.mutate_image = lambda image: image["Config"].update(Env=environment.copy(), Labels=labels.copy())

    def populate(runner: FakeDockerRunner) -> None:
        for index, container in enumerate(runner.containers.values(), 1):
            container["Config"]["Env"] = environment.copy()
            container["Config"]["Labels"].update(labels)
            container["State"].update(Pid=100 + index, Paused=False, Restarting=False, Dead=False)
        fake.containers = runner.containers
        fake.networks = runner.networks

    command.mutate_up = populate
    profile = DockerAgentCliProfile(
        image=agent.image,
        executable=PurePosixPath("/opt/pinned-cli"),
        config=approved,
        artifact_policy=AgentArtifactPolicy.TARGET_SIDE_ONLY,
        codex_config_sha256=config_digest,
    )
    lease = DockerStopOnlyAgentLease(
        run_id=route.run_id,
        spec=spec,
        runner=command,
        engine=engine,
        profile=profile,
        guest_auth=auth,
        stop_timeout_seconds=stop_timeout,
    )
    return lease, command, fake, engine, approved


def kills(fake: FakeEngine) -> list[str]:
    return [request.url.path.split("/")[-2] for request in fake.requests if request.url.path.endswith("/kill")]


@pytest.mark.parametrize("count", [2, 4])
async def test_real_runner_consumes_engine_bytes_and_stops_only_agent_before_grader_async(count: int) -> None:
    lease, command, fake, engine, approved = make_agent(count=count)
    allocation = await lease.acquire_async()
    target_before = {
        name: copy.deepcopy(fake.containers[identifier])
        for name, identifier in allocation.containers
        if name != "agent"
    }
    sink = Recorder()
    outcome = await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=sink).run_async(
        config=approved, prompt="literal $(not executed by host)\n--danger"
    )
    assert outcome.exit_code == 0 and outcome.coverage_complete
    assert b"".join(chunk.data for chunk in sink.raw if chunk.stream is NativeCliStream.STDOUT) == codex_output()
    assert b"".join(chunk.data for chunk in sink.raw if chunk.stream is NativeCliStream.STDERR) == b"diagnostic\xff\r\n"
    assert [chunk.sequence for chunk in sink.raw] == list(range(1, len(sink.raw) + 1))
    assert fake.created["Cmd"] == [
        "/opt/pinned-cli",
        "exec",
        "--json",
        "--",
        "literal $(not executed by host)\n--danger",
    ]
    assert {item.split("=", 1)[0] for item in fake.created["Env"]} == {"PYRIT_GUEST_MODEL_TOKEN", "PYRIT_RUN_ID"}
    assert fake.created["AttachStdin"] is False and fake.created["Privileged"] is False
    assert kills(fake) == [allocation.container_id("agent")]
    assert lease.agent_stop.stopped and fake.containers[allocation.container_id("agent")]["State"]["Pid"] == 0
    # The target-side grader is safe to run only after NativeCliRunner has returned.
    for name, identifier in lease.agent_stop.preserved_services:
        assert fake.containers[identifier] == target_before[name]
        assert fake.containers[identifier]["State"]["Running"] is True
    assert len(command.documents) == 1
    with pytest.raises(RuntimeError, match="unused"):
        await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="not a retained session")
    await lease.stop_agent_async()
    assert len(kills(fake)) == 1
    await lease.close_async()
    assert not command.containers and not command.networks
    await engine.close_async()


@pytest.mark.parametrize("exit_code", [1, 42, 126, 137, 255])
async def test_wait_uses_daemon_guest_exit_not_host_control_success_async(exit_code: int) -> None:
    lease, _, fake, engine, approved = make_agent()
    await lease.acquire_async()
    fake.exit_code = exit_code
    outcome = await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
        config=approved, prompt="fixture"
    )
    assert outcome.exit_code == exit_code and not outcome.coverage_complete
    assert lease.agent_stop.stopped
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize("defect", ["exit-missing", "exit-bool", "wrong-exec", "wrong-container", "different-argv"])
async def test_post_stream_exec_inspection_mismatch_never_returns_an_outcome_async(defect: str) -> None:
    lease, _, fake, engine, approved = make_agent()
    allocation = await lease.acquire_async()

    def change(value: dict[str, Any]) -> None:
        if not fake.started:
            return
        if defect == "exit-missing":
            value.pop("ExitCode")
        elif defect == "exit-bool":
            value["ExitCode"] = True
        elif defect == "wrong-exec":
            value["ID"] = "f" * 64
        elif defect == "wrong-container":
            value["ContainerID"] = allocation.container_id("target-1")
        else:
            value["ProcessConfig"]["arguments"] = ["forged"]

    fake.inspection_change = change
    with pytest.raises(DockerEngineError):
        await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
            config=approved, prompt="fixture"
        )
    assert kills(fake) == [allocation.container_id("agent")]
    assert fake.containers[allocation.container_id("target-1")]["State"]["Running"]
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize("status", [101, 500])
async def test_unsupported_engine_attach_still_stops_possibly_started_guest_async(status: int) -> None:
    lease, _, fake, engine, approved = make_agent()
    allocation = await lease.acquire_async()
    fake.start_status = status
    with pytest.raises(DockerEngineError, match="no fallback"):
        await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
            config=approved, prompt="fixture"
        )
    assert fake.started and lease.agent_stop.stopped
    assert kills(fake) == [allocation.container_id("agent")]
    assert fake.last_stream.closed
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize("failure", ["read", "sink", "timeout", "cancel"])
async def test_failure_paths_verify_guest_stop_and_preserve_target_async(failure: str) -> None:
    approved = config(timeout=0.15 if failure == "timeout" else 2)
    lease, _, fake, engine, _ = make_agent(run_config=approved, stop_timeout=0.1)
    allocation = await lease.acquire_async()
    fake.pause_stream = failure in {"timeout", "cancel"}
    if failure == "read":
        fake.stream_failure = httpx.ReadError("inert disconnected attach")
    sink = Recorder(fail_raw=failure == "sink")
    task = asyncio.create_task(
        NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=sink).run_async(
            config=approved, prompt="fixture"
        )
    )
    if failure == "cancel":
        await asyncio.wait_for(fake.start_entered.wait(), timeout=1)
        task.cancel()
    expected = {"read": DockerEngineError, "sink": OSError, "timeout": TimeoutError, "cancel": asyncio.CancelledError}[
        failure
    ]
    with pytest.raises(expected):
        await task
    assert lease.agent_stop.stopped and kills(fake) == [allocation.container_id("agent")]
    assert fake.containers[allocation.container_id("target-1")]["State"]["Running"]
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize("phase", ["create", "start"])
async def test_cancelled_launch_before_return_owns_guest_cleanup_async(phase: str) -> None:
    lease, _, fake, engine, approved = make_agent()
    allocation = await lease.acquire_async()
    gate = asyncio.Event()
    if phase == "create":
        fake.create_gate = gate
        entered = fake.create_entered
    else:
        fake.start_gate = gate
        entered = fake.start_entered
    task = asyncio.create_task(DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="fixture"))
    await asyncio.wait_for(entered.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lease.agent_stop.stopped
    assert kills(fake) == [allocation.container_id("agent")]
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize(
    "defect", ["wrong-owner", "privileged", "mount", "restart-policy", "wrong-image", "extra-network"]
)
async def test_prelaunch_ownership_and_security_drift_prevents_engine_exec_async(defect: str) -> None:
    lease, _, fake, engine, approved = make_agent()
    allocation = await lease.acquire_async()
    container = fake.containers[allocation.container_id("agent")]
    original = copy.deepcopy(container)
    if defect == "wrong-owner":
        container["Config"]["Labels"]["org.pyrit.native.run"] = "another-run"
    elif defect == "privileged":
        container["HostConfig"]["Privileged"] = True
    elif defect == "mount":
        container["Mounts"] = [{"Type": "bind", "Source": "/host", "Destination": "/tmp"}]
    elif defect == "restart-policy":
        container["HostConfig"]["RestartPolicy"]["Name"] = "always"
    elif defect == "wrong-image":
        container["Image"] = "sha256:" + "c" * 64
    else:
        container["NetworkSettings"]["Networks"]["bridge"] = {"NetworkID": "d" * 64}
    with pytest.raises(DockerEngineError):
        await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="fixture")
    assert fake.created is None and not kills(fake)
    fake.containers[allocation.container_id("agent")] = original
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize(
    "defect", ["kill-http-error", "still-running", "target-died", "pid-remains", "paused", "foreign-network"]
)
async def test_stop_failure_is_latched_and_blocks_successful_runner_return_async(defect: str) -> None:
    lease, _, fake, engine, approved = make_agent(stop_timeout=0.1)
    allocation = await lease.acquire_async()
    if defect == "kill-http-error":
        fake.kill_status = 500
    elif defect == "still-running":
        fake.kill_keeps_running = True
    elif defect == "target-died":
        fake.after_kill = lambda: fake.containers[allocation.container_id("target-1")]["State"].update(
            Running=False, Pid=0
        )
    elif defect == "pid-remains":
        fake.after_kill = lambda: fake.containers[allocation.container_id("agent")]["State"].update(Pid=123)
    elif defect == "paused":
        fake.after_kill = lambda: fake.containers[allocation.container_id("agent")]["State"].update(Paused=True)
    else:
        fake.after_kill = lambda: fake.networks[allocation.network_id]["Containers"].update({"e" * 64: {}})
    with pytest.raises((DockerEngineError, TimeoutError)):
        await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
            config=approved, prompt="fixture"
        )
    assert lease.agent_stop is not None and not lease.agent_stop.stopped
    request_count = len(fake.requests)
    with pytest.raises((DockerEngineError, TimeoutError)):
        await lease.stop_agent_async()
    assert len(fake.requests) == request_count
    await engine.close_async()


async def test_close_is_serialized_behind_agent_stop_without_killing_targets_early_async() -> None:
    lease, command, fake, engine, approved = make_agent()
    allocation = await lease.acquire_async()
    fake.kill_gate = asyncio.Event()
    running = asyncio.create_task(
        NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
            config=approved, prompt="fixture"
        )
    )
    await asyncio.wait_for(fake.kill_entered.wait(), timeout=1)
    closing = asyncio.create_task(lease.close_async())
    await asyncio.sleep(0)
    assert len(command.documents) == 1
    assert fake.containers[allocation.container_id("target-1")]["State"]["Running"]
    fake.kill_gate.set()
    outcome = await running
    await closing
    assert outcome.exit_code == 0 and lease.agent_stop.stopped
    assert len(command.documents) == 2 and not command.containers
    await engine.close_async()


async def test_repeated_cancellation_does_not_skip_the_stop_barrier_async() -> None:
    lease, _, fake, engine, approved = make_agent()
    await lease.acquire_async()
    session = await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="fixture")
    fake.kill_gate = asyncio.Event()
    stopping = asyncio.create_task(session.stop_async())
    await asyncio.wait_for(fake.kill_entered.wait(), timeout=1)
    stopping.cancel()
    await asyncio.sleep(0)
    stopping.cancel()
    fake.kill_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await stopping
    assert lease.agent_stop.stopped
    await session.stop_async()
    assert len(kills(fake)) == 1
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize(
    "roles",
    [
        (frozenset({"agent", "target"}), frozenset({"target"})),
        (frozenset({"agent", "grader"}), frozenset({"target"})),
        (frozenset({"agent"}), frozenset({"agent"})),
        (frozenset({"target"}), frozenset({"grader"})),
        (frozenset({"agent"}), frozenset({"unrelated"})),
    ],
)
async def test_role_collisions_are_rejected_before_any_io_async(roles: tuple[frozenset[str], ...]) -> None:
    services = tuple(
        ComposeServiceSpec.model_validate({**service_spec(f"s{index}").model_dump(), "roles": role})
        for index, role in enumerate(roles)
    )
    fake, command = FakeEngine(), FakeDockerRunner()
    engine = fake.client()
    profile = DockerAgentCliProfile(
        image=services[0].image,
        executable=PurePosixPath("/opt/cli"),
        config=config(),
        artifact_policy=AgentArtifactPolicy.TARGET_SIDE_ONLY,
        codex_config_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="agent-only"):
        DockerStopOnlyAgentLease(
            run_id="run",
            spec=ComposeEnvironmentSpec(services=services),
            runner=command,
            engine=engine,
            profile=profile,
            guest_auth=DockerGuestAuth.from_route(
                route=GatewayRoute(run_id="run", model="inert", guest_token="g" * 40),
                protocol=NativeCliProtocol.CODEX_EXEC_JSON,
                gateway_endpoint=config().model_gateway_endpoint,
            ),
        )
    assert not fake.requests and not command.calls
    await engine.close_async()


def test_freeze_and_post_stop_agent_artifacts_are_not_silently_enabled() -> None:
    with pytest.raises(NotImplementedError, match="target-side"):
        DockerAgentCliProfile(
            image=service_spec("agent").image,
            executable=PurePosixPath("/opt/cli"),
            config=config(),
            artifact_policy=AgentArtifactPolicy.FROZEN_AGENT_ARTIFACTS,
        )


@pytest.mark.parametrize("field", ["cli_version", "cli_profile", "model_gateway_endpoint", "agent_workdir"])
async def test_config_cannot_replace_the_pinned_launch_profile_async(field: str) -> None:
    lease, _, fake, engine, approved = make_agent()
    await lease.acquire_async()
    values = {
        "cli_version": "99.0.0",
        "cli_profile": "different",
        "model_gateway_endpoint": "http://other.invalid/v1",
        "agent_workdir": PurePosixPath("/tmp/other"),
    }
    with pytest.raises(ValueError, match="exactly match"):
        await DockerSandboxLauncher(lease=lease).launch_async(
            config=replace(approved, **{field: values[field]}), prompt="fixture"
        )
    assert fake.created is None and not kills(fake)
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize(
    "environment",
    [
        [
            "PATH=/usr/bin",
            "HOME=/tmp/home",
            "OPENAI_BASE_URL=http://gateway.sandbox/v1",
            "OPENAI_API_KEY=not-a-real-key",
        ],
        ["PATH=/usr/bin", "HOME=/root", "OPENAI_BASE_URL=http://gateway.sandbox/v1"],
        ["PATH=/usr/bin", "HOME=/tmp/home", "OPENAI_BASE_URL=http://other.invalid/v1"],
        ["PATH=/usr/bin", "HOME=/tmp/home", "HOME=/tmp/second", "OPENAI_BASE_URL=http://gateway.sandbox/v1"],
    ],
)
async def test_primary_credentials_or_unapproved_home_and_routing_fail_before_exec_async(
    environment: list[str],
) -> None:
    lease, command, fake, engine, _ = make_agent()
    populate = command.mutate_up
    command.mutate_image = lambda image: image["Config"].update(Env=environment.copy())

    def changed(runner: FakeDockerRunner) -> None:
        assert populate is not None
        populate(runner)
        for container in runner.containers.values():
            container["Config"]["Env"] = environment.copy()

    command.mutate_up = changed
    with pytest.raises(DockerEngineError):
        await lease.acquire_async()
    assert fake.created is None and not kills(fake)
    assert not command.containers
    await engine.close_async()


async def test_compose_and_engine_must_observe_the_same_verified_allocation_async() -> None:
    lease, command, fake, engine, _ = make_agent()
    fake.container_change = lambda observed: observed["Config"].update(User="0:0")
    with pytest.raises(DockerEngineError, match="same approved"):
        await lease.acquire_async()
    assert fake.created is None and not command.containers
    await engine.close_async()


async def test_launcher_does_not_spawn_any_host_cli_or_change_compose_security_async() -> None:
    lease, command, fake, engine, approved = make_agent()
    with patch("asyncio.create_subprocess_exec", side_effect=AssertionError("No host subprocess in Engine launcher.")):
        await lease.acquire_async()
        await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
            config=approved, prompt="fixture"
        )
        await lease.close_async()
    manifest = command.documents[0]
    for spec in manifest["services"].values():
        assert spec["read_only"] and not spec["privileged"] and spec["cap_drop"] == ["ALL"]
        assert not set(spec) & {"volumes", "environment", "ports", "devices"}
    assert all(request.url.host == "docker-engine.invalid" for request in fake.requests)
    await engine.close_async()


async def test_eof_without_observed_guest_exit_times_out_then_stops_agent_async() -> None:
    approved = config(timeout=0.15)
    lease, _, fake, engine, _ = make_agent(run_config=approved, stop_timeout=0.1)
    allocation = await lease.acquire_async()
    with patch.object(fake, "_exited"):
        with pytest.raises(TimeoutError):
            await NativeCliRunner(launcher=DockerSandboxLauncher(lease=lease), sink=Recorder()).run_async(
                config=approved, prompt="fixture"
            )
    assert kills(fake) == [allocation.container_id("agent")]
    assert lease.agent_stop.stopped
    await lease.close_async()
    await engine.close_async()


async def test_already_stopped_agent_is_observed_without_inventing_a_second_kill_async() -> None:
    lease, _, fake, engine, approved = make_agent()
    allocation = await lease.acquire_async()
    session = await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="fixture")
    assert session.exec_id == fake.exec_id and session.container_id == allocation.container_id("agent")
    _ = [part async for part in session.read_chunks_async()]
    assert await session.wait_async() == 0
    fake.containers[allocation.container_id("agent")]["State"].update(Running=False, Status="exited", Pid=0)
    fake.containers[allocation.container_id("agent")]["NetworkSettings"]["Networks"] = {}
    fake.networks[allocation.network_id]["Containers"].pop(allocation.container_id("agent"))
    await session.stop_async()
    assert lease.agent_stop.stopped and not kills(fake)
    assert lease.agent_stop.exec_id == session.exec_id
    assert fake.containers[allocation.container_id("target-1")]["State"]["Running"]
    await lease.close_async()
    await engine.close_async()


async def test_stop_refuses_ownership_drift_after_exec_without_touching_another_service_async() -> None:
    lease, _, fake, engine, approved = make_agent()
    await lease.acquire_async()
    session = await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="fixture")
    _ = [part async for part in session.read_chunks_async()]
    assert await session.wait_async() == 0
    fake.container_change = lambda observed: observed["Config"]["Labels"].update({"org.pyrit.native.run": "foreign"})
    with pytest.raises(DockerEngineError, match="identity"):
        await session.stop_async()
    assert not lease.agent_stop.stopped and not kills(fake)
    assert fake.last_stream.closed
    fake.container_change = None
    await lease.close_async()
    await engine.close_async()


async def test_close_during_create_prevents_start_and_is_not_a_successful_session_async() -> None:
    lease, command, fake, engine, approved = make_agent()
    await lease.acquire_async()
    fake.create_gate = asyncio.Event()
    launching = asyncio.create_task(DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="fixture"))
    await asyncio.wait_for(fake.create_entered.wait(), timeout=1)
    closing = asyncio.create_task(lease.close_async())
    await asyncio.sleep(0)
    assert len(command.documents) == 1
    fake.create_gate.set()
    with pytest.raises(RuntimeError, match="release was requested"):
        await launching
    await closing
    assert not fake.started
    assert not command.containers and not command.networks
    await engine.close_async()


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
async def test_both_pinned_cli_jsonl_argv_profiles_keep_the_prompt_as_data_async(protocol: NativeCliProtocol) -> None:
    approved = config(protocol=protocol)
    lease, _, fake, engine, _ = make_agent(run_config=approved)
    await lease.acquire_async()
    session = await DockerSandboxLauncher(lease=lease).launch_async(
        config=approved, prompt="--model unapproved; echo host"
    )
    expected = (
        ["exec", "--json"]
        if protocol is NativeCliProtocol.CODEX_EXEC_JSON
        else ["--print", "--output-format", "stream-json", "--verbose"]
    )
    assert fake.created["Cmd"] == ["/opt/pinned-cli", *expected, "--", "--model unapproved; echo host"]
    await session.stop_async()
    await lease.close_async()
    await engine.close_async()
