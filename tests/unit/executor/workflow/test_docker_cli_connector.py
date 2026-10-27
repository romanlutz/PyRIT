# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Actual Docker/Engine/native components over explicit in-memory HTTP and control-plane fixtures."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import socket
import subprocess
import tarfile
from dataclasses import replace
from ipaddress import IPv4Address, IPv4Network
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pyrit.executor.workflow.docker_agent import AgentArtifactPolicy, DockerAgentCliProfile
from pyrit.executor.workflow.docker_cli_connector import (
    DockerCliGateway,
    DockerCliQualification,
    DockerTargetDeadline,
    DockerTargetOperation,
    DockerTargetPhase,
    OwnedDockerCliConnector,
    OwnedDockerCliRuntime,
    OwnedDockerRuntimeLease,
)
from pyrit.executor.workflow.docker_compose import ComposeEnvironmentSpec
from pyrit.executor.workflow.docker_named_workspace import DockerNamedWorkspaceSpec
from pyrit.executor.workflow.native_cyber_eval import NativeCyberEvaluation, NativeCyberTaskBinding
from pyrit.executor.workflow.native_harness import ClaudeCliHarnessAdapter
from pyrit.models.native_cyber import NativeCyberJudgment, NativeCyberRequest
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute
from pyrit.prompt_target.native_cli_models import NativeCliProtocol, NativeCliRunConfig
from tests.unit.executor.workflow.test_docker_compose import FakeDockerRunner, service_spec
from tests.unit.executor.workflow.test_docker_engine import FakeEngine, WireStream, wire_frame
from tests.unit.executor.workflow.test_native_cyber_eval import FixtureBinding
from tests.unit.memory.test_native_cli_evidence import _stdout
from tests.unit.prompt_target.gateway.messages_mocks import FakeMessagesBackend

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from typing import Any

    from starlette.types import ASGIApp

    from pyrit.executor.workflow.docker_cli_connector import DockerTargetResult
    from pyrit.memory import SQLiteMemory
    from pyrit.models.native_cyber import NativeCyberReadiness
    from pyrit.models.native_harness import NativeHarnessEvidence
    from pyrit.prompt_target.gateway.run_listener import GatewayBridgeBinding

pytestmark = pytest.mark.usefixtures("patch_central_database")


@pytest.fixture(autouse=True)
async def no_live_io_async() -> AsyncIterator[None]:
    original_send = httpx.AsyncClient.send

    async def send_async(client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any) -> httpx.Response:
        assert isinstance(client._transport, httpx.MockTransport), "Only in-memory Engine HTTP is allowed"
        assert request.url.host == "docker-engine.invalid"
        return await original_send(client, request, **kwargs)

    with (
        patch.object(socket.socket, "connect", side_effect=AssertionError("No live socket")),
        patch.object(socket.socket, "connect_ex", side_effect=AssertionError("No live socket")),
        patch.object(socket.socket, "sendto", side_effect=AssertionError("No live socket")),
        patch.object(socket, "getaddrinfo", side_effect=AssertionError("No live DNS")),
        patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess or CLI")),
        patch.object(os, "system", side_effect=AssertionError("No host shell")),
        patch.object(
            asyncio, "create_subprocess_exec", new_callable=AsyncMock, side_effect=AssertionError("No process")
        ),
        patch.object(
            asyncio, "create_subprocess_shell", new_callable=AsyncMock, side_effect=AssertionError("No process")
        ),
        patch.object(httpx.AsyncClient, "send", new=send_async),
        patch.object(httpx.Client, "send", side_effect=AssertionError("No live provider")),
        patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            new_callable=AsyncMock,
            side_effect=AssertionError("No real transport"),
        ),
    ):
        yield


class InertListener:
    """No sockets: retain the actual ASGI app for the in-memory agent Engine fixture."""

    def __init__(self, *, binding: GatewayBridgeBinding, app: ASGIApp, owner: ConnectorBinding) -> None:
        self.binding, self.app, self.owner = binding, app, owner
        self.is_locally_running = False
        self.start_count = self.close_count = 0

    @property
    def base_url(self) -> str:
        return f"http://{self.binding.address}:{self.binding.port}"

    async def start_async(self) -> None:
        self.start_count += 1
        if self.owner.defect == "listener-start":
            raise RuntimeError("Inert listener startup failed")
        self.is_locally_running = True
        self.owner.order.append("listener-start")

    async def close_async(self) -> None:
        self.close_count += 1
        self.is_locally_running = False
        self.owner.order.append("listener-close")
        if self.owner.defect == "listener-close":
            raise RuntimeError("Inert listener settlement failed")
        if self.owner.defect == "stop-receipt":
            lease = self.owner.runtime.lease
            lease._stop_observation = replace(lease.agent_stop, exec_id="e" * 64)


class ConnectorEngine(FakeEngine):
    """Multiple daemon-issued exec IDs, with separate real Engine process specifications."""

    def __init__(self, owner: ConnectorBinding) -> None:
        super().__init__()
        self.owner = owner
        self.execs: dict[str, dict[str, Any]] = {}
        self.target_state = b"pristine target fixture"
        self.agent_stream: WireStream | None = None
        self.agent_created = asyncio.Event()
        self.agent_stream_ready = asyncio.Event()
        self.volume_create_entered = asyncio.Event()
        self.volumes: dict[str, dict[str, Any]] = {}
        self.volume_files: dict[str, dict[str, bytes]] = {}
        self.volume_owners: dict[str, tuple[int, int, int]] = {}

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path.removeprefix("/v1.47")
        if path.startswith("/volumes") or path == "/containers/json":
            response = self._volume_control(request)
            if path == "/volumes/create":
                self.volume_create_entered.set()
                if self.owner.defect == "cancel-volume-create":
                    await asyncio.Event().wait()
            return response
        if path.startswith("/containers/") and path.endswith("/archive") and request.url.params["path"] == "/work":
            return self._workspace_archive(request)
        if path.startswith("/containers/") and path.endswith("/exec"):
            self.requests.append(request)
            identifier = f"{100 + len(self.execs):064x}"
            container = path.split("/")[2]
            body = json.loads(request.content)
            agent = self.containers[container]["Config"]["Labels"]["com.docker.compose.service"] == "agent"
            self.execs[identifier] = {
                "container": container,
                "body": body,
                "running": False,
                "started": False,
                "agent": agent,
            }
            if agent:
                self.agent_created.set()
            return httpx.Response(201, json={"Id": identifier})
        if path.startswith("/exec/"):
            self.requests.append(request)
            identifier = path.split("/")[2]
            state = self.execs[identifier]
            body = state["body"]
            if path.endswith("/json"):
                value = {
                    "ID": identifier,
                    "ContainerID": state["container"],
                    "Running": state["running"],
                    "ExitCode": 0,
                    "OpenStdin": False,
                    "OpenStdout": True,
                    "OpenStderr": True,
                    "ProcessConfig": {
                        "entrypoint": body["Cmd"][0],
                        "arguments": body["Cmd"][1:],
                        "user": body["User"],
                        "tty": False,
                        "privileged": False,
                    },
                }
                if not state["agent"] and body["Cmd"][1] == "read-facts" and self.owner.defect == "target-exit":
                    value["ExitCode"] = 41
                if state["agent"] and state["started"]:
                    if self.owner.defect == "wrong-exec":
                        value["ID"] = "d" * 64
                    elif self.owner.defect == "wrong-container":
                        value["ContainerID"] = self.owner.runtime.allocation.container_id("target")
                return httpx.Response(200, json=value)
            assert path.endswith("/start") and json.loads(request.content) == {"Detach": False, "Tty": False}
            state["started"] = state["running"] = True

            async def guest_work_async() -> None:
                if state["agent"]:
                    if self.owner.defect != "no-model":
                        await self._model_request_async(body)
                    if self.owner.workspace:
                        name = self.containers[state["container"]]["Mounts"][0]["Name"]
                        self.volume_files[name]["task.py"] = b"mutated by SYNTHETIC agent"
                        if self.owner.defect == "deleted-protected":
                            del self.volume_files[name]["protected.py"]
                        if self.owner.defect == "volume-drift":
                            self.volumes[name]["Labels"]["org.pyrit.native.run"] = "foreign"
                    else:
                        self.target_state = b"target state mutated by SYNTHETIC agent API action"
                elif body["Cmd"][1] == "read-facts" and self.owner.defect == "long-operation":
                    await asyncio.sleep(0.75)

            if state["agent"]:
                self.started = True
                self.owner.order.append("agent-start")
                raw = _stdout(protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE, tool=True)
                if self.owner.defect == "malformed":
                    raw += b"{invalid\n"
                wire = (wire_frame(1, raw), wire_frame(2, b"synthetic agent stderr\xff\n"))
            else:
                assert "Env" not in body
                self.owner.order.append("target-" + body["Cmd"][1])
                if body["Cmd"][1] == "setup":
                    if self.owner.workspace:
                        name = self.containers[state["container"]]["Mounts"][0]["Name"]
                        self.volume_files[name].update({"task.py": b"initial", "protected.py": b"protected fixture"})
                    else:
                        self.target_state = b"prepared target fixture"
                    result = b"prepared"
                else:
                    if self.owner.workspace:
                        name = self.containers[state["container"]]["Mounts"][0]["Name"]
                        result = json.dumps(
                            {key: value.decode() for key, value in self.volume_files[name].items()}, sort_keys=True
                        ).encode()
                    else:
                        result = self.target_state
                wire = (wire_frame(1, result), wire_frame(2, b""))

            def exited() -> None:
                state["running"] = False

            stream = WireStream(
                wire,
                on_eof=exited,
                on_read=guest_work_async,
                pause=(state["agent"] and self.owner.defect == "cancel")
                or (not state["agent"] and body["Cmd"][1] == "read-facts" and self.owner.defect == "target-timeout"),
            )
            if state["agent"]:
                self.agent_stream = stream
                self.agent_stream_ready.set()
            return httpx.Response(200, stream=stream, headers={"content-type": "application/vnd.docker.raw-stream"})
        return await super().handle_async(request)

    def _volume_control(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path.removeprefix("/v1.47")
        if path == "/containers/json":
            assert request.url.params["all"] == "1"
            name = json.loads(request.url.params["filters"])["volume"][0]
            ids = [
                identifier
                for identifier, item in self.containers.items()
                if any(mount.get("Name") == name for mount in item["Mounts"])
            ]
            if self.owner.defect == "foreign-attachment" and self.started:
                ids.append("d" * 64)
            return httpx.Response(200, json=[{"Id": value} for value in ids])
        if path == "/volumes/create":
            body = json.loads(request.content)
            assert body["Driver"] == "local" and body["DriverOpts"] == {}
            name = body["Name"]
            value = {
                "Name": name,
                "Driver": "local",
                "Scope": "local",
                "Options": None,
                "Labels": body["Labels"],
                "CreatedAt": "2026-10-26T00:00:00Z",
                "Mountpoint": "/daemon-private/never-a-bind-path/" + name,
            }
            if self.owner.defect == "volume-label":
                value["Labels"]["org.pyrit.native.run"] = "foreign"
            if self.owner.defect == "volume-env":
                value["Labels"]["org.pyrit.native.lease"] = "foreign"
            if self.owner.defect == "volume-name":
                value["Name"] = "foreign"
            if self.owner.defect == "volume-driver":
                value["Driver"] = "remote-plugin"
            if self.owner.defect == "volume-options":
                value["Options"] = {"device": "/host/not-approved"}
            self.volumes[name] = value
            self.volume_files[name] = {"unexpected.txt": b"not empty"} if self.owner.defect == "nonempty-volume" else {}
            self.volume_owners[name] = (0, 0, 0o755)
            self.owner.order.append("volume-create")
            return httpx.Response(500 if self.owner.defect == "ambiguous-create" else 201, json=value)
        name = path.split("/")[2]
        if self.owner.defect == "preexisting-volume" and not self.volumes:
            self.volumes[name] = {"Name": name, "Driver": "local", "Labels": {"foreign": "owner"}}
        if request.method == "GET":
            return (
                httpx.Response(200, json=self.volumes[name])
                if name in self.volumes
                else httpx.Response(404, json={"message": "absent"})
            )
        assert request.method == "DELETE" and request.url.query == b"force=false"
        assert not self.containers
        self.owner.order.append("volume-delete")
        if self.owner.defect == "volume-delete":
            return httpx.Response(500)
        if self.owner.defect == "delete-not-absent":
            return httpx.Response(204)
        del self.volumes[name]
        del self.volume_files[name]
        return httpx.Response(204)

    def _workspace_archive(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        identifier = request.url.path.split("/")[3]
        name = self.containers[identifier]["Mounts"][0]["Name"]
        if request.method == "PUT":
            assert request.url.params["copyUIDGID"] == "true" and request.url.params["noOverwriteDirNonDir"] == "true"
            with tarfile.open(fileobj=io.BytesIO(request.content), mode="r:") as archive:
                members = archive.getmembers()
            assert len(members) == 1 and members[0].name == "." and members[0].isdir()
            root = members[0]
            assert root.size == 0 and not root.linkname and not root.pax_headers
            self.volume_owners[name] = (root.uid, root.gid, root.mode)
            self.owner.order.append("workspace-root-init")
            return httpx.Response(200)
        assert request.method == "GET"
        output = io.BytesIO()
        uid, gid, mode = self.volume_owners[name]
        if self.owner.defect == "root-readback" and uid == 1000:
            mode = 0o755
        with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            root = tarfile.TarInfo("work")
            root.type, root.uid, root.gid, root.mode = tarfile.DIRTYPE, uid, gid, mode
            archive.addfile(root)
            for path, data in self.volume_files[name].items():
                member = tarfile.TarInfo("work/" + path)
                member.uid, member.gid, member.mode, member.size = uid, gid, 0o600, len(data)
                archive.addfile(member, io.BytesIO(data))
        stat = {"name": "work", "mode": (1 << 31) | mode, "linkTarget": "", "size": 0}
        return httpx.Response(
            200,
            stream=WireStream((output.getvalue(),)),
            headers={
                "content-type": "application/x-tar",
                "x-docker-container-path-stat": base64.b64encode(json.dumps(stat).encode()).decode(),
            },
        )

    async def _model_request_async(self, body: dict[str, Any]) -> None:
        listener = self.owner.listener
        assert listener is not None and listener.is_locally_running
        environment = dict(value.split("=", 1) for value in body["Env"])
        header_name, run_id = environment["ANTHROPIC_CUSTOM_HEADERS"].split(": ", 1)
        assert environment["ANTHROPIC_BASE_URL"] == listener.base_url
        assert environment["ANTHROPIC_MODEL"] == self.owner.route.model
        raw = json.dumps(
            {
                "model": environment["ANTHROPIC_MODEL"],
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "synthetic model request"}],
            }
        ).encode()
        token = environment["ANTHROPIC_AUTH_TOKEN"]
        if self.owner.defect == "wrong-token":
            token = "foreign-" + "x" * 40
        if self.owner.defect == "wrong-run-header":
            run_id = "foreign-run"
        request = httpx.Request(
            "POST",
            listener.base_url + "/v1/messages",
            content=raw,
            headers={
                "Authorization": "Bearer " + token,
                header_name: run_id,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        async with httpx.ASGITransport(app=listener.app) as transport:
            response = await transport.handle_async_request(request)
            await response.aread()
        if self.owner.defect not in {"wrong-token", "wrong-run-header"}:
            assert response.status_code == 200
        else:
            assert response.status_code in {401, 403}


class ConnectorBinding(NativeCyberTaskBinding):
    create_host_storage_async = FixtureBinding.create_host_storage_async
    validate_host_storage_async = FixtureBinding.validate_host_storage_async

    def __init__(self, *, defect: str | None = None, workspace: bool = False) -> None:
        super().__init__(
            name="synthetic_target_state",
            version="1",
            description="OFFLINE fixture only",
            max_ttl_seconds=1800 if defect == "long-operation" else 180,
        )
        self.defect = defect
        self.workspace = workspace
        self.order: list[str] = []
        self.fake = ConnectorEngine(self)
        self.engine = self.fake.client(control_timeout_seconds=0.5)
        self.runner = FakeDockerRunner()
        self.fake.volumes = self.runner.volumes
        self.backend = FakeMessagesBackend()
        self.listener: InertListener | None = None
        self.runtime: OwnedDockerCliRuntime | None = None
        self.route: GatewayRoute | None = None
        self.grade_count = 0
        self.result = None
        self.spec = ComposeEnvironmentSpec(services=(service_spec("agent", role="agent"), service_spec("target")))
        self.config = NativeCliRunConfig(
            protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
            cli_version="2.1.220",
            cli_profile="synthetic-stop-only",
            agent_workdir=PurePosixPath("/work" if workspace else "/tmp/work"),
            model_gateway_endpoint="http://172.30.0.1:58321",
            max_steps=4,
            timeout_seconds=5,
        )
        self.profile = DockerAgentCliProfile(
            image=self.spec.services[0].image,
            executable=PurePosixPath("/opt/pinned-cli"),
            config=self.config,
            artifact_policy=AgentArtifactPolicy.TARGET_SIDE_ONLY,
        )
        self.qualification = DockerCliQualification(
            engine=self.engine,
            engine_identity="synthetic:engine-api-1.47-http200",
            profile=self.profile,
            backend_identity="synthetic:messages-backend-v1",
            backend=self.backend,
            qualification_ref="synthetic:qualified-only-for-inert-tests",
            ready=defect != "unqualified",
            simulated=True,
            workspace=(
                DockerNamedWorkspaceSpec(
                    root=PurePosixPath("/work"), qualification_ref="synthetic:volume-v1", ready=True
                )
                if workspace
                else None
            ),
        )
        self.runner.mutate_image = lambda image: image["Config"].update(Env=["PATH=/usr/bin", "HOME=/tmp/home"])
        self.runner.mutate_up = self._populate
        self.runner.mutate_down = self._released
        if defect == "cleanup":
            self.runner.down_returncode = 1
        if defect == "stop":
            self.fake.kill_status = 500
        if defect == "target-died":

            def target_died() -> None:
                target = self.runtime.allocation.container_id("target")
                self.fake.containers[target]["State"].update(Running=False, Status="exited", Pid=0)

            self.fake.after_kill = target_died

    def _populate(self, runner: FakeDockerRunner) -> None:
        for index, container in enumerate(runner.containers.values()):
            container["Config"]["Env"] = ["PATH=/usr/bin", "HOME=/tmp/home"]
            container["State"].update(Pid=100 + index, Paused=False, Restarting=False, Dead=False)
            if self.workspace:
                name = runner.documents[0]["volumes"]["task_workspace"]["name"]
                container["HostConfig"]["Mounts"] = [
                    {
                        "Type": "volume",
                        "Source": name,
                        "Target": "/work",
                        "ReadOnly": False,
                        "VolumeOptions": {"NoCopy": True},
                    }
                ]
                container["Mounts"] = [
                    {
                        "Type": "volume",
                        "Name": name,
                        "Source": "/daemon-private/never-a-bind-path/" + name,
                        "Destination": "/work",
                        "Driver": "local",
                        "Mode": "",
                        "RW": True,
                        "Propagation": "",
                    }
                ]
                if self.defect == "mount-path":
                    container["Mounts"][0]["Destination"] = "/foreign"
                if self.defect == "mount-name":
                    container["Mounts"][0]["Name"] = "foreign"
                if self.defect == "mount-mode":
                    container["Mounts"][0]["RW"] = False
        for network in runner.networks.values():
            plan = runner.documents[0]["networks"]["challenge"]["ipam"]
            network["IPAM"] = {
                "Driver": plan["driver"],
                "Config": [{"Subnet": item["subnet"], "Gateway": item["gateway"]} for item in plan["config"]],
            }
            if self.defect == "wrong-network":
                network["IPAM"]["Config"][0]["Gateway"] = "172.30.0.2"
        self.fake.containers, self.fake.networks = runner.containers, runner.networks
        self.order.append("created-two-services")

    def _released(self, runner: FakeDockerRunner) -> None:
        self.fake.containers, self.fake.networks = runner.containers, runner.networks
        for state in self.fake.execs.values():
            if state["container"] not in runner.containers:
                state["running"] = False
        if not runner.containers:
            self.order.append("owned-cleanup")

    async def readiness_async(self) -> NativeCyberReadiness:
        return self.qualification.readiness()

    def _listener_factory(self, *, route: GatewayRoute, binding: GatewayBridgeBinding, app: ASGIApp) -> InertListener:
        assert route is self.route
        if self.defect == "foreign-listener":
            binding = replace(binding, run_id="foreign-run")
        self.listener = InertListener(binding=binding, app=app, owner=self)
        return self.listener

    async def _grade_async(
        self, *, runtime: OwnedDockerCliRuntime, evidence: NativeHarnessEvidence
    ) -> NativeCyberJudgment:
        self.grade_count += 1
        assert evidence.quiescent and evidence.coverage_complete
        assert runtime.lease.agent_stop.stopped and self.listener.close_count == 1
        self.result = await runtime.run_target_operation_async("facts")
        assert self.result.exit_code == 0
        grade = 0.625
        if self.workspace:
            identity = await runtime.workspace_identity_async()
            self.workspace_identity = identity
            data = json.loads(self.result.stdout)
            assert data["task.py"] == "mutated by SYNTHETIC agent"
            assert identity.task_root == "/work" and identity.run_id == runtime.run_id
            grade = 0.0 if "protected.py" not in data else 0.625
        else:
            assert self.result.stdout == self.fake.target_state
            assert self.result.stdout != b"prepared target fixture"
        target = await runtime.inspect_target_async()
        assert target.running is True and target.pid > 0
        self.order.append("original-fixture-grade")
        if self.defect == "postgrade-target-died":
            target_id = runtime.allocation.container_id("target")
            self.fake.containers[target_id]["State"].update(Running=False, Status="exited", Pid=0)
        return NativeCyberJudgment(
            value=grade,
            complete=True,
            rationale="SYNTHETIC target-service-state fixture, not JWT file grading",
            evidence={
                "fixture": True,
                "agent_exec_id": runtime.agent_exec_id,
                "target_exec_id": self.result.exec_id,
                "target_sha256": hashlib.sha256(self.result.stdout).hexdigest(),
            },
        )

    def create_environment_lease(self, *, run_id: str, request: NativeCyberRequest) -> OwnedDockerRuntimeLease:
        self.route = GatewayRoute(
            run_id=run_id, model="claude-offline-model", guest_token="synthetic-guest-" + "g" * 40
        )
        gateway = DockerCliGateway(
            route=self.route,
            limits=GatewayLimits(),
            backend=self.backend,
            backend_identity=self.qualification.backend_identity,
            address=IPv4Address("172.30.0.1"),
            subnet=IPv4Network("172.30.0.0/24"),
            port=58321,
            listener_factory=self._listener_factory,
        )
        self.runtime = OwnedDockerCliRuntime(
            run_id=run_id,
            spec=self.spec,
            runner=self.runner,
            qualification=self.qualification,
            gateway=gateway,
            grader=self._grade_async,
            stop_timeout_seconds=0.5,
            require_agent_workspace=self.workspace,
            target_operations=(
                DockerTargetOperation(
                    name="setup",
                    argv=("/opt/taskctl", "setup"),
                    working_directory=PurePosixPath("/tmp"),
                    phase=DockerTargetPhase.SETUP,
                    timeout_seconds=2,
                ),
                DockerTargetOperation(
                    name="facts",
                    argv=("/opt/taskctl", "read-facts"),
                    working_directory=PurePosixPath("/tmp"),
                    phase=DockerTargetPhase.AFTER_AGENT_STOP,
                    timeout_seconds=0.1
                    if self.defect == "target-timeout"
                    else 600
                    if self.defect == "long-operation"
                    else 2,
                ),
            ),
        )
        return OwnedDockerRuntimeLease(runtime=self.runtime, prepare_async=self._prepare_async)

    async def _prepare_async(self, runtime: OwnedDockerCliRuntime) -> None:
        result = await runtime.run_target_operation_async("setup")
        assert result.exit_code == 0


@pytest.fixture
async def binding_async() -> AsyncIterator[ConnectorBinding]:
    binding = ConnectorBinding()
    try:
        yield binding
    finally:
        await binding.engine.close_async()


async def test_actual_owned_provider_engine_launcher_gateway_and_score_composition_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, binding_async: ConnectorBinding
) -> None:
    binding = binding_async
    run = NativeCyberEvaluation(
        binding=binding,
        request=NativeCyberRequest(instruction="SYNTHETIC target API task"),
        directory=tmp_path,
        harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
    )
    await run.start_async()
    assert run.status == "completed", run.report
    assert run.score.get_value() == 0.625 and binding.grade_count == 1
    assert binding.order == [
        "created-two-services",
        "target-setup",
        "listener-start",
        "agent-start",
        "listener-close",
        "target-read-facts",
        "original-fixture-grade",
        "owned-cleanup",
    ]
    assert binding.runtime.lease.snapshot().state == "closed"
    assert binding.runtime.lease.agent_stop.exec_id == binding.runtime.agent_exec_id
    assert binding.runtime.lease.agent_stop.preserved_services == (
        ("target", binding.runtime.allocation.container_id("target")),
    )
    assert binding.result.container_id == binding.runtime.allocation.container_id("target")
    assert binding.result.exec_id != binding.runtime.agent_exec_id and binding.result.exit_code == 0
    assert len(binding.fake.execs) == 3 and not binding.runner.containers and not binding.runner.networks
    assert (
        sum("up" in command for command in binding.runner.calls)
        == sum("down" in command for command in binding.runner.calls)
        == 1
    )
    assert len(binding.backend.requests) == 1
    document = binding.runner.documents[0]
    assert document["networks"]["challenge"]["internal"] is True
    assert all(
        not set(service) & {"volumes", "ports", "environment", "privileged_mode"}
        for service in document["services"].values()
    )
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert episode.coverage_complete and episode.report_sha256 == run.report.sha256()
    assert episode.score_id == run.score.id and len(episode.turns) == 1
    assert episode.turns[0].request_piece_ids and episode.turns[0].response_piece_ids
    assert not binding.runtime.agent_workspace_available_after_stop
    await run.finish_async()
    assert binding.grade_count == 1


@pytest.mark.parametrize(
    "defect",
    [
        "no-model",
        "malformed",
        "wrong-exec",
        "wrong-container",
        "wrong-network",
        "wrong-token",
        "wrong-run-header",
        "stop",
        "target-died",
        "stop-receipt",
        "listener-start",
        "listener-close",
        "foreign-listener",
        "cleanup",
        "target-exit",
        "target-timeout",
        "postgrade-target-died",
    ],
)
async def test_concrete_binding_failures_do_not_publish_complete_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, defect: str
) -> None:
    binding = ConnectorBinding(defect=defect)
    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC task"),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        await run.start_async()
        assert run.score is None or run.score.is_undetermined
        assert binding.grade_count == int(
            defect in {"cleanup", "target-exit", "target-timeout", "postgrade-target-died"}
        )
        if defect == "postgrade-target-died":
            assert run.report.judgment.value == 0.625
        assert not any(
            not score.is_undetermined
            for score in await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
        )
        if defect in {"wrong-token", "wrong-run-header"}:
            assert not binding.backend.requests
        if defect == "foreign-listener":
            assert binding.listener.close_count == 0
        if defect != "cleanup":
            assert not binding.runner.containers and not binding.runner.networks
    finally:
        await binding.engine.close_async()


@pytest.mark.parametrize("workspace", [False, True])
async def test_actual_engine_stream_cancellation_stops_agent_before_owned_project_cleanup_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, workspace: bool
) -> None:
    binding = ConnectorBinding(defect="cancel", workspace=workspace)
    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC task"),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        running = asyncio.create_task(run.start_async())
        await asyncio.wait_for(binding.fake.agent_created.wait(), timeout=5)
        await asyncio.wait_for(binding.fake.agent_stream_ready.wait(), timeout=5)
        await asyncio.wait_for(binding.fake.agent_stream.paused.wait(), timeout=5)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert binding.grade_count == 0 and run.score.is_undetermined
        assert binding.runtime.lease.agent_stop.stopped and binding.fake.agent_stream.closed
        assert not binding.runner.containers and not binding.runner.networks
        assert not binding.fake.volumes
    finally:
        await binding.engine.close_async()


async def test_uncertain_volume_create_cancellation_preserves_cancel_and_quarantines_resource_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = ConnectorBinding(workspace=True, defect="cancel-volume-create")
    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC workspace"),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        task = asyncio.create_task(run.start_async())
        await asyncio.wait_for(binding.fake.volume_create_entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert binding.fake.volumes and not binding.runner.containers
        assert not any(request.method == "DELETE" for request in binding.fake.requests)
        assert binding.grade_count == 0 and not binding.backend.requests
        assert not await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
    finally:
        await binding.engine.close_async()


async def test_unqualified_binding_is_blocked_without_provider_or_model_io_async(*, tmp_path: Path) -> None:
    binding = ConnectorBinding(defect="unqualified")
    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC task"),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        await run.start_async()
        assert run.status == "blocked" and not binding.runner.calls and not binding.fake.requests
        assert not binding.backend.requests and binding.grade_count == 0
    finally:
        await binding.engine.close_async()


@pytest.mark.parametrize("capability", ["workspace", "codex", "backend", "backend-object", "route"])
async def test_unsupported_or_mismatched_connection_is_refused_before_allocation_async(capability: str) -> None:
    binding = ConnectorBinding()
    try:
        route = GatewayRoute(run_id="synthetic-run", model="claude-offline-model", guest_token="fake-only-" + "x" * 40)
        gateway = DockerCliGateway(
            route=route,
            limits=GatewayLimits(),
            backend=FakeMessagesBackend() if capability == "backend-object" else binding.backend,
            backend_identity="foreign" if capability == "backend" else binding.qualification.backend_identity,
            address=IPv4Address("172.30.0.1"),
            subnet=IPv4Network("172.30.0.0/24"),
            port=58321,
            listener_factory=binding._listener_factory,
        )
        qualification = binding.qualification
        if capability == "codex":
            config = replace(binding.config, protocol=NativeCliProtocol.CODEX_EXEC_JSON)
            profile = replace(binding.profile, config=config, codex_config_sha256="a" * 64)
            qualification = replace(qualification, profile=profile)
        expected = NotImplementedError if capability in {"workspace", "codex"} else ValueError
        with pytest.raises(expected):
            OwnedDockerCliRuntime(
                run_id="foreign-run" if capability == "route" else route.run_id,
                spec=binding.spec,
                runner=binding.runner,
                qualification=qualification,
                gateway=gateway,
                grader=binding._grade_async,
                stop_timeout_seconds=0.5,
                require_agent_workspace=capability == "workspace",
            )
        assert not binding.fake.requests and not binding.runner.calls
    finally:
        await binding.engine.close_async()


@pytest.mark.parametrize("deleted", [False, True])
async def test_named_workspace_keeps_actual_agent_mutations_and_deletions_for_original_grade_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, deleted: bool
) -> None:
    binding = ConnectorBinding(workspace=True, defect="deleted-protected" if deleted else None)
    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC shared task workspace"),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        await run.start_async()
        assert run.status == "completed", run.report
        assert binding.grade_count == 1 and run.score.get_value() == (0.0 if deleted else 0.625)
        identity = binding.workspace_identity
        assert identity.run_id == run.run_id and identity.environment_id == binding.runtime.environment_id
        assert identity.task_root == "/work" and identity.capability == "run-owned-task-volume-v1"
        assert len(identity.container_ids) == 2 and identity.volume_name == binding.runtime.lease.workspace_name
        assert identity.nonce in identity.volume_name
        result = json.loads(binding.result.stdout)
        assert result["task.py"] == "mutated by SYNTHETIC agent"
        assert ("protected.py" not in result) == deleted
        assert binding.order.index("volume-create") < binding.order.index("created-two-services")
        assert binding.order.index("agent-start") < binding.order.index("target-read-facts")
        assert binding.order.index("listener-close") < binding.order.index("original-fixture-grade")
        assert (
            binding.order.index("original-fixture-grade")
            < binding.order.index("owned-cleanup")
            < binding.order.index("volume-delete")
        )
        assert not binding.runner.containers and not binding.runner.networks and not binding.runner.volumes
        assert not binding.fake.volumes
        assert (
            len(
                [
                    request
                    for request in binding.fake.requests
                    if request.method == "POST" and request.url.path.endswith("/volumes/create")
                ]
            )
            == 1
        )
        deletes = [request for request in binding.fake.requests if request.method == "DELETE"]
        assert len(deletes) == 1 and deletes[0].url.path.endswith("/" + identity.volume_name)
        document = binding.runner.documents[0]
        assert document["volumes"] == {"task_workspace": {"external": True, "name": identity.volume_name}}
        for service in document["services"].values():
            assert service["volumes"] == [
                {
                    "type": "volume",
                    "source": "task_workspace",
                    "target": "/work",
                    "read_only": False,
                    "volume": {"nocopy": True},
                }
            ]
        assert "/daemon-private/" not in run.report.canonical_json()
        episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
        assert episode.coverage_complete and episode.score_id == run.score.id
        assert binding.runtime.lease.snapshot().state == "closed"
    finally:
        await binding.engine.close_async()


@pytest.mark.parametrize(
    "defect",
    [
        "preexisting-volume",
        "volume-label",
        "volume-env",
        "volume-name",
        "volume-driver",
        "volume-options",
        "ambiguous-create",
        "nonempty-volume",
        "root-readback",
        "mount-name",
        "mount-path",
        "mount-mode",
        "foreign-attachment",
        "volume-delete",
        "delete-not-absent",
        "volume-drift",
        "stop",
        "target-died",
        "cleanup",
    ],
)
async def test_named_volume_ownership_mount_and_release_failures_never_publish_complete_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, defect: str
) -> None:
    binding = ConnectorBinding(workspace=True, defect=defect)
    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC workspace"),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        try:
            await run.start_async()
        except PermissionError:
            assert run.score is None and run.report is None
        assert run.score is None or run.score.is_undetermined
        assert binding.grade_count == int(defect in {"volume-delete", "delete-not-absent", "cleanup"})
        assert not any(
            not score.is_undetermined
            for score in await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")
        )
        deletes = [request for request in binding.fake.requests if request.method == "DELETE"]
        if defect in {
            "preexisting-volume",
            "volume-label",
            "volume-env",
            "volume-name",
            "volume-driver",
            "volume-options",
            "ambiguous-create",
            "mount-name",
            "mount-path",
            "mount-mode",
            "foreign-attachment",
            "cleanup",
            "volume-drift",
        }:
            assert not deletes and binding.fake.volumes
        if defect == "preexisting-volume":
            assert not any(request.url.path.endswith("/volumes/create") for request in binding.fake.requests)
        if defect in {"volume-delete", "delete-not-absent"}:
            assert len(deletes) == 1 and not binding.runner.containers and binding.fake.volumes
    finally:
        await binding.engine.close_async()


@pytest.mark.parametrize("root", ["/", "/tmp/work", "/opt/taskctl", "/etc", "/work/../escape"])
async def test_workspace_scope_is_explicit_and_never_uses_credential_or_helper_paths_async(root: str) -> None:
    binding = ConnectorBinding(workspace=True)
    try:
        with pytest.raises(ValueError):
            workspace = DockerNamedWorkspaceSpec(root=PurePosixPath(root), qualification_ref="synthetic", ready=True)
            qualification = replace(binding.qualification, workspace=workspace)
            route = GatewayRoute(
                run_id="scope-fixture", model="claude-offline-model", guest_token="fake-only-" + "x" * 40
            )
            gateway = DockerCliGateway(
                route=route,
                limits=GatewayLimits(),
                backend=binding.backend,
                backend_identity=qualification.backend_identity,
                address=IPv4Address("172.30.0.1"),
                subnet=IPv4Network("172.30.0.0/24"),
                port=58321,
                listener_factory=binding._listener_factory,
            )
            OwnedDockerCliRuntime(
                run_id=route.run_id,
                spec=binding.spec,
                runner=binding.runner,
                qualification=qualification,
                gateway=gateway,
                grader=binding._grade_async,
                stop_timeout_seconds=0.5,
            )
        assert not binding.runner.calls and not binding.fake.requests
    finally:
        await binding.engine.close_async()


async def test_explicit_600_second_operation_uses_the_caller_deadline_not_the_engine_header_timeout_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory
) -> None:
    binding = ConnectorBinding(defect="long-operation")
    supplied: list[DockerTargetDeadline] = []
    observed_deadlines: list[float | None] = []
    original_operation = OwnedDockerCliRuntime.run_target_operation_async
    original_timeout_at = asyncio.timeout_at

    def timeout_at(when: float | None) -> asyncio.Timeout:
        observed_deadlines.append(when)
        return original_timeout_at(when)

    async def execute_async(self: OwnedDockerCliRuntime, name: str) -> DockerTargetResult:
        if name == "facts":
            deadline = DockerTargetDeadline.start(timeout_seconds=600)
            supplied.append(deadline)
            return await self.run_target_operation_until_async(name=name, deadline=deadline)
        return await original_operation(self, name)

    try:
        run = NativeCyberEvaluation(
            binding=binding,
            request=NativeCyberRequest(instruction="SYNTHETIC long approved task operation", ttl_seconds=1800),
            directory=tmp_path,
            harness=ClaudeCliHarnessAdapter(config=binding.config, connect=OwnedDockerCliConnector()),
        )
        with (
            patch.object(OwnedDockerCliRuntime, "run_target_operation_async", new=execute_async),
            patch.object(asyncio, "timeout_at", side_effect=timeout_at),
        ):
            await run.start_async()
        assert run.status == "completed" and binding.grade_count == 1
        assert len(supplied) == 1 and supplied[0].expires_at - supplied[0].started_at == 600
        assert observed_deadlines.count(supplied[0].expires_at) == 1
        assert asyncio.get_running_loop().time() - supplied[0].started_at >= 0.75
        assert binding.engine._control_timeout == 0.5
        assert (
            len(
                [
                    state
                    for state in binding.fake.execs.values()
                    if state["body"]["Cmd"] == ["/opt/taskctl", "read-facts"]
                ]
            )
            == 1
        )
        assert binding.result.exit_code == 0
    finally:
        await binding.engine.close_async()


@pytest.mark.parametrize("kind", ["short", "extended", "expired"])
async def test_short_extended_or_expired_caller_target_deadline_is_refused_before_engine_io_async(kind: str) -> None:
    binding = ConnectorBinding(defect="long-operation")
    lease = binding.create_environment_lease(
        run_id="deadline-fixture", request=NativeCyberRequest(instruction="fixture")
    )
    try:
        runtime = await lease.acquire_async()
        now = asyncio.get_running_loop().time()
        deadline = (
            DockerTargetDeadline(started_at=now - 601, expires_at=now - 1)
            if kind == "expired"
            else DockerTargetDeadline(started_at=now, expires_at=now + (60 if kind == "short" else 601))
        )
        before = len(binding.fake.requests)
        with pytest.raises(TimeoutError if kind == "expired" else ValueError):
            await runtime.run_target_operation_until_async(name="facts", deadline=deadline)
        assert len(binding.fake.requests) == before
        assert not any(state["agent"] for state in binding.fake.execs.values())
    finally:
        await lease.close_async()
        await binding.engine.close_async()


@pytest.mark.parametrize("seconds", [0, -1, 601, float("inf"), True])
def test_installed_operation_budget_remains_explicitly_bounded(seconds: float) -> None:
    with pytest.raises(ValueError):
        DockerTargetOperation(
            name="bounded",
            argv=("/opt/taskctl", "fixed"),
            working_directory=PurePosixPath("/tmp"),
            phase=DockerTargetPhase.SETUP,
            timeout_seconds=seconds,
        )
