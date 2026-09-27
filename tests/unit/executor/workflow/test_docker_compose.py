# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

from pyrit.executor.workflow.docker_command import DockerCommandResult
from pyrit.executor.workflow.docker_compose import (
    ComposeEnvironmentSpec,
    ComposeServiceSpec,
    ComposeTmpfsSpec,
    DockerComposeEnvironmentLease,
    DockerComposeError,
)
from pyrit.executor.workflow.environment_lease import EnvironmentCleanupError
from pyrit.models.environment_lease import EnvironmentCapability

if TYPE_CHECKING:
    from collections.abc import Callable


def service_spec(name: str, *, parent_name: str | None = None, role: str = "target") -> ComposeServiceSpec:
    return ComposeServiceSpec(
        name=name,
        roles=frozenset({role}),
        parent_name=parent_name,
        image="registry.invalid/task:approved@sha256:" + "a" * 64,
        command=("/app/start",),
        healthcheck=("/app/check",),
        uid=1000,
        gid=1000,
        cpu_millis=500,
        memory_bytes=134_217_728,
        pids_limit=64,
        tmpfs_bytes=16_777_216,
    )


class FakeDockerRunner:
    """In-memory Docker responses only; never invokes a CLI, socket, network or container."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.documents: list[dict[str, Any]] = []
        self.containers: dict[str, dict[str, Any]] = {}
        self.networks: dict[str, dict[str, Any]] = {}
        self.volumes: dict[str, dict[str, Any]] = {}
        self.mutate_up: Callable[[FakeDockerRunner], None] | None = None
        self.mutate_down: Callable[[FakeDockerRunner], None] | None = None
        self.up_returncode = 0
        self.down_returncode = 0
        self.bad_image = False
        self.mutate_image: Callable[[dict[str, Any]], None] | None = None
        self.bad_result: DockerCommandResult | None = None
        self.wait_in_up = False
        self.up_entered = asyncio.Event()

    async def run_async(
        self, *, arguments: tuple[str, ...], input_text: str | None = None, timeout_seconds: float
    ) -> DockerCommandResult:
        assert 0 < timeout_seconds <= 130
        self.calls.append(arguments)
        if self.bad_result is not None:
            return self.bad_result
        if arguments[0] == "compose":
            assert arguments[1:3] == ("--project-name", arguments[2])
            assert "--file" in arguments and arguments[arguments.index("--file") + 1] == "-"
            document = json.loads(input_text or "")
            self.documents.append(document)
            if "up" in arguments:
                assert "--no-build" in arguments and arguments[arguments.index("--pull") + 1] == "never"
                self._populate(document=document, project=arguments[2])
                if self.mutate_up:
                    self.mutate_up(self)
                self.up_entered.set()
                if self.wait_in_up:
                    await asyncio.Event().wait()
                return DockerCommandResult(stdout="", stderr="inert up", returncode=self.up_returncode)
            assert "down" in arguments and "--remove-orphans" in arguments
            assert "--volumes" not in arguments and "--rmi" not in arguments
            if self.down_returncode:
                if self.mutate_down:
                    self.mutate_down(self)
                return DockerCommandResult(stdout="", stderr="inert down", returncode=self.down_returncode)
            project = arguments[2]
            self.containers = {
                key: value
                for key, value in self.containers.items()
                if value["Config"]["Labels"].get("com.docker.compose.project") != project
            }
            self.networks = {
                key: value
                for key, value in self.networks.items()
                if value["Labels"].get("com.docker.compose.project") != project
            }
            if self.mutate_down:
                self.mutate_down(self)
            return DockerCommandResult(stdout="", stderr="", returncode=0)
        kind, operation = arguments[:2]
        if kind == "image":
            if self.bad_image:
                return DockerCommandResult(stdout="", stderr="No local image", returncode=1)
            image = {
                "Id": "sha256:" + "b" * 64,
                "RepoDigests": ["registry.invalid/task@sha256:" + "a" * 64],
                "Os": "linux",
                "Architecture": "amd64",
                "Config": {"Env": ["PATH=/usr/bin"], "Volumes": None},
            }
            if self.mutate_image:
                self.mutate_image(image)
            return DockerCommandResult(stdout=json.dumps([image]), stderr="", returncode=0)
        resources = {"container": self.containers, "network": self.networks, "volume": self.volumes}[kind]
        if operation == "ls":
            selector = arguments[arguments.index("--filter") + 1]
            found = []
            for identifier, value in resources.items():
                labels = value["Config"]["Labels"] if kind == "container" else value["Labels"]
                if selector.startswith("label="):
                    key, expected = selector[6:].split("=", 1)
                    matches = labels.get(key) == expected
                else:
                    matches = selector[5:] in value["Name"]
                if matches:
                    found.append(identifier)
            return DockerCommandResult(stdout="\n".join(found), stderr="", returncode=0)
        assert operation == "inspect"
        return DockerCommandResult(
            stdout=json.dumps([resources[key] for key in arguments[2:]]), stderr="", returncode=0
        )

    def _populate(self, *, document: dict[str, Any], project: str) -> None:
        network_spec = document["networks"]["challenge"]
        network_id = "f" * 64
        attached = {}
        for index, (name, spec) in enumerate(document["services"].items(), 1):
            identifier = f"{index:064x}"
            attached[identifier] = {"Name": spec["container_name"]}
            self.containers[identifier] = {
                "Id": identifier,
                "Name": "/" + spec["container_name"],
                "Image": "sha256:" + "b" * 64,
                "Config": {
                    "Labels": {
                        **spec["labels"],
                        "com.docker.compose.project": project,
                        "com.docker.compose.service": name,
                    },
                    "Image": spec["image"],
                    "Env": ["PATH=/usr/bin"],
                    "User": spec["user"],
                    "Cmd": spec["command"],
                    "Entrypoint": [],
                    "Healthcheck": {"Test": spec["healthcheck"]["test"]},
                },
                "HostConfig": {
                    "ReadonlyRootfs": spec["read_only"],
                    "Privileged": spec["privileged"],
                    "PublishAllPorts": False,
                    "NanoCpus": int(spec["cpus"] * 1_000_000_000),
                    "Memory": spec["mem_limit"],
                    "MemorySwap": spec["memswap_limit"],
                    "PidsLimit": spec["pids_limit"],
                    "NetworkMode": network_spec["name"],
                    "IpcMode": spec["ipc"],
                    "CapDrop": spec["cap_drop"],
                    "SecurityOpt": spec["security_opt"],
                    "Tmpfs": dict(mount.split(":", 1) for mount in spec["tmpfs"]),
                    "RestartPolicy": {"Name": "no"},
                },
                "Mounts": [],
                "NetworkSettings": {"Networks": {network_spec["name"]: {"NetworkID": network_id}}},
                "State": {"Running": True, "Status": "running", "Health": {"Status": "healthy"}},
            }
        self.networks[network_id] = {
            "Id": network_id,
            "Name": network_spec["name"],
            "Internal": network_spec["internal"],
            "Driver": "bridge",
            "EnableIPv6": False,
            "Scope": "local",
            "Options": {},
            "Labels": {
                **network_spec["labels"],
                "com.docker.compose.project": project,
                "com.docker.compose.network": "challenge",
            },
            "Containers": attached,
        }


def make_lease(count: int = 2) -> tuple[DockerComposeEnvironmentLease, FakeDockerRunner]:
    services = tuple(
        service_spec(
            f"service-{index}",
            parent_name=f"service-{index - 1}" if index else None,
            role="agent" if index == 0 else "target",
        )
        for index in range(count)
    )
    runner = FakeDockerRunner()
    return DockerComposeEnvironmentLease(
        run_id="run-1", spec=ComposeEnvironmentSpec(services=services), runner=runner
    ), runner


@pytest.mark.parametrize("count", [1, 2, 4])
async def test_compose_role_topologies_and_exact_owned_cleanup_async(count: int) -> None:
    lease, runner = make_lease(count)
    allocation = await lease.acquire_async()
    assert len(allocation.containers) == len(lease.snapshot().services) == count
    assert allocation.project_name == lease.project_name
    assert allocation.container_id("service-0") == "0" * 63 + "1"
    assert lease.capabilities == frozenset({EnvironmentCapability.HEALTH_CHECK})
    assert lease.snapshot().resources[0].resource.resource_id == lease.project_name
    assert all(health.healthy for health in lease.snapshot().health)
    document = runner.documents[0]
    assert document["networks"]["challenge"]["internal"] is True
    for spec in document["services"].values():
        assert spec["pull_policy"] == "never" and spec["entrypoint"] == []
        assert spec["cap_drop"] == ["ALL"] and spec["security_opt"] == ["no-new-privileges:true"]
        assert not set(spec) & {"volumes", "environment", "ports", "build", "devices", "cap_add", "network_mode"}
    await lease.close_async()
    await lease.close_async()
    assert runner.documents == [document, document]
    assert not runner.containers and not runner.networks and not runner.volumes
    assert lease.snapshot().state == "closed"
    assert (
        sum("up" in arguments for arguments in runner.calls)
        == sum("down" in arguments for arguments in runner.calls)
        == 1
    )
    assert all("prune" not in arguments and arguments[0] not in {"pull", "build"} for arguments in runner.calls)


@pytest.mark.parametrize(
    "field",
    [
        "environment",
        "env_file",
        "volumes",
        "ports",
        "privileged",
        "cap_add",
        "devices",
        "build",
        "network_mode",
        "extra_hosts",
        "secrets",
        "profiles",
    ],
)
def test_arbitrary_compose_fields_are_rejected(field: str) -> None:
    data = service_spec("task").model_dump()
    data[field] = "not-approved"
    with pytest.raises(ValueError):
        ComposeServiceSpec.model_validate(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("image", "unversioned:latest"),
        ("uid", 0),
        ("gid", 0),
        ("cpu_millis", 0),
        ("pids_limit", -1),
        ("name", "${GH_TOKEN}"),
        ("command", ("echo", "${GH_TOKEN}")),
        ("healthcheck", ("echo", "$SECRET")),
        ("command", ("cmd\x00tail",)),
    ],
)
def test_unpinned_unsafe_or_interpolated_service_values_are_rejected(*, field: str, value: Any) -> None:
    data = service_spec("task").model_dump()
    data[field] = value
    with pytest.raises(ValueError):
        ComposeServiceSpec.model_validate(data)


def test_topology_and_run_labels_cannot_override_identity_or_interpolate() -> None:
    spec = service_spec("task")
    with pytest.raises(ValueError, match="unique names"):
        ComposeEnvironmentSpec(services=(spec, spec))
    with pytest.raises(ValueError, match="parent"):
        ComposeEnvironmentSpec(services=(service_spec("task", parent_name="missing"),))
    with pytest.raises(ValueError, match="literal identifier"):
        DockerComposeEnvironmentLease(
            run_id="${GH_TOKEN}", spec=ComposeEnvironmentSpec(services=(spec,)), runner=FakeDockerRunner()
        )


@pytest.mark.parametrize("kind", ["container", "network", "volume"])
async def test_preexisting_namespace_collision_is_never_adopted_or_deleted_async(kind: str) -> None:
    lease, runner = make_lease()
    resource = {"Name": lease.project_name + "-foreign", "Labels": {"com.docker.compose.project": lease.project_name}}
    identifier = "e" * 64 if kind != "volume" else lease.project_name + "-foreign"
    if kind == "container":
        resource = {"Id": identifier, "Name": resource["Name"], "Config": {"Labels": resource["Labels"]}}
        runner.containers[identifier] = resource
    elif kind == "network":
        resource["Id"] = identifier
        runner.networks[identifier] = resource
    else:
        resource["Name"] = identifier
        runner.volumes[identifier] = resource
    before = copy.deepcopy((runner.containers, runner.networks, runner.volumes))
    with pytest.raises(DockerComposeError, match="occupied"):
        await lease.acquire_async()
    assert (runner.containers, runner.networks, runner.volumes) == before
    assert not runner.documents


async def test_missing_local_image_does_not_pull_build_or_create_async() -> None:
    lease, runner = make_lease()
    runner.bad_image = True
    with pytest.raises(DockerComposeError):
        await lease.acquire_async()
    assert not runner.documents
    assert all(arguments[0] not in {"pull", "build"} for arguments in runner.calls)


@pytest.mark.parametrize("defect", ["digest", "platform", "volumes"])
async def test_unapproved_local_image_metadata_stops_before_up_async(defect: str) -> None:
    lease, runner = make_lease()

    def mutate(image: dict[str, Any]) -> None:
        if defect == "digest":
            image["RepoDigests"] = ["registry.invalid/task@sha256:" + "c" * 64]
        elif defect == "platform":
            image["Architecture"] = "unapproved"
        else:
            image["Config"]["Volumes"] = {"/data": {}}

    runner.mutate_image = mutate
    with pytest.raises(DockerComposeError, match="Local pinned"):
        await lease.acquire_async()
    assert not runner.documents


async def test_collision_appearing_during_image_preflight_does_not_dispatch_up_async() -> None:
    lease, runner = make_lease()

    def collide(image: dict[str, Any]) -> None:
        runner.volumes[lease.project_name + "-collision"] = {
            "Name": lease.project_name + "-collision",
            "Labels": {},
        }

    runner.mutate_image = collide
    with pytest.raises(DockerComposeError, match="became occupied"):
        await lease.acquire_async()
    assert not runner.documents and runner.volumes


async def test_lease_label_inventory_detects_resources_outside_project_name_async() -> None:
    lease, runner = make_lease()
    runner.volumes["unexpected"] = {"Name": "unexpected", "Labels": {"org.pyrit.native.lease": lease.lease_id}}
    with pytest.raises(DockerComposeError, match="occupied"):
        await lease.acquire_async()
    assert "unexpected" in runner.volumes and not runner.documents


@pytest.mark.parametrize(
    "defect", ["missing", "extra", "duplicate", "unhealthy", "starting", "missing-network", "external-network"]
)
async def test_incomplete_or_unhealthy_effective_topology_never_becomes_ready_async(defect: str) -> None:
    lease, runner = make_lease()

    def mutate(fake: FakeDockerRunner) -> None:
        first = next(iter(fake.containers.values()))
        network = next(iter(fake.networks.values()))
        if defect == "missing":
            identifier = next(iter(fake.containers))
            fake.containers.pop(identifier)
            network["Containers"].pop(identifier)
        elif defect in {"extra", "duplicate"}:
            extra = copy.deepcopy(first)
            extra["Id"] = "e" * 64
            extra["Name"] += "-extra"
            if defect == "extra":
                extra["Config"]["Labels"]["com.docker.compose.service"] = "unapproved"
            fake.containers[extra["Id"]] = extra
            network["Containers"][extra["Id"]] = {}
        elif defect == "missing-network":
            fake.networks.clear()
        elif defect == "external-network":
            network["Internal"] = False
        else:
            first["State"]["Health"]["Status"] = defect

    runner.mutate_up = mutate
    with pytest.raises(DockerComposeError):
        await lease.acquire_async()
    if defect == "extra":
        assert lease.snapshot().state == "cleanup_failed"
        assert runner.containers and runner.networks
        assert all("down" not in arguments for arguments in runner.calls)
    else:
        assert lease.snapshot().state == "closed"
        assert not runner.containers and not runner.networks
        assert sum("down" in arguments for arguments in runner.calls) == 1


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("HostConfig", "Privileged", True),
        ("HostConfig", "CapAdd", ["SYS_ADMIN"]),
        ("HostConfig", "Binds", ["/host:/guest"]),
        ("HostConfig", "PortBindings", {"80/tcp": [{"HostPort": "80"}]}),
        ("HostConfig", "ReadonlyRootfs", False),
        ("HostConfig", "SecurityOpt", []),
        ("HostConfig", "Memory", 0),
        ("HostConfig", "PidsLimit", -1),
        ("HostConfig", "NetworkMode", "host"),
        ("HostConfig", "Devices", [{"PathOnHost": "/dev/sda"}]),
        ("HostConfig", "PidMode", "host"),
        ("Config", "User", "0:0"),
        ("Config", "Env", ["GH_TOKEN=inert-value"]),
        ("Config", "Cmd", ["/changed"]),
        ("Config", "Entrypoint", ["/unapproved"]),
        ("Config", "Image", "unpinned:latest"),
    ],
)
async def test_effective_container_mismatch_blocks_readiness_async(*, section: str, field: str, value: Any) -> None:
    lease, runner = make_lease()
    runner.mutate_up = lambda fake: next(iter(fake.containers.values()))[section].update({field: value})
    with pytest.raises(DockerComposeError):
        await lease.acquire_async()
    if field in {"Privileged", "CapAdd", "Binds", "ReadonlyRootfs", "SecurityOpt", "User", "Image"}:
        assert lease.snapshot().state == "cleanup_failed" and runner.containers
        assert all("down" not in arguments for arguments in runner.calls)
    else:
        assert lease.snapshot().state == "closed" and not runner.containers


@pytest.mark.parametrize("defect", ["mount", "image-id", "network-attachment", "published-port", "health-command"])
async def test_effective_mount_image_network_and_health_are_audited_async(defect: str) -> None:
    lease, runner = make_lease()

    def mutate(fake: FakeDockerRunner) -> None:
        first = next(iter(fake.containers.values()))
        if defect == "mount":
            first["Mounts"] = [{"Type": "bind", "Source": "/var/run/docker.sock", "Destination": "/socket"}]
        elif defect == "image-id":
            first["Image"] = "sha256:" + "c" * 64
        elif defect == "network-attachment":
            first["NetworkSettings"]["Networks"]["bridge"] = {"NetworkID": "e" * 64}
        elif defect == "published-port":
            first["NetworkSettings"]["Ports"] = {"80/tcp": [{"HostIp": "0.0.0.0", "HostPort": "12345"}]}
        else:
            first["Config"]["Healthcheck"]["Test"] = ["NONE"]

    runner.mutate_up = mutate
    with pytest.raises(DockerComposeError):
        await lease.acquire_async()
    if defect in {"mount", "image-id"}:
        assert lease.snapshot().state == "cleanup_failed" and runner.containers
        assert all("down" not in arguments for arguments in runner.calls)
    else:
        assert not runner.containers


async def test_partial_up_failure_still_cleans_owned_project_once_async() -> None:
    lease, runner = make_lease(4)
    runner.up_returncode = 1

    def partial(fake: FakeDockerRunner) -> None:
        for identifier in list(fake.containers)[1:]:
            fake.containers.pop(identifier)
            fake.networks["f" * 64]["Containers"].pop(identifier)

    runner.mutate_up = partial
    with pytest.raises(DockerComposeError, match="exit=1"):
        await lease.acquire_async()
    assert not runner.containers and not runner.networks
    assert lease.snapshot().resources[0].status == "confirmed"
    assert not lease.snapshot().resources[0].acquisition_completed


async def test_cancelled_up_rolls_back_after_runner_cancellation_async() -> None:
    lease, runner = make_lease()
    runner.wait_in_up = True
    task = asyncio.create_task(lease.acquire_async())
    await asyncio.wait_for(runner.up_entered.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runner.containers and not runner.networks
    assert lease.snapshot().state == "closed"


@pytest.mark.parametrize("foreign", ["labels", "attached-container", "volume"])
async def test_cleanup_refuses_unverified_project_membership_async(foreign: str) -> None:
    lease, runner = make_lease()
    await lease.acquire_async()
    if foreign == "labels":
        next(iter(runner.containers.values()))["Config"]["Labels"]["org.pyrit.native.lease"] = "foreign"
    elif foreign == "attached-container":
        runner.networks["f" * 64]["Containers"]["e" * 64] = {"Name": "foreign"}
    else:
        runner.volumes[lease.project_name + "-extra"] = {
            "Name": lease.project_name + "-extra",
            "Labels": {
                **runner.networks["f" * 64]["Labels"],
            },
        }
    with pytest.raises(EnvironmentCleanupError):
        await lease.close_async()
    assert all("down" not in arguments for arguments in runner.calls)
    assert runner.containers and runner.networks


@pytest.mark.parametrize("failure", ["down-exit", "residual"])
async def test_down_failure_or_residual_is_not_clean_success_or_retried_async(failure: str) -> None:
    lease, runner = make_lease()
    await lease.acquire_async()
    if failure == "down-exit":
        runner.down_returncode = 1
        runner.mutate_down = lambda fake: fake.containers.pop(next(iter(fake.containers)))
    else:
        network = copy.deepcopy(runner.networks["f" * 64])
        network["Containers"] = {}
        runner.mutate_down = lambda fake: fake.networks.update({"f" * 64: network})
    for _ in range(2):
        with pytest.raises(EnvironmentCleanupError):
            await lease.close_async()
    assert lease.snapshot().state == "cleanup_failed"
    assert sum("down" in arguments for arguments in runner.calls) == 1
    if failure == "down-exit":
        assert len(runner.containers) == 1 and runner.networks


@pytest.mark.parametrize(
    "result",
    [
        DockerCommandResult(stdout="", stderr="", returncode=1),
        DockerCommandResult(stdout="", stderr="", returncode=None, timed_out=True),
        DockerCommandResult(stdout="", stderr="", returncode=0, truncated=True),
    ],
)
async def test_incomplete_control_output_is_never_treated_as_empty_inventory_async(result: DockerCommandResult) -> None:
    lease, runner = make_lease()
    runner.bad_result = result
    with pytest.raises(DockerComposeError):
        await lease.acquire_async()
    assert not runner.documents


async def test_unrelated_resources_survive_owned_down_async() -> None:
    lease, runner = make_lease()
    runner.containers["e" * 64] = {
        "Id": "e" * 64,
        "Name": "/unrelated",
        "Config": {"Labels": {"com.docker.compose.project": "unrelated"}},
    }
    await lease.acquire_async()
    await lease.close_async()
    assert list(runner.containers) == ["e" * 64]


async def test_post_acquisition_health_cannot_accept_replaced_container_identity_async() -> None:
    lease, runner = make_lease()
    allocation = await lease.acquire_async()
    previous = allocation.container_id("service-0")
    container = runner.containers.pop(previous)
    container["Id"] = "d" * 64
    runner.containers[container["Id"]] = container
    runner.networks["f" * 64]["Containers"].pop(previous)
    runner.networks["f" * 64]["Containers"][container["Id"]] = {}
    with pytest.raises(DockerComposeError, match="identity changed"):
        await lease.check_health_async()
    await lease.close_async()


@pytest.mark.parametrize("payload", ["not-json", "{}", "[]", '[{"Id":"wrong"}]'])
async def test_malformed_or_mismatched_inspection_is_rejected_async(payload: str) -> None:
    lease, runner = make_lease()
    runner.bad_result = DockerCommandResult(stdout=payload, stderr="", returncode=0)
    with pytest.raises(DockerComposeError):
        await lease._inspect_async("container", ("a" * 64,))


def state_service(name: str = "cli") -> ComposeServiceSpec:
    return ComposeServiceSpec.model_validate(
        {
            **service_spec(name, role="agent").model_dump(),
            "tmpfs_bytes": 67_108_864,
            "runtime_state": (
                ComposeTmpfsSpec(path="/tmp", size_bytes=16_777_216),
                ComposeTmpfsSpec(path="/workspace", size_bytes=33_554_432, executable=True),
                ComposeTmpfsSpec(path="/home/runner", size_bytes=16_777_216),
            ),
        }
    )


def report_tmpfs_mounts(runner: FakeDockerRunner) -> None:
    for container in runner.containers.values():
        container["Mounts"] = [
            {
                "Type": "tmpfs",
                "Source": "",
                "Destination": path,
                "Mode": options,
                "RW": True,
                "Propagation": "",
            }
            for path, options in container["HostConfig"]["Tmpfs"].items()
        ]


async def test_absent_runtime_policy_keeps_the_exact_strict_default_async() -> None:
    lease, runner = make_lease(1)
    await lease.acquire_async()
    service = runner.documents[0]["services"]["service-0"]
    assert service["tmpfs"] == ["/tmp:rw,noexec,nosuid,nodev,size=16777216,uid=1000,gid=1000,mode=0700"]
    assert service["read_only"] is True and service["privileged"] is False
    assert service["cap_drop"] == ["ALL"] and service["security_opt"] == ["no-new-privileges:true"]
    assert service["networks"] == ["challenge"] and runner.documents[0]["networks"]["challenge"]["internal"] is True
    assert not set(service) & {"volumes", "environment", "ports", "cap_add", "devices"}
    assert service_spec("default").runtime_state is None
    await lease.close_async()


@pytest.mark.parametrize("include_mount_records", [False, True])
async def test_opt_in_cli_and_grafana_state_are_bounded_per_service_async(include_mount_records: bool) -> None:
    cli = state_service()
    target = ComposeServiceSpec.model_validate(
        {
            **service_spec("grafana-state-model").model_dump(),
            "tmpfs_bytes": 67_108_864,
            "runtime_state": (
                ComposeTmpfsSpec(path="/tmp", size_bytes=16_777_216),
                ComposeTmpfsSpec(path="/var/lib/grafana", size_bytes=33_554_432),
                ComposeTmpfsSpec(path="/var/log/grafana", size_bytes=16_777_216),
            ),
        }
    )
    runner = FakeDockerRunner()
    if include_mount_records:
        runner.mutate_up = report_tmpfs_mounts
    lease = DockerComposeEnvironmentLease(
        run_id="state-profile", spec=ComposeEnvironmentSpec(services=(cli, target)), runner=runner
    )
    allocation = await lease.acquire_async()
    assert len(allocation.containers) == 2 and lease.snapshot().state == "ready"
    specifications = runner.documents[0]["services"]
    assert specifications["cli"]["tmpfs"] == [
        "/tmp:rw,noexec,nosuid,nodev,size=16777216,uid=1000,gid=1000,mode=0700",
        "/workspace:rw,exec,nosuid,nodev,size=33554432,uid=1000,gid=1000,mode=0700",
        "/home/runner:rw,noexec,nosuid,nodev,size=16777216,uid=1000,gid=1000,mode=0700",
    ]
    assert all("rw,noexec,nosuid,nodev" in options for options in specifications[target.name]["tmpfs"])
    assert set(
        next(
            value
            for value in runner.containers.values()
            if value["Config"]["Labels"]["com.docker.compose.service"] == target.name
        )["HostConfig"]["Tmpfs"]
    ) == {"/tmp", "/var/lib/grafana", "/var/log/grafana"}
    for service in specifications.values():
        assert service["mem_limit"] == service["memswap_limit"] == 134_217_728
        assert service["cpus"] == 0.5 and service["pids_limit"] == 64
        assert service["read_only"] and not service["privileged"]
        assert not set(service) & {"volumes", "ports", "environment", "cap_add"}
    await lease.close_async()
    assert runner.documents[0] == runner.documents[1]
    assert not runner.containers and not runner.networks and not runner.volumes
    assert lease.capabilities == frozenset({EnvironmentCapability.HEALTH_CHECK})


async def test_tmp_executable_requires_explicit_path_level_opt_in_async() -> None:
    service = ComposeServiceSpec.model_validate(
        {
            **service_spec("cli").model_dump(),
            "runtime_state": (ComposeTmpfsSpec(path="/tmp", size_bytes=16_777_216, executable=True),),
        }
    )
    runner = FakeDockerRunner()
    lease = DockerComposeEnvironmentLease(
        run_id="exec-tmp", spec=ComposeEnvironmentSpec(services=(service,)), runner=runner
    )
    await lease.acquire_async()
    assert runner.documents[0]["services"]["cli"]["tmpfs"] == [
        "/tmp:rw,exec,nosuid,nodev,size=16777216,uid=1000,gid=1000,mode=0700"
    ]
    await lease.close_async()


@pytest.mark.parametrize(
    "path",
    [
        "/",
        "relative",
        r"C:\workspace",
        "//workspace",
        "/workspace/",
        "/workspace/../cache",
        "/workspace/./cache",
        "/tmp/$HOME",
        "/tmp/${HOME}",
        "/tmp/a,b",
        "/tmp/a:b",
        "/tmp/a b",
        "/tmp/x\x00y",
        "/tmp/x\ny",
        "/tmp/" + "x" * 256,
        "/dev",
        "/dev/shm",
        "/proc/self",
        "/sys",
        "/etc/hosts",
        "/usr/local",
        "/var",
        "/var/run/task",
        "/run",
    ],
)
def test_unsafe_or_aliased_runtime_state_paths_are_rejected(path: str) -> None:
    with pytest.raises(ValueError):
        ComposeTmpfsSpec(path=path, size_bytes=1024)


@pytest.mark.parametrize(
    "paths",
    [
        ("/tmp", "/tmp"),
        ("/tmp", "/tmp/cache"),
        ("/tmp/cache", "/tmp"),
        ("/tmp", "/workspace", "/workspace/cache"),
        ("/workspace/cache", "/workspace", "/tmp"),
        ("/workspace",),
        ("/tmp", "/app"),
        (),
    ],
)
def test_runtime_state_requires_nonoverlapping_paths_and_preserves_executables(paths: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        ComposeServiceSpec.model_validate(
            {
                **service_spec("task").model_dump(),
                "runtime_state": tuple(ComposeTmpfsSpec(path=path, size_bytes=1024) for path in paths),
            }
        )


def test_similar_path_prefixes_are_not_mistaken_for_nested_mounts() -> None:
    service = ComposeServiceSpec.model_validate(
        {
            **service_spec("task").model_dump(),
            "runtime_state": (
                ComposeTmpfsSpec(path="/tmp", size_bytes=1024),
                ComposeTmpfsSpec(path="/workspace", size_bytes=1024),
                ComposeTmpfsSpec(path="/workspace-cache", size_bytes=1024),
            ),
        }
    )
    assert len(service.tmpfs_mounts) == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("size_bytes", 0),
        ("size_bytes", 1_073_741_825),
        ("size_bytes", True),
        ("executable", "true"),
        ("mode", "0777"),
        ("uid", 0),
        ("gid", 0),
        ("source", "/host"),
        ("type", "volume"),
        ("options", "suid,dev"),
        ("propagation", "rshared"),
    ],
)
def test_tmpfs_policy_does_not_accept_unsafe_permissions_or_arbitrary_options(*, field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        ComposeTmpfsSpec.model_validate({"path": "/tmp", "size_bytes": 1024, field: value})


def test_total_runtime_state_size_cannot_exceed_service_budget_or_memory() -> None:
    values = state_service().model_dump()
    with pytest.raises(ValueError, match="Combined runtime-state"):
        ComposeServiceSpec.model_validate({**values, "tmpfs_bytes": 16_777_216})
    with pytest.raises(ValueError, match="memory limit"):
        ComposeServiceSpec.model_validate({**values, "tmpfs_bytes": 268_435_456})


@pytest.mark.parametrize("mount_count", [8, 9])
def test_runtime_state_mount_count_has_an_explicit_upper_bound(mount_count: int) -> None:
    mounts = (
        ComposeTmpfsSpec(path="/tmp", size_bytes=1024),
        *(ComposeTmpfsSpec(path=f"/state-{index}", size_bytes=1024) for index in range(mount_count - 1)),
    )
    values = {**service_spec("task").model_dump(), "runtime_state": mounts}
    if mount_count == 8:
        assert len(ComposeServiceSpec.model_validate(values).tmpfs_mounts) == 8
    else:
        with pytest.raises(ValueError):
            ComposeServiceSpec.model_validate(values)


@pytest.mark.parametrize("value", [{"/workspace": {}}, {"/unmatched-volume": {}}])
async def test_opt_in_tmpfs_never_authorizes_an_image_volume_async(value: dict[str, Any]) -> None:
    runner = FakeDockerRunner()
    runner.mutate_image = lambda image: image["Config"].update({"Volumes": value})
    lease = DockerComposeEnvironmentLease(
        run_id="volume-blocked", spec=ComposeEnvironmentSpec(services=(state_service(),)), runner=runner
    )
    with pytest.raises(DockerComposeError, match="Local pinned"):
        await lease.acquire_async()
    assert not runner.documents and not runner.containers and not runner.volumes


@pytest.mark.parametrize(
    "drift",
    [
        "extra-path",
        "missing-path",
        "alias-path",
        "unbounded",
        "exec",
        "mode",
        "uid",
        "gid",
        "suid",
        "dev",
        "bind",
        "long-mount",
        "volumes-from",
        "image-volume",
        "privileged",
        "cap-add",
        "no-security",
    ],
)
async def test_opt_in_state_drift_blocks_readiness_and_cleanup_async(drift: str) -> None:
    runner = FakeDockerRunner()
    lease = DockerComposeEnvironmentLease(
        run_id="drift", spec=ComposeEnvironmentSpec(services=(state_service(),)), runner=runner
    )

    def mutate(fake: FakeDockerRunner) -> None:
        container = next(iter(fake.containers.values()))
        host = container["HostConfig"]
        options = host["Tmpfs"]
        if drift == "extra-path":
            options["/extra"] = options["/tmp"]
        elif drift == "missing-path":
            options.pop("/workspace")
        elif drift == "alias-path":
            options["/workspace/../workspace"] = options.pop("/workspace")
        elif drift in {"unbounded", "exec", "mode", "uid", "gid", "suid", "dev"}:
            old, new = {
                "unbounded": ("size=16777216", "size=0"),
                "exec": ("noexec", "exec"),
                "mode": ("mode=0700", "mode=0777"),
                "uid": ("uid=1000", "uid=0"),
                "gid": ("gid=1000", "gid=0"),
                "suid": ("nosuid", "suid"),
                "dev": ("nodev", "dev"),
            }[drift]
            options["/tmp"] = options["/tmp"].replace(old, new)
        elif drift == "bind":
            host["Binds"] = ["/host:/workspace"]
        elif drift == "long-mount":
            host["Mounts"] = [{"Type": "volume", "Target": "/workspace"}]
        elif drift == "volumes-from":
            host["VolumesFrom"] = ["another-container"]
        elif drift == "image-volume":
            container["Config"]["Volumes"] = {"/workspace": {}}
        elif drift == "privileged":
            host["Privileged"] = True
        elif drift == "cap-add":
            host["CapAdd"] = ["SYS_ADMIN"]
        else:
            host["SecurityOpt"] = []

    runner.mutate_up = mutate
    with pytest.raises(DockerComposeError) as caught:
        await lease.acquire_async()
    assert isinstance(caught.value.__cause__, EnvironmentCleanupError)
    assert lease.snapshot().state == "cleanup_failed"
    assert all("down" not in arguments for arguments in runner.calls)
    assert runner.containers and runner.networks


@pytest.mark.parametrize(
    "drift",
    [
        "type",
        "destination",
        "duplicate",
        "partial",
        "source",
        "volume-name",
        "driver",
        "readonly",
        "exec",
        "mode",
        "propagation",
        "unknown-option",
        "malformed",
        "unknown-list",
    ],
)
async def test_reported_effective_mount_records_cannot_override_tmpfs_policy_async(drift: str) -> None:
    runner = FakeDockerRunner()
    lease = DockerComposeEnvironmentLease(
        run_id="mount-drift", spec=ComposeEnvironmentSpec(services=(state_service(),)), runner=runner
    )

    def mutate(fake: FakeDockerRunner) -> None:
        report_tmpfs_mounts(fake)
        container = next(iter(fake.containers.values()))
        mount = container["Mounts"][0]
        changes: dict[str, tuple[str, Any]] = {
            "type": ("Type", "volume"),
            "destination": ("Destination", "/tmp/../tmp"),
            "source": ("Source", "/var/run/docker.sock"),
            "volume-name": ("Name", "anonymous-volume"),
            "driver": ("Driver", "local"),
            "readonly": ("RW", False),
            "exec": ("Mode", "rw,exec"),
            "mode": ("Mode", "rw,mode=0777"),
            "propagation": ("Propagation", "shared"),
            "unknown-option": ("Options", ["suid"]),
        }
        if drift in changes:
            field, value = changes[drift]
            mount[field] = value
        elif drift == "duplicate":
            container["Mounts"].append(copy.deepcopy(mount))
        elif drift == "partial":
            container["Mounts"].pop()
        elif drift == "malformed":
            container["Mounts"].append(None)
        else:
            container["Mounts"] = None

    runner.mutate_up = mutate
    with pytest.raises(DockerComposeError):
        await lease.acquire_async()
    assert lease.snapshot().state == "cleanup_failed"
    assert all("down" not in arguments for arguments in runner.calls)


async def test_runtime_state_is_rechecked_before_teardown_without_requiring_healthy_services_async() -> None:
    runner = FakeDockerRunner()
    lease = DockerComposeEnvironmentLease(
        run_id="stopped", spec=ComposeEnvironmentSpec(services=(state_service(),)), runner=runner
    )
    await lease.acquire_async()
    container = next(iter(runner.containers.values()))
    container["State"] = {"Running": False, "Status": "exited"}
    await lease.close_async()
    assert lease.snapshot().state == "closed" and not runner.containers

    changed_runner = FakeDockerRunner()
    changed = DockerComposeEnvironmentLease(
        run_id="changed", spec=ComposeEnvironmentSpec(services=(state_service(),)), runner=changed_runner
    )
    await changed.acquire_async()
    next(iter(changed_runner.containers.values()))["HostConfig"]["Tmpfs"]["/tmp"] += ",mode=0777"
    with pytest.raises(EnvironmentCleanupError):
        await changed.close_async()
    assert all("down" not in arguments for arguments in changed_runner.calls)


@pytest.mark.parametrize("cancel", [False, True])
async def test_opt_in_policy_partial_start_preserves_owned_rollback_async(cancel: bool) -> None:
    runner = FakeDockerRunner()
    runner.up_returncode = 1 if not cancel else 0
    runner.wait_in_up = cancel
    lease = DockerComposeEnvironmentLease(
        run_id="partial-state", spec=ComposeEnvironmentSpec(services=(state_service(),)), runner=runner
    )
    task = asyncio.create_task(lease.acquire_async())
    await asyncio.wait_for(runner.up_entered.wait(), timeout=2)
    if cancel:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(DockerComposeError, match="exit=1"):
            await task
    assert lease.snapshot().state == "closed"
    assert not runner.containers and not runner.networks and not runner.volumes
    assert runner.documents[0] == runner.documents[1]
