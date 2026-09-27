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
                    "Tmpfs": {"/tmp": spec["tmpfs"][0].split(":", 1)[1]},
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
