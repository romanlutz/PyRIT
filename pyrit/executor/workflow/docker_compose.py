# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strict, prebuilt-only Compose provider for a run-owned native environment."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyrit.executor.workflow.environment_lease import EnvironmentLease
from pyrit.models.environment_lease import (
    EnvironmentCapability,
    EnvironmentHealth,
    EnvironmentResourceHandle,
    EnvironmentServiceHandle,
)

if TYPE_CHECKING:
    from pyrit.executor.workflow.docker_command import DockerCommandRunner


class ComposePlatform(str, Enum):
    """The prebuilt Linux image platforms qualified by image inspection."""

    AMD64 = "linux/amd64"
    ARM64 = "linux/arm64"


class ComposeServiceSpec(BaseModel):
    """Trusted approved argv and limits, not an arbitrary Compose service mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9-]{0,39}$")
    roles: frozenset[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    parent_name: str | None = None
    image: str = Field(pattern=r"^[a-z0-9][a-z0-9._:/-]*@sha256:[0-9a-f]{64}$")
    platform: ComposePlatform = ComposePlatform.AMD64
    command: tuple[str, ...] = Field(min_length=1, max_length=128)
    healthcheck: tuple[str, ...] = Field(min_length=1, max_length=128)
    uid: int = Field(ge=1, le=2_147_483_647, strict=True)
    gid: int = Field(ge=1, le=2_147_483_647, strict=True)
    cpu_millis: int = Field(ge=1, le=8000, strict=True)
    memory_bytes: int = Field(ge=33_554_432, le=34_359_738_368, strict=True)
    pids_limit: int = Field(ge=1, le=4096, strict=True)
    tmpfs_bytes: int = Field(ge=1, le=1_073_741_824, strict=True)

    @field_validator("command", "healthcheck")
    @classmethod
    def _validate_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not part or any(character in part for character in ("\x00", "\n", "\r", "$")) for part in value):
            raise ValueError("Approved argv must be nonempty literals without Compose interpolation or line breaks.")
        return value

    @model_validator(mode="after")
    def _validate_tmpfs(self) -> ComposeServiceSpec:
        if self.tmpfs_bytes > self.memory_bytes:
            raise ValueError("Temporary filesystem size cannot exceed the service memory limit.")
        return self


class ComposeEnvironmentSpec(BaseModel):
    """An approved static topology with no mounts, ports, env, builds or arbitrary options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    services: tuple[ComposeServiceSpec, ...] = Field(min_length=1, max_length=32)
    wait_timeout_seconds: int = Field(default=60, ge=1, le=120, strict=True)

    @model_validator(mode="after")
    def _validate_topology(self) -> ComposeEnvironmentSpec:
        names: set[str] = set()
        for service in self.services:
            if service.name in names or (service.parent_name is not None and service.parent_name not in names):
                raise ValueError("Services need unique names and earlier declared parent references.")
            names.add(service.name)
        return self


class DockerComposeError(RuntimeError):
    """A control, ownership, health or isolation check failed without an implicit fallback."""


@dataclass(frozen=True, kw_only=True)
class ComposeAllocation:
    """Only verified IDs are returned; these are not permission to execute arbitrary host commands."""

    project_name: str
    network_id: str
    containers: tuple[tuple[str, str], ...]

    def container_id(self, name: str) -> str:
        """
        Resolve one exact named allocation.

        Returns:
            str: The full inspected container ID.

        Raises:
            KeyError: If the named service is absent.
        """
        return dict(self.containers)[name]


@dataclass(frozen=True, kw_only=True)
class _Inventory:
    containers: tuple[dict[str, Any], ...]
    networks: tuple[dict[str, Any], ...]
    volumes: tuple[dict[str, Any], ...]

    @property
    def empty(self) -> bool:
        """Whether the inspected project namespace is empty."""
        return not (self.containers or self.networks or self.volumes)


class DockerComposeEnvironmentLease(EnvironmentLease[ComposeAllocation]):
    """Own a strict Compose project, checking resource labels before any scoped teardown."""

    _RUN_LABEL = "org.pyrit.native.run"
    _LEASE_LABEL = "org.pyrit.native.lease"
    _PROJECT_LABEL = "com.docker.compose.project"
    _SERVICE_LABEL = "com.docker.compose.service"
    _NETWORK_LABEL = "com.docker.compose.network"
    _COMMAND_TIMEOUT = 5

    def __init__(self, *, run_id: str, spec: ComposeEnvironmentSpec, runner: DockerCommandRunner) -> None:
        """
        Reserve a unique project identity without contacting Docker.

        Args:
            run_id (str): Owning native evaluation.
            spec (ComposeEnvironmentSpec): Trusted immutable service/resource approval.
            runner (DockerCommandRunner): Host-owned bounded control transport.

        Raises:
            ValueError: If run identity could be interpreted as Compose interpolation.
        """
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}", run_id):
            raise ValueError("Compose run identity must be a literal identifier without interpolation.")
        super().__init__(
            run_id=run_id, capabilities=frozenset({EnvironmentCapability.HEALTH_CHECK}), cleanup_timeout_seconds=150
        )
        self._spec = spec
        self._runner = runner
        self._project = "pyrit-" + self.lease_id.replace("-", "")
        self._network = self._project + "_challenge"
        self._up_attempted = False
        self._images: dict[str, dict[str, Any]] = {}
        self._allocation: ComposeAllocation | None = None
        self._manifest = json.dumps(self._document(), sort_keys=True, separators=(",", ":"))

    @property
    def project_name(self) -> str:
        """The unique project reserved for this run before control I/O."""
        return self._project

    async def _acquire_async(self) -> ComposeAllocation:
        resource = EnvironmentResourceHandle(
            run_id=self.run_id, provider="docker_compose", resource_id=self._project, kind="compose_project"
        )
        allocation = await self._acquire_resource_async(
            handle=resource, acquire_async=self._up_async, release_async=self._down_async
        )
        for service in self._spec.services:
            self._register_service(
                EnvironmentServiceHandle(
                    name=service.name, roles=service.roles, parent_name=service.parent_name, resource=resource
                )
            )
        self._allocation = allocation
        return allocation

    async def _up_async(self, handle: EnvironmentResourceHandle) -> ComposeAllocation:
        if not (await self._inventory_async()).empty:
            raise DockerComposeError("Compose project label/name namespace is already occupied; no mutation allowed.")
        for service in self._spec.services:
            image = (await self._inspect_async("image", (service.image,)))[0]
            self._validate_image(service=service, image=image)
            self._images[service.name] = image
        if not (await self._inventory_async()).empty:
            raise DockerComposeError("Compose namespace became occupied before dispatch; no mutation allowed.")
        # Reserve before dispatch: even a failed/cancelled up may have created resources.
        self._up_attempted = True
        await self._compose_async(
            (
                "up",
                "--detach",
                "--no-build",
                "--pull",
                "never",
                "--no-recreate",
                "--wait",
                "--wait-timeout",
                str(self._spec.wait_timeout_seconds),
            ),
            timeout_seconds=self._spec.wait_timeout_seconds + 10,
        )
        inventory = await self._inventory_async()
        self._validate_inventory(inventory, require_ready=True)
        return ComposeAllocation(
            project_name=self._project,
            network_id=inventory.networks[0]["Id"],
            containers=tuple(
                (self._labels(container, "container")[self._SERVICE_LABEL], container["Id"])
                for container in inventory.containers
            ),
        )

    async def _check_health_async(self) -> tuple[EnvironmentHealth, ...]:
        inventory = await self._inventory_async()
        self._validate_inventory(inventory, require_ready=True)
        assert self._allocation is not None
        current = {
            self._labels(container, "container")[self._SERVICE_LABEL]: container["Id"]
            for container in inventory.containers
        }
        if current != dict(self._allocation.containers) or inventory.networks[0]["Id"] != self._allocation.network_id:
            raise DockerComposeError("Compose allocation identity changed after acquisition.")
        return tuple(
            EnvironmentHealth(
                service_name=service.name, healthy=True, reason="Inspected running and healthy owned container."
            )
            for service in self._spec.services
        )

    async def _down_async(self, handle: EnvironmentResourceHandle) -> None:
        if not self._up_attempted:
            return
        inventory = await self._inventory_async()
        self._validate_inventory(inventory, require_ready=False)
        if not inventory.empty:
            await self._compose_async(("down", "--remove-orphans", "--timeout", "10"), timeout_seconds=15)
        if not (await self._inventory_async()).empty:
            raise DockerComposeError("Owned Compose cleanup left residual resources; cleanup is unconfirmed.")

    def _labels(self, value: dict[str, Any], kind: str) -> dict[str, Any]:
        config = value.get("Config")
        labels = config.get("Labels") if kind == "container" and isinstance(config, dict) else value.get("Labels")
        if not isinstance(labels, dict):
            raise DockerComposeError("Resource ownership labels are missing.")
        return labels

    def _validate_inventory(self, inventory: _Inventory, *, require_ready: bool) -> None:
        for kind, resources in (
            ("container", inventory.containers),
            ("network", inventory.networks),
            ("volume", inventory.volumes),
        ):
            for resource in resources:
                labels = self._labels(resource, kind)
                if any(labels.get(key) != value for key, value in self._ownership_labels().items()):
                    raise DockerComposeError("Foreign or unverified resources occupy the reserved Compose namespace.")
                if labels.get(self._PROJECT_LABEL) != self._project:
                    raise DockerComposeError("Owned resource belongs to a different Compose project.")
        container_ids = {container["Id"] for container in inventory.containers}
        for network in inventory.networks:
            attached = network.get("Containers")
            if not isinstance(attached, dict) or set(attached) - container_ids:
                raise DockerComposeError("Challenge network contains unverified attached containers.")
        if inventory.volumes:
            raise DockerComposeError(
                "Volumes are not approved by this provider; refusing project teardown or readiness."
            )
        if not require_ready:
            return
        expected = {service.name for service in self._spec.services}
        names = [self._labels(container, "container").get(self._SERVICE_LABEL) for container in inventory.containers]
        if len(names) != len(expected) or set(names) != expected:
            raise DockerComposeError("Missing, duplicate or extra Compose services.")
        if len(inventory.networks) != 1:
            raise DockerComposeError("Expected exactly one internal challenge network.")
        network = inventory.networks[0]
        if (
            network.get("Name") != self._network
            or network.get("Internal") is not True
            or network.get("Driver") != "bridge"
            or network.get("EnableIPv6") is not False
            or network.get("Scope") != "local"
            or network.get("Options") not in ({}, None)
            or set(network["Containers"]) != container_ids
            or self._labels(network, "network").get(self._NETWORK_LABEL) != "challenge"
        ):
            raise DockerComposeError("Challenge network configuration is not the approved internal bridge.")
        for container in inventory.containers:
            name = self._labels(container, "container")[self._SERVICE_LABEL]
            service = next(service for service in self._spec.services if service.name == name)
            self._validate_container(service=service, container=container, network_id=network["Id"])

    def _validate_image(self, *, service: ComposeServiceSpec, image: dict[str, Any]) -> None:
        expected_os, architecture = service.platform.value.split("/")
        config = image.get("Config")
        digests = image.get("RepoDigests")
        if (
            not isinstance(digests, list)
            or not any(
                isinstance(digest, str) and digest.endswith("@" + service.image.split("@")[1]) for digest in digests
            )
            or image.get("Os") != expected_os
            or image.get("Architecture") != architecture
            or not isinstance(config, dict)
            or config.get("Volumes")
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(image.get("Id", "")))
        ):
            raise DockerComposeError(
                "Local pinned image identity/platform is unverified or declares unapproved volumes."
            )

    def _validate_container(self, *, service: ComposeServiceSpec, container: dict[str, Any], network_id: str) -> None:
        config, host, state = (container.get(name) for name in ("Config", "HostConfig", "State"))
        if not isinstance(config, dict) or not isinstance(host, dict) or not isinstance(state, dict):
            raise DockerComposeError("Container inspection lacks configuration or state.")
        expected_config = {
            "Image": service.image,
            "User": f"{service.uid}:{service.gid}",
            "Cmd": list(service.command),
            "Env": self._images[service.name]["Config"].get("Env") or [],
        }
        if any(config.get(key) != value for key, value in expected_config.items()) or config.get("Entrypoint"):
            raise DockerComposeError("Container image, user, command or environment differs from approved input.")
        if container.get("Image") != self._images[service.name]["Id"] or container.get(
            "Name"
        ) != "/" + self._container_name(service):
            raise DockerComposeError("Container name or immutable image identity differs from its reservation.")
        expected_host = {
            "ReadonlyRootfs": True,
            "Privileged": False,
            "PublishAllPorts": False,
            "NanoCpus": service.cpu_millis * 1_000_000,
            "Memory": service.memory_bytes,
            "MemorySwap": service.memory_bytes,
            "PidsLimit": service.pids_limit,
            "NetworkMode": self._network,
            "IpcMode": "private",
        }
        if any(host.get(key) != value for key, value in expected_host.items()):
            raise DockerComposeError("Container isolation or resource limits differ from the approved specification.")
        forbidden = (
            "Binds",
            "Mounts",
            "VolumesFrom",
            "PortBindings",
            "Devices",
            "DeviceRequests",
            "DeviceCgroupRules",
            "CapAdd",
            "ExtraHosts",
            "PidMode",
            "UsernsMode",
            "Links",
            "GroupAdd",
        )
        if any(host.get(key) for key in forbidden):
            raise DockerComposeError("Unapproved mount, port, device, capability or host namespace was observed.")
        if host.get("CapDrop") != ["ALL"] or host.get("SecurityOpt") not in (
            ["no-new-privileges:true"],
            ["no-new-privileges"],
        ):
            raise DockerComposeError("Capability dropping and no-new-privileges are required.")
        if host.get("Tmpfs") != {"/tmp": self._tmpfs(service)}:
            raise DockerComposeError("Unexpected writable filesystem configuration.")
        if host.get("RestartPolicy", {}).get("Name") != "no":
            raise DockerComposeError("An unapproved restart policy was observed.")
        if any(
            mount.get("Type") != "tmpfs" or mount.get("Destination") != "/tmp" for mount in container.get("Mounts", [])
        ):
            raise DockerComposeError("A host or volume mount was observed.")
        networks = container.get("NetworkSettings", {}).get("Networks")
        ports = container.get("NetworkSettings", {}).get("Ports") or {}
        if not isinstance(ports, dict) or any(ports.values()):
            raise DockerComposeError("Effective network settings expose a published port.")
        if (
            not isinstance(networks, dict)
            or set(networks) != {self._network}
            or networks[self._network].get("NetworkID") != network_id
        ):
            raise DockerComposeError("Container is attached to an unapproved network.")
        if (
            state.get("Running") is not True
            or state.get("Status") != "running"
            or any(state.get(key) for key in ("Paused", "Restarting", "Dead", "OOMKilled"))
            or state.get("Health", {}).get("Status") != "healthy"
            or config.get("Healthcheck", {}).get("Test") != ["CMD", *service.healthcheck]
        ):
            raise DockerComposeError(
                "Every named service requires its approved health check and healthy running state."
            )

    async def _inventory_async(self) -> _Inventory:
        found: dict[str, tuple[dict[str, Any], ...]] = {}
        for kind in ("container", "network", "volume"):
            identifiers: set[str] = set()
            for selector in (
                f"label={self._PROJECT_LABEL}={self._project}",
                f"label={self._LEASE_LABEL}={self.lease_id}",
                f"name={self._project}",
            ):
                arguments = (
                    kind,
                    "ls",
                    *(("--all", "--no-trunc") if kind == "container" else ("--no-trunc",) if kind == "network" else ()),
                    "--filter",
                    selector,
                    "--format",
                    "{{.Name}}" if kind == "volume" else "{{.ID}}",
                )
                output = await self._command_async(arguments)
                for identifier in output.splitlines():
                    pattern = r"[a-zA-Z0-9][a-zA-Z0-9_.-]*" if kind == "volume" else r"[0-9a-f]{64}"
                    if not re.fullmatch(pattern, identifier):
                        raise DockerComposeError("Docker inventory returned an invalid resource identity.")
                    identifiers.add(identifier)
            if len(identifiers) > 64:
                raise DockerComposeError("Unexpected resource count in reserved Compose namespace.")
            found[kind] = await self._inspect_async(kind, tuple(sorted(identifiers))) if identifiers else ()
        return _Inventory(containers=found["container"], networks=found["network"], volumes=found["volume"])

    async def _inspect_async(self, kind: str, identifiers: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        output = await self._command_async((kind, "inspect", *identifiers))
        try:
            items = json.loads(output)
        except json.JSONDecodeError as error:
            raise DockerComposeError("Docker inspection returned invalid JSON.") from error
        if (
            not isinstance(items, list)
            or len(items) != len(identifiers)
            or not all(isinstance(item, dict) for item in items)
        ):
            raise DockerComposeError("Docker inspection returned missing or malformed resource records.")
        if kind != "image":
            actual = [item.get("Name") if kind == "volume" else item.get("Id") for item in items]
            if len(set(actual)) != len(actual) or set(actual) != set(identifiers):
                raise DockerComposeError("Docker inspection identity does not match the requested resources.")
        return tuple(items)

    async def _command_async(
        self, arguments: tuple[str, ...], *, input_text: str | None = None, timeout_seconds: float = _COMMAND_TIMEOUT
    ) -> str:
        result = await self._runner.run_async(
            arguments=arguments, input_text=input_text, timeout_seconds=timeout_seconds
        )
        if result.timed_out or result.truncated or result.returncode != 0:
            raise DockerComposeError(
                f"Docker control failed: exit={result.returncode}, "
                f"timeout={result.timed_out}, truncated={result.truncated}."
            )
        return result.stdout

    async def _compose_async(self, arguments: tuple[str, ...], *, timeout_seconds: float) -> str:
        return await self._command_async(
            ("compose", "--project-name", self._project, "--env-file", os.devnull, "--file", "-", *arguments),
            input_text=self._manifest,
            timeout_seconds=timeout_seconds,
        )

    def _ownership_labels(self) -> dict[str, str]:
        return {self._RUN_LABEL: self.run_id, self._LEASE_LABEL: self.lease_id}

    def _container_name(self, service: ComposeServiceSpec) -> str:
        return f"{self._project}-{service.name}"

    @staticmethod
    def _tmpfs(service: ComposeServiceSpec) -> str:
        return f"rw,noexec,nosuid,nodev,size={service.tmpfs_bytes},uid={service.uid},gid={service.gid},mode=0700"

    def _document(self) -> dict[str, Any]:
        return {
            "services": {
                service.name: {
                    "image": service.image,
                    "platform": service.platform.value,
                    "pull_policy": "never",
                    "container_name": self._container_name(service),
                    "labels": self._ownership_labels(),
                    "entrypoint": [],
                    "command": list(service.command),
                    "user": f"{service.uid}:{service.gid}",
                    "read_only": True,
                    "privileged": False,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges:true"],
                    "restart": "no",
                    "ipc": "private",
                    "cpus": service.cpu_millis / 1000,
                    "mem_limit": service.memory_bytes,
                    "memswap_limit": service.memory_bytes,
                    "pids_limit": service.pids_limit,
                    "tmpfs": ["/tmp:" + self._tmpfs(service)],
                    "networks": ["challenge"],
                    "healthcheck": {
                        "test": ["CMD", *service.healthcheck],
                        "interval": "1s",
                        "timeout": "2s",
                        "retries": 3,
                    },
                }
                for service in self._spec.services
            },
            "networks": {
                "challenge": {
                    "name": self._network,
                    "driver": "bridge",
                    "internal": True,
                    "enable_ipv6": False,
                    "labels": self._ownership_labels(),
                }
            },
        }
