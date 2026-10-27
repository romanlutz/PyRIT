# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Optional fresh, local, run-owned task volume; strict default Compose mounts remain unchanged."""

from __future__ import annotations

import base64
import copy
import io
import json
import re
import secrets
import tarfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

from pyrit.executor.workflow.docker_agent import DockerAgentCliProfile, DockerStopOnlyAgentLease
from pyrit.executor.workflow.docker_compose import (
    ComposeEnvironmentSpec,
    ComposeServiceSpec,
    ComposeTmpfsSpec,
    _Inventory,
)
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.models.environment_lease import EnvironmentResourceHandle

if TYPE_CHECKING:
    from ipaddress import IPv4Address, IPv4Network

    from pyrit.executor.workflow.docker_command import DockerCommandRunner
    from pyrit.executor.workflow.docker_compose import ComposeAllocation
    from pyrit.executor.workflow.docker_guest_auth import DockerGuestAuth


@dataclass(frozen=True, kw_only=True)
class DockerNamedWorkspaceSpec:
    """Explicit task-root capability and separate qualification, never a caller-selected volume name."""

    root: PurePosixPath
    qualification_ref: str
    ready: bool = False
    capability: str = field(default="run-owned-task-volume-v1", init=False)

    def __post_init__(self) -> None:
        """
        Require one canonical non-system task root and explicit profile qualification.

        Raises:
            ValueError: If the root or qualification is unsafe.
        """
        if (
            not isinstance(self.root, PurePosixPath)
            or not self.root.is_absolute()
            or not re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", str(self.root))
            or ".." in self.root.parts
            or len(str(self.root)) > 128
            or any(
                self.root.is_relative_to(path) or PurePosixPath(path).is_relative_to(self.root)
                for path in ComposeTmpfsSpec._RESERVED_PATHS
            )
            or not self.qualification_ref
            or len(self.qualification_ref) > 256
            or type(self.ready) is not bool
        ):
            raise ValueError("A named task workspace needs an exact non-system root and bounded qualification.")


@dataclass(frozen=True, kw_only=True)
class DockerWorkspaceIdentity:
    """Verified volume generation/mount identity, without daemon-host mount paths or filesystem claims."""

    run_id: str
    environment_id: str
    volume_name: str
    nonce: str
    created_at: str
    task_root: str
    container_ids: tuple[str, ...]
    capability: str = "run-owned-task-volume-v1"


class DockerNamedWorkspaceLease(DockerStopOnlyAgentLease):
    """Opt in to one fresh named volume, reserved before the project and released after every container."""

    _VOLUME_NONCE_LABEL = "org.pyrit.native.workspace-nonce"
    _VOLUME_CAPABILITY_LABEL = "org.pyrit.native.workspace-capability"

    def __init__(
        self,
        *,
        workspace: DockerNamedWorkspaceSpec,
        run_id: str,
        spec: ComposeEnvironmentSpec,
        runner: DockerCommandRunner,
        engine: DockerEngineClient,
        profile: DockerAgentCliProfile,
        guest_auth: DockerGuestAuth,
        stop_timeout_seconds: float,
        bridge_address: IPv4Address,
        bridge_subnet: IPv4Network,
    ) -> None:
        """
        Accept a separately qualified two-service workspace capability, preserving all other guards.

        Raises:
            ValueError: If topology, root, profile, owners or helper paths cannot share this exact volume.
        """
        root = workspace.root
        if (
            not workspace.ready
            or len(spec.services) != 2
            or profile.config.agent_workdir != root
            or profile.executable.is_relative_to(root)
            or len({(service.uid, service.gid) for service in spec.services}) != 1
            or any(
                root.is_relative_to(mount.path) or PurePosixPath(mount.path).is_relative_to(root)
                for service in spec.services
                for mount in service.tmpfs_mounts
            )
            or any(
                PurePosixPath(argv[0]).is_relative_to(root)
                for service in spec.services
                for argv in (service.command, service.healthcheck)
            )
        ):
            raise ValueError("Named workspace requires its separately qualified exact root and same nonroot owner.")
        self.workspace_spec = workspace
        self._workspace_nonce = secrets.token_hex(24)
        self._workspace_name = "pyrit-workspace-" + self._workspace_nonce
        self._volume_attempted = False
        self._volume_confirmed = False
        self._volume_signature: str | None = None
        self._volume_created_at = ""
        self._volume_mountpoint = ""
        self._workspace_initialized = False
        self._bridge_address, self._bridge_subnet = bridge_address, bridge_subnet
        super().__init__(
            run_id=run_id,
            spec=spec,
            runner=runner,
            engine=engine,
            profile=profile,
            guest_auth=guest_auth,
            stop_timeout_seconds=stop_timeout_seconds,
        )

    @property
    def workspace_name(self) -> str:
        """The generated nonce resource name, never a caller-selected mount path."""
        return self._workspace_name

    def _workspace_allowed(self, *, agent: ComposeServiceSpec, workdir: PurePosixPath) -> bool:
        return workdir == self.workspace_spec.root

    def _document(self) -> dict[str, Any]:
        document = super()._document()
        document["networks"]["challenge"]["ipam"] = {
            "driver": "default",
            "config": [{"subnet": str(self._bridge_subnet), "gateway": str(self._bridge_address)}],
        }
        document["volumes"] = {"task_workspace": {"external": True, "name": self.workspace_name}}
        for service in document["services"].values():
            service["volumes"] = [
                {
                    "type": "volume",
                    "source": "task_workspace",
                    "target": str(self.workspace_spec.root),
                    "read_only": False,
                    "volume": {"nocopy": True},
                }
            ]
        return document

    def _volume_labels(self) -> dict[str, str]:
        return {
            self._RUN_LABEL: self.run_id,
            self._LEASE_LABEL: self.lease_id,
            self._VOLUME_NONCE_LABEL: self._workspace_nonce,
            self._VOLUME_CAPABILITY_LABEL: self.workspace_spec.capability,
        }

    def _volume_metadata(self, value: dict[str, Any] | None) -> str:
        if (
            value is None
            or value.get("Name") != self.workspace_name
            or value.get("Driver") != "local"
            or value.get("Scope") != "local"
            or value.get("Options") not in (None, {})
            or value.get("Labels") != self._volume_labels()
            or not isinstance(value.get("CreatedAt"), str)
            or not 1 <= len(value["CreatedAt"]) <= 64
        ):
            raise DockerEngineError("Workspace volume ownership/driver/generation is unverified; quarantine required.")
        return json.dumps(
            {key: value.get(key) for key in ("Name", "Driver", "Scope", "Labels", "Options", "CreatedAt")},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    async def _create_workspace_async(self, handle: EnvironmentResourceHandle) -> str:
        if await self._engine.inspect_volume_async(self.workspace_name) is not None:
            raise DockerEngineError("Workspace name already exists; it must not be adopted or deleted.")
        if await self._engine.containers_using_volume_async(self.workspace_name):
            raise DockerEngineError("Workspace name already has attachments.")
        self._volume_attempted = True
        created = await self._engine.create_volume_async(name=self.workspace_name, labels=self._volume_labels())
        signature = self._volume_metadata(created)
        if self._volume_metadata(await self._engine.inspect_volume_async(self.workspace_name)) != signature:
            raise DockerEngineError("Workspace create acknowledgement and post-read differ; quarantine required.")
        if await self._engine.containers_using_volume_async(self.workspace_name):
            raise DockerEngineError("Workspace was attached before the owned project existed; quarantine required.")
        self._volume_signature = signature
        self._volume_created_at = created["CreatedAt"]
        mountpoint = created.get("Mountpoint")
        observed = await self._engine.inspect_volume_async(self.workspace_name)
        if (
            not isinstance(mountpoint, str)
            or not mountpoint
            or observed is None
            or observed.get("Mountpoint") != mountpoint
        ):
            raise DockerEngineError("Volume daemon mountpoint attribution differs; no host path will be adopted.")
        self._volume_mountpoint = mountpoint
        self._volume_confirmed = True
        return self.workspace_name

    async def _release_workspace_async(self, handle: EnvironmentResourceHandle) -> None:
        if not self._volume_attempted:
            return
        if not self._volume_confirmed:
            raise DockerEngineError("Workspace create was ambiguous; refusing deletion without verified ownership.")
        if (
            self._volume_metadata(await self._engine.inspect_volume_async(self.workspace_name))
            != self._volume_signature
        ):
            raise DockerEngineError("Workspace generation changed; refusing deletion.")
        if await self._engine.containers_using_volume_async(self.workspace_name):
            raise DockerEngineError("Workspace still has attached containers; refusing volume deletion.")
        inventory = await super()._inventory_async()
        if inventory.containers:
            raise DockerEngineError("Owned containers remain; workspace deletion is forbidden.")
        await self._engine.remove_volume_async(self.workspace_name)
        if await self._engine.inspect_volume_async(self.workspace_name) is not None:
            raise DockerEngineError("Workspace delete was not observed absent; no retry or adoption.")

    async def _acquire_async(self) -> ComposeAllocation:
        handle = EnvironmentResourceHandle(
            run_id=self.run_id,
            provider="docker_volume",
            resource_id=self.workspace_name,
            kind=self.workspace_spec.capability,
        )
        await self._acquire_resource_async(
            handle=handle, acquire_async=self._create_workspace_async, release_async=self._release_workspace_async
        )
        allocation = await super()._acquire_async()
        await self.verify_workspace_async()
        target = next(service for service in self._spec.services if "agent" not in service.roles)
        container_id = allocation.container_id(target.name)
        await self._empty_root_async(container_id=container_id, initialized=False)
        archive = self._root_archive(uid=target.uid, gid=target.gid)
        await self._engine._config_archive_async(
            container_id=container_id, method="PUT", path=self.workspace_spec.root, archive=archive
        )
        await self._empty_root_async(container_id=container_id, initialized=True)
        self._workspace_initialized = True
        return allocation

    async def verify_workspace_async(self) -> DockerWorkspaceIdentity:
        """
        Revalidate the exact volume generation and both service mount identities, without reading task bytes.

        Returns:
            DockerWorkspaceIdentity: Run-owned shared storage identity, not a content/audit verdict.

        Raises:
            DockerEngineError: If volume ownership or any attachment is missing, foreign, or changed.
        """
        if not self._volume_confirmed or self._allocation is None:
            raise DockerEngineError("Workspace ownership and allocation are not confirmed.")
        if (
            self._volume_metadata(await self._engine.inspect_volume_async(self.workspace_name))
            != self._volume_signature
        ):
            raise DockerEngineError("Workspace volume generation changed.")
        identifiers = tuple(identifier for _, identifier in self._allocation.containers)
        if set(await self._engine.containers_using_volume_async(self.workspace_name)) != set(identifiers):
            raise DockerEngineError("Workspace attachments are not exactly the two owned services.")
        for service in self._spec.services:
            observed = await self._engine.inspect_container_async(self._allocation.container_id(service.name))
            self._validate_runtime_state(service=service, container=observed)
        return DockerWorkspaceIdentity(
            run_id=self.run_id,
            environment_id=self.lease_id,
            volume_name=self.workspace_name,
            nonce=self._workspace_nonce,
            created_at=self._volume_created_at,
            task_root=str(self.workspace_spec.root),
            container_ids=identifiers,
        )

    async def _audit_engine_async(self, *, require_agent_running: bool) -> dict[str, dict[str, Any]]:
        result = await super()._audit_engine_async(require_agent_running=require_agent_running)
        await self.verify_workspace_async()
        return result

    async def _inventory_async(self) -> _Inventory:
        observed = await super()._inventory_async()
        others = []
        for volume in observed.volumes:
            if volume.get("Name") == self.workspace_name:
                if not self._volume_confirmed or self._volume_metadata(volume) != self._volume_signature:
                    raise DockerEngineError("Compose inventory observed an unverified workspace volume.")
            else:
                others.append(volume)
        return _Inventory(containers=observed.containers, networks=observed.networks, volumes=tuple(others))

    def _validate_runtime_state(self, *, service: ComposeServiceSpec, container: dict[str, Any]) -> None:
        host, mounts = container.get("HostConfig"), container.get("Mounts")
        expected = {
            "Type": "volume",
            "Source": self.workspace_name,
            "Target": str(self.workspace_spec.root),
            "ReadOnly": False,
            "VolumeOptions": {"NoCopy": True},
        }
        if (
            not isinstance(host, dict)
            or host.get("Mounts") != [expected]
            or host.get("Binds")
            or not isinstance(mounts, list)
        ):
            raise DockerEngineError(
                "Workspace requires exactly the approved named mount, never binds or anonymous volumes."
            )
        selected = [mount for mount in mounts if mount.get("Destination") == str(self.workspace_spec.root)]
        if (
            len(selected) != 1
            or any(
                selected[0].get(key) != value
                for key, value in {
                    "Type": "volume",
                    "Name": self.workspace_name,
                    "Driver": "local",
                    "RW": True,
                    "Mode": "",
                    "Propagation": "",
                    "Source": self._volume_mountpoint,
                }.items()
            )
            or set(selected[0]) - {"Type", "Name", "Source", "Driver", "Destination", "Mode", "RW", "Propagation"}
        ):
            raise DockerEngineError("Effective workspace name/type/path/mode differs from its owned capability.")
        for value in container.get("Config", {}).get("Env", []):
            name, _, path = value.partition("=")
            if name in {"HOME", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "DOCKER_CONFIG", "AZURE_CONFIG_DIR"} and (
                PurePosixPath(path).is_relative_to(self.workspace_spec.root)
                or self.workspace_spec.root.is_relative_to(PurePosixPath(path))
            ):
                raise DockerEngineError("Task workspace cannot mount over a home, cache or credential path.")
        normalized = copy.deepcopy(container)
        normalized["HostConfig"]["Mounts"] = []
        normalized["Mounts"] = [mount for mount in mounts if mount is not selected[0]]
        super()._validate_runtime_state(service=service, container=normalized)

    async def _empty_root_async(self, *, container_id: str, initialized: bool) -> None:
        _, headers, raw = await self._engine._config_archive_async(
            container_id=container_id, method="GET", path=self.workspace_spec.root
        )
        try:
            stat = json.loads(base64.b64decode(headers["x-docker-container-path-stat"], validate=True))
            with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
                members = archive.getmembers()
            member = members[0] if len(members) == 1 else None
            expected_owner = (self._spec.services[0].uid, self._spec.services[0].gid)
            if (
                not isinstance(stat, dict)
                or stat.get("name") != self.workspace_spec.root.name
                or stat.get("linkTarget") != ""
                or type(stat.get("mode")) is not int
                or stat["mode"] & ~0o777 != 1 << 31
                or member is None
                or not member.isdir()
                or member.name.rstrip("/") != self.workspace_spec.root.name
                or member.size != 0
                or member.linkname
                or member.pax_headers
                or member.sparse is not None
                or member.mode & ~0o777
                or (initialized and ((member.uid, member.gid) != expected_owner or member.mode != 0o700))
                or len(raw) % 512
                or len(raw) - member.offset_data < 1024
                or any(raw[member.offset_data :])
            ):
                raise DockerEngineError("Workspace root is not the verified fresh empty non-link directory.")
        except (ValueError, KeyError, tarfile.TarError) as error:
            raise DockerEngineError("Workspace empty-root metadata/readback is unverified.") from error

    @staticmethod
    def _root_archive(*, uid: int, gid: int) -> bytes:
        result = io.BytesIO()
        with tarfile.open(fileobj=result, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            root = tarfile.TarInfo(".")
            root.type, root.mode, root.uid, root.gid, root.mtime = tarfile.DIRTYPE, 0o700, uid, gid, 0
            archive.addfile(root)
        return result.getvalue()
