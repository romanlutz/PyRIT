# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Pinned public Inspect file-probe setup on an owned native Compose sandbox."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
import stat
import tarfile
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pyrit.executor.workflow.docker_compose import (
    ComposeEnvironmentSpec,
    ComposePlatform,
    DockerComposeEnvironmentLease,
)
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.models.environment_lease import EnvironmentCapability, EnvironmentLeaseState

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.executor.workflow.docker_command import DockerCommandRunner


class InspectFileProbeComposeLease(DockerComposeEnvironmentLease):
    """Reuse only Inspect's pinned ``touch foo.txt`` setup, never its Task Python or sandbox lifecycle."""

    SOURCE_REVISION = "674e62823b226b5cf2449db8dd014276533841f7"
    SOURCE_PATH = "tests/util/sandbox/sandbox_setup.sh"
    SOURCE_SHA256 = "5ddb5ffce1c1af64195f8be5780fa01e1db418e42cea1bdf9d7bac61eb40df6e"
    IMAGE = "python:3.12-slim@sha256:44ff437bba879d4941b710a369a8f19266aea34b29002807f0c487fabc9eec9b"
    WORKDIR = PurePosixPath("/tmp")
    FILE = WORKDIR / "foo.txt"
    _FILE_LIMIT = 32_768
    _SETUP_BYTES = b"#!/usr/bin/env bash\n\ntouch foo.txt\n"

    def __init__(
        self,
        *,
        run_id: str,
        spec: ComposeEnvironmentSpec,
        runner: DockerCommandRunner,
        engine: DockerEngineClient,
        setup_script_path: Path,
    ) -> None:
        """Require one pinned, nonroot sandbox and the exact local copy of the upstream setup script."""
        self._validate_spec(spec)
        self._validate_source(setup_script_path)
        super().__init__(
            run_id=run_id,
            spec=spec,
            runner=runner,
            capabilities=frozenset({EnvironmentCapability.SETUP, EnvironmentCapability.HEALTH_CHECK}),
        )
        self._engine = engine
        self._service = spec.services[0]

    async def observe_file_async(self) -> bool:
        """
        Observe the original file-existence rule while the owned sandbox is still live.

        Returns:
            bool: Whether the original ``check_file`` rule can read ``foo.txt`` in the approved working directory.

        Raises:
            RuntimeError: If setup has not completed or the sandbox has already been released.
            DockerEngineError: If the file or sandbox identity cannot be verified.
        """
        if self.snapshot().state is not EnvironmentLeaseState.READY or not self.snapshot().setup_completed:
            raise RuntimeError("File-probe observation requires an acquired, setup-complete sandbox.")
        container_id = await self._verified_container_id_async()
        return await self._read_file_async(container_id=container_id) is not None

    async def _setup_async(self) -> None:
        container_id = await self._verified_container_id_async()
        if await self._read_file_async(container_id=container_id) is not None:
            raise DockerEngineError("File-probe setup refuses an existing file rather than adopting or touching it.")
        archive = await asyncio.to_thread(self._empty_file_archive, uid=self._service.uid, gid=self._service.gid)
        status, _, body = await self._engine._config_archive_async(
            container_id=container_id, method="PUT", path=self.WORKDIR, archive=archive
        )
        if status != 200 or body:
            raise DockerEngineError("Engine did not acknowledge the exact empty file-probe asset.")
        if await self._read_file_async(container_id=container_id) != b"":
            raise DockerEngineError("Staged file-probe asset is not the pinned empty regular file.")

    async def _verified_container_id_async(self) -> str:
        if self._allocation is None:
            raise RuntimeError("File-probe setup requires an owned Compose allocation.")
        inventory = await self._inventory_async()
        self._validate_inventory(inventory, require_ready=True)
        container_id = self._allocation.container_id(self._service.name)
        expected = inventory.containers[0]
        observed = await self._engine.inspect_container_async(container_id)
        immutable = ("Id", "Name", "Image", "Config", "HostConfig", "Mounts")
        if any(
            field not in expected or field not in observed or expected[field] != observed[field] for field in immutable
        ):
            raise DockerEngineError("Engine asset staging and Compose do not observe the same approved container.")
        self._validate_container(service=self._service, container=observed, network_id=self._allocation.network_id)
        self._validate_runtime_state(service=self._service, container=observed)
        return container_id

    async def _read_file_async(self, *, container_id: str) -> bytes | None:
        status, headers, _ = await self._engine._config_archive_async(
            container_id=container_id, method="HEAD", path=self.WORKDIR
        )
        if status != 200:
            raise DockerEngineError("The approved file-probe tmpfs is missing.")
        self._engine._check_archive_directory_stat(headers=headers, path=self.WORKDIR, private=True)
        status, headers, _ = await self._engine._config_archive_async(
            container_id=container_id, method="HEAD", path=self.FILE
        )
        if status == 404:
            return None
        size, mode = self._check_file_stat(headers)
        status, headers, archive = await self._engine._config_archive_async(
            container_id=container_id, method="GET", path=self.FILE
        )
        if status != 200 or self._check_file_stat(headers) != (size, mode):
            raise DockerEngineError("File-probe readback identity changed during observation.")
        return await asyncio.to_thread(
            self._read_archive, archive=archive, size=size, mode=mode, uid=self._service.uid, gid=self._service.gid
        )

    @classmethod
    def _validate_spec(cls, spec: ComposeEnvironmentSpec) -> None:
        if not isinstance(spec, ComposeEnvironmentSpec) or len(spec.services) != 1:
            raise ValueError("The public file-probe fixture requires exactly one sandbox service.")
        service = spec.services[0]
        if (
            service.name != "default"
            or service.roles != frozenset({"target", "grader"})
            or service.parent_name is not None
            or service.image != cls.IMAGE
            or service.platform is not ComposePlatform.AMD64
            or service.command != ("/usr/bin/sleep", "infinity")
            or service.healthcheck != ("/usr/bin/true",)
            or service.uid != 1000
            or service.gid != 1000
            or service.runtime_state is not None
        ):
            raise ValueError("File-probe roles, image, commands, owner and default tmpfs must match the pin.")

    @classmethod
    def _validate_source(cls, path: Path) -> None:
        if (
            not path.is_absolute()
            or path.name != "sandbox_setup.sh"
            or ".." in path.parts
            or path.resolve(strict=True) != path
            or not stat.S_ISREG(path.lstat().st_mode)
            or path.stat().st_size != len(cls._SETUP_BYTES)
        ):
            raise ValueError("File-probe setup requires the exact regular, unaliased upstream script path.")
        with path.open("rb") as stream:
            source = stream.read(129)
        if source != cls._SETUP_BYTES or hashlib.sha256(source).hexdigest() != cls.SOURCE_SHA256:
            raise ValueError("File-probe setup differs from the pinned public Inspect source; no script is executed.")

    @classmethod
    def _check_file_stat(cls, headers: dict[str, str]) -> tuple[int, int]:
        encoded = headers.get("x-docker-container-path-stat", "")
        if not encoded or len(encoded) > 4096:
            raise DockerEngineError("File-probe path stat is missing or oversized.")
        try:
            value = json.loads(base64.b64decode(encoded, validate=True))
        except (ValueError, UnicodeError, binascii.Error):
            raise DockerEngineError("File-probe path stat is malformed.") from None
        if not isinstance(value, dict):
            raise DockerEngineError("File-probe path stat is not a structured file observation.")
        size, mode = value.get("size"), value.get("mode")
        if (
            value.get("name") != cls.FILE.name
            or value.get("linkTarget") != ""
            or type(mode) is not int
            or type(size) is not int
            or mode & ~0o777
            or not mode & 0o400
            or not 0 <= size <= cls._FILE_LIMIT
        ):
            raise DockerEngineError("File-probe observation requires a bounded, owner-readable regular file.")
        return size, mode

    @classmethod
    def _empty_file_archive(cls, *, uid: int, gid: int) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            file = tarfile.TarInfo(cls.FILE.name)
            file.type, file.mode, file.uid, file.gid = tarfile.REGTYPE, 0o600, uid, gid
            archive.addfile(file, io.BytesIO())
        return buffer.getvalue()

    @classmethod
    def _read_archive(cls, *, archive: bytes, size: int, mode: int, uid: int, gid: int) -> bytes:
        if not archive or len(archive) > 65_536:
            raise DockerEngineError("File-probe readback archive is missing or exceeds its fixed bound.")
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as contents:
                entries = list(contents)
                if len(entries) != 1:
                    raise DockerEngineError("File-probe readback has missing or extra paths.")
                entry = entries[0]
                if (
                    entry.name != cls.FILE.name
                    or not entry.isreg()
                    or entry.linkname
                    or entry.pax_headers
                    or entry.uid != uid
                    or entry.gid != gid
                    or entry.mode != mode
                    or entry.size != size
                ):
                    raise DockerEngineError("File-probe readback has unapproved path, type, owner or metadata.")
                stream = contents.extractfile(entry)
                if stream is None:
                    raise DockerEngineError("File-probe readback has no file content.")
                with stream:
                    data = stream.read(size + 1)
                data_end = entry.offset_data + ((size + 511) // 512) * 512
        except (tarfile.TarError, ValueError, EOFError):
            raise DockerEngineError("File-probe readback is not a valid uncompressed archive.") from None
        if len(data) != size or len(archive) < data_end + 1024 or any(archive[data_end:]):
            raise DockerEngineError("File-probe readback is truncated or contains unapproved trailing data.")
        return data
