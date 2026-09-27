# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Exact credential-free Codex configuration staging, with no host archive extraction."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import io
import json
import tarfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from pyrit.executor.workflow.docker_engine import DockerEngineError
from pyrit.executor.workflow.docker_guest_auth import codex_gateway_config

if TYPE_CHECKING:
    from pyrit.executor.workflow.docker_engine import DockerEngineClient


@dataclass(frozen=True, kw_only=True)
class CodexConfigStaging:
    """An exact expected subtree and successful daemon readback, not live model-route qualification."""

    container_id: str
    mount_path: PurePosixPath
    subtree: PurePosixPath
    directories: tuple[str, ...]
    config_path: str
    uid: int
    gid: int
    sha256: str
    content: bytes = field(repr=False)


class DockerCodexConfigStager:
    """Write only a generated user configuration under a fresh, approved tmpfs subtree."""

    _GO_DIRECTORY = 1 << 31
    _ARCHIVE_LIMIT = 65_536

    def __init__(self, *, engine: DockerEngineClient) -> None:
        """Use the same host-only Engine transport and ownership boundary as the agent lease."""
        self._engine = engine

    async def stage_async(
        self,
        *,
        container_id: str,
        home: PurePosixPath,
        mount_path: PurePosixPath,
        uid: int,
        gid: int,
        model: str,
        base_url: str,
    ) -> CodexConfigStaging:
        """
        Create a new credential-free .codex subtree and verify its entire bounded archive.

        Existing non-mount HOME directories are rejected: Engine path stat has no
        UID/GID, so this slice does not infer their ownership. Missing HOME ancestry
        can be created only below an approved, owner-only tmpfs root.

        Returns:
            CodexConfigStaging: Verified content digest and subtree identity.

        Raises:
            DockerEngineError: If ancestry, vacancy, extraction acknowledgement or readback is unverified.
            ValueError: If the staging paths or owner are not explicit.
        """
        if (
            not home.is_absolute()
            or not mount_path.is_absolute()
            or ".." in home.parts
            or ".." in mount_path.parts
            or not home.is_relative_to(mount_path)
            or type(uid) is not int
            or type(gid) is not int
            or uid <= 0
            or gid <= 0
        ):
            raise ValueError("Config staging requires an explicit approved tmpfs HOME and nonroot owner.")
        relative = home.relative_to(mount_path)
        if len(relative.parts) > 8:
            raise ValueError("Config staging supports at most eight fresh HOME path components.")
        content = codex_gateway_config(model=model, base_url=base_url).encode("utf-8")
        if len(content) > 8192:
            raise ValueError("Generated Codex configuration exceeds its fixed size limit.")
        await self._check_ancestors_async(container_id=container_id, path=mount_path)
        names = list(relative.parts) + [".codex"]
        directories = tuple("/".join(names[:index]) for index in range(1, len(names) + 1))
        subtree = mount_path / names[0]
        status, _, _ = await self._engine._config_archive_async(container_id=container_id, method="HEAD", path=subtree)
        if status != 404:
            raise DockerEngineError("Config staging refuses existing HOME/config state instead of overwriting it.")
        expected = CodexConfigStaging(
            container_id=container_id,
            mount_path=mount_path,
            subtree=subtree,
            directories=directories,
            config_path=directories[-1] + "/config.toml",
            uid=uid,
            gid=gid,
            content=content,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        archive = await asyncio.to_thread(self._build_archive, expected)
        status, _, body = await self._engine._config_archive_async(
            container_id=container_id, method="PUT", path=mount_path, archive=archive
        )
        if status != 200 or body:
            raise DockerEngineError("Engine did not acknowledge the exact credential-free config archive.")
        await self.verify_async(expected)
        return expected

    async def verify_async(self, expected: CodexConfigStaging) -> None:
        """
        Recheck ancestry and exact subtree paths, types, ownership, modes and content without extraction.

        Raises:
            DockerEngineError: If the daemon readback cannot establish the pinned user configuration.
        """
        await self._check_ancestors_async(container_id=expected.container_id, path=expected.mount_path)
        for name in expected.directories:
            path = expected.mount_path / name
            status, headers, _ = await self._engine._config_archive_async(
                container_id=expected.container_id, method="HEAD", path=path
            )
            if status != 200:
                raise DockerEngineError("Staged config directory is missing.")
            self._check_directory_stat(headers=headers, path=path, private=True)
        status, headers, archive = await self._engine._config_archive_async(
            container_id=expected.container_id, method="GET", path=expected.subtree
        )
        if status != 200:
            raise DockerEngineError("Engine config readback failed.")
        self._check_directory_stat(headers=headers, path=expected.subtree, private=True)
        await asyncio.to_thread(self._verify_archive, expected=expected, archive=archive)

    async def _check_ancestors_async(self, *, container_id: str, path: PurePosixPath) -> None:
        for index in range(2, len(path.parts) + 1):
            ancestor = PurePosixPath(*path.parts[:index])
            status, headers, _ = await self._engine._config_archive_async(
                container_id=container_id, method="HEAD", path=ancestor
            )
            if status != 200:
                raise DockerEngineError("Approved config mount ancestry is missing.")
            self._check_directory_stat(headers=headers, path=ancestor, private=ancestor == path)

    @classmethod
    def _check_directory_stat(cls, *, headers: dict[str, str], path: PurePosixPath, private: bool) -> None:
        encoded = headers.get("x-docker-container-path-stat", "")
        if not encoded or len(encoded) > 4096:
            raise DockerEngineError("Engine archive path stat is missing or oversized.")
        try:
            value = json.loads(base64.b64decode(encoded, validate=True))
        except (ValueError, UnicodeError, binascii.Error):
            raise DockerEngineError("Engine archive path stat is malformed.") from None
        if (
            not isinstance(value, dict)
            or value.get("name") != path.name
            or value.get("linkTarget") != ""
            or type(value.get("mode")) is not int
        ):
            raise DockerEngineError("Engine archive path stat cannot prove exact non-link directory identity.")
        mode = value["mode"]
        if mode & ~0o777 != cls._GO_DIRECTORY or (private and mode & 0o777 != 0o700):
            raise DockerEngineError("Config ancestry must be directories, without links or unsafe private permissions.")

    @staticmethod
    def _build_archive(expected: CodexConfigStaging) -> bytes:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in expected.directories:
                entry = tarfile.TarInfo(name)
                entry.type, entry.mode, entry.uid, entry.gid = tarfile.DIRTYPE, 0o700, expected.uid, expected.gid
                archive.addfile(entry)
            entry = tarfile.TarInfo(expected.config_path)
            entry.type, entry.mode, entry.uid, entry.gid = tarfile.REGTYPE, 0o600, expected.uid, expected.gid
            entry.size = len(expected.content)
            archive.addfile(entry, io.BytesIO(expected.content))
        return buffer.getvalue()

    @classmethod
    def _verify_archive(cls, *, expected: CodexConfigStaging, archive: bytes) -> None:
        if not archive or len(archive) > cls._ARCHIVE_LIMIT:
            raise DockerEngineError("Config readback archive is missing or exceeds the fixed bound.")
        wanted = {*expected.directories, expected.config_path}
        seen: set[str] = set()
        data_end = 0
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as contents:
                for entry in contents:
                    if (
                        entry.name not in wanted
                        or entry.name in seen
                        or entry.linkname
                        or entry.pax_headers
                        or entry.uid != expected.uid
                        or entry.gid != expected.gid
                    ):
                        raise DockerEngineError(
                            "Config archive contains unapproved paths, links, metadata or ownership."
                        )
                    seen.add(entry.name)
                    data_end = max(data_end, entry.offset_data + ((entry.size + 511) // 512) * 512)
                    if entry.name == expected.config_path:
                        if (
                            entry.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                            or entry.mode != 0o600
                            or entry.size != len(expected.content)
                        ):
                            raise DockerEngineError("Config readback must be the exact owner-only regular file.")
                        file = contents.extractfile(entry)
                        if file is None:
                            raise DockerEngineError("Config archive did not expose its regular file bytes.")
                        with file:
                            value = file.read(len(expected.content) + 1)
                        if value != expected.content or hashlib.sha256(value).hexdigest() != expected.sha256:
                            raise DockerEngineError("Config readback bytes do not match the pinned template digest.")
                    elif entry.type != tarfile.DIRTYPE or entry.mode != 0o700 or entry.size != 0:
                        raise DockerEngineError("Config readback directories require exact owner-only modes.")
        except (tarfile.TarError, ValueError, EOFError):
            raise DockerEngineError("Config readback archive is invalid or unsupported.") from None
        if seen != wanted:
            raise DockerEngineError("Config readback omitted an expected directory or file.")
        if len(archive) < data_end + 1024 or any(archive[data_end:]):
            raise DockerEngineError("Config readback archive has missing terminators or unapproved trailing data.")
