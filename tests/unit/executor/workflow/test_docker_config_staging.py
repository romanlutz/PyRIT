# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import hashlib
import io
import tarfile
from pathlib import PurePosixPath
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from pyrit.executor.workflow.docker_agent import DockerSandboxLauncher
from pyrit.executor.workflow.docker_config_staging import DockerCodexConfigStager
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.executor.workflow.docker_guest_auth import codex_gateway_config
from pyrit.prompt_target.native_cli_models import NativeCliProtocol
from tests.unit.executor.workflow.test_docker_agent import config, make_agent
from tests.unit.executor.workflow.test_docker_engine import CONTAINER_ID, FakeEngine


def mutate_archive(data: bytes, defect: str) -> bytes:
    members: list[tuple[tarfile.TarInfo, bytes]] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as source:
        for entry in source:
            stream = source.extractfile(entry) if entry.isreg() else None
            members.append((copy.copy(entry), stream.read() if stream else b""))
            if stream:
                stream.close()
    entry, content = members[-1]
    if defect == "traversal":
        entry.name = "../outside"
    elif defect == "absolute":
        entry.name = "/host/file"
    elif defect == "symlink":
        entry.type, entry.linkname = tarfile.SYMTYPE, "/etc/passwd"
        entry.size, content = 0, b""
    elif defect == "hardlink":
        entry.type, entry.linkname = tarfile.LNKTYPE, "other"
        entry.size, content = 0, b""
    elif defect == "device":
        entry.type, entry.size, content = tarfile.CHRTYPE, 0, b""
    elif defect == "mode":
        entry.mode = 0o666
    elif defect == "uid":
        entry.uid = 0
    elif defect == "gid":
        entry.gid = 0
    elif defect == "digest":
        content = b"x" * len(content)
    elif defect == "length":
        content += b"x"
        entry.size = len(content)
    elif defect == "directory-mode":
        members[0][0].mode = 0o777
    elif defect == "directory-owner":
        members[0][0].uid = 0
    elif defect == "pax":
        entry.pax_headers = {"comment": "unapproved"}
    members[-1] = entry, content
    if defect == "missing":
        members.pop()
    elif defect == "duplicate":
        members.append(members[-1])
    elif defect == "extra":
        extra = tarfile.TarInfo("unexpected")
        members.append((extra, b""))
    result = io.BytesIO()
    with tarfile.open(
        fileobj=result,
        mode="w:gz" if defect == "compressed" else "w",
        format=tarfile.PAX_FORMAT if defect == "pax" else tarfile.USTAR_FORMAT,
    ) as archive:
        for member, value in members:
            archive.addfile(member, io.BytesIO(value) if member.isreg() else None)
    return result.getvalue()


@pytest.mark.parametrize("home", ["/tmp", "/tmp/home", "/tmp/users/agent"])
async def test_config_staging_creates_only_new_owned_directories_and_verified_file_async(home: str) -> None:
    fake = FakeEngine()
    engine = fake.client()
    stager = DockerCodexConfigStager(engine=engine)
    with patch.object(tarfile.TarFile, "extractall", side_effect=AssertionError("Never extract on host")):
        expected = await stager.stage_async(
            container_id=CONTAINER_ID,
            home=PurePosixPath(home),
            mount_path=PurePosixPath("/tmp"),
            uid=1000,
            gid=1000,
            model="codex-fixture",
            base_url="http://gateway.sandbox/v1",
        )
        await stager.verify_async(expected)
    assert len(fake.config_archive.writes) == 1 and fake.config_archive.reads == 2
    assert expected.sha256 == hashlib.sha256(expected.content).hexdigest()
    assert expected.content.decode() == codex_gateway_config(
        model="codex-fixture", base_url="http://gateway.sandbox/v1"
    )
    with tarfile.open(fileobj=io.BytesIO(fake.config_archive.writes[0]), mode="r:") as archive:
        entries = list(archive)
        assert [entry.name for entry in entries] == [*expected.directories, expected.config_path]
        assert all(entry.uid == entry.gid == 1000 and not entry.linkname for entry in entries)
        assert all(entry.mode == 0o700 for entry in entries[:-1]) and entries[-1].mode == 0o600
        assert archive.extractfile(entries[-1]).read() == expected.content
    assert all(CONTAINER_ID in request.url.path for request in fake.requests)
    await engine.close_async()


@pytest.mark.parametrize("existing", ["/tmp/home", "/tmp/.codex"])
async def test_existing_home_or_config_is_not_overwritten_async(existing: str) -> None:
    fake = FakeEngine()
    fake.config_archive.directories[existing] = 0o700
    engine = fake.client()
    with pytest.raises(DockerEngineError, match="existing HOME/config"):
        await DockerCodexConfigStager(engine=engine).stage_async(
            container_id=CONTAINER_ID,
            home=PurePosixPath("/tmp" if existing.endswith(".codex") else "/tmp/home"),
            mount_path=PurePosixPath("/tmp"),
            uid=1000,
            gid=1000,
            model="model",
            base_url="http://gateway.sandbox/v1",
        )
    assert not fake.config_archive.writes and fake.config_archive.directories[existing] == 0o700
    await engine.close_async()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mode", (1 << 31) | 0o777),
        ("mode", (1 << 27) | 0o700),
        ("mode", 0o700),
        ("mode", "2147484096"),
        ("name", "other"),
        ("linkTarget", "/outside"),
    ],
)
async def test_unverified_tmpfs_ancestry_blocks_before_write_async(*, field: str, value: Any) -> None:
    fake = FakeEngine()
    fake.config_archive.stat_change = lambda _, stat: stat.update({field: value})
    engine = fake.client()
    with pytest.raises(DockerEngineError):
        await DockerCodexConfigStager(engine=engine).stage_async(
            container_id=CONTAINER_ID,
            home=PurePosixPath("/tmp/home"),
            mount_path=PurePosixPath("/tmp"),
            uid=1000,
            gid=1000,
            model="model",
            base_url="http://gateway.sandbox/v1",
        )
    assert not fake.config_archive.writes
    await engine.close_async()


@pytest.mark.parametrize(
    "defect",
    [
        "traversal",
        "absolute",
        "symlink",
        "hardlink",
        "device",
        "mode",
        "uid",
        "gid",
        "digest",
        "length",
        "directory-mode",
        "directory-owner",
        "missing",
        "duplicate",
        "extra",
        "pax",
        "compressed",
    ],
)
async def test_readback_rejects_unsafe_or_inexact_tar_without_host_extraction_async(defect: str) -> None:
    fake = FakeEngine()
    fake.config_archive.readback = lambda data: mutate_archive(data, defect)
    engine = fake.client()
    with pytest.raises(DockerEngineError):
        await DockerCodexConfigStager(engine=engine).stage_async(
            container_id=CONTAINER_ID,
            home=PurePosixPath("/tmp/home"),
            mount_path=PurePosixPath("/tmp"),
            uid=1000,
            gid=1000,
            model="model",
            base_url="http://gateway.sandbox/v1",
        )
    assert len(fake.config_archive.writes) == fake.config_archive.reads == 1
    await engine.close_async()


@pytest.mark.parametrize("defect", ["oversize", "truncated", "trailing", "encoding"])
async def test_archive_byte_bounds_terminators_and_encoding_fail_closed_async(defect: str) -> None:
    fake = FakeEngine()
    if defect == "encoding":
        fake.config_archive.get_headers = {"content-encoding": "gzip"}
    else:
        fake.config_archive.readback = {
            "oversize": lambda _: b"x" * 65_537,
            "truncated": lambda data: data[:1000],
            "trailing": lambda data: data + b"hidden data",
        }[defect]
    engine = fake.client()
    with pytest.raises(DockerEngineError):
        await DockerCodexConfigStager(engine=engine).stage_async(
            container_id=CONTAINER_ID,
            home=PurePosixPath("/tmp/home"),
            mount_path=PurePosixPath("/tmp"),
            uid=1000,
            gid=1000,
            model="model",
            base_url="http://gateway.sandbox/v1",
        )
    await engine.close_async()


async def test_codex_is_staged_and_read_back_before_exec_but_claude_never_stages_async() -> None:
    for protocol in NativeCliProtocol:
        lease, _, fake, engine, approved = make_agent(run_config=config(protocol=protocol))
        await lease.acquire_async()
        if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
            assert len(fake.config_archive.writes) == fake.config_archive.reads == 1
            archive = fake.config_archive.writes[0]
            assert lease._guest_auth.token.get_secret_value().encode() not in archive
        else:
            assert not fake.config_archive.writes and fake.config_archive.reads == 0
        assert fake.created is None
        session = await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="inert")
        if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
            assert fake.config_archive.reads == 2
        await session.stop_async()
        await lease.close_async()
        await engine.close_async()


async def test_bad_staging_readback_rolls_back_owned_project_and_never_launches_cli_async() -> None:
    lease, command, fake, engine, _ = make_agent()
    fake.config_archive.readback = lambda data: mutate_archive(data, "digest")
    with pytest.raises(DockerEngineError):
        await lease.acquire_async()
    assert fake.created is None
    assert lease.snapshot().state == "closed" and not command.containers and not command.networks
    await engine.close_async()


async def test_config_drift_after_acquire_is_rejected_before_cli_exec_async() -> None:
    lease, _, fake, engine, approved = make_agent()
    await lease.acquire_async()
    fake.config_archive.readback = lambda data: mutate_archive(data, "mode")
    with pytest.raises(DockerEngineError):
        await DockerSandboxLauncher(lease=lease).launch_async(config=approved, prompt="inert")
    assert fake.created is None and not fake.started
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize("failure", ["put-error", "cancel-after-write"])
async def test_failed_or_cancelled_staging_never_claims_success_and_rolls_back_owned_resources_async(
    failure: str,
) -> None:
    lease, command, fake, engine, _ = make_agent()
    original = engine._config_archive_async
    written = asyncio.Event()

    async def partial_async(**kwargs: Any) -> tuple[int, dict[str, str], bytes]:
        result = await original(**kwargs)
        if kwargs["method"] == "PUT":
            written.set()
            await asyncio.Event().wait()
        return result

    if failure == "put-error":
        fake.config_archive.put_status = 500
        with pytest.raises(DockerEngineError):
            await lease.acquire_async()
    else:
        with patch.object(engine, "_config_archive_async", side_effect=partial_async):
            acquiring = asyncio.create_task(lease.acquire_async())
            await asyncio.wait_for(written.wait(), timeout=2)
            acquiring.cancel()
            with pytest.raises(asyncio.CancelledError):
                await acquiring
    assert lease.staged_codex_config is None and fake.created is None
    assert lease.snapshot().state == "closed"
    assert not command.containers and not command.networks
    await engine.close_async()


@pytest.mark.parametrize("encoded", ["", "not-base64", "e30=", "bnVsbA==", "eA=="])
async def test_missing_or_malformed_engine_path_stat_blocks_configuration_write_async(encoded: str) -> None:
    requests: list[httpx.Request] = []

    def malformed(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, headers={"x-docker-container-path-stat": encoded})

    engine = DockerEngineClient(transport=httpx.MockTransport(malformed))
    with pytest.raises(DockerEngineError):
        await DockerCodexConfigStager(engine=engine).stage_async(
            container_id=CONTAINER_ID,
            home=PurePosixPath("/tmp/home"),
            mount_path=PurePosixPath("/tmp"),
            uid=1000,
            gid=1000,
            model="model",
            base_url="http://gateway.sandbox/v1",
        )
    assert requests and all(request.method == "HEAD" for request in requests)
    await engine.close_async()
