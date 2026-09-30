# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import tarfile
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from pydantic import ValidationError

from pyrit.executor.workflow.docker_compose import (
    ComposeEnvironmentSpec,
    ComposeServiceSpec,
    ComposeTmpfsSpec,
    DockerComposeEnvironmentLease,
)
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.executor.workflow.inspect_file_probe import InspectFileProbeComposeLease
from pyrit.models.environment_lease import EnvironmentCapability
from tests.unit.executor.workflow.test_docker_compose import FakeDockerRunner
from tests.unit.executor.workflow.test_docker_engine import FakeConfigArchive, FakeEngine

if TYPE_CHECKING:
    from pathlib import Path

PUBLIC_SETUP = b"#!/usr/bin/env bash\n\ntouch foo.txt\n"


@pytest.fixture
def setup_script_path(tmp_path: Path) -> Path:
    path = tmp_path / "sandbox_setup.sh"
    path.write_bytes(PUBLIC_SETUP)
    return path


class FakeFileArchive(FakeConfigArchive):
    """In-memory guest file state, not a Docker daemon or Inspect sandbox."""

    def __init__(self) -> None:
        super().__init__()
        self.content: bytes | None = None
        self.bad_readback: str | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.params["path"]
        if path == "/tmp":
            response = super().handle(request)
            if request.method == "PUT" and response.status_code == 200:
                with tarfile.open(fileobj=io.BytesIO(request.content), mode="r:") as archive:
                    entries = list(archive)
                    assert [entry.name for entry in entries] == ["foo.txt"]
                    stream = archive.extractfile(entries[0])
                    assert stream is not None
                    with stream:
                        self.content = stream.read()
            return response
        assert path == "/tmp/foo.txt"
        if self.content is None:
            return httpx.Response(404)
        metadata = {
            "name": "foo.txt",
            "mode": 0o600,
            "size": len(self.content),
            "linkTarget": "",
        }
        if self.stat_change:
            self.stat_change(path, metadata)
        headers = {"x-docker-container-path-stat": base64.b64encode(json.dumps(metadata).encode()).decode()}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        assert request.method == "GET"
        self.reads += 1
        data = self._file_archive()
        return httpx.Response(
            200,
            headers={**headers, "content-type": "application/x-tar"},
            content=self.readback(data) if self.readback else data,
        )

    def _file_archive(self) -> bytes:
        content = self.content
        assert content is not None
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            entry = tarfile.TarInfo("foo.txt")
            entry.type, entry.mode, entry.uid, entry.gid = tarfile.REGTYPE, 0o600, 1000, 1000
            if self.bad_readback == "wrong-path":
                entry.name = "../other"
            elif self.bad_readback == "wrong-owner":
                entry.uid = 0
            elif self.bad_readback == "symlink":
                entry.type, entry.linkname = tarfile.SYMTYPE, "/etc/passwd"
            elif self.bad_readback == "wrong-bytes":
                content = b"not empty"
            entry.size = len(content) if entry.isreg() else 0
            archive.addfile(entry, io.BytesIO(content) if entry.isreg() else None)
            if self.bad_readback == "extra":
                archive.addfile(tarfile.TarInfo("unapproved"))
        data = buffer.getvalue()
        return data + b"unapproved" if self.bad_readback == "trailing" else data


class FakeFileProbeEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.asset_archive = FakeFileArchive()
        self.config_archive = self.asset_archive


def fixture_spec() -> ComposeEnvironmentSpec:
    return ComposeEnvironmentSpec(
        services=(
            ComposeServiceSpec(
                name="default",
                roles=frozenset({"target", "grader"}),
                image=InspectFileProbeComposeLease.IMAGE,
                command=("/usr/bin/sleep", "infinity"),
                healthcheck=("/usr/bin/true",),
                uid=1000,
                gid=1000,
                cpu_millis=500,
                memory_bytes=134_217_728,
                pids_limit=64,
                tmpfs_bytes=16_777_216,
            ),
        )
    )


def make_lease(
    *, path: Path, spec: ComposeEnvironmentSpec | None = None
) -> tuple[InspectFileProbeComposeLease, FakeDockerRunner, FakeFileProbeEngine, DockerEngineClient]:
    runner = FakeDockerRunner()

    def pinned_image(image: dict[str, Any]) -> None:
        image["RepoDigests"] = ["python@" + InspectFileProbeComposeLease.IMAGE.split("@")[1]]

    runner.mutate_image = pinned_image
    fake = FakeFileProbeEngine()
    fake.containers = runner.containers
    fake.networks = runner.networks
    engine = fake.client()
    lease = InspectFileProbeComposeLease(
        run_id="file-probe-1",
        spec=spec or fixture_spec(),
        runner=runner,
        engine=engine,
        setup_script_path=path,
    )
    return lease, runner, fake, engine


async def test_pinned_setup_stages_only_one_empty_file_before_original_rule_observation_and_owned_cleanup_async(
    setup_script_path: Path,
) -> None:
    lease, runner, fake, engine = make_lease(path=setup_script_path)
    assert hashlib.sha256(PUBLIC_SETUP).hexdigest() == lease.SOURCE_SHA256
    try:
        allocation = await lease.acquire_async()
        assert allocation.container_id("default") == "0" * 63 + "1"
        snapshot = lease.snapshot()
        assert snapshot.setup_completed and snapshot.state == "ready"
        assert snapshot.capabilities == frozenset({EnvironmentCapability.SETUP, EnvironmentCapability.HEALTH_CHECK})
        assert snapshot.services[0].roles == frozenset({"target", "grader"})
        assert len(fake.asset_archive.writes) == 1
        assert fake.asset_archive.content == b""
        assert await lease.observe_file_async() is True  # Upstream check_file returns FOUND.
        fake.asset_archive.content = b"changed by the inert agent"
        assert await lease.observe_file_async() is True  # The original rule tests existence, not content.
        fake.asset_archive.content = None  # The same sandbox's file was removed by the inert agent.
        assert await lease.observe_file_async() is False  # Upstream check_file returns MISSING.
        document = runner.documents[0]
        service = document["services"]["default"]
        assert service["image"] == lease.IMAGE and service["read_only"] is True
        assert service["tmpfs"] == ["/tmp:rw,noexec,nosuid,nodev,size=16777216,uid=1000,gid=1000,mode=0700"]
        assert not set(service) & {"build", "volumes", "secrets", "ports", "environment"}
        assert all("/exec" not in request.url.path for request in fake.requests)
        assert all(
            request.url.params.get("path") in ("/tmp", "/tmp/foo.txt")
            for request in fake.requests
            if "/archive" in request.url.path
        )
        await lease.close_async()
        await lease.close_async()
        assert runner.documents == [document, document]
        assert not runner.containers and not runner.networks
        assert sum("down" in arguments for arguments in runner.calls) == 1
        with pytest.raises(RuntimeError, match="setup-complete"):
            await lease.observe_file_async()
    finally:
        await engine.close_async()


@pytest.mark.parametrize(
    ("source_name", "source_bytes"),
    [
        ("sandbox_setup.sh", b"#!/usr/bin/env bash\n\ntouch foo.txt; echo surprise\n"),
        ("sandbox_setup.sh", PUBLIC_SETUP + b"\n"),
        ("setup.py", PUBLIC_SETUP),
    ],
)
async def test_unpinned_source_or_wrong_source_path_is_rejected_before_any_provider_io_async(
    *, tmp_path: Path, source_name: str, source_bytes: bytes
) -> None:
    path = tmp_path / source_name
    await asyncio.to_thread(path.write_bytes, source_bytes)
    runner = FakeDockerRunner()
    engine = FakeEngine().client()
    try:
        with pytest.raises(ValueError, match="File-probe setup"):
            InspectFileProbeComposeLease(
                run_id="invalid-source", spec=fixture_spec(), runner=runner, engine=engine, setup_script_path=path
            )
        assert not runner.calls
    finally:
        await engine.close_async()


async def test_source_path_with_a_traversal_alias_is_rejected_before_provider_io_async(
    *, tmp_path: Path, setup_script_path: Path
) -> None:
    await asyncio.to_thread((tmp_path / "alias").mkdir)
    path = tmp_path / "alias" / ".." / setup_script_path.name
    runner = FakeDockerRunner()
    engine = FakeEngine().client()
    try:
        with pytest.raises(ValueError, match="unaliased"):
            InspectFileProbeComposeLease(
                run_id="aliased-source", spec=fixture_spec(), runner=runner, engine=engine, setup_script_path=path
            )
        assert not runner.calls
    finally:
        await engine.close_async()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("roles", frozenset({"agent"})),
        ("image", "python:3.12-slim@sha256:" + "a" * 64),
        ("command", ("/bin/sh", "-c", "touch foo.txt")),
        ("runtime_state", (ComposeTmpfsSpec(path="/tmp", size_bytes=16_777_216, executable=True),)),
    ],
)
async def test_fixture_rejects_other_roles_images_commands_or_runtime_mounts_async(
    *, setup_script_path: Path, field: str, value: Any
) -> None:
    original = fixture_spec().services[0]
    spec = ComposeEnvironmentSpec(services=(original.model_copy(update={field: value}),))
    runner = FakeDockerRunner()
    engine = FakeEngine().client()
    try:
        with pytest.raises(ValueError, match="File-probe roles"):
            InspectFileProbeComposeLease(
                run_id="invalid-spec", spec=spec, runner=runner, engine=engine, setup_script_path=setup_script_path
            )
        assert not runner.calls
    finally:
        await engine.close_async()


@pytest.mark.parametrize("field", ["build", "volumes", "setup", "files", "network_mode"])
def test_inspect_compose_extensions_cannot_be_treated_as_approved_setup(field: str) -> None:
    service = fixture_spec().services[0]
    with pytest.raises(ValidationError, match=field):
        ComposeServiceSpec.model_validate({**service.model_dump(), field: "unapproved"})


def test_base_compose_provider_cannot_advertise_unimplemented_setup() -> None:
    runner = FakeDockerRunner()
    capabilities = frozenset({EnvironmentCapability.SETUP, EnvironmentCapability.HEALTH_CHECK})
    with pytest.raises(NotImplementedError, match="setup requires"):
        DockerComposeEnvironmentLease(
            run_id="unimplemented-setup", spec=fixture_spec(), runner=runner, capabilities=capabilities
        )
    assert not runner.calls


@pytest.mark.parametrize("defect", ["wrong-path", "wrong-owner", "symlink", "wrong-bytes", "extra", "trailing"])
async def test_unverifiable_setup_readback_rolls_back_owned_sandbox_async(
    *, setup_script_path: Path, defect: str
) -> None:
    lease, runner, fake, engine = make_lease(path=setup_script_path)
    fake.asset_archive.bad_readback = defect
    try:
        with pytest.raises(DockerEngineError, match="File-probe|Staged"):
            await lease.acquire_async()
        assert len(fake.asset_archive.writes) == 1
        assert lease.snapshot().state == "closed" and not lease.snapshot().setup_completed
        assert not runner.containers and not runner.networks
        assert sum("down" in arguments for arguments in runner.calls) == 1
        with pytest.raises(RuntimeError, match="setup-complete"):
            await lease.observe_file_async()
    finally:
        await engine.close_async()


async def test_setup_and_cleanup_failures_never_look_like_ready_or_released_async(setup_script_path: Path) -> None:
    lease, runner, fake, engine = make_lease(path=setup_script_path)
    fake.asset_archive.bad_readback = "wrong-bytes"
    runner.down_returncode = 1
    try:
        with pytest.raises(DockerEngineError, match="File-probe"):
            await lease.acquire_async()
        assert lease.snapshot().state == "cleanup_failed" and not lease.snapshot().setup_completed
        assert runner.containers and runner.networks
        assert sum("down" in arguments for arguments in runner.calls) == 1
    finally:
        await engine.close_async()


async def test_existing_file_cannot_be_adopted_or_modified_async(setup_script_path: Path) -> None:
    lease, runner, fake, engine = make_lease(path=setup_script_path)
    fake.asset_archive.content = b"old"
    try:
        with pytest.raises(DockerEngineError, match="existing file"):
            await lease.acquire_async()
        assert fake.asset_archive.content == b"old" and not fake.asset_archive.writes
        assert not runner.containers and not runner.networks
    finally:
        await engine.close_async()


async def test_engine_compose_mismatch_prevents_staging_and_rolls_back_async(setup_script_path: Path) -> None:
    lease, runner, fake, engine = make_lease(path=setup_script_path)

    def mutate(container: dict[str, Any]) -> None:
        container["Config"]["User"] = "0:0"

    fake.container_change = mutate
    try:
        with pytest.raises(DockerEngineError, match="same approved container"):
            await lease.acquire_async()
        assert not fake.asset_archive.writes
        assert not runner.containers and not runner.networks
    finally:
        await engine.close_async()
