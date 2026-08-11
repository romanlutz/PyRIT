# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker-backed sandbox conformance tests."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from unit.sandbox.conformance import ProviderConformanceSuite

from pyrit.sandbox import (
    DockerLifecycleError,
    DockerSandboxProvider,
    DockerSandboxProviderConfig,
    DockerServiceBuildSpec,
    SandboxSessionSpec,
)
from pyrit.sandbox.docker_provider import DockerSandboxSession

if TYPE_CHECKING:
    from pathlib import Path


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True,
            capture_output=True,
            timeout=10,
        )
        subprocess.run(
            ["docker", "info"],
            check=True,
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return True


def _docker_provider(tmp_path: Path) -> DockerSandboxProvider:
    (tmp_path / "Dockerfile").write_text(
        'FROM python:3.12-alpine\nRUN apk add --no-cache coreutils\nWORKDIR /workspace\nCMD ["sleep", "infinity"]\n',
        encoding="utf-8",
    )
    return DockerSandboxProvider(
        config=DockerSandboxProviderConfig(
            services=(DockerServiceBuildSpec(service_name="default", build_context=tmp_path),),
            project_context=tmp_path,
            state_dir=tmp_path / "state",
            readiness_timeout_seconds=60,
        )
    )


@pytest.mark.docker
@pytest.mark.run_only_if_all_tests
@pytest.mark.skipif(not _docker_available(), reason="Docker CLI, Compose v2, and a running daemon are required.")
class TestDockerSandboxProviderConformance(ProviderConformanceSuite):
    """Run the provider-neutral suite against Docker Compose."""

    provider_factory = staticmethod(_docker_provider)
    python_command = staticmethod(lambda code: ("python", "-c", code))


@pytest.mark.docker
@pytest.mark.run_only_if_all_tests
@pytest.mark.skipif(not _docker_available(), reason="Docker CLI, Compose v2, and a running daemon are required.")
async def test_failed_owned_cleanup_is_retried_without_leaving_resources(tmp_path: Path) -> None:
    provider = _docker_provider(tmp_path)
    await provider.prepare_async()
    session = await provider.create_session_async(spec=SandboxSessionSpec(session_id="owned-cleanup"))
    assert isinstance(session, DockerSandboxSession)
    project_name = session.project_name
    try:
        await session.initialize_async()
        assert await provider._list_containers_by_project_async(
            project_name=project_name,
            ownership_id=provider._ownership_id,
        )
        remove_owned_resources = provider._force_remove_project_resources_async
        cleanup_attempt = 0

        async def fail_once(*, project_name: str, ownership_id: str) -> None:
            nonlocal cleanup_attempt
            cleanup_attempt += 1
            if cleanup_attempt == 1:
                raise DockerLifecycleError(operation="rm", detail="injected failure")
            await remove_owned_resources(project_name=project_name, ownership_id=ownership_id)

        with patch.object(provider, "_force_remove_project_resources_async", new=AsyncMock(side_effect=fail_once)):
            with pytest.raises(DockerLifecycleError, match="injected failure"):
                await session.close_async()
            await provider.cleanup_async()
        assert cleanup_attempt == 2
        assert not await provider._list_containers_by_project_async(
            project_name=project_name,
            ownership_id=provider._ownership_id,
        )
        assert not await provider._list_project_resource_ids_async(
            resource="network",
            project_name=project_name,
            ownership_id=provider._ownership_id,
            format_field="{{.ID}}",
        )
        assert not await provider._list_project_resource_ids_async(
            resource="volume",
            project_name=project_name,
            ownership_id=provider._ownership_id,
            format_field="{{.Name}}",
        )
    finally:
        await provider.cleanup_async()
