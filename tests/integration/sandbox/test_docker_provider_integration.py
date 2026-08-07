# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker-backed sandbox conformance tests."""

from __future__ import annotations

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from unit.sandbox.conformance import ProviderConformanceSuite

from pyrit.sandbox import DockerSandboxProvider, DockerSandboxProviderConfig, DockerServiceBuildSpec

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
