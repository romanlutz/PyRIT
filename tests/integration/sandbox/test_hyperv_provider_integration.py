# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Opt-in live Hyper-V sandbox conformance tests."""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest
from unit.sandbox.conformance import ProviderConformanceSuite

from pyrit.sandbox import (
    HyperVEnvironmentConfig,
    HyperVSandboxProvider,
    HyperVSandboxProviderConfig,
    HyperVSecretReference,
)

if TYPE_CHECKING:
    from pathlib import Path


def _live_hyperv_available() -> bool:
    if os.environ.get("PYRIT_RUN_HYPERV_TESTS") != "1" or os.name != "nt":
        return False
    executable = shutil.which("powershell.exe")
    if executable is None:
        return False
    command = (
        "$module = Get-Module -ListAvailable Hyper-V; "
        "$cmd = Get-Command Get-VM -ErrorAction SilentlyContinue; "
        "if ($null -eq $module -or $null -eq $cmd) { exit 1 }; "
        "try { Get-VM -ErrorAction Stop | Out-Null } catch { exit 1 }"
    )
    try:
        subprocess.run(
            [executable, "-NoProfile", "-NonInteractive", "-Command", command],
            check=True,
            capture_output=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(os.environ.get("PYRIT_HYPERV_BASE_VHDX") and os.environ.get("PYRIT_HYPERV_GUEST_SECRET"))


def _hyperv_provider(tmp_path: Path) -> HyperVSandboxProvider:
    base_vhdx = os.environ["PYRIT_HYPERV_BASE_VHDX"]
    return HyperVSandboxProvider(
        config=HyperVSandboxProviderConfig(
            environments=(
                HyperVEnvironmentConfig(
                    name="default",
                    default=True,
                    base_vhdx=base_vhdx,
                    credential=HyperVSecretReference(secret_id="PYRIT_HYPERV_GUEST_SECRET"),
                ),
            ),
            state_dir=tmp_path / "state",
        )
    )


@pytest.mark.hyperv
@pytest.mark.run_only_if_all_tests
@pytest.mark.skipif(
    not _live_hyperv_available(),
    reason=(
        "Set PYRIT_RUN_HYPERV_TESTS=1, PYRIT_HYPERV_BASE_VHDX, and PYRIT_HYPERV_GUEST_SECRET on a "
        "Windows host with Hyper-V management permission to run live tests."
    ),
)
class TestHyperVSandboxProviderLiveConformance(ProviderConformanceSuite):
    """Run provider-neutral conformance against a real Windows Hyper-V guest."""

    provider_factory = staticmethod(_hyperv_provider)
    python_command = staticmethod(lambda code: ("python.exe", "-c", code))
