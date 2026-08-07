# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Hyper-V sandbox tests that do not require Hyper-V or external CLIs."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from pyrit.executor.capability import InMemoryCapabilityEvidenceSink
from pyrit.sandbox import (
    HyperVComposeDelegationConfig,
    HyperVComposeDelegationUnsupportedError,
    HyperVEnvironmentConfig,
    HyperVGuestOS,
    HyperVGuestTransport,
    HyperVGuestTransportKind,
    HyperVLifecycleError,
    HyperVNetworkMode,
    HyperVPermissionError,
    HyperVPlatformError,
    HyperVPrerequisiteError,
    HyperVSandboxProvider,
    HyperVSandboxProviderConfig,
    HyperVSecretReference,
    HyperVSecurityPolicy,
    HyperVTransportError,
    HyperVUnsupportedCapabilityError,
    PowerShellCommandRunner,
    PowerShellDirectGuestTransport,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxSessionSpec,
)
from pyrit.sandbox.hyperv_provider import _HyperVScripts, _safe_resource_name
from pyrit.sandbox.hyperv_transport import (
    ExternalCommandResult,
    GuestExecResult,
    OpenSSHGuestTransport,
)
from unit.sandbox.conformance import ProviderConformanceSuite

if TYPE_CHECKING:
    from collections.abc import Mapping


class _FakePowerShellRunner(PowerShellCommandRunner):
    def __init__(self, *, fail_create_at: int | None = None, fail_remove: bool = False) -> None:
        self.calls: list[dict[str, object]] = []
        self.fail_create_at = fail_create_at
        self.fail_remove = fail_remove
        self.create_count = 0

    async def run_json_async(
        self,
        *,
        script: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None = None,
        max_output_bytes: int | None = None,
    ) -> dict[str, object]:
        _ = script, timeout_seconds, cancellation_event, max_output_bytes
        call = dict(payload)
        self.calls.append(call)
        action = payload.get("action")
        if "process_id" in payload:
            process_id = payload["process_id"]
            return {
                "alive": process_id == os.getpid(),
                "start_time": "2026-01-01T00:00:00.0000000Z" if process_id == os.getpid() else None,
            }
        if action == "create":
            self.create_count += 1
            if self.fail_create_at == self.create_count:
                raise HyperVTransportError(message="create failed", error_code="fake_create_failed")
        if action == "remove" and self.fail_remove:
            raise HyperVTransportError(message="remove failed", error_code="fake_remove_failed")
        if action is None:
            return {
                "module_available": True,
                "missing_cmdlets": [],
                "permission_denied": False,
                "missing_switches": [],
                "mismatched_switches": [],
                "missing_images": [],
                "missing_templates": [],
                "ssh_available": True,
            }
        return {"ok": True}


class _MockGuestTransport(HyperVGuestTransport):
    def __init__(self, *, root: Path) -> None:
        self._root = root

    @property
    def name(self) -> str:
        return "mock-guest"

    @property
    def remote_process_cleanup_guaranteed(self) -> bool:
        return True

    async def connect_async(self) -> None:
        await asyncio.to_thread(self._root.mkdir, parents=True, exist_ok=True)

    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None,
        stdout_limit: int,
        stderr_limit: int,
    ) -> GuestExecResult:
        if request.user is not None:
            raise HyperVUnsupportedCapabilityError(capability="alternate_user", transport=self.name)
        command = request.argv
        if command is None:
            command = (
                (os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", request.shell_script or "")
                if os.name == "nt"
                else ("/bin/sh", "-c", request.shell_script or "")
            )
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=self._local_path(request.cwd or "/workspace"),
            env={**os.environ, **request.environment},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        communicate = asyncio.create_task(process.communicate(request.stdin))
        cancellation = asyncio.create_task(cancellation_event.wait()) if cancellation_event else None
        waiting = (communicate,) if cancellation is None else (communicate, cancellation)
        done, _ = await asyncio.wait(waiting, timeout=timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
        timed_out = not done
        cancelled = cancellation is not None and cancellation in done
        if timed_out or cancelled:
            process.kill()
        stdout, stderr = await communicate
        if cancellation is not None:
            cancellation.cancel()
            await asyncio.gather(cancellation, return_exceptions=True)
        return GuestExecResult(
            stdout=stdout[:stdout_limit],
            stderr=stderr[:stderr_limit],
            exit_code=process.returncode,
            stdout_truncated=len(stdout) > stdout_limit,
            stderr_truncated=len(stderr) > stderr_limit,
            timed_out=timed_out,
            cancelled=cancelled,
        )

    async def read_file_async(self, *, path: str, max_bytes: int) -> bytes:
        try:
            data = await asyncio.to_thread(self._local_path(path).read_bytes)
        except FileNotFoundError as error:
            raise HyperVTransportError(message="missing", error_code="file_not_found") from error
        if len(data) > max_bytes:
            raise HyperVTransportError(message="large", error_code="read_limit_exceeded")
        return data

    async def write_file_async(self, *, path: str, data: bytes) -> None:
        local = self._local_path(path)
        await asyncio.to_thread(local.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(local.write_bytes, data)

    def _local_path(self, path: str) -> Path:
        relative = PurePosixPath(path).relative_to("/workspace")
        return self._root.joinpath(*relative.parts)


class _MockTransportFactory:
    def __init__(self, *, root: Path) -> None:
        self._root = root

    def __call__(self, **kwargs: object) -> HyperVGuestTransport:
        vm_name = str(kwargs["vm_name"])
        return _MockGuestTransport(root=self._root / vm_name)


class _StaticSecretResolver:
    def __init__(self, value: Mapping[str, str]) -> None:
        self._value = value

    async def resolve_secret_async(self, reference: HyperVSecretReference) -> Mapping[str, str]:
        _ = reference
        return self._value


@pytest.fixture(autouse=True)
def _mock_windows_host():
    with patch("pyrit.sandbox.hyperv_provider._is_windows_host", return_value=True):
        yield


def _environment(name: str, *, default: bool = False) -> HyperVEnvironmentConfig:
    return HyperVEnvironmentConfig(
        name=name,
        default=default,
        base_vhdx=Path(f"{name}.vhdx"),
        guest_os=HyperVGuestOS.LINUX,
        transport=HyperVGuestTransportKind.SSH,
        ssh_host="192.0.2.1",
        credential=HyperVSecretReference(secret_id="HYPERV_TEST_SECRET"),
        workspace_root="/workspace",
    )


def _provider(
    tmp_path: Path,
    *,
    runner: _FakePowerShellRunner | None = None,
    environments: tuple[HyperVEnvironmentConfig, ...] | None = None,
    evidence_sink: InMemoryCapabilityEvidenceSink | None = None,
) -> HyperVSandboxProvider:
    return HyperVSandboxProvider(
        config=HyperVSandboxProviderConfig(
            environments=environments
            or (
                _environment("default", default=True),
                _environment("alpha"),
                _environment("zeta"),
            ),
            state_dir=tmp_path / "state",
        ),
        evidence_sink=evidence_sink,
        power_shell_runner=runner or _FakePowerShellRunner(),
        guest_transport_factory=_MockTransportFactory(root=tmp_path / "guests"),
    )


class TestHyperVSandboxProviderConformance(ProviderConformanceSuite):
    """Run the provider-neutral suite through a mocked Hyper-V guest transport."""

    provider_factory = staticmethod(_provider)
    python_command = staticmethod(lambda code: (sys.executable, "-c", code))


def test_hyperv_config_defaults_are_secure_and_deterministic() -> None:
    config = HyperVSandboxProviderConfig(
        environments=(_environment("zeta"), _environment("alpha")),
    )
    assert config.resolve_default_environment() == "alpha"
    assert not config.security_policy.allow_external_switch
    assert not config.security_policy.allow_internet_egress
    assert not config.security_policy.allow_host_filesystem_sharing
    assert not config.security_policy.allow_nested_virtualization
    assert not config.security_policy.allow_device_passthrough
    assert not config.security_policy.allow_mac_spoofing
    with pytest.raises(ValidationError, match="frozen"):
        config.default_environment = "zeta"


def test_hyperv_config_rejects_unsafe_network_and_resource_bounds() -> None:
    external = _environment("external").model_copy(
        update={"network_mode": HyperVNetworkMode.EXTERNAL, "switch_name": "Internet"}
    )
    with pytest.raises(ValidationError, match="explicit switch"):
        HyperVSandboxProviderConfig(environments=(external,), allowed_switches=("Internet",))
    with pytest.raises(ValidationError, match="processor policy"):
        HyperVSandboxProviderConfig(
            environments=(_environment("large").model_copy(update={"processor_count": 4}),),
            security_policy=HyperVSecurityPolicy(max_processor_count=2),
        )


def test_hyperv_environment_requires_one_template_and_transport_prerequisites() -> None:
    with pytest.raises(ValidationError, match="Exactly one"):
        HyperVEnvironmentConfig(
            name="invalid",
            guest_os=HyperVGuestOS.LINUX,
            transport=HyperVGuestTransportKind.SSH,
            ssh_host="host",
            credential=HyperVSecretReference(secret_id="secret"),
        )
    with pytest.raises(ValidationError, match="Windows guest"):
        HyperVEnvironmentConfig(
            name="invalid",
            base_vhdx=Path("base.vhdx"),
            guest_os=HyperVGuestOS.LINUX,
            credential=HyperVSecretReference(secret_id="secret"),
        )


async def test_powershell_runner_encodes_fixed_script_and_passes_data_on_stdin() -> None:
    runner = PowerShellCommandRunner(executable="powershell.exe", max_output_bytes=4096)
    secret = "credential-value-must-not-enter-script"
    result = ExternalCommandResult(stdout=b'{"ok":true}', stderr=b"", returncode=0)
    with patch(
        "pyrit.sandbox.hyperv_transport.run_external_command_async",
        new=AsyncMock(return_value=result),
    ) as run:
        assert await runner.run_json_async(
            script="param(); @{ ok = $true } | ConvertTo-Json",
            payload={"password": secret},
            timeout_seconds=1,
        ) == {"ok": True}
    argv = run.await_args.kwargs["argv"]
    decoded_script = base64.b64decode(argv[-1]).decode("utf-16-le")
    assert secret not in decoded_script
    assert json.loads(run.await_args.kwargs["stdin"]) == {"password": secret}


async def test_provider_reports_platform_and_compose_limitations(tmp_path: Path) -> None:
    provider = _provider(tmp_path)
    with (
        patch("pyrit.sandbox.hyperv_provider._is_windows_host", return_value=False),
        pytest.raises(HyperVPlatformError),
    ):
        await provider.prepare_async()
    compose = HyperVComposeDelegationConfig(
        enabled=True,
        docker_host="ssh://guest",
        provider_config={
            "services": ({"service_name": "default", "build_context": tmp_path},),
        },
    )
    provider = HyperVSandboxProvider(
        config=HyperVSandboxProviderConfig(
            environments=(_environment("default"),),
            compose_delegation=compose,
        ),
        power_shell_runner=_FakePowerShellRunner(),
    )
    with pytest.raises(HyperVComposeDelegationUnsupportedError):
        await provider.prepare_async()


@pytest.mark.parametrize(
    ("preflight_update", "error_code"),
    [
        ({"module_available": False}, "module"),
        ({"permission_denied": True}, "permission"),
        ({"missing_cmdlets": ["Get-VM"]}, "cmdlets"),
        ({"missing_images": ["base.vhdx"]}, "images"),
        ({"missing_templates": ["template"]}, "templates"),
        ({"missing_switches": ["switch"]}, "switches"),
        ({"mismatched_switches": ["switch"]}, "switch_types"),
        ({"ssh_available": False}, "ssh"),
    ],
)
def test_preflight_errors_are_typed_and_actionable(
    tmp_path: Path,
    preflight_update: dict[str, object],
    error_code: str,
) -> None:
    provider = _provider(tmp_path)
    preflight = {
        "module_available": True,
        "permission_denied": False,
        "missing_cmdlets": [],
        "missing_images": [],
        "missing_templates": [],
        "missing_switches": [],
        "mismatched_switches": [],
        "ssh_available": True,
        **preflight_update,
    }
    expected = HyperVPrerequisiteError if error_code != "permission" else HyperVPermissionError
    with pytest.raises(expected) as exc_info:
        provider._validate_preflight(preflight)
    assert error_code in exc_info.value.error_code


async def test_partial_creation_rolls_back_every_owned_resource(tmp_path: Path) -> None:
    runner = _FakePowerShellRunner(fail_create_at=2)
    provider = _provider(
        tmp_path,
        runner=runner,
        environments=(_environment("alpha"), _environment("zeta")),
    )
    await provider.prepare_async()
    spec = SandboxSessionSpec(
        session_id="rollback",
        environments=(SandboxEnvironmentSpec(name="alpha"), SandboxEnvironmentSpec(name="zeta")),
    )
    with pytest.raises(HyperVLifecycleError, match="create"):
        async with provider.session_async(spec=spec):
            pytest.fail("Partially initialized Hyper-V session should not be entered.")
    remove_calls = [call for call in runner.calls if call.get("action") == "remove"]
    assert len(remove_calls) == 2
    assert json.loads(provider._state_path.read_text(encoding="utf-8")) == {}
    await provider.cleanup_async()


def test_lifecycle_stamps_vm_ownership_before_mutable_configuration() -> None:
    script = _HyperVScripts.LIFECYCLE
    new_vm = script.index("$vm = New-VM @parameters")
    ownership = script.index("Set-VM -VM $vm -Notes $data.ownership_json")
    processor = script.index("Set-VMProcessor -VM $vm")
    assert new_vm < ownership < processor


async def test_cleanup_is_exactly_once_and_names_are_attempt_unique(tmp_path: Path) -> None:
    runner = _FakePowerShellRunner()
    provider = _provider(tmp_path, runner=runner)
    await provider.prepare_async()
    first = SandboxSessionSpec(session_id="same", attempt_id=uuid.uuid4())
    second = SandboxSessionSpec(session_id="same", attempt_id=uuid.uuid4())
    first_name = _safe_resource_name(
        prefix="pyrit",
        session_id=first.session_id,
        attempt_id=str(first.attempt_id),
        environment_name="default",
    )
    second_name = _safe_resource_name(
        prefix="pyrit",
        session_id=second.session_id,
        attempt_id=str(second.attempt_id),
        environment_name="default",
    )
    assert first_name != second_name
    session = await provider.create_session_async(spec=first)
    await session.initialize_async()
    await asyncio.gather(session.close_async(), session.close_async())
    assert len([call for call in runner.calls if call.get("action") == "remove"]) == 1
    await provider.cleanup_async()


async def test_orphan_cleanup_preserves_state_after_owned_cleanup_failure(tmp_path: Path) -> None:
    runner = _FakePowerShellRunner(fail_remove=True)
    provider = _provider(tmp_path, runner=runner)
    await provider.prepare_async()
    session = await provider.create_session_async(spec=SandboxSessionSpec(session_id="orphan"))
    await provider._register_session_state_async(session=session)
    await provider._remove_session_async("orphan")
    state = json.loads(provider._state_path.read_text(encoding="utf-8"))
    state["orphan"]["owner_process_id"] = 2_147_483_647
    provider._state_path.write_text(json.dumps(state), encoding="utf-8")
    assert await provider.cleanup_orphans_async() == 0
    state = json.loads(provider._state_path.read_text(encoding="utf-8"))
    assert "orphan" in state
    runner.fail_remove = False
    assert await provider.cleanup_orphans_async() == 1
    assert json.loads(provider._state_path.read_text(encoding="utf-8")) == {}
    await provider.cleanup_async()


async def test_orphan_cleanup_never_removes_a_live_owner_session(tmp_path: Path) -> None:
    runner = _FakePowerShellRunner()
    provider = _provider(tmp_path, runner=runner)
    await provider.prepare_async()
    session = await provider.create_session_async(spec=SandboxSessionSpec(session_id="live-owner"))
    await provider._register_session_state_async(session=session)
    await provider._remove_session_async("live-owner")
    assert await provider.cleanup_orphans_async() == 0
    assert not [call for call in runner.calls if call.get("action") == "remove"]
    await provider._unregister_session_state_async(session_id="live-owner")
    await provider.cleanup_async()


async def test_session_limits_must_fit_provider_command_output_bound(tmp_path: Path) -> None:
    provider = HyperVSandboxProvider(
        config=HyperVSandboxProviderConfig(
            environments=(_environment("default"),),
            state_dir=tmp_path / "state",
            max_command_output_bytes=1024,
        ),
        power_shell_runner=_FakePowerShellRunner(),
        guest_transport_factory=_MockTransportFactory(root=tmp_path / "guests"),
    )
    await provider.prepare_async()
    with pytest.raises(HyperVPrerequisiteError, match="larger"):
        await provider.create_session_async(spec=SandboxSessionSpec())
    await provider.cleanup_async()


async def test_evidence_records_security_posture_without_credentials(tmp_path: Path) -> None:
    sink = InMemoryCapabilityEvidenceSink()
    provider = _provider(tmp_path, evidence_sink=sink)
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            info = session.get_environment().connection_info
            assert info.metadata["security_boundary"] is True
            assert info.metadata["host_filesystem_sharing"] is False
            write = await session.get_environment().write_file_async(path="artifact.bin", data=b"\x00\xff")
            assert write.sha256
    serialized = json.dumps([event.model_dump(mode="json") for event in await sink.snapshot_async()])
    assert "credential-value" not in serialized
    assert "password" not in serialized


async def test_powershell_direct_transport_is_binary_safe_and_rejects_alternate_user() -> None:
    environment = HyperVEnvironmentConfig(
        name="windows",
        base_vhdx=Path("base.vhdx"),
        credential=HyperVSecretReference(secret_id="secret"),
    )
    runner = PowerShellCommandRunner(executable="powershell.exe", max_output_bytes=4096)
    secret_resolver = _StaticSecretResolver({"username": "guest", "password": "credential-value"})
    transport = PowerShellDirectGuestTransport(
        vm_name="vm",
        config=environment,
        runner=runner,
        secret_resolver=secret_resolver,
        command_timeout_seconds=10,
        max_output_bytes=4096,
    )
    binary = b"\x00\xff"
    with patch.object(
        runner,
        "run_json_async",
        new=AsyncMock(
            side_effect=[
                {"data_base64": base64.b64encode(binary).decode("ascii")},
                {"size_bytes": len(binary)},
                {"stdout_base64": "", "stderr_base64": "", "exit_code": 0, "timed_out": False},
            ]
        ),
    ):
        assert await transport.read_file_async(path=r"C:\workspace\file", max_bytes=2) == binary
        await transport.write_file_async(path=r"C:\workspace\file", data=binary)
        result = await transport.exec_async(
            request=SandboxExecRequest(argv=("cmd.exe", "/c", "exit", "0")),
            timeout_seconds=1,
            cancellation_event=None,
            stdout_limit=10,
            stderr_limit=10,
        )
        assert result.exit_code == 0
    with pytest.raises(HyperVUnsupportedCapabilityError, match="alternate_user"):
        await transport.exec_async(
            request=SandboxExecRequest(argv=("whoami",), user="other"),
            timeout_seconds=1,
            cancellation_event=None,
            stdout_limit=10,
            stderr_limit=10,
        )


async def test_openssh_transport_uses_argv_and_binary_stdin() -> None:
    config = _environment("ssh")
    transport = OpenSSHGuestTransport(
        config=config,
        secret_resolver=_StaticSecretResolver({"username": "guest", "identity_file": "key-path"}),
        executable="ssh",
        command_timeout_seconds=10,
        max_output_bytes=4096,
    )
    outcome = ExternalCommandResult(stdout=b"ok", stderr=b"", returncode=0)
    with patch(
        "pyrit.sandbox.hyperv_transport.run_external_command_async",
        new=AsyncMock(return_value=outcome),
    ) as run:
        result = await transport.exec_async(
            request=SandboxExecRequest(
                argv=("printf", "%s", "hello world"),
                stdin=b"\x00\xff",
                environment={"VALUE": "a b"},
                cwd="/workspace",
            ),
            timeout_seconds=1,
            cancellation_event=None,
            stdout_limit=10,
            stderr_limit=10,
        )
    assert result.stdout == b"ok"
    assert run.await_args.kwargs["stdin"] == b"\x00\xff"
    argv = run.await_args.kwargs["argv"]
    assert argv[0] == "ssh"
    assert "credential-value" not in " ".join(argv)


def test_registry_discovers_hyperv_provider(tmp_path: Path) -> None:
    from pyrit.sandbox import SandboxProviderRegistry

    registry = SandboxProviderRegistry()
    assert "HyperVSandboxProvider" in registry.get_class_names()
    provider = _provider(tmp_path)
    registry.instances.register(provider, name="hyperv-test")
    assert registry.instances.get("hyperv-test") is provider
