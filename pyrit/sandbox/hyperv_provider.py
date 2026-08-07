# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Production-oriented Hyper-V sandbox provider.

The implementation uses only standard-library Python plus PyRIT's existing model
stack. Hyper-V and guest access are performed through external PowerShell and
OpenSSH facilities. Importing or constructing the provider performs no host probe,
requires no administrator rights, and does not import Inspect, WinRM, SSH, Docker,
or Hyper-V Python packages.

Compose-inside-VM is represented by ``HyperVComposeDelegationConfig`` but is
deliberately rejected when enabled. Layer 5's Docker provider has no injectable
remote Docker endpoint/context, so safe delegation cannot yet preserve its lifecycle
and evidence guarantees. This layer exposes a typed, tested limitation instead of a
success-shaped stub.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import ntpath
import os
import posixpath
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyrit.executor.capability.models import SandboxOperationEvidence
from pyrit.sandbox.contracts import SandboxEnvironment, SandboxProcess, SandboxProvider, SandboxSession
from pyrit.sandbox.hyperv_transport import (
    EnvironmentHyperVSecretResolver,
    GuestExecResult,
    HyperVComposeDelegationUnsupportedError,
    HyperVGuestTransport,
    HyperVLifecycleError,
    HyperVPermissionError,
    HyperVPlatformError,
    HyperVPrerequisiteError,
    HyperVSandboxError,
    HyperVSecretResolver,
    HyperVTransportError,
    HyperVUnsupportedCapabilityError,
    PowerShellCommandRunner,
    create_default_guest_transport,
)
from pyrit.sandbox.local import SandboxPathEscapeError, SandboxSetupError
from pyrit.sandbox.models import (
    HyperVEnvironmentConfig,
    HyperVGuestOS,
    HyperVGuestTransportKind,
    HyperVNetworkMode,
    HyperVSandboxProviderConfig,
    SandboxArtifact,
    SandboxConnectionInfo,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxExecResult,
    SandboxOperationContext,
    SandboxOperationStatus,
    SandboxReadResult,
    SandboxSessionSpec,
    SandboxTaskSpec,
    SandboxWriteResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pyrit.executor.capability.evidence import CapabilityEvidenceSink
    from pyrit.models import JSONValue

    HyperVGuestTransportFactory = Callable[..., HyperVGuestTransport]


class _HyperVScripts:
    """Fixed PowerShell templates used by the provider."""

    PREFLIGHT = r"""
$ErrorActionPreference = 'Stop'
$data = [Console]::In.ReadToEnd() | ConvertFrom-Json
try {
    $required = @(
        'Get-VM', 'New-VM', 'Set-VM', 'Start-VM', 'Stop-VM', 'Remove-VM',
        'Get-VMSwitch', 'New-VMSwitch', 'Remove-VMSwitch',
        'Get-VMHardDiskDrive', 'New-VHD', 'Get-VHD', 'Set-VMProcessor',
        'Get-VMNetworkAdapter', 'Set-VMNetworkAdapter', 'Set-VMFirmware'
    )
    $missingCmdlets = @($required | Where-Object { -not (Get-Command $_ -ErrorAction SilentlyContinue) })
    $moduleAvailable = $null -ne (Get-Module -ListAvailable -Name Hyper-V)
    $permissionDenied = $false
    if ($moduleAvailable -and $missingCmdlets.Count -eq 0) {
        try { Get-VM -ErrorAction Stop | Out-Null } catch { $permissionDenied = $true }
    }
    $switches = @()
    if ($moduleAvailable -and -not $permissionDenied -and $missingCmdlets.Count -eq 0) {
        $switches = @(Get-VMSwitch -ErrorAction Stop | ForEach-Object {
            @{ name = $_.Name; switch_type = [string]$_.SwitchType }
        })
    }
    $missingSwitches = @($data.switches | Where-Object {
        $requestedName = $_.name
        -not ($switches | Where-Object { $_.name -eq $requestedName })
    } | ForEach-Object { $_.name })
    $mismatchedSwitches = @($data.switches | Where-Object {
        $requestedName = $_.name
        $requestedType = $_.switch_type
        $actual = $switches | Where-Object { $_.name -eq $requestedName } | Select-Object -First 1
        $null -ne $actual -and $actual.switch_type -ne $requestedType
    } | ForEach-Object { $_.name })
    $missingImages = @($data.images | Where-Object { -not (Test-Path -LiteralPath $_ -PathType Leaf) })
    $allVMs = @()
    if (-not $permissionDenied -and $missingCmdlets.Count -eq 0) {
        $allVMs = @(Get-VM -ErrorAction Stop | ForEach-Object { $_.Name })
    }
    $missingTemplates = @($data.template_vms | Where-Object { $_ -notin $allVMs })
    $sshAvailable = $true
    if ($data.require_ssh) {
        $sshAvailable = $null -ne (Get-Command $data.ssh_executable -ErrorAction SilentlyContinue)
    }
    @{
        module_available = $moduleAvailable
        missing_cmdlets = $missingCmdlets
        permission_denied = $permissionDenied
        missing_switches = $missingSwitches
        mismatched_switches = $mismatchedSwitches
        missing_images = $missingImages
        missing_templates = $missingTemplates
        ssh_available = $sshAvailable
    } | ConvertTo-Json -Compress -Depth 6
} catch {
    [Console]::Error.WriteLine((@{ error_code = 'hyperv_preflight_failed' } | ConvertTo-Json -Compress))
    exit 1
}
"""

    LIFECYCLE = r"""
$ErrorActionPreference = 'Stop'
$data = [Console]::In.ReadToEnd() | ConvertFrom-Json
function Write-StructuredError([string]$code) {
    [Console]::Error.WriteLine((@{ error_code = $code } | ConvertTo-Json -Compress))
}
function Read-Owner([string]$path) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}
try {
    switch ($data.action) {
        'create' {
            $parentPath = $data.base_vhdx
            if ($data.template_vm) {
                $template = Get-VM -Name $data.template_vm -ErrorAction Stop
                if ($template.State -ne 'Off') { throw 'Template VM must be off.' }
                if ($data.template_checkpoint) {
                    $snapshot = Get-VMSnapshot -VM $template -Name $data.template_checkpoint -ErrorAction Stop
                    $parentPath = (
                        Get-VMHardDiskDrive -VMSnapshot $snapshot -ErrorAction Stop |
                            Select-Object -First 1
                    ).Path
                } else {
                    $parentPath = (Get-VMHardDiskDrive -VM $template -ErrorAction Stop | Select-Object -First 1).Path
                }
            }
            $vhd = Get-VHD -Path $parentPath -ErrorAction Stop
            if ($vhd.Size -gt [int64]$data.max_disk_bytes) { throw 'Template disk exceeds configured bound.' }
            [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName([string]$data.disk_path)) | Out-Null
            if (Test-Path -LiteralPath $data.disk_path) { throw 'Owned disk path already exists.' }
            if ($data.disk_strategy -eq 'differencing') {
                New-VHD -Path $data.disk_path -ParentPath $parentPath -Differencing -ErrorAction Stop | Out-Null
            } else {
                Copy-Item -LiteralPath $parentPath -Destination $data.disk_path -ErrorAction Stop
            }
            @{ ownership_id = $data.ownership_id; resource = 'disk'; name = $data.disk_path } |
                ConvertTo-Json -Compress | Set-Content -LiteralPath $data.disk_marker -Encoding UTF8

            $switchName = $data.switch_name
            if ($data.create_switch) {
                if (Get-VMSwitch -Name $switchName -ErrorAction SilentlyContinue) {
                    throw 'Owned switch name collision.'
                }
                New-VMSwitch -Name $switchName -SwitchType $data.switch_type -ErrorAction Stop | Out-Null
                @{ ownership_id = $data.ownership_id; resource = 'switch'; name = $switchName } |
                    ConvertTo-Json -Compress | Set-Content -LiteralPath $data.switch_marker -Encoding UTF8
            }

            $parameters = @{
                Name = $data.vm_name
                Generation = [int]$data.generation
                MemoryStartupBytes = [int64]$data.memory_bytes
                VHDPath = $data.disk_path
                ErrorAction = 'Stop'
            }
            if ($switchName) { $parameters['SwitchName'] = $switchName }
            $vm = New-VM @parameters
            Set-VM -VM $vm -Notes $data.ownership_json -ErrorAction Stop
            Set-VM -VM $vm -ProcessorCount ([int]$data.processor_count) `
                -DynamicMemoryEnabled ([bool]$data.dynamic_memory) -AutomaticCheckpointsEnabled $false -ErrorAction Stop
            Set-VMProcessor -VM $vm `
                -ExposeVirtualizationExtensions ([bool]$data.nested_virtualization) -ErrorAction Stop
            Get-VMNetworkAdapter -VM $vm -ErrorAction SilentlyContinue |
                Set-VMNetworkAdapter `
                    -MacAddressSpoofing $(if ($data.mac_spoofing) { 'On' } else { 'Off' }) -ErrorAction Stop
            if ([int]$data.generation -eq 2) {
                if ($data.secure_boot -eq 'disabled') {
                    Set-VMFirmware -VM $vm -EnableSecureBoot Off -ErrorAction Stop
                } else {
                    $templateName = if ($data.secure_boot -eq 'microsoft_uefi_ca') {
                        'MicrosoftUEFICertificateAuthority'
                    } else {
                        'MicrosoftWindows'
                    }
                    Set-VMFirmware -VM $vm -EnableSecureBoot On -SecureBootTemplate $templateName -ErrorAction Stop
                }
            }
            Start-VM -VM $vm -ErrorAction Stop | Out-Null
            @{ ok = $true; vm_name = $data.vm_name; disk_path = $data.disk_path; switch_name = $switchName } |
                ConvertTo-Json -Compress
        }
        'remove' {
            $vm = Get-VM -Name $data.vm_name -ErrorAction SilentlyContinue
            if ($vm) {
                $owner = $null
                try { $owner = $vm.Notes | ConvertFrom-Json } catch { }
                if ($null -eq $owner -or $owner.ownership_id -ne $data.ownership_id) {
                    Write-StructuredError 'hyperv_ownership_mismatch'
                    exit 1
                }
                if ($vm.State -ne 'Off') { Stop-VM -VM $vm -TurnOff -Force -ErrorAction Stop }
                Remove-VM -VM $vm -Force -ErrorAction Stop
            }
            $diskOwner = Read-Owner $data.disk_marker
            if ($diskOwner -and $diskOwner.ownership_id -eq $data.ownership_id) {
                Remove-Item -LiteralPath $data.disk_path -Force -ErrorAction SilentlyContinue
                Remove-Item -LiteralPath $data.disk_marker -Force -ErrorAction SilentlyContinue
            } elseif (Test-Path -LiteralPath $data.disk_path) {
                Write-StructuredError 'hyperv_disk_ownership_mismatch'
                exit 1
            }
            if ($data.owns_switch) {
                $switchOwner = Read-Owner $data.switch_marker
                if ($switchOwner -and $switchOwner.ownership_id -eq $data.ownership_id) {
                    $ownedSwitch = Get-VMSwitch -Name $data.switch_name -ErrorAction SilentlyContinue
                    if ($ownedSwitch) { Remove-VMSwitch -VMSwitch $ownedSwitch -Force -ErrorAction Stop }
                    Remove-Item -LiteralPath $data.switch_marker -Force -ErrorAction SilentlyContinue
                } elseif (Get-VMSwitch -Name $data.switch_name -ErrorAction SilentlyContinue) {
                    Write-StructuredError 'hyperv_switch_ownership_mismatch'
                    exit 1
                }
            }
            @{ ok = $true } | ConvertTo-Json -Compress
        }
        default { throw 'Unsupported lifecycle action.' }
    }
} catch {
    Write-StructuredError 'hyperv_lifecycle_failed'
    exit 1
}
"""

    PROCESS_STATUS = r"""
$ErrorActionPreference = 'Stop'
$data = [Console]::In.ReadToEnd() | ConvertFrom-Json
try {
    $process = Get-Process -Id ([int]$data.process_id) -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        @{ alive = $false; start_time = $null } | ConvertTo-Json -Compress
    } else {
        @{
            alive = $true
            start_time = $process.StartTime.ToUniversalTime().ToString('o')
        } | ConvertTo-Json -Compress
    }
} catch {
    [Console]::Error.WriteLine((@{ error_code = 'hyperv_process_status_failed' } | ConvertTo-Json -Compress))
    exit 1
}
"""


def _is_windows_host() -> bool:
    """Return whether the current host can support Hyper-V."""
    return os.name == "nt"


def _now() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(tz=timezone.utc)


def _hash_bytes(data: bytes) -> str:
    """Return the SHA-256 digest for ``data``."""
    return hashlib.sha256(data).hexdigest()


def _safe_resource_name(*, prefix: str, session_id: str, attempt_id: str, environment_name: str) -> str:
    """
    Build a bounded, collision-resistant Hyper-V resource name.

    Returns:
        str: A Hyper-V-compatible owned resource name.
    """
    readable = re.sub(r"[^A-Za-z0-9-]", "-", f"{session_id}-{environment_name}").strip("-") or "session"
    digest = _hash_bytes(f"{session_id}\0{attempt_id}\0{environment_name}".encode())[:12]
    available = max(1, 63 - len(prefix) - len(digest) - 2)
    return f"{prefix}-{readable[:available]}-{digest}"


def _lock_file(handle: Any) -> None:
    """Acquire an advisory lock on an open state file."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        if handle.read(1) == "":
            handle.seek(0)
            handle.write(" ")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    """Release an advisory state-file lock."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_state_file(path: Path) -> dict[str, Any]:
    """
    Read provider state under a cross-process lock.

    Returns:
        dict[str, Any]: The session state mapping.
    """
    if not path.exists():
        return {}
    with open(path, "a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            handle.seek(0)
            raw = handle.read()
        finally:
            _unlock_file(handle)
    return json.loads(raw) if raw.strip() else {}


def _update_state_file(*, path: Path, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """Read-modify-write provider state under a cross-process lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", encoding="utf-8") as handle:
        _lock_file(handle)
        try:
            handle.seek(0)
            raw = handle.read()
            state = json.loads(raw) if raw.strip() else {}
            updated = mutate(state)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(updated, indent=2, sort_keys=True))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)


def _evidence(
    *,
    operation: str,
    outcome: SandboxOperationStatus,
    started_at: datetime,
    session_id: str | None = None,
    environment_name: str | None = None,
    operation_context: SandboxOperationContext | None = None,
    error_code: str | None = None,
    input_size_bytes: int | None = None,
    output_size_bytes: int | None = None,
    sha256: str | None = None,
    artifact_ids: tuple[str, ...] = (),
    metadata: dict[str, JSONValue] | None = None,
) -> SandboxOperationEvidence:
    """
    Build content-free Hyper-V operation evidence.

    Returns:
        SandboxOperationEvidence: The immutable evidence event.
    """
    return SandboxOperationEvidence(
        provider="hyperv",
        operation=operation,
        outcome=outcome.value,
        started_at=started_at,
        ended_at=_now(),
        session_id=session_id,
        environment_name=environment_name,
        call_id=operation_context.call_id if operation_context else None,
        attempt_id=operation_context.attempt_id if operation_context else None,
        error_code=error_code,
        input_size_bytes=input_size_bytes,
        output_size_bytes=output_size_bytes,
        sha256=sha256,
        artifact_ids=artifact_ids,
        metadata=metadata or {},
    )


async def _emit_evidence_async(*, sink: CapabilityEvidenceSink | None, evidence: SandboxOperationEvidence) -> None:
    """Emit evidence when a sink is configured."""
    if sink is not None:
        await sink.emit_async(evidence)


class HyperVSandboxProcess(SandboxProcess):
    """A transport process with explicit cancellation and bounded output."""

    def __init__(
        self,
        *,
        environment: HyperVSandboxEnvironment,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None,
    ) -> None:
        """Start the guest execution task immediately."""
        self._environment = environment
        self._request = request
        self._operation_context = operation_context
        self._started_at = _now()
        self._cancellation_event = asyncio.Event()
        self._task = asyncio.create_task(self._run_async())
        self._termination_lock = asyncio.Lock()

    async def communicate_async(
        self,
        *,
        stdin: bytes | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> SandboxExecResult:
        """
        Wait for completion and propagate external cancellation.

        Returns:
            SandboxExecResult: The bounded guest process result.

        Raises:
            HyperVUnsupportedCapabilityError: If stdin differs from the started request.
            asyncio.CancelledError: If the calling task is cancelled.
        """
        if stdin is not None and stdin != self._request.stdin:
            raise HyperVUnsupportedCapabilityError(
                capability="late_process_stdin",
                transport=self._environment.transport.name,
            )
        bridge = (
            asyncio.create_task(self._bridge_cancellation_async(cancellation_event)) if cancellation_event else None
        )
        try:
            return await self._task
        except asyncio.CancelledError:
            await asyncio.shield(self.terminate_async())
            raise
        finally:
            if bridge is not None:
                bridge.cancel()
                await asyncio.gather(bridge, return_exceptions=True)

    async def terminate_async(self) -> None:
        """Cancel the host transport process exactly once."""
        async with self._termination_lock:
            self._cancellation_event.set()
        await asyncio.gather(self._task, return_exceptions=True)

    async def _bridge_cancellation_async(self, event: asyncio.Event) -> None:
        await event.wait()
        self._cancellation_event.set()

    async def _run_async(self) -> SandboxExecResult:
        try:
            raw = await self._environment.transport.exec_async(
                request=self._request,
                timeout_seconds=self._environment.exec_timeout(self._request),
                cancellation_event=self._cancellation_event,
                stdout_limit=self._environment.spec.limits.max_stdout_bytes,
                stderr_limit=self._environment.spec.limits.max_stderr_bytes,
            )
            return await self._environment.finalize_exec_async(
                request=self._request,
                raw=raw,
                started_at=self._started_at,
                operation_context=self._operation_context,
            )
        except HyperVSandboxError as error:
            return await self._environment.exec_error_async(
                error=error,
                started_at=self._started_at,
                operation_context=self._operation_context,
            )
        finally:
            await self._environment.remove_process_async(self)


class HyperVSandboxEnvironment(SandboxEnvironment):
    """A named guest execution surface backed by one owned Hyper-V VM."""

    def __init__(
        self,
        *,
        session: HyperVSandboxSession,
        spec: SandboxEnvironmentSpec,
        config: HyperVEnvironmentConfig,
        vm_name: str,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> None:
        """Initialize the environment without connecting to the guest."""
        self._session = session
        self._spec = spec
        self._config = config
        self._vm_name = vm_name
        self._evidence_sink = evidence_sink
        self._transport: HyperVGuestTransport | None = None
        self._operation_lock = asyncio.Lock()
        self._process_lock = asyncio.Lock()
        self._processes: set[HyperVSandboxProcess] = set()
        self._closed = False

    @property
    def name(self) -> str:
        """The environment name."""
        return self._spec.name

    @property
    def spec(self) -> SandboxEnvironmentSpec:
        """The provider-neutral environment specification."""
        return self._spec

    @property
    def transport(self) -> HyperVGuestTransport:
        """
        The connected guest transport.

        Raises:
            RuntimeError: If VM initialization has not bound a transport.
        """
        if self._transport is None:
            raise RuntimeError("Hyper-V guest transport is not connected.")
        return self._transport

    @property
    def connection_info(self) -> SandboxConnectionInfo:
        """Non-secret VM connection and security metadata."""
        policy = self._session.provider.config.security_policy
        return SandboxConnectionInfo(
            provider="hyperv",
            session_id=self._session.session_id,
            environment_name=self.name,
            transport=self._config.transport.value,
            endpoint=self._vm_name,
            metadata={
                "security_boundary": True,
                "network_mode": self._config.network_mode.value,
                "external_egress": policy.allow_internet_egress,
                "host_filesystem_sharing": False,
                "nested_virtualization": policy.allow_nested_virtualization,
                "device_passthrough": False,
                "mac_spoofing": policy.allow_mac_spoofing,
                "secure_boot": self._config.secure_boot.value,
                "remote_process_cleanup_guaranteed": (
                    self._transport.remote_process_cleanup_guaranteed if self._transport else False
                ),
            },
        )

    def bind_transport(self, transport: HyperVGuestTransport) -> None:
        """Bind the configured transport after VM creation."""
        self._transport = transport

    async def initialize_async(self) -> None:
        """
        Wait for readiness, create the workspace, and apply setup content.

        Raises:
            SandboxSetupError: If a setup file, mode update, or command fails.
        """
        started_at = _now()
        await self._wait_ready_async()
        await self.transport.write_file_async(path=self._resolve_path(".pyrit-owner"), data=b"pyrit")
        for setup_file in (*self._config.setup_files, *self._spec.setup_files):
            result = await self.write_file_async(path=setup_file.path, data=setup_file.content)
            if result.status is not SandboxOperationStatus.SUCCEEDED:
                raise SandboxSetupError(f"A Hyper-V setup file failed with status '{result.status.value}'.")
            if setup_file.executable and self._config.guest_os is HyperVGuestOS.LINUX:
                chmod = await self.exec_async(request=SandboxExecRequest(argv=("chmod", "+x", setup_file.path)))
                if chmod.status is not SandboxOperationStatus.SUCCEEDED:
                    raise SandboxSetupError("A Hyper-V setup executable mode update failed.")
        for setup_script in (*self._config.setup_scripts, *self._spec.setup_scripts):
            result = await self.exec_async(request=setup_script.request)
            if result.status is not SandboxOperationStatus.SUCCEEDED:
                raise SandboxSetupError(
                    f"Hyper-V setup command failed with status '{result.status.value}' "
                    f"and exit code {result.exit_code}."
                )
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=self._operation_evidence(
                operation="environment_setup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata=self.connection_info.metadata,
            ),
        )

    async def _wait_ready_async(self) -> None:
        deadline = asyncio.get_running_loop().time() + self._config.readiness.timeout_seconds
        last_error: HyperVSandboxError | None = None
        while True:
            try:
                await self.transport.connect_async()
                if self._config.readiness.probe_argv is not None:
                    probe = await self.transport.exec_async(
                        request=SandboxExecRequest(argv=self._config.readiness.probe_argv),
                        timeout_seconds=min(30.0, self._config.readiness.timeout_seconds),
                        cancellation_event=None,
                        stdout_limit=4096,
                        stderr_limit=4096,
                    )
                    if probe.exit_code != 0:
                        raise HyperVTransportError(
                            message="Guest readiness command failed.",
                            error_code="hyperv_guest_probe_failed",
                        )
                return
            except HyperVSandboxError as error:
                last_error = error
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise HyperVLifecycleError(
                    operation="readiness",
                    detail=f"Transport did not become ready ({last_error.error_code if last_error else 'unknown'}).",
                ) from last_error
            await asyncio.sleep(min(self._config.readiness.poll_interval_seconds, remaining))

    async def start_process_async(
        self,
        *,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxProcess:
        """
        Start one guest process.

        Returns:
            SandboxProcess: The running transport-backed process.

        Raises:
            RuntimeError: If the environment is closed.
        """
        async with self._operation_lock:
            if self._closed:
                raise RuntimeError("Sandbox environment is closed.")
            resolved = self._resolve_request(request)
            process = HyperVSandboxProcess(environment=self, request=resolved, operation_context=operation_context)
            async with self._process_lock:
                self._processes.add(process)
            return process

    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        cancellation_event: asyncio.Event | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxExecResult:
        """
        Execute one bounded guest process.

        Returns:
            SandboxExecResult: The process outcome and evidence.
        """
        started_at = _now()
        try:
            process = await self.start_process_async(request=request, operation_context=operation_context)
        except (HyperVSandboxError, SandboxPathEscapeError, RuntimeError, ValueError) as error:
            typed = (
                error
                if isinstance(error, HyperVSandboxError)
                else HyperVTransportError(
                    message="Guest process could not be started.",
                    error_code="process_start_failed",
                )
            )
            return await self.exec_error_async(
                error=typed,
                started_at=started_at,
                operation_context=operation_context,
            )
        return await process.communicate_async(stdin=request.stdin, cancellation_event=cancellation_event)

    async def finalize_exec_async(
        self,
        *,
        request: SandboxExecRequest,
        raw: GuestExecResult,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxExecResult:
        """
        Map a guest result to the provider-neutral result and evidence.

        Returns:
            SandboxExecResult: The normalized execution result.
        """
        if raw.cancelled:
            status, error_code = SandboxOperationStatus.CANCELLED, "cancelled"
        elif raw.timed_out:
            status, error_code = SandboxOperationStatus.TIMED_OUT, "timeout"
        elif raw.stdout_truncated or raw.stderr_truncated:
            status, error_code = SandboxOperationStatus.TRUNCATED, "output_limit_exceeded"
        elif raw.exit_code == 0:
            status, error_code = SandboxOperationStatus.SUCCEEDED, None
        else:
            status, error_code = SandboxOperationStatus.FAILED, "nonzero_exit"
        evidence = self._operation_evidence(
            operation="exec",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error_code,
            input_size_bytes=len(request.stdin or b""),
            output_size_bytes=len(raw.stdout) + len(raw.stderr),
            metadata={
                "exit_code": raw.exit_code,
                "stdout_truncated": raw.stdout_truncated,
                "stderr_truncated": raw.stderr_truncated,
                "transport": self.transport.name,
                "remote_process_cleanup_guaranteed": self.transport.remote_process_cleanup_guaranteed,
            },
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxExecResult(
            status=status,
            stdout=raw.stdout,
            stderr=raw.stderr,
            exit_code=raw.exit_code,
            stdout_truncated=raw.stdout_truncated,
            stderr_truncated=raw.stderr_truncated,
            timed_out=raw.timed_out,
            cancelled=raw.cancelled,
            error_code=error_code,
            evidence=(evidence,),
        )

    async def exec_error_async(
        self,
        *,
        error: HyperVSandboxError,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxExecResult:
        """Return a sanitized typed exec failure."""
        status = (
            SandboxOperationStatus.CANCELLED
            if error.error_code == "hyperv_powershell_cancelled"
            else SandboxOperationStatus.TIMED_OUT
            if error.error_code == "hyperv_powershell_timeout"
            else SandboxOperationStatus.FAILED
        )
        evidence = self._operation_evidence(
            operation="exec",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error.error_code,
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxExecResult(
            status=status,
            timed_out=status is SandboxOperationStatus.TIMED_OUT,
            cancelled=status is SandboxOperationStatus.CANCELLED,
            error_code=error.error_code,
            error_message=str(error),
            evidence=(evidence,),
        )

    async def read_file_async(
        self,
        *,
        path: str,
        max_bytes: int | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxReadResult:
        """
        Read a bounded binary guest file.

        Returns:
            SandboxReadResult: The file data or an explicit error status.
        """
        started_at = _now()
        if max_bytes is not None and max_bytes <= 0:
            return await self._read_error_async(
                status=SandboxOperationStatus.FAILED,
                error_code="invalid_read_limit",
                error_message="Read limit must be greater than zero.",
                started_at=started_at,
                operation_context=operation_context,
            )
        limit = min(max_bytes or self._spec.limits.max_read_bytes, self._spec.limits.max_read_bytes)
        try:
            resolved = self._resolve_path(path)
            data = await self.transport.read_file_async(path=resolved, max_bytes=limit)
        except SandboxPathEscapeError:
            return await self._read_error_async(
                status=SandboxOperationStatus.PATH_ESCAPE,
                error_code="path_escape",
                error_message="Requested path is outside the sandbox environment.",
                started_at=started_at,
                operation_context=operation_context,
            )
        except HyperVSandboxError as error:
            status = (
                SandboxOperationStatus.NOT_FOUND
                if error.error_code == "file_not_found"
                else SandboxOperationStatus.PERMISSION_DENIED
                if error.error_code == "permission_denied"
                else SandboxOperationStatus.TOO_LARGE
                if error.error_code == "read_limit_exceeded"
                else SandboxOperationStatus.FAILED
            )
            return await self._read_error_async(
                status=status,
                error_code=error.error_code,
                error_message=str(error),
                started_at=started_at,
                operation_context=operation_context,
            )
        sha256 = _hash_bytes(data)
        artifact = self._artifact(path=path, size_bytes=len(data), sha256=sha256)
        evidence = self._operation_evidence(
            operation="read",
            outcome=SandboxOperationStatus.SUCCEEDED,
            started_at=started_at,
            operation_context=operation_context,
            output_size_bytes=len(data),
            sha256=sha256,
            artifact_ids=(artifact.artifact_id,),
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxReadResult(
            status=SandboxOperationStatus.SUCCEEDED,
            data=data,
            size_bytes=len(data),
            sha256=sha256,
            artifacts=(artifact,),
            evidence=(evidence,),
        )

    async def write_file_async(
        self,
        *,
        path: str,
        data: bytes,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxWriteResult:
        """
        Write a bounded binary guest file.

        Returns:
            SandboxWriteResult: The write metadata or an explicit error status.
        """
        started_at = _now()
        if len(data) > self._spec.limits.max_write_bytes:
            return await self._write_error_async(
                status=SandboxOperationStatus.TOO_LARGE,
                error_code="write_limit_exceeded",
                error_message=f"Data size {len(data)} exceeds write limit {self._spec.limits.max_write_bytes}.",
                started_at=started_at,
                operation_context=operation_context,
                input_size_bytes=len(data),
            )
        try:
            resolved = self._resolve_path(path)
            await self.transport.write_file_async(path=resolved, data=data)
        except SandboxPathEscapeError:
            return await self._write_error_async(
                status=SandboxOperationStatus.PATH_ESCAPE,
                error_code="path_escape",
                error_message="Requested path is outside the sandbox environment.",
                started_at=started_at,
                operation_context=operation_context,
                input_size_bytes=len(data),
            )
        except HyperVSandboxError as error:
            return await self._write_error_async(
                status=(
                    SandboxOperationStatus.PERMISSION_DENIED
                    if error.error_code == "permission_denied"
                    else SandboxOperationStatus.FAILED
                ),
                error_code=error.error_code,
                error_message=str(error),
                started_at=started_at,
                operation_context=operation_context,
                input_size_bytes=len(data),
                side_effect_completed=None,
            )
        sha256 = _hash_bytes(data)
        artifact = self._artifact(path=path, size_bytes=len(data), sha256=sha256)
        evidence = self._operation_evidence(
            operation="write",
            outcome=SandboxOperationStatus.SUCCEEDED,
            started_at=started_at,
            operation_context=operation_context,
            input_size_bytes=len(data),
            sha256=sha256,
            artifact_ids=(artifact.artifact_id,),
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxWriteResult(
            status=SandboxOperationStatus.SUCCEEDED,
            size_bytes=len(data),
            sha256=sha256,
            artifacts=(artifact,),
            evidence=(evidence,),
        )

    async def close_async(self) -> None:
        """Terminate tracked host transport processes and close the transport."""
        async with self._operation_lock:
            self._closed = True
            async with self._process_lock:
                processes = tuple(self._processes)
        await asyncio.gather(*(process.terminate_async() for process in processes), return_exceptions=True)
        if self._transport is not None:
            await self._transport.close_async()

    async def remove_process_async(self, process: HyperVSandboxProcess) -> None:
        """Remove a completed process from tracking."""
        async with self._process_lock:
            self._processes.discard(process)

    def exec_timeout(self, request: SandboxExecRequest) -> float:
        """Return the environment-bounded execution timeout."""
        return min(request.timeout_seconds or self._spec.limits.max_exec_seconds, self._spec.limits.max_exec_seconds)

    def _resolve_request(self, request: SandboxExecRequest) -> SandboxExecRequest:
        cwd = self._resolve_path(request.cwd or "")
        return request.model_copy(update={"cwd": cwd})

    def _resolve_path(self, path: str) -> str:
        if self._config.guest_os is HyperVGuestOS.WINDOWS:
            if ntpath.isabs(path):
                raise SandboxPathEscapeError("Absolute guest paths are not allowed.")
            root = ntpath.normpath(self._config.workspace_root)
            candidate = ntpath.normpath(ntpath.join(root, path))
            if candidate.casefold() != root.casefold() and not candidate.casefold().startswith(root.casefold() + "\\"):
                raise SandboxPathEscapeError("Guest path leaves the environment workspace.")
            return candidate
        if posixpath.isabs(path):
            raise SandboxPathEscapeError("Absolute guest paths are not allowed.")
        root = posixpath.normpath(self._config.workspace_root)
        candidate = posixpath.normpath(posixpath.join(root, path))
        if candidate != root and not candidate.startswith(root + "/"):
            raise SandboxPathEscapeError("Guest path leaves the environment workspace.")
        return candidate

    def _artifact(self, *, path: str, size_bytes: int, sha256: str) -> SandboxArtifact:
        artifact_id = f"{self._session.session_id}:{self.name}:{path}"
        return SandboxArtifact(
            artifact_id=artifact_id,
            uri=f"sandbox://{self._session.session_id}/{self.name}/{path}",
            sha256=sha256,
            size_bytes=size_bytes,
        )

    def _operation_evidence(
        self,
        *,
        operation: str,
        outcome: SandboxOperationStatus,
        started_at: datetime,
        operation_context: SandboxOperationContext | None = None,
        error_code: str | None = None,
        input_size_bytes: int | None = None,
        output_size_bytes: int | None = None,
        sha256: str | None = None,
        artifact_ids: tuple[str, ...] = (),
        metadata: dict[str, JSONValue] | None = None,
    ) -> SandboxOperationEvidence:
        return _evidence(
            operation=operation,
            outcome=outcome,
            started_at=started_at,
            session_id=self._session.session_id,
            environment_name=self.name,
            operation_context=operation_context,
            error_code=error_code,
            input_size_bytes=input_size_bytes,
            output_size_bytes=output_size_bytes,
            sha256=sha256,
            artifact_ids=artifact_ids,
            metadata=metadata,
        )

    async def _read_error_async(
        self,
        *,
        status: SandboxOperationStatus,
        error_code: str,
        error_message: str,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxReadResult:
        evidence = self._operation_evidence(
            operation="read",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error_code,
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxReadResult(
            status=status,
            error_code=error_code,
            error_message=error_message,
            evidence=(evidence,),
        )

    async def _write_error_async(
        self,
        *,
        status: SandboxOperationStatus,
        error_code: str,
        error_message: str,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
        input_size_bytes: int,
        side_effect_completed: bool | None = False,
    ) -> SandboxWriteResult:
        evidence = self._operation_evidence(
            operation="write",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error_code,
            input_size_bytes=input_size_bytes,
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxWriteResult(
            status=status,
            error_code=error_code,
            error_message=error_message,
            side_effect_completed=side_effect_completed,
            evidence=(evidence,),
        )


class HyperVSandboxSession(SandboxSession):
    """A per-attempt owner of named VMs, disks, switches, and guest transports."""

    def __init__(
        self,
        *,
        provider: HyperVSandboxProvider,
        spec: SandboxSessionSpec,
        resources: tuple[dict[str, Any], ...],
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> None:
        """Initialize a Hyper-V session without creating host resources."""
        super().__init__(provider_name=provider.name, spec=spec, evidence_sink=evidence_sink)
        self.provider = provider
        self._resources = resources
        self._initialization_failed = False
        self._environments = tuple(
            HyperVSandboxEnvironment(
                session=self,
                spec=environment_spec,
                config=provider.config.get_environment(environment_spec.name),
                vm_name=str(resource["vm_name"]),
                evidence_sink=evidence_sink,
            )
            for environment_spec, resource in zip(spec.environments, resources, strict=True)
        )

    @property
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        """The session environments."""
        return self._environments

    async def _initialize_async(self) -> None:
        started_at = _now()
        await self.provider._register_session_state_async(session=self)
        try:
            for environment, resource in zip(self._environments, self._resources, strict=True):
                await self.provider._create_vm_async(resource=resource)
                transport = self.provider._create_guest_transport(
                    vm_name=str(resource["vm_name"]),
                    config=self.provider.config.get_environment(environment.name),
                )
                environment.bind_transport(transport)
                await environment.initialize_async()
        except BaseException:
            self._initialization_failed = True
            raise
        await self.emit_lifecycle_evidence_async(operation="session_setup", started_at=started_at)

    async def _close_async(self) -> None:
        started_at = _now()
        await asyncio.gather(*(environment.close_async() for environment in self._environments), return_exceptions=True)
        retain = self.provider.config.retain_resources_on_close or (
            self._initialization_failed and self.provider.config.retain_resources_on_failure
        )
        failures: list[BaseException] = []
        if not retain:
            for resource in reversed(self._resources):
                try:
                    await self.provider._remove_vm_async(resource=resource)
                except BaseException as error:
                    failures.append(error)
        if not failures:
            await self.provider._unregister_session_state_async(session_id=self.session_id)
        await self.provider._remove_session_async(self.session_id)
        outcome = SandboxOperationStatus.FAILED if failures else SandboxOperationStatus.SUCCEEDED
        await self.emit_lifecycle_evidence_async(
            operation="session_cleanup",
            started_at=started_at,
            outcome=outcome,
            error_code="hyperv_cleanup_failed" if failures else None,
            metadata={"resources_retained": retain, "cleanup_failures": len(failures)},
        )
        if failures:
            raise HyperVLifecycleError(operation="cleanup", detail=f"{len(failures)} owned resource set(s) failed.")

    async def emit_lifecycle_evidence_async(
        self,
        *,
        operation: str,
        started_at: datetime,
        outcome: SandboxOperationStatus = SandboxOperationStatus.SUCCEEDED,
        error_code: str | None = None,
        metadata: dict[str, JSONValue] | None = None,
    ) -> None:
        """Emit one session lifecycle event."""
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                operation=operation,
                outcome=outcome,
                started_at=started_at,
                session_id=self.session_id,
                error_code=error_code,
                metadata={
                    "attempt_id": str(self._spec.attempt_id),
                    "environment_count": len(self._environments),
                    **(metadata or {}),
                },
            ),
        )


class HyperVSandboxProvider(SandboxProvider):
    """Create isolated per-attempt Hyper-V VMs with durable owned-resource cleanup."""

    def __init__(
        self,
        *,
        config: HyperVSandboxProviderConfig,
        evidence_sink: CapabilityEvidenceSink | None = None,
        power_shell_runner: PowerShellCommandRunner | None = None,
        secret_resolver: HyperVSecretResolver | None = None,
        guest_transport_factory: HyperVGuestTransportFactory | None = None,
    ) -> None:
        """Initialize configuration without touching Hyper-V or credentials."""
        super().__init__()
        self.config = config
        self._evidence_sink = evidence_sink
        self._power_shell_runner = power_shell_runner or PowerShellCommandRunner(
            executable=config.powershell_executable,
            max_output_bytes=config.max_command_output_bytes,
        )
        self._secret_resolver = secret_resolver or EnvironmentHyperVSecretResolver()
        self._guest_transport_factory = guest_transport_factory or create_default_guest_transport
        state_dir = config.state_dir or Path(tempfile.gettempdir()) / "pyrit-sandbox-hyperv"
        self._state_path = state_dir / f"{config.vm_name_prefix}.state.json"
        self._resource_dir = state_dir / "resources"
        self._state_lock = asyncio.Lock()
        self._sessions_lock = asyncio.Lock()
        self._sessions: dict[str, HyperVSandboxSession] = {}
        self._owner_process_started_at: str | None = None

    @property
    def name(self) -> str:
        """The stable provider name."""
        return "hyperv"

    @property
    def is_security_boundary(self) -> bool:
        """Whether this provider isolates untrusted guest workloads."""
        return True

    async def _prepare_async(self) -> None:
        started_at = _now()
        if not _is_windows_host():
            raise HyperVPlatformError
        if self.config.compose_delegation.enabled:
            raise HyperVComposeDelegationUnsupportedError
        result = await self._run_preflight_async()
        self._validate_preflight(result)
        process_status = await self._process_status_async(process_id=os.getpid())
        if not process_status.get("alive") or not isinstance(process_status.get("start_time"), str):
            raise HyperVPrerequisiteError(
                error_code="hyperv_process_identity_unavailable",
                message="Could not establish a durable owner identity for Hyper-V orphan recovery.",
            )
        self._owner_process_started_at = str(process_status["start_time"])
        await asyncio.to_thread(self._resource_dir.mkdir, parents=True, exist_ok=True)
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                operation="provider_prepare",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={
                    "environment_count": len(self.config.environments),
                    "external_switch_allowed": self.config.security_policy.allow_external_switch,
                    "internet_egress_allowed": self.config.security_policy.allow_internet_egress,
                    "host_filesystem_sharing_allowed": False,
                    "nested_virtualization_allowed": self.config.security_policy.allow_nested_virtualization,
                    "device_passthrough_allowed": self.config.security_policy.allow_device_passthrough,
                    "device_passthrough_enabled": False,
                    "mac_spoofing_allowed": self.config.security_policy.allow_mac_spoofing,
                },
            ),
        )

    async def _run_preflight_async(self) -> dict[str, object]:
        images = [str(environment.base_vhdx) for environment in self.config.environments if environment.base_vhdx]
        templates = [environment.template_vm for environment in self.config.environments if environment.template_vm]
        switches = [
            {"name": environment.switch_name, "switch_type": environment.network_mode.value.capitalize()}
            for environment in self.config.environments
            if environment.switch_name
        ]
        try:
            return await self._power_shell_runner.run_json_async(
                script=_HyperVScripts.PREFLIGHT,
                payload={
                    "images": images,
                    "template_vms": templates,
                    "switches": switches,
                    "require_ssh": any(
                        environment.transport is HyperVGuestTransportKind.SSH
                        for environment in self.config.environments
                    ),
                    "ssh_executable": self.config.ssh_executable,
                },
                timeout_seconds=self.config.cli_timeout_seconds,
            )
        except HyperVTransportError as error:
            raise HyperVPrerequisiteError(
                error_code=error.error_code,
                message="Hyper-V host prerequisite validation could not complete.",
            ) from error

    def _validate_preflight(self, result: Mapping[str, object]) -> None:
        if result.get("permission_denied"):
            raise HyperVPermissionError
        if not result.get("module_available"):
            raise HyperVPrerequisiteError(
                error_code="hyperv_module_missing",
                message="The Windows Hyper-V PowerShell module is not installed. Enable the Hyper-V feature and tools.",
            )
        checks = (
            ("missing_cmdlets", "hyperv_cmdlets_missing", "Required Hyper-V cmdlets are unavailable"),
            ("missing_images", "hyperv_images_missing", "Configured base VHDX files are missing"),
            ("missing_templates", "hyperv_templates_missing", "Configured template VMs are missing"),
            ("missing_switches", "hyperv_switches_missing", "Configured allow-listed switches are missing"),
            (
                "mismatched_switches",
                "hyperv_switch_types_mismatched",
                "Configured switches do not match their required isolation type",
            ),
        )
        for key, code, message in checks:
            values = result.get(key)
            if isinstance(values, list) and values:
                raise HyperVPrerequisiteError(error_code=code, message=f"{message}: {', '.join(map(str, values))}.")
        if not result.get("ssh_available", True):
            raise HyperVPrerequisiteError(
                error_code="hyperv_ssh_missing",
                message="OpenSSH client is required by an SSH guest environment but was not found.",
            )

    async def _process_status_async(self, *, process_id: int) -> dict[str, object]:
        try:
            return await self._power_shell_runner.run_json_async(
                script=_HyperVScripts.PROCESS_STATUS,
                payload={"process_id": process_id},
                timeout_seconds=min(self.config.cli_timeout_seconds, 30.0),
            )
        except HyperVTransportError as error:
            raise HyperVPrerequisiteError(
                error_code=error.error_code,
                message="Could not verify a Hyper-V state owner process.",
            ) from error

    async def _prepare_task_async(self, task: SandboxTaskSpec) -> None:
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                operation="task_prepare",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=_now(),
                metadata={"task_id_sha256": _hash_bytes(task.task_id.encode())},
            ),
        )

    async def _cleanup_task_async(self, task: SandboxTaskSpec) -> None:
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                operation="task_cleanup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=_now(),
                metadata={"task_id_sha256": _hash_bytes(task.task_id.encode())},
            ),
        )

    async def _create_session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> SandboxSession:
        started_at = _now()
        self._validate_session_limits(spec)
        resources = tuple(
            self._resource_for_environment(spec=spec, environment=environment) for environment in spec.environments
        )
        session = HyperVSandboxSession(
            provider=self,
            spec=spec,
            resources=resources,
            evidence_sink=evidence_sink or self._evidence_sink,
        )
        async with self._sessions_lock:
            self._sessions[spec.session_id] = session
        await session.emit_lifecycle_evidence_async(operation="session_create", started_at=started_at)
        return session

    def _validate_session_limits(self, spec: SandboxSessionSpec) -> None:
        for environment in spec.environments:
            limits = environment.limits
            transport = self.config.get_environment(environment.name).transport
            required_output = max(
                limits.max_read_bytes,
                limits.max_stdout_bytes,
                limits.max_stderr_bytes,
            )
            if transport is HyperVGuestTransportKind.POWERSHELL_DIRECT:
                required_output = (required_output * 4 // 3) + 65_536
            if required_output > self.config.max_command_output_bytes:
                raise HyperVPrerequisiteError(
                    error_code="hyperv_command_output_limit_too_small",
                    message=(
                        f"Environment '{environment.name}' limits require a larger "
                        "Hyper-V max_command_output_bytes setting."
                    ),
                )

    def _resource_for_environment(
        self,
        *,
        spec: SandboxSessionSpec,
        environment: SandboxEnvironmentSpec,
    ) -> dict[str, Any]:
        config = self.config.get_environment(environment.name)
        vm_name = _safe_resource_name(
            prefix=self.config.vm_name_prefix,
            session_id=spec.session_id,
            attempt_id=str(spec.attempt_id),
            environment_name=environment.name,
        )
        ownership_id = _hash_bytes(f"{spec.session_id}\0{spec.attempt_id}\0{environment.name}".encode())
        disk_path = self._resource_dir / f"{vm_name}.vhdx"
        owns_switch = config.switch_name is None and config.network_mode in {
            HyperVNetworkMode.PRIVATE,
            HyperVNetworkMode.INTERNAL,
        }
        switch_name = (
            config.switch_name if config.switch_name is not None else f"{vm_name[:55]}-net" if owns_switch else None
        )
        ownership = {
            "provider": "pyrit",
            "resource_type": "hyperv-sandbox",
            "ownership_id": ownership_id,
            "session_id": spec.session_id,
            "attempt_id": str(spec.attempt_id),
            "environment_name": environment.name,
        }
        return {
            "ownership_id": ownership_id,
            "ownership_json": json.dumps(ownership, separators=(",", ":")),
            "vm_name": vm_name,
            "environment_name": environment.name,
            "base_vhdx": str(config.base_vhdx) if config.base_vhdx is not None else None,
            "template_vm": config.template_vm,
            "template_checkpoint": config.template_checkpoint,
            "generation": config.generation,
            "processor_count": config.processor_count,
            "memory_bytes": config.memory_mb * 1024 * 1024,
            "dynamic_memory": config.dynamic_memory,
            "disk_strategy": config.disk_strategy.value,
            "max_disk_bytes": config.max_disk_size_gb * 1024 * 1024 * 1024,
            "disk_path": str(disk_path),
            "disk_marker": str(disk_path.with_suffix(".owner.json")),
            "switch_name": switch_name,
            "create_switch": owns_switch,
            "owns_switch": owns_switch,
            "switch_type": "Private" if config.network_mode is HyperVNetworkMode.PRIVATE else "Internal",
            "switch_marker": str(self._resource_dir / f"{vm_name}.switch.owner.json"),
            "secure_boot": config.secure_boot.value,
            "nested_virtualization": self.config.security_policy.allow_nested_virtualization,
            "mac_spoofing": self.config.security_policy.allow_mac_spoofing,
        }

    async def _create_vm_async(self, *, resource: Mapping[str, object]) -> None:
        try:
            await self._power_shell_runner.run_json_async(
                script=_HyperVScripts.LIFECYCLE,
                payload={"action": "create", **resource},
                timeout_seconds=self.config.cli_timeout_seconds,
            )
        except HyperVTransportError as error:
            raise HyperVLifecycleError(operation="create", detail=error.error_code) from error

    async def _remove_vm_async(self, *, resource: Mapping[str, object]) -> None:
        try:
            await self._power_shell_runner.run_json_async(
                script=_HyperVScripts.LIFECYCLE,
                payload={"action": "remove", **resource},
                timeout_seconds=self.config.cli_timeout_seconds,
            )
        except HyperVTransportError as error:
            raise HyperVLifecycleError(operation="remove", detail=error.error_code) from error

    def _create_guest_transport(
        self,
        *,
        vm_name: str,
        config: HyperVEnvironmentConfig,
    ) -> HyperVGuestTransport:
        return self._guest_transport_factory(
            vm_name=vm_name,
            config=config,
            power_shell_runner=self._power_shell_runner,
            secret_resolver=self._secret_resolver,
            power_shell_timeout_seconds=self.config.cli_timeout_seconds,
            ssh_executable=self.config.ssh_executable,
            max_output_bytes=self.config.max_command_output_bytes,
        )

    async def _register_session_state_async(self, *, session: HyperVSandboxSession) -> None:
        if self._owner_process_started_at is None:
            raise RuntimeError("Hyper-V provider owner identity is not initialized.")

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state[session.session_id] = {
                "created_at": _now().isoformat(),
                "attempt_id": str(session.spec.attempt_id),
                "owner_process_id": os.getpid(),
                "owner_process_started_at": self._owner_process_started_at,
                "resources": list(session._resources),
            }
            return state

        async with self._state_lock:
            await asyncio.to_thread(_update_state_file, path=self._state_path, mutate=mutate)

    async def _unregister_session_state_async(self, *, session_id: str) -> None:
        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state.pop(session_id, None)
            return state

        async with self._state_lock:
            await asyncio.to_thread(_update_state_file, path=self._state_path, mutate=mutate)

    async def _remove_session_async(self, session_id: str) -> None:
        async with self._sessions_lock:
            self._sessions.pop(session_id, None)

    async def _cleanup_async(self) -> None:
        started_at = _now()
        async with self._sessions_lock:
            sessions = tuple(self._sessions.values())
        results = await asyncio.gather(*(session.close_async() for session in sessions), return_exceptions=True)
        failures = [result for result in results if isinstance(result, BaseException)]
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                operation="provider_cleanup",
                outcome=SandboxOperationStatus.FAILED if failures else SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                error_code="session_cleanup_failed" if failures else None,
                metadata={"session_cleanup_failures": len(failures)},
            ),
        )
        if failures:
            raise HyperVLifecycleError(operation="provider_cleanup", detail=f"{len(failures)} session(s) failed.")

    async def _cleanup_orphans_async(self) -> int:
        started_at = _now()
        state = await asyncio.to_thread(_read_state_file, self._state_path)
        async with self._sessions_lock:
            active = set(self._sessions)
        cleaned = 0
        discovered = 0
        for session_id, record in state.items():
            if session_id in active or not isinstance(record, dict):
                continue
            if await self._state_owner_is_alive_async(record=record):
                continue
            discovered += 1
            resources = record.get("resources")
            if not isinstance(resources, list):
                continue
            failed = False
            for resource in reversed(resources):
                if not isinstance(resource, dict):
                    failed = True
                    break
                normalized_resource: dict[str, object] = {}
                for key, value in resource.items():
                    if not isinstance(key, str):
                        failed = True
                        break
                    normalized_resource[key] = value
                if failed:
                    break
                try:
                    await self._remove_vm_async(resource=normalized_resource)
                except HyperVSandboxError:
                    failed = True
                    break
            if failed:
                continue
            await self._unregister_session_state_async(session_id=session_id)
            cleaned += 1
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                operation="orphan_cleanup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"resources_discovered": discovered, "resources_cleaned": cleaned},
            ),
        )
        return cleaned

    async def _state_owner_is_alive_async(self, *, record: Mapping[str, object]) -> bool:
        process_id = record.get("owner_process_id")
        expected_start = record.get("owner_process_started_at")
        if not isinstance(process_id, int) or not isinstance(expected_start, str):
            return True
        try:
            status = await self._process_status_async(process_id=process_id)
        except HyperVPrerequisiteError:
            return True
        return bool(status.get("alive")) and status.get("start_time") == expected_start
