# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
External-command transports for the Hyper-V sandbox provider.

No command is run at import time. PowerShell templates are fixed and encoded as
UTF-16LE; every dynamic value is serialized to JSON on stdin. Guest credentials
are resolved only in memory and are never included in errors, evidence, or state.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
from abc import ABC, abstractmethod
from contextlib import suppress
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Protocol

from pyrit.sandbox.models import (
    HyperVEnvironmentConfig,
    HyperVGuestTransportKind,
    HyperVSecretReference,
    SandboxExecRequest,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class HyperVSandboxError(RuntimeError):
    """The base error for the Hyper-V sandbox provider."""

    def __init__(self, *, message: str, error_code: str) -> None:
        """Initialize a typed Hyper-V error."""
        self.error_code = error_code
        super().__init__(message)


class HyperVPlatformError(HyperVSandboxError):
    """The host platform cannot run Hyper-V."""

    def __init__(self) -> None:
        """Initialize a platform error."""
        super().__init__(
            message="HyperVSandboxProvider requires a Windows host with Hyper-V enabled.",
            error_code="hyperv_windows_required",
        )


class HyperVCliUnavailableError(HyperVSandboxError):
    """A required external host executable is unavailable."""

    def __init__(self, *, executable: str) -> None:
        """Initialize an executable availability error."""
        super().__init__(
            message=f"Required host executable '{executable}' was not found. Install it and ensure it is on PATH.",
            error_code="hyperv_cli_unavailable",
        )


class HyperVPrerequisiteError(HyperVSandboxError):
    """A required Hyper-V host prerequisite is missing."""

    def __init__(self, *, error_code: str, message: str) -> None:
        """Initialize an actionable prerequisite error."""
        super().__init__(message=message, error_code=error_code)


class HyperVPermissionError(HyperVSandboxError):
    """The caller lacks Hyper-V management permission."""

    def __init__(self) -> None:
        """Initialize a permission error."""
        super().__init__(
            message=(
                "The current identity cannot manage Hyper-V. Run from an elevated shell or add the identity to "
                "the local 'Hyper-V Administrators' group, then start a new sign-in session."
            ),
            error_code="hyperv_permission_denied",
        )


class HyperVLifecycleError(HyperVSandboxError):
    """A VM, disk, or switch lifecycle operation failed."""

    def __init__(self, *, operation: str, detail: str | None = None) -> None:
        """Initialize a sanitized lifecycle error."""
        self.operation = operation
        suffix = f" Detail: {detail}" if detail else ""
        super().__init__(
            message=f"Hyper-V lifecycle operation '{operation}' failed.{suffix}",
            error_code=f"hyperv_{operation}_failed",
        )


class HyperVTransportError(HyperVSandboxError):
    """A guest transport operation failed."""


class HyperVUnsupportedCapabilityError(HyperVTransportError):
    """The selected guest transport cannot honor an operation."""

    def __init__(self, *, capability: str, transport: str) -> None:
        """Initialize an explicit unsupported-capability error."""
        super().__init__(
            message=f"Guest transport '{transport}' does not support capability '{capability}'.",
            error_code="hyperv_unsupported_capability",
        )


class HyperVComposeDelegationUnsupportedError(HyperVSandboxError):
    """Compose delegation was requested before the Docker provider supports remote endpoints."""

    def __init__(self) -> None:
        """Initialize the typed Compose delegation limitation."""
        super().__init__(
            message=(
                "Compose-inside-VM delegation is not available in this layer: DockerSandboxProvider currently "
                "owns a local Docker CLI invocation and has no injectable Docker endpoint/context. The typed "
                "HyperVComposeDelegationConfig seam preserves the intended endpoint and Compose assets without "
                "reporting a false success."
            ),
            error_code="hyperv_compose_delegation_unsupported",
        )


class HyperVSecretResolver(Protocol):
    """Resolve credential material without placing it in provider configuration."""

    async def resolve_secret_async(self, reference: HyperVSecretReference) -> Mapping[str, str]:
        """Resolve one opaque secret reference."""


class EnvironmentHyperVSecretResolver:
    """Resolve a JSON credential object from an environment variable named by the reference."""

    async def resolve_secret_async(self, reference: HyperVSecretReference) -> Mapping[str, str]:
        """
        Resolve a credential mapping from the process environment.

        Returns:
            Mapping[str, str]: The in-memory credential fields.

        Raises:
            HyperVPrerequisiteError: If the reference is absent or malformed.
        """
        raw = os.environ.get(reference.secret_id)
        if raw is None:
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_not_found",
                message=f"Credential secret reference '{reference.secret_id}' could not be resolved.",
            )
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as error:
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_invalid",
                message=f"Credential secret reference '{reference.secret_id}' is not a JSON object.",
            ) from error
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_invalid",
                message=f"Credential secret reference '{reference.secret_id}' must resolve to string fields.",
            )
        return value


@dataclass(frozen=True)
class ExternalCommandResult:
    """A bounded external command result."""

    stdout: bytes
    stderr: bytes
    returncode: int | None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class GuestExecResult:
    """A transport-neutral guest process result."""

    stdout: bytes
    stderr: bytes
    exit_code: int | None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False


async def _read_limited_async(*, stream: asyncio.StreamReader | None, limit: int) -> tuple[bytes, bool]:
    """
    Drain a stream while retaining at most ``limit`` bytes.

    Returns:
        tuple[bytes, bool]: Retained bytes and whether truncation occurred.
    """
    if stream is None:
        return b"", False
    retained = bytearray()
    truncated = False
    while chunk := await stream.read(65_536):
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > max(remaining, 0):
            truncated = True
    return bytes(retained), truncated


async def _write_stdin_async(*, process: asyncio.subprocess.Process, data: bytes | None) -> None:
    """Write optional stdin and close the stream."""
    if process.stdin is None:
        return
    if data:
        process.stdin.write(data)
        with suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.drain()
    process.stdin.close()


async def run_external_command_async(
    *,
    argv: Sequence[str],
    stdin: bytes | None,
    timeout_seconds: float,
    cancellation_event: asyncio.Event | None,
    stdout_limit: int,
    stderr_limit: int,
) -> ExternalCommandResult:
    """
    Run one argv-only host command with bounded output and cooperative cancellation.

    Returns:
        ExternalCommandResult: The bounded process outcome.

    Raises:
        HyperVCliUnavailableError: If the requested executable is not installed.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if stdin is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise HyperVCliUnavailableError(executable=argv[0]) from error
    stdout_task = asyncio.create_task(_read_limited_async(stream=process.stdout, limit=stdout_limit))
    stderr_task = asyncio.create_task(_read_limited_async(stream=process.stderr, limit=stderr_limit))
    stdin_task = asyncio.create_task(_write_stdin_async(process=process, data=stdin))
    wait_task = asyncio.create_task(process.wait())
    cancel_task = asyncio.create_task(cancellation_event.wait()) if cancellation_event is not None else None
    timed_out = False
    cancelled = False
    try:
        waiting = (wait_task,) if cancel_task is None else (wait_task, cancel_task)
        done, _ = await asyncio.wait(waiting, timeout=timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
        timed_out = not done
        cancelled = cancel_task is not None and cancel_task in done
        if timed_out or cancelled:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
        await asyncio.gather(stdin_task, return_exceptions=True)
        stdout, stderr = await asyncio.gather(stdout_task, stderr_task)
    except BaseException:
        with suppress(ProcessLookupError):
            process.kill()
        await asyncio.gather(process.wait(), return_exceptions=True)
        raise
    finally:
        if cancel_task is not None:
            cancel_task.cancel()
        for task in (stdin_task, wait_task, stdout_task, stderr_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(stdin_task, wait_task, stdout_task, stderr_task, return_exceptions=True)
    return ExternalCommandResult(
        stdout=stdout[0],
        stderr=stderr[0],
        returncode=process.returncode,
        stdout_truncated=stdout[1],
        stderr_truncated=stderr[1],
        timed_out=timed_out,
        cancelled=cancelled,
    )


class PowerShellCommandRunner:
    """Run fixed PowerShell templates with JSON data supplied only over stdin."""

    def __init__(self, *, executable: str, max_output_bytes: int) -> None:
        """Initialize the runner without invoking PowerShell."""
        self._executable = executable
        self._max_output_bytes = max_output_bytes

    async def run_json_async(
        self,
        *,
        script: str,
        payload: Mapping[str, object],
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None = None,
        max_output_bytes: int | None = None,
    ) -> dict[str, object]:
        """
        Execute an encoded fixed script and parse its JSON result.

        Returns:
            dict[str, object]: The structured command result.

        Raises:
            HyperVTransportError: If PowerShell fails or emits invalid output.
        """
        encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
        output_limit = max_output_bytes or self._max_output_bytes
        result = await run_external_command_async(
            argv=(
                self._executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-EncodedCommand",
                encoded,
            ),
            stdin=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            timeout_seconds=timeout_seconds,
            cancellation_event=cancellation_event,
            stdout_limit=output_limit,
            stderr_limit=output_limit,
        )
        if result.timed_out:
            raise HyperVTransportError(message="PowerShell command timed out.", error_code="hyperv_powershell_timeout")
        if result.cancelled:
            raise HyperVTransportError(
                message="PowerShell command was cancelled.", error_code="hyperv_powershell_cancelled"
            )
        if result.returncode != 0:
            code = _structured_error_code(result.stderr)
            raise HyperVTransportError(message="PowerShell command failed.", error_code=code)
        try:
            value = json.loads(result.stdout.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HyperVTransportError(
                message="PowerShell command returned invalid structured output.",
                error_code="hyperv_powershell_invalid_output",
            ) from error
        if not isinstance(value, dict):
            raise HyperVTransportError(
                message="PowerShell command did not return an object.",
                error_code="hyperv_powershell_invalid_output",
            )
        return value


def _structured_error_code(stderr: bytes) -> str:
    """
    Extract only a stable error code from structured PowerShell stderr.

    Returns:
        str: The structured code or a sanitized fallback.
    """
    for line in reversed(stderr.decode("utf-8", errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("error_code"), str):
            return value["error_code"]
    return "hyperv_powershell_failed"


class _PowerShellDirectScript:
    SCRIPT = r"""
$ErrorActionPreference = 'Stop'
$data = [Console]::In.ReadToEnd() | ConvertFrom-Json
try {
    $secure = ConvertTo-SecureString $data.password -AsPlainText -Force
    $credential = New-Object System.Management.Automation.PSCredential($data.username, $secure)
    $result = Invoke-Command -VMName $data.vm_name -Credential $credential -ScriptBlock {
        param($request)
        $ErrorActionPreference = 'Stop'
        switch ($request.action) {
            'probe' {
                return @{ ok = $true }
            }
            'read' {
                try {
                    $bytes = [IO.File]::ReadAllBytes([string]$request.path)
                    return @{ data_base64 = [Convert]::ToBase64String($bytes); size_bytes = $bytes.Length }
                } catch [System.IO.FileNotFoundException] {
                    return @{ error_code = 'file_not_found' }
                } catch [System.UnauthorizedAccessException] {
                    return @{ error_code = 'permission_denied' }
                }
            }
            'write' {
                try {
                    $path = [string]$request.path
                    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($path)) | Out-Null
                    $bytes = [Convert]::FromBase64String([string]$request.data_base64)
                    [IO.File]::WriteAllBytes($path, $bytes)
                    return @{ size_bytes = $bytes.Length }
                } catch [System.UnauthorizedAccessException] {
                    return @{ error_code = 'permission_denied' }
                }
            }
            'exec' {
                function Quote-WindowsArgument([string]$value) {
                    if ($value -notmatch '[\s"]') { return $value }
                    $escaped = $value -replace '(\\*)"', '$1$1\"'
                    $escaped = $escaped -replace '(\\+)$', '$1$1'
                    return '"' + $escaped + '"'
                }
                $psi = New-Object System.Diagnostics.ProcessStartInfo
                if ($null -ne $request.argv) {
                    $psi.FileName = [string]$request.argv[0]
                    $arguments = @()
                    for ($index = 1; $index -lt $request.argv.Count; $index++) {
                        $arguments += Quote-WindowsArgument ([string]$request.argv[$index])
                    }
                    $psi.Arguments = $arguments -join ' '
                } else {
                    $scriptBytes = [Text.Encoding]::Unicode.GetBytes([string]$request.shell_script)
                    $encoded = [Convert]::ToBase64String($scriptBytes)
                    $psi.FileName = 'powershell.exe'
                    $psi.Arguments = '-NoProfile -NonInteractive -EncodedCommand ' + $encoded
                }
                $psi.UseShellExecute = $false
                $psi.RedirectStandardInput = $true
                $psi.RedirectStandardOutput = $true
                $psi.RedirectStandardError = $true
                $psi.CreateNoWindow = $true
                if ($request.cwd) { $psi.WorkingDirectory = [string]$request.cwd }
                foreach ($property in $request.environment.PSObject.Properties) {
                    $psi.EnvironmentVariables[$property.Name] = [string]$property.Value
                }
                $process = New-Object System.Diagnostics.Process
                $process.StartInfo = $psi
                [void]$process.Start()
                if ($request.stdin_base64) {
                    $stdinBytes = [Convert]::FromBase64String([string]$request.stdin_base64)
                    $process.StandardInput.BaseStream.Write($stdinBytes, 0, $stdinBytes.Length)
                    $process.StandardInput.BaseStream.Flush()
                }
                $process.StandardInput.Close()
                $stdoutTask = $process.StandardOutput.ReadToEndAsync()
                $stderrTask = $process.StandardError.ReadToEndAsync()
                $finished = $process.WaitForExit([int]([double]$request.timeout_seconds * 1000))
                if (-not $finished) {
                    & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
                    $process.WaitForExit()
                }
                return @{
                    stdout_base64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($stdoutTask.Result))
                    stderr_base64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($stderrTask.Result))
                    exit_code = if ($finished) { $process.ExitCode } else { $null }
                    timed_out = -not $finished
                }
            }
            default { throw 'Unsupported PowerShell Direct action.' }
        }
    } -ArgumentList $data.request
    $result | ConvertTo-Json -Compress -Depth 8
} catch {
    [Console]::Error.WriteLine((@{ error_code = 'hyperv_powershell_direct_failed' } | ConvertTo-Json -Compress))
    exit 1
}
"""


class HyperVGuestTransport(ABC):
    """A pluggable external-facility transport to one guest VM."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The stable transport name."""

    @property
    @abstractmethod
    def remote_process_cleanup_guaranteed(self) -> bool:
        """Whether cancellation reliably terminates the remote process tree."""

    @abstractmethod
    async def connect_async(self) -> None:
        """Confirm the guest is ready."""

    @abstractmethod
    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None,
        stdout_limit: int,
        stderr_limit: int,
    ) -> GuestExecResult:
        """Execute one bounded guest process."""

    @abstractmethod
    async def read_file_async(self, *, path: str, max_bytes: int) -> bytes:
        """Read a bounded binary guest file."""

    @abstractmethod
    async def write_file_async(self, *, path: str, data: bytes) -> None:
        """Write a binary guest file."""

    async def close_async(self) -> None:
        """Close transport resources."""
        return


class PowerShellDirectGuestTransport(HyperVGuestTransport):
    """PowerShell Direct transport for compatible Windows guests."""

    def __init__(
        self,
        *,
        vm_name: str,
        config: HyperVEnvironmentConfig,
        runner: PowerShellCommandRunner,
        secret_resolver: HyperVSecretResolver,
        command_timeout_seconds: float,
        max_output_bytes: int,
    ) -> None:
        """Initialize a PowerShell Direct transport."""
        self._vm_name = vm_name
        self._config = config
        self._runner = runner
        self._secret_resolver = secret_resolver
        self._command_timeout_seconds = command_timeout_seconds
        self._max_output_bytes = max_output_bytes

    @property
    def name(self) -> str:
        """The stable transport name."""
        return HyperVGuestTransportKind.POWERSHELL_DIRECT.value

    @property
    def remote_process_cleanup_guaranteed(self) -> bool:
        """Whether cancellation reliably terminates the remote process tree."""
        return False

    async def connect_async(self) -> None:
        """Probe PowerShell Direct connectivity."""
        await self._invoke_async(request={"action": "probe"}, timeout_seconds=self._command_timeout_seconds)

    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None,
        stdout_limit: int,
        stderr_limit: int,
    ) -> GuestExecResult:
        """
        Execute a Windows guest process through PowerShell Direct.

        Returns:
            GuestExecResult: The bounded guest process result.

        Raises:
            HyperVUnsupportedCapabilityError: If an alternate guest user is requested.
        """
        credential = await self._credential_async()
        requested_user = request.user
        if requested_user is not None and requested_user.casefold() != credential["username"].casefold():
            raise HyperVUnsupportedCapabilityError(capability="alternate_user", transport=self.name)
        payload = {
            "action": "exec",
            "argv": list(request.argv) if request.argv is not None else None,
            "shell_script": request.shell_script,
            "stdin_base64": base64.b64encode(request.stdin).decode("ascii") if request.stdin is not None else None,
            "environment": request.environment,
            "cwd": request.cwd,
            "timeout_seconds": timeout_seconds,
        }
        value = await self._invoke_async(
            request=payload,
            timeout_seconds=timeout_seconds + 15,
            cancellation_event=cancellation_event,
            credential=credential,
            max_output_bytes=max(
                self._max_output_bytes,
                ((stdout_limit + stderr_limit) * 4 // 3) + 65_536,
            ),
        )
        stdout = base64.b64decode(str(value.get("stdout_base64", "")))
        stderr = base64.b64decode(str(value.get("stderr_base64", "")))
        exit_code_value = value.get("exit_code")
        exit_code = exit_code_value if isinstance(exit_code_value, int) else None
        return GuestExecResult(
            stdout=stdout[:stdout_limit],
            stderr=stderr[:stderr_limit],
            exit_code=exit_code,
            stdout_truncated=len(stdout) > stdout_limit,
            stderr_truncated=len(stderr) > stderr_limit,
            timed_out=bool(value.get("timed_out")),
        )

    async def read_file_async(self, *, path: str, max_bytes: int) -> bytes:
        """
        Read a binary file through PowerShell Direct.

        Returns:
            bytes: The bounded file data.

        Raises:
            HyperVTransportError: If the file exceeds the read limit.
        """
        value = await self._invoke_async(
            request={"action": "read", "path": path},
            timeout_seconds=self._command_timeout_seconds,
            max_output_bytes=max(self._max_output_bytes, (max_bytes * 4 // 3) + 65_536),
        )
        error_code = value.get("error_code")
        if isinstance(error_code, str):
            raise HyperVTransportError(message="PowerShell Direct guest read failed.", error_code=error_code)
        data = base64.b64decode(str(value.get("data_base64", "")))
        if len(data) > max_bytes:
            raise HyperVTransportError(message="Guest file exceeds the read limit.", error_code="read_limit_exceeded")
        return data

    async def write_file_async(self, *, path: str, data: bytes) -> None:
        """
        Write a binary file through PowerShell Direct.

        Raises:
            HyperVTransportError: If the guest denies the write.
        """
        value = await self._invoke_async(
            request={"action": "write", "path": path, "data_base64": base64.b64encode(data).decode("ascii")},
            timeout_seconds=self._command_timeout_seconds,
        )
        error_code = value.get("error_code")
        if isinstance(error_code, str):
            raise HyperVTransportError(message="PowerShell Direct guest write failed.", error_code=error_code)

    async def _credential_async(self) -> dict[str, str]:
        reference = self._config.credential
        if reference is None:
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_not_configured",
                message="PowerShell Direct requires a credential secret reference.",
            )
        value = dict(await self._secret_resolver.resolve_secret_async(reference))
        if not value.get("username") or not value.get("password"):
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_invalid",
                message="PowerShell Direct credentials require username and password fields.",
            )
        return value

    async def _invoke_async(
        self,
        *,
        request: Mapping[str, object],
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None = None,
        credential: Mapping[str, str] | None = None,
        max_output_bytes: int | None = None,
    ) -> dict[str, object]:
        secret = dict(credential) if credential is not None else await self._credential_async()
        return await self._runner.run_json_async(
            script=_PowerShellDirectScript.SCRIPT,
            payload={
                "vm_name": self._vm_name,
                "username": secret["username"],
                "password": secret["password"],
                "request": dict(request),
            },
            timeout_seconds=timeout_seconds,
            cancellation_event=cancellation_event,
            max_output_bytes=max_output_bytes,
        )


class OpenSSHGuestTransport(HyperVGuestTransport):
    """OpenSSH CLI transport for Linux and other SSH-capable guests."""

    def __init__(
        self,
        *,
        config: HyperVEnvironmentConfig,
        secret_resolver: HyperVSecretResolver,
        executable: str,
        command_timeout_seconds: float,
        max_output_bytes: int,
    ) -> None:
        """Initialize an OpenSSH CLI transport."""
        self._config = config
        self._secret_resolver = secret_resolver
        self._executable = executable
        self._command_timeout_seconds = command_timeout_seconds
        self._max_output_bytes = max_output_bytes

    @property
    def name(self) -> str:
        """The stable transport name."""
        return HyperVGuestTransportKind.SSH.value

    @property
    def remote_process_cleanup_guaranteed(self) -> bool:
        """Whether cancellation reliably terminates the remote process tree."""
        return False

    async def connect_async(self) -> None:
        """
        Probe SSH connectivity with a no-op command.

        Raises:
            HyperVTransportError: If the guest is not reachable.
        """
        result = await self._run_ssh_async(
            remote_command="true",
            stdin=None,
            timeout_seconds=self._command_timeout_seconds,
            cancellation_event=None,
            stdout_limit=1024,
            stderr_limit=4096,
        )
        if result.returncode != 0:
            raise HyperVTransportError(
                message="OpenSSH guest readiness probe failed.", error_code="hyperv_ssh_not_ready"
            )

    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None,
        stdout_limit: int,
        stderr_limit: int,
    ) -> GuestExecResult:
        """
        Execute one POSIX guest command over OpenSSH.

        Returns:
            GuestExecResult: The bounded guest process result.

        Raises:
            HyperVUnsupportedCapabilityError: If an alternate guest user is requested.
        """
        credential = await self._credential_async()
        requested_user = request.user
        if requested_user is not None and requested_user != credential["username"]:
            raise HyperVUnsupportedCapabilityError(capability="alternate_user", transport=self.name)
        command = _posix_command(request=request)
        result = await self._run_ssh_async(
            remote_command=f"sh -lc {shlex.quote(command)}",
            stdin=request.stdin,
            timeout_seconds=timeout_seconds,
            cancellation_event=cancellation_event,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            credential=credential,
        )
        return GuestExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
            timed_out=result.timed_out,
            cancelled=result.cancelled,
        )

    async def read_file_async(self, *, path: str, max_bytes: int) -> bytes:
        """
        Read a binary guest file over OpenSSH.

        Returns:
            bytes: The bounded file data.

        Raises:
            HyperVTransportError: If the read fails or exceeds its limit.
        """
        result = await self._run_ssh_async(
            remote_command=f"cat -- {shlex.quote(path)}",
            stdin=None,
            timeout_seconds=self._command_timeout_seconds,
            cancellation_event=None,
            stdout_limit=max_bytes + 1,
            stderr_limit=4096,
        )
        if result.stdout_truncated or len(result.stdout) > max_bytes:
            raise HyperVTransportError(message="Guest file exceeds the read limit.", error_code="read_limit_exceeded")
        if result.returncode != 0:
            code = (
                "file_not_found"
                if b"No such file" in result.stderr
                else "permission_denied"
                if b"Permission denied" in result.stderr
                else "hyperv_ssh_read_failed"
            )
            raise HyperVTransportError(message="OpenSSH guest file read failed.", error_code=code)
        return result.stdout

    async def write_file_async(self, *, path: str, data: bytes) -> None:
        """
        Write a binary guest file over OpenSSH.

        Raises:
            HyperVTransportError: If the write fails.
        """
        parent = str(PurePosixPath(path).parent)
        command = f"mkdir -p -- {shlex.quote(parent)} && cat > {shlex.quote(path)}"
        result = await self._run_ssh_async(
            remote_command=command,
            stdin=data,
            timeout_seconds=self._command_timeout_seconds,
            cancellation_event=None,
            stdout_limit=1024,
            stderr_limit=4096,
        )
        if result.returncode != 0:
            code = "permission_denied" if b"Permission denied" in result.stderr else "hyperv_ssh_write_failed"
            raise HyperVTransportError(message="OpenSSH guest file write failed.", error_code=code)

    async def _credential_async(self) -> dict[str, str]:
        reference = self._config.credential
        if reference is None:
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_not_configured",
                message="SSH requires a credential secret reference.",
            )
        value = dict(await self._secret_resolver.resolve_secret_async(reference))
        if not value.get("username"):
            raise HyperVPrerequisiteError(
                error_code="hyperv_secret_invalid",
                message="SSH credentials require a username field.",
            )
        return value

    async def _run_ssh_async(
        self,
        *,
        remote_command: str,
        stdin: bytes | None,
        timeout_seconds: float,
        cancellation_event: asyncio.Event | None,
        stdout_limit: int,
        stderr_limit: int,
        credential: Mapping[str, str] | None = None,
    ) -> ExternalCommandResult:
        secret = dict(credential) if credential is not None else await self._credential_async()
        host = self._config.ssh_host
        if host is None:
            raise HyperVPrerequisiteError(
                error_code="hyperv_ssh_host_missing",
                message="SSH transport requires an explicit guest host.",
            )
        argv = [
            self._executable,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=yes",
            "-p",
            str(self._config.ssh_port),
        ]
        identity_file = secret.get("identity_file")
        if identity_file:
            argv.extend(["-i", identity_file])
        argv.extend([f"{secret['username']}@{host}", "--", remote_command])
        return await run_external_command_async(
            argv=argv,
            stdin=stdin,
            timeout_seconds=timeout_seconds,
            cancellation_event=cancellation_event,
            stdout_limit=min(stdout_limit, self._max_output_bytes),
            stderr_limit=min(stderr_limit, self._max_output_bytes),
        )


def _posix_command(request: SandboxExecRequest) -> str:
    """
    Build a safely quoted POSIX command for an SSH guest.

    Returns:
        str: The remote shell command.

    Raises:
        ValueError: If no command is present.
    """
    command = shlex.join(request.argv) if request.argv is not None else request.shell_script
    if command is None:
        raise ValueError("Sandbox request is missing a command.")
    if request.cwd is not None:
        command = f"cd -- {shlex.quote(request.cwd)} && {command}"
    if request.environment:
        assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in request.environment.items())
        command = f"env {assignments} {command}"
    return command


def create_default_guest_transport(
    *,
    vm_name: str,
    config: HyperVEnvironmentConfig,
    power_shell_runner: PowerShellCommandRunner,
    secret_resolver: HyperVSecretResolver,
    power_shell_timeout_seconds: float,
    ssh_executable: str,
    max_output_bytes: int,
) -> HyperVGuestTransport:
    """
    Create the configured built-in guest transport.

    Returns:
        HyperVGuestTransport: PowerShell Direct or OpenSSH CLI transport.
    """
    if config.transport is HyperVGuestTransportKind.POWERSHELL_DIRECT:
        return PowerShellDirectGuestTransport(
            vm_name=vm_name,
            config=config,
            runner=power_shell_runner,
            secret_resolver=secret_resolver,
            command_timeout_seconds=power_shell_timeout_seconds,
            max_output_bytes=max_output_bytes,
        )
    return OpenSSHGuestTransport(
        config=config,
        secret_resolver=secret_resolver,
        executable=ssh_executable,
        command_timeout_seconds=power_shell_timeout_seconds,
        max_output_bytes=max_output_bytes,
    )
