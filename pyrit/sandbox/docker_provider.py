# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Docker/Compose sandbox provider offering real container isolation.

This module shells out to the external ``docker`` and ``docker compose`` command-line
tools using safe argv subprocesses only. It never imports the third-party ``docker``
Python SDK and performs no Docker calls at import time, so importing PyRIT never
requires Docker to be installed or running.

Compose service definitions may be supplied as explicit Compose files, synthesized from
typed Docker service entries (Dockerfile-based or immutable prebuilt-image services,
rendered as temporary Compose JSON using only the standard library ``json``
module), or both. Every resolved definition is validated with
``docker compose config --format json`` and scanned against
``DockerSecurityPolicy`` before any container is created.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import os
import posixpath
import re
import secrets
import shutil
import signal
import stat
import subprocess
import tempfile
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from appdirs import user_state_dir
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from pyrit.executor.capability.models import SandboxOperationEvidence
from pyrit.sandbox.contracts import SandboxEnvironment, SandboxProcess, SandboxProvider, SandboxSession
from pyrit.sandbox.local import SandboxPathEscapeError, SandboxSetupError
from pyrit.sandbox.models import (
    DockerSandboxProviderConfig,
    DockerSecurityPolicy,
    DockerServiceBuildSpec,
    DockerServiceImageSpec,
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
    from asyncio.subprocess import Process
    from collections.abc import Callable, Mapping, Sequence

    from pyrit.executor.capability.evidence import CapabilityEvidenceSink
    from pyrit.models import JSONValue

_ENVIRONMENT_LABEL = "com.pyrit.sandbox.environment"
_RESOURCE_OWNER_LABEL = "com.pyrit.sandbox.owner"
_COMPOSE_PROJECT_LABEL = "com.docker.compose.project"
_COMPOSE_SERVICE_LABEL = "com.docker.compose.service"
_MAX_CLI_OUTPUT_BYTES = 4_194_304
_DEFAULT_IDLE_COMMAND: tuple[str, ...] = ("sleep", "infinity")
_PROJECT_NAME_INVALID_CHARS = re.compile(r"[^a-z0-9_-]")
_OWNERSHIP_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_DANGEROUS_CAPABILITIES = frozenset(
    {
        "ALL",
        "SYS_ADMIN",
        "SYS_PTRACE",
        "SYS_MODULE",
        "SYS_RAWIO",
        "SYS_BOOT",
        "NET_ADMIN",
        "NET_RAW",
        "DAC_READ_SEARCH",
        "DAC_OVERRIDE",
        "MKNOD",
        "BPF",
        "PERFMON",
    }
)
_TRANSIENT_STDERR_MARKERS = (
    "i/o timeout",
    "connection reset by peer",
    "unexpected eof",
    "context deadline exceeded",
    "resource temporarily unavailable",
    "please try again",
    "temporary failure in name resolution",
    "tls handshake timeout",
    "connection refused",
    "broken pipe",
)
_DAEMON_UNAVAILABLE_MARKERS = (
    "cannot connect to the docker daemon",
    "docker daemon is not running",
    "is the docker daemon running",
    "error during connect",
    "dockerdesktoplinuxengine",
    "pipe/docker_engine",
)
_RETRYABLE_LIFECYCLE_OPERATIONS = frozenset({"config", "network_ls", "ps", "volume_ls"})


class DockerSandboxError(RuntimeError):
    """Base error for the Docker/Compose sandbox provider."""

    def __init__(self, *, message: str, error_code: str) -> None:
        """Initialize a typed, actionable Docker sandbox error."""
        self.error_code = error_code
        super().__init__(message)


class DockerCliUnavailableError(DockerSandboxError):
    """The ``docker`` or ``docker compose`` command-line tool is missing or broken."""

    def __init__(self, *, tool: str, detail: str) -> None:
        """Initialize a CLI-unavailable error naming the missing tool."""
        self.tool = tool
        super().__init__(
            message=(
                f"The '{tool}' command-line tool is required by DockerSandboxProvider but was not found or "
                f"failed to run ({detail}). Install Docker Desktop or the Docker Engine CLI together with the "
                "Docker Compose v2 plugin ('docker compose'), and ensure both are on PATH."
            ),
            error_code="docker_cli_unavailable",
        )


class DockerDaemonUnavailableError(DockerSandboxError):
    """The Docker daemon could not be reached."""

    def __init__(self, *, detail: str) -> None:
        """Initialize a daemon-unavailable error with the underlying detail."""
        super().__init__(
            message=(
                "Could not reach the Docker daemon. Start Docker Desktop (or the dockerd service) and confirm "
                f"'docker version' succeeds before using DockerSandboxProvider. Detail: {detail}"
            ),
            error_code="docker_daemon_unavailable",
        )


class DockerComposeConfigError(DockerSandboxError):
    """A Compose source failed to resolve or parse."""

    def __init__(self, *, detail: str) -> None:
        """Initialize a Compose configuration error with the resolution detail."""
        super().__init__(
            message=f"Docker Compose configuration is invalid: {detail}",
            error_code="docker_compose_config_invalid",
        )


class DockerSecurityPolicyViolationError(DockerSandboxError):
    """A resolved Compose service violates the configured security policy."""

    def __init__(self, *, violations: Sequence[str]) -> None:
        """Initialize a security policy violation error naming every offending field."""
        self.violations = tuple(violations)
        joined = "; ".join(violations)
        super().__init__(
            message=(
                f"Docker Compose service definition violates the sandbox security policy: {joined}. Enable the "
                "corresponding DockerSecurityPolicy 'allow_*' flag to opt in explicitly."
            ),
            error_code="docker_security_policy_violation",
        )


class DockerStateSecurityError(DockerSandboxError):
    """Durable Docker sandbox state is not safe to trust."""

    def __init__(self, *, detail: str) -> None:
        """Initialize a fail-closed state security error."""
        super().__init__(
            message=f"Docker sandbox state is not safe to use: {detail}",
            error_code="docker_state_untrusted",
        )


class DockerLifecycleError(DockerSandboxError):
    """A Compose lifecycle operation (build, up, down, ...) failed."""

    def __init__(self, *, operation: str, detail: str) -> None:
        """Initialize a lifecycle error naming the failed operation."""
        self.operation = operation
        super().__init__(
            message=f"Docker Compose '{operation}' failed: {detail}",
            error_code=f"docker_{operation}_failed",
        )


class DockerServiceSelectionError(DockerSandboxError):
    """No Compose service could be resolved for a sandbox environment."""

    def __init__(self, *, environment_name: str, available: Sequence[str]) -> None:
        """Initialize a service selection error listing the available services."""
        self.environment_name = environment_name
        super().__init__(
            message=(
                f"Could not resolve a Docker Compose service for sandbox environment '{environment_name}'. "
                f"Available services: {', '.join(available) or '(none)'}. Name a Compose service after the "
                f"environment, or label it '{_ENVIRONMENT_LABEL}={environment_name}' to bind it explicitly."
            ),
            error_code="docker_service_selection_failed",
        )


class _CliInvocationError(DockerSandboxError):
    """A raw CLI invocation returned a non-zero exit code or timed out."""

    def __init__(
        self, *, argv: Sequence[str], returncode: int | None, stdout: bytes, stderr: bytes, timed_out: bool
    ) -> None:
        """Initialize the internal raw invocation failure carrying captured output."""
        self.argv = tuple(argv)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        detail = "timed out" if timed_out else f"exit code {returncode}"
        super().__init__(
            message=f"Command {' '.join(argv)!r} {detail}. stderr: {_stderr_excerpt(stderr)}",
            error_code="docker_cli_invocation_failed",
        )


@dataclass(frozen=True)
class _CliOutput:
    """Captured, bounded output from one successful CLI invocation."""

    stdout: bytes
    stderr: bytes


@dataclass(frozen=True)
class _RawRun:
    """Mechanical, evidence-free outcome of one bounded container command."""

    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    exit_code: int | None
    timed_out: bool
    cancelled: bool


@dataclass
class _SessionState:
    """One durable, on-disk record for crash and orphan recovery."""

    session_id: str
    task_id: str | None
    created_at: str
    metadata: dict[str, str] = field(default_factory=dict)


async def _run_cli_async(
    *,
    argv: Sequence[str],
    cwd: Path | None,
    timeout_seconds: float,
    input_bytes: bytes | None,
    semaphore: asyncio.Semaphore,
) -> _CliOutput:
    """
    Run one bounded, semaphore-gated CLI invocation to completion.

    Returns:
        _CliOutput: Captured stdout and stderr, bounded to an upper byte limit.

    Raises:
        asyncio.CancelledError: If the caller cancels the invocation after its process starts.
        DockerCliUnavailableError: If the executable could not be found.
        _CliInvocationError: If the command timed out or exited non-zero.
    """
    async with semaphore:
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(cwd) if cwd is not None else None,
                stdin=asyncio.subprocess.PIPE if input_bytes is not None else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )
        except FileNotFoundError as error:
            raise DockerCliUnavailableError(tool=argv[0], detail=str(error)) from error
        communication = asyncio.create_task(process.communicate(input=input_bytes))
        try:
            stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            await _terminate_cli_process_tree_async(process=process)
            stdout, stderr = await communication
            raise _CliInvocationError(
                argv=argv,
                returncode=None,
                stdout=stdout[:_MAX_CLI_OUTPUT_BYTES],
                stderr=stderr[:_MAX_CLI_OUTPUT_BYTES],
                timed_out=True,
            ) from None
        except asyncio.CancelledError:
            await _terminate_cli_process_tree_async(process=process)
            await communication
            raise
    stdout = stdout[:_MAX_CLI_OUTPUT_BYTES]
    stderr = stderr[:_MAX_CLI_OUTPUT_BYTES]
    if process.returncode != 0:
        raise _CliInvocationError(
            argv=argv, returncode=process.returncode, stdout=stdout, stderr=stderr, timed_out=False
        )
    return _CliOutput(stdout=stdout, stderr=stderr)


async def _terminate_cli_process_tree_async(*, process: Process) -> None:
    if process.returncode is not None:
        return
    if os.name != "nt":
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        return
    try:
        terminator = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await terminator.wait()
        if terminator.returncode != 0 and process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
    except OSError:
        with suppress(ProcessLookupError):
            process.kill()


def _is_transient_cli_error(error: BaseException) -> bool:
    """
    Identify infrastructure-level failures safe to retry before any side effect lands.

    Returns:
        bool: True if the failure looks transient (network blip, daemon warm-up, ...).
    """
    if not isinstance(error, _CliInvocationError):
        return False
    if error.timed_out:
        return False
    stderr_text = error.stderr.decode("utf-8", errors="replace").lower()
    return any(marker in stderr_text for marker in _TRANSIENT_STDERR_MARKERS)


def _cli_failure_detail(*, error: _CliInvocationError, timeout_seconds: float) -> str:
    output = _stderr_excerpt(error.stderr) or _stderr_excerpt(error.stdout)
    if error.timed_out:
        prefix = f"timed out after {timeout_seconds:g} seconds"
        return f"{prefix}: {output}" if output else prefix
    return output or f"exit code {error.returncode}"


def _is_daemon_unavailable_error(error: _CliInvocationError) -> bool:
    """
    Identify failures caused by an unreachable Docker daemon.

    Returns:
        bool: True if stderr indicates the daemon is not running or not reachable.
    """
    stderr_text = error.stderr.decode("utf-8", errors="replace").lower()
    return any(marker in stderr_text for marker in _DAEMON_UNAVAILABLE_MARKERS)


def _stderr_excerpt(stderr: bytes) -> str:
    """
    Return a bounded, human-readable stderr excerpt for error messages.

    Returns:
        str: A decoded and length-limited stderr excerpt.
    """
    return stderr.decode("utf-8", errors="replace").strip()[:2000]


def _project_name_for_session(*, prefix: str, session_id: str, attempt_id: str, ownership_id: str) -> str:
    """
    Build a collision-resistant, Compose-legal project name for one attempt.

    Returns:
        str: A project name matching Compose's ``^[a-z0-9][a-z0-9_-]*$`` requirement.
    """
    sanitized = _PROJECT_NAME_INVALID_CHARS.sub("-", session_id.lower()).strip("-_") or "session"
    collision_suffix = _hash_bytes(f"{session_id}\0{attempt_id}\0{ownership_id}".encode())[:12]
    available_session_length = max(1, 63 - len(prefix) - len(collision_suffix) - 2)
    return f"{prefix}-{sanitized[:available_session_length]}-{collision_suffix}"


def _is_provider_project_name(*, project_name: str, prefix: str) -> bool:
    """
    Check whether a durable-state key could have been generated by this provider.

    Returns:
        bool: True for a bounded Compose project name under the configured prefix.
    """
    return (
        0 < len(project_name) <= 63
        and project_name.startswith(f"{prefix}-")
        and _PROJECT_NAME_INVALID_CHARS.search(project_name) is None
    )


def _resolve_build_context(*, build_context: Path, project_context: Path) -> Path:
    """
    Resolve a service build context to an absolute path before synthesis.

    Resolving here (rather than leaving the path relative in the synthesized Compose
    document) preserves the caller's intended relative build context regardless of
    where the temporary Compose file itself is written.

    Returns:
        Path: The absolute, resolved build context directory.
    """
    candidate = build_context if build_context.is_absolute() else project_context / build_context
    return candidate.resolve()


def _synthesize_service_document(
    *,
    spec: DockerServiceBuildSpec | DockerServiceImageSpec,
    project_context: Path,
    policy: DockerSecurityPolicy,
) -> dict[str, Any]:
    """
    Build one synthesized, secure-by-default Compose service definition.

    Returns:
        dict[str, Any]: The JSON-serializable service definition.
    """
    service: dict[str, Any] = {
        "init": True,
        "command": list(spec.command) if spec.command is not None else list(_DEFAULT_IDLE_COMMAND),
        "labels": {**spec.labels, _ENVIRONMENT_LABEL: spec.service_name},
        "security_opt": ["no-new-privileges:true"],
    }
    if isinstance(spec, DockerServiceBuildSpec):
        build: dict[str, Any] = {
            "context": str(_resolve_build_context(build_context=spec.build_context, project_context=project_context)),
            "dockerfile": spec.dockerfile,
        }
        if spec.build_args:
            build["args"] = dict(spec.build_args)
        if spec.target is not None:
            build["target"] = spec.target
        service["build"] = build
        service["image"] = f"pyrit-sandbox/{spec.service_name}:synthesized"
    else:
        service["image"] = spec.image
    if policy.drop_all_capabilities:
        service["cap_drop"] = ["ALL"]
    if policy.read_only_root_filesystem:
        service["read_only"] = True
    if policy.workspace_tmpfs_size_mb is not None:
        service["tmpfs"] = [f"/workspace:rw,nosuid,nodev,size={policy.workspace_tmpfs_size_mb}m,mode=1777"]
    if policy.default_pids_limit is not None:
        service["pids_limit"] = policy.default_pids_limit
    if policy.default_memory_limit is not None:
        service["mem_limit"] = policy.default_memory_limit
    if policy.default_cpus is not None:
        service["cpus"] = policy.default_cpus
    if spec.environment:
        service["environment"] = dict(spec.environment)
    if spec.working_dir is not None:
        service["working_dir"] = spec.working_dir
    if spec.depends_on:
        service["depends_on"] = list(spec.depends_on)
    if policy.isolate_interservice_network:
        service["networks"] = ["pyrit_sandbox"]
    return service


def _synthesize_compose_document(
    *,
    services: Sequence[DockerServiceBuildSpec | DockerServiceImageSpec],
    project_context: Path,
    security_policy: DockerSecurityPolicy,
) -> dict[str, Any]:
    """
    Synthesize a full Compose document (as plain JSON, no third-party YAML library).

    Returns:
        dict[str, Any]: The JSON-serializable synthesized Compose document.
    """
    document: dict[str, Any] = {
        "services": {
            spec.service_name: _synthesize_service_document(
                spec=spec, project_context=project_context, policy=security_policy
            )
            for spec in services
        }
    }
    if security_policy.isolate_interservice_network:
        document["networks"] = {"pyrit_sandbox": {"driver": "bridge", "internal": not security_policy.allow_egress}}
    return document


def _service_network_names(service: Mapping[str, Any]) -> tuple[str, ...]:
    """
    Return the Compose networks used by one service.

    Returns:
        tuple[str, ...]: Referenced top-level network names. An empty tuple means
            networking is explicitly disabled with ``network_mode: none``.
    """
    network_mode = str(service.get("network_mode") or "")
    if network_mode == "none":
        return ()
    raw_networks = service.get("networks")
    if raw_networks is None:
        return ("default",)
    if isinstance(raw_networks, dict):
        return tuple(str(name) for name in raw_networks)
    if isinstance(raw_networks, list):
        return tuple(str(name) for name in raw_networks)
    return ()


def _synthesize_policy_overlay(
    *,
    resolved_document: Mapping[str, Any],
    policy: DockerSecurityPolicy,
    ownership_id: str,
) -> dict[str, Any]:
    """
    Build the final Compose overlay for network and cleanup ownership policy.

    Existing service-to-network relationships are left intact. When egress is
    denied, every provider-managed network is made internal, preserving service
    DNS and connectivity without retaining an external route.

    Returns:
        dict[str, Any]: A JSON-serializable Compose overlay.
    """
    resolved_services = resolved_document.get("services")
    services = resolved_services if isinstance(resolved_services, dict) else {}
    overlay: dict[str, Any] = {
        "services": {str(name): {"labels": {_RESOURCE_OWNER_LABEL: ownership_id}} for name in services}
    }

    resolved_networks = resolved_document.get("networks")
    networks = resolved_networks if isinstance(resolved_networks, dict) else {}
    network_names = {str(name) for name in networks}
    for service in services.values():
        if isinstance(service, dict):
            network_names.update(_service_network_names(service))
    network_overlay: dict[str, Any] = {}
    for name in sorted(network_names):
        definition = networks.get(name)
        if isinstance(definition, dict) and definition.get("external"):
            continue
        network_overlay[name] = {"labels": {_RESOURCE_OWNER_LABEL: ownership_id}}
        if not policy.allow_egress:
            network_overlay[name]["internal"] = True
    if network_overlay:
        overlay["networks"] = network_overlay

    resolved_volumes = resolved_document.get("volumes")
    volumes = resolved_volumes if isinstance(resolved_volumes, dict) else {}
    volume_overlay = {
        str(name): {"labels": {_RESOURCE_OWNER_LABEL: ownership_id}}
        for name, definition in volumes.items()
        if not isinstance(definition, dict) or not definition.get("external")
    }
    if volume_overlay:
        overlay["volumes"] = volume_overlay
    return overlay


def _write_json_file(*, path: Path, document: dict[str, Any]) -> None:
    """Write a JSON document to disk (invoked only via ``asyncio.to_thread``)."""
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def _service_labels(service: Mapping[str, Any]) -> dict[str, str]:
    """
    Return a service's labels as a plain string-to-string mapping.

    Handles both the dict form and the ``"key=value"`` list form that different
    Compose versions may resolve labels to.

    Returns:
        dict[str, str]: The service's labels.
    """
    raw = service.get("labels")
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    labels: dict[str, str] = {}
    for entry in raw:
        key, _, value = str(entry).partition("=")
        labels[key] = value
    return labels


def _select_service_name(
    *, resolved_services: Mapping[str, Any], environment_name: str, service_order: Sequence[str]
) -> str:
    """
    Deterministically select a Compose service for one sandbox environment.

    Selection precedence: (1) an exact name match, (2) a service labeled
    ``com.pyrit.sandbox.environment`` with the environment's name, (3) the first
    service in resolved declaration order.

    Returns:
        str: The selected Compose service name.

    Raises:
        DockerServiceSelectionError: If no service could be resolved.
    """
    if environment_name in resolved_services:
        return environment_name
    for name in service_order:
        if _service_labels(resolved_services[name]).get(_ENVIRONMENT_LABEL) == environment_name:
            return name
    if service_order:
        return service_order[0]
    raise DockerServiceSelectionError(environment_name=environment_name, available=tuple(service_order))


def _is_docker_socket_source(source: str) -> bool:
    """
    Identify whether a bind-mount source is the Docker control socket.

    Returns:
        bool: True if the source targets the Docker daemon socket or named pipe.
    """
    normalized = source.replace("\\", "/").lower()
    return normalized.endswith("docker.sock") or "pipe/docker_engine" in normalized


def _scan_security_violations(*, resolved_document: Mapping[str, Any], policy: DockerSecurityPolicy) -> list[str]:
    """
    Scan a resolved Compose document for security-policy violations.

    Returns:
        list[str]: Human-readable descriptions of every violation found.
    """
    violations: list[str] = []
    services: Mapping[str, Any] = resolved_document.get("services", {})
    for name, service in services.items():
        violations.extend(_scan_service_violations(name=name, service=service, policy=policy))
    if not policy.allow_egress:
        networks: Mapping[str, Any] = resolved_document.get("networks", {})
        for name, service in services.items():
            network_mode = str(service.get("network_mode") or "")
            if network_mode and network_mode != "none":
                violations.append(
                    f"service '{name}' uses network_mode: {network_mode}, which bypasses internal Compose networks"
                )
                continue
            for network_name in _service_network_names(service):
                definition = networks.get(network_name)
                if not isinstance(definition, dict):
                    violations.append(
                        f"service '{name}' uses network '{network_name}' without an internal network definition"
                    )
                elif definition.get("external"):
                    violations.append(f"service '{name}' uses external network '{network_name}'")
                elif definition.get("internal") is not True:
                    violations.append(f"service '{name}' uses network '{network_name}' with internal: false")
    secrets: Mapping[str, Any] = resolved_document.get("secrets", {})
    if not policy.allow_unrestricted_secrets:
        for secret_name, secret_def in secrets.items():
            if isinstance(secret_def, dict) and "environment" in secret_def:
                violations.append(f"secret '{secret_name}' is sourced from a plain environment variable")
    return violations


def _scan_service_violations(*, name: str, service: Mapping[str, Any], policy: DockerSecurityPolicy) -> list[str]:
    """
    Scan one resolved Compose service definition for security-policy violations.

    Returns:
        list[str]: Human-readable descriptions of every violation found.
    """
    violations: list[str] = []
    if not policy.allow_privileged and service.get("privileged"):
        violations.append(f"service '{name}' sets privileged: true")
    if not policy.allow_host_namespaces:
        for namespace_key in ("network_mode", "pid", "ipc"):
            namespace_value = str(service.get(namespace_key) or "")
            if namespace_value == "host" or namespace_value.startswith(("container:", "service:")):
                violations.append(f"service '{name}' uses {namespace_key}: {namespace_value}")
        if service.get("userns_mode") == "host":
            violations.append(f"service '{name}' uses userns_mode: host")
    for volume in service.get("volumes") or ():
        if not isinstance(volume, dict) or volume.get("type") != "bind":
            continue
        source = str(volume.get("source") or "")
        if _is_docker_socket_source(source) and not policy.allow_docker_socket_mount:
            violations.append(f"service '{name}' mounts the Docker socket ('{source}')")
        elif not policy.allow_bind_mounts:
            violations.append(f"service '{name}' bind-mounts host path '{source}'")
    if not policy.allow_device_mounts:
        for device in service.get("devices") or ():
            source = device.get("source") if isinstance(device, dict) else device
            violations.append(f"service '{name}' mounts host device '{source}'")
    ports = service.get("ports") or ()
    if not policy.allow_published_ports and ports:
        violations.append(f"service '{name}' publishes {len(ports)} host port(s)")
    cap_add = {str(capability).upper().removeprefix("CAP_") for capability in (service.get("cap_add") or ())}
    if not policy.allow_dangerous_capabilities and cap_add & _DANGEROUS_CAPABILITIES:
        violations.append(f"service '{name}' adds capabilities {sorted(cap_add)}")
    security_opt = [str(option) for option in (service.get("security_opt") or ())]
    if not policy.allow_unconfined_seccomp and any("seccomp:unconfined" in option for option in security_opt):
        violations.append(f"service '{name}' disables seccomp confinement")
    return violations


def _lock_file(handle: Any) -> None:
    """Acquire an OS-level advisory exclusive lock on an open file handle."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_file(handle: Any) -> None:
    """Release an OS-level advisory lock on an open file handle."""
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _validate_state_owner(*, metadata: os.stat_result, description: str) -> None:
    """
    Reject a POSIX state object owned by another account.

    Raises:
        DockerStateSecurityError: If the state object belongs to another user.
    """
    get_effective_user_id = getattr(os, "geteuid", None)
    if get_effective_user_id is not None and metadata.st_uid != get_effective_user_id():
        raise DockerStateSecurityError(detail=f"{description} is owned by another user")


def _ensure_private_state_directory(path: Path) -> None:
    """
    Create or validate a private, non-symlink state directory.

    Raises:
        DockerStateSecurityError: If the directory cannot be trusted.
    """
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise DockerStateSecurityError(detail=f"state directory '{path}' is not a real directory")
    _validate_state_owner(metadata=metadata, description=f"state directory '{path}'")
    if os.name != "nt":
        path.chmod(0o700)


def _open_state_file(path: Path) -> TextIO:
    """
    Open a private regular state file without following POSIX symlinks.

    Returns:
        TextIO: The securely opened state file.

    Raises:
        DockerStateSecurityError: If the file cannot be trusted or opened securely.
    """
    _ensure_private_state_directory(path.parent)
    if path.is_symlink():
        raise DockerStateSecurityError(detail=f"state file '{path}' is a symbolic link")
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise DockerStateSecurityError(detail=f"state file '{path}' could not be opened securely ({error})") from error
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DockerStateSecurityError(detail=f"state file '{path}' is not a regular file")
        _validate_state_owner(metadata=metadata, description=f"state file '{path}'")
        if os.name != "nt":
            os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "r+", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def _decode_state(raw: str) -> dict[str, Any]:
    """
    Decode and validate the durable state document.

    Returns:
        dict[str, Any]: The decoded state mapping.

    Raises:
        DockerStateSecurityError: If the state root is not an object.
    """
    decoded = json.loads(raw) if raw.strip() else {}
    if not isinstance(decoded, dict):
        raise DockerStateSecurityError(detail="state document root is not an object")
    return decoded


def _read_state_file(path: Path) -> dict[str, Any]:
    """
    Read the durable provider state file under a cross-process advisory lock.

    Invoked only via ``asyncio.to_thread``.

    Returns:
        dict[str, Any]: The current project-name-to-record state mapping.
    """
    with _open_state_file(path) as handle:
        _lock_file(handle)
        try:
            handle.seek(0)
            raw = handle.read()
        finally:
            _unlock_file(handle)
    return _decode_state(raw)


def _update_state_file(path: Path, mutate: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
    """
    Read-modify-write the durable provider state file under a cross-process lock.

    Invoked only via ``asyncio.to_thread``.
    """
    with _open_state_file(path) as handle:
        _lock_file(handle)
        try:
            handle.seek(0)
            raw = handle.read()
            state = _decode_state(raw)
            updated = mutate(state)
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps(updated, indent=2))
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            _unlock_file(handle)


def _parse_docker_labels(raw: str) -> dict[str, str]:
    """
    Parse the comma-joined label string emitted by ``docker ps --format {{json .}}``.

    Returns:
        dict[str, str]: The parsed label mapping.
    """
    labels: dict[str, str] = {}
    for entry in raw.split(","):
        if not entry:
            continue
        key, _, value = entry.partition("=")
        labels[key] = value
    return labels


def _parse_ndjson(data: bytes) -> list[dict[str, Any]]:
    """
    Parse newline-delimited JSON records, as emitted by ``docker ps --format {{json .}}``.

    Returns:
        list[dict[str, Any]]: One dict per non-empty line.
    """
    records: list[dict[str, Any]] = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped:
            records.append(json.loads(stripped))
    return records


def _is_container_running(record: Mapping[str, Any]) -> bool:
    """
    Determine whether a ``docker ps`` record describes a running container.

    Returns:
        bool: True if the container's reported state is "running".
    """
    return str(record.get("State", "")).lower() == "running"


def _process_is_alive(process_id: int) -> bool:
    """
    Conservatively determine whether a durable-state owner may still be alive.

    Permission errors and PID reuse fail closed by retaining the state record.

    Returns:
        bool: True when cleanup must not treat the owner as crashed.
    """
    if process_id <= 0:
        return True
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        invalid_parameter = 87
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(process_query_limited_information, False, process_id)
        if not handle:
            return ctypes.get_last_error() != invalid_parameter
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _container_health_state(record: Mapping[str, Any]) -> str | None:
    """
    Extract Compose's human-readable container health state when one is present.

    Returns:
        str | None: ``healthy``, ``unhealthy``, ``starting``, or None for services
            without a health check.
    """
    status = str(record.get("Status", "")).lower()
    if "(healthy)" in status:
        return "healthy"
    if "(unhealthy)" in status:
        return "unhealthy"
    if "(health: starting)" in status:
        return "starting"
    return None


def _is_container_ready(record: Mapping[str, Any]) -> bool:
    """
    Determine whether a container is running and, when configured, healthy.

    Returns:
        bool: True when the service is ready for setup operations.
    """
    return _is_container_running(record) and _container_health_state(record) not in {"starting", "unhealthy"}


def _remaining_seconds(deadline: float) -> float:
    """
    Return non-negative time remaining before a monotonic deadline.

    Returns:
        float: Remaining seconds.
    """
    return max(0.0, deadline - asyncio.get_running_loop().time())


async def _read_limited_async(*, stream: Any, limit: int) -> tuple[bytes, bool]:
    """
    Drain a process stream while retaining at most the configured byte limit.

    Returns:
        tuple[bytes, bool]: Retained bytes and whether output was truncated.
    """
    if stream is None:
        return b"", False
    retained = bytearray()
    truncated = False
    while chunk := await stream.read(65_536):
        remaining = limit - len(retained)
        if remaining > 0:
            retained.extend(chunk[:remaining])
        if len(chunk) > remaining:
            truncated = True
    return bytes(retained), truncated


async def _finish_streams_async(
    *,
    stdout_task: asyncio.Task[tuple[bytes, bool]],
    stderr_task: asyncio.Task[tuple[bytes, bool]],
    grace_seconds: float,
) -> tuple[tuple[bytes, bool], tuple[bytes, bool]]:
    """
    Give in-flight stream readers a short grace window after termination.

    Returns:
        tuple[tuple[bytes, bool], tuple[bytes, bool]]: Best-effort stdout and stderr results.
    """
    try:
        return await asyncio.wait_for(asyncio.gather(stdout_task, stderr_task), timeout=max(1.0, grace_seconds))
    except asyncio.TimeoutError:
        stdout_task.cancel()
        stderr_task.cancel()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        return (b"", True), (b"", True)


async def _write_stdin_async(process: Process, data: bytes | None) -> None:
    """Write optional stdin bytes and close the stream."""
    if process.stdin is None:
        return
    if data:
        process.stdin.write(data)
        with suppress(BrokenPipeError, ConnectionResetError):
            await process.stdin.drain()
    process.stdin.close()


def _hash_bytes(data: bytes) -> str:
    """
    Return a SHA-256 digest.

    Returns:
        str: The hex-encoded digest.
    """
    return hashlib.sha256(data).hexdigest()


def _now() -> datetime:
    """
    Return an aware UTC timestamp.

    Returns:
        datetime: The current UTC time.
    """
    return datetime.now(tz=timezone.utc)


def _evidence(
    *,
    provider: str,
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
    Build a content-free authoritative sandbox evidence event.

    Returns:
        SandboxOperationEvidence: The immutable evidence event.
    """
    return SandboxOperationEvidence(
        provider=provider,
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
    """Emit evidence when a sink was configured."""
    if sink is not None:
        await sink.emit_async(evidence)


async def _spawn_container_process_async(
    *,
    docker_executable: str,
    argv: Sequence[str],
    semaphore: asyncio.Semaphore,
    needs_stdin: bool,
) -> tuple[Process, Callable[[], None]]:
    """
    Spawn one host-side ``docker exec`` client process under the shared CLI semaphore.

    The semaphore is held for the process's entire lifetime (spawn through completion),
    so the caller must invoke the returned release callback exactly once when done.

    Returns:
        tuple[Process, Callable[[], None]]: The spawned process and an idempotent
        release callback for the acquired semaphore slot.

    Raises:
        DockerCliUnavailableError: If the executable could not be found.
    """
    await semaphore.acquire()
    released = False

    def release() -> None:
        nonlocal released
        if not released:
            released = True
            semaphore.release()

    try:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE if needs_stdin else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as error:
        release()
        raise DockerCliUnavailableError(tool=docker_executable, detail=str(error)) from error
    except BaseException:
        release()
        raise
    return process, release


class DockerSandboxProcess(SandboxProcess):
    """
    A host-side ``docker exec`` client process wrapping a live in-container command.

    Only the host-side client process is tracked here; the workload itself runs inside
    the container. Killing this client process stops PyRIT's view of the command but
    does not, by itself, guarantee the in-container process stops (a known Docker CLI
    limitation). When the container has the ``timeout`` coreutil available, the exec
    layer wraps commands with an in-container ``timeout`` for a genuine kill guarantee;
    otherwise the container's eventual removal at session close is the outer bound.
    """

    def __init__(
        self,
        *,
        process: Process,
        environment: DockerSandboxEnvironment,
        started_at: datetime,
        release_semaphore: Callable[[], None],
        timeout_seconds: float,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        request: SandboxExecRequest | None = None,
        operation_context: SandboxOperationContext | None = None,
        container_timeout_wrapped: bool = False,
    ) -> None:
        """Initialize a Docker exec process handle."""
        self._process = process
        self._environment = environment
        self._started_at = started_at
        self._release_semaphore = release_semaphore
        self._timeout_seconds = timeout_seconds
        self._stdout_limit = stdout_limit or environment._spec.limits.max_stdout_bytes
        self._stderr_limit = stderr_limit or environment._spec.limits.max_stderr_bytes
        self._request = request
        self._operation_context = operation_context
        self._container_timeout_wrapped = container_timeout_wrapped
        self._termination_lock = asyncio.Lock()
        self._termination_task: asyncio.Task[None] | None = None

    async def communicate_async(
        self,
        *,
        stdin: bytes | None = None,
        cancellation_event: asyncio.Event | None = None,
    ) -> SandboxExecResult:
        """
        Wait for completion while bounding output and honoring cancellation.

        Returns:
            SandboxExecResult: Buffered output and terminal status.

        Raises:
            RuntimeError: If this handle was not created for an ``exec`` request.
            asyncio.CancelledError: If the calling task is cancelled.
        """
        if self._request is None:
            raise RuntimeError("DockerSandboxProcess.communicate_async requires a bound exec request.")
        try:
            raw = await self._communicate_raw_async(stdin=stdin, cancellation_event=cancellation_event)
        except asyncio.CancelledError:
            evidence = self._environment._operation_evidence(
                operation="exec",
                outcome=SandboxOperationStatus.CANCELLED,
                started_at=self._started_at,
                operation_context=self._operation_context,
                error_code="task_cancelled",
            )
            await asyncio.shield(_emit_evidence_async(sink=self._environment._evidence_sink, evidence=evidence))
            raise
        if self._container_timeout_wrapped and not raw.cancelled and raw.exit_code in {124, 137, 143}:
            raw = replace(raw, timed_out=True)
        return await self._environment._finalize_exec_result_async(
            request=self._request,
            started_at=self._started_at,
            operation_context=self._operation_context,
            raw=raw,
        )

    async def terminate_async(self) -> None:
        """Terminate the host-side exec client exactly once."""
        async with self._termination_lock:
            if self._termination_task is None:
                self._termination_task = asyncio.create_task(self._terminate_once_async())
            termination_task = self._termination_task
        await asyncio.shield(termination_task)

    async def _terminate_once_async(self) -> None:
        with suppress(ProcessLookupError):
            self._process.kill()
        await self._process.wait()

    async def _communicate_raw_async(
        self,
        *,
        stdin: bytes | None,
        cancellation_event: asyncio.Event | None,
    ) -> _RawRun:
        """
        Wait for the exec client to finish, bounding output and honoring cancellation.

        Returns:
            _RawRun: Mechanical, evidence-free process outcome.
        """
        deadline = asyncio.get_running_loop().time() + self._timeout_seconds
        stdout_task = asyncio.create_task(_read_limited_async(stream=self._process.stdout, limit=self._stdout_limit))
        stderr_task = asyncio.create_task(_read_limited_async(stream=self._process.stderr, limit=self._stderr_limit))
        stdin_task = asyncio.create_task(_write_stdin_async(self._process, stdin))
        wait_task = asyncio.create_task(self._process.wait())
        cancellation_task = asyncio.create_task(cancellation_event.wait()) if cancellation_event is not None else None
        timed_out = False
        cancelled = False
        stdout: tuple[bytes, bool] = (b"", False)
        stderr: tuple[bytes, bool] = (b"", False)
        try:
            waiting = (wait_task,) if cancellation_task is None else (wait_task, cancellation_task)
            done, _ = await asyncio.wait(
                waiting, timeout=_remaining_seconds(deadline), return_when=asyncio.FIRST_COMPLETED
            )
            timed_out = not done
            cancelled = cancellation_task is not None and cancellation_task in done
            if not timed_out and not cancelled:
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.gather(stdout_task, stderr_task), timeout=_remaining_seconds(deadline)
                    )
                    await asyncio.wait_for(stdin_task, timeout=_remaining_seconds(deadline))
                except asyncio.TimeoutError:
                    timed_out = True
            if timed_out or cancelled:
                await self.terminate_async()
                stdin_task.cancel()
                stdout, stderr = await _finish_streams_async(
                    stdout_task=stdout_task,
                    stderr_task=stderr_task,
                    grace_seconds=self._environment._spec.limits.terminate_grace_seconds,
                )
        except BaseException:
            await asyncio.shield(self.terminate_async())
            raise
        finally:
            if cancellation_task is not None:
                cancellation_task.cancel()
            for task in (stdin_task, wait_task, stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdin_task, wait_task, stdout_task, stderr_task, return_exceptions=True)
            self._release_semaphore()
            await self._environment.remove_process_async(self)
        stdout_bytes, stdout_truncated = stdout
        stderr_bytes, stderr_truncated = stderr
        return _RawRun(
            stdout=stdout_bytes,
            stderr=stderr_bytes,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
            exit_code=self._process.returncode,
            timed_out=timed_out,
            cancelled=cancelled,
        )


def _wrap_with_timeout(*, command: Sequence[str], budget_seconds: float, grace_seconds: float) -> tuple[str, ...]:
    """
    Wrap a container command with an in-container ``timeout`` for a genuine kill guarantee.

    Killing the host-side ``docker exec`` client does not reliably terminate the
    in-container process. When the container has the ``timeout`` coreutil available,
    wrapping the command lets the container itself enforce the deadline.

    Returns:
        tuple[str, ...]: The wrapped command.
    """
    budget = format(budget_seconds, ".6g")
    grace = format(grace_seconds, ".6g")
    return ("timeout", "-k", grace, budget, *command)


def _classify_start_error(error: BaseException) -> tuple[SandboxOperationStatus, str, str]:
    """
    Map process-start failures to a stable content-free status, code, and message.

    Returns:
        tuple[SandboxOperationStatus, str, str]: Status, error code, and error message.
    """
    if isinstance(error, DockerCliUnavailableError):
        return SandboxOperationStatus.FAILED, "docker_cli_unavailable", str(error)
    if isinstance(error, SandboxPathEscapeError):
        return SandboxOperationStatus.PATH_ESCAPE, "path_escape", "Requested path is outside the sandbox environment."
    if isinstance(error, DockerServiceSelectionError):
        return SandboxOperationStatus.FAILED, "docker_service_selection_failed", str(error)
    if isinstance(error, ValueError):
        return (
            SandboxOperationStatus.FAILED,
            "unsupported_process_option",
            "The requested process option is not supported.",
        )
    if isinstance(error, RuntimeError):
        return SandboxOperationStatus.FAILED, "environment_closed", "Sandbox environment is closed."
    return SandboxOperationStatus.FAILED, "process_start_failed", "The process could not be started."


class DockerSandboxEnvironment(SandboxEnvironment):
    """
    A named execution surface bound to one running Compose service container.

    Unlike ``LocalSandboxEnvironment``, this environment IS a
    security boundary: commands run inside an isolated container subject to the
    provider's ``DockerSecurityPolicy``. Path containment
    checks below operate purely lexically against the container's declared root (no
    host filesystem is touched) and are a defense-in-depth measure layered on top of
    Docker's own isolation, not the sole protection as in the local provider.
    """

    def __init__(
        self,
        *,
        session: DockerSandboxSession,
        spec: SandboxEnvironmentSpec,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> None:
        """Initialize a Docker-backed environment (container binding is deferred)."""
        self._session = session
        self._spec = spec
        self._evidence_sink = evidence_sink
        self._container_root = str(spec.metadata.get("docker_workdir", "/workspace"))
        self._container_id: str | None = None
        self._service_name: str | None = None
        self._timeout_available = False
        self._operation_lock = asyncio.Lock()
        self._closed = False
        self._process_lock = asyncio.Lock()
        self._processes: set[DockerSandboxProcess] = set()

    @property
    def name(self) -> str:
        """The environment name."""
        return self._spec.name

    @property
    def connection_info(self) -> SandboxConnectionInfo:
        """Non-secret Docker connection metadata."""
        return SandboxConnectionInfo(
            provider="docker",
            session_id=self._session.session_id,
            environment_name=self.name,
            transport="docker-exec",
            endpoint=self._container_id,
            metadata={
                "security_boundary": True,
                "compose_service": self._service_name,
                "compose_project": self._session.project_name,
            },
        )

    def bind_container(self, *, container_id: str, service_name: str) -> None:
        """Bind this environment to its resolved running container."""
        self._container_id = container_id
        self._service_name = service_name

    async def initialize_async(self) -> None:
        """
        Probe container capabilities, materialize setup files, and run setup scripts.

        Raises:
            SandboxSetupError: If a setup file or command fails.
        """
        started_at = _now()
        self._timeout_available = await self._probe_timeout_available_async()
        await self._exec_container_command_async(("mkdir", "-p", self._container_root))
        if self._session._provider._config.security_policy.require_secure_file_operations:
            try:
                await self._exec_container_command_async(("python", "-c", "import os, stat"))
            except SandboxSetupError as error:
                raise SandboxSetupError(
                    "Secure Docker file operations require a Python runtime in the isolated service image."
                ) from error
        for setup_file in self._spec.setup_files:
            result = await self.write_file_async(path=setup_file.path, data=setup_file.content)
            if result.status is not SandboxOperationStatus.SUCCEEDED:
                raise SandboxSetupError(f"A setup file failed with status '{result.status.value}'.")
            if setup_file.executable:
                container_path = self._resolve_container_path(setup_file.path)
                await self._exec_container_command_async(("chmod", "+x", container_path))
        for setup_script in self._spec.setup_scripts:
            result = await self.exec_async(request=setup_script.request)
            if result.status is not SandboxOperationStatus.SUCCEEDED:
                raise SandboxSetupError(
                    f"Setup command failed with status '{result.status.value}' and exit code {result.exit_code}."
                )
        evidence = self._operation_evidence(
            operation="environment_setup",
            outcome=SandboxOperationStatus.SUCCEEDED,
            started_at=started_at,
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)

    async def _probe_timeout_available_async(self) -> bool:
        """
        Probe whether the container's shell has the ``timeout`` coreutil available.

        Returns:
            bool: True if ``timeout`` can be used to enforce in-container deadlines.
        """
        argv = self._container_exec_argv(("sh", "-c", "command -v timeout >/dev/null 2>&1"))
        semaphore = self._session._provider._cli_semaphore
        try:
            process, release = await _spawn_container_process_async(
                docker_executable=self._session._provider._config.docker_executable,
                argv=argv,
                semaphore=semaphore,
                needs_stdin=False,
            )
        except DockerCliUnavailableError:
            return False
        try:
            await asyncio.wait_for(process.communicate(), timeout=10.0)
            return process.returncode == 0
        except asyncio.TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
            return False
        finally:
            release()

    async def _exec_container_command_async(self, command: Sequence[str]) -> None:
        """
        Run a fixed, trusted setup command inside the container, raising on failure.

        Raises:
            SandboxSetupError: If the container command exits non-zero.
        """
        argv = self._container_exec_argv(command)
        semaphore = self._session._provider._cli_semaphore
        process, release = await _spawn_container_process_async(
            docker_executable=self._session._provider._config.docker_executable,
            argv=argv,
            semaphore=semaphore,
            needs_stdin=False,
        )
        try:
            _, stderr = await process.communicate()
        finally:
            release()
        if process.returncode != 0:
            raise SandboxSetupError(f"Container setup command {command!r} failed: {_stderr_excerpt(stderr)}")

    def _container_exec_argv(self, command: Sequence[str]) -> tuple[str, ...]:
        if self._container_id is None:
            raise RuntimeError(f"Sandbox environment '{self.name}' is not bound to a running container.")
        docker_executable = self._session._provider._config.docker_executable
        return (docker_executable, "exec", "-i", self._container_id, *command)

    def _resolve_container_path(self, path: str) -> str:
        """
        Resolve a requested path to an absolute, contained container path.

        Purely lexical (``posixpath``); no host filesystem access is possible or
        attempted here, since the container filesystem is remote.

        Returns:
            str: The absolute, contained container path.

        Raises:
            SandboxPathEscapeError: If the path leaves the container root, or is
                absolute without ``allow_absolute_container_paths``.
        """
        policy = self._session._provider._config.security_policy
        if posixpath.isabs(path):
            if not policy.allow_absolute_container_paths:
                raise SandboxPathEscapeError(
                    f"Absolute container path '{path}' requires allow_absolute_container_paths=True."
                )
            candidate = posixpath.normpath(path)
        else:
            candidate = posixpath.normpath(posixpath.join(self._container_root, path))
        root = posixpath.normpath(self._container_root)
        if candidate != root and not candidate.startswith(root + "/"):
            raise SandboxPathEscapeError(f"Path '{path}' leaves sandbox environment '{self.name}'.")
        return candidate

    def _resolve_cwd(self, cwd: str | None) -> str:
        return self._container_root if cwd is None else self._resolve_container_path(cwd)

    @staticmethod
    def _command(request: SandboxExecRequest) -> tuple[str, ...]:
        if request.argv is not None:
            return request.argv
        if request.shell_script is None:
            raise ValueError("Shell-script request is missing script content.")
        return ("sh", "-c", request.shell_script)

    async def start_process_async(
        self,
        *,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxProcess:
        """
        Start a ``docker exec`` process for a buffered request.

        Returns:
            SandboxProcess: The running process handle.

        Raises:
            RuntimeError: If the environment is closed or not yet bound to a container.
            SandboxPathEscapeError: If the requested cwd leaves the environment root.
        """
        async with self._operation_lock:
            if self._closed:
                raise RuntimeError("Sandbox environment is closed.")
            return await self._start_process_unlocked_async(request=request, operation_context=operation_context)

    async def _start_process_unlocked_async(
        self,
        *,
        request: SandboxExecRequest,
        operation_context: SandboxOperationContext | None,
    ) -> DockerSandboxProcess:
        """
        Start and register a container process while holding the operation gate.

        Returns:
            DockerSandboxProcess: The running process handle.
        """
        cwd = self._resolve_cwd(request.cwd)
        command = list(self._command(request))
        budget = min(request.timeout_seconds or self._spec.limits.max_exec_seconds, self._spec.limits.max_exec_seconds)
        if self._timeout_available:
            command = list(
                _wrap_with_timeout(
                    command=command,
                    budget_seconds=budget,
                    grace_seconds=self._spec.limits.terminate_grace_seconds,
                )
            )
        argv = [
            self._session._provider._config.docker_executable,
            "exec",
            "-i",
        ]
        if request.user is not None:
            argv.extend(["-u", request.user])
        argv.extend(["-w", cwd])
        for key, value in request.environment.items():
            argv.extend(["-e", f"{key}={value}"])
        argv.append(self._require_container_id())
        argv.extend(command)
        process, release = await _spawn_container_process_async(
            docker_executable=self._session._provider._config.docker_executable,
            argv=argv,
            semaphore=self._session._provider._cli_semaphore,
            needs_stdin=True,
        )
        process_handle = DockerSandboxProcess(
            process=process,
            environment=self,
            started_at=_now(),
            release_semaphore=release,
            timeout_seconds=budget + self._spec.limits.terminate_grace_seconds,
            request=request,
            operation_context=operation_context,
            container_timeout_wrapped=self._timeout_available,
        )
        async with self._process_lock:
            self._processes.add(process_handle)
        return process_handle

    def _require_container_id(self) -> str:
        if self._container_id is None:
            raise RuntimeError(f"Sandbox environment '{self.name}' is not bound to a running container.")
        return self._container_id

    async def exec_async(
        self,
        *,
        request: SandboxExecRequest,
        cancellation_event: asyncio.Event | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxExecResult:
        """
        Execute a bounded ``docker exec`` process.

        Returns:
            SandboxExecResult: Buffered output and terminal status.
        """
        started_at = _now()
        try:
            process = await self.start_process_async(request=request, operation_context=operation_context)
        except (DockerCliUnavailableError, SandboxPathEscapeError, RuntimeError, ValueError) as error:
            status, error_code, error_message = _classify_start_error(error)
            evidence = self._operation_evidence(
                operation="exec",
                outcome=status,
                started_at=started_at,
                operation_context=operation_context,
                error_code=error_code,
            )
            await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
            return SandboxExecResult(
                status=status,
                error_code=error_code,
                error_message=error_message,
                evidence=(evidence,),
            )
        return await process.communicate_async(stdin=request.stdin, cancellation_event=cancellation_event)

    async def _finalize_exec_result_async(
        self,
        *,
        request: SandboxExecRequest,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
        raw: _RawRun,
    ) -> SandboxExecResult:
        """
        Map a raw container run onto a typed, evidence-backed exec result.

        Returns:
            SandboxExecResult: Buffered output and terminal status.
        """
        if raw.cancelled:
            status = SandboxOperationStatus.CANCELLED
            error_code = "cancelled"
        elif raw.timed_out:
            status = SandboxOperationStatus.TIMED_OUT
            error_code = "timeout"
        elif raw.stdout_truncated or raw.stderr_truncated:
            status = SandboxOperationStatus.TRUNCATED
            error_code = "output_limit_exceeded"
        else:
            status = SandboxOperationStatus.SUCCEEDED if raw.exit_code == 0 else SandboxOperationStatus.FAILED
            error_code = None if status is SandboxOperationStatus.SUCCEEDED else "nonzero_exit"
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
                "command_form": "argv" if request.argv is not None else "shell_script",
                "compose_service": self._service_name,
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

    async def read_file_async(
        self,
        *,
        path: str,
        max_bytes: int | None = None,
        operation_context: SandboxOperationContext | None = None,
    ) -> SandboxReadResult:
        """
        Read a bounded binary file from the container via ``docker exec``.

        Returns:
            SandboxReadResult: File data or an explicit error status.
        """
        async with self._operation_lock:
            if self._closed:
                return await self._read_error_async(
                    status=SandboxOperationStatus.FAILED,
                    path=path,
                    started_at=_now(),
                    operation_context=operation_context,
                    error_code="environment_closed",
                    error_message="Sandbox environment is closed.",
                )
            return await self._read_file_unlocked_async(
                path=path, max_bytes=max_bytes, operation_context=operation_context
            )

    async def _read_file_unlocked_async(
        self,
        *,
        path: str,
        max_bytes: int | None,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxReadResult:
        started_at = _now()
        if max_bytes is not None and max_bytes <= 0:
            return await self._read_error_async(
                status=SandboxOperationStatus.FAILED,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="invalid_read_limit",
                error_message="Read limit must be greater than zero.",
            )
        limit = min(max_bytes or self._spec.limits.max_read_bytes, self._spec.limits.max_read_bytes)
        try:
            container_path = self._resolve_container_path(path)
        except SandboxPathEscapeError:
            return await self._read_error_async(
                status=SandboxOperationStatus.PATH_ESCAPE,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="path_escape",
                error_message="Requested path is outside the sandbox environment.",
            )
        relative_path = posixpath.relpath(container_path, self._container_root)
        policy = self._session._provider._config.security_policy
        command = (
            ("python", "-c", _SECURE_CONTAINER_READ_SCRIPT, self._container_root, relative_path)
            if policy.require_secure_file_operations
            else (
                "sh",
                "-c",
                (
                    'root=$(readlink -f -- "$1") || exit 1; '
                    'target=$(readlink -f -- "$2") || exit 1; '
                    'case "$target" in "$root"/*) exec cat -- "$target" ;; '
                    '*) printf "pyrit_path_escape\\n" >&2; exit 1 ;; esac'
                ),
                "sh",
                self._container_root,
                container_path,
            )
        )
        raw, release_error = await self._run_container_read_write_async(
            command=command,
            stdin=None,
            stdout_limit=limit,
        )
        if release_error is not None:
            return await self._read_error_async(
                status=SandboxOperationStatus.FAILED,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="docker_cli_unavailable",
                error_message=str(release_error),
            )
        if raw.stdout_truncated:
            return await self._read_error_async(
                status=SandboxOperationStatus.TOO_LARGE,
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code="read_limit_exceeded",
                error_message=f"File exceeds read limit {limit}.",
            )
        if raw.exit_code != 0:
            return await self._read_error_async(
                status=_classify_container_io_error(raw.stderr),
                path=path,
                started_at=started_at,
                operation_context=operation_context,
                error_code=_classify_container_io_error_code(raw.stderr),
                error_message=_stderr_excerpt(raw.stderr) or "The container read command failed.",
            )
        data = raw.stdout
        sha256 = _hash_bytes(data)
        artifact = self._file_artifact(path=path, size_bytes=len(data), sha256=sha256)
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
        Write a bounded binary file into the container via ``docker exec``.

        Returns:
            SandboxWriteResult: Write metadata or an explicit error status.
        """
        async with self._operation_lock:
            if self._closed:
                return await self._write_error_async(
                    status=SandboxOperationStatus.FAILED,
                    started_at=_now(),
                    operation_context=operation_context,
                    error_code="environment_closed",
                    error_message="Sandbox environment is closed.",
                    input_size_bytes=len(data),
                )
            return await self._write_file_unlocked_async(path=path, data=data, operation_context=operation_context)

    async def _write_file_unlocked_async(
        self,
        *,
        path: str,
        data: bytes,
        operation_context: SandboxOperationContext | None,
    ) -> SandboxWriteResult:
        started_at = _now()
        if len(data) > self._spec.limits.max_write_bytes:
            return await self._write_error_async(
                status=SandboxOperationStatus.TOO_LARGE,
                started_at=started_at,
                operation_context=operation_context,
                error_code="write_limit_exceeded",
                error_message=f"Data size {len(data)} exceeds write limit {self._spec.limits.max_write_bytes}.",
                input_size_bytes=len(data),
            )
        try:
            container_path = self._resolve_container_path(path)
        except SandboxPathEscapeError:
            return await self._write_error_async(
                status=SandboxOperationStatus.PATH_ESCAPE,
                started_at=started_at,
                operation_context=operation_context,
                error_code="path_escape",
                error_message="Requested path is outside the sandbox environment.",
                input_size_bytes=len(data),
            )
        relative_path = posixpath.relpath(container_path, self._container_root)
        policy = self._session._provider._config.security_policy
        command = (
            ("python", "-c", _SECURE_CONTAINER_WRITE_SCRIPT, self._container_root, relative_path)
            if policy.require_secure_file_operations
            else (
                "sh",
                "-c",
                (
                    'root=$(readlink -f -- "$1") || exit 1; target=$2; parent=${target%/*}; '
                    'case "$parent" in "$1") relative="" ;; "$1"/*) relative=${parent#"$1"/} ;; '
                    '*) printf "pyrit_path_escape\\n" >&2; exit 1 ;; esac; '
                    "current=$root; old_ifs=$IFS; IFS=/; set -f; "
                    "for part in $relative; do "
                    '[ -n "$part" ] || continue; next=$current/$part; '
                    'if [ -L "$next" ]; then printf "pyrit_path_escape\\n" >&2; exit 1; fi; '
                    'if [ -e "$next" ]; then [ -d "$next" ] || exit 1; else mkdir -- "$next" || exit 1; fi; '
                    "current=$next; done; IFS=$old_ifs; destination=$current/${target##*/}; "
                    'if [ -L "$destination" ]; then printf "pyrit_path_escape\\n" >&2; exit 1; fi; '
                    'cat > "$destination"'
                ),
                "sh",
                self._container_root,
                container_path,
            )
        )
        raw, release_error = await self._run_container_read_write_async(
            command=command,
            stdin=data,
            stdout_limit=0,
        )
        if release_error is not None:
            return await self._write_error_async(
                status=SandboxOperationStatus.FAILED,
                started_at=started_at,
                operation_context=operation_context,
                error_code="docker_cli_unavailable",
                error_message=str(release_error),
                input_size_bytes=len(data),
                side_effect_completed=False,
            )
        if raw.exit_code != 0:
            return await self._write_error_async(
                status=_classify_container_io_error(raw.stderr),
                started_at=started_at,
                operation_context=operation_context,
                error_code=_classify_container_io_error_code(raw.stderr),
                error_message=_stderr_excerpt(raw.stderr) or "The container write command failed.",
                input_size_bytes=len(data),
                side_effect_completed=None,
            )
        sha256 = _hash_bytes(data)
        artifact = self._file_artifact(path=path, size_bytes=len(data), sha256=sha256)
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

    async def _run_container_read_write_async(
        self,
        *,
        command: Sequence[str],
        stdin: bytes | None,
        stdout_limit: int,
    ) -> tuple[_RawRun, DockerCliUnavailableError | None]:
        """
        Run one internal read/write container command outside the public exec path.

        Bypasses ``SandboxProcess.communicate_async`` (which is exec-specific and
        would mislabel evidence) by driving ``DockerSandboxProcess`` directly.

        Returns:
            tuple[_RawRun, DockerCliUnavailableError | None]: The raw outcome, or a
            spawn-time error when the command could never be started (in which case no
            side effect was attempted).
        """
        argv = self._container_exec_argv(command)
        try:
            process, release = await _spawn_container_process_async(
                docker_executable=self._session._provider._config.docker_executable,
                argv=argv,
                semaphore=self._session._provider._cli_semaphore,
                needs_stdin=stdin is not None,
            )
        except DockerCliUnavailableError as error:
            empty = _RawRun(
                stdout=b"",
                stderr=b"",
                stdout_truncated=False,
                stderr_truncated=False,
                exit_code=None,
                timed_out=False,
                cancelled=False,
            )
            return empty, error
        process_handle = DockerSandboxProcess(
            process=process,
            environment=self,
            started_at=_now(),
            release_semaphore=release,
            timeout_seconds=self._spec.limits.max_exec_seconds,
            stdout_limit=max(stdout_limit, 1),
            stderr_limit=self._spec.limits.max_stderr_bytes,
        )
        async with self._process_lock:
            self._processes.add(process_handle)
        raw = await process_handle._communicate_raw_async(stdin=stdin, cancellation_event=None)
        return raw, None

    def _file_artifact(self, *, path: str, size_bytes: int, sha256: str) -> SandboxArtifact:
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
            provider="docker",
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
        path: str,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
        error_code: str,
        error_message: str,
    ) -> SandboxReadResult:
        evidence = self._operation_evidence(
            operation="read",
            outcome=status,
            started_at=started_at,
            operation_context=operation_context,
            error_code=error_code,
            metadata={"path_sha256": _hash_bytes(path.encode())},
        )
        await _emit_evidence_async(sink=self._evidence_sink, evidence=evidence)
        return SandboxReadResult(
            status=status, error_code=error_code, error_message=error_message, evidence=(evidence,)
        )

    async def _write_error_async(
        self,
        *,
        status: SandboxOperationStatus,
        started_at: datetime,
        operation_context: SandboxOperationContext | None,
        error_code: str,
        error_message: str,
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

    async def close_async(self) -> None:
        """Terminate every active process before container/session teardown."""
        async with self._operation_lock:
            self._closed = True
            async with self._process_lock:
                processes = tuple(self._processes)
        await asyncio.gather(*(process.terminate_async() for process in processes))
        async with self._process_lock:
            self._processes.difference_update(processes)

    async def remove_process_async(self, process: DockerSandboxProcess) -> None:
        """Remove a completed process from environment tracking."""
        async with self._process_lock:
            self._processes.discard(process)


_SECURE_CONTAINER_READ_SCRIPT = """
import errno
import os
import stat
import sys

fds = []
try:
    parts = sys.argv[2].split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OSError(errno.EXDEV, "path escape")
    current = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fds.append(current)
    for part in parts[:-1]:
        current = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
        fds.append(current)
    file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW, dir_fd=current)
    fds.append(file_fd)
    file_stat = os.fstat(file_fd)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise OSError(errno.EXDEV, "path escape")
    while chunk := os.read(file_fd, 65536):
        os.write(1, chunk)
except FileNotFoundError:
    os.write(2, b"pyrit_file_not_found\\n")
    sys.exit(1)
except PermissionError:
    os.write(2, b"pyrit_permission_denied\\n")
    sys.exit(1)
except OSError as error:
    marker = (
        b"pyrit_path_escape\\n"
        if error.errno in {errno.ELOOP, errno.EXDEV, errno.ENOTDIR}
        else b"pyrit_container_io_error\\n"
    )
    os.write(2, marker)
    sys.exit(1)
finally:
    for descriptor in reversed(fds):
        try:
            os.close(descriptor)
        except OSError:
            pass
"""

_SECURE_CONTAINER_WRITE_SCRIPT = """
import errno
import os
import stat
import sys

fds = []
try:
    parts = sys.argv[2].split("/")
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise OSError(errno.EXDEV, "path escape")
    current = os.open(sys.argv[1], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    fds.append(current)
    for part in parts[:-1]:
        try:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
        except FileNotFoundError:
            os.mkdir(part, mode=0o755, dir_fd=current)
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=current)
        current = next_fd
        fds.append(current)
    file_fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, mode=0o644, dir_fd=current)
    fds.append(file_fd)
    file_stat = os.fstat(file_fd)
    if not stat.S_ISREG(file_stat.st_mode) or file_stat.st_nlink != 1:
        raise OSError(errno.EXDEV, "path escape")
    os.ftruncate(file_fd, 0)
    while chunk := os.read(0, 65536):
        view = memoryview(chunk)
        while view:
            view = view[os.write(file_fd, view):]
except FileNotFoundError:
    os.write(2, b"pyrit_file_not_found\\n")
    sys.exit(1)
except PermissionError:
    os.write(2, b"pyrit_permission_denied\\n")
    sys.exit(1)
except OSError as error:
    marker = (
        b"pyrit_path_escape\\n"
        if error.errno in {errno.ELOOP, errno.EXDEV, errno.ENOTDIR}
        else b"pyrit_container_io_error\\n"
    )
    os.write(2, marker)
    sys.exit(1)
finally:
    for descriptor in reversed(fds):
        try:
            os.close(descriptor)
        except OSError:
            pass
"""


def _classify_container_io_error(stderr: bytes) -> SandboxOperationStatus:
    """
    Heuristically classify a failed container read/write from its stderr text.

    There is no structured errno channel over ``docker exec``, so this is an honest,
    documented heuristic rather than a precise mapping.

    Returns:
        SandboxOperationStatus: The best-effort classified status.
    """
    text = stderr.decode("utf-8", errors="replace").lower()
    if "pyrit_path_escape" in text:
        return SandboxOperationStatus.PATH_ESCAPE
    if "pyrit_file_not_found" in text:
        return SandboxOperationStatus.NOT_FOUND
    if "pyrit_permission_denied" in text:
        return SandboxOperationStatus.PERMISSION_DENIED
    if "no such file" in text:
        return SandboxOperationStatus.NOT_FOUND
    if "permission denied" in text:
        return SandboxOperationStatus.PERMISSION_DENIED
    return SandboxOperationStatus.FAILED


def _classify_container_io_error_code(stderr: bytes) -> str:
    """
    Return the stable error code paired with ``_classify_container_io_error``.

    Returns:
        str: The stable error code.
    """
    status = _classify_container_io_error(stderr)
    return {
        SandboxOperationStatus.PATH_ESCAPE: "path_escape",
        SandboxOperationStatus.NOT_FOUND: "file_not_found",
        SandboxOperationStatus.PERMISSION_DENIED: "permission_denied",
    }.get(status, "container_io_error")


class DockerSandboxSession(SandboxSession):
    """A per-attempt Compose project owning one or more running service containers."""

    def __init__(
        self,
        *,
        provider: DockerSandboxProvider,
        spec: SandboxSessionSpec,
        project_name: str,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> None:
        """Initialize a Docker session (containers are not yet started)."""
        super().__init__(provider_name=provider.name, spec=spec, evidence_sink=evidence_sink)
        self._provider = provider
        self.project_name = project_name
        self._environments = tuple(
            DockerSandboxEnvironment(session=self, spec=environment_spec, evidence_sink=evidence_sink)
            for environment_spec in spec.environments
        )

    @property
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        """The session environments."""
        return self._environments

    async def _initialize_async(self) -> None:
        started_at = _now()
        await self._provider._ensure_images_ready_async()
        await self._provider._register_session_state_async(
            project_name=self.project_name,
            session_id=self.session_id,
            task_id=self._spec.task.task_id if self._spec.task is not None else None,
        )
        await self._up_async()
        await self._wait_ready_async()
        containers = await self._provider._list_containers_by_project_async(
            project_name=self.project_name,
            ownership_id=self._provider._ownership_id,
        )
        container_by_service: dict[str, str] = {}
        for record in containers:
            service_name = _parse_docker_labels(record.get("Labels", "")).get(_COMPOSE_SERVICE_LABEL)
            if service_name is not None:
                container_by_service[service_name] = record["ID"]
        for environment in self._environments:
            service_name = _select_service_name(
                resolved_services=self._provider._resolved_services,
                environment_name=environment.name,
                service_order=self._provider._service_names,
            )
            container_id = container_by_service.get(service_name)
            if container_id is None:
                raise DockerServiceSelectionError(
                    environment_name=environment.name,
                    available=tuple(container_by_service),
                )
            environment.bind_container(container_id=container_id, service_name=service_name)
        for environment in self._environments:
            await environment.initialize_async()
        await self.emit_lifecycle_evidence_async(operation="session_setup", started_at=started_at)

    async def _up_async(self) -> None:
        argv = [
            *self._provider._compose_base_argv(project_name=self.project_name),
            "up",
            "-d",
            "--remove-orphans",
            "--pull",
            self._provider._config.pull_policy.value,
        ]
        await self._provider._run_lifecycle_cli_async(
            operation="up",
            argv=argv,
            cwd=self._provider._project_context,
        )

    async def _wait_ready_async(self) -> None:
        """
        Poll ``docker ps`` until every resolved service has a running container.

        Raises:
            DockerLifecycleError: If services are not all running before the deadline.
        """
        deadline = asyncio.get_running_loop().time() + self._provider._config.readiness_timeout_seconds
        expected = set(self._provider._service_names)
        while True:
            containers = await self._provider._list_containers_by_project_async(
                project_name=self.project_name,
                ownership_id=self._provider._ownership_id,
            )
            unhealthy_services: set[str] = set()
            for record in containers:
                service_name = _parse_docker_labels(record.get("Labels", "")).get(_COMPOSE_SERVICE_LABEL)
                if service_name is not None and _container_health_state(record) == "unhealthy":
                    unhealthy_services.add(service_name)
            if unhealthy_services:
                raise DockerLifecycleError(
                    operation="readiness",
                    detail=f"Service(s) {sorted(unhealthy_services)} reported an unhealthy status.",
                )
            running_services = {
                _parse_docker_labels(record.get("Labels", "")).get(_COMPOSE_SERVICE_LABEL)
                for record in containers
                if _is_container_ready(record)
            }
            running_services.discard(None)
            if expected <= running_services:
                return
            if _remaining_seconds(deadline) <= 0:
                missing = expected - running_services
                raise DockerLifecycleError(
                    operation="readiness",
                    detail=f"Service(s) {sorted(missing)} did not reach a running state in time.",
                )
            poll_interval = min(self._provider._config.readiness_poll_interval_seconds, _remaining_seconds(deadline))
            await asyncio.sleep(poll_interval)

    async def _close_async(self) -> None:
        started_at = _now()
        await asyncio.gather(*(environment.close_async() for environment in self._environments), return_exceptions=True)
        cleanup_error: DockerSandboxError | None = None
        if not self._provider._config.retain_resources_on_close:
            try:
                await self._provider._force_remove_project_resources_async(
                    project_name=self.project_name,
                    ownership_id=self._provider._ownership_id,
                )
            except DockerSandboxError as error:
                cleanup_error = error
        if cleanup_error is None:
            await self._provider._unregister_session_state_async(project_name=self.project_name)
            await self._provider._remove_session_async(self.session_id)
        outcome = SandboxOperationStatus.FAILED if cleanup_error is not None else SandboxOperationStatus.SUCCEEDED
        await self.emit_lifecycle_evidence_async(
            operation="session_cleanup",
            started_at=started_at,
            outcome=outcome,
            error_code="docker_resource_cleanup_failed" if cleanup_error is not None else None,
        )
        if cleanup_error is not None:
            raise cleanup_error

    async def emit_lifecycle_evidence_async(
        self,
        *,
        operation: str,
        started_at: datetime,
        outcome: SandboxOperationStatus = SandboxOperationStatus.SUCCEEDED,
        error_code: str | None = None,
    ) -> None:
        """Emit one session lifecycle event."""
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self._provider_name,
                operation=operation,
                outcome=outcome,
                started_at=started_at,
                session_id=self.session_id,
                error_code=error_code,
                metadata={"compose_project": self.project_name},
            ),
        )


class DockerSandboxProvider(SandboxProvider):
    """
    Create real, isolated per-attempt sandbox sessions backed by Docker Compose.

    Every Docker/Compose call is a safe argv subprocess against the external ``docker``
    and ``docker compose`` CLIs; this class never imports the third-party ``docker``
    Python SDK and issues no subprocess calls at import time (construction alone is
    also side-effect-free — all validation happens in ``_prepare_async``), so
    importing PyRIT never requires Docker to be installed or running.

    Configuration sources are combined: ``DockerSandboxProviderConfig.services``
    entries are synthesized into a temporary Compose document (Dockerfile build
    contexts are resolved to absolute paths before synthesis, preserving the caller's
    intended relative context regardless of where the temporary file lives), and any
    ``DockerSandboxProviderConfig.compose_files`` are layered on top via repeated
    ``-f`` flags. The combined definition is resolved with
    ``docker compose config --format json`` (which works without a running daemon),
    hardened with a final network and ownership-label overlay, resolved again, and
    scanned against ``DockerSandboxProviderConfig.security_policy`` before any
    container is ever created.

    Per-attempt sessions get a collision-resistant Compose project name derived from
    the session's UUID4 identifier, so concurrent attempts never collide. A durable,
    advisory-locked JSON state file in the user's state directory records every live
    project so that ``cleanup_orphans_async`` can find resources left behind by a
    crashed process. Cleanup requires both that private record and the corresponding
    random PyRIT ownership label on each Docker resource.
    """

    def __init__(
        self,
        *,
        config: DockerSandboxProviderConfig,
        evidence_sink: CapabilityEvidenceSink | None = None,
    ) -> None:
        """Initialize provider configuration (no subprocess calls happen here)."""
        super().__init__()
        self._config = config
        self._evidence_sink = evidence_sink
        self._cli_semaphore = asyncio.Semaphore(self._config.max_concurrent_cli_calls)
        self._project_context = (self._config.project_context or Path.cwd()).resolve()
        self._compose_file_paths: tuple[Path, ...] = ()
        self._synthesized_compose_path: Path | None = None
        self._policy_overlay_path: Path | None = None
        self._synthesized_temp_dir: Path | None = None
        self._ownership_id = secrets.token_hex(16)
        self._resolved_services: dict[str, Any] = {}
        self._service_names: tuple[str, ...] = ()
        self._images_ready = False
        self._images_lock = asyncio.Lock()
        self._sessions: dict[str, DockerSandboxSession] = {}
        self._sessions_lock = asyncio.Lock()
        self._state_path = self._resolve_state_path()
        self._state_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        """The stable provider name."""
        return "docker"

    def _resolve_state_path(self) -> Path:
        state_dir = self._config.state_dir or Path(user_state_dir("pyrit")) / "sandbox" / "docker"
        return state_dir / f"{self._config.project_name_prefix}.state.json"

    async def _ensure_synthesized_temp_dir_async(self) -> Path:
        """
        Create the private temporary Compose directory once.

        Returns:
            Path: The provider's temporary Compose directory.
        """
        if self._synthesized_temp_dir is None:
            temp_dir = await asyncio.to_thread(
                tempfile.mkdtemp,
                "",
                f"{self._config.project_name_prefix}-compose-",
            )
            self._synthesized_temp_dir = Path(str(temp_dir))
        return self._synthesized_temp_dir

    async def _check_cli_available_async(self) -> None:
        """
        Confirm the ``docker compose`` CLI is present, without requiring the daemon.

        Raises:
            DockerCliUnavailableError: If the Compose v2 plugin is missing or broken.
        """
        argv = [self._config.docker_executable, "compose", "version"]
        try:
            await _run_cli_async(
                argv=argv,
                cwd=None,
                timeout_seconds=self._config.cli_timeout_seconds,
                input_bytes=None,
                semaphore=self._cli_semaphore,
            )
        except _CliInvocationError as error:
            raise DockerCliUnavailableError(tool="docker compose", detail=_stderr_excerpt(error.stderr)) from error

    async def _synthesize_compose_if_needed_async(self) -> None:
        """Resolve explicit compose file paths and synthesize services, if configured."""
        self._compose_file_paths = tuple(
            path if path.is_absolute() else (self._project_context / path).resolve()
            for path in self._config.compose_files
        )
        if not self._config.services:
            return
        document = _synthesize_compose_document(
            services=self._config.services,
            project_context=self._project_context,
            security_policy=self._config.security_policy,
        )
        temp_dir = await self._ensure_synthesized_temp_dir_async()
        self._synthesized_compose_path = temp_dir / "compose.json"
        await asyncio.to_thread(_write_json_file, path=self._synthesized_compose_path, document=document)

    async def _synthesize_policy_overlay_async(self, *, resolved_document: Mapping[str, Any]) -> None:
        """Write the final network and resource-ownership Compose overlay."""
        document = _synthesize_policy_overlay(
            resolved_document=resolved_document,
            policy=self._config.security_policy,
            ownership_id=self._ownership_id,
        )
        temp_dir = await self._ensure_synthesized_temp_dir_async()
        self._policy_overlay_path = temp_dir / "policy-overlay.json"
        await asyncio.to_thread(_write_json_file, path=self._policy_overlay_path, document=document)

    def _compose_file_args(self) -> list[str]:
        """
        Build the ordered ``-f`` argument list: synthesized file first, then explicit files.

        Ordering is deliberate: explicit user-supplied Compose files are layered on top
        of (and can intentionally override) synthesized service definitions.

        Returns:
            list[str]: The ``-f <path>`` argument pairs, in resolution order.
        """
        args: list[str] = []
        if self._synthesized_compose_path is not None:
            args.extend(["-f", str(self._synthesized_compose_path)])
        for compose_file in self._compose_file_paths:
            args.extend(["-f", str(compose_file)])
        if self._policy_overlay_path is not None:
            args.extend(["-f", str(self._policy_overlay_path)])
        return args

    def _compose_base_argv(self, *, project_name: str) -> list[str]:
        """
        Build the common ``docker compose`` argument prefix for one project.

        Every invocation always passes explicit ``-p`` and ``--project-directory`` for
        full determinism, never relying on Compose's directory-basename defaults.

        Returns:
            list[str]: The common argv prefix, ready to append a subcommand to.
        """
        return [
            self._config.docker_executable,
            "compose",
            "--project-directory",
            str(self._project_context),
            "-p",
            project_name,
            *self._compose_file_args(),
        ]

    async def _run_lifecycle_cli_async(
        self,
        *,
        operation: str,
        argv: Sequence[str],
        cwd: Path | None = None,
        timeout_seconds: float | None = None,
        error_factory: Callable[[str], DockerSandboxError] | None = None,
    ) -> _CliOutput:
        """
        Run one lifecycle CLI command with narrow, pre-side-effect transient retry.

        This is the only place a retry policy is applied. Retries are restricted to
        read-only ``config``/``ps``/resource-list operations. Mutating lifecycle commands
        and ``docker exec`` use a single attempt so side effects are not duplicated.

        Returns:
            _CliOutput: Captured stdout and stderr.

        Raises:
            DockerDaemonUnavailableError: If stderr indicates the daemon is unreachable.
            DockerLifecycleError: Generic fallback naming ``operation`` when no
                ``error_factory`` is supplied.
            DockerSandboxError: A typed error built by ``error_factory``.
        """
        timeout = timeout_seconds or self._config.cli_timeout_seconds
        retryer: AsyncRetrying[Any] = AsyncRetrying(
            stop=stop_after_attempt(3 if operation in _RETRYABLE_LIFECYCLE_OPERATIONS else 1),
            wait=wait_exponential(multiplier=0.5, max=4.0),
            retry=retry_if_exception(_is_transient_cli_error),
            reraise=True,
        )
        try:
            return await retryer(
                _run_cli_async,
                argv=argv,
                cwd=cwd,
                timeout_seconds=timeout,
                input_bytes=None,
                semaphore=self._cli_semaphore,
            )
        except _CliInvocationError as error:
            detail = _cli_failure_detail(error=error, timeout_seconds=timeout)
            if not error.timed_out and _is_daemon_unavailable_error(error):
                raise DockerDaemonUnavailableError(detail=_stderr_excerpt(error.stderr)) from error
            if error_factory is not None:
                raise error_factory(detail) from error
            raise DockerLifecycleError(operation=operation, detail=detail) from error

    async def _resolve_compose_config_async(self) -> dict[str, Any]:
        """
        Resolve the combined Compose definition to plain JSON.

        This succeeds without a running Docker daemon (pure client-side resolution).

        Returns:
            dict[str, Any]: The resolved Compose document.

        Raises:
            DockerComposeConfigError: If resolution fails or the output is not valid JSON.
        """
        base_argv = self._compose_base_argv(project_name=f"{self._config.project_name_prefix}-prep")
        argv = [*base_argv, "config", "--format", "json"]
        output = await self._run_lifecycle_cli_async(
            operation="config",
            argv=argv,
            cwd=self._project_context,
            error_factory=lambda detail: DockerComposeConfigError(detail=detail),
        )
        try:
            return json.loads(output.stdout.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise DockerComposeConfigError(detail=f"Could not parse resolved Compose JSON: {error}") from error

    async def _prepare_async(self) -> None:
        started_at = _now()
        await self._check_cli_available_async()
        await self._synthesize_compose_if_needed_async()
        resolved = await self._resolve_compose_config_async()
        await self._synthesize_policy_overlay_async(resolved_document=resolved)
        resolved = await self._resolve_compose_config_async()
        self._resolved_services = resolved.get("services", {})
        self._service_names = tuple(self._resolved_services)
        violations = _scan_security_violations(resolved_document=resolved, policy=self._config.security_policy)
        if violations:
            raise DockerSecurityPolicyViolationError(violations=violations)
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="provider_prepare",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"service_count": len(self._service_names)},
            ),
        )

    async def _prepare_task_async(self, task: SandboxTaskSpec) -> None:
        started_at = _now()
        await self._ensure_images_ready_async()
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="task_prepare",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"task_id_sha256": _hash_bytes(task.task_id.encode())},
            ),
        )

    async def _cleanup_task_async(self, task: SandboxTaskSpec) -> None:
        started_at = _now()
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="task_cleanup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"task_id_sha256": _hash_bytes(task.task_id.encode())},
            ),
        )

    async def _ensure_images_ready_async(self) -> None:
        """Build configured services exactly once, deterministically tagged for reuse."""
        if self._images_ready or not any(
            isinstance(service, DockerServiceBuildSpec) for service in self._config.services
        ):
            self._images_ready = True
            return
        async with self._images_lock:
            if self._images_ready:
                return
            argv = [
                *self._compose_base_argv(project_name=f"{self._config.project_name_prefix}-prep"),
                "build",
            ]
            await self._run_lifecycle_cli_async(
                operation="build",
                argv=argv,
                cwd=self._project_context,
                timeout_seconds=max(self._config.cli_timeout_seconds, 600.0),
            )
            self._images_ready = True

    async def _create_session_async(
        self,
        *,
        spec: SandboxSessionSpec,
        evidence_sink: CapabilityEvidenceSink | None,
    ) -> SandboxSession:
        started_at = _now()
        project_name = _project_name_for_session(
            prefix=self._config.project_name_prefix,
            session_id=spec.session_id,
            attempt_id=str(spec.attempt_id),
            ownership_id=self._ownership_id,
        )
        session = DockerSandboxSession(
            provider=self,
            spec=spec,
            project_name=project_name,
            evidence_sink=evidence_sink or self._evidence_sink,
        )
        async with self._sessions_lock:
            self._sessions[spec.session_id] = session
        await session.emit_lifecycle_evidence_async(operation="session_create", started_at=started_at)
        return session

    async def _remove_session_async(self, session_id: str) -> None:
        async with self._sessions_lock:
            self._sessions.pop(session_id, None)

    async def _cleanup_async(self) -> None:
        started_at = _now()
        async with self._sessions_lock:
            sessions = tuple(self._sessions.values())
        results = await asyncio.gather(*(session.close_async() for session in sessions), return_exceptions=True)
        failures: list[BaseException] = []
        for session, result in zip(sessions, results, strict=True):
            if not isinstance(result, BaseException):
                continue
            if self._config.retain_resources_on_close:
                try:
                    await self._unregister_session_state_async(project_name=session.project_name)
                    await self._remove_session_async(session.session_id)
                except BaseException as recovery_error:
                    failures.append(recovery_error)
                failures.append(result)
                continue
            try:
                await self._force_remove_project_resources_async(
                    project_name=session.project_name,
                    ownership_id=self._ownership_id,
                )
                await self._unregister_session_state_async(project_name=session.project_name)
                await self._remove_session_async(session.session_id)
            except BaseException as recovery_error:
                failures.append(recovery_error)
        if self._synthesized_temp_dir is not None:
            await asyncio.to_thread(shutil.rmtree, self._synthesized_temp_dir, True)
        outcome = SandboxOperationStatus.FAILED if failures else SandboxOperationStatus.SUCCEEDED
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="provider_cleanup",
                outcome=outcome,
                started_at=started_at,
                error_code="session_cleanup_failed" if failures else None,
                metadata={"session_cleanup_failures": len(failures)},
            ),
        )
        if failures:
            raise RuntimeError(f"{len(failures)} Docker sandbox session cleanup operation(s) failed.")

    async def _register_session_state_async(self, *, project_name: str, session_id: str, task_id: str | None) -> None:
        """Record one live project in the durable state file for crash recovery."""

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state[project_name] = {
                "session_id": session_id,
                "task_id": task_id,
                "created_at": _now().isoformat(),
                "owner_process_id": os.getpid(),
                "ownership_id": self._ownership_id,
                "retain_resources": self._config.retain_resources_on_close,
            }
            return state

        async with self._state_lock:
            await asyncio.to_thread(_update_state_file, self._state_path, mutate)

    async def _unregister_session_state_async(self, *, project_name: str) -> None:
        """Remove one project's durable state record after it has been torn down."""

        def mutate(state: dict[str, Any]) -> dict[str, Any]:
            state.pop(project_name, None)
            return state

        async with self._state_lock:
            await asyncio.to_thread(_update_state_file, self._state_path, mutate)

    async def _list_containers_by_project_async(
        self,
        *,
        project_name: str,
        ownership_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        List every container (any state) belonging to one Compose project.

        Uses raw ``docker ps`` (not ``docker compose ps``) so this works even after a
        crash where the original/synthesized Compose file no longer exists.

        Returns:
            list[dict[str, Any]]: One NDJSON-parsed record per matching container.
        """
        argv = [
            self._config.docker_executable,
            "ps",
            "-a",
            "--filter",
            f"label={_COMPOSE_PROJECT_LABEL}={project_name}",
            "--filter",
            f"label={_RESOURCE_OWNER_LABEL}={ownership_id or self._ownership_id}",
            "--format",
            "{{json .}}",
        ]
        output = await self._run_lifecycle_cli_async(operation="ps", argv=argv)
        return _parse_ndjson(output.stdout)

    async def _force_remove_project_resources_async(self, *, project_name: str, ownership_id: str) -> None:
        """
        Forcibly remove every container, network, and volume for one Compose project.

        Every lookup requires both the Compose project label and the unguessable
        PyRIT ownership label recorded in private durable state.
        """
        containers = await self._list_containers_by_project_async(
            project_name=project_name,
            ownership_id=ownership_id,
        )
        container_ids = [record["ID"] for record in containers if "ID" in record]
        if container_ids:
            await self._run_lifecycle_cli_async(
                operation="rm",
                argv=[self._config.docker_executable, "rm", "-f", "-v", *container_ids],
            )
        network_ids = await self._list_project_resource_ids_async(
            resource="network",
            project_name=project_name,
            ownership_id=ownership_id,
            format_field="{{.ID}}",
        )
        if network_ids:
            await self._run_lifecycle_cli_async(
                operation="network_rm",
                argv=[self._config.docker_executable, "network", "rm", *network_ids],
            )
        volume_names = await self._list_project_resource_ids_async(
            resource="volume",
            project_name=project_name,
            ownership_id=ownership_id,
            format_field="{{.Name}}",
        )
        if volume_names:
            await self._run_lifecycle_cli_async(
                operation="volume_rm",
                argv=[self._config.docker_executable, "volume", "rm", *volume_names],
            )

    async def _list_project_resource_ids_async(
        self,
        *,
        resource: str,
        project_name: str,
        ownership_id: str,
        format_field: str,
    ) -> list[str]:
        """
        List resource identifiers (networks/volumes) scoped to one Compose project.

        Returns:
            list[str]: The matching resource identifiers.
        """
        argv = [
            self._config.docker_executable,
            resource,
            "ls",
            "--filter",
            f"label={_COMPOSE_PROJECT_LABEL}={project_name}",
            "--filter",
            f"label={_RESOURCE_OWNER_LABEL}={ownership_id}",
            "--format",
            format_field,
        ]
        output = await self._run_lifecycle_cli_async(operation=f"{resource}_ls", argv=argv)
        return [line for line in output.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]

    async def _cleanup_orphans_async(self) -> int:
        """
        Remove containers/networks/volumes left behind by a crashed provider instance.

        Returns:
            int: The number of orphaned projects cleaned.
        """
        started_at = _now()
        state = await asyncio.to_thread(_read_state_file, self._state_path)
        async with self._sessions_lock:
            active_projects = {session.project_name for session in self._sessions.values()}
        orphan_projects: list[tuple[str, str]] = []
        for project_name, record in state.items():
            if project_name in active_projects or not isinstance(record, dict):
                continue
            if record.get("retain_resources") is not False:
                continue
            owner_process_id = record.get("owner_process_id")
            ownership_id = record.get("ownership_id")
            if (
                not _is_provider_project_name(
                    project_name=project_name,
                    prefix=self._config.project_name_prefix,
                )
                or not isinstance(owner_process_id, int)
                or not isinstance(ownership_id, str)
                or _OWNERSHIP_ID_PATTERN.fullmatch(ownership_id) is None
            ):
                continue
            if await asyncio.to_thread(_process_is_alive, owner_process_id):
                continue
            orphan_projects.append((project_name, ownership_id))
        cleaned_count = 0
        for project_name, ownership_id in orphan_projects:
            try:
                await self._force_remove_project_resources_async(
                    project_name=project_name,
                    ownership_id=ownership_id,
                )
            except DockerSandboxError:
                continue
            await self._unregister_session_state_async(project_name=project_name)
            cleaned_count += 1
        await _emit_evidence_async(
            sink=self._evidence_sink,
            evidence=_evidence(
                provider=self.name,
                operation="orphan_cleanup",
                outcome=SandboxOperationStatus.SUCCEEDED,
                started_at=started_at,
                metadata={"resources_discovered": len(orphan_projects), "resources_cleaned": cleaned_count},
            ),
        )
        return cleaned_count


def register_process_exit_cleanup(provider: DockerSandboxProvider) -> None:
    """
    Register a best-effort, opt-in ``atexit`` hook to clean up abandoned resources.

    This is never invoked automatically anywhere else in this module — callers must
    explicitly opt in by calling this function once per provider instance if they want
    a last-resort safety net for orphaned Docker resources on interpreter exit.
    """

    def _cleanup_on_exit() -> None:
        with suppress(Exception):
            asyncio.run(provider.cleanup_orphans_async())

    atexit.register(_cleanup_on_exit)
