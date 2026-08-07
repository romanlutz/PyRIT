# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Immutable provider-neutral sandbox models."""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path  # noqa: TC003 (Pydantic resolves this annotation at runtime)

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrit.executor.capability.models import SandboxOperationEvidence  # noqa: TC001
from pyrit.models import JSONValue  # noqa: TC001


class SandboxOperationStatus(str, Enum):
    """The terminal status of a sandbox operation."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    TRUNCATED = "truncated"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    TOO_LARGE = "too_large"
    PATH_ESCAPE = "path_escape"


class SandboxLimits(BaseModel):
    """Resource limits enforced by a sandbox environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_stdout_bytes: int = Field(default=1_048_576, gt=0)
    max_stderr_bytes: int = Field(default=1_048_576, gt=0)
    max_read_bytes: int = Field(default=8_388_608, gt=0)
    max_write_bytes: int = Field(default=8_388_608, gt=0)
    max_exec_seconds: float = Field(default=300.0, gt=0)
    terminate_grace_seconds: float = Field(default=2.0, ge=0)


class SandboxArtifact(BaseModel):
    """A provider-neutral reference to data produced by a sandbox."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str
    uri: str
    media_type: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SandboxConnectionInfo(BaseModel):
    """Non-secret metadata describing how a sandbox environment is reached."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str
    session_id: str
    environment_name: str
    transport: str
    endpoint: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SandboxOperationContext(BaseModel):
    """Correlation identifiers supplied by a capability tool execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    call_id: str | None = None
    attempt_id: uuid.UUID | None = None


class SandboxExecRequest(BaseModel):
    """A buffered process request using argv or an explicit shell script."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    argv: tuple[str, ...] | None = None
    shell_script: str | None = None
    stdin: bytes | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    user: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_command(self) -> SandboxExecRequest:
        if (self.argv is None) == (self.shell_script is None):
            raise ValueError("Exactly one of argv or shell_script must be provided.")
        if self.argv is not None and not self.argv:
            raise ValueError("argv must contain at least one element.")
        return self


class SandboxSetupFile(BaseModel):
    """A file materialized before an environment is used."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    content: bytes = b""
    executable: bool = False


class SandboxSetupScript(BaseModel):
    """A process request executed while preparing an environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request: SandboxExecRequest


class SandboxEnvironmentSpec(BaseModel):
    """Configuration for one named environment in a session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    default: bool = False
    limits: SandboxLimits = Field(default_factory=SandboxLimits)
    setup_files: tuple[SandboxSetupFile, ...] = ()
    setup_scripts: tuple[SandboxSetupScript, ...] = ()
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SandboxTaskSpec(BaseModel):
    """Provider preparation inputs shared by every attempt for one task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_id: str = Field(min_length=1)
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class SandboxSessionSpec(BaseModel):
    """Per-attempt session configuration and deterministic environment selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()), min_length=1)
    attempt_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    task: SandboxTaskSpec | None = None
    environments: tuple[SandboxEnvironmentSpec, ...] = (SandboxEnvironmentSpec(name="default", default=True),)
    default_environment: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_environments(self) -> SandboxSessionSpec:
        names = [environment.name for environment in self.environments]
        if not names:
            raise ValueError("At least one sandbox environment is required.")
        if len(names) != len(set(names)):
            raise ValueError("Sandbox environment names must be unique.")
        marked_defaults = [environment.name for environment in self.environments if environment.default]
        if len(marked_defaults) > 1:
            raise ValueError("At most one sandbox environment can be marked as default.")
        if self.default_environment is not None and self.default_environment not in names:
            raise ValueError(f"Default environment '{self.default_environment}' is not defined.")
        return self

    def resolve_default_environment(self) -> str:
        """Return the deterministic default environment name."""
        if self.default_environment is not None:
            return self.default_environment
        marked = [environment.name for environment in self.environments if environment.default]
        if marked:
            return marked[0]
        if "default" in {environment.name for environment in self.environments}:
            return "default"
        return min(environment.name for environment in self.environments)


class LocalSandboxProviderConfig(BaseModel):
    """Configuration for the trusted local development provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    workspace_root: Path | None = None
    retain_workspaces: bool = False
    allow_unrestricted_host_execution: bool = False


class DockerPullPolicy(str, Enum):
    """When the Docker sandbox provider pulls images before building or starting services."""

    MISSING = "missing"
    ALWAYS = "always"
    NEVER = "never"


class DockerServiceBuildSpec(BaseModel):
    """A Dockerfile-based service synthesized into a temporary Compose definition."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")
    build_context: Path
    dockerfile: str = "Dockerfile"
    build_args: dict[str, str] = Field(default_factory=dict)
    target: str | None = None
    command: tuple[str, ...] | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    working_dir: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


class DockerSecurityPolicy(BaseModel):
    """
    Security defaults enforced for every Docker sandbox service.

    Every ``allow_*`` flag defaults to a rejecting posture. A synthesized service is
    always compliant; an explicit user-supplied Compose file is validated against this
    policy after resolution and rejected (``DockerSecurityPolicyViolationError``) unless
    the corresponding flag is explicitly enabled.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_privileged: bool = False
    allow_host_namespaces: bool = False
    allow_docker_socket_mount: bool = False
    allow_device_mounts: bool = False
    allow_bind_mounts: bool = False
    allow_published_ports: bool = False
    allow_dangerous_capabilities: bool = False
    allow_unconfined_seccomp: bool = False
    allow_unrestricted_secrets: bool = False
    allow_absolute_container_paths: bool = False
    isolate_interservice_network: bool = True
    allow_egress: bool = True
    drop_all_capabilities: bool = True
    read_only_root_filesystem: bool = False
    default_pids_limit: int | None = Field(default=256, gt=0)
    default_memory_limit: str | None = None
    default_cpus: float | None = Field(default=None, gt=0)


class DockerSandboxProviderConfig(BaseModel):
    """Configuration for the Docker/Compose sandbox provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    compose_files: tuple[Path, ...] = ()
    services: tuple[DockerServiceBuildSpec, ...] = ()
    project_context: Path | None = None
    project_name_prefix: str = Field(default="pyrit-sbx", max_length=32, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    pull_policy: DockerPullPolicy = DockerPullPolicy.MISSING
    security_policy: DockerSecurityPolicy = Field(default_factory=DockerSecurityPolicy)
    readiness_timeout_seconds: float = Field(default=90.0, gt=0)
    readiness_poll_interval_seconds: float = Field(default=0.5, gt=0)
    max_concurrent_cli_calls: int = Field(default=4, gt=0)
    cli_timeout_seconds: float = Field(default=180.0, gt=0)
    state_dir: Path | None = None
    retain_resources_on_close: bool = False
    docker_executable: str = "docker"

    @model_validator(mode="after")
    def _validate_sources(self) -> DockerSandboxProviderConfig:
        if not self.compose_files and not self.services:
            raise ValueError("DockerSandboxProviderConfig requires at least one of 'compose_files' or 'services'.")
        return self


class SandboxExecResult(BaseModel):
    """Buffered process output with explicit limit and interruption facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SandboxOperationStatus
    stdout: bytes = b""
    stderr: bytes = b""
    exit_code: int | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False
    error_code: str | None = None
    error_message: str | None = None
    artifacts: tuple[SandboxArtifact, ...] = ()
    evidence: tuple[SandboxOperationEvidence, ...] = ()


class SandboxReadResult(BaseModel):
    """Binary-safe file-read result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SandboxOperationStatus
    data: bytes | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    artifacts: tuple[SandboxArtifact, ...] = ()
    evidence: tuple[SandboxOperationEvidence, ...] = ()


class SandboxWriteResult(BaseModel):
    """Binary-safe file-write result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SandboxOperationStatus
    size_bytes: int = Field(default=0, ge=0)
    sha256: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    side_effect_completed: bool | None = False
    artifacts: tuple[SandboxArtifact, ...] = ()
    evidence: tuple[SandboxOperationEvidence, ...] = ()
