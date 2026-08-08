# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Immutable provider-neutral sandbox models."""

from __future__ import annotations

import re
import uuid
from enum import Enum
from pathlib import Path  # noqa: TC003 (Pydantic resolves this annotation at runtime)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyrit.executor.capability.models import SandboxOperationEvidence  # noqa: TC001
from pyrit.models import JSONValue  # noqa: TC001

_ENVIRONMENT_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    @field_validator("environment")
    @classmethod
    def _validate_environment_keys(cls, environment: dict[str, str]) -> dict[str, str]:
        invalid_keys = sorted(key for key in environment if not _ENVIRONMENT_KEY_PATTERN.fullmatch(key))
        if invalid_keys:
            raise ValueError(
                "Environment variable names must match ^[A-Za-z_][A-Za-z0-9_]*$: "
                + ", ".join(repr(key) for key in invalid_keys)
            )
        return environment

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

    Host escape surfaces default to a rejecting posture. Network egress remains enabled
    by default for compatibility and can be disabled explicitly. Synthesized services
    are compliant by construction; explicit user-supplied Compose files are hardened
    with a final policy overlay, validated after resolution, and rejected
    (``DockerSecurityPolicyViolationError``) when enforcement cannot be proven.
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


class HyperVGuestOS(str, Enum):
    """The guest operating-system family."""

    WINDOWS = "windows"
    LINUX = "linux"


class HyperVGuestTransportKind(str, Enum):
    """The external host facility used to reach a Hyper-V guest."""

    POWERSHELL_DIRECT = "powershell_direct"
    SSH = "ssh"


class HyperVDiskStrategy(str, Enum):
    """How a per-attempt virtual disk is derived from its template."""

    DIFFERENCING = "differencing"
    COPY = "copy"


class HyperVNetworkMode(str, Enum):
    """The network isolation mode for a sandbox VM."""

    PRIVATE = "private"
    INTERNAL = "internal"
    EXTERNAL = "external"
    NONE = "none"


class HyperVSecureBootMode(str, Enum):
    """The Generation 2 VM secure-boot policy."""

    MICROSOFT_WINDOWS = "microsoft_windows"
    MICROSOFT_UEFI_CA = "microsoft_uefi_ca"
    DISABLED = "disabled"


class HyperVSecretReference(BaseModel):
    """An opaque reference resolved at runtime without persisting a credential value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    secret_id: str = Field(min_length=1)


class HyperVReadinessConfig(BaseModel):
    """Guest readiness polling settings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeout_seconds: float = Field(default=120.0, gt=0, le=1800)
    poll_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    probe_argv: tuple[str, ...] | None = None


class HyperVSecurityPolicy(BaseModel):
    """Secure-by-default host and VM settings requiring explicit opt-in."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allow_external_switch: bool = False
    allow_internet_egress: bool = False
    allow_host_filesystem_sharing: bool = False
    allow_nested_virtualization: bool = False
    allow_device_passthrough: bool = False
    allow_mac_spoofing: bool = False
    max_processor_count: int = Field(default=16, ge=1, le=64)
    max_memory_mb: int = Field(default=32768, ge=512, le=262144)
    max_disk_size_gb: int = Field(default=512, ge=1, le=4096)


class HyperVComposeDelegationConfig(BaseModel):
    """
    Typed seam for a future Compose-inside-VM delegation.

    Layer 5's Docker provider invokes a local Docker CLI and has no injectable remote
    Docker endpoint/context. Enabling this seam therefore raises a typed error rather
    than pretending that Compose workloads were delegated into the guest.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = False
    docker_host: str | None = None
    docker_context: str | None = None
    provider_config: DockerSandboxProviderConfig | None = None

    @model_validator(mode="after")
    def _validate_endpoint(self) -> HyperVComposeDelegationConfig:
        if self.enabled and self.docker_host is None and self.docker_context is None:
            raise ValueError("Enabled Hyper-V Compose delegation requires docker_host or docker_context.")
        if self.enabled and self.provider_config is None:
            raise ValueError("Enabled Hyper-V Compose delegation requires provider_config.")
        return self


class HyperVEnvironmentConfig(BaseModel):
    """Immutable VM template and guest-access configuration for one named environment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    default: bool = False
    base_vhdx: Path | None = None
    template_vm: str | None = None
    template_checkpoint: str | None = None
    generation: int = Field(default=2, ge=1, le=2)
    processor_count: int = Field(default=2, ge=1, le=64)
    memory_mb: int = Field(default=2048, ge=512, le=262144)
    dynamic_memory: bool = False
    disk_strategy: HyperVDiskStrategy = HyperVDiskStrategy.DIFFERENCING
    max_disk_size_gb: int = Field(default=128, ge=1, le=4096)
    network_mode: HyperVNetworkMode = HyperVNetworkMode.PRIVATE
    switch_name: str | None = None
    guest_os: HyperVGuestOS = HyperVGuestOS.WINDOWS
    transport: HyperVGuestTransportKind = HyperVGuestTransportKind.POWERSHELL_DIRECT
    secure_boot: HyperVSecureBootMode = HyperVSecureBootMode.MICROSOFT_WINDOWS
    credential: HyperVSecretReference | None = None
    ssh_host: str | None = None
    ssh_port: int = Field(default=22, ge=1, le=65535)
    workspace_root: str = Field(default=r"C:\PyRITSandbox", min_length=1)
    readiness: HyperVReadinessConfig = Field(default_factory=HyperVReadinessConfig)
    setup_files: tuple[SandboxSetupFile, ...] = ()
    setup_scripts: tuple[SandboxSetupScript, ...] = ()

    @model_validator(mode="after")
    def _validate_template_and_transport(self) -> HyperVEnvironmentConfig:
        if (self.base_vhdx is None) == (self.template_vm is None):
            raise ValueError("Exactly one of base_vhdx or template_vm must be configured.")
        if self.template_checkpoint is not None and self.template_vm is None:
            raise ValueError("template_checkpoint requires template_vm.")
        if self.transport is HyperVGuestTransportKind.POWERSHELL_DIRECT:
            if self.guest_os is not HyperVGuestOS.WINDOWS:
                raise ValueError("PowerShell Direct requires a Windows guest.")
            if self.credential is None:
                raise ValueError("PowerShell Direct requires a credential secret reference.")
        if self.transport is HyperVGuestTransportKind.SSH and self.credential is None:
            raise ValueError("SSH requires a credential secret reference.")
        if self.transport is HyperVGuestTransportKind.SSH and self.ssh_host is None:
            raise ValueError("SSH requires an explicit ssh_host.")
        if self.network_mode is HyperVNetworkMode.NONE and self.switch_name is not None:
            raise ValueError("network_mode='none' cannot specify switch_name.")
        return self


class HyperVSandboxProviderConfig(BaseModel):
    """Production-oriented Hyper-V provider configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environments: tuple[HyperVEnvironmentConfig, ...]
    default_environment: str | None = None
    allowed_switches: tuple[str, ...] = ()
    security_policy: HyperVSecurityPolicy = Field(default_factory=HyperVSecurityPolicy)
    compose_delegation: HyperVComposeDelegationConfig = Field(default_factory=HyperVComposeDelegationConfig)
    powershell_executable: str = "powershell.exe"
    ssh_executable: str = "ssh"
    sftp_executable: str = "sftp"
    state_dir: Path | None = None
    vm_name_prefix: str = Field(default="pyrit-sbx", min_length=1, max_length=24, pattern=r"^[A-Za-z0-9-]+$")
    cli_timeout_seconds: float = Field(default=180.0, gt=0, le=1800)
    max_command_output_bytes: int = Field(default=16_777_216, gt=0, le=67_108_864)
    retain_resources_on_close: bool = False
    retain_resources_on_failure: bool = False

    @model_validator(mode="after")
    def _validate_environments_and_security(self) -> HyperVSandboxProviderConfig:
        names = [environment.name for environment in self.environments]
        if not names:
            raise ValueError("At least one Hyper-V environment is required.")
        if len(names) != len(set(names)):
            raise ValueError("Hyper-V environment names must be unique.")
        marked = [environment.name for environment in self.environments if environment.default]
        if len(marked) > 1:
            raise ValueError("At most one Hyper-V environment can be marked as default.")
        if self.default_environment is not None and self.default_environment not in names:
            raise ValueError(f"Default Hyper-V environment '{self.default_environment}' is not defined.")
        self._validate_resource_bounds()
        self._validate_networks()
        return self

    def resolve_default_environment(self) -> str:
        """Return the deterministic default Hyper-V environment name."""
        if self.default_environment is not None:
            return self.default_environment
        marked = [environment.name for environment in self.environments if environment.default]
        if marked:
            return marked[0]
        if "default" in {environment.name for environment in self.environments}:
            return "default"
        return min(environment.name for environment in self.environments)

    def get_environment(self, name: str) -> HyperVEnvironmentConfig:
        """
        Return one named environment configuration.

        Returns:
            HyperVEnvironmentConfig: The matching environment configuration.

        Raises:
            KeyError: If ``name`` is not configured.
        """
        for environment in self.environments:
            if environment.name == name:
                return environment
        raise KeyError(name)

    def _validate_resource_bounds(self) -> None:
        policy = self.security_policy
        for environment in self.environments:
            if environment.processor_count > policy.max_processor_count:
                raise ValueError(f"Environment '{environment.name}' exceeds the processor policy bound.")
            if environment.memory_mb > policy.max_memory_mb:
                raise ValueError(f"Environment '{environment.name}' exceeds the memory policy bound.")
            if environment.max_disk_size_gb > policy.max_disk_size_gb:
                raise ValueError(f"Environment '{environment.name}' exceeds the disk policy bound.")

    def _validate_networks(self) -> None:
        policy = self.security_policy
        allowed = set(self.allowed_switches)
        for environment in self.environments:
            if environment.switch_name is not None and environment.switch_name not in allowed:
                raise ValueError(f"Environment '{environment.name}' uses a switch outside allowed_switches.")
            if environment.network_mode is HyperVNetworkMode.EXTERNAL:
                if not policy.allow_external_switch or not policy.allow_internet_egress:
                    raise ValueError("External networking requires explicit switch and Internet-egress opt-in.")
                if environment.switch_name is None:
                    raise ValueError("External networking requires an allow-listed switch_name.")


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
