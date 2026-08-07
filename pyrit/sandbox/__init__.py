# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provider-neutral sandbox contracts and trusted local implementation."""

from pyrit.sandbox.contracts import SandboxEnvironment, SandboxProcess, SandboxProvider, SandboxSession
from pyrit.sandbox.docker_provider import (
    DockerCliUnavailableError,
    DockerComposeConfigError,
    DockerDaemonUnavailableError,
    DockerLifecycleError,
    DockerSandboxError,
    DockerSandboxProvider,
    DockerSecurityPolicyViolationError,
    DockerServiceSelectionError,
    register_process_exit_cleanup,
)
from pyrit.sandbox.local import LocalSandboxProvider, SandboxPathEscapeError, SandboxSetupError
from pyrit.sandbox.models import (
    DockerPullPolicy,
    DockerSandboxProviderConfig,
    DockerSecurityPolicy,
    DockerServiceBuildSpec,
    LocalSandboxProviderConfig,
    SandboxArtifact,
    SandboxConnectionInfo,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxExecResult,
    SandboxLimits,
    SandboxOperationContext,
    SandboxOperationStatus,
    SandboxReadResult,
    SandboxSessionSpec,
    SandboxSetupFile,
    SandboxSetupScript,
    SandboxTaskSpec,
    SandboxWriteResult,
)
from pyrit.sandbox.registry import SandboxProviderMetadata, SandboxProviderRegistry
from pyrit.sandbox.tool_adapter import SandboxToolAdapter

__all__ = [
    "DockerCliUnavailableError",
    "DockerComposeConfigError",
    "DockerDaemonUnavailableError",
    "DockerLifecycleError",
    "DockerPullPolicy",
    "DockerSandboxError",
    "DockerSandboxProvider",
    "DockerSandboxProviderConfig",
    "DockerSecurityPolicy",
    "DockerSecurityPolicyViolationError",
    "DockerServiceBuildSpec",
    "DockerServiceSelectionError",
    "LocalSandboxProvider",
    "LocalSandboxProviderConfig",
    "SandboxArtifact",
    "SandboxConnectionInfo",
    "SandboxEnvironment",
    "SandboxEnvironmentSpec",
    "SandboxExecRequest",
    "SandboxExecResult",
    "SandboxLimits",
    "SandboxOperationContext",
    "SandboxOperationStatus",
    "SandboxPathEscapeError",
    "SandboxProcess",
    "SandboxProvider",
    "SandboxProviderMetadata",
    "SandboxProviderRegistry",
    "SandboxReadResult",
    "SandboxSession",
    "SandboxSessionSpec",
    "SandboxSetupError",
    "SandboxSetupFile",
    "SandboxSetupScript",
    "SandboxTaskSpec",
    "SandboxToolAdapter",
    "SandboxWriteResult",
    "register_process_exit_cleanup",
]
