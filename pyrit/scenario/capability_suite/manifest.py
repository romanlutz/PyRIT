# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Strict, immutable, JSON-serializable capability-suite manifest schema.

A capability-suite manifest is a fully static, declarative description of a set of
capability-task cases: their objectives/messages/modalities, staged assets, symbolic
tool declarations, sandbox provider configuration, setup, execution limits, scorers,
and run policy (attempts/epochs/concurrency). Manifests never carry Python import
paths, executable code, or references to Inspect AI / ``inspect_evals`` — every
resolvable name is a symbolic string resolved through an explicit, caller-populated
registry (see ``pyrit.scenario.capability_suite.registries``).

Every model in this module is frozen and rejects unknown fields
(``ConfigDict(frozen=True, extra="forbid")``), so a manifest is immutable once parsed
and malformed/unexpected JSON is rejected rather than silently ignored.
"""

from __future__ import annotations

import re
from enum import Enum
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrit.executor.capability import (
    CapabilityLimits,
    CapabilitySource,
    ExpectedEvidence,
    ToolDeclaration,
)
from pyrit.models import JSONValue  # noqa: TC001 (Pydantic resolves this annotation at runtime)
from pyrit.models.literals import ChatMessageRole, PromptDataType  # noqa: TC001 (Pydantic resolves at runtime)
from pyrit.sandbox import (
    DockerSandboxProviderConfig,
    HyperVSandboxProviderConfig,
    LocalSandboxProviderConfig,
)

#: The schema version this build of PyRIT natively understands. Raw JSON at an older
#: version is migrated forward (see ``serialization.load_manifest_json``); raw JSON at
#: a newer version is rejected.
CURRENT_MANIFEST_SCHEMA_VERSION = 2

_SYMBOL_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


def validate_safe_relative_path(path: str) -> str:
    """
    Validate that ``path`` is a safe, relative, non-escaping, non-empty path string.

    Rejects absolute paths, drive letters, UNC roots, home-directory shorthand, null
    bytes, and any ``..`` traversal component, so a manifest can never reference
    (or later stage a file to) a location outside of a declared containment root.

    Returns:
        str: The validated path, unchanged.

    Raises:
        ValueError: If the path is empty or unsafe by any of the rules above.
    """
    if not path:
        raise ValueError("Path must not be empty.")
    if "\x00" in path:
        raise ValueError(f"Path must not contain null bytes: {path!r}")
    if "\\" in path:
        raise ValueError(f"Path must use portable '/' separators: {path!r}")
    normalized = path
    if normalized.startswith(("/", "~")):
        raise ValueError(f"Path must be relative: {path!r}")
    if re.match(r"^[A-Za-z]:", normalized) or normalized.startswith("//"):
        raise ValueError(f"Path must not reference a drive or UNC root: {path!r}")
    segments = PurePosixPath(normalized).parts
    if ".." in segments:
        raise ValueError(f"Path must not contain '..' traversal components: {path!r}")
    if any(segment in ("", ".") for segment in normalized.split("/")):
        raise ValueError(f"Path must be normalized without empty or '.' components: {path!r}")
    return path


class SuiteProvenance(BaseModel):
    """Source/repository/revision/license provenance for a whole suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    source_id: str | None = None
    repository: str | None = None
    revision: str | None = None
    license: str | None = None
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class CaseMessageManifest(BaseModel):
    """
    A deterministic, JSON-serializable stand-in for one initial ``Message``.

    ``pyrit.models.Message``/``MessagePiece`` default to a random UUID and
    ``datetime.now()``, so they cannot be embedded in a frozen/hashed manifest.
    This model carries only deterministic content; the runner materializes real
    ``Message``/``MessagePiece`` objects (with fresh identifiers) per attempt.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: ChatMessageRole
    content: str
    data_type: PromptDataType = "text"
    metadata: dict[str, JSONValue] = Field(default_factory=dict)


class AssetMode(str, Enum):
    """How a staged asset's content may be used once materialized in a sandbox."""

    READ_ONLY = "read_only"
    EXECUTABLE = "executable"


class CaseAssetManifest(BaseModel):
    """An asset staged into a sandbox environment before a case runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset_id: str = Field(min_length=1, pattern=_SYMBOL_PATTERN)
    source: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    destination: str = Field(min_length=1)
    environment: str | None = None
    mode: AssetMode = AssetMode.READ_ONLY
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_paths(self) -> CaseAssetManifest:
        validate_safe_relative_path(self.source)
        validate_safe_relative_path(self.destination)
        return self


class BuildContextAssetKind(str, Enum):
    """The role a build-context asset plays for a Docker/Compose sandbox provider."""

    DOCKERFILE = "dockerfile"
    COMPOSE_FILE = "compose_file"
    BUILD_CONTEXT_FILE = "build_context_file"


class BuildContextAssetManifest(BaseModel):
    """Provenance/integrity metadata for one Dockerfile/Compose/build-context file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: BuildContextAssetKind
    source: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _validate_source_path(self) -> BuildContextAssetManifest:
        validate_safe_relative_path(self.source)
        return self


class ToolImplementationManifest(BaseModel):
    """A symbolic reference to a tool implementation, resolved through a registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1, pattern=_SYMBOL_PATTERN)
    config: dict[str, JSONValue] = Field(default_factory=dict)


class CaseToolManifest(BaseModel):
    """A model-visible tool declaration bound to a symbolic implementation reference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    declaration: ToolDeclaration
    implementation: ToolImplementationManifest


class CaseSetupStepManifest(BaseModel):
    """One command run in a sandbox environment before a case's conversation starts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    environment: str | None = None
    argv: tuple[str, ...] | None = None
    shell_script: str | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_command(self) -> CaseSetupStepManifest:
        if (self.argv is None) == (self.shell_script is None):
            raise ValueError("Exactly one of 'argv' or 'shell_script' must be provided.")
        if self.argv is not None and not self.argv:
            raise ValueError("'argv' must contain at least one element.")
        return self


class CaseScorerManifest(BaseModel):
    """A symbolic reference to a suite scorer, resolved through a registry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str = Field(min_length=1, pattern=_SYMBOL_PATTERN)
    config: dict[str, JSONValue] = Field(default_factory=dict)
    required_environments: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_unique_required_environments(self) -> CaseScorerManifest:
        if len(self.required_environments) != len(set(self.required_environments)):
            raise ValueError(f"Scorer '{self.kind}' has duplicate required environment names.")
        return self


class LocalSandboxProviderManifestConfig(BaseModel):
    """The trusted local-development sandbox provider, tagged for discriminated parsing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_type: Literal["local"] = "local"
    config: LocalSandboxProviderConfig = Field(default_factory=LocalSandboxProviderConfig)


class DockerSandboxProviderManifestConfig(BaseModel):
    """The Docker/Compose sandbox provider, tagged for discriminated parsing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_type: Literal["docker"] = "docker"
    config: DockerSandboxProviderConfig
    build_context_assets: tuple[BuildContextAssetManifest, ...] = ()


class HyperVSandboxProviderManifestConfig(BaseModel):
    """The Hyper-V sandbox provider, tagged for discriminated parsing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_type: Literal["hyperv"] = "hyperv"
    config: HyperVSandboxProviderConfig


SandboxProviderManifest = Annotated[
    LocalSandboxProviderManifestConfig | DockerSandboxProviderManifestConfig | HyperVSandboxProviderManifestConfig,
    Field(discriminator="provider_type"),
]


class RunPolicyManifest(BaseModel):
    """Attempts/epochs/concurrency/retry policy shared by every case in a suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    epochs: int = Field(default=1, gt=0)
    attempts: int = Field(default=1, gt=0)
    max_retries: int = Field(default=0, ge=0)
    max_concurrency: int = Field(default=1, gt=0)
    retryable_error_codes: tuple[str, ...] = ()


class CapabilityCaseManifest(BaseModel):
    """One immutable, symbolic capability-task case within a suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str = Field(min_length=1, pattern=_SYMBOL_PATTERN)
    objective: str = Field(min_length=1)
    messages: tuple[CaseMessageManifest, ...] = ()
    modalities: tuple[PromptDataType, ...] = ()
    assets: tuple[CaseAssetManifest, ...] = ()
    tools: tuple[CaseToolManifest, ...] = ()
    sandbox_tools_prefix: str | None = None
    sandbox_tools_default_environment: str | None = None
    sandbox_tools_allowed_environments: tuple[str, ...] = ()
    sandbox_tools_default_user: str | None = None
    sandbox_tools_allow_user_override: bool = True
    sandbox_tools_include_file_tools: bool = True
    setup: tuple[CaseSetupStepManifest, ...] = ()
    limits: CapabilityLimits = Field(default_factory=CapabilityLimits)
    scorers: tuple[CaseScorerManifest, ...] = ()
    expected_evidence: tuple[ExpectedEvidence, ...] = ()
    source: CapabilitySource | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = Field(default_factory=dict)
    runnable: bool = True
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def _validate_unique_asset_ids(self) -> CapabilityCaseManifest:
        asset_ids = [asset.asset_id for asset in self.assets]
        if len(asset_ids) != len(set(asset_ids)):
            raise ValueError(f"Case '{self.case_id}' has duplicate asset_id values.")
        tool_names = [tool.declaration.name for tool in self.tools]
        if len(tool_names) != len(set(tool_names)):
            raise ValueError(f"Case '{self.case_id}' has duplicate tool declaration names.")
        if self.sandbox_tools_prefix is not None:
            reserved = {
                f"{self.sandbox_tools_prefix}_exec",
                f"{self.sandbox_tools_prefix}_read_file",
                f"{self.sandbox_tools_prefix}_write_file",
            }
            collisions = sorted(reserved.intersection(tool_names))
            if collisions:
                raise ValueError(
                    f"Case '{self.case_id}' custom tools collide with sandbox tool names: {', '.join(collisions)}."
                )
        elif (
            self.sandbox_tools_default_environment is not None
            or self.sandbox_tools_allowed_environments
            or self.sandbox_tools_default_user is not None
        ):
            raise ValueError("Sandbox tool environment restrictions require 'sandbox_tools_prefix'.")
        if len(self.sandbox_tools_allowed_environments) != len(set(self.sandbox_tools_allowed_environments)):
            raise ValueError(f"Case '{self.case_id}' has duplicate sandbox tool environment names.")
        if self.sandbox_tools_default_environment is not None:
            if not self.sandbox_tools_allowed_environments:
                raise ValueError("Sandbox tool default environment requires a non-empty allowed environment list.")
            if self.sandbox_tools_default_environment not in self.sandbox_tools_allowed_environments:
                raise ValueError("Sandbox tool default environment must be present in the allowed environment list.")
        if self.sandbox_tools_default_user is not None and self.sandbox_tools_include_file_tools:
            raise ValueError("Sandbox file tools must be disabled when a default execution user is configured.")
        if self.runnable == (self.unsupported_reason is not None):
            raise ValueError("Non-runnable cases require exactly one 'unsupported_reason'.")
        return self


class CapabilitySuiteManifest(BaseModel):
    """The top-level, versioned, immutable capability-suite manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=CURRENT_MANIFEST_SCHEMA_VERSION, ge=1)
    suite_id: str = Field(min_length=1, pattern=_SYMBOL_PATTERN)
    name: str = Field(min_length=1)
    description: str | None = None
    provenance: SuiteProvenance
    sandbox_provider: SandboxProviderManifest
    run_policy: RunPolicyManifest = Field(default_factory=RunPolicyManifest)
    cases: tuple[CapabilityCaseManifest, ...] = Field(min_length=1)
    tags: tuple[str, ...] = ()
    metadata: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_case_ids(self) -> CapabilitySuiteManifest:
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("CapabilitySuiteManifest has duplicate case_id values.")
        return self
