# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest

from pyrit.sandbox import (
    DockerSandboxProvider,
    DockerSandboxProviderConfig,
    DockerServiceBuildSpec,
    HyperVEnvironmentConfig,
    HyperVSandboxProvider,
    HyperVSandboxProviderConfig,
    HyperVSecretReference,
    LocalSandboxProvider,
)
from pyrit.scenario.capability_suite.manifest import (
    DockerSandboxProviderManifestConfig,
    HyperVSandboxProviderManifestConfig,
    LocalSandboxProviderManifestConfig,
)
from pyrit.scenario.capability_suite.registries import (
    CapabilitySuiteScorerFactoryRegistry,
    SandboxProviderFactoryRegistry,
    ToolImplementationFactoryRegistry,
    UnknownRegistryKeyError,
    build_default_sandbox_provider_registry,
    build_default_scorer_registry,
)
from pyrit.scenario.capability_suite.scorers import (
    ResultOnlyScorerAdapter,
    SandboxCommandScorer,
    SandboxFileScorer,
)


def test_sandbox_provider_registry_unknown_key_raises() -> None:
    registry = SandboxProviderFactoryRegistry()
    with pytest.raises(UnknownRegistryKeyError):
        registry.build(LocalSandboxProviderManifestConfig())


def test_sandbox_provider_registry_rejects_duplicate_registration() -> None:
    registry = SandboxProviderFactoryRegistry()
    registry.register(provider_type="local", factory=lambda config: LocalSandboxProvider())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(provider_type="local", factory=lambda config: LocalSandboxProvider())


def test_build_default_sandbox_provider_registry_builds_local() -> None:
    registry = build_default_sandbox_provider_registry()
    provider = registry.build(LocalSandboxProviderManifestConfig())
    assert isinstance(provider, LocalSandboxProvider)


def test_build_default_sandbox_provider_registry_builds_docker(tmp_path) -> None:
    registry = build_default_sandbox_provider_registry()
    build_spec = DockerServiceBuildSpec(service_name="svc", build_context=tmp_path)
    config = DockerSandboxProviderManifestConfig(config=DockerSandboxProviderConfig(services=(build_spec,)))
    provider = registry.build(config)
    assert isinstance(provider, DockerSandboxProvider)


def test_build_default_sandbox_provider_registry_builds_hyperv() -> None:
    registry = build_default_sandbox_provider_registry()
    config = HyperVSandboxProviderManifestConfig(
        config=HyperVSandboxProviderConfig(
            environments=(
                HyperVEnvironmentConfig(
                    name="default",
                    template_vm="template",
                    credential=HyperVSecretReference(secret_id="cred"),
                ),
            )
        )
    )
    provider = registry.build(config)
    assert isinstance(provider, HyperVSandboxProvider)


def test_tool_implementation_registry_unknown_key_raises() -> None:
    registry = ToolImplementationFactoryRegistry()
    with pytest.raises(UnknownRegistryKeyError):
        registry.build(kind="missing", config={}, session=object())  # type: ignore[arg-type]


def test_tool_implementation_registry_rejects_duplicate_registration() -> None:
    registry = ToolImplementationFactoryRegistry()
    registry.register(kind="echo", factory=lambda config, session: object())
    with pytest.raises(ValueError, match="already registered"):
        registry.register(kind="echo", factory=lambda config, session: object())


def test_scorer_registry_unknown_key_raises() -> None:
    registry = CapabilitySuiteScorerFactoryRegistry()
    with pytest.raises(UnknownRegistryKeyError):
        registry.build(kind="missing", config={})


def test_build_default_scorer_registry_resolves_all_builtin_kinds(patch_central_database) -> None:
    registry = build_default_scorer_registry()

    text_match = registry.build(kind="text_match", config={"expected_value": "done"})
    assert isinstance(text_match, ResultOnlyScorerAdapter)

    tool_evidence = registry.build(kind="tool_evidence", config={"tool_name": "lookup"})
    assert isinstance(tool_evidence, ResultOnlyScorerAdapter)

    sandbox_file = registry.build(kind="sandbox_file", config={"path": "out.txt"})
    assert isinstance(sandbox_file, SandboxFileScorer)

    sandbox_command = registry.build(kind="sandbox_command", config={"argv": ["echo", "hi"]})
    assert isinstance(sandbox_command, SandboxCommandScorer)
