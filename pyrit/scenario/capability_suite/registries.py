# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Explicit, injected factory registries for capability-suite resolution.

A manifest never carries a Python import path — only symbolic string keys (a sandbox
``provider_type``, a tool ``implementation.kind``, a scorer ``kind``). Resolving those
keys into real objects always goes through one of the registries below, which the
caller populates explicitly (opt-in) before running a suite. This is the seam that
keeps manifest loading free of arbitrary imports or code execution.

``build_default_*_registry`` helpers pre-register PyRIT's own built-in
providers/scorers for convenience, but callers must call them explicitly -- nothing
here runs at import time, and a caller may always build an empty registry and
register only what they choose to allow.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.executor.capability import CapabilityEvidenceSink, ToolImplementation
    from pyrit.models import JSONValue
    from pyrit.sandbox import SandboxProvider, SandboxSession
    from pyrit.scenario.capability_suite.manifest import (
        DockerSandboxProviderManifestConfig,
        SandboxProviderManifest,
    )
    from pyrit.scenario.capability_suite.scorers import CapabilitySuiteScorer


class UnknownRegistryKeyError(KeyError):
    """Raised when a manifest references a symbolic key with no registered factory."""

    def __init__(self, *, registry_name: str, key: str, known_keys: tuple[str, ...]) -> None:
        """Initialize the error with the offending registry, key, and known alternatives."""
        known = ", ".join(sorted(known_keys)) or "(none)"
        message = f"{registry_name} has no factory registered for '{key}'. Known keys: {known}."
        super().__init__(message)
        self.registry_name = registry_name
        self.key = key
        self.known_keys = known_keys


SandboxProviderFactory = Callable[["SandboxProviderManifest"], "SandboxProvider"]


class SandboxProviderFactoryRegistry:
    """Resolve a manifest's symbolic ``provider_type`` into a live ``SandboxProvider``."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._factories: dict[str, SandboxProviderFactory] = {}

    def register(self, *, provider_type: str, factory: SandboxProviderFactory) -> None:
        """
        Register a factory for one provider type.

        Raises:
            ValueError: If ``provider_type`` is already registered.
        """
        if provider_type in self._factories:
            raise ValueError(f"Sandbox provider factory for '{provider_type}' is already registered.")
        self._factories[provider_type] = factory

    def build(self, config: SandboxProviderManifest) -> SandboxProvider:
        """
        Build a ``SandboxProvider`` for the manifest's declared provider type.

        Returns:
            SandboxProvider: A newly constructed, unprepared provider.

        Raises:
            UnknownRegistryKeyError: If no factory is registered for the provider type.
        """
        factory = self._factories.get(config.provider_type)
        if factory is None:
            raise UnknownRegistryKeyError(
                registry_name="SandboxProviderFactoryRegistry",
                key=config.provider_type,
                known_keys=tuple(self._factories),
            )
        return factory(config)


def build_default_sandbox_provider_registry(
    *, evidence_sink: CapabilityEvidenceSink | None = None
) -> SandboxProviderFactoryRegistry:
    """
    Build a registry with PyRIT's built-in local/docker/hyperv providers pre-registered.

    This is an explicit, opt-in convenience. Callers may instead build an empty
    ``SandboxProviderFactoryRegistry`` and register only the providers they want to
    allow a manifest to select.

    Returns:
        SandboxProviderFactoryRegistry: A registry with all three built-in providers.
    """
    from pyrit.scenario.capability_suite.manifest import (
        DockerSandboxProviderManifestConfig,
        HyperVSandboxProviderManifestConfig,
        LocalSandboxProviderManifestConfig,
    )

    registry = SandboxProviderFactoryRegistry()

    def _build_local(config: SandboxProviderManifest) -> SandboxProvider:
        from pyrit.sandbox import LocalSandboxProvider

        assert isinstance(config, LocalSandboxProviderManifestConfig)
        return LocalSandboxProvider(config=config.config, evidence_sink=evidence_sink)

    def _build_docker(config: SandboxProviderManifest) -> SandboxProvider:
        from pyrit.sandbox import DockerSandboxProvider

        assert isinstance(config, DockerSandboxProviderManifestConfig)
        _verify_docker_build_context(config)
        return DockerSandboxProvider(config=config.config, evidence_sink=evidence_sink)

    def _build_hyperv(config: SandboxProviderManifest) -> SandboxProvider:
        from pyrit.sandbox import HyperVSandboxProvider

        assert isinstance(config, HyperVSandboxProviderManifestConfig)
        return HyperVSandboxProvider(config=config.config, evidence_sink=evidence_sink)

    registry.register(provider_type="local", factory=_build_local)
    registry.register(provider_type="docker", factory=_build_docker)
    registry.register(provider_type="hyperv", factory=_build_hyperv)
    return registry


def _verify_docker_build_context(config: DockerSandboxProviderManifestConfig) -> None:
    if not config.build_context_assets:
        return
    root = config.config.project_context
    if root is None:
        raise ValueError("Docker build-context integrity records require config.project_context.")
    resolved_root = root.resolve()
    for asset in config.build_context_assets:
        path = (resolved_root / asset.source).resolve()
        if resolved_root not in path.parents:
            raise ValueError(f"Docker build-context asset '{asset.source}' escapes project_context.")
        if not path.is_file():
            raise ValueError(f"Docker build-context asset '{asset.source}' does not exist.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != asset.sha256:
            raise ValueError(
                f"Docker build-context asset '{asset.source}' sha256 mismatch: expected {asset.sha256}, got {actual}."
            )


ToolImplementationFactory = Callable[[Mapping[str, "JSONValue"], "SandboxSession"], "ToolImplementation"]


class ToolImplementationFactoryRegistry:
    """Resolve a manifest's symbolic tool ``implementation.kind`` into a ``ToolImplementation``."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._factories: dict[str, ToolImplementationFactory] = {}

    def register(self, *, kind: str, factory: ToolImplementationFactory) -> None:
        """
        Register a factory for one implementation kind.

        Raises:
            ValueError: If ``kind`` is already registered.
        """
        if kind in self._factories:
            raise ValueError(f"Tool implementation factory for '{kind}' is already registered.")
        self._factories[kind] = factory

    def build(self, *, kind: str, config: Mapping[str, JSONValue], session: SandboxSession) -> ToolImplementation:
        """
        Build a ``ToolImplementation`` for a symbolic kind bound to a live session.

        Returns:
            ToolImplementation: The resolved tool implementation.

        Raises:
            UnknownRegistryKeyError: If no factory is registered for ``kind``.
        """
        factory = self._factories.get(kind)
        if factory is None:
            raise UnknownRegistryKeyError(
                registry_name="ToolImplementationFactoryRegistry",
                key=kind,
                known_keys=tuple(self._factories),
            )
        return factory(config, session)


ScorerFactory = Callable[[Mapping[str, "JSONValue"]], "CapabilitySuiteScorer"]


class CapabilitySuiteScorerFactoryRegistry:
    """Resolve a manifest's symbolic scorer ``kind`` into a ``CapabilitySuiteScorer``."""

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._factories: dict[str, ScorerFactory] = {}

    def register(self, *, kind: str, factory: ScorerFactory) -> None:
        """
        Register a factory for one scorer kind.

        Raises:
            ValueError: If ``kind`` is already registered.
        """
        if kind in self._factories:
            raise ValueError(f"Scorer factory for '{kind}' is already registered.")
        self._factories[kind] = factory

    def build(self, *, kind: str, config: Mapping[str, JSONValue]) -> CapabilitySuiteScorer:
        """
        Build a ``CapabilitySuiteScorer`` for a symbolic kind.

        Returns:
            CapabilitySuiteScorer: The resolved scorer.

        Raises:
            UnknownRegistryKeyError: If no factory is registered for ``kind``.
        """
        factory = self._factories.get(kind)
        if factory is None:
            raise UnknownRegistryKeyError(
                registry_name="CapabilitySuiteScorerFactoryRegistry",
                key=kind,
                known_keys=tuple(self._factories),
            )
        return factory(config)


def build_default_scorer_registry() -> CapabilitySuiteScorerFactoryRegistry:
    """
    Build a registry with PyRIT's built-in native scorer kinds pre-registered.

    Registers ``"text_match"``, ``"tool_evidence"``, ``"sandbox_file"``,
    ``"sandbox_command"``, and ``"sandbox_state_match"``. To compose an *existing*
    ``pyrit.score.Scorer`` (the
    "existing scorer adapter" seam), register an additional kind whose factory wraps
    an already-constructed scorer, e.g.::

        from pyrit.executor.capability import MessageScorerAdapter
        from pyrit.scenario.capability_suite.scorers import ResultOnlyScorerAdapter

        registry.register(
            kind="my_existing_scorer",
            factory=lambda config: ResultOnlyScorerAdapter(
                scorer=MessageScorerAdapter(scorer=my_already_built_scorer)
            ),
        )

    Returns:
        CapabilitySuiteScorerFactoryRegistry: A registry with all built-in native scorers.
    """
    from pyrit.scenario.capability_suite.code_evaluation import CodeEvaluationScorer
    from pyrit.scenario.capability_suite.scorers import (
        ResultOnlyScorerAdapter,
        SandboxCommandScorer,
        SandboxFileScorer,
        SandboxStateMatchScorer,
        TextMatchScorer,
        ToolEvidenceScorer,
    )

    registry = CapabilitySuiteScorerFactoryRegistry()
    registry.register(kind="code_evaluation", factory=lambda config: CodeEvaluationScorer.from_config(config))
    registry.register(
        kind="text_match",
        factory=lambda config: ResultOnlyScorerAdapter(scorer=TextMatchScorer.from_config(config)),
    )
    registry.register(
        kind="tool_evidence",
        factory=lambda config: ResultOnlyScorerAdapter(scorer=ToolEvidenceScorer.from_config(config)),
    )
    registry.register(kind="sandbox_file", factory=lambda config: SandboxFileScorer.from_config(config))
    registry.register(kind="sandbox_command", factory=lambda config: SandboxCommandScorer.from_config(config))
    registry.register(
        kind="sandbox_state_match",
        factory=lambda config: SandboxStateMatchScorer.from_config(config),
    )
    return registry
