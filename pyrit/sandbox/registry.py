# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Sandbox provider discovery and instance registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrit.registry.instance_registry import DefaultInstanceRegistry, InstanceRegistry
from pyrit.registry.registry import Registry
from pyrit.registry.registry_metadata import RegistryMetadata

if TYPE_CHECKING:
    from types import ModuleType

    from pyrit.sandbox.contracts import SandboxProvider


@dataclass(frozen=True)
class SandboxProviderMetadata(RegistryMetadata):
    """Metadata describing a discoverable sandbox provider."""


class SandboxProviderRegistry(Registry["SandboxProvider", SandboxProviderMetadata]):
    """Discover, build, and hold sandbox provider instances."""

    def __init__(self, *, lazy_discovery: bool = True) -> None:
        """Initialize the provider registry."""
        super().__init__(lazy_discovery=lazy_discovery)
        self.instances: InstanceRegistry[SandboxProvider] = DefaultInstanceRegistry(instance_type=self._base_type)

    def _base_type(self) -> type[SandboxProvider]:
        """Return the provider contract."""
        from pyrit.sandbox.contracts import SandboxProvider

        return SandboxProvider

    def _discovery_package(self) -> ModuleType:
        """Return the sandbox package scanned for providers."""
        from pyrit import sandbox

        return sandbox

    def _metadata_class(self) -> type[SandboxProviderMetadata]:
        """Return sandbox provider metadata."""
        return SandboxProviderMetadata
