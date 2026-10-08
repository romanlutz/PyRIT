# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host-installed service credentials and scoped operator delegation, never browser input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class EvaluationWorkerAuthContext:
    """The audience and exact HTTP operation to authorize using host-owned credentials."""

    audience: str
    actor_id: str
    method: str
    path: str
    body_sha256: str
    job_id: UUID | None = None
    request_sha256: str | None = None
    gateway_fence_id: UUID | None = None
    binding_sha256: str | None = None


@dataclass(frozen=True, kw_only=True)
class EvaluationWorkerCredentials:
    """Two separately verified credentials; their contents must never be logged or persisted."""

    service_token: str = field(repr=False)
    operator_delegation: str = field(repr=False)

    def headers(self) -> dict[str, str]:
        """
        Expose credentials only to the explicitly configured worker authority.

        Returns:
            dict[str, str]: Service and scoped-delegation authentication headers.

        Raises:
            ValueError: If either credential is empty, too long, or multiline.
        """
        if any(
            not 1 <= len(value) <= 16384 or "\r" in value or "\n" in value
            for value in (self.service_token, self.operator_delegation)
        ):
            raise ValueError("Worker credentials must be bounded, nonempty, single-line values.")
        return {
            "Authorization": f"Bearer {self.service_token}",
            "X-PyRIT-Operator-Delegation": self.operator_delegation,
        }


class EvaluationWorkerCredentialProvider(Protocol):
    """A reviewed host binding, not an Entra/OBO implementation or actor-header shortcut."""

    audience: str
    fixture_only: bool

    async def credentials_async(self, *, context: EvaluationWorkerAuthContext) -> EvaluationWorkerCredentials:
        """Supply service identity and verified, operation-bound operator delegation."""
        ...

    async def close_async(self) -> None:
        """Close only credentials owned by this installed provider."""
        ...


class EvaluationWorkerCredentialRegistry:
    """Empty by default; named providers must be deliberately installed by the host."""

    _factories: dict[str, Callable[[str], EvaluationWorkerCredentialProvider]] = {}

    @classmethod
    def register(cls, *, name: str, factory: Callable[[str], EvaluationWorkerCredentialProvider]) -> None:
        """
        Install one provider without arbitrary import or ambient credential discovery.

        Raises:
            ValueError: If the name is invalid or already installed.
        """
        if not name or len(name) > 128 or name in cls._factories:
            raise ValueError("Worker credential provider names must be bounded and unique.")
        cls._factories[name] = factory

    @classmethod
    def create(cls, *, name: str, audience: str) -> EvaluationWorkerCredentialProvider:
        """
        Refuse identity/delegation until a reviewed provider is installed.

        Returns:
            EvaluationWorkerCredentialProvider: The explicitly audience-bound host provider.

        Raises:
            ValueError: If provider installation or audience binding is absent.
        """
        factory = cls._factories.get(name)
        if factory is None:
            raise ValueError("No reviewed worker service identity/delegation provider is installed.")
        provider = factory(audience)
        if provider.audience != audience:
            raise ValueError("Installed worker credentials are bound to another audience.")
        return provider
