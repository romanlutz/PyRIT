# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Provider-neutral identities and observations for a run-owned environment."""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class EnvironmentCapability(str, Enum):
    """Optional environment operations, independent of service roles or provider technology."""

    SETUP = "setup"
    HEALTH_CHECK = "health_check"
    DYNAMIC_SERVICES = "dynamic_services"
    NETWORK_PHASE_TRANSITIONS = "network_phase_transitions"


class EnvironmentLeaseState(str, Enum):
    """Acquisition and cleanup state, not a task outcome."""

    NEW = "new"
    ACQUIRING = "acquiring"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"
    CLEANUP_FAILED = "cleanup_failed"


class EnvironmentCleanupStatus(str, Enum):
    """Whether the provider confirmed release of one exact owned resource."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"


class EnvironmentResourceHandle(BaseModel):
    """An opaque provider-scoped resource reserved for one run, not a command or endpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    resource_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)


class EnvironmentServiceHandle(BaseModel):
    """A named service with open-ended roles; multiple services may share one owned resource."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    roles: frozenset[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    resource: EnvironmentResourceHandle
    parent_name: str | None = Field(default=None, min_length=1)


class EnvironmentHealth(BaseModel):
    """A provider observation; unknown health is not successful health."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    service_name: str = Field(min_length=1)
    healthy: bool | None
    reason: str


class EnvironmentResourceCleanup(BaseModel):
    """Cleanup evidence for one reserved resource, including failed acquisitions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource: EnvironmentResourceHandle
    acquisition_completed: bool
    status: EnvironmentCleanupStatus
    error: str | None = None


class EnvironmentLeaseSnapshot(BaseModel):
    """In-memory lifecycle metadata, not raw logs, runtime access, or a durable episode schema."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    lease_id: str
    state: EnvironmentLeaseState
    capabilities: frozenset[EnvironmentCapability]
    services: tuple[EnvironmentServiceHandle, ...]
    resources: tuple[EnvironmentResourceCleanup, ...]
    setup_completed: bool
    health: tuple[EnvironmentHealth, ...]
    errors: tuple[str, ...]
