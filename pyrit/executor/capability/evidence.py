# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Evidence sinks for provider-neutral capability execution."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pyrit.executor.capability.models import CapabilityEvidence


class CapabilityEvidenceSink(Protocol):
    """A destination for authoritative capability evidence events."""

    async def emit_async(self, evidence: CapabilityEvidence) -> None:
        """Store one immutable evidence event."""


class InMemoryCapabilityEvidenceSink:
    """A concurrency-safe in-memory evidence sink for composition and tests."""

    def __init__(self) -> None:
        """Initialize an empty sink."""
        self._evidence: list[CapabilityEvidence] = []
        self._lock = asyncio.Lock()

    async def emit_async(self, evidence: CapabilityEvidence) -> None:
        """Store one evidence event."""
        async with self._lock:
            self._evidence.append(evidence)

    async def snapshot_async(self) -> tuple[CapabilityEvidence, ...]:
        """Return the evidence emitted so far."""
        async with self._lock:
            return tuple(self._evidence)
