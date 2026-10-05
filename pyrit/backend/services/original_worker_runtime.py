# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Normal application lifespan ownership for the opt-in trusted cohost pilot."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import httpx

from pyrit.backend.services.original_evidence_admission import (
    install_original_evidence_provider,
    uninstall_original_evidence_provider,
)
from pyrit.backend.services.original_model_relay import OriginalModelRelay
from pyrit.backend.services.original_run_admission import (
    install_trusted_original_runner,
    uninstall_trusted_original_runner,
)
from pyrit.backend.services.original_worker_state import CohostValidationState
from pyrit.backend.services.original_worker_supervisor import OriginalWorkerSupervisor

if TYPE_CHECKING:
    from pyrit.backend.services.original_worker_preflight import CohostPreflight


class OriginalWorkerRuntime:
    """Own the single worker, relay and provider without a separate server/cookie harness."""

    INTERNAL_MODEL_PATH = re.compile(
        r"^/api/internal/original-model/[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}/(chat/completions|close)$"
    )

    def __init__(self, *, preflight: CohostPreflight) -> None:
        """Construct only native local helpers, not private Task/scorer classes."""
        self.preflight = preflight
        self.validation_state = None if preflight.config.local_test else CohostValidationState(preflight=preflight)
        self.relay = OriginalModelRelay(
            config=preflight.config.relay,
            preflight=preflight,
            client=httpx.AsyncClient(
                trust_env=False,
                follow_redirects=False,
                timeout=preflight.config.relay.request_timeout_seconds,
                limits=httpx.Limits(max_connections=1, max_keepalive_connections=1),
            ),
            validation_state=self.validation_state,
        )
        self.supervisor = OriginalWorkerSupervisor(
            preflight=preflight, relay=self.relay, validation_state=self.validation_state
        )
        self._installed = False

    async def startup_async(self) -> None:
        """Qualify staging/aggregate state, then install this lifespan's two authorities once."""
        if self.validation_state is not None:
            await self.validation_state.startup_async()
        await self.supervisor.startup_async()
        await self.relay.startup_async()
        install_original_evidence_provider(provider=self.supervisor)
        try:
            install_trusted_original_runner(runner=self.supervisor)
        except ValueError:
            uninstall_original_evidence_provider(provider=self.supervisor)
            raise
        self._installed = True

    def has_active_work(self) -> bool:
        """
        Keep runtime memory/credentials alive during original cleanup and real inflight drain.

        Returns:
            bool: Whether lifespan-owned work still exists.
        """
        return self.supervisor.has_active_work() or self.relay.has_active_work()

    async def shutdown_async(self) -> None:
        """Observe owned work before uninstalling authorities or disposing canonical memory."""
        try:
            try:
                await self.supervisor.shutdown_async()
            finally:
                await self.relay.shutdown_async()
        finally:
            if self._installed:
                uninstall_trusted_original_runner(runner=self.supervisor)
                uninstall_original_evidence_provider(provider=self.supervisor)
                self._installed = False
            if self.validation_state is not None:
                await self.validation_state.shutdown_async()

    @classmethod
    def is_internal_model_post(cls, *, method: str, path: str) -> bool:
        """
        Bypass Graph only for exact capability-authenticated POSTs.

        Returns:
            bool: Whether the path owns its separate evaluated-role authorization.
        """
        return method == "POST" and cls.INTERNAL_MODEL_PATH.fullmatch(path) is not None
