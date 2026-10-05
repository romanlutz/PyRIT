# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Read-only exact-owned native sandbox closure, separate from original scoring."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from time import monotonic
from typing import TYPE_CHECKING

from pyrit.backend.services.original_worker_preflight import CohostPreflightError

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic import JsonValue

    from pyrit.backend.models.original_worker import CohostSandboxConfig
    from pyrit.backend.services.original_worker_preflight import CohostPreflight


class OriginalSandboxObserver:
    """Observe only this run; never delete/recreate a sandbox, snapshot, group or resource group."""

    REQUIRED_TRUE = (
        "sandbox_absent",
        "snapshots_absent",
        "inner_absent",
        "owned_child_artifacts_absent",
        "commands_terminal",
        "cleanup_scope_exact_owned_only",
        "within_cleanup_reserve",
        "model_capability_revoked",
        "model_upstream_drained",
        "shared_resource_group_retained",
        "shared_sandbox_group_retained",
        "owned_snapshot_inventory_empty",
    )

    def __init__(self, *, preflight: CohostPreflight) -> None:
        """Use the frozen exact group/client, with no evaluated-guest credentials."""
        self.preflight = preflight
        self.tasks: set[asyncio.Task[dict[str, JsonValue]]] = set()

    async def verify_async(
        self, *, run_id: UUID, closure: dict[str, JsonValue], timeout_seconds: float
    ) -> dict[str, JsonValue]:
        """
        Retain the SDK thread until its actual end.

        Returns:
            dict[str, JsonValue]: Independent scoped absence, not source-score authority.
        """
        self._validate_closure(run_id=run_id, closure=closure)
        self.preflight.verify_identity_environment()
        task = asyncio.create_task(
            asyncio.to_thread(self._observe, run_id=run_id, closure=closure, timeout_seconds=timeout_seconds)
        )
        self.tasks.add(task)
        task.add_done_callback(self._observed)
        async with asyncio.timeout(timeout_seconds):
            return await asyncio.shield(task)

    def _observed(self, task: asyncio.Task[dict[str, JsonValue]]) -> None:
        self.tasks.discard(task)
        if not task.cancelled():
            task.exception()

    def _validate_closure(self, *, run_id: UUID, closure: dict[str, JsonValue]) -> None:
        if (
            closure.get("schema") != "cohost-owned-sandbox-cleanup/v1"
            or closure.get("run_id") != str(run_id)
            or any(closure.get(key) is not True for key in self.REQUIRED_TRUE)
            or closure.get("errors") != []
            or closure.get("unresolved_operation_ids") != []
            or type(closure.get("create_attempted")) is not bool
            or closure.get("create_outcome_known") is not True
        ):
            raise CohostPreflightError("Original source has not proved exact terminal physical cleanup.")
        owned = self._owned_ids(closure.get("owned_sandbox_ids"))
        self._owned_ids(closure.get("owned_snapshot_ids"))
        if len(owned) > 1 or (closure["create_attempted"] is True and len(owned) != 1):
            raise CohostPreflightError("Original run exceeded one allocation or has an unknown create outcome.")
        if closure["create_attempted"] is False and owned:
            raise CohostPreflightError("Preexisting matching labels do not establish ownership.")
        config = self.preflight.config.sandbox
        if config is not None and (
            closure.get("resource_group") != config.resource_group
            or closure.get("sandbox_group") != config.sandbox_group
        ):
            raise CohostPreflightError("Original cleanup scope differs from the startup-owned group.")

    def _observe(self, *, run_id: UUID, closure: dict[str, JsonValue], timeout_seconds: float) -> dict[str, JsonValue]:
        if self.preflight.config.local_test:
            if closure["create_attempted"] is not False:
                raise CohostPreflightError("Public local fixtures cannot claim or create a hosted sandbox.")
            return {"authority": "public-no-sandbox-fixture", "run_id": str(run_id), "sdk_calls_drained": True}
        config = self.preflight.config.sandbox
        if config is None:
            raise CohostPreflightError("Independent native sandbox observation is unavailable.")
        return self._native_observe(config=config, run_id=run_id, closure=closure, timeout_seconds=timeout_seconds)

    def _native_observe(
        self,
        *,
        config: CohostSandboxConfig,
        run_id: UUID,
        closure: dict[str, JsonValue],
        timeout_seconds: float,
    ) -> dict[str, JsonValue]:
        from azure.containerapps.sandbox import SandboxGroupClient
        from azure.core.exceptions import AzureError, HttpResponseError
        from azure.identity import ManagedIdentityCredential

        owned = self._owned_ids(closure.get("owned_sandbox_ids"))
        snapshots = self._owned_ids(closure.get("owned_snapshot_ids"))
        observations: list[JsonValue] = []
        deadline = monotonic() + timeout_seconds
        credential = ManagedIdentityCredential(client_id=str(self.preflight.config.managed_identity_client_id))
        try:
            with SandboxGroupClient(
                config.endpoint,
                credential,
                subscription_id=str(config.subscription_id),
                resource_group=config.resource_group,
                sandbox_group=config.sandbox_group,
                connection_timeout=15,
                read_timeout=min(30, timeout_seconds),
                retry_total=0,
            ) as client:
                for sandbox_id in owned:
                    self._require_remaining(deadline)
                    try:
                        client.get_sandbox(sandbox_id)
                    except HttpResponseError as error:
                        if error.status_code != 404:
                            raise
                    else:
                        raise CohostPreflightError("An exact owned evaluated sandbox still exists.")
                    observations.append({"sandbox_id": sandbox_id, "http_status": 404})
                # Fully consume every page; do not infer absence from a first page.
                self._require_remaining(deadline)
                inventory = iter(client.list_sandboxes(labels={"pilotRunId": str(run_id), "purpose": config.purpose}))
                while True:
                    self._require_remaining(deadline)
                    try:
                        next(inventory)
                    except StopIteration:
                        break
                    raise CohostPreflightError("The owned run's sandbox label inventory is not empty.")
                for snapshot_id in snapshots:
                    self._require_remaining(deadline)
                    try:
                        client.get_snapshot(snapshot_id)
                    except HttpResponseError as error:
                        if error.status_code != 404:
                            raise
                    else:
                        raise CohostPreflightError("An exact owned evaluated snapshot still exists.")
                    observations.append({"snapshot_id": snapshot_id, "http_status": 404})
                self._require_remaining(deadline)
                snapshot_inventory = iter(client.list_snapshots())
                while True:
                    self._require_remaining(deadline)
                    try:
                        snapshot = next(snapshot_inventory)
                    except StopIteration:
                        break
                    if snapshot.sandbox_id in owned or snapshot.id in snapshots:
                        raise CohostPreflightError("Owned evaluated snapshot inventory is not empty.")
                self._require_remaining(deadline)
        except AzureError as error:
            raise CohostPreflightError(
                "Native owned-resource observation failed; only exact404 proves absence."
            ) from error
        finally:
            credential.close()
        return {
            "authority": "native-backend-exact-owned-sdk-readback",
            "run_id": str(run_id),
            "resource_group": config.resource_group,
            "sandbox_group": config.sandbox_group,
            "observations": observations,
            "owned_inventory_empty": True,
            "foreign_resources_retained": True,
            "sdk_calls_drained": True,
            "observed_at": datetime.now(UTC).isoformat(),
        }

    @staticmethod
    def _require_remaining(deadline: float) -> None:
        if monotonic() >= deadline:
            raise CohostPreflightError("Native observation cannot issue another SDK read after the cleanup deadline.")

    @staticmethod
    def _owned_ids(value: JsonValue) -> tuple[str, ...]:
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", item) for item in value
        ):
            raise CohostPreflightError("Original owned child IDs are not exact bounded native identifiers.")
        ids = tuple(item for item in value if isinstance(item, str))
        if len(set(ids)) != len(ids):
            raise CohostPreflightError("Original owned child identifiers are duplicated.")
        return ids
