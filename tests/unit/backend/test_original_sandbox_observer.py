# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Read-only native physical closure cannot turn authorization or uncertain threads into absence."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest
from azure.containerapps.sandbox import SandboxGroupClient
from azure.core.exceptions import HttpResponseError
from azure.identity import ManagedIdentityCredential

from pyrit.backend.models.original_worker import CohostSandboxConfig
from pyrit.backend.services.original_sandbox_observer import OriginalSandboxObserver
from pyrit.backend.services.original_worker_preflight import CohostPreflight, CohostPreflightError

if TYPE_CHECKING:
    from pydantic import JsonValue

    from pyrit.backend.models.original_worker import CohostBackendConfig


def _closure(*, run_id: UUID, created: bool) -> dict[str, JsonValue]:
    return {
        "schema": "cohost-owned-sandbox-cleanup/v1",
        "run_id": str(run_id),
        **dict.fromkeys(OriginalSandboxObserver.REQUIRED_TRUE, True),
        "errors": [],
        "unresolved_operation_ids": [],
        "create_attempted": created,
        "create_outcome_known": True,
        "owned_sandbox_ids": ["public-owned"] if created else [],
        "owned_snapshot_ids": ["public-snapshot"] if created else [],
        "resource_group": "public-fixture",
        "sandbox_group": "public-group",
    }


@pytest.mark.parametrize("status", [404, 403, 500, None])
def test_native_only_exact_404_is_absence_and_foreign_snapshots_are_retained(
    *, worker_config: CohostBackendConfig, status: int | None
) -> None:
    sandbox = CohostSandboxConfig(
        endpoint="https://management.westus2.azuredevcompute.io",
        subscription_id=uuid4(),
        resource_group="public-fixture",
        sandbox_group="public-group",
    )
    config = worker_config.model_copy(update={"sandbox": sandbox})
    observer = OriginalSandboxObserver(preflight=CohostPreflight(config=config, environment={}))
    run_id = uuid4()
    closure = _closure(run_id=run_id, created=True)
    client = MagicMock(spec=SandboxGroupClient)
    client.__enter__.return_value = client
    failure = HttpResponseError("Public native mock")
    failure.status_code = status
    client.get_sandbox.side_effect = failure
    client.get_snapshot.side_effect = failure
    client.list_sandboxes.return_value = []
    client.list_snapshots.return_value = [SimpleNamespace(id="foreign-snapshot", sandbox_id="foreign-sandbox")]
    credential = MagicMock(spec=ManagedIdentityCredential)
    with (
        patch("azure.containerapps.sandbox.SandboxGroupClient", return_value=client),
        patch("azure.identity.ManagedIdentityCredential", return_value=credential) as construct,
    ):
        if status == 404:
            receipt = observer._native_observe(config=sandbox, run_id=run_id, closure=closure, timeout_seconds=5)
            assert receipt["owned_inventory_empty"] is True and receipt["foreign_resources_retained"] is True
            assert receipt["sdk_calls_drained"] is True
            assert client.get_sandbox.call_args.args == ("public-owned",)
            assert client.get_snapshot.call_args.args == ("public-snapshot",)
        else:
            with pytest.raises(CohostPreflightError, match="only exact404"):
                observer._native_observe(config=sandbox, run_id=run_id, closure=closure, timeout_seconds=5)
        assert construct.call_args.kwargs["client_id"] == str(config.managed_identity_client_id)
    credential.close.assert_called_once()
    client.delete_sandbox.assert_not_called()
    client.delete_snapshot.assert_not_called()


def test_expired_observation_does_not_issue_another_sdk_read(worker_config: CohostBackendConfig) -> None:
    sandbox = CohostSandboxConfig(
        endpoint="https://management.westus2.azuredevcompute.io",
        subscription_id=uuid4(),
        resource_group="public-fixture",
        sandbox_group="public-group",
    )
    observer = OriginalSandboxObserver(preflight=CohostPreflight(config=worker_config, environment={}))
    client = MagicMock(spec=SandboxGroupClient)
    client.__enter__.return_value = client
    credential = MagicMock(spec=ManagedIdentityCredential)
    run_id = uuid4()
    with (
        patch("azure.containerapps.sandbox.SandboxGroupClient", return_value=client),
        patch("azure.identity.ManagedIdentityCredential", return_value=credential),
    ):
        with pytest.raises(CohostPreflightError, match="deadline"):
            observer._native_observe(
                config=sandbox, run_id=run_id, closure=_closure(run_id=run_id, created=True), timeout_seconds=0
            )
    client.get_sandbox.assert_not_called()
    client.list_sandboxes.assert_not_called()
    credential.close.assert_called_once()


async def test_caller_timeout_retains_actual_sdk_thread_until_it_really_finishes_async(
    worker_config: CohostBackendConfig,
) -> None:
    entered, finish = Event(), Event()
    run_id = uuid4()
    observer = OriginalSandboxObserver(preflight=CohostPreflight(config=worker_config, environment={}))

    def observe(*, run_id: UUID, closure: dict[str, JsonValue], timeout_seconds: float) -> dict[str, JsonValue]:
        entered.set()
        if not finish.wait(5):
            raise TimeoutError("Public fixture SDK thread deadline.")
        return {"run_id": str(run_id), "sdk_calls_drained": True}

    with patch.object(observer, "_observe", new=observe):
        try:
            with pytest.raises(TimeoutError):
                await observer.verify_async(
                    run_id=run_id, closure=_closure(run_id=run_id, created=False), timeout_seconds=0.02
                )
            assert await asyncio.to_thread(entered.wait, 1)
            assert len(observer.tasks) == 1
        finally:
            finish.set()
            await asyncio.gather(*tuple(observer.tasks))
    assert observer.tasks == set()
