# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Mock native credential use, refresh, standard memory clients and fail-closed startup identity."""

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
from azure.core.credentials import AccessToken
from azure.core.credentials_async import AsyncTokenCredential
from azure.storage.blob.aio import ContainerClient
from sqlalchemy.ext.asyncio import AsyncEngine

from pyrit.auth.azure_auth import AsyncAzureAuth
from pyrit.auth.azure_token_observer import ObservedAsyncTokenCredential
from pyrit.backend.models.original_worker import CohostBackendConfig, CohostSandboxConfig
from pyrit.backend.services.original_native_identity import OriginalNativeIdentity
from pyrit.memory import AzureSQLMemory
from pyrit.memory.storage import AzureBlobStorageIO

if TYPE_CHECKING:
    from collections.abc import Callable

    from pydantic import JsonValue

_TENANT = UUID("00000000-0000-4000-8000-000000000006")
_OBJECT = UUID("00000000-0000-4000-8000-000000000009")
_SUBSCRIPTION = UUID("00000000-0000-4000-8000-000000000010")


@pytest.fixture
def identity(worker_config: CohostBackendConfig) -> OriginalNativeIdentity:
    config = worker_config.model_copy(
        update={
            "sandbox": CohostSandboxConfig(
                endpoint="https://management.westus2.azuredevcompute.io",
                subscription_id=_SUBSCRIPTION,
                resource_group="public-owned-preview",
                sandbox_group="public-sandbox",
            )
        }
    )
    return OriginalNativeIdentity(config=config, tenant_id=_TENANT, verify_environment=lambda: None)


def _token(*, identity: OriginalNativeIdentity, path: str, changes: dict[str, JsonValue] | None = None) -> AccessToken:
    claims: dict[str, JsonValue] = {
        "appid": str(identity._config.managed_identity_client_id),
        "oid": str(_OBJECT),
        "tid": str(_TENANT),
        "aud": identity.SCOPES[path].removesuffix(".default"),
        "xms_mirid": (
            f"/subscriptions/{_SUBSCRIPTION}/resourceGroups/public-owned-preview/"
            "providers/Microsoft.ManagedIdentity/userAssignedIdentities/public-runtime"
        ),
    }
    claims.update(changes or {})
    segment = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return AccessToken(f"public-fixture.{segment}.not-a-signature", int(datetime.now(UTC).timestamp()) + 3600)


@pytest.mark.parametrize("path", list(OriginalNativeIdentity.SCOPES))
def test_native_claims_are_bound_to_owned_client_tenant_scope_and_group(
    *, identity: OriginalNativeIdentity, path: str
) -> None:
    token = _token(identity=identity, path=path)
    safe = identity.observe(access_token=token, scope=identity.SCOPES[path], path=path)
    assert safe["client_id"] == str(identity._config.managed_identity_client_id)
    assert safe["oid"] == str(_OBJECT) and safe["tid"] == str(_TENANT)
    serialized = json.dumps(identity.snapshot())
    assert token.token not in serialized
    assert token.token.split(".")[1] not in serialized
    assert "not-a-signature" not in serialized


@pytest.mark.parametrize(
    "changes",
    [
        {"appid": str(_OBJECT)},
        {"oid": "not-a-guid"},
        {"tid": str(_OBJECT)},
        {"aud": "https://graph.microsoft.com"},
        {"xms_mirid": None},
        {"xms_mirid": "/subscriptions/foreign/resourceGroups/shared"},
        {
            "xms_mirid": (
                f"/subscriptions/{_SUBSCRIPTION}/resourceGroups/shared/"
                "providers/Microsoft.ManagedIdentity/userAssignedIdentities/public-runtime"
            )
        },
    ],
)
def test_native_client_rejects_fallback_or_unowned_identity_without_evidence(
    *, identity: OriginalNativeIdentity, changes: dict[str, JsonValue]
) -> None:
    with pytest.raises(ValueError, match="identity_mismatch"):
        identity.observe(
            access_token=_token(identity=identity, path="sql_sync", changes=changes),
            scope=identity.SCOPES["sql_sync"],
            path="sql_sync",
        )
    assert identity.snapshot()["paths"] == {}


def test_snapshot_is_detached_and_oid_drift_cannot_replace_native_evidence(identity: OriginalNativeIdentity) -> None:
    scope = identity.SCOPES["sql_sync"]
    identity.observe(access_token=_token(identity=identity, path="sql_sync"), scope=scope, path="sql_sync")
    exported = identity.snapshot()
    exported["paths"]["sql_sync"]["claims"]["oid"] = "tampered-public-fixture"
    assert identity.snapshot()["paths"]["sql_sync"]["claims"]["oid"] == str(_OBJECT)
    with pytest.raises(ValueError, match="identity_drift"):
        identity.observe(
            access_token=_token(identity=identity, path="sql_sync", changes={"oid": str(_TENANT)}),
            scope=scope,
            path="sql_sync",
        )


async def test_environment_mutation_stops_before_native_refresh_async(identity: OriginalNativeIdentity) -> None:
    native = MagicMock(spec=AsyncTokenCredential)
    native.get_token = AsyncMock(return_value=_token(identity=identity, path="blob_results"))
    native.close = AsyncMock()
    wrapped = ObservedAsyncTokenCredential(credential=native, observer=identity, path="blob_results")
    with patch.object(identity, "_verify_environment", side_effect=ValueError("public environment changed")):
        with pytest.raises(ValueError, match="environment changed"):
            await wrapped.get_token(identity.SCOPES["blob_results"])
    native.get_token.assert_not_awaited()
    await wrapped.close()
    native.close.assert_awaited_once()


def test_standard_sync_sql_observes_initial_use_and_actual_refresh(identity: OriginalNativeIdentity) -> None:
    memory = object.__new__(AzureSQLMemory)
    memory.azure_token_observer = identity
    native = MagicMock()
    native.access_token = _token(identity=identity, path="sql_sync")
    with patch("pyrit.memory.azure_sql_memory.AzureAuth", return_value=native):
        memory._create_auth_token()
        memory._auth_token_expiry = int((datetime.now(UTC) - timedelta(seconds=1)).timestamp())
        memory._refresh_token_if_needed()
    assert identity.snapshot()["paths"]["sql_sync"]["observations"] == 2
    assert memory._auth_token is native.access_token
    assert native.azure_creds.close.call_count == 2


async def test_standard_async_sql_uses_observed_token_in_native_attrs1256_async(
    identity: OriginalNativeIdentity,
) -> None:
    memory = object.__new__(AzureSQLMemory)
    memory.azure_token_observer = identity
    memory._connection_string = (
        "mssql+pyodbc:///?odbc_connect=Driver%3D%7BODBC+Driver+18+for+SQL+Server%7D%3B"
        "Server%3Dpublic.test%3BDatabase%3Dpublic-fixture%3B"
    )
    memory._verbose = False
    memory._async_auth = {}
    native = MagicMock(spec=AsyncAzureAuth)
    token = _token(identity=identity, path="sql_async")
    native.get_access_token_async = AsyncMock(return_value=token)
    engine = MagicMock(spec=AsyncEngine)
    engine.dialect.create_connect_args.return_value = (
        [],
        {"dsn": "Driver={ODBC Driver 18 for SQL Server};Database=public-fixture;Trusted_Connection=Yes"},
    )
    with (
        patch("pyrit.memory.azure_sql_memory.AsyncAzureAuth", return_value=native),
        patch("pyrit.memory.azure_sql_memory.create_async_engine", return_value=engine) as create,
        patch("aioodbc.connect", new_callable=AsyncMock) as connect,
    ):
        assert memory._create_async_engine() is engine
        creator: Callable[[], object] = create.call_args.kwargs["async_creator"]
        await creator()
    assert identity.snapshot()["paths"]["sql_async"]["observations"] == 1
    attrs = connect.call_args.kwargs["attrs_before"][1256]
    encoded = token.token.encode("utf-16-le")
    assert attrs[4:] == encoded
    assert int.from_bytes(attrs[:4], "little") == len(encoded)
    assert "trusted_connection" not in connect.call_args.kwargs["dsn"].lower()


async def test_standard_result_blob_observes_token_use_refresh_and_owned_close_async(
    identity: OriginalNativeIdentity,
) -> None:
    token = _token(identity=identity, path="blob_results")
    credential = MagicMock(spec=AsyncTokenCredential)
    credential.get_token = AsyncMock(return_value=token)
    credential.close = AsyncMock()
    client = MagicMock(spec=ContainerClient)
    client.get_container_properties = AsyncMock()
    client.close = AsyncMock()
    storage = AzureBlobStorageIO(
        container_url="https://publicfixture.blob.core.windows.net/results", azure_token_observer=identity
    )
    with (
        patch("azure.identity.aio.DefaultAzureCredential", return_value=credential),
        patch("azure.storage.blob.aio.ContainerClient", return_value=client) as factory,
    ):
        await storage._create_container_client_async()
        wrapper = factory.call_args.kwargs["credential"]
        assert isinstance(wrapper, ObservedAsyncTokenCredential)
        assert await wrapper.get_token(identity.SCOPES["blob_results"]) is token
        assert await wrapper.get_token(identity.SCOPES["blob_results"]) is token
        await storage._close_client_async()
    assert identity.snapshot()["paths"]["blob_results"]["observations"] == 2
    credential.close.assert_awaited_once()
    client.close.assert_awaited_once()
    with pytest.raises(ValueError, match="unobserved"):
        identity.require_durable_paths()
