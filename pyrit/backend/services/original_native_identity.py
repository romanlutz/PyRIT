# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Safe evidence from tokens actually selected by the ordinary SQL, Blob and model clients."""

from __future__ import annotations

import base64
import copy
import re
from datetime import UTC, datetime
from threading import Lock
from typing import TYPE_CHECKING
from uuid import UUID

from pydantic import JsonValue, TypeAdapter

if TYPE_CHECKING:
    from collections.abc import Callable

    from azure.core.credentials import AccessToken

    from pyrit.backend.models.original_worker import CohostBackendConfig


class OriginalNativeIdentity:
    """Fail before client use on identity drift; retain bounded non-secret claims only."""

    SCOPES = {
        "sql_sync": "https://database.windows.net/.default",
        "sql_async": "https://database.windows.net/.default",
        "blob_results": "https://storage.azure.com/.default",
        "model_relay": "https://cognitiveservices.azure.com/.default",
    }
    MI_RESOURCE = re.compile(
        r"^/subscriptions/(?P<subscription>[0-9a-f-]{36})/resourceGroups/(?P<group>[^/]{1,90})/"
        r"providers/Microsoft\.ManagedIdentity/userAssignedIdentities/[A-Za-z0-9_-]{1,128}$",
        re.IGNORECASE,
    )

    def __init__(self, *, config: CohostBackendConfig, tenant_id: UUID, verify_environment: Callable[[], None]) -> None:
        """Bind the owned preview identity and immutable startup environment."""
        self._config = config
        self._tenant_id = tenant_id
        self._verify_environment = verify_environment
        self._paths: dict[str, dict[str, JsonValue]] = {}
        self._oid: str | None = None
        self._lock = Lock()

    def __call__(self, *, access_token: AccessToken, scope: str, path: str) -> None:
        """Validate a standard memory client's selected token without retaining its bearer."""
        self.observe(access_token=access_token, scope=scope, path=path)

    def before_token(self, *, scope: str, path: str) -> None:
        """Refuse environment drift before constructing or refreshing a native credential."""
        self._verify_environment()
        if scope != self.SCOPES.get(path):
            raise ValueError("original_native_scope_mismatch")

    def observe(self, *, access_token: AccessToken, scope: str, path: str) -> dict[str, JsonValue]:
        """
        Validate SDK-selected claims, not an independently verified JWT signature.

        Returns:
            dict[str, JsonValue]: Bounded non-secret identity claims.
        """
        self.before_token(scope=scope, path=path)
        claims = self._claims(access_token)
        client = claims.get("appid") or claims.get("azp")
        oid, tid, audience, mirid = (claims.get(key) for key in ("oid", "tid", "aud", "xms_mirid"))
        sandbox = self._config.sandbox
        resource = self.MI_RESOURCE.fullmatch(mirid) if isinstance(mirid, str) else None
        try:
            if (
                scope != self.SCOPES.get(path)
                or not isinstance(client, str)
                or UUID(client) != self._config.managed_identity_client_id
                or not isinstance(oid, str)
                or not isinstance(tid, str)
                or UUID(tid) != self._tenant_id
                or not isinstance(audience, str)
                or audience.rstrip("/") != scope.removesuffix("/.default")
                or access_token.expires_on <= datetime.now(UTC).timestamp()
                or resource is None
                or sandbox is None
                or UUID(resource["subscription"]) != sandbox.subscription_id
                or resource["group"].casefold() != sandbox.resource_group.casefold()
            ):
                raise ValueError
            canonical_oid = str(UUID(oid))
        except ValueError:
            raise ValueError("original_native_identity_mismatch") from None
        safe: dict[str, JsonValue] = {
            "client_id": str(self._config.managed_identity_client_id),
            "oid": canonical_oid,
            "tid": str(self._tenant_id),
            "aud": audience,
            "xms_mirid": mirid,
        }
        with self._lock:
            if self._oid is not None and self._oid != canonical_oid:
                raise ValueError("original_native_identity_drift")
            self._oid = canonical_oid
            previous = self._paths.get(path, {})
            count = previous.get("observations", 0)
            assert type(count) is int
            self._paths[path] = {
                "observations": count + 1,
                "observed_at": datetime.now(UTC).isoformat(),
                "claims": safe,
            }
        return copy.deepcopy(safe)

    def require_durable_paths(self) -> None:
        """Require use of both ordinary SQL engines and the actual result-container client."""
        with self._lock:
            if not {"sql_sync", "sql_async", "blob_results"} <= self._paths.keys():
                raise ValueError("original_native_durable_identity_unobserved")

    def snapshot(self) -> dict[str, JsonValue]:
        """
        Export selected identity evidence, never the token or its encoded claims.

        Returns:
            dict[str, JsonValue]: Safe native path observations and owned durable targets.
        """
        with self._lock:
            paths: dict[str, JsonValue] = {
                path: copy.deepcopy(observation) for path, observation in self._paths.items()
            }
            return {
                "schema": "original-native-client-identity/v1",
                "public_commit": self._config.public_commit,
                "credential_chain": "ManagedIdentityCredential",
                "database_name": self._config.expected_database_name,
                "result_container_url": self._config.result_container_url,
                "paths": paths,
            }

    @staticmethod
    def _claims(access_token: AccessToken) -> dict[str, JsonValue]:
        try:
            if len(access_token.token) > 65_536:
                raise ValueError
            segments = access_token.token.split(".")
            if len(segments) != 3 or not 1 <= len(segments[1]) <= 16_384:
                raise ValueError
            content = base64.b64decode(segments[1] + "=" * (-len(segments[1]) % 4), altchars=b"-_", validate=True)
            return TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
        except (UnicodeError, ValueError):
            raise ValueError("original_native_claims_unverified") from None
