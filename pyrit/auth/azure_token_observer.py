# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Optional observation at native credential use, without changing the credential chain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, Self

if TYPE_CHECKING:
    from types import TracebackType

    from azure.core.credentials import AccessToken
    from azure.core.credentials_async import AsyncTokenCredential


class AzureTokenObserver(Protocol):
    """Validate an ephemeral token before a native client uses it; never retain bearer bytes."""

    def __call__(self, *, access_token: AccessToken, scope: str, path: str) -> None:
        """Validate a native token immediately before the client receives it."""

    def before_token(self, *, scope: str, path: str) -> None:
        """Validate configuration before constructing or refreshing the credential."""


class ObservedAsyncTokenCredential:
    """Wrap the public Azure credential protocol and preserve its native refresh and ownership."""

    def __init__(self, *, credential: AsyncTokenCredential, observer: AzureTokenObserver, path: str) -> None:
        """Bind the caller's native credential and optional token-use policy."""
        self._credential = credential
        self._observer = observer
        self._path = path

    async def __aenter__(self) -> Self:
        """
        Enter the native credential context without bypassing observation.

        Returns:
            Self: The same observing credential context.
        """
        await self._credential.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Preserve the native public context-manager cleanup contract."""
        await self._credential.__aexit__(exc_type, exc_value, traceback)

    async def get_token(self, *scopes: str, **kwargs: Any) -> AccessToken:  # pyrit-async-suffix-exempt
        """
        Observe each actual token acquisition or refresh before returning it to the SDK.

        Returns:
            AccessToken: The unmodified native credential result.
        """
        for scope in scopes:
            self._observer.before_token(scope=scope, path=self._path)
        token = await self._credential.get_token(*scopes, **kwargs)
        for scope in scopes:
            self._observer(access_token=token, scope=scope, path=self._path)
        return token

    async def close(self) -> None:  # pyrit-async-suffix-exempt
        """Close the caller-owned native credential through its public protocol."""
        await self._credential.close()
