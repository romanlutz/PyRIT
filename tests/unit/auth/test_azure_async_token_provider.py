# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.credentials import AccessToken
from azure.core.credentials_async import AsyncTokenCredential

from pyrit.auth import get_azure_async_token_provider, get_azure_openai_auth


async def test_caller_owned_credential_preserves_refreshing_provider_async() -> None:
    credential = MagicMock(spec=AsyncTokenCredential)
    credential.get_token = AsyncMock(return_value=AccessToken("fixture-token", int(time.time()) + 3600))
    with patch("pyrit.auth.azure_auth.AsyncDefaultAzureCredential") as default_credential:
        provider = get_azure_openai_auth("https://fixture.openai.azure.com/openai/v1", credential=credential)
        assert await provider() == "fixture-token"
        assert await provider() == "fixture-token"
    default_credential.assert_not_called()
    credential.get_token.assert_awaited_once()
    assert credential.get_token.call_args.args == ("https://cognitiveservices.azure.com/.default",)
    credential.close.assert_not_called()


async def test_provider_refreshes_expired_token_async() -> None:
    credential = MagicMock(spec=AsyncTokenCredential)
    credential.get_token = AsyncMock(
        side_effect=[
            AccessToken("first-fixture-token", int(time.time()) + 3600),
            AccessToken("refreshed-fixture-token", int(time.time()) + 7200),
        ]
    )
    provider = get_azure_async_token_provider("https://cognitiveservices.azure.com/.default", credential=credential)
    assert await provider() == "first-fixture-token"
    later = time.time() + 4000
    with patch("time.time", return_value=later):
        assert await provider() == "refreshed-fixture-token"
    assert credential.get_token.await_count == 2


def test_default_auth_call_preserves_existing_construction() -> None:
    credential = MagicMock(spec=AsyncTokenCredential)
    with (
        patch("pyrit.auth.azure_auth.AsyncDefaultAzureCredential", return_value=credential) as default_credential,
        patch("pyrit.auth.azure_auth.get_async_bearer_token_provider") as build_provider,
    ):
        provider = get_azure_openai_auth("https://fixture.openai.azure.com/openai/v1")
    default_credential.assert_called_once()
    assert build_provider.call_args.args == (credential, "https://cognitiveservices.azure.com/.default")
    assert provider is build_provider.return_value


async def test_credential_failure_is_not_swallowed_async() -> None:
    credential = MagicMock(spec=AsyncTokenCredential)
    credential.get_token = AsyncMock(side_effect=ValueError("fixture authentication failure"))
    provider = get_azure_openai_auth("https://fixture.openai.azure.com/openai/v1", credential=credential)
    with pytest.raises(ValueError, match="fixture authentication failure"):
        await provider()
