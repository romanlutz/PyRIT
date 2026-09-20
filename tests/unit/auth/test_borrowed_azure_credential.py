# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from azure.core.credentials_async import AsyncTokenCredential

from pyrit.auth.azure_auth import get_azure_async_token_provider, get_azure_openai_auth


@pytest.mark.parametrize("helper", [get_azure_async_token_provider, get_azure_openai_auth])
async def test_async_auth_helper_borrows_credential_without_owning_lifetime_async(
    helper: Callable[..., Callable[[], Awaitable[str]]],
) -> None:
    credential = MagicMock(spec=AsyncTokenCredential)
    provider = AsyncMock(return_value="fixture-token")
    with (
        patch("pyrit.auth.azure_auth.AsyncDefaultAzureCredential") as default,
        patch("pyrit.auth.azure_auth.get_async_bearer_token_provider", return_value=provider) as create_provider,
    ):
        supplied = "https://fixture.openai.azure.com/openai/v1"
        result = helper(supplied, credential=credential)
        assert await result() == "fixture-token"
    default.assert_not_called()
    assert create_provider.call_args.args[0] is credential
    credential.close.assert_not_called()


def test_openai_auth_default_call_shape_is_backward_compatible() -> None:
    with patch("pyrit.auth.azure_auth.get_azure_async_token_provider") as provider:
        get_azure_openai_auth("https://fixture.openai.azure.com/openai/v1")
    assert provider.call_args.args == ("https://cognitiveservices.azure.com/.default",)
    assert provider.call_args.kwargs == {}
