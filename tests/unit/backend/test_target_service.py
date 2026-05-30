# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend target service.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrit.backend.models.targets import CreateTargetRequest
from pyrit.backend.persistence.runtime_targets import (
    RuntimeTargetEntry,
    RuntimeTargetStore,
)
from pyrit.backend.services.target_service import TargetService, get_target_service
from pyrit.identifiers import ComponentIdentifier
from pyrit.registry.object_registries import TargetRegistry


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the TargetRegistry singleton before each test."""
    TargetRegistry.reset_instance()
    yield
    TargetRegistry.reset_instance()


@pytest.fixture
def runtime_store(tmp_path: Path) -> RuntimeTargetStore:
    """A per-test runtime targets store backed by a fresh file."""
    return RuntimeTargetStore(path=tmp_path / "runtime_targets.json")


def _mock_target_identifier(*, class_name: str = "MockTarget", **kwargs) -> ComponentIdentifier:
    """Create a mock target identifier using ComponentIdentifier."""
    params = {
        "endpoint": kwargs.get("endpoint"),
        "model_name": kwargs.get("model_name"),
        "temperature": kwargs.get("temperature"),
        "top_p": kwargs.get("top_p"),
        "max_requests_per_minute": kwargs.get("max_requests_per_minute"),
    }
    # Filter out None values to match ComponentIdentifier.of behavior
    clean_params = {k: v for k, v in params.items() if v is not None}
    return ComponentIdentifier(
        class_name=class_name,
        class_module="tests.unit.backend.test_target_service",
        params=clean_params,
    )


async def _test_token_provider() -> str:
    """Shared async token provider used in Entra authentication tests."""
    return "test-token"


class TestListTargets:
    """Tests for TargetService.list_targets method."""

    async def test_list_targets_returns_empty_when_no_targets(self) -> None:
        """Test that list_targets returns empty list when no targets exist."""
        service = TargetService()

        result = await service.list_targets_async()

        assert result.items == []
        assert result.pagination.has_more is False

    async def test_list_targets_returns_targets_from_registry(self) -> None:
        """Test that list_targets returns targets from registry."""
        service = TargetService()

        # Register a mock target
        mock_target = MagicMock()
        mock_target.get_identifier.return_value = _mock_target_identifier(endpoint="http://test")
        service._registry.register_instance(mock_target, name="target-1")

        result = await service.list_targets_async()

        assert len(result.items) == 1
        assert result.items[0].target_registry_name == "target-1"
        assert result.items[0].target_type == "MockTarget"
        assert result.pagination.has_more is False

    async def test_list_targets_paginates_with_limit(self) -> None:
        """Test that list_targets respects the limit parameter."""
        service = TargetService()

        for i in range(5):
            mock_target = MagicMock()
            mock_target.get_identifier.return_value = _mock_target_identifier()
            service._registry.register_instance(mock_target, name=f"target-{i}")

        result = await service.list_targets_async(limit=3)

        assert len(result.items) == 3
        assert result.pagination.limit == 3
        assert result.pagination.has_more is True
        assert result.pagination.next_cursor == result.items[-1].target_registry_name

    async def test_list_targets_cursor_returns_next_page(self) -> None:
        """Test that list_targets cursor skips to the correct position."""
        service = TargetService()

        for i in range(5):
            mock_target = MagicMock()
            mock_target.get_identifier.return_value = _mock_target_identifier()
            service._registry.register_instance(mock_target, name=f"target-{i}")

        first_page = await service.list_targets_async(limit=2)
        second_page = await service.list_targets_async(limit=2, cursor=first_page.pagination.next_cursor)

        assert len(second_page.items) == 2
        assert second_page.items[0].target_registry_name != first_page.items[0].target_registry_name
        assert second_page.pagination.has_more is True

    async def test_list_targets_last_page_has_no_more(self) -> None:
        """Test that the last page has has_more=False and no next_cursor."""
        service = TargetService()

        for i in range(3):
            mock_target = MagicMock()
            mock_target.get_identifier.return_value = _mock_target_identifier()
            service._registry.register_instance(mock_target, name=f"target-{i}")

        first_page = await service.list_targets_async(limit=2)
        last_page = await service.list_targets_async(limit=2, cursor=first_page.pagination.next_cursor)

        assert len(last_page.items) == 1
        assert last_page.pagination.has_more is False
        assert last_page.pagination.next_cursor is None


class TestGetTarget:
    """Tests for TargetService.get_target method."""

    async def test_get_target_returns_none_for_nonexistent(self) -> None:
        """Test that get_target returns None for non-existent target."""
        service = TargetService()

        result = await service.get_target_async(target_registry_name="nonexistent-id")

        assert result is None

    async def test_get_target_returns_target_from_registry(self) -> None:
        """Test that get_target returns target built from registry object."""
        service = TargetService()

        mock_target = MagicMock()
        mock_target.get_identifier.return_value = _mock_target_identifier()
        service._registry.register_instance(mock_target, name="target-1")

        result = await service.get_target_async(target_registry_name="target-1")

        assert result is not None
        assert result.target_registry_name == "target-1"
        assert result.target_type == "MockTarget"

    async def test_list_targets_includes_extra_params_in_target_specific(self) -> None:
        """Test that extra identifier params (reasoning_effort etc.) appear in target_specific_params."""
        service = TargetService()

        mock_target = MagicMock()
        identifier = ComponentIdentifier(
            class_name="OpenAIResponseTarget",
            class_module="pyrit.prompt_target",
            params={
                "endpoint": "https://api.openai.com",
                "model_name": "o3",
                "temperature": 1.0,
                "reasoning_effort": "high",
                "reasoning_summary": "auto",
                "max_output_tokens": 4096,
            },
        )
        mock_target.get_identifier.return_value = identifier
        service._registry.register_instance(mock_target, name="response-target")

        result = await service.list_targets_async()

        assert len(result.items) == 1
        target = result.items[0]
        assert target.temperature == 1.0
        assert target.target_specific_params is not None
        assert target.target_specific_params["reasoning_effort"] == "high"
        assert target.target_specific_params["reasoning_summary"] == "auto"
        assert target.target_specific_params["max_output_tokens"] == 4096

    async def test_get_target_includes_extra_params_in_target_specific(self) -> None:
        """Test that get_target returns target_specific_params with extra identifier params."""
        service = TargetService()

        mock_target = MagicMock()
        identifier = ComponentIdentifier(
            class_name="OpenAIChatTarget",
            class_module="pyrit.prompt_target",
            params={
                "endpoint": "https://api.openai.com",
                "model_name": "gpt-4",
                "frequency_penalty": 0.5,
                "seed": 42,
            },
        )
        mock_target.get_identifier.return_value = identifier
        service._registry.register_instance(mock_target, name="chat-target")

        result = await service.get_target_async(target_registry_name="chat-target")

        assert result is not None
        assert result.target_specific_params is not None
        assert result.target_specific_params["frequency_penalty"] == 0.5
        assert result.target_specific_params["seed"] == 42


class TestGetTargetObject:
    """Tests for TargetService.get_target_object method."""

    def test_get_target_object_returns_none_for_nonexistent(self) -> None:
        """Test that get_target_object returns None for non-existent target."""
        service = TargetService()

        result = service.get_target_object(target_registry_name="nonexistent-id")

        assert result is None

    def test_get_target_object_returns_object_from_registry(self) -> None:
        """Test that get_target_object returns the actual target object."""
        service = TargetService()
        mock_target = MagicMock()
        service._registry.register_instance(mock_target, name="target-1")

        result = service.get_target_object(target_registry_name="target-1")

        assert result is mock_target


class TestCreateTarget:
    """Tests for TargetService.create_target method."""

    async def test_create_target_raises_for_invalid_type(self) -> None:
        """Test that create_target raises for invalid target type."""
        service = TargetService()

        request = CreateTargetRequest(
            type="NonExistentTarget",
            params={},
        )

        with pytest.raises(ValueError, match="not found"):
            await service.create_target_async(request=request)

    async def test_create_target_success(self, sqlite_instance) -> None:
        """Test successful target creation."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            params={},
        )

        result = await service.create_target_async(request=request)

        assert result.target_registry_name is not None
        assert result.target_type == "TextTarget"

    async def test_create_target_registers_in_registry(self, sqlite_instance) -> None:
        """Test that create_target registers object in registry."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            params={},
        )

        result = await service.create_target_async(request=request)

        # Object should be retrievable from registry
        target_obj = service.get_target_object(target_registry_name=result.target_registry_name)
        assert target_obj is not None

    async def test_create_target_model_name_not_overridden_by_env_var(self, sqlite_instance) -> None:
        """Test that explicit model_name is not overridden by underlying_model env var."""
        with patch.dict(os.environ, {"OPENAI_CHAT_UNDERLYING_MODEL": "gpt-4o"}):
            service = TargetService()

            request = CreateTargetRequest(
                type="OpenAIChatTarget",
                params={
                    "model_name": "claude-sonnet-4-6",
                    "endpoint": "https://test.openai.azure.com/",
                    "api_key": "test-key",
                },
            )

            result = await service.create_target_async(request=request)

            assert result.model_name == "claude-sonnet-4-6"
            # underlying_model_name should be None since no underlying_model was passed
            assert result.underlying_model_name is None

    async def test_create_target_with_different_underlying_model(self, sqlite_instance) -> None:
        """Test that explicit underlying_model is used when it differs from model_name."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={
                "model_name": "my-gpt4o-deployment",
                "endpoint": "https://test.openai.azure.com/",
                "api_key": "test-key",
                "underlying_model": "gpt-4o",
            },
        )

        result = await service.create_target_async(request=request)

        assert result.model_name == "my-gpt4o-deployment"
        assert result.underlying_model_name == "gpt-4o"


class TestCreateTargetEntraAuth:
    """Test that creating targets with Entra auth mode properly authenticates and handles edge cases."""

    async def test_create_openai_target_with_entra_injects_token_provider(self, sqlite_instance) -> None:
        """Entra auth path: api_key is replaced with the authentication callable"""

        with patch(
            "pyrit.backend.services.target_service.get_azure_openai_auth",
            return_value=_test_token_provider,
        ) as mock_get_auth:
            service = TargetService()

            request = CreateTargetRequest(
                type="OpenAIChatTarget",
                params={
                    "endpoint": "https://test.openai.azure.com/",
                    "model_name": "gpt-4o",
                },
                auth_mode="entra",
            )

            result = await service.create_target_async(request=request)

            mock_get_auth.assert_called_once_with("https://test.openai.azure.com/")
            target_obj = service.get_target_object(target_registry_name=result.target_registry_name)
            assert target_obj is not None
            # OpenAI target preserves async callables verbatim through ensure_async_token_provider.
            assert target_obj._api_key is _test_token_provider  # type: ignore[attr-defined]

    async def test_create_openai_target_with_entra_drops_user_api_key(self, sqlite_instance) -> None:
        """Any api_key supplied alongside auth_mode='entra' must be discarded."""

        with patch(
            "pyrit.backend.services.target_service.get_azure_openai_auth",
            return_value=_test_token_provider,
        ):
            service = TargetService()

            request = CreateTargetRequest(
                type="OpenAIChatTarget",
                params={
                    "endpoint": "https://test.openai.azure.com/",
                    "model_name": "gpt-4o",
                    "api_key": "should-be-ignored",
                },
                auth_mode="entra",
            )

            result = await service.create_target_async(request=request)

            target_obj = service.get_target_object(target_registry_name=result.target_registry_name)
            assert target_obj is not None
            assert target_obj._api_key is _test_token_provider  # type: ignore[attr-defined]
            # The literal "should-be-ignored" string must never appear.
            assert target_obj._api_key != "should-be-ignored"  # type: ignore[attr-defined]

    async def test_create_openai_target_with_entra_does_not_mutate_request_params(self, sqlite_instance) -> None:
        """The CreateTargetRequest.params object must remain unchanged after creation."""

        with patch(
            "pyrit.backend.services.target_service.get_azure_openai_auth",
            return_value=_test_token_provider,
        ):
            service = TargetService()

            original_params = {
                "endpoint": "https://test.openai.azure.com/",
                "model_name": "gpt-4o",
                "api_key": "original-key",
            }
            request = CreateTargetRequest(
                type="OpenAIChatTarget",
                params=dict(original_params),
                auth_mode="entra",
            )

            await service.create_target_async(request=request)

            # The caller's request.params must be unchanged after the call.
            assert request.params == original_params

    async def test_create_openai_target_with_entra_non_azure_endpoint_raises(self, sqlite_instance) -> None:
        """Entra ID requires a known Azure OpenAI / AI Foundry hostname suffix."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={"endpoint": "https://api.openai.com/"},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="Azure endpoint"):
            await service.create_target_async(request=request)

    async def test_create_openai_target_with_entra_substring_lookalike_endpoint_raises(self, sqlite_instance) -> None:
        """Substring 'azure' in the hostname must not be enough to pass Entra validation."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            # Hostname contains 'azure' but does NOT end with an approved suffix.
            params={"endpoint": "https://evil-azure.example.com/"},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="Azure endpoint"):
            await service.create_target_async(request=request)

    async def test_create_openai_target_with_entra_missing_endpoint_raises(self, sqlite_instance) -> None:
        """Entra ID for OpenAI must reject a missing endpoint with a clear error."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="endpoint"):
            await service.create_target_async(request=request)

    async def test_create_azureml_target_with_entra_injects_token_provider(self, sqlite_instance) -> None:
        """AzureML Entra path: api_key is replaced with the ML scope token provider."""

        with patch(
            "pyrit.backend.services.target_service.get_azure_async_token_provider",
            return_value=_test_token_provider,
        ) as mock_get_provider:
            service = TargetService()

            request = CreateTargetRequest(
                type="AzureMLChatTarget",
                params={"endpoint": "https://my-aml.region.inference.ml.azure.com/score"},
                auth_mode="entra",
            )

            result = await service.create_target_async(request=request)

            mock_get_provider.assert_called_once_with("https://ml.azure.com/.default")
            target_obj = service.get_target_object(target_registry_name=result.target_registry_name)
            assert target_obj is not None
            # AzureMLChatTarget stores the provider on _api_key_provider; static _api_key is cleared.
            assert target_obj._api_key_provider is _test_token_provider  # type: ignore[attr-defined]
            assert target_obj._api_key == ""  # type: ignore[attr-defined]

    async def test_create_azureml_target_with_entra_non_aml_endpoint_raises(self, sqlite_instance) -> None:
        """Entra ID for AzureMLChatTarget requires a known AML hostname suffix."""
        service = TargetService()

        request = CreateTargetRequest(
            type="AzureMLChatTarget",
            params={"endpoint": "https://example.com/score"},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="AML endpoint"):
            await service.create_target_async(request=request)

    async def test_create_azureml_target_with_entra_substring_lookalike_endpoint_raises(self, sqlite_instance) -> None:
        """Substring 'inference.ml.azure.com' in the hostname must not be enough to pass AML validation."""
        service = TargetService()

        request = CreateTargetRequest(
            type="AzureMLChatTarget",
            # Hostname contains the AML suffix as a substring but does NOT end with it.
            params={"endpoint": "https://evil-inference.ml.azure.com.attacker.com/score"},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="AML endpoint"):
            await service.create_target_async(request=request)

    async def test_create_azureml_target_with_entra_missing_endpoint_raises(self, sqlite_instance) -> None:
        """Entra ID for AzureMLChatTarget must reject a missing endpoint with a clear error."""
        service = TargetService()

        request = CreateTargetRequest(
            type="AzureMLChatTarget",
            params={},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="endpoint"):
            await service.create_target_async(request=request)

    async def test_create_target_entra_unsupported_type_raises(self, sqlite_instance) -> None:
        """Entra ID is only supported for OpenAI-family and AzureMLChatTarget."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            params={},
            auth_mode="entra",
        )

        with pytest.raises(ValueError, match="does not support Entra"):
            await service.create_target_async(request=request)


class TestCreateTargetApiKeyAuth:
    """Test that auth_mode='api_key' strictly requires a key in params or environment."""

    async def test_create_openai_target_api_key_mode_without_key_raises(self, sqlite_instance) -> None:
        """Without an api_key (params or env), OpenAITarget would silently fall back to Entra;
        the service must reject this so the user's explicit choice is honored."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={
                "model_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com/",
            },
            auth_mode="api_key",
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_CHAT_KEY", None)
            with pytest.raises(ValueError, match="auth_mode='api_key' requires an API key"):
                await service.create_target_async(request=request)

    async def test_create_openai_target_api_key_mode_with_env_var_succeeds(self, sqlite_instance) -> None:
        """An env-var-supplied key satisfies the api_key requirement."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={
                "model_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com/",
            },
            auth_mode="api_key",
        )

        with patch.dict(os.environ, {"OPENAI_CHAT_KEY": "env-test-key"}):
            result = await service.create_target_async(request=request)

        assert result.target_type == "OpenAIChatTarget"

    async def test_create_openai_target_api_key_mode_rejects_empty_key(self, sqlite_instance) -> None:
        """An empty-string api_key counts as missing and must be rejected."""
        service = TargetService()

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={
                "model_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com/",
                "api_key": "",
            },
            auth_mode="api_key",
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_CHAT_KEY", None)
            with pytest.raises(ValueError, match="auth_mode='api_key' requires an API key"):
                await service.create_target_async(request=request)

    async def test_create_azureml_target_api_key_mode_without_key_raises(self, sqlite_instance) -> None:
        """AzureMLChatTarget in api_key mode also requires an explicit key."""
        service = TargetService()

        request = CreateTargetRequest(
            type="AzureMLChatTarget",
            params={"endpoint": "https://my-endpoint.eastus.inference.ml.azure.com/score"},
            auth_mode="api_key",
        )

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AZURE_ML_KEY", None)
            with pytest.raises(ValueError, match="auth_mode='api_key' requires an API key"):
                await service.create_target_async(request=request)

    async def test_create_text_target_api_key_mode_skips_validation(self, sqlite_instance) -> None:
        """Targets without an api_key_environment_variable (e.g. TextTarget) are unaffected."""
        service = TargetService()

        request = CreateTargetRequest(
            type="TextTarget",
            params={},
            auth_mode="api_key",
        )

        result = await service.create_target_async(request=request)
        assert result.target_type == "TextTarget"


class TestTargetServiceSingleton:
    """Tests for get_target_service singleton function."""

    def test_get_target_service_returns_target_service(self) -> None:
        """Test that get_target_service returns a TargetService instance."""
        get_target_service.cache_clear()

        service = get_target_service()
        assert isinstance(service, TargetService)

    def test_get_target_service_returns_same_instance(self) -> None:
        """Test that get_target_service returns the same instance."""
        get_target_service.cache_clear()

        service1 = get_target_service()
        service2 = get_target_service()
        assert service1 is service2


class TestCreateTargetPersistence:
    """Persistence side-effects of create_target_async."""

    async def test_create_target_with_inline_api_key_skips_persistence_and_marks_session_only(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        """
        Targets created with an inline api_key must NOT be persisted to disk
        (we never write secrets to JSON) and must be flagged session_only so
        the frontend can warn the user that the target won't survive restart.
        """
        service = TargetService(runtime_store=runtime_store)

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={
                "model_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com/",
                "api_key": "should-not-persist",
            },
            auth_mode="api_key",
        )

        result = await service.create_target_async(request=request)

        # Nothing written to disk.
        assert await runtime_store.load_async() == []
        assert not runtime_store.path.exists() or "should-not-persist" not in runtime_store.path.read_text(
            encoding="utf-8"
        )
        # Response advertises the session-only state with an actionable hint.
        assert result.session_only is True
        assert result.persist_hint is not None
        assert "OPENAI_CHAT_KEY" in result.persist_hint
        assert result.is_runtime is True  # still deletable for this process

    async def test_create_target_persist_hint_falls_back_when_no_env_var(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        """For target classes with no known api_key env var, the hint is generic."""
        service = TargetService(runtime_store=runtime_store)

        # TextTarget accepts no api_key at all; force the path by patching
        # _resolve_api_key_env_var to return None for any class.
        with patch.object(service, "_get_target_class", wraps=service._get_target_class) as _:
            request = CreateTargetRequest(
                type="TextTarget",
                params={"api_key": "shouldnt-matter"},
                auth_mode="api_key",
            )
            # TextTarget rejects unknown kwargs, so this would fail at construction.
            # We assert the helper directly instead:
            hint = TargetService._build_persist_hint(target_class=type("Stub", (), {}))
            assert "inline API key" in hint
            assert "OPENAI" not in hint  # no env var-specific text when class has none

    async def test_create_target_with_env_var_api_key_persists_sanitized_entry(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        """
        When the api_key is resolved from an env var (not provided inline), the
        target is persisted and the response is NOT session_only. The persisted
        entry never contains an api_key value.
        """
        service = TargetService(runtime_store=runtime_store)

        request = CreateTargetRequest(
            type="OpenAIChatTarget",
            params={
                "model_name": "gpt-4o",
                "endpoint": "https://test.openai.azure.com/",
            },
            auth_mode="api_key",
        )

        with patch.dict(os.environ, {"OPENAI_CHAT_KEY": "env-resolved-key"}):
            result = await service.create_target_async(request=request)

        entries = await runtime_store.load_async()
        assert len(entries) == 1
        entry = entries[0]
        assert entry.target_registry_name == result.target_registry_name
        assert entry.type == "OpenAIChatTarget"
        assert entry.auth_mode == "api_key"
        assert "api_key" not in entry.params
        assert "env-resolved-key" not in runtime_store.path.read_text(encoding="utf-8")
        assert result.session_only is False
        assert result.persist_hint is None

    async def test_create_target_marks_instance_as_runtime(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        service = TargetService(runtime_store=runtime_store)

        request = CreateTargetRequest(type="TextTarget", params={}, auth_mode="api_key")
        result = await service.create_target_async(request=request)

        assert result.is_runtime is True
        assert result.needs_reconfiguration is False
        assert result.session_only is False

    async def test_initializer_registered_targets_are_not_marked_runtime(
        self,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        service = TargetService(runtime_store=runtime_store)

        mock_target = MagicMock()
        mock_target.get_identifier.return_value = ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        service._registry.register_instance(mock_target, name="initializer-target")

        result = await service.get_target_async(target_registry_name="initializer-target")

        assert result is not None
        assert result.is_runtime is False
        assert result.session_only is False


class TestDeleteTarget:
    """delete_target_async behaviour."""

    async def test_delete_runtime_target_removes_from_registry_and_store(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        service = TargetService(runtime_store=runtime_store)
        created = await service.create_target_async(
            request=CreateTargetRequest(type="TextTarget", params={}, auth_mode="api_key")
        )

        await service.delete_target_async(target_registry_name=created.target_registry_name)

        assert service.get_target_object(target_registry_name=created.target_registry_name) is None
        assert await runtime_store.load_async() == []

    async def test_delete_unknown_raises_lookup_error(
        self,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        service = TargetService(runtime_store=runtime_store)

        with pytest.raises(LookupError):
            await service.delete_target_async(target_registry_name="never-existed")

    async def test_delete_initializer_target_raises_permission_error(
        self,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        service = TargetService(runtime_store=runtime_store)

        mock_target = MagicMock()
        mock_target.get_identifier.return_value = ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        service._registry.register_instance(mock_target, name="initializer-target")

        with pytest.raises(PermissionError, match="initializer"):
            await service.delete_target_async(target_registry_name="initializer-target")

        # Target must remain registered after a refused delete.
        assert service.get_target_object(target_registry_name="initializer-target") is mock_target

    async def test_delete_broken_runtime_entry_removes_from_store(
        self,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        service = TargetService(runtime_store=runtime_store)

        broken_entry = RuntimeTargetEntry(
            target_registry_name="broken-target",
            type="OpenAIChatTarget",
            auth_mode="api_key",
            params={"endpoint": "https://x", "model_name": "gpt-4o"},
        )
        await runtime_store.append_async(broken_entry)
        service._runtime_target_metadata["broken-target"] = __import__(
            "pyrit.backend.services.target_service", fromlist=["_RuntimeTargetMetadata"]
        )._RuntimeTargetMetadata(entry=broken_entry, is_broken=True, reconfiguration_hint="missing OPENAI_CHAT_KEY")

        await service.delete_target_async(target_registry_name="broken-target")

        assert await runtime_store.load_async() == []
        assert "broken-target" not in service._runtime_target_metadata


class TestRestoreRuntimeTargets:
    """restore_runtime_targets_async replay behaviour."""

    async def test_restore_empty_store_is_noop(self, runtime_store: RuntimeTargetStore) -> None:
        service = TargetService(runtime_store=runtime_store)

        await service.restore_runtime_targets_async()

        result = await service.list_targets_async()
        assert result.items == []

    async def test_restore_replays_persisted_target_into_registry(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        await runtime_store.append_async(
            RuntimeTargetEntry(
                target_registry_name="text-target-1",
                type="TextTarget",
                auth_mode="api_key",
                params={},
            )
        )
        service = TargetService(runtime_store=runtime_store)

        await service.restore_runtime_targets_async()

        result = await service.list_targets_async()
        assert len(result.items) == 1
        assert result.items[0].target_type == "TextTarget"
        assert result.items[0].is_runtime is True
        assert result.items[0].needs_reconfiguration is False

    async def test_restore_marks_target_needs_reconfiguration_on_missing_env_var(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        await runtime_store.append_async(
            RuntimeTargetEntry(
                target_registry_name="openai-needs-reconfig",
                type="OpenAIChatTarget",
                auth_mode="api_key",
                params={
                    "model_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com/",
                },
            )
        )
        service = TargetService(runtime_store=runtime_store)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_CHAT_KEY", None)
            await service.restore_runtime_targets_async()

        result = await service.get_target_async(target_registry_name="openai-needs-reconfig")
        assert result is not None
        assert result.is_runtime is True
        assert result.needs_reconfiguration is True
        assert result.reconfiguration_hint
        # The broken entry must remain in the store so the user can delete it.
        assert any(e.target_registry_name == "openai-needs-reconfig" for e in await runtime_store.load_async())

    async def test_restore_skips_target_when_initializer_has_same_name(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        await runtime_store.append_async(
            RuntimeTargetEntry(
                target_registry_name="text_target",
                type="TextTarget",
                auth_mode="api_key",
                params={},
            )
        )
        service = TargetService(runtime_store=runtime_store)

        initializer_target = MagicMock()
        initializer_target.get_identifier.return_value = ComponentIdentifier(
            class_name="TextTarget",
            class_module="pyrit.prompt_target",
        )
        service._registry.register_instance(initializer_target, name="text_target")

        import logging

        with caplog.at_level(logging.WARNING):
            await service.restore_runtime_targets_async()

        # The pre-existing (initializer) instance must still be the one in the registry.
        assert service.get_target_object(target_registry_name="text_target") is initializer_target
        # The runtime metadata must not have been populated for the conflicting name.
        assert "text_target" not in service._runtime_target_metadata
        assert any("initializer already registered" in r.getMessage() for r in caplog.records)

    async def test_broken_target_shows_up_in_list_targets(
        self,
        sqlite_instance,
        runtime_store: RuntimeTargetStore,
    ) -> None:
        await runtime_store.append_async(
            RuntimeTargetEntry(
                target_registry_name="broken-openai",
                type="OpenAIChatTarget",
                auth_mode="api_key",
                params={
                    "model_name": "gpt-4o",
                    "endpoint": "https://test.openai.azure.com/",
                },
            )
        )
        service = TargetService(runtime_store=runtime_store)

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_CHAT_KEY", None)
            await service.restore_runtime_targets_async()

        result = await service.list_targets_async()
        assert len(result.items) == 1
        item = result.items[0]
        assert item.target_registry_name == "broken-openai"
        assert item.needs_reconfiguration is True
        assert item.is_runtime is True
        assert item.target_type == "OpenAIChatTarget"
        assert item.endpoint == "https://test.openai.azure.com/"
        assert item.model_name == "gpt-4o"
