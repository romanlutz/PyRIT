# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target service for managing target instances.

Handles creation and retrieval of target instances.
Uses TargetRegistry as the source of truth for instances.

Targets can be:
- Created via API request (instantiated from request params, then registered)
- Retrieved from registry (pre-registered at startup or created earlier)
- Restored from the runtime targets file at startup (replay of API-created targets)
"""

import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from pyrit import prompt_target
from pyrit.auth import get_azure_async_token_provider, get_azure_openai_auth
from pyrit.backend.mappers.target_mappers import target_object_to_instance
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    TargetCapabilitiesInfo,
    TargetInstance,
    TargetListResponse,
)
from pyrit.backend.persistence.runtime_targets import (
    RuntimeTargetEntry,
    RuntimeTargetStore,
    get_runtime_target_store,
    sanitize_params,
)
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.azure_ml_chat_target import AzureMLChatTarget
from pyrit.prompt_target.openai.openai_target import OpenAITarget
from pyrit.registry.object_registries import TargetRegistry

logger = logging.getLogger(__name__)

# Scope for Azure Machine Learning managed online endpoints.
_AZURE_ML_SCOPE = "https://ml.azure.com/.default"

# Recognised Azure OpenAI / AI Foundry hostname suffixes. Used for strict
# endpoint validation when Entra ID auth is requested, so a bearer token is
# only ever issued for a known Microsoft-operated endpoint.
_AZURE_OPENAI_HOSTNAME_SUFFIXES = (
    ".openai.azure.com",
    ".ai.azure.com",
    ".services.ai.azure.com",
    ".cognitiveservices.azure.com",
)

# Recognised Azure Machine Learning managed online endpoint hostname suffixes.
# Used for the same strict endpoint validation when issuing Entra ID tokens
# against an AML scope.
_AZURE_ML_HOSTNAME_SUFFIXES = (".inference.ml.azure.com",)


@dataclass
class _RuntimeTargetMetadata:
    """
    Per-target metadata for runtime-created targets.

    Attributes:
        entry: The persisted ``RuntimeTargetEntry`` (used both as the source of
            truth for delete-permission and for rendering broken targets in
            list views without a real target object).
        is_broken: True when the target could not be reconstructed on restart
            (e.g., missing api_key env var). Broken targets are not added to
            ``TargetRegistry`` but still appear in ``list_targets_async`` so
            the user can see and delete them.
        reconfiguration_hint: Human-readable hint for the broken case
            (e.g., the missing env var name).
        session_only: True when the target was created with an inline api_key
            and therefore deliberately not persisted to disk. Mutually
            exclusive with ``is_broken``.
        persist_hint: Human-readable hint for the ``session_only`` case
            (e.g., the env var to set so the target can be persisted).
    """

    entry: RuntimeTargetEntry
    is_broken: bool = False
    reconfiguration_hint: str | None = None
    session_only: bool = False
    persist_hint: str | None = None


def _is_azure_openai_endpoint(endpoint: str) -> bool:
    """
    Return True if ``endpoint`` resolves to a known Azure OpenAI / AI Foundry host.
    Uses a strict hostname-suffix check (not a substring search).

    Args:
        endpoint (str): The endpoint URL to validate.

    Returns:
        bool: True if the endpoint's hostname ends with a recognised Azure suffix;
            False otherwise
    """
    hostname = (urlparse(endpoint).hostname or "").lower()
    return any(hostname.endswith(suffix) for suffix in _AZURE_OPENAI_HOSTNAME_SUFFIXES)


def _is_azure_ml_endpoint(endpoint: str) -> bool:
    """
    Return True if ``endpoint`` resolves to a known AML managed host.
    Uses a strict hostname-suffix check (not a substring search).

    Args:
        endpoint (str): The endpoint URL to validate.

    Returns:
        bool: True if the endpoint's hostname ends with a recognised AML suffix;
            False otherwise.
    """
    hostname = (urlparse(endpoint).hostname or "").lower()
    return any(hostname.endswith(suffix) for suffix in _AZURE_ML_HOSTNAME_SUFFIXES)


def _resolve_api_key_env_var(target_class: type) -> str | None:
    """
    Return the api_key environment variable name for a target class.

    Args:
        target_class (type): The target class to inspect.

    Returns:
        str | None: The env var name, or None if the class does not declare one.
    """
    if issubclass(target_class, AzureMLChatTarget):
        env_var = getattr(target_class, "api_key_environment_variable", None)
        return env_var if isinstance(env_var, str) and env_var else None
    if issubclass(target_class, OpenAITarget):
        try:
            instance = target_class.__new__(target_class)
            instance._set_openai_env_configuration_vars()
        except Exception:
            return None
        env_var = getattr(instance, "api_key_environment_variable", None)
        return env_var if isinstance(env_var, str) and env_var else None
    return None


def _build_target_class_registry() -> dict[str, type]:
    """
    Build a registry mapping target class names to their classes.

    Uses the prompt_target module's __all__ to discover all available targets.

    Returns:
        Dict mapping class name (str) to class (type).
    """
    registry: dict[str, type] = {}
    for name in prompt_target.__all__:
        cls = getattr(prompt_target, name, None)
        if cls is not None and isinstance(cls, type) and issubclass(cls, PromptTarget):
            registry[name] = cls
    return registry


# Module-level class registry (built once on import)
_TARGET_CLASS_REGISTRY: dict[str, type] = _build_target_class_registry()


class TargetService:
    """
    Service for managing target instances.

    Uses TargetRegistry as the sole source of truth for live target objects.
    API metadata is derived from the target objects' identifiers.

    For targets created via the API at runtime, also persists a sanitized
    metadata record to a ``RuntimeTargetStore`` so they can be replayed on the
    next backend start. The persisted record never contains ``api_key`` values
    — credentials are re-resolved at restart via the target class's documented
    environment variable or Entra ID.
    """

    def __init__(self, *, runtime_store: RuntimeTargetStore | None = None) -> None:
        """
        Args:
            runtime_store: Override the runtime targets store. Defaults to the
                process-wide singleton from ``get_runtime_target_store``.
        """
        self._registry = TargetRegistry.get_registry_singleton()
        self._runtime_store = runtime_store or get_runtime_target_store()
        # Per-target side table for runtime-created targets. Keyed by registry name.
        # Includes both healthy (registered in TargetRegistry) and broken (registered
        # only here, surfaced as needs_reconfiguration ghosts in list views) targets.
        self._runtime_target_metadata: dict[str, _RuntimeTargetMetadata] = {}

    def _get_target_class(self, *, target_type: str) -> type:
        """
        Get the target class for a given type name.

        Looks up the class in the module-level target class registry.

        Args:
            target_type: The exact class name of the target (e.g., 'TextTarget').

        Returns:
            The target class.

        Raises:
            ValueError: If the target type is not found.
        """
        cls = _TARGET_CLASS_REGISTRY.get(target_type)
        if cls is None:
            raise ValueError(
                f"Target type '{target_type}' not found. Available types: {sorted(_TARGET_CLASS_REGISTRY.keys())}"
            )
        return cls

    def _build_instance_from_object(self, *, target_registry_name: str, target_obj: Any) -> TargetInstance:
        """
        Build a ``TargetInstance`` from a live registry object, layering on
        any runtime-target metadata we have for that name.

        Returns:
            TargetInstance with metadata derived from the object plus
            ``is_runtime`` / ``needs_reconfiguration`` / ``session_only`` flags
            from the side table.
        """
        meta = self._runtime_target_metadata.get(target_registry_name)
        return target_object_to_instance(
            target_registry_name,
            target_obj,
            is_runtime=meta is not None,
            needs_reconfiguration=meta.is_broken if meta else False,
            reconfiguration_hint=meta.reconfiguration_hint if meta else None,
            session_only=meta.session_only if meta else False,
            persist_hint=meta.persist_hint if meta else None,
        )

    def _build_broken_instance(self, meta: _RuntimeTargetMetadata) -> TargetInstance:
        """
        Build a ``TargetInstance`` for a runtime target that failed to restore.

        Pulls the displayable fields straight from the persisted entry's params
        rather than from a live target object (we don't have one) so the user
        still sees the type / endpoint / model and can choose to re-create or
        delete it.

        Args:
            meta: The runtime-target metadata for a broken entry.

        Returns:
            TargetInstance flagged with ``needs_reconfiguration=True`` and the
            stored hint, with capabilities defaulted to a minimal record.
        """
        params = meta.entry.params
        return TargetInstance(
            target_registry_name=meta.entry.target_registry_name,
            target_type=meta.entry.type,
            endpoint=params.get("endpoint") or None,
            model_name=params.get("model_name") or None,
            underlying_model_name=None,
            temperature=params.get("temperature"),
            top_p=params.get("top_p"),
            max_requests_per_minute=params.get("max_requests_per_minute"),
            capabilities=TargetCapabilitiesInfo(),
            target_specific_params=None,
            is_runtime=True,
            needs_reconfiguration=True,
            reconfiguration_hint=meta.reconfiguration_hint,
        )

    async def list_targets_async(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> TargetListResponse:
        """
        List all target instances with pagination.

        Combines live registry targets (initializer- or runtime-created) with
        broken runtime targets (those that failed to restore on startup but
        remain in the runtime targets file). Sorted by ``target_registry_name``.

        Args:
            limit: Maximum items to return.
            cursor: Pagination cursor (target_registry_name to start after).

        Returns:
            TargetListResponse containing paginated targets.
        """
        live_items = [
            self._build_instance_from_object(target_registry_name=entry.name, target_obj=entry.instance)
            for entry in self._registry.get_all_instances()
        ]
        registered_names = {item.target_registry_name for item in live_items}
        broken_items = [
            self._build_broken_instance(meta)
            for name, meta in self._runtime_target_metadata.items()
            if meta.is_broken and name not in registered_names
        ]
        items = sorted(live_items + broken_items, key=lambda t: t.target_registry_name)
        page, has_more = self._paginate(items=items, cursor=cursor, limit=limit)
        next_cursor = page[-1].target_registry_name if has_more and page else None
        return TargetListResponse(
            items=page,
            pagination=PaginationInfo(limit=limit, has_more=has_more, next_cursor=next_cursor, prev_cursor=cursor),
        )

    @staticmethod
    def _paginate(*, items: list[TargetInstance], cursor: str | None, limit: int) -> tuple[list[TargetInstance], bool]:
        """
        Apply cursor-based pagination.

        Returns:
            Tuple of (paginated items, has_more flag).
        """
        start_idx = 0
        if cursor:
            for i, item in enumerate(items):
                if item.target_registry_name == cursor:
                    start_idx = i + 1
                    break

        page = items[start_idx : start_idx + limit]
        has_more = len(items) > start_idx + limit
        return page, has_more

    async def get_target_async(self, *, target_registry_name: str) -> TargetInstance | None:
        """
        Get a target instance by registry name.

        Returns a ``needs_reconfiguration`` placeholder if the name belongs to a
        runtime target that failed to restore on startup.

        Returns:
            TargetInstance if found, None otherwise.
        """
        obj = self._registry.get_instance_by_name(target_registry_name)
        if obj is not None:
            return self._build_instance_from_object(target_registry_name=target_registry_name, target_obj=obj)
        meta = self._runtime_target_metadata.get(target_registry_name)
        if meta is not None and meta.is_broken:
            return self._build_broken_instance(meta)
        return None

    def get_target_object(self, *, target_registry_name: str) -> Any | None:
        """
        Get the actual target object for use in attacks.

        Returns:
            The PromptTarget object if found, None otherwise.
        """
        return self._registry.get_instance_by_name(target_registry_name)

    async def create_target_async(self, *, request: CreateTargetRequest) -> TargetInstance:
        """
        Create a new target instance from an API request.

        Instantiates the target with the given type and params and registers it
        in the registry under its registry name.

        Persistence policy:
            - If the request supplies an inline ``api_key`` in ``params``, the
              target is treated as **session-only**: it lives for the current
              backend process but is NOT written to the runtime targets file
              (we deliberately never persist secrets to disk in plain text).
              The returned ``TargetInstance`` is flagged ``session_only=True``
              with a ``persist_hint`` describing how the user can promote it
              to a persisted target (typically by setting the relevant api_key
              environment variable and recreating the target).
            - In all other cases (env-var-backed api_key, ``entra`` auth, or
              targets without an api_key concept like ``TextTarget``) a
              sanitized metadata record is persisted so the target can be
              replayed on the next backend start.

        Args:
            request: The create target request with type, params, and auth_mode.

        Returns:
            TargetInstance with the new target's details.

        Raises:
            ValueError: if any of the following occur:
                - Target type in request is not found in the class registry;
                - Entra ID auth is requested but the target type does not support it;
                - Entra ID auth is requested for an OpenAI target or AzureMLChatTarget
                    but the endpoint is not valid (not managed by correct hosts);
                - If auth_mode='api_key' is set for a target but no key is supplied
        """
        had_inline_api_key = request.auth_mode == "api_key" and bool(request.params.get("api_key"))

        target_obj, target_registry_name = self._instantiate_and_register(request=request)

        entry = RuntimeTargetEntry(
            target_registry_name=target_registry_name,
            type=request.type,
            auth_mode=request.auth_mode,
            params=sanitize_params(request.params),
        )

        if had_inline_api_key:
            target_class = self._get_target_class(target_type=request.type)
            persist_hint = self._build_persist_hint(target_class=target_class)
            self._runtime_target_metadata[target_registry_name] = _RuntimeTargetMetadata(
                entry=entry,
                session_only=True,
                persist_hint=persist_hint,
            )
        else:
            await self._runtime_store.append_async(entry)
            self._runtime_target_metadata[target_registry_name] = _RuntimeTargetMetadata(entry=entry)

        return self._build_instance_from_object(target_registry_name=target_registry_name, target_obj=target_obj)

    @staticmethod
    def _build_persist_hint(*, target_class: type) -> str:
        """
        Build a user-facing hint explaining why an inline-api-key target is
        session-only and how to make it persist across restarts.

        Args:
            target_class: The target class being instantiated, used to look
                up the canonical api_key environment variable name when one
                is declared.

        Returns:
            A human-readable hint string suitable for display in the UI.
        """
        env_var = _resolve_api_key_env_var(target_class)
        base = (
            "This target was created with an inline API key and will not survive a backend restart "
            "(API keys are never written to disk for security)."
        )
        if env_var:
            return (
                f"{base} To persist it across restarts, set the {env_var} environment variable "
                "in your shell or in ~/.pyrit/.env and recreate the target without the inline key. "
                "For Azure endpoints you can also use Microsoft Entra authentication."
            )
        return base

    def _instantiate_and_register(
        self,
        *,
        request: CreateTargetRequest,
    ) -> tuple[Any, str]:
        """
        Apply auth handling, construct the target, register it, and return the
        (object, registry name) pair. Shared by the create and restore paths.

        Returns:
            tuple[Any, str]: The instantiated PromptTarget and its registry name.
        """
        target_class = self._get_target_class(target_type=request.type)

        # Copy params so we can modify values (eg api_key) without changing request.params.
        params: dict[str, Any] = dict(request.params)

        if request.auth_mode == "entra":
            params = self._apply_entra_auth(target_class=target_class, target_type=request.type, params=params)
        else:
            self._validate_api_key_auth(target_class=target_class, params=params)

        target_obj = target_class(**params)
        self._registry.register_instance(target_obj)
        return target_obj, target_obj.get_identifier().unique_name

    async def delete_target_async(self, *, target_registry_name: str) -> None:
        """
        Remove a runtime-created target from the registry and the runtime store.

        Args:
            target_registry_name: The registry name of the target to remove.

        Raises:
            LookupError: If no target with that name is currently registered
                and there is no broken runtime entry under that name.
            PermissionError: If the target is registered but was not created at
                runtime (i.e., it belongs to an initializer; remove it from
                ``.pyrit_conf`` instead).
        """
        meta = self._runtime_target_metadata.get(target_registry_name)
        live = self._registry.get_instance_by_name(target_registry_name)

        if meta is None:
            if live is None:
                raise LookupError(f"Target '{target_registry_name}' not found.")
            raise PermissionError(
                f"Target '{target_registry_name}' was registered by an initializer and "
                "cannot be deleted via the API. Remove it from ~/.pyrit/.pyrit_conf instead."
            )

        if live is not None:
            self._registry.unregister(target_registry_name)
        self._runtime_target_metadata.pop(target_registry_name, None)
        await self._runtime_store.remove_async(target_registry_name)

    async def restore_runtime_targets_async(self) -> None:
        """
        Replay persisted runtime targets after initializers have seeded the registry.

        For each persisted entry:
        - If the registry already contains an instance with that name (e.g., an
          initializer registered one first), the runtime entry is skipped with
          a warning.
        - Otherwise the entry is reconstructed via the same code path as
          ``create_target_async``. Failures (typically a missing api_key env
          var) are recorded as ``needs_reconfiguration`` placeholders rather
          than aborting startup.
        """
        entries = await self._runtime_store.load_async()
        for entry in entries:
            if entry.target_registry_name in self._registry:
                logger.warning(
                    "Skipping runtime target %r: an initializer already registered a target with that name.",
                    entry.target_registry_name,
                )
                continue

            request = CreateTargetRequest(
                type=entry.type,
                params=dict(entry.params),
                auth_mode=entry.auth_mode,  # type: ignore[arg-type]
            )
            try:
                _, registry_name = self._instantiate_and_register(request=request)
            except Exception as exc:  # noqa: BLE001 — we deliberately catch all to keep startup resilient
                hint = self._build_reconfiguration_hint(entry=entry, exc=exc)
                logger.warning(
                    "Could not restore runtime target %r (%s): %s. "
                    "It will appear as 'needs reconfiguration' in the UI.",
                    entry.target_registry_name,
                    entry.type,
                    exc,
                )
                self._runtime_target_metadata[entry.target_registry_name] = _RuntimeTargetMetadata(
                    entry=entry,
                    is_broken=True,
                    reconfiguration_hint=hint,
                )
                continue

            if registry_name != entry.target_registry_name:
                # The identifier-derived name changed (e.g., params evolved); track under both
                # so DELETE on the persisted name still works.
                logger.info(
                    "Runtime target restored under registry name %r (persisted as %r).",
                    registry_name,
                    entry.target_registry_name,
                )
            self._runtime_target_metadata[registry_name] = _RuntimeTargetMetadata(entry=entry)

    @staticmethod
    def _build_reconfiguration_hint(*, entry: RuntimeTargetEntry, exc: BaseException) -> str:
        """
        Build a short user-facing hint for a target that failed to restore.

        Tries to surface the missing env var name when the failure is the
        standard ``_validate_api_key_auth`` ValueError. Falls back to the
        exception message for anything else.

        Returns:
            str: Human-readable hint to display next to the broken target.
        """
        message = str(exc).strip()
        if isinstance(exc, ValueError) and "environment variable" in message:
            return message
        return message or f"{type(exc).__name__} raised while restoring target {entry.target_registry_name!r}."

    @staticmethod
    def _apply_entra_auth(*, target_class: type, target_type: str, params: dict[str, Any]) -> dict[str, Any]:
        """
        Replace ``api_key`` in ``params`` with an Entra ID token provider for
        the given target class.

        Args:
            target_class (type): The target class being instantiated
            target_type (str): The user-facing target type name
            params (dict[str, Any]): The target constructor parameters from the request

        Returns:
            dict[str, Any]: A new params dict with ``api_key`` replaced by an async
            token-provider callable suitable for the target class.

        Raises:
            ValueError: If the target type does not support Entra ID, if an
                OpenAI target is given a non-Azure endpoint, or if an
                AzureMLChatTarget is given a non-AML endpoint.
        """
        new_params = dict(params)
        if "api_key" in new_params:
            logger.debug("Discarding 'api_key' from params because auth_mode='entra'.")
            new_params.pop("api_key", None)

        if issubclass(target_class, OpenAITarget):
            endpoint = new_params.get("endpoint")
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError("Entra ID authentication requires an 'endpoint' in params.")
            if not _is_azure_openai_endpoint(endpoint):
                raise ValueError(
                    "Entra ID authentication requires an Azure endpoint "
                    f"(*.openai.azure.com or *.ai.azure.com). Got: {endpoint}"
                )
            new_params["api_key"] = get_azure_openai_auth(endpoint)
            return new_params

        if issubclass(target_class, AzureMLChatTarget):
            endpoint = new_params.get("endpoint")
            if not isinstance(endpoint, str) or not endpoint:
                raise ValueError("Entra ID authentication requires an 'endpoint' in params.")
            if not _is_azure_ml_endpoint(endpoint):
                raise ValueError(
                    "Entra ID authentication for AzureMLChatTarget requires an AML endpoint "
                    f"(*.inference.ml.azure.com). Got: {endpoint}"
                )
            new_params["api_key"] = get_azure_async_token_provider(_AZURE_ML_SCOPE)
            return new_params

        raise ValueError(
            f"Target type '{target_type}' does not support Entra ID authentication. "
            "Supported types are OpenAI-family targets and AzureMLChatTarget."
        )

    @staticmethod
    def _validate_api_key_auth(*, target_class: type, params: dict[str, Any]) -> None:
        """
        Enforce that ``auth_mode='api_key'`` actually has a usable key.

        Targets that do not authenticate via an api_key (e.g. ``TextTarget``)
        are skipped since they have no env var and the underlying
        constructor does not take any ``api_key`` arguments.

        Args:
            target_class (type): The target class being instantiated.
            params (dict[str, Any]): The constructor parameters from the request.

        Raises:
            ValueError: If no API key is provided in params or in the relevant
                environment variable for a target class that authenticates via
                an API key.
        """
        env_var = _resolve_api_key_env_var(target_class)
        if env_var is None:
            return

        if params.get("api_key"):
            return
        if os.environ.get(env_var):
            return

        raise ValueError(
            f"auth_mode='api_key' requires an API key but none was provided. "
            f"Pass 'api_key' in params or set the {env_var} environment variable. "
            "To authenticate with Microsoft Entra ID instead, set auth_mode='entra'."
        )


@lru_cache(maxsize=1)
def get_target_service() -> TargetService:
    """
    Get the global target service instance.

    Returns:
        The singleton TargetService instance.
    """
    return TargetService()
