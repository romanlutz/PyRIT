# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target service for managing target instances.

Uses ``TargetRegistry`` as the source of truth for live instances and persists
sanitized metadata for API-created targets so they can be restored after a
backend restart.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from pyrit.backend.mappers.target_mappers import target_object_to_instance
from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.models.targets import (
    CreateTargetRequest,
    TargetCatalogEntry,
    TargetCatalogResponse,
    TargetListResponse,
)
from pyrit.backend.persistence.runtime_targets import (
    RuntimeTargetEntry,
    RuntimeTargetStore,
    contains_sensitive_params,
    get_runtime_target_store,
    sanitize_params,
)
from pyrit.models import TargetCapabilities, TargetIdentifier
from pyrit.models.catalog.target import TargetInstance
from pyrit.prompt_target import PromptTarget
from pyrit.registry import TargetRegistry

logger = logging.getLogger(__name__)


@dataclass
class _RuntimeTargetMetadata:
    """In-memory state for an API-created target."""

    entry: RuntimeTargetEntry
    is_broken: bool = False
    reconfiguration_hint: str | None = None
    session_only: bool = False
    persist_hint: str | None = None


def _resolve_credential_env_var(target_class: type[PromptTarget]) -> str | None:
    """
    Resolve the API-key environment variable declared by a target class.

    OpenAI target subclasses initialize this value through a lightweight setter,
    so an uninitialized instance is sufficient and avoids running constructors.

    Args:
        target_class (type[PromptTarget]): The target class to inspect.

    Returns:
        str | None: The environment-variable name, if the target declares one.
    """
    for attribute_name in (
        "api_key_environment_variable",
        "API_KEY_ENVIRONMENT_VARIABLE",
        "SAS_TOKEN_ENVIRONMENT_VARIABLE",
        "HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE",
    ):
        env_var = getattr(target_class, attribute_name, None)
        if isinstance(env_var, str) and env_var:
            return env_var

    try:
        instance = target_class.__new__(target_class)
        setter = getattr(instance, "_set_openai_env_configuration_vars", None)
        if callable(setter):
            setter()
    except Exception:
        return None

    env_var = getattr(instance, "api_key_environment_variable", None)
    return env_var if isinstance(env_var, str) and env_var else None


def _optional_string(value: Any) -> str | None:
    """Return a non-empty string value, or None."""
    return value if isinstance(value, str) and value else None


def _optional_float(value: Any) -> float | None:
    """Return a numeric value as float, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    """Return an integer value, or None."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


class TargetService:
    """
    Service for managing target instances.

    Class discovery, parameter coercion, reference resolution, and endpoint
    validation remain owned by the current ``TargetRegistry`` and target
    classes. This service adds runtime persistence and deletion policy.
    """

    def __init__(self, *, runtime_store: RuntimeTargetStore | None = None) -> None:
        """
        Initialize the target service.

        Args:
            runtime_store (RuntimeTargetStore | None): Optional persistence store
                override, primarily for tests.
        """
        self._registry = TargetRegistry.get_registry_singleton()
        self._runtime_store = runtime_store or get_runtime_target_store()
        self._runtime_target_metadata: dict[str, _RuntimeTargetMetadata] = {}

    def _build_instance_from_object(
        self,
        *,
        target_registry_name: str,
        target_obj: PromptTarget,
    ) -> TargetInstance:
        """
        Build a target DTO and layer on runtime metadata.

        Returns:
            TargetInstance: The target's current API representation.
        """
        metadata = self._runtime_target_metadata.get(target_registry_name)
        return target_object_to_instance(
            target_registry_name,
            target_obj,
            is_runtime=metadata is not None,
            needs_reconfiguration=metadata.is_broken if metadata else False,
            reconfiguration_hint=metadata.reconfiguration_hint if metadata else None,
            session_only=metadata.session_only if metadata else False,
            persist_hint=metadata.persist_hint if metadata else None,
        )

    @staticmethod
    def _build_broken_instance(metadata: _RuntimeTargetMetadata) -> TargetInstance:
        """
        Build a selectable-disabled DTO for a target that failed to restore.

        Returns:
            TargetInstance: A target representation marked as needing reconfiguration.
        """
        params = metadata.entry.params
        identifier = TargetIdentifier(
            class_name=metadata.entry.type,
            class_module="pyrit.prompt_target",
            endpoint=_optional_string(params.get("endpoint")),
            model_name=_optional_string(params.get("model_name")),
            underlying_model_name=_optional_string(params.get("underlying_model_name"))
            or _optional_string(params.get("underlying_model")),
            temperature=_optional_float(params.get("temperature")),
            top_p=_optional_float(params.get("top_p")),
            max_requests_per_minute=_optional_int(params.get("max_requests_per_minute")),
        )
        return TargetInstance(
            target_registry_name=metadata.entry.target_registry_name,
            identifier=identifier,
            capabilities=TargetCapabilities(),
            is_runtime=True,
            needs_reconfiguration=True,
            reconfiguration_hint=metadata.reconfiguration_hint,
        )

    async def list_targets_async(
        self,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> TargetListResponse:
        """
        List live and unrestorable runtime targets with pagination.

        Returns:
            TargetListResponse: The requested page of target representations.
        """
        live_items = [
            self._build_instance_from_object(target_registry_name=entry.name, target_obj=entry.instance)
            for entry in self._registry.instances.get_all_instances()
        ]
        registered_names = {item.target_registry_name for item in live_items}
        broken_items = [
            self._build_broken_instance(metadata)
            for name, metadata in self._runtime_target_metadata.items()
            if metadata.is_broken and name not in registered_names
        ]
        items = sorted(live_items + broken_items, key=lambda item: item.target_registry_name)
        page, has_more = self._paginate(items=items, cursor=cursor, limit=limit)
        next_cursor = page[-1].target_registry_name if has_more and page else None
        return TargetListResponse(
            items=page,
            pagination=PaginationInfo(
                limit=limit,
                has_more=has_more,
                next_cursor=next_cursor,
                prev_cursor=cursor,
            ),
        )

    @staticmethod
    def _paginate(
        *,
        items: list[TargetInstance],
        cursor: str | None,
        limit: int,
    ) -> tuple[list[TargetInstance], bool]:
        """
        Apply cursor-based pagination.

        Returns:
            tuple[list[TargetInstance], bool]: The page and whether another page exists.
        """
        start_index = 0
        if cursor:
            for index, item in enumerate(items):
                if item.target_registry_name == cursor:
                    start_index = index + 1
                    break

        page = items[start_index : start_index + limit]
        return page, len(items) > start_index + limit

    async def get_target_async(self, *, target_registry_name: str) -> TargetInstance | None:
        """
        Get a live target or an unrestorable runtime-target placeholder.

        Returns:
            TargetInstance | None: The target representation, or None if unknown.
        """
        target_obj = self._registry.instances.get(target_registry_name)
        if target_obj is not None:
            return self._build_instance_from_object(
                target_registry_name=target_registry_name,
                target_obj=target_obj,
            )

        metadata = self._runtime_target_metadata.get(target_registry_name)
        if metadata is not None and metadata.is_broken:
            return self._build_broken_instance(metadata)
        return None

    def get_target_object(self, *, target_registry_name: str) -> PromptTarget | None:
        """
        Get the live target object used by attacks.

        Returns:
            PromptTarget | None: The live target, or None if it is unavailable.
        """
        return self._registry.instances.get(target_registry_name)

    async def list_target_catalog_async(self) -> TargetCatalogResponse:
        """
        List all constructible target classes and their current contracts.

        Returns:
            TargetCatalogResponse: The target class catalog.
        """
        items: list[TargetCatalogEntry] = [
            TargetCatalogEntry(
                target_type=metadata.class_name,
                parameters=[parameter for parameter in metadata.parameters if parameter.is_string_coercible],
                supported_auth_modes=cast(
                    "list[Literal['api_key', 'identity']]",
                    list(metadata.supported_auth_modes),
                ),
                description=metadata.class_description or None,
            )
            for metadata in self._registry.get_all_registered_class_metadata()
        ]
        return TargetCatalogResponse(items=items)

    async def create_target_async(self, *, request: CreateTargetRequest) -> TargetInstance:
        """
        Create, register, and conditionally persist a runtime target.

        Inline credentials are never written to disk. Such targets remain
        deletable for the current process and are returned with
        ``session_only=True``.
        Identity-backed, environment-backed, and no-key targets persist sanitized
        constructor metadata for startup replay.

        Args:
            request (CreateTargetRequest): Target type, parameters, and auth mode.

        Returns:
            TargetInstance: The newly registered target.
        """
        target_obj = self._instantiate_target(request=request)
        target_registry_name = target_obj.get_identifier().unique_name
        self._ensure_runtime_name_available(target_registry_name=target_registry_name)

        entry = RuntimeTargetEntry(
            target_registry_name=target_registry_name,
            type=request.type,
            auth_mode=request.auth_mode,
            params=sanitize_params(request.params),
        )
        has_inline_credential = request.auth_mode == "api_key" and contains_sensitive_params(request.params)

        if has_inline_credential:
            await self._runtime_store.remove_async(target_registry_name)
            metadata = _RuntimeTargetMetadata(
                entry=entry,
                session_only=True,
                persist_hint=self._build_persist_hint(target_class=type(target_obj)),
            )
        else:
            await self._runtime_store.append_async(entry)
            metadata = _RuntimeTargetMetadata(entry=entry)

        self._registry.instances.register(target_obj, name=target_registry_name)
        self._runtime_target_metadata[target_registry_name] = metadata
        return self._build_instance_from_object(
            target_registry_name=target_registry_name,
            target_obj=target_obj,
        )

    def _instantiate_target(self, *, request: CreateTargetRequest) -> PromptTarget:
        """
        Construct a target through the current registry contract.

        Returns:
            PromptTarget: The configured target, not yet registered.
        """
        if request.type not in self._registry:
            raise ValueError(
                f"Target type '{request.type}' not found. Available types: {self._registry.get_class_names()}"
            )

        target_class = self._registry.get_class(request.type)
        params: dict[str, Any] = dict(request.params)
        if request.auth_mode == "identity":
            if "identity" not in target_class.supported_auth_modes:
                raise ValueError(f"Target type '{request.type}' does not support identity-based authentication.")
            params = sanitize_params(params)

        if self._has_reference_params(target_type=request.type):
            return self._registry.create_instance(request.type, **params)
        return target_class(**params)

    def _ensure_runtime_name_available(self, *, target_registry_name: str) -> None:
        """Prevent an API request from converting an initializer target into a deletable target."""
        existing = self._registry.instances.get(target_registry_name)
        if existing is not None and target_registry_name not in self._runtime_target_metadata:
            raise ValueError(
                f"Target '{target_registry_name}' is already registered by an initializer and cannot be replaced."
            )

    @staticmethod
    def _build_persist_hint(*, target_class: type[PromptTarget]) -> str:
        """
        Build the warning shown for a target created with an inline credential.

        Returns:
            str: The user-facing persistence guidance.
        """
        env_var = _resolve_credential_env_var(target_class)
        base = (
            "This target was created with an inline API key or credential and will not survive a backend restart "
            "(credentials are never written to disk for security)."
        )
        if not env_var:
            return base
        return (
            f"{base} To persist it across restarts, set the {env_var} environment variable "
            "in your shell or in ~/.pyrit/.env and recreate the target without the inline key. "
            "For Azure endpoints you can also use Microsoft Entra authentication."
        )

    async def delete_target_async(self, *, target_registry_name: str) -> None:
        """
        Delete a runtime-created target from memory and persistent storage.

        Args:
            target_registry_name (str): The registry name to delete.

        Raises:
            LookupError: If no target or broken runtime entry has this name.
            PermissionError: If the name belongs to an initializer target.
        """
        metadata = self._runtime_target_metadata.get(target_registry_name)
        live_target = self._registry.instances.get(target_registry_name)
        if metadata is None:
            if live_target is None:
                raise LookupError(f"Target '{target_registry_name}' not found.")
            raise PermissionError(
                f"Target '{target_registry_name}' was registered by an initializer and "
                "cannot be deleted via the API. Remove it from ~/.pyrit/.pyrit_conf instead."
            )

        if not metadata.session_only:
            await self._runtime_store.remove_async(target_registry_name)
        if live_target is not None:
            self._registry.instances.unregister(target_registry_name)
        self._runtime_target_metadata.pop(target_registry_name, None)

    async def restore_runtime_targets_async(self) -> None:
        """Restore persisted targets in dependency-independent passes."""
        pending = await self._runtime_store.load_async()
        while pending:
            failures: list[tuple[RuntimeTargetEntry, Exception]] = []
            restored_any = False
            for entry in pending:
                if entry.target_registry_name in self._registry.instances:
                    if entry.target_registry_name not in self._runtime_target_metadata:
                        await self._remove_initializer_conflict_async(entry=entry)
                    continue
                try:
                    restored_any = self._restore_runtime_target(entry=entry) or restored_any
                except Exception as exc:  # noqa: BLE001
                    failures.append((entry, exc))

            if not failures:
                return
            if not restored_any:
                for entry, exc in failures:
                    self._record_restore_failure(entry=entry, exc=exc)
                return
            pending = [entry for entry, _ in failures]

    async def _remove_initializer_conflict_async(self, *, entry: RuntimeTargetEntry) -> None:
        """Remove a persisted entry whose name is now owned by an initializer."""
        logger.warning(
            "Skipping runtime target %r: an initializer already registered a target with that name. "
            "Removing the conflicting persisted entry.",
            entry.target_registry_name,
        )
        try:
            await self._runtime_store.remove_async(entry.target_registry_name)
        except OSError as exc:
            logger.error(
                "Could not remove conflicting runtime target %r from %s: %s",
                entry.target_registry_name,
                self._runtime_store.path,
                exc,
            )

    def _restore_runtime_target(self, *, entry: RuntimeTargetEntry) -> bool:
        """
        Restore one persisted runtime target.

        Returns:
            bool: False when the name is already registered, otherwise True
                after successful restoration.
        """
        if entry.target_registry_name in self._registry.instances:
            return False

        request = CreateTargetRequest(type=entry.type, params=dict(entry.params), auth_mode=entry.auth_mode)
        target_obj = self._instantiate_target(request=request)
        self._registry.instances.register(target_obj, name=entry.target_registry_name)
        self._runtime_target_metadata[entry.target_registry_name] = _RuntimeTargetMetadata(entry=entry)
        return True

    def _record_restore_failure(self, *, entry: RuntimeTargetEntry, exc: Exception) -> None:
        """Record a target that still cannot be restored after dependency retries."""
        hint = self._build_reconfiguration_hint(entry=entry, exc=exc)
        logger.warning(
            "Could not restore runtime target %r (%s): %s. It will appear as 'needs reconfiguration' in the UI.",
            entry.target_registry_name,
            entry.type,
            exc,
        )
        self._runtime_target_metadata[entry.target_registry_name] = _RuntimeTargetMetadata(
            entry=entry,
            is_broken=True,
            reconfiguration_hint=hint,
        )

    @staticmethod
    def _build_reconfiguration_hint(*, entry: RuntimeTargetEntry, exc: BaseException) -> str:
        """
        Build a user-facing hint for a target that failed startup restoration.

        Returns:
            str: The restoration failure message.
        """
        message = str(exc).strip()
        return message or f"{type(exc).__name__} raised while restoring target {entry.target_registry_name!r}."

    def _has_reference_params(self, *, target_type: str) -> bool:
        """Return whether construction must resolve references through the registry."""
        metadata = self._registry.get_registered_class_metadata(target_type)
        if metadata is None:
            return False
        return any(parameter.reference is not None for parameter in metadata.parameters)


@lru_cache(maxsize=1)
def get_target_service() -> TargetService:
    """
    Get the process-wide target service.

    Returns:
        TargetService: The cached service instance.
    """
    return TargetService()
