# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Target mappers – domain → DTO translation for target-related models.
"""

from pyrit.backend.models.targets import TargetCapabilitiesInfo, TargetInstance
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import CapabilityName, TargetCapabilities

# Capability flag names that should never be surfaced as identifier-level params:
# they are sourced from `target_obj.capabilities` instead.
_CAPABILITY_PARAM_NAMES = frozenset(cap.value for cap in CapabilityName)


def _target_capabilities_to_info(capabilities: TargetCapabilities) -> TargetCapabilitiesInfo:
    """
    Build a TargetCapabilitiesInfo DTO from a domain TargetCapabilities object.

    Modality combinations are flattened into sorted unique modality lists since
    the frontend uses them only for per-piece modality checks.

    Args:
        capabilities: The domain TargetCapabilities object.

    Returns:
        TargetCapabilitiesInfo DTO mirroring the capability flags and flattened
        input/output modalities.
    """
    return TargetCapabilitiesInfo(
        supports_multi_turn=capabilities.supports_multi_turn,
        supports_multi_message_pieces=capabilities.supports_multi_message_pieces,
        supports_json_schema=capabilities.supports_json_schema,
        supports_json_output=capabilities.supports_json_output,
        supports_editable_history=capabilities.supports_editable_history,
        supports_system_prompt=capabilities.supports_system_prompt,
        supported_input_modalities=sorted({str(t) for combo in capabilities.input_modalities for t in combo}),
        supported_output_modalities=sorted({str(t) for combo in capabilities.output_modalities for t in combo}),
    )


def target_object_to_instance(
    target_registry_name: str,
    target_obj: PromptTarget,
    *,
    is_runtime: bool = False,
    needs_reconfiguration: bool = False,
    reconfiguration_hint: str | None = None,
    session_only: bool = False,
    persist_hint: str | None = None,
) -> TargetInstance:
    """
    Build a TargetInstance DTO from a registry target object.

    Extracts only the frontend-relevant fields from the internal identifier,
    avoiding leakage of internal PyRIT core structures.

    Args:
        target_registry_name: The human-friendly target registry name.
        target_obj: The domain PromptTarget object from the registry.
        is_runtime: True if this target was created at runtime via the API and
            (when ``session_only`` is False) persisted to the runtime targets file.
            Runtime targets are deletable via DELETE /api/targets/{name}.
        needs_reconfiguration: True if the target was restored from disk on
            startup but could not be fully reconstructed (e.g., the required
            api_key environment variable is missing).
        reconfiguration_hint: Optional human-readable hint for the
            ``needs_reconfiguration`` case (e.g., the missing env var name).
        session_only: True for runtime targets created with an inline api_key
            and therefore intentionally not persisted to disk.
        persist_hint: Optional human-readable hint explaining how the user can
            promote a session-only target to a persistent one.

    Returns:
        TargetInstance DTO with metadata derived from the object.
    """
    identifier = target_obj.get_identifier()
    params = identifier.params

    # Keys that are extracted as top-level TargetInstance fields, are internal-only
    # (e.g., target_configuration is the verbose capabilities blob), or duplicate
    # capability flags (filtered via _CAPABILITY_PARAM_NAMES) — those are sourced
    # solely from target_obj.capabilities and must not leak into target_specific_params.
    extracted_keys = {
        "endpoint",
        "model_name",
        "underlying_model_name",
        "temperature",
        "top_p",
        "max_requests_per_minute",
        "target_specific_params",
        "target_configuration",
    } | _CAPABILITY_PARAM_NAMES

    # Collect remaining params as target_specific_params so the frontend can display them
    explicit_specific = params.get("target_specific_params") or {}
    extra = {k: v for k, v in params.items() if k not in extracted_keys and v is not None}
    combined_specific = {**extra, **explicit_specific} or None

    return TargetInstance(
        target_registry_name=target_registry_name,
        target_type=identifier.class_name,
        endpoint=params.get("endpoint") or None,
        model_name=params.get("model_name") or None,
        underlying_model_name=params.get("underlying_model_name") or None,
        temperature=params.get("temperature"),
        top_p=params.get("top_p"),
        max_requests_per_minute=params.get("max_requests_per_minute"),
        capabilities=_target_capabilities_to_info(target_obj.capabilities),
        target_specific_params=combined_specific,
        is_runtime=is_runtime,
        needs_reconfiguration=needs_reconfiguration,
        reconfiguration_hint=reconfiguration_hint,
        session_only=session_only,
        persist_hint=persist_hint,
    )
