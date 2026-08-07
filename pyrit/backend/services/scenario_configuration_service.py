# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared launch-aligned Scenario configuration resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from pyrit.registry import ConverterRegistry

if TYPE_CHECKING:
    from pyrit.converter import Converter
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario import Scenario

_CONVERTER_MODIFIER_PREFIX = "converter."


class ScenarioConfiguration(Protocol):
    """Fields shared by run and estimate requests."""

    techniques: list[str] | None
    dataset_names: list[str] | None
    max_dataset_size: int | None
    dataset_filters: dict[str, list[str]] | None
    include_baseline: bool | None


class ScenarioConfigurationService:
    """Resolve API selections through a Scenario's own technique and dataset types."""

    def __init__(self, *, converter_registry: type[ConverterRegistry] = ConverterRegistry) -> None:
        """
        Initialize with the converter registry class used by the API process.

        Args:
            converter_registry: Registry class supplying converter instances.
        """
        self._converter_registry = converter_registry

    def build_initialization_kwargs(
        self,
        *,
        configuration: ScenarioConfiguration,
        scenario_name: str,
        scenario_class: type[Scenario],
        objective_target: PromptTarget,
    ) -> dict[str, Any]:
        """
        Build the launch-aligned Scenario initialization arguments.

        Returns:
            dict[str, Any]: Arguments for Scenario estimation or initialization.
        """
        init_kwargs: dict[str, Any] = {"objective_target": objective_target}
        if configuration.include_baseline is not None:
            init_kwargs["include_baseline"] = configuration.include_baseline

        dataset_filters = configuration.dataset_filters or {}
        needs_introspection = (
            bool(configuration.techniques)
            or bool(configuration.dataset_names)
            or configuration.max_dataset_size is not None
            or bool(dataset_filters)
        )
        if not needs_introspection:
            return init_kwargs

        try:
            introspection_instance = scenario_class()  # type: ignore[ty:missing-argument]
        except Exception as exc:
            raise ValueError(
                f"Cannot resolve runtime configuration for scenario '{scenario_name}': "
                f"scenario class is not instantiable without arguments ({exc})."
            ) from exc

        if configuration.techniques:
            technique_enums, technique_converters = self.resolve_techniques_and_converters(
                tokens=configuration.techniques,
                technique_class=introspection_instance._technique_class,
                scenario_name=scenario_name,
            )
            init_kwargs["scenario_techniques"] = technique_enums
            if technique_converters:
                init_kwargs["technique_converters"] = technique_converters

        if configuration.dataset_names or configuration.max_dataset_size is not None or dataset_filters:
            default_config = introspection_instance._default_dataset_config
            if configuration.dataset_names:
                default_config_class = type(default_config)
                try:
                    init_kwargs["dataset_config"] = default_config_class(
                        dataset_names=configuration.dataset_names,
                        max_dataset_size=configuration.max_dataset_size,
                        filters=dataset_filters or None,
                    )
                except TypeError as exc:
                    raise ValueError(
                        f"Scenario '{scenario_name}' does not support overriding dataset names through "
                        f"its {default_config_class.__name__} configuration: {exc}"
                    ) from exc
            else:
                if configuration.max_dataset_size is not None:
                    default_config.max_dataset_size = configuration.max_dataset_size
                if dataset_filters:
                    default_config.update_filters(filters=dataset_filters)
                init_kwargs["dataset_config"] = default_config

        return init_kwargs

    def resolve_techniques_and_converters(
        self,
        *,
        tokens: list[str],
        technique_class: type[Any],
        scenario_name: str,
    ) -> tuple[list[Any], dict[str, list[Converter]]]:
        """
        Resolve technique tokens and converter modifiers using the launch path.

        Returns:
            tuple: Resolved technique enums and concrete-technique converter mappings.
        """
        technique_enums: list[Any] = []
        technique_converters: dict[str, list[Converter]] = {}
        for token in tokens:
            base_name, _, remainder = token.partition(":")
            modifiers = [modifier for modifier in remainder.split(":") if modifier] if remainder else []
            try:
                technique_enum = technique_class(base_name)
            except ValueError:
                available_techniques = [technique.value for technique in technique_class]
                raise ValueError(
                    f"Technique '{base_name}' not found for scenario '{scenario_name}'. "
                    f"Available: {', '.join(available_techniques)}"
                ) from None
            technique_enums.append(technique_enum)

            converters = self.resolve_converter_modifiers(modifiers=modifiers, token=token)
            if not converters:
                continue
            for concrete in technique_class.expand({technique_enum}):
                technique_converters.setdefault(concrete.value, []).extend(converters)

        return technique_enums, technique_converters

    def resolve_converter_modifiers(self, *, modifiers: list[str], token: str) -> list[Converter]:
        """
        Resolve converter modifiers against registered converter instances.

        Returns:
            list[Converter]: Registered converter instances in token order.
        """
        if not modifiers:
            return []

        instances = self._converter_registry.get_registry_singleton().instances
        converters: list[Converter] = []
        for modifier in modifiers:
            if not modifier.startswith(_CONVERTER_MODIFIER_PREFIX):
                raise ValueError(
                    f"Unknown technique modifier '{modifier}' in '{token}'. "
                    f"Supported modifiers must use the '{_CONVERTER_MODIFIER_PREFIX}' prefix "
                    f"(e.g. '{_CONVERTER_MODIFIER_PREFIX}translation_spanish')."
                )
            converter_name = modifier[len(_CONVERTER_MODIFIER_PREFIX) :]
            converter = instances.get(converter_name)
            if converter is None:
                available = instances.get_names()
                available_text = ", ".join(available) if available else "(none registered)"
                raise ValueError(
                    f"Converter '{converter_name}' in '{token}' is not a registered converter "
                    f"instance. Available converters: {available_text}"
                )
            converters.append(converter)
        return converters
