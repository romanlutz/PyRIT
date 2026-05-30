# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Identifiers module for PyRIT components."""

from typing import Any

from pyrit.common.deprecation import print_deprecation_message
from pyrit.identifiers.atomic_attack_identifier import (
    build_atomic_attack_identifier,
    build_seed_identifier,
)
from pyrit.identifiers.class_name_utils import (
    REGISTRY_NAME_PATTERN,
    class_name_to_snake_case,
    snake_case_to_class_name,
    validate_registry_name,
)
from pyrit.identifiers.component_identifier import ComponentIdentifier, Identifiable, config_hash
from pyrit.identifiers.evaluation_identifier import (
    TARGET_EVAL_PARAM_FALLBACKS,
    TARGET_EVAL_PARAMS,
    AtomicAttackEvaluationIdentifier,
    ChildEvalRule,
    EvaluationIdentifier,
    ScorerEvaluationIdentifier,
    compute_eval_hash,
)
from pyrit.identifiers.identifier_filters import IdentifierFilter, IdentifierType

__all__ = [
    "AtomicAttackEvaluationIdentifier",
    "build_atomic_attack_identifier",
    "build_seed_identifier",
    "ChildEvalRule",
    "class_name_to_snake_case",
    "ComponentIdentifier",
    "compute_eval_hash",
    "EvaluationIdentifier",
    "Identifiable",
    "REGISTRY_NAME_PATTERN",
    "ScorerEvaluationIdentifier",
    "snake_case_to_class_name",
    "TARGET_EVAL_PARAM_FALLBACKS",
    "TARGET_EVAL_PARAMS",
    "validate_registry_name",
    "config_hash",
    "IdentifierFilter",
    "IdentifierType",
]


# Deprecated aliases for names removed in #1387 (ScorerIdentifier et al. were collapsed into
# ComponentIdentifier). Kept temporarily so external partners that depend on the old import path
# (e.g. azure-ai-evaluation) keep working until they migrate. Will be removed in 0.16.0.
_DEPRECATED_ALIASES: dict[str, Any] = {
    "ScorerIdentifier": ComponentIdentifier,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        target = _DEPRECATED_ALIASES[name]
        print_deprecation_message(
            old_item=f"pyrit.identifiers.{name}",
            new_item=target,
            removed_in="0.16.0",
        )
        return target
    raise AttributeError(f"module 'pyrit.identifiers' has no attribute {name!r}")
