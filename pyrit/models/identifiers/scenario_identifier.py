# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strongly-typed projection of a scenario's identifier."""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, ClassVar

from pydantic import Field, model_validator

from pyrit.models.eval_case import EvalSourceKind
from pyrit.models.identifiers.component_identifier import ComponentIdentifier
from pyrit.models.identifiers.evaluation_markers import Evaluate
from pyrit.models.identifiers.param_markers import Param
from pyrit.models.identifiers.scorer_identifier import (  # noqa: TC001
    ScorerIdentifier,  # runtime-required by Pydantic field annotations
)
from pyrit.models.identifiers.target_identifier import (  # noqa: TC001
    TargetIdentifier,  # runtime-required by Pydantic field annotations
)
from pyrit.models.parameter import ComponentType


class ScenarioExecutionOwner(str, Enum):
    """Which component owns the target and grade for this Scenario run."""

    TASK_OWNED = "task_owned"
    APPROVED_ORIGINAL = "approved_original"


class ScenarioIdentifier(ComponentIdentifier):
    """
    Strongly-typed projection of a ``Scenario``'s ``ComponentIdentifier``.

    Like the sibling projections (``TargetIdentifier`` / ``ScorerIdentifier``),
    this is produced by the scenario registry when a scenario is built. It is also
    the canonical per-run identity carried on the ``ScenarioResult`` aggregate and
    persisted with it: the scenario class name (``class_name``), definition
    ``version``, resolved ``techniques`` / ``datasets``, the resolved scenario
    ``params``, and the ``objective_target`` / ``objective_scorer`` child
    references all live here rather than as separate denormalized fields. Its eval
    hash (via ``ScenarioEvaluationIdentifier``) backs resume drift detection.

    Promotes the scenario's behavioral identity to typed ``params`` fields that
    feed both the content and eval hash: the definition ``version`` and the
    resolved ``techniques`` / ``datasets`` the scenario runs (a v1 vs a v2, or a
    different technique / dataset selection, is a different identity). The two
    run-resolved reference slots — ``objective_target`` (a ``PromptTarget``) and
    ``objective_scorer`` (a ``Scorer``) — are promoted children the registry
    resolves by name from the target / scorer registries when building a scenario.
    Explicitly task-owned scenarios instead identify their safe public source,
    execution profile, and case selection in ``params``; neither child exists.
    """

    component_type: ClassVar[ComponentType] = ComponentType.SCENARIO
    TASK_OWNED_PARAM_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "execution_owner",
            "eval_spec_sha256",
            "source_kind",
            "source_name",
            "source_sha256",
            "harness_name",
            "harness_sha256",
            "model_route_name",
            "model_route_sha256",
            "case_set_sha256",
            "server_admission_required",
            "input_variant_sha256",
            "input_surface_id",
            "version",
            "techniques",
            "datasets",
        }
    )

    #: Scenario definition version. Behavioral identity (a v1 and a v2 of the same
    #: scenario are different identities); not a constructor input.
    version: Annotated[int | None, Evaluate.Include(), Param.Exclude()] = None
    #: Resolved technique names the scenario runs. Behavioral identity; not a
    #: constructor input (the registry populates it from the selected strategies).
    techniques: Annotated[list[str] | None, Evaluate.Include(), Param.Exclude()] = None
    #: Resolved dataset names the scenario runs. Behavioral identity; not a
    #: constructor input (the registry populates it from the dataset config).
    datasets: Annotated[list[str] | None, Evaluate.Include(), Param.Exclude()] = None
    #: Target the scenario attacks. Run-resolved reference resolved by name from
    #: the target registry.
    objective_target: Annotated[TargetIdentifier | None, Evaluate.Include(), Param.Include()] = Field(default=None)
    #: Primary scorer the scenario evaluates with. Run-resolved reference resolved
    #: by name from the scorer registry.
    objective_scorer: Annotated[ScorerIdentifier | None, Evaluate.Include(), Param.Include()] = Field(default=None)

    @model_validator(mode="after")
    def _validate_task_owned_identity(self) -> ScenarioIdentifier:
        """
        Require an honest target-free, scorer-free, fingerprinted task identity.

        Returns:
            ScenarioIdentifier: The validated identity.

        Raises:
            ValueError: If a task-owned identity supplies a target, scorer, or unsafe reference.
        """
        if self.params.get("execution_owner") == ScenarioExecutionOwner.APPROVED_ORIGINAL.value:
            if (
                self.class_name != "ServerApprovedOriginalScenario"
                or self.objective_target is not None
                or self.objective_scorer is not None
                or set(self.params) != {"execution_owner", "version", "techniques", "datasets"}
                or self.version != 1
                or self.techniques
                or self.datasets
            ):
                raise ValueError("Approved original job identities may contain only the server-owned reference.")
            return self
        if self.params.get("execution_owner") != ScenarioExecutionOwner.TASK_OWNED.value:
            return self
        if self.objective_target is not None or self.objective_scorer is not None:
            raise ValueError("Task-owned Scenario identifiers cannot contain an external target or scorer")
        unknown = set(self.params) - self.TASK_OWNED_PARAM_NAMES
        if unknown:
            raise ValueError(f"Task-owned Scenario identifiers contain unsupported parameters: {sorted(unknown)}")
        if "server_admission_required" in self.params and self.params["server_admission_required"] is not True:
            raise ValueError("A server-admitted Task must declare an explicit protected Scenario identity")
        if self.params.get("source_kind") not in {kind.value for kind in EvalSourceKind}:
            raise ValueError("Task-owned Scenario identifiers require a known source_kind")
        for name in ("eval_spec_sha256", "source_sha256", "harness_sha256", "model_route_sha256", "case_set_sha256"):
            value = self.params.get(name)
            if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
                raise ValueError(f"Task-owned Scenario identifiers require a valid {name}")
        for name in ("source_name", "harness_name", "model_route_name"):
            value = self.params.get(name)
            if not isinstance(value, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", value) is None:
                raise ValueError(f"Task-owned Scenario identifiers require a public {name} alias")
        variant_sha = self.params.get("input_variant_sha256")
        surface = self.params.get("input_surface_id")
        if (variant_sha is None) != (surface is None):
            raise ValueError("Task-owned Scenario input variants require both a digest and a surface")
        if variant_sha is not None:
            if not isinstance(variant_sha, str) or re.fullmatch(r"[0-9a-f]{64}", variant_sha) is None:
                raise ValueError("Task-owned Scenario identifiers require a valid input_variant_sha256")
            if not isinstance(surface, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", surface) is None:
                raise ValueError("Task-owned Scenario identifiers require a public input_surface_id alias")
        return self
