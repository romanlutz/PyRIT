# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strict compatibility profiles for pinned Inspect-evals source."""

from __future__ import annotations

from dataclasses import dataclass


class InspectProfileMismatchError(ValueError):
    """Raised when a source or API profile does not match the pinned contract."""


class UnsupportedInspectFeatureError(NotImplementedError):
    """Raised when source requests an Inspect symbol outside the active profile."""

    def __init__(
        self,
        *,
        symbol: str,
        source_profile: str,
        remediation: str | None = None,
    ) -> None:
        """Initialize a precise unsupported-feature failure."""
        self.symbol = symbol
        self.source_profile = source_profile
        self.remediation = remediation or (
            "Use a supported symbol, select a native PyRIT adapter, or add an explicitly tested compatibility profile."
        )
        super().__init__(
            f"Inspect symbol '{symbol}' is unsupported by source profile '{source_profile}'. "
            f"Remediation: {self.remediation}"
        )


@dataclass(frozen=True)
class InspectCompatibilityProfile:
    """A closed set of symbols and capabilities for one pinned source revision."""

    profile_id: str
    inspect_evals_revision: str
    inspect_api_profile: str
    supported_symbols: frozenset[str]
    capabilities: tuple[tuple[str, bool], ...]

    def require_symbol(self, symbol: str) -> None:
        """
        Require ``symbol`` to be inside this closed profile.

        Raises:
            UnsupportedInspectFeatureError: If the symbol is not implemented.
        """
        if symbol not in self.supported_symbols:
            raise UnsupportedInspectFeatureError(symbol=symbol, source_profile=self.profile_id)

    def capability_report(self) -> dict[str, bool]:
        """Return deterministic capability flags."""
        return dict(self.capabilities)


_CORE_SYMBOLS = frozenset(
    {
        "inspect_ai",
        "inspect_ai.Task",
        "inspect_ai.Epochs",
        "inspect_ai.task",
        "inspect_ai.agent",
        "inspect_ai.agent.Agent",
        "inspect_ai.agent.AgentPrompt",
        "inspect_ai.agent.AgentSubmit",
        "inspect_ai.agent.as_solver",
        "inspect_ai.agent.react",
        "inspect_ai._util",
        "inspect_ai._util.registry",
        "inspect_ai._util.registry.registry_find",
        "inspect_ai._util.registry.registry_info",
        "inspect_ai.dataset",
        "inspect_ai.dataset.Dataset",
        "inspect_ai.dataset.FieldSpec",
        "inspect_ai.dataset.MemoryDataset",
        "inspect_ai.dataset.Sample",
        "inspect_ai.dataset.csv_dataset",
        "inspect_ai.dataset.hf_dataset",
        "inspect_ai.dataset.json_dataset",
        "inspect_ai.model",
        "inspect_ai.model.ChatMessage",
        "inspect_ai.model.ChatMessageAssistant",
        "inspect_ai.model.ChatMessageSystem",
        "inspect_ai.model.ChatMessageTool",
        "inspect_ai.model.ChatMessageUser",
        "inspect_ai.model.ContentImage",
        "inspect_ai.model.ContentText",
        "inspect_ai.model.GenerateConfig",
        "inspect_ai.model.Model",
        "inspect_ai.model.ModelName",
        "inspect_ai.model.get_model",
        "inspect_ai.sandbox",
        "inspect_ai.sandbox.SandboxSpec",
        "inspect_ai.scorer",
        "inspect_ai.scorer.CORRECT",
        "inspect_ai.scorer.INCORRECT",
        "inspect_ai.scorer.Score",
        "inspect_ai.scorer.Scorer",
        "inspect_ai.scorer.ScorerSpec",
        "inspect_ai.scorer.Target",
        "inspect_ai.scorer.accuracy",
        "inspect_ai.scorer.at_least",
        "inspect_ai.scorer.choice",
        "inspect_ai.scorer.grouped",
        "inspect_ai.scorer.includes",
        "inspect_ai.scorer.match",
        "inspect_ai.scorer.mean",
        "inspect_ai.scorer.mean_score",
        "inspect_ai.scorer.pass_at",
        "inspect_ai.scorer.pass_k",
        "inspect_ai.scorer.scorer",
        "inspect_ai.scorer.stderr",
        "inspect_ai.solver",
        "inspect_ai.solver.Generate",
        "inspect_ai.solver.Solver",
        "inspect_ai.solver.SolverSpec",
        "inspect_ai.solver.TaskState",
        "inspect_ai.solver.assistant_message",
        "inspect_ai.solver.chain",
        "inspect_ai.solver.chain_of_thought",
        "inspect_ai.solver.generate",
        "inspect_ai.solver.multiple_choice",
        "inspect_ai.solver.prompt_template",
        "inspect_ai.solver.solver",
        "inspect_ai.solver.system_message",
        "inspect_ai.solver.user_message",
        "inspect_ai.tool",
        "inspect_ai.tool.Tool",
        "inspect_ai.tool.ToolSpec",
        "inspect_ai.tool.bash",
        "inspect_ai.tool.python",
        "inspect_ai.tool.tool",
        "inspect_ai.util",
        "inspect_ai.util.SandboxEnvironmentSpec",
        "inspect_ai.util.message_limit",
        "inspect_ai.util.sandbox",
    }
)


PINNED_INSPECT_EVALS_PROFILE = InspectCompatibilityProfile(
    profile_id="inspect-evals-b935c0e-inspect-api-0.3.233",
    inspect_evals_revision="b935c0e5cfa04710f016f925db75d8e81413e2cf",
    inspect_api_profile="inspect_ai==0.3.233 construction API profile for the pinned inspect_evals revision",
    supported_symbols=_CORE_SYMBOLS,
    capabilities=(
        ("arc_multiple_choice_execution", True),
        ("choice_scoring", True),
        ("dataset_record_mapping", True),
        ("declarative_solver_composition", True),
        ("multimodal_message_mapping", True),
        ("typed_metric_and_reducer_specs", True),
        ("task_registration_and_parameters", True),
        ("task_registry_lookup", True),
        ("custom_scorer_execution", True),
        ("agent_or_react_loops", True),
        ("bash_or_python_tools", True),
        ("eval_log_or_store_hooks", False),
        ("inspect_model_providers", False),
        ("non_azure_cloud_providers", False),
    ),
)


def resolve_profile(profile_id: str) -> InspectCompatibilityProfile:
    """
    Resolve the one explicitly supported compatibility profile.

    Returns:
        InspectCompatibilityProfile: The pinned profile.

    Raises:
        InspectProfileMismatchError: If ``profile_id`` is not pinned and supported.
    """
    if profile_id != PINNED_INSPECT_EVALS_PROFILE.profile_id:
        raise InspectProfileMismatchError(
            f"Unknown Inspect compatibility profile '{profile_id}'. "
            f"Supported profile: '{PINNED_INSPECT_EVALS_PROFILE.profile_id}'."
        )
    return PINNED_INSPECT_EVALS_PROFILE
