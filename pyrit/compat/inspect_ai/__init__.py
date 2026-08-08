# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Pinned, opt-in Inspect source compatibility without an Inspect dependency.

No top-level ``inspect_ai`` distribution is shipped. A dedicated worker process
temporarily aliases a strict facade, executes trusted source, and exits without
exposing compatibility/source aliases to the caller.
"""

from pyrit.compat.inspect_ai.inventory import (
    InspectApiInventory,
    InspectApiUsage,
    inventory_inspect_api_usage,
)
from pyrit.compat.inspect_ai.loader import (
    InspectCompatibilityReport,
    InspectEvalRun,
    LoadedInspectEval,
    load_inspect_eval,
    run_inspect_eval_async,
)
from pyrit.compat.inspect_ai.profile import (
    PINNED_INSPECT_EVALS_PROFILE,
    InspectCompatibilityProfile,
    InspectProfileMismatchError,
    UnsupportedInspectFeatureError,
    resolve_profile,
)
from pyrit.compat.inspect_ai.types import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
    Dataset,
    GenerateConfig,
    MemoryDataset,
    Model,
    ModelName,
    Sample,
    SandboxSpec,
    Score,
    ScorerSpec,
    SolverSpec,
    Target,
    Task,
    ToolSpec,
)

__all__ = [
    "PINNED_INSPECT_EVALS_PROFILE",
    "ChatMessage",
    "ChatMessageAssistant",
    "ChatMessageSystem",
    "ChatMessageTool",
    "ChatMessageUser",
    "ContentImage",
    "ContentText",
    "Dataset",
    "GenerateConfig",
    "InspectApiInventory",
    "InspectApiUsage",
    "InspectCompatibilityProfile",
    "InspectCompatibilityReport",
    "InspectEvalRun",
    "InspectProfileMismatchError",
    "LoadedInspectEval",
    "MemoryDataset",
    "Model",
    "ModelName",
    "Sample",
    "SandboxSpec",
    "Score",
    "ScorerSpec",
    "SolverSpec",
    "Target",
    "Task",
    "ToolSpec",
    "UnsupportedInspectFeatureError",
    "inventory_inspect_api_usage",
    "load_inspect_eval",
    "resolve_profile",
    "run_inspect_eval_async",
]
