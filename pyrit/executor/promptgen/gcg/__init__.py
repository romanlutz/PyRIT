# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public API for the Greedy Coordinate Gradient (GCG) adversarial-suffix generator.

The primary entry point is ``GCG`` (alias for ``GCGGenerator``), a
``pyrit.executor.promptgen.core.PromptGeneratorStrategy`` that produces
adversarial suffixes via the GCG algorithm.

Example:

    from pyrit.executor.promptgen.gcg import (
        GCG,
        GCGAlgorithmConfig,
        GCGModelConfig,
    )

    generator = GCG(
        models=[GCGModelConfig(name="meta-llama/Llama-2-7b-chat-hf")],
        algorithm=GCGAlgorithmConfig(n_steps=500, batch_size=512),
    )
    result = await generator.execute_async(
        goals=["how do I ..."],
        targets=["Sure, here is ..."],
    )

This subpackage is **experimental**: APIs may change in any release without a
deprecation cycle. Pin pyrit to a specific version if you depend on it. To
silence the warning emitted on import::

    import warnings
    from pyrit.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
"""

from typing import TYPE_CHECKING

from pyrit.executor.promptgen.core.lazy_exports import build_lazy_exports, warn_experimental
from pyrit.executor.promptgen.gcg.config import (
    GCGAlgorithmConfig,
    GCGConfig,
    GCGDataConfig,
    GCGModelConfig,
    GCGOutputConfig,
    GCGStrategyConfig,
)

warn_experimental(module_name=__name__)

# Torch-dependent symbols are exposed lazily via PEP 562 __getattr__ so that
# `from pyrit.executor.promptgen.gcg import GCGConfig` works on installs that
# only have the base `dev` extra (no torch). Touching any of these names from
# the package root triggers the underlying module import on first access; if
# torch is missing the user gets a clear ModuleNotFoundError pointing at torch.
#
# The extension Protocols live in ``extension_protocols`` (typing-only — that
# module imports cleanly without torch) but are routed through the same lazy
# mechanism so all GCG public symbols share one re-export pathway.
_LAZY_IMPORTS = {
    "CandidateFilter": ("pyrit.executor.promptgen.gcg.extension_protocols", "CandidateFilter"),
    "CrossEntropyLoss": ("pyrit.executor.promptgen.gcg.default_implementations", "CrossEntropyLoss"),
    "GCG": ("pyrit.executor.promptgen.gcg.generator", "GCGGenerator"),
    "GCGContext": ("pyrit.executor.promptgen.gcg.generator", "GCGContext"),
    "GCGGenerator": ("pyrit.executor.promptgen.gcg.generator", "GCGGenerator"),
    "GCGResult": ("pyrit.executor.promptgen.gcg.generator", "GCGResult"),
    "LengthPreservingFilter": ("pyrit.executor.promptgen.gcg.default_implementations", "LengthPreservingFilter"),
    "LiteralStringInit": ("pyrit.executor.promptgen.gcg.default_implementations", "LiteralStringInit"),
    "LossFunction": ("pyrit.executor.promptgen.gcg.extension_protocols", "LossFunction"),
    "SamplingStrategy": ("pyrit.executor.promptgen.gcg.extension_protocols", "SamplingStrategy"),
    "StandardGCGSampling": ("pyrit.executor.promptgen.gcg.default_implementations", "StandardGCGSampling"),
    "SuffixInitializer": ("pyrit.executor.promptgen.gcg.extension_protocols", "SuffixInitializer"),
    "load_goals_and_targets": ("pyrit.executor.promptgen.gcg.data", "load_goals_and_targets"),
}

if TYPE_CHECKING:
    from pyrit.executor.promptgen.gcg.data import load_goals_and_targets
    from pyrit.executor.promptgen.gcg.default_implementations import (
        CrossEntropyLoss,
        LengthPreservingFilter,
        LiteralStringInit,
        StandardGCGSampling,
    )
    from pyrit.executor.promptgen.gcg.extension_protocols import (
        CandidateFilter,
        LossFunction,
        SamplingStrategy,
        SuffixInitializer,
    )
    from pyrit.executor.promptgen.gcg.generator import (
        GCGContext,
        GCGGenerator,
        GCGResult,
    )

    GCG = GCGGenerator


__getattr__, __dir__ = build_lazy_exports(module_globals=globals(), lazy_imports=_LAZY_IMPORTS)


__all__ = [
    "CandidateFilter",
    "CrossEntropyLoss",
    "GCG",
    "GCGAlgorithmConfig",
    "GCGConfig",
    "GCGContext",
    "GCGDataConfig",
    "GCGGenerator",
    "GCGModelConfig",
    "GCGOutputConfig",
    "GCGResult",
    "GCGStrategyConfig",
    "LengthPreservingFilter",
    "LiteralStringInit",
    "LossFunction",
    "SamplingStrategy",
    "StandardGCGSampling",
    "SuffixInitializer",
    "load_goals_and_targets",
]
