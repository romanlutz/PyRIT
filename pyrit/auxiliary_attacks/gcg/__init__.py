# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Public API for the Greedy Coordinate Gradient (GCG) auxiliary attack.

The primary entry point is :class:`GCG` (alias for :class:`GCGGenerator`), a
:class:`pyrit.executor.promptgen.core.PromptGeneratorStrategy` that produces
adversarial suffixes via the GCG algorithm.

Example:

    from pyrit.auxiliary_attacks.gcg import (
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
"""

from pyrit.auxiliary_attacks.gcg.config import (
    GCGAlgorithmConfig,
    GCGConfig,
    GCGDataConfig,
    GCGModelConfig,
    GCGOutputConfig,
    GCGStrategyConfig,
)
from pyrit.auxiliary_attacks.gcg.data import load_goals_and_targets
from pyrit.auxiliary_attacks.gcg.generator import GCGContext, GCGGenerator, GCGResult

GCG = GCGGenerator

__all__ = [
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
    "load_goals_and_targets",
]
