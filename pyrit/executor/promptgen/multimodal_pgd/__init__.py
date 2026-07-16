# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Public API for the Multimodal PGD adversarial-image generator.

The primary entry point is ``MultiModalPGD`` (alias for ``MultiModalPGDGenerator``),
a ``pyrit.executor.promptgen.core.PromptGeneratorStrategy`` that perturbs a seed
image so a white-box VLM begins its reply with a target affirmative string. Output
is a JSONL manifest (see ``manifest``) plus cached PNGs, consumed downstream by a
``VisualPromptInjection`` scenario.

Example::

    from pyrit.executor.promptgen.multimodal_pgd import (
        MultiModalPGD,
        MultiModalPGDModelConfig,
    )

    generator = MultiModalPGD(
        model=MultiModalPGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf"),
    )
    result = await generator.execute_async(
        seed_image_path="cat.png",
        behavior="Describe how to ...",
        target_text="Sure, here is how to ...",
    )

This subpackage is **experimental**: APIs may change in any release without a
deprecation cycle. Pin pyrit to a specific version if you depend on it. To silence
the warning emitted on import::

    import warnings
    from pyrit.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
"""

import warnings
from typing import TYPE_CHECKING, Any

from pyrit.exceptions import ExperimentalWarning

# Torch-free symbols are imported eagerly so config / manifest / data helpers work
# on installs without the `multimodal_pgd` extra (no torch).
from pyrit.executor.promptgen.multimodal_pgd.config import (
    MultiModalPGDAlgorithmConfig,
    MultiModalPGDConfig,
    MultiModalPGDDataConfig,
    MultiModalPGDModelConfig,
    MultiModalPGDOutputConfig,
    MultiModalPGDVariantConfig,
    PGDVariant,
)
from pyrit.executor.promptgen.multimodal_pgd.data import BehaviorRow, load_behaviors
from pyrit.executor.promptgen.multimodal_pgd.manifest import (
    SCHEMA_VERSION,
    PGDManifestEntry,
    append_manifest_entry,
    read_manifest,
    write_manifest,
)
from pyrit.executor.promptgen.multimodal_pgd.targets import (
    augment_target,
    default_affirmative_target,
)

warnings.warn(
    "pyrit.executor.promptgen.multimodal_pgd is experimental: APIs may change in any "
    "release without a deprecation cycle. Pin pyrit to a specific version if you "
    "depend on this module. To silence: "
    "warnings.filterwarnings('ignore', category=pyrit.exceptions.ExperimentalWarning).",
    ExperimentalWarning,
    stacklevel=2,
)

# Torch-dependent symbols are exposed lazily via PEP 562 __getattr__ so that
# `from pyrit.executor.promptgen.multimodal_pgd import MultiModalPGDConfig` works on
# installs that only have the base dependencies (no torch). Touching any of these
# names triggers the underlying module import on first access.
_LAZY_IMPORTS = {
    "MultiModalPGD": ("pyrit.executor.promptgen.multimodal_pgd.generator", "MultiModalPGDGenerator"),
    "MultiModalPGDGenerator": ("pyrit.executor.promptgen.multimodal_pgd.generator", "MultiModalPGDGenerator"),
    "MultiModalPGDContext": ("pyrit.executor.promptgen.multimodal_pgd.generator", "MultiModalPGDContext"),
    "MultiModalPGDResult": ("pyrit.executor.promptgen.multimodal_pgd.generator", "MultiModalPGDResult"),
}

if TYPE_CHECKING:
    from pyrit.executor.promptgen.multimodal_pgd.generator import (
        MultiModalPGDContext,
        MultiModalPGDGenerator,
        MultiModalPGDResult,
    )

    MultiModalPGD = MultiModalPGDGenerator


def __getattr__(name: str) -> Any:
    if name in _LAZY_IMPORTS:
        import importlib

        module_name, attr = _LAZY_IMPORTS[name]
        value = getattr(importlib.import_module(module_name), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals().keys()) + list(_LAZY_IMPORTS.keys())))


__all__ = [
    "SCHEMA_VERSION",
    "BehaviorRow",
    "MultiModalPGD",
    "MultiModalPGDAlgorithmConfig",
    "MultiModalPGDConfig",
    "MultiModalPGDContext",
    "MultiModalPGDDataConfig",
    "MultiModalPGDGenerator",
    "MultiModalPGDModelConfig",
    "MultiModalPGDOutputConfig",
    "MultiModalPGDResult",
    "MultiModalPGDVariantConfig",
    "PGDManifestEntry",
    "PGDVariant",
    "append_manifest_entry",
    "augment_target",
    "default_affirmative_target",
    "load_behaviors",
    "read_manifest",
    "write_manifest",
]
