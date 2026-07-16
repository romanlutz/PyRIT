# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Typed configuration objects for the Multimodal PGD image generator.

Mirrors the ``pyrit.executor.promptgen.gcg.config`` pattern: plain dataclasses with
``__post_init__`` validation plus JSON round-tripping for AzureML transport. These
objects are intentionally free of ``torch`` / ``transformers`` imports so that
``from pyrit.executor.promptgen.multimodal_pgd import MultiModalPGDConfig`` works on
installs without the ``multimodal_pgd`` extra.

A minimal call is::

    generator = MultiModalPGDGenerator(
        model=MultiModalPGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf"),
    )
    await generator.execute_async(
        seed_image_path="cat.png",
        behavior="Describe how to ...",
        target_text="Sure, here is how to ...",
    )
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


class PGDVariant(Enum):
    """
    The three HarmBench white-box image-perturbation variants.

    Attributes:
        EPS_BOUNDED: Perturb a real seed photo within an epsilon ball around it.
        BLANK_IMAGE: Start from random noise with no epsilon bound.
        PATCH: Restrict perturbation to a random rectangular patch of the image.
    """

    EPS_BOUNDED = "eps_bounded"
    BLANK_IMAGE = "blank_image"
    PATCH = "patch"


def _json_default(value: Any) -> Any:
    """
    JSON encoder hook that renders ``Enum`` members as their ``.value``.

    Returns:
        Any: The ``.value`` of an ``Enum`` member.

    Raises:
        TypeError: If ``value`` is not a JSON-serializable type.
    """
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class MultiModalPGDModelConfig:
    """
    Identity and loading options for the vision model PGD optimizes against.

    Attributes:
        vlm_id (str): HuggingFace model identifier such as
            ``"llava-hf/llava-1.5-7b-hf"``. Loads both the processor and weights.
        device (str): Torch device string. Defaults to ``"cuda:0"``.
        dtype (str): Torch dtype name used to load the model. Defaults to
            ``"float16"``.
    """

    vlm_id: str
    device: str = "cuda:0"
    dtype: str = "float16"

    def __post_init__(self) -> None:
        """
        Validate that ``vlm_id`` is set.

        Raises:
            ValueError: If ``vlm_id`` is empty.
        """
        if not self.vlm_id:
            raise ValueError("MultiModalPGDModelConfig.vlm_id must be a non-empty HuggingFace model identifier.")


@dataclass
class MultiModalPGDAlgorithmConfig:
    """
    Hyper-parameters of the PGD optimization loop.

    Attributes:
        num_steps (int): Number of optimization steps. Defaults to 500.
        step_size (float): Per-step gradient-sign step size in normalized model
            space. Defaults to ``2 / 255``.
        epsilon (float): Maximum per-pixel perturbation (normalized model space)
            for the ``EPS_BOUNDED`` variant. Defaults to ``16 / 255``.
        stop_loss (float): Early-stop threshold; the loop halts once the loss is at
            or below this value. Defaults to ``0.05``.
        random_seed (int): Seed for reproducible random initialization / patch
            placement. Defaults to 42.
    """

    num_steps: int = 500
    step_size: float = 2 / 255
    epsilon: float = 16 / 255
    stop_loss: float = 0.05
    random_seed: int = 42

    def __post_init__(self) -> None:
        """
        Validate the optimization hyper-parameters.

        Raises:
            ValueError: If ``num_steps``, ``step_size``, or ``epsilon`` is not positive.
        """
        if self.num_steps <= 0:
            raise ValueError(f"MultiModalPGDAlgorithmConfig.num_steps must be > 0, got {self.num_steps}.")
        if self.step_size <= 0:
            raise ValueError(f"MultiModalPGDAlgorithmConfig.step_size must be > 0, got {self.step_size}.")
        if self.epsilon <= 0:
            raise ValueError(f"MultiModalPGDAlgorithmConfig.epsilon must be > 0, got {self.epsilon}.")


@dataclass
class MultiModalPGDVariantConfig:
    """
    Selects which PGD variant runs and its variant-specific parameters.

    Attributes:
        kind (PGDVariant): Which variant to run. Defaults to
            ``PGDVariant.EPS_BOUNDED``.
        patch_fraction (float): Side length of the square patch as a fraction of
            ``min(height, width)`` for the ``PATCH`` variant. Must be in ``(0, 1]``.
            Defaults to ``0.2``.
    """

    kind: PGDVariant = PGDVariant.EPS_BOUNDED
    patch_fraction: float = 0.2

    def __post_init__(self) -> None:
        """
        Coerce ``kind`` from a string and validate ``patch_fraction``.

        Raises:
            ValueError: If ``patch_fraction`` is outside ``(0, 1]``.
        """
        if isinstance(self.kind, str):
            self.kind = PGDVariant(self.kind)
        if not 0 < self.patch_fraction <= 1:
            raise ValueError(f"MultiModalPGDVariantConfig.patch_fraction must be in (0, 1], got {self.patch_fraction}.")


@dataclass
class MultiModalPGDOutputConfig:
    """
    Where the run writes its manifest / log artefacts.

    Attributes:
        result_prefix (str): Prefix for the per-run JSON log and default manifest
            path. The log filename is ``{result_prefix}_{YYYYMMDD-HHMMSS}.json``.
        manifest_path (str): Explicit manifest file path. When empty a default of
            ``{result_prefix}_manifest_{YYYYMMDD-HHMMSS}.jsonl`` is used.
        verbose (bool): Verbose progress logging during the run. Defaults to True.
    """

    result_prefix: str = ""
    manifest_path: str = ""
    verbose: bool = True


@dataclass
class MultiModalPGDDataConfig:
    """
    CSV dataset configuration for a batch of behaviors.

    Used as a typed bundle for AML transport (a job ships its data config as a
    separate JSON file alongside the strategy ``MultiModalPGDConfig``).

    Attributes:
        behaviors_csv (str): Path or URL to a CSV with ``behavior``,
            ``target``, and ``seed_image_path`` columns (plus an optional
            ``behavior_id``).
        n_behaviors (int): Maximum number of rows to use. ``0`` means all rows.
    """

    behaviors_csv: str = ""
    n_behaviors: int = 0

    def __post_init__(self) -> None:
        """
        Validate ``n_behaviors``.

        Raises:
            ValueError: If ``n_behaviors`` is negative.
        """
        if self.n_behaviors < 0:
            raise ValueError(f"MultiModalPGDDataConfig.n_behaviors must be >= 0, got {self.n_behaviors}.")

    def to_json(self) -> str:
        """
        Serialize this config to a JSON string.

        Returns:
            str: The indented JSON representation.
        """
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, payload: str) -> MultiModalPGDDataConfig:
        """
        Deserialize a config previously produced by ``to_json``.

        Returns:
            MultiModalPGDDataConfig: The reconstructed config.

        Raises:
            ValueError: If ``payload`` is not valid JSON.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"MultiModalPGDDataConfig.from_json: payload is not valid JSON: {e}") from e
        return cls(**data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> MultiModalPGDDataConfig:
        """
        Load a config from a JSON file.

        Returns:
            MultiModalPGDDataConfig: The reconstructed config.
        """
        with open(path) as f:
            return cls.from_json(f.read())

    def to_json_file(self, path: str | Path) -> None:
        """Write this config to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


@dataclass
class MultiModalPGDConfig:
    """
    Top-level strategy configuration for one Multimodal PGD run.

    Bundles everything ``MultiModalPGDGenerator``'s constructor needs for the
    serialization / AML path. Per-execution data (behaviors, targets, seed images)
    flows through ``execute_async`` or a separate ``MultiModalPGDDataConfig``.

    Attributes:
        model (MultiModalPGDModelConfig): The vision model to optimize against.
        algorithm (MultiModalPGDAlgorithmConfig): Optimization hyper-parameters.
        variant (MultiModalPGDVariantConfig): Variant selection.
        output (MultiModalPGDOutputConfig): Log / manifest file locations.
        hf_token (str | None): HuggingFace token for gated models. ``None`` falls
            back to the ``HUGGINGFACE_TOKEN`` environment variable.
    """

    model: MultiModalPGDModelConfig
    algorithm: MultiModalPGDAlgorithmConfig = field(default_factory=MultiModalPGDAlgorithmConfig)
    variant: MultiModalPGDVariantConfig = field(default_factory=MultiModalPGDVariantConfig)
    output: MultiModalPGDOutputConfig = field(default_factory=MultiModalPGDOutputConfig)
    hf_token: str | None = None

    def to_json(self) -> str:
        """
        Serialize this config to a JSON string (enum members render as values).

        Returns:
            str: The indented JSON representation.
        """
        return json.dumps(asdict(self), indent=2, default=_json_default)

    @classmethod
    def from_json(cls, payload: str) -> MultiModalPGDConfig:
        """
        Deserialize a config previously produced by ``to_json``.

        Returns:
            MultiModalPGDConfig: The reconstructed config.

        Raises:
            ValueError: If ``payload`` is not valid JSON or is missing ``model``.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as e:
            raise ValueError(f"MultiModalPGDConfig.from_json: payload is not valid JSON: {e}") from e
        return cls._from_dict(data)

    @classmethod
    def from_json_file(cls, path: str | Path) -> MultiModalPGDConfig:
        """
        Load a config from a JSON file produced by ``to_json_file``.

        Returns:
            MultiModalPGDConfig: The reconstructed config.
        """
        with open(path) as f:
            return cls.from_json(f.read())

    def to_json_file(self, path: str | Path) -> None:
        """Write this config to a JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> MultiModalPGDConfig:
        if "model" not in data or not isinstance(data["model"], dict):
            raise ValueError("MultiModalPGDConfig payload must contain a 'model' object.")
        return cls(
            model=MultiModalPGDModelConfig(**data["model"]),
            algorithm=MultiModalPGDAlgorithmConfig(**data.get("algorithm", {})),
            variant=MultiModalPGDVariantConfig(**data.get("variant", {})),
            output=MultiModalPGDOutputConfig(**data.get("output", {})),
            hf_token=data.get("hf_token"),
        )


__all__ = [
    "MultiModalPGDAlgorithmConfig",
    "MultiModalPGDConfig",
    "MultiModalPGDDataConfig",
    "MultiModalPGDModelConfig",
    "MultiModalPGDOutputConfig",
    "MultiModalPGDVariantConfig",
    "PGDVariant",
]
