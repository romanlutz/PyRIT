# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
MultiModalPGDGenerator — a typed PromptGeneratorStrategy that crafts adversarial
images via Projected Gradient Descent against a white-box VLM.

Mirrors the ``GCGGenerator`` lifecycle:

- Strategy configuration (model / algorithm / variant / output) goes in ``__init__``.
- Per-execution data (one behavior + target + seed image) flows through
  ``execute_async``.
- ``_setup_async`` builds the white-box target (or adopts a caller-supplied one),
  ``_perform_async`` runs the optimization loop and writes the manifest row + PNG,
  ``_teardown_async`` releases resources for generator-owned targets only.

The generator accepts EITHER a pre-built ``target`` (which implements the
``WhiteBoxTarget`` Protocol) OR a ``model`` config used to construct a
``HuggingFaceVisionTarget`` at setup time. Exactly one must be provided.
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, overload

import torch
from pydantic import Field

from pyrit.common.utils import combine_dict
from pyrit.datasets.seed_datasets.remote._image_cache import fetch_and_cache_image_async
from pyrit.executor.promptgen.core.prompt_generator_strategy import (
    PromptGeneratorStrategy,
    PromptGeneratorStrategyContext,
    PromptGeneratorStrategyResult,
)
from pyrit.executor.promptgen.multimodal_pgd.attack.pgd_core import run_pgd
from pyrit.executor.promptgen.multimodal_pgd.attack.variants import make_variant_strategy
from pyrit.executor.promptgen.multimodal_pgd.config import (
    MultiModalPGDAlgorithmConfig,
    MultiModalPGDModelConfig,
    MultiModalPGDOutputConfig,
    MultiModalPGDVariantConfig,
)
from pyrit.executor.promptgen.multimodal_pgd.manifest import PGDManifestEntry, append_manifest_entry
from pyrit.executor.promptgen.multimodal_pgd.targets import response_matches_target
from pyrit.models import ComponentIdentifier, Identifiable
from pyrit.prompt_target.common.white_box_target import SupportsResponseGeneration

if TYPE_CHECKING:
    import PIL.Image

    from pyrit.prompt_target.common.white_box_target import WhiteBoxTarget

logger = logging.getLogger(__name__)


@dataclass
class MultiModalPGDContext(PromptGeneratorStrategyContext):
    """
    Per-execution state for one MultiModalPGDGenerator run.

    Attributes:
        behavior (str): The carrier behavior paired with the image.
        target_text (str): The affirmative target string to optimize for.
        seed_image_path (str): Path to the seed image. Optional for the blank-image
            variant (a neutral gray image is synthesized when absent).
        behavior_id (str): Stable identifier for the behavior, used in file names.
        memory_labels (dict[str, str]): Optional labels echoed onto the result.
        built_target (WhiteBoxTarget | None): The resolved white-box target for this
            run (either caller-supplied or generator-constructed).
        manifest_path (str | None): Resolved manifest file path for this run.
    """

    behavior: str = ""
    target_text: str = ""
    seed_image_path: str = ""
    behavior_id: str = ""
    memory_labels: dict[str, str] = field(default_factory=dict)

    built_target: Any | None = None
    manifest_path: str | None = None


class MultiModalPGDResult(PromptGeneratorStrategyResult):
    """
    Result of one MultiModalPGDGenerator run.

    Attributes:
        image_path (str): Path to the cached perturbed PNG.
        final_loss (float): Loss at the final optimization step.
        step_count (int): Number of steps actually executed.
        loss_history (list[float]): Per-step loss values.
        succeeded (bool): Whether the run reached ``final_loss <= stop_loss``.
        vlm_id (str): HuggingFace id of the VLM the image was crafted against.
        variant (str): PGD variant value used.
        deployed_loss (float | None): Loss recomputed on the reloaded 8-bit PNG, or
            ``None`` when the recomputation was skipped or failed.
        model_response (str | None): The VLM's reply to the crafted image, or ``None``
            when verification was skipped.
        target_emitted (bool | None): Whether ``model_response`` begins with the target
            string, or ``None`` when verification was skipped.
        manifest_entry (PGDManifestEntry | None): The manifest row written for the run.
        manifest_path (str | None): Path of the JSONL manifest the row was appended to.
        memory_labels (dict[str, str]): Echo of the labels passed via the context.
    """

    image_path: str = ""
    final_loss: float = float("nan")
    step_count: int = 0
    loss_history: list[float] = Field(default_factory=list)
    succeeded: bool = False
    vlm_id: str = ""
    variant: str = ""
    deployed_loss: float | None = None
    model_response: str | None = None
    target_emitted: bool | None = None
    manifest_entry: PGDManifestEntry | None = None
    manifest_path: str | None = None
    memory_labels: dict[str, str] = Field(default_factory=dict)


class MultiModalPGDGenerator(
    PromptGeneratorStrategy[MultiModalPGDContext, MultiModalPGDResult],
    Identifiable,
):
    """
    Projected Gradient Descent adversarial-image generator.

    Perturbs a seed image so a white-box VLM begins its reply to ``behavior`` with
    ``target_text``. Ports HarmBench's ``MultiModalPGD`` / ``MultiModalPGDBlankImage``
    / ``MultiModalPGDPatch`` baselines behind a single ``variant`` config slot.
    """

    def __init__(
        self,
        *,
        target: WhiteBoxTarget | None = None,
        model: MultiModalPGDModelConfig | None = None,
        algorithm: MultiModalPGDAlgorithmConfig | None = None,
        variant: MultiModalPGDVariantConfig | None = None,
        output: MultiModalPGDOutputConfig | None = None,
        hf_token: str | None = None,
    ) -> None:
        """
        Initialize the Multimodal PGD generator.

        Args:
            target (WhiteBoxTarget | None): A pre-built white-box target. Mutually
                exclusive with ``model``; when supplied the caller owns its lifecycle
                (it is NOT released on teardown).
            model (MultiModalPGDModelConfig | None): Config used to construct a
                ``HuggingFaceVisionTarget`` at setup time. Mutually exclusive with
                ``target``; a generator-owned target IS released on teardown.
            algorithm (MultiModalPGDAlgorithmConfig | None): Optimization
                hyper-parameters. ``None`` uses dataclass defaults.
            variant (MultiModalPGDVariantConfig | None): Variant selection. ``None``
                uses ``EPS_BOUNDED``.
            output (MultiModalPGDOutputConfig | None): Log / manifest locations.
                ``None`` uses dataclass defaults.
            hf_token (str | None): HuggingFace token for gated models.

        Raises:
            ValueError: If neither or both of ``target`` and ``model`` are provided.
        """
        super().__init__(logger=logger, context_type=MultiModalPGDContext)
        if (target is None) == (model is None):
            raise ValueError("MultiModalPGDGenerator: provide exactly one of 'target' or 'model'.")
        self._provided_target = target
        self._model = model
        self._algorithm = algorithm or MultiModalPGDAlgorithmConfig()
        self._variant_config = variant or MultiModalPGDVariantConfig()
        self._output = output or MultiModalPGDOutputConfig()
        self._hf_token = hf_token
        self._owns_target = model is not None

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build a behavioral identifier exposing model identity + key hyper-params.

        Returns:
            ComponentIdentifier: Identifier capturing the model id, variant, and
            optimization hyper-parameters.
        """
        return ComponentIdentifier.of(
            self,
            params={
                "vlm_id": self._resolve_vlm_id(),
                "variant": self._variant_config.kind.value,
                "patch_fraction": self._variant_config.patch_fraction,
                "num_steps": self._algorithm.num_steps,
                "step_size": self._algorithm.step_size,
                "epsilon": self._algorithm.epsilon,
                "stop_loss": self._algorithm.stop_loss,
            },
        )

    def _validate_context(self, *, context: MultiModalPGDContext) -> None:
        from pyrit.executor.promptgen.multimodal_pgd.config import PGDVariant

        if not context.behavior:
            raise ValueError("MultiModalPGDContext.behavior must be non-empty.")
        if not context.target_text:
            raise ValueError("MultiModalPGDContext.target_text must be non-empty.")
        if self._variant_config.kind is not PGDVariant.BLANK_IMAGE and not context.seed_image_path:
            raise ValueError(
                f"MultiModalPGDContext.seed_image_path is required for the {self._variant_config.kind.value} variant."
            )

    async def _setup_async(self, *, context: MultiModalPGDContext) -> None:
        """Resolve the white-box target and manifest path for this run."""
        context.memory_labels = combine_dict({}, context.memory_labels)
        if self._provided_target is not None:
            context.built_target = self._provided_target
        else:
            context.built_target = await asyncio.to_thread(self._build_target)
        context.manifest_path = self._resolve_manifest_path()

    async def _perform_async(self, *, context: MultiModalPGDContext) -> MultiModalPGDResult:
        """
        Run the PGD loop, cache the PNG, and append the manifest row.

        Returns:
            MultiModalPGDResult: The optimization outcome plus the appended manifest row.
        """
        target = context.built_target
        assert target is not None  # populated by _setup_async
        manifest_path = context.manifest_path
        assert manifest_path is not None  # populated by _setup_async
        seed_image = await asyncio.to_thread(self._load_seed_image, context.seed_image_path)
        variant_strategy = make_variant_strategy(config=self._variant_config)
        rng = torch.Generator(device=torch.device(target.device)).manual_seed(self._algorithm.random_seed)

        core = await asyncio.to_thread(
            run_pgd,
            target=target,
            variant=variant_strategy,
            behavior=context.behavior,
            target_text=context.target_text,
            seed_image=seed_image,
            num_steps=self._algorithm.num_steps,
            step_size=self._algorithm.step_size,
            epsilon=self._algorithm.epsilon,
            stop_loss=self._algorithm.stop_loss,
            rng=rng,
            verbose=self._output.verbose,
        )

        vlm_id = self._resolve_vlm_id() or getattr(target, "vlm_id", "")
        behavior_id = context.behavior_id or "behavior"
        variant_value = self._variant_config.kind.value
        filename = f"pgd_{self._slugify(vlm_id)}_{behavior_id}_{variant_value}.png"

        png_bytes = await asyncio.to_thread(self._encode_png, core.image)
        image_path = await fetch_and_cache_image_async(
            filename=filename,
            image_bytes=png_bytes,
            log_prefix="MultiModalPGD",
        )

        model_response, target_emitted = await self._verify_async(
            target=target, image=core.image, behavior=context.behavior, target_text=context.target_text
        )

        deployed_loss = await self._deployed_loss_async(
            target=target, image=core.image, behavior=context.behavior, target_text=context.target_text
        )

        entry = PGDManifestEntry(
            id=filename[:-4],
            behavior_id=behavior_id,
            behavior_text=context.behavior,
            target_text=context.target_text,
            image_path=image_path,
            vlm_id=vlm_id,
            variant=variant_value,
            num_steps_run=core.step_count,
            final_loss=core.final_loss,
            epsilon=self._algorithm.epsilon,
            step_size=self._algorithm.step_size,
            stop_loss=self._algorithm.stop_loss,
            succeeded_stop_criterion=core.succeeded,
            deployed_loss=deployed_loss,
            seed_image_path=context.seed_image_path,
            model_response=model_response,
            target_emitted=target_emitted,
        )
        await asyncio.to_thread(append_manifest_entry, entry=entry, path=manifest_path)

        return MultiModalPGDResult(
            image_path=image_path,
            final_loss=core.final_loss,
            step_count=core.step_count,
            loss_history=core.loss_history,
            succeeded=core.succeeded,
            vlm_id=vlm_id,
            variant=variant_value,
            deployed_loss=deployed_loss,
            model_response=model_response,
            target_emitted=target_emitted,
            manifest_entry=entry,
            manifest_path=manifest_path,
            memory_labels=dict(context.memory_labels),
        )

    async def _verify_async(
        self, *, target: Any, image: PIL.Image.Image, behavior: str, target_text: str
    ) -> tuple[str | None, bool | None]:
        """
        Feed the crafted image back through the target and check for the target string.

        Returns ``(None, None)`` when verification is disabled or the target cannot
        generate responses, so the optimization outcome is never blocked on it.

        Returns:
            tuple[str | None, bool | None]: The model's reply and whether it begins with
            ``target_text``.
        """
        if not self._output.verify_response or not isinstance(target, SupportsResponseGeneration):
            return None, None
        try:
            model_response = await asyncio.to_thread(target.generate_response, behavior=behavior, image=image)
        except Exception as e:  # verification is best-effort; never fail the run over it
            self._logger.warning(f"Response verification failed: {e}")
            return None, None
        return model_response, response_matches_target(response=model_response, target_text=target_text)

    async def _deployed_loss_async(
        self, *, target: Any, image: PIL.Image.Image, behavior: str, target_text: str
    ) -> float | None:
        """
        Recompute the loss on the reloaded 8-bit PNG as an honesty check on ``final_loss``.

        The optimizer already scores a straight-through quantized tensor, so this should
        track ``final_loss`` closely; a large gap flags a remaining optimize-vs-deploy
        mismatch. Best-effort: returns ``None`` on any failure so the run is never blocked.

        Returns:
            float | None: The loss measured on the re-preprocessed image, or ``None``.
        """
        try:
            return await asyncio.to_thread(
                self._deployed_loss, target=target, image=image, behavior=behavior, target_text=target_text
            )
        except Exception as e:  # honesty metric is best-effort; never fail the run over it
            self._logger.warning(f"Deployed-loss recomputation failed: {e}")
            return None

    @staticmethod
    def _deployed_loss(*, target: Any, image: PIL.Image.Image, behavior: str, target_text: str) -> float:
        """
        Re-preprocess ``image`` and compute the loss on the exact pixels that ship.

        Returns:
            float: The loss measured on the reloaded 8-bit image.
        """
        import torch

        inputs = target.preprocess(behavior=behavior, image=image)
        with torch.no_grad():
            loss = target.compute_loss(inputs=inputs, target_text=target_text)
        return float(loss.detach().item())

    async def _teardown_async(self, *, context: MultiModalPGDContext) -> None:
        """Release the target only when the generator constructed (owns) it."""
        target = context.built_target
        if target is not None and self._owns_target:
            try:
                await asyncio.to_thread(target.release_white_box_resources)
            except Exception as e:
                self._logger.warning(f"Failed to release white-box target resources: {e}")
        context.built_target = None

    @overload
    async def execute_async(
        self,
        *,
        behavior: str,
        target_text: str,
        seed_image_path: str | None = None,
        behavior_id: str | None = None,
        memory_labels: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> MultiModalPGDResult: ...

    @overload
    async def execute_async(self, **kwargs: Any) -> MultiModalPGDResult: ...

    async def execute_async(self, **kwargs: Any) -> MultiModalPGDResult:
        """
        Craft an adversarial image for one behavior.

        Args:
            behavior (str): The carrier behavior. Required.
            target_text (str): The affirmative target string. Required.
            seed_image_path (str | None): Path to the seed image. Required for all
                variants except blank-image.
            behavior_id (str | None): Stable behavior identifier used in file names.
            memory_labels (dict[str, str] | None): Optional labels echoed on the result.
            **kwargs: Forwarded to the base ``Strategy.execute_async``.

        Returns:
            MultiModalPGDResult: The optimization result plus manifest row.
        """
        kwargs.setdefault("seed_image_path", "")
        kwargs.setdefault("behavior_id", "")
        kwargs.setdefault("memory_labels", {})
        return await super().execute_async(**kwargs)

    def _build_target(self) -> WhiteBoxTarget:
        """
        Construct a ``HuggingFaceVisionTarget`` from the model config.

        Returns:
            WhiteBoxTarget: A freshly loaded vision target owned by this generator.
        """
        from pyrit.prompt_target.hugging_face.hugging_face_vision_target import HuggingFaceVisionTarget

        assert self._model is not None  # guaranteed by __init__ when _owns_target
        return HuggingFaceVisionTarget(
            model_id=self._model.vlm_id,
            device=self._model.device,
            dtype=self._model.dtype,
            hf_token=self._hf_token,
        )

    def _resolve_vlm_id(self) -> str:
        if self._model is not None:
            return self._model.vlm_id
        return getattr(self._provided_target, "vlm_id", "")

    def _resolve_manifest_path(self) -> str:
        if self._output.manifest_path:
            return self._output.manifest_path
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        return f"{self._output.result_prefix}_manifest_{timestamp}.jsonl"

    @staticmethod
    def _load_seed_image(seed_image_path: str) -> PIL.Image.Image:
        import PIL.Image

        if not seed_image_path:
            return PIL.Image.new("RGB", (336, 336), (128, 128, 128))
        return PIL.Image.open(seed_image_path).convert("RGB")

    @staticmethod
    def _encode_png(image: PIL.Image.Image) -> bytes:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    @staticmethod
    def _slugify(vlm_id: str) -> str:
        return vlm_id.replace("/", "_").replace(".", "_").replace("-", "_") or "vlm"


__all__ = ["MultiModalPGDContext", "MultiModalPGDGenerator", "MultiModalPGDResult"]
