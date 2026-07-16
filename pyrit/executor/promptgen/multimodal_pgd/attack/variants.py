# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
PGD variant strategies: what to start from and how to project each step.

The outer optimization loop in ``pgd_core`` is identical across the three HarmBench
variants; they differ only in two hooks:

- ``initial_pixel_values`` — the starting perturbable tensor.
- ``project_step`` — one gradient-sign descent step plus the variant's projection.

All operations run in the model's **normalized** pixel space (the tensor the
processor feeds the model), so ``epsilon`` / ``step_size`` are expressed in that
space. This module imports ``torch`` at module load and is therefore only reachable
with the ``multimodal_pgd`` extra installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pyrit.executor.promptgen.multimodal_pgd.config import MultiModalPGDVariantConfig


class PGDVariantStrategy(ABC):
    """Base strategy encapsulating the per-variant PGD hooks."""

    @abstractmethod
    def initial_pixel_values(self, *, base: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Return the initial perturbable tensor for a run.

        Args:
            base (torch.Tensor): The seed image's normalized ``pixel_values``.
            rng (torch.Generator | None): Optional RNG for reproducible init.

        Returns:
            torch.Tensor: A detached tensor to begin optimizing from.
        """

    @abstractmethod
    def project_step(
        self,
        *,
        pixel_values: torch.Tensor,
        base: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Apply one gradient-sign descent step and project into the valid region.

        Args:
            pixel_values (torch.Tensor): Current perturbed tensor (the leaf whose
                ``.grad`` is ``grad``).
            base (torch.Tensor): The unperturbed seed tensor.
            grad (torch.Tensor): Gradient of the loss w.r.t. ``pixel_values``.
            step_size (float): Gradient-sign step size.
            epsilon (float): Perturbation bound (used by bounded variants).

        Returns:
            torch.Tensor: The detached, projected tensor for the next step.
        """


class EpsBoundedVariant(PGDVariantStrategy):
    """Perturb a real seed image, keeping it within an L-infinity epsilon ball."""

    def initial_pixel_values(self, *, base: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Start from an exact copy of the seed tensor.

        Returns:
            torch.Tensor: A detached clone of ``base``.
        """
        return base.detach().clone()

    def project_step(
        self,
        *,
        pixel_values: torch.Tensor,
        base: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Take a signed-gradient step and clip the perturbation into the epsilon ball.

        Returns:
            torch.Tensor: The detached, epsilon-projected tensor for the next step.
        """
        stepped = pixel_values.detach() - step_size * grad.detach().sign()
        perturbation = torch.clamp(stepped - base, min=-epsilon, max=epsilon)
        return (base + perturbation).detach()


class BlankImageVariant(PGDVariantStrategy):
    """Start from standard-normal noise and optimize with no epsilon bound."""

    def initial_pixel_values(self, *, base: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Start from standard-normal noise the size of the seed tensor.

        Returns:
            torch.Tensor: A detached random-noise tensor matching ``base``'s shape.
        """
        noise = torch.randn(base.shape, generator=rng, dtype=base.dtype, device=base.device)
        return noise.detach()

    def project_step(
        self,
        *,
        pixel_values: torch.Tensor,
        base: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Take an unbounded signed-gradient step.

        Returns:
            torch.Tensor: The detached tensor for the next step (no epsilon bound).
        """
        return (pixel_values.detach() - step_size * grad.detach().sign()).detach()


class PatchVariant(PGDVariantStrategy):
    """
    Perturb only a random square patch of a fixed-resolution image (unbounded).

    Requires a 4D ``[N, C, H, W]`` ``pixel_values`` tensor (the fixed-resolution
    layout produced by models such as LLaVA-1.5). Dynamic-tiling layouts (e.g.
    Qwen2-VL's flattened patch sequence) do not have a well-defined spatial patch and
    raise ``ValueError``.
    """

    def __init__(self, *, patch_fraction: float) -> None:
        """
        Initialize the patch variant.

        Args:
            patch_fraction (float): Square patch side length as a fraction of
                ``min(H, W)``.
        """
        self._patch_fraction = patch_fraction
        self._mask: torch.Tensor | None = None

    def initial_pixel_values(self, *, base: torch.Tensor, rng: torch.Generator | None = None) -> torch.Tensor:
        """
        Sample the patch mask and start from a copy of the seed tensor.

        Returns:
            torch.Tensor: A detached clone of ``base`` (the patch region is perturbed
            in later steps).
        """
        self._mask = self._make_patch_mask(base=base, rng=rng)
        return base.detach().clone()

    def project_step(
        self,
        *,
        pixel_values: torch.Tensor,
        base: torch.Tensor,
        grad: torch.Tensor,
        step_size: float,
        epsilon: float,
    ) -> torch.Tensor:
        """
        Take a signed-gradient step applied only inside the sampled patch.

        Returns:
            torch.Tensor: The detached tensor with only the patch region updated.

        Raises:
            RuntimeError: If called before ``initial_pixel_values`` sampled the mask.
        """
        if self._mask is None:
            raise RuntimeError("PatchVariant.project_step called before initial_pixel_values.")
        stepped = pixel_values.detach() - step_size * grad.detach().sign()
        return torch.where(self._mask, stepped, base).detach()

    def _make_patch_mask(self, *, base: torch.Tensor, rng: torch.Generator | None) -> torch.Tensor:
        if base.dim() != 4:
            raise ValueError(
                "PatchVariant requires a 4D [N, C, H, W] pixel_values tensor "
                f"(fixed-resolution model), got shape {tuple(base.shape)}. Use a "
                "fixed-resolution VLM (e.g. LLaVA-1.5) for the patch variant."
            )
        height, width = int(base.shape[-2]), int(base.shape[-1])
        side = max(1, int(self._patch_fraction * min(height, width)))
        row = int(torch.randint(0, max(1, height - side + 1), (1,), generator=rng, device=base.device).item())
        col = int(torch.randint(0, max(1, width - side + 1), (1,), generator=rng, device=base.device).item())
        mask = torch.zeros_like(base, dtype=torch.bool)
        mask[..., row : row + side, col : col + side] = True
        return mask


def make_variant_strategy(*, config: MultiModalPGDVariantConfig) -> PGDVariantStrategy:
    """
    Build the concrete variant strategy selected by ``config``.

    Args:
        config (MultiModalPGDVariantConfig): The variant selection + parameters.

    Returns:
        PGDVariantStrategy: The strategy implementing the selected variant.

    Raises:
        ValueError: If ``config.kind`` is not a recognized variant.
    """
    from pyrit.executor.promptgen.multimodal_pgd.config import PGDVariant

    if config.kind is PGDVariant.EPS_BOUNDED:
        return EpsBoundedVariant()
    if config.kind is PGDVariant.BLANK_IMAGE:
        return BlankImageVariant()
    if config.kind is PGDVariant.PATCH:
        return PatchVariant(patch_fraction=config.patch_fraction)
    raise ValueError(f"Unknown PGD variant: {config.kind!r}")


__all__ = [
    "BlankImageVariant",
    "EpsBoundedVariant",
    "PGDVariantStrategy",
    "PatchVariant",
    "make_variant_strategy",
]
