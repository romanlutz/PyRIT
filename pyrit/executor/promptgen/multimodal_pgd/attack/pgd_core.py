# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
The shared Multimodal PGD optimization loop.

``run_pgd`` drives the forward -> loss -> backward -> project cycle that is common to
all three variants, delegating the two points of variation (initial tensor and
per-step projection) to a ``PGDVariantStrategy``. It depends only on the
``WhiteBoxTarget`` Protocol, so tests can substitute a tiny CPU model.

This module imports ``torch`` at module load and is therefore only reachable with
the ``multimodal_pgd`` extra installed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    import PIL.Image

    from pyrit.executor.promptgen.multimodal_pgd.attack.variants import PGDVariantStrategy
    from pyrit.prompt_target.common.white_box_target import WhiteBoxTarget

logger = logging.getLogger(__name__)


@dataclass
class PGDCoreResult:
    """
    Outcome of one ``run_pgd`` call.

    Attributes:
        image (PIL.Image.Image): The final perturbed image.
        loss_history (list[float]): Per-step loss values.
        final_loss (float): Loss at the final executed step (``nan`` if no steps).
        step_count (int): Number of optimization steps actually executed.
        succeeded (bool): Whether the run reached ``final_loss <= stop_loss``.
    """

    image: PIL.Image.Image
    loss_history: list[float]
    final_loss: float
    step_count: int
    succeeded: bool


def run_pgd(
    *,
    target: WhiteBoxTarget,
    variant: PGDVariantStrategy,
    behavior: str,
    target_text: str,
    seed_image: PIL.Image.Image,
    num_steps: int,
    step_size: float,
    epsilon: float,
    stop_loss: float,
    rng: torch.Generator | None = None,
    verbose: bool = False,
    log_every: int = 25,
) -> PGDCoreResult:
    """
    Run the PGD optimization loop for a single behavior.

    Args:
        target (WhiteBoxTarget): The white-box VLM surface providing preprocessing,
            loss, and denormalization.
        variant (PGDVariantStrategy): The variant-specific init/projection hooks.
        behavior (str): The carrier behavior paired with the image.
        target_text (str): The affirmative target string to minimize loss against.
        seed_image (PIL.Image.Image): The seed image (also used to derive tensor
            shape for the blank-image variant).
        num_steps (int): Maximum number of optimization steps.
        step_size (float): Gradient-sign step size.
        epsilon (float): Perturbation bound for bounded variants.
        stop_loss (float): Early-stop threshold on the loss.
        rng (torch.Generator | None): Optional RNG for reproducible init.
        verbose (bool): Whether to log periodic progress.
        log_every (int): Step interval for progress logging when ``verbose``.

    Returns:
        PGDCoreResult: The perturbed image plus loss history and stop status.

    Raises:
        RuntimeError: If ``compute_loss`` yields no gradient w.r.t. ``pixel_values``.
    """
    inputs = target.preprocess(behavior=behavior, image=seed_image)
    base = inputs.pixel_values.detach()
    pixel_values = variant.initial_pixel_values(base=base, rng=rng)

    loss_history: list[float] = []
    final_loss = float("nan")
    succeeded = False
    steps_run = 0

    for step in range(num_steps):
        pixel_values = pixel_values.detach()
        pixel_values.requires_grad_(True)
        if pixel_values.grad is not None:
            pixel_values.grad = None

        loss = target.compute_loss(inputs=inputs.with_pixel_values(pixel_values), target_text=target_text)
        loss.backward()

        grad = pixel_values.grad
        if grad is None:
            raise RuntimeError(
                "compute_loss produced no gradient w.r.t. pixel_values. Ensure the loss "
                "is a differentiable function of inputs.pixel_values."
            )

        loss_value = float(loss.detach().item())
        loss_history.append(loss_value)
        final_loss = loss_value
        steps_run = step + 1

        with torch.no_grad():
            pixel_values = variant.project_step(
                pixel_values=pixel_values,
                base=base,
                grad=grad,
                step_size=step_size,
                epsilon=epsilon,
            )

        if verbose and (step % log_every == 0 or steps_run == num_steps):
            logger.info("PGD step %d/%d loss=%.6f", steps_run, num_steps, loss_value)

        if loss_value <= stop_loss:
            succeeded = True
            break

    image = target.to_pil(inputs=inputs.with_pixel_values(pixel_values.detach()))
    return PGDCoreResult(
        image=image,
        loss_history=loss_history,
        final_loss=final_loss,
        step_count=steps_run,
        succeeded=succeeded,
    )


__all__ = ["PGDCoreResult", "run_pgd"]
