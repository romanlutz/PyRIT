# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Capability interface for prompt targets that expose model internals.

``WhiteBoxTarget`` is a structural ``Protocol`` (not an ABC) so that a target can
declare it *in addition to* ``PromptTarget`` without a deeper inheritance tree, and
so that lightweight test doubles can satisfy the gradient surface without
implementing the full asynchronous ``PromptTarget`` send flow.

The module imports cleanly without ``torch`` / ``PIL``: those are referenced only
under ``TYPE_CHECKING``. Consumers (e.g. the multimodal PGD generator) type their
dependency on the ``WhiteBoxTarget`` Protocol, so importing this module never pulls
in the heavy machine-learning stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import PIL.Image
    import torch


@dataclass
class WhiteBoxInputs:
    """
    Model inputs for a single white-box forward/backward pass.

    Separates the one differentiable leaf tensor that gradient-based image attacks
    perturb (``pixel_values``) from the remaining, non-differentiable model inputs
    that are threaded through unchanged (token ids, attention masks, and any
    model-specific extras such as ``image_grid_thw`` for Qwen2-VL or ``image_sizes``
    for LLaVA-Next).

    Attributes:
        pixel_values (torch.Tensor): The normalized image tensor consumed by the
            model. This is the leaf an image attack sets ``requires_grad_(True)`` on.
        model_inputs (dict[str, Any]): All other keyword arguments forwarded to the
            model's forward pass (e.g. ``input_ids``, ``attention_mask``, ``labels``,
            and model-specific image-grid metadata).
    """

    pixel_values: torch.Tensor
    model_inputs: dict[str, Any] = field(default_factory=dict)

    def with_pixel_values(self, pixel_values: torch.Tensor) -> WhiteBoxInputs:
        """
        Return a copy of these inputs with ``pixel_values`` swapped out.

        The ``model_inputs`` mapping is shallow-copied so callers can iterate the
        optimization loop without mutating the original container.

        Args:
            pixel_values (torch.Tensor): The replacement pixel tensor.

        Returns:
            WhiteBoxInputs: A new container sharing ``model_inputs`` (shallow copy)
            with the supplied ``pixel_values``.
        """
        return replace(self, pixel_values=pixel_values, model_inputs=dict(self.model_inputs))


@runtime_checkable
class WhiteBoxTarget(Protocol):
    """
    Capability interface for prompt targets that expose model gradients.

    Implemented by *locally-loaded* targets (HuggingFace transformers backends, …)
    where the model's forward pass and input gradients are accessible. NOT
    implementable by remote / API targets (OpenAI, Azure ML endpoints, Ollama HTTP).

    A concrete user-facing class typically inherits ``PromptTarget`` for the
    black-box ``send_prompt_async`` flow AND implements this Protocol for the
    white-box gradient flow, against the same in-process model object. Consumers
    such as the multimodal PGD generator depend on this Protocol rather than on any
    concrete target class.
    """

    @property
    def vlm_id(self) -> str:
        """
        Identifier of the loaded vision-language model (e.g. its HuggingFace id).

        Returns:
            str: The model identifier echoed into manifest rows and file names.
        """
        ...

    device: str

    def preprocess(self, *, behavior: str, image: PIL.Image.Image) -> WhiteBoxInputs:
        """
        Build model inputs pairing a carrier ``behavior`` with an ``image``.

        Args:
            behavior (str): The benign carrier prompt sent alongside the image.
            image (PIL.Image.Image): The seed image to perturb.

        Returns:
            WhiteBoxInputs: The differentiable ``pixel_values`` leaf plus the
            pass-through model inputs.
        """
        ...

    def compute_loss(self, *, inputs: WhiteBoxInputs, target_text: str) -> torch.Tensor:
        """
        Compute the scalar cross-entropy loss of ``target_text`` given ``inputs``.

        The returned tensor must be part of the autograd graph rooted at
        ``inputs.pixel_values`` so that ``loss.backward()`` populates
        ``inputs.pixel_values.grad``.

        Args:
            inputs (WhiteBoxInputs): Model inputs whose ``pixel_values`` leaf has
                ``requires_grad=True``.
            target_text (str): The affirmative target string the attack maximizes
                the likelihood of.

        Returns:
            torch.Tensor: A scalar loss tensor connected to ``inputs.pixel_values``.
        """
        ...

    def to_pil(self, *, inputs: WhiteBoxInputs) -> PIL.Image.Image:
        """
        Denormalize ``inputs.pixel_values`` back into a viewable PIL image.

        Args:
            inputs (WhiteBoxInputs): Model inputs holding the (possibly perturbed)
                ``pixel_values`` tensor in normalized model space.

        Returns:
            PIL.Image.Image: The reconstructed RGB image.
        """
        ...

    def release_white_box_resources(self) -> None:
        """Release any GPU / model resources held for the white-box surface."""
        ...


__all__ = ["WhiteBoxInputs", "WhiteBoxTarget"]
