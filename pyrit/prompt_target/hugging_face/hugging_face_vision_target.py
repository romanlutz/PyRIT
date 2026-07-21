# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
A generic, locally-loaded HuggingFace vision-language target.

``HuggingFaceVisionTarget`` loads any image-text-to-text model by ``model_id`` via
``AutoProcessor`` + ``AutoModelForImageTextToText`` and exposes two surfaces against
the same in-process model:

- **Black-box** (``PromptTarget``): ``send_prompt_async`` runs generation on a
  text (+ optional image) prompt, like ``HuggingFaceChatTarget`` but multimodal.
- **White-box** (``WhiteBoxTarget`` Protocol): ``preprocess`` / ``compute_loss`` /
  ``to_pil`` / ``release_white_box_resources`` provide the differentiable gradient
  path that image attacks such as PGD optimize against.

The class is intentionally model-agnostic: model-specific behavior (image-token
expansion, extra processor tensors like ``image_grid_thw`` / ``image_sizes``,
normalization constants) is discovered from the processor at runtime rather than
hard-coded per model. ``epsilon`` / ``step_size`` for downstream attacks are
expressed in the processor's **normalized** ``pixel_values`` space (the differentiable
model input).

Requires the ``gradient`` (or any torch-bearing) extra; ``torch`` and
``transformers`` vision auto-classes are imported at module load.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from transformers import (
    AutoModelForImageTextToText,  # type: ignore[ty:possibly-missing-import]
    AutoProcessor,  # type: ignore[ty:possibly-missing-import]
)

from pyrit.common import default_values
from pyrit.common.download_hf_model import download_specific_files_async
from pyrit.exceptions import EmptyResponseException, pyrit_target_retry
from pyrit.models import ComponentIdentifier, Message, construct_response_from_request
from pyrit.models.target.target_capabilities import TargetCapabilities
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.common.utils import limit_requests_per_minute
from pyrit.prompt_target.common.white_box_target import WhiteBoxInputs

if TYPE_CHECKING:
    import PIL.Image

logger = logging.getLogger(__name__)

_TEXT_IMAGE_INPUT = frozenset({frozenset({"text"}), frozenset({"image_path"}), frozenset({"text", "image_path"})})
_TEXT_OUTPUT = frozenset({frozenset({"text"})})


def _reconstruct_image_from_flattened_patches(
    *,
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    num_channels: int,
) -> torch.Tensor:
    """
    Invert the Qwen2-VL patchify to recover a ``[C, H, W]`` normalized image.

    Qwen2-VL-style processors flatten an image into
    ``[grid_t*grid_h*grid_w, num_channels*temporal_patch_size*patch_size**2]`` patch
    rows (plus an ``image_grid_thw``) instead of a dense ``[C, H, W]`` tensor. This
    reverses the exact reshape/permute the image processor applies so a perturbed
    ``pixel_values`` can be rendered back to a PNG. Only single still images
    (``grid_t == 1``, one grid row) are supported.

    Args:
        pixel_values (torch.Tensor): The flattened ``[num_patches, patch_dim]`` tensor.
        image_grid_thw (torch.Tensor): The ``[1, 3]`` grid ``(t, h, w)`` in patch units.
        patch_size (int): Spatial patch size.
        temporal_patch_size (int): Temporal patch size (frames folded per patch).
        merge_size (int): Spatial merge factor.
        num_channels (int): Number of image channels.

    Returns:
        torch.Tensor: The reconstructed ``[C, H, W]`` tensor in normalized model space.

    Raises:
        ValueError: If more than one image or a temporal (video) grid is present.
    """
    if image_grid_thw.shape[0] != 1:
        raise ValueError(
            f"Expected a single image grid, got {image_grid_thw.shape[0]} rows. "
            "Rendering a PNG is only supported for a single perturbed image."
        )
    grid_t, grid_h, grid_w = (int(x) for x in image_grid_thw[0].tolist())
    if grid_t != 1:
        raise ValueError(f"Only still images (grid_t == 1) can be rendered, got grid_t={grid_t}.")

    merged_h, merged_w = grid_h // merge_size, grid_w // merge_size
    # Split the flattened rows/features back into the (grid, merge, patch) axes the
    # processor collapsed, drop the duplicated temporal frame, then regroup into [C, H, W].
    reshaped = pixel_values.reshape(
        merged_h, merged_w, merge_size, merge_size, num_channels, temporal_patch_size, patch_size, patch_size
    )
    frame = reshaped[:, :, :, :, :, 0, :, :]  # [merged_h, merged_w, mh, mw, C, ph, pw]
    image = frame.permute(4, 0, 2, 5, 1, 3, 6).contiguous()  # [C, merged_h, mh, ph, merged_w, mw, pw]
    return image.reshape(num_channels, grid_h * patch_size, grid_w * patch_size)


class HuggingFaceVisionTarget(PromptTarget):
    """
    A locally-loaded HuggingFace VLM exposing black-box and white-box surfaces.

    Loads any vision-language model by ``model_id`` and serves both PyRIT's
    ``send_prompt_async`` flow and the ``WhiteBoxTarget`` gradient flow used by
    image attacks. See the module docstring for the design rationale.
    """

    _DEFAULT_CONFIGURATION: TargetConfiguration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=False,
            supports_multi_message_pieces=True,
            input_modalities=cast("Any", _TEXT_IMAGE_INPUT),
            output_modalities=cast("Any", _TEXT_OUTPUT),
        )
    )

    HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE = "HUGGINGFACE_TOKEN"

    model: Any

    def __init__(
        self,
        *,
        model_id: str,
        device: str = "cuda:0",
        dtype: str = "float16",
        hf_token: str | None = None,
        trust_remote_code: bool = False,
        max_new_tokens: int = 64,
        necessary_files: list[str] | None = None,
        max_requests_per_minute: int | None = None,
        custom_configuration: TargetConfiguration | None = None,
    ) -> None:
        """
        Initialize the vision target.

        Args:
            model_id (str): HuggingFace model identifier (e.g.
                ``"llava-hf/llava-1.5-7b-hf"``).
            device (str): Torch device string. Defaults to ``"cuda:0"``.
            dtype (str): Torch dtype name used to load the model (e.g. ``"float16"``,
                ``"bfloat16"``, ``"float32"``). Defaults to ``"float16"``.
            hf_token (str | None): HuggingFace token for gated models. ``None`` falls
                back to the ``HUGGINGFACE_TOKEN`` environment variable (optional).
            trust_remote_code (bool): Whether to trust remote model code. Defaults to
                False.
            max_new_tokens (int): Max new tokens for the black-box generation path.
                Defaults to 64.
            necessary_files (list[str] | None): Optional subset of model files to
                download. ``None`` downloads all files.
            max_requests_per_minute (int | None): Optional rate limit for the
                black-box send path.
            custom_configuration (TargetConfiguration | None): Per-instance
                capability override.

        Raises:
            ValueError: If ``model_id`` is empty.
        """
        super().__init__(
            max_requests_per_minute=max_requests_per_minute,
            model_name=model_id,
            custom_configuration=custom_configuration,
        )
        if not model_id:
            raise ValueError("HuggingFaceVisionTarget: 'model_id' must be a non-empty HuggingFace model identifier.")

        self.model_id = model_id
        self.device = device
        self._dtype_name = dtype
        self._trust_remote_code = trust_remote_code
        self._max_new_tokens = max_new_tokens
        self._necessary_files = necessary_files
        self._torch_dtype = getattr(torch, dtype)

        self._hf_token = default_values.get_non_required_value(
            env_var_name=self.HUGGINGFACE_TOKEN_ENVIRONMENT_VARIABLE, passed_value=hf_token
        )

        self.model = None
        self._processor: Any = None
        self._loaded = False

    # ------------------------------------------------------------------ identity

    @property
    def vlm_id(self) -> str:
        """The HuggingFace id of the loaded model (satisfies ``WhiteBoxTarget``)."""
        return self.model_id

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "model_id": self.model_id,
                "device": self.device,
                "dtype": self._dtype_name,
                "trust_remote_code": self._trust_remote_code,
                "max_new_tokens": self._max_new_tokens,
            },
        )

    # -------------------------------------------------------------- model loading

    def _ensure_loaded(self) -> None:
        """Download (if needed) and load the processor + model, once, synchronously."""
        if self._loaded:
            return
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{self.model_id.replace('/', '--')}"
        self._processor = AutoProcessor.from_pretrained(
            self.model_id, cache_dir=cache_dir, trust_remote_code=self._trust_remote_code
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            cache_dir=cache_dir,
            torch_dtype=self._torch_dtype,
            trust_remote_code=self._trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True
        logger.info("Loaded vision model %s on %s (%s).", self.model_id, self.device, self._dtype_name)

    async def _ensure_loaded_async(self) -> None:
        """Async loader that downloads model files before the blocking load."""
        if self._loaded:
            return
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{self.model_id.replace('/', '--')}"
        await download_specific_files_async(self.model_id, self._necessary_files, self._hf_token or "", cache_dir)
        self._ensure_loaded()

    # ------------------------------------------------------------- white-box path

    def preprocess(self, *, behavior: str, image: PIL.Image.Image) -> WhiteBoxInputs:
        """
        Build model inputs pairing ``behavior`` (as the user prompt) with ``image``.

        The differentiable normalized ``pixel_values`` leaf is separated from the
        prompt token ids and any model-specific extras (``image_grid_thw`` for
        Qwen2-VL, ``image_sizes`` for LLaVA-Next, ...), which are threaded through
        ``compute_loss`` unchanged.

        Args:
            behavior (str): The benign carrier prompt.
            image (PIL.Image.Image): The seed image to perturb.

        Returns:
            WhiteBoxInputs: The ``pixel_values`` leaf plus pass-through model inputs.
        """
        self._ensure_loaded()
        prompt_text = self._build_prompt_text(behavior=behavior)
        encoded = self._processor(images=image, text=prompt_text, return_tensors="pt")
        encoded = encoded.to(self.device)

        model_inputs = dict(encoded)
        pixel_values = model_inputs.pop("pixel_values").to(torch.float32)
        return WhiteBoxInputs(pixel_values=pixel_values, model_inputs=model_inputs)

    def compute_loss(self, *, inputs: WhiteBoxInputs, target_text: str) -> torch.Tensor:
        """
        Cross-entropy of ``target_text`` given the (image + behavior) prompt.

        Appends the target tokens to the prompt, masks the prompt positions in the
        labels, and runs a single forward pass so the returned scalar is a
        differentiable function of ``inputs.pixel_values``.

        Args:
            inputs (WhiteBoxInputs): Inputs whose ``pixel_values`` leaf has
                ``requires_grad=True``.
            target_text (str): The affirmative target string.

        Returns:
            torch.Tensor: A scalar loss connected to ``inputs.pixel_values``.
        """
        self._ensure_loaded()
        tokenizer = self._processor.tokenizer

        prompt_ids = inputs.model_inputs["input_ids"]
        attention_mask = inputs.model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids)

        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt").input_ids.to(self.device)

        full_ids = torch.cat([prompt_ids, target_ids], dim=1)
        full_attention = torch.cat([attention_mask, torch.ones_like(target_ids)], dim=1)
        labels = torch.cat([torch.full_like(prompt_ids, -100), target_ids], dim=1)

        prompt_length = prompt_ids.shape[1]
        target_length = target_ids.shape[1]
        extras: dict[str, Any] = {}
        for key, value in inputs.model_inputs.items():
            if key in ("input_ids", "attention_mask", "pixel_values"):
                continue
            # Token-aligned extras (e.g. Qwen2-VL's mm_token_type_ids) must grow to cover the
            # appended target tokens, which are text (type 0), so the model's mRoPE and image
            # placeholder masks stay valid. Non-token tensors (image_grid_thw) pass through.
            if torch.is_tensor(value) and value.dim() >= 1 and value.shape[-1] == prompt_length:
                value = torch.cat([value, value.new_zeros((*value.shape[:-1], target_length))], dim=-1)
            extras[key] = value

        outputs = self.model(
            input_ids=full_ids,
            attention_mask=full_attention,
            pixel_values=inputs.pixel_values.to(self._torch_dtype),
            labels=labels,
            **extras,
        )
        return outputs.loss

    def to_pil(self, *, inputs: WhiteBoxInputs) -> PIL.Image.Image:
        """
        Denormalize ``inputs.pixel_values`` back into a viewable RGB image.

        Uses the processor's ``image_mean`` / ``image_std`` to invert normalization.
        Supports the fixed-resolution ``[N, C, H, W]`` / ``[C, H, W]`` layouts (e.g.
        LLaVA-1.5) and the patch-flattened dynamic-resolution layout used by the
        Qwen2-VL family (a 2D ``pixel_values`` plus ``image_grid_thw``), which is
        rebuilt into ``[C, H, W]`` before denormalizing. Layouts that cannot be
        reshaped to a single still image raise ``ValueError``.

        Args:
            inputs (WhiteBoxInputs): Inputs holding the (possibly perturbed)
                ``pixel_values`` tensor in normalized model space.

        Returns:
            PIL.Image.Image: The reconstructed RGB image.

        Raises:
            ValueError: If ``pixel_values`` is not a reshapeable image tensor.
        """
        import numpy as np
        import PIL.Image

        self._ensure_loaded()
        image_processor = self._processor.image_processor
        pixel_values = inputs.pixel_values.detach().to(torch.float32).cpu()

        image_grid_thw = inputs.model_inputs.get("image_grid_thw")
        if pixel_values.dim() == 2 and image_grid_thw is not None:
            pixel_values = _reconstruct_image_from_flattened_patches(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw.detach().cpu(),
                patch_size=int(getattr(image_processor, "patch_size", 14)),
                temporal_patch_size=int(getattr(image_processor, "temporal_patch_size", 2)),
                merge_size=int(getattr(image_processor, "merge_size", 2)),
                num_channels=len(image_processor.image_mean),
            )
        elif pixel_values.dim() == 4:
            pixel_values = pixel_values[0]

        if pixel_values.dim() != 3:
            raise ValueError(
                f"to_pil expects a [C, H, W] / [N, C, H, W] tensor, or a 2D patch-flattened tensor "
                f"with image_grid_thw, got shape {tuple(inputs.pixel_values.shape)}. This model uses a "
                "pixel layout that cannot be rendered back to a single PNG."
            )

        mean = torch.tensor(image_processor.image_mean, dtype=torch.float32).view(-1, 1, 1)
        std = torch.tensor(image_processor.image_std, dtype=torch.float32).view(-1, 1, 1)
        denormalized = (pixel_values * std + mean).clamp(0.0, 1.0)
        array = (denormalized.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return PIL.Image.fromarray(array)

    def clamp_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        """
        Project normalized ``pixel_values`` onto the displayable ``[0, 1]`` image range.

        PGD optimizes in the processor's normalized space, but ``to_pil`` clamps the
        denormalized image to ``[0, 1]`` before writing a PNG. Clamping every step keeps
        the optimized tensor on the renderable manifold so the loss reflects the image
        actually saved and queried back, not an out-of-gamut tensor that rendering
        discards. Handles both the fixed-resolution ``[C, H, W]`` / ``[N, C, H, W]``
        layout and the Qwen2-VL patch-flattened ``[num_patches, C*T*P*P]`` layout, where
        the channel is the slowest-varying axis within each patch row.

        Args:
            inputs (WhiteBoxInputs): Model inputs whose ``pixel_values`` tensor is projected.

        Returns:
            torch.Tensor: A detached ``pixel_values`` tensor clamped into each channel's
            displayable bound, matching the input shape and normalization.
        """
        self._ensure_loaded()
        image_processor = self._processor.image_processor
        pixel_values = inputs.pixel_values.detach()

        mean = torch.tensor(image_processor.image_mean, dtype=pixel_values.dtype, device=pixel_values.device)
        std = torch.tensor(image_processor.image_std, dtype=pixel_values.dtype, device=pixel_values.device)
        low = (0.0 - mean) / std
        high = (1.0 - mean) / std
        num_channels = mean.shape[0]

        if pixel_values.dim() == 2:
            # Qwen2-VL flattened [num_patches, C*T*P*P]: the channel occupies the slowest
            # axis, so each channel spans a contiguous block of columns.
            channels_per_column = pixel_values.shape[1] // num_channels
            column_channel = torch.arange(pixel_values.shape[1], device=pixel_values.device) // channels_per_column
            low = low[column_channel]
            high = high[column_channel]
        else:
            broadcast_shape = [1] * pixel_values.dim()
            broadcast_shape[-3] = num_channels
            low = low.view(broadcast_shape)
            high = high.view(broadcast_shape)

        return torch.maximum(torch.minimum(pixel_values, high), low)

    def quantize_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        """
        Snap ``inputs.pixel_values`` to the deployed 8-bit image via a straight-through estimator.

        PGD minimizes the loss on the *continuous* normalized ``pixel_values``, but the
        artifact that actually ships is an 8-bit PNG that the model re-preprocesses at
        inference. Any degree of freedom the optimizer exploits that the render round-trip
        discards — 1/255 quantization noise, or a Qwen2-VL image's duplicated temporal
        patches diverging when only slot 0 is rendered — leaves a near-zero training loss
        that the shipped image no longer reproduces. This method returns exactly the pixels
        the model will see at deployment by running the real ``to_pil`` -> ``image_processor``
        round-trip, so quantization, temporal re-duplication, and any resize are all folded
        in. The rounding is non-differentiable, so a straight-through estimator makes the
        forward value the deployed tensor while the backward pass is identity w.r.t. the
        input, letting the optimizer descend the *deployed* image's loss.

        Args:
            inputs (WhiteBoxInputs): Model inputs whose ``pixel_values`` leaf (with any
                layout metadata) is projected to its deployed 8-bit form.

        Returns:
            torch.Tensor: A ``pixel_values`` tensor whose forward value equals the
            re-preprocessed 8-bit image and whose gradient flows straight through to the
            input, matching the input shape and normalization.

        Raises:
            ValueError: If the re-preprocessed tensor's shape does not match the input.
        """
        self._ensure_loaded()
        pixel_values = inputs.pixel_values
        rendered = self.to_pil(inputs=inputs)
        reprocessed = self._processor.image_processor(images=rendered, return_tensors="pt")
        deployed = reprocessed["pixel_values"].to(device=pixel_values.device, dtype=pixel_values.dtype)
        if deployed.shape != pixel_values.shape:
            raise ValueError(
                f"quantize_to_displayable: re-preprocessed pixel_values shape {tuple(deployed.shape)} "
                f"does not match the optimized tensor shape {tuple(pixel_values.shape)}."
            )
        return pixel_values + (deployed - pixel_values).detach()

    def release_white_box_resources(self) -> None:
        """Free the loaded model and empty the CUDA cache, if any."""
        self.model = None
        self._processor = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------- black-box path

    @limit_requests_per_minute
    @pyrit_target_retry
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Generate a response to the last message's text (+ optional image).

        Args:
            normalized_conversation (list[Message]): The normalized conversation; the
                current request is the last element.

        Returns:
            list[Message]: A single-element list with the model's text response.

        Raises:
            EmptyResponseException: If the model produces an empty response.
        """
        await self._ensure_loaded_async()

        message = normalized_conversation[-1]
        request_piece = message.message_pieces[0]

        text_pieces = message.get_pieces_by_type(data_type="text")
        image_pieces = message.get_pieces_by_type(data_type="image_path")
        prompt = "\n".join(piece.converted_value for piece in text_pieces).strip()

        image = None
        if image_pieces:
            import PIL.Image

            image = PIL.Image.open(image_pieces[0].converted_value).convert("RGB")

        prompt_text = self._build_prompt_text(behavior=prompt, has_image=image is not None)
        response_text = self._generate_text(behavior=prompt, image=image, prompt_text=prompt_text)

        if not response_text:
            raise EmptyResponseException

        response = construct_response_from_request(
            request=request_piece,
            response_text_pieces=[response_text],
            prompt_metadata={"model_id": self.model_id, "device": self.device},
        )
        return [response]

    def generate_response(self, *, behavior: str, image: PIL.Image.Image) -> str:
        """
        Greedily decode the model's black-box reply to ``behavior`` paired with ``image``.

        Synchronous companion to the white-box gradient surface (``preprocess`` /
        ``compute_loss`` / ``to_pil``): after an image attack perturbs ``pixel_values`` and
        renders the result to PIL, callers pass it here to check whether the model now
        actually emits the target string. Unlike ``send_prompt_async`` this returns the raw
        decoded text without wrapping it in a ``Message`` or touching memory.

        Args:
            behavior (str): The benign carrier prompt sent alongside the image.
            image (PIL.Image.Image): The image to send (typically the perturbed result).

        Returns:
            str: The model's greedily-decoded response text (possibly empty).
        """
        self._ensure_loaded()
        return self._generate_text(behavior=behavior, image=image)

    def _generate_text(self, *, behavior: str, image: PIL.Image.Image | None, prompt_text: str | None = None) -> str:
        """
        Greedy-decode the model's reply to ``behavior`` (+ optional ``image``).

        Shared core of the black-box send path and the white-box verification helper;
        assumes the processor + model are already loaded.

        Returns:
            str: The decoded, stripped response text.
        """
        if prompt_text is None:
            prompt_text = self._build_prompt_text(behavior=behavior, has_image=image is not None)
        processor_kwargs: dict[str, Any] = {"text": prompt_text, "return_tensors": "pt"}
        if image is not None:
            processor_kwargs["images"] = image
        encoded = self._processor(**processor_kwargs).to(self.device)

        input_length = encoded["input_ids"].shape[-1]
        with torch.no_grad():
            generated = self.model.generate(**encoded, max_new_tokens=self._max_new_tokens, do_sample=False)
        return self._processor.tokenizer.decode(generated[0][input_length:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------- internals

    def _build_prompt_text(self, *, behavior: str, has_image: bool = True) -> str:
        """
        Render the chat prompt string, applying the processor's chat template.

        Falls back to the raw behavior text when the processor lacks a chat template.

        Returns:
            str: The rendered prompt string ready for the processor.
        """
        content: list[dict[str, str]] = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": behavior})
        messages = [{"role": "user", "content": content}]
        try:
            return self._processor.apply_chat_template(messages, add_generation_prompt=True)
        except (ValueError, AttributeError, TypeError) as e:
            logger.debug("apply_chat_template unavailable for %s (%s); using raw text.", self.model_id, e)
            return behavior


__all__ = ["HuggingFaceVisionTarget"]
