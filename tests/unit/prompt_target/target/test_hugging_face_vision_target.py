# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for HuggingFaceVisionTarget's dynamic-resolution (Qwen2-VL) white-box path.

Network-free and model-free: the patch-reconstruction math is locked against a local
mirror of the Qwen2-VL image processor's forward patchify, and ``compute_loss`` /
``to_pil`` are exercised with stubbed processor + model so no weights are downloaded.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="gradient extra (torch) not installed")

from pyrit.prompt_target.common.white_box_target import WhiteBoxInputs  # noqa: E402
from pyrit.prompt_target.hugging_face.hugging_face_vision_target import (  # noqa: E402
    HuggingFaceVisionTarget,
    _reconstruct_image_from_flattened_patches,
)

_PATCH_SIZE = 14
_TEMPORAL_PATCH_SIZE = 2
_MERGE_SIZE = 2


def _forward_patchify(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror Qwen2VLImageProcessor's flatten so tests invert the real transform.

    Takes a ``[1, C, H, W]`` normalized image and returns the flattened
    ``[num_patches, patch_dim]`` pixel_values plus the ``[1, 3]`` image_grid_thw.
    """
    batch, channel = image.shape[:2]
    grid_h, grid_w = image.shape[2] // _PATCH_SIZE, image.shape[3] // _PATCH_SIZE
    patches = image.reshape(
        batch,
        channel,
        grid_h // _MERGE_SIZE,
        _MERGE_SIZE,
        _PATCH_SIZE,
        grid_w // _MERGE_SIZE,
        _MERGE_SIZE,
        _PATCH_SIZE,
    )
    patches = patches.permute(0, 2, 5, 3, 6, 1, 4, 7)
    flatten = (
        patches.unsqueeze(6)
        .expand(-1, -1, -1, -1, -1, -1, _TEMPORAL_PATCH_SIZE, -1, -1)
        .reshape(batch, grid_h * grid_w, channel * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE)
    )
    return flatten[0], torch.tensor([[1, grid_h, grid_w]])


def test_reconstruct_inverts_qwen_patchify_exactly() -> None:
    image = torch.rand(1, 3, 4 * _PATCH_SIZE, 6 * _PATCH_SIZE)
    pixel_values, grid = _forward_patchify(image)

    reconstructed = _reconstruct_image_from_flattened_patches(
        pixel_values=pixel_values,
        image_grid_thw=grid,
        patch_size=_PATCH_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        num_channels=3,
    )

    assert reconstructed.shape == (3, 4 * _PATCH_SIZE, 6 * _PATCH_SIZE)
    assert torch.allclose(reconstructed, image[0], atol=1e-6)


def test_reconstruct_rejects_multiple_images() -> None:
    grid = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixel_values = torch.zeros(2 * 2 * 2, 3 * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE)
    with pytest.raises(ValueError, match="single perturbed image"):
        _reconstruct_image_from_flattened_patches(
            pixel_values=pixel_values,
            image_grid_thw=grid,
            patch_size=_PATCH_SIZE,
            temporal_patch_size=_TEMPORAL_PATCH_SIZE,
            merge_size=_MERGE_SIZE,
            num_channels=3,
        )


def test_reconstruct_rejects_video_grid() -> None:
    grid = torch.tensor([[2, 2, 2]])
    pixel_values = torch.zeros(2 * 2 * 2, 3 * _TEMPORAL_PATCH_SIZE * _PATCH_SIZE * _PATCH_SIZE)
    with pytest.raises(ValueError, match="grid_t == 1"):
        _reconstruct_image_from_flattened_patches(
            pixel_values=pixel_values,
            image_grid_thw=grid,
            patch_size=_PATCH_SIZE,
            temporal_patch_size=_TEMPORAL_PATCH_SIZE,
            merge_size=_MERGE_SIZE,
            num_channels=3,
        )


@pytest.fixture
def stub_image_processor() -> SimpleNamespace:
    return SimpleNamespace(
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        patch_size=_PATCH_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
    )


@pytest.mark.usefixtures("patch_central_database")
def test_to_pil_renders_patch_flattened_layout(stub_image_processor: SimpleNamespace) -> None:
    target = HuggingFaceVisionTarget(model_id="stub/qwen-vl", device="cpu")
    target._loaded = True
    target._processor = SimpleNamespace(image_processor=stub_image_processor)

    original = torch.randint(0, 256, (3, 2 * _PATCH_SIZE, 2 * _PATCH_SIZE), dtype=torch.uint8)
    mean = torch.tensor(stub_image_processor.image_mean).view(-1, 1, 1)
    std = torch.tensor(stub_image_processor.image_std).view(-1, 1, 1)
    normalized = (original.float() / 255.0 - mean) / std
    pixel_values, grid = _forward_patchify(normalized.unsqueeze(0))

    rendered = target.to_pil(inputs=WhiteBoxInputs(pixel_values=pixel_values, model_inputs={"image_grid_thw": grid}))

    assert rendered.size == (2 * _PATCH_SIZE, 2 * _PATCH_SIZE)
    rendered_tensor = torch.frombuffer(bytearray(rendered.tobytes()), dtype=torch.uint8)
    rendered_tensor = rendered_tensor.reshape(2 * _PATCH_SIZE, 2 * _PATCH_SIZE, 3).permute(2, 0, 1)
    assert torch.equal(rendered_tensor, original)


class _RecordingModel:
    """A stand-in model that records the kwargs its forward receives."""

    def __init__(self) -> None:
        self.received: dict = {}

    def __call__(self, **kwargs: object) -> SimpleNamespace:
        self.received = kwargs
        return SimpleNamespace(loss=torch.tensor(1.0, requires_grad=True))


@pytest.mark.usefixtures("patch_central_database")
def test_compute_loss_extends_token_aligned_extras() -> None:
    target = HuggingFaceVisionTarget(model_id="stub/qwen-vl", device="cpu", dtype="float32")
    target._loaded = True

    target_ids = torch.tensor([[7, 8, 9]])  # three target tokens
    tokenizer = lambda text, add_special_tokens, return_tensors: SimpleNamespace(input_ids=target_ids)  # noqa: E731
    target._processor = SimpleNamespace(tokenizer=tokenizer)
    recording_model = _RecordingModel()
    target.model = recording_model

    prompt_length = 5
    prompt_ids = torch.arange(prompt_length).reshape(1, prompt_length)
    model_inputs = {
        "input_ids": prompt_ids,
        "attention_mask": torch.ones(1, prompt_length, dtype=torch.long),
        "mm_token_type_ids": torch.tensor([[0, 1, 1, 1, 0]]),  # token-aligned, must be extended
        "image_grid_thw": torch.tensor([[1, 2, 2]]),  # NOT token-aligned, passes through
    }
    inputs = WhiteBoxInputs(pixel_values=torch.rand(4, 12, requires_grad=True), model_inputs=model_inputs)

    target.compute_loss(inputs=inputs, target_text="unused")

    received = recording_model.received
    assert received["mm_token_type_ids"].shape[-1] == prompt_length + 3
    assert torch.equal(received["mm_token_type_ids"][0, prompt_length:], torch.zeros(3, dtype=torch.long))
    assert torch.equal(received["image_grid_thw"], model_inputs["image_grid_thw"])
    assert received["input_ids"].shape[-1] == prompt_length + 3
