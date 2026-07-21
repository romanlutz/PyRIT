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


class _Encoded(dict):
    """A processor output stand-in whose ``.to(device)`` returns itself."""

    def to(self, device: object) -> _Encoded:
        return self


class _StubProcessor:
    """Callable processor stub with a chat template and a fixed decode."""

    def __init__(self, *, decoded: str) -> None:
        self.tokenizer = SimpleNamespace(decode=lambda ids, skip_special_tokens: decoded)

    def apply_chat_template(self, messages: object, add_generation_prompt: bool) -> str:
        return "PROMPT"

    def __call__(self, **kwargs: object) -> _Encoded:
        encoded = _Encoded()
        encoded["input_ids"] = torch.tensor([[1, 2, 3]])
        return encoded


class _GenModel:
    """A model stub whose ``generate`` echoes the prompt plus two new tokens."""

    def generate(self, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[1, 2, 3, 4, 5]])


@pytest.mark.usefixtures("patch_central_database")
def test_generate_response_greedy_decodes_and_strips() -> None:
    import PIL.Image

    target = HuggingFaceVisionTarget(model_id="stub/vlm", device="cpu")
    target._loaded = True
    target._processor = _StubProcessor(decoded="  Sure, here is the plan  ")
    target.model = _GenModel()

    response = target.generate_response(behavior="carrier", image=PIL.Image.new("RGB", (4, 4)))

    assert response == "Sure, here is the plan"


def _clamp_target(*, image_mean: list[float], image_std: list[float]) -> HuggingFaceVisionTarget:
    target = HuggingFaceVisionTarget(model_id="stub/vlm", device="cpu")
    target._loaded = True
    target._processor = SimpleNamespace(
        image_processor=SimpleNamespace(
            image_mean=image_mean,
            image_std=image_std,
            patch_size=_PATCH_SIZE,
            temporal_patch_size=_TEMPORAL_PATCH_SIZE,
            merge_size=_MERGE_SIZE,
        )
    )
    return target


@pytest.mark.usefixtures("patch_central_database")
def test_clamp_to_displayable_bounds_fixed_resolution_layout() -> None:
    image_mean = [0.1, 0.5, 0.9]
    image_std = [0.2, 0.5, 0.5]
    target = _clamp_target(image_mean=image_mean, image_std=image_std)

    mean = torch.tensor(image_mean)
    std = torch.tensor(image_std)
    low = (0.0 - mean) / std
    high = (1.0 - mean) / std

    pixel_values = torch.empty(1, 3, 2, 2)
    for channel in range(3):
        pixel_values[0, channel, 0, :] = 100.0  # above the displayable ceiling
        pixel_values[0, channel, 1, :] = -100.0  # below the displayable floor

    clamped = target.clamp_to_displayable(inputs=WhiteBoxInputs(pixel_values=pixel_values))

    for channel in range(3):
        assert torch.allclose(clamped[0, channel, 0, :], high[channel].expand(2))
        assert torch.allclose(clamped[0, channel, 1, :], low[channel].expand(2))
    denormalized = clamped[0] * std.view(3, 1, 1) + mean.view(3, 1, 1)
    assert denormalized.min().item() >= -1e-6
    assert denormalized.max().item() <= 1 + 1e-6


@pytest.mark.usefixtures("patch_central_database")
def test_clamp_to_displayable_bounds_flattened_qwen_layout() -> None:
    image_mean = [0.1, 0.5, 0.9]
    image_std = [0.2, 0.5, 0.5]
    target = _clamp_target(image_mean=image_mean, image_std=image_std)

    # Out-of-gamut normalized image with both extremes present in every channel.
    normalized = torch.full((1, 3, 2 * _PATCH_SIZE, 2 * _PATCH_SIZE), 50.0)
    normalized[..., ::2] = -50.0
    pixel_values, grid = _forward_patchify(normalized)

    clamped = target.clamp_to_displayable(
        inputs=WhiteBoxInputs(pixel_values=pixel_values, model_inputs={"image_grid_thw": grid})
    )

    reconstructed = _reconstruct_image_from_flattened_patches(
        pixel_values=clamped,
        image_grid_thw=grid,
        patch_size=_PATCH_SIZE,
        temporal_patch_size=_TEMPORAL_PATCH_SIZE,
        merge_size=_MERGE_SIZE,
        num_channels=3,
    )

    low = (0.0 - torch.tensor(image_mean)) / torch.tensor(image_std)
    high = (1.0 - torch.tensor(image_mean)) / torch.tensor(image_std)
    # Each channel saturates to ITS OWN displayable bound, proving the column->channel map.
    for channel in range(3):
        assert torch.isclose(reconstructed[channel].min(), low[channel], atol=1e-6)
        assert torch.isclose(reconstructed[channel].max(), high[channel], atol=1e-6)


class _RoundTripImageProcessor:
    """Callable stub that re-preprocesses a PIL image the way Qwen2VLImageProcessor does."""

    def __init__(self, *, image_mean: list[float], image_std: list[float]) -> None:
        self.image_mean = image_mean
        self.image_std = image_std
        self.patch_size = _PATCH_SIZE
        self.temporal_patch_size = _TEMPORAL_PATCH_SIZE
        self.merge_size = _MERGE_SIZE

    def __call__(self, *, images: object, return_tensors: str) -> dict:
        import PIL.Image

        assert isinstance(images, PIL.Image.Image)
        width, height = images.size
        raw = torch.frombuffer(bytearray(images.tobytes()), dtype=torch.uint8).float() / 255.0
        chw = raw.reshape(height, width, 3).permute(2, 0, 1)
        mean = torch.tensor(self.image_mean).view(-1, 1, 1)
        std = torch.tensor(self.image_std).view(-1, 1, 1)
        normalized = (chw - mean) / std
        pixel_values, _ = _forward_patchify(normalized.unsqueeze(0))
        return {"pixel_values": pixel_values}


@pytest.mark.usefixtures("patch_central_database")
def test_quantize_to_displayable_round_trips_and_passes_gradient() -> None:
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    target = HuggingFaceVisionTarget(model_id="stub/qwen-vl", device="cpu")
    target._loaded = True
    target._processor = SimpleNamespace(
        image_processor=_RoundTripImageProcessor(image_mean=image_mean, image_std=image_std)
    )

    # Start from an exact 8-bit image so the render -> re-preprocess round-trip is lossless.
    original = torch.randint(0, 256, (3, 2 * _PATCH_SIZE, 2 * _PATCH_SIZE), dtype=torch.uint8)
    mean = torch.tensor(image_mean).view(-1, 1, 1)
    std = torch.tensor(image_std).view(-1, 1, 1)
    normalized = (original.float() / 255.0 - mean) / std
    base, grid = _forward_patchify(normalized.unsqueeze(0))

    pixel_values = base.clone().requires_grad_(True)
    deployed = target.quantize_to_displayable(
        inputs=WhiteBoxInputs(pixel_values=pixel_values, model_inputs={"image_grid_thw": grid})
    )

    # Forward value equals the deployed (re-preprocessed) tensor; lossless for an 8-bit-exact image.
    assert torch.allclose(deployed.detach(), base, atol=1e-6)
    # Straight-through estimator: gradient flows identically back to the optimized tensor.
    deployed.sum().backward()
    assert pixel_values.grad is not None
    assert torch.allclose(pixel_values.grad, torch.ones_like(pixel_values))


@pytest.mark.usefixtures("patch_central_database")
def test_quantize_to_displayable_rejects_shape_mismatch() -> None:
    target = HuggingFaceVisionTarget(model_id="stub/qwen-vl", device="cpu")
    target._loaded = True

    class _WrongShapeProcessor(_RoundTripImageProcessor):
        def __call__(self, *, images: object, return_tensors: str) -> dict:
            result = super().__call__(images=images, return_tensors=return_tensors)
            return {"pixel_values": result["pixel_values"][:1]}  # drop rows to force a mismatch

    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]
    target._processor = SimpleNamespace(
        image_processor=_WrongShapeProcessor(image_mean=image_mean, image_std=image_std)
    )

    original = torch.randint(0, 256, (3, 2 * _PATCH_SIZE, 2 * _PATCH_SIZE), dtype=torch.uint8)
    mean = torch.tensor(image_mean).view(-1, 1, 1)
    std = torch.tensor(image_std).view(-1, 1, 1)
    normalized = (original.float() / 255.0 - mean) / std
    base, grid = _forward_patchify(normalized.unsqueeze(0))

    with pytest.raises(ValueError, match="does not match"):
        target.quantize_to_displayable(inputs=WhiteBoxInputs(pixel_values=base, model_inputs={"image_grid_thw": grid}))
