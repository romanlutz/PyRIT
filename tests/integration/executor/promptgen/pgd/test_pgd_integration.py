# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""CPU integration test for the PGD generator.

Exercises the full pipeline — config -> generator -> pgd_core -> variant -> manifest
-> cached PNG — against a ``TinyWhiteBoxTarget`` (one linear layer, real
``loss.backward()``). No GPU, no network, no real VLM: runnable in CI when
``RUN_ALL_TESTS=true``. GPU smoke tests against a real LLaVA live alongside, guarded
by ``@pytest.mark.gpu`` so they are collected only on a CUDA box.
"""

from __future__ import annotations

from pathlib import Path

import PIL.Image
import pytest

torch = pytest.importorskip("torch", reason="gradient extra (torch) not installed")

from pyrit.executor.promptgen.pgd.config import (  # noqa: E402
    PGDAlgorithmConfig,
    PGDModelConfig,
    PGDOutputConfig,
    PGDVariant,
    PGDVariantConfig,
)
from pyrit.executor.promptgen.pgd.generator import PGDGenerator  # noqa: E402
from pyrit.executor.promptgen.pgd.manifest import read_manifest  # noqa: E402
from pyrit.prompt_target.common.white_box_target import WhiteBoxInputs  # noqa: E402


class TinyWhiteBoxTarget:
    """A CPU ``WhiteBoxTarget`` with a single linear layer and a real gradient path.

    The loss is the cross-entropy of a fixed target class given the flattened
    ``pixel_values``, so ``loss.backward()`` populates ``pixel_values.grad`` exactly
    like a real VLM would, letting PGD actually reduce the loss.
    """

    vlm_id = "tiny/whitebox-1.0"
    device = "cpu"

    def __init__(self, *, image_size: int = 8, vocab: int = 8, seed: int = 0) -> None:
        self._image_size = image_size
        self._vocab = vocab
        generator = torch.Generator().manual_seed(seed)
        self._weight = torch.randn(vocab, 3 * image_size * image_size, generator=generator) * 0.1
        self._bias = torch.zeros(vocab)
        self.released = False

    def preprocess(self, *, behavior: str, image: PIL.Image.Image) -> WhiteBoxInputs:
        resized = image.convert("RGB").resize((self._image_size, self._image_size))
        raw = torch.frombuffer(bytearray(resized.tobytes()), dtype=torch.uint8).float() / 255.0
        pixel_values = raw.reshape(1, self._image_size, self._image_size, 3).permute(0, 3, 1, 2).contiguous()
        return WhiteBoxInputs(pixel_values=pixel_values)

    def _target_index(self, target_text: str) -> int:
        return sum(ord(c) for c in target_text) % self._vocab

    def compute_loss(self, *, inputs: WhiteBoxInputs, target_text: str) -> torch.Tensor:
        logits = inputs.pixel_values.reshape(1, -1) @ self._weight.t() + self._bias
        target = torch.tensor([self._target_index(target_text)])
        return torch.nn.functional.cross_entropy(logits, target)

    def to_pil(self, *, inputs: WhiteBoxInputs) -> PIL.Image.Image:
        arr = inputs.pixel_values.detach().clamp(0, 1)[0].permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
        return PIL.Image.fromarray(arr)

    def clamp_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        return inputs.pixel_values.detach().clamp(0.0, 1.0)

    def quantize_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        pixel_values = inputs.pixel_values
        rendered = self.to_pil(inputs=inputs)
        deployed = self.preprocess(behavior="", image=rendered).pixel_values.to(
            device=pixel_values.device, dtype=pixel_values.dtype
        )
        return pixel_values + (deployed - pixel_values).detach()

    def release_white_box_resources(self) -> None:
        self.released = True


def _algorithm() -> PGDAlgorithmConfig:
    return PGDAlgorithmConfig(num_steps=25, step_size=0.02, epsilon=0.5, stop_loss=-1.0)


@pytest.mark.run_only_if_all_tests
async def test_eps_bounded_pipeline_writes_manifest_and_png(tmp_path: Path, patch_central_database) -> None:
    seed_path = tmp_path / "seed.png"
    PIL.Image.new("RGB", (16, 16), (120, 120, 120)).save(seed_path)
    manifest_path = tmp_path / "manifest.jsonl"

    generator = PGDGenerator(
        target=TinyWhiteBoxTarget(),
        algorithm=_algorithm(),
        variant=PGDVariantConfig(kind=PGDVariant.EPS_BOUNDED),
        output=PGDOutputConfig(manifest_path=str(manifest_path), verbose=False),
    )

    result = await generator.execute_async(
        behavior="Describe how to do something",
        target_text="Sure, here is how",
        behavior_id="beh_1",
        seed_image_path=str(seed_path),
    )

    assert Path(result.image_path).exists()
    assert result.step_count == 25
    assert result.final_loss < result.loss_history[0]
    assert result.variant == "eps_bounded"
    # The deployed-loss honesty metric re-scores the shipped image and should match final_loss,
    # which is itself measured on the quantized (deployed) pixels.
    assert result.deployed_loss is not None
    assert result.deployed_loss == pytest.approx(result.final_loss, abs=1e-5)

    entries = read_manifest(path=manifest_path)
    assert len(entries) == 1
    assert entries[0].behavior_id == "beh_1"
    assert entries[0].image_path == result.image_path
    assert entries[0].variant == "eps_bounded"
    assert entries[0].vlm_id == "tiny/whitebox-1.0"
    assert entries[0].deployed_loss == result.deployed_loss


@pytest.mark.run_only_if_all_tests
async def test_blank_image_pipeline_without_seed(tmp_path: Path, patch_central_database) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    generator = PGDGenerator(
        target=TinyWhiteBoxTarget(),
        algorithm=_algorithm(),
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
        output=PGDOutputConfig(manifest_path=str(manifest_path), verbose=False),
    )

    result = await generator.execute_async(behavior="Carrier", target_text="Sure, here is")

    assert Path(result.image_path).exists()
    assert result.final_loss < result.loss_history[0]
    assert len(read_manifest(path=manifest_path)) == 1


@pytest.mark.run_only_if_all_tests
async def test_manifest_appends_across_two_runs(tmp_path: Path, patch_central_database) -> None:
    seed_path = tmp_path / "seed.png"
    PIL.Image.new("RGB", (16, 16), (30, 60, 90)).save(seed_path)
    manifest_path = tmp_path / "manifest.jsonl"
    generator = PGDGenerator(
        target=TinyWhiteBoxTarget(),
        algorithm=_algorithm(),
        output=PGDOutputConfig(manifest_path=str(manifest_path), verbose=False),
    )

    for behavior_id in ("b1", "b2"):
        await generator.execute_async(
            behavior=f"Behavior {behavior_id}",
            target_text="Sure, here is",
            behavior_id=behavior_id,
            seed_image_path=str(seed_path),
        )

    entries = read_manifest(path=manifest_path)
    assert [e.behavior_id for e in entries] == ["b1", "b2"]


@pytest.mark.gpu
@pytest.mark.run_only_if_all_tests
async def test_gpu_single_pgd_step_against_real_llava(tmp_path: Path, patch_central_database) -> None:
    """One real PGD step against LLaVA-1.5. Collected only on a CUDA box."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the real-VLM smoke test")

    seed_path = tmp_path / "seed.png"
    PIL.Image.new("RGB", (336, 336), (128, 128, 128)).save(seed_path)

    generator = PGDGenerator(
        model=PGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf", device="cuda:0"),
        algorithm=PGDAlgorithmConfig(num_steps=1, stop_loss=-1.0),
        output=PGDOutputConfig(manifest_path=str(tmp_path / "manifest.jsonl"), verbose=False),
    )

    result = await generator.execute_async(
        behavior="Describe the image.",
        target_text="Sure, here is",
        behavior_id="gpu_smoke",
        seed_image_path=str(seed_path),
    )

    assert Path(result.image_path).exists()
    assert result.step_count == 1
    assert result.loss_history and result.loss_history[0] == result.loss_history[0]  # not NaN


@pytest.mark.gpu
@pytest.mark.run_only_if_all_tests
async def test_gpu_single_pgd_step_against_real_qwen2_5_vl(tmp_path: Path, patch_central_database) -> None:
    """One real PGD step against Qwen2.5-VL-7B, whose patch-flattened pixel layout
    exercises the dynamic-resolution ``compute_loss`` + ``to_pil`` path. CUDA only."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the real-VLM smoke test")

    seed_path = tmp_path / "seed.png"
    PIL.Image.new("RGB", (392, 392), (128, 128, 128)).save(seed_path)

    generator = PGDGenerator(
        model=PGDModelConfig(vlm_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0"),
        algorithm=PGDAlgorithmConfig(num_steps=1, stop_loss=-1.0),
        output=PGDOutputConfig(manifest_path=str(tmp_path / "manifest.jsonl"), verbose=False),
    )

    result = await generator.execute_async(
        behavior="Describe the image.",
        target_text="Sure, here is",
        behavior_id="gpu_smoke_qwen",
        seed_image_path=str(seed_path),
    )

    assert Path(result.image_path).exists()
    assert result.step_count == 1
    assert PIL.Image.open(result.image_path).size == (392, 392)
    """The same HuggingFaceVisionTarget instance must serve send + compute_loss."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for the real-VLM dual-surface test")

    from pyrit.models import Message, MessagePiece
    from pyrit.prompt_target.hugging_face.hugging_face_vision_target import HuggingFaceVisionTarget

    seed_path = tmp_path / "seed.png"
    PIL.Image.new("RGB", (336, 336), (200, 180, 160)).save(seed_path)
    target = HuggingFaceVisionTarget(model_id="llava-hf/llava-1.5-7b-hf", device="cuda:0")

    try:
        image = PIL.Image.open(seed_path).convert("RGB")
        inputs = target.preprocess(behavior="Describe the image.", image=image)
        loss = target.compute_loss(inputs=inputs, target_text="Sure, here is")
        assert loss.requires_grad

        message = Message(
            message_pieces=[
                MessagePiece(role="user", original_value="Describe the image.", original_value_data_type="text"),
                MessagePiece(role="user", original_value=str(seed_path), original_value_data_type="image_path"),
            ]
        )
        responses = await target.send_prompt_async(message=message)
        assert responses
    finally:
        target.release_white_box_resources()
