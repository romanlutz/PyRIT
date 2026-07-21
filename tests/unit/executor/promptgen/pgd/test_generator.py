# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for :class:`PGDGenerator` lifecycle, identity, and validation.

The heavy ``run_pgd`` loop and the image cache are patched so these tests exercise the
strategy plumbing (target ownership, manifest wiring, teardown) without a real VLM.
"""

from __future__ import annotations

import types
from unittest.mock import AsyncMock, MagicMock, patch

import PIL.Image
import pytest

torch = pytest.importorskip("torch", reason="gradient extra (torch) not installed")

generator_mod = pytest.importorskip(
    "pyrit.executor.promptgen.pgd.generator",
    reason="pgd optional dependencies (torch) not installed",
)
PGDGenerator = generator_mod.PGDGenerator
PGDContext = generator_mod.PGDContext

from pyrit.executor.promptgen.pgd.config import (  # noqa: E402
    PGDAlgorithmConfig,
    PGDModelConfig,
    PGDVariant,
    PGDVariantConfig,
)
from pyrit.prompt_target.common.white_box_target import WhiteBoxInputs  # noqa: E402


class FakeWhiteBoxTarget:
    """A caller-owned white-box target stub that records resource release.

    ``run_pgd`` is patched out in these tests, so ``preprocess`` / ``compute_loss`` are
    exercised only by the best-effort deployed-loss recomputation and return trivial
    tensors rather than raising.
    """

    vlm_id = "fake/vlm-1.0"
    device = "cpu"

    def __init__(self) -> None:
        self.released = False

    def preprocess(self, *, behavior, image):
        import torch

        return WhiteBoxInputs(pixel_values=torch.zeros(1, 3, 2, 2))

    def compute_loss(self, *, inputs, target_text):
        import torch

        return torch.tensor(0.25)

    def to_pil(self, *, inputs):  # pragma: no cover
        raise AssertionError("run_pgd should be patched in these tests")

    def clamp_to_displayable(self, *, inputs):
        return inputs.pixel_values

    def quantize_to_displayable(self, *, inputs):
        return inputs.pixel_values

    def release_white_box_resources(self) -> None:
        self.released = True


class FakeVerifyingTarget(FakeWhiteBoxTarget):
    """A white-box target that also supports response generation (verification)."""

    def __init__(self, *, response: str) -> None:
        super().__init__()
        self._response = response
        self.generate_calls: list = []

    def generate_response(self, *, behavior, image):
        self.generate_calls.append((behavior, image))
        return self._response


def _fake_core_result() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        image=PIL.Image.new("RGB", (4, 4), (10, 20, 30)),
        loss_history=[1.0, 0.4],
        final_loss=0.4,
        step_count=2,
        succeeded=True,
    )


def _patched_perform_dependencies():
    """Patch the image cache + manifest writer used inside ``_perform_async``."""
    return (
        patch.object(generator_mod, "fetch_and_cache_image_async", new=AsyncMock(return_value="/cache/pgd.png")),
        patch.object(generator_mod, "append_manifest_entry", new=MagicMock()),
        patch.object(generator_mod, "run_pgd", new=MagicMock(return_value=_fake_core_result())),
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_requires_exactly_one_of_target_or_model() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        PGDGenerator()
    with pytest.raises(ValueError, match="exactly one"):
        PGDGenerator(target=FakeWhiteBoxTarget(), model=PGDModelConfig(vlm_id="a/b"))


def test_init_with_target_does_not_own_it() -> None:
    gen = PGDGenerator(target=FakeWhiteBoxTarget())
    assert gen._owns_target is False


def test_init_with_model_owns_target() -> None:
    gen = PGDGenerator(model=PGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf"))
    assert gen._owns_target is True


# ---------------------------------------------------------------------------
# identifier / helpers
# ---------------------------------------------------------------------------


def test_build_identifier_exposes_hyperparameters() -> None:
    gen = PGDGenerator(
        model=PGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf"),
        algorithm=PGDAlgorithmConfig(num_steps=17, step_size=0.1, epsilon=0.2, stop_loss=0.01),
        variant=PGDVariantConfig(kind=PGDVariant.PATCH, patch_fraction=0.3),
    )
    ident = gen._build_identifier()
    assert ident.params["vlm_id"] == "llava-hf/llava-1.5-7b-hf"
    assert ident.params["variant"] == "patch"
    assert ident.params["patch_fraction"] == 0.3
    assert ident.params["num_steps"] == 17


def test_slugify_replaces_path_and_punctuation() -> None:
    assert PGDGenerator._slugify("llava-hf/llava-1.5-7b-hf") == "llava_hf_llava_1_5_7b_hf"


def test_resolve_manifest_path_uses_explicit_value() -> None:
    from pyrit.executor.promptgen.pgd.config import PGDOutputConfig

    gen = PGDGenerator(
        target=FakeWhiteBoxTarget(),
        output=PGDOutputConfig(manifest_path="/tmp/m.jsonl"),
    )
    assert gen._resolve_manifest_path() == "/tmp/m.jsonl"


# ---------------------------------------------------------------------------
# _validate_context
# ---------------------------------------------------------------------------


def test_validate_context_requires_behavior() -> None:
    gen = PGDGenerator(target=FakeWhiteBoxTarget())
    with pytest.raises(ValueError, match="behavior"):
        gen._validate_context(context=PGDContext(behavior="", target_text="t", seed_image_path="s.png"))


def test_validate_context_requires_target_text() -> None:
    gen = PGDGenerator(target=FakeWhiteBoxTarget())
    with pytest.raises(ValueError, match="target_text"):
        gen._validate_context(context=PGDContext(behavior="b", target_text="", seed_image_path="s.png"))


def test_validate_context_requires_seed_image_for_bounded_variant() -> None:
    gen = PGDGenerator(target=FakeWhiteBoxTarget())
    with pytest.raises(ValueError, match="seed_image_path"):
        gen._validate_context(context=PGDContext(behavior="b", target_text="t", seed_image_path=""))


def test_validate_context_allows_missing_seed_for_blank_variant() -> None:
    gen = PGDGenerator(
        target=FakeWhiteBoxTarget(),
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    gen._validate_context(context=PGDContext(behavior="b", target_text="t", seed_image_path=""))


# ---------------------------------------------------------------------------
# execute_async lifecycle
# ---------------------------------------------------------------------------


async def test_execute_async_does_not_release_caller_target() -> None:
    target = FakeWhiteBoxTarget()
    gen = PGDGenerator(
        target=target,
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    cache_patch, manifest_patch, run_patch = _patched_perform_dependencies()
    with cache_patch, manifest_patch as mock_append, run_patch:
        result = await gen.execute_async(behavior="carrier", target_text="Sure, here is", behavior_id="beh_1")

    assert result.image_path == "/cache/pgd.png"
    assert result.final_loss == 0.4
    assert result.step_count == 2
    assert result.succeeded is True
    assert result.vlm_id == "fake/vlm-1.0"
    assert result.variant == "blank_image"
    assert result.deployed_loss == 0.25
    assert result.manifest_entry is not None
    assert result.manifest_entry.behavior_id == "beh_1"
    assert result.manifest_entry.deployed_loss == 0.25
    mock_append.assert_called_once()
    assert target.released is False


async def test_execute_async_releases_owned_target() -> None:
    built_target = FakeWhiteBoxTarget()
    gen = PGDGenerator(
        model=PGDModelConfig(vlm_id="fake/vlm-1.0"),
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    cache_patch, manifest_patch, run_patch = _patched_perform_dependencies()
    with cache_patch, manifest_patch, run_patch, patch.object(gen, "_build_target", return_value=built_target):
        await gen.execute_async(behavior="carrier", target_text="Sure, here is")

    assert built_target.released is True


async def test_execute_async_releases_owned_target_on_failure() -> None:
    built_target = FakeWhiteBoxTarget()
    gen = PGDGenerator(
        model=PGDModelConfig(vlm_id="fake/vlm-1.0"),
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    with (
        patch.object(generator_mod, "run_pgd", side_effect=RuntimeError("boom")),
        patch.object(gen, "_build_target", return_value=built_target),
    ):
        with pytest.raises(Exception, match="boom"):
            await gen.execute_async(behavior="carrier", target_text="Sure, here is")

    assert built_target.released is True


# ---------------------------------------------------------------------------
# response verification
# ---------------------------------------------------------------------------


async def test_execute_async_verifies_response_when_supported() -> None:
    target = FakeVerifyingTarget(response="Sure, here is the plan")
    gen = PGDGenerator(
        target=target,
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    cache_patch, manifest_patch, run_patch = _patched_perform_dependencies()
    with cache_patch, manifest_patch, run_patch:
        result = await gen.execute_async(behavior="carrier", target_text="Sure, here is the plan", behavior_id="beh_1")

    assert result.model_response == "Sure, here is the plan"
    assert result.target_emitted is True
    assert result.manifest_entry is not None
    assert result.manifest_entry.model_response == "Sure, here is the plan"
    assert result.manifest_entry.target_emitted is True
    assert len(target.generate_calls) == 1


async def test_execute_async_verification_records_miss() -> None:
    target = FakeVerifyingTarget(response="I cannot help with that.")
    gen = PGDGenerator(
        target=target,
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    cache_patch, manifest_patch, run_patch = _patched_perform_dependencies()
    with cache_patch, manifest_patch, run_patch:
        result = await gen.execute_async(behavior="carrier", target_text="Sure, here is the plan")

    assert result.model_response == "I cannot help with that."
    assert result.target_emitted is False


async def test_execute_async_skips_verification_when_target_unsupported() -> None:
    target = FakeWhiteBoxTarget()  # no generate_response
    gen = PGDGenerator(
        target=target,
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    cache_patch, manifest_patch, run_patch = _patched_perform_dependencies()
    with cache_patch, manifest_patch, run_patch:
        result = await gen.execute_async(behavior="carrier", target_text="Sure, here is", behavior_id="b")

    assert result.model_response is None
    assert result.target_emitted is None


async def test_execute_async_verification_can_be_disabled() -> None:
    from pyrit.executor.promptgen.pgd.config import PGDOutputConfig

    target = FakeVerifyingTarget(response="Sure, here is the plan")
    gen = PGDGenerator(
        target=target,
        variant=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
        output=PGDOutputConfig(verify_response=False),
    )
    cache_patch, manifest_patch, run_patch = _patched_perform_dependencies()
    with cache_patch, manifest_patch, run_patch:
        result = await gen.execute_async(behavior="carrier", target_text="Sure, here is the plan")

    assert result.model_response is None
    assert result.target_emitted is None
    assert target.generate_calls == []
