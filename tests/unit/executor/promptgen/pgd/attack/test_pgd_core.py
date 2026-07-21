# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the shared PGD loop (requires the ``pgd`` extra).

The loop is exercised against a ``FakeWhiteBoxTarget`` whose loss is a simple convex
``||pixel_values - target||^2``, so optimization behavior is fully deterministic and
CPU-only.
"""

from __future__ import annotations

import PIL.Image
import pytest

torch = pytest.importorskip("torch", reason="gradient extra (torch) not installed")

from pyrit.executor.promptgen.pgd.attack.pgd_core import run_pgd  # noqa: E402
from pyrit.executor.promptgen.pgd.attack.variants import (  # noqa: E402
    BlankImageVariant,
    EpsBoundedVariant,
    PatchVariant,
)
from pyrit.prompt_target.common.white_box_target import WhiteBoxInputs  # noqa: E402


class FakeWhiteBoxTarget:
    """Minimal ``WhiteBoxTarget`` whose loss is ``||pixel_values - target||^2``."""

    vlm_id = "fake/vlm"
    device = "cpu"

    def __init__(self, *, base: torch.Tensor, target: torch.Tensor) -> None:
        self._base = base
        self._target = target
        self.last_pixel_values: torch.Tensor | None = None
        self.quantize_calls = 0
        self.released = False

    def preprocess(self, *, behavior: str, image: PIL.Image.Image) -> WhiteBoxInputs:
        return WhiteBoxInputs(pixel_values=self._base.clone())

    def compute_loss(self, *, inputs: WhiteBoxInputs, target_text: str) -> torch.Tensor:
        return ((inputs.pixel_values - self._target) ** 2).sum()

    def to_pil(self, *, inputs: WhiteBoxInputs) -> PIL.Image.Image:
        self.last_pixel_values = inputs.pixel_values.detach().clone()
        arr = inputs.pixel_values.detach().clamp(0, 1)[0].permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
        return PIL.Image.fromarray(arr)

    def clamp_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        return inputs.pixel_values

    def quantize_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        self.quantize_calls += 1
        return inputs.pixel_values

    def release_white_box_resources(self) -> None:
        self.released = True


def _seed_image() -> PIL.Image.Image:
    return PIL.Image.new("RGB", (4, 4), (0, 0, 0))


def _run(*, target: FakeWhiteBoxTarget, variant, **overrides):
    params = {
        "target": target,
        "variant": variant,
        "behavior": "carrier",
        "target_text": "Sure, here is",
        "seed_image": _seed_image(),
        "num_steps": 5,
        "step_size": 0.05,
        "epsilon": 1.0,
        "stop_loss": -1.0,
    }
    params.update(overrides)
    return run_pgd(**params)


def test_run_pgd_decreases_loss_monotonically() -> None:
    base = torch.zeros(1, 3, 4, 4)
    target = FakeWhiteBoxTarget(base=base, target=torch.full_like(base, 0.5))
    result = _run(target=target, variant=EpsBoundedVariant())

    assert result.step_count == 5
    assert len(result.loss_history) == 5
    # final_loss re-scores the image after the last step, so it is at least as low as the
    # last logged (pre-step) loss in the history.
    assert result.final_loss <= result.loss_history[-1]
    assert not result.succeeded
    assert all(b < a for a, b in zip(result.loss_history, result.loss_history[1:], strict=False))


def test_run_pgd_respects_epsilon_bound() -> None:
    base = torch.zeros(1, 3, 4, 4)
    target = FakeWhiteBoxTarget(base=base, target=torch.full_like(base, 0.5))
    _run(target=target, variant=EpsBoundedVariant(), epsilon=0.12, num_steps=10)

    assert target.last_pixel_values is not None
    assert (target.last_pixel_values - base).abs().max().item() <= 0.12 + 1e-6


def test_run_pgd_blank_image_is_unbounded_and_reproducible() -> None:
    base = torch.zeros(1, 3, 4, 4)
    goal = torch.full_like(base, 3.0)

    first = _run(
        target=FakeWhiteBoxTarget(base=base, target=goal),
        variant=BlankImageVariant(),
        step_size=0.2,
        num_steps=8,
        rng=torch.Generator().manual_seed(7),
    )
    second = _run(
        target=FakeWhiteBoxTarget(base=base, target=goal),
        variant=BlankImageVariant(),
        step_size=0.2,
        num_steps=8,
        rng=torch.Generator().manual_seed(7),
    )
    assert first.loss_history == second.loss_history
    assert first.final_loss < first.loss_history[0]


def test_run_pgd_patch_only_changes_patch_region() -> None:
    base = torch.zeros(1, 3, 4, 4)
    target = FakeWhiteBoxTarget(base=base, target=torch.ones_like(base))
    _run(
        target=target,
        variant=PatchVariant(patch_fraction=0.5),
        step_size=0.3,
        num_steps=1,
        rng=torch.Generator().manual_seed(0),
    )

    changed = target.last_pixel_values != base
    assert int(changed[0, 0].sum().item()) == 4  # 2x2 patch (0.5 * 4)


def test_run_pgd_early_stops_when_loss_below_threshold() -> None:
    base = torch.zeros(1, 3, 4, 4)
    target = FakeWhiteBoxTarget(base=base, target=base.clone())
    result = _run(target=target, variant=EpsBoundedVariant(), stop_loss=0.05)

    assert result.succeeded
    assert result.step_count == 1
    assert result.final_loss == pytest.approx(0.0)


def test_run_pgd_raises_when_loss_has_no_gradient() -> None:
    class DetachedLossTarget(FakeWhiteBoxTarget):
        def compute_loss(self, *, inputs: WhiteBoxInputs, target_text: str) -> torch.Tensor:
            return torch.tensor(1.0, requires_grad=True)

    base = torch.zeros(1, 3, 4, 4)
    target = DetachedLossTarget(base=base, target=base.clone())
    with pytest.raises(RuntimeError, match="no gradient"):
        _run(target=target, variant=EpsBoundedVariant())


class _ClampingTarget(FakeWhiteBoxTarget):
    """Records clamp calls and caps ``pixel_values`` at a displayable ceiling."""

    def __init__(self, *, base: torch.Tensor, target: torch.Tensor, clamp_max: float) -> None:
        super().__init__(base=base, target=target)
        self._clamp_max = clamp_max
        self.clamp_calls = 0

    def clamp_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        self.clamp_calls += 1
        return inputs.pixel_values.detach().clamp(0.0, self._clamp_max)


def test_run_pgd_applies_clamp_on_initial_and_each_step() -> None:
    base = torch.zeros(1, 3, 4, 4)
    target = _ClampingTarget(base=base, target=torch.full_like(base, 5.0), clamp_max=0.3)
    _run(
        target=target,
        variant=BlankImageVariant(),
        step_size=0.5,
        num_steps=4,
        rng=torch.Generator().manual_seed(1),
    )

    # One projection for the initial tensor plus one after each of the 4 steps.
    assert target.clamp_calls == 5
    assert target.last_pixel_values is not None
    assert target.last_pixel_values.max().item() <= 0.3 + 1e-6


class _QuantizingTarget(FakeWhiteBoxTarget):
    """Straight-through target that rounds ``pixel_values`` to integers before scoring."""

    def quantize_to_displayable(self, *, inputs: WhiteBoxInputs) -> torch.Tensor:
        self.quantize_calls += 1
        pixel_values = inputs.pixel_values
        return pixel_values + (pixel_values.round() - pixel_values).detach()


def test_run_pgd_quantizes_before_scoring_each_step() -> None:
    base = torch.zeros(1, 3, 4, 4)
    target = FakeWhiteBoxTarget(base=base, target=torch.full_like(base, 0.5))
    _run(target=target, variant=EpsBoundedVariant(), num_steps=3)

    # quantize_to_displayable runs once per step plus a final deployed-loss re-score.
    assert target.quantize_calls == 4


def test_run_pgd_computes_loss_on_quantized_pixels() -> None:
    # Raw pixels sit at 0.4 (loss 0.16 * 48 vs a zero target); the deployed pixels round to
    # 0.0, so scoring on the quantized image drives the recorded loss to 0 instead.
    base = torch.full((1, 3, 4, 4), 0.4)
    target = _QuantizingTarget(base=base, target=torch.zeros_like(base))
    result = _run(target=target, variant=EpsBoundedVariant(), num_steps=1, step_size=0.01)

    # One quantize in the single step plus the final deployed-loss re-score.
    assert target.quantize_calls == 2
    assert result.final_loss == pytest.approx(0.0)
