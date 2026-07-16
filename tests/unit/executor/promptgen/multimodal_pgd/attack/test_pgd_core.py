# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the shared PGD loop (requires the ``multimodal_pgd`` extra).

The loop is exercised against a ``FakeWhiteBoxTarget`` whose loss is a simple convex
``||pixel_values - target||^2``, so optimization behavior is fully deterministic and
CPU-only.
"""

from __future__ import annotations

import PIL.Image
import pytest

torch = pytest.importorskip("torch", reason="multimodal_pgd extra (torch) not installed")

from pyrit.executor.promptgen.multimodal_pgd.attack.pgd_core import run_pgd  # noqa: E402
from pyrit.executor.promptgen.multimodal_pgd.attack.variants import (  # noqa: E402
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
        self.released = False

    def preprocess(self, *, behavior: str, image: PIL.Image.Image) -> WhiteBoxInputs:
        return WhiteBoxInputs(pixel_values=self._base.clone())

    def compute_loss(self, *, inputs: WhiteBoxInputs, target_text: str) -> torch.Tensor:
        return ((inputs.pixel_values - self._target) ** 2).sum()

    def to_pil(self, *, inputs: WhiteBoxInputs) -> PIL.Image.Image:
        self.last_pixel_values = inputs.pixel_values.detach().clone()
        arr = inputs.pixel_values.detach().clamp(0, 1)[0].permute(1, 2, 0).mul(255).to(torch.uint8).numpy()
        return PIL.Image.fromarray(arr)

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
    assert result.final_loss == result.loss_history[-1]
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
