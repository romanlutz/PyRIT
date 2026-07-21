# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the PGD variant strategies (requires the ``pgd`` extra)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="gradient extra (torch) not installed")

from pyrit.executor.promptgen.pgd.attack.variants import (  # noqa: E402
    BlankImageVariant,
    EpsBoundedVariant,
    PatchVariant,
    make_variant_strategy,
)
from pyrit.executor.promptgen.pgd.config import (  # noqa: E402
    PGDVariant,
    PGDVariantConfig,
)


def _base_4d() -> torch.Tensor:
    return torch.zeros(1, 3, 4, 4)


def test_eps_bounded_initial_is_copy_of_base() -> None:
    variant = EpsBoundedVariant()
    base = _base_4d()
    initial = variant.initial_pixel_values(base=base)
    assert torch.equal(initial, base)
    assert initial is not base


def test_eps_bounded_projects_within_epsilon() -> None:
    variant = EpsBoundedVariant()
    base = _base_4d()
    pv = base.clone().requires_grad_(True)
    grad = torch.ones_like(base)
    stepped = variant.project_step(pixel_values=pv, base=base, grad=grad, step_size=0.5, epsilon=0.1)
    assert (stepped - base).abs().max().item() <= 0.1 + 1e-6


def test_blank_image_initial_uses_rng_and_is_reproducible() -> None:
    variant = BlankImageVariant()
    base = _base_4d()
    first = variant.initial_pixel_values(base=base, rng=torch.Generator().manual_seed(0))
    second = variant.initial_pixel_values(base=base, rng=torch.Generator().manual_seed(0))
    assert first.shape == base.shape
    assert torch.equal(first, second)
    assert not torch.equal(first, base)


def test_blank_image_project_is_unbounded() -> None:
    variant = BlankImageVariant()
    base = _base_4d()
    pv = torch.zeros_like(base)
    grad = torch.ones_like(base)
    stepped = variant.project_step(pixel_values=pv, base=base, grad=grad, step_size=1.0, epsilon=0.1)
    assert stepped.abs().max().item() == pytest.approx(1.0)


def test_patch_variant_only_perturbs_patch() -> None:
    variant = PatchVariant(patch_fraction=0.5)
    base = _base_4d()
    variant.initial_pixel_values(base=base, rng=torch.Generator().manual_seed(0))
    pv = base.clone()
    grad = torch.ones_like(base)
    stepped = variant.project_step(pixel_values=pv, base=base, grad=grad, step_size=1.0, epsilon=1.0)

    changed = stepped != base
    assert changed.any()
    # Exactly the masked square (side = 0.5 * 4 = 2) changed across all channels.
    assert int(changed[0, 0].sum().item()) == 4


def test_patch_variant_requires_4d_tensor() -> None:
    variant = PatchVariant(patch_fraction=0.2)
    with pytest.raises(ValueError, match="4D"):
        variant.initial_pixel_values(base=torch.zeros(3, 4, 4))


def test_make_variant_strategy_dispatch() -> None:
    assert isinstance(
        make_variant_strategy(config=PGDVariantConfig(kind=PGDVariant.EPS_BOUNDED)),
        EpsBoundedVariant,
    )
    assert isinstance(
        make_variant_strategy(config=PGDVariantConfig(kind=PGDVariant.BLANK_IMAGE)),
        BlankImageVariant,
    )
    assert isinstance(
        make_variant_strategy(config=PGDVariantConfig(kind=PGDVariant.PATCH)),
        PatchVariant,
    )
