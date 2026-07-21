# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the PGD configuration objects (torch-free)."""

from __future__ import annotations

import pytest

from pyrit.executor.promptgen.pgd.config import (
    PGDAlgorithmConfig,
    PGDConfig,
    PGDDataConfig,
    PGDModelConfig,
    PGDVariant,
    PGDVariantConfig,
)


def test_model_config_requires_vlm_id() -> None:
    with pytest.raises(ValueError, match="vlm_id"):
        PGDModelConfig(vlm_id="")


def test_algorithm_defaults() -> None:
    algo = PGDAlgorithmConfig()
    assert algo.num_steps == 500
    assert algo.stop_loss == 0.05
    assert algo.epsilon == pytest.approx(16 / 255)


@pytest.mark.parametrize("field,value", [("num_steps", 0), ("step_size", 0.0), ("epsilon", -1.0)])
def test_algorithm_validation_rejects_non_positive(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        PGDAlgorithmConfig(**{field: value})


def test_variant_config_defaults_to_eps_bounded() -> None:
    assert PGDVariantConfig().kind is PGDVariant.EPS_BOUNDED


def test_variant_config_coerces_string_kind() -> None:
    variant = PGDVariantConfig(kind="patch")
    assert variant.kind is PGDVariant.PATCH


@pytest.mark.parametrize("patch_fraction", [0.0, -0.1, 1.5])
def test_variant_config_rejects_out_of_range_patch_fraction(patch_fraction: float) -> None:
    with pytest.raises(ValueError, match="patch_fraction"):
        PGDVariantConfig(patch_fraction=patch_fraction)


def test_data_config_rejects_negative_count() -> None:
    with pytest.raises(ValueError, match="n_behaviors"):
        PGDDataConfig(n_behaviors=-1)


def test_top_level_config_json_round_trip(tmp_path) -> None:
    config = PGDConfig(
        model=PGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf", dtype="bfloat16"),
        algorithm=PGDAlgorithmConfig(num_steps=17, stop_loss=0.1),
        variant=PGDVariantConfig(kind=PGDVariant.PATCH, patch_fraction=0.3),
    )
    path = tmp_path / "config.json"
    config.to_json_file(path)
    restored = PGDConfig.from_json_file(path)

    assert restored.model.vlm_id == "llava-hf/llava-1.5-7b-hf"
    assert restored.model.dtype == "bfloat16"
    assert restored.algorithm.num_steps == 17
    assert restored.algorithm.stop_loss == 0.1
    assert restored.variant.kind is PGDVariant.PATCH
    assert restored.variant.patch_fraction == 0.3


def test_config_json_serializes_enum_as_value() -> None:
    config = PGDConfig(model=PGDModelConfig(vlm_id="x"))
    payload = config.to_json()
    assert '"eps_bounded"' in payload
    assert "PGDVariant" not in payload


def test_config_from_json_requires_model() -> None:
    with pytest.raises(ValueError, match="must contain a 'model'"):
        PGDConfig.from_json('{"algorithm": {}}')


def test_config_from_json_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        PGDConfig.from_json("{not json}")
