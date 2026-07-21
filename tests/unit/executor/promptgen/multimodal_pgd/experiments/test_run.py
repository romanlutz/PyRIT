# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the Multimodal PGD AzureML entry point (``experiments.run``).

These are pure wiring tests: the vision target and generator are mocked so the tests
assert how ``_main_async`` orchestrates them. The key guarantee is that the VLM is
loaded exactly once and shared across every behavior (rather than reloaded per
behavior), matching the GCG runner's single-load batch pattern.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ``patch`` imports the real vision-target module (torch + transformers) to swap the
# class, so skip gracefully when the gradient extra is absent, like the sibling tests.
pytest.importorskip("torch", reason="gradient extra (torch) not installed")

from pyrit.executor.promptgen.multimodal_pgd.config import (  # noqa: E402
    MultiModalPGDConfig,
    MultiModalPGDDataConfig,
    MultiModalPGDModelConfig,
    MultiModalPGDOutputConfig,
    MultiModalPGDVariantConfig,
    PGDVariant,
)
from pyrit.executor.promptgen.multimodal_pgd.experiments import run as run_module  # noqa: E402

_VLM_ID = "llava-hf/llava-1.5-7b-hf"
_VISION_TARGET_PATH = "pyrit.prompt_target.hugging_face.hugging_face_vision_target.HuggingFaceVisionTarget"
_GENERATOR_PATH = "pyrit.executor.promptgen.multimodal_pgd.generator.MultiModalPGDGenerator"


def _write_configs(tmp_path: Path, *, n_behaviors: int) -> tuple[Path, Path]:
    """Write a 2-row behaviors CSV plus config/data JSON and return their paths."""
    csv_path = tmp_path / "behaviors.csv"
    csv_path.write_text("behavior,target\nfirst behavior,Sure here\nsecond behavior,Sure here\n", encoding="utf-8")

    config = MultiModalPGDConfig(
        model=MultiModalPGDModelConfig(vlm_id=_VLM_ID, device="cpu", dtype="float32"),
        variant=MultiModalPGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    )
    data = MultiModalPGDDataConfig(behaviors_csv=str(csv_path), n_behaviors=n_behaviors)

    config_path = tmp_path / "config.json"
    data_path = tmp_path / "data.json"
    config.to_json_file(config_path)
    data.to_json_file(data_path)
    return config_path, data_path


def _fake_result(*, result_id: str, loss_history: list[float]) -> SimpleNamespace:
    """Minimal stand-in for MultiModalPGDResult with the fields the curves sidecar reads."""
    return SimpleNamespace(
        manifest_entry=SimpleNamespace(id=result_id),
        vlm_id=_VLM_ID,
        variant="blank_image",
        step_count=len(loss_history),
        final_loss=loss_history[-1] if loss_history else float("nan"),
        succeeded=bool(loss_history) and loss_history[-1] <= 0.05,
        target_emitted=(bool(loss_history) and loss_history[-1] <= 0.05) or None,
        loss_history=loss_history,
    )


async def test_main_async_loads_model_once_and_releases(tmp_path: Path) -> None:
    config_path, data_path = _write_configs(tmp_path, n_behaviors=2)

    fake_target = MagicMock()
    fake_target_cls = MagicMock(return_value=fake_target)
    fake_generator = MagicMock()
    fake_generator.execute_async = AsyncMock(return_value=_fake_result(result_id="pgd_0", loss_history=[0.5, 0.1]))
    fake_generator_cls = MagicMock(return_value=fake_generator)

    with (
        patch(_VISION_TARGET_PATH, fake_target_cls),
        patch(_GENERATOR_PATH, fake_generator_cls),
        patch("pyrit.setup.initialize_pyrit_async", new=AsyncMock()),
    ):
        await run_module._main_async(str(config_path), str(data_path), output_dir=None)

    # The VLM is constructed exactly once regardless of the number of behaviors.
    fake_target_cls.assert_called_once()
    assert fake_target_cls.call_args.kwargs["model_id"] == _VLM_ID
    # The generator adopts the pre-loaded target and is NOT given the model config
    # (which would make it reload the model on every execute_async).
    gen_kwargs = fake_generator_cls.call_args.kwargs
    assert gen_kwargs["target"] is fake_target
    assert "model" not in gen_kwargs
    # One optimization per behavior; the runner-owned target is released once.
    assert fake_generator.execute_async.await_count == 2
    fake_target.release_white_box_resources.assert_called_once()


async def test_main_async_seed_image_overrides_every_behavior(tmp_path: Path) -> None:
    config_path, data_path = _write_configs(tmp_path, n_behaviors=2)

    fake_generator = MagicMock()
    fake_generator.execute_async = AsyncMock(
        side_effect=[
            _fake_result(result_id="pgd_0", loss_history=[0.5, 0.1]),
            _fake_result(result_id="pgd_1", loss_history=[0.5, 0.1]),
        ]
    )

    with (
        patch(_VISION_TARGET_PATH, MagicMock(return_value=MagicMock())),
        patch(_GENERATOR_PATH, MagicMock(return_value=fake_generator)),
        patch("pyrit.setup.initialize_pyrit_async", new=AsyncMock()),
    ):
        await run_module._main_async(str(config_path), str(data_path), output_dir=None, seed_image="/mnt/seed.png")

    # The shipped seed image is applied to every behavior, overriding any CSV seed path.
    seed_paths = [call.kwargs["seed_image_path"] for call in fake_generator.execute_async.await_args_list]
    assert seed_paths == ["/mnt/seed.png", "/mnt/seed.png"]


async def test_main_async_releases_target_even_when_a_run_fails(tmp_path: Path) -> None:
    config_path, data_path = _write_configs(tmp_path, n_behaviors=2)

    fake_target = MagicMock()
    fake_generator = MagicMock()
    fake_generator.execute_async = AsyncMock(side_effect=RuntimeError("boom"))

    with (
        patch(_VISION_TARGET_PATH, MagicMock(return_value=fake_target)),
        patch(_GENERATOR_PATH, MagicMock(return_value=fake_generator)),
        patch("pyrit.setup.initialize_pyrit_async", new=AsyncMock()),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            await run_module._main_async(str(config_path), str(data_path), output_dir=None)

    fake_target.release_white_box_resources.assert_called_once()


async def test_main_async_routes_results_path_to_output_dir(tmp_path: Path) -> None:
    config_path, data_path = _write_configs(tmp_path, n_behaviors=1)
    out_dir = tmp_path / "results"

    fake_generator = MagicMock()
    fake_generator.execute_async = AsyncMock(return_value=_fake_result(result_id="pgd_0", loss_history=[0.9, 0.5, 0.1]))
    fake_memory = MagicMock()

    with (
        patch(_VISION_TARGET_PATH, MagicMock(return_value=MagicMock())),
        patch(_GENERATOR_PATH, MagicMock(return_value=fake_generator)),
        patch("pyrit.setup.initialize_pyrit_async", new=AsyncMock()),
        patch("pyrit.memory.CentralMemory") as mock_central_memory,
    ):
        mock_central_memory.get_memory_instance.return_value = fake_memory
        await run_module._main_async(str(config_path), str(data_path), output_dir=str(out_dir))

    # PNGs land in the AzureML output mount because the seed-prompt cache is rerouted.
    assert fake_memory.results_path == str(out_dir)


def test_resolve_output_roots_prefix_and_manifest_under_dir() -> None:
    output = MultiModalPGDOutputConfig(result_prefix="pgd", manifest_path="run/pgd_manifest.jsonl")
    resolved = run_module._resolve_output(output=output, output_dir="/mnt/out")
    assert Path(resolved.result_prefix) == Path("/mnt/out/pgd")
    assert Path(resolved.manifest_path) == Path("/mnt/out/pgd_manifest.jsonl")


def test_resolve_output_none_returns_original_unchanged() -> None:
    output = MultiModalPGDOutputConfig(result_prefix="pgd")
    assert run_module._resolve_output(output=output, output_dir=None) is output


def test_loss_curves_path_swaps_manifest_marker() -> None:
    resolved = run_module._loss_curves_path(manifest_path="/mnt/out/pgd_manifest_20260101-000000.jsonl")
    assert Path(resolved) == Path("/mnt/out/pgd_loss_curves_20260101-000000.jsonl")


def test_loss_curves_path_fallback_when_no_marker() -> None:
    resolved = run_module._loss_curves_path(manifest_path="/mnt/out/custom.jsonl")
    assert Path(resolved) == Path("/mnt/out/custom_loss_curves.jsonl")


async def test_main_async_writes_loss_curves_sidecar(tmp_path: Path) -> None:
    config_path, data_path = _write_configs(tmp_path, n_behaviors=2)
    out_dir = tmp_path / "results"
    histories = [[0.9, 0.4, 0.05], [1.2, 0.7, 0.6, 0.55]]

    fake_generator = MagicMock()
    fake_generator.execute_async = AsyncMock(
        side_effect=[
            _fake_result(result_id="pgd_0", loss_history=histories[0]),
            _fake_result(result_id="pgd_1", loss_history=histories[1]),
        ]
    )

    with (
        patch(_VISION_TARGET_PATH, MagicMock(return_value=MagicMock())),
        patch(_GENERATOR_PATH, MagicMock(return_value=fake_generator)),
        patch("pyrit.setup.initialize_pyrit_async", new=AsyncMock()),
        patch("pyrit.memory.CentralMemory"),
    ):
        await run_module._main_async(str(config_path), str(data_path), output_dir=str(out_dir))

    curves_files = list(out_dir.glob("*_loss_curves_*.jsonl"))
    assert len(curves_files) == 1
    rows = [json.loads(line) for line in curves_files[0].read_text(encoding="utf-8").splitlines()]
    # One JSONL row per behavior, preserving order and the full per-step trajectory.
    assert [row["id"] for row in rows] == ["pgd_0", "pgd_1"]
    assert [row["loss_history"] for row in rows] == histories
    assert [row["num_steps_run"] for row in rows] == [3, 4]
    assert rows[0]["succeeded"] is True
    assert rows[1]["succeeded"] is False


async def test_main_async_skips_loss_curves_without_output_dir(tmp_path: Path) -> None:
    config_path, data_path = _write_configs(tmp_path, n_behaviors=1)

    fake_generator = MagicMock()
    fake_generator.execute_async = AsyncMock(return_value=_fake_result(result_id="pgd_0", loss_history=[0.5, 0.1]))

    with (
        patch(_VISION_TARGET_PATH, MagicMock(return_value=MagicMock())),
        patch(_GENERATOR_PATH, MagicMock(return_value=fake_generator)),
        patch("pyrit.setup.initialize_pyrit_async", new=AsyncMock()),
    ):
        await run_module._main_async(str(config_path), str(data_path), output_dir=None)

    # No output mount -> no sidecar written anywhere under the tmp tree.
    assert list(tmp_path.rglob("*_loss_curves*.jsonl")) == []
