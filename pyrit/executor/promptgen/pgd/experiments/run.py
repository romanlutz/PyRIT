# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Thin CLI wrapper around ``PGDGenerator.execute_async`` for AzureML jobs.

The notebook (or any user) builds a ``PGDConfig`` (strategy) and a
``PGDDataConfig`` (data) locally, serializes both with their respective
``to_json_file`` methods, ships them to Azure ML as job inputs, and the job's command
line is::

    python -m pyrit.executor.promptgen.pgd.experiments.run \\
        --config inputs/config.json \\
        --data inputs/data.json \\
        --output-dir ${{outputs.results}}

This file deserializes both configs inside the job, loads behaviors from the
configured CSV, and runs one PGD optimization per behavior against a single
generator (one loaded model), appending each result to the shared manifest.

An optional ``--seed-image inputs/seed.png`` overrides the seed for every behavior,
so a job can ship one benign photo as an input mount and attack it with many
behaviors (used by the eps_bounded before/after demo); the blank_image variant
ignores it and starts from random noise.

Heavy imports (torch via the generator) are deferred into ``_main_async`` so
``--help`` stays cheap.
"""

import argparse
import asyncio
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from pyrit.executor.promptgen.pgd.config import (
    PGDConfig,
    PGDDataConfig,
    PGDOutputConfig,
)
from pyrit.executor.promptgen.pgd.data import load_behaviors

if TYPE_CHECKING:
    from pyrit.executor.promptgen.pgd.generator import PGDResult


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run PGD from serialized PGDConfig + PGDDataConfig "
            "JSON files. Intended as the AzureML job entry point; for local development construct "
            "a PGDGenerator and call execute_async directly."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON file produced by PGDConfig.to_json_file() (strategy).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to a JSON file produced by PGDDataConfig.to_json_file() (CSV path + count).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Optional output directory. When set, the manifest and per-run logs are written under "
            "this directory by overriding config.output.result_prefix. AzureML jobs pass "
            "${{outputs.<name>}} here so results land in the named output mount."
        ),
    )
    parser.add_argument(
        "--seed-image",
        type=str,
        default=None,
        help=(
            "Optional path to a single seed image applied to every behavior, overriding any "
            "per-row seed_image_path from the CSV. Lets a job ship one benign photo as an input "
            "mount and attack it with many behaviors (the eps_bounded / patch variants); ignored "
            "by the blank_image variant, which starts from random noise."
        ),
    )
    return parser.parse_args()


def _resolve_output(*, output: PGDOutputConfig, output_dir: str | None) -> PGDOutputConfig:
    """
    Combine ``output_dir`` with the basename of the config's existing result_prefix.

    Returns:
        PGDOutputConfig: The original config when ``output_dir`` is ``None``,
        otherwise a copy whose result_prefix and manifest_path are rooted under it.
    """
    if output_dir is None:
        return output
    base = Path(output.result_prefix).name or "pgd"
    resolved_prefix = str(Path(output_dir) / base)
    resolved_manifest = str(Path(output_dir) / Path(output.manifest_path).name) if output.manifest_path else ""
    return replace(output, result_prefix=resolved_prefix, manifest_path=resolved_manifest)


def _loss_curves_path(*, manifest_path: str) -> str:
    """
    Derive the loss-curves JSONL path that sits next to the manifest.

    Returns:
        str: ``manifest_path`` with ``_manifest_`` swapped for ``_loss_curves_`` when that
        marker is present, otherwise ``<manifest-stem>_loss_curves.jsonl`` in the same dir.
    """
    manifest = Path(manifest_path)
    if "_manifest_" in manifest.name:
        curves_name = manifest.name.replace("_manifest_", "_loss_curves_", 1)
    else:
        curves_name = f"{manifest.stem}_loss_curves.jsonl"
    return str(manifest.with_name(curves_name))


def _write_loss_curves(*, results: "list[PGDResult]", manifest_path: str) -> str:
    """
    Persist each behavior's full loss trajectory as JSONL for offline convergence plots.

    The manifest only records the final loss and step count; this sidecar keeps the whole
    per-step ``loss_history`` so the AzureML notebook can render convergence curves after
    downloading the job outputs.

    Returns:
        str: The path the curves were written to.
    """
    curves_path = _loss_curves_path(manifest_path=manifest_path)
    Path(curves_path).parent.mkdir(parents=True, exist_ok=True)
    with open(curves_path, "w", encoding="utf-8") as handle:
        for result in results:
            row = {
                "id": result.manifest_entry.id if result.manifest_entry is not None else "",
                "vlm_id": result.vlm_id,
                "variant": result.variant,
                "num_steps_run": result.step_count,
                "final_loss": result.final_loss,
                "succeeded": result.succeeded,
                "target_emitted": result.target_emitted,
                "loss_history": result.loss_history,
            }
            handle.write(json.dumps(row))
            handle.write("\n")
    return curves_path


async def _main_async(
    config_path: str, data_path: str, output_dir: str | None = None, seed_image: str | None = None
) -> None:
    from pyrit.executor.promptgen.pgd.generator import PGDGenerator
    from pyrit.memory import CentralMemory
    from pyrit.prompt_target.hugging_face.hugging_face_vision_target import HuggingFaceVisionTarget
    from pyrit.setup import IN_MEMORY, initialize_pyrit_async

    # initialize_pyrit_async loads the environment files (.env / .env.local) up front,
    # so env vars such as HUGGINGFACE_TOKEN are available to the target below without a
    # separate load step. It also stands up CentralMemory, which the generator uses to
    # cache perturbed PNGs through its seed-prompt cache, so it must run before any
    # optimization. Route that cache at the output directory (when provided) so the PNGs
    # land in the AzureML output mount next to the manifest instead of an ephemeral
    # compute-local path.
    await initialize_pyrit_async(memory_db_type=IN_MEMORY, load_defaults=False)
    if output_dir is not None:
        CentralMemory.get_memory_instance().results_path = output_dir

    config = PGDConfig.from_json_file(config_path)
    data = PGDDataConfig.from_json_file(data_path)

    output = _resolve_output(output=config.output, output_dir=output_dir)
    if not output.manifest_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        prefix = output.result_prefix or "pgd"
        output = replace(output, manifest_path=f"{prefix}_manifest_{timestamp}.jsonl")

    # Load the VLM once and share it across every behavior. Passing a pre-built
    # ``target`` (rather than the ``model`` config) makes the generator treat the
    # model as externally owned, so it is loaded a single time here instead of being
    # reloaded and released on each ``execute_async`` — matching the GCG runner's
    # single-load batch pattern. We own the target, so we release it at the end.
    target = HuggingFaceVisionTarget(
        model_id=config.model.vlm_id,
        device=config.model.device,
        dtype=config.model.dtype,
        hf_token=config.hf_token,
    )
    try:
        generator = PGDGenerator(
            target=target,
            algorithm=config.algorithm,
            variant=config.variant,
            output=output,
        )
        behaviors = load_behaviors(data=data)
        results: list[PGDResult] = []
        for row in behaviors:
            result = await generator.execute_async(
                behavior=row.behavior,
                target_text=row.target_text,
                seed_image_path=seed_image or row.seed_image_path,
                behavior_id=row.behavior_id,
            )
            results.append(result)
            emitted = result.target_emitted
            emitted_flag = "n/a" if emitted is None else ("YES" if emitted else "no")
            print(
                f"[{result.variant}] {row.behavior_id}: "
                f"final_loss={result.final_loss:.4f} steps={result.step_count} "
                f"target_emitted={emitted_flag}"
            )
        # Only persist the curves sidecar when writing to an output mount (the AzureML
        # batch path). Local direct-generator usage reads loss_history off the returned
        # result, so there is no need to litter the working directory.
        if output_dir is not None:
            curves_path = _write_loss_curves(results=results, manifest_path=output.manifest_path)
            print(f"Wrote loss curves for {len(results)} behavior(s): {curves_path}")
    finally:
        target.release_white_box_resources()


if __name__ == "__main__":
    args = _parse_arguments()
    asyncio.run(_main_async(args.config, args.data, args.output_dir, args.seed_image))
