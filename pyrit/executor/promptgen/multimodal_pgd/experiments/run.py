# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Thin CLI wrapper around ``MultiModalPGDGenerator.execute_async`` for AzureML jobs.

The notebook (or any user) builds a ``MultiModalPGDConfig`` (strategy) and a
``MultiModalPGDDataConfig`` (data) locally, serializes both with their respective
``to_json_file`` methods, ships them to Azure ML as job inputs, and the job's command
line is::

    python -m pyrit.executor.promptgen.multimodal_pgd.experiments.run \\
        --config inputs/config.json \\
        --data inputs/data.json \\
        --output-dir ${{outputs.results}}

This file deserializes both configs inside the job, loads behaviors from the
configured CSV, and runs one PGD optimization per behavior against a single
generator (one loaded model), appending each result to the shared manifest.

Heavy imports (torch via the generator) are deferred into ``_main_async`` so
``--help`` stays cheap.
"""

import argparse
import asyncio
import time
from dataclasses import replace
from pathlib import Path

from pyrit.executor.promptgen.multimodal_pgd.config import (
    MultiModalPGDConfig,
    MultiModalPGDDataConfig,
    MultiModalPGDOutputConfig,
)
from pyrit.executor.promptgen.multimodal_pgd.data import load_behaviors


def _parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Multimodal PGD from serialized MultiModalPGDConfig + MultiModalPGDDataConfig "
            "JSON files. Intended as the AzureML job entry point; for local development construct "
            "a MultiModalPGDGenerator and call execute_async directly."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a JSON file produced by MultiModalPGDConfig.to_json_file() (strategy).",
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to a JSON file produced by MultiModalPGDDataConfig.to_json_file() (CSV path + count).",
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
    return parser.parse_args()


def _resolve_output(*, output: MultiModalPGDOutputConfig, output_dir: str | None) -> MultiModalPGDOutputConfig:
    """
    Combine ``output_dir`` with the basename of the config's existing result_prefix.

    Returns:
        MultiModalPGDOutputConfig: The original config when ``output_dir`` is ``None``,
        otherwise a copy whose result_prefix and manifest_path are rooted under it.
    """
    if output_dir is None:
        return output
    base = Path(output.result_prefix).name or "multimodal_pgd"
    resolved_prefix = str(Path(output_dir) / base)
    resolved_manifest = str(Path(output_dir) / Path(output.manifest_path).name) if output.manifest_path else ""
    return replace(output, result_prefix=resolved_prefix, manifest_path=resolved_manifest)


async def _main_async(config_path: str, data_path: str, output_dir: str | None = None) -> None:
    from pyrit.executor.promptgen.multimodal_pgd.generator import MultiModalPGDGenerator
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

    config = MultiModalPGDConfig.from_json_file(config_path)
    data = MultiModalPGDDataConfig.from_json_file(data_path)

    output = _resolve_output(output=config.output, output_dir=output_dir)
    if not output.manifest_path:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        prefix = output.result_prefix or "multimodal_pgd"
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
        generator = MultiModalPGDGenerator(
            target=target,
            algorithm=config.algorithm,
            variant=config.variant,
            output=output,
        )
        behaviors = load_behaviors(data=data)
        for row in behaviors:
            await generator.execute_async(
                behavior=row.behavior,
                target_text=row.target_text,
                seed_image_path=row.seed_image_path,
                behavior_id=row.behavior_id,
            )
    finally:
        target.release_white_box_resources()


if __name__ == "__main__":
    args = _parse_arguments()
    asyncio.run(_main_async(args.config, args.data, args.output_dir))
