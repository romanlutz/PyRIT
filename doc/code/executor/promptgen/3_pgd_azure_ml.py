# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
# ---

# %% [markdown]
# # Generating PGD Adversarial Images Using Azure Machine Learning

# %% [markdown]
# PGD [@mazeika2024harmbench] is a **white-box** image attack: given
# gradient access to an open-source vision-language model (VLM), it perturbs an image
# so that the model begins its reply to a carrier behavior with a target affirmative
# string ("Sure, here is ..."). PyRIT ports HarmBench's three baselines behind a
# single `variant` slot on `PGDGenerator`:
#
# - `eps_bounded` — perturb a real seed image within an L-infinity epsilon ball.
# - `blank_image` — start from random noise with no epsilon bound.
# - `patch` — restrict the perturbation to a random square patch of the image.
#
# Because the optimization needs a GPU and loads a full VLM, PyRIT ships it as a
# **prompt generator** that runs offline (like GCG) rather than an in-conversation
# attack. Each run appends a row to a JSONL **manifest** and caches the perturbed
# **PNG**; downstream a `VisualPromptInjection` scenario consumes that manifest to
# replay the images against a target. This notebook mirrors the
# [GCG AML notebook](../2_gcg_azure_ml.ipynb) and has three steps:
# 1. Connect to an Azure Machine Learning (AML) workspace.
# 2. Create an AML environment with the GPU Python dependencies.
# 3. Submit a job that optimizes one adversarial image per behavior.
#
# > This notebook requires a CUDA GPU compute target and AML access, so — like the
# > GCG AML notebook — it is not executed in CI; run it in an AML-connected
# > environment to populate the outputs.

# %% [markdown]
# ## Connect to Azure Machine Learning Workspace

# %% [markdown]
# The [workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace)
# is the top-level AML resource. To connect we need a subscription id, resource
# group, and workspace name, which we feed to `MLClient` from `azure.ai.ml` using the
# [default Azure CLI credential](https://learn.microsoft.com/en-us/python/api/azure-identity/azure.identity.azureclicredential).

# %%
import os

from pyrit.setup.initialization import _load_environment_files

_load_environment_files(env_files=None)

subscription_id = os.environ.get("AZURE_ML_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_ML_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_ML_WORKSPACE_NAME")
print(workspace)

# %% [markdown]
# The Azure ML SDK emits a fair amount of benign telemetry to stderr (an
# `ActivityCompleted: ... HowEnded=Failure` line for every expected `UserError`, plus
# a one-line warning for each preview class). Quiet all of it so the notebook output
# stays focused.

# %%
import logging
import warnings

logging.getLogger("azure.ai.ml").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", module=r"azure\.ai\.ml.*")

# %%
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

ml_client = MLClient(AzureCliCredential(), subscription_id, resource_group, workspace)

# %% [markdown]
# ## Create AML Environment

# %% [markdown]
# We build the environment from a
# [Dockerfile](https://github.com/microsoft/PyRIT/blob/main/pyrit/executor/promptgen/src/Dockerfile)
# that uses an NVIDIA CUDA base image with Python 3.11 and installs PyRIT with the
# `gradient` extra (torch + accelerate). The build context is the repo root so
# the Dockerfile can `COPY pyproject.toml` and `pyrit/` for the editable install.

# %%
from pathlib import Path

from azure.ai.ml.entities import BuildContext, Environment

from pyrit.common.path import HOME_PATH

env_docker_context = Environment(
    build=BuildContext(
        path=Path(HOME_PATH),
        dockerfile_path="pyrit/executor/promptgen/src/Dockerfile",
    ),
    name="pyrit-pgd",
    description="PyRIT PGD environment: CUDA 12.1 + Python 3.11 + pip install -e .[gradient]",
    tags={"Owner": os.environ.get("USER", "unknown")},
)

ml_client.environments.create_or_update(env_docker_context)

# %% [markdown]
# ## Submit the Job to AML

# %% [markdown]
# The entry point is
# [`pyrit.executor.promptgen.pgd.experiments.run`](https://github.com/microsoft/PyRIT/blob/main/pyrit/executor/promptgen/pgd/experiments/run.py),
# invoked as a module so the uploaded code snapshot takes priority over the
# Docker-installed package.
#
# The public API takes a typed `PGDConfig` (strategy: model + algorithm +
# variant + output) and a separate `PGDDataConfig` (behaviors CSV + count).
# We build both locally, serialize each into a JSON file the AML job reads as an
# input, and ship those paths through the job command. Defaults come from the
# dataclasses in `pyrit.executor.promptgen.pgd.config`.
#
# The example below runs the `eps_bounded` variant against Qwen2.5-VL-7B-Instruct for
# two AdvBench behaviors, perturbing a **benign photo** — a three-masted sailing ship —
# within an L-infinity ball of radius `epsilon` around it. It optimizes for `num_steps`
# (default 500, or until the loss drops below `stop_loss`) so the convergence curve is
# visible; set `PYRIT_PGD_NUM_STEPS`, `PYRIT_PGD_STOP_LOSS`, or `PYRIT_PGD_EPSILON`
# to override. The end-to-end test forces a tiny step count so it exercises the full
# pipeline quickly, while real image generation wants the full 500+. The benign photo is
# shipped once as a job input and shared across every behavior through the runner's
# `--seed-image` override, so the text-only AdvBench CSV needs no seed column;
# `blank_image` instead needs no seed at all and starts from random noise.
#
# A GPU instance with >= 24 GB of vRAM is recommended (e.g. `Standard_NC24ads_A100_v4`).
# The `compute` below points at `gcg-gpu-a100`, the same A100 pool the
# [GCG AML notebook](../2_gcg_azure_ml.ipynb) uses; change it to any GPU compute
# target in your workspace. If you hit out-of-memory errors, lower `algorithm.num_steps`,
# use a smaller model, or reduce `data.n_behaviors`.

# %%
import tempfile

from pyrit.executor.promptgen.pgd import (
    PGDAlgorithmConfig,
    PGDConfig,
    PGDDataConfig,
    PGDModelConfig,
    PGDOutputConfig,
    PGDVariant,
    PGDVariantConfig,
)

# The benign carrier photo PGD perturbs: a three-masted sailing ship already shipped in
# the repo. It is sent to the job once as an input mount and shared across every behavior
# through the runner's `--seed-image` override.
SEED_IMAGE_PATH = Path(HOME_PATH) / "doc" / "code" / "executor" / "assets" / "three_masted_ship_color.jpg"

config = PGDConfig(
    model=PGDModelConfig(vlm_id="Qwen/Qwen2.5-VL-7B-Instruct", device="cuda:0", dtype="float16"),
    algorithm=PGDAlgorithmConfig(
        num_steps=int(os.environ.get("PYRIT_PGD_NUM_STEPS", "500")),
        stop_loss=float(os.environ.get("PYRIT_PGD_STOP_LOSS", "0.05")),
        epsilon=float(os.environ.get("PYRIT_PGD_EPSILON", str(64 / 255))),
    ),
    variant=PGDVariantConfig(kind=PGDVariant.EPS_BOUNDED),
    output=PGDOutputConfig(result_prefix="pgd", verbose=True),
)
data_config = PGDDataConfig(
    behaviors_csv="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    n_behaviors=2,
)

# Write the configs into a tempdir so AML can mount them as separate job inputs.
config_dir = Path(tempfile.mkdtemp(prefix="pgd-aml-config-"))
config_path = config_dir / "config.json"
data_path = config_dir / "data.json"
config.to_json_file(config_path)
data_config.to_json_file(data_path)

# %%
from azure.ai.ml import Input, Output, command

job = command(
    code=Path(HOME_PATH),
    command=(
        "python -m pyrit.executor.promptgen.pgd.experiments.run"
        " --config ${{inputs.config}}"
        " --data ${{inputs.data}}"
        " --seed-image ${{inputs.seed_image}}"
        " --output-dir ${{outputs.results}}"
    ),
    inputs={
        "config": Input(type="uri_file", path=str(config_path)),
        "data": Input(type="uri_file", path=str(data_path)),
        "seed_image": Input(type="uri_file", path=str(SEED_IMAGE_PATH)),
    },
    outputs={"results": Output(type="uri_folder")},
    environment=f"{env_docker_context.name}:{env_docker_context.version}",
    environment_variables={"HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")},
    compute="gcg-gpu-a100",
    display_name="pgd_image_generation",
    description="Generate adversarial images using PGD on Qwen2.5-VL-7B-Instruct.",
    tags={"Owner": os.environ.get("USER", "unknown")},
)

# %%
returned_job = ml_client.create_or_update(job)
print(f"Job: {returned_job.name}")
print(f"Status: {returned_job.status}")
print(f"Studio URL: {returned_job.studio_url}")

# %% [markdown]
# ## Wait for the Job to Complete and Inspect the Manifest
#
# The next cell polls the job until it reaches a terminal state, then downloads the
# named `results` output. The job writes a JSONL **manifest**
# (`pgd_manifest_<timestamp>.jsonl`) plus the cached perturbed **PNGs** under
# `seed-prompt-entries/`; both land in the `results` mount because the runner points
# PyRIT's results path at `--output-dir`. Each manifest row records the behavior,
# target text, variant, final loss, and the PNG path — exactly the schema the
# Phase 1 `VisualPromptInjection` scenario reads back.

# %%
import tempfile
import time

_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested"}

last_status = None
while True:
    current_status = ml_client.jobs.get(returned_job.name).status
    if current_status != last_status:
        print(f"Job status: {current_status}", flush=True)
        last_status = current_status
    if current_status in _TERMINAL_STATES:
        break
    time.sleep(60)

assert current_status == "Completed", f"Job did not complete successfully: {current_status}"

download_dir = Path(tempfile.mkdtemp(prefix="pgd-aml-"))
ml_client.jobs.download(name=returned_job.name, download_path=str(download_dir), all=True)

# %% [markdown]
# Load the downloaded manifest, then compare the **benign seed photo** with the crafted
# **adversarial image**. Because `eps_bounded` keeps every pixel within `epsilon` of the
# seed, the two look alike; a third panel amplifies their per-pixel difference so the
# injected perturbation is visible. The manifest stores each PNG's job-side path; we
# resolve it against the downloaded artifacts by filename so the preview works regardless
# of the mount layout.

# %%
from IPython.display import display  # runpy-safe: no-ops outside a notebook kernel
from PIL import Image

from pyrit.executor.promptgen.pgd import read_manifest

manifest_files = list(download_dir.rglob("*_manifest_*.jsonl"))
if not manifest_files:
    print(f"No manifest found under {download_dir}. Files captured:")
    for p in sorted(download_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(download_dir)}")
    raise FileNotFoundError("Manifest JSONL not in downloaded artifacts")

entries = read_manifest(path=manifest_files[0])
print(f"Manifest: {manifest_files[0].name}  ({len(entries)} entries)")
for entry in entries:
    emitted = entry.target_emitted
    flag = "n/a" if emitted is None else ("emitted ✓" if emitted else "not emitted ✗")
    print(
        f"  [{entry.variant}] {entry.behavior_id!r} final_loss={entry.final_loss:.4f} "
        f"({flag}) -> {Path(entry.image_path).name}"
    )

first = entries[0]
png_matches = list(download_dir.rglob(Path(first.image_path).name))

before = Image.open(SEED_IMAGE_PATH).convert("RGB")
if png_matches:
    import matplotlib.pyplot as plt
    import numpy as np

    after = Image.open(png_matches[0]).convert("RGB")
    # The model preprocesses the photo to its own grid before perturbing it, so resize the
    # seed to the adversarial image's size to line the two up for a per-pixel difference.
    before_resized = before.resize(after.size)
    diff = np.abs(np.asarray(after, dtype=np.int16) - np.asarray(before_resized, dtype=np.int16))
    amplified = np.clip(diff * 5, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    panels = (
        (before, "Benign seed (before)"),
        (after, "Adversarial image (after)"),
        (Image.fromarray(amplified), "Perturbation (|after - before| x5)"),
    )
    for ax, (img, title) in zip(axes, panels, strict=True):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.show()
else:
    print(f"PNG {Path(first.image_path).name} not found in the downloaded artifacts.")
    display(before)  # the benign seed we attempted to perturb

# %% [markdown]
# ## Verify the Attack Worked
#
# The optimization loss is only a proxy — the real question is whether the crafted image
# actually makes the model comply. Because `verify_response` is on by default, the runner
# feeds each adversarial PNG **back through the VLM** and records the model's reply
# (`model_response`) plus whether it begins with the requested `target_text`
# (`target_emitted`). Below we print the target beside the model's actual response so you
# can see the attack land: a benign-looking image steering the model into opening its
# answer with the affirmative string it was optimized for.

# %%
for entry in entries:
    verdict = "✓ EMITTED TARGET" if entry.target_emitted else "✗ did not emit target"
    print(f"[{entry.variant}] {entry.behavior_id!r}  ({verdict})")
    print(f"  behavior : {entry.behavior_text}")
    print(f"  target   : {entry.target_text!r}")
    print(f"  response : {entry.model_response!r}")
    print()

# %% [markdown]
# ## Plot the Convergence Curves
#
# Alongside the manifest the runner writes a `pgd_loss_curves_<timestamp>.jsonl`
# sidecar — one row per behavior carrying the full per-step `loss_history` (the
# manifest itself only keeps the final loss). Plotting it shows PGD driving the
# cross-entropy against the target string down over the optimization: a run that
# early-stops converged below `stop_loss` before exhausting `num_steps`.

# %%
import json

import matplotlib.pyplot as plt

curve_files = list(download_dir.rglob("*_loss_curves_*.jsonl"))
if not curve_files:
    print(f"No loss-curves sidecar found under {download_dir}.")
else:
    curves = [json.loads(line) for line in curve_files[0].read_text(encoding="utf-8").splitlines() if line.strip()]
    fig, ax = plt.subplots(figsize=(7, 4))
    for row in curves:
        history = row["loss_history"]
        ax.plot(
            range(1, len(history) + 1),
            history,
            marker=".",
            markersize=4,
            label=f"{row['id']} (final={row['final_loss']:.3f}, {row['num_steps_run']} steps)",
        )
    ax.set_xlabel("PGD step")
    ax.set_ylabel("Cross-entropy loss vs. target string")
    ax.set_title(f"PGD convergence — {curves[0]['vlm_id']} ({curves[0]['variant']})")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.3)
    plt.show()
