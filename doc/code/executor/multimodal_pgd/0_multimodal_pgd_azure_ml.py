# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
# ---

# %% [markdown]
# # Generating Multimodal PGD Adversarial Images Using Azure Machine Learning

# %% [markdown]
# Multimodal PGD [@mazeika2024harmbench] is a **white-box** image attack: given
# gradient access to an open-source vision-language model (VLM), it perturbs an image
# so that the model begins its reply to a carrier behavior with a target affirmative
# string ("Sure, here is ..."). PyRIT ports HarmBench's three baselines behind a
# single `variant` slot on `MultiModalPGDGenerator`:
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
# [GCG AML notebook](../gcg/1_gcg_azure_ml.ipynb) and has three steps:
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
    name="pyrit-multimodal-pgd",
    description="PyRIT Multimodal PGD environment: CUDA 12.1 + Python 3.11 + pip install -e .[gradient]",
    tags={"Owner": os.environ.get("USER", "unknown")},
)

ml_client.environments.create_or_update(env_docker_context)

# %% [markdown]
# ## Submit the Job to AML

# %% [markdown]
# The entry point is
# [`pyrit.executor.promptgen.multimodal_pgd.experiments.run`](https://github.com/microsoft/PyRIT/blob/main/pyrit/executor/promptgen/multimodal_pgd/experiments/run.py),
# invoked as a module so the uploaded code snapshot takes priority over the
# Docker-installed package.
#
# The public API takes a typed `MultiModalPGDConfig` (strategy: model + algorithm +
# variant + output) and a separate `MultiModalPGDDataConfig` (behaviors CSV + count).
# We build both locally, serialize each into a JSON file the AML job reads as an
# input, and ship those paths through the job command. Defaults come from the
# dataclasses in `pyrit.executor.promptgen.multimodal_pgd.config`.
#
# The example below runs the `blank_image` variant against LLaVA-1.5-7B for two
# AdvBench behaviors. `blank_image` needs no seed image, so a text-only behaviors CSV
# works; for `eps_bounded` or `patch`, add a `seed_image_path` column pointing at
# images the job can read.
#
# A GPU instance with >= 24 GB of vRAM is recommended (e.g. `Standard_NC24ads_A100_v4`).
# The `compute` below points at `gcg-gpu-a100`, the same A100 pool the
# [GCG AML notebook](../gcg/1_gcg_azure_ml.ipynb) uses; change it to any GPU compute
# target in your workspace. If you hit out-of-memory errors, lower `algorithm.num_steps`,
# use a smaller model, or reduce `data.n_behaviors`.

# %%
import tempfile

from pyrit.executor.promptgen.multimodal_pgd import (
    MultiModalPGDAlgorithmConfig,
    MultiModalPGDConfig,
    MultiModalPGDDataConfig,
    MultiModalPGDModelConfig,
    MultiModalPGDOutputConfig,
    MultiModalPGDVariantConfig,
    PGDVariant,
)

config = MultiModalPGDConfig(
    model=MultiModalPGDModelConfig(vlm_id="llava-hf/llava-1.5-7b-hf", device="cuda:0", dtype="float16"),
    algorithm=MultiModalPGDAlgorithmConfig(num_steps=100, stop_loss=0.05),
    variant=MultiModalPGDVariantConfig(kind=PGDVariant.BLANK_IMAGE),
    output=MultiModalPGDOutputConfig(result_prefix="pgd", verbose=True),
)
data_config = MultiModalPGDDataConfig(
    behaviors_csv="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    n_behaviors=2,
)

# Write the configs into a tempdir so AML can mount them as separate job inputs.
config_dir = Path(tempfile.mkdtemp(prefix="mmpgd-aml-config-"))
config_path = config_dir / "config.json"
data_path = config_dir / "data.json"
config.to_json_file(config_path)
data_config.to_json_file(data_path)

# %%
from azure.ai.ml import Input, Output, command

job = command(
    code=Path(HOME_PATH),
    command=(
        "python -m pyrit.executor.promptgen.multimodal_pgd.experiments.run"
        " --config ${{inputs.config}}"
        " --data ${{inputs.data}}"
        " --output-dir ${{outputs.results}}"
    ),
    inputs={
        "config": Input(type="uri_file", path=str(config_path)),
        "data": Input(type="uri_file", path=str(data_path)),
    },
    outputs={"results": Output(type="uri_folder")},
    environment=f"{env_docker_context.name}:{env_docker_context.version}",
    environment_variables={"HUGGINGFACE_TOKEN": os.environ.get("HUGGINGFACE_TOKEN", "")},
    compute="gcg-gpu-a100",
    display_name="multimodal_pgd_image_generation",
    description="Generate adversarial images using Multimodal PGD on LLaVA-1.5.",
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

download_dir = Path(tempfile.mkdtemp(prefix="mmpgd-aml-"))
ml_client.jobs.download(name=returned_job.name, download_path=str(download_dir), all=True)

# %% [markdown]
# Load the downloaded manifest and preview the first adversarial image. The manifest
# stores each PNG's job-side path; we resolve it against the downloaded artifacts by
# filename so the preview works regardless of the mount layout.

# %%
from PIL import Image

from pyrit.executor.promptgen.multimodal_pgd import read_manifest

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
    print(
        f"  [{entry.variant}] {entry.behavior_id!r} final_loss={entry.final_loss:.4f} -> {Path(entry.image_path).name}"
    )

first = entries[0]
png_matches = list(download_dir.rglob(Path(first.image_path).name))
if png_matches:
    display(Image.open(png_matches[0]))  # noqa: F821 - display is provided by the notebook kernel
else:
    print(f"PNG {Path(first.image_path).name} not found in the downloaded artifacts.")
