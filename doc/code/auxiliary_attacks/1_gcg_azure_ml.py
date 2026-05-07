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
# # 1. Generating GCG Suffixes Using Azure Machine Learning

# %% [markdown]
# This notebook shows how to generate GCG [@zou2023gcg] suffixes using Azure Machine Learning (AML), which consists of three main steps:
# 1. Connect to an Azure Machine Learning (AML) workspace.
# 2. Create AML Environment with the Python dependencies.
# 3. Submit a training job to AML.

# %% [markdown]
# ## Connect to Azure Machine Learning Workspace

# %% [markdown]
# The [workspace](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace) is the top-level resource for Azure Machine Learning (AML), providing a centralized place to work with all the artifacts you create when using AML. In this section, we will connect to the workspace in which the job will be run.
#
# To connect to a workspace, we need identifier parameters - a subscription, resource group and workspace name. We will use these details in the `MLClient` from `azure.ai.ml` to get a handle to the required AML workspace. We use the [default Azure authentication](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity.defaultazurecredential?view=azure-python) for this tutorial.

# %%
import os

from pyrit.setup.initialization import _load_environment_files

_load_environment_files(env_files=None)

subscription_id = os.environ.get("AZURE_ML_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_ML_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_ML_WORKSPACE_NAME")
print(workspace)

# %%
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

ml_client = MLClient(AzureCliCredential(), subscription_id, resource_group, workspace)

# %% [markdown]
# ## Create AML Environment

# %% [markdown]
# To install the dependencies needed to run GCG, we create an AML environment from a
# [Dockerfile](../../../pyrit/auxiliary_attacks/gcg/src/Dockerfile). The Dockerfile uses
# an NVIDIA CUDA base image with Python 3.11 and installs PyRIT with the `gcg` extra.

# %%
from pathlib import Path

from azure.ai.ml.entities import BuildContext, Environment

from pyrit.common.path import HOME_PATH

# Configure the AML environment — build context is the repo root so the Dockerfile
# can COPY pyproject.toml and pyrit/ for pip install -e ".[gcg]"
env_docker_context = Environment(
    build=BuildContext(
        path=Path(HOME_PATH),
        dockerfile_path="pyrit/auxiliary_attacks/gcg/src/Dockerfile",
    ),
    name="pyrit-gcg",
    description="PyRIT GCG environment: CUDA 12.1 + Python 3.11 + pip install -e .[gcg]",
    tags={"Owner": os.environ.get("USER", "unknown")},
)

ml_client.environments.create_or_update(env_docker_context)

# %% [markdown]
# ## Submit Training Job to AML

# %% [markdown]
# Finally, we configure the command to run the GCG algorithm. The entry point is
# [`pyrit.auxiliary_attacks.gcg.experiments.run`](../../../pyrit/auxiliary_attacks/gcg/experiments/run.py),
# invoked as a module so the uploaded code snapshot takes priority over the
# Docker-installed package (Python's `-m` flag puts the cwd at the front of `sys.path`).
#
# We also have to specify a GPU compute target. In our experience, a GPU instance with
# at least 24GB of vRAM is required (e.g., Standard_NC24ads_A100_v4).
#
# Depending on the compute instance you use, you may encounter "out of memory" errors.
# In this case, we recommend training on a smaller model or lowering `n_train_data` or `batch_size`.

# %%
from azure.ai.ml import command

job = command(
    code=Path(HOME_PATH),
    command=(
        "python -m pyrit.auxiliary_attacks.gcg.experiments.run"
        " --model_name llama_2"
        " --setup single"
        " --n_train_data 5"
        " --n_test_data 0"
        " --n_steps 5"
        " --batch_size 64"
    ),
    inputs={},
    environment=f"{env_docker_context.name}:{env_docker_context.version}",
    environment_variables={"HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"]},
    compute="gcg-gpu-a100",
    display_name="gcg_suffix_generation",
    description="Generate adversarial suffixes using GCG on Llama-2.",
    tags={"Owner": os.environ.get("USER", "unknown")},
)

# %%
returned_job = ml_client.create_or_update(job)
print(f"Job: {returned_job.name}")
print(f"Status: {returned_job.status}")
print(f"Studio URL: {returned_job.studio_url}")
