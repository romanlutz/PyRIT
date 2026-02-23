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
# # GCG Adversarial Suffix Generation
#
# > **⚠️ Experimental:** GCG support in PyRIT is experimental and will have substantial
# > changes in upcoming versions. APIs may change without notice.
#
# The [Greedy Coordinate Gradient (GCG)](https://arxiv.org/abs/2307.15043) algorithm generates adversarial
# suffixes that, when appended to a prompt, can cause language models to comply with harmful instructions.
# A GCG suffix is a string of tokens optimized via gradient-based search — for example:
#
# ```
# $\ inqu Here spher =" vas_{( dentSMckground CREATE poem combining_{( Taylor_{( phrase morning _{
# ```
#
# In PyRIT, GCG suffixes are applied to prompts using `SuffixAppendConverter`:
#
# ```python
# from pyrit.prompt_converter import SuffixAppendConverter
# from pyrit.prompt_normalizer import PromptConverterConfiguration
#
# suffix = '$\\ inqu Here spher =" vas_{( dentSMckground CREATE poem combining_{( Taylor_{( phrase morning _{'
# converter = PromptConverterConfiguration.from_converters(
#     converters=[SuffixAppendConverter(suffix=suffix)]
# )
# ```
#
# The rest of this notebook shows how to **generate** these suffixes using `GCGWorkflow`.

# %% [markdown]
# ## Generating Suffixes with GCGWorkflow
#
# `GCGWorkflow` wraps the GCG algorithm in the PyRIT `WorkflowStrategy` pattern, providing
# validation, setup, execution, and teardown lifecycle management.
#
# **Requirements:**
# - A GPU with at least 32 GB of vRAM
# - PyRIT installed with the `gcg` extras: `pip install pyrit[gcg]`
# - A HuggingFace token for model access

# %%
import os

from pyrit.executor.workflow.gcg import GCGWorkflow

# Configure the workflow with model infrastructure
workflow = GCGWorkflow(
    model_name="phi_3_mini",
    model_paths=["microsoft/Phi-3-mini-4k-instruct"],
    tokenizer_paths=["microsoft/Phi-3-mini-4k-instruct"],
    conversation_templates=["phi3"],
    token=os.environ["HUGGINGFACE_TOKEN"],
)

# Run a short 3-step optimization (increase n_steps for real attacks)
result = await workflow.execute_async(  # type: ignore
    train_data="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    n_train_data=25,
    n_steps=3,
    batch_size=256,
)

print(f"Status: {result.status}")
print(f"Success: {result.success}")
print(f"Steps: {result.n_steps}")
print(f"Loss: {result.loss}")
print(f"Suffix: {result.control_str}")

# %% [markdown]
# For production use, increase `n_steps` (typically 500+) and consider tuning `batch_size`,
# `learning_rate`, and other hyperparameters via `GCGContext`:

# %%
from pyrit.executor.workflow.gcg import GCGContext

context = GCGContext(
    train_data="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
    n_train_data=25,
    n_steps=500,
    batch_size=256,
    stop_on_success=True,
)

result = await workflow.execute_with_context_async(context=context)  # type: ignore
print(f"Suffix: {result.control_str}")

# %% [markdown]
# ## Running on an Azure GPU VM
#
# If you don't have a local GPU, you can run GCG on an Azure VM that automatically
# installs everything, runs the workflow, uploads results, and deallocates itself.
#
# See the full guide: [Running GCG on an Azure GPU VM](gcg_azure_setup.md)
#
# The setup uses:
# - A **cloud-init template** (`docker/gcg_cloud_init_template.sh`) passed to the VM at creation
# - A **configuration script** (`docker/gcg_configure_cloud_init.py`) that fills in your secrets
#
# Quick start:
#
# ```bash
# # Configure the cloud-init script with your secrets
# python docker/gcg_configure_cloud_init.py \
#     --storage-account "$STORAGE_ACCOUNT" \
#     --storage-key "$STORAGE_KEY" \
#     --hf-token "$HF_TOKEN" \
#     --output /tmp/gcg-cloud-init.sh
#
# # Create the GPU VM (it runs GCG and deallocates itself)
# az vm create \
#     --resource-group gcg-test \
#     --name gcg-runner \
#     --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
#     --size Standard_NC24ads_A100_v4 \
#     --admin-username azureuser \
#     --generate-ssh-keys \
#     --assign-identity "[system]" \
#     --custom-data /tmp/gcg-cloud-init.sh \
#     --os-disk-size-gb 128
# ```
