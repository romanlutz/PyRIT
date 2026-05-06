# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Submit a GCG baseline job to Azure ML.

Reads workspace configuration from PyRIT's .env files
(AZURE_ML_SUBSCRIPTION_ID, AZURE_ML_RESOURCE_GROUP, AZURE_ML_WORKSPACE_NAME).

Usage:
    python scripts/submit_gcg_job.py
"""

import os
from pathlib import Path

from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import BuildContext, Environment
from azure.identity import AzureCliCredential

from pyrit.common.path import HOME_PATH
from pyrit.setup.initialization import _load_environment_files


def main() -> None:
    _load_environment_files(env_files=None)

    subscription_id = os.environ["AZURE_ML_SUBSCRIPTION_ID"]
    resource_group = os.environ["AZURE_ML_RESOURCE_GROUP"]
    workspace_name = os.environ["AZURE_ML_WORKSPACE_NAME"]
    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    job = command(
        code=Path(HOME_PATH),
        command=(
            "python scripts/run_gcg_aml.py"
            " --model_name phi_3_mini"
            " --setup single"
            " --n_train_data 5"
            " --n_test_data 0"
            " --n_steps 5"
            " --batch_size 64"
        ),
        inputs={},
        environment="pyrit-gcg:6",
        environment_variables={"HUGGINGFACE_TOKEN": hf_token},
        compute="gcg-gpu-a100",
        display_name="gcg_baseline",
        description="GCG baseline: phi-3-mini, 5 steps, 5 train data",
        tags={"Owner": "romanlutz"},
    )

    returned_job = ml_client.jobs.create_or_update(job)
    print(f"Job: {returned_job.name}")
    print(f"Status: {returned_job.status}")
    print(f"Studio URL: {returned_job.studio_url}")


if __name__ == "__main__":
    main()
