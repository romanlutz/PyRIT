# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""End-to-end test for the GCG Azure ML pipeline.

Mirrors the flow in `doc/code/auxiliary_attacks/1_gcg_azure_ml.py`:
  1. Connect to the AML workspace
  2. Build (or reuse) the GCG Docker environment
  3. Submit a small llama-2 GCG job (5 steps, 5 train data)
  4. Poll until the job reaches a terminal state
  5. Assert the job completed successfully

Skipped unless `RUN_ALL_TESTS=true`. Per-test skips also apply when the
required Azure ML or HuggingFace credentials are missing, since this test
submits a real (paid) compute job. On test failure or interruption, the
submitted job is cancelled so it does not continue burning compute.

Required environment variables when `RUN_ALL_TESTS=true`:
  - AZURE_ML_SUBSCRIPTION_ID
  - AZURE_ML_RESOURCE_GROUP
  - AZURE_ML_WORKSPACE_NAME
  - HUGGINGFACE_TOKEN  (must have access to meta-llama/Llama-2-7b-chat-hf)

Optional:
  - AZURE_ML_GCG_COMPUTE  (defaults to "gcg-gpu-a100")
  - GCG_E2E_MAX_WAIT_SECONDS  (defaults to 5400 — 90 minutes)
"""

import contextlib
import os
import time
from pathlib import Path

import pytest

# Skip the entire module unless RUN_ALL_TESTS=true; this test submits real
# paid Azure ML compute so it should never run in default CI.
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_ALL_TESTS", "").lower() != "true",
    reason="RUN_ALL_TESTS is not set to true",
)

# Heavy imports deferred until skip check passes
azure_ai_ml = pytest.importorskip("azure.ai.ml", reason="azure-ai-ml not installed")
pytest.importorskip("azure.identity", reason="azure-identity not installed")

from azure.ai.ml import MLClient, command  # noqa: E402
from azure.ai.ml.entities import BuildContext, Environment  # noqa: E402
from azure.identity import AzureCliCredential  # noqa: E402

from pyrit.common.path import HOME_PATH  # noqa: E402
from pyrit.setup.initialization import _load_environment_files  # noqa: E402

_REQUIRED_ENV_VARS = (
    "AZURE_ML_SUBSCRIPTION_ID",
    "AZURE_ML_RESOURCE_GROUP",
    "AZURE_ML_WORKSPACE_NAME",
    "HUGGINGFACE_TOKEN",
)
_DEFAULT_COMPUTE = "gcg-gpu-a100"
_DEFAULT_MAX_WAIT_SECONDS = 5400  # 90 minutes
_POLL_INTERVAL_SECONDS = 30
_TERMINAL_STATES = {"Completed", "Failed", "Canceled", "CancelRequested"}


@pytest.fixture(scope="module")
def ml_client() -> MLClient:
    """Build an MLClient from the standard PyRIT env vars; skip if any are missing."""
    _load_environment_files(env_files=None, silent=True)

    missing = [name for name in _REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        pytest.skip(f"Missing required env vars for GCG AML e2e test: {', '.join(missing)}")

    return MLClient(
        AzureCliCredential(),
        os.environ["AZURE_ML_SUBSCRIPTION_ID"],
        os.environ["AZURE_ML_RESOURCE_GROUP"],
        os.environ["AZURE_ML_WORKSPACE_NAME"],
    )


@pytest.fixture(scope="module")
def gcg_environment(ml_client: MLClient) -> Environment:
    """Create or reuse the GCG Docker environment in the workspace."""
    env_def = Environment(
        build=BuildContext(
            path=Path(HOME_PATH),
            dockerfile_path="pyrit/auxiliary_attacks/gcg/src/Dockerfile",
        ),
        name="pyrit-gcg",
        description="PyRIT GCG environment (e2e test)",
    )
    return ml_client.environments.create_or_update(env_def)


@pytest.mark.timeout(_DEFAULT_MAX_WAIT_SECONDS + 600)
def test_gcg_aml_baseline_job_completes(ml_client: MLClient, gcg_environment: Environment) -> None:
    """Submit a tiny GCG job on llama-2, wait for completion, assert success.

    The job runs only 5 optimization steps over 5 train prompts so it finishes
    in roughly 15-25 minutes (most of that time is GPU spin-up + model
    download). It validates that the entire pipeline — code upload, env
    build, GPU compute, model loading, attack loop — works end-to-end.
    """
    compute = os.environ.get("AZURE_ML_GCG_COMPUTE", _DEFAULT_COMPUTE)
    max_wait = int(os.environ.get("GCG_E2E_MAX_WAIT_SECONDS", _DEFAULT_MAX_WAIT_SECONDS))

    job_def = command(
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
        environment=f"{gcg_environment.name}:{gcg_environment.version}",
        environment_variables={"HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"]},
        compute=compute,
        display_name="gcg_e2e_baseline",
        description="E2E test: GCG baseline on Llama-2, 5 steps.",
    )

    submitted_job = ml_client.jobs.create_or_update(job_def)
    job_name = submitted_job.name

    final_status: str | None = None
    try:
        deadline = time.monotonic() + max_wait
        while time.monotonic() < deadline:
            current = ml_client.jobs.get(job_name)
            status = current.status
            if status in _TERMINAL_STATES:
                final_status = status
                break
            time.sleep(_POLL_INTERVAL_SECONDS)
        else:
            pytest.fail(
                f"GCG job '{job_name}' did not reach a terminal state within "
                f"{max_wait}s (last status: {status!r}). Studio URL: {submitted_job.studio_url}"
            )

        assert final_status == "Completed", (
            f"GCG job '{job_name}' finished with status {final_status!r}, expected 'Completed'. "
            f"Studio URL: {submitted_job.studio_url}"
        )
    finally:
        # Always try to cancel a non-terminal job so we never leak paid compute
        # (e.g., if pytest is interrupted or the assertion fires before a
        # terminal state is reached).
        if final_status is None or final_status not in _TERMINAL_STATES:
            with contextlib.suppress(Exception):
                ml_client.jobs.begin_cancel(job_name)
