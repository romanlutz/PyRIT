#!/bin/bash
# Cloud-init script for running GCG adversarial suffix generation on an Azure GPU VM.
#
# This script is designed to be passed as --custom-data to `az vm create`.
# It installs NVIDIA drivers, Python 3.12, PyRIT with GCG extras, runs the
# GCG workflow, uploads results to Azure Blob Storage, and deallocates the VM.
#
# Placeholders (replace before use -- see gcg_configure_cloud_init.py):
#   {{STORAGE_ACCOUNT}}  - Azure Storage account name
#   {{STORAGE_KEY}}       - Azure Storage account key
#   {{CONTAINER}}         - Blob container name
#   {{HF_TOKEN}}          - HuggingFace API token
#   {{PYRIT_REPO}}        - PyRIT git repo URL (default: https://github.com/Azure/PyRIT.git)
#   {{PYRIT_BRANCH}}      - PyRIT git branch to install from (default: main)

set -euo pipefail
exec > /var/log/gcg-setup.log 2>&1

echo "=== $(date) Starting GCG VM setup ==="

# Environment
export STORAGE_ACCOUNT="{{STORAGE_ACCOUNT}}"
export STORAGE_KEY="{{STORAGE_KEY}}"
export CONTAINER="{{CONTAINER}}"
export HF_TOKEN="{{HF_TOKEN}}"
export PYRIT_REPO="{{PYRIT_REPO}}"
export PYRIT_BRANCH="{{PYRIT_BRANCH}}"
export DEBIAN_FRONTEND=noninteractive

# Wait for any existing apt/dpkg locks to release (e.g., unattended-upgrades on first boot)
wait_for_apt() {
    local max_wait=300
    local waited=0
    while fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || fuser /var/lib/apt/lists/lock >/dev/null 2>&1; do
        if [ $waited -ge $max_wait ]; then
            echo "WARNING: apt lock held for over ${max_wait}s, proceeding anyway"
            break
        fi
        echo "Waiting for apt lock to release... (${waited}s)"
        sleep 10
        waited=$((waited + 10))
    done
}

# Install NVIDIA drivers + CUDA toolkit
echo "=== $(date) Installing NVIDIA drivers ==="
wait_for_apt
apt-get update -qq
apt-get install -y -qq linux-headers-$(uname -r) build-essential
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    -o /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update -qq
apt-get install -y -qq cuda-toolkit-12-4 nvidia-driver-550

echo "=== $(date) Loading NVIDIA driver ==="
modprobe nvidia || true
nvidia-smi || echo "nvidia-smi failed, driver may need reboot"

# Install Python 3.12 + pip
echo "=== $(date) Installing Python 3.12 ==="
wait_for_apt
add-apt-repository -y ppa:deadsnakes/ppa
wait_for_apt
apt-get update -qq
apt-get install -y -qq python3.12 python3.12-venv python3.12-dev

# Create venv and install PyRIT with GCG extras
echo "=== $(date) Setting up Python environment ==="
python3.12 -m venv /opt/gcg-env
source /opt/gcg-env/bin/activate
pip install --upgrade pip
pip install "pyrit[gcg] @ git+${PYRIT_REPO}@${PYRIT_BRANCH}"

# Install Azure CLI for blob upload
echo "=== $(date) Installing Azure CLI ==="
curl -sL https://aka.ms/InstallAzureCLIDeb | bash

# Write the GCG runner script
cat > /opt/run_gcg.py << 'PYEOF'
import asyncio
import json
import os
import sys
import traceback


async def main():
    from pyrit.executor.workflow.gcg import GCGContext, GCGWorkflow

    token = os.environ.get("HF_TOKEN", "")

    workflow = GCGWorkflow(
        model_name="vikhyatk/moondream2",
        model_paths=["vikhyatk/moondream2"],
        tokenizer_paths=["vikhyatk/moondream2"],
        conversation_templates=["moondream2"],
        token=token,
    )

    context = GCGContext(
        train_data="https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        n_train_data=25,
        n_steps=50,
        batch_size=128,
        topk=128,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
    )

    result = await workflow.execute_with_context_async(context=context)

    output = {
        "success": result.success,
        "status": result.status.value if hasattr(result.status, "value") else str(result.status),
        "loss": result.loss,
        "suffix": result.control_str,
        "error": result.error,
    }
    with open("/opt/gcg_result.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"GCG Result: success={result.success}, loss={result.loss}, suffix={result.control_str}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        traceback.print_exc()
        with open("/opt/gcg_result.json", "w") as f:
            json.dump({"success": False, "error": str(e), "traceback": traceback.format_exc()}, f, indent=2)
        sys.exit(1)
PYEOF

# Run GCG
echo "=== $(date) Running GCG workflow ==="
source /opt/gcg-env/bin/activate
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python /opt/run_gcg.py || true

# Upload results
echo "=== $(date) Uploading results ==="
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --container-name "$CONTAINER" \
    --name "gcg_result_${TIMESTAMP}.json" \
    --file /opt/gcg_result.json \
    --overwrite 2>&1 || echo "Blob upload failed"

az storage blob upload \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --container-name "$CONTAINER" \
    --name "gcg_setup_log_${TIMESTAMP}.txt" \
    --file /var/log/gcg-setup.log \
    --overwrite 2>&1 || echo "Log upload failed"

# Deallocate VM to free quota
echo "=== $(date) Deallocating VM ==="
RESOURCE_GROUP=$(curl -sH Metadata:true \
    "http://169.254.169.254/metadata/instance/compute/resourceGroupName?api-version=2021-02-01&format=text")
VM_NAME=$(curl -sH Metadata:true \
    "http://169.254.169.254/metadata/instance/compute/name?api-version=2021-02-01&format=text")
SUB_ID=$(curl -sH Metadata:true \
    "http://169.254.169.254/metadata/instance/compute/subscriptionId?api-version=2021-02-01&format=text")

TOKEN=$(curl -sH Metadata:true \
    "http://169.254.169.254/metadata/identity/oauth2/token?api-version=2018-02-01&resource=https://management.azure.com/" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

curl -X POST \
    "https://management.azure.com/subscriptions/${SUB_ID}/resourceGroups/${RESOURCE_GROUP}/providers/Microsoft.Compute/virtualMachines/${VM_NAME}/deallocate?api-version=2024-03-01" \
    -H "Authorization: Bearer ${TOKEN}" \
    -H "Content-Type: application/json" \
    -d '{}' || echo "Deallocate failed"

echo "=== $(date) Done ==="
