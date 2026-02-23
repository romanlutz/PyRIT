# Running GCG on an Azure GPU VM

This guide walks through provisioning an Azure GPU VM to run GCG adversarial suffix
generation using `GCGWorkflow`. The VM installs all dependencies, runs the workflow,
uploads results to Azure Blob Storage, and deallocates itself to free GPU quota.

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- An Azure subscription with **NCADS_A100_v4** (or similar GPU) compute quota
- A [HuggingFace](https://huggingface.co/) API token
- Python 3 (for the configuration script)
- PyRIT installed with GCG extras: `pip install pyrit[gcg]`

> **Branch note:** The GCG workflow is experimental. If it has not yet been merged to
> `main`, use `--pyrit-repo` and `--pyrit-branch` in Step 5 to point at the branch
> or fork that contains `GCGWorkflow`.

> **Quota note:** Azure Compute quota and Azure Machine Learning quota are separate.
> `az vm create` uses **Microsoft.Compute** quota. Check yours with:
>
> ```bash
> az vm list-usage --location westus3 --subscription $SUB \
>     --query "[?contains(name.value,'NCADS')]" -o table
> ```

## 1. Authenticate

```bash
az login
# Select the subscription with GPU quota
```

Set variables used throughout (adjust to your environment):

```bash
export SUB="<your-subscription-id>"
export RG="gcg-test"
export LOC="westus3"
export SA="gcgresults$(openssl rand -hex 4)"  # must be globally unique
```

## 2. Create Resource Group and Storage Account

```bash
az group create --name $RG --location $LOC --subscription $SUB

az storage account create \
    --name $SA \
    --resource-group $RG \
    --location $LOC \
    --subscription $SUB \
    --sku Standard_LRS \
    --kind StorageV2 \
    --allow-shared-key-access true
```

## 3. Create Blob Container

```bash
KEY=$(az storage account keys list \
    --account-name $SA --resource-group $RG --subscription $SUB \
    --query "[0].value" -o tsv)

az storage container create \
    --name gcg-results \
    --account-name $SA \
    --account-key "$KEY"
```

## 4. (Optional) Network Security Perimeter

For production use, restrict storage access with a Network Security Perimeter (NSP).
Use **Learning** mode during setup so that container creation and blob uploads work.
Switch to **Enforced** only after the run completes (Step 8).

> **Important:** If you set the NSP to Enforced mode before creating the blob container
> in Step 3, the container creation will be blocked. Always create the container first
> or use Learning mode.

```bash
# Create NSP + profile
az network perimeter create \
    --name gcg-nsp --resource-group $RG --location $LOC --subscription $SUB

az network perimeter profile create \
    --perimeter-name gcg-nsp --resource-group $RG --name gcg-profile --subscription $SUB

# Associate storage account (Learning mode)
SA_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Storage/storageAccounts/$SA"
PROFILE_ID="/subscriptions/$SUB/resourceGroups/$RG/providers/Microsoft.Network/networkSecurityPerimeters/gcg-nsp/profiles/gcg-profile"

az network perimeter association create \
    --perimeter-name gcg-nsp --resource-group $RG --subscription $SUB \
    --association-name gcg-storage-assoc \
    --access-mode Learning \
    --private-link-resource "{id:$SA_ID}" \
    --profile "{id:$PROFILE_ID}"

# Allow traffic from the same subscription
az network perimeter profile access-rule create \
    --perimeter-name gcg-nsp --profile-name gcg-profile \
    --resource-group $RG --subscription $SUB \
    --access-rule-name allow-same-subscription \
    --direction Inbound \
    --subscriptions "[{id:/subscriptions/$SUB}]"
```

## 5. Configure the Cloud-Init Script

The template at `docker/gcg_cloud_init_template.sh` has placeholders for secrets.
Use the configuration script to fill them in:

```bash
python docker/gcg_configure_cloud_init.py \
    --storage-account "$SA" \
    --storage-key "$KEY" \
    --hf-token "$HF_TOKEN" \
    --pyrit-repo "https://github.com/Azure/PyRIT.git" \
    --pyrit-branch "main" \
    --output /tmp/gcg-cloud-init.sh
```

Or use environment variables:

```bash
export STORAGE_ACCOUNT="$SA"
export STORAGE_KEY="$KEY"
export HF_TOKEN="<your-huggingface-token>"
export PYRIT_REPO="https://github.com/Azure/PyRIT.git"  # or your fork

python docker/gcg_configure_cloud_init.py --output /tmp/gcg-cloud-init.sh
```

## 6. Create the GPU VM

```bash
az vm create \
    --resource-group $RG \
    --subscription $SUB \
    --name gcg-runner \
    --location $LOC \
    --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
    --size Standard_NC24ads_A100_v4 \
    --admin-username azureuser \
    --generate-ssh-keys \
    --assign-identity "[system]" \
    --custom-data /tmp/gcg-cloud-init.sh \
    --os-disk-size-gb 128 \
    --public-ip-sku Standard
```

The `--assign-identity "[system]"` flag gives the VM a managed identity, which it uses
to call the Azure Management API and deallocate itself after the run completes.

> **VM size:** `Standard_NC24ads_A100_v4` provides 1× A100 GPU (24 cores).
> For multi-GPU runs, use `Standard_NC48ads_A100_v4` (2× A100) or
> `Standard_NC96ads_A100_v4` (4× A100).

### Grant the VM Permission to Deallocate Itself

The VM needs at least **Virtual Machine Contributor** on itself (or the resource group)
to call the deallocate API. If you have **Owner** or **User Access Administrator** role:

```bash
VM_PRINCIPAL=$(az vm show -g $RG -n gcg-runner --subscription $SUB \
    --query "identity.principalId" -o tsv)

az role assignment create \
    --assignee "$VM_PRINCIPAL" \
    --role "Virtual Machine Contributor" \
    --scope "/subscriptions/$SUB/resourceGroups/$RG" \
    --subscription $SUB
```

If you only have **Contributor** role, the VM will still run GCG and upload results,
but the self-deallocation step will fail. You can deallocate it manually:

```bash
az vm deallocate --resource-group $RG --name gcg-runner --subscription $SUB
```

## 7. Monitor Progress

The cloud-init log is streamed to `/var/log/gcg-setup.log` on the VM. To check
progress over SSH:

```bash
VM_IP=$(az vm show -g $RG -n gcg-runner --subscription $SUB -d \
    --query publicIps -o tsv)
ssh azureuser@$VM_IP tail -f /var/log/gcg-setup.log
```

Once the run completes, results are uploaded to blob storage:

```bash
az storage blob list \
    --account-name $SA --account-key "$KEY" \
    --container-name gcg-results -o table

# Download results
az storage blob download \
    --account-name $SA --account-key "$KEY" \
    --container-name gcg-results \
    --name "gcg_result_<timestamp>.json" \
    --file gcg_result.json
```

## 8. Cleanup

After retrieving results, clean up to stop incurring costs:

```bash
# Deallocate VM (if it didn't self-deallocate)
az vm deallocate --resource-group $RG --name gcg-runner --subscription $SUB

# Disable local auth on storage
az storage account update --name $SA --resource-group $RG --subscription $SUB \
    --allow-shared-key-access false

# Switch NSP to Enforced mode (if using NSP)
az network perimeter association create \
    --perimeter-name gcg-nsp --resource-group $RG --subscription $SUB \
    --association-name gcg-storage-assoc \
    --access-mode Enforced \
    --private-link-resource "{id:$SA_ID}" \
    --profile "{id:$PROFILE_ID}"

# Or delete everything
az group delete --name $RG --subscription $SUB --yes --no-wait
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `QuotaExceeded` on VM create | Check compute quota with `az vm list-usage`. Azure ML quota is separate from Compute quota. |
| Storage operations blocked | NSP in Enforced mode blocks external traffic. Switch to Learning mode or add inbound access rules. |
| `nvidia-smi` not found | Driver installation may require a reboot. SSH in and run `sudo reboot`. |
| Blob upload fails | Verify local auth is enabled (`--allow-shared-key-access true`) and the storage key is correct. |
| VM doesn't deallocate | The managed identity needs **Virtual Machine Contributor** role. Deallocate manually if RBAC can't be assigned. |
