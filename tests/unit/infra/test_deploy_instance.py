# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for CoPyRIT deployment configuration."""

import argparse
import json
import logging
import subprocess
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from infra import deploy_instance


def test_managed_identity_blob_uri_accepts_credential_free_azure_uri() -> None:
    uri = "https://account.blob.core.windows.net/config/config.yaml"

    assert deploy_instance._managed_identity_blob_uri(uri) == uri


@pytest.mark.parametrize(
    "uri",
    [
        "https://account.blob.core.windows.net/config/config.yaml?sig=secret",
        "https://account.blob.core.windows.net/config/config.yaml#fragment",
        "https://attacker.blob.example.com/config/config.yaml",
        "http://account.blob.core.windows.net/config/config.yaml",
    ],
)
def test_managed_identity_blob_uri_rejects_credentialed_or_untrusted_uri(uri: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="credential-free"):
        deploy_instance._managed_identity_blob_uri(uri)


def test_post_deploy_enables_spa_and_device_code_authentication() -> None:
    with patch.object(deploy_instance, "run_az") as run_az:
        deploy_instance.post_deploy(
            app_object_id="app-object-id",
            fqdn="copyrit.example.com",
        )

    args = run_az.call_args.kwargs["args"]
    body = json.loads(args[args.index("--body") + 1])
    assert body == {
        "spa": {"redirectUris": ["https://copyrit.example.com"]},
        "isFallbackPublicClient": True,
    }


def test_deploy_bicep_deploys_infrastructure_before_application() -> None:
    infrastructure_outputs = {
        "egressPublicIpAddress": {"value": "203.0.113.10"},
        "managedIdentityResourceId": {"value": "/subscriptions/sub/resourceGroups/rg/providers/identity"},
    }
    application_outputs = {"appFqdn": {"value": "copyrit.example.com"}}

    with patch.object(
        deploy_instance,
        "_deploy_bicep_template",
        side_effect=[infrastructure_outputs, application_outputs],
    ) as deploy_template:
        outputs = deploy_instance.deploy_bicep(
            resource_group="rg",
            app_name="copyrit-demo",
            container_image="acr.azurecr.io/pyrit:abc123",
            tenant_id="tenant-id",
            client_id="client-id",
            group_ids="group-id",
            admin_group_id="admin-group-id",
            allowed_cidr="",
            sql_server_fqdn="server.database.windows.net",
            sql_database_name="database",
            kv_resource_id="/subscriptions/sub/resourceGroups/rg/providers/Microsoft.KeyVault/vaults/vault",
            acr_name="acr",
            managed_identity_resource_id=(
                "/subscriptions/sub/resourceGroups/rg/providers/"
                "Microsoft.ManagedIdentity/userAssignedIdentities/copyrit-demo-identity"
            ),
            env_file_contents="KEY=value",
            pyrit_config_file_uri="",
            tags={"Service": "pyrit-gui"},
        )

    assert deploy_template.call_count == 2
    infrastructure_call, application_call = deploy_template.call_args_list
    assert infrastructure_call.kwargs["template_file"] == deploy_instance.INFRASTRUCTURE_BICEP_TEMPLATE
    assert infrastructure_call.kwargs["deployment_name"] == "copyrit-demo-infrastructure"
    assert set(infrastructure_call.kwargs["parameters"]) == {
        "appName",
        "acrName",
        "existingManagedIdentityResourceId",
        "sandboxGroupName",
        "tags",
    }
    assert application_call.kwargs["template_file"] == deploy_instance.APPLICATION_BICEP_TEMPLATE
    assert application_call.kwargs["deployment_name"] == "copyrit-demo-application"
    assert application_call.kwargs["parameters"]["containerImage"]["value"] == "acr.azurecr.io/pyrit:abc123"
    assert infrastructure_call.kwargs["parameters"]["sandboxGroupName"]["value"] == ""
    assert application_call.kwargs["parameters"]["cpuCores"]["value"] == "1.0"
    assert application_call.kwargs["parameters"]["memoryGb"]["value"] == "2.0"
    assert application_call.kwargs["parameters"]["pyritInitializer"]["value"] == "target,technique"
    assert outputs == infrastructure_outputs | application_outputs


@pytest.fixture
def deployment_arguments(tmp_path: Path) -> list[str]:
    env_file = tmp_path / "preview.env"
    env_file.write_text("AZURE_TOKEN_CREDENTIALS=ManagedIdentityCredential\n", encoding="utf-8")
    return [
        "--instance-name",
        "preview",
        "--env-file",
        str(env_file),
        "--subscription",
        "11111111-1111-1111-1111-111111111111",
        "--acr-name",
        "sharedacr",
        "--container-image",
        "sharedacr.azurecr.io/pyrit:abc123",
        "--allowed-groups",
        "22222222-2222-2222-2222-222222222222",
        "--admin-group",
        "33333333-3333-3333-3333-333333333333",
    ]


@pytest.mark.parametrize(
    ("cpu", "memory"),
    [("0.25", "0.5"), ("1.0", "2.0"), ("2", "4"), ("2.0", "4.0"), ("4.0", "8.0")],
)
def test_container_resources_accept_consumption_pairs(*, cpu: str, memory: str) -> None:
    deploy_instance._validate_container_resources(cpu_cores=cpu, memory_gb=memory)


@pytest.mark.parametrize(
    ("cpu", "memory"),
    [
        ("0", "0"),
        ("-1", "-2"),
        ("0.3", "0.6"),
        ("4.25", "8.5"),
        ("2", "2"),
        ("1", "NaN"),
        ("Infinity", "2"),
        ("invalid", "2"),
    ],
)
def test_invalid_container_resources_fail_before_azure(
    *, deployment_arguments: list[str], cpu: str, memory: str
) -> None:
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        result = deploy_instance.main([*deployment_arguments, "--cpu-cores", cpu, "--memory-gb", memory])

    assert result == 1
    run_az.assert_not_called()


def test_preview_dry_run_reports_explicit_regions_resources_and_lifetime(
    *, deployment_arguments: list[str], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=deploy_instance.__name__)
    journal_file = tmp_path / "journal.json"
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        result = deploy_instance.main(
            [
                *deployment_arguments,
                "--location",
                "westus2",
                "--sql-location",
                "westus3",
                "--sql-service-objective",
                "S0",
                "--cpu-cores",
                "2.0",
                "--memory-gb",
                "4.0",
                "--pyrit-initializer",
                "target",
                "--sandbox-group-name",
                "preview-sandbox",
                "--validation-state-container",
                "validation-state",
                "--journal-file",
                str(journal_file),
                "--retention-hours",
                "24",
                "--dry-run",
            ]
        )

    assert result == 0
    run_az.assert_not_called()
    assert not journal_file.exists()
    assert "SQL location: westus3 (service objective: S0)" in caplog.text
    assert "Container resources: 2.0 CPU cores / 4.0 GiB" in caplog.text
    assert "Replicas: 1 minimum / 1 maximum" in caplog.text
    assert "Preview retention hours: 24" in caplog.text
    assert "Owned sandbox group: preview-sandbox" in caplog.text
    assert "Validation state container: validation-state" in caplog.text


def test_default_dry_run_preserves_same_region_basic_and_original_sizing(
    *, deployment_arguments: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=deploy_instance.__name__)
    assert deploy_instance.main([*deployment_arguments, "--dry-run"]) == 0
    assert "SQL location: eastus2 (service objective: Basic)" in caplog.text
    assert "Container resources: 1.0 CPU cores / 2.0 GiB" in caplog.text
    assert "PyRIT initializers: target,technique" in caplog.text
    assert "Validation state container: (not configured)" in caplog.text


@pytest.mark.parametrize("group_exists", [True, "false", None, {}])
def test_create_only_workflow_rejects_existing_or_invalid_group_read(
    *, deployment_arguments: list[str], group_exists: object
) -> None:
    with (
        patch.object(deploy_instance, "set_subscription", autospec=True),
        patch.object(
            deploy_instance,
            "run_az_json",
            side_effect=["11111111-1111-1111-1111-111111111111", group_exists],
            autospec=True,
        ),
        patch.object(deploy_instance, "create_resource_group", autospec=True) as create_group,
    ):
        result = deploy_instance.main(deployment_arguments)

    assert result == 1
    create_group.assert_not_called()


@pytest.mark.parametrize(("objective", "edition", "capacity"), [("Basic", "Basic", "5"), ("S0", "Standard", "10")])
def test_sql_service_objective_preserves_basic_and_explicit_s0(*, objective: str, edition: str, capacity: str) -> None:
    with (
        patch.object(deploy_instance, "run_az", autospec=True) as run_az,
        patch.object(
            deploy_instance,
            "run_az_json",
            side_effect=[
                {"displayName": "Preview Owner", "id": "owner-id"},
                "preview.database.windows.net",
            ],
            autospec=True,
        ),
    ):
        deploy_instance.create_sql_server_and_db(
            resource_group="copyrit-preview",
            location="westus3",
            server_name="preview",
            database_name="preview",
            service_objective=objective,
        )

    server_command, database_command = [call.kwargs["args"] for call in run_az.call_args_list]
    assert server_command[server_command.index("--location") + 1] == "westus3"
    assert database_command[database_command.index("--edition") + 1] == edition
    assert database_command[database_command.index("--capacity") + 1] == capacity


def test_main_forwards_explicit_preview_parameters(*, deployment_arguments: list[str], tmp_path: Path) -> None:
    responses: dict[str, object] = {
        "set_subscription": None,
        "create_resource_group": None,
        "create_entra_app": {
            "app_id": "new-client",
            "app_object_id": "new-app",
            "tenant_id": "tenant",
            "sp_id": "new-sp",
        },
        "assign_groups_to_app": None,
        "create_sql_server_and_db": {"server_fqdn": "preview.database.windows.net", "database_name": "preview"},
        "create_storage_account": {"account_id": "own-storage", "container_url": "https://own-storage/dbdata"},
        "create_key_vault": "own-vault",
        "create_managed_identity_and_grant_roles": "new-preview-principal",
        "deploy_bicep": {
            "appFqdn": {"value": "preview.azurecontainerapps.io"},
            "egressPublicIpAddress": {"value": "203.0.113.10"},
        },
        "configure_sql_network_access": None,
        "post_deploy": None,
    }
    journal_file = tmp_path / "journal.json"
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(
                deploy_instance,
                "run_az_json",
                side_effect=["11111111-1111-1111-1111-111111111111", False],
                autospec=True,
            )
        )
        mocks = {
            name: stack.enter_context(patch.object(deploy_instance, name, return_value=response, autospec=True))
            for name, response in responses.items()
        }
        result = deploy_instance.main(
            [
                *deployment_arguments,
                "--location",
                "westus2",
                "--sql-location",
                "westus3",
                "--sql-service-objective",
                "S0",
                "--cpu-cores",
                "2.0",
                "--memory-gb",
                "4.0",
                "--pyrit-initializer",
                "target",
                "--sandbox-group-name",
                "preview-sandbox",
                "--validation-state-container",
                "validation-state",
                "--journal-file",
                str(journal_file),
                "--retention-hours",
                "24",
            ]
        )

    assert result == 0
    assert mocks["create_resource_group"].call_args.kwargs["location"] == "westus2"
    assert mocks["create_sql_server_and_db"].call_args.kwargs["location"] == "westus3"
    assert mocks["create_sql_server_and_db"].call_args.kwargs["service_objective"] == "S0"
    assert mocks["create_storage_account"].call_args.kwargs["state_container_name"] == "validation-state"
    application = mocks["deploy_bicep"].call_args.kwargs
    assert (application["cpu_cores"], application["memory_gb"], application["pyrit_initializer"]) == (
        "2.0",
        "4.0",
        "target",
    )
    assert "ExpiresAt" in application["tags"]
    assert application["sandbox_group_name"] == "preview-sandbox"
    document = json.loads(journal_file.read_text(encoding="utf-8"))
    assert document["status"] == "provisioned_manual_sql_and_runtime_validation_required"
    assert (
        datetime.fromisoformat(document["expires_at"])
        - datetime.fromisoformat(document["first_creation_request_not_before"])
    ).total_seconds() == 86400
    assert deploy_instance._CURRENT_JOURNAL.get() is None


@pytest.mark.parametrize("state_container", ["", "validation-state"])
def test_storage_state_container_is_opt_in_private_and_separate(state_container: str) -> None:
    with (
        patch.object(deploy_instance, "run_az", autospec=True) as run_az,
        patch.object(deploy_instance, "run_az_json", return_value="owned-storage-id", autospec=True),
    ):
        result = deploy_instance.create_storage_account(
            resource_group="copyrit-preview",
            location="westus2",
            account_name="copyritpreviewsa",
            state_container_name=state_container,
        )

    commands = [call.kwargs["args"] for call in run_az.call_args_list]
    containers = [command for command in commands if command[:3] == ["storage", "container-rm", "create"]]
    assert [command[command.index("--name") + 1] for command in containers] == (
        ["dbdata", state_container] if state_container else ["dbdata"]
    )
    for command in containers:
        assert command[command.index("--storage-account") + 1] == "copyritpreviewsa"
        assert command[command.index("--resource-group") + 1] == "copyrit-preview"
        assert command[command.index("--public-access") + 1] == "off"
    assert result["container_url"] == "https://copyritpreviewsa.blob.core.windows.net/dbdata"
    assert ("state_container_url" in result) == bool(state_container)
    if state_container:
        assert result["state_container_url"] == "https://copyritpreviewsa.blob.core.windows.net/validation-state"
    account = commands[0]
    assert account[account.index("--allow-blob-public-access") + 1] == "false"
    assert account[account.index("--sku") + 1] == "Standard_LRS"


@pytest.mark.parametrize("name", ["dbdata", "ab", "a" * 64, "Uppercase", "two--hyphens", "-state", "state/other"])
def test_invalid_state_container_is_rejected_before_azure(*, deployment_arguments: list[str], name: str) -> None:
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        with pytest.raises(SystemExit) as error:
            deploy_instance.main([*deployment_arguments, "--validation-state-container", name])
    assert error.value.code == 2
    run_az.assert_not_called()


def test_storage_state_container_cannot_overlap_custom_result_container() -> None:
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        with pytest.raises(ValueError, match="separate containers"):
            deploy_instance.create_storage_account(
                resource_group="copyrit-preview",
                location="westus2",
                account_name="copyritpreviewsa",
                container_name="custom-media",
                state_container_name="custom-media",
            )
    run_az.assert_not_called()


def test_journal_excludes_payloads_and_captures_actual_creation_ids(tmp_path: Path) -> None:
    journal = deploy_instance._DeploymentJournal(
        path=tmp_path / "journal.json",
        instance="preview",
        resource_group_id="/subscriptions/sub/resourceGroups/copyrit-preview",
        retention_hours=24,
    )
    secret = "DO_NOT_RETAIN_THIS_SECRET"
    operation = journal.begin(
        ["rest", "--method", "POST", "--url", "https://graph.microsoft.com/v1.0/applications", "--body", secret]
    )
    assert operation == 0
    journal.complete(
        operation=operation,
        result=subprocess.CompletedProcess(
            ["az"],
            0,
            json.dumps({"id": "new-app-object", "appId": "new-client", "secret": {"value": secret}}),
            "",
        ),
    )

    text = journal.path.read_text(encoding="utf-8")
    assert secret not in text
    event = json.loads(text)["operations"][0]
    assert event["returned_references"] == [{"id": "new-app-object", "appId": "new-client"}]
    assert event["status"] == "succeeded"
    assert journal.begin(["group", "exists", "--name", "copyrit-preview"]) is None


def test_run_az_journals_request_before_mutation_and_actual_cli_failure(tmp_path: Path) -> None:
    journal = deploy_instance._DeploymentJournal(
        path=tmp_path / "journal.json",
        instance="preview",
        resource_group_id="/subscriptions/sub/resourceGroups/copyrit-preview",
        retention_hours=24,
    )

    def fail_cli(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        document = json.loads(journal.path.read_text(encoding="utf-8"))
        assert document["operations"][0]["status"] == "requested"
        assert command[-2:] == ["-o", "json"]
        raise subprocess.CalledProcessError(1, command, stderr="AuthorizationFailed")

    token = deploy_instance._CURRENT_JOURNAL.set(journal)
    try:
        with patch.object(deploy_instance.subprocess, "run", side_effect=fail_cli):
            with pytest.raises(subprocess.CalledProcessError):
                deploy_instance.run_az(args=["group", "create", "--name", "copyrit-preview"])
    finally:
        deploy_instance._CURRENT_JOURNAL.reset(token)

    event = json.loads(journal.path.read_text(encoding="utf-8"))["operations"][0]
    assert event["status"] == "failed"
    assert event["exit_code"] == 1
    assert "AuthorizationFailed" not in journal.path.read_text(encoding="utf-8")
    assert "stderr_sha256" in event


@pytest.mark.parametrize("hours", ["0", "-1", "24"])
def test_preview_retention_requires_positive_hours_and_new_journal(
    *, deployment_arguments: list[str], hours: str
) -> None:
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        assert deploy_instance.main([*deployment_arguments, "--retention-hours", hours]) == 1
    run_az.assert_not_called()


def test_existing_journal_is_not_overwritten(*, deployment_arguments: list[str], tmp_path: Path) -> None:
    journal_file = tmp_path / "journal.json"
    journal_file.write_text("retained journal", encoding="utf-8")
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        result = deploy_instance.main([*deployment_arguments, "--journal-file", str(journal_file)])

    assert result == 1
    run_az.assert_not_called()
    assert journal_file.read_text(encoding="utf-8") == "retained journal"


def test_journal_replace_failure_preserves_previous_receipt_and_removes_temporary_paths(tmp_path: Path) -> None:
    journal = deploy_instance._DeploymentJournal(
        path=tmp_path / "journal.json",
        instance="preview",
        resource_group_id="/subscriptions/sub/resourceGroups/copyrit-preview",
        retention_hours=24,
    )
    previous = journal.path.read_bytes()
    with patch.object(Path, "replace", side_effect=PermissionError("journal is locked")):
        with pytest.raises(PermissionError, match="journal is locked"):
            journal.finish("test-status")

    assert journal.path.read_bytes() == previous
    assert list(tmp_path.iterdir()) == [journal.path]


@pytest.mark.parametrize("name", ["-preview", "preview/group", "p" * 33, "preview_group"])
def test_invalid_sandbox_group_fails_before_azure(*, deployment_arguments: list[str], name: str) -> None:
    with patch.object(deploy_instance, "run_az", autospec=True) as run_az:
        assert deploy_instance.main([*deployment_arguments, f"--sandbox-group-name={name}"]) == 1
    run_az.assert_not_called()


def test_journal_extracts_arm_output_resource_references() -> None:
    resource_id = "/subscriptions/sub/resourceGroups/copyrit-preview/providers/Microsoft.App/sandboxGroups/preview"
    response = {"properties": {"outputs": {"sandboxGroupResourceId": {"type": "String", "value": resource_id}}}}
    assert deploy_instance._DeploymentJournal._references(response) == [{"resourceId": resource_id}]


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("invalid creation response"),
        subprocess.CalledProcessError(1, ["az", "ad", "app", "create"], stderr="AuthorizationFailed"),
    ],
)
def test_main_retains_failed_provisioning_status_and_original_expiry(
    *, deployment_arguments: list[str], tmp_path: Path, error: Exception
) -> None:
    journal_file = tmp_path / "failed-journal.json"
    with (
        patch.object(deploy_instance, "set_subscription", autospec=True),
        patch.object(
            deploy_instance,
            "run_az_json",
            side_effect=["11111111-1111-1111-1111-111111111111", False],
            autospec=True,
        ),
        patch.object(deploy_instance, "create_resource_group", autospec=True),
        patch.object(deploy_instance, "create_entra_app", side_effect=error, autospec=True),
    ):
        result = deploy_instance.main(
            [*deployment_arguments, "--journal-file", str(journal_file), "--retention-hours", "24"]
        )
    assert result == 1
    document = json.loads(journal_file.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert (
        datetime.fromisoformat(document["expires_at"])
        - datetime.fromisoformat(document["first_creation_request_not_before"])
    ).total_seconds() == 86400
    assert deploy_instance._CURRENT_JOURNAL.get() is None
