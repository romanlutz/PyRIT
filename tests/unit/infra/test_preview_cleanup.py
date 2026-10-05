# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline coverage of exact journal-bound preview cleanup."""

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from infra import deploy_instance, teardown_instance

INSTANCE = "preview"
SUBSCRIPTION = "11111111-1111-1111-1111-111111111111"
TENANT = "22222222-2222-2222-2222-222222222222"
PRINCIPAL = "33333333-3333-3333-3333-333333333333"
CLIENT = "44444444-4444-4444-4444-444444444444"
APPLICATION = "55555555-5555-5555-5555-555555555555"
SERVICE_PRINCIPAL = "66666666-6666-6666-6666-666666666666"
ASSIGNMENT = "77777777-7777-7777-7777-777777777777"
GROUP = f"copyrit-{INSTANCE}"
GROUP_ID = f"/subscriptions/{SUBSCRIPTION}/resourceGroups/{GROUP}"
IDENTITY_ID = f"{GROUP_ID}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{GROUP}-identity"
MODEL_SCOPE = (
    f"/subscriptions/{SUBSCRIPTION}/resourceGroups/shared-model/providers/Microsoft.CognitiveServices/accounts/model"
)
ASSIGNMENT_ID = f"{MODEL_SCOPE}/providers/Microsoft.Authorization/roleAssignments/{ASSIGNMENT}"
MODEL_ROLE = "Cognitive Services OpenAI User"


def _record_creation(*, journal: deploy_instance._DeploymentJournal, args: list[str], response: object) -> None:
    operation = journal.begin(args)
    assert operation is not None
    journal.complete(
        operation=operation, result=subprocess.CompletedProcess(["az", *args], 0, json.dumps(response), "")
    )


@pytest.fixture
def journal(tmp_path: Path) -> deploy_instance._DeploymentJournal:
    result = deploy_instance._DeploymentJournal(
        path=tmp_path / "deployment.json", instance=INSTANCE, resource_group_id=GROUP_ID, retention_hours=24
    )
    result.bind_tenant(TENANT)
    _record_creation(journal=result, args=["group", "create", "--name", GROUP], response={"id": GROUP_ID})
    _record_creation(
        journal=result,
        args=["ad", "app", "create", "--display-name", f"CoPyRIT GUI ({INSTANCE})"],
        response={"id": APPLICATION, "appId": CLIENT},
    )
    _record_creation(
        journal=result,
        args=["ad", "sp", "create", "--id", CLIENT],
        response={"id": SERVICE_PRINCIPAL, "appId": CLIENT},
    )
    _record_creation(
        journal=result,
        args=["identity", "create", "--name", f"{GROUP}-identity", "--resource-group", GROUP],
        response={"id": IDENTITY_ID, "principalId": PRINCIPAL},
    )
    _record_creation(
        journal=result,
        args=[
            "role",
            "assignment",
            "create",
            "--assignee-object-id",
            PRINCIPAL,
            "--scope",
            MODEL_SCOPE,
            "--role",
            MODEL_ROLE,
        ],
        response={"id": ASSIGNMENT_ID, "scope": MODEL_SCOPE, "principalId": PRINCIPAL},
    )
    return result


class _FakeAzure:
    def __init__(self, *, journal: deploy_instance._DeploymentJournal, receipt: Path) -> None:
        self.journal = journal
        self.receipt = receipt
        self.commands: list[list[str]] = []
        self.group_present = True
        self.identity_present = True
        self.application_present = True
        self.principal_present = True
        self.role_present = True
        self.group_tags = deploy_instance._deployment_tags(instance=INSTANCE, owner="", expires_at=journal.expires_at)
        self.identity_tags = self.group_tags.copy()
        self.tenant = TENANT
        self.role_principal = PRINCIPAL
        self.role_scope = MODEL_SCOPE
        self.role_name = MODEL_ROLE
        self.role_assignment_id = ASSIGNMENT_ID
        self.duplicate_role = False
        self.app_object = APPLICATION
        self.sp_object = SERVICE_PRINCIPAL
        self.fail_prefix: list[str] | None = None
        self.fail_stderr = "ERROR: (AuthorizationFailed) Denied"
        self.retain_directory = False
        self.role_reappears = False

    def run(self, *, args: list[str], check: bool, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        assert check is False
        assert timeout is not None and 0 < timeout <= 60
        assert args[args.index("--subscription") + 1] == SUBSCRIPTION
        self.commands.append(args)
        if self.receipt.exists():
            lines = [json.loads(line) for line in self.receipt.read_text(encoding="utf-8").splitlines()]
            assert lines[-1]["event"] == "cli_request"
            assert lines[-1]["command"] == ["az", *args]
        if self.fail_prefix is not None and args[: len(self.fail_prefix)] == self.fail_prefix:
            return subprocess.CompletedProcess(["az", *args], 1, "", self.fail_stderr)
        value: object
        if args[:2] == ["account", "show"]:
            value = {"id": SUBSCRIPTION, "tenantId": self.tenant}
        elif args[:2] == ["group", "exists"]:
            value = self.group_present
        elif args[:2] == ["group", "show"]:
            value = {"id": GROUP_ID, "tags": self.group_tags}
        elif args[:2] == ["identity", "show"]:
            if not self.identity_present:
                return subprocess.CompletedProcess(["az", *args], 1, "", "ERROR: (ResourceNotFound) Identity absent")
            value = {"id": IDENTITY_ID, "principalId": PRINCIPAL, "tags": self.identity_tags}
        elif args[:3] == ["role", "assignment", "list"]:
            value = (
                [
                    {
                        "id": self.role_assignment_id,
                        "scope": self.role_scope,
                        "principalId": self.role_principal,
                        "roleDefinitionName": self.role_name,
                    }
                ]
                if self.role_present
                else []
            )
            if self.duplicate_role and isinstance(value, list) and value:
                duplicate = dict(value[0])
                duplicate["id"] = (
                    f"{MODEL_SCOPE}/providers/Microsoft.Authorization/roleAssignments/"
                    "88888888-8888-8888-8888-888888888888"
                )
                value.append(duplicate)
        elif args[:3] == ["ad", "app", "list"]:
            value = (
                [{"id": self.app_object, "appId": CLIENT, "displayName": f"CoPyRIT GUI ({INSTANCE})"}]
                if self.application_present
                else []
            )
        elif args[:3] == ["ad", "sp", "list"]:
            value = (
                [
                    {
                        "id": self.sp_object,
                        "appId": CLIENT,
                        "displayName": f"CoPyRIT GUI ({INSTANCE})",
                        "servicePrincipalType": "Application",
                    }
                ]
                if self.principal_present
                else []
            )
        elif args[:3] == ["role", "assignment", "delete"]:
            assert args[args.index("--ids") + 1] == ASSIGNMENT_ID
            self.role_present = False
            value = None
        elif args[:2] == ["group", "delete"]:
            assert not self.role_present
            assert "--no-wait" in args
            self.group_present = self.identity_present = False
            self.role_present = self.role_reappears
            value = None
        elif args[:3] == ["ad", "sp", "delete"]:
            assert args[args.index("--id") + 1] == SERVICE_PRINCIPAL
            self.principal_present = self.retain_directory
            value = None
        elif args[:3] == ["ad", "app", "delete"]:
            assert args[args.index("--id") + 1] == APPLICATION
            self.application_present = self.retain_directory
            value = None
        elif args[:2] == ["keyvault", "list-deleted"]:
            value = [
                {
                    "id": (
                        f"/subscriptions/{SUBSCRIPTION}/providers/Microsoft.KeyVault/"
                        f"locations/westus2/deletedVaults/{GROUP}-kv"
                    ),
                    "name": f"{GROUP}-kv",
                    "vaultId": f"{GROUP_ID}/providers/Microsoft.KeyVault/vaults/{GROUP}-kv",
                    "scheduledPurgeDate": "2027-01-01T00:00:00Z",
                }
            ]
        else:
            raise AssertionError(f"Unexpected Azure command: {args}")
        return subprocess.CompletedProcess(["az", *args], 0, json.dumps(value) if value is not None else "", "")


def _arguments(*, journal: Path, receipt: Path) -> list[str]:
    return [
        "--instance-name",
        INSTANCE,
        "--subscription",
        SUBSCRIPTION,
        "--resource-group-id",
        GROUP_ID,
        "--acknowledge-egress-ip-release",
        "--journal-file",
        str(journal),
        "--cleanup-receipt",
        str(receipt),
        "--yes",
    ]


def _delete_commands(commands: list[list[str]]) -> list[list[str]]:
    return [args for args in commands if "delete" in args[:3]]


def test_preview_cleanup_orders_exact_references_and_records_absence(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    original = journal.path.read_bytes()
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 0

    assert [args[:3] for args in _delete_commands(cloud.commands)] == [
        ["role", "assignment", "delete"],
        ["group", "delete", "--name"],
        ["ad", "sp", "delete"],
        ["ad", "app", "delete"],
    ]
    assert journal.path.read_bytes() == original
    events = [json.loads(line) for line in receipt.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "cleanup_verified"
    assert events[-1]["sandbox_data_plane_absence_claimed"] is False
    tombstone = next(event for event in events if event["event"] == "keyvault_soft_deletion_observation")
    assert tombstone["physical_purge_claimed"] is False
    assert tombstone["tombstones"][0]["vaultId"].startswith(GROUP_ID)


def test_preview_cleanup_read_only_inventory_does_not_delete(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "inventory.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main([*_arguments(journal=journal.path, receipt=receipt), "--dry-run"]) == 0
    assert not _delete_commands(cloud.commands)
    assert json.loads(receipt.read_text(encoding="utf-8").splitlines()[-1])["event"] == "cleanup_dry_run"


def test_preview_cleanup_recovers_when_group_and_identity_already_absent(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    cloud.group_present = cloud.identity_present = False
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 0
        cloud.receipt = tmp_path / "repeat.jsonl"
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=cloud.receipt)) == 0
    assert not any(args[:2] == ["group", "delete"] for args in cloud.commands)
    assert [args[:3] for args in _delete_commands(cloud.commands)].count(["role", "assignment", "delete"]) == 1


@pytest.mark.parametrize(
    "field",
    ["group_tags", "identity_tags", "tenant", "role_principal", "role_scope", "role_name", "app_object", "sp_object"],
)
def test_preview_cleanup_foreign_live_bindings_block_all_deletion(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path, field: str
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    if field.endswith("_tags"):
        setattr(cloud, field, {"Service": "foreign"})
    else:
        setattr(cloud, field, "88888888-8888-8888-8888-888888888888")
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not _delete_commands(cloud.commands)


@pytest.mark.parametrize(
    "prefix", [["group", "exists"], ["identity", "show"], ["role", "assignment", "list"], ["ad", "app", "list"]]
)
def test_preview_cleanup_permission_error_is_not_absence(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path, prefix: list[str], caplog: pytest.LogCaptureFixture
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    cloud.fail_prefix = prefix
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not _delete_commands(cloud.commands)
    assert "AuthorizationFailed" in caplog.text
    assert "cleanup_verified" not in receipt.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure", ["directory_remains", "role_reappears", "keyvault_denied"])
def test_preview_cleanup_failed_final_observation_does_not_claim_success(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path, failure: str
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    cloud.retain_directory = failure == "directory_remains"
    cloud.role_reappears = failure == "role_reappears"
    cloud.fail_prefix = ["keyvault", "list-deleted"] if failure == "keyvault_denied" else None
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert "cleanup_verified" not in receipt.read_text(encoding="utf-8")


@pytest.mark.parametrize("field", ["schema", "instance", "resource_group_id", "tenant_id"])
def test_preview_cleanup_invalid_journal_fails_before_cli(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path, field: str
) -> None:
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    document[field] = "foreign"
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    with patch.object(teardown_instance, "run_az", autospec=True) as cli:
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=tmp_path / "cleanup.jsonl")) == 1
    cli.assert_not_called()


def test_preview_cleanup_requires_creation_intent_before_deleting_live_group(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    document["operations"] = []
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not _delete_commands(cloud.commands)


def test_preview_cleanup_unbound_application_is_not_deleted_by_name(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    document["operations"] = document["operations"][:2]
    document["operations"][1].pop("returned_references")
    document["operations"][1]["status"] = "requested"
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    cloud.identity_present = cloud.principal_present = cloud.role_present = False
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not cloud.group_present
    assert cloud.application_present
    assert not any(args[:2] == ["ad", "app"] and "delete" in args for args in cloud.commands)
    assert "unbound_directory_candidates" in receipt.read_text(encoding="utf-8")
    assert "cleanup_verified" not in receipt.read_text(encoding="utf-8")


def test_preview_cleanup_requires_expiry_for_scheduled_deletion(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main([*_arguments(journal=journal.path, receipt=receipt), "--require-expired"]) == 1
    assert not _delete_commands(cloud.commands)

    expiry = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    document["expires_at"] = expiry
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    cloud.group_tags["ExpiresAt"] = expiry
    cloud.receipt = tmp_path / "expired.jsonl"
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert (
            teardown_instance.main([*_arguments(journal=journal.path, receipt=cloud.receipt), "--require-expired"]) == 0
        )


def test_preview_cleanup_timeout_is_persisted_as_failure(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=subprocess.TimeoutExpired("az", 60)):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert "cleanup_failed" in receipt.read_text(encoding="utf-8")
    assert "cleanup_verified" not in receipt.read_text(encoding="utf-8")


def test_preview_cleanup_receipt_is_never_overwritten(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    receipt.write_text("preserved\n", encoding="utf-8")
    with patch.object(teardown_instance, "run_az", autospec=True) as cli:
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    cli.assert_not_called()
    assert receipt.read_text(encoding="utf-8") == "preserved\n"


def test_journal_tenant_binding_cannot_change(journal: deploy_instance._DeploymentJournal) -> None:
    original = journal.path.read_bytes()
    with pytest.raises(RuntimeError, match="another tenant"):
        journal.bind_tenant("99999999-9999-9999-9999-999999999999")
    assert journal.path.read_bytes() == original


@pytest.mark.parametrize("missing", ["identity", "role"])
def test_preview_cleanup_accepts_nested_arm_creation_references(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path, missing: str
) -> None:
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    index = 3 if missing == "identity" else 4
    flat = document["operations"][index]["returned_references"][0]
    document["operations"][index]["returned_references"] = [
        {"id": flat["id"]},
        {key: value for key, value in flat.items() if key != "id"},
    ]
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 0


@pytest.mark.parametrize("confirmed", [True, False])
def test_preview_cleanup_duplicate_external_grant_is_not_deleted(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path, confirmed: bool
) -> None:
    if not confirmed:
        document = json.loads(journal.path.read_text(encoding="utf-8"))
        document["operations"][-1].pop("returned_references")
        journal.path.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    cloud.duplicate_role = True
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not _delete_commands(cloud.commands)


def test_preview_cleanup_unconfirmed_single_external_grant_can_be_bound(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    document["operations"][-1].pop("returned_references")
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 0


def test_preview_cleanup_unjournalled_external_grant_blocks_identity_deletion(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    document = json.loads(journal.path.read_text(encoding="utf-8"))
    document["operations"] = document["operations"][:-1]
    journal.path.write_text(json.dumps(document), encoding="utf-8")
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)
    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=cloud.run):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not _delete_commands(cloud.commands)


def test_preview_cleanup_identity_null_success_is_not_confirmed_absence(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    receipt = tmp_path / "cleanup.jsonl"
    cloud = _FakeAzure(journal=journal, receipt=receipt)

    def null_identity(
        *, args: list[str], check: bool, timeout: float | None = None
    ) -> subprocess.CompletedProcess[str]:
        response = cloud.run(args=args, check=check, timeout=timeout)
        if args[:2] == ["identity", "show"]:
            return subprocess.CompletedProcess(response.args, 0, "null", "")
        return response

    with patch.object(teardown_instance, "run_az", autospec=True, side_effect=null_identity):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=receipt)) == 1
    assert not _delete_commands(cloud.commands)


def test_preview_cleanup_rejects_other_subscription_before_cli(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    arguments = _arguments(journal=journal.path, receipt=tmp_path / "cleanup.jsonl")
    arguments[arguments.index("--subscription") + 1] = "88888888-8888-8888-8888-888888888888"
    with patch.object(teardown_instance, "run_az", autospec=True) as cli:
        assert teardown_instance.main(arguments) == 1
    cli.assert_not_called()


def test_preview_cleanup_deadline_expiry_prevents_cli_dispatch(
    *, journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    with (
        patch.object(teardown_instance.time, "monotonic", side_effect=[0, 1000]),
        patch.object(teardown_instance, "run_az", autospec=True) as cli,
    ):
        assert teardown_instance.main(_arguments(journal=journal.path, receipt=tmp_path / "cleanup.jsonl")) == 1
    cli.assert_not_called()


def test_bounded_windows_cli_uses_native_interpreter_not_shell(tmp_path: Path) -> None:
    interpreter = tmp_path / "python.exe"
    interpreter.touch()
    launcher = tmp_path / "wbin" / "az.cmd"
    with (
        patch.object(teardown_instance, "_SHELL", True),
        patch.object(teardown_instance.shutil, "which", return_value=str(launcher)),
        patch.object(teardown_instance.subprocess, "run", autospec=True) as process,
    ):
        teardown_instance.run_az(args=["version"], timeout=60)
    assert process.call_args.args[0] == [str(interpreter), "-IBm", "azure.cli", "version"]
    assert process.call_args.kwargs["shell"] is False
    assert process.call_args.kwargs["timeout"] == 60


def test_bounded_windows_cli_refuses_missing_native_interpreter(tmp_path: Path) -> None:
    with (
        patch.object(teardown_instance, "_SHELL", True),
        patch.object(teardown_instance.shutil, "which", return_value=str(tmp_path / "wbin" / "az.cmd")),
        patch.object(teardown_instance.subprocess, "run", autospec=True) as process,
    ):
        with pytest.raises(RuntimeError, match="native Python"):
            teardown_instance.run_az(args=["version"], timeout=60)
    process.assert_not_called()


def test_create_entra_app_binds_tenant_before_directory_mutation(
    *, journal: deploy_instance._DeploymentJournal
) -> None:
    token = deploy_instance._CURRENT_JOURNAL.set(journal)
    try:
        with (
            patch.object(
                deploy_instance,
                "run_az_json",
                side_effect=[TENANT, {"id": APPLICATION, "appId": CLIENT}, {"id": SERVICE_PRINCIPAL}],
            ) as cli_json,
            patch.object(deploy_instance, "run_az", autospec=True),
            patch.object(journal, "bind_tenant", wraps=journal.bind_tenant) as bind,
        ):
            result = deploy_instance.create_entra_app(display_name=f"CoPyRIT GUI ({INSTANCE})")
    finally:
        deploy_instance._CURRENT_JOURNAL.reset(token)
    assert result["tenant_id"] == TENANT
    bind.assert_called_once()
    assert cli_json.call_args_list[0].kwargs["args"] == ["account", "show", "--query", "tenantId"]
