# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for bound post-deployment journal resumption."""

import contextvars
import hashlib
import json
import subprocess
from contextlib import AbstractContextManager, nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from infra import deploy_instance, teardown_instance

if TYPE_CHECKING:
    from collections.abc import Callable

SUBSCRIPTION = "11111111-1111-1111-1111-111111111111"
TENANT = "22222222-2222-2222-2222-222222222222"
PRINCIPAL = "33333333-3333-3333-3333-333333333333"
GROUP_ID = f"/subscriptions/{SUBSCRIPTION}/resourceGroups/copyrit-preview"
IDENTITY_ID = f"{GROUP_ID}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/copyrit-preview-identity"
STORAGE_ID = f"{GROUP_ID}/providers/Microsoft.Storage/storageAccounts/copyritpreviewsa"
MODEL_SCOPE = f"/subscriptions/{SUBSCRIPTION}/resourceGroups/model/providers/Microsoft.CognitiveServices/accounts/model"
ARTIFACT_URI = "https://copyritpreviewsa.blob.core.windows.net/private-source/immutable.tar.gz"
ARTIFACT_SHA = "a" * 64


@pytest.fixture
def provisioned_journal(tmp_path: Path) -> deploy_instance._DeploymentJournal:
    journal = deploy_instance._DeploymentJournal(
        path=tmp_path / "journal.json",
        instance="preview",
        resource_group_id=GROUP_ID,
        retention_hours=24,
    )
    journal.bind_tenant(TENANT)
    for args, response in [
        (["group", "create", "--name", "copyrit-preview"], {"id": GROUP_ID}),
        (
            ["identity", "create", "--name", "copyrit-preview-identity", "--resource-group", "copyrit-preview"],
            {"id": IDENTITY_ID, "principalId": PRINCIPAL},
        ),
        (
            ["storage", "account", "create", "--name", "copyritpreviewsa", "--resource-group", "copyrit-preview"],
            {"id": STORAGE_ID},
        ),
    ]:
        operation = journal.begin(args)
        assert operation is not None
        journal.complete(operation=operation, result=subprocess.CompletedProcess(["az"], 0, json.dumps(response), ""))
    journal.finish("provisioned_manual_sql_and_runtime_validation_required")
    return journal


def _resume(journal: deploy_instance._DeploymentJournal) -> AbstractContextManager[deploy_instance._DeploymentJournal]:
    return deploy_instance.resume_deployment_journal(
        path=journal.path,
        expected_sha256=hashlib.sha256(journal.path.read_bytes()).hexdigest(),
        instance="preview",
        resource_group_id=GROUP_ID,
        tenant_id=TENANT,
    )


def _grant_args() -> list[str]:
    return [
        "role",
        "assignment",
        "create",
        "--assignee-object-id",
        PRINCIPAL,
        "--assignee-principal-type",
        "ServicePrincipal",
        "--role",
        "Cognitive Services OpenAI User",
        "--scope",
        MODEL_SCOPE,
        "--subscription",
        SUBSCRIPTION,
    ]


@pytest.fixture(params=["normal", "exception"])
def resumed_after_exit(
    *, provisioned_journal: deploy_instance._DeploymentJournal, request: pytest.FixtureRequest
) -> tuple[deploy_instance._DeploymentJournal, int, int]:
    exit_check = (
        pytest.raises(ValueError, match="owned context exit") if request.param == "exception" else nullcontext()
    )
    result: tuple[deploy_instance._DeploymentJournal, int, int] | None = None
    with exit_check, _resume(provisioned_journal) as retained:
        grant_operation = retained.begin(_grant_args())
        assert grant_operation is not None
        artifact_operation = retained.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
        result = retained, grant_operation, artifact_operation
        if request.param == "exception":
            raise ValueError("owned context exit")
    assert result is not None
    assert deploy_instance._CURRENT_JOURNAL.get() is None
    retained = result[0]
    assert not retained.path.with_name(f"{retained.path.name}.resume-lock").exists()
    return result


@pytest.mark.parametrize(
    "method", ["begin", "complete", "finish", "bind_tenant", "begin_artifact", "verify_artifact", "_write"]
)
def test_retained_resume_denies_every_mutation_after_context_exit(
    *, resumed_after_exit: tuple[deploy_instance._DeploymentJournal, int, int], method: str
) -> None:
    retained, grant_operation, artifact_operation = resumed_after_exit
    actions: dict[str, Callable[[], object]] = {
        "begin": lambda: retained.begin(_grant_args()),
        "complete": lambda: retained.complete(
            operation=grant_operation, result=subprocess.CompletedProcess(["az"], 0, "{}", "")
        ),
        "finish": lambda: retained.finish("success"),
        "bind_tenant": lambda: retained.bind_tenant(TENANT),
        "begin_artifact": lambda: retained.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100),
        "verify_artifact": lambda: retained.verify_artifact(
            operation=artifact_operation, sha256=ARTIFACT_SHA, size_bytes=100, etag='"etag"'
        ),
        "_write": retained._write,
    }
    previous_bytes = retained.path.read_bytes()
    previous_document = json.dumps(retained._document, sort_keys=True)
    with patch.object(deploy_instance.subprocess, "run", autospec=True) as run:
        with pytest.raises(RuntimeError, match="active held-lock context"):
            actions[method]()
    run.assert_not_called()
    assert retained.path.read_bytes() == previous_bytes
    assert json.dumps(retained._document, sort_keys=True) == previous_document


@pytest.mark.parametrize("command", ["run_az", "run_az_json"])
def test_retained_resume_cannot_reactivate_cli_after_context_exit(
    *, resumed_after_exit: tuple[deploy_instance._DeploymentJournal, int, int], command: str
) -> None:
    retained = resumed_after_exit[0]
    before = retained.path.read_bytes()
    token = deploy_instance._CURRENT_JOURNAL.set(retained)
    try:
        with patch.object(deploy_instance.subprocess, "run", autospec=True) as run:
            run.return_value = subprocess.CompletedProcess(["az"], 0, "{}", "")
            with pytest.raises(RuntimeError, match="active held-lock context"):
                getattr(deploy_instance, command)(args=_grant_args())
    finally:
        deploy_instance._CURRENT_JOURNAL.reset(token)
    run.assert_not_called()
    assert retained.path.read_bytes() == before


def test_retained_resume_denial_precedes_artifact_sdk_call(
    resumed_after_exit: tuple[deploy_instance._DeploymentJournal, int, int],
) -> None:
    retained = resumed_after_exit[0]
    before = retained.path.read_bytes()
    upload_sdk = MagicMock(spec=["__call__"])
    with pytest.raises(RuntimeError, match="active held-lock context"):
        operation = retained.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
        upload_sdk(operation=operation)
    upload_sdk.assert_not_called()
    assert retained.path.read_bytes() == before


def test_copied_resume_context_cannot_dispatch_after_exit(
    provisioned_journal: deploy_instance._DeploymentJournal,
) -> None:
    with _resume(provisioned_journal):
        copied = contextvars.copy_context()
    before = provisioned_journal.path.read_bytes()
    with patch.object(deploy_instance.subprocess, "run", autospec=True) as run:
        with pytest.raises(RuntimeError, match="active held-lock context"):
            copied.run(deploy_instance.run_az, args=_grant_args())
    run.assert_not_called()
    assert provisioned_journal.path.read_bytes() == before


@pytest.mark.parametrize("invalid_context", ["unset", "closed_lock"])
def test_resumed_mutation_requires_active_context_and_open_lock(
    *, provisioned_journal: deploy_instance._DeploymentJournal, invalid_context: str
) -> None:
    with _resume(provisioned_journal) as resumed:
        before = resumed.path.read_bytes()
        if invalid_context == "closed_lock":
            assert resumed._resume_lock is not None
            resumed._resume_lock.close()
        token = deploy_instance._CURRENT_JOURNAL.set(None) if invalid_context == "unset" else None
        try:
            with pytest.raises(RuntimeError, match="active held-lock context"):
                resumed.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
        finally:
            if token is not None:
                deploy_instance._CURRENT_JOURNAL.reset(token)
        assert resumed.path.read_bytes() == before


def test_old_resume_cannot_write_in_next_session(*, provisioned_journal: deploy_instance._DeploymentJournal) -> None:
    with _resume(provisioned_journal) as old:
        pass
    with _resume(provisioned_journal) as current:
        before = current.path.read_bytes()
        with pytest.raises(RuntimeError, match="active held-lock context"):
            old.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
        assert current.path.read_bytes() == before
        current.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)


def test_resume_records_actual_grant_before_cli_without_changing_lifetime(
    provisioned_journal: deploy_instance._DeploymentJournal,
) -> None:
    before = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))
    assignment_id = f"{MODEL_SCOPE}/providers/Microsoft.Authorization/roleAssignments/{TENANT}"

    def grant_cli(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        pending = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))["operations"][-1]
        assert pending["status"] == "requested"
        assert pending["targets"]["--scope"] == MODEL_SCOPE
        return subprocess.CompletedProcess(command, 0, json.dumps({"id": assignment_id, "principalId": PRINCIPAL}), "")

    with _resume(provisioned_journal) as resumed:
        assert deploy_instance._CURRENT_JOURNAL.get() is resumed
        with patch.object(deploy_instance.subprocess, "run", side_effect=grant_cli):
            deploy_instance.run_az(args=_grant_args())
    after = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))
    assert {key: value for key, value in before.items() if key != "operations"} == {
        key: value for key, value in after.items() if key != "operations"
    }
    assert after["operations"][:-1] == before["operations"]
    assert after["operations"][-1]["returned_references"][0]["id"] == assignment_id
    assert deploy_instance._CURRENT_JOURNAL.get() is None
    assert list(provisioned_journal.path.parent.iterdir()) == [provisioned_journal.path]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema", "foreign/v1"),
        ("instance", "foreign"),
        ("resource_group_id", GROUP_ID + "-other"),
        ("tenant_id", PRINCIPAL),
        ("status", "provisioning"),
        ("expires_at", None),
        ("operations", [None]),
    ],
)
def test_resume_rejects_mismatched_binding_and_preserves_document(
    provisioned_journal: deploy_instance._DeploymentJournal, *, field: str, value: object
) -> None:
    document = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))
    document[field] = value
    provisioned_journal.path.write_text(json.dumps(document), encoding="utf-8")
    before = provisioned_journal.path.read_bytes()
    with pytest.raises(RuntimeError):
        with _resume(provisioned_journal):
            pytest.fail("Mismatched journal must not open")
    assert provisioned_journal.path.read_bytes() == before
    assert deploy_instance._CURRENT_JOURNAL.get() is None
    assert list(provisioned_journal.path.parent.iterdir()) == [provisioned_journal.path]


def test_resume_refuses_stale_sha_without_azure(provisioned_journal: deploy_instance._DeploymentJournal) -> None:
    with patch.object(deploy_instance.subprocess, "run", autospec=True) as run:
        with pytest.raises(RuntimeError, match="SHA256"):
            with deploy_instance.resume_deployment_journal(
                path=provisioned_journal.path,
                expected_sha256="0" * 64,
                instance="preview",
                resource_group_id=GROUP_ID,
                tenant_id=TENANT,
            ):
                pytest.fail("Stale journal must not open")
    run.assert_not_called()


def test_resume_lock_refuses_other_writer(provisioned_journal: deploy_instance._DeploymentJournal) -> None:
    lock_path = provisioned_journal.path.with_name("journal.json.resume-lock")
    lock_path.write_text("existing owner", encoding="utf-8")
    with pytest.raises(FileExistsError):
        with _resume(provisioned_journal):
            pytest.fail("Existing lock must not be replaced")
    assert lock_path.read_text(encoding="utf-8") == "existing owner"


def test_resume_context_cannot_nest_and_releases_after_failure(
    provisioned_journal: deploy_instance._DeploymentJournal,
) -> None:
    with pytest.raises(ValueError, match="upload failed"):
        with _resume(provisioned_journal):
            with pytest.raises(RuntimeError, match="already active"):
                with _resume(provisioned_journal):
                    pytest.fail("Nested contexts must fail")
            raise ValueError("upload failed")
    assert deploy_instance._CURRENT_JOURNAL.get() is None
    with _resume(provisioned_journal):
        pass


def test_resumed_write_does_not_overwrite_external_change(
    provisioned_journal: deploy_instance._DeploymentJournal,
) -> None:
    with _resume(provisioned_journal) as resumed:
        original = provisioned_journal.path.read_bytes()
        changed = original + b"\n"
        provisioned_journal.path.write_bytes(changed)
        with pytest.raises(RuntimeError, match="changed outside"):
            resumed.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
    assert provisioned_journal.path.read_bytes() == changed


@pytest.mark.parametrize("when", ["open", "intent"])
def test_expiry_denies_new_operations_without_rewriting_deadline(
    provisioned_journal: deploy_instance._DeploymentJournal, when: str
) -> None:
    document = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))
    expiry = datetime.fromisoformat(document["expires_at"])
    if when == "open":
        document["expires_at"] = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        provisioned_journal.path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(RuntimeError, match="expiry"):
            with _resume(provisioned_journal):
                pytest.fail("Expired preview must not open")
    else:
        with _resume(provisioned_journal) as resumed:
            with patch.object(deploy_instance, "datetime") as clock:
                clock.fromisoformat.return_value = expiry
                clock.now.return_value = expiry + timedelta(seconds=1)
                with pytest.raises(RuntimeError, match="expiry"):
                    resumed.begin(_grant_args())
    assert json.loads(provisioned_journal.path.read_text(encoding="utf-8"))["expires_at"] == document["expires_at"]


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("--assignee-object-id", TENANT),
        ("--assignee-principal-type", "User"),
        ("--subscription", TENANT),
        ("--scope", f"/subscriptions/{TENANT}/resourceGroups/model"),
        ("--role", ""),
    ],
)
def test_resumed_grant_requires_owned_identity_and_exact_subscription(
    provisioned_journal: deploy_instance._DeploymentJournal, *, option: str, value: str
) -> None:
    args = _grant_args()
    args[args.index(option) + 1] = value
    with _resume(provisioned_journal):
        with patch.object(deploy_instance.subprocess, "run", autospec=True) as run:
            with pytest.raises(RuntimeError, match="exact preview identity"):
                deploy_instance.run_az(args=args)
    run.assert_not_called()


@pytest.mark.parametrize(
    "args",
    [
        ["group", "create", "--name", "copyrit-other"],
        ["ad", "app", "create", "--display-name", "other"],
        ["rest", "--method", "PATCH", "--url", "https://graph.microsoft.com/v1.0/applications/id"],
        ["containerapp", "update", "--name", "shared"],
        ["storage", "blob", "upload", "--account-name", "shared"],
    ],
)
def test_resumed_cli_rejects_unbound_mutations_before_execution(
    provisioned_journal: deploy_instance._DeploymentJournal, args: list[str]
) -> None:
    with _resume(provisioned_journal):
        with patch.object(deploy_instance.subprocess, "run", autospec=True) as run:
            with pytest.raises(RuntimeError, match="limited to"):
                deploy_instance.run_az(args=args)
    run.assert_not_called()


def test_artifact_intent_is_not_verified_until_exact_readback(
    provisioned_journal: deploy_instance._DeploymentJournal,
) -> None:
    with _resume(provisioned_journal) as resumed:
        operation = resumed.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
        pending = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))["operations"][operation]
        assert pending["status"] == "requested"
        with pytest.raises(RuntimeError, match="readback"):
            resumed.verify_artifact(operation=operation, sha256="b" * 64, size_bytes=100, etag='"version1"')
        resumed.verify_artifact(operation=operation, sha256=ARTIFACT_SHA, size_bytes=100, etag='"version1"')
    event = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))["operations"][operation]
    assert event["status"] == "verified"
    assert event["readback"] == {"sha256": ARTIFACT_SHA, "size_bytes": 100, "etag": '"version1"'}


@pytest.mark.parametrize("etag", ["", "a" * 1025, "etag\nother", "etag\rother"])
def test_artifact_rejects_invalid_readback_etag(
    provisioned_journal: deploy_instance._DeploymentJournal, etag: str
) -> None:
    with _resume(provisioned_journal) as resumed:
        operation = resumed.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=1)
        with pytest.raises(RuntimeError, match="readback"):
            resumed.verify_artifact(operation=operation, sha256=ARTIFACT_SHA, size_bytes=1, etag=etag)
        with pytest.raises(RuntimeError, match="readback"):
            resumed.verify_artifact(operation=operation, sha256=ARTIFACT_SHA, size_bytes=True, etag='"etag"')
    event = json.loads(provisioned_journal.path.read_text(encoding="utf-8"))["operations"][operation]
    assert event["status"] == "requested"


@pytest.mark.parametrize(
    ("uri", "sha", "size"),
    [
        ("https://foreign.blob.core.windows.net/source/file", ARTIFACT_SHA, 100),
        ("https://copyritpreviewsa.blob.core.chinacloudapi.cn/source/file", ARTIFACT_SHA, 100),
        (ARTIFACT_URI + "?sig=SECRET", ARTIFACT_SHA, 100),
        (ARTIFACT_URI, "A" * 64, 100),
        (ARTIFACT_URI, ARTIFACT_SHA, 0),
        (ARTIFACT_URI, ARTIFACT_SHA, True),
    ],
)
def test_artifact_rejects_foreign_secret_or_invalid_metadata(
    provisioned_journal: deploy_instance._DeploymentJournal, *, uri: str, sha: str, size: int
) -> None:
    before = provisioned_journal.path.read_bytes()
    with _resume(provisioned_journal) as resumed:
        with pytest.raises((RuntimeError, ValueError)):
            resumed.begin_artifact(blob_uri=uri, sha256=sha, size_bytes=size)
    assert provisioned_journal.path.read_bytes() == before
    assert b"SECRET" not in before


def test_resumed_finish_cannot_rewrite_provisioning_status(
    provisioned_journal: deploy_instance._DeploymentJournal,
) -> None:
    before = provisioned_journal.path.read_bytes()
    with _resume(provisioned_journal) as resumed:
        with pytest.raises(RuntimeError, match="cannot rewrite"):
            resumed.finish("success")
    assert provisioned_journal.path.read_bytes() == before


@pytest.mark.parametrize("when", ["open", "write"])
def test_resume_bounds_journal_reads(provisioned_journal: deploy_instance._DeploymentJournal, when: str) -> None:
    size = provisioned_journal.path.stat().st_size
    if when == "open":
        with patch.object(deploy_instance._DeploymentJournal, "_MAX_BYTES", size - 1):
            with pytest.raises(RuntimeError, match="2 MiB"):
                with _resume(provisioned_journal):
                    pytest.fail("Oversized journal must not open")
    else:
        with _resume(provisioned_journal) as resumed:
            provisioned_journal.path.write_bytes(b"x" * (2_097_152 + 1))
            with pytest.raises(RuntimeError, match="2 MiB"):
                resumed.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)


def test_journal_resume_and_cleanup_share_actual_grant_reference(
    provisioned_journal: deploy_instance._DeploymentJournal, tmp_path: Path
) -> None:
    assignment_id = f"{MODEL_SCOPE}/providers/Microsoft.Authorization/roleAssignments/{TENANT}"
    with _resume(provisioned_journal) as resumed:
        operation = resumed.begin(_grant_args())
        assert operation is not None
        resumed.complete(
            operation=operation,
            result=subprocess.CompletedProcess(
                ["az"], 0, json.dumps({"id": assignment_id, "principalId": PRINCIPAL}), ""
            ),
        )
        resumed.begin_artifact(blob_uri=ARTIFACT_URI, sha256=ARTIFACT_SHA, size_bytes=100)
    arguments = teardown_instance.parse_args(
        [
            "--instance-name",
            "preview",
            "--subscription",
            SUBSCRIPTION,
            "--resource-group-id",
            GROUP_ID,
            "--journal-file",
            str(provisioned_journal.path),
            "--cleanup-receipt",
            str(tmp_path / "cleanup.jsonl"),
            "--acknowledge-egress-ip-release",
            "--dry-run",
        ]
    )
    cleanup = teardown_instance._PreviewCleanup(arguments)
    assert cleanup.principal_id == PRINCIPAL
    assert cleanup.role_intents[0]["assignmentId"] == assignment_id
    assert cleanup.document["expires_at"] == provisioned_journal.expires_at
