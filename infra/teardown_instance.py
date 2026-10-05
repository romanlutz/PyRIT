# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
CoPyRIT GUI — Tear down an isolated instance.

Removes all Azure resources for an instance deployed by deploy_instance.py.
Entra resources (app registration, service principal) must be deleted separately
since they live outside the resource group.

Usage:
    python infra/teardown_instance.py --instance-name partners-demo \\
        --subscription "<subscription-id>" \
        --resource-group-id "/subscriptions/<subscription-id>/resourceGroups/copyrit-partners-demo" \
        --acknowledge-egress-ip-release

    # Include Entra cleanup:
    python infra/teardown_instance.py --instance-name partners-demo \\
        --subscription "<subscription-id>" \
        --resource-group-id "/subscriptions/<subscription-id>/resourceGroups/copyrit-partners-demo" \
        --acknowledge-egress-ip-release \
        --delete-entra-app --entra-app-id "<application-client-id>"

"""

import argparse
import hashlib
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TextIO, cast
from uuid import UUID

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# On Windows, az CLI is a .cmd script that requires shell=True for subprocess to find it.
_SHELL = platform.system() == "Windows"
_INSTANCE_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,11}[a-z0-9])?$")
_RESOURCE_GROUP_ID_RE = re.compile(
    r"^/subscriptions/([^/]+)/resourceGroups/([^/]+)$",
    re.IGNORECASE,
)
_OWNERSHIP_TAGS: dict[str, str] = {
    "Service": "pyrit-gui",
    "ManagedBy": "infra/deploy_instance.py",
}


def run_az(
    *,
    args: list[str],
    capture: bool = True,
    check: bool = True,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """
    Run an Azure CLI command.

    Args:
        args (list[str]): The az CLI arguments (without the leading 'az').
        capture (bool): Whether to capture stdout/stderr. Defaults to True.
        check (bool): Whether to raise on non-zero exit. Defaults to True.
        timeout (float | None): Optional subprocess bound in seconds.

    Returns:
        subprocess.CompletedProcess[str]: The completed process.

    Raises:
        subprocess.CalledProcessError: If the command fails and check is True.
    """
    cmd = ["az"] + args
    shell = _SHELL
    if timeout is not None and _SHELL:
        launcher = shutil.which("az")
        if launcher is None:
            raise RuntimeError("Azure CLI launcher is not installed")
        if Path(launcher).suffix.casefold() in (".cmd", ".bat"):
            interpreter = Path(launcher).parent.parent / "python.exe"
            if not interpreter.is_file():
                raise RuntimeError("Bounded cleanup requires Azure CLI's native Python, not a shell-process timeout")
            cmd = [str(interpreter), "-IBm", "azure.cli", *args]
        else:
            cmd = [launcher, *args]
        shell = False
    logger.debug("Running: %s", " ".join(cmd))
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=check,
        shell=shell,
        timeout=timeout,
    )


def _expect_json_object(value: object, *, context: str) -> dict[str, object]:
    """Require a JSON object with string keys at an Azure CLI response boundary."""
    if not isinstance(value, dict):
        raise RuntimeError(f"Azure CLI returned invalid {context} data")
    return cast("dict[str, object]", value)


def _expect_json_array(value: object, *, context: str) -> list[object]:
    """Require a JSON array at an Azure CLI response boundary."""
    if not isinstance(value, list):
        raise RuntimeError(f"Azure CLI returned invalid {context} data")
    return cast("list[object]", value)


def _expect_string(value: object, *, context: str) -> str:
    """Require a nonempty string at an Azure CLI response boundary."""
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"Azure CLI returned invalid {context} data")
    return value


def run_az_json(*, args: list[str]) -> object | None:
    """
    Run an Azure CLI command and parse JSON output.

    Args:
        args (list[str]): The az CLI arguments (without the leading 'az').

    Returns:
        object | None: The parsed JSON output, or None on command failure.
    """
    result = run_az(args=args + ["-o", "json"], check=False)
    if result.returncode != 0:
        return None
    parsed: object = json.loads(result.stdout)
    return parsed


class _AzureAbsence(Enum):
    CONFIRMED = "confirmed_resource_absence"


class _PreviewCleanup:
    """Clean only journal-bound preview references and retain independent observations."""

    def __init__(self, arguments: argparse.Namespace) -> None:
        self.arguments = arguments
        if arguments.journal_file.stat().st_size > 2_097_152:
            raise RuntimeError("Deployment journal exceeds the 2 MiB bound")
        raw = arguments.journal_file.read_bytes()
        if len(raw) > 2_097_152:
            raise RuntimeError("Deployment journal exceeds the 2 MiB bound")
        self.document = _expect_json_object(json.loads(raw), context="deployment journal")
        if (
            self.document.get("schema") != "copyrit-deployment-journal/v1"
            or self.document.get("instance") != arguments.instance_name
            or self.document.get("resource_group_id") != arguments.resource_group_id
        ):
            raise RuntimeError("Journal does not bind the exact requested instance and resource group")
        match = _RESOURCE_GROUP_ID_RE.fullmatch(arguments.resource_group_id)
        if match is None or match[2] != f"copyrit-{arguments.instance_name}":
            raise RuntimeError("Journal resource group is not the canonical instance resource group")
        self.subscription = str(UUID(match[1]))
        if str(UUID(arguments.subscription)) != self.subscription:
            raise RuntimeError("--subscription does not match the exact journalled subscription ID")
        self.group = match[2]
        self.group_id = arguments.resource_group_id
        self.identity_id = (
            f"{self.group_id}/providers/Microsoft.ManagedIdentity/userAssignedIdentities/{self.group}-identity"
        )
        self.app_name = f"CoPyRIT GUI ({arguments.instance_name})"
        self.operations = [
            _expect_json_object(value, context="journal operation")
            for value in _expect_json_array(self.document.get("operations"), context="journal operations")
        ]
        self.journal_sha256 = hashlib.sha256(raw).hexdigest()
        self.deadline = time.monotonic() + arguments.cleanup_timeout_seconds
        self.receipt: TextIO | None = None
        self.principal_id: str | None = None
        self.application: dict[str, str] | None = None
        self.service_principal_id: str | None = None
        self.group_creation_requested = False
        self.unbound_application_request = False
        self.directory_blocker: str | None = None
        self.role_intents: list[dict[str, str]] = []
        self._load_references()
        tenant = self.document.get("tenant_id")
        self.tenant_id = self._uuid(tenant, context="journal tenant") if tenant is not None else None
        if (self.application is not None or self.unbound_application_request) and self.tenant_id is None:
            raise RuntimeError("Directory cleanup requires a tenant-bound journal, not an inferred tenant")

    def _load_references(self) -> None:
        for operation in self.operations:
            command = _expect_json_array(operation.get("command"), context="journal command")
            targets = _expect_json_object(operation.get("targets"), context="journal targets")
            references = [
                _expect_json_object(value, context="journal returned reference")
                for value in _expect_json_array(operation.get("returned_references", []), context="returned references")
            ]
            if command == ["az", "group", "create"]:
                if targets.get("--name") != self.group:
                    raise RuntimeError("Journal contains a foreign resource-group creation")
                if any(str(reference.get("id", "")).casefold() != self.group_id.casefold() for reference in references):
                    raise RuntimeError("Journal resource-group creation returned a foreign ID")
                self.group_creation_requested = True
            elif command == ["az", "identity", "create"]:
                if targets.get("--name") != f"{self.group}-identity" or targets.get("--resource-group") != self.group:
                    raise RuntimeError("Journal contains a foreign identity creation")
                identity_ids = [reference["id"] for reference in references if "id" in reference]
                if any(str(value).casefold() != self.identity_id.casefold() for value in identity_ids):
                    raise RuntimeError("Journal identity reference does not match its owned resource")
                for reference in references:
                    if "principalId" not in reference:
                        continue
                    if not identity_ids:
                        raise RuntimeError("Journal principal lacks its owned managed-identity resource reference")
                    principal = self._uuid(reference["principalId"], context="managed identity principal")
                    if self.principal_id is not None and self.principal_id != principal:
                        raise RuntimeError("Journal contains conflicting managed identity principals")
                    self.principal_id = principal
            elif command == ["az", "ad", "app"]:
                if targets.get("--display-name") != self.app_name:
                    raise RuntimeError("Journal contains a foreign application creation")
                applications = [reference for reference in references if "appId" in reference]
                if len(applications) > 1:
                    raise RuntimeError("Journal contains ambiguous application creation references")
                if not applications:
                    self.unbound_application_request = True
                else:
                    application = {
                        "id": self._uuid(applications[0].get("id"), context="application object"),
                        "appId": self._uuid(applications[0]["appId"], context="application client"),
                    }
                    if self.application is not None and self.application != application:
                        raise RuntimeError("Journal contains conflicting application identities")
                    self.application = application
            elif command == ["az", "ad", "sp"]:
                if self.application is None:
                    raise RuntimeError("Service-principal mutation is not bound to an application creation")
                for reference in references:
                    if "appId" not in reference:
                        continue
                    if reference["appId"] != self.application["appId"]:
                        raise RuntimeError("Journal contains a foreign service-principal application")
                    principal = self._uuid(reference.get("id"), context="service principal object")
                    if self.service_principal_id is not None and self.service_principal_id != principal:
                        raise RuntimeError("Journal contains conflicting service-principal identities")
                    self.service_principal_id = principal
                target = self._uuid(targets.get("--id"), context="service-principal mutation target")
                if target != self.application["appId"]:
                    if self.service_principal_id is not None and target != self.service_principal_id:
                        raise RuntimeError("Service-principal mutation target differs from its creation reference")
                    self.service_principal_id = target
            elif command == ["az", "role", "assignment"]:
                scope = _expect_string(targets.get("--scope"), context="role scope")
                if not scope.casefold().startswith(f"/subscriptions/{self.subscription}/".casefold()):
                    raise RuntimeError("Journal role scope belongs to another subscription")
                intent = {
                    "scope": scope,
                    "principalId": self._uuid(targets.get("--assignee-object-id"), context="role principal"),
                    "role": _expect_string(targets.get("--role"), context="role name"),
                    "assignmentId": "",
                }
                prefix = f"{scope}/providers/Microsoft.Authorization/roleAssignments/"
                for reference in references:
                    if reference.get("principalId") not in (None, intent["principalId"]):
                        raise RuntimeError("Role creation reference contains a foreign principal")
                    if "scope" in reference and str(reference["scope"]).casefold() != scope.casefold():
                        raise RuntimeError("Role creation reference contains a foreign scope")
                    if "id" not in reference:
                        continue
                    assignment_id = _expect_string(reference["id"], context="role creation reference")
                    if not assignment_id.casefold().startswith(prefix.casefold()):
                        raise RuntimeError("Role creation reference is outside its requested scope")
                    self._uuid(assignment_id[len(prefix) :], context="role assignment name")
                    if intent["assignmentId"] and intent["assignmentId"] != assignment_id:
                        raise RuntimeError("Role creation returned conflicting assignment IDs")
                    intent["assignmentId"] = assignment_id
                self.role_intents.append(intent)

    @staticmethod
    def _uuid(value: object, *, context: str) -> str:
        return str(UUID(_expect_string(value, context=context)))

    def _record(self, *, event: str, **fields: object) -> None:
        if self.receipt is None:
            return
        self.receipt.write(json.dumps({"event": event, "at": datetime.now(UTC).isoformat(), **fields}) + "\n")
        self.receipt.flush()
        os.fsync(self.receipt.fileno())

    def _call(self, *, args: list[str], missing_codes: tuple[str, ...] = ()) -> object | None:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("Bounded preview cleanup deadline expired")
        command = [*args, "--subscription", self.subscription, "--only-show-errors", "-o", "json"]
        self._record(event="cli_request", command=["az", *command])
        result = run_az(args=command, check=False, timeout=min(60, remaining))
        if result.returncode != 0:
            self._record(
                event="cli_failure",
                exit_code=result.returncode,
                stderr_sha256=hashlib.sha256(result.stderr.encode("utf-8")).hexdigest(),
            )
            for code in missing_codes:
                if re.search(rf"\({re.escape(code)}\)", result.stderr):
                    self._record(event="resource_absent", command=["az", *command], code=code)
                    return _AzureAbsence.CONFIRMED
            raise subprocess.CalledProcessError(
                result.returncode, ["az", *command], output=result.stdout, stderr=result.stderr
            )
        self._record(event="cli_success", command=["az", *command])
        return json.loads(result.stdout) if result.stdout.strip() else None

    def _group_exists(self) -> bool:
        value = self._call(args=["group", "exists", "--name", self.group])
        if not isinstance(value, bool):
            raise RuntimeError("Azure CLI returned invalid resource-group absence data")
        self._record(event="resource_group_observation", exists=value, resource_group_id=self.group_id)
        return value

    def _validate_group(self) -> bool:
        exists = self._group_exists()
        if exists:
            if not self.group_creation_requested:
                raise RuntimeError("Live resource group has no creation request in this journal")
            group = _expect_json_object(
                self._call(args=["group", "show", "--name", self.group, "--query", "{id:id,tags:tags}"]),
                context="resource group",
            )
            if str(group.get("id", "")).casefold() != self.group_id.casefold():
                raise RuntimeError("Live resource group does not match journal")
            tags = _expect_json_object(group.get("tags"), context="ownership tags")
            expected_tags = {**_OWNERSHIP_TAGS, "Instance": self.arguments.instance_name}
            if any(tags.get(key) != value for key, value in expected_tags.items()):
                raise RuntimeError("Live resource group ownership tags do not match the deployment journal")
            if self.document.get("expires_at") and tags.get("ExpiresAt") != self.document["expires_at"]:
                raise RuntimeError("Live resource group expiry does not match journal")
        if self.arguments.require_expired:
            expiry = datetime.fromisoformat(_expect_string(self.document.get("expires_at"), context="preview expiry"))
            if expiry.tzinfo is None or datetime.now(UTC) < expiry:
                raise RuntimeError("Preview has not reached its absolute journalled UTC expiry")
        return exists

    def _validate_identity(self) -> None:
        identity_value = self._call(
            args=[
                "identity",
                "show",
                "--ids",
                self.identity_id,
                "--query",
                "{id:id,principalId:principalId,tags:tags}",
            ],
            missing_codes=("ResourceNotFound", "ResourceGroupNotFound"),
        )
        if identity_value is _AzureAbsence.CONFIRMED:
            return
        identity = _expect_json_object(identity_value, context="managed identity")
        if str(identity.get("id", "")).casefold() != self.identity_id.casefold():
            raise RuntimeError("Azure returned a foreign managed identity")
        tags = _expect_json_object(identity.get("tags"), context="managed identity tags")
        expected_tags = {**_OWNERSHIP_TAGS, "Instance": self.arguments.instance_name}
        if any(tags.get(key) != value for key, value in expected_tags.items()):
            raise RuntimeError("Managed identity is not owned by this instance")
        principal = self._uuid(identity.get("principalId"), context="managed identity principal")
        if self.principal_id is not None and principal != self.principal_id:
            raise RuntimeError("Live managed identity principal does not match its creation reference")
        self.principal_id = principal

    def _role_assignments(self) -> list[dict[str, str]]:
        for intent in self.role_intents:
            if intent["principalId"] != self.principal_id:
                raise RuntimeError("Role creation is not bound to the instance managed identity")
        if self.principal_id is None:
            return []
        values = _expect_json_array(
            self._call(
                args=[
                    "role",
                    "assignment",
                    "list",
                    "--all",
                    "--assignee-object-id",
                    self.principal_id,
                    "--fill-principal-name",
                    "false",
                    "--query",
                    "[].{id:id,scope:scope,principalId:principalId,roleDefinitionName:roleDefinitionName}",
                ]
            ),
            context="managed-identity role assignments",
        )
        assignments: list[dict[str, str]] = []
        unconfirmed: dict[tuple[str, str], int] = {}
        for value in values:
            assignment = _expect_json_object(value, context="role assignment")
            if self._uuid(assignment.get("principalId"), context="role principal") != self.principal_id:
                raise RuntimeError("Azure returned a foreign role principal")
            scope = _expect_string(assignment.get("scope"), context="role scope")
            assignment_id = _expect_string(assignment.get("id"), context="role assignment ID")
            prefix = f"{scope}/providers/Microsoft.Authorization/roleAssignments/"
            if not assignment_id.casefold().startswith(prefix.casefold()):
                raise RuntimeError("Role assignment ID is not within its returned scope")
            self._uuid(assignment_id[len(prefix) :], context="role assignment name")
            if scope.casefold() == self.group_id.casefold() or scope.casefold().startswith(
                f"{self.group_id}/".casefold()
            ):
                continue
            role = _expect_string(assignment.get("roleDefinitionName"), context="role name")
            matching = [
                intent
                for intent in self.role_intents
                if intent["scope"].casefold() == scope.casefold()
                and intent["role"] == role
                and (not intent["assignmentId"] or intent["assignmentId"].casefold() == assignment_id.casefold())
            ]
            if not matching:
                raise RuntimeError("Unjournalled external role requires explicit ownership binding before cleanup")
            if not any(intent["assignmentId"] for intent in matching):
                key = (scope.casefold(), role)
                unconfirmed[key] = unconfirmed.get(key, 0) + 1
                if unconfirmed[key] > 1:
                    raise RuntimeError("Unconfirmed external role has multiple candidates; require exact ID binding")
            assignments.append({"id": assignment_id, "scope": scope, "principalId": self.principal_id})
        self._record(event="external_role_observation", principal_id=self.principal_id, assignments=assignments)
        return assignments

    def _application_objects(self) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        if self.application is None:
            if self.unbound_application_request:
                candidates = _expect_json_array(
                    self._call(
                        args=[
                            "ad",
                            "app",
                            "list",
                            "--display-name",
                            self.app_name,
                            "--query",
                            "[].{id:id,appId:appId,displayName:displayName}",
                        ]
                    ),
                    context="unbound application candidates",
                )
                if candidates:
                    self.directory_blocker = (
                        "Application creation had no returned IDs; "
                        "candidate applications require manual ownership binding"
                    )
                    self._record(event="unbound_directory_candidates", candidates=candidates, deletion_authorized=False)
                    logger.error("%s", self.directory_blocker)
            return [], []
        client_id = self.application["appId"]
        applications = [
            _expect_json_object(value, context="application")
            for value in _expect_json_array(
                self._call(
                    args=[
                        "ad",
                        "app",
                        "list",
                        "--filter",
                        f"appId eq '{client_id}'",
                        "--query",
                        "[].{id:id,appId:appId,displayName:displayName}",
                    ]
                ),
                context="applications",
            )
        ]
        principals = [
            _expect_json_object(value, context="service principal")
            for value in _expect_json_array(
                self._call(
                    args=[
                        "ad",
                        "sp",
                        "list",
                        "--filter",
                        f"appId eq '{client_id}'",
                        "--query",
                        "[].{id:id,appId:appId,displayName:displayName,servicePrincipalType:servicePrincipalType}",
                    ]
                ),
                context="service principals",
            )
        ]
        if len(applications) > 1 or len(principals) > 1:
            raise RuntimeError("Entra lookup returned ambiguous application or service-principal identities")
        for application in applications:
            if application.get("appId") != client_id or application.get("id") != self.application["id"]:
                raise RuntimeError("Entra application lookup does not match exact creation IDs")
            if application.get("displayName") != self.app_name:
                raise RuntimeError("Entra application no longer has this instance's display name")
        for principal in principals:
            if principal.get("appId") != client_id or principal.get("displayName") != self.app_name:
                raise RuntimeError("Service principal is not bound to this instance application")
            if principal.get("servicePrincipalType") != "Application":
                raise RuntimeError("Refusing to delete a non-application service principal")
            principal_id = self._uuid(principal.get("id"), context="service principal object")
            if self.service_principal_id is not None and principal_id != self.service_principal_id:
                raise RuntimeError("Live service principal differs from its exact creation reference")
        self._record(
            event="directory_observation", application_count=len(applications), service_principal_count=len(principals)
        )
        return applications, principals

    def _keyvault_tombstone(self) -> None:
        values = _expect_json_array(
            self._call(
                args=[
                    "keyvault",
                    "list-deleted",
                    "--resource-type",
                    "vault",
                    "--query",
                    (
                        f"[?name=='{self.group}-kv'].{{id:id,name:name,vaultId:properties.vaultId,"
                        "location:properties.location,deletionDate:properties.deletionDate,"
                        "scheduledPurgeDate:properties.scheduledPurgeDate}"
                    ),
                ]
            ),
            context="deleted vault metadata",
        )
        if len(values) > 1:
            raise RuntimeError("Deleted-vault lookup returned ambiguous tombstones")
        for value in values:
            vault = _expect_json_object(value, context="deleted vault")
            expected_id = f"{self.group_id}/providers/Microsoft.KeyVault/vaults/{self.group}-kv"
            if str(vault.get("vaultId", "")).casefold() != expected_id.casefold():
                raise RuntimeError("Deleted-vault metadata belongs to another resource group")
        self._record(event="keyvault_soft_deletion_observation", tombstones=values, physical_purge_claimed=False)

    def _wait_for_group_absence(self) -> None:
        while self._group_exists():
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Resource group deletion did not complete before cleanup deadline")
            time.sleep(min(10, remaining))

    def run(self) -> int:
        """Verify all targets before deletion and independently verify their absence."""
        if self.arguments.cleanup_receipt is not None:
            self.receipt = self.arguments.cleanup_receipt.open("x", encoding="utf-8")
        try:
            self._record(
                event="cleanup_started",
                schema="copyrit-preview-cleanup/v1",
                journal_sha256=self.journal_sha256,
                instance=self.arguments.instance_name,
                resource_group_id=self.group_id,
                dry_run=self.arguments.dry_run,
            )
            account = _expect_json_object(
                self._call(args=["account", "show", "--query", "{id:id,tenantId:tenantId}"]),
                context="subscription",
            )
            if str(account.get("id", "")).casefold() != self.subscription.casefold():
                raise RuntimeError("Active account does not match the exact journalled subscription")
            if self.tenant_id is not None and account.get("tenantId") != self.tenant_id:
                raise RuntimeError("Active subscription tenant differs from the directory creation binding")
            group_exists = self._validate_group()
            self._validate_identity()
            assignments = self._role_assignments()
            applications, principals = self._application_objects()
            logger.info(
                "Verified preview: %s; external roles: %d; applications: %d; service principals: %d",
                self.group,
                len(assignments),
                len(applications),
                len(principals),
            )
            if self.arguments.dry_run:
                self._record(
                    event="cleanup_dry_run",
                    external_roles=assignments,
                    applications=applications,
                    service_principals=principals,
                )
                return 1 if self.directory_blocker else 0
            if (
                not self.arguments.yes
                and input(f"Delete exact journalled preview '{self.group}' and its external bindings? [y/N] ").lower()
                != "y"
            ):
                self._record(event="cleanup_aborted")
                return 0
            for assignment in assignments:
                self._call(args=["role", "assignment", "delete", "--ids", assignment["id"]])
            if self._role_assignments():
                raise RuntimeError("Owned external role assignments remain after deletion")
            if group_exists:
                self._call(args=["group", "delete", "--name", self.group, "--yes", "--no-wait"])
                self._wait_for_group_absence()
            for principal in principals:
                self._call(args=["ad", "sp", "delete", "--id", str(principal["id"])])
            for application in applications:
                self._call(args=["ad", "app", "delete", "--id", str(application["id"])])
            remaining_applications, remaining_principals = self._application_objects()
            if remaining_applications or remaining_principals:
                raise RuntimeError("Entra application or service principal still exists after deletion")
            if self._group_exists() or self._role_assignments():
                raise RuntimeError("Resource or external role reappeared during final absence verification")
            self._keyvault_tombstone()
            if self.directory_blocker:
                raise RuntimeError(self.directory_blocker)
            self._record(
                event="cleanup_verified",
                resource_group_absent=True,
                external_roles_absent=True,
                application_absent=True,
                service_principal_absent=True,
                sandbox_data_plane_absence_claimed=False,
            )
            logger.info(
                "Preview ARM/role/directory absence verified; "
                "sandbox data-plane and process cleanup remain separate gates"
            )
            return 0
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            self._record(event="cleanup_failed", error_type=type(error).__name__)
            raise
        finally:
            if self.receipt is not None:
                self.receipt.close()


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Args:
        args (list[str] | None): Arguments to parse. Defaults to sys.argv.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Tear down an isolated CoPyRIT GUI instance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--instance-name",
        required=True,
        help="Instance name (must match the name used during deployment)",
    )
    parser.add_argument(
        "--subscription",
        required=True,
        help="Azure subscription name or ID",
    )
    parser.add_argument(
        "--resource-group-id",
        required=True,
        help="Exact resource ID of the instance resource group",
    )
    parser.add_argument(
        "--delete-entra-app",
        action="store_true",
        help="Also delete the Entra app registration and service principal",
    )
    parser.add_argument(
        "--entra-app-id",
        default="",
        help="Exact Application (client) ID; required with --delete-entra-app",
    )
    parser.add_argument(
        "--acknowledge-egress-ip-release",
        action="store_true",
        help="Confirm that external allowlists have been updated before the static egress IP is released",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive prompt; ownership and egress-release checks still apply",
    )
    parser.add_argument("--journal-file", type=Path, help="Opt-in exact preview cleanup using its deployment journal")
    parser.add_argument("--cleanup-receipt", type=Path, help="New append-only JSONL cleanup evidence file")
    parser.add_argument("--dry-run", action="store_true", help="Read-only inventory for --journal-file mode")
    parser.add_argument("--require-expired", action="store_true", help="Require the journalled UTC preview expiry")
    parser.add_argument(
        "--cleanup-timeout-seconds", type=int, default=900, help="Journal-mode cleanup wall-clock bound"
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """
    Main entry point for the teardown script.

    Args:
        args (list[str] | None): CLI arguments. Defaults to sys.argv.

    Returns:
        int: Exit code (0 for success).
    """
    parsed = parse_args(args)

    instance = _expect_string(parsed.instance_name, context="instance name")
    rg_name = f"copyrit-{instance}"
    entra_app_name = f"CoPyRIT GUI ({instance})"

    if not _INSTANCE_NAME_RE.fullmatch(instance):
        logger.error(
            "--instance-name must be 1-13 lowercase letters, numbers, or internal hyphens, "
            "and must start and end with a letter or number"
        )
        return 1
    if not parsed.acknowledge_egress_ip_release:
        logger.error(
            "--acknowledge-egress-ip-release is required after removing the instance IP from external allowlists"
        )
        return 1
    if parsed.journal_file is not None:
        if parsed.delete_entra_app or parsed.entra_app_id:
            logger.error("Journal mode derives exact app/SP IDs; do not combine it with manual Entra deletion flags")
            return 1
        if not 1 <= parsed.cleanup_timeout_seconds <= 3600:
            logger.error("--cleanup-timeout-seconds must be between 1 and 3600")
            return 1
        if not parsed.dry_run and parsed.cleanup_receipt is None:
            logger.error("--cleanup-receipt is required for journal-mode deletion")
            return 1
        if parsed.cleanup_receipt is not None and parsed.cleanup_receipt.exists():
            logger.error("Cleanup receipt already exists; use a fresh output path")
            return 1
        try:
            return _PreviewCleanup(parsed).run()
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as error:
            logger.error("Preview cleanup failed: %s", error)
            if isinstance(error, subprocess.CalledProcessError) and error.stderr:
                logger.error("stderr: %s", error.stderr.strip())
            return 1
    if parsed.cleanup_receipt or parsed.dry_run or parsed.require_expired:
        logger.error("Preview cleanup flags require --journal-file")
        return 1
    if parsed.delete_entra_app != bool(parsed.entra_app_id):
        logger.error("--delete-entra-app and --entra-app-id must be provided together")
        return 1

    try:
        logger.info("Setting subscription to: %s", parsed.subscription)
        run_az(args=["account", "set", "--subscription", parsed.subscription])

        account = _expect_json_object(
            run_az_json(args=["account", "show", "--query", "{id:id,name:name}"]),
            context="active subscription",
        )
        account_id = _expect_string(account.get("id"), context="active subscription ID")
        account_name = _expect_string(account.get("name"), context="active subscription name")

        resource_group_match = _RESOURCE_GROUP_ID_RE.fullmatch(parsed.resource_group_id)
        if resource_group_match is None:
            raise RuntimeError("--resource-group-id is not a canonical Azure resource group ID")
        resource_group_subscription_id, resource_group_name = resource_group_match.groups()
        if resource_group_subscription_id.casefold() != account_id.casefold() or resource_group_name != rg_name:
            raise RuntimeError("--resource-group-id does not match the active subscription and derived instance name")

        group_info = _expect_json_object(
            run_az_json(args=["group", "show", "--name", rg_name, "--query", "{id:id,name:name,tags:tags}"]),
            context="resource group",
        )
        group_id = _expect_string(group_info.get("id"), context="resource group ID")
        if group_id.casefold() != parsed.resource_group_id.casefold():
            raise RuntimeError("Azure returned a resource group ID different from --resource-group-id")
        tags = _expect_json_object(group_info.get("tags"), context="resource group tags")
        expected_tags = {**_OWNERSHIP_TAGS, "Instance": instance}
        if any(tags.get(key) != value for key, value in expected_tags.items()):
            raise RuntimeError(
                "Resource group ownership tags do not match deploy_instance.py; refuse automatic deletion"
            )

        entra_app: dict[str, object] | None = None
        if parsed.delete_entra_app:
            entra_app = _expect_json_object(
                run_az_json(
                    args=[
                        "ad",
                        "app",
                        "show",
                        "--id",
                        parsed.entra_app_id,
                        "--query",
                        "{appId:appId,displayName:displayName}",
                    ]
                ),
                context="Entra application",
            )
            if entra_app.get("appId") != parsed.entra_app_id or entra_app.get("displayName") != entra_app_name:
                raise RuntimeError("--entra-app-id does not identify the expected instance application")

        egress_ip_value = run_az_json(
            args=[
                "network",
                "public-ip",
                "show",
                "--resource-group",
                rg_name,
                "--name",
                f"{rg_name}-egress-pip",
                "--query",
                "ipAddress",
            ]
        )
        egress_ip = egress_ip_value if isinstance(egress_ip_value, str) and egress_ip_value else None
        principal_id = _expect_string(
            run_az_json(
                args=[
                    "identity",
                    "show",
                    "--resource-group",
                    rg_name,
                    "--name",
                    f"{rg_name}-identity",
                    "--query",
                    "principalId",
                ]
            ),
            context="managed identity principal ID",
        )
        assignment_values = _expect_json_array(
            run_az_json(
                args=[
                    "role",
                    "assignment",
                    "list",
                    "--assignee-object-id",
                    principal_id,
                    "--all",
                    "--fill-principal-name",
                    "false",
                    "--query",
                    "[].{id:id,scope:scope}",
                ]
            ),
            context="managed identity role assignments",
        )
        assignments: list[dict[str, str]] = []
        for value in assignment_values:
            assignment = _expect_json_object(value, context="role assignment")
            assignment_id = _expect_string(assignment.get("id"), context="role assignment ID")
            scope_value = assignment.get("scope")
            scope = scope_value if isinstance(scope_value, str) and scope_value else "<unknown scope>"
            assignments.append({"id": assignment_id, "scope": scope})

        logger.info("Instance:          %s", instance)
        logger.info("Subscription:      %s (%s)", account_name, account_id)
        logger.info("Resource group ID: %s", group_id)
        logger.info("Static egress IP:  %s (will be released)", egress_ip or "<not allocated>")
        logger.info("Role assignments:  %d (will be removed)", len(assignments))
        if entra_app is not None:
            logger.info("Entra app ID:      %s (will be deleted)", parsed.entra_app_id)

        if not parsed.yes:
            confirm = input(f"\nDelete verified instance resource group '{rg_name}' and all its resources? [y/N] ")
            if confirm.lower() != "y":
                logger.info("Aborted.")
                return 0

        for assignment in assignments:
            logger.info("Deleting role assignment on %s", assignment["scope"])
            run_az(args=["role", "assignment", "delete", "--ids", assignment["id"]])

        logger.info("Deleting resource group: %s (this may take several minutes)", rg_name)
        run_az(args=["group", "delete", "--name", rg_name, "--yes"])
        group_exists = run_az_json(args=["group", "exists", "--name", rg_name])
        if group_exists is not False:
            raise RuntimeError(f"Resource group '{rg_name}' still exists after deletion returned")

        if parsed.delete_entra_app:
            logger.info("Deleting Entra app registration: %s", parsed.entra_app_id)
            run_az(args=["ad", "app", "delete", "--id", parsed.entra_app_id])
            logger.info("Entra app deleted")

        logger.info("")
        logger.info("=" * 60)
        logger.info("TEARDOWN COMPLETE")
        logger.info("=" * 60)
        logger.info("Resource group '%s' was deleted.", rg_name)
        logger.info("This includes: Container App, SQL server, Key Vault, MI, networking, logs.")
        logger.info("Static egress IP '%s' was released.", egress_ip or "<not allocated>")
        logger.info("")
        logger.info("Note: Key Vault uses purge protection. The vault name '%s'", f"copyrit-{instance}-kv")
        logger.info("will be reserved for ~90 days after deletion.")
        logger.info("=" * 60)

        return 0

    except RuntimeError as error:
        logger.error("%s", error)
        return 1
    except subprocess.CalledProcessError as e:
        logger.error("Command failed (exit code %d): %s", e.returncode, " ".join(e.cmd))
        if e.stderr:
            logger.error("stderr: %s", e.stderr.strip())
        return 1


if __name__ == "__main__":
    sys.exit(main())
