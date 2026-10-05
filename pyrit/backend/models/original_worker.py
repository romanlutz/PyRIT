# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Server-owned configuration and versioned messages for one trusted original worker."""

from __future__ import annotations

import re
from datetime import datetime, timedelta  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID  # noqa: TC003

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StrictInt, model_validator

from pyrit.models import EvalCaseRef, EvalSpecRef, config_hash

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
Alias = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]

ORIGINAL_WORKER_ARTIFACTS = (
    "source.eval",
    "backend-intake-envelope.json",
    "worker-scenario.json",
    "source-manifest.json",
    "runner-closure.json",
    "worker-provenance.json",
)
CANCEL_FILENAME = "cancel.request.json"


class _FrozenMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


class CohostSourcePolicy(_FrozenMessage):
    """Exactly one approved source/case and its unchanged original scorer."""

    contract_sha256: Digest
    spec: EvalSpecRef
    cases: tuple[EvalCaseRef, ...] = Field(min_length=1, max_length=1)
    primary_scorer: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    display_values: frozenset[str] = Field(max_length=16)

    @model_validator(mode="after")
    def _validate_source(self) -> CohostSourcePolicy:
        """
        Require one unchanged, source-bound case without an input override.

        Returns:
            CohostSourcePolicy: The bounded approved source policy.
        """
        if self.spec.input_variant is not None or self.cases[0].package != self.spec.package:
            raise ValueError("Original worker policy must select one unchanged case from its approved source.")
        if any(re.fullmatch(r"[A-Za-z0-9._+-]{1,64}", value) is None for value in self.display_values):
            raise ValueError("Original score display values must be bounded reviewed scalars.")
        return self


class CohostRelayConfig(_FrozenMessage):
    """One fixed native AOAI pilot route; limits include every dispatched attempt."""

    endpoint: str
    account: Literal["pyrit-github-pipeline"] = "pyrit-github-pipeline"
    deployment: Literal["gpt-4-32"] = "gpt-4-32"
    model: Literal["gpt-4o"] = "gpt-4o"
    model_version: Literal["2024-11-20"] = "2024-11-20"
    api_version: Literal["2024-10-21"] = "2024-10-21"
    content_filter: Literal["Microsoft.Default"] = "Microsoft.Default"
    max_requests: StrictInt = Field(default=50, ge=1, le=50)
    max_observed_tokens: StrictInt = Field(default=100_000, ge=1, le=100_000)
    max_request_bytes: StrictInt = Field(default=524_288, ge=1024, le=524_288)
    max_response_bytes: StrictInt = Field(default=2_097_152, ge=1024, le=2_097_152)
    max_completion_tokens: StrictInt = Field(default=4096, ge=1, le=8192)
    request_timeout_seconds: StrictInt = Field(default=180, ge=1, le=180)
    aggregate_max_requests: StrictInt = Field(default=100, ge=1, le=100)
    aggregate_max_observed_tokens: StrictInt = Field(default=200_000, ge=1, le=200_000)
    min_request_interval_seconds: StrictInt = Field(default=1, ge=1, le=60)
    minute_observed_token_limit: StrictInt = Field(default=20_000, ge=1, le=30_000)

    @model_validator(mode="after")
    def _validate_endpoint(self) -> CohostRelayConfig:
        """
        Reject arbitrary destinations, alternate deployments and credentials.

        Returns:
            CohostRelayConfig: The exact approved native route.
        """
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "https"
            or parsed.hostname != f"{self.account}.openai.azure.com"
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Original model relay requires the exact approved Azure OpenAI account origin.")
        return self

    @property
    def route_metadata(self) -> dict[str, JsonValue]:
        """The fixed route fingerprint shared with the original source descriptor."""
        return {
            "role": "evaluated",
            "account": self.account,
            "deployment": self.deployment,
            "model": self.model,
            "model_version": self.model_version,
            "filter": self.content_filter,
            "api_version": self.api_version,
            "execution_profile": "COHOST-TRUSTED/v1",
            "credential_authority": "trusted-preview-uami",
            "model_auth_placement": "cohost-trusted-api-relay",
            "guest_receives_model_credentials": False,
            "native_iam_chat_only": False,
        }

    @property
    def route_sha256(self) -> str:
        """Canonical source-policy identity, not a credential or request digest."""
        return config_hash(self.route_metadata)


class CohostSandboxConfig(_FrozenMessage):
    """Read-only native cleanup observation under one startup-owned preview group."""

    endpoint: str
    subscription_id: UUID
    resource_group: str = Field(min_length=1, max_length=90)
    sandbox_group: str = Field(min_length=1, max_length=90)
    region: Literal["westus2"] = "westus2"
    purpose: Literal["synthetic-jwt-mode1"] = "synthetic-jwt-mode1"

    @model_validator(mode="after")
    def _validate_endpoint(self) -> CohostSandboxConfig:
        """
        Use only the pinned native regional data-plane origin.

        Returns:
            CohostSandboxConfig: The approved native observer scope.
        """
        if self.endpoint != f"https://management.{self.region}.azuredevcompute.io":
            raise ValueError("Sandbox observation must use the approved native regional endpoint.")
        return self


class CohostValidationScope(_FrozenMessage):
    """Opt-in two-job pilot authority; not a permanent original-Scenario product limit."""

    instance_id: UUID
    expires_at: datetime
    max_original_jobs: Literal[2] = 2
    state_container_url: str
    authority_key_sha256: Digest | None = None

    @model_validator(mode="after")
    def _validate_scope(self) -> CohostValidationScope:
        """
        Keep lease/state authority in one private owned container and UTC lifetime.

        Returns:
            CohostValidationScope: The explicit validation-scope policy.
        """
        parsed = urlsplit(self.state_container_url)
        if (
            self.expires_at.utcoffset() != timedelta(0)
            or parsed.scheme != "https"
            or not (parsed.hostname or "").endswith(".blob.core.windows.net")
            or not parsed.path.strip("/")
            or "/" in parsed.path.strip("/")
            or parsed.port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Validation state requires its exact private owned Blob container and UTC expiry.")
        return self


class CohostModelBudget(_FrozenMessage):
    """Authenticated cumulative native model state, never a byte-based token estimate."""

    requests: StrictInt = Field(ge=0)
    observed_tokens: StrictInt = Field(ge=0)
    unresolved: StrictBool


class CohostBackendConfig(_FrozenMessage):
    """Opt-in deployment settings, never accepted through a browser request."""

    schema_version: Literal[1] = 1
    profile_ref: Alias
    source_alias: Alias
    public_commit: Commit
    worker_python: Path
    worker_python_version: tuple[StrictInt, StrictInt] = (3, 12)
    worker_inspect_version: str = Field(default="0.3.247", pattern=r"^0\.3\.[0-9]+$")
    worker_entrypoint: Path
    worker_entrypoint_sha256: Digest
    source_root: Path
    source_contract: Path | None = None
    jobs_root: Path
    relay_origin: str = "http://127.0.0.1:8000"
    source: CohostSourcePolicy
    relay: CohostRelayConfig
    sandbox: CohostSandboxConfig | None = None
    validation_scope: CohostValidationScope | None = None
    allowed_operator_oids: frozenset[str] = Field(min_length=1, max_length=64)
    allowed_group_ids: frozenset[str] = Field(min_length=1, max_length=20)
    managed_identity_client_id: UUID
    worker_environment: dict[str, str] = Field(default_factory=dict, repr=False)
    active_timeout_seconds: StrictInt = Field(default=1500, ge=1, le=1500)
    cleanup_timeout_seconds: Literal[300] = 300
    retention_seconds: Literal[86400] = 86400
    min_free_bytes: StrictInt = Field(default=1_073_741_824, ge=67_108_864)
    max_memory_current_bytes: StrictInt = Field(default=3_221_225_472, ge=67_108_864)
    expected_database_name: str = Field(min_length=1, max_length=128)
    result_container_url: str
    local_test: StrictBool = False
    allow_ordinary_model_calls: Literal[False] = False

    @model_validator(mode="after")
    def _validate_configuration(self) -> CohostBackendConfig:
        """
        Keep paths and native route selection entirely on the trusted host.

        Returns:
            CohostBackendConfig: The closed server-owned deployment policy.
        """
        if any(
            not path.is_absolute()
            for path in (self.worker_python, self.worker_entrypoint, self.source_root, self.jobs_root)
        ):
            raise ValueError("Original worker staging and run roots must be absolute.")
        if self.jobs_root == self.source_root or self.jobs_root.is_relative_to(self.source_root):
            raise ValueError("Original run roots must be separate from the staged source tree.")
        if self.source_root.is_relative_to(self.jobs_root):
            raise ValueError("Original source staging cannot be inside the run inventory.")
        if not self.worker_entrypoint.is_relative_to(self.source_root):
            raise ValueError("Only the pinned staged worker entrypoint may execute.")
        parsed = urlsplit(self.relay_origin)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError("The trusted child must use this replica's explicit loopback API origin.")
        if not self.local_test and self.relay.max_completion_tokens != 8192:
            raise ValueError("Hosted original preview requires its unchanged 8192-token completion configuration.")
        if not self.local_test and self.source.spec.model_route.config_sha256 != self.relay.route_sha256:
            raise ValueError("The approved source and fixed evaluated model route differ.")
        if not self.local_test and self.sandbox is None:
            raise ValueError("Hosted original workers require an independent owned-sandbox observer.")
        if not self.local_test and self.validation_scope is None:
            raise ValueError("This hosted pilot requires explicit durable two-job validation-scope admission.")
        if not self.local_test and (
            self.source_contract is None
            or not self.source_contract.is_absolute()
            or not self.source_contract.is_relative_to(self.source_root)
        ):
            raise ValueError("Hosted original workers require the exact staged data-only qualified contract.")
        if not self.local_test and (self.worker_python_version != (3, 12) or self.worker_inspect_version != "0.3.247"):
            raise ValueError("The qualified hosted source requires its separate Python 3.12/Inspect 0.3.247 runtime.")
        if any(not value.strip() for value in (*self.allowed_operator_oids, *self.allowed_group_ids)):
            raise ValueError("Original source and actor authorization must be explicit.")
        return self

    @property
    def harness_metadata(self) -> dict[str, JsonValue]:
        """The fixed original harness fingerprint, shared with the staged source contract."""
        return {
            "manifest_sha256": "$server_manifest_sha256",
            "worker_placement": "cohost-trusted-supervised-child",
            "model_auth_placement": "cohost-trusted-api-relay",
            "execution_profile": "COHOST-TRUSTED/v1",
            "scratch_memory_authority": "transient-worker-staging",
            "durable_memory_authority": "authenticated-backend-only",
        }

    def harness_metadata_for(self, manifest_sha256: str) -> dict[str, JsonValue]:
        """
        Bind the fixed harness template to one server-owned manifest.

        Returns:
            dict[str, JsonValue]: Concrete original per-job harness identity.
        """
        return {**self.harness_metadata, "manifest_sha256": manifest_sha256}


class CohostWorkerRequest(_FrozenMessage):
    """One bounded stdin request; run_instance_id is the supervisor control UUID."""

    abi_version: Literal[1] = 1
    app_run_id: UUID
    job_ref: UUID
    run_instance_id: UUID
    operator_oid: str = Field(min_length=1, max_length=128)
    profile_ref: Alias
    source_alias: Alias
    source_sha256: Digest
    manifest_sha256: Digest
    source_root: Path
    run_root: Path
    active_timeout_seconds: StrictInt = Field(ge=1, le=1500)
    cleanup_timeout_seconds: Literal[300] = 300
    active_deadline_utc: datetime
    cleanup_deadline_utc: datetime
    relay_url: str
    relay_capability: str = Field(min_length=43, max_length=128, repr=False)

    @model_validator(mode="after")
    def _validate_deadlines(self) -> CohostWorkerRequest:
        """
        Require the same absolute UTC active/cleanup clock.

        Returns:
            CohostWorkerRequest: One launch-owned 300-second cleanup reserve.
        """
        if (
            self.active_deadline_utc.utcoffset() != timedelta(0)
            or self.cleanup_deadline_utc.utcoffset() != timedelta(0)
            or self.cleanup_deadline_utc - self.active_deadline_utc != timedelta(seconds=300)
        ):
            raise ValueError("The worker must use the supervisor's exact absolute UTC cleanup reserve.")
        return self


class CohostWorkerTerminal(_FrozenMessage):
    """Safe stdout, not a grade or evidence authority."""

    abi_version: Literal[1] = 1
    job_ref: UUID
    run_instance_id: UUID
    state: Literal["success", "error", "cancelled"]


class CohostArtifactDigest(_FrozenMessage):
    """Exact retained bytes; no worker-selected paths."""

    sha256: Digest
    bytes: StrictInt = Field(ge=1, le=16 * 1024 * 1024)


class CohostWorkerProvenance(_FrozenMessage):
    """Bind supervisor and framework identities without reading scratch database rows."""

    abi_version: Literal[1] = 1
    execution_profile: Literal["COHOST-TRUSTED/v1"]
    app_run_id: UUID
    job_ref: UUID
    control_run_instance_id: UUID
    framework_run_instance_id: UUID
    operator_oid: str = Field(min_length=1, max_length=128)
    profile_ref: Alias
    source_alias: Alias
    source_sha256: Digest
    manifest_sha256: Digest
    model_route: dict[str, JsonValue]
    public_commit: Commit
    state: Literal["success", "error", "cancelled"]
    failure_type: str | None = Field(default=None, max_length=128)
    process_identity: dict[str, JsonValue]
    private_imports_after_memory: Literal[True]
    authority: str = Field(min_length=1, max_length=128)
    guest_receives_preview_credentials: Literal[False]
    scratch_memory_authority: Literal["transient-worker-staging"]
    durable_memory_authority: Literal["authenticated-backend-only"]
    scratch_grade_counts: dict[str, StrictInt]
    artifacts: dict[str, CohostArtifactDigest]

    @model_validator(mode="after")
    def _validate_artifacts(self) -> CohostWorkerProvenance:
        """
        Refuse foreign files or missing references without trusting a worker grade.

        Returns:
            CohostWorkerProvenance: Exact fixed artifact references.
        """
        expected = set(ORIGINAL_WORKER_ARTIFACTS) - {"worker-provenance.json"}
        if self.state != "success":
            expected.discard("source.eval")
        if set(self.artifacts) not in (expected, expected | {"source.eval"}):
            raise ValueError("Original worker provenance must bind only its fixed evidence artifacts.")
        if any(value < 0 for value in self.scratch_grade_counts.values()):
            raise ValueError("Scratch row counts cannot be negative.")
        return self
