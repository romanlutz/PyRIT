# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Immutable Inspect-owned task results and observed GHCP evidence summary."""

from __future__ import annotations

import hashlib
import json
from enum import Enum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator


class InspectGhcpStatus(str, Enum):
    """Whether a real original scorer can safely determine the final run."""

    COMPLETED = "completed"
    INCOMPLETE = "incomplete"
    ERROR = "error"


class InspectGhcpTaskKind(str, Enum):
    """A qualified original cyber task versus a non-scored protocol smoke."""

    CYBER_BENCHMARK = "cyber_benchmark"
    PROTOCOL_SMOKE = "protocol_smoke"


class InspectGhcpJudgment(BaseModel):
    """An acquired original Inspect Score, not a second PyRIT grading pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scorer_name: str = Field(min_length=1)
    source_event_id: str | None = Field(default=None, min_length=1)
    normalization_version: int = Field(default=1, ge=1, le=1)
    raw_value: JsonValue
    numeric_value: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False, strict=True)
    explanation: str | None = None
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class InspectGhcpReport(BaseModel):
    """Canonical, content-anchored result for one complete original Inspect Task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1, 2, 3] = 3
    run_id: str = Field(min_length=1, max_length=128)
    task_name: str = Field(min_length=1)
    task_version: str = Field(min_length=1)
    sample_id: str = Field(min_length=1)
    sample_epoch: int = Field(default=1, ge=1)
    task_kind: InspectGhcpTaskKind = InspectGhcpTaskKind.PROTOCOL_SMOKE
    benchmark_verified: bool = False
    provider_verified: bool = False
    task_assets_sha256: dict[str, str] = Field(default_factory=dict)
    image_ids: dict[str, str] = Field(default_factory=dict)
    target_image: str | None = None
    cli_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str = Field(min_length=1)
    wire_model: str = Field(min_length=1)
    prompt_cache_key_policy: str = "reject"
    started_at: AwareDatetime
    ended_at: AwareDatetime
    inspect_log_id: str | None = None
    inspect_log_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    inspect_sample_uuid: str | None = None
    agent_container_id: str | None = None
    model_container_id: str | None = None
    target_container_id: str | None = None
    ghcp_session_id: str | None = None
    agent_process: dict[str, JsonValue] | None = None
    conversation_id: str | None = None
    attack_identifier: dict[str, JsonValue] | None = None
    turn_count: int = Field(ge=0)
    sdk_event_count: int = Field(ge=0)
    sdk_event_raw_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_request_count: int = Field(ge=0)
    model_http_200_count: int = Field(ge=0)
    host_model_request_count: int = Field(ge=0)
    host_model_http_200_count: int = Field(ge=0)
    adversarial_request_count: int = Field(ge=0)
    adversarial_http_200_count: int = Field(ge=0)
    tool_start_count: int = Field(ge=0)
    tool_complete_count: int = Field(ge=0)
    successful_tool_execution_count: int = Field(default=0, ge=0)
    required_tool_executions: int = Field(default=0, ge=0)
    gateway_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    host_model_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    adversarial_audit_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    control_receipt_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    token_files_absent_before_turn: bool = False
    phase_durations_seconds: dict[str, float] = Field(default_factory=dict)
    stopped_before_scoring: bool = False
    gateway_alive_before_scoring: bool = False
    gateway_alive_after_scoring: bool = False
    target_alive_before_scoring: bool = False
    target_alive_after_scoring: bool = False
    original_cleanup_called: bool = False
    original_cleanup_succeeded: bool = False
    sandbox_cleanup_observed: bool = False
    judgment: InspectGhcpJudgment | None = None
    required_gaps: tuple[str, ...] = ()
    optional_gaps: tuple[str, ...] = ()
    status: InspectGhcpStatus = InspectGhcpStatus.INCOMPLETE

    @model_validator(mode="after")
    def _validate_report(self) -> InspectGhcpReport:
        if self.ended_at < self.started_at:
            raise ValueError("An Inspect evaluation cannot finish before it starts.")
        if self.model_http_200_count > self.model_request_count:
            raise ValueError("GHCP model successes cannot exceed recorded requests.")
        if self.host_model_http_200_count > self.host_model_request_count:
            raise ValueError("Trusted host model successes cannot exceed observed requests.")
        if self.adversarial_http_200_count > self.adversarial_request_count:
            raise ValueError("PyRIT adversarial-model successes cannot exceed observed requests.")
        if self.tool_complete_count > self.tool_start_count:
            raise ValueError("GHCP tool completions cannot exceed observed starts.")
        if self.successful_tool_execution_count > self.tool_complete_count:
            raise ValueError("GHCP successful tools cannot exceed their observed completions.")
        if self.benchmark_verified and (
            self.task_kind is not InspectGhcpTaskKind.CYBER_BENCHMARK
            or not self.task_assets_sha256
            or not self.target_image
            or not self.image_ids
        ):
            raise ValueError("A verified cyber task needs its original assets and pinned target image.")
        if self.status is InspectGhcpStatus.COMPLETED and (
            not self.benchmark_verified
            or not self.provider_verified
            or self.judgment is None
            or self.judgment.numeric_value is None
            or (self.schema_version >= 2 and not self.judgment.source_event_id)
            or not self.inspect_log_sha256
            or not self.gateway_audit_sha256
            or not self.sdk_event_raw_sha256
            or not self.host_model_audit_sha256
            or (self.turn_count > 1 and not self.adversarial_audit_sha256)
            or (
                self.schema_version == 3
                and (not self.control_receipt_sha256 or not self.token_files_absent_before_turn)
            )
            or not all(
                isinstance(container_id, str) and len(container_id) == 64
                for container_id in (
                    self.agent_container_id,
                    self.model_container_id,
                    self.target_container_id,
                )
            )
            or len({self.agent_container_id, self.model_container_id, self.target_container_id}) != 3
            or self.agent_process is None
            or not isinstance(self.agent_process.get("cli_pid"), int)
            or not self.agent_process.get("net_namespace")
            or not self.stopped_before_scoring
            or not self.gateway_alive_before_scoring
            or not self.gateway_alive_after_scoring
            or not self.target_alive_before_scoring
            or not self.target_alive_after_scoring
            or not self.original_cleanup_called
            or not self.original_cleanup_succeeded
            or not self.sandbox_cleanup_observed
            or self.turn_count < 1
            or self.model_http_200_count < self.turn_count
            or self.host_model_http_200_count < self.turn_count
            or self.adversarial_http_200_count < self.turn_count - 1
            or self.tool_start_count != self.tool_complete_count
            or self.successful_tool_execution_count < self.required_tool_executions
            or self.required_gaps
        ):
            raise ValueError("A completed cyber Score requires full source coverage, original grading and cleanup.")
        return self

    def canonical_json(self) -> str:
        """
        Serialize the exact recorded Inspect outcome.

        Returns:
            str: Stable JSON for immutable content-anchored scoring.
        """
        return json.dumps(
            self.model_dump(mode="json", exclude_unset=self.schema_version in {1, 2}),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def sha256(self) -> str:
        """
        Hash the canonical Inspect outcome.

        Returns:
            str: Lowercase SHA256 of the report bytes.
        """
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()
