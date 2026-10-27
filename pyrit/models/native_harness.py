# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Harness-neutral identities and evidence views, without flattening provider transcripts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pyrit.models.native_cli_report import NativeCliReportEvidence  # noqa: TC001 (Pydantic resolves the source model)
from pyrit.models.native_cyber import NativeAgentEvidence  # noqa: TC001 (Pydantic resolves the source model)


class NativeModelResponseOrigin(str, Enum):
    """Trusted original provider bytes, distinct from host or synthetic replacements."""

    PROVIDER = "provider"
    HOST = "host"
    SYNTHETIC = "synthetic"


@dataclass(frozen=True, kw_only=True)
class NativeModelCaptureIdentity:
    """Controller identity; a source session is absent until actually observed."""

    run_id: str
    session_id: str | None
    turn_id: str
    turn_index: int
    environment_id: str | None = None
    harness_id: str | None = None


class NativeHarnessIdentity(BaseModel):
    """Immutable owned environment and selected harness, with nullable observed session."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1, max_length=128)
    environment_id: str = Field(min_length=1, max_length=128)
    harness_id: str = Field(min_length=1, max_length=128)
    source_session_id: str | None = Field(default=None, min_length=1, max_length=128)


class NativeHarnessEvidence(BaseModel):
    """Common grading gate plus the unchanged, explicitly typed harness evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    identity: NativeHarnessIdentity
    simulated: bool
    quiescent: bool
    coverage_complete: bool
    gaps: tuple[str, ...] = ()
    sdk: NativeAgentEvidence | None = None
    cli: NativeCliReportEvidence | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> NativeHarnessEvidence:
        if (self.sdk is None) == (self.cli is None):
            raise ValueError("Harness evidence requires exactly one original source representation.")
        if self.sdk is not None and (
            self.sdk.environment_id != self.identity.environment_id
            or self.sdk.session_id != self.identity.source_session_id
            or self.sdk.simulated != self.simulated
        ):
            raise ValueError("SDK evidence disagrees with the owned harness identity.")
        if self.cli is not None and self.cli.source_session_id != self.identity.source_session_id:
            raise ValueError("CLI evidence disagrees with the observed harness session.")
        source = self.sdk if self.sdk is not None else self.cli
        assert source is not None
        if self.coverage_complete and (self.gaps or not source.coverage_complete):
            raise ValueError("Complete harness evidence requires complete original source coverage.")
        return self
