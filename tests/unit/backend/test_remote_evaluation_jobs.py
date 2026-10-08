# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Explicit remote startup configuration and empty production identity/delegation installation."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pyrit.backend.services.evaluation_job_service import (
    RemoteEvaluationJobSettings,
    evaluation_job_settings_from_environment,
)
from pyrit.executor.jobs.worker_auth import EvaluationWorkerCredentialRegistry
from pyrit.executor.jobs.worker_client import EvaluationWorkerHttpSettings
from pyrit.models.evaluation_worker import evaluation_worker_schema_sha256

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import SQLiteMemory

ACTOR = "00000000-0000-4000-8000-000000000001"


def _environment(root: Path) -> dict[str, str]:
    return {
        "PYRIT_EVALUATION_JOB_BACKEND": "remote",
        "PYRIT_EVALUATION_JOB_ROOT": str(root),
        "PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS": ACTOR,
        "PYRIT_EVALUATION_JOB_REMOTE_URL": "https://worker.example.invalid",
        "PYRIT_EVALUATION_JOB_REMOTE_AUDIENCE": "configured-fixture-audience",
        "PYRIT_EVALUATION_JOB_REMOTE_IDENTITY": "not-installed",
        "PYRIT_EVALUATION_JOB_REMOTE_SERVICE_ID": "public_fixture",
        "PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_VERSION": "1",
        "PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_SHA256": evaluation_worker_schema_sha256(),
    }


def test_remote_configuration_is_startup_owned_and_completely_default_off(tmp_path: Path) -> None:
    assert evaluation_job_settings_from_environment({}) is None
    settings = evaluation_job_settings_from_environment(_environment(tmp_path))
    assert isinstance(settings, RemoteEvaluationJobSettings)
    assert settings.root == tmp_path and settings.allowed_actor_ids == frozenset({ACTOR})
    assert settings.transport.audience == "configured-fixture-audience"
    assert settings.identity_name == "not-installed"
    assert not settings.transport.allow_loopback_http


@pytest.mark.parametrize(
    "key",
    [
        "PYRIT_EVALUATION_JOB_ROOT",
        "PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS",
        "PYRIT_EVALUATION_JOB_REMOTE_URL",
        "PYRIT_EVALUATION_JOB_REMOTE_AUDIENCE",
        "PYRIT_EVALUATION_JOB_REMOTE_IDENTITY",
        "PYRIT_EVALUATION_JOB_REMOTE_SERVICE_ID",
        "PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_VERSION",
        "PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_SHA256",
    ],
)
def test_partial_remote_configuration_never_falls_back_to_local_or_disabled(*, tmp_path: Path, key: str) -> None:
    environment = _environment(tmp_path)
    del environment[key]
    with pytest.raises(ValueError):
        evaluation_job_settings_from_environment(environment)


@pytest.mark.parametrize("backend", ["", "local", "azure"])
def test_remote_fragments_require_explicit_remote_selection(*, tmp_path: Path, backend: str) -> None:
    environment = _environment(tmp_path)
    environment["PYRIT_EVALUATION_JOB_BACKEND"] = backend
    with pytest.raises(ValueError):
        evaluation_job_settings_from_environment(environment)


@pytest.mark.parametrize(
    "url",
    [
        "http://worker.example.invalid",
        "http://127.0.0.1",
        "https://worker.example.invalid/api",
        "https://name:password@worker.example.invalid",
        "https://worker.example.invalid?token=not-a-token",
        "https://worker.example.invalid/#fragment",
    ],
)
def test_remote_authority_rejects_insecure_implicit_or_credential_bearing_endpoints(
    *, tmp_path: Path, url: str
) -> None:
    environment = _environment(tmp_path)
    environment["PYRIT_EVALUATION_JOB_REMOTE_URL"] = url
    with pytest.raises(ValueError):
        evaluation_job_settings_from_environment(environment)


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "601"])
def test_remote_poll_limits_are_finite_and_bounded(*, tmp_path: Path, value: str) -> None:
    environment = _environment(tmp_path)
    environment["PYRIT_EVALUATION_JOB_REMOTE_POLL_DEADLINE_SECONDS"] = value
    with pytest.raises(ValueError):
        evaluation_job_settings_from_environment(environment)


def test_loopback_http_is_explicitly_fixture_only(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment["PYRIT_EVALUATION_JOB_REMOTE_URL"] = "http://127.0.0.1:12345"
    environment["PYRIT_EVALUATION_JOB_REMOTE_ALLOW_LOOPBACK_HTTP"] = "true"
    settings = evaluation_job_settings_from_environment(environment)
    assert isinstance(settings, RemoteEvaluationJobSettings) and settings.transport.allow_loopback_http


@pytest.mark.parametrize("version", ["2", "true", "1.0"])
def test_remote_protocol_version_is_explicit_and_not_implicitly_upgraded(*, tmp_path: Path, version: str) -> None:
    environment = _environment(tmp_path)
    environment["PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_VERSION"] = version
    with pytest.raises(ValueError, match="version 1"):
        evaluation_job_settings_from_environment(environment)


async def test_unknown_production_identity_and_delegation_fail_before_any_dispatch_async(
    *, sqlite_instance: SQLiteMemory, tmp_path: Path
) -> None:
    settings = RemoteEvaluationJobSettings.from_environment(_environment(tmp_path))
    with patch.dict(EvaluationWorkerCredentialRegistry._factories, {}, clear=True):
        with pytest.raises(ValueError, match="No reviewed"):
            await settings.create_port_async(memory=sqlite_instance)
    assert not (tmp_path / "remote.sqlite").exists()


@pytest.mark.parametrize(
    ("timeout", "schema"),
    [(True, evaluation_worker_schema_sha256()), (10, "f" * 64)],
)
def test_transport_settings_reject_unchecked_bool_timeouts_and_protocol_mismatch(
    *, timeout: float, schema: str
) -> None:
    with pytest.raises(ValueError):
        EvaluationWorkerHttpSettings(
            base_url="https://worker.example.invalid",
            audience="configured-audience",
            service_id="public_fixture",
            schema_sha256=schema,
            request_timeout_seconds=timeout,
        )


@pytest.mark.parametrize("value", ["true", "false", 1, 0])
def test_transport_loopback_opt_in_requires_an_actual_boolean(value: object) -> None:
    settings = EvaluationWorkerHttpSettings(
        base_url="https://worker.example.invalid",
        audience="configured-audience",
        service_id="public_fixture",
        schema_sha256=evaluation_worker_schema_sha256(),
    )
    with pytest.raises(ValueError):
        replace(settings, allow_loopback_http=value)
