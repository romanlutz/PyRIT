# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker sandbox provider tests that do not require a Docker daemon."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from pyrit.sandbox import (
    DockerComposeConfigError,
    DockerLifecycleError,
    DockerSandboxProvider,
    DockerSandboxProviderConfig,
    DockerSecurityPolicy,
    DockerSecurityPolicyViolationError,
    DockerServiceBuildSpec,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxOperationStatus,
    SandboxSessionSpec,
)
from pyrit.sandbox.docker_provider import (
    DockerSandboxEnvironment,
    DockerSandboxProcess,
    DockerSandboxSession,
    _CliInvocationError,
    _CliOutput,
    _container_health_state,
    _project_name_for_session,
    _RawRun,
    _scan_security_violations,
    _select_service_name,
    _synthesize_compose_document,
    _wrap_with_timeout,
)

if TYPE_CHECKING:
    from pathlib import Path


def _provider_config(tmp_path: Path) -> DockerSandboxProviderConfig:
    compose_file = tmp_path / "compose.json"
    compose_file.write_text('{"services":{"default":{"image":"python:3.12-alpine"}}}', encoding="utf-8")
    return DockerSandboxProviderConfig(
        compose_files=(compose_file,),
        project_context=tmp_path,
        state_dir=tmp_path / "state",
        cli_timeout_seconds=0.1,
    )


def _resolved_document(service: dict[str, object]) -> dict[str, object]:
    return {"services": {"default": service}}


def test_config_requires_compose_or_dockerfile_service() -> None:
    with pytest.raises(ValidationError, match="requires at least one"):
        DockerSandboxProviderConfig()


def test_synthesized_compose_preserves_relative_context_and_isolates_network(tmp_path: Path) -> None:
    context = tmp_path / "relative-context"
    context.mkdir()
    policy = DockerSecurityPolicy(allow_egress=False, default_memory_limit="512m", default_cpus=1.5)
    document = _synthesize_compose_document(
        services=(DockerServiceBuildSpec(service_name="worker", build_context=context.name),),
        project_context=tmp_path,
        security_policy=policy,
    )
    service = document["services"]["worker"]
    assert service["build"]["context"] == str(context.resolve())
    assert service["cap_drop"] == ["ALL"]
    assert service["mem_limit"] == "512m"
    assert service["cpus"] == 1.5
    assert document["networks"]["pyrit_sandbox"]["internal"] is True


def test_project_names_are_bounded_and_attempt_unique() -> None:
    first = _project_name_for_session(prefix="pyrit", session_id="S" * 200, attempt_id="attempt-one")
    second = _project_name_for_session(prefix="pyrit", session_id="S" * 200, attempt_id="attempt-two")
    assert first != second
    assert len(first) <= 63
    assert len(second) <= 63


def test_service_selection_is_deterministic() -> None:
    services = {
        "first": {},
        "marked": {"labels": {"com.pyrit.sandbox.environment": "analysis"}},
        "default": {},
    }
    order = ("first", "marked", "default")
    assert (
        _select_service_name(resolved_services=services, environment_name="default", service_order=order) == "default"
    )
    assert (
        _select_service_name(resolved_services=services, environment_name="analysis", service_order=order) == "marked"
    )
    assert _select_service_name(resolved_services=services, environment_name="other", service_order=order) == "first"


@pytest.mark.parametrize(
    ("service", "expected"),
    [
        ({"privileged": True}, "privileged"),
        ({"network_mode": "host"}, "network_mode"),
        ({"pid": "container:outside"}, "pid"),
        ({"ipc": "service:outside"}, "ipc"),
        ({"userns_mode": "host"}, "userns_mode"),
        ({"volumes": [{"type": "bind", "source": "/var/run/docker.sock"}]}, "Docker socket"),
        ({"volumes": [{"type": "bind", "source": "/host"}]}, "bind-mounts"),
        ({"devices": ["/dev/kvm"]}, "device"),
        ({"cap_add": ["SYS_ADMIN"]}, "capabilities"),
        ({"cap_add": ["CAP_SYS_ADMIN"]}, "capabilities"),
        ({"security_opt": ["seccomp:unconfined"]}, "seccomp"),
        ({"ports": [{"target": 8080}]}, "host port"),
    ],
)
def test_security_policy_rejects_host_escape_surfaces(service: dict[str, object], expected: str) -> None:
    violations = _scan_security_violations(
        resolved_document=_resolved_document(service),
        policy=DockerSecurityPolicy(),
    )
    assert any(expected in violation for violation in violations)


def test_security_policy_rejects_environment_backed_secrets() -> None:
    violations = _scan_security_violations(
        resolved_document={"services": {"default": {}}, "secrets": {"token": {"environment": "TOKEN"}}},
        policy=DockerSecurityPolicy(),
    )
    assert violations == ["secret 'token' is sourced from a plain environment variable"]


def test_container_health_states() -> None:
    assert _container_health_state({"Status": "Up 2 seconds (health: starting)"}) == "starting"
    assert _container_health_state({"Status": "Up 2 seconds (healthy)"}) == "healthy"
    assert _container_health_state({"Status": "Up 2 seconds (unhealthy)"}) == "unhealthy"
    assert _container_health_state({"Status": "Up 2 seconds"}) is None


def test_container_timeout_wrapper_preserves_subsecond_budget() -> None:
    command = _wrap_with_timeout(command=("sleep", "10"), budget_seconds=0.05, grace_seconds=0.25)
    assert command == ("timeout", "-k", "0.25", "0.05", "sleep", "10")


async def test_prepare_resolves_config_and_checks_security_before_lifecycle(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    safe_config = json.dumps({"services": {"default": {"image": "python:3.12-alpine"}}}).encode()
    with patch(
        "pyrit.sandbox.docker_provider._run_cli_async",
        new=AsyncMock(side_effect=[_CliOutput(stdout=b"", stderr=b""), _CliOutput(stdout=safe_config, stderr=b"")]),
    ) as run:
        await provider.prepare_async()
    assert run.await_count == 2
    config_argv = run.await_args_list[1].kwargs["argv"]
    assert config_argv[-3:] == ["config", "--format", "json"]
    await provider.cleanup_async()


async def test_prepare_rejects_unsafe_resolved_config(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    unsafe_config = json.dumps({"services": {"default": {"privileged": True}}}).encode()
    with (
        patch(
            "pyrit.sandbox.docker_provider._run_cli_async",
            new=AsyncMock(
                side_effect=[_CliOutput(stdout=b"", stderr=b""), _CliOutput(stdout=unsafe_config, stderr=b"")]
            ),
        ),
        pytest.raises(DockerSecurityPolicyViolationError, match="privileged"),
    ):
        await provider.prepare_async()


async def test_prepare_reports_invalid_compose_json(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    with (
        patch(
            "pyrit.sandbox.docker_provider._run_cli_async",
            new=AsyncMock(side_effect=[_CliOutput(stdout=b"", stderr=b""), _CliOutput(stdout=b"not-json", stderr=b"")]),
        ),
        pytest.raises(DockerComposeConfigError, match="Could not parse"),
    ):
        await provider.prepare_async()


async def test_lifecycle_retries_only_known_transient_failures(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    transient = _CliInvocationError(
        argv=("docker", "ps"),
        returncode=1,
        stdout=b"",
        stderr=b"connection reset by peer",
        timed_out=False,
    )
    with patch(
        "pyrit.sandbox.docker_provider._run_cli_async",
        new=AsyncMock(side_effect=[transient, _CliOutput(stdout=b"ok", stderr=b"")]),
    ) as run:
        output = await provider._run_lifecycle_cli_async(operation="ps", argv=("docker", "ps"))
    assert output.stdout == b"ok"
    assert run.await_count == 2


async def test_lifecycle_does_not_retry_ordinary_failure(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    ordinary = _CliInvocationError(
        argv=("docker", "compose", "up"),
        returncode=1,
        stdout=b"",
        stderr=b"invalid project",
        timed_out=False,
    )
    with (
        patch("pyrit.sandbox.docker_provider._run_cli_async", new=AsyncMock(side_effect=ordinary)) as run,
        pytest.raises(DockerLifecycleError, match="invalid project"),
    ):
        await provider._run_lifecycle_cli_async(operation="up", argv=("docker", "compose", "up"))
    assert run.await_count == 1


async def test_orphan_cleanup_uses_only_durable_provider_projects(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    await provider._register_session_state_async(project_name="pyrit-owned", session_id="session", task_id=None)
    with patch.object(provider, "_force_remove_project_resources_async", new_callable=AsyncMock) as remove:
        assert await provider.cleanup_orphans_async() == 1
    remove.assert_awaited_once_with(project_name="pyrit-owned")
    assert json.loads(provider._state_path.read_text(encoding="utf-8")) == {}


async def test_orphan_cleanup_retains_state_when_removal_fails(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    await provider._register_session_state_async(project_name="pyrit-owned", session_id="session", task_id=None)
    with patch.object(
        provider,
        "_force_remove_project_resources_async",
        new=AsyncMock(side_effect=DockerLifecycleError(operation="rm", detail="daemon unavailable")),
    ):
        assert await provider.cleanup_orphans_async() == 0
    assert "pyrit-owned" in json.loads(provider._state_path.read_text(encoding="utf-8"))


async def test_session_down_failure_preserves_state_for_orphan_recovery(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    spec = SandboxSessionSpec(session_id="stateful")
    session = DockerSandboxSession(
        provider=provider,
        spec=spec,
        project_name="pyrit-stateful",
        evidence_sink=None,
    )
    await provider._register_session_state_async(
        project_name=session.project_name,
        session_id=spec.session_id,
        task_id=None,
    )
    with (
        patch.object(
            session,
            "_down_async",
            new=AsyncMock(side_effect=DockerLifecycleError(operation="down", detail="failed")),
        ),
        pytest.raises(DockerLifecycleError, match="failed"),
    ):
        await session.close_async()
    assert session.project_name in json.loads(provider._state_path.read_text(encoding="utf-8"))


async def test_readiness_fails_fast_for_unhealthy_service(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    provider._service_names = ("default",)
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(),
        project_name="pyrit-health",
        evidence_sink=None,
    )
    unhealthy = {
        "ID": "container",
        "State": "running",
        "Status": "Up 1 second (unhealthy)",
        "Labels": "com.docker.compose.service=default",
    }
    with (
        patch.object(provider, "_list_containers_by_project_async", new=AsyncMock(return_value=[unhealthy])),
        pytest.raises(DockerLifecycleError, match="unhealthy"),
    ):
        await session._wait_ready_async()


class _FakeProcess:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.stdin = None
        self._killed = asyncio.Event()
        self.kill_count = 0

    async def wait(self) -> int:
        await self._killed.wait()
        return -9

    def kill(self) -> None:
        self.kill_count += 1
        self.returncode = -9
        self._killed.set()


class _CompletedProcess(_FakeProcess):
    def __init__(self, *, returncode: int) -> None:
        super().__init__()
        self.returncode = returncode
        self._killed.set()


async def test_docker_exec_timeout_terminates_host_client_and_is_typed(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(environments=(SandboxEnvironmentSpec(name="default"),)),
        project_name="pyrit-timeout",
        evidence_sink=None,
    )
    environment = session.get_environment()
    assert isinstance(environment, DockerSandboxEnvironment)
    process = _FakeProcess()
    handle = DockerSandboxProcess(
        process=process,
        environment=environment,
        started_at=datetime.now(tz=timezone.utc),
        release_semaphore=lambda: None,
        timeout_seconds=0.01,
        request=SandboxExecRequest(argv=("sleep", "10")),
    )
    result = await handle.communicate_async()
    assert result.status is SandboxOperationStatus.TIMED_OUT
    assert result.timed_out
    assert process.kill_count == 1


async def test_container_timeout_exit_is_typed_as_timeout(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(),
        project_name="pyrit-timeout-exit",
        evidence_sink=None,
    )
    environment = session.get_environment()
    assert isinstance(environment, DockerSandboxEnvironment)
    handle = DockerSandboxProcess(
        process=_CompletedProcess(returncode=124),
        environment=environment,
        started_at=datetime.now(tz=timezone.utc),
        release_semaphore=lambda: None,
        timeout_seconds=1,
        request=SandboxExecRequest(argv=("sleep", "10")),
        container_timeout_wrapped=True,
    )
    result = await handle.communicate_async()
    assert result.status is SandboxOperationStatus.TIMED_OUT
    assert result.timed_out
    assert result.exit_code == 124


async def test_binary_read_write_results_and_limits_without_docker(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(),
        project_name="pyrit-files",
        evidence_sink=None,
    )
    environment = session.get_environment()
    assert isinstance(environment, DockerSandboxEnvironment)
    environment.bind_container(container_id="container", service_name="default")
    binary = b"\x00\xff"
    successful_read = _RawRun(
        stdout=binary,
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
        exit_code=0,
        timed_out=False,
        cancelled=False,
    )
    successful_write = successful_read.__class__(
        stdout=b"",
        stderr=b"",
        stdout_truncated=False,
        stderr_truncated=False,
        exit_code=0,
        timed_out=False,
        cancelled=False,
    )
    with patch.object(
        environment,
        "_run_container_read_write_async",
        new=AsyncMock(side_effect=[(successful_write, None), (successful_read, None)]),
    ):
        write = await environment.write_file_async(path="binary.bin", data=binary)
        read = await environment.read_file_async(path="binary.bin")
    assert write.status is SandboxOperationStatus.SUCCEEDED
    assert read.data == binary
    assert (await environment.write_file_async(path="../escape", data=b"x")).status is (
        SandboxOperationStatus.PATH_ESCAPE
    )
