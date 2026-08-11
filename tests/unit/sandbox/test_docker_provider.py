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
    DockerStateSecurityError,
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
    _run_cli_async,
    _scan_security_violations,
    _select_service_name,
    _synthesize_compose_document,
    _synthesize_policy_overlay,
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


def test_default_state_path_is_scoped_to_user_state_directory(tmp_path: Path) -> None:
    config = _provider_config(tmp_path).model_copy(update={"state_dir": None})
    with patch("pyrit.sandbox.docker_provider.user_state_dir", return_value=str(tmp_path / "user-state")):
        provider = DockerSandboxProvider(config=config)
    assert provider._state_path == tmp_path / "user-state" / "sandbox" / "docker" / "pyrit-sbx.state.json"


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


def test_policy_overlay_preserves_topology_and_marks_owned_networks_internal() -> None:
    resolved = {
        "services": {
            "attacker": {"networks": {"arena": {"aliases": ["operator"]}}},
            "victim": {"networks": {"arena": None}},
        },
        "networks": {"arena": {"driver": "bridge"}},
        "volumes": {"evidence": {}},
    }
    overlay = _synthesize_policy_overlay(
        resolved_document=resolved,
        policy=DockerSecurityPolicy(allow_egress=False),
        ownership_id="owner-token",
    )
    assert overlay["services"]["attacker"] == {"labels": {"com.pyrit.sandbox.owner": "owner-token"}}
    assert overlay["networks"]["arena"] == {
        "internal": True,
        "labels": {"com.pyrit.sandbox.owner": "owner-token"},
    }
    assert overlay["volumes"]["evidence"] == {"labels": {"com.pyrit.sandbox.owner": "owner-token"}}


def test_project_names_are_bounded_and_execution_owner_unique() -> None:
    first = _project_name_for_session(
        prefix="pyrit",
        session_id="S" * 200,
        attempt_id="attempt-one",
        ownership_id="owner-one",
    )
    second = _project_name_for_session(
        prefix="pyrit",
        session_id="S" * 200,
        attempt_id="attempt-two",
        ownership_id="owner-two",
    )
    same_attempt_new_owner = _project_name_for_session(
        prefix="pyrit",
        session_id="S" * 200,
        attempt_id="attempt-one",
        ownership_id="owner-two",
    )
    assert first != second
    assert first != same_attempt_new_owner
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


@pytest.mark.parametrize(
    "network",
    [
        {"internal": False},
        {"external": True},
    ],
)
def test_security_policy_rejects_network_without_enforced_egress_isolation(
    network: dict[str, object],
) -> None:
    violations = _scan_security_violations(
        resolved_document={
            "services": {"default": {"networks": {"default": None}}},
            "networks": {"default": network},
        },
        policy=DockerSecurityPolicy(allow_egress=False),
    )
    assert any("network 'default'" in violation for violation in violations)


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
        new=AsyncMock(
            side_effect=[
                _CliOutput(stdout=b"", stderr=b""),
                _CliOutput(stdout=safe_config, stderr=b""),
                _CliOutput(stdout=safe_config, stderr=b""),
            ]
        ),
    ) as run:
        await provider.prepare_async()
    assert run.await_count == 3
    config_argv = run.await_args_list[2].kwargs["argv"]
    assert config_argv[-3:] == ["config", "--format", "json"]
    await provider.cleanup_async()


async def test_prepare_enforces_internal_network_for_explicit_compose(tmp_path: Path) -> None:
    config = _provider_config(tmp_path).model_copy(update={"security_policy": DockerSecurityPolicy(allow_egress=False)})
    provider = DockerSandboxProvider(config=config)
    initial = json.dumps(
        {
            "services": {"default": {"image": "python:3.12-alpine", "networks": {"default": None}}},
            "networks": {"default": {"internal": False}},
        }
    ).encode()
    hardened = json.dumps(
        {
            "services": {"default": {"image": "python:3.12-alpine", "networks": {"default": None}}},
            "networks": {"default": {"internal": True}},
        }
    ).encode()
    with patch(
        "pyrit.sandbox.docker_provider._run_cli_async",
        new=AsyncMock(
            side_effect=[
                _CliOutput(stdout=b"", stderr=b""),
                _CliOutput(stdout=initial, stderr=b""),
                _CliOutput(stdout=hardened, stderr=b""),
            ]
        ),
    ) as run:
        await provider.prepare_async()
    assert provider._policy_overlay_path is not None
    overlay = json.loads(provider._policy_overlay_path.read_text(encoding="utf-8"))
    assert overlay["networks"]["default"]["internal"] is True
    assert overlay["services"]["default"]["labels"] == {"com.pyrit.sandbox.owner": provider._ownership_id}
    assert str(provider._policy_overlay_path) in run.await_args_list[2].kwargs["argv"]
    await provider.cleanup_async()


async def test_prepare_rejects_unsafe_resolved_config(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    unsafe_config = json.dumps({"services": {"default": {"privileged": True}}}).encode()
    with (
        patch(
            "pyrit.sandbox.docker_provider._run_cli_async",
            new=AsyncMock(
                side_effect=[
                    _CliOutput(stdout=b"", stderr=b""),
                    _CliOutput(stdout=unsafe_config, stderr=b""),
                    _CliOutput(stdout=unsafe_config, stderr=b""),
                ]
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


async def test_lifecycle_does_not_retry_timeout_and_preserves_diagnostics(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    timeout = _CliInvocationError(
        argv=("docker", "ps"),
        returncode=None,
        stdout=b"partial stdout",
        stderr=b"partial diagnostic",
        timed_out=True,
    )
    with (
        patch("pyrit.sandbox.docker_provider._run_cli_async", new=AsyncMock(side_effect=timeout)) as run,
        pytest.raises(DockerLifecycleError, match="timed out after 0.01 seconds: partial diagnostic"),
    ):
        await provider._run_lifecycle_cli_async(
            operation="ps",
            argv=("docker", "ps"),
            timeout_seconds=0.01,
        )
    assert run.await_count == 1


async def test_lifecycle_does_not_retry_mutating_transient_failure(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    transient = _CliInvocationError(
        argv=("docker", "compose", "up"),
        returncode=1,
        stdout=b"",
        stderr=b"connection reset by peer",
        timed_out=False,
    )
    with (
        patch("pyrit.sandbox.docker_provider._run_cli_async", new=AsyncMock(side_effect=transient)) as run,
        pytest.raises(DockerLifecycleError, match="connection reset by peer"),
    ):
        await provider._run_lifecycle_cli_async(operation="up", argv=("docker", "compose", "up"))
    assert run.await_count == 1


class _TimedOutCliProcess:
    def __init__(self) -> None:
        self.pid = 123
        self.returncode: int | None = None
        self._terminated = asyncio.Event()
        self.communicate_started = asyncio.Event()

    async def communicate(self, **kwargs: bytes | None) -> tuple[bytes, bytes]:
        del kwargs
        self.communicate_started.set()
        await self._terminated.wait()
        return b"partial stdout", b"partial stderr"

    def kill(self) -> None:
        self.returncode = -9
        self._terminated.set()


async def test_cli_timeout_terminates_tree_and_drains_partial_output() -> None:
    process = _TimedOutCliProcess()

    async def terminate(*, process: _TimedOutCliProcess) -> None:
        process.kill()

    with (
        patch("pyrit.sandbox.docker_provider.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
        patch(
            "pyrit.sandbox.docker_provider._terminate_cli_process_tree_async",
            new=AsyncMock(side_effect=terminate),
        ) as terminate_tree,
        pytest.raises(_CliInvocationError) as exc_info,
    ):
        await _run_cli_async(
            argv=("docker", "ps"),
            cwd=None,
            timeout_seconds=0.001,
            input_bytes=None,
            semaphore=asyncio.Semaphore(1),
        )

    assert exc_info.value.timed_out
    assert exc_info.value.stdout == b"partial stdout"
    assert exc_info.value.stderr == b"partial stderr"
    terminate_tree.assert_awaited_once_with(process=process)


async def test_cli_cancellation_terminates_process_tree() -> None:
    process = _TimedOutCliProcess()

    async def terminate(*, process: _TimedOutCliProcess) -> None:
        process.kill()

    with (
        patch("pyrit.sandbox.docker_provider.asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)),
        patch(
            "pyrit.sandbox.docker_provider._terminate_cli_process_tree_async",
            new=AsyncMock(side_effect=terminate),
        ) as terminate_tree,
    ):
        invocation = asyncio.create_task(
            _run_cli_async(
                argv=("docker", "ps"),
                cwd=None,
                timeout_seconds=10,
                input_bytes=None,
                semaphore=asyncio.Semaphore(1),
            )
        )
        await process.communicate_started.wait()
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await invocation

    terminate_tree.assert_awaited_once_with(process=process)


async def test_orphan_cleanup_uses_only_durable_provider_projects(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    project_name = "pyrit-sbx-owned"
    await provider._register_session_state_async(project_name=project_name, session_id="session", task_id=None)
    with (
        patch("pyrit.sandbox.docker_provider._process_is_alive", return_value=False),
        patch.object(provider, "_force_remove_project_resources_async", new_callable=AsyncMock) as remove,
    ):
        assert await provider.cleanup_orphans_async() == 1
    remove.assert_awaited_once_with(project_name=project_name, ownership_id=provider._ownership_id)
    assert json.loads(provider._state_path.read_text(encoding="utf-8")) == {}


async def test_orphan_cleanup_retains_state_when_removal_fails(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    project_name = "pyrit-sbx-owned"
    await provider._register_session_state_async(project_name=project_name, session_id="session", task_id=None)
    with (
        patch("pyrit.sandbox.docker_provider._process_is_alive", return_value=False),
        patch.object(
            provider,
            "_force_remove_project_resources_async",
            new=AsyncMock(side_effect=DockerLifecycleError(operation="rm", detail="daemon unavailable")),
        ),
    ):
        assert await provider.cleanup_orphans_async() == 0
    assert project_name in json.loads(provider._state_path.read_text(encoding="utf-8"))


async def test_orphan_cleanup_skips_explicitly_retained_projects(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(
        config=_provider_config(tmp_path).model_copy(update={"retain_resources_on_close": True})
    )
    project_name = "pyrit-sbx-retained"
    await provider._register_session_state_async(project_name=project_name, session_id="session", task_id=None)
    with (
        patch("pyrit.sandbox.docker_provider._process_is_alive", return_value=False),
        patch.object(provider, "_force_remove_project_resources_async", new_callable=AsyncMock) as remove,
    ):
        assert await provider.cleanup_orphans_async() == 0
    remove.assert_not_awaited()
    assert project_name in json.loads(provider._state_path.read_text(encoding="utf-8"))


async def test_orphan_cleanup_skips_live_and_unowned_state_records(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    live_project = "pyrit-sbx-live"
    legacy_project = "pyrit-sbx-legacy"
    await provider._register_session_state_async(project_name=live_project, session_id="live", task_id=None)
    state = json.loads(provider._state_path.read_text(encoding="utf-8"))
    state[legacy_project] = {
        "session_id": "legacy",
        "owner_process_id": 0,
        "ownership_id": provider._ownership_id,
    }
    provider._state_path.write_text(json.dumps(state), encoding="utf-8")
    with patch.object(provider, "_force_remove_project_resources_async", new_callable=AsyncMock) as remove:
        assert await provider.cleanup_orphans_async() == 0
    remove.assert_not_awaited()


async def test_state_file_symlink_is_rejected(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    provider._state_path.parent.mkdir(parents=True)
    target = tmp_path / "attacker-controlled.json"
    target.write_text("{}", encoding="utf-8")
    try:
        provider._state_path.symlink_to(target)
    except OSError:
        pytest.skip("Symbolic links are unavailable on this host")
    with pytest.raises(DockerStateSecurityError, match="symbolic link"):
        await provider._register_session_state_async(
            project_name="pyrit-sbx-owned",
            session_id="session",
            task_id=None,
        )


async def test_container_discovery_requires_resource_ownership_label(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    with patch.object(
        provider,
        "_run_lifecycle_cli_async",
        new=AsyncMock(return_value=_CliOutput(stdout=b"", stderr=b"")),
    ) as run:
        await provider._list_containers_by_project_async(
            project_name="pyrit-sbx-owned",
            ownership_id="owner-token",
        )
    assert "label=com.pyrit.sandbox.owner=owner-token" in run.await_args.kwargs["argv"]


@pytest.mark.parametrize("detail", ["failed", "timed out after 0.1 seconds"])
async def test_session_cleanup_failure_preserves_durable_state(tmp_path: Path, detail: str) -> None:
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
    cleanup_failure = DockerLifecycleError(operation="rm", detail=detail)
    with patch.object(
        provider,
        "_force_remove_project_resources_async",
        new=AsyncMock(side_effect=cleanup_failure),
    ) as force_remove:
        with pytest.raises(DockerLifecycleError, match=detail):
            await session.close_async()
    force_remove.assert_awaited_once_with(
        project_name=session.project_name,
        ownership_id=provider._ownership_id,
    )
    assert session.project_name in json.loads(provider._state_path.read_text(encoding="utf-8"))


async def test_session_success_sweeps_only_owned_project_resources(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(session_id="successful"),
        project_name="pyrit-successful",
        evidence_sink=None,
    )
    with patch.object(
        provider,
        "_force_remove_project_resources_async",
        new_callable=AsyncMock,
    ) as force_remove:
        await session.close_async()
    force_remove.assert_awaited_once_with(
        project_name=session.project_name,
        ownership_id=provider._ownership_id,
    )


async def test_session_cleanup_failure_preserves_state_for_provider_retry(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    spec = SandboxSessionSpec(session_id="stateful")
    session = DockerSandboxSession(
        provider=provider,
        spec=spec,
        project_name="pyrit-stateful",
        evidence_sink=None,
    )
    provider._sessions[spec.session_id] = session
    await provider._register_session_state_async(
        project_name=session.project_name,
        session_id=spec.session_id,
        task_id=None,
    )
    cleanup_failure = DockerLifecycleError(operation="rm", detail="daemon unavailable")
    with patch.object(
        provider,
        "_force_remove_project_resources_async",
        new=AsyncMock(side_effect=[cleanup_failure, None]),
    ) as force_remove:
        with pytest.raises(DockerLifecycleError, match="daemon unavailable"):
            await session.close_async()
        await provider._cleanup_async()
    assert force_remove.await_count == 2
    assert session.project_name not in json.loads(provider._state_path.read_text(encoding="utf-8"))
    assert spec.session_id not in provider._sessions


async def test_session_close_cancellation_waits_for_owned_cleanup(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(config=_provider_config(tmp_path))
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(session_id="cancelled"),
        project_name="pyrit-cancelled",
        evidence_sink=None,
    )
    cleanup_started = asyncio.Event()
    release_cleanup = asyncio.Event()

    async def cleanup_owned_resources(*, project_name: str, ownership_id: str) -> None:
        del project_name, ownership_id
        cleanup_started.set()
        await release_cleanup.wait()

    with patch.object(
        provider,
        "_force_remove_project_resources_async",
        new=AsyncMock(side_effect=cleanup_owned_resources),
    ) as force_remove:
        close = asyncio.create_task(session.close_async())
        await cleanup_started.wait()
        close.cancel()
        await asyncio.sleep(0)
        assert not close.done()
        close.cancel()
        await asyncio.sleep(0)
        assert not close.done()
        release_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await close
    force_remove.assert_awaited_once_with(
        project_name=session.project_name,
        ownership_id=provider._ownership_id,
    )


async def test_provider_recovery_preserves_explicitly_retained_resources(tmp_path: Path) -> None:
    provider = DockerSandboxProvider(
        config=_provider_config(tmp_path).model_copy(update={"retain_resources_on_close": True})
    )
    session = DockerSandboxSession(
        provider=provider,
        spec=SandboxSessionSpec(session_id="retained"),
        project_name="pyrit-retained",
        evidence_sink=None,
    )
    provider._sessions[session.session_id] = session
    await provider._register_session_state_async(
        project_name=session.project_name,
        session_id=session.session_id,
        task_id=None,
    )
    with (
        patch.object(session, "close_async", new=AsyncMock(side_effect=RuntimeError("bookkeeping failed"))),
        patch.object(provider, "_force_remove_project_resources_async", new_callable=AsyncMock) as force_remove,
        pytest.raises(RuntimeError, match="1 Docker sandbox session cleanup"),
    ):
        await provider._cleanup_async()
    force_remove.assert_not_awaited()
    assert session.project_name not in json.loads(provider._state_path.read_text(encoding="utf-8"))
    assert session.session_id not in provider._sessions


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
