# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Local provider conformance and host-process behavior tests."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pyrit.executor.capability import CapabilityToolRuntime, InMemoryCapabilityEvidenceSink, ToolRegistry
from pyrit.models import ToolCallRequest
from pyrit.sandbox import (
    LocalSandboxProvider,
    LocalSandboxProviderConfig,
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxOperationStatus,
    SandboxProviderRegistry,
    SandboxSessionSpec,
    SandboxSetupFile,
    SandboxSetupScript,
    SandboxToolAdapter,
)
from unit.sandbox.conformance import ProviderConformanceSuite

if TYPE_CHECKING:
    from pathlib import Path


class TestLocalSandboxProviderConformance(ProviderConformanceSuite):
    """Run the provider-neutral suite against trusted local execution."""

    provider_factory = staticmethod(
        lambda path: LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=path))
    )
    python_command = staticmethod(lambda code: (sys.executable, "-c", code))


async def test_local_permission_errors_are_explicit(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            environment = session.get_environment()
            await environment.write_file_async(path="file.bin", data=b"x")
            with patch("pyrit.sandbox.local.aiofiles.open", side_effect=PermissionError("denied")):
                read = await environment.read_file_async(path="file.bin")
                write = await environment.write_file_async(path="other.bin", data=b"x")
            assert read.status is SandboxOperationStatus.PERMISSION_DENIED
            assert write.status is SandboxOperationStatus.PERMISSION_DENIED


async def test_local_other_filesystem_errors_are_sanitized(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            environment = session.get_environment()
            with patch("pyrit.sandbox.local.aiofiles.open", side_effect=OSError("secret-host-path")):
                read = await environment.read_file_async(path="file.bin")
                write = await environment.write_file_async(path="other.bin", data=b"x")
            assert read.status is SandboxOperationStatus.FAILED
            assert write.status is SandboxOperationStatus.FAILED
            assert "secret-host-path" not in (read.error_message or "")
            assert "secret-host-path" not in (write.error_message or "")


async def test_local_timeout_terminates_descendant_process(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    child_code = "import pathlib,time;time.sleep(2);pathlib.Path('orphan.txt').write_text('alive')"
    parent_code = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child_code!r}]);time.sleep(10)"
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            environment = session.get_environment()
            result = await environment.exec_async(
                request=SandboxExecRequest(
                    argv=(sys.executable, "-c", parent_code),
                    timeout_seconds=0.1,
                )
            )
            assert result.status is SandboxOperationStatus.TIMED_OUT
            await asyncio.sleep(2.2)
            assert (await environment.read_file_async(path="orphan.txt")).status is SandboxOperationStatus.NOT_FOUND


async def test_session_cleanup_terminates_unawaited_process_tree(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path, retain_workspaces=True))
    child_code = "import pathlib,time;time.sleep(1);pathlib.Path('unawaited.txt').write_text('alive')"
    parent_code = f"import subprocess,sys,time;subprocess.Popen([sys.executable,'-c',{child_code!r}]);time.sleep(10)"
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            await session.get_environment().start_process_async(
                request=SandboxExecRequest(argv=(sys.executable, "-c", parent_code))
            )
    await asyncio.sleep(1.2)
    assert not tuple(tmp_path.glob("session-*/environments/default/unawaited.txt"))


async def test_local_partial_setup_failure_cleans_session(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    spec = SandboxSessionSpec(
        environments=(
            SandboxEnvironmentSpec(
                name="default",
                setup_scripts=(
                    SandboxSetupScript(request=SandboxExecRequest(argv=(sys.executable, "-c", "raise SystemExit(2)"))),
                ),
            ),
        )
    )
    async with provider.managed_async():
        with pytest.raises(RuntimeError, match="Setup command failed"):
            async with provider.session_async(spec=spec):
                pytest.fail("Session setup should not complete.")
        assert not tuple(tmp_path.glob("session-*"))


async def test_local_orphan_cleanup(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    await provider.prepare_async()
    orphan = tmp_path / "session-orphan"
    orphan.mkdir()
    assert await provider.cleanup_orphans_async() == 1
    assert not orphan.exists()
    await provider.cleanup_async()
    await provider.cleanup_async()


async def test_local_retained_orphans_are_not_reported_as_cleaned(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path, retain_workspaces=True))
    await provider.prepare_async()
    orphan = tmp_path / "session-retained"
    orphan.mkdir()
    assert await provider.cleanup_orphans_async() == 0
    assert orphan.exists()
    await provider.cleanup_async()


async def test_closed_environment_rejects_new_operations(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    await provider.prepare_async()
    session = await provider.create_session_async(spec=SandboxSessionSpec())
    await session.initialize_async()
    environment = session.get_environment()
    await session.close_async()
    write = await environment.write_file_async(path="recreated.bin", data=b"x")
    read = await environment.read_file_async(path="recreated.bin")
    exec_result = await environment.exec_async(
        request=SandboxExecRequest(argv=(sys.executable, "-c", "print('unexpected')"))
    )
    assert write.error_code == "environment_closed"
    assert read.error_code == "environment_closed"
    assert exec_result.error_code == "environment_closed"
    assert not tuple(tmp_path.glob("session-*"))
    await provider.cleanup_async()


def test_local_registry_discovers_and_holds_provider(tmp_path: Path) -> None:
    registry = SandboxProviderRegistry()
    assert registry.get_class_names() == ["DockerSandboxProvider", "LocalSandboxProvider"]
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    registry.instances.register(provider, name="development")
    assert registry.instances.get("development") is provider


async def test_tool_adapter_links_sandbox_evidence_to_call_and_attempt(tmp_path: Path) -> None:
    sink = InMemoryCapabilityEvidenceSink()
    provider = LocalSandboxProvider(
        config=LocalSandboxProviderConfig(workspace_root=tmp_path),
        evidence_sink=sink,
    )
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            registry = ToolRegistry()
            SandboxToolAdapter(session=session).register(registry=registry)
            runtime = CapabilityToolRuntime(registry=registry)
            request_piece_id = uuid.uuid4()
            request = ToolCallRequest(
                call_id="call-1",
                name="sandbox_write_file",
                arguments=json.dumps({"path": "tool.bin", "data_base64": "AAE="}),
            )
            prepared, _ = await runtime.prepare_calls_async(
                calls=((request, request_piece_id),),
                case_id=uuid.uuid4(),
                conversation_id="conversation",
                asset_references=(),
                environment_requirement_references=(),
                cancellation_event=None,
            )
            record = await runtime.execute_call_async(
                call=prepared[0],
                case_id=uuid.uuid4(),
                conversation_id="conversation",
                asset_references=(),
                environment_requirement_references=(),
                cancellation_event=None,
            )
            assert record.result.error is None
            evidence = record.additional_evidence[0]
            assert evidence.evidence_type == "sandbox_operation"
            assert evidence.call_id == "call-1"
            assert evidence.attempt_id == record.execution_evidence[0].attempt_id
            assert record.artifact_evidence[0].created_by_call_id == "call-1"


async def test_tool_adapter_sanitizes_invalid_inputs(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            registry = ToolRegistry()
            SandboxToolAdapter(session=session).register(registry=registry)
            runtime = CapabilityToolRuntime(registry=registry)
            secret = "do-not-persist-this-command"
            request = ToolCallRequest(
                call_id="invalid",
                name="sandbox_exec",
                arguments=json.dumps(
                    {
                        "argv": [sys.executable],
                        "shell_script": secret,
                        "environment": "missing-secret-environment",
                    }
                ),
            )
            prepared, _ = await runtime.prepare_calls_async(
                calls=((request, uuid.uuid4()),),
                case_id=uuid.uuid4(),
                conversation_id="conversation",
                asset_references=(),
                environment_requirement_references=(),
                cancellation_event=None,
            )
            record = await runtime.execute_call_async(
                call=prepared[0],
                case_id=uuid.uuid4(),
                conversation_id="conversation",
                asset_references=(),
                environment_requirement_references=(),
                cancellation_event=None,
            )
            persisted = json.dumps(
                {
                    "result": record.result.model_dump(mode="json"),
                    "errors": [evidence.model_dump(mode="json") for evidence in record.error_evidence],
                }
            )
            assert secret not in persisted
            assert "missing-secret-environment" not in persisted


def test_local_provider_is_not_security_boundary() -> None:
    provider = LocalSandboxProvider()
    assert not provider.is_security_boundary


async def test_errors_and_evidence_do_not_expose_paths_or_secrets(tmp_path: Path) -> None:
    sink = InMemoryCapabilityEvidenceSink()
    provider = LocalSandboxProvider(
        config=LocalSandboxProviderConfig(workspace_root=tmp_path),
        evidence_sink=sink,
    )
    secret = "sandbox-secret-value"
    async with provider.managed_async():
        async with provider.session_async(spec=SandboxSessionSpec()) as session:
            environment = session.get_environment()
            missing = await environment.read_file_async(path="secret-filename.txt")
            escaped = await environment.exec_async(
                request=SandboxExecRequest(
                    argv=(sys.executable, "-c", "pass"),
                    cwd=str(tmp_path.parent / "host-secret-path"),
                )
            )
            await environment.exec_async(
                request=SandboxExecRequest(
                    argv=(sys.executable, "-c", "pass"),
                    environment={"TOKEN": secret},
                )
            )
            assert "secret-filename" not in (missing.error_message or "")
            assert "host-secret-path" not in (escaped.error_message or "")
    serialized_evidence = json.dumps([evidence.model_dump(mode="json") for evidence in await sink.snapshot_async()])
    assert secret not in serialized_evidence
    assert "host-secret-path" not in serialized_evidence


async def test_local_absolute_cwd_requires_explicit_opt_in(tmp_path: Path) -> None:
    restricted = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path / "restricted"))
    async with restricted.managed_async():
        async with restricted.session_async(spec=SandboxSessionSpec()) as session:
            result = await session.get_environment().exec_async(
                request=SandboxExecRequest(argv=(sys.executable, "-c", "print('no')"), cwd=str(tmp_path))
            )
            assert result.status is SandboxOperationStatus.FAILED
    unrestricted = LocalSandboxProvider(
        config=LocalSandboxProviderConfig(
            workspace_root=tmp_path / "unrestricted",
            allow_unrestricted_host_execution=True,
        )
    )
    async with unrestricted.managed_async():
        async with unrestricted.session_async(spec=SandboxSessionSpec()) as session:
            result = await session.get_environment().exec_async(
                request=SandboxExecRequest(argv=(sys.executable, "-c", "print('yes')"), cwd=str(tmp_path))
            )
            assert result.status is SandboxOperationStatus.SUCCEEDED


@pytest.mark.skipif(os.name == "nt", reason="Executable mode bits are POSIX-specific.")
async def test_setup_file_executable_mode(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    spec = SandboxSessionSpec(
        environments=(
            SandboxEnvironmentSpec(
                name="default",
                setup_files=(
                    SandboxSetupFile(path="script", content=b"#!/bin/sh\nprintf executable", executable=True),
                ),
            ),
        )
    )
    async with provider.managed_async():
        async with provider.session_async(spec=spec) as session:
            result = await session.get_environment().exec_async(request=SandboxExecRequest(argv=("./script",)))
            assert result.status is SandboxOperationStatus.SUCCEEDED
            assert result.stdout == b"executable"
