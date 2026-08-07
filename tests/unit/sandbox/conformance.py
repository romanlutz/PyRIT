# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Reusable provider-neutral sandbox conformance tests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from pyrit.executor.capability import InMemoryCapabilityEvidenceSink
from pyrit.sandbox import (
    SandboxEnvironmentSpec,
    SandboxExecRequest,
    SandboxLimits,
    SandboxOperationStatus,
    SandboxSessionSpec,
    SandboxSetupFile,
    SandboxSetupScript,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from pyrit.sandbox import SandboxProvider


class ProviderConformanceSuite:
    """Provider-neutral tests inherited by each provider implementation."""

    provider_factory: Callable[[Path], SandboxProvider]
    python_command: Callable[[str], tuple[str, ...]]

    async def test_named_and_default_environments(self, tmp_path: Path) -> None:
        provider = self.provider_factory(tmp_path)
        spec = SandboxSessionSpec(
            environments=(
                SandboxEnvironmentSpec(name="zeta"),
                SandboxEnvironmentSpec(name="alpha"),
            )
        )
        async with provider.managed_async():
            async with provider.session_async(spec=spec) as session:
                assert session.get_environment().name == "alpha"
                assert session.get_environment("zeta").name == "zeta"

    async def test_binary_empty_missing_escape_and_file_limits(self, tmp_path: Path) -> None:
        provider = self.provider_factory(tmp_path)
        limits = SandboxLimits(max_read_bytes=4, max_write_bytes=4)
        spec = SandboxSessionSpec(environments=(SandboxEnvironmentSpec(name="default", limits=limits),))
        async with provider.managed_async():
            async with provider.session_async(spec=spec) as session:
                environment = session.get_environment()
                binary = await environment.write_file_async(path="binary.bin", data=b"\x00\xff")
                assert binary.status is SandboxOperationStatus.SUCCEEDED
                assert (await environment.read_file_async(path="binary.bin")).data == b"\x00\xff"
                assert (await environment.write_file_async(path="empty.bin", data=b"")).status is (
                    SandboxOperationStatus.SUCCEEDED
                )
                assert (await environment.read_file_async(path="empty.bin")).data == b""
                assert (await environment.read_file_async(path="missing.bin")).status is (
                    SandboxOperationStatus.NOT_FOUND
                )
                assert (await environment.read_file_async(path="../escape.bin")).status is (
                    SandboxOperationStatus.PATH_ESCAPE
                )
                assert (await environment.write_file_async(path="../escape.bin", data=b"x")).status is (
                    SandboxOperationStatus.PATH_ESCAPE
                )
                assert (await environment.write_file_async(path="large.bin", data=b"12345")).status is (
                    SandboxOperationStatus.TOO_LARGE
                )
                await environment.write_file_async(path="read-limit.bin", data=b"1234")
                assert (await environment.read_file_async(path="read-limit.bin", max_bytes=2)).status is (
                    SandboxOperationStatus.TOO_LARGE
                )
                assert (await environment.read_file_async(path="read-limit.bin", max_bytes=-1)).status is (
                    SandboxOperationStatus.FAILED
                )

    async def test_exec_forms_and_inputs(self, tmp_path: Path) -> None:
        provider = self.provider_factory(tmp_path)
        async with provider.managed_async():
            async with provider.session_async(spec=SandboxSessionSpec()) as session:
                environment = session.get_environment()
                await environment.write_file_async(path="sub/input.txt", data=b"cwd")
                code = (
                    "import os,sys;"
                    "data=sys.stdin.buffer.read();"
                    "sys.stdout.buffer.write(data+b'|'+os.environ['CONFORMANCE'].encode()+b'|'+"
                    "os.path.basename(os.getcwd()).encode());"
                    "sys.stderr.write('stderr')"
                )
                result = await environment.exec_async(
                    request=SandboxExecRequest(
                        argv=self.python_command(code),
                        stdin=b"stdin",
                        environment={"CONFORMANCE": "env"},
                        cwd="sub",
                    )
                )
                assert result.status is SandboxOperationStatus.SUCCEEDED
                assert result.stdout == b"stdin|env|sub"
                assert result.stderr == b"stderr"
                assert result.exit_code == 0
                failed = await environment.exec_async(
                    request=SandboxExecRequest(argv=self.python_command("raise SystemExit(7)"))
                )
                assert failed.status is SandboxOperationStatus.FAILED
                assert failed.exit_code == 7
                shell = await environment.exec_async(request=SandboxExecRequest(shell_script="echo shell-output"))
                assert shell.status is SandboxOperationStatus.SUCCEEDED
                assert b"shell-output" in shell.stdout

    async def test_timeout_cancellation_and_output_limits(self, tmp_path: Path) -> None:
        provider = self.provider_factory(tmp_path)
        limits = SandboxLimits(max_stdout_bytes=16, max_stderr_bytes=8, max_exec_seconds=2)
        spec = SandboxSessionSpec(environments=(SandboxEnvironmentSpec(name="default", limits=limits),))
        async with provider.managed_async():
            async with provider.session_async(spec=spec) as session:
                environment = session.get_environment()
                truncated = await environment.exec_async(
                    request=SandboxExecRequest(
                        argv=self.python_command("import sys;print('x'*100);sys.stderr.write('y'*100)")
                    )
                )
                assert truncated.status is SandboxOperationStatus.TRUNCATED
                assert len(truncated.stdout) == 16
                assert len(truncated.stderr) == 8
                timed_out = await environment.exec_async(
                    request=SandboxExecRequest(
                        argv=self.python_command("import time;time.sleep(10)"),
                        timeout_seconds=0.05,
                    )
                )
                assert timed_out.status is SandboxOperationStatus.TIMED_OUT
                assert timed_out.timed_out
                cancellation = asyncio.Event()
                task = asyncio.create_task(
                    environment.exec_async(
                        request=SandboxExecRequest(argv=self.python_command("import time;time.sleep(10)")),
                        cancellation_event=cancellation,
                    )
                )
                await asyncio.sleep(0.05)
                cancellation.set()
                cancelled = await task
                assert cancelled.status is SandboxOperationStatus.CANCELLED
                assert cancelled.cancelled

    async def test_setup_evidence_concurrency_and_cleanup(self, tmp_path: Path) -> None:
        sink = InMemoryCapabilityEvidenceSink()
        provider = self.provider_factory(tmp_path)
        spec = SandboxSessionSpec(
            environments=(
                SandboxEnvironmentSpec(
                    name="default",
                    setup_files=(SandboxSetupFile(path="setup.bin", content=b"setup"),),
                    setup_scripts=(
                        SandboxSetupScript(
                            request=SandboxExecRequest(
                                argv=self.python_command(
                                    "from pathlib import Path;Path('script.bin').write_bytes(b'ok')"
                                )
                            )
                        ),
                    ),
                ),
            )
        )
        async with provider.managed_async():
            async with provider.session_async(spec=spec, evidence_sink=sink) as session:
                environment = session.get_environment()
                assert (await environment.read_file_async(path="setup.bin")).data == b"setup"
                assert (await environment.read_file_async(path="script.bin")).data == b"ok"
                writes = await asyncio.gather(
                    *(environment.write_file_async(path=f"concurrent/{index}.bin", data=b"x") for index in range(8))
                )
                assert all(result.status is SandboxOperationStatus.SUCCEEDED for result in writes)
            await session.close_async()
        events = await sink.snapshot_async()
        operations = {event.operation for event in events if event.evidence_type == "sandbox_operation"}
        assert {"session_create", "environment_setup", "session_setup", "session_cleanup"} <= operations
