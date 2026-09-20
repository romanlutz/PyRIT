# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping


class DockerCommandError(RuntimeError):
    """A Docker control operation failed, without changing any unrelated resource."""


class DockerExecutionError(DockerCommandError):
    """A terminal execution failure with retained command and transport evidence."""

    def __init__(
        self,
        *,
        reason: str,
        execution_id: str,
        execution_started: bool,
        result: dict[str, Any] | None,
        diagnostics: dict[str, Any],
    ) -> None:
        """Retain partial evidence for the harness to journal before termination."""
        super().__init__(f"Terminal Docker execution failure: {reason}.")
        self.reason = reason
        self.execution_id = execution_id
        self.execution_started = execution_started
        self.result = result
        self.diagnostics = diagnostics


@dataclass(frozen=True, kw_only=True)
class CommandResult:
    """Bounded output from one process, with explicit incomplete-output indicators."""

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False
    truncated: bool = False


def _decode_output(*, value: bytes | bytearray, limit: int) -> tuple[str, bool]:
    decoded = value.decode("utf-8", errors="replace").encode("utf-8")
    return decoded[:limit].decode("utf-8", errors="ignore"), len(decoded) > limit


@dataclass(kw_only=True)
class _ExecutionCapture:
    execution_id: str
    output_limit: int
    started: bool = False
    finished: bool = False
    returncode: int | None = None
    timed_out: bool = False
    truncated: bool = False
    protocol_error: str | None = None
    launch_error: str | None = None
    streams: dict[str, bytearray] = field(default_factory=lambda: {"stdout": bytearray(), "stderr": bytearray()})

    def accept(self, line: str) -> None:
        record = json.loads(line)
        if not isinstance(record, dict) or record.get("execution_id") != self.execution_id:
            raise ValueError("Missing or mismatched runner execution identity.")
        if self.finished or self.launch_error is not None:
            raise ValueError("Unexpected runner data after its terminal record.")
        event = record.get("event")
        if event == "started":
            if self.started or set(record) != {"event", "execution_id", "pid"}:
                raise ValueError("Invalid or repeated runner start record.")
            if type(record["pid"]) is not int or record["pid"] <= 0:
                raise ValueError("Invalid child process identity.")
            self.started = True
        elif event == "launch_error":
            if self.started or set(record) != {"event", "execution_id", "message"}:
                raise ValueError("Invalid runner launch-error record.")
            if not isinstance(record["message"], str):
                raise ValueError("Invalid runner launch-error message.")
            self.launch_error = record["message"]
        elif not self.started:
            raise ValueError("Runner output is missing its child-start record.")
        elif event == "output":
            self._accept_output(record)
        elif event == "finished":
            self._accept_finish(record)
        else:
            raise ValueError("Unsupported runner event.")

    def tool_result(self, *, transport_timed_out: bool) -> dict[str, Any] | None:
        if not self.started:
            return None
        stdout, stdout_truncated = _decode_output(value=self.streams["stdout"], limit=self.output_limit)
        stderr, stderr_truncated = _decode_output(value=self.streams["stderr"], limit=self.output_limit)
        timed_out = self.timed_out or transport_timed_out
        truncated = self.truncated or stdout_truncated or stderr_truncated
        returncode = None if timed_out else self.returncode
        error = "tool_timeout" if timed_out else "output_truncated" if truncated else None
        if error is None and returncode != 0:
            error = "nonzero_exit"
        return {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "timed_out": timed_out,
            "truncated": truncated,
            "error": error,
            "execution_id": self.execution_id,
        }

    def _accept_output(self, record: dict[str, Any]) -> None:
        if set(record) != {"event", "execution_id", "stream", "data"}:
            raise ValueError("Invalid runner output record.")
        stream, encoded = record["stream"], record["data"]
        if not isinstance(stream, str) or stream not in self.streams or not isinstance(encoded, str):
            raise ValueError("Invalid runner output stream or encoding.")
        chunk = base64.b64decode(encoded, validate=True)
        if len(self.streams[stream]) + len(chunk) > self.output_limit:
            raise ValueError("Runner output exceeded the agreed capture limit.")
        self.streams[stream].extend(chunk)

    def _accept_finish(self, record: dict[str, Any]) -> None:
        if set(record) != {"event", "execution_id", "returncode", "timed_out", "truncated"}:
            raise ValueError("Invalid runner completion record.")
        if type(record["timed_out"]) is not bool or type(record["truncated"]) is not bool:
            raise ValueError("Invalid runner completion flags.")
        returncode = record["returncode"]
        if record["timed_out"]:
            if returncode is not None:
                raise ValueError("A timed-out command cannot have a completed exit code.")
        elif type(returncode) is not int:
            raise ValueError("A completed command must have an integer exit code.")
        self.returncode, self.timed_out, self.truncated = returncode, record["timed_out"], record["truncated"]
        self.finished = True


async def _drain_async(*, stream: asyncio.StreamReader, buffer: bytearray, limit: int) -> bool:
    truncated = False
    while chunk := await stream.read(8192):
        remaining = limit - len(buffer)
        buffer.extend(chunk[:remaining])
        truncated |= len(chunk) > remaining
    return truncated


async def run_process_async(*, arguments: list[str], timeout: float, output_limit: int = 16384) -> CommandResult:
    """
    Run an argv directly, never through the host shell, and drain bounded output.

    Returns:
        CommandResult: Captured output and completion information.
    """
    process = await asyncio.create_subprocess_exec(
        *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    assert process.stdout is not None and process.stderr is not None
    stdout, stderr = bytearray(), bytearray()
    readers = [
        asyncio.create_task(_drain_async(stream=process.stdout, buffer=stdout, limit=output_limit)),
        asyncio.create_task(_drain_async(stream=process.stderr, buffer=stderr, limit=output_limit)),
    ]
    timed_out = False
    try:
        await asyncio.wait_for(process.wait(), timeout=timeout)
    except TimeoutError:
        timed_out = True
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
        truncated = any(await asyncio.gather(*readers))
    decoded = [_decode_output(value=buffer, limit=output_limit) for buffer in (stdout, stderr)]
    truncated |= any(item[1] for item in decoded)
    return CommandResult(
        stdout=decoded[0][0],
        stderr=decoded[1][0],
        returncode=None if timed_out else process.returncode,
        timed_out=timed_out,
        truncated=truncated,
    )


class DockerCTFEnvironment:
    """Own one fresh Linux container and route tool arguments only to Docker exec."""

    OWNER_LABEL = "pyrit.native-ctf.owner"
    OUTPUT_LIMIT = 16384
    TRANSPORT_OUTPUT_LIMIT = 65536

    def __init__(
        self,
        *,
        image: str,
        run_id: str,
        platform: str = "linux/amd64",
        cpus: float = 1,
        memory_bytes: int = 512 * 1024 * 1024,
        pids_limit: int = 128,
        max_tool_executions: int = 8,
        tool_timeout_seconds: float = 30,
    ) -> None:
        """
        Initialize a narrowly scoped environment with a digest-pinned image.

        Raises:
            ValueError: If image, platform, identity, or resource bounds are invalid.
        """
        UUID(run_id)
        if not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", image):
            raise ValueError("The Docker image must include an immutable sha256 digest.")
        if platform != "linux/amd64":
            raise ValueError("This prototype supports only linux/amd64 containers.")
        if not (0 < cpus <= 1 and 0 < memory_bytes <= 512 * 1024 * 1024 and 0 < pids_limit <= 128):
            raise ValueError("Resources must not exceed 1 CPU, 512 MiB, or 128 PIDs.")
        if not (0 < max_tool_executions <= 8 and 0 < tool_timeout_seconds <= 30):
            raise ValueError("Tool bounds must not exceed eight executions or thirty seconds.")
        self.image = image
        self.run_id = run_id
        self.platform = platform
        self.cpus = cpus
        self.memory_bytes = memory_bytes
        self.pids_limit = pids_limit
        self.max_tool_executions = max_tool_executions
        self.tool_timeout_seconds = tool_timeout_seconds
        self._command_timeout_seconds = max(tool_timeout_seconds - 1, tool_timeout_seconds / 2)
        self.name = f"pyrit-native-ctf-{run_id}"
        self.container_id: str | None = None
        self.image_identity: dict[str, Any] = {}
        self.container_identity: dict[str, Any] = {}
        self.executions = 0
        self.execution_attempts = 0
        self.timed_out = False
        self.terminal_error: str | None = None
        self.cleanup_status = "not_created"
        self._create_started = False
        self._execution_lock = asyncio.Lock()
        self._execution_deadline: float | None = None
        self._runner_source = Path(__file__).with_name("_command_runner.py").read_text(encoding="utf-8")

    async def acquire_image_async(self) -> None:
        """
        Acquire the pinned image before the episode clock starts.

        Raises:
            DockerCommandError: If acquisition fails or the platform differs.
        """
        await self._control_async(arguments=["pull", "--platform", self.platform, self.image], timeout=300)
        result = await self._control_async(arguments=["image", "inspect", self.image])
        image = json.loads(result.stdout)[0]
        if image["Os"] != "linux" or image["Architecture"] != "amd64":
            raise DockerCommandError("The acquired image does not match linux/amd64.")
        self.image_identity = {key: image[key] for key in ("Id", "RepoDigests", "Os", "Architecture")}
        version = await self._control_async(arguments=["version", "--format", "{{json .Server}}"])
        self.image_identity["docker_server"] = json.loads(version.stdout)

    async def start_async(self, *, files: Mapping[str, Path]) -> None:
        """
        Create a fresh container, verify isolation, and copy only named task files.

        Raises:
            DockerCommandError: If Docker fails or its effective isolation differs.
            ValueError: If a task filename is not a flat, regular file.
        """
        if self._create_started:
            raise ValueError("An environment is single-use; create a new one for another episode.")
        for name, path in files.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise ValueError(f"Task files must have safe flat names: {name!r}")
            if not await asyncio.to_thread(path.is_file) or await asyncio.to_thread(path.is_symlink):
                raise ValueError(f"Task asset must be a regular, non-symlink file: {path}")
        self._create_started = True
        self.cleanup_status = "pending"
        result = await self._control_async(arguments=self._create_arguments())
        self.container_id = result.stdout.strip()
        if not re.fullmatch(r"[0-9a-f]{64}", self.container_id):
            raise DockerCommandError("Docker create did not return a full container ID.")
        await self._control_async(arguments=["start", self.container_id])
        inspected = await self._control_async(arguments=["container", "inspect", self.container_id])
        self.container_identity = json.loads(inspected.stdout)[0]
        self._verify_isolation()
        for name, path in files.items():
            source = str(await asyncio.to_thread(path.resolve))
            await self._control_async(arguments=["cp", "--", source, f"{self.container_id}:/workspace/{name}"])

    async def execute_async(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Execute one bash or Python command inside the owned container.

        A validated child-start record proves execution; Docker's exit code alone
        does not. The harness must journal a terminal failure before calling
        ``terminate_async`` and the outer run must still perform owned cleanup.

        Returns:
            dict[str, Any]: The shared JSON tool-result envelope.

        Raises:
            ValueError: If the tool name or argument schema is invalid.
            RuntimeError: If the container is unavailable or the execution budget is exhausted.
            DockerExecutionError: If Docker execution, the runner protocol, or a tool timeout fails.
        """
        argument_name = {"bash": "command", "python": "code"}.get(name)
        if argument_name is None or not isinstance(arguments, dict) or set(arguments) != {argument_name}:
            raise ValueError("Expected bash(command: string) or python(code: string), without extra arguments.")
        value = arguments[argument_name]
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("Tool input must be a nonempty string without NUL bytes.")
        async with self._execution_lock:
            if self.container_id is None or self.terminal_error or self.cleanup_status != "pending":
                raise RuntimeError("The task container is not available.")
            if self.execution_attempts >= self.max_tool_executions:
                raise RuntimeError("The episode exhausted its Docker tool-execution budget.")
            self.execution_attempts += 1
            execution_id = str(uuid4())
            self._execution_deadline = time.monotonic() + self.tool_timeout_seconds
            executable = ["/bin/bash", "--noprofile", "--norc", "-c"] if name == "bash" else ["python", "-c"]
            try:
                transport = await run_process_async(
                    arguments=[
                        "docker",
                        "exec",
                        "--workdir",
                        "/workspace",
                        self.container_id,
                        "python",
                        "-I",
                        "-u",
                        "-c",
                        self._runner_source,
                        execution_id,
                        json.dumps([*executable, value]),
                        str(self._command_timeout_seconds),
                        str(self.OUTPUT_LIMIT),
                    ],
                    timeout=self.tool_timeout_seconds,
                    output_limit=self.TRANSPORT_OUTPUT_LIMIT,
                )
            except OSError as error:
                self.terminal_error = "docker_execution_error"
                raise DockerExecutionError(
                    reason=self.terminal_error,
                    execution_id=execution_id,
                    execution_started=False,
                    result=None,
                    diagnostics={"transport_error": {"type": type(error).__name__, "message": str(error)}},
                ) from error
            return self._validate_execution(execution_id=execution_id, transport=transport)

    async def terminate_async(self) -> None:
        """
        Terminate only the owned container after terminal evidence has been journaled.

        Raises:
            RuntimeError: If no owned container or terminal failure exists.
            DockerCommandError: If Docker cannot confirm termination.
        """
        if self.container_id is None or self.terminal_error is None or self._execution_deadline is None:
            raise RuntimeError("Termination requires an owned container and a terminal execution failure.")
        remaining = self._execution_deadline - time.monotonic()
        if remaining <= 0:
            raise DockerCommandError("The tool deadline expired; outer owned-resource cleanup must terminate it.")
        await self._control_async(arguments=["kill", self.container_id], timeout=remaining)

    async def cleanup_async(self) -> None:
        """
        Remove only this run's labeled container, then verify it is absent.

        Raises:
            DockerCommandError: If ownership or removal cannot be verified.
        """
        if not self._create_started or self.cleanup_status == "removed":
            return
        self.cleanup_status = "failed"
        found = await self._control_async(
            arguments=["container", "ls", "--all", "--quiet", "--no-trunc", "--filter", f"name=^/{self.name}$"]
        )
        if not found.stdout.strip():
            self.cleanup_status = "removed"
            return
        inspected = await self._control_async(arguments=["container", "inspect", self.name])
        container = json.loads(inspected.stdout)[0]
        if container["Config"]["Labels"].get(self.OWNER_LABEL) != self.run_id:
            raise DockerCommandError("Refusing cleanup: container ownership label does not match.")
        owned_id = container["Id"]
        if self.container_id is not None and owned_id != self.container_id:
            raise DockerCommandError("Refusing cleanup: container ID changed.")
        self.container_id = owned_id
        await self._control_async(arguments=["rm", "--force", owned_id])
        remaining = await self._control_async(
            arguments=["container", "ls", "--all", "--quiet", "--no-trunc", "--filter", f"id={owned_id}"]
        )
        if remaining.stdout.strip():
            raise DockerCommandError(f"Owned container {owned_id} still exists after removal.")
        self.cleanup_status = "removed"

    def _validate_execution(self, *, execution_id: str, transport: CommandResult) -> dict[str, Any]:
        capture = _ExecutionCapture(execution_id=execution_id, output_limit=self.OUTPUT_LIMIT)
        try:
            for line in transport.stdout.splitlines():
                capture.accept(line)
        except ValueError as error:
            capture.protocol_error = str(error)
        if capture.started:
            self.executions += 1
        result = capture.tool_result(transport_timed_out=transport.timed_out)
        self.timed_out = capture.timed_out or transport.timed_out
        invalid = (
            transport.returncode != 0
            or transport.truncated
            or bool(transport.stderr)
            or capture.protocol_error is not None
            or not capture.finished
        )
        if self.timed_out or invalid:
            self.terminal_error = "tool_timeout" if self.timed_out else "docker_execution_error"
            if result is not None:
                result["error"] = self.terminal_error
            raise DockerExecutionError(
                reason=self.terminal_error,
                execution_id=execution_id,
                execution_started=capture.started,
                result=result,
                diagnostics={
                    "transport": asdict(transport),
                    "protocol_error": capture.protocol_error,
                    "launch_error": capture.launch_error,
                    "completion_confirmed": capture.finished,
                },
            )
        assert result is not None
        return result

    def _create_arguments(self) -> list[str]:
        return [
            "create",
            "--name",
            self.name,
            "--label",
            f"{self.OWNER_LABEL}={self.run_id}",
            "--platform",
            self.platform,
            "--network",
            "none",
            "--cpus",
            str(self.cpus),
            "--memory",
            str(self.memory_bytes),
            "--memory-swap",
            str(self.memory_bytes),
            "--pids-limit",
            str(self.pids_limit),
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--init",
            "--workdir",
            "/workspace",
            self.image,
            "/bin/sleep",
            "infinity",
        ]

    def _verify_isolation(self) -> None:
        container = self.container_identity
        host = container["HostConfig"]
        expected = {
            "NetworkMode": "none",
            "NanoCpus": int(self.cpus * 1_000_000_000),
            "Memory": self.memory_bytes,
            "MemorySwap": self.memory_bytes,
            "PidsLimit": self.pids_limit,
            "Privileged": False,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges"],
            "Init": True,
        }
        if any(host.get(key) != value for key, value in expected.items()):
            raise DockerCommandError("Docker did not apply the requested isolation/resource profile.")
        if container["Mounts"] or host.get("Binds") or host.get("Devices") or host.get("PortBindings"):
            raise DockerCommandError("Unexpected mounts, devices, or published ports on the task container.")
        if container["Config"]["WorkingDir"] != "/workspace":
            raise DockerCommandError("Docker did not apply the requested working directory.")
        if container["Config"]["Labels"].get(self.OWNER_LABEL) != self.run_id:
            raise DockerCommandError("Docker did not apply the ownership label.")
        if self.image_identity and container["Image"] != self.image_identity["Id"]:
            raise DockerCommandError("The container does not use the acquired pinned image.")

    async def _control_async(self, *, arguments: list[str], timeout: float = 30) -> CommandResult:
        result = await run_process_async(arguments=["docker", *arguments], timeout=timeout, output_limit=65536)
        if result.returncode != 0 or result.timed_out or result.truncated:
            raise DockerCommandError(
                f"Docker {arguments[0]} failed: returncode={result.returncode}, "
                f"timeout={result.timed_out}, truncated={result.truncated}; {result.stderr}"
            )
        return result
