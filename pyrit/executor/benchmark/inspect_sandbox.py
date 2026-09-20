# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
    from pyrit.executor.benchmark._inspect_response_trace import InspectResponseTrace

logger = logging.getLogger(__name__)


class InspectToolResult(BaseModel):
    """The bounded tool result shared with the native Docker example."""

    model_config = ConfigDict(extra="forbid", strict=True)

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool
    truncated: bool
    error: str | None
    execution_id: str


class _CaptureFrame(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    protocol: str
    event: str
    execution_id: str
    result: InspectToolResult | None = None
    termination_error: str | None = None


class InspectSandboxTools:
    """Execute sequential model-requested tools through the active Inspect sandbox."""

    _SCHEMAS: list[dict[str, Any]] = [
        {
            "type": "function",
            "name": "bash",
            "description": "Run a bash command in the isolated task container.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "python",
            "description": "Run Python code in the isolated task container.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]

    def __init__(
        self,
        *,
        artifacts: InspectRunArtifacts,
        trace: InspectResponseTrace | None,
        max_executions: int,
        timeout_seconds: int,
        output_limit_bytes: int,
    ) -> None:
        """
        Configure local profile limits; a missing trace is reserved for no-model smoke tests.

        Raises:
            ValueError: If any configured limit is invalid.
        """
        if max_executions < 1 or timeout_seconds < 2 or output_limit_bytes < 1:
            raise ValueError("Tool limits must be positive, with at least two seconds for the timeout.")
        self.artifacts = artifacts
        self.trace = trace
        self.max_executions = max_executions
        self.timeout_seconds = timeout_seconds
        self.output_limit_bytes = output_limit_bytes
        self.executions: list[dict[str, Any]] = []
        self._worker = Path(__file__).with_name("_inspect_tool_worker.py").read_text(encoding="utf-8")
        self._active = False

    @classmethod
    def schemas(cls) -> list[dict[str, Any]]:
        """Return independent copies of the shared Responses API function schemas."""
        return copy.deepcopy(cls._SCHEMAS)

    async def bash_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Run a bash command in the current Inspect task container.

        Returns:
            dict[str, Any]: The bounded, correlated execution envelope.
        """
        return await self._execute_async(name="bash", key="command", arguments=arguments)

    async def python_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Run Python code in the current Inspect task container.

        Returns:
            dict[str, Any]: The bounded, correlated execution envelope.
        """
        return await self._execute_async(name="python", key="code", arguments=arguments)

    async def _execute_async(self, *, name: str, key: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from inspect_ai.util import sandbox

        if set(arguments) != {key} or not isinstance(arguments[key], str) or not arguments[key]:
            raise ValueError(f"{name} requires exactly one nonempty string argument named {key}.")
        if self._active or len(self.executions) >= self.max_executions:
            raise RuntimeError("Sequential tool execution budget exhausted or another execution is active.")
        call_id = self.trace.claim_call(name=name, arguments=arguments) if self.trace is not None else None
        execution_id = str(uuid4())
        execution: dict[str, Any] = {
            "execution_id": execution_id,
            "provider_call_id": call_id,
            "origin": "model" if self.trace is not None else "smoke",
            "name": name,
            "arguments": arguments,
            "status": "dispatching",
            "confirmed_started": False,
        }
        self._active = True
        self.executions.append(execution)
        await self.artifacts.append_async(event="tool_dispatch", data=execution)
        command = ["bash", "-c", arguments[key]] if name == "bash" else ["python", "-c", arguments[key]]
        options = json.dumps(
            {
                "command": command,
                "timeout": self.timeout_seconds - 1,
                "limit": self.output_limit_bytes,
                "execution_id": execution_id,
            }
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await sandbox("default").exec(
                    cmd=["python", "-I", "-u", "-c", self._worker],
                    input=options,
                    cwd="/workspace",
                    timeout=self.timeout_seconds,
                    timeout_retry=False,
                )
            execution["transport"] = {
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            envelope, termination_error = self._parse_capture(stdout=result.stdout, execution=execution)
            if not result.success:
                raise RuntimeError(f"Sandbox capture worker failed with exit status {result.returncode}.")
            if envelope.returncode is None and not envelope.timed_out:
                raise RuntimeError("Sandbox capture has no completed command status.")
            expected_error = (
                "tool_timeout"
                if envelope.timed_out
                else "output_truncated"
                if envelope.truncated
                else "nonzero_exit"
                if envelope.returncode
                else None
            )
            if envelope.error != expected_error or (envelope.timed_out and envelope.returncode is not None):
                raise ValueError("Sandbox returned inconsistent error, timeout, or exit-status evidence.")
            if any(
                len(value.encode("utf-8")) > self.output_limit_bytes for value in (envelope.stdout, envelope.stderr)
            ):
                raise ValueError("Sandbox returned output above the configured per-stream byte limit.")
            execution.update(status="completed", result=envelope.model_dump(), termination_error=termination_error)
            await self.artifacts.append_async(event="tool_completed", data=execution)
            if envelope.timed_out or termination_error:
                raise RuntimeError("Tool timed out; do not continue or grade this mutated sandbox.")
            return envelope.model_dump()
        except BaseException as error:
            if self.trace is not None:
                self.trace.abort(type(error).__name__)
            execution.update(status="error", error_type=type(error).__name__, error=str(error))
            partial = getattr(error, "truncated_output", None)
            if isinstance(partial, str):
                execution["transport_partial_output"] = partial
            await self.artifacts.append_async(event="tool_error", data=execution)
            if isinstance(error, TimeoutError):
                raise RuntimeError(
                    "Sandbox transport timed out; capture is incomplete and this episode must stop."
                ) from error
            raise
        finally:
            self._active = False

    @staticmethod
    def _parse_capture(*, stdout: str, execution: dict[str, Any]) -> tuple[InspectToolResult, str | None]:
        lines = stdout.splitlines()
        if not lines:
            raise RuntimeError("Sandbox did not return a capture start frame; command execution is unconfirmed.")
        started = _CaptureFrame.model_validate_json(lines[0])
        execution_id = execution["execution_id"]
        if (
            started.protocol != "pyrit-inspect-tool-v1"
            or started.event != "started"
            or started.execution_id != execution_id
            or started.result is not None
            or started.termination_error is not None
        ):
            raise ValueError("Sandbox capture start frame does not match this dispatch.")
        execution["confirmed_started"] = True
        if len(lines) != 2:
            raise RuntimeError("Sandbox command started but no unique completion frame was retained.")
        completed = _CaptureFrame.model_validate_json(lines[1])
        if (
            completed.protocol != started.protocol
            or completed.event != "completed"
            or completed.execution_id != execution_id
            or completed.result is None
            or completed.result.execution_id != execution_id
        ):
            raise ValueError("Sandbox capture completion frame does not match this execution.")
        execution["result"] = completed.result.model_dump()
        execution["termination_error"] = completed.termination_error
        return completed.result, completed.termination_error


class InspectDockerProfile:
    """Describe and verify one owned Docker Compose service; Inspect alone provisions it."""

    _LABEL = "org.pyrit.inspect.run"
    _METADATA_TIMEOUT_SECONDS = 30
    _METADATA_REAP_TIMEOUT_SECONDS = 5

    def __init__(self, *, artifacts: InspectRunArtifacts, image: str) -> None:
        """
        Bind an immutable Linux image and a unique run label.

        Raises:
            ValueError: If the image is not pinned by digest.
        """
        if "@sha256:" not in image:
            raise ValueError("The Docker profile requires an image pinned by digest.")
        self.artifacts = artifacts
        self.image = image
        self.compose_file = artifacts.directory / "compose.yaml"
        self.container_evidence: dict[str, Any] | None = None

    async def write_compose_async(self) -> None:
        """Write the minimal-image variant without mounts, ports, or a fixed container name."""
        import aiofiles

        config = {
            "services": {
                "default": {
                    "image": self.image,
                    "platform": "linux/amd64",
                    "command": ["sleep", "infinity"],
                    "working_dir": "/workspace",
                    "network_mode": "none",
                    "cpus": 1,
                    "mem_limit": "512m",
                    "memswap_limit": "512m",
                    "pids_limit": 128,
                    "cap_drop": ["ALL"],
                    "security_opt": ["no-new-privileges:true"],
                    "labels": {self._LABEL: self.artifacts.run_id},
                }
            }
        }
        async with aiofiles.open(self.compose_file, "w", encoding="utf-8") as stream:
            await stream.write(json.dumps(config, indent=2))

    async def verify_running_async(self) -> None:
        """
        Reject an unexpected or under-isolated container before any model call.

        Raises:
            RuntimeError: If Docker did not apply the complete resource/isolation profile.
        """
        container_ids = (await self._docker_async("ps", "-aq", "--filter", self._label_filter())).split()
        if len(container_ids) != 1:
            raise RuntimeError("Expected exactly one running container owned by this Inspect run.")
        records = json.loads(await self._docker_async("inspect", container_ids[0]))
        record = records[0]
        host = record["HostConfig"]
        config = record["Config"]
        expected = {
            "NetworkMode": "none",
            "NanoCpus": 1_000_000_000,
            "Memory": 512 * 1024 * 1024,
            "MemorySwap": 512 * 1024 * 1024,
            "PidsLimit": 128,
            "Privileged": False,
        }
        if any(host.get(name) != value for name, value in expected.items()):
            raise RuntimeError("Docker did not apply every requested isolation/resource setting.")
        if host.get("CapDrop") != ["ALL"] or "no-new-privileges:true" not in host.get("SecurityOpt", []):
            raise RuntimeError("Docker capabilities or no-new-privileges settings do not match the profile.")
        if (
            record["Mounts"]
            or host.get("PortBindings")
            or host.get("Devices")
            or host.get("DeviceRequests")
            or host.get("CapAdd")
            or host.get("PidMode")
        ):
            raise RuntimeError("The Inspect profile forbids mounts, ports, devices, and host socket access.")
        if config["WorkingDir"] != "/workspace" or config["Image"] != self.image or not record["State"]["Running"]:
            raise RuntimeError("Unexpected Docker image, working directory, or container state.")
        self.container_evidence = {
            "id": record["Id"],
            "name": record["Name"],
            "image_id": record["Image"],
            "compose_project": config["Labels"]["com.docker.compose.project"],
            "host_config": {name: host[name] for name in (*expected, "CapDrop", "SecurityOpt")},
            "mounts": record["Mounts"],
            "working_directory": config["WorkingDir"],
        }
        await self.artifacts.append_async(event="container_verified", data=self.container_evidence)

    async def verify_cleanup_async(self) -> None:
        """
        Check for owned leftovers without deleting resources or running Docker prune.

        Raises:
            RuntimeError: If Inspect left any identified run resources behind.
            OSError: If the Docker metadata command cannot be started.
            TimeoutError: If Docker cannot answer the cleanup query in time.
        """
        try:
            containers = (await self._docker_async("ps", "-aq", "--filter", self._label_filter())).split()
            leftovers: dict[str, list[str]] = {"containers": containers}
            if self.container_evidence is not None:
                project = self.container_evidence["compose_project"]
                for resource in ("network", "volume"):
                    leftovers[resource] = (
                        await self._docker_async(
                            resource, "ls", "-q", "--filter", f"label=com.docker.compose.project={project}"
                        )
                    ).split()
        except (OSError, RuntimeError, TimeoutError) as error:
            self.artifacts.manifest.update(cleanup_status="unknown", cleanup_error=str(error))
            await self.artifacts.save_async()
            raise
        self.artifacts.manifest["cleanup"] = leftovers
        self.artifacts.manifest["cleanup_status"] = "failed" if any(leftovers.values()) else "verified"
        await self.artifacts.save_async()
        if any(leftovers.values()):
            raise RuntimeError("Inspect left owned Docker resources behind; see manifest cleanup identities.")

    def _label_filter(self) -> str:
        return f"label={self._LABEL}={self.artifacts.run_id}"

    @staticmethod
    async def _docker_async(*arguments: str) -> str:
        process = await asyncio.create_subprocess_exec(
            "docker", *arguments, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            async with asyncio.timeout(InspectDockerProfile._METADATA_TIMEOUT_SECONDS):
                stdout, stderr = await process.communicate()
        except (TimeoutError, asyncio.CancelledError) as error:
            await InspectDockerProfile._stop_metadata_process_async(process=process, original_error=error)
            raise
        if process.returncode:
            raise RuntimeError(f"Docker metadata query failed: {stderr.decode('utf-8', errors='replace')}")
        return stdout.decode("utf-8")

    @staticmethod
    async def _stop_metadata_process_async(
        *, process: asyncio.subprocess.Process, original_error: TimeoutError | asyncio.CancelledError
    ) -> None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
        except OSError as error:
            InspectDockerProfile._note_metadata_cleanup_error(
                original_error=original_error, pid=process.pid, detail=f"kill failed: {error}"
            )

        reaper = asyncio.create_task(InspectDockerProfile._reap_metadata_process_async(process))
        cancellation = original_error if isinstance(original_error, asyncio.CancelledError) else None
        # Repeated caller cancellation must not abandon the bounded, owned reaper.
        while not reaper.done():
            try:
                await asyncio.shield(reaper)
            except asyncio.CancelledError as error:
                cancellation = cancellation or error
        try:
            cleanup_error = reaper.result()
        except asyncio.CancelledError:
            cleanup_error = "reaper was cancelled before the process exit was observed"
        if cleanup_error:
            InspectDockerProfile._note_metadata_cleanup_error(
                original_error=original_error, pid=process.pid, detail=cleanup_error
            )
        if cancellation is not None and cancellation is not original_error:
            raise cancellation from original_error

    @staticmethod
    async def _reap_metadata_process_async(process: asyncio.subprocess.Process) -> str | None:
        try:
            async with asyncio.timeout(InspectDockerProfile._METADATA_REAP_TIMEOUT_SECONDS):
                await process.wait()
        except TimeoutError:
            return f"reaping exceeded {InspectDockerProfile._METADATA_REAP_TIMEOUT_SECONDS} seconds"
        except (OSError, RuntimeError) as error:
            return f"reaping failed: {error}"
        return None

    @staticmethod
    def _note_metadata_cleanup_error(*, original_error: BaseException, pid: int, detail: str) -> None:
        diagnostic = f"Docker metadata subprocess {pid} cleanup: {detail}"
        original_error.add_note(diagnostic)
        logger.warning("%s", diagnostic)
