# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inspect sandbox process control; neither a task scorer nor an attack strategy."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiofiles
from inspect_ai.util import (
    ExecCompleted,
    ExecRemoteProcess,
    ExecRemoteStreamingOptions,
    ExecResult,
    ExecStderr,
    ExecStdout,
    sandbox,
)
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from inspect_ai.util import SandboxEnvironment


@dataclass(frozen=True, kw_only=True)
class InspectGhcpLimits:
    """Hard per-sample limits for process, model traffic, and capture."""

    min_turns: int = 2
    min_tool_executions: int = 1
    max_turns: int = 2
    max_model_requests: int = 12
    max_model_bytes: int = 1_048_576
    max_raw_bytes: int = 16_777_216
    max_prompt_tokens: int = 8192
    max_output_tokens: int = 1024
    turn_timeout_seconds: int = 60
    run_timeout_seconds: int = 600

    def __post_init__(self) -> None:
        """
        Reject unbounded or nonsensical resource limits.

        Raises:
            ValueError: If any resource cap is invalid.
        """
        if any(
            value < 1
            for value in (
                self.min_turns,
                self.max_turns,
                self.max_model_requests,
                self.max_model_bytes,
                self.max_raw_bytes,
                self.max_prompt_tokens,
                self.max_output_tokens,
                self.turn_timeout_seconds,
                self.run_timeout_seconds,
            )
        ):
            raise ValueError("Inspect GHCP limits must all be positive.")
        if self.min_turns > self.max_turns or self.turn_timeout_seconds > self.run_timeout_seconds:
            raise ValueError("A GHCP turn cannot outlast its run.")
        if self.min_tool_executions < 0:
            raise ValueError("Task-required GHCP tool executions cannot be negative.")


class _SandboxRpc:
    """Read bounded newline-delimited frames from one in-sandbox process."""

    MAX_FRAME_BYTES = 8_388_608
    MAX_STDERR_BYTES = 65_536

    def __init__(self, *, process: ExecRemoteProcess, monitor: bool = True) -> None:
        if type(process.pid) is not int or process.pid < 1:
            raise ValueError("Inspect sandbox process must expose its actual started job ID.")
        self._process = process
        self._job_pid = process.pid
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader = asyncio.create_task(self._read_async()) if monitor else None
        self._exit_code: int | None = None
        self._stderr = bytearray()
        self._stderr_received = 0

    @property
    def stderr_capture(self) -> tuple[bytes, int]:
        """The bounded guest stderr bytes and their explicit omitted byte count."""
        return bytes(self._stderr), self._stderr_received - len(self._stderr)

    async def send_async(self, *, frame: dict[str, Any]) -> None:
        await self._process.write_stdin(json.dumps(frame, separators=(",", ":")) + "\n")

    async def receive_async(self, *, kind: str, timeout: int) -> dict[str, Any]:
        try:
            if self._reader is None:
                item = await asyncio.wait_for(self._read_initial_async(), timeout=timeout)
            else:
                item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except TimeoutError as error:
            raise RuntimeError(
                f"Contained process did not report {kind} within its {timeout}s startup/turn limit."
            ) from error
        if item.get("kind") == "terminated":
            raise RuntimeError(f"Contained process exited before {kind} (code {item['exit_code']}).")
        if item.get("kind") == "process_error":
            raise RuntimeError(f"Contained process reader failed before {kind}: {item['error_type']}.")
        if item.get("kind") != kind:
            raise ValueError(f"Expected a contained {kind} frame; got {item.get('kind')!r}.")
        return item

    async def finish_async(self, *, timeout: int) -> None:
        if self._reader is None:
            raise RuntimeError("A gateway process cannot be finished as an interactive agent.")
        if self._process.pid != self._job_pid:
            raise RuntimeError("The GHCP worker's original Inspect job identity changed before stop.")
        try:
            await asyncio.wait_for(self._reader, timeout=timeout)
        except TimeoutError as error:
            raise RuntimeError("Guest worker did not exit within the approved stop budget.") from error
        if self._exit_code != 0:
            raise RuntimeError(f"Contained GHCP worker exited with code {self._exit_code}.")

    async def kill_async(self) -> None:
        await self._process.kill()
        if self._reader is None:
            return
        try:
            await asyncio.wait_for(self._reader, timeout=10)
        except TimeoutError as error:
            raise RuntimeError("Contained sandbox process termination was not observed within 10s.") from error

    async def _read_async(self) -> None:
        buffered = ""
        try:
            async for item in self._process:
                if isinstance(item, ExecStderr):
                    self._append_stderr(data=item.data)
                elif isinstance(item, ExecStdout):
                    buffered += item.data
                    if len(buffered.encode("utf-8")) > self.MAX_FRAME_BYTES:
                        raise ValueError("Sandbox process exceeded the bounded frame size.")
                    while "\n" in buffered:
                        line, buffered = buffered.split("\n", 1)
                        frame = json.loads(line)
                        if not isinstance(frame, dict):
                            raise ValueError("Sandbox process returned a non-object frame.")
                        await self._queue.put(frame)
                else:
                    if self._process.pid != self._job_pid:
                        raise RuntimeError("A foreign Inspect job cannot complete the GHCP worker.")
                    self._exit_code = item.exit_code
                    await self._queue.put({"kind": "terminated", "exit_code": item.exit_code})
            if buffered.strip():
                raise ValueError("Sandbox process terminated with a partial JSON frame.")
        except (ValueError, UnicodeError, RuntimeError, OSError) as error:
            await self._queue.put({"kind": "process_error", "error_type": type(error).__name__})
            raise

    async def _read_initial_async(self) -> dict[str, Any]:
        buffered = ""
        async for item in self._process:
            if isinstance(item, ExecStderr):
                self._append_stderr(data=item.data)
            elif isinstance(item, ExecStdout):
                buffered += item.data
                if len(buffered.encode("utf-8")) > self.MAX_FRAME_BYTES:
                    raise ValueError("Gateway startup exceeded the bounded source frame size.")
                if "\n" in buffered:
                    line, remaining = buffered.split("\n", 1)
                    if remaining.strip():
                        raise ValueError("Gateway startup emitted an unexpected second source frame.")
                    frame = json.loads(line)
                    if not isinstance(frame, dict):
                        raise ValueError("Gateway startup returned a non-object source frame.")
                    return frame
            else:
                raise RuntimeError(f"Model gateway exited before readiness (code {item.exit_code}).")
        raise RuntimeError("Model gateway ended without an observed ready frame.")

    def _append_stderr(self, *, data: str) -> None:
        source = data.encode("utf-8")
        self._stderr_received += len(source)
        self._stderr.extend(source)
        if len(self._stderr) > self.MAX_STDERR_BYTES:
            del self._stderr[: len(self._stderr) - self.MAX_STDERR_BYTES]


async def _stage_script_async(*, environment: SandboxEnvironment, filename: str) -> str:
    """
    Write only public runner code, never the run-scoped token, into the sandbox.

    Returns:
        str: The installed sandbox path.
    """
    source = Path(__file__).with_name(filename)
    async with aiofiles.open(source, "rb") as stream:
        contents = await stream.read()
    destination = f"/tmp/pyrit-inspect/{filename}"
    await environment.write_file(destination, contents)
    return destination


class InspectGhcpSandboxRuntime:
    """Keep a model gateway and one GHCP SDK process alive for an Inspect sample."""

    def __init__(
        self,
        *,
        run_id: str,
        token: str,
        model_id: str,
        wire_model: str,
        cli_path: str,
        cli_sha256: str,
        gateway_image_service: str,
        limits: InspectGhcpLimits,
        allowed_tools: tuple[str, ...],
        control_receipt_sink: Callable[[dict[str, Any]], Awaitable[None]],
        verify_image_async: Callable[[str, str], Awaitable[str | None]],
        approved_image_ids: dict[str, str],
        prompt_cache_key_policy: str = "reject",
    ) -> None:
        """
        Bind a per-sample token and pinned agent image to supported Inspect sandboxes.

        Raises:
            ValueError: If the model service or scoped identity is invalid.
        """
        if (
            gateway_image_service != "model-bridge"
            or not re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
            or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", run_id)
            or any(
                not re.fullmatch(r"sha256:[0-9a-f]{64}", approved_image_ids.get(service, ""))
                for service in ("agent", "model-bridge")
            )
        ):
            raise ValueError("The GHCP runtime requires a scoped UUID and pinned, isolated agent/bridge images.")
        self.run_id = run_id
        self._token = token
        self._token_file = f"/tmp/pyrit-inspect-token-{run_id}"
        self._control_receipt_sink = control_receipt_sink
        self._verify_image_async = verify_image_async
        self._approved_image_ids = dict(approved_image_ids)
        self._model_id = model_id
        self._wire_model = wire_model
        self._cli_path = cli_path
        self._cli_sha256 = cli_sha256
        self._limits = limits
        self._allowed_tools = allowed_tools
        self._prompt_cache_key_policy = prompt_cache_key_policy
        self._gateway: _SandboxRpc | None = None
        self._agent: _SandboxRpc | None = None
        self._gateway_pid: int | None = None
        self._agent_identity: dict[str, Any] | None = None
        self._session_id: str | None = None
        self._container_ref: str | None = None
        self._model_container_ref: str | None = None
        self._attested_container_ids: dict[str, str] = {}
        self._token_helpers: dict[str, str] = {}
        self._consumed_tokens: set[str] = set()
        self._pending_token_write: asyncio.Task[ExecResult[str]] | None = None
        self._pending_token_service: str | None = None
        self._turn_index = 0
        self._stopped = False
        self._closed = False
        self.phase = "not_started"
        self.phase_durations: dict[str, float] = {}
        self._phase_started = time.monotonic()

    @property
    def container_ref(self) -> str | None:
        """The Compose agent container name reported by Inspect, not its full Docker ID."""
        return self._container_ref

    @property
    def session_id(self) -> str | None:
        """The SDK session ID observed at startup."""
        return self._session_id

    @property
    def model_container_ref(self) -> str | None:
        """The separate Compose model-bridge name, not its full Docker ID."""
        return self._model_container_ref

    @property
    def attested_container_ids(self) -> dict[str, str]:
        """Full Docker IDs independently verified before private token delivery."""
        return dict(self._attested_container_ids)

    @property
    def closed(self) -> bool:
        """Whether owned guest processes were already stopped inside the live sample."""
        return self._closed

    @property
    def private_tokens_consumed(self) -> bool:
        """Whether both sandbox token files were observed absent before the first turn."""
        return self._consumed_tokens == {"agent", "model-bridge"}

    @property
    def agent_identity(self) -> dict[str, Any] | None:
        """The observed guest worker/CLI PID and namespace identity."""
        return self._agent_identity

    @property
    def agent_stderr_capture(self) -> tuple[bytes, int]:
        """Exact bounded worker stderr, never shown in safe task output."""
        return self._agent.stderr_capture if self._agent is not None else (b"", 0)

    async def start_async(self, *, inspect_proxy_port: int) -> None:
        """
        Launch the authenticated bridge before starting the guest SDK worker.

        Raises:
            RuntimeError: If this runtime was already started.
            ValueError: If distinct containers or the GHCP process cannot be proven.
        """
        if self._agent is not None or self._gateway is not None:
            raise RuntimeError("An Inspect GHCP runtime may be started only once.")
        self._mark_phase("connecting_inspect_sandboxes")
        bridge_env = sandbox("model-bridge")
        agent_env = sandbox("agent")
        model_connection = await bridge_env.connection()
        agent_connection = await agent_env.connection()
        if (
            not model_connection.container
            or not agent_connection.container
            or model_connection.container == agent_connection.container
        ):
            raise ValueError("Inspect did not provision distinct model and agent containers.")
        self._container_ref = agent_connection.container
        self._model_container_ref = model_connection.container
        raw_bridge = await self._scoped_raw_sandbox_async(
            environment=bridge_env,
            service="model-bridge",
            expected_container_ref=model_connection.container,
        )
        raw_agent = await self._scoped_raw_sandbox_async(
            environment=agent_env,
            service="agent",
            expected_container_ref=agent_connection.container,
        )
        self._mark_phase("attesting_private_sandboxes")
        await self._attest_container_ids_async()
        self._mark_phase("staging_model_gateway")
        gateway_script = await _stage_script_async(environment=bridge_env, filename="inspect_ghcp_gateway.py")
        bridge_token_helper = await _stage_script_async(environment=bridge_env, filename="inspect_ghcp_token_file.py")
        self._token_helpers["model-bridge"] = bridge_token_helper
        self._mark_phase("delivering_bridge_token")
        await self._deliver_private_token_async(
            raw=raw_bridge,
            service="model-bridge",
            helper=bridge_token_helper,
            expected_container_ref=model_connection.container,
        )
        gateway_env = self._gateway_env(proxy_port=inspect_proxy_port)
        self._mark_phase("starting_model_gateway_process")
        process = await bridge_env.exec_remote(
            ["python3", gateway_script],
            ExecRemoteStreamingOptions(env=gateway_env),
        )
        self._gateway = _SandboxRpc(process=process, monitor=False)
        self._mark_phase("awaiting_model_gateway_ready")
        ready = await self._gateway.receive_async(kind="ready", timeout=10)
        pid = ready.get("pid")
        if type(pid) is not int or pid < 1:
            raise ValueError("Inspect model gateway did not report an observed process ID.")
        self._gateway_pid = pid
        await self._assert_token_absent_async(service="model-bridge", helper=bridge_token_helper)
        self._mark_phase("staging_guest_sdk_worker")
        worker_script = await _stage_script_async(environment=agent_env, filename="inspect_ghcp_guest.py")
        agent_token_helper = await _stage_script_async(environment=agent_env, filename="inspect_ghcp_token_file.py")
        self._token_helpers["agent"] = agent_token_helper
        self._mark_phase("starting_guest_sdk_worker")
        worker = await agent_env.exec_remote(
            ["python3", worker_script],
            ExecRemoteStreamingOptions(stdin_open=True),
        )
        self._agent = _SandboxRpc(process=worker)
        self._mark_phase("delivering_agent_token")
        await self._deliver_private_token_async(
            raw=raw_agent,
            service="agent",
            helper=agent_token_helper,
            expected_container_ref=agent_connection.container,
        )
        self._mark_phase("sending_guest_sdk_start")
        await self._agent.send_async(frame=self._start_frame())
        self._mark_phase("awaiting_guest_sdk_ready")
        started = await self._agent.receive_async(kind="ready", timeout=30)
        self._session_id = self._validate_agent_ready(started=started)
        await self._assert_token_absent_async(service="agent", helper=agent_token_helper)
        self._mark_phase("guest_sdk_ready")

    async def send_turn_async(self, *, instruction: str, turn_index: int) -> dict[str, Any]:
        """
        Send a PyRIT-selected instruction to the same guest process and SDK session.

        Returns:
            dict[str, Any]: Source-observed SDK event and model exchange frame.

        Raises:
            RuntimeError: If the process has stopped or has not started.
            ValueError: If turn ordering, run or process identity changes.
        """
        if self._agent is None or self._session_id is None or self._stopped:
            raise RuntimeError("The contained GHCP session must be running before any send.")
        if turn_index != self._turn_index + 1 or turn_index > self._limits.max_turns:
            raise ValueError("GHCP outer turns must be consecutive and below the approved cap.")
        self._mark_phase(f"awaiting_guest_turn_{turn_index}")
        await self._agent.send_async(frame={"op": "send", "turn_index": turn_index, "instruction": instruction})
        frame = await self._agent.receive_async(kind="turn", timeout=self._limits.turn_timeout_seconds + 10)
        if (
            frame.get("run_id") != self.run_id
            or frame.get("turn_index") != turn_index
            or frame.get("session_id") != self._session_id
            or frame.get("identity") != self._agent_identity
        ):
            raise ValueError("A GHCP turn came from a different session or agent process.")
        self._turn_index = turn_index
        self._mark_phase(f"guest_turn_{turn_index}_complete")
        return frame

    async def stop_agent_async(self) -> None:
        """
        Prove that the CLI and worker stopped before the original Inspect scorer.

        Raises:
            RuntimeError: If the worker is absent or CLI remains running.
        """
        if self._agent is None or self._stopped:
            raise RuntimeError("The GHCP agent cannot be stopped twice or before it starts.")
        await self._agent.send_async(frame={"op": "stop"})
        stopped = await self._agent.receive_async(kind="stopped", timeout=20)
        if stopped.get("cli_exited") is not True:
            raise RuntimeError("The guest did not attest that the GHCP CLI exited.")
        await self._agent.finish_async(timeout=20)
        self._stopped = True
        self._mark_phase("guest_sdk_stopped")

    async def gateway_alive_async(self) -> bool:
        """
        Verify the separate gateway PID and inert local /health response.

        Returns:
            bool: Whether the authenticated model gateway remains live in its sandbox.
        """
        if self._gateway_pid is None:
            return False
        script = (
            "import os,sys,urllib.request;"
            "os.kill(int(sys.argv[1]),0);"
            "opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));"
            "response=opener.open('http://127.0.0.1:18181/health',timeout=3);"
            "assert response.status==200 and response.read()==b'{\"ready\":true}'"
        )
        observed = await sandbox("model-bridge").exec(
            ["python3", "-c", script, str(self._gateway_pid)], timeout=5, timeout_retry=False
        )
        return observed.success

    async def read_gateway_audit_async(self) -> list[dict[str, Any]]:
        """
        Retrieve bounded authoritative model bytes without putting them in Inspect's log.

        Returns:
            list[dict[str, Any]]: Actual gateway request/response byte records.

        Raises:
            RuntimeError: If the process or audit read fails.
            ValueError: If the audit is incomplete, invalid or over quota.
        """
        if self._gateway is None or not self._stopped:
            raise RuntimeError("Read model bytes only after the agent has stopped.")
        bridge_env = sandbox("model-bridge")
        script = "/tmp/pyrit-inspect/inspect_ghcp_gateway.py"
        audit_proc = await bridge_env.exec_remote(
            ["python3", script, "--audit"],
            ExecRemoteStreamingOptions(env={"PYRIT_GATEWAY_AUDIT_PATH": self._audit_path()}),
        )
        records: list[dict[str, Any]] = []
        pending = ""
        total_bytes = 0
        try:
            async for output in audit_proc:
                if isinstance(output, ExecStderr):
                    raise RuntimeError("Inspect model gateway audit read failed.")
                if isinstance(output, ExecCompleted) and output.exit_code != 0:
                    raise RuntimeError("Inspect model gateway audit is incomplete.")
                if isinstance(output, ExecStdout):
                    total_bytes += len(output.data.encode("utf-8"))
                    if total_bytes > self._limits.max_raw_bytes:
                        raise ValueError("Inspect gateway raw capture exceeded the approved quota.")
                    pending += output.data
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        record = json.loads(line)
                        if not isinstance(record, dict):
                            raise ValueError("Inspect model gateway returned a non-object audit row.")
                        records.append(record)
        finally:
            await audit_proc.kill()
        if pending or len(records) > self._limits.max_model_requests:
            raise ValueError("Inspect gateway returned partial or excessive model audit.")
        return records

    async def close_async(self) -> None:
        """
        Stop residual processes and remove unconsumed token files.

        Raises:
            RuntimeError: If cleanup is not observed; require exact-container teardown.
        """
        if self._closed:
            return
        errors: list[Exception] = []
        try:
            pending = self._pending_token_write
            if pending is not None:
                try:
                    await asyncio.wait_for(asyncio.shield(pending), timeout=40)
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append(error)
                except asyncio.CancelledError:
                    errors.append(RuntimeError("Private token writer was cancelled before sandbox cleanup."))
                except TimeoutError:
                    errors.append(RuntimeError("Private token writer did not stop before sandbox cleanup."))
                finally:
                    if pending.done():
                        self._pending_token_write = None
                        self._pending_token_service = None
            for process in (self._agent if not self._stopped else None, self._gateway):
                if process is not None:
                    try:
                        await process.kill_async()
                    except (OSError, RuntimeError, ValueError) as error:
                        errors.append(error)
        finally:
            for service, helper in self._token_helpers.items():
                if self._pending_token_write is not None and service == self._pending_token_service:
                    continue
                try:
                    await self._clear_token_file_async(service=service, helper=helper)
                except (OSError, RuntimeError, ValueError) as error:
                    errors.append(error)
        if self._pending_token_write is not None:
            errors.append(RuntimeError("An unfinished private writer requires exact container teardown."))
        if errors:
            raise RuntimeError("Inspect guest stop or private token-file removal was not observed.") from None
        self._closed = True

    def _audit_path(self) -> str:
        return f"/tmp/pyrit-inspect/{self.run_id}/model.jsonl"

    def _gateway_env(self, *, proxy_port: int) -> dict[str, str]:
        return {
            "PYRIT_GATEWAY_TOKEN_FILE": self._token_file,
            "PYRIT_GATEWAY_RUN_ID": self.run_id,
            "PYRIT_GATEWAY_MODEL": self._wire_model,
            "PYRIT_GATEWAY_PORT": "18181",
            "PYRIT_INSPECT_PROXY_PORT": str(proxy_port),
            "PYRIT_GATEWAY_MAX_REQUESTS": str(self._limits.max_model_requests),
            "PYRIT_GATEWAY_MAX_BODY": str(self._limits.max_model_bytes),
            "PYRIT_GATEWAY_MAX_RESPONSE": str(self._limits.max_model_bytes),
            "PYRIT_GATEWAY_DEADLINE": str(self._limits.run_timeout_seconds),
            "PYRIT_GATEWAY_TIMEOUT": str(self._limits.turn_timeout_seconds),
            "PYRIT_GATEWAY_AUDIT_PATH": self._audit_path(),
            "PYRIT_PROMPT_CACHE_KEY_POLICY": self._prompt_cache_key_policy,
        }

    def _start_frame(self) -> dict[str, Any]:
        return {
            "op": "start",
            "run_id": self.run_id,
            "token_file": self._token_file,
            "gateway_url": "http://model-bridge:18181",
            "model_id": self._model_id,
            "wire_model": self._wire_model,
            "cli_path": self._cli_path,
            "cli_sha256": self._cli_sha256,
            "allowed_tools": list(self._allowed_tools),
            "max_turns": self._limits.max_turns,
            "timeout_seconds": self._limits.turn_timeout_seconds,
            "max_model_bytes": self._limits.max_model_bytes,
            "max_prompt_tokens": self._limits.max_prompt_tokens,
            "max_output_tokens": self._limits.max_output_tokens,
        }

    async def _scoped_raw_sandbox_async(
        self, *, environment: SandboxEnvironment, service: str, expected_container_ref: str
    ) -> SandboxEnvironment:
        raw = environment.as_type(DockerSandboxEnvironment)
        if type(raw) is not DockerSandboxEnvironment or getattr(raw, "_service", None) != service:
            raise TypeError("Inspect secret bootstrap requires this exact pinned Docker provider and service.")
        observed = await raw.connection()
        if observed.container != expected_container_ref:
            raise ValueError("Raw token bootstrap would address a different Inspect sample container.")
        return raw

    async def _attest_container_ids_async(self) -> None:
        for service, container_ref in (
            ("model-bridge", self._model_container_ref),
            ("agent", self._container_ref),
        ):
            if container_ref is None:
                raise ValueError("Inspect did not name both private per-sample services.")
            full_id = await self._verify_image_async(container_ref, self._approved_image_ids[service])
            if (
                not isinstance(full_id, str)
                or not re.fullmatch(r"[0-9a-f]{64}", full_id)
                or full_id in self._attested_container_ids.values()
            ):
                raise ValueError("Private token delivery requires two distinct verified full Docker IDs.")
            self._attested_container_ids[service] = full_id

    async def _deliver_private_token_async(
        self, *, raw: SandboxEnvironment, service: str, helper: str, expected_container_ref: str
    ) -> None:
        container_ref = self._container_ref if service == "agent" else self._model_container_ref
        full_id = self._attested_container_ids.get(service)
        if (
            service not in {"agent", "model-bridge"}
            or type(raw) is not DockerSandboxEnvironment
            or container_ref != expected_container_ref
            or getattr(raw, "_service", None) != service
            or not full_id
        ):
            raise ValueError("A token may be delivered only to an attested per-sample agent or model-bridge.")
        if self._pending_token_write is not None:
            raise RuntimeError("Only one bounded Inspect token handoff may run at a time.")
        if await self._verify_image_async(expected_container_ref, self._approved_image_ids[service]) != full_id:
            raise ValueError("Private token delivery lost its verified full Docker ID before the write.")
        secret_bytes = self._token.encode("ascii")
        write = asyncio.create_task(
            raw.exec(
                ["python3", helper, "write", self.run_id],
                input=self._token,
                timeout=20,
                timeout_retry=False,
            )
        )
        self._pending_token_write = write
        self._pending_token_service = service
        try:
            observed = await asyncio.wait_for(
                asyncio.shield(write),
                timeout=40,
            )
        except TimeoutError:
            raise RuntimeError(f"Private {service} bootstrap exceeded its bounded controller budget.") from None
        except asyncio.CancelledError:
            raise asyncio.CancelledError from None
        except (OSError, RuntimeError, ValueError) as error:
            raise RuntimeError(f"Private {service} bootstrap failed: {type(error).__name__}.") from None
        finally:
            if write.done():
                self._pending_token_write = None
                self._pending_token_service = None
        if (
            not observed.success
            or observed.returncode != 0
            or observed.stderr
            or len(observed.stdout.encode("utf-8")) > 16
        ):
            raise RuntimeError(f"Private {service} bootstrap did not complete with bounded, token-free output.")
        matched = re.fullmatch(r"([1-9][0-9]{0,9})\r?\n", observed.stdout)
        if matched is None:
            raise ValueError(f"Private {service} bootstrap did not expose only its process ID.")
        job_id = int(matched[1])
        if (await raw.connection()).container != expected_container_ref:
            raise ValueError("Private token bootstrap changed Inspect sample containers.")
        if await self._verify_image_async(expected_container_ref, self._approved_image_ids[service]) != full_id:
            raise ValueError("Private token bootstrap changed its verified full Docker ID.")
        await self._control_receipt_sink(
            {
                "run_id": self.run_id,
                "service": service,
                "source_id": f"{self.run_id}:{service}:{job_id}",
                "observed_job_id": job_id,
                "container_id": full_id,
                "frame_size_bytes": len(secret_bytes),
                "frame_sha256": hashlib.sha256(secret_bytes).hexdigest(),
                "completed_exit_code": observed.returncode,
                "inspect_raw_control_elided": True,
                "provenance": "Inspect as_type Docker provider; owner-only tmpfs token file removed on read",
            }
        )

    async def _assert_token_absent_async(self, *, service: str, helper: str) -> None:
        observed = await sandbox(service).exec(
            ["python3", helper, "absent", self.run_id], timeout=5, timeout_retry=False
        )
        if not observed.success or observed.returncode != 0 or observed.stdout or observed.stderr:
            raise RuntimeError(f"Inspect {service} token file was not observed absent before GHCP inference.")
        self._consumed_tokens.add(service)

    async def _clear_token_file_async(self, *, service: str, helper: str) -> None:
        observed = await sandbox(service).exec(
            ["python3", helper, "clear", self.run_id], timeout=5, timeout_retry=False
        )
        if not observed.success or observed.returncode != 0 or observed.stdout or observed.stderr:
            raise RuntimeError(f"Inspect {service} token-file cleanup could not be confirmed.")
        await self._assert_token_absent_async(service=service, helper=helper)

    def _validate_agent_ready(self, *, started: dict[str, Any]) -> str:
        if started.get("run_id") != self.run_id:
            raise ValueError("GHCP SDK startup returned a different run identity.")
        session_id = started.get("session_id")
        identity = started.get("identity")
        if (
            not isinstance(session_id, str)
            or not session_id
            or not isinstance(identity, dict)
            or identity.get("cli_sha256") != self._cli_sha256
            or type(identity.get("uid")) is not int
            or identity["uid"] == 0
            or type(identity.get("worker_pid")) is not int
            or type(identity.get("cli_pid")) is not int
            or not isinstance(identity.get("net_namespace"), str)
        ):
            raise ValueError("GHCP SDK startup has no qualified nonroot CLI process evidence.")
        self._agent_identity = identity
        return session_id

    def _mark_phase(self, phase: str) -> None:
        now = time.monotonic()
        if self.phase != "not_started":
            self.phase_durations[self.phase] = round(now - self._phase_started, 3)
        self.phase = phase
        self._phase_started = now

    def phase_snapshot(self) -> dict[str, float]:
        """
        Include the current unfinished phase in a bounded, content-free timing report.

        Returns:
            dict[str, float]: Per-phase monotonic durations in seconds.
        """
        result = dict(self.phase_durations)
        if self.phase != "not_started":
            result[self.phase] = round(time.monotonic() - self._phase_started, 3)
        return result
