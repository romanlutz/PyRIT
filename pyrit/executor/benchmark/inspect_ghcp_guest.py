# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Stdio-only GHCP worker copied into Inspect's isolated agent sandbox."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from copilot import CopilotClient, RuntimeConnection
from copilot.copilot_request_handler import CopilotRequestContext, CopilotRequestHandler, CopilotWebSocketHandler
from copilot.generated.session_events import session_event_to_dict
from copilot.session import PermissionHandler

if __package__:
    from pyrit.executor.benchmark.inspect_ghcp_token_file import read_scoped_token
else:
    from inspect_ghcp_token_file import read_scoped_token  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from copilot.session_events import SessionEvent


class _ScopedModelHandler(CopilotRequestHandler):
    """Forward actual CLI model bytes only to this run's authenticated gateway."""

    def __init__(self, *, url: str, token: str, run_id: str, max_bytes: int, timeout: int) -> None:
        self._url = url.rstrip("/") + "/v1/responses"
        self._token = token
        self._run_id = run_id
        self._max_bytes = max_bytes
        self._client = httpx.AsyncClient(timeout=timeout, trust_env=False, follow_redirects=False)
        self.exchanges: list[dict[str, Any]] = []

    async def send_request(  # pyrit-async-suffix-exempt
        self, request: httpx.Request, ctx: CopilotRequestContext
    ) -> httpx.Response:
        if request.method != "POST" or str(request.url) != self._url:
            raise ValueError("GHCP attempted an unapproved model endpoint.")
        body = await request.aread()
        if len(body) > self._max_bytes:
            raise ValueError("GHCP model request exceeded the run-scoped byte quota.")
        record: dict[str, Any] = {
            "request_id": ctx.request_id,
            "source_session_id": ctx.session_id,
            "request_base64": base64.b64encode(body).decode("ascii"),
            "response_base64": None,
            "status": None,
            "error": None,
        }
        try:
            response = await self._client.post(
                self._url,
                content=body,
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "X-PyRIT-Run": self._run_id,
                    "X-PyRIT-Request-ID": ctx.request_id,
                    "Content-Type": "application/json",
                    "Accept-Encoding": "identity",
                },
            )
            if len(response.content) > self._max_bytes:
                raise ValueError("GHCP model response exceeded the run-scoped byte quota.")
            record["status"] = response.status_code
            record["response_base64"] = base64.b64encode(response.content).decode("ascii")
            return httpx.Response(
                response.status_code,
                headers={"Content-Type": response.headers.get("Content-Type", "application/json")},
                content=response.content,
                request=request,
            )
        except (httpx.HTTPError, ValueError) as error:
            record["error"] = type(error).__name__
            raise
        finally:
            self.exchanges.append(record)

    async def open_websocket(  # pyrit-async-suffix-exempt
        self, ctx: CopilotRequestContext
    ) -> CopilotWebSocketHandler:
        raise ValueError("The approved Inspect model bridge supports HTTP Responses only.")

    async def close_async(self) -> None:
        await self._client.aclose()


def _read_frame() -> dict[str, Any]:
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError("The trusted controller closed the GHCP command stream.")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("GHCP command frames must be JSON objects.")
    return value


def _write_frame(frame: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(frame, separators=(",", ":"), ensure_ascii=True) + "\n")
    sys.stdout.flush()


def _cli_identity(*, path: Path, expected_sha256: str) -> dict[str, str | int]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("The GHCP CLI must be an installed regular file in the agent image.")
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("The agent image's GHCP CLI does not match the approved SHA256.")
    getuid = getattr(os, "getuid", None)
    if getuid is None:
        raise RuntimeError("The GHCP guest worker requires a Linux agent sandbox.")
    uid = getuid()
    if uid == 0:
        raise ValueError("The GHCP worker must not run as root.")
    children = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()
    if len(children) != 1:
        raise RuntimeError("A single GHCP CLI child process was not observed in the agent sandbox.")
    cli_pid = int(children[0])
    return {
        "worker_pid": os.getpid(),
        "cli_pid": cli_pid,
        "cli_exe": os.readlink(f"/proc/{cli_pid}/exe"),
        "uid": uid,
        "pid_namespace": os.readlink("/proc/self/ns/pid"),
        "net_namespace": os.readlink("/proc/self/ns/net"),
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
        "cli_sha256": actual_sha256,
    }


def _prepare_guest_home(*, run_id: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", run_id):
        raise ValueError("GHCP HOME requires the trusted run's canonical UUID.")
    getuid = getattr(os, "getuid", None)
    if getuid is None or getuid() != 10001:
        raise RuntimeError("GHCP HOME can only be created by the nonroot agent UID10001.")
    root = Path("/tmp")
    root_stat = root.lstat()
    if not stat.S_ISDIR(root_stat.st_mode) or not root_stat.st_mode & stat.S_ISVTX:
        raise ValueError("GHCP HOME requires the already-approved sticky /tmp tmpfs.")
    home = root / f"pyrit-inspect-home-{run_id}"
    os.mkdir(home, mode=0o700)
    copilot_home = home / "copilot-state"
    os.mkdir(copilot_home, mode=0o700)
    for directory in (home, copilot_home):
        observed = directory.lstat()
        if (
            not stat.S_ISDIR(observed.st_mode)
            or observed.st_uid != 10001
            or stat.S_IMODE(observed.st_mode) != 0o700
            or directory.is_symlink()
        ):
            raise ValueError("GHCP HOME is not a fresh owner-only nonroot directory.")
    return home, copilot_home


def _validated_config(frame: dict[str, Any]) -> dict[str, Any]:
    if frame.get("op") != "start":
        raise ValueError("The first GHCP command must start a run.")
    for key in ("run_id", "token_file", "gateway_url", "model_id", "wire_model", "cli_path", "cli_sha256"):
        if not isinstance(frame.get(key), str) or not frame[key]:
            raise ValueError(f"Missing required GHCP setting: {key}.")
    if not frame["gateway_url"].startswith("http://model-bridge:"):
        raise ValueError("GHCP requires the internal model-bridge endpoint.")
    for key in ("max_turns", "timeout_seconds", "max_model_bytes", "max_prompt_tokens", "max_output_tokens"):
        if type(frame.get(key)) is not int or frame[key] < 1:
            raise ValueError(f"GHCP limit {key} must be a positive integer.")
    tools = frame.get("allowed_tools")
    if (
        not isinstance(tools, list)
        or not tools
        or any(not isinstance(tool, str) or tool not in {"bash", "read_file", "write_file"} for tool in tools)
    ):
        raise ValueError("GHCP can use only explicitly approved in-container tools.")
    return frame


class _GuestSession:
    def __init__(self, *, config: dict[str, Any]) -> None:
        self._config = config
        self._handler: _ScopedModelHandler | None = None
        self._events: list[dict[str, Any]] = []
        self._event_cursor = 0
        self._model_cursor = 0
        self._turn_count = 0
        self._client: CopilotClient | None = None
        self._session: Any = None
        self._identity: dict[str, str | int] | None = None

    async def start_async(self) -> None:
        config = self._config
        cli = Path(config["cli_path"])
        home, copilot_home = await asyncio.to_thread(_prepare_guest_home, run_id=config["run_id"])
        pinned_path = "/opt/pyrit/guest-local/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        if os.environ.get("PATH") != pinned_path:
            raise ValueError("GHCP guest process PATH differs from the approved immutable image.")
        token = await asyncio.to_thread(
            read_scoped_token,
            run_id=config["run_id"],
            token_file=config["token_file"],
        )
        self._handler = _ScopedModelHandler(
            url=config["gateway_url"],
            token=token,
            run_id=config["run_id"],
            max_bytes=config["max_model_bytes"],
            timeout=config["timeout_seconds"],
        )
        with_metadata = {
            "PATH": pinned_path,
            "HOME": str(home),
            "COPILOT_PROVIDER_BASE_URL": config["gateway_url"].rstrip("/") + "/v1",
            "COPILOT_MODEL": config["model_id"],
            "COPILOT_SKIP_CLI_DOWNLOAD": "1",
            "COPILOT_HOME": str(copilot_home),
        }
        self._client = CopilotClient(
            connection=RuntimeConnection.for_stdio(path=str(cli)),
            working_directory="/workspace",
            base_directory=str(copilot_home),
            env=with_metadata,
            use_logged_in_user=False,
            request_handler=self._handler,
            mode="copilot-cli",
        )
        await self._client.start()
        self._identity = await asyncio.to_thread(_cli_identity, path=cli, expected_sha256=config["cli_sha256"])
        self._session = await self._client.create_session(
            model=config["model_id"],
            provider={
                "type": "openai",
                "wire_api": "responses",
                "transport": "http",
                "base_url": with_metadata["COPILOT_PROVIDER_BASE_URL"],
                "bearer_token": token,
                "model_id": config["model_id"],
                "wire_model": config["wire_model"],
                "max_prompt_tokens": config["max_prompt_tokens"],
                "max_output_tokens": config["max_output_tokens"],
            },
            available_tools=config["allowed_tools"],
            on_permission_request=PermissionHandler.approve_all,
            on_event=self._on_event,
            enable_managed_settings=False,
            enable_host_git_operations=False,
            enable_session_store=False,
            enable_skills=False,
            enable_config_discovery=False,
            enable_file_hooks=False,
            skip_embedding_retrieval=True,
            enable_session_telemetry=False,
            enable_experimental_mode=False,
            mcp_servers={},
            tool_search={"enabled": False},
            streaming=True,
        )
        await asyncio.to_thread(
            _write_frame,
            {
                "kind": "ready",
                "run_id": config["run_id"],
                "session_id": self._session.session_id,
                "identity": self._identity,
            },
        )

    async def send_async(self, *, frame: dict[str, Any]) -> None:
        index = frame.get("turn_index")
        instruction = frame.get("instruction")
        if (
            type(index) is not int
            or index != self._turn_count + 1
            or index > self._config["max_turns"]
            or not isinstance(instruction, str)
            or not instruction.strip()
            or len(instruction.encode("utf-8")) > 32768
        ):
            raise ValueError("GHCP turn index, count or instruction is invalid.")
        self._turn_count = index
        assert self._session is not None
        response = await self._session.send_and_wait(instruction, timeout=self._config["timeout_seconds"])
        if self._handler is None:
            raise RuntimeError("GHCP model capture handler disappeared during a retained turn.")
        observed = self._events[self._event_cursor :]
        exchanges = self._handler.exchanges[self._model_cursor :]
        self._event_cursor = len(self._events)
        self._model_cursor = len(self._handler.exchanges)
        if not observed or not any(
            event.get("type") == "session.idle" and not event.get("agentId") for event in observed
        ):
            raise RuntimeError("The GHCP turn ended without an observed root session.idle event.")
        if response is None or response.type.value != "assistant.message":
            raise RuntimeError("The GHCP turn has no genuine final assistant message.")
        content = response.data.content
        if not isinstance(content, str) or not content:
            raise RuntimeError("The GHCP assistant response is not nonempty text.")
        await asyncio.to_thread(
            _write_frame,
            {
                "kind": "turn",
                "run_id": self._config["run_id"],
                "turn_index": index,
                "session_id": self._session.session_id,
                "identity": self._identity,
                "assistant_text": content,
                "events": observed,
                "model_exchanges": exchanges,
            },
        )

    async def stop_async(self) -> None:
        if self._session is not None:
            await self._session.disconnect()
        if self._client is not None:
            await self._client.stop()
        if self._handler is not None:
            await self._handler.close_async()
        cli_pid = self._identity["cli_pid"] if self._identity is not None else None
        if cli_pid is not None and Path(f"/proc/{cli_pid}").exists():
            raise RuntimeError("The GHCP CLI was still running when the scorer was about to start.")
        await asyncio.to_thread(_write_frame, {"kind": "stopped", "cli_exited": True})

    def _on_event(self, event: SessionEvent) -> None:
        self._events.append(session_event_to_dict(event))


async def _main_async() -> None:
    config = _validated_config(await asyncio.to_thread(_read_frame))
    worker = _GuestSession(config=config)
    try:
        await worker.start_async()
        while True:
            frame = await asyncio.to_thread(_read_frame)
            if frame.get("op") == "stop":
                await worker.stop_async()
                return
            if frame.get("op") != "send":
                raise ValueError("Only start, send, and stop are valid GHCP commands.")
            await worker.send_async(frame=frame)
    finally:
        if worker._client is not None:
            await worker._client.stop()
        if worker._handler is not None:
            await worker._handler.close_async()


if __name__ == "__main__":
    asyncio.run(_main_async())
