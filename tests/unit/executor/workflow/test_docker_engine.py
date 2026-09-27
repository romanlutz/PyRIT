# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import base64
import copy
import io
import json
import os
import tarfile
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import httpx
import pytest

from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.prompt_target.native_cli_models import NativeCliStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from pyrit.executor.workflow.docker_engine import DockerExecHandle


CONTAINER_ID = "a" * 64
EXEC_ID = "b" * 64
NETWORK_ID = "c" * 64


def wire_frame(stream: int, data: bytes) -> bytes:
    return bytes((stream, 0, 0, 0)) + len(data).to_bytes(4, "big") + data


class WireStream(httpx.AsyncByteStream):
    """Inert Engine output, including fragmented protocol headers and incomplete responses."""

    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        before_first_async: Callable[[], Awaitable[None]] | None = None,
        on_eof: Callable[[], None] | None = None,
        pause: bool = False,
        failure: Exception | None = None,
    ) -> None:
        self.chunks = chunks
        self.before_first_async = before_first_async
        self.on_eof = on_eof
        self.pause = pause
        self.failure = failure
        self.paused = asyncio.Event()
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self.before_first_async is not None:
            await self.before_first_async()
        for chunk in self.chunks:
            await asyncio.sleep(0)
            yield chunk
        if self.pause:
            self.paused.set()
            await asyncio.Event().wait()
        if self.failure:
            raise self.failure
        if self.on_eof:
            self.on_eof()

    async def aclose(self) -> None:
        self.closed = True


class FakeEngine:
    """All HTTP requests terminate in memory; no daemon/socket/client process is used."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.exec_id: Any = EXEC_ID
        self.container_id = CONTAINER_ID
        self.created: dict[str, Any] | None = None
        self.running = False
        self.started = False
        self.on_start: Callable[[], Awaitable[None]] | None = None
        self.exit_code: Any = 0
        self.wire: tuple[bytes, ...] = (wire_frame(1, b"stdout\xff\n"), wire_frame(2, b"stderr\r\n"))
        self.last_stream: WireStream | None = None
        self.pause_stream = False
        self.stream_failure: Exception | None = None
        self.start_status = 200
        self.content_type = "application/vnd.docker.raw-stream"
        self.content_encoding: str | None = None
        self.inspection_change: Callable[[dict[str, Any]], None] | None = None
        self.create_gate: asyncio.Event | None = None
        self.create_entered = asyncio.Event()
        self.start_gate: asyncio.Event | None = None
        self.start_entered = asyncio.Event()
        self.kill_gate: asyncio.Event | None = None
        self.kill_entered = asyncio.Event()
        self.kill_status = 204
        self.kill_keeps_running = False
        self.after_kill: Callable[[], None] | None = None
        self.containers: dict[str, dict[str, Any]] = {}
        self.networks: dict[str, dict[str, Any]] = {}
        self.container_change: Callable[[dict[str, Any]], None] | None = None
        self.config_archive = FakeConfigArchive()

    def client(self, **kwargs: Any) -> DockerEngineClient:
        return DockerEngineClient(transport=httpx.MockTransport(self.handle_async), **kwargs)

    async def handle_async(self, request: httpx.Request) -> httpx.Response:
        assert request.url.host == "docker-engine.invalid"
        assert "authorization" not in request.headers and "cookie" not in request.headers
        self.requests.append(request)
        path = request.url.path.removeprefix("/v1.47")
        if path.startswith("/containers/") and path.endswith("/archive"):
            return self.config_archive.handle(request)
        if path.startswith("/containers/") and path.endswith("/exec"):
            self.container_id = path.split("/")[2]
            self.created = json.loads(request.content)
            self.create_entered.set()
            if self.create_gate is not None:
                await self.create_gate.wait()
            return httpx.Response(201, json={"Id": self.exec_id})
        if path == f"/exec/{EXEC_ID}/json":
            assert self.created is not None
            value = {
                "ID": self.exec_id,
                "ContainerID": self.container_id,
                "Running": self.running,
                "ExitCode": self.exit_code,
                "Pid": 123 if self.running else 0,
                "OpenStdin": False,
                "OpenStdout": True,
                "OpenStderr": True,
                "ProcessConfig": {
                    "entrypoint": self.created["Cmd"][0],
                    "arguments": self.created["Cmd"][1:],
                    "user": self.created["User"],
                    "tty": False,
                    "privileged": False,
                },
            }
            if self.inspection_change:
                self.inspection_change(value)
            return httpx.Response(200, json=value)
        if path == f"/exec/{EXEC_ID}/start":
            assert json.loads(request.content) == {"Detach": False, "Tty": False}
            self.started = self.running = True
            self.start_entered.set()
            if self.start_gate is not None:
                await self.start_gate.wait()
            self.last_stream = WireStream(
                self.wire,
                before_first_async=self.on_start,
                on_eof=self._exited,
                pause=self.pause_stream,
                failure=self.stream_failure,
            )
            headers = {"content-type": self.content_type}
            if self.content_encoding:
                headers["content-encoding"] = self.content_encoding
            return httpx.Response(self.start_status, stream=self.last_stream, headers=headers)
        if path.startswith("/containers/") and path.endswith("/json"):
            identifier = path.split("/")[2]
            if identifier not in self.containers:
                return httpx.Response(404)
            container = copy.deepcopy(self.containers[identifier])
            if self.container_change:
                self.container_change(container)
            return httpx.Response(200, json=container)
        if path.startswith("/networks/"):
            return httpx.Response(200, json=self.networks[path.split("/")[2]])
        if path.startswith("/containers/") and path.endswith("/kill"):
            assert request.url.query == b"signal=SIGKILL"
            identifier = path.split("/")[2]
            self.kill_entered.set()
            if self.kill_gate is not None:
                await self.kill_gate.wait()
            if self.kill_status == 204 and not self.kill_keeps_running:
                self.containers[identifier]["State"].update(Running=False, Status="exited", Pid=0)
                if self.running:
                    self.running = False
                    self.exit_code = 137
            if self.after_kill:
                self.after_kill()
            return httpx.Response(self.kill_status)
        raise AssertionError(f"Unexpected inert Engine operation: {request.method} {path}")

    def _exited(self) -> None:
        self.running = False


class FakeConfigArchive:
    """Model archive bytes in memory without extracting files on host or guest."""

    def __init__(self) -> None:
        self.directories: dict[str, int] = {"/tmp": 0o700}
        self.writes: list[bytes] = []
        self.reads = 0
        self.stat_change: Callable[[str, dict[str, Any]], None] | None = None
        self.readback: Callable[[bytes], bytes] | None = None
        self.put_status = 200
        self.get_headers: dict[str, str] = {}

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.params["path"]
        if request.method == "PUT":
            assert dict(request.url.params) == {
                "path": path,
                "noOverwriteDirNonDir": "true",
                "copyUIDGID": "true",
            }
            assert request.headers["content-type"] == "application/x-tar"
            if self.put_status != 200:
                return httpx.Response(self.put_status)
            self.writes.append(request.content)
            with tarfile.open(fileobj=io.BytesIO(request.content), mode="r:") as archive:
                for member in archive:
                    if member.isdir():
                        self.directories[str(PurePosixPath(path) / member.name)] = member.mode
            return httpx.Response(200)
        if path not in self.directories:
            return httpx.Response(404)
        stat = {
            "name": PurePosixPath(path).name,
            "mode": (1 << 31) | self.directories[path],
            "size": 0,
            "mtime": "2026-09-26T00:00:00Z",
            "linkTarget": "",
        }
        if self.stat_change:
            self.stat_change(path, stat)
        headers = {"x-docker-container-path-stat": base64.b64encode(json.dumps(stat).encode()).decode()}
        if request.method == "HEAD":
            return httpx.Response(200, headers=headers)
        assert request.method == "GET" and self.writes
        self.reads += 1
        content = self.readback(self.writes[-1]) if self.readback else self.writes[-1]
        return httpx.Response(
            200,
            headers={**headers, "content-type": "application/x-tar", **self.get_headers},
            stream=WireStream((content,)),
        )


async def create_exec(engine: DockerEngineClient) -> DockerExecHandle:
    return await engine.create_exec_async(
        container_id=CONTAINER_ID,
        argv=("/opt/cli", "--", "literal $(not-a-shell)"),
        user="1000:1000",
        working_directory="/tmp/work",
    )


async def test_engine_exec_identity_direct_argv_and_raw_multiplex_order_async() -> None:
    fake = FakeEngine()
    raw = wire_frame(1, b"out\xc3\xa9\n") + wire_frame(2, b"err\xff\r\n") + wire_frame(1, b"last")
    fake.wire = (raw[:3], raw[3:10], raw[10:19], raw[19:])
    engine = fake.client()
    handle = await create_exec(engine)
    stream = await engine.start_exec_async(handle)
    chunks = [chunk async for chunk in stream.read_chunks_async()]
    assert b"".join(chunk.data for chunk in chunks if chunk.stream is NativeCliStream.STDOUT) == b"out\xc3\xa9\nlast"
    assert b"".join(chunk.data for chunk in chunks if chunk.stream is NativeCliStream.STDERR) == b"err\xff\r\n"
    assert chunks[0].stream is NativeCliStream.STDOUT and chunks[-1].data == b"last"
    assert stream.eof_observed
    assert fake.created == {
        "Cmd": ["/opt/cli", "--", "literal $(not-a-shell)"],
        "User": "1000:1000",
        "WorkingDir": "/tmp/work",
        "AttachStdin": False,
        "AttachStdout": True,
        "AttachStderr": True,
        "Privileged": False,
        "Tty": False,
    }
    observed = await engine.inspect_exec_async(handle)
    assert observed["ID"] == EXEC_ID and observed["ContainerID"] == CONTAINER_ID
    assert observed["Running"] is False and observed["ExitCode"] == 0
    with pytest.raises(DockerEngineError, match="once"):
        await engine.start_exec_async(handle)
    await stream.close_async()
    await engine.close_async()
    assert fake.last_stream.closed


@pytest.mark.parametrize("value", [None, "", "friendly-exec", "b" * 63, "B" * 64])
async def test_exec_create_requires_full_daemon_id_async(value: Any) -> None:
    fake = FakeEngine()
    fake.exec_id = value
    engine = fake.client()
    with pytest.raises(DockerEngineError, match="daemon-issued"):
        await create_exec(engine)
    assert not fake.started
    await engine.close_async()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ID", "d" * 64),
        ("ContainerID", "d" * 64),
        ("Running", None),
        ("Running", 0),
        ("OpenStdin", True),
        ("OpenStdout", False),
        ("OpenStderr", False),
        ("ProcessConfig", None),
    ],
)
async def test_exec_inspection_cannot_misattribute_process_or_pipes_async(*, field: str, value: Any) -> None:
    fake = FakeEngine()
    fake.inspection_change = lambda observed: observed.update({field: value})
    engine = fake.client()
    with pytest.raises(DockerEngineError, match="attribution"):
        await create_exec(engine)
    assert not fake.started
    await engine.close_async()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("entrypoint", "/bin/sh"),
        ("arguments", ["different"]),
        ("user", "0:0"),
        ("tty", True),
        ("tty", 0),
        ("privileged", True),
    ],
)
async def test_exec_process_config_is_bound_to_the_approved_handle_async(*, field: str, value: Any) -> None:
    fake = FakeEngine()
    fake.inspection_change = lambda observed: observed["ProcessConfig"].update({field: value})
    engine = fake.client()
    with pytest.raises(DockerEngineError, match="process differs"):
        await create_exec(engine)
    assert not fake.started
    await engine.close_async()


async def test_forged_or_reused_exec_handle_cannot_start_async() -> None:
    fake = FakeEngine()
    engine = fake.client()
    handle = await create_exec(engine)
    with pytest.raises(DockerEngineError, match="exact"):
        await engine.start_exec_async(replace(handle, container_id="c" * 64))
    with pytest.raises(DockerEngineError, match="reused"):
        await create_exec(engine)
    assert not fake.started
    await engine.close_async()


@pytest.mark.parametrize(
    ("status", "content_type", "encoding"),
    [
        (101, "application/vnd.docker.raw-stream", None),
        (500, "application/vnd.docker.raw-stream", None),
        (200, "text/plain", None),
        (200, "application/vnd.docker.raw-stream", "gzip"),
    ],
)
async def test_unsupported_attach_upgrade_or_encoding_fails_without_output_async(
    *, status: int, content_type: str, encoding: str | None
) -> None:
    fake = FakeEngine()
    fake.start_status, fake.content_type, fake.content_encoding = status, content_type, encoding
    engine = fake.client()
    handle = await create_exec(engine)
    with pytest.raises(DockerEngineError, match="no fallback"):
        await engine.start_exec_async(handle)
    assert fake.last_stream.closed
    with pytest.raises(DockerEngineError, match="once"):
        await engine.start_exec_async(handle)
    await engine.close_async()


@pytest.mark.parametrize(
    "wire",
    [
        b"\x01\x00",
        wire_frame(1, b"truncated")[:-1],
        wire_frame(3, b"daemon error, not guest stderr"),
        b"\x01\x01\x00\x00\x00\x00\x00\x00",
        b"\x00\x00\x00\x00\x00\x00\x00\x00",
    ],
)
async def test_invalid_or_truncated_multiplex_stream_never_has_clean_eof_async(wire: bytes) -> None:
    fake = FakeEngine()
    fake.wire = (wire,)
    engine = fake.client()
    stream = await engine.start_exec_async(await create_exec(engine))
    with pytest.raises(DockerEngineError):
        _ = [chunk async for chunk in stream.read_chunks_async()]
    assert not stream.eof_observed
    await stream.close_async()
    await engine.close_async()


@pytest.mark.parametrize("limit", ["frame", "wire"])
async def test_output_limits_are_errors_not_successful_truncation_async(limit: str) -> None:
    fake = FakeEngine()
    fake.wire = (wire_frame(1, b"x" * 40),)
    engine = fake.client(**({"max_frame_bytes": 8} if limit == "frame" else {"max_wire_bytes": 16}))
    stream = await engine.start_exec_async(await create_exec(engine))
    with pytest.raises(DockerEngineError, match="limit"):
        _ = [chunk async for chunk in stream.read_chunks_async()]
    assert not stream.eof_observed
    await stream.close_async()
    await engine.close_async()


async def test_stream_network_error_is_not_guest_exit_async() -> None:
    fake = FakeEngine()
    fake.stream_failure = httpx.ReadError("inert failure")
    engine = fake.client()
    stream = await engine.start_exec_async(await create_exec(engine))
    with pytest.raises(DockerEngineError, match="guest exit is unknown"):
        _ = [chunk async for chunk in stream.read_chunks_async()]
    assert not stream.eof_observed
    await stream.close_async()
    await engine.close_async()


@pytest.mark.parametrize("payload", [b"[]", b"{", b'{"large":"' + b"x" * 100 + b'"}'])
async def test_bad_control_response_is_not_an_inspection_async(payload: bytes) -> None:
    engine = DockerEngineClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=payload)), max_control_bytes=32
    )
    with pytest.raises(DockerEngineError):
        await engine.inspect_container_async(CONTAINER_ID)
    await engine.close_async()


async def test_fixed_origin_ignores_ambient_auth_and_never_follows_redirects_async() -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(307, headers={"Location": "https://unapproved.invalid"})

    with patch.dict(os.environ, {"HTTP_PROXY": "http://unapproved.invalid", "GH_TOKEN": "inert-token"}):
        engine = DockerEngineClient(transport=httpx.MockTransport(handle))
        with pytest.raises(DockerEngineError, match="HTTP 307"):
            await engine.inspect_container_async(CONTAINER_ID)
        await engine.close_async()
    assert len(requests) == 1 and requests[0].url.host == "docker-engine.invalid"
    assert "authorization" not in requests[0].headers and "cookie" not in requests[0].headers


async def test_engine_cookies_are_never_forwarded_into_exec_requests_async() -> None:
    fake = FakeEngine()

    async def cookie_response_async(request: httpx.Request) -> httpx.Response:
        response = await fake.handle_async(request)
        response.headers["set-cookie"] = "inert_cookie=value; Path=/"
        return response

    engine = DockerEngineClient(transport=httpx.MockTransport(cookie_response_async))
    handle = await create_exec(engine)
    stream = await engine.start_exec_async(handle)
    _ = [part async for part in stream.read_chunks_async()]
    await engine.inspect_exec_async(handle)
    assert all("cookie" not in request.headers for request in fake.requests)
    await stream.close_async()
    await engine.close_async()


def test_windows_named_pipe_transport_is_explicitly_unsupported() -> None:
    with patch("pyrit.executor.workflow.docker_engine.os", spec=os) as platform:
        platform.name = "nt"
        with pytest.raises(NotImplementedError, match="Unix sockets only"):
            DockerEngineClient.for_unix_socket(socket_path=Path("unused"))
