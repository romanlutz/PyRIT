# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Host-only Docker Engine exec transport with daemon-attributed exit status."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any

import httpx

from pyrit.prompt_target.native_cli_models import NativeCliProcessChunk, NativeCliStream

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
    from pathlib import Path


class DockerEngineError(RuntimeError):
    """Engine identity, framing or control failure, never a guest CLI exit code."""


@dataclass(frozen=True, kw_only=True)
class DockerExecHandle:
    """One daemon-issued exec identity and the exact process specification it names."""

    exec_id: str
    container_id: str
    argv: tuple[str, ...]
    user: str
    working_directory: str


class DockerExecStream:
    """Non-TTY Docker multiplex frames, yielding original bytes in observed wire order."""

    def __init__(
        self,
        *,
        response: httpx.Response,
        max_frame_bytes: int,
        max_wire_bytes: int,
    ) -> None:
        """Own an already-accepted exec-start response and bounded output framing."""
        self._response = response
        self._max_frame_bytes = max_frame_bytes
        self._max_wire_bytes = max_wire_bytes
        self._consumed = False
        self._closed = False
        self._eof_observed = False

    @property
    def eof_observed(self) -> bool:
        """Whether the original daemon stream ended on a complete multiplex boundary."""
        return self._eof_observed

    async def read_chunks_async(self) -> AsyncGenerator[NativeCliProcessChunk, None]:
        """
        Demultiplex stdout/stderr without decoding or manufacturing guest output.

        Yields:
            NativeCliProcessChunk: Actual bytes from one daemon-framed guest pipe.

        Raises:
            DockerEngineError: If framing or output bounds are violated.
            RuntimeError: If the response was already consumed or closed.
        """
        if self._consumed or self._closed:
            raise RuntimeError("An Engine exec stream can be consumed only once.")
        self._consumed = True
        header = bytearray()
        remaining = wire_bytes = 0
        stream = NativeCliStream.STDOUT
        try:
            async for chunk in self._response.aiter_raw():
                wire_bytes += len(chunk)
                if wire_bytes > self._max_wire_bytes:
                    raise DockerEngineError("Engine exec output exceeded its wire-byte limit; coverage is incomplete.")
                offset = 0
                while offset < len(chunk):
                    if remaining == 0:
                        length = min(8 - len(header), len(chunk) - offset)
                        header.extend(chunk[offset : offset + length])
                        offset += length
                        if len(header) < 8:
                            continue
                        if header[0] not in (1, 2) or header[1:4] != b"\x00\x00\x00":
                            raise DockerEngineError("Engine exec returned unsupported multiplex framing.")
                        stream = NativeCliStream.STDOUT if header[0] == 1 else NativeCliStream.STDERR
                        remaining = int.from_bytes(header[4:8], "big")
                        header.clear()
                        if remaining > self._max_frame_bytes:
                            raise DockerEngineError("Engine exec frame exceeded its declared output limit.")
                        if remaining == 0:
                            continue
                    length = min(remaining, len(chunk) - offset, 16_384)
                    if length:
                        yield NativeCliProcessChunk(stream=stream, data=chunk[offset : offset + length])
                        remaining -= length
                        offset += length
            if header or remaining:
                raise DockerEngineError("Engine exec stream ended in a partial multiplex frame.")
            self._eof_observed = True
        except httpx.HTTPError as error:
            raise DockerEngineError("Engine exec output transport failed; guest exit is unknown.") from error

    async def close_async(self) -> None:
        """Release the host HTTP response; this does not stop any guest process."""
        if not self._closed:
            self._closed = True
            await self._response.aclose()


class DockerEngineClient:
    """Fixed-origin Engine HTTP client; production construction is local Unix-socket only."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport,
        api_version: str = "1.47",
        control_timeout_seconds: float = 5,
        max_control_bytes: int = 1_048_576,
        max_frame_bytes: int = 8_388_608,
        max_wire_bytes: int = 33_554_432,
    ) -> None:
        """
        Bind a trusted transport; injected transports are for host-owned configuration and tests.

        Raises:
            ValueError: If API version or resource limits are invalid.
        """
        if not re.fullmatch(r"1\.[0-9]{2,3}", api_version):
            raise ValueError("A concrete Docker Engine API version is required.")
        if (
            not math.isfinite(control_timeout_seconds)
            or not 0 < control_timeout_seconds <= 10
            or not 1 <= max_control_bytes <= 4_194_304
            or not 1 <= max_frame_bytes <= 16_777_216
            or not 8 <= max_wire_bytes <= 67_108_864
        ):
            raise ValueError("Engine control, frame and wire limits must be positive and bounded.")
        self._prefix = f"/v{api_version}"
        self._control_timeout = control_timeout_seconds
        self._max_control_bytes = max_control_bytes
        self._max_frame_bytes = max_frame_bytes
        self._max_wire_bytes = max_wire_bytes
        self._execs: dict[str, DockerExecHandle] = {}
        self._start_requested: set[str] = set()
        self._client = httpx.AsyncClient(
            base_url="http://docker-engine.invalid",
            transport=transport,
            trust_env=False,
            follow_redirects=False,
            timeout=httpx.Timeout(control_timeout_seconds, read=None),
        )

    @classmethod
    def for_unix_socket(
        cls, *, socket_path: Path, api_version: str = "1.47", control_timeout_seconds: float = 5
    ) -> DockerEngineClient:
        """
        Use a host-local Engine socket without exposing it inside the agent.

        Returns:
            DockerEngineClient: A caller-owned client requiring explicit close.

        Raises:
            NotImplementedError: If the host is not POSIX; Windows named pipes are not implemented.
            ValueError: If the socket path is not absolute.
        """
        if os.name != "posix":
            raise NotImplementedError("Engine exec transport supports POSIX host Unix sockets only.")
        if not socket_path.is_absolute() or ".." in socket_path.parts:
            raise ValueError("The host Engine socket must be an explicit absolute path.")
        return cls(
            transport=httpx.AsyncHTTPTransport(uds=str(socket_path), retries=0, trust_env=False, http2=False),
            api_version=api_version,
            control_timeout_seconds=control_timeout_seconds,
        )

    async def create_exec_async(
        self, *, container_id: str, argv: tuple[str, ...], user: str, working_directory: str
    ) -> DockerExecHandle:
        """
        Create one nonprivileged, non-TTY exec with no stdin or credential environment override.

        Returns:
            DockerExecHandle: The daemon-issued identity after exact process inspection.

        Raises:
            ValueError: If process arguments or identity are invalid.
            DockerEngineError: If create or identity inspection fails.
        """
        self._validate_id(container_id)
        directory = PurePosixPath(working_directory)
        if (
            not argv
            or any("\x00" in arg for arg in argv)
            or sum(len(arg.encode("utf-8")) for arg in argv) > 262_144
            or not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", user)
            or not PurePosixPath(argv[0]).is_absolute()
            or not directory.is_absolute()
            or ".." in directory.parts
            or str(directory) != working_directory
            or "\\" in working_directory
            or any(ord(char) < 32 for char in working_directory)
        ):
            raise ValueError("Engine execution requires explicit bounded argv, nonroot user and guest workdir.")
        result = await self._json_async(
            "POST",
            f"/containers/{container_id}/exec",
            status=201,
            body={
                "AttachStdin": False,
                "AttachStdout": True,
                "AttachStderr": True,
                "Tty": False,
                "Privileged": False,
                "Cmd": list(argv),
                "User": user,
                "WorkingDir": working_directory,
            },
        )
        exec_id = result.get("Id")
        if not isinstance(exec_id, str) or not re.fullmatch(r"[0-9a-f]{64}", exec_id):
            raise DockerEngineError("Engine exec create did not return a full daemon-issued ID.")
        if exec_id in self._execs:
            raise DockerEngineError("Engine exec create reused a previously observed identity.")
        handle = DockerExecHandle(
            exec_id=exec_id, container_id=container_id, argv=argv, user=user, working_directory=working_directory
        )
        self._execs[exec_id] = handle
        inspection = await self.inspect_exec_async(handle)
        if inspection["Running"] is not False:
            raise DockerEngineError("New Engine exec was already running before start.")
        return handle

    async def start_exec_async(self, handle: DockerExecHandle) -> DockerExecStream:
        """
        Attach to exactly the inspected exec; only non-TTY HTTP 200 framing is supported.

        Returns:
            DockerExecStream: Original daemon-multiplexed output.

        Raises:
            DockerEngineError: If the identity changed or stream upgrade/type is unsupported.
        """
        if self._execs.get(handle.exec_id) != handle or handle.exec_id in self._start_requested:
            raise DockerEngineError("Only a newly created, exact Engine exec may be started once.")
        self._start_requested.add(handle.exec_id)
        inspection = await self.inspect_exec_async(handle)
        if inspection["Running"] is not False:
            raise DockerEngineError("Engine exec start requires an unstarted, inspected process.")
        request = self._request(
            method="POST", path=f"/exec/{handle.exec_id}/start", body={"Detach": False, "Tty": False}
        )
        try:
            async with asyncio.timeout(self._control_timeout):
                response = await self._client.send(request, stream=True)
        except httpx.HTTPError as error:
            raise DockerEngineError("Engine exec start transport failed; start outcome is unknown.") from error
        if (
            response.status_code != 200
            or response.headers.get("content-type", "").split(";", 1)[0].strip()
            not in {
                "application/vnd.docker.raw-stream",
                "application/vnd.docker.multiplexed-stream",
            }
            or response.headers.get("content-encoding", "identity") != "identity"
        ):
            await response.aclose()
            raise DockerEngineError("Engine exec start requires an unencoded HTTP 200 non-TTY stream; no fallback.")
        return DockerExecStream(
            response=response, max_frame_bytes=self._max_frame_bytes, max_wire_bytes=self._max_wire_bytes
        )

    async def inspect_exec_async(self, handle: DockerExecHandle) -> dict[str, Any]:
        """
        Verify daemon identity and process configuration before interpreting status.

        Returns:
            dict[str, Any]: The exact exec observation; ExitCode is not a Docker client result.

        Raises:
            DockerEngineError: If any required identity or process field is absent or changed.
        """
        self._validate_id(handle.exec_id)
        result = await self._json_async("GET", f"/exec/{handle.exec_id}/json")
        config = result.get("ProcessConfig")
        if (
            result.get("ID") != handle.exec_id
            or result.get("ContainerID") != handle.container_id
            or type(result.get("Running")) is not bool
            or not isinstance(config, dict)
            or result.get("OpenStdin") is not False
            or result.get("OpenStdout") is not True
            or result.get("OpenStderr") is not True
        ):
            raise DockerEngineError("Engine exec identity or pipe attribution is missing or mismatched.")
        expected = {
            "entrypoint": handle.argv[0],
            "arguments": list(handle.argv[1:]),
            "user": handle.user,
            "privileged": False,
            "tty": False,
        }
        if (
            any(config.get(key) != value for key, value in expected.items())
            or config.get("tty") is not False
            or config.get("privileged") is not False
        ):
            raise DockerEngineError("Engine exec process differs from its approved argv, user or privilege profile.")
        return result

    async def inspect_network_async(self, network_id: str) -> dict[str, Any]:
        """
        Inspect the allocation's exact network without implying host/gateway reachability.

        Returns:
            dict[str, Any]: The daemon network metadata.

        Raises:
            DockerEngineError: If the daemon returns a different network.
        """
        self._validate_id(network_id)
        result = await self._json_async("GET", f"/networks/{network_id}")
        if result.get("Id") != network_id:
            raise DockerEngineError("Engine network inspection returned a different identity.")
        return result

    async def inspect_container_async(self, container_id: str) -> dict[str, Any]:
        """
        Inspect an exact full container ID through the same Engine as exec.

        Returns:
            dict[str, Any]: The daemon observation for that ID.

        Raises:
            DockerEngineError: If the daemon returns a different container.
        """
        self._validate_id(container_id)
        result = await self._json_async("GET", f"/containers/{container_id}/json")
        if result.get("Id") != container_id:
            raise DockerEngineError("Engine container inspection returned a different identity.")
        return result

    async def kill_container_async(self, container_id: str) -> None:
        """
        Force-stop exactly one container; callers must independently inspect stopped state.

        Raises:
            DockerEngineError: If the kill request is not acknowledged; no not-found fallback.
        """
        self._validate_id(container_id)
        await self._json_async("POST", f"/containers/{container_id}/kill?signal=SIGKILL", status=204)

    async def close_async(self) -> None:
        """Close host transport resources, not guest processes or the daemon."""
        await self._client.aclose()

    async def _json_async(
        self, method: str, path: str, *, status: int = 200, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        request = self._request(method=method, path=path, body=body)
        try:
            async with asyncio.timeout(self._control_timeout):
                response = await self._client.send(request, stream=True)
                try:
                    if response.status_code != status:
                        raise DockerEngineError(
                            f"Engine control returned HTTP {response.status_code}; expected {status}."
                        )
                    content = bytearray()
                    async for part in response.aiter_bytes(chunk_size=16_384):
                        if len(content) + len(part) > self._max_control_bytes:
                            raise DockerEngineError("Engine control response exceeded its size limit.")
                        content.extend(part)
                finally:
                    await response.aclose()
        except httpx.HTTPError as error:
            raise DockerEngineError("Engine control transport failed.") from error
        if status == 204:
            if content:
                raise DockerEngineError("Engine kill acknowledgement unexpectedly contained a body.")
            return {}
        try:
            value = json.loads(content)
        except (ValueError, UnicodeError) as error:
            raise DockerEngineError("Engine control returned malformed JSON.") from error
        if not isinstance(value, dict):
            raise DockerEngineError("Engine control returned no structured observation.")
        return value

    def _request(self, *, method: str, path: str, body: dict[str, Any] | None) -> httpx.Request:
        request = self._client.build_request(method, self._prefix + path, json=body)
        request.headers.pop("cookie", None)
        return request

    @staticmethod
    def _validate_id(value: str) -> None:
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("Engine operations require a full immutable Docker ID.")
