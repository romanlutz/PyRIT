# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-owned ASGI listener on a prequalified private Docker bridge socket."""

from __future__ import annotations

import asyncio
import math
import re
import socket
from dataclasses import dataclass
from ipaddress import IPv4Address, IPv4Network
from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from starlette.types import ASGIApp

    from pyrit.prompt_target.gateway.responses_contract import GatewayRoute


@dataclass(frozen=True, kw_only=True)
class GatewayBridgeBinding:
    """Provider-observed identity, not proof that a guest can reach the host."""

    run_id: str
    network_id: str
    address: IPv4Address
    subnet: IPv4Network
    port: int

    def __post_init__(self) -> None:
        """
        Reject wildcard, public, loopback, or ambiguously scoped listener addresses.

        Raises:
            ValueError: If this is not one bounded, run-owned private IPv4 bridge identity.
        """
        private_ranges = (
            IPv4Network("10.0.0.0/8"),
            IPv4Network("172.16.0.0/12"),
            IPv4Network("192.168.0.0/16"),
        )
        if (
            not isinstance(self.run_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", self.run_id)
            or not isinstance(self.network_id, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.network_id)
            or not isinstance(self.address, IPv4Address)
            or not isinstance(self.subnet, IPv4Network)
            or self.address not in self.subnet
            or not any(self.subnet.subnet_of(private) for private in private_ranges)
            or self.address.is_loopback
            or self.address.is_link_local
            or self.address.is_multicast
            or type(self.port) is not int
            or not 1 <= self.port <= 65535
        ):
            raise ValueError("A run-scoped gateway requires a verified private IPv4 bridge address and port.")


class GatewayListenerCleanupError(RuntimeError):
    """The run-owned listener did not stop within its bounded cleanup deadline."""


class RunScopedModelGatewayListener:
    """Serve one model-only ASGI app on a caller-provided, already-listening socket."""

    def __init__(
        self,
        *,
        route: GatewayRoute,
        binding: GatewayBridgeBinding,
        app: ASGIApp,
        listening_socket: socket.socket,
        startup_seconds: float = 5,
        cleanup_seconds: float = 10,
    ) -> None:
        """
        Accept ownership of an already-bound provider-approved listener socket.

        Raises:
            ValueError: If the socket, run identity, or finite deadlines do not match approval.
        """
        if route.run_id != binding.run_id:
            raise ValueError("The gateway app route and listener binding must belong to the same run.")
        if (
            type(startup_seconds) not in (int, float)
            or not math.isfinite(startup_seconds)
            or not 0 < startup_seconds <= 60
            or type(cleanup_seconds) not in (int, float)
            or not math.isfinite(cleanup_seconds)
            or not 0 < cleanup_seconds <= 120
        ):
            raise ValueError("Gateway listener startup and cleanup require bounded positive deadlines.")
        if (
            listening_socket.family != socket.AF_INET
            or listening_socket.type & socket.SOCK_STREAM != socket.SOCK_STREAM
            or listening_socket.fileno() < 0
            or listening_socket.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN) != 1
            or listening_socket.getsockname()[:2] != (str(binding.address), binding.port)
        ):
            raise ValueError("The gateway listener socket is not the approved bound IPv4 bridge socket.")
        self.binding = binding
        self._socket = listening_socket
        self._startup_seconds = startup_seconds
        self._cleanup_seconds = cleanup_seconds
        self._server = uvicorn.Server(
            uvicorn.Config(
                app=app,
                host=str(binding.address),
                port=binding.port,
                loop="asyncio",
                http="h11",
                ws="none",
                lifespan="off",
                access_log=False,
                log_config=None,
                log_level="error",
                proxy_headers=False,
                server_header=False,
                date_header=False,
                workers=1,
                backlog=32,
                limit_concurrency=16,
                timeout_keep_alive=2,
                timeout_graceful_shutdown=5,
            )
        )
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._cleanup_failed = False

    @property
    def base_url(self) -> str:
        """A candidate guest route, not a claim that firewall or guest reachability was checked."""
        return f"http://{self.binding.address}:{self.binding.port}"

    @property
    def is_locally_running(self) -> bool:
        """Whether the local ASGI server started and is still running."""
        return bool(self._task and self._server.started and not self._task.done() and not self._closed)

    async def start_async(self) -> None:
        """
        Start exactly one local server on the transferred socket.

        Raises:
            RuntimeError: If startup fails or stops before readiness.
            TimeoutError: If the server does not start within the approved deadline.
            asyncio.CancelledError: If the caller cancels after owned cleanup settles.
        """
        if self._task is not None or self._closed:
            raise RuntimeError("The model gateway listener may be started only once.")
        self._task = asyncio.create_task(self._server.serve(sockets=[self._socket]))
        try:
            async with asyncio.timeout(self._startup_seconds):
                while not self._server.started:
                    if self._task.done():
                        self._task.result()
                        raise RuntimeError("Model gateway server exited before its listening boundary.")
                    await asyncio.sleep(0.01)
            if self._task.done():
                self._task.result()
                raise RuntimeError("Model gateway server exited during startup.")
        except (Exception, asyncio.CancelledError) as error:
            try:
                await self.close_async()
            except (Exception, asyncio.CancelledError) as cleanup_error:
                error.add_note(f"Gateway listener cleanup failed ({type(cleanup_error).__name__}).")
                raise error from cleanup_error
            raise

    async def close_async(self) -> None:
        """
        Stop and release only this listener's socket, never another run's app.

        Raises:
            GatewayListenerCleanupError: If the server cannot be confirmed stopped.
        """
        if self._closed:
            if self._cleanup_failed:
                raise GatewayListenerCleanupError("Gateway listener cleanup remains unconfirmed.")
            return
        self._closed = True
        self._server.should_exit = True
        failure: BaseException | None = None
        try:
            if self._task is not None:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=self._cleanup_seconds)
        except (Exception, asyncio.CancelledError) as error:
            self._cleanup_failed = True
            if self._task is not None and not self._task.done():
                self._task.cancel()
            failure = error
        try:
            await asyncio.to_thread(self._socket.close)
        except OSError as error:
            self._cleanup_failed = True
            if failure is not None:
                failure.add_note(f"Gateway listener socket close failed ({type(error).__name__}).")
            else:
                failure = error
        if failure is not None:
            if isinstance(failure, asyncio.CancelledError):
                raise failure
            raise GatewayListenerCleanupError("Gateway listener did not confirm clean shutdown.") from failure
