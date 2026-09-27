# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-scoped gateway startup/cleanup with fake sockets and no network I/O."""

from __future__ import annotations

import asyncio
import math
import socket
from dataclasses import replace
from ipaddress import IPv4Address, IPv4Network
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest
import uvicorn
from starlette.applications import Starlette

from pyrit.prompt_target.gateway.responses_contract import GatewayRoute
from pyrit.prompt_target.gateway.run_listener import (
    GatewayBridgeBinding,
    GatewayListenerCleanupError,
    RunScopedModelGatewayListener,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _FakeServer:
    def __init__(self, *, start: bool = True, fail: bool = False, ignore_exit: bool = False) -> None:
        self.started = False
        self._exit = asyncio.Event()
        self._should_exit = False
        self.start = start
        self.fail = fail
        self.ignore_exit = ignore_exit
        self.sockets: Sequence[socket.socket] | None = None

    @property
    def should_exit(self) -> bool:
        return self._should_exit

    @should_exit.setter
    def should_exit(self, value: bool) -> None:
        self._should_exit = value
        if value:
            self._exit.set()

    async def serve(self, sockets: list[socket.socket] | None = None) -> None:
        self.sockets = sockets
        if self.fail:
            raise OSError("Inert server startup failed")
        self.started = self.start
        if self.ignore_exit:
            await asyncio.Event().wait()
        else:
            await self._exit.wait()


def _binding() -> GatewayBridgeBinding:
    return GatewayBridgeBinding(
        run_id="run-inert-1",
        network_id="a" * 64,
        address=IPv4Address("172.30.0.1"),
        subnet=IPv4Network("172.30.0.0/16"),
        port=43123,
    )


def _socket(*, binding: GatewayBridgeBinding | None = None) -> MagicMock:
    spec = binding or _binding()
    result = MagicMock(spec=socket.socket)
    result.family = socket.AF_INET
    result.type = socket.SOCK_STREAM
    result.fileno.return_value = 42
    result.getsockopt.return_value = 1
    result.getsockname.return_value = (str(spec.address), spec.port)
    return result


def _listener(
    *,
    server: _FakeServer,
    listening_socket: MagicMock | None = None,
    binding: GatewayBridgeBinding | None = None,
    startup_seconds: float = 0.05,
    cleanup_seconds: float = 0.05,
) -> RunScopedModelGatewayListener:
    selected = binding or _binding()
    route = GatewayRoute(run_id=selected.run_id, model="offline-model", guest_token="g" * 40)
    with patch.object(uvicorn, "Server", return_value=server):
        return RunScopedModelGatewayListener(
            route=route,
            binding=selected,
            app=Starlette(),
            listening_socket=listening_socket or _socket(binding=selected),
            startup_seconds=startup_seconds,
            cleanup_seconds=cleanup_seconds,
        )


async def test_gateway_listener_accepts_only_prebound_private_socket_and_shuts_down_once_async() -> None:
    server = _FakeServer()
    sock = _socket()
    listener = _listener(server=server, listening_socket=sock)
    assert not listener.is_locally_running
    assert listener.base_url == "http://172.30.0.1:43123"
    await listener.start_async()
    assert listener.is_locally_running and server.sockets == [sock]
    await listener.close_async()
    assert not listener.is_locally_running
    sock.close.assert_called_once()
    await listener.close_async()
    sock.close.assert_called_once()
    with pytest.raises(RuntimeError, match="only once"):
        await listener.start_async()


def test_gateway_listener_disables_access_logs_proxy_headers_and_websockets() -> None:
    binding = _binding()
    route = GatewayRoute(run_id=binding.run_id, model="offline-model", guest_token="g" * 40)
    with patch.object(uvicorn, "Server", return_value=_FakeServer()) as factory:
        RunScopedModelGatewayListener(
            route=route,
            binding=binding,
            app=Starlette(),
            listening_socket=_socket(binding=binding),
        )
    config = factory.call_args.args[0]
    assert config.access_log is False and config.proxy_headers is False
    assert config.ws == "none" and config.server_header is False and config.date_header is False
    assert config.log_config is None and config.workers == 1 and config.limit_concurrency == 16


@pytest.mark.parametrize(
    ("startup", "cleanup"),
    [(True, 1), (1, False), (math.nan, 1), (1, math.inf), (0, 1), (1, -1)],
)
def test_gateway_listener_requires_real_finite_deadlines(startup: float, cleanup: float) -> None:
    binding = _binding()
    with pytest.raises(ValueError, match="bounded positive deadlines"):
        RunScopedModelGatewayListener(
            route=GatewayRoute(run_id=binding.run_id, model="offline-model", guest_token="g" * 40),
            binding=binding,
            app=Starlette(),
            listening_socket=_socket(binding=binding),
            startup_seconds=startup,
            cleanup_seconds=cleanup,
        )


@pytest.mark.parametrize(
    "change",
    [
        {"network_id": "invalid"},
        {"port": 0},
        {"port": True},
        {"address": IPv4Address("127.0.0.1")},
        {"address": IPv4Address("0.0.0.0")},
        {"address": IPv4Address("8.8.8.8"), "subnet": IPv4Network("8.8.8.0/24")},
        {"address": IPv4Address("172.31.0.1")},
        {"subnet": IPv4Network("172.0.0.0/8")},
    ],
)
def test_gateway_binding_rejects_unapproved_network_identity(change: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="verified private IPv4"):
        replace(_binding(), **change)


@pytest.mark.parametrize("defect", ["wrong_run", "wrong_socket", "not_listening", "wildcard", "closed"])
def test_gateway_listener_rejects_foreign_or_unbound_socket_before_server(defect: str) -> None:
    binding = _binding()
    sock = _socket()
    route = GatewayRoute(
        run_id="another-run" if defect == "wrong_run" else binding.run_id,
        model="offline-model",
        guest_token="g" * 40,
    )
    if defect == "wrong_socket":
        sock.getsockname.return_value = ("172.30.0.2", binding.port)
    elif defect == "not_listening":
        sock.getsockopt.return_value = 0
    elif defect == "wildcard":
        sock.getsockname.return_value = ("0.0.0.0", binding.port)
    elif defect == "closed":
        sock.fileno.return_value = -1
    with (
        patch.object(uvicorn, "Server") as server,
        pytest.raises(ValueError, match="same run|approved bound IPv4"),
    ):
        RunScopedModelGatewayListener(
            route=route,
            binding=binding,
            app=Starlette(),
            listening_socket=sock,
        )
    server.assert_not_called()


async def test_gateway_listener_startup_timeout_closes_owned_socket_async() -> None:
    server = _FakeServer(start=False)
    sock = _socket()
    listener = _listener(server=server, listening_socket=sock, startup_seconds=0.03)
    with pytest.raises(TimeoutError):
        await listener.start_async()
    sock.close.assert_called_once()
    assert not listener.is_locally_running


async def test_gateway_listener_startup_failure_propagates_without_success_async() -> None:
    server = _FakeServer(fail=True)
    sock = _socket()
    listener = _listener(server=server, listening_socket=sock)
    with pytest.raises(OSError, match="Inert server startup failed"):
        await listener.start_async()
    sock.close.assert_called_once()
    assert not listener.is_locally_running


async def test_gateway_listener_stop_timeout_is_latched_and_never_reports_running_async() -> None:
    server = _FakeServer(ignore_exit=True)
    sock = _socket()
    listener = _listener(server=server, listening_socket=sock, cleanup_seconds=0.02)
    await listener.start_async()
    with pytest.raises(GatewayListenerCleanupError, match="did not confirm clean shutdown"):
        await listener.close_async()
    sock.close.assert_called_once()
    assert not listener.is_locally_running
    with pytest.raises(GatewayListenerCleanupError, match="remains unconfirmed"):
        await listener.close_async()


async def test_gateway_listener_socket_close_error_is_latched_async() -> None:
    server = _FakeServer()
    sock = _socket()
    sock.close.side_effect = OSError("inert socket close failure")
    listener = _listener(server=server, listening_socket=sock)
    await listener.start_async()
    with pytest.raises(GatewayListenerCleanupError, match="did not confirm clean shutdown"):
        await listener.close_async()
    with pytest.raises(GatewayListenerCleanupError, match="remains unconfirmed"):
        await listener.close_async()
    sock.close.assert_called_once()


async def test_gateway_listener_cancellation_cleans_up_prebound_socket_async() -> None:
    server = _FakeServer(start=False)
    sock = _socket()
    listener = _listener(server=server, listening_socket=sock, startup_seconds=1)
    task = asyncio.create_task(listener.start_async())
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sock.close.assert_called_once()
    assert not listener.is_locally_running
