# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Qualify only exact fake Docker network metadata for a candidate listener."""

from __future__ import annotations

import copy
from ipaddress import IPv4Address, IPv4Network
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uvicorn

from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.executor.workflow.docker_gateway_bridge import (
    acquire_gateway_bridge_binding_async,
    derive_gateway_bridge_binding,
)
from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute
from pyrit.prompt_target.gateway.run_listener import RunScopedModelGatewayListener
from tests.unit.executor.workflow.test_docker_compose import make_lease
from tests.unit.prompt_target.gateway.test_run_listener import _FakeServer, _socket

if TYPE_CHECKING:
    from pyrit.executor.workflow.docker_compose import ComposeAllocation, DockerComposeEnvironmentLease


async def _bridge_case_async() -> tuple[DockerComposeEnvironmentLease, ComposeAllocation, dict[str, object]]:
    lease, runner = make_lease(2)
    allocation = await lease.acquire_async()
    network: dict[str, object] = copy.deepcopy(runner.networks[allocation.network_id])
    network["IPAM"] = {
        "Driver": "default",
        "Config": [{"Subnet": "172.30.0.0/16", "Gateway": "172.30.0.1"}],
    }
    return lease, allocation, network


async def test_gateway_bridge_candidate_uses_exact_owned_internal_network_async() -> None:
    lease, allocation, network = await _bridge_case_async()
    try:
        binding = derive_gateway_bridge_binding(lease=lease, allocation=allocation, network=network, port=43123)
        assert binding.run_id == lease.run_id and binding.network_id == allocation.network_id
        assert binding.address == IPv4Address("172.30.0.1")
        assert binding.subnet == IPv4Network("172.30.0.0/16")
        assert binding.port == 43123
    finally:
        await lease.close_async()


async def test_gateway_bridge_candidate_is_accepted_by_run_owned_fake_listener_async() -> None:
    lease, allocation, network = await _bridge_case_async()
    binding = derive_gateway_bridge_binding(lease=lease, allocation=allocation, network=network, port=43123)
    sock = _socket(binding=binding)
    route = GatewayRoute(run_id=lease.run_id, model="offline-model", guest_token="g" * 40)
    with patch.object(uvicorn, "Server", return_value=_FakeServer()):
        listener = RunScopedModelGatewayListener(
            route=route,
            binding=binding,
            app=create_codex_responses_app(route=route, limits=GatewayLimits()),
            listening_socket=sock,
        )
    try:
        await listener.start_async()
        assert listener.is_locally_running and listener.base_url == "http://172.30.0.1:43123"
    finally:
        await listener.close_async()
        await lease.close_async()
    sock.close.assert_called_once()


async def test_gateway_bridge_reinspects_exact_network_before_listener_setup_async() -> None:
    lease, allocation, network = await _bridge_case_async()
    engine = MagicMock(spec=DockerEngineClient)
    engine.inspect_network_async = AsyncMock(return_value=network)
    try:
        binding = await acquire_gateway_bridge_binding_async(
            lease=lease, allocation=allocation, engine=engine, port=43123
        )
        assert binding.network_id == allocation.network_id
        engine.inspect_network_async.assert_awaited_once_with(allocation.network_id)
    finally:
        await lease.close_async()


async def test_gateway_bridge_engine_failure_is_not_replaced_by_cached_metadata_async() -> None:
    lease, allocation, _network = await _bridge_case_async()
    engine = MagicMock(spec=DockerEngineClient)
    engine.inspect_network_async = AsyncMock(side_effect=DockerEngineError("inert inspection unavailable"))
    try:
        with pytest.raises(DockerEngineError, match="inspection unavailable"):
            await acquire_gateway_bridge_binding_async(lease=lease, allocation=allocation, engine=engine, port=43123)
        engine.inspect_network_async.assert_awaited_once_with(allocation.network_id)
    finally:
        await lease.close_async()


@pytest.mark.parametrize(
    "defect",
    [
        "wrong_id",
        "external",
        "wrong_driver",
        "foreign_label",
        "extra_container",
        "many_subnets",
        "no_gateway",
        "public_gateway",
        "ipv6",
        "bad_port",
        "missing_ipam",
        "unexpected_ipam_field",
    ],
)
async def test_gateway_bridge_rejects_ambiguous_or_foreign_network_async(defect: str) -> None:
    lease, allocation, network = await _bridge_case_async()
    port = 43123
    if defect == "wrong_id":
        network["Id"] = "e" * 64
    elif defect == "external":
        network["Internal"] = False
    elif defect == "wrong_driver":
        network["Driver"] = "overlay"
    elif defect == "foreign_label":
        labels = network["Labels"]
        assert isinstance(labels, dict)
        labels["org.pyrit.native.run"] = "foreign-run"
    elif defect == "extra_container":
        containers = network["Containers"]
        assert isinstance(containers, dict)
        containers["f" * 64] = {}
    elif defect == "many_subnets":
        network["IPAM"]["Config"].append({"Subnet": "172.31.0.0/16", "Gateway": "172.31.0.1"})
    elif defect == "no_gateway":
        del network["IPAM"]["Config"][0]["Gateway"]
    elif defect == "public_gateway":
        network["IPAM"]["Config"][0] = {"Subnet": "8.8.8.0/24", "Gateway": "8.8.8.8"}
    elif defect == "ipv6":
        network["IPAM"]["Config"][0] = {"Subnet": "fd00::/64", "Gateway": "fd00::1"}
    elif defect == "bad_port":
        port = 0
    elif defect == "missing_ipam":
        del network["IPAM"]
    else:
        network["IPAM"]["Config"][0]["IPRange"] = "172.30.1.0/24"
    try:
        with pytest.raises(ValueError, match="(?i)gateway bridge|run-scoped gateway"):
            derive_gateway_bridge_binding(lease=lease, allocation=allocation, network=network, port=port)
    finally:
        await lease.close_async()


async def test_gateway_bridge_refuses_unacquired_lease_before_network_metadata_async() -> None:
    lease, _runner = make_lease(2)
    with pytest.raises(ValueError, match="ready run-owned lease"):
        derive_gateway_bridge_binding(
            lease=lease,
            allocation=object(),
            network={},
            port=43123,
        )
