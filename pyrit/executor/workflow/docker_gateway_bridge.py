# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Derive a candidate model-listener address from one owned Docker bridge."""

from __future__ import annotations

from ipaddress import IPv4Address, IPv4Network
from typing import TYPE_CHECKING

from pyrit.models.environment_lease import EnvironmentLeaseState
from pyrit.prompt_target.gateway.run_listener import GatewayBridgeBinding

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.executor.workflow.docker_compose import ComposeAllocation, DockerComposeEnvironmentLease


def derive_gateway_bridge_binding(
    *,
    lease: DockerComposeEnvironmentLease,
    allocation: ComposeAllocation,
    network: Mapping[str, object],
    port: int,
) -> GatewayBridgeBinding:
    """
    Bind the listener candidate to an Engine-inspected, still-owned challenge network.

    This checks daemon metadata, not the host interface/firewall or guest
    reachability. The caller must separately provision and verify its
    prebound socket before advertising a model route.

    Returns:
        GatewayBridgeBinding: Candidate private address tied to the current run.

    Raises:
        ValueError: If any identity, attachment, IPv4 gateway, or network policy is ambiguous.
    """
    if lease.snapshot().state is not EnvironmentLeaseState.READY:
        raise ValueError("A gateway bridge candidate needs an acquired, ready run-owned lease.")
    if (
        allocation.project_name != lease.project_name
        or not isinstance(network, dict)
        or network.get("Id") != allocation.network_id
        or network.get("Name") != allocation.project_name + "_challenge"
        or network.get("Driver") != "bridge"
        or network.get("Internal") is not True
        or network.get("EnableIPv6") is not False
        or network.get("Scope") != "local"
        or network.get("Options") not in ({}, None)
    ):
        raise ValueError("Gateway bridge inspection disagrees with the owned internal Compose allocation.")
    labels = network.get("Labels")
    if not isinstance(labels, dict) or any(
        labels.get(key) != value
        for key, value in {
            "org.pyrit.native.run": lease.run_id,
            "org.pyrit.native.lease": lease.lease_id,
            "com.docker.compose.project": allocation.project_name,
            "com.docker.compose.network": "challenge",
        }.items()
    ):
        raise ValueError("Gateway bridge labels do not identify the exact owned run and lease.")
    attached = network.get("Containers")
    if not isinstance(attached, dict) or set(attached) != {container_id for _, container_id in allocation.containers}:
        raise ValueError("Gateway bridge attachments differ from the currently owned services.")
    ipam = network.get("IPAM")
    if (
        not isinstance(ipam, dict)
        or ipam.get("Driver") != "default"
        or not isinstance(ipam.get("Config"), list)
        or len(ipam["Config"]) != 1
    ):
        raise ValueError("Gateway bridge has no single approved IPv4 IPAM allocation.")
    config = ipam["Config"][0]
    if not isinstance(config, dict) or set(config) != {"Subnet", "Gateway"}:
        raise ValueError("Gateway bridge subnet and gateway metadata are missing or ambiguous.")
    try:
        subnet = IPv4Network(config["Subnet"], strict=True)
        address = IPv4Address(config["Gateway"])
    except (ValueError, TypeError, KeyError) as error:
        raise ValueError("Gateway bridge has no canonical IPv4 subnet and gateway.") from error
    return GatewayBridgeBinding(
        run_id=lease.run_id,
        network_id=allocation.network_id,
        address=address,
        subnet=subnet,
        port=port,
    )
