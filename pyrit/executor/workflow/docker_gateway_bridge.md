# Candidate model-gateway bridge address

`derive_gateway_bridge_binding` takes an already-ready, run-owned Compose lease,
its verified allocation, a fresh Engine inspection of that same network, and
a selected listener port. It rejects foreign project/run/lease labels,
external or non-bridge networks, unknown attachments, IPv6 and ambiguous IPAM
configurations. It returns a `GatewayBridgeBinding` for the host-only
`RunScopedModelGatewayListener`, whose separately provided socket must already
be bound to that exact private address and port.
For runtime construction, call `acquire_gateway_bridge_binding_async` to fetch
the exact network ID again through the host-only Engine client before applying
these checks; never trust a candidate-supplied or cached inspection.

This is a **candidate address, not a connectivity or firewall proof**. Docker
IPAM metadata does not establish that the host can bind the address, that
only the owned agent container can reach it, or that model traffic will use
it. The trusted task binding must provision a protected socket, verify host
firewall policy and agent-side reachability for the exact run, and keep live
readiness blocked until those checks pass. Current tests inspect only fake
Compose and Engine-shaped metadata; they open no socket and run no container.
