# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""No-network checks of authenticated, bounded Inspect model forwarding."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pyrit.executor.benchmark.inspect_ghcp_gateway import GatewayResponse, ModelGateway

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _gateway(
    *,
    audit_path: Path,
    forward: Callable[[bytes], GatewayResponse],
    max_requests: int = 2,
    cache_policy: str = "reject",
) -> ModelGateway:
    return ModelGateway(
        run_id="run-1",
        token="guest-token-" + "x" * 32,
        model="approved-model",
        audit_path=audit_path,
        forward=forward,
        max_requests=max_requests,
        max_body_bytes=1024,
        max_response_bytes=1024,
        deadline_seconds=60,
        prompt_cache_key_policy=cache_policy,
    )


def _headers() -> dict[str, str]:
    return {
        "authorization": "Bearer guest-token-" + "x" * 32,
        "x-pyrit-run": "run-1",
        "x-pyrit-request-id": "sdk-request-1",
    }


def test_gateway_auth_quota_and_raw_rejections(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def forward(body: bytes) -> GatewayResponse:
        calls.append(body)
        return GatewayResponse(status=200, body=b'{"output":"real model bytes"}')

    audit = tmp_path / "audit.jsonl"
    gateway = _gateway(audit_path=audit, forward=forward, max_requests=1)
    body = b'{"model":"approved-model","input":"hello"}'
    assert gateway.process(path="/v1/responses", method="POST", headers={}, body=body).status == 401
    assert gateway.process(path="/v1/responses", method="POST", headers=_headers(), body=body).status == 200
    assert gateway.process(path="/v1/responses", method="POST", headers=_headers(), body=body).status == 429
    assert calls == [body]
    assert b"guest-token" not in audit.read_bytes()
    stored = [json.loads(line) for line in audit.read_text().splitlines()]
    assert [row["response_status"] for row in stored] == [200, 429]
    assert stored[0]["request_sha256"]
    assert stored[0]["source_request_id"] == "sdk-request-1"


def test_gateway_only_routes_approved_function_calls_and_explicit_cache_policy(tmp_path: Path) -> None:
    calls: list[bytes] = []

    def forward(body: bytes) -> GatewayResponse:
        calls.append(body)
        return GatewayResponse(status=200, body=b"{}")

    reject = _gateway(audit_path=tmp_path / "reject.jsonl", forward=forward)
    body = b'{"model":"approved-model","prompt_cache_key":"cross-run"}'
    assert reject.process(path="/v1/responses", method="POST", headers=_headers(), body=body).status == 400
    remote = b'{"model":"approved-model","tools":[{"type":"web_search"}]}'
    assert reject.process(path="/v1/responses", method="POST", headers=_headers(), body=remote).status == 400
    assert not calls
    omit = _gateway(audit_path=tmp_path / "omit.jsonl", forward=forward, cache_policy="omit_after_capture")
    assert omit.process(path="/v1/responses", method="POST", headers=_headers(), body=body).status == 200
    assert calls == [b'{"model":"approved-model"}']
    stored = json.loads((tmp_path / "omit.jsonl").read_text().splitlines()[0])
    assert stored["request_sha256"] != stored["forwarded_sha256"]
