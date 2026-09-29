# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-scoped, no-egress gateway in Inspect's separate model-bridge sandbox."""

from __future__ import annotations

import base64
import hashlib
import hmac
import http.client
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

if __package__:
    from pyrit.executor.benchmark.inspect_ghcp_token_file import read_scoped_token
else:
    from inspect_ghcp_token_file import read_scoped_token  # ty: ignore[unresolved-import]

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, kw_only=True)
class GatewayResponse:
    """An exact HTTP response body and its status from the approved bridge."""

    status: int
    body: bytes
    content_type: str = "application/json"


class ModelGateway:
    """Allow only bounded, authenticated Responses requests to Inspect's local proxy."""

    def __init__(
        self,
        *,
        run_id: str,
        token: str,
        model: str,
        audit_path: Path,
        forward: Callable[[bytes], GatewayResponse],
        max_requests: int,
        max_body_bytes: int,
        max_response_bytes: int,
        deadline_seconds: int,
        prompt_cache_key_policy: str = "reject",
    ) -> None:
        """
        Bind one token, model, capture file, and finite request budget.

        Raises:
            ValueError: If a scoped identity or bound is invalid.
        """
        if (
            not run_id
            or len(token) < 32
            or not model
            or max_requests < 1
            or max_body_bytes < 1
            or max_response_bytes < 1
            or deadline_seconds < 1
            or prompt_cache_key_policy not in {"reject", "omit_after_capture"}
        ):
            raise ValueError("Inspect model gateway requires a scoped token, model, and positive limits.")
        self.run_id = run_id
        self._token = token
        self._model = model
        self._audit_path = audit_path
        self._forward = forward
        self._max_requests = max_requests
        self._max_body_bytes = max_body_bytes
        self._max_response_bytes = max_response_bytes
        self._deadline = time.monotonic() + deadline_seconds
        self._prompt_cache_key_policy = prompt_cache_key_policy
        self._lock = threading.Lock()
        self._requests = 0
        self._audit_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._audit_path.touch(mode=0o600, exist_ok=False)
        self._audit_path.chmod(0o600)

    def process(self, *, path: str, method: str, headers: dict[str, str], body: bytes) -> GatewayResponse:
        """
        Capture both sides of a permitted model request, including rejections.

        Returns:
            GatewayResponse: The response sent back to the CLI.

        Raises:
            ValueError: If the audit cannot be written completely.
        """
        authorization = headers.get("authorization", "")
        if not hmac.compare_digest(authorization, f"Bearer {self._token}"):
            return self._error("unauthorized", 401)
        if not hmac.compare_digest(headers.get("x-pyrit-run", ""), self.run_id):
            return self._error("wrong_run", 403)
        request_id = headers.get("x-pyrit-request-id")
        if not isinstance(request_id, str) or not 0 < len(request_id) <= 128:
            return self._error("missing_source_request_id", 400)
        with self._lock:
            self._requests += 1
            sequence = self._requests
        error, forwarded = self._validate_request(path=path, method=method, body=body, sequence=sequence)
        if error:
            result = self._error(error, 429 if error == "model_quota_exceeded" else 400)
        else:
            try:
                result = self._forward(forwarded)
                if len(result.body) > self._max_response_bytes:
                    raise ValueError("Model response exceeds the approved raw-byte limit.")
            except ValueError:
                result = self._error("model_response_too_large", 502)
                error = "model_response_too_large"
            except (OSError, TimeoutError, http.client.HTTPException):
                result = self._error("model_bridge_unavailable", 502)
                error = "model_bridge_unavailable"
        self._record(
            sequence=sequence,
            request_id=request_id,
            request=body,
            forwarded=forwarded,
            response=result,
            error=error,
        )
        return result

    def _validate_request(self, *, path: str, method: str, body: bytes, sequence: int) -> tuple[str | None, bytes]:
        if method != "POST" or path != "/v1/responses":
            return "unsupported_model_route", b""
        if time.monotonic() > self._deadline or sequence > self._max_requests:
            return "model_quota_exceeded", b""
        if len(body) > self._max_body_bytes:
            return "model_request_too_large", b""
        try:
            payload = json.loads(body)
        except (UnicodeError, json.JSONDecodeError):
            return "invalid_model_json", b""
        if not isinstance(payload, dict) or payload.get("model") != self._model:
            return "unapproved_model", b""
        tools = payload.get("tools", [])
        if not isinstance(tools, list) or any(
            not isinstance(tool, dict) or tool.get("type") != "function" for tool in tools
        ):
            return "unapproved_remote_tool", b""
        if "prompt_cache_key" not in payload:
            return None, body
        if self._prompt_cache_key_policy == "reject":
            return "prompt_cache_key_not_approved", b""
        del payload["prompt_cache_key"]
        return None, json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def _record(
        self,
        *,
        sequence: int,
        request_id: str,
        request: bytes,
        forwarded: bytes,
        response: GatewayResponse,
        error: str | None,
    ) -> None:
        entry = {
            "sequence": sequence,
            "source_request_id": request_id,
            "run_id": self.run_id,
            "request_sha256": hashlib.sha256(request).hexdigest(),
            "request_base64": base64.b64encode(request).decode("ascii"),
            "forwarded_sha256": hashlib.sha256(forwarded).hexdigest(),
            "forwarded_base64": base64.b64encode(forwarded).decode("ascii"),
            "response_sha256": hashlib.sha256(response.body).hexdigest(),
            "response_base64": base64.b64encode(response.body).decode("ascii"),
            "response_status": response.status,
            "content_type": response.content_type,
            "error": error,
        }
        encoded = json.dumps(entry, separators=(",", ":"), sort_keys=True).encode("utf-8") + b"\n"
        with self._lock, self._audit_path.open("ab") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _error(code: str, status: int) -> GatewayResponse:
        return GatewayResponse(
            status=status,
            body=json.dumps({"error": {"code": code}}, separators=(",", ":")).encode("ascii"),
        )


def _forward_to_inspect(
    *, proxy_port: int, timeout_seconds: int, max_response_bytes: int, body: bytes
) -> GatewayResponse:
    connection = http.client.HTTPConnection("127.0.0.1", proxy_port, timeout=timeout_seconds)
    try:
        connection.request(
            "POST",
            "/v1/responses",
            body=body,
            headers={"Content-Type": "application/json", "Accept-Encoding": "identity"},
        )
        response = connection.getresponse()
        result = response.read(max_response_bytes + 1)
        return GatewayResponse(
            status=response.status,
            body=result,
            content_type=response.getheader("Content-Type", "application/json"),
        )
    finally:
        connection.close()


class _GatewayHandler(BaseHTTPRequestHandler):
    gateway: ModelGateway
    max_body_bytes: int

    def do_POST(self) -> None:
        """Dispatch one bounded model request."""
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > self.max_body_bytes:
            self._send(ModelGateway._error("invalid_body_length", 413))
            return
        body = self.rfile.read(length)
        result = self.gateway.process(
            path=self.path,
            method="POST",
            headers={key.lower(): value for key, value in self.headers.items()},
            body=body,
        )
        self._send(result)

    def do_GET(self) -> None:
        """Expose only an inert liveness check; forbid audit and model metadata."""
        if self.path == "/health":
            self._send(GatewayResponse(status=200, body=b'{"ready":true}'))
        else:
            self._send(ModelGateway._error("unsupported_model_route", 405))

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        """Avoid logging credentials or model data to the service's stderr."""

    def _send(self, result: GatewayResponse) -> None:
        self.send_response(result.status)
        self.send_header("Content-Type", result.content_type)
        self.send_header("Content-Length", str(len(result.body)))
        self.end_headers()
        self.wfile.write(result.body)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or not value:
        raise ValueError(f"Missing required model-gateway setting: {name}.")
    return value


def main() -> None:
    """
    Start an authenticated gateway, or read its private audit through Inspect exec_remote.

    Raises:
        ValueError: If a required configuration value or command is invalid.
    """
    if len(sys.argv) == 2 and sys.argv[1] == "--audit":
        audit_path = Path(_required_env("PYRIT_GATEWAY_AUDIT_PATH"))
        with audit_path.open("rb") as stream:
            sys.stdout.buffer.write(stream.read())
        return
    if len(sys.argv) != 1:
        raise ValueError("Unsupported model-gateway command.")
    port = int(_required_env("PYRIT_GATEWAY_PORT"))
    proxy_port = int(_required_env("PYRIT_INSPECT_PROXY_PORT"))
    timeout_seconds = int(_required_env("PYRIT_GATEWAY_TIMEOUT"))
    max_body_bytes = int(_required_env("PYRIT_GATEWAY_MAX_BODY"))
    max_response_bytes = int(_required_env("PYRIT_GATEWAY_MAX_RESPONSE"))
    run_id = _required_env("PYRIT_GATEWAY_RUN_ID")
    gateway = ModelGateway(
        run_id=run_id,
        token=read_scoped_token(
            run_id=run_id,
            token_file=_required_env("PYRIT_GATEWAY_TOKEN_FILE"),
        ),
        model=_required_env("PYRIT_GATEWAY_MODEL"),
        audit_path=Path(_required_env("PYRIT_GATEWAY_AUDIT_PATH")),
        forward=lambda body: _forward_to_inspect(
            proxy_port=proxy_port,
            timeout_seconds=timeout_seconds,
            max_response_bytes=max_response_bytes,
            body=body,
        ),
        max_requests=int(_required_env("PYRIT_GATEWAY_MAX_REQUESTS")),
        max_body_bytes=max_body_bytes,
        max_response_bytes=max_response_bytes,
        deadline_seconds=int(_required_env("PYRIT_GATEWAY_DEADLINE")),
        prompt_cache_key_policy=_required_env("PYRIT_PROMPT_CACHE_KEY_POLICY"),
    )
    _GatewayHandler.gateway = gateway
    _GatewayHandler.max_body_bytes = max_body_bytes
    with ThreadingHTTPServer(("0.0.0.0", port), _GatewayHandler) as server:
        sys.stdout.write(json.dumps({"kind": "ready", "pid": os.getpid()}) + "\n")
        sys.stdout.flush()
        server.serve_forever(poll_interval=0.2)


if __name__ == "__main__":
    main()
