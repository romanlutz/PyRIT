# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-scoped guest authentication, independent of host model credentials and evidence storage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from pydantic import SecretStr

from pyrit.prompt_target.gateway.responses_contract import GatewayRoute
from pyrit.prompt_target.native_cli_models import NativeCliProtocol


@dataclass(frozen=True, kw_only=True)
class DockerGuestAuth:
    """An independently generated guest-only token; no primary host credential is accepted here."""

    run_id: str
    model: str
    protocol: NativeCliProtocol
    gateway_endpoint: str
    token: SecretStr = field(repr=False)

    def __post_init__(self) -> None:
        """
        Reuse gateway route validation without retaining an unredacted route object.

        Raises:
            ValueError: If the token, protocol or routing identity is invalid.
        """
        if not isinstance(self.token, SecretStr) or not isinstance(self.protocol, NativeCliProtocol):
            raise ValueError("Guest authentication requires a redacted token and supported CLI protocol.")
        GatewayRoute(run_id=self.run_id, model=self.model, guest_token=self.token.get_secret_value())
        if not isinstance(self.gateway_endpoint, str) or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127 for char in self.gateway_endpoint
        ):
            raise ValueError("The guest gateway must be an explicit nonsecret URL.")
        parts = urlsplit(self.gateway_endpoint)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.query
            or parts.fragment
        ):
            raise ValueError("The guest gateway URL cannot contain credentials, queries or fragments.")
        _ = parts.port
        if any(self.token.get_secret_value() in value for value in (self.run_id, self.model, self.gateway_endpoint)):
            raise ValueError("The guest token must not appear in public route identifiers.")

    @classmethod
    def from_route(cls, *, route: GatewayRoute, protocol: NativeCliProtocol, gateway_endpoint: str) -> DockerGuestAuth:
        """
        Bind the same host-created route used by the budget-capped gateway app.

        Returns:
            DockerGuestAuth: A redacted guest-only credential, never the host backend credential.
        """
        return cls(
            run_id=route.run_id,
            model=route.model,
            protocol=protocol,
            gateway_endpoint=gateway_endpoint,
            token=SecretStr(route.guest_token),
        )

    def exec_environment(self) -> list[str]:
        """
        Materialize only the approved per-run exec environment for Engine request serialization.

        Returns:
            list[str]: Sensitive guest env values; do not log, persist or include in a process handle.
        """
        token = self.token.get_secret_value()
        if self.protocol is NativeCliProtocol.CODEX_EXEC_JSON:
            return [f"PYRIT_GUEST_MODEL_TOKEN={token}", f"PYRIT_RUN_ID={self.run_id}"]
        return [
            f"ANTHROPIC_AUTH_TOKEN={token}",
            f"ANTHROPIC_CUSTOM_HEADERS=X-PyRIT-Run-ID: {self.run_id}",
            f"ANTHROPIC_BASE_URL={self.gateway_endpoint}",
            f"ANTHROPIC_MODEL={self.model}",
        ]


def codex_gateway_config(*, model: str, base_url: str) -> str:
    """
    Render the exact credential-free per-run Codex configuration for ephemeral staging.

    Returns:
        str: TOML that uses only the two run-scoped exec variables for guest authentication.
    """
    return (
        f"model = {json.dumps(model)}\n"
        'model_provider = "pyrit_gateway"\n\n'
        "[model_providers.pyrit_gateway]\n"
        'name = "PyRIT run gateway"\n'
        f"base_url = {json.dumps(base_url)}\n"
        'wire_api = "responses"\n'
        'env_key = "PYRIT_GUEST_MODEL_TOKEN"\n'
        "requires_openai_auth = false\n"
        'env_http_headers = { "X-PyRIT-Run-ID" = "PYRIT_RUN_ID" }\n'
    )


def codex_gateway_template_sha256() -> str:
    """
    Hash the static template contract with URL/model placeholders, never a particular run route.

    Returns:
        str: Image-appropriate template pin, distinct from the rendered per-run config SHA256.
    """
    return hashlib.sha256(codex_gateway_config(model="<model>", base_url="<gateway-url>").encode("utf-8")).hexdigest()
