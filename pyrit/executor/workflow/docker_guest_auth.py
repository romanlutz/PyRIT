# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run-scoped guest authentication, independent of host model credentials and evidence storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import SecretStr

from pyrit.prompt_target.gateway.responses_contract import GatewayRoute
from pyrit.prompt_target.native_cli_models import NativeCliProtocol


@dataclass(frozen=True, kw_only=True)
class DockerGuestAuth:
    """An independently generated guest-only token; no primary host credential is accepted here."""

    run_id: str
    model: str
    protocol: NativeCliProtocol
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
        if self.token.get_secret_value() in self.run_id or self.token.get_secret_value() in self.model:
            raise ValueError("The guest token must not appear in public route identifiers.")

    @classmethod
    def from_route(cls, *, route: GatewayRoute, protocol: NativeCliProtocol) -> DockerGuestAuth:
        """
        Bind the same host-created route used by the budget-capped gateway app.

        Returns:
            DockerGuestAuth: A redacted guest-only credential, never the host backend credential.
        """
        return cls(run_id=route.run_id, model=route.model, protocol=protocol, token=SecretStr(route.guest_token))

    def exec_environment(self) -> list[str]:
        """
        Materialize only the two approved exec environment entries for Engine request serialization.

        Returns:
            list[str]: Sensitive guest env values; do not log, persist or include in a process handle.
        """
        token = self.token.get_secret_value()
        if self.protocol is NativeCliProtocol.CODEX_EXEC_JSON:
            return [f"PYRIT_GUEST_MODEL_TOKEN={token}", f"PYRIT_RUN_ID={self.run_id}"]
        return [f"ANTHROPIC_AUTH_TOKEN={token}", f"ANTHROPIC_CUSTOM_HEADERS=X-PyRIT-Run-ID: {self.run_id}"]


def codex_gateway_config(*, model: str, base_url: str) -> str:
    """
    Render the exact credential-free user-level Codex provider configuration for image staging.

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
