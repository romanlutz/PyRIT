# Host-only Codex Responses transport

`create_codex_responses_app` exposes only `POST /v1/responses` for one run and one model alias. Codex's custom provider uses `wire_api = "responses"`, a `base_url` ending in `/v1`, an ephemeral **guest-only** bearer token, and an `X-PyRIT-Run-ID` header. Do not put the upstream provider credential in the sandbox.

The host can supply `HttpxResponsesBackend` as the gateway's `ModelOnlyResponsesBackend`:

```python
from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.httpx_responses_backend import HttpxResponsesBackend
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute

route = GatewayRoute(run_id=run_id, model=model_alias, guest_token=guest_token)
limits = GatewayLimits()
backend = HttpxResponsesBackend(
    route=route,
    endpoint=pinned_https_responses_url,
    auth_token=host_provider_token,
    client=host_owned_httpx_async_client,
    limits=limits,
    capabilities=verified_provider_capabilities,
)
app = create_codex_responses_app(
    route=route,
    limits=limits,
    backend=backend,
    observation_callback=host_observation_callback,
)
```

Generate `guest_token` independently for each run. Supply `host_provider_token` and the pinned **HTTPS** `/responses` URL only on the host; the two tokens must differ. The caller owns the injected `httpx.AsyncClient` and closes it when the run ends. Pass an explicitly verified `BackendCapabilities` value; do not assume the upstream model supports tools or reasoning. Neither the gateway nor this backend executes tools: Codex sends tool results in subsequent model requests.

The backend enforces the gateway's request/response byte ceilings and a bounded upstream deadline. Set `timeout_seconds` below `limits.timeout_seconds` if upstream timeouts must be distinguished from the gateway's outer deadline. It neither follows redirects nor accepts guest-supplied endpoints or headers. It preserves complete upstream JSON/SSE frames and never fabricates `[DONE]`. Malformed, truncated, oversized, or credential-bearing responses fail closed; possible credential prefixes in streamed text are held until safe to forward. This guards the configured bearer token in raw and decoded JSON/text, not arbitrary encodings or other sensitive data. The model must report usage within the reserved token limit. The backend rejects endpoint query parameters (including Azure `api-version`); use a compatible pinned endpoint or implement a separate reviewed host-owned configuration.

Upstream HTTP, network, payload/stream, and timeout failures now return sanitized `upstream_http_error`, `upstream_network_error`, `upstream_stream_error`, and `upstream_timeout` codes (`502` or `504`). This intentionally replaces the previous generic `backend_failed` code **only for these typed backend failures**. `GatewayFrameKind.GATEWAY_ERROR` observations identify host-generated error frames separately from raw provider frames; no upstream body, URL, header, or host credential is sent to the guest or callback. Unsupported Responses features still fail explicitly at the gateway.

This transport is tested only with injected fake HTTP and ASGI transports. No live model, CLI, container, or network compatibility is claimed. The parent workflow must still mount the ASGI app in the isolated run, provide host-only credentials and a run-scoped listener, and wire the observation callback to persistence.
