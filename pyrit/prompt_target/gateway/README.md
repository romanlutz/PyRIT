# Host-only coding-agent model transports

These internal transports relay a **model** wire format. They do not manage the CLI's conversation, execute tools, provide a sandbox, or expose a provider key to guest code. Each app instance is bound to one run and one exact model ID.

## Codex: OpenAI Responses

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

## Claude Code: Anthropic Messages

Anthropic [documents](https://code.claude.com/docs/en/llm-gateway-protocol#api-formats) the Anthropic Messages route for `claude -p`: configure `ANTHROPIC_BASE_URL` as the gateway's base URL (before `/v1`) and `ANTHROPIC_AUTH_TOKEN` as an **independently generated, per-run guest token**. Set `ANTHROPIC_CUSTOM_HEADERS` to `X-PyRIT-Run-ID: <run_id>` so the gateway can check the routing identity as well. Claude Code calls `POST /v1/messages?beta=true`; the gateway also accepts the path with no query, and forwards the documented `beta=true` query unchanged. [Non-interactive mode](https://code.claude.com/docs/en/headless) documents `claude -p`. Its optional `--bare` mode changes startup and credential behavior; selecting that mode for a sandbox requires separate validation. These are configuration instructions, not a command to run from this package.

Create a separate Anthropic-format model backend **on the host**:

```python
from pyrit.prompt_target.gateway.claude_messages import create_claude_messages_app
from pyrit.prompt_target.gateway.httpx_messages_backend import HttpxMessagesBackend
from pyrit.prompt_target.gateway.messages_contract import MessagesCapabilities
from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute

route = GatewayRoute(run_id=run_id, model=pinned_claude_model_id, guest_token=guest_token)
limits = GatewayLimits(max_output_tokens_per_request=verified_model_token_limit)
capabilities = MessagesCapabilities(
    streaming=True,
    tool_use=True,
    thinking=verified_model_thinking,
    prompt_caching=verified_model_prompt_caching,
    effort=verified_model_effort,
    allowed_beta_values=frozenset(verified_upstream_beta_values),
)
backend = HttpxMessagesBackend(
    route=route,
    endpoint=pinned_https_anthropic_messages_url,
    host_api_key=host_only_provider_key,
    client=host_owned_httpx_async_client,
    limits=limits,
    capabilities=capabilities,
    timeout_seconds=upstream_deadline_seconds,
)
app = create_claude_messages_app(
    route=route,
    limits=limits,
    backend=backend,
    observation_callback=host_observation_callback,
)
```

The host owns the injected `httpx.AsyncClient` and closes it after the run. The endpoint must be a pinned HTTPS Anthropic-format `/v1/messages` URL, with no query, redirect, userinfo, or provider-format translation. The backend sends a host-only API key in the upstream `x-api-key` header; it never forwards the guest's Authorization header, routing header, or arbitrary URLs. It forwards the accepted original JSON bytes and the verified `anthropic-version` and `anthropic-beta` header values without rewriting them, and relays genuine provider SSE frames, keep-alive pings, usage, `tool_use`, and CLI-supplied `tool_result` history. The CLI owns the tool loop. Real provider HTTP error bodies and retry headers are forwarded unchanged **only after** validating the Anthropic error envelope and checking for a credential echo; network, timeout, malformed, and truncated upstream responses become separately labeled host-generated errors.

The gateway bounds per-run request count, incoming bytes, and reserved output tokens, plus per-request provider response bytes and a wall-clock deadline. It never lowers `max_tokens` silently: a request over budget fails before upstream I/O. The backend requires uncompressed provider responses so it can enforce byte limits on exact wire frames. The observer receives original request/response bodies and SSE frames plus safe, selected headers, or `GatewayFrameKind.GATEWAY_ERROR` for a host-generated failure; it is not a DB recorder. Even if that recorder fails after an upstream error, the client receives the original failure rather than a fabricated success. Host credentials are checked in original and decoded provider bodies and held SSE text fragments before being observed or sent to the guest; this is not a general DLP filter for other sensitive material or arbitrary encodings.

The scope is deliberately narrower than the full evolving Claude Code API: unknown betas/headers/body fields, server-side tools, tool search, images/documents, experimental context management, structured outputs, and one-hour cache TTL fail explicitly, not silently. Only five-minute cache breakpoints, client text tools and results, signed thinking when verified, and documented output effort are accepted. The [gateway compatibility guide](https://code.claude.com/docs/en/llm-gateway-protocol#forward-as-open-lists) recommends open-list forwarding for unrestricted compatibility; this strict subset instead requires host verification and will reject new Claude Code features until they are reviewed. `POST /v1/messages/count_tokens` is intentionally absent, which the [documentation](https://code.claude.com/docs/en/llm-gateway-protocol#optional-endpoints-and-startup-traffic) says falls back to a character estimate. The best-effort `HEAD /api/hello` and optional model-discovery endpoint are also absent.

Only fake in-process ASGI and HTTP transports test this path. It makes **no live Claude Code or provider compatibility claim**. A parent integration must provision the isolated run, mount the ASGI app, restrict guest egress to the run-scoped gateway, configure the guest token and route header without a provider key, verify the selected upstream model's feature flags, and persist observations separately.
