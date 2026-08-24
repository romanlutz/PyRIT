---
applyTo: "pyrit/prompt_target/**"
---

# Prompt Target Development Guidelines

**Responsibility**: A prompt target is "the thing we're sending the prompt to" — often an LLM, but it can be any endpoint (e.g. a storage account for cross-domain prompt injection). Targets use `message_normalizer` together with `TargetConfiguration` to transform `Message`s into the format the target supports.

**Does not own** (see [framework.md](../../doc/code/framework.md)): what to send or what to do with the response. A target sends a prepared `Message` and returns a response; it must not convert prompts (converters), score (scorers), or manage the conversation / decide the next turn (attacks). Flag such bleed in review.

## Base Class Contract

All targets MUST inherit from ``PromptTarget`` (or one of its public
subclasses such as ``OpenAITarget`` / ``HTTPTarget``) and implement
``_send_prompt_to_target_async``:

```python
from pyrit.prompt_target import PromptTarget, ProviderAttempt


class MyTarget(PromptTarget):
    def __init__(
        self,
        *,
        endpoint: str,
        api_key: str,
        max_requests_per_minute: int | None = None,
        custom_configuration: TargetConfiguration | None = None,
    ) -> None:
        super().__init__(
            endpoint=endpoint,
            max_requests_per_minute=max_requests_per_minute,
            custom_configuration=custom_configuration,
        )
        self._api_key = api_key

    async def _send_prompt_to_target_async(
        self,
        *,
        normalized_conversation: list[Message],
        provider_attempt: ProviderAttempt,
    ) -> list[Message]:
        body = await self._build_request_body_async(
            normalized_conversation=normalized_conversation,
        )
        response = await provider_attempt.run_async(
            operation=lambda: self._client.send_async(body=body),
        )
        return self._parse_response(response=response)
```

``send_prompt_async`` (the public entry point) is ``@final`` and MUST NOT
be overridden. Override ``_send_prompt_to_target_async`` instead.

## Provider-attempt boundary

``PromptTarget.send_prompt_async`` constructs a one-shot ``ProviderAttempt`` for
each concrete target invocation. It owns the shared requests-per-minute wait and
the caller lifecycle notification without exposing attack-owned history state to
the target.

Provider-backed targets MUST call ``provider_attempt.start_async()`` immediately
before their first irreversible operation that may deliver the current request,
or wrap that operation with ``provider_attempt.run_async(...)``. Perform
cancellable setup first, including validation, body construction, authentication,
local file reads, model loading and tokenization, connection/session setup,
history restoration, and browser DOM staging. Repeated starts are idempotent, so
retries and multi-operation sends reuse the same token. Providerless targets do
not start it.

For one compatibility window, ``PromptTarget`` inspects the most-derived
protected method signature. Existing overrides that do not accept
``provider_attempt`` are called with their legacy signature after the token is
started conservatively at method entry. Migrated built-ins accept
``provider_attempt: ProviderAttempt | None = None`` only so a legacy subclass can
call ``super()._send_prompt_to_target_async(normalized_conversation=...)`` without
a second wait or mark. New targets must require the token. Direct calls to
``_send_prompt_to_target_async`` are unsupported.

``limit_requests_per_minute`` remains exported temporarily as a deprecated no-op
decorator. Rate limiting is applied by ``ProviderAttempt`` when a target is
invoked through ``send_prompt_async``.

## Keyword-only ``__init__`` is enforced

Every ``PromptTarget`` subclass MUST make all ``__init__`` parameters
keyword-only (i.e., place ``*`` as the first parameter after ``self``).
``PromptTarget.__init_subclass__`` validates this at class-definition time
via ``enforce_keyword_only_init`` and raises ``TypeError`` on violations.

The check is satisfied by either of:

```python
def __init__(self, *, endpoint: str, api_key: str) -> None: ...

def __init__(self, *args: Any, **kwargs: Any) -> None: ...  # *args after self
```

It rejects:

```python
def __init__(self, endpoint: str, api_key: str) -> None: ...    # missing *
```

> [!NOTE]
> ``PromptTarget.__init__`` *itself* is now keyword-only as well (``*`` after
> ``self``), so both the base class and its subclasses enforce the same
> contract.

## Configuration and Capabilities

- Set ``_DEFAULT_CONFIGURATION`` at the class level when your target's
  capabilities differ from the base defaults (multi-turn support, non-text
  modalities, JSON-mode responses, etc.).
- Accept ``custom_configuration: TargetConfiguration | None = None`` in
  ``__init__`` and forward it to ``super().__init__`` so callers can
  override capabilities per-instance (this is required for HTTP / Playwright
  targets whose capabilities depend on deployment configuration).

## Identifiable Pattern

All targets inherit ``Identifiable``. Override ``_build_identifier()`` to
include parameters that affect target behaviour:

```python
def _build_identifier(self) -> ComponentIdentifier:
    return self._create_identifier(
        params={"endpoint": self._endpoint, "model_name": self._model_name},
    )
```

Include: endpoint, model_name, deployment identifiers, custom headers that
affect routing.
Exclude: API keys, retry counts, logging config, timeouts.

## Exports

New targets MUST be added to ``pyrit/prompt_target/__init__.py`` — both
the import and the ``__all__`` list.
