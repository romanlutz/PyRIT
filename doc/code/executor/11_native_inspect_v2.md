# Caller-owned native Inspect v2 wiring

`InspectNativeTaskBridgeV2` is an **additive, source-prepared facade**, separate
from the [offline v1 bridge](10_native_inspect.md) and the CTF example. It uses the
same native Task/Solver/Generate/tool path while accepting the explicitly
versioned `strict-submission-v2` report contract.

The source supports a caller-owned model client, credentials, task tools and
environment hooks. **Only inert, mocked-I/O invocation tests have been run for
this facade.** Validating a `real` provenance object does not authenticate a
deployment, execute an artifact, establish service readiness, or authorize a run.
There are no bundled endpoints, credentials, evaluator clients, datasets,
containers, VM providers, or infrastructure defaults.

## Exact public API

The new facade uses a direct import so the accepted v1 public package remains
unchanged:

```python
from pyrit.executor.benchmark.inspect_native_task_v2 import InspectNativeTaskBridgeV2
from pyrit.executor.benchmark.submission.hooks_v2 import (
    SubmissionEnvironmentHooksV2,
    SubmissionLimitsV2,
)

bridge = InspectNativeTaskBridgeV2(
    target_factory=target_context,
    read_report=binding.read_report,
    directory=new_run_directory,
    environment=SubmissionEnvironmentHooksV2(
        audit_async=audit_async,
        cleanup_async=cleanup_async,
    ),
    limits=SubmissionLimitsV2(
        max_requests=8,
        max_tool_calls=8,
        max_messages=20,
        max_tokens=4096,
        episode_timeout_seconds=180,
    ),
    max_tool_output_bytes=16384,
)
result = await bridge.execute_async(
    task=original_task,
    sample_id=selected_sample_id,
    native_scorer="original_numeric_scorer",
    expectation=None,
)
```

All dependencies above are supplied explicitly. `directory` must be a fresh owned
directory. `read_report()` is a non-I/O snapshot of the binding-owned ledger.
The task has one selected sample and keeps its original setup, solver, prompts,
tools, Store, and native numeric scorer. The task must be model-free
(`Task(model=None)` without model roles); the bridge explicitly calls
`eval_async(model=None)` and does not resolve an ambient Inspect model.

`target_context` implements the shared `SubmissionTargetFactoryV2` protocol:

```python
@asynccontextmanager
async def target_context(*, tools, request_hook, response_hook):
    async with owned_client_and_credentials(...) as owned:
        yield OpenAIResponseTarget(
            endpoint=owned.endpoint,
            model_name=owned.model,
            api_key=owned.token_provider,
            auto_execute_tools=False,
            extra_body_parameters={
                "tools": tools,
                "parallel_tool_calls": False,
                "store": False,
            },
            httpx_client_kwargs={
                "http_client": owned.http_client,
                "max_retries": 0,
            },
        )
```

This is an ownership sketch, not a preconfigured credential/client implementation.
The caller attaches `request_hook` and `response_hook` to the owned HTTPX client's
respective event-hook lists. In tests, that client uses `httpx.MockTransport`.
The context opens only after the native solver has installed its tool schemas.
The model identity comes from the returned target's public identifier, never an
Inspect model replacement or a guessed deployment. The context must remain alive
through generation and must close its client/credential resources on exit.

The hooks admit exactly one actual HTTP request per explicit generation. They
reject hidden SDK retries, uncorrelated requests, and model/schema/protocol
substitution before another transport dispatch. A successful generation that
omits either hook is rejected. Bodies, real call identities, and explicitly
reported usage are retained; request headers and credentials are not logged.
`max_response_bytes` checks the response body delivered to the hook and retained
provider-response evidence. It is not a promise about allocations inside a
caller-supplied HTTP transport.

`audit_async()` returns JSON-safe caller environment evidence.
It has its own five-second owned deadline, separate from the episode bound.
Expiry stops before factory/model dispatch, retains interruption diagnostics,
and attempts caller cleanup once. Cancelling the local audit await is not
evidence that an external resource or audit operation stopped.
`cleanup_async()` returns `SubmissionCleanupStatus`; unknown or failed cleanup
does not become a clean run. These hooks have no implicit sandbox, network,
Hyper-V, or cancellation implementation. Client-context failures and environment
errors remain separate lifecycle evidence. Local cancellation never proves that
remote evaluation work stopped.

## Provenance is a contract, not a flag override

The native-owned shared models validate these exact pairs:

| Contract | Mode | Simulated | Label |
| --- | --- | --- | --- |
| `strict-submission-v2` | `offline` | `true` | `OFFLINE/SIMULATED` |
| `strict-submission-v2` | `real` | `false` | `REAL/UNSIMULATED` |

The pair must match across the binding report, native log metadata, event writer,
retained outer report, and score metadata. It cannot change after binding. No
temporary v1/offline report is constructed for a real v2 report. V1's fixed
offline literals and validation remain unchanged.

All facade invocation fixtures use `offline/true`. Real-pair tests only construct
and validate models/facade bindings, without creating a target, authenticating,
making requests, or creating a run directory. The label records declared
provenance, not evidence that a dispatch happened.

## Preserved strict semantics

Original string callback results and recoverable `ToolError` feedback use the
native tool execution pipeline. There is no CTF stdout/stderr wrapper, alternate
solver, fake final answer, or fabricated continuation input. A no-input native
Generate re-entry uses the public PyRIT conversation-continuation method.

The report retains ordered submission IDs and artifact hashes supplied by the
binding. `.75` followed by `.25` selects `.25`; a missing artifact causes no
evaluator dispatch and does not erase a prior valid observation. Full success
stops before another generation. Unknown outcomes, partial evidence, malformed
grades, lifecycle failures, and uncertain cleanup cannot publish a clean score.
The facade does not implement or retry the private evaluator policy.

Limits are distinct: provider generations, tool calls, authentic message
envelopes, and actual input-plus-output token usage. Missing usage required for a
token bound is an explicit incomplete result, not zero. Native task message,
turn, supported token, and time bounds are also enforced by the reused Generate
bridge. The final allowed generation's genuine tool result is retained when
tool/message capacity remains.

`max_tool_output_bytes` retains the v1 explicit UTF-8 contract. Below the bound,
feedback must be byte-exact. Oversized or rewritten feedback is retained as
acquired evidence but stops the run as incomplete rather than forwarding a
truncation notice with a clean grade.

## Evidence, cancellation and publication

The original typed `.eval` log is materialized through public Inspect readers.
The selected native numeric grade must equal the validated latest binding grade.
Final model text is optional presentation, not the scoring anchor.

The shared `SubmissionEvidenceWriterV2` retains a canonical
`RetainedSubmissionReportV2`; the read-only `SubmissionReportScorerV2` persists a
`ContentEntryScorable` with the exact report digest and truthful provenance.
Replay reads stored content and never invokes the model, native tools, or
evaluator. The report records actual provider/generation/message counts,
observed usage, target identity, caller audit, cleanup status, and lifecycle
errors.
Every admitted provider request reserves an unknown usage entry until its
response hook reports usage. If a later request fails or is cancelled before
that response, earlier known entries remain evidence but aggregate token usage
is unknown rather than their misleading partial sum or a fabricated zero.

The tested publication path requires the unchanged synchronous `SQLiteMemory`
instance. All awaited native-log checks, file retention, client-context exit,
and environment cleanup precede the final text-report score transaction.
That SQLite score commit is the publication boundary, followed immediately by
`last_result` assignment and return without another await. Prepared files or
manifests are not commit receipts.

Caller or callback cancellation is re-raised after bounded retention;
`bridge.last_result` exposes an acquired undetermined report when publication
succeeded. Actual returned feedback is captured before a later cancellation can
erase it. Native log/sample/evaluation IDs are retained only when observed and
verified. Unknown remote disposition is not rewritten as cancelled.
Late cancellation of result delivery cannot roll back an already committed
score. This is not cross-store atomicity or a claim about other storage backends.

## Compatibility and validation

This version keeps the reviewed text/string, sequential-tool, fixed-schema,
automatic-tool-choice, append-only-history constraints. It rejects unsupported
native model configuration and tool-result protocols rather than silently
adapting them. Existing v1 and CTF code paths remain separate.

```powershell
uv run --no-sync pytest -q tests\unit\executor\benchmark\test_inspect_native_task_v2.py `
  tests\unit\score\test_submission_report_scorer_v2.py
```

These tests run actual Inspect/PyRIT code with newly authored inert callbacks and
mocked HTTP only. No result here establishes real deployment, evaluator-service,
build-container, or VM readiness.
