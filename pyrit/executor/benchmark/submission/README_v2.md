# V2 caller-owned submission wiring

V2 is an additive source interface for a configured caller-owned setup. It does
not provide a model deployment, credentials, evaluator client, artifact builder,
sandbox, VM pool, or resource manager. Creating these types does not authorize
execution. The implementation has only been exercised here with inert fixtures
and `httpx.MockTransport`; no real v2 provider or evaluator run is claimed.

The accepted v1 `NativeSubmissionRunner.run_offline_async` and
`strict-submission-v1` remain unchanged. V1 still requires `MockTransport` and
offline/simulated reports. V2 uses separate module paths and retains the v1
attempt-value types, lifecycle coherence rules, exact string tool semantics and
content-anchored SQLite publication contract. It deliberately does not refactor
the frozen v1 coordinator.

## Entry points and ownership

```python
from pyrit.executor.benchmark.submission.runner_v2 import NativeSubmissionRunnerV2
from pyrit.executor.benchmark.submission.hooks_v2 import (
    SubmissionEnvironmentHooksV2,
    SubmissionLimitsV2,
    SubmissionTargetFactoryV2,
)
from pyrit.models.submission_v2 import (
    StrictSubmissionReportV2,
    RetainedSubmissionReportV2,
)
from pyrit.score.float_scale.submission_report_scorer_v2 import SubmissionReportScorerV2
```

Construction:

```python
runner = NativeSubmissionRunnerV2(
    hooks=caller_submission_hooks,
    directory=caller_evidence_directory,
    limits=SubmissionLimitsV2(
        max_requests=12,
        max_tool_calls=12,
        max_messages=30,
        max_tokens=10_000,
        episode_timeout_seconds=30,
    ),
)
result = await runner.run_with_target_factory_async(
    seed=caller_seed,
    system_prompt=caller_system_prompt,
    target_factory=caller_target_factory,
    environment=caller_environment_hooks,
)
```

`SubmissionHooks` and `SubmissionTool` are reused from `submission.hooks`.
The binding still owns artifact bytes/digests, one evaluator call per submission,
ordered submission identity, exact returned strings, recoverable-error mapping,
raw acquired results and the latest-valid selection policy. No private tool
names, arguments, categories or service formats are built into the public runner.

The target factory implements this keyword-only contract:

```python
def caller_target_factory(
    *,
    tools: list[dict],
    request_hook,
    response_hook,
) -> AsyncContextManager[OpenAIResponseTarget]: ...
```

It must yield an actual `OpenAIResponseTarget(auto_execute_tools=False)` from an
async context manager. The factory supplies the explicit model/deployment,
endpoint and credential through public constructor arguments and owns their
entire lifetime. Caller-owned refreshing authentication remains inside this
factory; the runner never creates credentials or selects an account or endpoint.
The target's public identifier is retained as model provenance.

Attach the supplied hooks to the caller's HTTPX client:

```python
event_hooks = {"request": [request_hook], "response": [response_hook]}
```

Pass that client through `httpx_client_kwargs={"http_client": client,
"max_retries": 0}`. Configure the supplied tool definitions unchanged and set
`parallel_tool_calls=False`. Missing or bypassed hooks fail explicitly. The
request hook admits at most one provider request per deliberate generation and
rejects implicit retries before another transport dispatch. This also prevents a
misconfigured target retry from silently consuming a second request. An observed
request boundary does not prove the server accepted the request.

Only bodies, model identity, status and correlation are retained, never request
headers or bearer tokens. Tool and response content can still be sensitive;
storage location and retention are the caller's responsibility. Do not put
private evidence or configuration into public source control.

`SubmissionEnvironmentHooksV2` requires both callbacks:

| Callback | Contract |
| --- | --- |
| `audit_async()` | Return JSON-safe caller audit evidence before target entry |
| `cleanup_async()` | Return an explicit `SubmissionCleanupStatus` |

There is no default environment. A no-resource offline fixture must explicitly
return `NOT_REQUIRED`. Audit has its own owned, cancellation-safe five-second
deadline within the episode limit; expiry or caller cancellation blocks target
factory entry. Cleanup is attempted once after target-context exit, even on audit,
acquisition or generation failure. Each lifecycle operation has its own bounded
five-second grace. Failure or unknown cleanup prevents a clean score.
These are local lifecycle observations, not remote query/cancel/idempotency
guarantees. Ending a local audit wait does not prove that any external work stopped.
The caller hooks and factory are trusted Python code, not a sandbox.

## Truthful versioned provenance

Every binding report must declare `contract_version="strict-submission-v2"` and
exactly one pair:

| Mode | Simulated | Retained label |
| --- | --- | --- |
| `offline` | `true` | `OFFLINE/SIMULATED` |
| `real` | `false` | `REAL/UNSIMULATED` |

The initial binding report is validated before any factory, audit or cleanup
callback is invoked. This validated pair determines the writer's fixed
provenance. It cannot change during the run, and the outer schema-version-2
report must agree. Labels declare the configured path; they are not proof that a
model call, artifact execution, or remote job happened. Those facts require
actual acquired evidence. No real report is relabeled as simulated.

V2 keeps the exact v1 ledger fields and coherent dispatch, acceptance and
disposition semantics. Optional receipt IDs are not manufactured. A known
completed result can be valid without a separately observed acknowledgement.
Unknown outcome, partial required evidence or uncertain cleanup remains
incomplete. A prior valid grade is retained but not projected as a clean result.

## Boundaries, continuation and budgets

The native executor owns the sequential task/tool loop. The real Responses target
performs one generation at a time. No-tool output continues from authentic
persisted history through the public continuation API, without an inserted prompt,
duplicate message or converter replay. Full success and incomplete binding states
stop before another generation. The last actual tool result is persisted without
fabricating an assistant final.

`SubmissionLimitsV2` keeps separate bounds:

| Field | Count |
| --- | --- |
| `max_requests` | Explicit generations and admitted provider requests |
| `max_tool_calls` | Actual injected-tool invocations |
| `max_messages` | Authentic conversation message envelopes, including system/user/assistant/tool |
| `max_tokens` | Sum of actual provider input and output token usage |

Message and token limits are optional, not aliases for request limits. Missing or
malformed token usage is an explicit failure when a token bound is configured,
never a fabricated zero. Without a token bound, missing usage remains unknown.
Every generation exit also finalizes usage for its admitted request. Transport
failure, cancellation or an interrupted response hook adds an unknown entry when
no usable usage was acquired. Earlier observed usage remains retained, but the
aggregate total is unknown rather than a misleading known prefix. Blocked
pre-transport retries do not add request or usage entries. Request admission is
not a billing or server-acceptance claim.
Initial system/user messages count toward the message bound. Each allowed
generation's tool/result finishes before the next healthy budget check.

Healthy between-operation exhaustion records `termination_reason=budget` and the
specific `termination_limit`, then projects the latest valid result, including
zero. No valid submission stays `no_submission`. Cancellation, in-flight timeout,
uncertain remote state or required-evidence gaps do not become clean budget stops.

## Durable publication and replay

The runner requires the exact synchronous `SQLiteMemory` backend and a stable
`CentralMemory` instance. The fixed text-report scorer commits without yielding;
that transaction is the publication boundary, immediately followed by assigning
`last_result` without another await. Awaitable report files are unscored
candidates until that commit.

The report, true returned tool string and submission correlation are captured
synchronously before cancellable post-call retention. Bounded retention preserves
the original cancellation and does not reinvoke tools/evaluators. A cancellation
accepted before publication produces one cancelled/undetermined outer result
while retaining the actual acquired inner observation. A later delivery
cancellation cannot roll back already-published facts. `runner.last_result`
exposes the published result when available; `SubmissionRunV2Error.result`
retains operational-failure evidence.

`SubmissionReportScorerV2(report_sha256=...)` accepts only retained text content or
its `ContentEntryScorable`. It checks canonical bytes/digest, v2 provenance,
selection, lifecycle coherence and cleanup before projecting a grade. Score
metadata includes contract, mode, evidence label, selected submission, run ID,
report digest and the existing final-run/publication-boundary markers. The
`simulated` metadata flag uses the score model's integer representation; the
retained report keeps the strict boolean.

Replay never calls a provider, tool, environment callback or evaluator. It replays
the acquired report, not artifact behavior, and is not a new generic job or
`Observation` schema. No database migration or distributed transaction is
introduced. Failed storage can leave an unscored candidate; it must not be treated
as a successful published run.

## Mock-only validation

```powershell
uv run --frozen pytest -q `
  tests\unit\executor\benchmark\submission\test_runner_v2.py `
  tests\unit\score\test_submission_report_scorer_v2.py
```

The source includes no live configuration or default service. Real-provenance
schema tests only construct and validate types. All executed integration fixtures
use the offline provenance pair, inert bytes, fake evaluator callbacks and
`MockTransport`, without authentication or resource operations.
