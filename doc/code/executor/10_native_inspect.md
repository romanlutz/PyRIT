# Offline native Inspect task bridge

`InspectNativeTaskBridge` is a separate, **OFFLINE/SIMULATED** extension of the
[CTF prototype](9_inspect.md). It preserves an injected Inspect task's solver,
setup, tools, prompts, store, and numeric native grader. It replaces only the
solver's public `Generate` callback with calls to the existing PyRIT
`OpenAIResponseTarget`. It does not bundle an evaluator, task dataset, remote
client, VM provider, or deployment configuration.

No real model, evaluator, container, VM, or generated workload was used to
validate this extension. The tests use inert byte strings, `httpx.MockTransport`,
injected evaluators, and genuine Inspect/PyRIT APIs. Previous CTF live results do
not establish compatibility with a remote submission service.

## Public boundaries

```python
from pyrit.executor.benchmark import InspectNativeTaskBridge, InspectTaskEnvironment
from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark.submission.hooks import SubmissionLimits

bridge = InspectNativeTaskBridge(
    target_factory=target_factory,
    model_name="offline-fixture",
    read_report=binding.read_report,
    artifacts=InspectRunArtifacts(
        directory=new_run_directory,
        provenance={"mode": "OFFLINE/SIMULATED"},
    ),
    environment=InspectTaskEnvironment(
        audit_async=audit_native_sample_async,
        cleanup_async=check_owned_cleanup_async,
    ),
    limits=SubmissionLimits(max_requests=8, max_tool_calls=8),
    max_tool_output_bytes=16384,
)
result = await bridge.execute_async(
    task=native_task,
    sample_id="offline-case",
    native_scorer="latest_native_grade",
)
```

The caller owns every dependency above. `native_task` contains one selected sample
and its real solver and scorer. The solver supplies its system/user messages and
tools, then calls `await generate(state, tool_calls="loop")`. The bridge invokes
that same solver with a public callback, not a copied solver or a fabricated
Inspect model provider. Grader-only criteria remain with the task.

`target_factory(schemas)` receives the function definitions derived from the
task's original `ToolDef`s after native setup. It must return an
`OpenAIResponseTarget` configured with:

- `auto_execute_tools=False`, so each call returns one actual provider generation.
- The supplied schemas unchanged, `parallel_tool_calls=False`, and `store=False`.
- An explicitly owned mock HTTP client for offline validation and SDK retries zero.

Set `RETRY_MAX_NUM_ATTEMPTS=1` in the isolated caller process or test fixture.
The bridge verifies this PyRIT retry setting. The caller remains responsible for
the injected SDK client's retry configuration and lifetime.

This offline variant requires the existing synchronous `SQLiteMemory` backend,
with the same `CentralMemory` instance throughout the run. Other storage
implementations must not be assumed to share its final publication behavior.

Native callbacks are invoked through public `inspect_ai.model.execute_tools`.
Within the configured feedback bound, string results pass through unchanged.
Recoverable native `ToolError` results use
Inspect's OpenAI wire representation, `Error: ` followed by the original message.
Native error metadata and raw feedback are retained alongside this representation.
Terminal errors must propagate, not be converted to a recoverable `ToolError`
or `TimeoutError` that invites another model submission.

The tested seam supports sequential string-returning tools, text system/user
messages, fixed tool definitions, and automatic tool choice. Custom tool
`model_input`, structured/multimodal tool results, handoffs, tool-set replacement,
and edits to already retained history are rejected explicitly. These are
prototype compatibility limits, not universal restrictions in Inspect or PyRIT.

`max_tool_output_bytes` is an optional, positive UTF-8 byte limit on **exact
model-visible tool feedback**. Its default is the supplied
`SubmissionLimits.max_response_bytes`; it can be configured independently.
`max_response_bytes` separately bounds a retained provider response. No shared
submission-report or private evaluator policy is changed by either setting.
If a native callback returns more feedback than the explicit tool bound, or
Inspect would rewrite its string, the bridge retains the original acquired
feedback and a source/limit diagnostic, then stops as incomplete. It does not send
Inspect's truncation notice to the provider and publish a clean score. Inert
UTF-8/newline feedback larger than Inspect's usual 16 KiB default is tested
byte-for-byte below the explicitly configured bound.

## Honest continuation and termination

`OpenAIResponseTarget(auto_execute_tools=True)` remains the default. Its existing
tool loop, dictionary serialization, and default identifier are unchanged.
String-returning callbacks are also supported without a JSON wrapper.

In external mode, callers send genuine tool results as `function_call_output`
messages through `PromptNormalizer.send_prompt_async`. A terminal tool result is
hashed and stored once through public PyRIT memory APIs without triggering another
model call. There is no fake final assistant message.

If the native solver asks to generate again without adding a message, the bridge
uses `PromptNormalizer.continue_conversation_async(target=..., conversation_id=...)`.
That calls the target's public continuation method over already retained history.
It adds no request, duplicates no old turn, reruns no prompt converters, and
persists only actual new provider responses. Targets without explicit continuation
support reject the operation. Empty history, missing output, and transport failures
do not create synthetic error or continuation messages.

Backend parity compares successful and recoverable model-visible feedback exactly.
For terminal cancellation, Inspect may produce its own diagnostic rather than a
callback result. The bridge retains that unforwarded diagnostic separately with
native provenance, leaves returned-feedback evidence unset, and makes no further
provider call. This is not a claim that both backends have identical raw logs.

A transparent `ToolDef` observer captures an original callback's returned string
and synchronous report snapshot before the next async write. Cancellation at that
boundary does not erase the new grade, replace its genuine feedback with a generic
diagnostic, or resubmit the artifact. Public memory persistence is joined exactly
once, including a real terminal tool result that was never forwarded.

Every continuation counts toward the generation limit. A healthy limit reached
between completed operations ends normally with the latest valid grade, including
zero. The outer report records `termination_reason="budget"` and actual generation
count. With no valid submission, it records `no_submission`, not a zero capability
score. A deadline or interruption during an operation is not this healthy boundary.

The same pre-dispatch check enforces native `Task.turn_limit`, the current
`TaskState.message_limit`, and native `token_limit` modes `all` and `output`.
The bridge counts actual provider calls and reported token usage explicitly;
Inspect's automatic model-provider accounting does not meter the external PyRIT
target. Per-response `ModelOutput.usage` contains the observed values when
available. Missing usage required by a configured limit stops further dispatch
as incomplete instead of inventing zero usage. Unsupported token formulas, cost
limits, and working-time budgets are rejected before callbacks.
These token thresholds stop between generations: the last allowed generation
can cross a threshold, but another generation is not dispatched afterward.
They are not a claim of a hard per-request token or cost cap.

Stops, native completion, and applicable bounds are checked again after awaited
dispatch preparation and immediately before an original tool callback. Exhausting
the tool-call budget blocks another provider generation. A genuine tool requested
by the last allowed generation still completes if tool and message capacity
remain; its exact returned feedback is retained. A prepared call skipped at a
healthy boundary stays visible as `not_dispatched` in the journal, not as a
fabricated callback result.

**Strict variant behavior:** after a returned full-success submission, stop before
another provider or evaluator call. After partial required evidence, unknown remote
disposition, or uncertain cleanup, retain the actual feedback and valid numeric
observation but stop as incomplete. This differs from tasks whose legacy solver
checks completion only after an unrestricted Generate/tool loop returns.

## Report contract and scoring

The caller's non-I/O `read_report()` supplies a defensive JSON-safe snapshot using
`strict-submission-v1`. The binding, not the public adapter, owns acquisition and
evaluation policy: freeze artifact bytes before dispatch; at most one evaluator
call per deliberate submit; expose every attempt; keep submission and observed
remote identities separate; use the latest valid returned result, never best-of.
Missing artifacts are recoverable and cause no dispatch. A returned grade must
be a finite number in `[0, 1]`, not a boolean or coerced string.

`StrictSubmissionReport` validates the projection and preserves ordered submission
IDs, artifact SHA256/length, acquisition/acceptance/cleanup status, exact feedback,
and opaque returned evidence. It does not implement a remote service protocol.
Reported query, cancel, and idempotency capabilities are false. Local cancellation
does not establish that remote work stopped.

After materializing the original typed `.eval` log and attachments, the bridge
checks run/sample/epoch identity and the native numeric score against the selected
latest valid submission. Optional final text is presentation only. The deterministic
`RetainedSubmissionReport` is retained in an exclusively created file and scored as
`ContentScorable(text)`. PyRIT persists it as `ContentEntryScorable`, with no database
migration and no attachment to unrelated final chat.

`SubmissionReportScorer` is read-only replay: it checks the expected report digest,
stored content identity, and selected submission, then returns the numeric value
only for a clean completed acquisition. Error, unknown, cancelled, incomplete, and
no-submission outcomes are undetermined. Prior valid grades remain observations
inside an incomplete report, not clean overall verdicts. Replay performs no model,
tool, or evaluator call.

Caller cancellation is re-raised after bounded local retention. Inspect may not
finalize a native log on this path: `bridge.last_result` exposes the acquired
report and its persisted undetermined score, with unavailable native log/evaluation
identities left `None`. It does not fabricate a native completion or remote stop.
If retention itself fails, the original exception carries diagnostics rather than
being replaced with a success-shaped result.

One owned, shielded finalization task joins pending authentic-message writes,
checks cleanup, and uses a single five-second deadline. Repeated caller
cancellation neither abandons that join nor restarts its timeout. Before the
deadline, the caller waits for the same write rather than issuing a duplicate.
If it expires, the manifest records `canonical_messages_status="unknown"` and
the pending native message IDs, and the original cancellation carries the
persistence diagnostic. An uninterruptible database write may later complete;
the result does not claim that persistence succeeded or that the write stopped.

After local Inspect evaluation settles, recovery uses public log listing and
typed readers scoped to the owned run directory. A finalized log must have a
completion timestamp and matching run, attempt, sample, epoch, and observed
sample UUID. Its genuine log path/evaluation ID are retained even for a cancelled
run. Unrelated, unfinished, or unreadable candidates cannot supply those IDs.
Lookup failure is diagnostic and cannot replace the caller's cancellation.

### Publication and recovery

Report-file writes, manifest writes, native-log checks, and cleanup occur before
score publication. They are **unscored candidate evidence**, not a committed run
verdict. A cancellation checkpoint follows these awaited operations. For the
specific current `SubmissionReportScorer(ContentScorable(text))` plus unchanged
`SQLiteMemory` path, validation and the durable score transaction do not yield to
the event loop. That commit is the publication boundary. There is no later await
before `last_result` is assigned and returned.

An accepted precommit cancellation retains the known behavior observation but
publishes only a cancelled, undetermined run report. A later cancellation of
result delivery cannot retract an already published observation. This is not an
atomic transaction spanning report files and the database. The prepared on-disk
manifest explicitly identifies itself as `unscored_candidate`; the authoritative
publication is the retained PyRIT score and its `ContentEntryScorable`. Its durable
metadata records `publication_role="final_run_result"` and
`publication_boundary="pyrit_score_commit"` alongside the offline/simulated label.

Recovery does not require the old bridge object or `last_result`. After reopening
the SQLite database, use public `get_scores` with the
`SubmissionReportScorer` identifier, select the matching `run_id` and
`report_sha256` in score metadata, then resolve the stored content through
`get_scorable_content(content_ids=[score.scorable.content_id])`. Parse it as
`RetainedSubmissionReport` and verify its canonical hash against the prepared
manifest. No matching score means the candidate was not published; the existence
of a JSON file alone must never be interpreted as a completed run.

Interrupted candidate files are retained under distinct content-hash filenames.
No scored fact is deleted or rewritten. Successful and precommit-cancelled
recovery are covered with a closed-and-reopened SQLite fixture, without an
in-process result pointer.

The run directory contains the original native log, canonical PyRIT conversation
and score through the caller's memory backend, `submission-report-<sha256>.json`, and the
adapter journal/manifest. The journal correlates original provider-call arguments,
native string/error results, newly observed submission IDs, and explicit generation
counts. Keep acquired reports and any binding-owned artifact snapshots private.
Public examples must never contain private clients, task prompts, endpoints,
credentials, or generated artifacts.

Environment audit and cleanup are explicitly caller-supplied and distinct from
remote-work disposition. The CTF Docker profile remains unchanged; it is not used
as a substitute for a task's build environment or behavior containment.

## Offline validation

The tested optional environment is Inspect AI `0.3.259`, OpenAI `2.54.0`, and
Python `3.12`. PyRIT core and bridge definitions remain importable without Inspect.
The string/error formatter intentionally uses public `ChatMessageTool` fields:
Inspect's provider-specific `messages_to_openai` helper in this version requires
OpenAI `>=3.1.0`, which this extension does not require or silently install.

```powershell
uv run --no-sync pytest -q tests\unit\executor\benchmark\test_inspect_native_task.py `
  tests\unit\executor\benchmark\test_inspect_native_generate.py `
  tests\unit\prompt_target\target\test_openai_response_target_external_tools.py
```

These fixtures exercise real native solver setup, public Generate and tool
execution, mocked provider requests, deliberate resubmission, the native grader,
content persistence/replay, and exact continuation histories. They are not evidence
of real Hyper-V or evaluator-service readiness.
