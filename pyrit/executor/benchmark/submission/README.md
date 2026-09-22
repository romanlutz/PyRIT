# Offline submission evaluation

This experimental executor runs **OFFLINE/SIMULATED** conformance fixtures through
the real PyRIT Responses target, normalizer, memory and scoring APIs. It does not
bundle an evaluator, task, build environment, artifact executor, service client,
or live-provider mode. Existing CTF behavior is unchanged.

The caller supplies async string-returning tools and a synchronous, non-I/O
`read_report()` snapshot. The binding owns artifact acquisition, the strict
submission policy, evaluator invocation, feedback, and final selection. The
executor does not reproduce that state machine or inspect private task state.

## Public seams

Import `SubmissionTool`, `SubmissionHooks` and `SubmissionLimits` from
`pyrit.executor.benchmark.submission.hooks`, and `NativeSubmissionRunner` from
`pyrit.executor.benchmark.submission.runner`.

`SubmissionTool` accepts `name`, `description`, an object `parameters` JSON schema,
and `callback_async`. The callback is invoked with keyword arguments and must
return a string. Successful strings remain byte-for-byte unchanged. Tool names,
descriptions and schemas are binding-owned.

`SubmissionHooks` accepts:

| Field | Responsibility |
| --- | --- |
| `tools` | Tuple of injected tools |
| `read_report` | Defensive snapshot of the versioned strict report |
| `recoverable_errors` | Explicit pre-dispatch exception classes |
| `terminal_errors` | Explicit operational exception classes |
| `error_feedback` | Exact model-visible string for a recoverable tool error |

No error prefix is silently added. For example, a binding comparing against a
tool system that renders `Error: <message>` must explicitly supply that mapping.
Terminal and unexpected errors stop the loop; they are not recovery feedback.
Exception policies must not overlap. Python callbacks are trusted local test
code: this interface is **not isolation against a malicious callback**.

```python
runner = NativeSubmissionRunner(
    hooks=fixture_hooks,
    directory=Path("results") / "offline-submissions",
    limits=SubmissionLimits(max_requests=4, max_tool_calls=4),
)
result = await runner.run_offline_async(
    seed=fixture_seed,
    system_prompt="OFFLINE/SIMULATED fixture instructions.",
    transport=httpx.MockTransport(fixture_provider),
)
```

The factory requires `httpx.MockTransport` and supplies a dummy endpoint and
credential through public constructors. It never initializes Azure credentials.
There is no URL, socket, container, VM, or authentication option to enable live
execution. Supplying a custom Python hook is not evidence of service isolation.

## Conversation and stopping behavior

The new executor uses `OpenAIResponseTarget(auto_execute_tools=False)`. It owns
the task/tool loop, while the target returns one genuine provider generation.
Other integrations may instead let their native task runtime own the loop.
The target's existing automatic loop remains the default for existing callers.

Each generation is sent through the public normalizer. A recoverable tool failure
produces the binding's exact tool-error feedback; a later corrected submission is
a new invocation, not an evaluator retry. Sequential execution is explicit and
parallel provider tool calls are rejected.

A no-tool assistant response is not a completion signal. Further generation uses
the public continuation API over authentic persisted history, without appending a
`continue` prompt, replaying input converters, or duplicating the prior message.

Returned full success stops before another provider or evaluator invocation.
The real final tool output is hashed with `set_message_piece_sha256_async` and
persisted with `CentralMemory.add_message_to_memory` when there will not be a
subsequent normalizer send. It is not a fabricated assistant final.

A healthy configured generation/tool bound is checked **between** completed
operations. It ends the task normally with `termination_reason=budget`: the latest
valid grade remains the result, and no valid submission remains `no_submission`.
An in-flight timeout, cancellation, protocol error, or uncertain acquisition is
different and never projects a clean earlier score.

## Evidence and scoring

`StrictSubmissionReport` in `pyrit.models.submission` validates the
`strict-submission-v1` projection. It does not define a general remote-job API.
Reports must explicitly declare `mode=offline` and `simulated=true`.

The binding is responsible for freezing inert file bytes and their digest before
dispatch and retaining those bytes in approved storage. Every invocation,
including rejection and guarded calls, has a distinct local submission identity.
Repeated bytes do not imply an idempotent submission. The executor journals
provider call identity before invoking a hook and links newly observed submission
IDs after the hook. It retains every raw report projection, exact feedback and
earlier valid observation, including on errors.

The selected submission is the **latest valid completed result**, never the best:
`0.75` followed by `0.25` selects `0.25`. Boolean, missing, nonfinite and out-of-range
grades are invalid. A missing artifact is not an executed behavioral failure.
A genuine completed zero is distinct from infrastructure failure.

Partial evidence, uncertain remote disposition, or unknown/failed cleanup stops
as incomplete while retaining the acquired grade and successful tool feedback.
Cancellation of a local await does not confirm remote cancellation. Query,
remote cancellation and idempotency capabilities remain explicitly absent.
Lifecycle fields must agree: a returned completed observation cannot say it was
never dispatched, rejected, or remotely cancelled. A separately observed
acceptance acknowledgement and receipt ID are not required when the final outcome
is known complete. Unknown acknowledgement provenance is distinct from an unknown
remote outcome.

Each run creates a fresh local directory containing a labeled `events.jsonl`
journal and a canonical JSON report named by its SHA256. Awaitable file writes
are explicitly unscored candidates, not publication receipts. Private evidence is not
appropriate for public commits. The default example location is already ignored.
Raw headers and credentials are not journaled.

`SubmissionReportScorer(report_sha256=...)` reads that report via
`ContentScorable`, which memory converts to `ContentEntryScorable`. A real PyRIT
`Score` references the retained report, not the last chat sentence. Only a clean
completed report projects a numeric value; other states use `ScoreStatus.UNDETERMINED`
and retain the prior grade separately as an observation. No `Observation`-union
extension or database migration is introduced.

Replay uses the stored content reference and original report digest. It verifies
the stored content digest, selected submission and embedded artifact/evidence
identities without invoking an evaluator or tool. This replays the acquired report,
**not the artifact's behavior**. It neither re-executes nor attests the contents of
binding-owned artifact storage. Tampering is an error, not a new grade.

`SubmissionRunError.result` and `runner.last_result` expose retained operational
failure evidence. Cancellation is re-raised after retaining its partial result.
Filesystem retention errors still propagate; the executor does not invent a
durable receipt when a write failed.

The non-I/O report snapshot, call correlation, and any genuine returned tool string
are captured synchronously before the next cancellable await. Post-call journaling
and final retention each have a five-second deadline and survive repeated caller
cancellation without reinvoking a callback. The original cancellation is re-raised;
retention failure is surfaced separately, and no durable `last_result` is claimed
if final retention failed.

**Publication boundary:** this offline executor supports the exact synchronous
`SQLiteMemory` backend and the fixed text-content `SubmissionReportScorer` path.
Other configured memory implementations are rejected. After all awaited candidate
writes, cancellation is reconciled before scoring; an owned unscored candidate may
be replaced by a cancelled report. No previously scored file or earlier run is
removed. The text-content scorer's successful SQLite transaction is the durable
publication point, followed immediately by `last_result` assignment without an
await. A cancellation accepted before that point publishes one cancelled,
undetermined score retaining the true acquired observation. A later cancellation
can interrupt delivery but cannot roll back an already published result.

The persisted score and `ContentEntryScorable` are authoritative even without the
in-process result: metadata includes run identity, report digest, status,
`publication_role=final_run_result` and `publication_boundary=pyrit_score_commit`.
These are provenance labels, not live-execution attestations. There is no atomic
filesystem/database transaction or remote cancellation guarantee. Storage failures
can leave an unscored candidate; without a committed score it is not a completed run.

## Local conformance checks

Initialize an isolated `SQLiteMemory` backend before constructing the runner.
No additional Python dependency or Inspect installation is required for native
execution. The public tests use inert byte strings and fake async evaluators:

```powershell
uv run --frozen pytest -q tests\unit\executor\benchmark\submission `
  tests\unit\score\test_submission_report_scorer.py `
  tests\unit\prompt_target\target\test_openai_response_target_external_tools.py
```

These are simulated protocol/conversation checks, not a real remote evaluation
result, an artifact execution result, or evidence of production containment.
