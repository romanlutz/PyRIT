# Native agent cyber evaluation

This slice composes existing PyRIT bricks instead of routing through another
evaluation framework. It is **not evidence of a qualified GHCP container run**.
The shipped tests use explicitly simulated native SDK events. A real binding
must establish container image, transport, credential isolation and tool
execution before advertising readiness.

## Ownership

| Component | Owns |
| --- | --- |
| `NativeCyberTaskBinding` | Task seed, allowed edits/techniques, readiness and one fresh runtime context |
| `NativeCyberEvaluation` | Literal `PromptSendingAttack` or an allowed `AttackTechniqueFactory`, converters, run/step/finish policy, TTL and lineage |
| `NativeAgentTarget` | Sending prepared instructions to an existing native session; no tools, prompts, grading or environment construction |
| `CopilotSdkAgentSession` | Public SDK send/event adaptation and observed tool correlation |
| Runtime `grade_async` | Original grader and immutable artifact acquisition while the workspace still exists |
| `NativeCyberReportScorer` | Read-only grade projection from retained canonical report content |
| Runtime context owner | Agent environment creation and cleanup, exactly once |
| `EnvironmentLease` | Provider-neutral service handles, optional setup/health, run-owned rollback and release |
| Binding `create_host_storage_async` | Verifying an existing private parent before exclusively creating the exact empty run child |
| Binding `validate_host_storage_async` | Effective host access controls on the run directory and its private parent |
| Runtime `validate_agent_storage_async` | Actual guest exclusion from host reports and PyRIT memory, including mounts and host tools |

The literal baseline has no objective or auxiliary scorer inside the attack.
The original grader is invoked once after the agent is quiescent and before
environment cleanup. The report, its one `ContentEntryScorable`-linked `Score`, and the episode link
are committed together in PyRIT memory, not scored from the final assistant
sentence. The normal attack result and conversation remain in memory and are
referenced by the report. Cleanup failure prevents clean success while
retaining an already-acquired judgment.

Bindings are trusted code registered via `get_native_cyber_bindings().register`.
Only registry names and allow-listed configuration should cross an API boundary.
Private task assets, credentials, graders and outcome weights do not belong in
public presets or source.

## Binding contract

```python
from contextlib import asynccontextmanager

from pyrit.executor.workflow.native_cyber_eval import (
    NativeCyberEvaluation,
    NativeCyberRuntime,
    NativeCyberTaskBinding,
    get_native_cyber_bindings,
)
from pyrit.models.native_cyber import NativeCyberRequest


class TaskBinding(NativeCyberTaskBinding):
    async def create_host_storage_async(self, *, directory):
        await self.host_storage_policy.create_child_async(directory=directory)

    async def validate_host_storage_async(self, *, directory):
        await self.host_storage_policy.verify_async(directory=directory)

    async def readiness_async(self):
        # Return actual qualification or explicit blockers, not a successful fallback.
        return self.qualified_readiness

    @asynccontextmanager
    async def open_runtime(self, *, run_id, request):
        async with self.owned_environment_factory(run_id=run_id) as runtime:
            yield runtime


binding = TaskBinding(name="registered_task", version="1", description="A caller-owned task")
get_native_cyber_bindings().register(binding, name=binding.name)
run = NativeCyberEvaluation(
    binding=binding,
    request=NativeCyberRequest(instruction="The exact approved task input"),
    directory=private_host_evidence_root,
)
view = await run.start_async()
```

`NativeCyberRuntime.target` is a `NativeAgentTarget` around a fresh
`NativeAgentSession`. Its required `validate_agent_storage_async(*, directory)`
must verify that the actual runtime cannot read or write the host report directory
or PyRIT memory. This check runs before any instruction or raw event-file write;
missing or failed verification prevents execution and triggers owned cleanup.
`grade_async(*, evidence)` returns `NativeCyberJudgment`
with an optional numeric value, rationale, complete flag, retained artifact
references and acquired original evidence. A missing or failed acquisition has
no numeric value. Artifact references describe bytes the binding has already
retained and hashed; they do not authorize arbitrary host file access.

`NativeCyberRequest` limits edits to instruction, label, allowed technique and
converter registry names, stepping, bounded TTL/turn timeout and parent lineage.
The binding caps TTL and explicitly allows converters and technique factories.
Task-specific interfaces must further constrain instruction variants and protect
the original grading rules, trust policy, fixture and credentials.

## Opt-in unified Copilot SDK/model evidence

`NativeCyberTaskBinding(..., evidence_contract=NativeCyberEvidenceContract.COPILOT_SDK_AND_MODEL)`
selects unified capture explicitly. The default `SDK_EVENTS` contract retains
the existing SDK-only behavior and **does not claim GHCP model-wire coverage**.
The unified contract currently supports the literal attack and retained operator
steps with `CopilotSdkAgentSession`; it does not create a Copilot SDK, model
client, relay, credential, task image or alternate grader.

Before dispatch, the controller declares `ghcp.sdk.events`,
`ghcp.model.requests` and `ghcp.model.responses` as required sources in the
existing `NativeCyberEvidenceStore` episode. It attaches one
`CopilotSdkDatabaseEvidenceSink` to the session before any event, opens each
outer turn and its raw streams, then calls the binding's
`bind_model_observer(*, runtime, observer)` hook. A binding opting in without
that trusted host-transport injection fails, rather than falling back to
SDK-only completion.

The injected `CopilotSdkModelObserver` has an immutable
`CopilotSdkCaptureIdentity(run_id, session_id, turn_id, turn_index)` and two
async observation methods:

- `record_request_async(*, request_id, body: bytes, accepted: bool)` must finish
  **before** the trusted transport dispatches the original request. Rejected
  ingress is a HARNESS source and cannot satisfy model coverage.
- `record_response_async(*, request_id, body: bytes, status_code, origin,
  complete: bool)` must finish **before** that transport returns bytes to the
  SDK. `CopilotModelResponseOrigin.PROVIDER`, `HOST` and `SYNTHETIC` are distinct.
  Only a complete, paired, successful original provider response can satisfy
  model coverage. This is a trusted source assertion, not inferred from a
  response body or a 200 status.

Only original bodies belong in these methods, not URLs, authorization headers,
tokens or constructed success receipts. Raw bodies can still contain sensitive
task data and remain in the existing private raw-stream store. Public model
event summaries retain source request identity, byte count, digest, origin and
status, never the body. This whole-body, bounded observation seam does not
claim arbitrary provider or streaming-protocol compatibility.

SDK callbacks synchronously enqueue detached original SDK envelopes. Their
durable writes and the awaited model boundaries drain a single ordered queue.
One run-wide controller sequence spans all sources and retained turns. SDK
event IDs and model request IDs cannot be reused, including across turns;
foreign run/session/turn identity, late observations, incomplete pairing,
malformed events and failed or quota-limited writes latch required gaps.
Cancellation joins an in-flight store write before sealing/finalizing so it
cannot silently finish after the workflow returns.

The controller owns turn creation, authentic normalizer-persisted request,
response and tool `MessagePiece` links, finalization, and grading decisions.
The target/session only forwards prepared instructions and observed SDK events;
it has no DB, model dispatch or grading logic. In unified mode the legacy
after-delivery SDK DB capture and SDK sidecar journal are not also run.
There is one SDK stream per turn, not replayed cumulative session history.
Model sources, host-rejected requests and host/synthetic responses have
separate stream identities in the same turn.

The recorder projects the canonical report into the same controller event
order as the DB, preserving every original SDK ID/payload and remapping tool
request/start/completion ordinals. Raw SDK ordinals remain in the session
snapshot. The report's `scope="native_sdk_and_model"` explicitly distinguishes
this projection from SDK-only evidence; no second event table or evidence DB
is introduced. A watermark mismatch between the adapter and persisted SDK
observations is incomplete capture, not a recovered success.

After the final turn, the agent must successfully quiesce and the store's
pre-grading coverage assessment must pass before the binding's original grader
is called once. That call occurs while the owned environment still exists.
Cancellation, missing actual model capture, tool-correlation gaps, failed
quiescence or capture failures prevent grading. Late/post-grade capture failure
or failed cleanup prevents a COMPLETE result. Only after owned cleanup does
the existing atomic finalizer commit the report, standard Score and episode
reference. A finalizer failure is surfaced without retrying the grader or
manufacturing a published result.

The regression suite uses explicitly **synthetic source fixtures** and isolated
SQLite with network/provider/subprocess calls blocked during tests. It proves
offline lifecycle/capture behavior, not a genuine JWT judgment, benchmark
equivalence, live GHCP qualification or authorization to replay a prior task.

### Trusted whole-body model observation transport

`CopilotSdkModelTransport(observer=observer, forwarder=admitted_forwarder)`
enforces the before-dispatch/before-return obligations above instead of leaving
that ordering to every binding. A binding creates a fresh immutable adapter for
each supplied observer; its already-admitted forwarder may be reused. There is
no SDK, route, credential lookup, provider client, listener, authorization policy,
tool executor, retry or run-budget allocator in this adapter.

The caller supplies a `CopilotSdkModelRequest` with the exact observer identity,
one original request ID, original `body: bytes`, and explicit `streaming=False`.
`forward_async(*, request, deadline)` first durably records that body. Only then
does it call the injected `CopilotSdkModelForwarder.forward_async(*, request,
deadline, max_response_bytes)` once. The forwarder returns a
`CopilotSdkModelResponse` with matching identity/request ID, original body,
status, immutable ordered header pairs, explicit `CopilotModelResponseOrigin`
and `complete` flag. Response capture finishes before the identical result
object is returned. Headers are preserved for the caller, not sent to the
recorder or included in representations. Request URLs/auth headers cannot be
passed through this body-only request type.

No JSON is parsed or reserialized. Fields such as `prompt_cache_key`, unknown
body fields, whitespace and response header ordering are preserved. The caller
must validate the exact upstream protocol and wire features before admission.
Streaming requests are rejected rather than silently coerced into non-streaming
requests. This whole-body boundary does **not** implement SSE, WebSocket or the
Copilot SDK's request-handler types/call shape. No pinned SDK handler package is
available in this implementation's local environment, so an independently
qualified caller-owned SDK shim is still required. This is not proof that a
particular CLI/SDK/provider combination works.

An admitted forwarder must label HOST/SYNTHETIC responses explicitly, even at
HTTP 200. They cannot satisfy MODEL coverage. For ingress the caller already
rejected, `record_rejection_async(*, request, response, deadline)` records the
original rejected request and supplied non-provider error response without ever
calling the forwarder. The adapter never fabricates a provider or rejection
body. Incomplete responses, wrong identities, oversize bodies, durable-write
failure and forwarder exceptions fail closed; the observed request and any
retained partial evidence remain available.

`deadline` is one finite, absolute `asyncio` loop time supplied by the caller,
covering request capture, forwarding and response capture. The same value and
the store's body-size cap are passed to the forwarder. Boundaries are rechecked
before forwarding and returning; there is no renewed timeout or implicit retry.
Cancellation/expiry cancels and joins the one local forwarding task, including
its cancellation cleanup, and latches a content-free evidence gap. In-flight
durable writes are also joined. Joining can exceed the deadline: this is
cooperative cancellation, not a hard wall-clock or remote-generation-stop
guarantee. A trusted forwarder must honor cancellation, release its I/O, join
its own workers and raise sanitized errors; an uncooperative implementation
cannot be force-stopped by this library. The existing controller checkpoints
the failure and retains ownership of grading/cleanup/publication decisions.

## Harness-neutral task composition

`NativeCyberEvaluation(..., harness=adapter)` explicitly selects an agent
adapter independently of the original task binding and sandbox provider.
Without that argument, the existing `SDK_EVENTS` and
`COPILOT_SDK_AND_MODEL` paths retain their call contracts. Unknown adapters,
non-literal attack loops and unsupported retained operator turns are rejected
before allocating an environment or invoking a model.

An explicitly adapted task supplies `NativeTaskRuntime`: immutable `run_id`,
`environment_id`, `simulated`, the existing host-storage exclusion check, and
`grade_async(*, evidence: NativeHarnessEvidence) -> NativeCyberJudgment`.
The original task grader does not construct an SDK, select stream names or
know a CLI parser. The task binding still owns instruction/response policy,
the original judgment and artifact acquisition. The provider still owns
allocation, stop/freeze guarantees and cleanup.

Adapters declare capabilities and required sources before dispatch, attach a
target/recorder to the exact allocated runtime, and expose an explicitly typed
source snapshot. `NativeHarnessEvidence` holds an immutable
`NativeHarnessIdentity(run_id, environment_id, harness_id, source_session_id)`,
quiescence, source coverage/gaps and **either** unchanged SDK evidence **or**
native CLI evidence. A CLI source session is absent until observed; a
controller turn ID is not a fabricated provider session or event ID.

| Adapter | Outer turns | Source and tool semantics | Model evidence |
| --- | --- | --- | --- |
| `CopilotSdkHarnessAdapter` | Retained steps only when explicitly qualified | Actual SDK callbacks and correlated request/start/completion, with original model-visible output | Unified SDK recorder plus trusted host model observer |
| `ClaudeCliHarnessAdapter` | One-shot `-p --output-format stream-json --verbose` | Actual assistant `tool_use` and user `tool_result`; no invented execution-start/completion | Original Anthropic Messages bodies/SSE through the existing admitted gateway |
| `CodexCliHarnessAdapter` | One-shot `exec --json` | Actual item start/completion and `aggregated_output`; no provable mapping from item ID to a model request | Original Responses bodies/SSE through the existing admitted gateway, or an admitted whole-body forwarder |

Claude is the second tool-bearing conformance path because its existing
transcript exposes the model request/result IDs. Codex tool-free episodes can
complete, but **tool-bearing Codex episodes remain incomplete** under the
existing causal-coverage rule. `aggregated_output` is an observed CLI tool
receipt, not evidence that the model saw those exact bytes. This implementation
does not relax that rule or manufacture SDK events/request IDs. CLI command
and tool-output correlations remain source event links in the existing store;
the target does not invent conversation `MessagePiece`s for them. Actual
normalizer-persisted request/assistant pieces retain their IDs and content
digests. Missing fields remain absent.

The shared controller runs the same converter/normalizer/attack lifecycle for
all adapters. It seals sources, confirms agent quiescence, and assesses
required DB coverage **before** calling the task's original grader once while
the target environment is alive. CLI PREGRADING validates raw pipes, actual
parser/tool/model evidence and message links without requiring a future
judgment or artifact. FINAL still requires the acquired judgment and verified
closure. The CLI runner distinguishes observed guest exit from a successful
owned stop barrier; exit zero alone cannot authorize grading.
The explicit SDK connector also supplies a provider-owned
`quiesce_agent_async` barrier, awaited after SDK disconnect and before grading.
SDK disconnection alone is not treated as proof that guest tools stopped.

After cleanup, the existing SDK or CLI atomic finalizer stores one native
report, one standard Score and the episode reference. The CLI representation
keeps its version/profile, original event fields, stdout offsets/digests and
real process outcome, rather than translating them into SDK shapes. When the
shared task returns a `NativeCyberJudgment`, its complete original object is
retained inline in the CLI judgment with a verified `inline-sha256:` reference;
this is an observed grading result, not a second invocation or invented
artifact. Task-supplied artifact references remain unchanged. Artifact-only
turns require explicit task policy and actual native terminal events, never
a manufactured assistant answer.

Canonical v1 CLI serialization omits only `environment_id`, `harness_id` and
judgment `retained_evidence` when their value is `None`. Existing v1 null fields
remain explicit, preserving pre-extension canonical bytes, hashes and stored
Score/content links without rewriting historical records. Non-null extensions,
including empty/false retained evidence, remain part of the bound digest.

The raw capture sink now checks each write receipt and stops progress on
quota loss or misattribution instead of allowing a clean parser exit after
truncation. In-flight writes are joined under cancellation. Post-finish
observations latch gaps; partial reports cannot acquire a COMPLETE Score.
Read-only `get_episode`/`read_event_payloads`/`read_raw_chunks` expose the
existing run identity, controller order, stream UUID/offset and verified
digests. Sensitive payload reads still require `allow_sensitive=True`.

The original-body wrapper is now `NativeModelTransport`, with
`NativeModelObserver`, `NativeModelForwarder`, `NativeModelRequest`,
`NativeModelResponse` and `NativeModelCaptureIdentity`. Old
`CopilotSdkModel*` names and `CopilotSdkCaptureIdentity` remain aliases.
Identity additionally binds an environment/harness; for CLI whole-body
capture the observed session is explicitly `None` until CLI evidence supplies
it. `NativeCliModelObserver` supports the admitted non-streaming Responses
boundary only. Anthropic Messages retains its separate existing gateway
contract, including approved headers and streaming semantics. Neither path
coerces an unsupported wire protocol or adds auth, retries or budget windows.

These are ordinary offline library implementations and synthetic source/store
conformance tests. They do not qualify any actual CLI, SDK, image, model route,
sandbox or original benchmark. A production connector must provide its
already-admitted launcher/gateway and proven stop boundary. No private task
implementation, accepted task result or operational database is copied into
these adapters.

## Provider-neutral environment lease

This path is native PyRIT only. It neither imports nor delegates to Inspect or
Inspect SWE. Providers and future sandbox-local harness adapters remain separate
from attacks, targets, converters and original graders.

Bindings may override
`create_environment_lease(*, run_id, request) -> EnvironmentLease[NativeCyberRuntime]`
instead of `open_runtime`. The factory performs no acquisition. The workflow
first completes host storage and readiness checks, then acquires the lease under
the episode deadline. Provider setup and health checks, when declared, complete
before the runtime is returned. The existing runtime storage guard, native
attack, retained-session stepping, original grade, cleanup and report projection
retain their ordering. The lease remains open through original grading.

`pyrit.models.environment_lease` defines opaque provider-scoped resource handles
and named service handles with open-ended roles and optional parent names. There
is no fixed service count, agent/target pair, container ID format, socket, mount,
VM type or controller client in these canonical models. Several services may
share one owned provider allocation. An external controller can release only
its per-run allocation; the shared controller is not implicitly owned.

Providers subclass `EnvironmentLease` and use `_acquire_resource_async` before
each provider acquisition. It reserves the exact run-owned handle and release
callback **before** awaiting provider I/O. Callbacks receive that same handle.
Failed or cancelled acquisition, setup or health checks roll back all
reservations in reverse order, including the failing acquisition. Provider
release must be safe for an absent or partially created reserved resource and
confirm release, not delete by a broad name or prune unrelated resources.
Duplicate cleanup ownership and foreign-run handles are rejected.
This first lease implementation acquires serially in its owning task. Spawning
untracked acquisition tasks is rejected before provider I/O, so rollback cannot
race an allocation still running in a child task.

Cleanup attempts every reservation even if one release fails. It is once-only,
has a cooperative per-resource timeout, and cannot become successful on a later
call merely because an earlier attempt failed. Caller cancellation is propagated
after release settles. Providers must honor cancellation; this abstraction cannot
terminate an uncooperative external process or recover an unknown allocation ID.
Reserved IDs must therefore be chosen before acquisition, or the provider must
own and confirm its own compensating rollback.
The workflow's cleanup grace follows the lease's per-resource deadline and
reservation count instead of always cutting off after ten seconds. Host storage
creation retains its original ten-second grace plus five-second settling
allowance. This does not relax failure reporting: expired or unconfirmed
release is never recorded as clean cleanup.

Explicit `SETUP` and `HEALTH_CHECK` capabilities are optional; calling an
unsupported operation raises rather than silently succeeding. A health check
must observe every named service exactly once, with `healthy=True`; missing,
unknown or unhealthy observations fail acquisition. Dynamic service acquisition
and network-phase transitions are represented as future capabilities but are
rejected by this first implementation, not advertised as working. Role names
and parent handles leave room for those later providers without fixing a
two-service topology.

Existing `open_runtime` bindings are wrapped as a single `runtime` service with
an owned **context** ID, not an invented physical resource ID. This compatibility
wrapper declares no setup or health capability. It calls the original context
exit once after successful entry. The binding still owns rollback inside a
failed `__aenter__`; the wrapper cannot observe its partial resources and reports
cleanup uncertainty rather than claiming that an entry failure left no resources.
New multi-resource bindings should use the reservation helper for observable
rollback.

`evaluation.environment_lease` exposes an immutable in-memory lifecycle snapshot.
It is not a durable raw-log or resource schema. The workflow records native
events separately through `memory.native_cyber_evidence`. Tests here use
only fake one-, two- and four-service allocations and an inert external
controller. No Docker/VM provider, live harness, network transition or model
transport is implemented or qualified by these tests.

## Private host evidence boundary

The caller supplies a private host evidence root and an already-working, private
PyRIT memory backend. Neither may be exposed through agent mounts, host tools,
working directories or artifact-download endpoints. An absolute path, a user
profile location and a safe UI DTO are not access-control evidence.

Before readiness or runtime creation, the controller calls
`binding.create_host_storage_async(*, directory: Path) -> None`, then independently
calls `binding.validate_host_storage_async(*, directory)`. Creation must verify
the existing private parent before creating only the exact absent run child.
The POSIX default verifies parent ownership/type/mode, uses `mkdir(mode=0o700,
exist_ok=False)` without recursive parent creation, and checks both parent and
child again during validation. Both defaults reject Windows.

A trusted Windows binding must implement guarded creation as well as strict
validation. Before creation it verifies effective owner+SYSTEM parent permissions
and protected root ancestry, rejecting redirected/reparse paths and broad grants.
The direct parent may inherit its exact approved ACL from a protected ancestor;
it need not itself have a protected DACL. Only then may it create the absent
child with `Path.mkdir(exist_ok=False)`, omitting `mode`, so that approved
permissions are inherited. Explicit `mode=0o700` is not portable ACL hardening:
it can replace Windows inheritance with an unexpected DACL. The strict child
validator must still run; do not loosen its ACE checks or mutate old roots,
children or ACLs. Public tests exercise injected inert creation/verifier behavior,
not Windows DACL inheritance. The coordinated private OS measurements remain
separate evidence.

Successful creation transfers ownership of that empty child to the controller.
A creator that fails or is cancelled must clean up only any empty child it
created, never an existing path. The controller settles creation before handling
caller cancellation, then removes its owned empty child if host validation did
not succeed. Failed creation or validation never permits raw report/event writes.
Host verification and the runtime's guest-exclusion check are distinct
responsibilities, not promises inferred from a directory name or preset.

The host validator is read-only. A Windows override must check both the child
and provisioned root, reject redirected/reparse paths, and verify a non-null
effective DACL granting data access only to the trusted service identity and
SYSTEM. Broad inherited grants must fail verification. The generic controller
does not modify ACLs or consider a user-profile location sufficient.

Preparation errors, including readiness exceptions and directory creation
failure, end in an explicit error/expired/cancelled state. If qualification never
returned, report `readiness` and `simulated` are `null`, and score metadata says
`simulated: "unknown"`. They never imply a simulated or live run. Error reports
and undetermined scores are retained through working PyRIT memory even if the
directory cannot be used. `directory` is only the intended location, not proof
that a report file exists. Failure of memory retention propagates to the caller;
there is no invented receipt or numeric fallback.
If guest exclusion cannot be verified and environment removal also fails, the
controller propagates that failure without publishing raw content to potentially
agent-visible storage. That case has no report or score receipt.

## True session continuation, not cloning

Stepping is available only when both the qualification and runtime capabilities
declare the same retained session and workspace. It is currently supported only
with the literal baseline. The first instruction uses `PromptSendingAttack`;
subsequent operator instructions use the public normalizer in the same recorded
conversation and native session.

`view().can_step` is false while a native turn is running, after the turn limit,
after TTL expiry, on cancellation and after finalization. `step_async` rejects
unsupported states. `finish_async` stops the agent, grades and cleans up.
`cancel_async` requests termination at the next completed turn boundary, not
individual tool approval or mid-tool interruption. TTL/turn deadlines provide a
separate safety bound; expiration yields an undetermined outcome and owned
cleanup, never an asserted cloned session.

`rerun(request=...)` creates a fresh run with immutable `parent_run_id`; it does
not reuse the old target, environment or conversation. The binding must produce
fresh resources for every context. Runtime environment identity reuse is rejected
along with native session identity reuse, including ancestry through a blocked
child with no runtime. Direct constructor `parent_run_id` is rejected.
Cross-restart reruns are currently unsupported: the backend must retain the
verified parent controller or disable rerun. A client-provided parent ID or a
reconstructed controller without ancestry must not substitute for that state.

## Event fidelity and coverage

The adapter subscribes before the first send and preserves actual event IDs,
observed ordering, native session ID and the unmodified JSON envelope, including
ephemeral usage and idle events. Tool request/start/completion identities and
arguments are correlated. Missing, repeated or mismatched events are explicit
coverage gaps and cannot produce a clean grade.
Execution observed before its model request is a permanent ordering gap even
if an otherwise matching request arrives later; the adapter never backfills
causality.

`NativeToolTrace.model_visible_output` is the actual `result.content`, while
`detailed_output` retains `detailedContent` separately. The complete original
result and error payloads remain in evidence. Unknown exit code, truncation,
provider usage or omitted protocol fields are not invented. Coverage means
**the observed native-session event stream**, not a guarantee of every action
inside an arbitrary CLI or operating system. Original graders can require
additional source-specific coverage.

The target returns authentic assistant/tool messages only. A tool-only result
does not get a fabricated assistant receipt. Events from a failed turn stay in
the session evidence and are retained before owned cleanup.

The controller declares a required, bounded
`harness/jsonl/controller-serialized-sdk-events` stream before sending the
first instruction. Each completed outer turn links real user requests,
assistant messages, assistant function calls and model-visible tool results as
distinct MessagePiece directions, then appends observed SDK events and their
serialized JSONL bytes to PyRIT memory. Tool calls/results never count as a
fabricated assistant reply. This is a **controller serialization of SDK events**,
not the original SDK wire stream or a claim to have observed every guest
process. Sensitive payload/byte reads require an explicit opt-in. The file
`native-events.jsonl` remains a private supplementary copy, not the source
of the database Score.

The stream is sealed and task-required DB coverage is assessed before the
original grader runs. Missing event/tool phases, unsealed turns, omitted raw
bytes or mismatched digests block a clean grade. The scorer only constructs
an unpersisted verdict; memory atomically stores the canonical report, one
Score and episode link, downgrading that Score to undetermined on required
capture gaps. If memory finalization fails, the caller receives the error
and must explicitly recover the unfinished episode; the workflow does not
retry grading or publish a score-shaped fallback.

Chat turns require a genuine non-tool assistant response. A task with
artifact-only output must declare
`NativeCyberResponsePolicy(allow_artifact_only=True)` in its trusted binding
and override `response_mode(*, request, turn_index)` for the intended turns;
missing assistant text does **not** select that mode automatically. An
artifact-only turn must have a persisted request and observed root terminal
event. Final publication additionally requires a completed original judgment
with a retained typed artifact. The pre-grading capture check does not require
that not-yet-acquired judgment.

## GHCP container transport: source-supported path, unqualified authentication

Inspected public SDK source:
[`github/copilot-sdk` at `4001c1da`](https://github.com/github/copilot-sdk/tree/4001c1da7d832c51bad1d38619c1a082af390efb).
The adapter's subset is `session.on`, `send_and_wait(prompt, timeout=...)` and
`disconnect`. It does not initialize an SDK, forward credentials or install host
custom tool callbacks.

At that revision `RuntimeConnection.for_stdio(path, args)` places caller args
before the SDK-managed headless/stdio flags. The following is mechanically
supported for an already-created, qualified container:

```python
args = CopilotSdkAgentSession.docker_stdio_arguments(
    container_id=owned_full_container_id,
    cli_path="/opt/copilot/copilot",
)
connection = RuntimeConnection.for_stdio(path="docker", args=args)
```

This runs the real CLI through `docker exec -i`, with no host shell, forwarded
environment, published TCP port or broad filesystem mount. The caller owns SDK
and Docker context cleanup. PyRIT and the original grader stay outside the agent
image, while CLI, toolchain/dependencies and task workspace belong inside it.

This does **not** qualify authentication. The SDK's ordinary `github_token`
option creates `COPILOT_SDK_AUTH_TOKEN` in the spawned process environment.
Per-session `github_token_provider` returns a token over RPC to the runtime.
Neither proves that a reusable bearer is inaccessible to untrusted tools in
the same guest. Never pass ambient `GH_TOKEN`, credential HOME mounts or token
environment variables into this agent container.

The SDK also exposes `request_handler` for host-side model HTTP/WebSocket
forwarding. This is a potential model backchannel, but token-free CLI bootstrap,
model discovery, exact CLI/SDK compatibility and a constrained authenticated
host proxy have **not been qualified here**. A `--network none` guest cannot
directly reach GitHub or a BYOK provider. Until a source-supported and tested
credential-isolated backchannel exists, readiness must remain blocked.

A Linux CLI distribution exists:
[`github/copilot-cli` v1.0.88](https://github.com/github/copilot-cli/releases/tag/v1.0.88),
asset `copilot-linux-x64.tar.gz`, published SHA256
`42f40c08ff8a8ff78522161e4b5e2b86340ad8bb0853a5f1aa64ce65b48d007b`.
It was inspected, not downloaded or executed by this implementation. A separately
pinned, approved image/layer is still required; the prior minimal Python image
contains no Copilot CLI/toolchain.

The coordinated private pinned-asset probe initially paused on a local
`sec.endpointdlp` marker. This per-file metadata alone does not establish an access
prohibition: later normal policy-aware reads succeeded with the marker intact.
Respect an actual content-exclusion denial; never remove or route around a marker
to evade policy. Permitted local reads or a matching release digest do not qualify
the image, authentication, model backchannel, or an end-to-end GHCP run.

The existing InterCode/GDM task 4 binding is the lowest-effort additional candidate:
one tiny file, no network task dependency and an existing inclusion grader.
It is excluded from a **new live** demonstration for now because its prior image
and host-dispatched Responses harness do not exercise the GHCP container
architecture. Re-running that model-only harness would not establish parity.

## Validation and limits

The focused native tests use inert SDK-event fixtures through the actual target,
normalizer, `PromptSendingAttack`, converter registry, memory and report scorer.
They are not a real GHCP or private task evaluation:

```powershell
uv run --frozen pytest -q `
  tests\unit\prompt_target\target\test_native_agent_target.py `
  tests\unit\executor\workflow\test_native_cyber_eval.py
```

The saved-preset/run/evidence/rerun UI is a dependent surface over this registry,
typed request and capability-gated run view. Raw report/artifact access needs
the operator API's authorization and private evidence policy; the generic view
is not an authorization boundary.
