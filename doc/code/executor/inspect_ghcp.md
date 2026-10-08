# Inspect-owned cyber tasks with a contained GHCP agent

This optional adapter runs **one original text Sample at a time in a GHCP harness**,
not a rewrite of Inspect or a general adapter for Claude, Codex, or arbitrary
Inspect tasks. A trusted-local source can enumerate several Tasks and Samples,
but that new source/Scenario fan-out is **inert until each task profile is separately
qualified**; only the original named benign case has a live protocol proof.
Inspect owns the original `Task`, sample files and setup, Docker Compose
sandbox, model provider, original scorer (called once), and cleanup. PyRIT
selects the `RedTeamingAttack`, adversarial model, feedback scorer, converters,
and next user instruction. The registered Inspect solver replaces only the
task's solver: it does not replace or call the original scorer. The original
task and scorer must be trusted and pinned before they run in the controller.

The solver starts one GHCP SDK process **inside** a nonroot agent container and
keeps one session alive across PyRIT turns. A separate `model-bridge` container
accepts only authenticated, run-scoped `/v1/responses` calls and forwards them
to Inspect's host-held model provider. The agent has no Docker socket, host
mount, GitHub login, provider credential, remote MCP tool, web search, or
general host shell. Inspect's sandbox proxy talks to the trusted controller;
the separately named `target` service remains alive through original grading.
The SDK worker stops and its process exits before the original scorer starts.

## One-click benign Eval Scenario and inert source fan-out

The registered `benchmark.inspect_eval` Scenario selects authored Tasks, not
manually constructed `InspectGhcpTaskBinding` objects or a synthetic victim
`PromptTarget`. After normal PyRIT initialization in an **approved isolated
environment**, set `eval_family` to `benign_protocol` and run
`initialize_async()` / `run_async()` for the unchanged named, single-case smoke.
The Scenario is available from the PyRIT scenario catalog without importing
executable Eval code. It derives one PyRIT objective from each supported,
nonempty text `Sample.input`, retaining the original Task dataset, setup,
scorer and cleanup. `Sample.target` remains original scorer data, never a
second attack objective. For the one qualified live case, the GHCP solver
runs PyRIT's retained-session `RedTeamingAttack` once across two user turns.

No live Task, Docker or provider access occurs during scenario catalog
discovery. Runtime selection requires the already-reviewed `DOCKER_HOST`
child SSH alias and `PYRIT_INSPECT_AGENT_IMAGE` /
`PYRIT_INSPECT_AGENT_IMAGE_ID` settings from the direct pilot. The scenario
uses the fixed `ghcp_protocol_v1` harness profile and host-only
`qwen3_loopback_v1` model route. Unknown profiles, routes, Eval families,
invalid image IDs, undeclared Tasks/Samples/assets, edits to Sample files/
setup/sandbox and authored solver initialization are explicitly unsupported.
These are capability rejections, **not** a license to skip the original
Task's scorer or its protections. The existing `PROTOCOL_SMOKE` binding gate
still refuses a cyber benchmark Score.

The approved single-case Scenario entry point is
`examples.inspect_eval_scenario_smoke`; invoke it from the repository root
with `uv run --project build_scripts\inspect_ghcp_controller python -m
examples.inspect_eval_scenario_smoke` **only after** a new isolated
Docker/model lease has been granted. Module invocation keeps the pinned
authored Task import rooted in this checkout. It uses its own ignored
`.venv\inspect-ghcp\oneclick-protocol.db` so historical pilot evidence is
never rewritten. The existing token audit accepts `--database
oneclick-protocol.db` for that run (the default remains
`protocol-smoke.db`). Both the Scenario and audit print only safe
identifiers/counts; the original `.eval`/SQLite source data remains private.

An operator may instead select `trusted_eval_dir` with `trust_local=True`
and `eval_revision` equal to the SHA256 of that directory's exact
`inspect-eval.json` bytes. The local directory contains explicitly trusted
Python code; validating a manifest does **not** make arbitrary Python safe
to import. V1's manifest declares `schema_version: 1`, a public `name`,
relative `task_file`, `factory`, `files_sha256` (including the factory),
`task_name`, `task_version`, `sample_id`, `sample_input_sha256`,
`sample_target_sha256`, `scorer_name` and its registry-parameter
fingerprint, the ordered `setup_fingerprints`, `cleanup_name`, a bounded
target-side `health_command`, and whether `allow_initial_input_override`
is enabled. These setup/scorer/cleanup fingerprints are checked again
against the materialized Task before a case starts; an authored solver
with initialization or nondefault model/checkpoint configuration is
outside V1. V1 rejects parameterized setup/scorer callbacks whose
arguments could refer to further unpinned grading or preparation files.
The schema-1 named benign manifest and its source/case/sandbox hashes remain
unchanged.

An explicitly trusted local directory can instead declare a SHA-pinned
`schema_version: 2` manifest with an ordered `tasks` array. Each entry
declares `task_name`, `task_version`, the original scorer/setup/cleanup
fingerprints, and an ordered `samples` array of IDs, original text-input
SHA256s, and target SHA256s. The finite inventory admits 2 to 32 cases across
at most eight Tasks and 16 Samples per Task. Task and per-Task Sample IDs
must be unique. The factory receives `agent_image` and `target_image`
keyword arguments and must materialize **exactly** those Tasks and Samples;
it never silently selects `dataset[0]` or deduplicates identical inputs.
All Tasks, their full datasets, callbacks, pinned files, effective Compose
sandboxes and declared SHA256 image IDs/tag bindings are checked before any
case starts. Actual running image IDs are verified only for an authorized
live runtime; a schema-2 inventory alone does not attest Docker images.
The Scenario creates one independent case-run identity and task-owned work
item per Sample in manifest order; they share one source/profile fingerprint
and run serially, with no automatic retries or resume. Multi-case
`initial_user_input` overlays are rejected until an explicit single-case
selector and run variant contract are qualified. The case executor
**refuses to launch schema-2 Tasks** without separate runtime qualification;
unit tests use an inert per-case committed-Score stub, never Docker or a
model. No new live benchmark claim follows from code-only fan-out.

The resolver checks declared source file hashes **before** importing a
trusted local factory; it rechecks source files and each Task's complete
inventory before the Scenario starts cases, and each case is rechecked
again at execution. Symlinks, missing declared files, extra returned
Tasks/Samples and per-Sample overrides fail closed. This is *not* a sandbox
for arbitrary Python imports: the operator must trust the local factory
and any code it loads; no URL-based loader or network fetch is provided.

When that source allows it, `initial_user_input` replaces only the selected
text `Sample.input` (max 32,768 UTF-8 bytes). The original input SHA and
source package SHA remain unchanged; `InputVariantRef` records a separate
overlay digest in the scenario/spec identity and the already-committed Score
metadata. This is a labeled GHCP Task **variant**, not unchanged benchmark
parity. Solver init, scorer, target image, Sample files and sandbox are
never user-editable through this parameter. The schema-1 Scenario runs one
sample; schema-2 fan-out does not permit overlays. Both forbid
baseline/retry/resume and run only one case at a time.

For the qualified live case, the original Inspect ScoreEvent is verified
once against the selected Task/Sample/epoch/scorer and sealed raw EvalLog.
The evaluator labels its single **pre-committed** PyRIT Score with the typed
`EvalScoreProvenance(BENCHMARK_ORIGINAL, case_run_id, mapper_hash)` before
the same atomic DB transaction that stores the report. Here
`BENCHMARK_ORIGINAL` identifies the **source** of the benign lifecycle
marker, not a qualified cyber achievement: the Score and TaskOwned
`AttackResult` remain UND/NULL. Each `TaskOwnedAtomicAttack` re-reads its
own exact Score ID and links it via FK, without creating a second Score or
calling the original scorer again. The provider/model and GHCP SDK/token
safety limits below remain unchanged.
The benign pilot's PyRIT `RedTeamingAttack` also persists its two
`SubStringScorer` impossible-canary **false** control Scores, attached to
the two MessagePieces. They have no `pyrit_eval_role` metadata and are
neither benchmark verdicts nor typed progress achievements; the
TaskOwned AttackResult links only the one original-report UND Score.
No ManualScorer stop sentinel was observed in the accepted one-click run.

## Mode 1: import an original Inspect run without PyRIT steering

`InspectOriginalEvalImporter` reads a **preexisting `.eval`** directly, without
loading its authored Task, solver, sandbox, model, or source factory. An
initialized PyRIT memory backend is required:

```python
from pathlib import Path

from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter
from pyrit.memory import CentralMemory

imported = await InspectOriginalEvalImporter(memory=CentralMemory.get_memory_instance()).import_eval_log_async(
    path=Path("approved-local-original.eval")
)
```

The importer stores the **exact original `.eval` bytes and SHA256** plus a
separately resolved typed `EvalLog` (including attachments) in owner-controlled,
sensitive PyRIT raw streams. It projects original `EvalSample.messages` text and
assistant function calls/tool replies to
MessagePieces with a **distinct conversation per Sample/epoch**, and stores
typed Inspect events in original retry-attempt and sample order with observed
UUIDs when present. Original model/tool/score events, errors, intermediate
scores and final `EvalSample.scores` remain source evidence, not synthetic
PyRIT attack turns. ModelEvent output is not inserted a second time as an
assistant message. Calls use the existing `function_call` convention, and
replies use `function_call_output` with the exact original call ID. Mixed text
and multiple calls retain their original message/part order; tool errors retain
their original source metadata. CoPyRIT labels calls and replies separately and
renders their arguments/output as read-only data, not executable content.
Unsupported content and calls without source IDs stay in the resolved log and
are reported as a projection gap. The original `.eval` is the only source of truth for
offline imports; `EvalSourceFactory` and task materialization are **not**
required. A caller can supply a matching `EvalCaseRef` inventory and
`EvalRunRef` to label source case-run identities, but this does not qualify an
external scorer.

New imports use projection schema 3 (native binding version `2`), which is part
of the import identity. Existing sealed schema-2 imports (binding version `1`)
remain readable using their original text-only projection, including the old
Score/AttackResult IDs. `InspectProjectionVersion.TEXT_ONLY` explicitly selects
that historical import schema. Reimporting the same archive with the new
default creates a distinct projection; it never backfills or changes an old
sealed episode, conversation, source archive or result pair. No database
schema or Alembic revision changes are required.

An ungraded `inspect-original` episode is sealed only after DB byte/digest,
sample, event-order and MessagePiece-link checks. `coverage_complete` says
the **Inspect-visible log** was retained and projected, not that an external
CLI, OS action, provider request, or benchmark result was independently
attested. An error, incomplete attempt, missing event ID, unmatched final
ScoreEvent or truncated source remains explicitly incomplete. The **episode**
remains unscored (`episode.score_id is None`), while the offline importer
persists one separately linked PyRIT `Score` and `AttackResult` per retained
Sample/epoch in `imported.case_results`. These are projections of the
**original** Inspect scorer, not a second call to a Task, solver or scorer.
They include the exact archive digest and Inspect run/eval/Sample identity;
when a final ScoreEvent matches, its ID and typed-event digest are also
recorded. A case-run ID is included when matching `EvalCaseRef`/`EvalRunRef`
bindings are provided. All intermediate, retry and additional final scorer
events remain in the retained raw and typed streams.

For one final scorer, its score is the primary by default. For several,
provide an `InspectOriginalScorePolicy` naming the **reviewed** primary
scorer and the exact original Task name/version; without one, the PyRIT
Score is UNDETERMINED rather than silently choosing a grade. **Every**
declared final sample score must match exactly one non-intermediate final
ScoreEvent in the final attempt, even when it is not the primary. Matched
booleans become PyRIT true/false values and finite numbers in `[0, 1]`
become float-scale values; unsupported values, missing/mismatched events,
incomplete Sample or run-level capture and missing selected primaries create
a value-less UNDETERMINED Score. The original value still survives in the
exact `.eval`.

Even a COMPLETE imported Score leaves `AttackOutcome` UNDETERMINED until a
caller supplies **both** a matching source case inventory/run and a
task-specific `InspectOriginalScorePolicy` with `success_direction` set to
`InspectSuccessDirection.AT_LEAST` or `AT_MOST` and a finite
`success_threshold` in `[0, 1]`. Only then does that explicit threshold
map a verified scalar to SUCCESS or FAILURE. Equivalent numeric thresholds
(for example, `1`/`1.0` or `-0.0`/`0.0`) share one canonical import identity.
An actual recorded Inspect Sample/run exception maps to ERROR with an
undetermined Score, never to defender FAILURE; a non-success status
without an exception remains UNDETERMINED. An offline AttackResult has
zero executed PyRIT turns; the original Inspect turn count remains
source metadata when available.
Offline Scores are source-attributed, **not** task-owned
`benchmark_original` scorer attestations or independent external-activity
proof. The allowlisted live original Task runner projects that same archive
after original cleanup to obtain its separately linked Score/AttackResult;
it never invokes a second scorer.

Re-import with the same archive, case bindings and policy rechecks the
exact source bytes, run/Sample coverage derived from the typed source,
every original event's capture timestamp, typed event payloads, and
persisted Score/AttackResult links without rerunning Inspect or creating
duplicates. The synthetic per-Sample summary has no original event
timestamp to check. Missing or tampered projection rows fail closed.
An original run-level gap applies to every Sample: moving it into one
turn's stored gaps cannot make another Sample's Score complete. Each
stored turn gap must match that Sample's evidence or observed native
tool-capture gap.

Score and AttackResult rows are committed atomically; an interrupted
import after the evidence episode is sealed but before projection
publication requires explicit reconciliation rather than an automatic
repair.

For an unchanged-runner proof, `run_original_inert_eval_async(memory=...,
log_dir=...)` invokes only the separately SHA-pinned public
`inspect_original_inert` Task. That Task's authored setup, solver, scorer and
cleanup run in Inspect without a sandbox, without a model-generation call and
without replacing a solver or running PyRIT attack turns. The `mockllm/model`
identifier is Task metadata for this **inert** fixture, not a model answer.
Import the runner with
`from pyrit.executor.benchmark.inspect_original_runner import
run_original_inert_eval_async`; provide an initialized PyRIT memory instance
and an existing, private local `log_dir`. Inspect writes its original `.eval`
there before PyRIT imports it. Local `file://` log URIs are accepted only
after their resolved path is checked inside that directory; remote URIs and
network shares are rejected.
This entry point is **not an arbitrary-Task runner**: other Task profiles,
Docker Compose guests, network models, credentials and private cyber Tasks
remain unapproved. The existing GHCP benign case retains its separately
reviewed execution path and protocol-only gate.
Only this public runner's source pin uses Git's LF-normalized Python bytes
so the same checked-in file remains approved in CRLF Windows worktrees;
the original `.eval` archive is always hashed and stored **byte for byte**.

### One-click public Mode 1 entry

With the backend configured for PyRIT SQLite, CoPyRIT's Scenario catalog
automatically discovers `benchmark.inspect_original_inert`. If unrelated
model-backed catalog introspection fails without credentials, the catalog
still links to this Task's independently readable detail page. That page
shows the sole approved Task ID, `inspect_original_inert`, and a **Run original
Inspect Task** button; no target, Python code, path, model, sandbox, secret, or
scorer editing is exposed. The same registry-backed action is available via
`POST /api/scenarios/runs`:

```json
{
  "scenario_name": "benchmark.inspect_original_inert",
  "scenario_params": {"eval_family": "inspect_original_inert"}
}
```

The Scenario checks the pinned public source and profile before allocating a
fresh case-run ID and private local `.eval` directory. It calls the unchanged
runner once and checks its live evidence. **After** the original setup, solver,
scorer and cleanup, the strict offline importer reads that same `.eval` with
the approved case/run inventory and `original_inert_scorer` as its primary
scorer. The Scenario requires identical archive/run/Sample identities and a
linked, persisted source Score and AttackResult before marking the run complete.
No success direction or threshold is supplied: the source-attributed PyRIT
Score is COMPLETE with the original value, while the AttackResult outcome is
UNDETERMINED, not SUCCESS or FAILURE. The original live episode remains
unscored. A successful run removes its temporary disk log; a failed run retains
its unique local log for reconciliation.
No automatic retry or resume is allowed. The backend rejects other Task IDs,
source paths and URLs, arbitrary request fields, target/model/sandbox settings,
initializers (including secret-bearing arguments), labels, overlays, datasets,
and extra execution techniques **before** running any initializer or Task.

`GET /api/scenarios/runs/{scenario_result_id}` and the run-progress endpoint
expose `original_inspect_import` after successful projection: source SHA256,
case-run ID, Inspect run/eval IDs, live and offline evidence episode IDs, exact
archive SHA256, and the linked PyRIT Score/AttackResult IDs. The fields
`score_status: "complete"` and `outcome: "undetermined"` are deliberately
distinct. CoPyRIT displays the original scorer value and a link to the
source-attributed AttackResult instead of a misleading attack-success rate.
The offline AttackResult is not a task-owned `benchmark_original` attestation
or a second Scenario attack; its original `.eval` remains the source of truth.
The Scenario progress read model counts the one planned case as completed
**only** when its persisted verified import matches the plan's run ID, source
SHA and case-run ID. Readback also checks the fixed public source SHA and
recomputes the planned `EvalCaseRef.case_id` from the verified `.eval` Task
name/version and Sample ID/epoch, without executing a Task factory. The
planned objective text and SHA256 must equal the verified typed Sample input;
extra prompts or input variants are not approved. It reports 1/1 completed,
zero successes and no success percentage; this is import completion, not a
success verdict or a synthetic Scenario AttackResult. The history API
validates the same persisted run and case metadata before counting 1/1.
Detail, progress and history also read back the referenced Score,
AttackResult and sealed offline `.eval` episode: their foreign key, source
case/run/archive metadata and undetermined outcome must still agree. The
**entire required-stream set** must be exactly the approved
archive (`harness` / `eval_log`) and resolved typed log (`harness` / `jsonl`).
An extra missing required stream cannot be excused by stale coverage metadata.
Both bounded private streams are re-read through the integrity-checking
memory reader: archive length/SHA256 and resolved-log bytes/length/SHA256 must
match the verified `.eval`. Its typed final ScoreEvent ID, event hash and value
must match the linked Score, and the projected native event stream is checked
against that `.eval` again. The imported Score and AttackResult timestamps,
Sample UUID, original objective and optional original turn count must match
the same typed Sample, including whether the turn count was present. Projected
MessagePieces must retain the Sample's exact text, role, order and source
metadata; matching mutable link digests and piece IDs alone is insufficient. The
AttackResult must point to the deterministic imported conversation for that
verified Sample, with no last-response or related-conversation links and no
extra conversation MessagePieces. Direct attack detail, messages,
conversations and list reads resolve its ID through the Scenario's independent
persisted import reference and rerun the same typed-evidence verification,
rather than trusting matching Score/AttackResult metadata. Missing or
ambiguous import bindings fail closed, including removed source markers.
The imported original result cannot be edited through generic attack PATCH,
human-score deletion, manual scoring, message send/storage or conversation
branch/promote routes; rejection occurs before any scorer, target dispatch or
write. Ordinary attacks retain these actions. Human judgments for the
original run need a separate, explicitly approved representation.
Raw bytes never enter the REST response. Missing or substituted IDs or altered
event/stream bytes, planned cases or conversation links fail closed without
rerunning the Task or inventing another result. Detail and history return
`objective_achieved_rate: null`, and CLI output says "undetermined"; other
Scenarios retain their numeric rate. A rejected `.eval` leaves the planned
case incomplete. Detail, history and the progress header expose only a vetted
failure diagnosis (or a safe generic failure message), while full exception
details remain in backend logs for reconciliation.
Without separate server admission, internal/private Tasks, arbitrary Python
and cyber Evals are not selectable.

### Server-admitted original preview (disabled by default)

`benchmark.approved_original` is absent until the ordinary backend lifespan
installs its trusted supervisor and evidence provider. Setting the
server-owned `PYRIT_ORIGINAL_WORKER_CONFIG` and
`PYRIT_ORIGINAL_WORKER_CONFIG_SHA256` opts into one hash-bound approved
source/profile. Leaving both unset preserves ordinary CoPyRIT behavior.
The config is a closed `CohostBackendConfig`, not a browser-authored command
or general private-Task catalog.

The frontend, API, native model relay and one trusted worker **child process**
share one host/ACA replica. Only the evaluated sandbox is separate and must
receive no real model, database or provisioning credentials. The trusted
worker may use the preview identity and necessary credentials. It receives
fresh HOME/app-data/cache/temp/result roots before imports and initializes
its own scratch SQLite before importing private Task code. The web process
does not import that Task, solver, scorer or private Scenario. Protected
Scenario classes stay excluded from public discovery and direct launch.
There is no VM/UID/IMDS isolation claim for this trusted child.

For an authenticated operator whom the host runner explicitly admits, the
facade returns only a safe profile reference, the fixed `evaluated` model
role, readiness and finite unmet-condition codes. When ready, its detail
page gets a short-lived, single-use, server-issued `admission_ref`, bound to
that operator and the approved profile. A request contains exactly:

```json
{
  "scenario_name": "benchmark.approved_original",
  "original_admission_ref": "<server-issued-one-use-reference>"
}
```

The backend consumes the reference before allocation and rechecks actor,
group, source qualification, immutable configuration and one-slot capacity.
The supervisor prepares, starts, waits for, cooperatively cancels and observes
the exit of one fixed hashed worker executable. App/job/control UUIDs are
distinct from the framework original-run UUID. The worker executes unchanged
original setup, solver, scorer and cleanup once. After its real exit, the
backend independently validates fixed artifact names, byte/hash bindings,
actual process identity, typed ModelCall correlation, relay drain and exact
physical closure. Release observes cleanup; it does not run another
destructive cleanup command.

The supervisor passes the exact `.eval` and authorized envelope to the same
`OriginalEvidenceService.intake_async` used by authenticated HTTP intake.
Only the backend writes canonical memory and result storage. It never opens
worker SQLite or copies its rows. Its strict offline projection creates one
linked source-attributed Score/AttackResult, not another scorer invocation.
Later views recheck retained authority and source-derived projection.
An original scalar is displayed only under the reviewed display policy.
Null success direction/threshold leave AttackOutcome UNDETERMINED even when
the authentic original Score is COMPLETE.

Detail, progress and history return a redacted `original_source_result`
with the allowed original grade, coverage, PyRIT Score status,
undetermined attack outcome and **separate** cleanup status. Authorized
canonical attack detail/conversations/messages are read-only. Private Task
configuration, raw Scenario plans and evidence streams are not REST payloads.
A source grade without proven cleanup is
`cleanup_uncertain` and 0/1 completed; no Sample or source score despite
proved physical closure is `failed_ungraded`, also 0/1. A cancelled run
requires an exact job abort and closure observation; lack of either proof
fails visibly, never as a completed evaluation. Physical absence alone
does **not** imply that the original Task reached its final scorer.
An authenticated original grade retained after incomplete execution is
`failed_source_verified` with 0/1 completed, not a success claim.
Cancellation accepted before canonical handoff revokes new relay posts
immediately and cannot publish a grade, including numeric zero. Once the
exited original source enters irreversible canonical publication, cancellation
returns HTTP 409 rather than falsely accepting cancellation of completed work.
Application shutdown also waits for this irreversible handoff instead of
cancelling publication; earlier shutdown cancellation drains the owned worker
and releases the scheduler slot before memory closes.
While verification/publication is pending, status/history expose no source
result or grade. Failed publication remains a readable failed run; incomplete
canonical projection is not available through source-result detail.
After a web-process interruption, the projected job is marked failed with
cleanup unverified; a later conflicting broker proof requires explicit
reconciliation rather than retroactively assuming completion.
Receipt IDs, Task names, case IDs, private paths, endpoints and
credentials remain on the host.

The public harmless regression exercises the actual ordinary app/API,
supervised child, unchanged original Task, exact archive, fresh canonical
SQLite, linked Score/AttackResult and read-only view. It requires no model,
network or sandbox and uses a mocked authenticated actor, not live Graph.
This code proof is not hosted SQL/MI, private-worker qualification or a
deployed GUI proof. A physical cleanup receipt after a pre-Sample failure
is still not a source grade. The disabled public default makes no resource
or model call.

#### Startup and private adapter contract

Use one backend ASGI worker/ACA replica. `.pyrit_conf` must set
`memory_db_type: azure_sql`, `env_files: []`, `initialization_scripts: []`,
`initializers: []`, `max_concurrent_scenario_runs: 1`,
`enable_live_reinitialization: false` and `allow_custom_initializers: false`.
Key Vault environment reload is not admitted. Set:

| Process setting | Required authority |
| --- | --- |
| `PYRIT_CONFIG_FILE` | Absolute immutable backend configuration path |
| `PYRIT_ORIGINAL_WORKER_CONFIG` / `_SHA256` | Absolute closed JSON descriptor and SHA256 of its exact bytes |
| `AZURE_TOKEN_CREDENTIALS` | Exactly `ManagedIdentityCredential`, with azure-identity >=1.24 |
| `AZURE_CLIENT_ID` | Exact newly owned preview UAMI client/application ID |
| `ENTRA_TENANT_ID`, `ENTRA_CLIENT_ID` | Approved tenant and this instance's SPA client ID |
| `ENTRA_ALLOWED_GROUP_IDS` | Approved groups, including the source policy groups |
| `ENTRA_ADMIN_GROUP_ID` | Approved separate administrator group |
| `PYRIT_ALLOW_UNAUTHENTICATED_ADMIN` | `false` |
| `AZURE_SQL_DB_CONNECTION_STRING` / `_PROD` | Identical native-token Driver18 URL, explicit owned database, encryption and certificate verification |
| `AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL` | Exact owned result container HTTPS URL |
| `AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN` | Absent/empty; no shared SAS fallback |

The descriptor binds the final public wheel commit; fixed absolute
`worker_python`, `worker_entrypoint` and entrypoint SHA256; Python3.12 and
qualified worker Inspect0.3.247; source root/contract digest; jobs root;
opaque source/profile aliases; one exact source package/case/primary scorer;
reviewed display values; actor/group allowlists; MI client ID; database and
result container; exact owned sandbox group/subscription; and the validation
instance's expiry, private state container and frozen authority-key digest.
The hosted preview must explicitly set `relay.max_completion_tokens` to `8192`
to preserve the accepted original `GenerateConfig(max_tokens=8192)`.
The generic relay default remains `4096`; it is not the approved private worker
profile and cannot admit hosted original startup. Do not lower the unchanged
worker's request, introduce a harness variant, or reuse a `4096`-based completion
quote as the bound for this profile.
Backend and worker must report the **same final PyRIT compatibility ID**,
even though their qualified Inspect versions differ.
The source descriptor must declare `qualified-updated-public-wheel`, that
final public commit, exact package/model/harness/case/policy and sorted display
values. It is data-only qualification, not permission to import private code
in the web process.

Run Alembic under a **separate schema actor** before starting runtime:
set `AZURE_CLIENT_ID` to that actor's client ID, keep the exact owned SQL/result
settings, then run PyRIT's native migration entry point:

```text
uv run python -c "from pyrit.memory import AzureSQLMemory; from pyrit.memory.migration import run_schema_migrations, check_schema_migrations; m = AzureSQLMemory(skip_schema_migration=True); run_schema_migrations(engine=m.engine); check_schema_migrations(engine=m.engine); m.cleanup()"
```

This deliberately runs upgrade plus strict schema comparison under the schema
actor; it is not a runtime command. PyRIT binds Alembic to its native-token engine
programmatically and has no `pyrit/memory/alembic.ini` CLI configuration.
Detach that actor before runtime DML. The backend explicitly initializes
AzureSQLMemory with `skip_schema_migration=True`, requires exact single head
and `alembic check` compatibility, and fails rather than accepting a schema
warning. Runtime gets necessary DML/metadata visibility, never CREATE/ALTER.
A SQL contained identity must use the UAMI client/application ID for its
TYPE=E SID; a principal/object-ID SID is not interchangeable.

Actual standard sync/async SQL and result-container credentials are observed
before readiness, with the same frozen MI-only environment checked before
construction/use/refresh. Only bounded client_id/oid/tid/aud/xms_mirid claims
are retained; no bearer, HTTP DEBUG logs or private response payload.
The relay observes the same identity on its actual model-token path.
Provisioning receipts must independently bind the observed OID to this
instance's UAMI. A direct pyodbc probe is not proof of these standard paths.

Worker stdin is one JSON line <=64KiB: ABI1, app/job/control UUIDs, actor,
source/profile aliases, source/per-job manifest digests, absolute source/run
roots, absolute UTC active/cleanup deadlines, timeout values, loopback relay
URL and ephemeral capability. Active time is at most1500s; cleanup gets the
original additional300s, never a fresh clock. `cancel.request.json` contains
ABI1/job/control IDs. Stdout is one <=4KiB terminal with
`state: success|error|cancelled`; stderr is drained within16KiB without logging
its private contents. Fixed outputs are `source.eval`,
`backend-intake-envelope.json`, `worker-scenario.json`,
`source-manifest.json`, `runner-closure.json` and `worker-provenance.json`.
The backend rejects arbitrary filenames, substituted IDs/digests, second
allocation and unverified exit/closure.

#### Relay, restart and acceptance boundaries

The native relay has only capability-authenticated POST completion/close
routes for the exact job UUID. The worker sends `PyRIT-Compatibility-ID`
from its installed same-wheel `pyrit._compatibility.get_compatibility_id()`
on completion and close requests, never a marker copied from the backend.
It fixes `pyrit-github-pipeline`, `gpt-4-32`, `gpt-4o` version
`2024-11-20`, `Microsoft.Default` and API `2024-10-21`.
No retries, alternate endpoints/models or streaming. Limits are512KiB
request/2MiB response,8192 completion tokens for the explicitly configured
hosted original profile,180s generic dispatch including
credential acquisition,10s authenticated body read, one inflight request,
one-second minimum spacing and20000 observed tokens/minute.
Each job permits at most50 attempts/100000 observed tokens; the validation
pair permits100 attempts/200000 observed tokens. These are observed stop
thresholds, not invoice ceilings. An authentic final response may overshoot
a threshold; it is retained unchanged. Unknown/unmetered dispatch blocks
later admission. HTTP disconnect/cancellation does not prove upstream drain.
The child's first same-clock qualification may request only `Reply only OK`
with max_tokens8, at most once per job, and consumes the same budgets.

A selected profile can explicitly set `relay.request_timeout_seconds` to60s
without changing the original8192-token Task or the private caller's90s read
and10s connect/write/pool limits. Its owned-task drain wait is nominally65s,
not an end-to-end bound: evidence-file and final budget persistence occur
outside the dispatch timer, and the authenticated close body has its own10s
bound. The generic180s policy remains a different profile and must not be
silently relabeled. Freeze the selected descriptor and policy separately
before consumer qualification; this timing configuration alone is not hosted
admission.
If the drain wait times out or its waiter is cancelled before positive
closure is observed, the relay permanently latches
aggregate uncertainty, owns its durable uncertainty commit independently of
that caller, and refuses later runs even after persistence settles or a
restart restores the budget. Authentic late token usage is still counted;
it cannot clear the unverified-close latch or establish complete closure.
Persisted dispatch intent remains unresolved through metering until positive
close commits, so this restart barrier does not depend on corrective writes
succeeding. Persistence of an already observed positive close is also owned
independently of its caller; caller cancellation cannot cancel that commit.
Missing evidence/budget persistence or lost Blob lease likewise cannot
publish a verified drain or reset prior dispatch intent.

The opt-in initial validation instance admits cancel-first plus one clean
success, not a permanent product-wide two-run limit. Bootstrap fresh
private state **offline**:

```text
uv run python -m pyrit.backend.services.original_worker_state --config <new-unbound-config.json> --bound-config-output <new-bound-config.json> --private-state-output <new-private-state.json>
```

Upload that private packet once with If-None-Match:* to
`instances/<instanceUUID>/state.json` in its separate owned private state
container. It contains an authority key: never publish it. Runtime acquires
a60-second lease, renews every20s and conditionally commits ETag+lease state
before reservation, child spawn and every model dispatch. Missing/modified
state, uncertain commits, lost/expired lease or incomplete prior job fail
closed. Runtime does not create/reset state or recycle failed admissions.
Retained envelopes and budgets survive restart without copying worker rows.

Source display values and operator/group allowlists are unordered memberships.
Their JSON serialization sorts only those fields, so bootstrap, fresh
interpreters and retained-policy restoration compute the same fingerprint
without pinning `PYTHONHASHSEED`. Source cases and signed job/history lists
keep their original order. A changed member, scorer, case, native identity,
durable target or relay limit still changes policy identity and refuses
foreign signed state. A packet from an older, differently fingerprinted
policy is not migrated or re-signed at runtime; create and bind a new owned
validation scope for the final qualified source/wheel.

Run the normal backend, not a standalone harness:
`uv run uvicorn pyrit.backend.main:app --host 0.0.0.0 --port 8000 --workers 1`.
Select the approved alias in ordinary CoPyRIT and launch with its one-use
reference. Read status/history under the approved actor, then canonical
`/api/attacks/{id}`, `/conversations` and
`/messages?conversation_id=<canonical-id>`. Readback derives piece/tool
counts, role/value order, hashes, final scorer identity/value and canonical
links from **this exact typed archive**. It must preserve genuine COMPLETE
or value-less UND semantics. The retained e73 fixture's20 pieces/nine
calls/nine replies/1.0 are only that offline regression, never hosted
acceptance constants or a reason to retry a Task.

Export safe receipts/hashes plus separately protected original archives,
canonical database and owned result/state containers before the fixed
24-hour expiry; preserve actual identity, model usage/drain and physical
closure evidence. Revoke new admission and drain the owned child/relay
before exact journal-bound instance cleanup. Never modify/delete shared
CoPyRIT apps, environments, identities or retained fixtures.
Private same-wheel qualification and real hosted Graph/MI/SQL/Blob/model/
sandbox/browser acceptance remain deployment gates, not claims from
offline tests.

### Authenticated retained-evidence intake and read-only viewing

A separately approved host adapter may install
`install_original_evidence_provider(provider=...)` at trusted backend startup.
This capability is **absent by default** and is independent of runner/catalog
admission: retaining an already completed case does not authorize another
Task, model call, lease or deployment. The web process never imports the
private Scenario, Task or scorer. It accepts an explicitly authorized copy of
the original evidence, not a worker SQLite file or a private filesystem handle.

The worker-only endpoint is `POST /api/internal/original-evidence/{job_ref}`.
It requires a distinct short-lived upload bearer, a base64url JSON
`X-PyRIT-Original-Envelope` header and an `application/octet-stream` body
containing the **exact** original `.eval`. A legitimate pre-Sample startup
failure has an empty body and no fabricated archive/Score identifiers.
The envelope binds the app job, operator, approved profile, original generated
run-instance UUID, pinned source/worker manifest, original Scenario snapshot,
evaluated model-role receipt, final ScoreEvent and **distinct**
terminal-operation and physical-closure receipts. The original run UUID is
not an app-job or provider-lease UUID. The adapter must authenticate these
receipts independently, atomically consume the actor/job/source-bound
capability across backend processes, and permit only exact immutable replays.
Body-supplied hashes are not authority. The upload token is never forwarded
to Graph, and this header is not part of the browser CORS allowlist.

The backend checks archive length/SHA, typed Task/Sample/epoch, final original
ScoreEvent and the server-resolved source/scoring policy. Its event digest is
`config_hash({"event": final_event.model_dump(mode="json", exclude_none=True)})`;
the envelope digest is
`config_hash({"original_evidence": envelope.model_dump(mode="json", exclude_none=True)})`.
No success threshold is admitted. Only source completion, a unique final
original score, authenticated operation-terminal proof and independently
proved physical closure permit the canonical grade import. Otherwise
`capture_only=True` retains the archive/messages under a distinct import
identity with **zero** Score/AttackResult rows. Physical cleanup can be proved
while execution is `failed_ungraded`; unproved containment is
`cleanup_uncertain`, not a fabricated grade.

The configured **backend-owned** `MemoryInterface` imports the original bytes
and derives its own canonical Score/AttackResult IDs. Worker IDs are provenance
only. An independent persisted Scenario reference binds the exact pair,
original archive/event digests and projected messages. Publication is staged
with `persistence_verified=False`, checked by exact readback, and finalized
only after the persisted source agrees. Provisional, ambiguous or tampered
imports remain unreadable and require explicit reconciliation, not reexecution.
For Azure SQL, a separate provisioner runs migrations; the backend DML identity
opens with `skip_schema_migration=True` and checks
`check_schema_migrations(engine=memory.engine)` before serving. No new ORM
columns or Alembic revisions are needed; no SQL credentials reach the worker.

Existing attack detail/messages/conversations and Scenario history/progress
routes recheck the authenticated actor and source binding. Stored provenance
and raw evidence remain intact, while API views expose only approved
correlation metadata and an allowlisted scalar display value. Unsupported
categorical/structured source scores remain exact in the archive and map to
UNDETERMINED rather than numeric success. An original score whose scalar is
not approved for display is shown as recorded, not as missing or ungraded.
CoPyRIT marks the source conversation read-only, renders original function
calls/results with their source IDs, and disables editing, manual scoring,
branching and sends even if another usable target is selected. Viewing or
exporting does not rerun the evaluation.

This public contract and harmless HTTP/SQLite fixtures do not supply or
qualify a private production admission provider, shared OS/credential
boundary, SQL deployment or live execution. A local owner-authorized retained
view is not Entra/SFI certification or fresh launch authority. Hosted adapters
must supply their own reviewed authentication, isolation and durable receipt
verification.

The runner optionally captures `Hooks.on_sample_event` and `on_sample_end`
into a bounded run-scoped stream before the final `.eval` is read. Inspect
emits these callbacks only for completed events, and hook exceptions are
warnings rather than evaluation failures. A process-wide hook instance stays
default-off outside that run and checks Inspect run IDs. Hook event coverage
is reconciled with the final typed log and differences are **optional gaps**,
never evidence of complete external execution. Offline import never
registers the live hook. The importer bounds each original archive to
16 MiB, its resolved typed log to 32 MiB, optional live frames to 2 MiB,
and one import to 32 Samples. The ZIP preflight also limits uncompressed
archive members to 64 MiB across at most 256 members before Inspect parses
them. Re-logged duplicate Sample ZIP members retain exact raw bytes but
cannot claim complete projected event coverage. These are private memory
bytes, not public output; do not publish raw `.eval` files or their
sensitive transcript data.

### Opt-in local evaluation job port

`pyrit.models.evaluation_job` defines a versioned **PyRIT-owned** queue
contract. It is not an Inspect or sandbox-platform protocol. Immutable requests
contain source/case/run/attempt references, an installed runtime kind and an
execution-profile digest. Initial input stays in the source. Requests cannot
supply Python, URLs, model/sandbox settings, secrets, or additional Mode 1
adversarial controls.

The producer, receiver, runtime and canonical writer protocols live in
`pyrit.executor.jobs.port`. `LocalEvaluationJobPort` uses a separate local
`queue.sqlite` journal and an exclusive owner lock. An exact actor-bound
redelivery is a duplicate, not another original execution. Changed same-ID
requests and different job/attempt aliases for the same source case/run are
rejected. The receiver returns `started`, `duplicate`, or `busy`; a broker must
not settle a busy delivery. None of these acknowledgments is a grade.
Previously dispatched attempts found after restart become `interrupted`.
Unknown or failed closure blocks new dispatch; there is no automatic retry or
success-shaped recovery.

Only `PublicOriginalInspectJobRuntime` is installed by the local factory.
It runs the existing SHA-pinned, model-free `inspect_original_inert` Task
unchanged, in independently disposed scratch SQLite. Exact original binary
bytes and a fenced manifest cross to `OriginalInspectArtifactWriter`, which
imports into canonical memory and verifies linked Score/AttackResult/archive
readback. No worker database rows are merged. The original grade is `1.0`;
without a reviewed success threshold, the AttackOutcome remains UNDETERMINED.
Job `succeeded` means source completion and canonical import, not benchmark
success.

Run the actual local CLI example from the repository root, using a **new**
absolute local evidence directory:

```powershell
uv run --no-sync --offline python -m examples.evaluation_job_inert --root C:\local\public-evaluation-proof
```

The example requires the existing Inspect optional dependencies. It does not
restore dependencies, call a model, contact Azure, or build an image. It retains
`canonical.sqlite`, the separate queue journal, scratch evidence, original
logs, and immutable named artifacts/manifest. Do not publish those runtime
artifacts merely because this source fixture is harmless.

The additive backend routes are disabled unless startup explicitly configures:

```text
PYRIT_EVALUATION_JOB_BACKEND=local
PYRIT_EVALUATION_JOB_ROOT=<absolute local owner directory>
PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS=<explicit authenticated operator UUIDs>
```

The backend requires canonical SQLite, one worker/replica, and no simultaneous
original-preview backend. The routes still require a server-authenticated
operator when ordinary authentication is disabled; setting an actor header or
an unauthenticated-admin option does not grant job access. Changing startup
job settings or canonical memory requires a restart, not live reinitialization.

`GET /api/evaluation-jobs/catalog`, `POST /api/evaluation-jobs`,
`GET /api/evaluation-jobs/{job_id}?after_sequence=...`, and the corresponding
`/cancel` and `/control` endpoints expose the producer port.
`EvaluationJobHttpClient` accepts a caller-owned authenticated `httpx.AsyncClient`
with the usual compatibility headers. The server derives the actor from
authentication, not request JSON. Framework callers can instead use
`create_public_original_job_port_async`, then `startup_async`,
`start_consumer`, and the same typed submit/status/cancel methods.
Ordinary Scenario behavior and existing GUI controls are unchanged. This is a
CLI/library/authenticated API PoC, **not a new wired CoPyRIT job button**.

Cancellation is joined before releasing owned runtime/storage work. The
original runner/importer uses owned thread writes; the original run-end hook
also shields its close inside Inspect's cancelled AnyIO scope. A cancelled
source does not acquire a canonical grade. The cancellation test transparently
instruments a real harmless solver hold; only the separate positive/fresh run
claims unchanged source lifecycle. Verified local closure means the owned
coroutines, writes and scratch engine drained, not that an interrupted source
scorer/cleanup completed. Once `finalizing` begins, cancel is refused and the
canonical writer is retained/joined.

The schema also names `reviewed_inspect_variant` and `native_binding`, but names
do not install runtimes. Mode 2 proof covers reviewed-boundary capability
validation and ordered command delivery only, not a live Inspect agent.
SendMessage/Nudge carry bounded text; Advance/Stop do not. Commands require the
actor, per-job capability, current runtime-created boundary and admitted
action. Delivery is not proof that an agent acted. No shell or follow-up Task
is created. Mode 3 has only a harmless native handler/artifact-port fixture,
with `native_evidence` rather than a fabricated `.eval`; it proves no private
native runtime parity. The default backend advertises neither Mode 2 nor 3.
No Azure broker transport, private provisioning, service identity, remote
exec/closure/reset, platform wait/agent transport, or production isolation is
provided or qualified by this local public PoC.

### Opt-in remote execution-only gateway

`RemoteEvaluationJobSettings` is a startup-only configuration for a generic
authenticated job service. It is **not** private platform configuration, a
provisioning client, an Inspect protocol, or an automatic implementation of all
runtime kinds. The public backend keeps its own durable queue, authenticated
operator admission and canonical SQLite. `RemoteEvaluationJobGateway` is a
deliberately installed local-owner facade; it does not proxy a worker database
or a worker's canonical result. Ordinary Scenario execution and existing UI
controls remain unchanged.

Configure the remote backend explicitly:

```text
PYRIT_EVALUATION_JOB_BACKEND=remote
PYRIT_EVALUATION_JOB_ROOT=<absolute local gateway owner directory>
PYRIT_EVALUATION_JOB_ALLOWED_OPERATOR_OIDS=<explicit authenticated operator UUIDs>
PYRIT_EVALUATION_JOB_REMOTE_URL=https://worker.example.invalid
PYRIT_EVALUATION_JOB_REMOTE_SERVICE_ID=<approved generic service ID>
PYRIT_EVALUATION_JOB_REMOTE_IDENTITY=<named host-installed credential provider>
PYRIT_EVALUATION_JOB_REMOTE_AUDIENCE=<configured service audience>
PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_VERSION=1
PYRIT_EVALUATION_JOB_REMOTE_PROTOCOL_SHA256=<evaluation_worker_schema_sha256()>
```

`EvaluationWorkerCredentialRegistry` is empty by default. A trusted host must
deliberately install a named factory returning an
`EvaluationWorkerCredentialProvider`. The provider supplies two separate
credentials: service identity and scoped operator delegation, bound to the
configured audience and exact method/path/body/job identity. The service must
verify both and authorize source/profile-to-operation mapping on its own side.
An actor header, browser Graph token, token string in settings, arbitrary
Python import path or ambient credential discovery is not a replacement.
Uninstalled production identity/delegation fails closed; this implementation
does not supply or qualify Entra/OBO or private platform identity.

The transport owns `httpx.AsyncClient` with HTTPS/TLS verification, redirects
disabled and environment proxies disabled. Credential-bearing URLs, queries
and fragments are rejected. Loopback HTTP requires
`PYRIT_EVALUATION_JOB_REMOTE_ALLOW_LOOPBACK_HTTP=true`; a fixture-only provider
is refused anywhere except that explicitly configured loopback test authority.
This flag is not an authentication bypass and installs no fixture credentials.
Request/artifact deadlines default to 10/30 seconds, polling to 0.25 seconds
with a 60-second deadline, and cancellation settlement to 10 seconds. The
corresponding `REMOTE_REQUEST_TIMEOUT_SECONDS`, `REMOTE_ARTIFACT_TIMEOUT_SECONDS`,
`REMOTE_POLL_INTERVAL_SECONDS`, `REMOTE_POLL_DEADLINE_SECONDS` and
`REMOTE_SETTLEMENT_TIMEOUT_SECONDS` suffixes use the same
`PYRIT_EVALUATION_JOB_` prefix. All limits must be finite and bounded.
Partial settings and incompatible versions/schemas fail before execution.

CoPyRIT/framework clients still use the public `/api/evaluation-jobs` producer
surface. The gateway intersects authenticated per-operator worker grants with
its own pinned registrations. Only the public harmless original and its
reviewed canonical writer are installed. Service-advertised Tasks, runtime
kinds, code, initial inputs, models, provisioning settings or Mode 1 steering
are never dynamically installed.

`pyrit.models.evaluation_worker` defines the separate execution-only
`/api/evaluation-worker/v1` wire. It provides protocol/catalog, job
admission/status/cancel/control, protected manifest/artifact retrieval and
gateway settlement messages. Worker `completed` means retained original
evidence and observed owned closure, not a grade or canonical import. Worker
messages forbid canonical receipts, projection IDs and Score/AttackResult IDs.
The gateway stores the actor-bound request, gateway fence, independent worker
fence/incarnation and a separate worker event cursor in `remote.sqlite`.
An ambiguous response or restart cannot replace that binding or replay the
original. The local and worker event sequences are distinct.

The service exports its original manifest **verbatim** under the worker fence.
The gateway verifies authenticated binding/terminal identity, request/source/
case/profile, gap-free events, exact inventory, media types, byte lengths and
SHA256, then downloads only declared flat names from the same service authority.
Existing per-artifact/aggregate bounds remain 16/32 MiB. It retains the exact
original manifest bytes and creates a separately identified **derived**
handoff manifest for the same inventory/bytes under its own gateway fence;
both digests and the original byte digest are linked durably. This is not a
silent fence restamp or provider-incarnation attestation.

Only `OriginalInspectArtifactWriter` in the API imports the exact binary `.eval`
and creates/verifies its own canonical Score, AttackResult and archive records.
It never copies worker SQLite IDs or reruns the source scorer. Gateway
settlement acknowledges that retained/imported handoff, not a guest shutdown.
A cancelled/failed source with verified joined closure and absent artifacts
can use `closure_observed` with **no artifact hashes**. Unknown closure/evidence
cannot use that disposition. A source runtime error after a requested cancel
remains a source error in the retained worker receipt; requested gateway
cancellation and source-grade absence are separate facts.

Remote cancellation, caller disconnect, control polling and finalization keep
retained owners. Verified closure requires the worker's reviewed task/storage
join, not HTTP ACK, process-client interruption or platform stop/status.
Cancellation after source retention but before API finalization keeps the
retained archive without a canonical grade and quarantines its unsettled
handoff; it does not automatically send a retention acknowledgment or retry.
Unknown dispatch/settlement quarantines new work. If API canonical import
commits but settlement is lost, the job retains its actual local canonical IDs
with failed/unknown settlement rather than reporting a successful run or
dropping the grade. Retained evidence and complete deterministic import
readback support explicit reconciliation, but there is no automatic restart
repair, source retry or execution renewal after authorization expires.

Real separate-process/TCP harmless proofs are distinct from mock HTTP unit
tests and fixture credential verification is not production identity
qualification. This public gateway provides no private HTTP host, operation
policy, provider-assignment/incarnation fencing, faithful guest exec/timeout/
termination/reset guarantees, managed agent controls, or production isolation.
Those must be supplied and independently reviewed before private-platform
execution. No cloud/model/private-original authority follows from configuring
a URL.

## Supported execution and qualification

- One selected, original text `Sample`, one original scorer, one epoch, one
  registered solver override, and a reviewed `ComposeConfig`. Per-sample
  sandbox overrides, multimodal input, altered GHCP history, artifact-only
  completion, and additional model providers are rejected rather than guessed.
- Pinned, prebuilt local images with `pull_policy=never`; a trusted controller
  verifies each running container's full ID, image ID, nonroot/read-only
  profile, and its sole Docker **internal** network. Both agent and bridge
  have bounded anonymous temporary storage; the executable `/var/tmp` tmpfs
  exists only for Inspect's hardcoded sandbox-tools injection. The target has
  no framework tmpfs, no published port, and no host volume.
- The guest worker creates a fresh mode-0700, UID-10001 HOME on its existing
  bounded `/tmp` tmpfs. The SDK child process receives only this HOME,
  COPILOT_HOME, the verified guest PATH, offline provider URL/model metadata,
  and `COPILOT_SKIP_CLI_DOWNLOAD=1`. `use_logged_in_user=False` and an
  explicit approved model/wire mapping prevent ambient GitHub authentication.
  The controller delivers its 43-character run token through exactly two
  bounded, single-use **raw Docker-provider stdin writes**, one to each
  already-attested agent/bridge container. Each creates an exclusive,
  no-follow, UID-10001/mode-0600 file on that container's existing `/tmp`
  tmpfs; the worker and gateway unlink their own file before starting
  inference. Inspect's `connection().container` is a **Compose name**, not
  a full Docker ID: the trusted image/profile verifier resolves that exact
  service to a distinct 64-hex ID before each write, again after it, and
  before the first attack turn. A name/ID change blocks the receipt and
  grading. Only the two token-bearing writes bypass Inspect's per-sample
  event proxy. Public setup, scorer, cleanup, liveness and token-file absence
  checks remain observable. An aborted startup removes any unconsumed private
  file or requires exact-container teardown before a recovered UND result.
  The model gateway strips the run token before forwarding, enforces quotas,
  rejects remote tools, and records the exact original and forwarded bytes.
  Prompt-cache-key removal, if approved, happens **after** original bytes are
  captured and is visible in their different digests.
- The bounded policy defaults to two retained user turns, at least one
  successful tool request/start/completion correlation, 12 model requests,
  60 seconds per user turn, 600 seconds per Inspect sample (240 seconds in
  the example), and a 16 MiB raw evidence quota. Change these only after
  reviewing the task and source coverage. A blank, truncated, mismatched or
  unsealed source is never a complete cyber Score.

**Controller environments are deliberately separate.** The main PyRIT
environment retains OpenAI 2 because the optional `all`/LiteLLM group requires
`openai<3`. Inspect 0.3.272's Responses agent bridge requires OpenAI 3.1 or
newer. The dedicated *trusted Windows controller* uv project
`build_scripts\inspect_ghcp_controller\pyproject.toml` installs **editable
PyRIT base only**, Inspect 0.3.272 and OpenAI 3 in its own `.venv`; it does not
install the `all` group or alter the main project `.venv`. The much smaller
`build_scripts\inspect_ghcp_guest\pyproject.toml` locks Linux CPython 3.12
Copilot SDK 1.0.14 and HTTP dependencies for the agent image. Both projects
have their own committed `uv.lock`. Use `uv sync` in each before executing
their code.

The offline image recipe in `build_scripts\inspect_ghcp_guest\Dockerfile`
starts only from an already-attested local image, checks the Linux uv and
GHCP executable SHA256s, verifies every public Linux wheel against the
**original** uv lock, then generates a separate local-only lock and runs
`uv sync --frozen --offline --no-index --find-links`. Neither a mutable
registry tag nor a Docker image ID is a registry RepoDigest; inspect the
*actual* image ID immediately before build and again when each container
starts. Do not pull a missing image, grant Docker TCP, or loosen the
agent/bridge network to install dependencies.

For an **authorized isolated environment only**, the bounded benign
protocol task is in `examples\inspect_ghcp_protocol_smoke.py`. Its original
Inspect setup writes a harmless marker inside the separate target. Its
original scorer reads that marker's SHA256 **inside the target** and checks
the observed GHCP final directory response; its cleanup removes the marker
after grading. A numeric original `1.0` here is a **lifecycle/readback
probe**, never a cyber achievement. This task remains a
`protocol_smoke` with an **UND/NULL PyRIT Score**, even if every source is
complete and the original scorer returns `1.0`.
The example exits **nonzero** if an error/UND report has any other required
gap, missed turn, missing original scorer, or unverified cleanup; it never
uses the presence of a persisted UND Score as a success signal.
An Inspect EvalLog with task/sample error or missing original scorer is
recorded as an error report even when `eval_async` returns normally.

```powershell
uv sync --extra inspect
uv sync --directory build_scripts\inspect_ghcp_controller --python 3.14
uv sync --directory build_scripts\inspect_ghcp_guest --python 3.12

# Only after a separately approved, per-session Docker SSH stdio identity,
# trusted host-key pin, offline image build and host-loopback model preflight:
uv run --project build_scripts\inspect_ghcp_controller python examples\inspect_ghcp_protocol_smoke.py
```

The example requires a reviewed Docker-SSH alias, exact local image IDs and a
real host-loopback Qwen provider. It refuses a wildcard listener, a Docker
TCP endpoint, missing image IDs, unexpected mounts, or a leftover Compose
project. Never copy a private SSH key or primary provider credential into an
image or log. Private live receipts, logs, wheelhouse and SQLite DB belong in
the ignored worktree `.venv\inspect-ghcp` directory, not in Git.
The controller records a SHA256 fingerprint (never the value) in the private
`controller-stage.jsonl` before either token delivery, so even an aborted
handoff remains auditable. Redirect a live controller's stdout and stderr
into separate files inside that same ignored directory. After the run, call
`uv run --project build_scripts\inspect_ghcp_controller python
build_scripts\inspect_ghcp_controller\audit_secret_retention.py --run-id
<run-uuid> --stdout <private-stdout-path> --stderr <private-stderr-path>`.
The verifier compares literal, padded/unpadded base64, hex, UTF-16 and
JSON/URL-escaped token windows against the private SHA256 without printing
the token or digest. It reads every run raw source
explicitly, the schema-3 report and receipts, the `.eval` file and
decompressed attachments, SQLite and its journals, the run folder, and
controller stdout/stderr. Its private JSON receipt requires a two-turn,
tool-using, original-scorer run with only the benchmark-unverified gap and
one UND Score with NULL numeric value.

## Evidence and finalization

The controller records one episode in the existing PyRIT evidence DB. Its
append-only turn rows link **actual** PyRIT user/assistant `MessagePiece` IDs.
The original SDK event IDs, ordering, session ID, tool request IDs, tool
names/arguments, start/completion success and model-visible results are
correlated with the same container/session before grading. Four additional
source streams retain the exact authenticated gateway request/response
bodies and status (including rejections), actual trusted-host model HTTP
bodies, actual PyRIT adversarial-model HTTP bodies, and the **single resolved
original Inspect EvalLog**. SDK events and the token-free, source-ID/size/SHA256/
process/container control receipts are two additional streams, six in all. Raw-byte
counts, omissions, SHA256s, event IDs and tool correlations are checked
against their stored rows. Safe episode snapshots do not expose raw content;
reading it requires an explicit `allow_sensitive=True` call to
`read_raw_chunks` or `read_event_payloads`.
New reports use schema 3 to link the two receipts and prove the token files
absent before the first turn. Historical schema 1/2 report hashes replay
without those new fields, and interrupted schema-2 episodes remain recoverable
as UND without fabricating control receipts.

The original Inspect ScoreEvent must identify the selected task, sample,
epoch, scorer and event ID exactly once. Its canonical typed judgment keeps
the original value and explanation, source event ID, normalization version,
and raw Score SHA256. PyRIT's `InspectGhcpReportScorer` only maps this
already-acquired judgment; it **never invokes** the original scorer or a
second grader. Report content, one prepared PyRIT Score, and their episode
link commit atomically. A complete numeric **cyber benchmark** Score would require
independently pinned original task/data/image/scorer assets, all required raw
sources, a real model and PyRIT decision loop, agent stop before grading,
target/gateway liveness across grading, original cleanup, and independent
removal of all three exact containers plus the Compose project/network.
The current schema 3 records the three exact full container IDs, but not
Inspect's random Compose project name or full network ID. Cleanup is
therefore proved as exact absence for each container **and separately**
absence of child Compose projects/containers/networks by name/label;
it is not an exact-network-ID deletion attestation. A future versioned
profile must capture project/network IDs while live, bind them to every
service, and verify exact absence after teardown if that stronger proof
is required. Existing sealed runs cannot gain such evidence retroactively.
Otherwise the Score has `status=undetermined` and no numeric value.
The present `InspectGhcpTaskBinding` explicitly **rejects**
`CYBER_BENCHMARK` even if a caller supplies toy asset hashes: no original
cyber task/image/harness has been qualified here. Complete-score model
validation is a future contract, not an enabled user-facing benchmark mode.

If the controller exits after retaining the original log but before the
atomic Score commit, `recover_interrupted_inspect_run_async` reopens that
same episode **only after** Docker project/container/network cleanup is
verified. It verifies the retained raw SHA256s, preserves an acquired
original judgment as an annotation when the ScoreEvent is unambiguous, and
commits one **UND** result without running a model, task or scorer again.
Repeating recovery returns the same linked Score ID. It cannot turn a smoke
or interrupted run into a cyber benchmark success.

This work reuses PyRIT's provider-neutral `Message`, `PromptNormalizer`,
`RedTeamingAttack`, `Score`, `CentralMemory`, and the **append-only parts** of
the committed evidence store and SQL tables, whose current names are
`NativeCyberEvidenceStore` / `NativeCyber*`. It does **not** import
`NativeCyberEvaluation`, `NativeCyberTaskBinding`,
`DockerStopOnlyAgentLease`, `NativeAgentTarget`, or the native report scorer.
No native evaluator, task lease, provider, or migration was modified for this
adapter. Before making this main-based independently of the native branch,
extract those shared evidence-row/atomic-score primitives under a
provider-neutral name (retaining compatibility aliases/migrations), rather
than introducing a parallel DB schema or importing native lifecycle code.
The Inspect-owned runner, sandbox transport, original log mapping and
scorer remain separate.

**Original public CTF qualification is still blocked.** At pinned
`inspect_evals` revision `8ddfea18ea7dabbac4d230b1fb0e7139655afb6f`,
`gdm_intercode_ctf` sample 4 uses the original `includes()` scorer and
InterCode dataset revision `c3e46d827cfc9d4c704ec078f7abf9f41e3191d8`,
but its dataset/asset and original target image are not among the reviewed
local resources. Its upstream Compose builds a distinct Ubuntu 24.04 image
with an extensive apt/pip toolchain and its default agent has `react` /
`submit` semantics. The older substitute `python:3.12-slim` run and this
benign marker task do **not** establish original benchmark parity. A scored
cyber evaluation needs a separate approved, offline-pinned original dataset,
image, task and grader qualification; do not use the present smoke Score for
that purpose.
