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
sensitive PyRIT raw streams. It projects original `EvalSample.messages` text to
MessagePieces with a **distinct conversation per Sample/epoch**, and stores
typed Inspect events in original retry-attempt and sample order with observed
UUIDs when present. Original model/tool/score events, errors, intermediate
scores and final `EvalSample.scores` remain source evidence, not synthetic
PyRIT attack turns. ModelEvent output is not inserted a second time as an
assistant message. Non-text messages stay in the resolved log and are reported
as a projection gap. The original `.eval` is the only source of truth for
offline imports; `EvalSourceFactory` and task materialization are **not**
required. A caller can supply a matching `EvalCaseRef` inventory and
`EvalRunRef` to label source case-run identities, but this does not qualify an
external scorer.

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
proof. The allowlisted live original Task runner still creates neither
per-case Score nor AttackResult.

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
SHA and case-run ID. It reports 1/1 completed, zero successes and no success
percentage; this is import completion, not a success verdict or a synthetic
Scenario AttackResult. The history API validates the same persisted run and
case metadata before counting 1/1. Detail, progress and history also read back
the referenced Score, AttackResult and sealed offline `.eval` episode: their
foreign key, source case/run/archive metadata and undetermined outcome must
still agree. Missing or substituted IDs fail closed without rerunning the Task
or inventing another result. Detail and history return
`objective_achieved_rate: null`, and CLI output says "undetermined"; other
Scenarios retain their numeric rate. A rejected `.eval` leaves the planned
case incomplete. Detail, history and the progress header expose only a vetted
failure diagnosis (or a safe generic failure message), while full exception
details remain in backend logs for reconciliation.
Internal/private Tasks, arbitrary Python and cyber Evals are not selectable.

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
