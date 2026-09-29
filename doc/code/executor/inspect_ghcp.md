# Inspect-owned cyber tasks with a contained GHCP agent

This optional adapter is a **single-task, text-only GHCP harness**, not a rewrite
of Inspect or a general adapter for Claude, Codex, or arbitrary Inspect tasks.
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
