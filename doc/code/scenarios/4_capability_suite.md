# Capability Suites

`pyrit.scenario.capability_suite` is a strict, native, JSON-serializable format and runner
for capability-style suites (multi-turn conversations, tools, and sandboxed
side effects), plus a static compiler for building suites from local data. It builds on
`pyrit.executor.capability` (the provider-neutral capability-task executor) and
`pyrit.sandbox` (Local/Docker/Hyper-V sandbox providers) from the framework's executor and
shared layers -- see [`doc/code/framework.md`](../framework.md) for how those pieces fit
together. Everything in this package is opt-in: existing scenarios, executors, and sandbox
usage are unaffected unless a caller explicitly builds and runs a
`CapabilitySuiteManifest`.

## Why a separate, native format?

Some capability-suite ecosystems (e.g. Inspect AI / `inspect_evals`) ship suites as
executable Python: tasks, solvers, and scorers are arbitrary code imported and run by the
harness. That is a large trust boundary to accept into an automated pipeline.
`capability_suite` takes a different approach:

> **No Inspect dependency exists.** PyRIT does not install or import the real
> `inspect_ai` package. Static compilers never execute `inspect_evals`. The separate,
> opt-in `pyrit.compat.inspect_ai` loader described below can execute trusted source
> from one pinned checkout before converting its construction graph to this native format.

- **The manifest is data, not code.** `CapabilitySuiteManifest` and everything it contains
  (`manifest.py`) is an immutable, `extra="forbid"` Pydantic model tree. It can be loaded
  from JSON, hashed, and diffed like any other data file -- there is no `eval`/`exec`/
  `importlib` anywhere in the loading path, and unknown fields or unsafe (absolute,
  UNC, or `..`-traversing) asset paths are rejected at parse time
  (`validate_safe_relative_path`).
- **Every symbolic reference is resolved through an explicit, injected registry.** A
  manifest names a sandbox `provider_type`, a tool `implementation.kind`, or a scorer
  `kind` as a plain string; turning that string into a live object always goes through a
  `SandboxProviderFactoryRegistry`, `ToolImplementationFactoryRegistry`, or
  `CapabilitySuiteScorerFactoryRegistry` (`registries.py`) that the *caller* builds and
  populates. Nothing is auto-discovered or imported by name from manifest content.
- **Compilation is static.** `compiler.py` builds manifests from local JSON/JSONL/CSV
  files, an already-fetched PyRIT `SeedDataset`, or a checked-out evaluation
  repository -- by reading declarative data and safe `string.Template` substitutions
  (lookup only, never executed code). `CheckedOutEvalRepoCompiler` scans a checked-out
  repo and produces a `CompatibilityReport`; if the repo has no statically-compilable
  data file (only executable indicators such as `inspect_ai`/`@task`/`@solver`/`@scorer`),
  it raises `UnsupportedExecutableMethodologyError` instead of ever importing the
  repository's Python.

## Inspect-evals source interoperability

`pyrit.scenario.capability_suite.inspect_evals` analyzes a user-supplied checkout using
YAML/JSON parsing, Python AST inspection, and Docker/Compose asset discovery only. Adapter
assumptions are pinned to
[`UKGovernmentBEIS/inspect_evals@b935c0e5cfa04710f016f925db75d8e81413e2cf`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/b935c0e5cfa04710f016f925db75d8e81413e2cf).
The source checkout is never added to `sys.path`, imported, or executed.

The dedicated capability-suite CLI exposes the local-only surface without changing the
`pyrit_scan` REST client's thin-client boundary:

```console
# Analyze every eval.yaml/task source/container asset in a checkout.
pyrit_capability_suite inspect-evals --source ./inspect_evals \
  --report ./compatibility.json

# Compile selected offline ARC records into the canonical manifest.
pyrit_capability_suite inspect-evals --source ./inspect_evals \
  --family arc \
  --data ./arc-records.json \
  --case-id Mercury_7175875 \
  --manifest ./arc-manifest.json
```

Network retrieval is disabled by default. `--allow-network` permits only the
adapter's fixed, revision-pinned Hugging Face source; local source files are preferred for
reproducible and licensed workflows. Reports and manifests distinguish the pinned adapter
schema revision from the detected checkout revision and include local data hashes; a revision
is marked verified only when its identity was actually established.

### Compatibility matrix

| Family | Source reused | Native adaptation | Fidelity | Unsupported/different semantics |
|---|---|---|---|---|
| ARC Easy/Challenge | `allenai/ai2_arc`, revision `210d026faf9955653af8916fad021475a3f00453`, CC-BY-SA-4.0 | Question/choice schema, answer-label normalization, exact text scoring, no sandbox | **native** | PyRIT asks for a letter directly rather than invoking Inspect's `multiple_choice()` solver. |
| GDM InterCode CTF | `princeton-nlp/intercode`, revision `c3e46d827cfc9d4c704ec078f7abf9f41e3191d8`, archive SHA-256 `32e552a468fd69efb7a2cfe13bc591a79246c5db46f3fb629f9cec6dbb1720d7`; picoCTF non-commercial terms | Challenge IDs/objectives, per-task files, Docker/Compose, shell/file tools, flag scoring, limits; statically materialized Dockerfile from the pinned template and AST-read literal package lists | **adapted** | Docker is mandatory; PyRIT does not redistribute the corpus. The pinned adapter's 22 Internet-dependent IDs (`1, 7, 9, 14, 16, 20, 28, 29, 35, 39, 41, 42, 43, 54, 57, 62, 66, 73, 87, 88, 89, 95`) are excluded. |
| GDM in-house CTF | Static `challenges/challenges.json` and per-challenge Compose assets from the pinned checkout; inspect-evals MIT, GDM source Apache-2.0 | One topology per manifest; model-visible target address; `x-default` tool routing; victim-only shell commands run as `app`; live victim-state flag scoring before cleanup; ten epochs | **adapted** | Only the exec binding is exposed, because provider file APIs cannot preserve the upstream `app` identity. Scorer-only victim services are not exposed to model tools. `marshw/*` images and their flags are floating external runtime state. |
| SWE-bench Verified | `princeton-nlp/SWE-bench_Verified`, revision `c104f840cc67f8b6eec6f759ebc8b2693d585d4a`, MIT | Problem statement, instance/base-commit/test metadata, patch evidence expectation, gold/test patch hashes | **partial** | Cases are explicitly non-runnable. Full scoring needs a distinct authenticated container image per instance, the SWE-bench test harness, and browser/custom agent semantics. |

The compatibility report uses exactly `native`, `adapted`, `partial`, or `unsupported`.
Unknown eval families are `unsupported` even when static files are present: field names,
solver behavior, and scoring semantics are not inferred or fabricated.

Results can differ from Inspect runs because PyRIT uses its own target transport,
conversation/tool loop, retry policy, evidence model, sandbox providers, and score
aggregation. A matching dataset and prompt does not by itself imply matching agent,
browser, tool timeout, container image, scorer, or epoch semantics.

### Opt-in unchanged-source compatibility

`pyrit.compat.inspect_ai` is an isolated construction facade for
[`inspect_evals@b935c0e5cfa04710f016f925db75d8e81413e2cf`](https://github.com/UKGovernmentBEIS/inspect_evals/tree/b935c0e5cfa04710f016f925db75d8e81413e2cf).
Its profile ID is `inspect-evals-b935c0e-inspect-api-0.3.233`; this implements the
construction API semantics of `inspect_ai==0.3.233` consumed by that exact source
revision, not arbitrary Inspect versions or current `inspect_evals` main.

The loader executes an unchanged `@task` module with temporary `inspect_ai` aliases,
materializes its dataset, and converts supported graph nodes to a
`CapabilitySuiteManifest`. No top-level `inspect_ai` package is distributed. Source
construction runs in a dedicated worker process, so its temporary `inspect_ai` and
`inspect_evals` import aliases are never visible in the caller, cannot shadow a real
Inspect installation, and disappear with the worker on success or failure. A contained
source finder validates every imported `inspect_evals.*` Python module before executing
it. Static API inventory runs before execution and unknown symbols raise
`UnsupportedInspectFeatureError` with the symbol, profile, and remediation.
Pinned Hugging Face and source-contained JSON/JSONL/CSV dataset helpers are bound directly
to offline-first facade loaders. GDM InterCode requires a pre-populated
`inspect_evals_cache_dir`; the compatibility worker never downloads its corpus.

This is **source containment, not a security sandbox**. The selected module path and
`inspect_evals.*` imports must resolve beneath the supplied checkout, but the worker is
arbitrary Python running with the PyRIT user's operating-system identity. Only run
reviewed, trusted source. Verified loads require both the exact pinned Git commit and a
clean tracked, untracked, and ignored worktree. Worker bytecode generation is disabled
so verified loads do not mutate the supplied checkout. If revision verification is
explicitly disabled, reports and manifests retain the detected revision but mark it
unverified. Offline dataset loading is the default; callers can inject local ARC records
or explicitly opt in to the pinned Hugging Face request. Worker execution is bounded by
`worker_timeout_seconds` (300 seconds by default), and Git verification is bounded by
`source_verification_timeout_seconds` (120 seconds by default).

```python
from pyrit.compat.inspect_ai import load_inspect_eval, run_inspect_eval_async

loaded = load_inspect_eval(
    source_root=checkout,
    task_spec="arc/arc.py@arc_challenge",
    dataset_loader=local_pinned_dataset_loader,
)

ctf = load_inspect_eval(
    source_root=checkout,
    task_spec="gdm_intercode_ctf/gdm_intercode_ctf.py@gdm_intercode_ctf",
    task_parameters={"sample_ids": [2]},
    inspect_evals_cache_dir=pinned_cache,
)

execution = await run_inspect_eval_async(
    source_root=checkout,
    task_spec="arc/arc.py@arc_challenge",
    target=target,
    dataset_loader=local_pinned_dataset_loader,
)
```

The unchanged-source surface supports ARC Easy/Challenge plus
`gdm_intercode_ctf@gdm_intercode_ctf` and
`gdm_in_house_ctf@gdm_in_house_ctf`. The GDM tasks compile standard ReAct,
`bash`, `python`, and `submit` construction nodes to the native capability executor.
InterCode preserves files, setup, submission retries, and includes scoring. In-house
preserves per-sample Compose topology, default service and user selection, target address,
epochs, and scorer-only live flag reads before sandbox cleanup. Tool implementations and
the in-house scorer are selected through explicit registries; the scorer proxy allows one
bounded command in its declared service and exposes no credentials or host-file access.

The compatibility layer also provides reusable declarative foundations for later adapters.
These foundations do **not** make another task family supported by themselves:

- `Task.setup` and `Task.solver` preserve ordered lists and nested `chain()` nodes.
  `system_message()`, `prompt_template()`, `user_message()`, `assistant_message()`, and a
  terminal `generate()` compile to native initial messages and target generation. Template
  parameters can use the current prompt and scalar sample metadata. Unsupported generation
  controls, unknown solver nodes, misplaced generation, and setup/files that cannot be
  represented fail during compilation with a graph path.
- `FieldSpec` maps IDs, input, targets, choices, metadata, setup, and files from records.
  Local JSON/JSONL and CSV sources remain confined to the supplied source root. Hugging Face
  sources require an explicit revision and split, reject remote code, and are offline unless
  callers inject records or opt in to network access. Dataset selection records deterministic
  shuffle, choice-shuffle, auto-ID, and limit settings in stable provenance, separate from
  per-run and per-attempt identities.
- String inputs and existing Inspect chat-message sequences retain role and order, including
  few-shot user/assistant turns. Text and image content parts compile to ordered native PyRIT
  `MessagePiece` objects rather than flattened strings. HTTP(S) and data-URI images remain URLs;
  source-contained local images are validated under trusted roots and embedded as deterministic,
  content-addressed data URIs so manifests do not retain machine-specific paths. Runner preflight
  checks the target's exact combined input modality and conversation-shape declarations before
  sandbox creation or model execution.
- Multiple scorers retain stable scorer IDs. Scalar boolean, integer, float, and string values,
  plus dictionary-valued scores, normalize to native PyRIT scores with explanations and
  metadata. Typed accuracy, mean, standard-error, grouped, clustered, mean-reducer,
  `at_least`, `pass_at`, and `pass_k` specifications aggregate deterministically. Unknown
  scorer, metric, reducer, or option nodes fail at compile time.

Manifest schema version 3 carries ordered multipart messages and typed metric/reducer
specifications; version 2 manifests migrate by retaining their legacy single-content shape.
ARC and CTF manifests and result aggregates retain their prior score counts, mean, and
distribution fields while optionally exposing the named metric and reducer maps.

Model calls only use the injected PyRIT `PromptTarget`, and Docker remains an external
runtime prerequisite. Stores, hooks, checkpoints, EvalLog parity, model providers, and
non-pinned callbacks remain unsupported. AWS/Bedrock/SageMaker/EC2, GCP, Modal, Daytona,
Kubernetes, and other cloud providers are not implemented. Every native case uses the
explicit `case_timeout_seconds` execution bound (300 seconds by default).
Static `Task.message_limit` is rejected because the native `max_turns` limit does not have
equivalent Inspect semantics.

### CLI setup and unchanged-task workflow

Install PyRIT's development environment with `uv`; do not install `inspect_ai` or
`inspect_evals`:

```console
uv sync --group dev
uv run pyrit_capability_suite inspect-evals source prepare --output ./inspect-source.json
```

`source prepare` fetches only
`UKGovernmentBEIS/inspect_evals@b935c0e5cfa04710f016f925db75d8e81413e2cf`
into a revision-keyed PyRIT user cache, then verifies the commit, Git tree, clean
worktree, package layout, and LICENSE hash. It never installs or executes Inspect.
Use `--offline` to require an existing cache. An already-available checkout can be
validated without modification:

```console
uv run pyrit_capability_suite inspect-evals source validate \
  --source ./inspect_evals-b935c0e \
  --output ./inspect-source.json
```

The JSON output's `source_root` is the value to pass as `<source>` below. Discovery,
compatibility diagnostics, the CI regression check, and compilation need no model
credentials:

```console
uv run pyrit_capability_suite inspect-evals tasks \
  --source <source> --family arc
uv run pyrit_capability_suite inspect-evals report \
  --source <source> --format json --output ./inspect-report.json
uv run pyrit_capability_suite inspect-evals catalog \
  --source <source> --check --format json --output ./inspect-catalog.json
uv run pyrit_capability_suite inspect-evals dry-run \
  --source <source> \
  --task arc/arc.py@arc_challenge \
  --data ./arc-records.json \
  --limit 10 \
  --manifest ./arc-manifest.json \
  --report ./arc-compatibility.json
```

`catalog --check` is intended for CI. At the pinned revision it expects 129
families, 249 task factories, and 262 referenced `inspect_ai` APIs with inventory
SHA-256
`4515d2aba8bedf78de2c0ee866f44de6167ab92aa4eccf9d8fc321b2e789cd34`.
It fails rather than silently expanding support if an API inventory or supported
task claim changes.

Configure targets through the normal [PyRIT configuration](../../getting_started/pyrit_conf.md)
and environment/secret providers. Credentials are never accepted as task arguments.
`--target` selects an exact `TargetRegistry` name; otherwise `--target-role`
selects exactly one registry tag and defaults to `default_objective_target`.
For example, ARC can use the configured default text target:

```console
uv run pyrit_capability_suite inspect-evals run \
  --config ./.pyrit_conf \
  --source <source> \
  --task arc/arc.py@arc_challenge \
  --data ./arc-records.json \
  --case-id Mercury_7175875 \
  --target openai_chat \
  --result ./results/arc.json
```

#### GDM InterCode CTF

InterCode data has separate picoCTF/non-commercial terms and is not fetched or
redistributed by PyRIT. After accepting the applicable license, prepare
`<intercode-cache>/gdm_intercode_ctf/data/ic_ctf.json` and its referenced
`task_assets/` from
`princeton-nlp/intercode@c3e46d827cfc9d4c704ec078f7abf9f41e3191d8`.
The reviewed archive SHA-256 is
`32e552a468fd69efb7a2cfe13bc591a79246c5db46f3fb629f9cec6dbb1720d7`.
Compile before starting Docker or making a model call:

```console
uv run pyrit_capability_suite inspect-evals dry-run \
  --source <source> \
  --task gdm_intercode_ctf/gdm_intercode_ctf.py@gdm_intercode_ctf \
  --inspect-evals-cache-dir <intercode-cache> \
  --case-id 2 \
  --submission-attempts 3 \
  --max-messages 50 \
  --manifest ./intercode-manifest.json
```

Then use a registered target whose declared capabilities include multiturn/editable
history, caller-owned external tool execution, `function_call` output, and
`function_call_output` input. The CLI currently supplies the tool declaration adapter
for targets using `OpenAIResponsesRequestOptions`:

```console
docker compose version
docker info
uv run pyrit_capability_suite inspect-evals run \
  --config ./.pyrit_conf \
  --source <source> \
  --task gdm_intercode_ctf/gdm_intercode_ctf.py@gdm_intercode_ctf \
  --inspect-evals-cache-dir <intercode-cache> \
  --case-id 2 \
  --attempts 1 \
  --concurrency 1 \
  --sandbox-provider docker \
  --target azure_openai_responses \
  --result ./results/intercode-2.json
```

#### GDM in-house CTF

In-house challenge metadata and Compose assets are in the pinned checkout. A
single-challenge dry-run and run are:

```console
uv run pyrit_capability_suite inspect-evals dry-run \
  --source <source> \
  --task gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf \
  --case-id ssh --epochs 1 \
  --manifest ./in-house-ssh.json
uv run pyrit_capability_suite inspect-evals run \
  --config ./.pyrit_conf \
  --source <source> \
  --task gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf \
  --case-id ssh --epochs 1 --attempts 1 --concurrency 1 \
  --sandbox-provider docker \
  --target azure_openai_responses \
  --result ./results/in-house-ssh.json
```

Docker resolves Compose, runs its preflight, and applies the deny-by-default
`DockerSecurityPolicy` before any model request. Policy failures name the rejected
field and required explicit `allow_*` opt-in. Temporary source snapshots and
sandboxes are cleaned after success, failure, timeout, signal, or cancellation.
`--retain-sandboxes` keeps Docker resources for investigation; `--sandbox-config`
accepts JSON overrides for the compiled `DockerSandboxProviderConfig`. Hyper-V is
rejected for these Compose-backed tasks because Compose-compatible Hyper-V
semantics are not implemented.

The `--result` JSON contains every native attempt, model/tool transcript evidence,
scores, aggregate counts, the unique run identity, stable suite/provenance and manifest
hashes, compatibility report, and cleanup error. A new run identity is generated on
every invocation so persistent-memory transcripts never collide. Pass `--resume-id`
only when intentionally reusing an existing execution identity.
`--manifest`, `--report`, and `--output` place compile and catalog evidence at the
specified paths. Common failures are actionable: use the exact clean pinned
checkout for revision errors, populate the licensed InterCode cache for missing
assets, start Docker/Compose for preflight failures, or select a target/adapter pair
whose declared behavior, request-options transport, and input/output modalities satisfy every
reported requirement. Preflight derives input and conversation-shape requirements (including
system prompts and editable multi-message history) from each case's actual messages and declared
modalities before preparing a sandbox or generating model output. The worker is trusted arbitrary
Python under the user's OS identity, not a security sandbox.
AWS/Bedrock/SageMaker/EC2, GCP, Modal, Daytona, Kubernetes, all other non-Azure
cloud runtimes, EvalLog/control-plane/TUI/checkpoint parity, and unreviewed task
families remain unsupported.

## Security and portability boundary

- **Local sandbox provider (`pyrit.sandbox.LocalSandboxProvider`) is not an isolation
  boundary.** It runs commands with the PyRIT process's own identity in a temporary
  workspace -- suitable for trusted development and CI, not for untrusted or adversarial
  case content. Path checks there guard against accidental escape, not malicious code.
- **Docker and Hyper-V sandbox providers are the actual isolation boundaries.** Use
  `DockerSandboxProviderManifestConfig` or `HyperVSandboxProviderManifestConfig` (both
  typed, validated manifest configs) when a suite's setup/tool commands must run
  untrusted content.
- **The default compiler/runner boundary never imports or executes third-party evaluation code.**
  Compilation only ever reads data files or scans file suffixes/text patterns; execution
  only ever runs a manifest's own declared messages/tools/setup commands through
  `pyrit.executor.capability.CapabilityTaskExecutor` inside the sandbox the caller chose.
  The opt-in compatibility loader is a separate, explicit trusted-code boundary that
  returns to this native manifest/runner boundary after source construction.

## Runner lifecycle

`CapabilitySuiteRunner.run_async()` expands a manifest's cases x epochs x independent attempts
(`expansion.expand_suite`) and runs each unit under a bounded `asyncio.Semaphore`
(`run_policy.max_concurrency`). Each physical attempt:

1. Builds a fresh sandbox session (`provider.create_session_async`) and stages the case's
   assets/setup commands into it.
2. Binds the case's tools (sandbox-command tools plus any registry-resolved custom tools).
3. Runs the case through `CapabilityTaskExecutor.execute_case_async` with a unique
   execution/attempt/conversation identity. Stable suite, manifest, and source case
   identifiers remain available as provenance; identities are reused only when an
   explicit resume identity is supplied.
4. Scores the result **while the sandbox session is still alive** (session-aware scorers
   can inspect sandbox file/command state), via the case's configured scorers.
5. Always closes the session (`finally`-equivalent cleanup), distinguishing a cleanup
   failure (`AttemptOutcomeKind.CLEANUP_FAILURE`, result preserved) from a run failure
   (`AttemptOutcomeKind.FAILURE`) in the preserved attempt record.

Only failures whose error code appears in `run_policy.retryable_error_codes` are retried,
up to `run_policy.max_retries`, each retry getting an entirely fresh sandbox session (never
reusing state from a failed attempt). Every attempt -- success, retry, failure, cancellation,
or cleanup failure -- is preserved as one `CapabilitySuiteAttemptRecord`; nothing is
overwritten or dropped. `aggregation.aggregate_attempts` then computes outcome counts,
success rate over final logical-run outcomes, and score mean/distribution over preserved results.

## Native scorers

`scorers.py` provides scorers that only evaluate (they never branch execution or retry
logic, which stays the runner's job): `TextMatchScorer` (exact/substring match against the
final message text), `ToolEvidenceScorer` (tool-call evidence on the result),
`SandboxFileScorer` (hash/content of a file read from the still-open sandbox),
`SandboxCommandScorer` (exit code of a command run in the still-open sandbox), and
`SandboxStateMatchScorer` (dynamic sandbox state compared with the final response). Any existing
`pyrit.score.Scorer` (via `MessageScorerAdapter`) composes into the same seam through
`ResultOnlyScorerAdapter`, which simply ignores the sandbox session.
