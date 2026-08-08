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

> **No Inspect dependency exists.** PyRIT does not install, import, or execute
> `inspect_ai` or `inspect_evals`. Arbitrary Python `@task` evaluations are executable
> programs and require a reviewed native adapter before PyRIT can compile them.

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

## Security and portability boundary

- **Local sandbox provider (`pyrit.sandbox.LocalSandboxProvider`) is not an isolation
  boundary.** It runs commands with the PyRIT process's own identity in a temporary
  workspace -- suitable for trusted development and CI, not for untrusted or adversarial
  case content. Path checks there guard against accidental escape, not malicious code.
- **Docker and Hyper-V sandbox providers are the actual isolation boundaries.** Use
  `DockerSandboxProviderManifestConfig` or `HyperVSandboxProviderManifestConfig` (both
  typed, validated manifest configs) when a suite's setup/tool commands must run
  untrusted content.
- **The compiler/runner boundary never imports or executes third-party evaluation code.**
  Compilation only ever reads data files or scans file suffixes/text patterns; execution
  only ever runs a manifest's own declared messages/tools/setup commands through
  `pyrit.executor.capability.CapabilityTaskExecutor` inside the sandbox the caller chose.

## Runner lifecycle

`CapabilitySuiteRunner.run_async()` expands a manifest's cases x epochs x independent attempts
(`expansion.expand_suite`) and runs each unit under a bounded `asyncio.Semaphore`
(`run_policy.max_concurrency`). Each physical attempt:

1. Builds a fresh sandbox session (`provider.create_session_async`) and stages the case's
   assets/setup commands into it.
2. Binds the case's tools (sandbox-command tools plus any registry-resolved custom tools).
3. Runs a deterministically identified case through `CapabilityTaskExecutor.execute_case_async`.
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
