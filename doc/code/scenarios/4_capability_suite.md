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
`SandboxFileScorer` (hash/content of a file read from the still-open sandbox), and
`SandboxCommandScorer` (exit code of a command run in the still-open sandbox). Any existing
`pyrit.score.Scorer` (via `MessageScorerAdapter`) composes into the same seam through
`ResultOnlyScorerAdapter`, which simply ignores the sandbox session.
