# Native Docker CTF prototype

This is a **single-sample, native-PyRIT evaluation**, not an Inspect adapter.
It uses the public `OpenAIResponseTarget`, `PromptNormalizer.send_prompt_async`,
`CentralMemory`/`SQLiteMemory`, `SeedPrompt`, `MessageScorable`, and `IncludesScorer`.
The existing Responses target owns the actual model/tool loop. No final methods
are overridden, and no Inspect package is imported or required.

## Run

From a repository checkout with Docker configured for Linux containers and an
existing authorized Azure identity:

```powershell
uv sync --locked
uv run --frozen python -m pyrit.executor.benchmark.ctf `
  --endpoint https://YOUR-RESOURCE.openai.azure.com/openai/v1 `
  --model YOUR-DEPLOYMENT
```

Each invocation is **one sample, epoch, and attempt**. There are no automatic
episode retries. Do not rerun a failed model episode without operator approval.
The operator binding explicitly passes
`api_key=get_azure_openai_auth(endpoint, credential=credential)`, which is PyRIT's
refreshing async DefaultAzureCredential-backed Entra provider. The binding owns
the credential in an async context manager and closes it after the last request.
The optional caller-owned credential argument preserves existing helper calls.
Interactive browser and broker authentication are explicitly excluded.
An ambient OpenAI API key cannot override it. It never logs tokens or request
headers, mounts credentials, changes Azure permissions, or initiates interactive
login. RBAC failures are real errors, not a reason to switch targets or accounts.

Docker and normal PyRIT dependencies are required; there is no Docker SDK extra.
Host execution uses argv-based async Docker CLI processes, not a host shell.
Windows paths are passed as host paths to `docker cp`; `/workspace` and tool
commands are Linux-container paths. Python 3.11+ is required.

## Deliberately narrow compatibility

| Area | Supported here | Not supported or claimed |
| --- | --- | --- |
| Task | `gdm_intercode_ctf` task 4, the in-the-clear file task | Other cases, setup scripts, broad dataset discovery |
| Environment | One fresh `linux/amd64` Docker container | Compose services, VMs, Kubernetes, remote/external targets |
| Agent | OpenAI-compatible Responses API with strict function calls | GHCP, Claude Code, Codex or other CLI-agent harnesses |
| Feedback | Final assistant answer, graded after the episode | Upstream `react`/three-submit behavior or interactive grading |
| Scoring | A persisted final PyRIT text message | Artifact-only runs, file-state/coverage grading |
| Evidence | SQLite plus a prototype-specific JSON journal/manifest | A new generic PyRIT artifact or episode schema |

The binding uses a **minimal-image/custom-harness/final-answer-only variant**.
It is not a reproduction of the full benchmark's Ubuntu image or package suite.
The model is instructed to inspect the asset using at least one tool and return
only the flag. Tool-less responses are invalid episodes. Gold is trusted host-side
grading data only; it is not included in the seed, prompt metadata, tools, or
feedback. `IncludesScorer` uses `casefold()` and substring containment, without
trimming whitespace or punctuation, matching upstream `includes()` defaults
in [Inspect 0.3.259](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0.3.259/src/inspect_ai/scorer/_match.py).
The raw `C`/`I`, casefolded answer and original explanation accompany the boolean
PyRIT score. Missing/error/truncated responses receive no fabricated answer or grade.

## Limits and isolation

The task binding pins
`python:3.12-slim@sha256:44ff437bba879d4941b710a369a8f19266aea34b29002807f0c487fabc9eec9b`.
Containers use 1 CPU, 512 MiB memory, memory-swap equal to memory, 128 PIDs,
`--cap-drop ALL`, `no-new-privileges`, `--network none`, and `/workspace`.
The effective Docker configuration is checked, not silently approximated.
There are no published ports, host networks, privileged mode, devices, bind
mounts, Docker sockets, or credential/home mounts. Only the named task asset is
copied. This is ordinary Docker isolation, not a claim that hostile code cannot
escape a container. Use a dedicated worker for untrusted benchmark extensions.

There are at most eight Docker tool dispatch attempts, thirty seconds each, and
180 seconds per episode after data/image acquisition, including memory initialization.
Demonstrated executions are counted separately, only after a validated child-start
record. A rejected `docker exec` does not demonstrate tool execution.
The model gets at most 2048 output tokens per request, not a made-up total token
budget. Actual per-response usage is recorded when supplied by the provider.
Requests are capped at nine and both SDK and PyRIT target retries are disabled.
`parallel_tool_calls=false` is required because the existing target executes only
one pending call per response. Unexpected parallel calls are retained and rejected.
Provider storage is disabled with `store=false`.

`bash(command: string)` and `python(code: string)` return JSON containing
`stdout`, `stderr`, `returncode`, `timed_out`, `truncated`, `error`, and
`execution_id`. Each stream is captured up to 16,384 bytes while additional output
is drained, not accumulated. UTF-8 output is also bounded after replacement of
invalid bytes. Nonzero exit, timeout, and truncation are explicit. A timeout
is terminal and is never graded. The fixed in-container Python runner reserves
one second of the thirty-second tool budget for capture and termination: the child
gets twenty-nine seconds. Smaller configured tool budgets reserve up to half
their time instead. The callback journals the bounded timeout result **before**
attempting to kill the owned container. A failed kill is recorded separately from
the retained output; outer cleanup still attempts owned-container removal.
The callback raises, and the request hook prohibits any later model request.
This profile-local terminal-timeout policy intentionally does not allow recovery
in the same container. Cleanup runs outside the episode timeout.

Docker transport/control failures are not command exit codes. The isolated
Python runner captures child stdout/stderr through pipes and sends validated
JSONL start, bounded base64 output, and completion records on its own stdout.
The host checks record fields, types, ordering, and the per-attempt execution ID.
Child output is only data inside that protocol, never a control record.
A valid runner completion can report an ordinary nonzero command exit, including
125, 126, or 127, without failing the environment. Missing or malformed receipts,
Docker transport errors, or a lost container terminate the episode before another
provider request or grade. Docker diagnostics remain separate from command output.
The runner itself requires Python in the image and is not a hostile-code-proof
attestation mechanism; the single approved task does not attempt to tamper with it.

## Evidence and failures

Each run prints its unique directory under the already-ignored
`results\native-ctf\<run-id>`. Keep it out of Git. It contains:

- `pyrit.db`: authentic messages/tool pieces and a message-anchored score.
- `events.jsonl`: model request/response bodies without authentication headers,
  provider request/call IDs, Docker arguments/results and execution IDs.
- `manifest.json`: source/input/asset hashes, implementation hashes, package
  versions, effective image/container configuration, usage, final answer, raw
  grade, errors, and cleanup status.
- The copied task asset, exact input, and upstream InterCode license.

The target buffers intermediate messages until its tool loop returns. On error,
SQLite can therefore have only the request and PyRIT's error message, while the
journal retains earlier authentic provider/tool evidence. A `requested` tool entry
does not prove that a command started. `execution_started_confirmed` records whether
the runner's child-start receipt was observed; rejected dispatches have no fabricated
command result. A transport failure after a confirmed start preserves received
output chunks without claiming completion. On an outer transport timeout, data
still buffered inside the runner is unknown, not reconstructed. No synthetic
assistant receipt is created to fill a gap. Provider call IDs are correlated using public HTTPX
hooks because custom-function callbacks receive only arguments. The manifest
is atomically replaced; the append-only journal is flushed after each event.
An abrupt host or Docker-daemon failure can still prevent cleanup. The owned
container name/ID and status allow an operator to investigate without pruning.
`termination_error` and final `cleanup_errors` are separate: failed or unconfirmed
cleanup is not reported as removed.

Cleanup verifies the unique run label and full container ID before removing
that container only. Images are retained; no networks or volumes are created,
and no unrelated Docker resources are touched.

## Provenance

Task definition and prompt formatting:
[inspect-evals at 8ddfea18](https://github.com/UKGovernmentBEIS/inspect_evals/tree/8ddfea18ea7dabbac4d230b1fb0e7139655afb6f/src/inspect_evals/gdm_intercode_ctf).
Dataset and asset:
[InterCode at c3e46d82](https://github.com/princeton-nlp/intercode/tree/c3e46d827cfc9d4c704ec078f7abf9f41e3191d8/data/ctf).
Both repositories are MIT licensed; see `THIRD_PARTY_NOTICES.txt`.

Only `data/ctf/ic_ctf.json`, `data/ctf/task_assets/4/flag`, and `LICENSE.md`
are downloaded from the pinned InterCode revision, with individual SHA256 checks.
No `solution/` path or archive is downloaded. The exact upstream input, including
leading/trailing newlines, hashes to
`15f4ba2c193c21420b781e3463a8edd8b51dbe2a2ec763e238bb74b4321ad3ee`.
Extending task coverage requires a separate binding and explicit assessment of
its files, setup, packages, privileges, network requirements, and grading evidence.
