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

The literal baseline has no objective or auxiliary scorer inside the attack.
The original grader is invoked once after the agent is quiescent and before
environment cleanup. The resulting report is persisted as `ContentScorable`,
and its one numeric `Score` is linked to `ContentEntryScorable`, not the final
assistant sentence. The normal attack result and conversation remain in memory
and are referenced by the report. Cleanup failure prevents clean success while
retaining an already-acquired judgment.

Bindings are trusted code registered via `get_native_cyber_bindings().register`.
Only registry names and allow-listed configuration should cross an API boundary.
Private task assets, credentials, graders and outcome weights do not belong in
public presets or source.

## Binding contract

```python
from contextlib import asynccontextmanager
from pathlib import Path

from pyrit.executor.workflow.native_cyber_eval import (
    NativeCyberEvaluation,
    NativeCyberRuntime,
    NativeCyberTaskBinding,
    get_native_cyber_bindings,
)
from pyrit.models.native_cyber import NativeCyberRequest


class TaskBinding(NativeCyberTaskBinding):
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
    directory=Path("results") / "native-agent-eval",
)
view = await run.start_async()
```

`NativeCyberRuntime.target` is a `NativeAgentTarget` around a fresh
`NativeAgentSession`. `grade_async(*, evidence)` returns `NativeCyberJudgment`
with an optional numeric value, rationale, complete flag, retained artifact
references and acquired original evidence. A missing or failed acquisition has
no numeric value. Artifact references describe bytes the binding has already
retained and hashed; they do not authorize arbitrary host file access.

`NativeCyberRequest` limits edits to instruction, label, allowed technique and
converter registry names, stepping, bounded TTL/turn timeout and parent lineage.
The binding caps TTL and explicitly allows converters and technique factories.
Task-specific interfaces must further constrain instruction variants and protect
the original grading rules, trust policy, fixture and credentials.

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
on this explicit rerun path.

## Event fidelity and coverage

The adapter subscribes before the first send and preserves actual event IDs,
observed ordering, native session ID and the unmodified JSON envelope, including
ephemeral usage and idle events. Tool request/start/completion identities and
arguments are correlated. Missing, repeated or mismatched events are explicit
coverage gaps and cannot produce a clean grade.

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
