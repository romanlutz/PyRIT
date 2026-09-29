# Executor

An **executor** is an *algorithm for interacting with an objective target*. You give it an objective
and some configuration, it drives the target, and it hands back a result. That's the whole job.

The important thing to notice up front is that **not every executor is an attack**. Sending a single
adversarial prompt is an executor, but so is running a Q&A benchmark over a dataset, fuzzing to
generate new prompts, or orchestrating a cross-domain injection workflow. Attacks are the largest and
most familiar family, but every category in this section — attacks, workflows, benchmarks, and prompt
generators — is the same kind of object running the same lifecycle.

## Executor vs. attack

An **executor** is the *algorithm* — for attacks, the **attack strategy** (e.g. `PromptSendingAttack`,
`CrescendoAttack`, `TreeOfAttacksWithPruningAttack`). It knows *how* to drive the objective target. An
**attack** is just the most common kind of executor.

Don't confuse the executor with an **[attack technique](../scenarios/0_attack_techniques.ipynb)** — a
configured recipe (a role-play framing, a many-shot priming set) that a
[scenario](../scenarios/0_scenarios.ipynb) selects by name. The technique is the *recipe*; the executor
is the *engine* that runs it. Techniques are defined fully in the scenarios docs.

## Executor categories

PyRIT ships several families of executor — attacks are the largest, alongside workflows, benchmarks,
and prompt generators. Attacks themselves split by a simple rule: **count requests to the objective
target** — a single-turn attack sends exactly one; a multi-turn attack sends more than one and adapts
as it goes.

- **[Single-Turn](1_single_turn.ipynb)** — sends a single prompt (**one attack turn**) to the
  objective target and scores the response. It may prepare that prompt elaborately (a role-play frame,
  many-shot priming, a prepended conversation), but only one crafted message is the actual ask, so no
  adversarial target is required to *drive* it.
- **[Multi-Turn](2_multi_turn.ipynb)** — sends **more than one** turn to the objective target,
  adapting until the objective is met or a turn limit is hit. Adaptive variants use an adversarial
  target to generate each next prompt from the responses; others send a fixed sequence, request the
  answer in chunks, or stream input — no adversarial target needed.
- **[Compound](4_compound.ipynb)** — doesn't add turns of its own; it orchestrates *other* attacks
  (running them in sequence) toward a single objective, after the building blocks it composes.
- **[Workflow](5_workflow.ipynb)** — generic multi-step orchestration that doesn't fit the
  attack/benchmark mould (e.g. cross-domain prompt injection / XPIA).
- **[Benchmark](6_benchmark.ipynb)** — evaluates an objective target against a fixed dataset and
  criteria (e.g. Q&A accuracy, bias).
- **[Prompt Generator](7_promptgen.ipynb)** — produces attack prompts (e.g. fuzzing, Anecdoctor) to
  augment datasets; some generate from a model alone, others probe a target to evolve effective
  prompts.
- **[Modality Feedback](8_modality_feedback.ipynb)** — shows how `TargetCapabilities` determine
  whether media is forwarded between objective and adversarial targets in multi-turn attacks, with a
  two-seed Crescendo image-edit example.

**[Attack Configuration](3_attack_configuration.ipynb)** isn't an executor — it's the cross-cutting
inputs every attack accepts (objective vs. adversarial target, prepended conversations, multimodal
seeds, next-turn messages, memory labels). It has its own page.

## The shape of an attack

Attacks — the most common executors — share a 4-component shape:

```{mermaid}
flowchart LR
    A(["Attack Strategy"])
    A --consumes--> B(["Attack Context <br>(objective, labels, prepended conversation)"])
    A --configured by--> D(["Attack Configurations <br>(Adversarial, Scoring, Converter)"])
    A --produces--> C(["Attack Result"])
```

To run one:

1. Initialize a **strategy** with optional **configurations** (converters, scorers, adversarial target).
2. Call `execute_async(...)` with an **objective** (and optional prepended conversation / next message).
3. Receive an **`AttackResult`** describing what happened and whether the objective was met.

The context is created for you from the `execute_async` arguments — you rarely build one by hand.
See [Attack Configuration](3_attack_configuration.ipynb) for what you can put in the context and
configs (prepended conversations, multimodal seeds, next-turn messages, memory labels).

The category pages above each walk through their executors with short runnable examples.

## When the task owns the final scorer

`RedTeamingAttack` normally requires a PyRIT objective scorer and scores responses during
its turn loop. A task with its own final scorer can opt into
`terminal_scoring=RedTeamingTerminalScoring.EXTERNAL_FINAL` instead. This mode accepts **no**
PyRIT objective or auxiliary scorer. It runs the bounded conversation without creating a
synthetic score or an early attack result, then returns a
`RedTeamingPendingExternalResult`: the final response and conversation are retained, but
`automated_score` is `None` and the outcome is `UNDETERMINED`.

The task must acquire `attack.external_final_scoring_session()` **before** the first turn and
execute through `session.execute_with_context_async(context=...)`. Direct execution without
the session is rejected. The session keeps target-side conversation state available until
the task has stopped its agent, independently observed exit or CLI Stop completion, and
run its **original scorer once** while the workspace is still alive. An accepted Stop
request alone does not establish that the agent exited. The task should quiesce its agent
in a bounded `finally` block, issuing at most one Stop request even when execution fails.
An original task scorer may inspect the full execution state or live workspace; PyRIT
does not import its executable definition or assume it is safe to call between turns.
Exiting the session resets each invoked target conversation once. Do this while the
workspace is still available: some targets may need it during reset. A separate outer
`finally` then tears down the workspace, including after cancellation or reset failure.
Only after grading, quiescence, reset, and teardown all succeed should the task persist
its original score and one graded `AttackResult`. Neither an errored/blocked target
response nor a failed cleanup is gradeable pending evidence.

This is an ownership sketch; `owner` methods are supplied by the calling task, not PyRIT:

```python
try:
    async with attack.external_final_scoring_session() as session:
        try:
            pending = await session.execute_with_context_async(context=context)
            await owner.request_stop_once_async()
            await owner.observe_agent_exit_async()
            verdict = await owner.original_scorer_async(pending)
        finally:
            await owner.ensure_quiesced_async()
finally:
    await owner.teardown_workspace_async()

# Persist the original score and graded result only if every step succeeded.
```

This mode intentionally has **no** per-turn progress scorer in its first version:
`Scorer.score_async` persists scores before returning, so an adapter cannot safely
label an arbitrary scorer's output as progress afterward. A future opt-in progress
signal needs provenance applied before that write; whether a true progress signal
stops the conversation should be configurable and off by default. The existing
internal-scoring mode and other attacks keep their current behavior.

## When do you actually need a new executor class?

Most of an executor's behavior comes from its *configuration and data*, not from new code. So before
writing a new executor class, ask whether the algorithm is genuinely new — or whether an existing
executor with different primitives would do.

For attacks specifically, the durable value of a new class is **adaptive decision-making**: branching
and backtracking based on the objective target's feedback, like searching a graph for a path that
works. Crescendo and TAP are the clearest examples — and you can reshape them substantially just by
swapping their *primitives* (system prompt, converters, scorers, prepended/simulated conversations)
rather than writing a new class.

A lot of what *looks* like a distinct executor isn't a new algorithm at all:

- **Pure prompt transformations** — obfuscating, or deconstructing-and-reconstructing a prompt — are
  better expressed as [converters](../converters/0_converters.ipynb) than as attack classes.
- **Fixed framings** — a role-play wrapper, a primed Q&A history — are really a prepended conversation
  plus seeds, i.e. an [attack technique](../scenarios/0_attack_techniques.ipynb) over an existing
  attack like `PromptSendingAttack`.
- **New datasets or criteria** — a different benchmark question set or a different scorer is data and
  configuration for an existing executor, not a new class.

Several of the single-turn attacks in this section predate this guidance and remain as classes for
compatibility. When you are building something new, prefer configuration, a converter, or a technique —
reach for a new executor class only when you genuinely need a new algorithm (most often a
feedback-driven loop).
