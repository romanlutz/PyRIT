# Memory Types

There are several types of data you can retrieve from memory at any point in time using the `MemoryInterface`.

## MessagePiece and Message

One of the most fundamental data structures in PyRIT is [MessagePiece](../../../pyrit/models/message_piece.py) and [Message](../../../pyrit/models/message.py). These classes provide the foundation for multi-modal interaction tracking throughout the framework.

### MessagePiece

`MessagePiece` represents a single piece of a request to a target. It is the atomic unit that is stored in the database and contains comprehensive metadata about each interaction.

**Key Fields:**

- **`id`**: Unique identifier for the piece (UUID)
- **`conversation_id`**: Identifier grouping pieces into a single conversation with a target
- **`sequence`**: Order of the piece within a conversation
- **`role`**: The role in the conversation (e.g., `user`, `assistant`, `system`)
- **`original_value`**: The original prompt text or file path (for images, audio, etc.)
- **`original_value_data_type`**: The data type of the original value (e.g., `text`, `image_path`, `audio_path`)
- **`converted_value`**: The prompt after any conversions/transformations have been applied
- **`converted_value_data_type`**: The data type of the converted value
- **`labels`**: Dictionary of labels for categorization and filtering
- **`prompt_metadata`**: Component-specific metadata (e.g., blob URIs, document types)
- **`converter_identifiers`**: List of converters applied to transform the prompt
- **`response_error`**: Error status (e.g., `none`, `blocked`, `processing`)
- **`timestamp`**: When the piece was created

This rich context allows PyRIT to track the full lifecycle of each interaction, including transformations, targeting, scoring, and error handling.

### Message

`Message` represents a single request or response to a target and can contain multiple `MessagePieces`. This allows for multi-modal interactions where, for example, you send both text and an image in a single request.

**Examples:**
- A text-only message: 1 `Message` containing 1 `MessagePiece`
- An image with caption: 1 `Message` containing 2 `MessagePieces` (text + image)
- A conversation: Multiple `Messages` linked by the same `conversation_id`

**Validation Rules:**
- All `MessagePieces` in a `Message` must share the same
   -  `conversation_id`
   - `sequence` number
   - `role`
- All `MessagePieces` have a non-null `converted_value`

### Conversation Structure

A conversation is a list of `Messages` that share the same `conversation_id`. The sequence of the `MessagePieces` and their corresponding `Messages` dictates the order of the conversation.

A conversation is always held with a single target. That target's identifier is recorded once per conversation in the `Conversations` table (`target_identifier`) rather than on every `MessagePiece`. Use `memory._get_conversation(conversation_id=...)` to retrieve it.

Here is a sample conversation made up of three `Messages` which all share the same conversation ID. The first `Message` is the `system` message, followed by a multi-modal `user` prompt with a text `MessagePiece` and an image `MessagePiece`, and finally the `assistant` response in the form of a text `MessagePiece`.

```{mermaid}
flowchart
   subgraph Conversation: 001
      subgraph Message: sequence 2
         subgraph "MessagePiece: <br>sequence: 2<br>conversation_id: 001<br>role: assistant<br>value: The image shows a wave ..."
         end
      end
      subgraph Message: sequence 1
         subgraph "MessagePiece: <br>sequence: 1<br>conversation_id: 001<br>role: user<br>value: tell me what's in this image"
         end
         subgraph "MessagePiece: <br>sequence: 1<br>conversation_id: 001<br>role: user<br>value: data/wave.png"
         end
      end
      subgraph Message: sequence 0
         subgraph "MessagePiece: <br>sequence: 0<br>conversation_id: 001<br>role: system<br>value: be a helpful assistant"
         end
      end
   end
```

This architecture is plumbed throughout PyRIT, providing flexibility to interact with various modalities seamlessly. All pieces are stored in the database as individual `MessagePieces` and are reassembled when needed. The `PromptNormalizer` automatically adds these to the database as prompts are sent.

## Seeds

All seed types inherit from [`Seed`](../../../pyrit/models/seeds/seed.py), which provides common fields (`value`, `value_sha256`, `dataset_name`, `harm_categories`, `is_general_technique`, `metadata`, etc.) along with Jinja2 templating and YAML loading support.

### Seed Types

- [`SeedPrompt`](../../../pyrit/models/seeds/seed_prompt.py) — A prompt to send to a target. Adds `data_type` (text, image_path, audio_path, etc.), `role` (user/assistant), `sequence` (for multi-turn ordering), and template `parameters`. This is the most common seed type and can be translated to and from `MessagePieces`.

- [`SeedObjective`](../../../pyrit/models/seeds/seed_objective.py) — The goal of an attack (e.g., "Generate hate speech content"). Always text. Cannot be a general technique.

- [`SeedSimulatedConversation`](../../../pyrit/models/seeds/seed_simulated_conversation.py) — Configuration for dynamically generating multi-turn conversations. Specifies system prompt paths, number of turns, and sequence offsets. The actual generation happens in the executor layer.

### Seed Groups

Seeds are organized into [`SeedGroup`](../../../pyrit/models/seeds/seed_group.py) containers that enforce consistency (shared `prompt_group_id`, valid role sequences, no duplicate sequence numbers). Two specialized subclasses add further constraints:

- [`AttackSeedGroup`](../../../pyrit/models/seeds/attack_seed_group.py) — Requires exactly one `SeedObjective`. Represents a complete attack specification: an objective plus optional prompts or simulated conversation config.

- [`AttackTechniqueSeedGroup`](../../../pyrit/models/seeds/attack_technique_seed_group.py) — All seeds must have `is_general_technique=True` and no `SeedObjective` is allowed. Represents reusable attack techniques (jailbreaks, role-plays, etc.) that can be composed with any objective.


## Scores

[`Score`](../../../pyrit/models/score.py) objects represent evaluations of prompts or responses. Scores are generated by scorer components and attached to `MessagePieces` to track the success or characteristics of attacks. When a prompt is scored, it is added to the database and can be queried later.

**Key Fields:**

- **`score_value`**: The actual score (e.g., `"true"`, `"0.75"`)
- **`score_value_description`**: Human-readable description of the score
- **`score_type`**: Type of score (`true_false` or `float_scale`)
- **`score_category`**: Categories the score applies to (e.g., `["hate", "violence"]`)
- **`score_rationale`**: Explanation of why the score was assigned
- **`scorer_class_identifier`**: Information about the scorer that generated this score
- **`message_piece_id`**: The ID of the piece/response being scored
- **`objective`**: The original attacker's objective being evaluated
- **`score_metadata`**: Custom metadata specific to the scorer

Scores enable automated evaluation of attack success, content harmfulness, and other metrics throughout PyRIT's red teaming workflows.

## Native cyber evidence

`memory.native_cyber_evidence` is the SQLite/Azure SQL persistence contract for native agent evaluations. A run has a stable `run_id`, numbered **outer** turns, source-tagged events with a controller-observed sequence and optional real provider event/session IDs and raw-stream byte offsets, and tool request/start/completion links. The `(run_id, sequence)` pair identifies an event; provider IDs may be absent or repeated across tool phases or message blocks and are never generated by memory. Only genuine user/assistant `MessagePiece` IDs are attached to turns. Tool arguments, results, and other native payloads stay in the event evidence rather than becoming invented chat messages.

Use `NativeCyberCapturedEvent.from_native_agent_event` for GHCP's existing typed session events. Other native recorders create `NativeCyberObservedEvent` values directly, leaving `source_event_id` null for frames without a real event ID and retaining any observed stream ID/byte offset or tool call ID. Repeated source event IDs and raw source-stream IDs are correlations, not uniqueness keys; each captured stream has its own `stream_id` and ordered chunks.

The current finalizers accept `NativeCyberReport`, whose embedded agent event type is GHCP-specific. Codex/Claude frames can be retained without invented IDs now, but a later report contract must represent those providers before their captured events can satisfy final report matching for a complete verdict.

Before capture, call `create_episode(start=NativeCyberEpisodeStart(...))` with the binding identity, observed source identity when available, a raw-byte quota, and the task's **required** `NativeCyberRawStreamKey` manifest. For each outer turn, call `begin_turn`, append captured `NativeCyberCapturedEvent` records in their original global sequence, and call `finish_turn` with the source's event count and coverage claim. Open each model/tool/harness stdout, stderr, JSONL, or native byte stream with its observed source ID, append the captured bytes, then close it with the source's length, SHA-256, and coverage claim. Calls to `append_raw` accept at most 1 MiB and write at most 64 KiB per database row; the default run quota is 256 MiB and is configurable. Its receipt reports stored and omitted bytes, and a capped, missing, corrupt, or unclosed stream remains explicitly incomplete. Optional telemetry gaps remain visible but do not change a verdict unless the stream was declared task-required.

The recorder must call `mark_capture_gap(run_id=..., reason=..., required=...)` if a source fails before its stream can be opened or a dynamically discovered segment is missed. Memory can verify declared streams and submitted bytes, not infer a source the recorder never reports.

The evidence API is append-only until finalization; linked message pieces carry integrity digests, and ordinary memory updates cannot rewrite those pieces. SQLite triggers also reject edits/deletes of linked pieces and the finalized Score/report content. On SQL Server, foreign keys protect deletion and ordinary prompt-update APIs guard linked messages; privileged direct database edits are outside this provenance boundary.

Raw DB appends and stream closure must occur **before** original grading. Once the lease is closed and the canonical `NativeCyberReport` includes the cleanup outcome, call `assess_required_coverage(report=..., expected_turns=...)` **before** scoring. If task-required capture is incomplete, the owning workflow must construct its final canonical report with an error status and those coverage errors before scoring. Keep any real `NativeCyberJudgment` in that report; an undetermined PyRIT Score does not erase the original judgment.

**Preferred single transaction:** When the caller-owned scorer can prepare a validated but **unpersisted** `Score` anchored to `ContentScorable(value=report.canonical_json())`, pass it to `finalize_episode_atomic(report=..., score=..., expected_turns=...)` instead of invoking `score_async`. Memory does not call the scorer. It rechecks required capture, prepares the standard content anchor, and commits the canonical report content, one standard generic Score row, and the episode link in one database transaction. A failed commit leaves no orphan Score/content; a missing required stream is stored as `UNDETERMINED` from the start. An already-persisted Score ID is rejected.

**Existing-score compatibility:** `NativeCyberReportScorer.score_async` already persists one generic `Score` and `ScorableContentEntry`. If that API was used, pass its *existing* Score to `finalize_episode(report=..., score=..., expected_turns=...)`, which checks the canonical anchor, links the episode, and defensively downgrades that same Score ID to `UNDETERMINED` when required evidence is missing. It never creates a second Score or content row. These scorer and link commits are **two separate transactions**. If linking fails after a complete Score was committed, that orphan is still visible through generic `get_scores`; the episode remains unfinalized, no run-facing success receipt is returned, and write failures are marked incomplete when the database can accept that marker. Run-facing callers must use `get_finalized_episode`, which refuses an unfinalized episode; after an uncertain commit, re-read the episode before retrying. A database outage cannot guarantee a durable undetermined Score and must be surfaced, not treated as success.

`get_episode` returns metadata, coverage gaps, report/Score links, and byte digests without raw event payloads or bytes. `read_event_payloads` and paged `read_raw_chunks` require an explicit `allow_sensitive=True`; this flag is **not authentication**, so callers must authorize access themselves and must not expose these methods through a generic GUI route. Call the synchronous memory methods from an async workflow using `asyncio.to_thread` to avoid blocking the event loop.

## AttackResults

[`AttackResult`](../../../pyrit/models/results/attack_result.py) objects encapsulate the complete outcome of an attack execution, including metrics, evidence, and success determination. When an attack is run, the AttackResult is added to the database and can be queried later.

**Key Fields:**

- **`conversation_id`**: The conversation that produced this result
- **`objective`**: Natural-language description of the attacker's goal
- **`atomic_attack_identifier`**: Composite `ComponentIdentifier` combining the attack technique with seed identifiers from the dataset (see [ComponentIdentifiers](#componentidentifiers) below)
- **`last_response`**: The final `MessagePiece` generated in the attack
- **`last_score`**: The final score assigned to the last response
- **`executed_turns`**: Number of turns executed in the attack
- **`execution_time_ms`**: Total execution time in milliseconds
- **`outcome`**: The attack outcome (`SUCCESS`, `FAILURE`, or `UNDETERMINED`)
- **`outcome_reason`**: Optional explanation for the outcome
- **`related_conversations`**: Set of related conversation references
- **`metadata`**: Arbitrary metadata about the attack execution
- **`targeted_harm_categories`**: Harm categories this attack targeted, auto-populated from the attack's seed group

`AttackResult` objects provide comprehensive reporting on attack campaigns, enabling analysis of red teaming effectiveness and vulnerability identification.

## ComponentIdentifiers

[`ComponentIdentifier`](../../../pyrit/identifiers/component_identifier.py) is an immutable snapshot of a component's behavioral configuration. A single type is used for all components — targets, scorers, converters, and attacks — enabling uniform storage and composition.

**Key Fields:**

- **`class_name`** / **`class_module`**: The Python class and module of the component
- **`params`**: Behavioral parameters (e.g., `temperature`, `model_name`)
- **`children`**: Named child identifiers for composition (e.g., a scorer's `prompt_target`)
- **`hash`**: Content-addressed SHA256 hash computed from class, params, and children

Identifiers are content-addressed: the same configuration always produces the same hash, and any change to params or children produces a different one. This is used throughout PyRIT to track which exact configuration produced a given result.

### Composite Identifiers

For atomic attacks, `AtomicAttackIdentifier.build` composes a tree of identifiers:

- **`attack_technique`** — the attack strategy and its children (target, converters, scorer, technique seeds)
- **`seed_identifiers`** — all seeds from the seed group, for traceability

### Eval Hashing

[`EvaluationIdentifier`](../../../pyrit/identifiers/evaluation_identifier.py) subclasses wrap a `ComponentIdentifier` and compute a separate **eval hash** that strips operational params (like endpoint URLs) so the same logical configuration on different deployments produces the same hash. This enables grouping equivalent runs for evaluation comparison.

What feeds the eval hash is declared **on the strongly-typed identifier fields themselves**, via `Evaluate.*` markers attached as [`typing.Annotated`](https://docs.python.org/3/library/typing.html#typing.Annotated) metadata. This keeps the identifier classes the single source of truth — you change what an eval hash includes by editing the field, not a separate rules table:

- `Evaluate.Include()` — keep this field in the eval hash (the default for an unmarked field). On a param, `fallback="model_name"` substitutes another param's value when this one is empty. On a child, `only_params={...}` restricts the child subtree to those params.
- `Evaluate.Exclude()` — drop the field (param or child) from the eval hash. Operational target params like `endpoint`, `model_name`, and `max_requests_per_minute` are excluded this way.
- `Evaluate.Unwrap()` — mark a wrapper passthrough slot (e.g. `TargetIdentifier.targets`). A multi-target like `RoundRobinTarget` is "looked through" to its inner target, so it eval-hashes the same as the bare inner target.

For example, `TargetIdentifier` excludes `endpoint` but includes `temperature`, and the `ObjectiveTargetEvaluationIdentifier` / `ScorerEvaluationIdentifier` / `AtomicAttackEvaluationIdentifier` subclasses derive their engine rules from these markers (via `derive_eval_config`). Markers affect **only** the eval hash — the identity `hash` always keeps distinct components (e.g. a wrapper vs. its inner target) distinct.
