# Native coding CLI evidence recorder

`NativeCliDatabaseEvidenceSink` is a caller-owned recorder for an existing
`NativeCyberEvidenceStore` episode. Before launching a Codex or Claude Code
process, the controller must create that episode with
`required_raw_streams=NativeCliDatabaseEvidenceSink.required_raw_streams(protocol=...)`
and any additional **task-required model gateway streams**, then call
`start_async` with a controller-assigned `turn_id`. The recorder opens that
outer turn and both process pipes before
the target runs. It does not start a CLI, create a sandbox, hold credentials,
grade, or finalize a report.

`NativeCliRunner` awaits every DB write before parsing the next chunk. The
recorder stores exact stdout/stderr bytes in bounded DB chunks and records
cross-pipe read order as separate typed source events with pipe, offset, size
and digest. Provider JSONL observations retain real source IDs, including
repeated Codex item IDs and Claude message IDs, with distinct observed
request/start/completion/result phases. No Codex model tool request is
invented from a CLI tool start. Normalized event payloads have the memory
layer's size limit; oversized observations fail capture instead of silently
truncating a complete run.

For a qualified model-only route, declare
`required_raw_streams(protocol=..., include_model_gateway=True)`, construct
the sink with `include_model_gateway=True`, and pass its
`record_gateway_observation_async` for Codex Responses, or its distinct
`record_messages_observation_async` for Claude Anthropic Messages, as the
gateway's host observation callback. Anthropic frames keep their own
`messages_gateway.*` event types, selected approved headers and `beta=true`
query, with a real `event: message_stop` terminal rather than a fabricated
Responses `[DONE]`. Genuine provider HTTP errors remain failed provider
responses; host-generated errors remain separate harness records.
Request bodies and original response/SSE frames go into separate required DB
streams, correlated by actual gateway request ID; host-generated failures
or provider responses without a completed-coverage claim remain required
gaps, never presented as successful provider bytes. A request gets at most
one terminal response; a second terminal cannot upgrade an incomplete one.
The third
optional error stream preserves the generated error frame. The fake ASGI
test proves this callback contract, **not** that a real CLI reached the
gateway. The caller must provision a run-isolated listener and model-only
provider separately; neither CLI's live gateway compatibility is qualified.

`PromptNormalizer` persists the user MessagePiece only *after* the target
returns (including its error path). The controller therefore calls
`finish_async` after that boundary with IDs of already-persisted request and
response pieces, plus the observed `NativeCliRunOutcome`. Missing pieces,
incomplete process coverage, a recorder failure, unsealed pipes, missing raw
bytes or quota omission remain required capture gaps. The controller must
not claim an original grade from this recorder. Once sealed,
`read_report_events_async` pages DB-verified event rows into bounded,
payload-free `NativeCliReportEvent` summaries for the pure CLI report
adapter; neither raw frames nor model/tool text are copied into that report.
Set the report's `raw_evidence_ref` to `db-episode:<run_id>`; the atomic
finalizer rejects a pointer to another run as complete evidence.
Capture rejects more than 10,000 parser observations, cross-pipe chunks or
model gateway frames instead of allowing an unbounded in-memory projection.
`memory.native_cyber_evidence.finalize_cli_episode_atomic` compares the
canonical CLI report to the persisted events, raw streams and host-side
model observations, then atomically publishes or downgrades a single PyRIT
Score. The caller must still gate the original grader on pre-grading
coverage, confirm guest stop, grade while the target lives and record
cleanup before finalizing. Existing GHCP-specific finalization does not
accept CLI event shapes.

Current tests use inert process bytes and SQLite only. A qualified sandbox
launcher, model gateway listener/observer, original grader and task image
remain separate prerequisites; no real CLI or Docker run is implied.
