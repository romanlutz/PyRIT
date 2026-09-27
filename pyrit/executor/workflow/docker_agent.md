# Stop-only native Engine launcher

This provider implements the existing `SandboxProcessLauncher` and
`SandboxProcessSession` protocols for `NativeCliRunner`. It is native PyRIT,
not an Inspect or guest-writable receipt adapter. The implementation and tests
use exact daemon exec/container identities; the tests use only mocked HTTP,
byte streams and Compose control responses. **No actual Docker/CLI, model,
credential, network or VM run qualifies this implementation.**

## Explicit profile and ownership

`DockerStopOnlyAgentLease` subclasses the strict Compose lease without changing
its defaults. It requires exactly one service whose role is **only** `agent`,
and at least one separate `target` or `grader`. Other services remain owned by
the original project lifecycle. An agent shared with a target/grader/controller
is rejected, and the provider never selects a container by a friendly name.

The trusted binding supplies `DockerAgentCliProfile(image, executable, config,
artifact_policy=AgentArtifactPolicy.TARGET_SIDE_ONLY, codex_config_sha256=...)`.
The configuration digest is mandatory only for Codex. The image reference is
digest-pinned, the executable is an absolute guest path outside writable tmpfs,
and `NativeCliRunConfig` pins the CLI version, protocol, profile identifier,
workdir, gateway and limits. The run config must match exactly at launch.
The version/profile assertion is trusted prebuilt-image provenance, not an
executable-version measurement. Image symlink targets and interpreters still
need independent qualification.

This first launcher uses fixed documented JSONL argv:

- Codex: `executable exec --json -- <one prepared prompt argument>`
- Claude: `executable --print --output-format stream-json --verbose -- <one prepared prompt argument>`

No shell, candidate flags, host coding CLI, primary credential, stdin protocol,
download, auth fallback or permission-bypass flags are added. The profile
identifier is a binding identity, not an implicit CLI `--profile` option.
Any required CLI configuration must already be staged by the trusted image.
The prepared prompt travels in Engine JSON over the host socket, not a host
subprocess command line.

The image environment is limited to PATH/locale, HOME/tmp/XDG paths, the
protocol's exact approved base-URL field, and Claude's pinned `ANTHROPIC_MODEL`.
Image-baked credentials and unknown fields are rejected. HOME/cache paths must
lie under approved ephemeral mounts; the base URL and model must match the
run route. Only the separate guest-only exec credentials described below are
injected dynamically. These checks do not prove live routing or connectivity.

Stopping the agent discards its tmpfs. **This profile is only for graders that
need target-side state and no agent-workspace artifacts after stop.**
`FROZEN_AGENT_ARTIFACTS` is an explicit rejected policy, not an alias for stop.
No pause, archive, `docker cp`, persistent volume or freeze-and-read lifetime
is implemented.

## Ephemeral guest gateway authentication

The host controller generates a **new** independent guest token for every run,
for example with `secrets.token_urlsafe(32)`, and creates one
`GatewayRoute(run_id, model, guest_token)` for the run's budget-capped app.
The provider receives
`DockerGuestAuth.from_route(route=route, protocol=config.protocol)` through the
required `DockerStopOnlyAgentLease(..., guest_auth=...)` argument. It validates
the exact run/protocol, retains the token as `SecretStr` with a redacted repr,
and accepts no primary host credential. The host-only `HttpxResponsesBackend`
or `HttpxMessagesBackend` constructor separately rejects reuse of this guest
token as its upstream credential. The host key is never passed to DockerAgent
or Engine even for that comparison.

Only the following two entries are sent in the **exec-create JSON body**:

| CLI | Exec-scoped environment |
| --- | --- |
| Codex | `PYRIT_GUEST_MODEL_TOKEN=<guest token>`, `PYRIT_RUN_ID=<exact run ID>` |
| Claude | `ANTHROPIC_AUTH_TOKEN=<guest token>`, `ANTHROPIC_CUSTOM_HEADERS=X-PyRIT-Run-ID: <exact run ID>` |

These are not image/Compose environment, host environment, CLI argv, profile
metadata or log fields. Engine handles and stop observations do not retain
them. Do not log or persist the sensitive Engine create body or the plaintext
values returned solely for serialization. The provider rejects a known guest
token embedded in public profile/image metadata or prompt argv. Both gateway
apps require the bearer **and** the exact run header; they enforce the
host-approved per-run request/token/byte budgets and never expose the host key.

Guest tools **can read this ephemeral bearer** from their exec environment and
can spend that run's remaining approved budget. It is not secret from the
evaluated agent, is not a primary credential, and does not prove caller/tool
identity inside the run. The controller must close/revoke its run-scoped app at
the end. Sensitive raw guest output remains governed by the evidence policy;
this provider does not promise arbitrary guest output redaction.

### Codex user-level configuration is a bootstrap requirement

`codex_gateway_config(model=route.model, base_url=config.model_gateway_endpoint)`
returns the exact **credential-free** TOML template for
`$HOME/.codex/config.toml`. It selects `model_provider = "pyrit_gateway"` with
`wire_api = "responses"`, `requires_openai_auth = false`,
`env_key = "PYRIT_GUEST_MODEL_TOKEN"`, and
`env_http_headers = { "X-PyRIT-Run-ID" = "PYRIT_RUN_ID" }`. No provider key,
literal guest token or workspace-level auth override belongs in this template.

The profile's `codex_config_sha256` must match its UTF-8 bytes for the exact
route model/base URL. The trusted prebuilt image and container must assert
`org.pyrit.native.codex-user-config-sha256` and
`org.pyrit.native.codex-user-config-path` for that digest and user-level path.
Missing or mismatched pins fail before exec; Claude instead requires the image
to pin the exact `ANTHROPIC_MODEL`.

**Those labels are bootstrap assertions, not proof that the file exists.**
HOME is on empty tmpfs: baking a file under image HOME would hide it at startup.
Compose explicitly clears the image entrypoint, so the separately approved
Compose service command must create HOME and copy the pinned template there,
or a separately reviewed exact staging primitive must do so before readiness.
No archive/copy/staging primitive is implemented by this patch. Actual bootstrap
behavior, file presence/ownership, CLI profile selection and authenticated
model-route use must be qualified before a real binding is declared runnable.

### Completion belongs to the controller and evidence store

ExecInspect exposes process/pipe identity and exit status, **not Env**. The
provider tests verify the exact exec-create request; they do not assert that
an Engine inspection proves env delivery or that a live CLI used it.
`wait_async` still reports the actual guest exit and `stop_async` still proves
only the owned stop barrier. An observed exit zero is not erased because model
traffic is missing, and the provider does not import/read the PyRIT DB.

The controller must declare required MODEL streams, start
`NativeCliDatabaseEvidenceSink(include_model_gateway=True)`, wire the actual
gateway observation callback to it, finish the sink, and check pregrading
coverage before calling the original grader. A real authenticated, run-specific
gateway request and completed response must be retained in DB. Missing traffic,
failed auth, response/capture gaps or a disconnected observer keep the result
incomplete; `finalize_cli_episode_atomic` must yield an UNDETERMINED Score rather
than treating exit zero or a metadata pin as model-route evidence.

Tests exercise both actual apps in-process using fake Engine Env and SQLite
capture, including missing/foreign tokens and run headers. They simulate CLI
header construction from the pinned template/environment, not a real CLI or
network call. Live Codex/Claude routing remains unqualified.

## Engine transport and attribution

`DockerEngineClient.for_unix_socket(socket_path=...)` constructs a host-only
`httpx` transport for a local POSIX Unix socket. The socket is never mounted
inside a container. Windows named pipes and remote Engine endpoints are not
implemented. API version is explicit (default `1.47`); no fallback or version
negotiation changes the requested behavior. The separately supplied Compose
control runner must name the same trusted daemon.

The client can also receive an explicit trusted `AsyncBaseTransport`, used by
all tests. It owns a fixed HTTP origin with no auth, cookies, ambient proxy or
redirect following. A production binding must use the Unix-socket construction,
not give a candidate control of that injected transport.

Before launch, the provider compares Engine and Compose observations of every
full container ID, immutable image, labels, user, HostConfig, and mounts. The
network identity/security and exact attached service IDs are also checked.
This binds both control surfaces to the same observed allocation. The daemon
and exclusive administrative control of it are trust prerequisites. Labels
and inspection cannot protect against a malicious daemon administrator or
prove containment of an escaped process.

`create_exec_async` uses the full agent ID with nonroot user, explicit workdir,
argv, stdout/stderr attachment, no stdin, no TTY, no privilege override, and
only the explicit guest gateway auth environment. The response must contain a new full daemon exec ID. Exec inspection
must match that ID, `ContainerID`, ProcessConfig argv/user/nonprivileged/no-TTY,
and the requested pipes. The exec is single-use even when start fails.

`start_exec_async` accepts **only an unencoded HTTP 200 Docker multiplex stream**.
ExecStart is a hijacked connection on some Engine/client versions; HTTP 101
Upgrade and other unsupported responses are rejected without interpreting them
as stdout, and the launcher still attempts the agent-stop barrier because start
may have happened. A future readiness check must qualify the approved daemon's
HTTP 200 attach behavior before any model run.

The decoder checks Docker's eight-byte headers and demultiplexes only stdout
and stderr. Original bytes, including invalid UTF-8, are yielded in observed
wire-frame order, not claimed wall-clock ordering between guest writes.
Partial/invalid frames, transport errors and byte limits fail explicitly; raw
daemon diagnostics are never relabeled as guest stderr. Output is bounded to
32 MiB of wire bytes and 8 MiB per daemon frame by default, with yielded pieces
at most 16 KiB. The run deadline bounds reading and exit polling.

After clean stream EOF, `wait_async()` inspects the **same daemon exec** until
`Running=false`, with an actual integer `ExitCode`. Missing/mismatched identity
or exit information raises. Docker-client exit codes and provider JSONL terminal
messages are not substitutes. A child retaining output pipes can prevent EOF;
that becomes a timeout and stop, not a fabricated successful exit.

## Stop before grading

`NativeCliRunner` calls the session's `stop_async()` in `finally`, even after
an exit-zero complete stream. The session calls lease-owned `stop_agent_async()`:

1. Serialize against launch and project close, and recheck exact allocation,
   labels, immutable security/image/profile and network membership.
2. Send Engine `kill?signal=SIGKILL` **only** for the bound agent container if
   it is running. A 204 response alone is not quiescence evidence.
3. Reinspect until the exact agent has `Running=false`, `Status=exited`,
   `Pid=0`, and is not paused/restarting/dead. The pinned restart policy is `no`.
   Every other allocated service must still have the same identity/configuration
   and be running with a live PID.

Under the trusted Engine/container-isolation contract, stopping that container
stops its contained agent/tool processes. It does not attest escaped host work
or prove target-side asynchronous effects have settled. The original grader
still owns any target-specific stabilization requirement.

The barrier has a separate bounded deadline (default 30 seconds, no more than
the runner's configured cleanup timeout). Stop failures/timeouts are latched;
repeat calls do not retry or convert unknown state to success. Repeated caller
cancellation waits for the barrier's bounded attempt and still propagates.
Only an observed stopped state populates a successful `agent_stop` observation.
No successful runner outcome reaches a caller when the stop barrier fails.

Failed/cancelled launch before a session can be returned also performs this
compensation, including an uncertain create or unsupported attach. Full-project
cleanup is a separate, serialized action that may remove target services only
when the owner closes the lease. The caller must retain that lease through
target-side grading, and must not concurrently close it while a grader is using
targets. The generic lease snapshot represents resource ownership; after stop,
the explicit agent-stop observation is authoritative and all-service readiness
checks are rejected. There is no retained session or restart path.

## Limits and future seams

- `ComposeAllocation` identity and the new stop observation are in-memory, not
  a new raw-log or task-result database schema. The existing evidence sink owns
  the actual raw chunks and parsed observations. Session `exec_id` and
  `container_id` are read-only daemon identities for caller-owned correlation;
  the stop observation records the known exec ID or `None` when creation was
  ambiguous. No identity or guest exit code is invented for that failure case.
- Current mocked tests are not a sandbox attestation, real image compatibility
  result, daemon API/attach qualification, or an auth/model-transport proof.
- Engine network inspection can expose `IPAM.Config[].Gateway`. This is network
  metadata, **not a qualified stable host bind address**. Reachability from the
  isolated agent, correct host namespace/interface, listener ownership, firewall
  rules and the no-egress boundary would need separate proof. This slice neither
  guesses an address nor opens a listener or changes network policy.
