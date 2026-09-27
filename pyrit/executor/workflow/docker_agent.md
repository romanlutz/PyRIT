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
artifact_policy=AgentArtifactPolicy.TARGET_SIDE_ONLY)`. The image reference is
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

The image environment is limited to PATH/locale, HOME/tmp/XDG paths and the
protocol's exact approved base-URL field. Credential and unknown environment
fields are rejected. HOME/cache paths must lie under approved ephemeral mounts;
the gateway base URL must match the config. No environment or credential is
forwarded from the host. These checks do not prove that a particular CLI version
uses that routing field, can authenticate without a primary guest credential,
or can reach the gateway.

Stopping the agent discards its tmpfs. **This profile is only for graders that
need target-side state and no agent-workspace artifacts after stop.**
`FROZEN_AGENT_ARTIFACTS` is an explicit rejected policy, not an alias for stop.
No pause, archive, `docker cp`, persistent volume or freeze-and-read lifetime
is implemented.

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
argv, stdout/stderr attachment, no stdin, no TTY, and no privilege or environment
override. The response must contain a new full daemon exec ID. Exec inspection
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
