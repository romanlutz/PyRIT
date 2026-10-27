# Concrete owned Docker CLI connection

`docker_cli_connector.py` connects the existing native workflow to the actual
`DockerStopOnlyAgentLease`, `DockerEngineClient`, `DockerSandboxLauncher` and
protocol-specific model gateway. It is not another CLI runner or grader.
Tests replace only the external Engine HTTP, Docker control plane, listener
and model backend with explicitly synthetic in-memory fixtures. No live
daemon, CLI, socket, model, task or original benchmark is qualified here.

## Explicit configuration and lifecycle

`DockerCliQualification` binds an explicitly supplied Engine client and host
qualification identity, exact `DockerAgentCliProfile`, admitted model-backend object and identity
and qualification reference. `ready` defaults to false. It does not discover
a Docker socket, host executable, credential, backend or model. The caller
retains ownership of the Engine client and host-only model backend.

`DockerCliGateway` receives an already-admitted Anthropic Messages backend,
run-scoped `GatewayRoute`, existing `GatewayLimits`, private IPv4 subnet,
gateway address and port, and an owned listener factory. That factory can
construct the existing `RunScopedModelGatewayListener` using a caller-prebound
socket. The connector never binds a socket automatically. A foreign returned
listener is not adopted or closed; factory creation failures remain
factory-owned. Only a matching listener is transferred to runtime cleanup.

The connector's Compose subclass adds only the explicitly approved IPAM
subnet/gateway to the existing internal challenge bridge. It does not change
the default Compose provider, publish a container port, enable host networking,
add a host mount or relax image/security/resource audits. Engine inspection
must match the network identity and IPAM plan. Listener address/URL/run,
backend identity, image/profile, source sink and guest environment all bind to
that exact run before agent exec creation.

Create `OwnedDockerCliRuntime` with those pins, a two-service spec, the existing
bounded Docker control runner and the original task grader callback. Use
`OwnedDockerRuntimeLease(runtime=..., prepare_async=...)` from the task binding's
`create_environment_lease` hook. It reserves rollback before acquisition and
accounts for the provider's cleanup budget even when acquisition fails. The
optional preparation callback invokes only the task's approved setup operations.

Select `ClaudeCliHarnessAdapter(config=qualification.profile.config,
connect=OwnedDockerCliConnector())`. The connector attaches the actual
`NativeCliDatabaseEvidenceSink`; it starts the original Messages app/listener
only after required sources are open, then delegates to `DockerSandboxLauncher`.
The original Engine exec ID, container ID, process spec, multiplex output and
daemon exit remain authoritative. Model/CLI evidence is not fabricated from
Docker control success.

The process stop wrapper invokes the actual lease-owned stop barrier, checks
the exact exec/container and preserved-target receipt, settles the run's model
listener, and re-inspects resource identity/liveness before grading. SDK/CLI
EOF and exit zero are not substitutes. Listener/stop failures cannot authorize
grading. The original task grader is invoked once while the target is alive;
then the existing controller closes the owned resources and atomically
publishes report, standard Score and episode references.

## Installed target operations

`DockerTargetOperation` pins an operation name, complete argv, absolute guest
working directory, phase, output cap and deadline. Its executable must remain
on the immutable target image, outside task-writable state. No candidate argv,
host shell, executable download, arbitrary file path, primary credential or
guest bearer is accepted by the execution method.

The default operation allowance stays 10 seconds; explicitly installed,
task-qualified operations may select up to **600 seconds**, preserving an
original task's 600-second install/test rule rather than substituting 60.
`run_target_operation_until_async(name=..., deadline=DockerTargetDeadline(...))` accepts
one caller-established monotonic start/expiry pair. Its total duration must
match the exact configured operation budget; shortened/extended or expired
windows are refused before target I/O. `DockerTargetDeadline.start(...)`
establishes that window once. The existing `run_target_operation_async(name)`
establishes the same exact installed budget once on entry. Waiting, inspection, exec creation, attached output and
exit inspection consume the same window, without phase resets or retries.
Already elapsed caller time is not replenished.

The controller's overall task TTL/cancellation is still an independent outer
ceiling. A task binding must admit enough overall time for its original
operations; this option neither extends that ceiling nor any old live clock.
Engine control/header requests retain their short transport deadlines, but
exec output is streamed under the operation deadline, not read in the header
acknowledgement window. Tests model guest/model work after the attach headers,
and verify a 600-second approved window through actual Engine/stream code
without running a real long command.

`run_target_operation_async(name)` uses Engine create/start/inspect on the
exact owned target container. `SETUP` is allowed before agent launch;
`AFTER_AGENT_STOP` requires the observed stop and settled model channel.
`DockerTargetResult` retains run/environment/operation/container/exec IDs,
original stdout/stderr and actual daemon exit code. This is process evidence,
not an asserted file read/write, filesystem audit or grader verdict. A task's
installed helper implements its approved setup/read/write/fact protocol and
the original task validates the returned bytes.

`inspect_target_async()` returns `DockerTargetInspection` with exact identity,
image and liveness, not container environment or daemon host mount paths.
Interrupted or over-limit target execution closes the entire owned runtime:
the current Engine surface has no separately attributed exec-kill primitive.
There is no success fallback, retry or claim that closing HTTP alone kills
guest work. Target callbacks and listener cleanup remain cooperative, with
the existing bounded provider/resource cleanup rules.

## Optional same-workspace named volume

The default profile remains per-container tmpfs and **cannot** preserve agent
workspace files for grading. `require_agent_workspace=True` without the
separately qualified capability is refused before allocation.

`DockerNamedWorkspaceSpec(root=PurePosixPath("/work"),
qualification_ref=..., ready=False)` selects the optional
`run-owned-task-volume-v1` capability through `DockerCliQualification.workspace`.
It is a new task/image/storage profile, not automatic compatibility with every
task. The exact configured agent workdir must equal the task root. Both services
use the same nonroot UID/GID; system, home/cache/credential, helper and tmpfs
overlaps are rejected. Only the two owned agent and target/grader services may
mount it. No model gateway, host evidence store or host credentials belong
there. The volume's contents are restricted, untrusted **task data**.

`DockerNamedWorkspaceLease` reserves a typed cleanup handle before a volume
create request. It generates a 192-bit nonce name internally; callers cannot
choose an external resource name, driver, driver options or host source path.
An actual 404 inspection and empty attachment inventory precede POST. Create
and post-read must match the name, nonce/run/lease/capability labels, creation
generation, `local` driver, local scope and empty options. POST is idempotent:
nonce + absence + post-read checks do **not** make it an atomic create/CAS API.
Exclusive trusted daemon administration remains a prerequisite. Ambiguous
create outcomes or foreign metadata are quarantined, not adopted or deleted.

Compose refers to this already-created owned volume through one generated
external name. Both services use exactly the approved long-syntax volume
mount at the task root, writable, with `nocopy=true`. Existing anonymous-volume,
host-bind, image `VOLUME`, extra mount, rootfs, capability, namespace and
network guards remain. Effective Mounts and HostConfig mount name/type/path/
mode must agree with the admitted volume. Daemon Source/Mountpoint metadata
is compared privately for attribution, never used as a caller bind path,
passed to the guest or included in public target inspection.

Before any candidate runs, the same actual mounted volume is checked empty
through bounded archive GET. A generated directory-only USTAR changes only
the fresh root's UID/GID/mode to the approved nonroot owner and `0700`; it
contains no task file, credential, link or arbitrary payload. Readback must
still contain exactly that empty non-link root with the approved metadata.
No image copy-up, adoption of nonempty data, broad delete, root guest exec or
host archive extraction is used. The private task preparation then creates
the authoritative initial workspace through its installed target operation.

The agent and target now access **the same named volume**. Mutations and
deletions remain visible to the target after the exact agent is stopped:
there is no snapshot merge or separately staged pristine fallback.
`workspace_identity_async()` re-verifies ownership/generation, both service
attachments and the stopped-agent/live-target boundary, returning
`DockerWorkspaceIdentity`. Its path, volume name, nonce and service IDs are
provenance, not a content/audit verdict. The original grader still owns
semantic checks and target-specific settlement.

Cleanup releases the Compose containers first. Only then may the exact
owned volume be removed without force, after both the volume attachment list
and owned-container inventory are empty, even after partial acquisition.
Deletion requires a successful acknowledgement followed by observed absence.
Foreign labels, changed generations, attachments, failed container cleanup or
ambiguous delete outcomes do not trigger retries or deletion of other assets.

Local Docker volumes do not supply a portable per-volume byte quota in this
no-driver-options interface. Real admission therefore also needs a reviewed
storage-capacity/isolation policy, image/setup/ownership qualification and
the existing Engine/network/CLI/model-route qualification. None is established
by fake tests or by setting a readiness flag.

## Why tmpfs archive handoff and concrete Codex remain refused

For the API-1.47-era official Moby v27.3.1 implementation:

- [`containerfs_linux.go`](https://github.com/moby/moby/blob/v27.3.1/daemon/containerfs_linux.go)
  documents that archive filesystem views have private tmpfs mounts; container
  tmpfs writes and view writes are not mutually visible.
- [`archive_unix.go`](https://github.com/moby/moby/blob/v27.3.1/daemon/archive_unix.go)
  uses that view for stat/read/extract and permits writing read-only-rootfs
  destinations only through writable named/bind MountPoints.
- [`volumes_unix.go`](https://github.com/moby/moby/blob/v27.3.1/daemon/volumes_unix.go)
  excludes tmpfs from those shared MountPoints.
- [`kill.go`](https://github.com/moby/moby/blob/v27.3.1/daemon/kill.go)
  can resume a paused task after sending SIGKILL. Omitting an explicit unpause
  request is not proof of no implicit resume.

Consequently this connector does not implement a purported tmpfs workspace
handoff or infer it from pause/HTTP success. The historical archive-based Codex
HOME config stager remains mock-only and cannot qualify guest-visible tmpfs
configuration. Concrete Codex attachment is explicitly refused pending a
separately implemented/qualified guest-visible staging primitive. The existing
native Codex parser/gateway path remains available, with its independent
tool-causality restrictions. No real daemon version or behavior was probed here.
