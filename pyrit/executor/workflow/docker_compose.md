# Native Docker Compose lease

`DockerComposeEnvironmentLease` is a concrete control-plane implementation of
`EnvironmentLease[ComposeAllocation]`. It creates no agent target, task prompt,
grader, credentials or model transport. A trusted binding can use its verified
named container IDs to construct a sandbox-local runtime later. This slice was
validated only with an in-memory command runner and mocked subprocesses. No
Docker engine, image, CLI or live task was exercised.

## Input and boundary

`ComposeEnvironmentSpec(services=..., wait_timeout_seconds=60)` accepts one
through 32 `ComposeServiceSpec` instances with unique names and optional
earlier-declared parent references. Each service supplies:

- Open-ended roles, a named SHA256-pinned prebuilt image and Linux platform.
- Literal command and exec-form health-check argv.
- Nonroot UID/GID and explicit CPU, memory, PID and bounded `/tmp` sizes.

There is no arbitrary YAML/JSON loader, extra-options dictionary, candidate
environment, host mount, volume, port, device, secret, build, privilege or capability option.
Unknown fields, unpinned images and interpolation tokens fail validation.
Commands and health checks may not contain `$`, NUL or line breaks: Compose
interpolates strings even when its input is JSON. Labels also use literal
validated run identities. Task images and their baked-in content/environment
remain trusted approved inputs, not something this provider can certify.

The default manifest uses an internal bridge challenge network, no published
ports, read-only root filesystems, nonroot users, all capabilities dropped,
no-new-privileges and a bounded noexec/nosuid/nodev `/tmp`. These defaults are
unchanged when `runtime_state` is omitted. Privileged tooling, writable root
filesystems, host services and Internet/model access remain unsupported. An
internal Docker network is not a claim of a fully qualified hostile-code sandbox
or isolation from every Docker-host service. Those boundaries require separate
provider qualification.
In particular, compatibility with the original GDM target image is unproven:
services such as Grafana may require writable runtime state, and sandbox-local
CLI harnesses may require an executable extraction/cache path. The opt-in policy
below models these needs; it does not qualify GDM, GHCP, Claude or Codex.
Original image/benchmark parity, safe staging and model transport, and a
reconciled cleanup budget remain readiness blockers for those future bindings.

## Opt-in ephemeral runtime state

`ComposeServiceSpec.runtime_state` optionally specifies a complete tuple of
`ComposeTmpfsSpec(path, size_bytes, executable=False)` mounts for that service.
It must include `/tmp` explicitly. The existing `tmpfs_bytes` field is the total
budget: all mount sizes combined must fit within it and the service memory
limit. At most eight nonoverlapping paths and at most 1 GiB total are supported.
No policy enables persistence or a host source.

Each path is a canonical absolute Linux directory with no lexical traversal or normalization aliases,
interpolation, option delimiters or whitespace. Duplicate paths, ancestor/child
overlaps, system/device/Docker-managed paths, and mounts hiding the approved
command or health-check executable are rejected. All mounts are writable, owned
by the service's nonroot UID/GID, `mode=0700`, `nosuid` and `nodev`. Only an
explicit `executable=True` changes `noexec` to `exec` for that one path.
Owner, permission and arbitrary mount-option overrides are not accepted.

For example, a **trusted image binding**, not candidate input, could select:

```python
from pyrit.executor.workflow.docker_compose import ComposeTmpfsSpec

runtime_state = (
    ComposeTmpfsSpec(path="/tmp", size_bytes=16 * 1024 * 1024),
    ComposeTmpfsSpec(path="/workspace", size_bytes=32 * 1024 * 1024, executable=True),
    ComposeTmpfsSpec(path="/home/runner", size_bytes=16 * 1024 * 1024),
)
```

This requires `tmpfs_bytes=64 * 1024 * 1024` and a sufficient service memory
limit. A separate target could instead approve noexec `/var/lib/grafana` and
`/var/log/grafana`. Those are storage-policy examples, not tested Grafana paths
or real pinned CLI images. Tmpfs starts empty and hides image content at its
mountpoint. A separately reviewed bootstrap in the prebuilt digest-pinned image
must stage any required workspace files, binaries and nonsecret HOME/cache
configuration. This policy adds no asset-transfer or guest-exec API and no
runtime environment override. Compose still starts the approved service command
and runs its declared health check. No downloads, authentication or SDK setup
are added by this policy.

Image-declared `VOLUME` remains a blocker even if its destination matches an
approved tmpfs path. Named/anonymous volumes and persistence are never silently
substituted. If the original target requires them or relies on mountpoint
content that cannot be safely staged into empty tmpfs, the binding must stop
and report that incompatibility. No original Grafana parity or CLI startup is
proven by the mocked profile tests.

## Control transport

Pass `runner: DockerCommandRunner` explicitly to the lease. The included
`SubprocessDockerRunner` takes absolute host-owned executable, empty working
directory and empty Docker-config directory paths, plus an explicit local
Unix-socket or Windows named-pipe daemon endpoint. It does not inherit Docker
contexts, proxies, candidate variables, API keys or the host HOME configuration.
Directory provisioning and ACL verification remain the trusted host binding's
responsibility. The CLI/plugin installation and local daemon are trusted.

Compose receives the same generated manifest through stdin with `--file -`,
an explicit project name and `--env-file` set to the platform's null device.
The subprocess environment also disables automatic `.env` loading. No host
shell is used. Output is drained with bounded capture; nonzero exit, timeout,
truncation or malformed inspection output cannot count as success.
Cancellation/timeout terminates and reaps the owned CLI process group/tree.
The runner never treats terminating a CLI process as proof that daemon-side
resources were removed.

## Acquisition and release

The lease reserves a UUID-derived project resource and release callback before
control I/O. It inventories the union of project labels, lease labels and the
reserved name prefix, rejecting any pre-existing resource,
inspects every pinned image locally (including rejecting image-declared
volumes), and checks vacancy again before dispatch. `compose up` explicitly
uses `--no-build --pull never --no-recreate --wait`. There is no image download,
build, registry-login or fallback code.

Before readiness, Docker inspection must show exactly one container per named
service and exactly one approved internal challenge network, without extra
volumes. Run/lease/project labels, full resource IDs, network membership,
immutable image IDs, effective user/command/environment, health commands and
running/healthy states are checked. Effective mounts, HostConfig privilege,
capabilities, security options, ports, host namespaces and resource limits are
also checked. Generating a safe-looking manifest alone is insufficient.

Runtime-state inspection compares the complete `HostConfig.Tmpfs` path/options
map, including each size, exec choice, UID/GID and permission. Any reported
`Mounts` entries must be the exact approved tmpfs destinations, writable without
host/volume sources, extra or repeated paths, conflicting modes or shared
propagation. Some Docker responses omit tmpfs entries from `Mounts`; an empty
list is accepted only with the exact approved `HostConfig.Tmpfs` map. These are
daemon inspection checks, not in-guest kernel or image-symlink attestation.
Their effective behavior still needs separate real-provider qualification.

Even failed/cancelled `up` can create resources. Rollback first re-inventories
and verifies all ownership labels, network attachments and the runtime-state,
image-VOLUME and related user/security policy for every remaining container.
Stopped or unhealthy owned services can be removed; unapproved or unmappable
runtime state, unknown mounts and image/permission drift refuse teardown and
remain explicit cleanup failures. Only after those checks may it run
`compose down` for this exact project and identical manifest, including verified
owned duplicates with an approved service policy. It never prunes, removes images,
removes volumes, or downs another project. Foreign/unknown resources or an attached foreign container stop cleanup
instead of broadening deletion. Nonzero `down` or residual resources leave an
explicit failed-cleanup snapshot, without implicit retry or a clean-success
claim. An operator may need to resolve that retained failure separately.
Control queries each have a five-second deadline, startup has the requested
health wait plus ten seconds, and project cleanup has a 150-second cooperative
ceiling. An integrating workflow must account for that ceiling in its outer
cleanup budget instead of assuming a single-container ten-second release.

Ownership labels and random names protect against accidental cross-run
collision, not a malicious actor with concurrent Docker-daemon administration.
Inspection and Compose mutation are not an atomic daemon transaction. Such an
actor can race or spoof labels, so exclusive/trusted control of the selected
daemon is a deployment prerequisite.

Only static acquisition and health observation are implemented. Dynamic service
creation and network-phase transitions remain unsupported. `ComposeAllocation`
returns verified IDs; it does not grant execution or bypass the native
workflow's host/guest storage checks, grading order, evidence policy or lineage.
