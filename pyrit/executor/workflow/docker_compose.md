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

## Azure cyber evaluation: first deployment profile (proposal, not approval)

**Start with one disposable, dedicated Linux VM per approved Inspect `Sample`,
hosting that sample's complete Compose challenge graph, rather than distributing
its services across managed containers.** The VM is a second, deletable failure
boundary around the shared Docker daemon, not proof that untrusted code cannot
escape a container or access the VM host. Run at most one sample/attempt per VM
until concurrent-run isolation has been separately qualified. Disable automatic
Inspect retries; an authorized rerun needs a fresh VM and run identity. Azure
Container Instances groups share a local network, storage and lifecycle; Azure
Container Apps sidecars share disk, network and lifecycle and restart on crashes. Separate
managed apps/groups could provide stronger network separation later, but would
need a new provider and a qualified remote execution, file, grading and cleanup
bridge. Neither service implements this lease's inspected Docker Engine/Compose
contract. See the [ACI container-group](https://learn.microsoft.com/en-us/azure/container-instances/container-instances-container-groups)
and [Container Apps container](https://learn.microsoft.com/en-us/azure/container-apps/containers)
documentation.

This is a **candidate operating envelope, not an approved Azure deployment**.
An authorized owner must sign the exact values below before provisioning;
substituting defaults, an unversioned image, or VM self-reporting for approval
must block execution:

| Gate | Candidate selection and required approval evidence |
| --- | --- |
| Placement | `eastus2`, one private-NIC VM per sample, no public IP or inbound SSH, dedicated run-scoped resources. Approve the subscription, resource group, VNet/subnet, region, permitted target/model endpoints and service policy. Confirm regional availability and organization rules. |
| Compute | `Standard_D4s_v5` (`linux/amd64`, 4 vCPU, 16 GiB) is a *sizing candidate*, not a special physically isolated Azure SKU. Approve the actual size and upper bound on provisioning/OS-disk spend. |
| Host image | Ubuntu 24.04 LTS Gen2 with Trusted Launch and a reviewed Docker Engine/Compose installation, from **one explicitly versioned organization Azure Compute Gallery image**. Supply its full image-version ARM resource ID, build provenance and component versions/digests; neither `latest` nor a generic Marketplace alias qualifies. No gallery image ID or approved build is present in this repository. |
| Worker identity | One **system-assigned managed identity** bound to that exact VM resource ID, with its tenant and principal ID verified by the trusted Azure controller after creation. Grant only approved ACR pull scope during image preloading; no guest access to the identity, PyRIT memory, grader, model/Key Vault roles or VM management. The independent controller/watchdog identity, not the worker, owns VM create/read/delete and protected evidence storage. Approve both identities and their exact RBAC scopes. |
| Inputs and limits | Sign the Inspect Task revision, one harmless sample/epoch/attempt and dataset/asset SHA256s, the full `@sha256` image references and platforms, grader revision, model route, per-run token/request/byte budget and maximum estimated spend. Deny anything not enumerated. |

The [Dsv5 size table](https://learn.microsoft.com/en-us/azure/virtual-machines/sizes/general-purpose/dsv5-series),
[Compute Gallery version model](https://learn.microsoft.com/en-us/azure/virtual-machines/shared-image-galleries),
[Trusted Launch](https://learn.microsoft.com/en-us/azure/virtual-machines/trusted-launch)
and [VM managed identity for ACR](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-authentication-managed-identity)
support these *choices*, not their authorization or security qualification.
The VM owner must approve a host-compromise threat model: container controls do
not protect host-held secrets against an actual container escape.

### Required phases and trust boundaries

1. **Prepare without the evaluated agent.** A trusted bootstrap identity uses
   private registry access to preload every approved image by digest and stages
   only hashed, licensed Task/dataset assets and reviewed setup scripts.
   ACR Private Link is an option for this phase. Stop/restrict registry access
   before evaluation; never put ACR auth, Docker socket, host HOME, dataset
   solutions or grader keys in the guest.
   The current lease checks local image identity and runs `compose up
   --no-build --pull never`; it cannot build images or fetch a missing one.
   Inspect's default Docker sandbox can build/pull, and replacing its generated
   Compose config also removes its `network_mode: none` default. Do not feed
   arbitrary Inspect Compose YAML into `ComposeEnvironmentSpec` or assume
   Inspect's `--sandbox-prebuilt` proves digest identity.
2. **Admit exactly one sample.** The host controller verifies approved ARM VM
   identity/image/network/permissions independently of guest IMDS, plus Docker
   versions, firewall rules, resource capacity and all input hashes. Reserve
   the exact VM resource ID and ownership tags before creation; record and
   verify the returned NIC/disk IDs before deletion. Then acquire a distinct
   Compose lease/project for the sample. Give each sample its own challenge
   network, empty bounded workspace/tmpfs and short-lived guest gateway
   token. The existing `ComposeEnvironmentSpec` permits one internal bridge and
   no persistent volumes, host mounts or ports; its per-service CPU/memory/PID limits do not
   enforce *aggregate* VM quotas. A Task requiring host/shared volumes or a
   second network is **unsupported**, not an excuse to accept arbitrary YAML.
3. **Separate outbound routes from grading.** Restrict the VM VNet/NSG and
   qualified *host* firewall so guests cannot reach the Internet, Azure IMDS
   (`169.254.169.254`), the Docker daemon, host-only memory or a grader. Azure
   NSGs allow outbound Internet traffic by default; explicitly override this.
   NSG rules alone do not prove container-to-host isolation; Docker installs its
   own bridge firewall rules. The candidate agent gets no model provider
   credential: a trusted host gateway holds it, accepts only a per-sample
   revocable guest token and enforces approved model, endpoint, requests,
   tokens, bytes and deadline. `DockerGuestAuth`/`DockerStopOnlyAgentLease`
   implement a *mock-tested* token/route seam, but live gateway reachability
   from the internal bridge, listener binding and egress enforcement remain
   unqualified. Do not place a credentialed router or the original grader on
   the **same** challenge network as the untrusted agent. Keep the grader
   host-owned with a private, non-guest control path, or require a newly
   qualified two-network topology; the present lease cannot make a private
   grader network. Target-side grading after agent stop is possible only if
   its inputs survive that stop; grading that needs agent workspace artifacts
   needs a separately qualified, bounded readback path.
4. **Grade, retain, then tear down.** An Inspect Task's `sandbox()` operations
   are per-sample, but only work explicitly sent through that interface runs
   inside the sandbox; tools/solvers/scorers otherwise run in the trusted
   evaluation process. A future binding must choose *one* sample lifecycle
   owner, map that exact Inspect `Sample` to one owned lease, and expose only
   the Task's approved sandbox operations. Run the **original Task scorer**
   once after the agent is quiescent and before the target/workspace goes
   away. Retain its original result and source/model/tool/harness events before
   sample cleanup. Make the future Inspect `sample_cleanup` bridge the **only**
   owner closing the Compose lease, after that original score. Let Inspect
   finalize the `.eval` log after its sandbox lifecycle; verify the `EvalLog`
   status and sample identity, then retain the exact `.eval` bytes in
   controller-owned PyRIT memory/results storage **outside the disposable VM**,
   with per-run digest, length and authorized readback. Normal VM deletion
   waits for that confirmation; the watchdog handles retention failure as
   described below. Never expose raw logs or scoring assets to the guest.
   `NativeCyberEvaluation._finalize_async` already grades before closing its
   runtime, and `NativeCyberEvidenceStore` retains bounded raw chunks and
   atomically links a canonical report and Score. Neither this Compose lease
   nor the native CTF prototype is an Inspect Task bridge; a
   `NativeCyberArtifact.evidence_ref` is **not** an `.eval` upload API. Missing
   bytes, incomplete coverage or a failed original scorer means undetermined,
   never a reconstructed score from final assistant text.
5. **Release on every outcome.** On success, error, timeout or cancellation,
   revoke the guest route, retain any partial evidence, and skip grading an
   incomplete/cancelled attempt. The sole lifecycle owner closes the exact
   Compose lease and verifies no residual containers, networks or volumes.
   After protected evidence transfer, delete only the exact owned VM/NIC/disks
   and poll Azure until their absence is confirmed; a separate TTL watchdog
   reconciles crashed controllers.
   Quarantine unknown/foreign Docker resources and report cleanup failure
   instead of global `inspect sandbox cleanup`, `docker prune` or falsely
   declaring success. Failed evidence upload must not silently erase the only
   copy; restrict and reconcile the VM until retention or an explicit failure
   is recorded. At the hard watchdog deadline, record irrecoverable retention
   failure and delete the VM anyway, never claiming a complete result. Neither
   Docker `down` nor a VM DELETE acceptance proves all resources were released.

For the first case, propose **one VM and one concurrent sample**, an overall
VM lifetime of at most 30 minutes including provisioning and cleanup, a
180-second agent/Task deadline, and a separate watchdog deadline of 35 minutes.
On the proposed 4-vCPU/16-GiB VM, admit only a graph with at most 3 total
vCPUs and 10 GiB container memory, 512 total PIDs, 1 GiB aggregate tmpfs
and a 64-GiB maximum OS disk, leaving headroom for Docker, the grader and
the host. Cap the per-run model route at 8 requests and 16,384 output tokens
with a proposed $5 maximum projected total cost per sample (VM, disk, data and
model); require owner approval of that amount and deny admission if region/model
prices or usage bounds are unknown. These are **proposed ceilings**, not
code-enforced Azure or billing caps. A budget alert
is only a notification, not a stop mechanism. The controller and independent
watchdog must enforce lifecycle and prevent new admissions when cost or quota
evidence is missing.

### Acceptance before any live cyber or model run

- **Offline, authorized-independent path:** existing fake
  `test_docker_compose.py` and `test_environment_lease.py` cover strict
  manifests/rollback; a future bridge should add deterministic fake ARM,
  gateway, Inspect sample and result-store tests. Exercise missing approvals,
  foreign resource, build/pull attempt, network/IMDS reachability claim,
  grader exposure, duplicate sample identity, over-budget graph, truncated
  `.eval`, scorer failure, upload failure and cancellation during acquisition
  and cleanup. None of those tests alone qualifies Azure containment.
- **After separate owner authorization:** one synthetic, non-cyber
  file-existence/equality Task with a preloaded harmless image and *no model
  or private dataset*. Observe the actual approved VM image/identity, guest
  egress denials, per-sample network/workspace isolation, verified host-only
  grader and score-before-teardown order. Require identical input/source
  hashes and sample/attempt/run IDs in the Inspect log and PyRIT report,
  byte-for-byte `.eval` retrieval by an authorized reader, bounded event
  coverage and confirmed removal of the Compose project *and* Azure resources.
  Repeat with forced failure/cancellation only after that experiment is
  authorized. No such live validation has been performed here.

**Why this remains a design record:** this branch has no approved subscription,
gallery image-version ID, principals/RBAC, VNet/firewall/gateway measurements,
private Task/dataset/grader authorization, Azure VM controller/attestation and
watchdog API, Inspect `SandboxEnvironment` sample-to-lease adapter or `.eval`
artifact writer/readback API. A Python preflight that accepts caller-provided
strings as proof of those properties would be success-shaped scaffolding. Keep
`readiness_async` blocked until those concrete authorities, ownership and
observable checks exist. The existing source boundaries are
[`DockerComposeEnvironmentLease`](docker_compose.py),
[`EnvironmentLease`](environment_lease.py),
[`NativeCyberEvaluation`](native_cyber_eval.py),
the [mock-tested gateway and stop-only provider](docker_agent.md) and the
[non-Inspect CTF prototype](../benchmark/ctf/README.md).
The [Inspect sandbox lifecycle/Compose behavior](https://inspect.aisi.org.uk/sandboxing.html),
[custom sample cleanup API](https://inspect.aisi.org.uk/extensions-sandboxes.html),
[`.eval` log API](https://inspect.aisi.org.uk/eval-logs.html),
[Azure IMDS boundary](https://learn.microsoft.com/en-us/azure/virtual-machines/instance-metadata-service),
[NSG defaults](https://learn.microsoft.com/en-us/azure/virtual-network/network-security-groups-overview),
[Docker firewall interactions](https://docs.docker.com/engine/network/packet-filtering-firewalls/),
[ACR private endpoints](https://learn.microsoft.com/en-us/azure/container-registry/container-registry-private-endpoints),
[exact VM deletion](https://learn.microsoft.com/en-us/rest/api/compute/virtual-machines/delete),
[Azure deletion confirmation](https://learn.microsoft.com/en-us/azure/azure-resource-manager/management/delete-resource-group)
and [budget alert semantics](https://learn.microsoft.com/en-us/azure/cost-management-billing/costs/cost-mgt-alerts-monitor-usage-spending)
are the external source contracts for these gates.
