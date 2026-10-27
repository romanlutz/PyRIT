# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Concrete run-owned Docker/Claude connection; no socket, credential or backend discovery."""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from ipaddress import IPv4Address, IPv4Network
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Protocol

from pyrit.executor.workflow.docker_agent import (
    DockerAgentCliProfile,
    DockerSandboxLauncher,
    DockerSandboxProcessSession,
    DockerStopOnlyAgentLease,
    _settle_task_async,
)
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError
from pyrit.executor.workflow.docker_guest_auth import DockerGuestAuth
from pyrit.executor.workflow.docker_named_workspace import (
    DockerNamedWorkspaceLease,
    DockerNamedWorkspaceSpec,
    DockerWorkspaceIdentity,
)
from pyrit.executor.workflow.environment_lease import EnvironmentLease
from pyrit.models.environment_lease import EnvironmentLeaseState, EnvironmentResourceHandle, EnvironmentServiceHandle
from pyrit.models.native_cyber import NativeAgentCapabilities, NativeCyberReadiness
from pyrit.prompt_target.gateway.claude_messages import create_claude_messages_app
from pyrit.prompt_target.gateway.messages_contract import MessagesCapabilities
from pyrit.prompt_target.gateway.run_listener import GatewayBridgeBinding
from pyrit.prompt_target.native_cli_models import NativeCliProtocol, NativeCliStream

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable
    from pathlib import Path

    from starlette.types import ASGIApp

    from pyrit.executor.workflow.docker_command import DockerCommandRunner
    from pyrit.executor.workflow.docker_compose import ComposeAllocation, ComposeEnvironmentSpec
    from pyrit.executor.workflow.docker_engine import DockerExecHandle
    from pyrit.executor.workflow.native_cli_evidence import NativeCliDatabaseEvidenceSink
    from pyrit.executor.workflow.native_harness import NativeTaskRuntime
    from pyrit.models.native_cyber import NativeCyberJudgment
    from pyrit.models.native_harness import NativeHarnessEvidence
    from pyrit.prompt_target.gateway.messages_contract import ModelOnlyMessagesBackend
    from pyrit.prompt_target.gateway.responses_contract import GatewayLimits, GatewayRoute
    from pyrit.prompt_target.native_cli_models import NativeCliProcessChunk, NativeCliRunConfig


@dataclass(frozen=True, kw_only=True)
class DockerCliQualification:
    """Caller-owned qualification pins; references are assertions, not a live readiness probe."""

    engine: DockerEngineClient = field(repr=False)
    engine_identity: str
    profile: DockerAgentCliProfile
    backend_identity: str
    backend: ModelOnlyMessagesBackend = field(repr=False)
    qualification_ref: str
    simulated: bool
    ready: bool = False
    workspace: DockerNamedWorkspaceSpec | None = None

    def __post_init__(self) -> None:
        """
        Require an actual explicitly injected Engine client and bounded qualification identities.

        Raises:
            ValueError: If qualification metadata or client identity is invalid.
        """
        if (
            not isinstance(self.engine, DockerEngineClient)
            or type(self.ready) is not bool
            or type(self.simulated) is not bool
            or any(
                not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}", value)
                for value in (self.engine_identity, self.backend_identity, self.qualification_ref)
            )
        ):
            raise ValueError("Docker CLI qualification needs explicit pinned host-owned identities.")

    def readiness(self) -> NativeCyberReadiness:
        """
        Report caller qualification without allocating, probing or changing a daemon.

        Returns:
            NativeCyberReadiness: Default-blocked qualification for this exact profile.
        """
        codex = self.profile.config.protocol is NativeCliProtocol.CODEX_EXEC_JSON
        blockers = (
            ("Docker archive views cannot qualify guest-visible Codex configuration in tmpfs.",)
            if codex
            else ("The named workspace profile is not separately qualified.",)
            if self.workspace is not None and not self.workspace.ready
            else ()
            if self.ready
            else ("Docker CLI/Engine/bridge profile is not qualified.",)
        )
        return NativeCyberReadiness(
            ready=not blockers,
            blockers=blockers,
            simulated=self.simulated,
            capabilities=NativeAgentCapabilities(evidence_scope="native_cli_jsonl"),
            provenance={
                "qualification_ref": self.qualification_ref,
                "engine_identity": self.engine_identity,
                "backend_identity": self.backend_identity,
            },
        )


class DockerGatewayListener(Protocol):
    """The existing RunScopedModelGatewayListener surface, injected without opening a socket here."""

    binding: GatewayBridgeBinding

    @property
    def base_url(self) -> str:
        """The exact bound host bridge URL."""
        ...

    @property
    def is_locally_running(self) -> bool:
        """Whether the owned listener has acknowledged startup and remains available."""
        ...

    async def start_async(self) -> None:
        """Start the one already-owned listener."""
        ...

    async def close_async(self) -> None:
        """Settle and release only this run's listener."""
        ...


class DockerGatewayListenerFactory(Protocol):
    """Transfer an explicit run-owned listener, normally RunScopedModelGatewayListener."""

    def __call__(self, *, route: GatewayRoute, binding: GatewayBridgeBinding, app: ASGIApp) -> DockerGatewayListener:
        """Use a caller-prebound socket; creation failure and foreign resources remain factory-owned."""
        ...


@dataclass(frozen=True, kw_only=True)
class DockerCliGateway:
    """One already-admitted Messages backend, guest route and explicit private bridge plan."""

    route: GatewayRoute = field(repr=False)
    limits: GatewayLimits
    backend: ModelOnlyMessagesBackend = field(repr=False)
    backend_identity: str
    address: IPv4Address
    subnet: IPv4Network
    port: int
    listener_factory: DockerGatewayListenerFactory = field(repr=False)

    def __post_init__(self) -> None:
        """
        Reject unbounded/public listener plans and a mismatched model wire backend.

        Raises:
            ValueError: If the private bridge or actual Messages backend is not explicit.
        """
        ranges = (IPv4Network("10.0.0.0/8"), IPv4Network("172.16.0.0/12"), IPv4Network("192.168.0.0/16"))
        if (
            not isinstance(self.address, IPv4Address)
            or not isinstance(self.subnet, IPv4Network)
            or not any(self.subnet.subnet_of(network) for network in ranges)
            or self.subnet.prefixlen > 30
            or self.address not in self.subnet
            or self.address in {self.subnet.network_address, self.subnet.broadcast_address}
            or type(self.port) is not int
            or not 1 <= self.port <= 65535
            or not isinstance(self.backend.capabilities, MessagesCapabilities)
            or not self.backend_identity
        ):
            raise ValueError("The gateway needs an explicit private bridge and admitted Anthropic Messages backend.")

    @property
    def base_url(self) -> str:
        """The explicitly approved HTTP bridge endpoint, never a connectivity claim."""
        return f"http://{self.address}:{self.port}"


class DockerTargetPhase(str, Enum):
    """Allowed installed target operation phases, separate from the evaluated agent."""

    SETUP = "setup"
    AFTER_AGENT_STOP = "after_agent_stop"


@dataclass(frozen=True, kw_only=True)
class DockerTargetOperation:
    """A fixed installed operation; no candidate-supplied argv, paths, shell or file payload."""

    name: str
    argv: tuple[str, ...] = field(repr=False)
    working_directory: PurePosixPath
    phase: DockerTargetPhase
    timeout_seconds: float = 10
    max_output_bytes: int = 65_536

    def __post_init__(self) -> None:
        """
        Require bounded literal operation pins.

        Raises:
            ValueError: If argv, operation phase, path or limits are unsafe.
        """
        if (
            not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.name)
            or not isinstance(self.phase, DockerTargetPhase)
            or not self.argv
            or not re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", self.argv[0])
            or any(type(arg) is not str or "\x00" in arg for arg in self.argv)
            or sum(len(arg.encode()) for arg in self.argv) > 8192
            or not isinstance(self.working_directory, PurePosixPath)
            or not self.working_directory.is_absolute()
            or ".." in self.working_directory.parts
            or type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 600
            or type(self.max_output_bytes) is not int
            or not 1 <= self.max_output_bytes <= 1_048_576
        ):
            raise ValueError("Target operations need fixed installed argv and bounded guest paths/output/deadlines.")


@dataclass(frozen=True, kw_only=True)
class DockerTargetResult:
    """Exact target exec evidence, not an asserted filesystem fact or grader verdict."""

    run_id: str
    environment_id: str
    operation: str
    container_id: str
    exec_id: str
    exit_code: int
    stdout: bytes = field(repr=False)
    stderr: bytes = field(repr=False)


@dataclass(frozen=True, kw_only=True)
class DockerTargetDeadline:
    """One caller-established monotonic operation window, not a fresh timeout for each phase."""

    started_at: float
    expires_at: float

    def __post_init__(self) -> None:
        """
        Require an explicit finite increasing time window.

        Raises:
            ValueError: If either timestamp is not finite or the interval is invalid.
        """
        if (
            any(
                type(value) not in (int, float) or not math.isfinite(value)
                for value in (self.started_at, self.expires_at)
            )
            or self.expires_at <= self.started_at
        ):
            raise ValueError("A target deadline requires finite increasing monotonic timestamps.")

    @classmethod
    def start(cls, *, timeout_seconds: float) -> DockerTargetDeadline:
        """
        Establish the approved operation deadline once, before dispatch/capture.

        Returns:
            DockerTargetDeadline: The single absolute start/expiry pair.

        Raises:
            ValueError: If the explicitly selected budget is outside the bounded operation profile.
        """
        if (
            type(timeout_seconds) not in (int, float)
            or not math.isfinite(timeout_seconds)
            or not 0 < timeout_seconds <= 600
        ):
            raise ValueError("Installed target operations require a positive budget of at most 600 seconds.")
        started = asyncio.get_running_loop().time()
        return cls(started_at=started, expires_at=started + timeout_seconds)

    def validate(self, operation: DockerTargetOperation) -> None:
        """
        Reject shortened, extended, future or expired caller windows before target I/O.

        Raises:
            ValueError: If the caller changed the original operation budget or supplied a future start.
            TimeoutError: If the established operation window already expired.
        """
        now = asyncio.get_running_loop().time()
        if self.started_at > now or not math.isclose(
            self.expires_at - self.started_at, operation.timeout_seconds, rel_tol=0, abs_tol=1e-6
        ):
            raise ValueError("Target deadline must preserve the exact installed operation budget.")
        if self.expires_at <= now:
            raise TimeoutError("The caller's target-operation deadline has expired.")


class DockerOriginalGrader(Protocol):
    """The task's existing original grader, supplied by its binding."""

    async def __call__(self, *, runtime: OwnedDockerCliRuntime, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        """Acquire the original judgment from actual owned target state."""
        ...


@dataclass(frozen=True, kw_only=True)
class DockerTargetInspection:
    """Verified target process identity, excluding daemon host paths and container environment."""

    run_id: str
    environment_id: str
    container_id: str
    image_id: str
    running: bool
    pid: int


class _GatewayLease(DockerStopOnlyAgentLease):
    def __init__(
        self,
        *,
        gateway: DockerCliGateway,
        run_id: str,
        spec: ComposeEnvironmentSpec,
        runner: DockerCommandRunner,
        engine: DockerEngineClient,
        profile: DockerAgentCliProfile,
        guest_auth: DockerGuestAuth,
        stop_timeout_seconds: float,
    ) -> None:
        self._gateway_plan = gateway
        super().__init__(
            run_id=run_id,
            spec=spec,
            runner=runner,
            engine=engine,
            profile=profile,
            guest_auth=guest_auth,
            stop_timeout_seconds=stop_timeout_seconds,
        )

    def _document(self) -> dict[str, Any]:
        document = super()._document()
        document["networks"]["challenge"]["ipam"] = {
            "driver": "default",
            "config": [{"subnet": str(self._gateway_plan.subnet), "gateway": str(self._gateway_plan.address)}],
        }
        return document


class OwnedDockerCliRuntime:
    """Concrete two-service task runtime; owns only its lease/listener, not host backend credentials."""

    def __init__(
        self,
        *,
        run_id: str,
        spec: ComposeEnvironmentSpec,
        runner: DockerCommandRunner,
        qualification: DockerCliQualification,
        gateway: DockerCliGateway,
        grader: DockerOriginalGrader,
        target_operations: tuple[DockerTargetOperation, ...] = (),
        stop_timeout_seconds: float = 30,
        require_agent_workspace: bool = False,
    ) -> None:
        """
        Pin the actual provider, image/profile, backend and guest-only route before allocation.

        Raises:
            NotImplementedError: If an agent-workspace handoff or archive-based Codex profile is requested.
            ValueError: If run, topology, qualification or gateway identity disagrees.
        """
        if require_agent_workspace and qualification.workspace is None:
            raise NotImplementedError(
                "Engine archive views do not expose live container tmpfs; "
                "no verified agent-workspace handoff is available."
            )
        profile = qualification.profile
        if profile.config.protocol is not NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE:
            raise NotImplementedError("Codex archive-config staging does not prove guest-visible tmpfs configuration.")
        if (
            not qualification.readiness().ready
            or run_id != gateway.route.run_id
            or qualification.backend_identity != gateway.backend_identity
            or qualification.backend is not gateway.backend
            or profile.config.model_gateway_endpoint != gateway.base_url
            or len(spec.services) != 2
            or len({operation.name for operation in target_operations}) != len(target_operations)
        ):
            raise ValueError("The exact qualified run/profile/backend/gateway and two-service topology are required.")
        agents = [service for service in spec.services if service.roles == frozenset({"agent"})]
        targets = [service for service in spec.services if service.roles and service.roles <= {"target", "grader"}]
        if len(agents) != 1 or len(targets) != 1:
            raise ValueError("The connector requires distinct agent-only and target/grader-only services.")
        for operation in target_operations:
            if any(PurePosixPath(operation.argv[0]).is_relative_to(mount.path) for mount in targets[0].tmpfs_mounts):
                raise ValueError("Installed target operations must remain outside candidate-writable tmpfs.")
            if qualification.workspace is not None and PurePosixPath(operation.argv[0]).is_relative_to(
                qualification.workspace.root
            ):
                raise ValueError("Installed target operations cannot execute from the shared untrusted workspace.")
            if gateway.route.guest_token in repr(operation.argv):
                raise ValueError("Guest credentials cannot be target operation arguments.")
        self.run_id, self.simulated = run_id, qualification.simulated
        self.qualification, self.gateway = qualification, gateway
        self._engine = qualification.engine
        self._agent, self._target = agents[0], targets[0]
        self._grader = grader
        self._operations = {operation.name: operation for operation in target_operations}
        auth = DockerGuestAuth.from_route(
            route=gateway.route, protocol=profile.config.protocol, gateway_endpoint=gateway.base_url
        )
        self.lease = (
            _GatewayLease(
                gateway=gateway,
                run_id=run_id,
                spec=spec,
                runner=runner,
                engine=self._engine,
                profile=profile,
                guest_auth=auth,
                stop_timeout_seconds=stop_timeout_seconds,
            )
            if qualification.workspace is None
            else DockerNamedWorkspaceLease(
                workspace=qualification.workspace,
                run_id=run_id,
                spec=spec,
                runner=runner,
                engine=self._engine,
                profile=profile,
                guest_auth=auth,
                stop_timeout_seconds=stop_timeout_seconds,
                bridge_address=gateway.address,
                bridge_subnet=gateway.subnet,
            )
        )
        self.environment_id = self.lease.lease_id
        self._allocation: ComposeAllocation | None = None
        self._baselines: dict[str, str] = {}
        self._network_baseline: str | None = None
        self._sink: NativeCliDatabaseEvidenceSink | None = None
        self._listener: DockerGatewayListener | None = None
        self._listener_closed = False
        self._session: DockerSandboxProcessSession | None = None
        self._lock = asyncio.Lock()
        self._launch_attempted = False
        self._graded = False
        self._failed = False
        self._close_task: asyncio.Task[None] | None = None

    @property
    def allocation(self) -> ComposeAllocation:
        """
        The exact acquired two-service IDs; unavailable before acquisition.

        Raises:
            RuntimeError: If the provider has not acquired its allocation.
        """
        if self._allocation is None:
            raise RuntimeError("The owned Docker allocation has not been acquired.")
        return self._allocation

    @property
    def agent_exec_id(self) -> str | None:
        """The daemon-issued agent exec ID, not a host-process or JSONL ID."""
        return self._session.exec_id if self._session else None

    @property
    def agent_workspace_available_after_stop(self) -> bool:
        """Whether the opted-in shared workspace remains available after confirmed agent stop."""
        return bool(
            isinstance(self.lease, DockerNamedWorkspaceLease)
            and self.lease.agent_stop is not None
            and self.lease.agent_stop.stopped
            and self._close_task is None
            and not self._failed
        )

    async def workspace_identity_async(self) -> DockerWorkspaceIdentity:
        """
        Verify the live target shares the stopped agent's exact admitted volume generation.

        Returns:
            DockerWorkspaceIdentity: Actual owned mount/lifetime identity, not a filesystem verdict.

        Raises:
            NotImplementedError: If this profile has no shared workspace capability.
        """
        if not isinstance(self.lease, DockerNamedWorkspaceLease):
            raise NotImplementedError("This profile does not expose the stopped agent's workspace.")
        async with self._lock:
            await self._verify_stopped_async()
            return await self.lease.verify_workspace_async()

    async def acquire_async(self) -> OwnedDockerCliRuntime:
        """
        Acquire and health-check the actual provider, binding Engine/Compose/network identities.

        Returns:
            OwnedDockerCliRuntime: This runtime, with exact owned service identities.

        Raises:
            RuntimeError: If the runtime is already acquired or closed.
            asyncio.CancelledError: After owned acquisition rollback.
        """
        if self._allocation is not None or self._close_task is not None:
            raise RuntimeError("The owned Docker runtime can be acquired only once.")
        try:
            async with self._lock:
                self._allocation = await self.lease.acquire_async()
                for name, identifier in self.allocation.containers:
                    value = await self._engine.inspect_container_async(identifier)
                    self._baselines[name] = DockerStopOnlyAgentLease._static_identity(value)
                self._network_baseline = DockerStopOnlyAgentLease._network_identity(
                    await self._engine.inspect_network_async(self.allocation.network_id)
                )
                await self._audit_async(stopped=False)
            return self
        except (Exception, asyncio.CancelledError) as error:
            self._failed = True
            try:
                await self.close_async()
            except (Exception, asyncio.CancelledError) as cleanup_error:
                raise error from cleanup_error
            raise

    async def validate_agent_storage_async(self, *, directory: Path) -> None:
        """
        Recheck the provider's no-host-mount/security boundary; host ACL checks remain task-binding-owned.

        Raises:
            ValueError: If the controller did not supply an absolute host evidence directory.
        """
        if not directory.is_absolute():
            raise ValueError("Host evidence storage must name an absolute controller directory.")
        await self._audit_async(stopped=False)

    def bind_sink(self, sink: NativeCliDatabaseEvidenceSink) -> _OwnedLauncher:
        """
        Attach the actual native CLI sink once; gateway startup waits until sources are opened.

        Returns:
            _OwnedLauncher: The exact DockerSandboxLauncher connection, not a caller-supplied runner.

        Raises:
            ValueError: If source identity/protocol differs from this runtime.
        """
        if (
            self._sink is not None
            or sink.run_id != self.run_id
            or sink.environment_id != self.environment_id
            or sink._protocol is not self.qualification.profile.config.protocol
        ):
            raise ValueError("The CLI sink must bind this exact run/environment/protocol once.")
        self._sink = sink
        return _OwnedLauncher(self)

    async def grade_async(self, *, evidence: NativeHarnessEvidence) -> NativeCyberJudgment:
        """
        Delegate once to the task-owned grader only after actual agent stop and gateway settlement.

        Returns:
            NativeCyberJudgment: The unchanged original task judgment.

        Raises:
            RuntimeError: If provider/source identity or stop/settlement is not complete.
        """
        if (
            self._graded
            or self._failed
            or not evidence.quiescent
            or not evidence.coverage_complete
            or evidence.identity.run_id != self.run_id
            or evidence.identity.environment_id != self.environment_id
            or evidence.identity.harness_id != self.qualification.profile.config.protocol.value
            or evidence.simulated != self.simulated
        ):
            raise RuntimeError("Original grading requires this ungraded, complete, quiescent owned runtime.")
        await self._verify_stopped_async()
        self._graded = True
        return await self._grader(runtime=self, evidence=evidence)

    async def inspect_target_async(self) -> DockerTargetInspection:
        """
        Inspect the actual target ID and security binding without interpreting task or filesystem state.

        Returns:
            DockerTargetInspection: Exact verified target state without host-private paths/env.
        """
        async with self._lock:
            await self._audit_async(stopped=self.lease.agent_stop is not None)
            observed = await self._engine.inspect_container_async(self.allocation.container_id(self._target.name))
            return DockerTargetInspection(
                run_id=self.run_id,
                environment_id=self.environment_id,
                container_id=observed["Id"],
                image_id=observed["Image"],
                running=observed["State"]["Running"],
                pid=observed["State"]["Pid"],
            )

    async def run_target_operation_async(self, name: str) -> DockerTargetResult:
        """
        Establish the installed operation's exact budget once and dispatch without retry.

        Returns:
            DockerTargetResult: Attributed target bytes and daemon exit.

        Raises:
            ValueError: If the operation is not in the installed allow-list.
        """
        if name not in self._operations:
            raise ValueError("The target operation is not in this run's installed allow-list.")
        return await self.run_target_operation_until_async(
            name=name, deadline=DockerTargetDeadline.start(timeout_seconds=self._operations[name].timeout_seconds)
        )

    async def run_target_operation_until_async(
        self, *, name: str, deadline: DockerTargetDeadline
    ) -> DockerTargetResult:
        """
        Run only a fixed installed target operation, never an agent CLI or arbitrary candidate command.

        File/setup/fact semantics belong to the pinned installed helper and original task.
        An interrupted exec closes the whole owned lease because Engine has no exec-kill primitive here.

        Returns:
            DockerTargetResult: Attributed original output and daemon exit, not an inferred filesystem fact.

        Raises:
            ValueError: If the operation or caller deadline differs from the approved budget.
            TimeoutError: If the caller deadline expired before I/O or during the operation.
            RuntimeError: If phase, target, output, or guest exit cannot be verified.
            DockerEngineError: If target output or daemon attribution is incomplete.
            asyncio.CancelledError: After owned cleanup is attempted.
        """
        if name not in self._operations:
            raise ValueError("The target operation is not in this run's installed allow-list.")
        operation = self._operations[name]
        budget = deadline
        budget.validate(operation)
        stream = None
        try:
            async with asyncio.timeout_at(budget.expires_at), self._lock:
                if operation.phase is DockerTargetPhase.SETUP:
                    if self._launch_attempted:
                        raise RuntimeError("Target setup cannot run after agent dispatch.")
                    await self._audit_async(stopped=False)
                else:
                    await self._verify_stopped_async()
                handle = await self._engine.create_exec_async(
                    container_id=self.allocation.container_id(self._target.name),
                    argv=operation.argv,
                    user=f"{self._target.uid}:{self._target.gid}",
                    working_directory=str(operation.working_directory),
                )
                stream = await self._engine.start_exec_async(handle)
                output = {NativeCliStream.STDOUT: bytearray(), NativeCliStream.STDERR: bytearray()}
                async for chunk in stream.read_chunks_async():
                    output[chunk.stream].extend(chunk.data)
                    if sum(map(len, output.values())) > operation.max_output_bytes:
                        raise DockerEngineError("Target operation output exceeded its approved byte limit.")
                if not stream.eof_observed:
                    raise DockerEngineError("Target operation lacks complete daemon-framed output.")
                exit_code = await self._target_exit_async(handle)
                await self._audit_async(stopped=operation.phase is DockerTargetPhase.AFTER_AGENT_STOP)
                return DockerTargetResult(
                    run_id=self.run_id,
                    environment_id=self.environment_id,
                    operation=name,
                    container_id=handle.container_id,
                    exec_id=handle.exec_id,
                    exit_code=exit_code,
                    stdout=bytes(output[NativeCliStream.STDOUT]),
                    stderr=bytes(output[NativeCliStream.STDERR]),
                )
        except (Exception, asyncio.CancelledError) as error:
            self._failed = True
            try:
                await self.close_async()
            except (Exception, asyncio.CancelledError) as cleanup_error:
                raise error from cleanup_error
            raise
        finally:
            if stream is not None:
                await stream.close_async()

    async def close_async(self) -> None:
        """Settle cleanup once for this listener and exact provider lease, never the caller's Engine/backend."""
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_owned_async())
        await _settle_task_async(self._close_task)

    async def _close_owned_async(self) -> None:
        async with self._lock:
            postgrade_error: Exception | None = None
            if self._graded and not self._failed:
                try:
                    await self.lease._audit_engine_async(require_agent_running=False)
                except Exception as error:
                    self._failed = True
                    postgrade_error = error
            try:
                await self._close_listener_async()
            finally:
                await self.lease.close_async()
            if postgrade_error is not None:
                raise postgrade_error

    async def _close_listener_async(self) -> None:
        if self._listener is not None and not self._listener_closed:
            try:
                await self._listener.close_async()
                if self._listener.is_locally_running:
                    raise RuntimeError("The owned model listener is still running after close.")
                self._listener_closed = True
            except (Exception, asyncio.CancelledError):
                self._failed = True
                raise

    async def _target_exit_async(self, handle: DockerExecHandle) -> int:
        while True:
            observed = await self._engine.inspect_exec_async(handle)
            if observed["Running"] is False:
                value = observed.get("ExitCode")
                if type(value) is not int or not 0 <= value <= 255:
                    raise DockerEngineError("Target operation has no actual daemon ExitCode.")
                return value
            await asyncio.sleep(0.01)

    async def _audit_async(self, *, stopped: bool) -> None:
        if (
            self._failed
            or self._close_task is not None
            or self.lease.run_id != self.run_id
            or self.lease.lease_id != self.environment_id
            or self.lease.snapshot().state is not EnvironmentLeaseState.READY
        ):
            raise RuntimeError("Owned Docker runtime is not ready or its identity changed.")
        for name, identifier in self.allocation.containers:
            value = await self._engine.inspect_container_async(identifier)
            if DockerStopOnlyAgentLease._static_identity(value) != self._baselines.get(name):
                raise DockerEngineError("Owned service identity, image, security or mounts changed.")
            state = value["State"]
            state_check = (
                DockerStopOnlyAgentLease._is_stopped
                if stopped and name == self._agent.name
                else DockerStopOnlyAgentLease._is_running
            )
            if not state_check(state):
                raise DockerEngineError("Owned agent/target liveness differs from the required operation phase.")
            networks = value.get("NetworkSettings", {}).get("Networks")
            if not (stopped and name == self._agent.name and networks == {}) and (
                not isinstance(networks, dict)
                or len(networks) != 1
                or next(iter(networks.values())).get("NetworkID") != self.allocation.network_id
            ):
                raise DockerEngineError("Owned container network membership changed.")
        network = await self._engine.inspect_network_async(self.allocation.network_id)
        if (
            network.get("Internal") is not True
            or network.get("Driver") != "bridge"
            or network.get("IPAM", {}).get("Driver") != "default"
            or network.get("IPAM", {}).get("Config")
            != [{"Subnet": str(self.gateway.subnet), "Gateway": str(self.gateway.address)}]
            or network.get("Labels", {}).get("org.pyrit.native.run") != self.run_id
            or network.get("Labels", {}).get("org.pyrit.native.lease") != self.environment_id
            or DockerStopOnlyAgentLease._network_identity(network) != self._network_baseline
            or not isinstance(network.get("Containers"), dict)
            or set(network["Containers"]) - {value for _, value in self.allocation.containers}
            or self.allocation.container_id(self._target.name) not in network["Containers"]
        ):
            raise DockerEngineError("Owned bridge identity/IPAM does not match the admitted listener plan.")
        if isinstance(self.lease, DockerNamedWorkspaceLease):
            await self.lease.verify_workspace_async()

    async def _verify_stopped_async(self) -> None:
        receipt = self.lease.agent_stop
        expected = tuple((name, value) for name, value in self.allocation.containers if name != self._agent.name)
        if (
            self._session is None
            or receipt is None
            or not receipt.stopped
            or receipt.error is not None
            or receipt.container_id != self._session.container_id
            or receipt.exec_id != self._session.exec_id
            or receipt.preserved_services != expected
            or not self._listener_closed
        ):
            raise RuntimeError("No exact owned agent exec/stop/target-preservation and gateway-close receipt.")
        await self._audit_async(stopped=True)

    async def _launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> _OwnedProcess:
        async with self._lock:
            if self._launch_attempted or config != self.qualification.profile.config or self._sink is None:
                raise ValueError("Launch requires the exact unused profile and bound source sink.")
            self._sink.ensure_active()
            await self._audit_async(stopped=False)
            self._launch_attempted = True
            binding = GatewayBridgeBinding(
                run_id=self.run_id,
                network_id=self.allocation.network_id,
                address=self.gateway.address,
                subnet=self.gateway.subnet,
                port=self.gateway.port,
            )
            app = create_claude_messages_app(
                route=self.gateway.route,
                limits=self.gateway.limits,
                backend=self.gateway.backend,
                observation_callback=self._sink.record_messages_observation_async,
            )
            listener = self.gateway.listener_factory(route=self.gateway.route, binding=binding, app=app)
            if listener.binding != binding:
                raise ValueError("A foreign listener cannot be adopted or closed by this run.")
            self._listener = listener
            if listener.base_url != self.gateway.base_url:
                raise ValueError("The listener URL differs from the pinned guest route.")
            await listener.start_async()
            if not listener.is_locally_running:
                raise RuntimeError("The owned model listener did not confirm local startup.")
            self._session = await DockerSandboxLauncher(lease=self.lease).launch_async(config=config, prompt=prompt)
            if self._session.container_id != self.allocation.container_id(self._agent.name):
                raise DockerEngineError("Agent exec belongs to a different container.")
            return _OwnedProcess(runtime=self, process=self._session)


class OwnedDockerCliConnector:
    """Connect only the actual owned Docker runtime to NativeCliHarnessAdapter."""

    def __call__(self, *, runtime: NativeTaskRuntime, sink: NativeCliDatabaseEvidenceSink) -> _OwnedLauncher:
        """
        Bind the concrete provider runtime, not an arbitrary supplied launcher.

        Returns:
            _OwnedLauncher: One actual Docker launcher connection.

        Raises:
            TypeError: If the environment is not this concrete owned runtime.
        """
        if not isinstance(runtime, OwnedDockerCliRuntime):
            raise TypeError("The Docker connector requires the concrete owned Docker CLI runtime.")
        return runtime.bind_sink(sink)


class OwnedDockerRuntimeLease(EnvironmentLease[OwnedDockerCliRuntime]):
    """Expose concrete runtime acquisition/rollback to the existing native controller."""

    def __init__(
        self,
        *,
        runtime: OwnedDockerCliRuntime,
        prepare_async: Callable[[OwnedDockerCliRuntime], Awaitable[None]] | None = None,
    ) -> None:
        """Reserve the exact runtime before its provider allocates anything."""
        super().__init__(run_id=runtime.run_id, cleanup_timeout_seconds=180)
        self._runtime = runtime
        self._prepare = prepare_async

    async def _acquire_async(self) -> OwnedDockerCliRuntime:
        resource = EnvironmentResourceHandle(
            run_id=self.run_id, provider="owned_docker_cli", resource_id=self._runtime.environment_id, kind="runtime"
        )
        runtime = await self._acquire_resource_async(
            handle=resource, acquire_async=self._enter_async, release_async=self._release_async
        )
        for service in (runtime._agent, runtime._target):
            self._register_service(EnvironmentServiceHandle(name=service.name, roles=service.roles, resource=resource))
        return runtime

    async def _enter_async(self, handle: EnvironmentResourceHandle) -> OwnedDockerCliRuntime:
        try:
            await self._runtime.acquire_async()
            if self._prepare is not None:
                await self._prepare(self._runtime)
            return self._runtime
        finally:
            self._cleanup_timeout_seconds = self._runtime.lease.cleanup_budget_seconds + 15

    async def _release_async(self, handle: EnvironmentResourceHandle) -> None:
        await self._runtime.close_async()


class _OwnedLauncher:
    def __init__(self, runtime: OwnedDockerCliRuntime) -> None:
        self.runtime = runtime

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> _OwnedProcess:
        return await self.runtime._launch_async(config=config, prompt=prompt)


class _OwnedProcess:
    def __init__(self, *, runtime: OwnedDockerCliRuntime, process: DockerSandboxProcessSession) -> None:
        self.runtime, self.process = runtime, process

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        async for part in self.process.read_chunks_async():
            yield part

    async def wait_async(self) -> int:
        return await self.process.wait_async()

    async def stop_async(self) -> None:
        async with self.runtime._lock:
            await self.process.stop_async()
            await self.runtime._close_listener_async()
            await self.runtime._verify_stopped_async()
