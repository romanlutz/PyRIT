# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-shot Engine launcher whose lease proves agent-container stop before target-side grading."""

from __future__ import annotations

import asyncio
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, TypeVar

from pyrit.executor.workflow.docker_compose import (
    ComposeAllocation,
    ComposeEnvironmentSpec,
    DockerComposeEnvironmentLease,
)
from pyrit.executor.workflow.docker_engine import DockerEngineClient, DockerEngineError, DockerExecHandle
from pyrit.models.environment_lease import EnvironmentLeaseState
from pyrit.prompt_target.native_cli_models import NativeCliProtocol, NativeCliRunConfig

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from pyrit.executor.workflow.docker_command import DockerCommandRunner
    from pyrit.executor.workflow.docker_engine import DockerExecStream
    from pyrit.models.environment_lease import EnvironmentHealth
    from pyrit.prompt_target.native_cli_models import NativeCliProcessChunk

ResultT = TypeVar("ResultT")


async def _settle_task_async(task: asyncio.Task[ResultT]) -> ResultT:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancellation = cancellation or error
        except Exception:
            break
    try:
        result = task.result()
    except BaseException as error:
        if cancellation is not None:
            raise cancellation from error
        raise
    if cancellation is not None:
        raise cancellation
    return result


class AgentArtifactPolicy(str, Enum):
    """Quiescence policies are explicit; frozen-workspace acquisition is not implemented."""

    TARGET_SIDE_ONLY = "target_side_only"
    FROZEN_AGENT_ARTIFACTS = "frozen_agent_artifacts"


@dataclass(frozen=True, kw_only=True)
class DockerAgentCliProfile:
    """Trusted image/config pin, not an authorization for candidate-supplied executable or flags."""

    image: str
    executable: PurePosixPath
    config: NativeCliRunConfig
    artifact_policy: AgentArtifactPolicy

    def __post_init__(self) -> None:
        """
        Validate the supported stop-only profile without inspecting or executing an image.

        Raises:
            ValueError: If the CLI image or executable is not pinned.
            NotImplementedError: If agent-workspace preservation is requested.
        """
        if self.artifact_policy is not AgentArtifactPolicy.TARGET_SIDE_ONLY:
            raise NotImplementedError("Only target-side grading without post-stop agent artifacts is supported.")
        if (
            not re.fullmatch(r"[^@\s]+@sha256:[0-9a-f]{64}", self.image)
            or not isinstance(self.executable, PurePosixPath)
            or not self.executable.is_absolute()
            or ".." in self.executable.parts
            or not re.fullmatch(r"/[A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*", str(self.executable))
        ):
            raise ValueError("The CLI requires a digest-pinned image and absolute immutable guest executable.")

    def argv(self, prompt: str) -> tuple[str, ...]:
        """
        Build the documented JSONL command with the prepared prompt as one final data argument.

        Returns:
            tuple[str, ...]: Direct guest argv, without shell or candidate-controlled flags.

        Raises:
            ValueError: If the prepared prompt is empty, contains NUL, or exceeds the input bound.
        """
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or "\x00" in prompt
            or len(prompt.encode("utf-8")) > 131_072
        ):
            raise ValueError("The prepared CLI prompt must be nonempty, NUL-free and at most 128 KiB.")
        arguments = (
            ("exec", "--json")
            if self.config.protocol is NativeCliProtocol.CODEX_EXEC_JSON
            else ("--print", "--output-format", "stream-json", "--verbose")
        )
        return (str(self.executable), *arguments, "--", prompt)


@dataclass(frozen=True, kw_only=True)
class AgentStopObservation:
    """Daemon observations, not a fabricated process receipt or a persistent evidence schema."""

    container_id: str
    stopped: bool
    exec_id: str | None = None
    preserved_services: tuple[tuple[str, str], ...] = ()
    error: str | None = None


class DockerStopOnlyAgentLease(DockerComposeEnvironmentLease):
    """Compose lease with one agent-only service and a serialized, latched stop barrier."""

    def __init__(
        self,
        *,
        run_id: str,
        spec: ComposeEnvironmentSpec,
        runner: DockerCommandRunner,
        engine: DockerEngineClient,
        profile: DockerAgentCliProfile,
        stop_timeout_seconds: float = 30,
    ) -> None:
        """
        Require a single-purpose agent container and target-side grading.

        Raises:
            ValueError: If roles, immutable executable, profile, or budgets are incompatible.
        """
        super().__init__(run_id=run_id, spec=spec, runner=runner)
        agents = [service for service in spec.services if "agent" in service.roles]
        if (
            len(agents) != 1
            or agents[0].roles != frozenset({"agent"})
            or not any(service.roles & {"target", "grader"} for service in spec.services if service is not agents[0])
        ):
            raise ValueError("Stop-only execution needs one agent-only service and a separate target/grader service.")
        agent = agents[0]
        if agent.image != profile.image or any(
            profile.executable.is_relative_to(mount.path) for mount in agent.tmpfs_mounts
        ):
            raise ValueError("The pinned CLI executable must remain on the approved read-only agent image.")
        if (
            not any(profile.config.agent_workdir.is_relative_to(mount.path) for mount in agent.tmpfs_mounts)
            or not math.isfinite(stop_timeout_seconds)
            or not 0 < stop_timeout_seconds <= 180
            or profile.config.timeout_seconds < stop_timeout_seconds
        ):
            raise ValueError(
                "The profile needs an approved agent workspace and a bounded stop budget within runner cleanup."
            )
        self._engine = engine
        self._profile = profile
        self._agent = agent
        self._stop_timeout = stop_timeout_seconds
        self._lifecycle_lock = asyncio.Lock()
        self._closing_requested = False
        self._launch_attempted = False
        self._exec_id: str | None = None
        self._baselines: dict[str, str] = {}
        self._network_baseline: str | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._stop_observation: AgentStopObservation | None = None

    @property
    def agent_stop(self) -> AgentStopObservation | None:
        """The most recent latched stop observation; absence is not confirmed quiescence."""
        return self._stop_observation

    async def _acquire_async(self) -> ComposeAllocation:
        allocation = await super()._acquire_async()
        inventory = await self._inventory_async()
        self._validate_inventory(inventory, require_ready=True)
        by_id = {container["Id"]: container for container in inventory.containers}
        for name, container_id in allocation.containers:
            observed = await self._engine.inspect_container_async(container_id)
            if self._static_identity(observed) != self._static_identity(by_id[container_id]):
                raise DockerEngineError("Engine exec and Compose controls do not observe the same approved container.")
            self._baselines[name] = self._static_identity(observed)
        network = await self._engine.inspect_network_async(allocation.network_id)
        self._network_baseline = self._network_identity(inventory.networks[0])
        if self._network_identity(network) != self._network_baseline:
            raise DockerEngineError("Engine exec and Compose controls do not observe the same approved network.")
        self._validate_agent_environment(by_id[allocation.container_id(self._agent.name)])
        await self._audit_engine_async(require_agent_running=True)
        return allocation

    async def launch_process_async(self, *, config: NativeCliRunConfig, prompt: str) -> DockerSandboxProcessSession:
        """
        Launch once under the lease lock, compensating uncertain create/start outcomes with agent stop.

        Returns:
            DockerSandboxProcessSession: A session bound to a daemon-issued exec ID.

        Raises:
            ValueError: If caller configuration differs from the approved profile.
            RuntimeError: If the lease cannot accept this one-shot launch.
            asyncio.CancelledError: If cancelled, after compensating guest stop is attempted.
        """
        if config != self._profile.config:
            raise ValueError("Run configuration must exactly match the image's trusted CLI profile.")
        argv = self._profile.argv(prompt)
        claimed = False
        stream: DockerExecStream | None = None
        try:
            async with self._lifecycle_lock:
                if (
                    self._closing_requested
                    or self._launch_attempted
                    or self._stop_task is not None
                    or self.snapshot().state is not EnvironmentLeaseState.READY
                ):
                    raise RuntimeError("Agent launch requires an unused, ready, stop-only lease.")
                await self._audit_engine_async(require_agent_running=True)
                self._launch_attempted = claimed = True
                assert self._allocation is not None
                handle = await self._engine.create_exec_async(
                    container_id=self._allocation.container_id(self._agent.name),
                    argv=argv,
                    user=f"{self._agent.uid}:{self._agent.gid}",
                    working_directory=str(config.agent_workdir),
                )
                self._exec_id = handle.exec_id
                if self._closing_requested:
                    raise RuntimeError("Lease release was requested before the guest exec could start.")
                stream = await self._engine.start_exec_async(handle)
                return DockerSandboxProcessSession(
                    lease=self, engine=self._engine, handle=handle, stream=stream, config=config
                )
        except (Exception, asyncio.CancelledError) as error:
            if claimed:
                try:
                    await self.stop_agent_async()
                except (Exception, asyncio.CancelledError) as stop_error:
                    raise error from stop_error
                finally:
                    if stream is not None:
                        await stream.close_async()
            raise

    async def stop_agent_async(self) -> None:
        """
        Force-stop only the exact owned agent and observe target/grader services still running.

        Failures are latched; repeated calls cannot turn uncertainty into success.
        Caller cancellation waits for the bounded barrier before propagating.

        Raises:
            RuntimeError: If ownership, security, termination or target preservation is unverified.
            TimeoutError: If the stop barrier exceeds its deadline.
            asyncio.CancelledError: If the caller cancels, after the barrier settles.
        """
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_agent_async())
        await _settle_task_async(self._stop_task)

    async def _stop_agent_async(self) -> None:
        container_id = self._allocation.container_id(self._agent.name) if self._allocation else ""
        try:
            async with asyncio.timeout(self._stop_timeout):
                async with self._lifecycle_lock:
                    if self._closing_requested or not self._launch_attempted:
                        raise RuntimeError("Agent stop requires a launched lease that is not closing.")
                    observations = await self._audit_engine_async(require_agent_running=False)
                    if observations[self._agent.name]["State"]["Running"]:
                        await self._engine.kill_container_async(container_id)
                    while True:
                        observations = await self._audit_engine_async(require_agent_running=False)
                        if self._is_stopped(observations[self._agent.name]["State"]):
                            break
                        await asyncio.sleep(0.05)
                    assert self._allocation is not None
                    self._stop_observation = AgentStopObservation(
                        container_id=container_id,
                        stopped=True,
                        exec_id=self._exec_id,
                        preserved_services=tuple(
                            (name, identifier)
                            for name, identifier in self._allocation.containers
                            if name != self._agent.name
                        ),
                    )
        except (Exception, asyncio.CancelledError) as error:
            self._stop_observation = AgentStopObservation(
                container_id=container_id,
                stopped=False,
                exec_id=self._exec_id,
                error=f"Agent stop failed: {type(error).__name__}",
            )
            raise

    async def close_async(self) -> None:
        """
        Serialize full-project release behind any active launch or stop barrier.

        Raises:
            RuntimeError: If another task is acquiring the lease.
        """
        if self._acquire_task is not None and self._acquire_task is not asyncio.current_task():
            raise RuntimeError("Cancel and await acquisition before releasing the stop-only lease.")
        if self._health_task is not None:
            raise RuntimeError("Cancel and await the active health check before releasing the stop-only lease.")
        self._closing_requested = True
        await super().close_async()

    async def _release_resources_async(self) -> None:
        async with self._lifecycle_lock:
            await super()._release_resources_async()

    async def check_health_async(self) -> tuple[EnvironmentHealth, ...]:
        """
        Serialize health observation with stop and full-project release.

        Returns:
            tuple[EnvironmentHealth, ...]: Normal Compose health observations.

        Raises:
            RuntimeError: If the lease is stopping or closing.
        """
        async with self._lifecycle_lock:
            if self._closing_requested or self._stop_task is not None:
                raise RuntimeError("A stopping or closing agent lease cannot advertise all-service readiness.")
            return await super().check_health_async()

    async def _audit_engine_async(self, *, require_agent_running: bool) -> dict[str, dict[str, Any]]:
        if self._allocation is None or len(self._baselines) != len(self._spec.services):
            raise DockerEngineError("Engine/Compose allocation binding has not been established.")
        observations: dict[str, dict[str, Any]] = {}
        for name, container_id in self._allocation.containers:
            observed = await self._engine.inspect_container_async(container_id)
            if self._static_identity(observed) != self._baselines[name]:
                raise DockerEngineError("Owned container identity, security or immutable configuration changed.")
            state = observed.get("State")
            if not isinstance(state, dict):
                raise DockerEngineError("Daemon container state is missing.")
            if name != self._agent.name or require_agent_running:
                if not self._is_running(state):
                    raise DockerEngineError("A required agent/target/grader service is not observed running.")
            elif not self._is_running(state) and not self._is_stopped(state):
                raise DockerEngineError("Agent state is neither safely running nor confirmed stopped.")
            settings = observed.get("NetworkSettings")
            if not isinstance(settings, dict):
                raise DockerEngineError("Engine container inspection has no network settings.")
            networks = settings.get("Networks")
            if name == self._agent.name and self._is_stopped(state) and networks == {}:
                observations[name] = observed
                continue
            if not isinstance(networks, dict) or set(networks) != {self._network}:
                raise DockerEngineError("An owned container's network membership changed.")
            if networks[self._network].get("NetworkID") != self._allocation.network_id:
                raise DockerEngineError("The owned container network identity changed.")
            observations[name] = observed
        network = await self._engine.inspect_network_async(self._allocation.network_id)
        if self._network_identity(network) != self._network_baseline:
            raise DockerEngineError("The challenge network's ownership or isolation changed.")
        attached = network.get("Containers")
        identifiers = {identifier for _, identifier in self._allocation.containers}
        required = {
            identifier
            for name, identifier in self._allocation.containers
            if self._is_running(observations[name]["State"])
        }
        if not isinstance(attached, dict) or not required.issubset(attached) or set(attached) - identifiers:
            raise DockerEngineError("Challenge network membership contains missing or unowned services.")
        return observations

    def _validate_agent_environment(self, container: dict[str, Any]) -> None:
        values = container["Config"].get("Env")
        if not isinstance(values, list) or not all(isinstance(value, str) and "=" in value for value in values):
            raise DockerEngineError("The pinned agent image must declare an explicit nonsecret environment.")
        environment = dict(value.split("=", 1) for value in values)
        gateway_key = (
            "OPENAI_BASE_URL"
            if self._profile.config.protocol is NativeCliProtocol.CODEX_EXEC_JSON
            else "ANTHROPIC_BASE_URL"
        )
        allowed = {"PATH", "LANG", "LC_ALL", "HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME", gateway_key}
        if len(environment) != len(values) or set(environment) - allowed:
            raise DockerEngineError(
                "Agent image environment contains duplicate or unapproved credential/configuration fields."
            )
        if environment.get(gateway_key) != self._profile.config.model_gateway_endpoint:
            raise DockerEngineError("The pinned agent gateway does not match the approved CLI profile.")
        if not environment.get("HOME"):
            raise DockerEngineError("The stop-only CLI requires an explicit ephemeral guest HOME.")
        for name in ("HOME", "TMPDIR", "XDG_CACHE_HOME", "XDG_CONFIG_HOME"):
            if name in environment:
                path = PurePosixPath(environment[name])
                if (
                    not path.is_absolute()
                    or ".." in path.parts
                    or str(path) != environment[name]
                    or not any(path.is_relative_to(mount.path) for mount in self._agent.tmpfs_mounts)
                ):
                    raise DockerEngineError("CLI home/cache paths must stay inside approved ephemeral guest state.")

    @staticmethod
    def _static_identity(container: dict[str, Any]) -> str:
        fields = ("Id", "Name", "Image", "Config", "HostConfig", "Mounts")
        if any(field not in container for field in fields):
            raise DockerEngineError("Container inspection lacks immutable identity or security fields.")
        return json.dumps(
            {field: container[field] for field in fields}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @staticmethod
    def _network_identity(network: dict[str, Any]) -> str:
        fields = ("Id", "Name", "Driver", "Internal", "EnableIPv6", "Scope", "Options", "Labels")
        if any(field not in network for field in fields):
            raise DockerEngineError("Network inspection lacks required identity/isolation fields.")
        return json.dumps(
            {field: network[field] for field in fields}, sort_keys=True, separators=(",", ":"), allow_nan=False
        )

    @staticmethod
    def _is_running(state: dict[str, Any]) -> bool:
        pid, status = state.get("Pid"), state.get("Status")
        return (
            type(pid) is int
            and type(status) is str
            and state.get("Running") is True
            and status == "running"
            and all(state.get(key) is False for key in ("Paused", "Restarting", "Dead"))
            and pid > 0
        )

    @staticmethod
    def _is_stopped(state: dict[str, Any]) -> bool:
        pid, status = state.get("Pid"), state.get("Status")
        return (
            type(pid) is int
            and type(status) is str
            and state.get("Running") is False
            and status == "exited"
            and all(state.get(key) is False for key in ("Paused", "Restarting", "Dead"))
            and pid == 0
        )


class DockerSandboxLauncher:
    """Adapt only the already-verified stop-only lease to NativeCliRunner's launcher protocol."""

    def __init__(self, *, lease: DockerStopOnlyAgentLease) -> None:
        """Bind the single-use lease; no executable or credentials are supplied by the candidate."""
        self._lease = lease

    async def launch_async(self, *, config: NativeCliRunConfig, prompt: str) -> DockerSandboxProcessSession:
        """
        Execute only the exact lease-approved image/profile in the agent container.

        Returns:
            DockerSandboxProcessSession: A daemon-attributed, stop-only guest execution.
        """
        return await self._lease.launch_process_async(config=config, prompt=prompt)


class DockerSandboxProcessSession:
    """Actual exec output and exit inspection; host transport exit is never a guest result."""

    def __init__(
        self,
        *,
        lease: DockerStopOnlyAgentLease,
        engine: DockerEngineClient,
        handle: DockerExecHandle,
        stream: DockerExecStream,
        config: NativeCliRunConfig,
    ) -> None:
        """Attach a started exec to its lease-owned stop barrier and deadline."""
        self._lease = lease
        self._engine = engine
        self._handle = handle
        self._stream = stream
        self._deadline = asyncio.get_running_loop().time() + config.timeout_seconds
        self._exit_code: int | None = None
        self._stop_task: asyncio.Task[None] | None = None

    @property
    def exec_id(self) -> str:
        """The daemon-issued exec identity for this single session."""
        return self._handle.exec_id

    @property
    def container_id(self) -> str:
        """The full owned agent container identity bound to this exec."""
        return self._handle.container_id

    async def read_chunks_async(self) -> AsyncIterator[NativeCliProcessChunk]:
        """
        Yield actual stdout/stderr bytes, bounded by the run deadline.

        Yields:
            NativeCliProcessChunk: Original daemon-demultiplexed pipe data.
        """
        iterator = self._stream.read_chunks_async()
        try:
            while True:
                async with asyncio.timeout_at(self._deadline):
                    try:
                        part = await anext(iterator)
                    except StopAsyncIteration:
                        break
                yield part
        finally:
            await iterator.aclose()

    async def wait_async(self) -> int:
        """
        Read the daemon's exact exec exit only after a complete output stream.

        Returns:
            int: Actual guest CLI exit code, including nonzero exits.

        Raises:
            DockerEngineError: If output is incomplete or the daemon provides no valid exit.
        """
        if not self._stream.eof_observed:
            raise DockerEngineError("Guest exit cannot be reported before complete exec output.")
        if self._exit_code is not None:
            return self._exit_code
        async with asyncio.timeout_at(self._deadline):
            while True:
                observed = await self._engine.inspect_exec_async(self._handle)
                if observed["Running"] is False:
                    value = observed.get("ExitCode")
                    if type(value) is not int or not 0 <= value <= 255:
                        raise DockerEngineError("Daemon exec inspection has no valid guest ExitCode.")
                    self._exit_code = value
                    return value
                await asyncio.sleep(0.05)

    async def stop_async(self) -> None:
        """Settle the lease-owned guest-stop barrier on success, failure, timeout or cancellation."""
        if self._stop_task is None:
            self._stop_task = asyncio.create_task(self._stop_async())
        await _settle_task_async(self._stop_task)

    async def _stop_async(self) -> None:
        try:
            await self._lease.stop_agent_async()
        finally:
            async with asyncio.timeout(5):
                await self._stream.close_async()
