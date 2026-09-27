# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded, shell-free Docker control transport, separate from task execution."""

from __future__ import annotations

import asyncio
import os
import signal
import subprocess
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True, kw_only=True)
class DockerCommandResult:
    """Control output is unusable when incomplete, even if its exit code is zero."""

    stdout: str
    stderr: str
    returncode: int | None
    timed_out: bool = False
    truncated: bool = False


class DockerCommandRunner(Protocol):
    """A trusted host runner, never a candidate-supplied command callback."""

    async def run_async(
        self, *, arguments: tuple[str, ...], input_text: str | None = None, timeout_seconds: float
    ) -> DockerCommandResult:
        """Run Docker argv and settle its process tree before returning or propagating cancellation."""
        ...


class SubprocessDockerRunner:
    """Invoke an explicit local Docker CLI without shell, ambient config, or credential forwarding."""

    def __init__(
        self,
        *,
        executable: Path,
        working_directory: Path,
        config_directory: Path,
        daemon_endpoint: str,
        output_limit: int = 1_048_576,
    ) -> None:
        """
        Configure host-owned paths and an explicit local daemon endpoint, without starting Docker.

        The config directory must already exist and remain empty. Provisioning its
        permissions belongs to the host binding, not this transport.

        Raises:
            ValueError: If paths, endpoint or capture bounds are not explicit.
        """
        if not all(path.is_absolute() for path in (executable, working_directory, config_directory)):
            raise ValueError("Docker control requires absolute executable, working and empty config paths.")
        if not daemon_endpoint.startswith(("unix:///", "npipe:////./pipe/")):
            raise ValueError("Only explicitly selected local Unix socket or Windows named-pipe daemons are supported.")
        if output_limit < 1 or output_limit > 4_194_304:
            raise ValueError("Docker control output must be bounded to at most 4 MiB per stream.")
        self._executable = executable
        self._working_directory = working_directory
        self._config_directory = config_directory
        self._daemon_endpoint = daemon_endpoint
        self._output_limit = output_limit

    async def run_async(
        self, *, arguments: tuple[str, ...], input_text: str | None = None, timeout_seconds: float
    ) -> DockerCommandResult:
        """
        Execute one bounded control operation and reap its owned process tree.

        Returns:
            DockerCommandResult: Bounded output and explicit timeout/truncation flags.

        Raises:
            ValueError: If the deadline is invalid or config is not empty.
            OSError: If spawning or terminating the owned process tree fails.
            asyncio.CancelledError: If cancelled, after process cleanup.
        """
        if not 0 < timeout_seconds <= 300:
            raise ValueError("Docker command timeout must be between zero and 300 seconds.")
        await asyncio.to_thread(self._validate_config)
        operation = asyncio.create_task(
            self._execute_async(arguments=arguments, input_text=input_text, timeout_seconds=timeout_seconds)
        )
        try:
            return await asyncio.shield(operation)
        except asyncio.CancelledError as cancellation:
            operation.cancel()
            while not operation.done():
                try:
                    await asyncio.shield(operation)
                except asyncio.CancelledError:
                    continue
                except Exception as cleanup_error:
                    raise cancellation from cleanup_error
            try:
                operation.result()
            except asyncio.CancelledError:
                pass
            except Exception as cleanup_error:
                raise cancellation from cleanup_error
            raise

    async def _execute_async(
        self, *, arguments: tuple[str, ...], input_text: str | None, timeout_seconds: float
    ) -> DockerCommandResult:
        spawn = asyncio.create_task(
            asyncio.create_subprocess_exec(
                str(self._executable),
                "--host",
                self._daemon_endpoint,
                *arguments,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self._working_directory,
                env=self._environment(),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                start_new_session=os.name != "nt",
            )
        )
        try:
            process = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            process = await spawn
            await self._terminate_async(process)
            raise
        return await self._capture_async(process=process, input_text=input_text, timeout_seconds=timeout_seconds)

    def _validate_config(self) -> None:
        for directory in (self._config_directory, self._working_directory):
            if directory.is_symlink() or not directory.is_dir() or any(directory.iterdir()):
                raise ValueError("Docker control requires existing empty host working and config directories.")

    def _environment(self) -> dict[str, str]:
        values = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT", "WINDIR") if key in os.environ}
        values.update(
            DOCKER_CONFIG=str(self._config_directory),
            COMPOSE_DISABLE_ENV_FILE="1",
            COMPOSE_BAKE="false",
            COMPOSE_PARALLEL_LIMIT="1",
            HOME=str(self._config_directory),
            USERPROFILE=str(self._config_directory),
        )
        return values

    async def _capture_async(
        self, *, process: asyncio.subprocess.Process, input_text: str | None, timeout_seconds: float
    ) -> DockerCommandResult:
        assert process.stdin is not None and process.stdout is not None and process.stderr is not None
        stdout, stderr = bytearray(), bytearray()
        readers = [
            asyncio.create_task(self._drain_async(stream=process.stdout, buffer=stdout)),
            asyncio.create_task(self._drain_async(stream=process.stderr, buffer=stderr)),
        ]
        timed_out = False
        try:
            async with asyncio.timeout(timeout_seconds):
                if input_text is not None:
                    process.stdin.write(input_text.encode("utf-8"))
                    await process.stdin.drain()
                process.stdin.close()
                await process.wait()
                truncated = any(await asyncio.shield(asyncio.gather(*readers)))
        except TimeoutError:
            timed_out = True
        finally:
            try:
                if process.returncode is None:
                    await self._terminate_async(process)
                async with asyncio.timeout(5):
                    truncated = any(await asyncio.shield(asyncio.gather(*readers)))
            except TimeoutError:
                raise OSError("Docker control output pipes did not close after process termination.") from None
            finally:
                for task in readers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*readers, return_exceptions=True)
        return DockerCommandResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            returncode=None if timed_out else process.returncode,
            timed_out=timed_out,
            truncated=truncated,
        )

    async def _drain_async(self, *, stream: asyncio.StreamReader, buffer: bytearray) -> bool:
        truncated = False
        while chunk := await stream.read(8192):
            remaining = self._output_limit - len(buffer)
            buffer.extend(chunk[:remaining])
            truncated |= len(chunk) > remaining
        return truncated

    async def _terminate_async(self, process: asyncio.subprocess.Process) -> None:
        if os.name == "nt":
            system_root = os.environ.get("SYSTEMROOT")
            if system_root is None:
                raise OSError("SystemRoot is required to terminate an owned Windows process tree.")
            killer = await asyncio.create_subprocess_exec(
                str(Path(system_root) / "System32" / "taskkill.exe"),
                "/PID",
                str(process.pid),
                "/T",
                "/F",
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                env=self._environment(),
            )
            try:
                await asyncio.wait_for(killer.wait(), timeout=5)
            finally:
                if killer.returncode is None:
                    killer.kill()
                    await killer.wait()
            if killer.returncode != 0 and process.returncode is None:
                raise OSError("Owned Docker process-tree termination could not be confirmed.")
        else:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
        await asyncio.wait_for(process.wait(), timeout=5)
