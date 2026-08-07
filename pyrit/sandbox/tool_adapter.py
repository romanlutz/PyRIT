# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Capability tool bindings for sandbox environment operations."""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pydantic import ValidationError

from pyrit.executor.capability import (
    ToolArtifact,
    ToolDeclaration,
    ToolExecutionContext,
    ToolExecutionError,
    ToolExecutionOutput,
    ToolExecutionStatus,
    ToolRegistry,
)
from pyrit.sandbox.models import (
    SandboxExecRequest,
    SandboxOperationContext,
    SandboxOperationStatus,
    SandboxWriteResult,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyrit.models import JSONValue
    from pyrit.sandbox import SandboxEnvironment, SandboxSession
    from pyrit.sandbox.models import SandboxArtifact, SandboxExecResult, SandboxReadResult


class SandboxToolAdapter:
    """Register exec, read, and write tools bound to one sandbox session."""

    def __init__(self, *, session: SandboxSession, default_environment: str | None = None) -> None:
        """Initialize a session-bound adapter."""
        self._session = session
        self._default_environment = default_environment

    def register(self, *, registry: ToolRegistry, prefix: str = "sandbox") -> tuple[str, ...]:
        """
        Register sandbox tools in the generic capability registry.

        Returns:
            tuple[str, ...]: Registered tool names.
        """
        bindings = (
            (
                ToolDeclaration(
                    name=f"{prefix}_exec",
                    description="Execute argv or an explicit shell script in a sandbox environment.",
                    input_schema=_exec_schema(),
                ),
                _SandboxExecTool(adapter=self),
            ),
            (
                ToolDeclaration(
                    name=f"{prefix}_read_file",
                    description="Read a binary file from a sandbox environment as base64.",
                    input_schema=_read_schema(),
                ),
                _SandboxReadTool(adapter=self),
            ),
            (
                ToolDeclaration(
                    name=f"{prefix}_write_file",
                    description="Write base64-encoded binary data to a sandbox environment.",
                    input_schema=_write_schema(),
                ),
                _SandboxWriteTool(adapter=self),
            ),
        )
        for declaration, implementation in bindings:
            registry.register(declaration=declaration, implementation=implementation)
        return tuple(declaration.name for declaration, _ in bindings)

    def environment(self, name: str | None) -> SandboxEnvironment:
        """
        Resolve an operation environment.

        Returns:
            SandboxEnvironment: The selected environment.

        Raises:
            ToolExecutionError: If the requested environment is unavailable.
        """
        try:
            return self._session.get_environment(name or self._default_environment)
        except KeyError as error:
            raise ToolExecutionError(
                code="environment_not_found",
                message="Requested sandbox environment is not available.",
            ) from error


class _SandboxExecTool:
    """Capability implementation for sandbox exec."""

    def __init__(self, *, adapter: SandboxToolAdapter) -> None:
        self._adapter = adapter

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        """
        Execute a sandbox process.

        Returns:
            ToolExecutionOutput: Binary-safe buffered process output.

        Raises:
            ToolExecutionError: If arguments are invalid or execution fails.
        """
        environment_name = _optional_string(arguments=arguments, name="environment")
        environment = self._adapter.environment(environment_name)
        try:
            request = SandboxExecRequest(
                argv=_optional_string_tuple(arguments=arguments, name="argv"),
                shell_script=_optional_string(arguments=arguments, name="shell_script"),
                stdin=_optional_base64(arguments=arguments, name="stdin_base64"),
                environment=_string_mapping(arguments=arguments, name="env"),
                cwd=_optional_string(arguments=arguments, name="cwd"),
                user=_optional_string(arguments=arguments, name="user"),
                timeout_seconds=_optional_float(arguments=arguments, name="timeout_seconds"),
            )
        except (ValidationError, ValueError) as error:
            raise ToolExecutionError(
                code="invalid_arguments",
                message="Sandbox exec arguments are invalid.",
            ) from error
        result = await environment.exec_async(
            request=request,
            cancellation_event=context.cancellation_event,
            operation_context=_operation_context(context),
        )
        if result.status is not SandboxOperationStatus.SUCCEEDED:
            _raise_exec_error(result)
        return ToolExecutionOutput(
            output={
                "status": result.status.value,
                "exit_code": result.exit_code,
                "stdout_base64": base64.b64encode(result.stdout).decode("ascii"),
                "stderr_base64": base64.b64encode(result.stderr).decode("ascii"),
                "stdout_truncated": result.stdout_truncated,
                "stderr_truncated": result.stderr_truncated,
            },
            artifacts=_tool_artifacts(result.artifacts),
            evidence=result.evidence,
            side_effect_completed=True,
        )


class _SandboxReadTool:
    """Capability implementation for sandbox reads."""

    def __init__(self, *, adapter: SandboxToolAdapter) -> None:
        self._adapter = adapter

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        """
        Read a sandbox file.

        Returns:
            ToolExecutionOutput: Base64 file data and metadata.

        Raises:
            ToolExecutionError: If arguments are invalid or the read fails.
        """
        environment = self._adapter.environment(_optional_string(arguments=arguments, name="environment"))
        path = _required_string(arguments=arguments, name="path")
        result = await environment.read_file_async(
            path=path,
            max_bytes=_optional_int(arguments=arguments, name="max_bytes"),
            operation_context=_operation_context(context),
        )
        if result.status is not SandboxOperationStatus.SUCCEEDED:
            _raise_file_error(result)
        return ToolExecutionOutput(
            output={
                "status": result.status.value,
                "data_base64": base64.b64encode(result.data or b"").decode("ascii"),
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
            },
            artifacts=_tool_artifacts(result.artifacts),
            evidence=result.evidence,
        )


class _SandboxWriteTool:
    """Capability implementation for sandbox writes."""

    def __init__(self, *, adapter: SandboxToolAdapter) -> None:
        self._adapter = adapter

    async def execute_async(
        self,
        *,
        arguments: Mapping[str, JSONValue],
        context: ToolExecutionContext,
    ) -> ToolExecutionOutput:
        """
        Write a sandbox file.

        Returns:
            ToolExecutionOutput: Write metadata.

        Raises:
            ToolExecutionError: If arguments are invalid or the write fails.
        """
        environment = self._adapter.environment(_optional_string(arguments=arguments, name="environment"))
        path = _required_string(arguments=arguments, name="path")
        data = _required_base64(arguments=arguments, name="data_base64")
        result = await environment.write_file_async(
            path=path,
            data=data,
            operation_context=_operation_context(context),
        )
        if result.status is not SandboxOperationStatus.SUCCEEDED:
            _raise_file_error(result)
        return ToolExecutionOutput(
            output={
                "status": result.status.value,
                "size_bytes": result.size_bytes,
                "sha256": result.sha256,
            },
            artifacts=_tool_artifacts(result.artifacts),
            evidence=result.evidence,
            side_effect_completed=True,
        )


def _operation_context(context: ToolExecutionContext) -> SandboxOperationContext:
    return SandboxOperationContext(call_id=context.call_id, attempt_id=context.attempt_id)


def _raise_exec_error(result: SandboxExecResult) -> None:
    status = (
        ToolExecutionStatus.TIMED_OUT
        if result.status is SandboxOperationStatus.TIMED_OUT
        else ToolExecutionStatus.CANCELLED
        if result.status is SandboxOperationStatus.CANCELLED
        else ToolExecutionStatus.FAILED
    )
    raise ToolExecutionError(
        code=result.error_code or result.status.value,
        message=result.error_message or f"Sandbox exec ended with status '{result.status.value}'.",
        side_effect_completed=result.exit_code is not None,
        status=status,
        evidence=result.evidence,
    )


def _raise_file_error(result: SandboxReadResult | SandboxWriteResult) -> None:
    raise ToolExecutionError(
        code=result.error_code or result.status.value,
        message=result.error_message or f"Sandbox file operation ended with status '{result.status.value}'.",
        side_effect_completed=result.side_effect_completed if isinstance(result, SandboxWriteResult) else False,
        evidence=result.evidence,
    )


def _tool_artifacts(artifacts: tuple[SandboxArtifact, ...]) -> tuple[ToolArtifact, ...]:
    return tuple(
        ToolArtifact(
            artifact_id=artifact.artifact_id,
            uri=artifact.uri,
            media_type=artifact.media_type,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            metadata=artifact.metadata,
        )
        for artifact in artifacts
    )


def _required_string(*, arguments: Mapping[str, JSONValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must be a string.")
    return value


def _optional_string(*, arguments: Mapping[str, JSONValue], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must be a string.")
    return value


def _optional_string_tuple(*, arguments: Mapping[str, JSONValue], name: str) -> tuple[str, ...] | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must be an array of strings.")
    return tuple(str(item) for item in value)


def _string_mapping(*, arguments: Mapping[str, JSONValue], name: str) -> dict[str, str]:
    value = arguments.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must map strings to strings.")
    return {key: str(item) for key, item in value.items()}


def _optional_float(*, arguments: Mapping[str, JSONValue], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must be a number.")
    return float(value)


def _optional_int(*, arguments: Mapping[str, JSONValue], name: str) -> int | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must be an integer.")
    if value <= 0:
        raise ToolExecutionError(code="invalid_arguments", message=f"Argument '{name}' must be greater than zero.")
    return value


def _required_base64(*, arguments: Mapping[str, JSONValue], name: str) -> bytes:
    value = _required_string(arguments=arguments, name=name)
    try:
        return base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ToolExecutionError(
            code="invalid_arguments", message=f"Argument '{name}' must be valid base64."
        ) from error


def _optional_base64(*, arguments: Mapping[str, JSONValue], name: str) -> bytes | None:
    if arguments.get(name) is None:
        return None
    return _required_base64(arguments=arguments, name=name)


def _exec_schema() -> dict[str, JSONValue]:
    return {
        "type": "object",
        "properties": {
            "environment": {"type": "string"},
            "argv": {"type": "array", "items": {"type": "string"}},
            "shell_script": {"type": "string"},
            "stdin_base64": {"type": "string"},
            "env": {"type": "object", "additionalProperties": {"type": "string"}},
            "cwd": {"type": "string"},
            "user": {"type": "string"},
            "timeout_seconds": {"type": "number", "exclusiveMinimum": 0},
        },
        "oneOf": [{"required": ["argv"]}, {"required": ["shell_script"]}],
        "additionalProperties": False,
    }


def _read_schema() -> dict[str, JSONValue]:
    return {
        "type": "object",
        "properties": {
            "environment": {"type": "string"},
            "path": {"type": "string"},
            "max_bytes": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
        "additionalProperties": False,
    }


def _write_schema() -> dict[str, JSONValue]:
    return {
        "type": "object",
        "properties": {
            "environment": {"type": "string"},
            "path": {"type": "string"},
            "data_base64": {"type": "string"},
        },
        "required": ["path", "data_base64"],
        "additionalProperties": False,
    }
