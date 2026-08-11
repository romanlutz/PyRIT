# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Typed native code evaluation over an existing sandbox session."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import JSONValue, Score
from pyrit.sandbox import (
    DockerPullPolicy,
    DockerServiceImageSpec,
    SandboxExecRequest,
    SandboxOperationStatus,
)
from pyrit.scenario.capability_suite.manifest import (
    CapabilitySuiteManifest,
    DockerSandboxProviderManifestConfig,
    validate_safe_relative_path,
)
from pyrit.scenario.capability_suite.scorers import _final_message_text

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Mapping

    from pyrit.executor.capability import CapabilityTaskResult
    from pyrit.sandbox import SandboxEnvironment, SandboxSession


class CodeExtractionMode(str, Enum):
    """Supported declarative extraction strategies for generated code."""

    RAW = "raw"
    FIRST_FENCED_BLOCK = "first_fenced_block"
    PYTHON_FENCED_BLOCK = "python_fenced_block"
    HUMAN_EVAL_COMPLETION = "human_eval_completion"
    LAST_NON_EMPTY_LINE = "last_non_empty_line"


class CodeComparisonMode(str, Enum):
    """Supported deterministic output comparison strategies."""

    EXACT = "exact"
    NORMALIZE_NEWLINES = "normalize_newlines"
    STRIP = "strip"


class CodeEvaluationFile(BaseModel):
    """One untrusted file materialized inside the sandbox for a test."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1)
    content: str

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        validate_safe_relative_path(value)
        return value


class CodeEvaluationTestCase(BaseModel):
    """One isolated code test with optional stdin, files, and expected output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    test_id: str = Field(min_length=1)
    environment: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    source_suffix: str = ""
    stdin: str | None = None
    expected_output: str | None = None
    output_path: str | None = None
    files: tuple[CodeEvaluationFile, ...] = ()
    run_argv: tuple[str, ...] | None = None
    timeout_seconds: float | None = Field(default=None, gt=0)
    comparison: CodeComparisonMode = CodeComparisonMode.EXACT

    @field_validator("output_path")
    @classmethod
    def _validate_output_path(cls, value: str | None) -> str | None:
        if value is not None:
            validate_safe_relative_path(value)
        return value

    @field_validator("run_argv")
    @classmethod
    def _validate_run_argv(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None:
            _validate_command_template(value)
        return value


class CodeEvaluationSpec(BaseModel):
    """Provider-neutral native code-evaluation specification."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    language: str = Field(min_length=1)
    runtime: str = Field(min_length=1)
    environment: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    extraction: CodeExtractionMode = CodeExtractionMode.RAW
    fence_languages: tuple[str, ...] = ()
    candidate_path: str = "candidate.py"
    source_prefix: str = ""
    source_suffix: str = ""
    source_suffix_lines: tuple[str, ...] = ()
    max_source_chars: int | None = Field(default=None, gt=0)
    compile_argv: tuple[str, ...] | None = None
    run_argv: tuple[str, ...]
    tests: tuple[CodeEvaluationTestCase, ...] = Field(max_length=100)
    timeout_seconds: float = Field(default=30.0, gt=0)
    stop_on_failure: bool = False
    category: str | None = None
    requires_isolation: bool = True
    requires_no_egress: bool = True
    required_dependencies: tuple[str, ...] = ()

    @field_validator("candidate_path")
    @classmethod
    def _validate_candidate_path(cls, value: str) -> str:
        validate_safe_relative_path(value)
        if value.split("/", maxsplit=1)[0].startswith("-"):
            raise ValueError("Code-evaluation candidate paths cannot begin with '-'.")
        return value

    @field_validator("compile_argv")
    @classmethod
    def _validate_compile_argv(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None:
            _validate_command_template(value)
        return value

    @field_validator("run_argv")
    @classmethod
    def _validate_run_argv(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _validate_command_template(value)
        return value

    @model_validator(mode="after")
    def _validate_tests(self) -> CodeEvaluationSpec:
        if not self.tests:
            raise ValueError("Code evaluation requires at least one test.")
        test_ids = [test.test_id for test in self.tests]
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("Code-evaluation test_id values must be unique.")
        output_paths = [test.output_path for test in self.tests if test.output_path is not None]
        if len(output_paths) != len(set(output_paths)):
            raise ValueError("Code-evaluation output_path values must be unique across tests.")
        for test in self.tests:
            staged_paths = {self.candidate_path}
            for file in test.files:
                if file.path in staged_paths:
                    raise ValueError(f"Code-evaluation test '{test.test_id}' reuses file path '{file.path}'.")
                staged_paths.add(file.path)
            if test.output_path in staged_paths:
                raise ValueError(
                    f"Code-evaluation test '{test.test_id}' output_path must not reuse a staged input path."
                )
        environments = [test.environment or self.environment or "default" for test in self.tests]
        if len(environments) != len(set(environments)):
            raise ValueError("Each code-evaluation test requires a distinct sandbox environment.")
        return self


def _validate_command_template(argv: tuple[str, ...]) -> None:
    if not argv:
        raise ValueError("Code-evaluation argv must contain at least one element.")
    allowed_tokens = {"candidate", "test_id"}
    for argument in argv:
        tokens = set(re.findall(r"\{([^{}]+)\}", argument))
        unknown = tokens - allowed_tokens
        if unknown:
            raise ValueError(f"Unknown code-evaluation command template token(s): {sorted(unknown)}.")
        if argument.count("{") != argument.count("}") or ("{" in argument and not tokens):
            raise ValueError(f"Invalid code-evaluation command template argument: {argument!r}.")


def extract_generated_code(*, text: str, spec: CodeEvaluationSpec) -> str:
    """
    Extract candidate content using only a declared, audited strategy.

    Returns:
        str: Extracted candidate content.
    """
    if spec.extraction is CodeExtractionMode.RAW:
        return text
    if spec.extraction is CodeExtractionMode.LAST_NON_EMPTY_LINE:
        lines = [line for line in text.splitlines() if line.strip()]
        return lines[-1] if lines else ""
    if spec.extraction is CodeExtractionMode.PYTHON_FENCED_BLOCK:
        matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        return matches[0] if matches else text
    if spec.extraction is CodeExtractionMode.HUMAN_EVAL_COMPLETION:
        matches = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        matches.extend(re.findall(r"```\n(.*?)```", text, re.DOTALL))
        if not matches:
            return text
        body = matches[0]
        return body[body.find(":\n    ") + 2 :]

    fence_pattern = re.compile(r"```(?P<language>[A-Za-z0-9_+.-]*)[ \t]*\r?\n(?P<body>.*?)```", re.DOTALL)
    matches = list(fence_pattern.finditer(text))
    if not matches:
        return text
    allowed = {language.casefold() for language in spec.fence_languages}
    selected = next(
        (
            match
            for match in matches
            if not allowed or not match.group("language") or match.group("language").casefold() in allowed
        ),
        matches[0],
    )
    return selected.group("body")


def _render_argv(*, template: tuple[str, ...], candidate_path: str, test_id: str) -> tuple[str, ...]:
    return tuple(argument.format(candidate=candidate_path, test_id=test_id) for argument in template)


def _compare_output(*, actual: str, expected: str, mode: CodeComparisonMode) -> bool:
    if mode is CodeComparisonMode.EXACT:
        return actual == expected
    if mode is CodeComparisonMode.NORMALIZE_NEWLINES:
        return actual.replace("\r\n", "\n") == expected.replace("\r\n", "\n")
    return actual.strip() == expected.strip()


def _decode_output(value: bytes) -> str:
    return value.decode("utf-8", errors="replace")


class CodeEvaluationScorer:
    """Execute generated code in a live isolation-capable sandbox session."""

    def __init__(self, *, spec: CodeEvaluationSpec, memory: MemoryInterface | None = None) -> None:
        """Initialize the scorer from a validated declarative specification."""
        self.spec = spec
        self._memory = memory or CentralMemory.get_memory_instance()

    @classmethod
    def from_config(cls, config: Mapping[str, JSONValue]) -> CodeEvaluationScorer:
        """
        Build a scorer from strict manifest configuration.

        Returns:
            CodeEvaluationScorer: The validated scorer.
        """
        return cls(spec=CodeEvaluationSpec.model_validate(dict(config)))

    async def score_async(
        self,
        *,
        result: CapabilityTaskResult,
        objective: str,
        session: SandboxSession,
        cancellation_event: asyncio.Event | None = None,
    ) -> list[Score]:
        """
        Extract, stage, execute, and score generated code.

        Returns:
            list[Score]: One native correctness score with per-test diagnostics.
        """
        text = _final_message_text(result=result, memory=self._memory)
        candidate = extract_generated_code(text=text, spec=self.spec)
        diagnostics: list[dict[str, str | int | bool | None]] = []
        passed = True

        for test in self.spec.tests:
            environment = session.get_environment(test.environment or self.spec.environment)
            diagnostic = await self._run_test_async(
                candidate=candidate,
                test=test,
                environment=environment,
                cancellation_event=cancellation_event,
            )
            diagnostics.append(diagnostic)
            if bool(diagnostic["cancelled"]):
                break
            test_passed = bool(diagnostic["passed"])
            passed = passed and test_passed
            if not test_passed and self.spec.stop_on_failure:
                break

        piece_id = result.final_message_piece_ids[-1] if result.final_message_piece_ids else str(result.case_id)
        return [
            Score(
                score_value=str(passed),
                score_type="true_false",
                score_category=[self.spec.category] if self.spec.category else [],
                score_rationale=(
                    f"Native code evaluation passed {sum(bool(item['passed']) for item in diagnostics)}"
                    f"/{len(self.spec.tests)} tests."
                ),
                score_metadata={
                    "language": self.spec.language,
                    "runtime": self.spec.runtime,
                    "tests_total": len(self.spec.tests),
                    "tests_passed": sum(bool(item["passed"]) for item in diagnostics),
                    "diagnostics_json": json.dumps(diagnostics, sort_keys=True),
                },
                message_piece_id=piece_id,
                objective=objective,
            )
        ]

    async def _run_test_async(
        self,
        *,
        candidate: str,
        test: CodeEvaluationTestCase,
        environment: SandboxEnvironment,
        cancellation_event: asyncio.Event | None,
    ) -> dict[str, str | int | bool | None]:
        source = f"{self.spec.source_prefix}{candidate}{self.spec.source_suffix}"
        for line in self.spec.source_suffix_lines:
            if self.spec.max_source_chars is not None and len(source) + len(line) > self.spec.max_source_chars:
                break
            source += line
        source += test.source_suffix
        write_result = await environment.write_file_async(
            path=self.spec.candidate_path,
            data=source.encode("utf-8"),
        )
        if write_result.status is not SandboxOperationStatus.SUCCEEDED:
            return _operation_diagnostic(
                test_id=test.test_id,
                phase="stage",
                status=write_result.status,
                error_code=write_result.error_code,
                error_message=write_result.error_message,
            )
        for file in test.files:
            file_result = await environment.write_file_async(path=file.path, data=file.content.encode("utf-8"))
            if file_result.status is not SandboxOperationStatus.SUCCEEDED:
                return _operation_diagnostic(
                    test_id=test.test_id,
                    phase="stage",
                    status=file_result.status,
                    error_code=file_result.error_code,
                    error_message=file_result.error_message,
                )

        if self.spec.compile_argv is not None:
            compile_result = await environment.exec_async(
                request=SandboxExecRequest(
                    argv=_render_argv(
                        template=self.spec.compile_argv,
                        candidate_path=self.spec.candidate_path,
                        test_id=test.test_id,
                    ),
                    timeout_seconds=test.timeout_seconds or self.spec.timeout_seconds,
                ),
                cancellation_event=cancellation_event,
            )
            if compile_result.status is not SandboxOperationStatus.SUCCEEDED:
                return {
                    **_operation_diagnostic(
                        test_id=test.test_id,
                        phase="compile",
                        status=compile_result.status,
                        error_code=compile_result.error_code,
                        error_message=compile_result.error_message,
                    ),
                    "exit_code": compile_result.exit_code,
                    "stdout": _diagnostic_excerpt(compile_result.stdout),
                    "stderr": _diagnostic_excerpt(compile_result.stderr),
                }

        run_template = test.run_argv or self.spec.run_argv
        run_result = await environment.exec_async(
            request=SandboxExecRequest(
                argv=_render_argv(
                    template=run_template,
                    candidate_path=self.spec.candidate_path,
                    test_id=test.test_id,
                ),
                stdin=test.stdin.encode("utf-8") if test.stdin is not None else None,
                timeout_seconds=test.timeout_seconds or self.spec.timeout_seconds,
            ),
            cancellation_event=cancellation_event,
        )
        actual_output = _decode_output(run_result.stdout)
        file_error: str | None = None
        if test.output_path is not None and run_result.status is SandboxOperationStatus.SUCCEEDED:
            read_result = await environment.read_file_async(path=test.output_path)
            if read_result.status is SandboxOperationStatus.SUCCEEDED and read_result.data is not None:
                actual_output = _decode_output(read_result.data)
            else:
                file_error = read_result.error_code or read_result.status.value
        output_matches = (
            True
            if test.expected_output is None
            else _compare_output(actual=actual_output, expected=test.expected_output, mode=test.comparison)
        )
        execution_succeeded = run_result.status is SandboxOperationStatus.SUCCEEDED or (
            run_result.status is SandboxOperationStatus.TRUNCATED
            and run_result.exit_code == 0
            and test.expected_output is None
            and test.output_path is None
        )
        test_passed = execution_succeeded and output_matches and file_error is None
        return {
            "test_id": test.test_id,
            "phase": "run",
            "passed": test_passed,
            "status": run_result.status.value,
            "exit_code": run_result.exit_code,
            "timed_out": run_result.timed_out,
            "cancelled": run_result.cancelled,
            "stdout_truncated": run_result.stdout_truncated,
            "stderr_truncated": run_result.stderr_truncated,
            "stdout": actual_output[:4096],
            "stderr": _diagnostic_excerpt(run_result.stderr),
            "error_code": file_error or run_result.error_code,
            "error_message": run_result.error_message,
        }


def _diagnostic_excerpt(value: bytes) -> str:
    decoded = _decode_output(value)
    return decoded if len(decoded) <= 4096 else decoded[:4096] + "\n<diagnostic truncated>"


def validate_code_evaluation_preflight(*, manifest: CapabilitySuiteManifest) -> None:
    """
    Fail closed before model execution when declared code isolation cannot be enforced.

    Raises:
        ValueError: If a code-evaluation case lacks the required isolation guarantees.
    """
    for case in manifest.cases:
        code_scorers = tuple(scorer for scorer in case.scorers if scorer.kind == "code_evaluation")
        if not code_scorers:
            continue
        for scorer in code_scorers:
            spec = CodeEvaluationSpec.model_validate(dict(scorer.config))
            test_environments = {test.environment or spec.environment or "default" for test in spec.tests}
            if test_environments != set(scorer.required_environments):
                raise ValueError(
                    f"Case '{case.case_id}' code-evaluation scorer must declare exactly one required environment "
                    "for every isolated test."
                )
        provider = case.sandbox_provider or manifest.sandbox_provider
        if not isinstance(provider, DockerSandboxProviderManifestConfig):
            raise ValueError(
                f"Case '{case.case_id}' uses native code evaluation and requires the Docker isolation profile; "
                f"provider '{provider.provider_type}' is not declared equivalent."
            )
        config = provider.config
        policy = config.security_policy
        service_names = {service.service_name for service in config.services}
        required_environments = {environment for scorer in code_scorers for environment in scorer.required_environments}
        if not required_environments.issubset(service_names):
            raise ValueError(
                f"Case '{case.case_id}' code-evaluation environments require matching immutable Docker services."
            )
        if config.retain_resources_on_close:
            raise ValueError(
                f"Case '{case.case_id}' uses native code evaluation and cannot retain untrusted Docker resources."
            )
        if config.compose_files or any(not isinstance(service, DockerServiceImageSpec) for service in config.services):
            raise ValueError(
                f"Case '{case.case_id}' uses native code evaluation and requires only content-addressed prebuilt "
                "Docker service images; Compose files and runtime builds are not allowed."
            )
        if config.pull_policy is not DockerPullPolicy.NEVER:
            raise ValueError(
                f"Case '{case.case_id}' uses native code evaluation and requires Docker pull_policy='never'."
            )
        unsafe_flags = (
            policy.allow_privileged,
            policy.allow_host_namespaces,
            policy.allow_docker_socket_mount,
            policy.allow_device_mounts,
            policy.allow_bind_mounts,
            policy.allow_published_ports,
            policy.allow_dangerous_capabilities,
            policy.allow_unconfined_seccomp,
            policy.allow_unrestricted_secrets,
            policy.allow_absolute_container_paths,
            policy.allow_egress,
        )
        if (
            any(unsafe_flags)
            or not policy.drop_all_capabilities
            or not policy.isolate_interservice_network
            or not policy.require_secure_file_operations
            or not policy.read_only_root_filesystem
            or policy.workspace_tmpfs_size_mb is None
        ):
            raise ValueError(
                f"Case '{case.case_id}' uses native code evaluation and requires no egress, no host integration, "
                "an internal network, all Linux capabilities dropped, secure descriptor-relative file operations, "
                "a read-only root filesystem, and a size-limited workspace tmpfs."
            )
        if policy.default_pids_limit is None or policy.default_memory_limit is None or policy.default_cpus is None:
            raise ValueError(
                f"Case '{case.case_id}' uses native code evaluation and requires declared PID, memory, and CPU quotas."
            )


def _operation_diagnostic(
    *,
    test_id: str,
    phase: str,
    status: SandboxOperationStatus,
    error_code: str | None,
    error_message: str | None,
) -> dict[str, str | int | bool | None]:
    return {
        "test_id": test_id,
        "phase": phase,
        "passed": False,
        "status": status.value,
        "exit_code": None,
        "timed_out": status is SandboxOperationStatus.TIMED_OUT,
        "cancelled": status is SandboxOperationStatus.CANCELLED,
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout": "",
        "stderr": "",
        "error_code": error_code,
        "error_message": error_message,
    }
