# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from pyrit.sandbox import (
    DockerPullPolicy,
    DockerSandboxProviderConfig,
    DockerSecurityPolicy,
    DockerServiceImageSpec,
)
from pyrit.scenario.capability_suite import (
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CapabilitySuiteRunner,
    CaseMessageManifest,
    CaseScorerManifest,
    CodeEvaluationSpec,
    CodeEvaluationTestCase,
    CodeExtractionMode,
    DockerSandboxProviderManifestConfig,
    LocalSandboxProviderManifestConfig,
    SuiteProvenance,
    extract_generated_code,
    validate_code_evaluation_preflight,
)

_IMAGE = "python@sha256:" + ("a" * 64)


def _spec(*, extraction: CodeExtractionMode = CodeExtractionMode.RAW) -> CodeEvaluationSpec:
    return CodeEvaluationSpec(
        language="python",
        runtime="CPython 3.12",
        extraction=extraction,
        candidate_path="candidate.py",
        run_argv=("python", "{candidate}"),
        tests=(CodeEvaluationTestCase(test_id="test"),),
    )


def _manifest(
    *,
    provider: LocalSandboxProviderManifestConfig | DockerSandboxProviderManifestConfig,
) -> CapabilitySuiteManifest:
    spec = _spec()
    return CapabilitySuiteManifest(
        suite_id="code-eval",
        name="code eval",
        provenance=SuiteProvenance(source="unit-test"),
        sandbox_provider=provider,
        cases=(
            CapabilityCaseManifest(
                case_id="case",
                objective="write code",
                messages=(CaseMessageManifest(role="user", content="write code"),),
                scorers=(
                    CaseScorerManifest(
                        kind="code_evaluation",
                        config=spec.model_dump(mode="json"),
                        required_environments=("default",),
                    ),
                ),
            ),
        ),
    )


def _secure_docker_provider() -> DockerSandboxProviderManifestConfig:
    return DockerSandboxProviderManifestConfig(
        config=DockerSandboxProviderConfig(
            services=(DockerServiceImageSpec(service_name="default", image=_IMAGE),),
            pull_policy=DockerPullPolicy.NEVER,
            security_policy=DockerSecurityPolicy(
                allow_egress=False,
                read_only_root_filesystem=True,
                require_secure_file_operations=True,
                workspace_tmpfs_size_mb=64,
                default_pids_limit=32,
                default_memory_limit="256m",
                default_cpus=1,
            ),
        )
    )


def test_extract_generated_code_preserves_pinned_humaneval_behavior() -> None:
    spec = _spec(extraction=CodeExtractionMode.HUMAN_EVAL_COMPLETION)
    completion = "```\ndef ignored():\n    return 0\n```\n```python\ndef f():\n    return 1\n```"

    assert extract_generated_code(text=completion, spec=spec) == "    return 1\n"


def test_extract_generated_code_preserves_python_only_fence_behavior() -> None:
    spec = _spec(extraction=CodeExtractionMode.PYTHON_FENCED_BLOCK)

    assert extract_generated_code(text="```\nvalue = 1\n```", spec=spec) == "```\nvalue = 1\n```"
    assert extract_generated_code(text="```python\nvalue = 1\n```", spec=spec) == "value = 1\n"


def test_code_evaluation_spec_rejects_path_escape_and_unknown_command_token() -> None:
    with pytest.raises(ValidationError, match="traversal"):
        CodeEvaluationSpec(
            language="python",
            runtime="CPython",
            candidate_path="../candidate.py",
            run_argv=("python", "{candidate}"),
            tests=(CodeEvaluationTestCase(test_id="test"),),
        )

    with pytest.raises(ValidationError, match="Unknown code-evaluation command template"):
        CodeEvaluationSpec(
            language="python",
            runtime="CPython",
            run_argv=("python", "{untrusted}"),
            tests=(CodeEvaluationTestCase(test_id="test"),),
        )


def test_code_evaluation_spec_requires_a_distinct_environment_per_test() -> None:
    with pytest.raises(ValidationError, match="distinct sandbox environment"):
        CodeEvaluationSpec(
            language="python",
            runtime="CPython",
            run_argv=("python", "{candidate}"),
            tests=(
                CodeEvaluationTestCase(test_id="one"),
                CodeEvaluationTestCase(test_id="two"),
            ),
        )


def test_preflight_rejects_local_sandbox_before_execution() -> None:
    with pytest.raises(ValueError, match="requires the Docker isolation profile"):
        validate_code_evaluation_preflight(manifest=_manifest(provider=LocalSandboxProviderManifestConfig()))


async def test_runner_rejects_local_code_evaluation_before_target_preflight() -> None:
    target = MagicMock()
    request_options_factory = MagicMock()
    provider_registry = MagicMock()

    with (
        patch("pyrit.scenario.capability_suite.runner.validate_capability_target") as target_preflight,
        pytest.raises(ValueError, match="requires the Docker isolation profile"),
    ):
        await CapabilitySuiteRunner(
            manifest=_manifest(provider=LocalSandboxProviderManifestConfig()),
            target=target,
            request_options_factory=request_options_factory,
            sandbox_provider_registry=provider_registry,
        ).run_async()

    target_preflight.assert_not_called()
    provider_registry.build.assert_not_called()


def test_preflight_rejects_egress_and_missing_resource_quotas() -> None:
    provider = _secure_docker_provider()
    insecure = provider.model_copy(
        update={
            "config": provider.config.model_copy(
                update={
                    "security_policy": provider.config.security_policy.model_copy(
                        update={"allow_egress": True, "default_memory_limit": None}
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="requires no egress"):
        validate_code_evaluation_preflight(manifest=_manifest(provider=insecure))


def test_preflight_rejects_retained_untrusted_resources() -> None:
    provider = _secure_docker_provider()
    retained = provider.model_copy(
        update={"config": provider.config.model_copy(update={"retain_resources_on_close": True})}
    )

    with pytest.raises(ValueError, match="cannot retain"):
        validate_code_evaluation_preflight(manifest=_manifest(provider=retained))


def test_preflight_rejects_unbounded_writable_container_storage() -> None:
    provider = _secure_docker_provider()
    unbounded = provider.model_copy(
        update={
            "config": provider.config.model_copy(
                update={
                    "security_policy": provider.config.security_policy.model_copy(
                        update={"read_only_root_filesystem": False, "workspace_tmpfs_size_mb": None}
                    )
                }
            )
        }
    )

    with pytest.raises(ValueError, match="size-limited workspace tmpfs"):
        validate_code_evaluation_preflight(manifest=_manifest(provider=unbounded))


def test_preflight_accepts_content_addressed_locked_docker_profile() -> None:
    validate_code_evaluation_preflight(manifest=_manifest(provider=_secure_docker_provider()))


def test_prebuilt_image_rejects_mutable_tag() -> None:
    with pytest.raises(ValidationError, match="immutable"):
        DockerServiceImageSpec(service_name="default", image="python:3.12")
