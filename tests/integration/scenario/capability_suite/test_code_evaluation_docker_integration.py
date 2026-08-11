# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Docker-backed security and behavior evidence for native code evaluation."""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

from pyrit.models import Message, TargetResponseMetadata
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration, TargetRequestOptions
from pyrit.sandbox import (
    DockerPullPolicy,
    DockerSandboxProviderConfig,
    DockerSecurityPolicy,
    DockerServiceImageSpec,
)
from pyrit.scenario.capability_suite import (
    AttemptOutcomeKind,
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CapabilitySuiteRunner,
    CaseMessageManifest,
    CaseScorerManifest,
    CaseSetupStepManifest,
    CodeEvaluationSpec,
    CodeEvaluationTestCase,
    DockerSandboxProviderManifestConfig,
    RunPolicyManifest,
    SuiteProvenance,
    build_default_sandbox_provider_registry,
    build_default_scorer_registry,
)

if TYPE_CHECKING:
    from pyrit.executor.capability import ToolDeclaration, ToolExecutionPolicy

_IMAGE = "python@sha256:519591d6871b7bc437060736b9f7456b8731f1499a57e22e6c285135ae657bf7"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "compose", "version"], check=True, capture_output=True, timeout=10)
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=10)
        subprocess.run(["docker", "image", "inspect", _IMAGE], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class _RequestOptionsFactory:
    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        return TargetRequestOptions()


class _CodeTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset({frozenset({"text"})}),
        )
    )

    def __init__(
        self,
        *,
        responses: dict[str, str],
        scoring_cancellation_event: asyncio.Event | None = None,
    ) -> None:
        super().__init__()
        self._responses = responses
        self._scoring_cancellation_event = scoring_cancellation_event

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        objective = normalized_conversation[-1].message_pieces[0].converted_value
        response = Message.from_prompt(prompt=self._responses[objective], role="assistant")
        response.message_pieces[0].conversation_id = normalized_conversation[-1].conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id=f"code-eval-{objective}",
                stop_reason="completed",
                provider_stop_reason="completed",
            )
        )
        if self._scoring_cancellation_event is not None:
            asyncio.get_running_loop().call_later(0.25, self._scoring_cancellation_event.set)
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        return


def _provider() -> DockerSandboxProviderManifestConfig:
    return DockerSandboxProviderManifestConfig(
        config=DockerSandboxProviderConfig(
            services=tuple(
                DockerServiceImageSpec(service_name=service_name, image=_IMAGE)
                for service_name in ("default", "test-one", "test-two")
            ),
            pull_policy=DockerPullPolicy.NEVER,
            security_policy=DockerSecurityPolicy(
                allow_egress=False,
                read_only_root_filesystem=True,
                require_secure_file_operations=True,
                workspace_tmpfs_size_mb=32,
                default_pids_limit=32,
                default_memory_limit="256m",
                default_cpus=1,
            ),
        )
    )


def _case(
    *,
    case_id: str,
    output_path: str | None = None,
    tests: tuple[CodeEvaluationTestCase, ...] | None = None,
) -> CapabilityCaseManifest:
    spec = CodeEvaluationSpec(
        language="python",
        runtime="CPython 3.12.11",
        candidate_path="candidate.py",
        run_argv=("python", "{candidate}"),
        tests=tests
        or (
            CodeEvaluationTestCase(
                test_id="verify",
                expected_output="ok\n",
                output_path=output_path,
                timeout_seconds=0.5,
            ),
        ),
    )
    return CapabilityCaseManifest(
        case_id=case_id,
        objective=case_id,
        messages=(CaseMessageManifest(role="user", content=case_id),),
        setup=(
            (
                CaseSetupStepManifest(
                    argv=("ln", "-s", "/etc/passwd", "candidate.py"),
                    environment="default",
                ),
            )
            if case_id == "prestage-symlink"
            else ()
        ),
        scorers=(
            CaseScorerManifest(
                kind="code_evaluation",
                scorer_id="code",
                config=spec.model_dump(mode="json"),
                required_environments=tuple(test.environment or "default" for test in spec.tests),
            ),
        ),
    )


def _owned_resources() -> dict[str, set[str]]:
    commands = {
        "containers": ["docker", "ps", "-aq", "--filter", "label=com.pyrit.sandbox.owner"],
        "networks": ["docker", "network", "ls", "-q", "--filter", "label=com.pyrit.sandbox.owner"],
        "volumes": ["docker", "volume", "ls", "-q", "--filter", "label=com.pyrit.sandbox.owner"],
    }
    return {
        kind: set(subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.splitlines())
        for kind, command in commands.items()
    }


@pytest.mark.docker
@pytest.mark.skipif(not _docker_available(), reason="Pinned Python image and a running Docker daemon are required.")
async def test_code_evaluation_enforces_runtime_network_output_path_and_cleanup() -> None:
    responses = {
        "correct": 'print("ok")',
        "wrong": 'print("wrong")',
        "compile-error": "def broken(:",
        "runtime-error": 'raise RuntimeError("boom")',
        "timeout": "while True:\n    pass",
        "excessive-output": "print('x' * 2000000)",
        "network": (
            "import socket\n"
            "try:\n"
            "    socket.create_connection(('1.1.1.1', 53), timeout=0.2)\n"
            "except OSError:\n"
            "    print('ok')\n"
            "else:\n"
            "    print('egress-enabled')\n"
        ),
        "symlink": "import os\nos.symlink('/etc/passwd', 'result.txt')",
        "hardlink": (
            "import os\n"
            "from pathlib import Path\n"
            "Path('source.txt').write_text('ok')\n"
            "os.link('source.txt', 'result.txt')"
        ),
        "prestage-symlink": 'print("ok")',
        "multi-isolated": (
            "from pathlib import Path\n"
            "state = Path('state')\n"
            "if state.exists():\n"
            "    print('stale')\n"
            "else:\n"
            "    state.write_text('created')\n"
            "    print('ok')"
        ),
    }
    before = _owned_resources()
    cases = tuple(
        _case(
            case_id=case_id,
            output_path="result.txt" if case_id in {"hardlink", "symlink"} else None,
            tests=(
                (
                    CodeEvaluationTestCase(
                        test_id="first",
                        environment="test-one",
                        expected_output="ok\n",
                    ),
                    CodeEvaluationTestCase(
                        test_id="second",
                        environment="test-two",
                        expected_output="ok\n",
                    ),
                )
                if case_id == "multi-isolated"
                else (
                    (CodeEvaluationTestCase(test_id="verify", timeout_seconds=0.5),)
                    if case_id == "excessive-output"
                    else None
                )
            ),
        )
        for case_id in responses
    )
    manifest = CapabilitySuiteManifest(
        suite_id="docker-code-eval",
        name="Docker code evaluation",
        provenance=SuiteProvenance(source="integration-test"),
        sandbox_provider=_provider(),
        run_policy=RunPolicyManifest(max_concurrency=2),
        cases=cases,
    )

    result = await CapabilitySuiteRunner(
        manifest=manifest,
        target=_CodeTarget(responses=responses),
        request_options_factory=_RequestOptionsFactory(),
        sandbox_provider_registry=build_default_sandbox_provider_registry(),
        scorer_registry=build_default_scorer_registry(),
    ).run_async()

    scores = {
        attempt.case_id: attempt.task_result.scores[0] for attempt in result.attempts if attempt.task_result is not None
    }
    assert scores["correct"].get_value() is True
    assert scores["network"].get_value() is True
    assert scores["excessive-output"].get_value() is True
    assert scores["multi-isolated"].get_value() is True
    assert all(
        scores[case_id].get_value() is False
        for case_id in ("wrong", "compile-error", "runtime-error", "timeout", "symlink", "hardlink", "prestage-symlink")
    )
    diagnostics = {case_id: json.loads(score.score_metadata["diagnostics_json"]) for case_id, score in scores.items()}
    assert diagnostics["timeout"][0]["timed_out"] is True
    assert diagnostics["excessive-output"][0]["status"] == "truncated"
    assert diagnostics["symlink"][0]["error_code"] == "path_escape"
    assert diagnostics["hardlink"][0]["error_code"] == "path_escape"
    assert diagnostics["prestage-symlink"][0]["error_code"] == "path_escape"
    assert _owned_resources() == before


@pytest.mark.docker
@pytest.mark.skipif(not _docker_available(), reason="Pinned Python image and a running Docker daemon are required.")
async def test_code_evaluation_cancellation_cleans_owned_resources() -> None:
    before = _owned_resources()
    manifest = CapabilitySuiteManifest(
        suite_id="docker-code-cancel",
        name="Docker code cancellation",
        provenance=SuiteProvenance(source="integration-test"),
        sandbox_provider=_provider(),
        cases=(_case(case_id="cancel"),),
    )
    cancellation_event = asyncio.Event()

    result = await CapabilitySuiteRunner(
        manifest=manifest,
        target=_CodeTarget(
            responses={"cancel": "while True:\n    pass"},
            scoring_cancellation_event=cancellation_event,
        ),
        request_options_factory=_RequestOptionsFactory(),
        sandbox_provider_registry=build_default_sandbox_provider_registry(),
        scorer_registry=build_default_scorer_registry(),
    ).run_async(cancellation_event=cancellation_event)
    assert result.attempts[0].outcome_kind is AttemptOutcomeKind.CANCELLED
    assert _owned_resources() == before
