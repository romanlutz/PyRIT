# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from pyrit.executor.capability import CapabilityOutcome, ToolExecutionPolicy
from pyrit.models import Message, TargetResponseMetadata
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration, TargetRequestOptions
from pyrit.sandbox import LocalSandboxProvider, LocalSandboxProviderConfig, SandboxProvider, SandboxSessionSpec
from pyrit.sandbox.contracts import SandboxEnvironment, SandboxSession
from pyrit.scenario.capability_suite.manifest import (
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CaseAssetManifest,
    CaseMessageManifest,
    CaseScorerManifest,
    LocalSandboxProviderManifestConfig,
    RunPolicyManifest,
    SuiteProvenance,
)
from pyrit.scenario.capability_suite.registries import (
    CapabilitySuiteScorerFactoryRegistry,
    SandboxProviderFactoryRegistry,
)
from pyrit.scenario.capability_suite.results import AttemptOutcomeKind, CapabilitySuiteProgress
from pyrit.scenario.capability_suite.runner import CapabilitySuiteRunner
from pyrit.scenario.capability_suite.serialization import manifest_hash

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("patch_central_database")


class RetryableError(Exception):
    """A test-only exception whose class name is registered as retryable."""


class FakeRequestOptionsFactory:
    def build_request_options(
        self,
        *,
        declarations: tuple[Any, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        self.declarations = declarations
        self.execution_policy = execution_policy
        return TargetRequestOptions()


@dataclass
class ConcurrencyProbe:
    active: int = 0
    maximum: int = 0


class FakeCapabilityTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset(
                {
                    frozenset({"text"}),
                    frozenset({"function_call_output"}),
                }
            ),
        )
    )

    def __init__(
        self,
        *,
        responses: list[Message | BaseException],
        probe: ConcurrencyProbe | None = None,
        delay_seconds: float = 0,
    ) -> None:
        super().__init__()
        self._responses = list(responses)
        self._probe = probe
        self._delay_seconds = delay_seconds

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        if self._probe is not None:
            self._probe.active += 1
            self._probe.maximum = max(self._probe.maximum, self._probe.active)
        if self._delay_seconds:
            await asyncio.sleep(self._delay_seconds)
        if self._probe is not None:
            self._probe.active -= 1
        item = self._responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        conversation_id = normalized_conversation[-1].conversation_id
        for piece in item.message_pieces:
            piece.conversation_id = conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="generation-1",
                stop_reason="completed",
                provider_stop_reason="completed",
            )
        )
        return [item]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        return


def _text_message(text: str = "done") -> Message:
    return Message.from_prompt(prompt=text, role="assistant")


class _FakeEnvironment(SandboxEnvironment):
    def __init__(self, *, name: str = "default") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def connection_info(self):  # pragma: no cover - not exercised by these tests
        raise NotImplementedError

    async def start_process_async(self, *, request, operation_context=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def exec_async(self, *, request, cancellation_event=None, operation_context=None):  # pragma: no cover
        raise NotImplementedError

    async def read_file_async(self, *, path, max_bytes=None, operation_context=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def write_file_async(self, *, path, data, operation_context=None):  # pragma: no cover - unused
        raise NotImplementedError


class FakeSandboxSession(SandboxSession):
    """A real ``SandboxSession`` subclass with configurable init/close outcomes."""

    def __init__(
        self,
        *,
        spec: SandboxSessionSpec,
        events: list[tuple[str, str]],
        label: str,
        close_error: Exception | None = None,
    ) -> None:
        super().__init__(provider_name="fake", spec=spec)
        self._events = events
        self.label = label
        self._close_error = close_error
        self._environment = _FakeEnvironment()

    @property
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        return (self._environment,)

    async def _initialize_async(self) -> None:
        self._events.append(("initialize", self.label))

    async def _close_async(self) -> None:
        self._events.append(("close", self.label))
        if self._close_error is not None:
            raise self._close_error


class FakeSandboxProvider(SandboxProvider):
    """A real ``SandboxProvider`` subclass whose session creation is fully injected."""

    def __init__(self, *, session_factory) -> None:
        super().__init__()
        self._session_factory = session_factory
        self.created_session_count = 0

    @property
    def name(self) -> str:
        return "fake"

    async def _prepare_async(self) -> None:
        return None

    async def _cleanup_async(self) -> None:
        return None

    async def _cleanup_orphans_async(self) -> int:
        return 0

    async def _create_session_async(self, *, spec, evidence_sink):
        self.created_session_count += 1
        return self._session_factory(spec)


class ProbeScorer:
    """A ``CapabilitySuiteScorer`` that records when it ran relative to session events."""

    def __init__(self, *, events: list[tuple[str, str]]) -> None:
        self._events = events

    async def score_async(self, *, result, objective, session):
        self._events.append(("score", getattr(session, "label", "?")))
        return []


class StaticAssetResolver:
    def __init__(self, *, content: bytes) -> None:
        self._content = content

    async def read_asset_async(self, *, asset: CaseAssetManifest) -> bytes:
        return self._content


class ProgressRecorder:
    def __init__(self) -> None:
        self.completed: list[int] = []

    async def report_progress_async(self, *, progress: CapabilitySuiteProgress) -> None:
        self.completed.append(progress.completed_units)


def _provenance() -> SuiteProvenance:
    return SuiteProvenance(source="unit-test")


def _case(case_id: str = "case-1", **overrides: object) -> CapabilityCaseManifest:
    defaults: dict[str, object] = {
        "case_id": case_id,
        "objective": "finish the task",
        "messages": (CaseMessageManifest(role="user", content="hello"),),
    }
    defaults.update(overrides)
    return CapabilityCaseManifest(**defaults)


def _manifest(*, cases: tuple[CapabilityCaseManifest, ...], run_policy: RunPolicyManifest | None = None):
    return CapabilitySuiteManifest(
        suite_id="suite-1",
        name="Example suite",
        provenance=_provenance(),
        sandbox_provider=LocalSandboxProviderManifestConfig(),
        run_policy=run_policy or RunPolicyManifest(),
        cases=cases,
    )


def _provider_registry(provider: SandboxProvider) -> SandboxProviderFactoryRegistry:
    registry = SandboxProviderFactoryRegistry()
    registry.register(provider_type="local", factory=lambda config: provider)
    return registry


async def test_runner_happy_path_single_case_success() -> None:
    events: list[tuple[str, str]] = []
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=events, label="s1")
    )
    manifest = _manifest(cases=(_case(),))
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )

    result = await runner.run_async()

    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.outcome_kind is AttemptOutcomeKind.SUCCESS
    assert attempt.task_result is not None
    assert attempt.task_result.outcome is CapabilityOutcome.COMPLETED
    assert events == [("initialize", "s1"), ("close", "s1")]
    assert provider.created_session_count == 1
    assert result.manifest_hash == manifest_hash(manifest)
    assert result.aggregate.total_attempts == 1
    assert result.aggregate.success_rate == 1.0


async def test_runner_bounded_concurrency_caps_active_attempts() -> None:
    events: list[tuple[str, str]] = []
    counter = itertools.count(1)
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=events, label=f"s{next(counter)}")
    )
    probe = ConcurrencyProbe()
    manifest = _manifest(
        cases=(_case(),),
        run_policy=RunPolicyManifest(epochs=4, max_concurrency=2),
    )
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message() for _ in range(4)], probe=probe, delay_seconds=0.05),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )

    result = await runner.run_async()

    assert len(result.attempts) == 4
    assert all(attempt.outcome_kind is AttemptOutcomeKind.SUCCESS for attempt in result.attempts)
    assert probe.maximum == 2
    assert provider.created_session_count == 4


async def test_runner_reports_monotonic_progress() -> None:
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=[], label=str(spec.session_id))
    )
    progress = ProgressRecorder()
    runner = CapabilitySuiteRunner(
        manifest=_manifest(
            cases=(_case(),),
            run_policy=RunPolicyManifest(epochs=2),
        ),
        target=FakeCapabilityTarget(responses=[_text_message(), _text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
        progress_sink=progress,
    )
    await runner.run_async()
    assert progress.completed == [1, 2]


async def test_runner_retries_only_known_retryable_failures_with_fresh_sandbox() -> None:
    events: list[tuple[str, str]] = []
    outcomes: list[Exception | None] = [RetryableError("transient"), RetryableError("transient"), None]
    counter = itertools.count(1)

    def _session_factory(spec: SandboxSessionSpec):
        outcome = outcomes.pop(0)
        if outcome is not None:
            raise outcome
        return FakeSandboxSession(spec=spec, events=events, label=f"s{next(counter)}")

    provider = FakeSandboxProvider(session_factory=_session_factory)
    manifest = _manifest(
        cases=(_case(),),
        run_policy=RunPolicyManifest(max_retries=2, retryable_error_codes=("RetryableError",)),
    )
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )

    result = await runner.run_async()

    assert len(result.attempts) == 3
    assert [a.outcome_kind for a in result.attempts] == [
        AttemptOutcomeKind.RETRY,
        AttemptOutcomeKind.RETRY,
        AttemptOutcomeKind.SUCCESS,
    ]
    assert provider.created_session_count == 3
    assert result.aggregate.outcome_counts == {"retry": 2, "success": 1}


async def test_runner_non_retryable_failure_is_not_retried() -> None:
    def _session_factory(spec: SandboxSessionSpec):
        raise ValueError("permanent failure")

    provider = FakeSandboxProvider(session_factory=_session_factory)
    manifest = _manifest(
        cases=(_case(),),
        run_policy=RunPolicyManifest(max_retries=2),  # retryable_error_codes defaults to empty
    )
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )

    result = await runner.run_async()

    assert len(result.attempts) == 1
    assert result.attempts[0].outcome_kind is AttemptOutcomeKind.FAILURE
    assert provider.created_session_count == 1


async def test_runner_scorer_runs_before_cleanup_while_session_still_open() -> None:
    events: list[tuple[str, str]] = []
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=events, label="s1")
    )
    scorer_registry = CapabilitySuiteScorerFactoryRegistry()
    scorer_registry.register(kind="probe_scorer", factory=lambda config: ProbeScorer(events=events))
    manifest = _manifest(cases=(_case(scorers=(CaseScorerManifest(kind="probe_scorer"),)),))
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
        scorer_registry=scorer_registry,
    )

    result = await runner.run_async()

    assert result.attempts[0].outcome_kind is AttemptOutcomeKind.SUCCESS
    assert events == [("initialize", "s1"), ("score", "s1"), ("close", "s1")]


async def test_runner_cleanup_failure_distinguished_from_run_failure() -> None:
    events: list[tuple[str, str]] = []
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(
            spec=spec, events=events, label="s1", close_error=RuntimeError("close boom")
        )
    )
    manifest = _manifest(cases=(_case(),))
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )

    result = await runner.run_async()

    attempt = result.attempts[0]
    assert attempt.outcome_kind is AttemptOutcomeKind.CLEANUP_FAILURE
    assert attempt.task_result is not None  # the run itself succeeded
    assert attempt.error is not None
    assert "cleanup failed" in attempt.error
    assert "run failed" not in attempt.error


async def test_runner_cleanup_failure_combined_with_run_failure_message() -> None:
    events: list[tuple[str, str]] = []
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(
            spec=spec, events=events, label="s1", close_error=RuntimeError("close boom")
        )
    )
    # Case declares a scorer, but no scorer_registry is given to the runner, so
    # `_apply_scorers_async` raises a plain ValueError -- a genuine run failure --
    # in addition to the sandbox's close failure.
    manifest = _manifest(cases=(_case(scorers=(CaseScorerManifest(kind="unregistered"),)),))
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
        scorer_registry=None,
    )

    result = await runner.run_async()

    attempt = result.attempts[0]
    assert attempt.outcome_kind is AttemptOutcomeKind.CLEANUP_FAILURE
    assert attempt.error is not None
    assert "run failed" in attempt.error
    assert "cleanup failed" in attempt.error


async def test_runner_preserves_completed_task_result_and_records_scoring_configuration_failure() -> None:
    events: list[tuple[str, str]] = []
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=events, label="s1")
    )
    manifest = _manifest(cases=(_case(scorers=(CaseScorerManifest(kind="unregistered"),)),))
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
        scorer_registry=CapabilitySuiteScorerFactoryRegistry(),
    )
    result = await runner.run_async()
    attempt = result.attempts[0]
    assert attempt.outcome_kind is AttemptOutcomeKind.SUCCESS
    assert attempt.task_result is not None
    assert attempt.task_result.outcome is CapabilityOutcome.COMPLETED
    scoring_errors = [
        evidence for evidence in attempt.task_result.evidence if getattr(evidence, "phase", None) == "suite_scoring"
    ]
    assert len(scoring_errors) == 1


async def test_runner_cancellation_pre_check_skips_attempt_entirely() -> None:
    provider = FakeSandboxProvider(session_factory=lambda spec: pytest.fail("session should never be created"))
    manifest = _manifest(cases=(_case(),))
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )
    cancellation_event = asyncio.Event()
    cancellation_event.set()

    result = await runner.run_async(cancellation_event=cancellation_event)

    assert len(result.attempts) == 1
    assert result.attempts[0].outcome_kind is AttemptOutcomeKind.CANCELLED
    assert provider.created_session_count == 0


async def test_runner_attempts_and_aggregation_across_multiple_cases_and_epochs() -> None:
    events: list[tuple[str, str]] = []
    counter = itertools.count(1)
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=events, label=f"s{next(counter)}")
    )
    manifest = _manifest(
        cases=(_case(case_id="case-1"), _case(case_id="case-2")),
        run_policy=RunPolicyManifest(epochs=2, attempts=2, max_concurrency=2),
    )
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[_text_message() for _ in range(8)]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )

    result = await runner.run_async()

    assert result.aggregate.total_attempts == 8
    assert result.aggregate.total_runs == 8
    assert result.aggregate.outcome_counts == {"success": 8}
    assert result.aggregate.task_outcome_counts == {"completed": 8}
    assert result.aggregate.success_rate == 1.0
    assert {attempt.case_id for attempt in result.attempts} == {"case-1", "case-2"}
    assert {attempt.epoch for attempt in result.attempts} == {1, 2}
    assert {attempt.repetition for attempt in result.attempts} == {1, 2}


async def test_runner_retries_failed_executor_result_in_fresh_session() -> None:
    events: list[tuple[str, str]] = []
    counter = itertools.count(1)
    provider = FakeSandboxProvider(
        session_factory=lambda spec: FakeSandboxSession(spec=spec, events=events, label=f"s{next(counter)}")
    )
    manifest = _manifest(
        cases=(_case(),),
        run_policy=RunPolicyManifest(max_retries=1, retryable_error_codes=("RetryableError",)),
    )
    runner = CapabilitySuiteRunner(
        manifest=manifest,
        target=FakeCapabilityTarget(responses=[RetryableError("transient"), _text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    )
    result = await runner.run_async()
    assert [attempt.outcome_kind for attempt in result.attempts] == [
        AttemptOutcomeKind.RETRY,
        AttemptOutcomeKind.SUCCESS,
    ], repr(result.attempts[0].task_result)
    assert provider.created_session_count == 2
    assert result.aggregate.success_rate == 1.0


async def test_runner_rejects_asset_content_hash_mismatch_before_session_creation() -> None:
    provider = FakeSandboxProvider(
        session_factory=lambda spec: pytest.fail("hash mismatch must fail before session creation")
    )
    asset = CaseAssetManifest(
        asset_id="payload",
        source="payload.bin",
        destination="payload.bin",
        sha256=hashlib.sha256(b"expected").hexdigest(),
    )
    runner = CapabilitySuiteRunner(
        manifest=_manifest(cases=(_case(assets=(asset,)),)),
        target=FakeCapabilityTarget(responses=[]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
        asset_resolver=StaticAssetResolver(content=b"tampered"),
    )
    result = await runner.run_async()
    assert result.attempts[0].outcome_kind is AttemptOutcomeKind.FAILURE
    assert "sha256 mismatch" in (result.attempts[0].error or "")
    assert provider.created_session_count == 0


async def test_runner_ids_are_stable_for_same_manifest() -> None:
    manifest = _manifest(cases=(_case(),))

    async def _run_once_async() -> object:
        provider = FakeSandboxProvider(
            session_factory=lambda spec: FakeSandboxSession(spec=spec, events=[], label="stable")
        )
        return await CapabilitySuiteRunner(
            manifest=manifest,
            target=FakeCapabilityTarget(responses=[_text_message()]),
            request_options_factory=FakeRequestOptionsFactory(),
            sandbox_provider_registry=_provider_registry(provider),
        ).run_async()

    first = await _run_once_async()
    second = await _run_once_async()
    assert first.run_id == second.run_id
    assert first.attempts[0].attempt_id == second.attempts[0].attempt_id
    assert first.attempts[0].task_result.case_id == second.attempts[0].task_result.case_id


async def test_runner_end_to_end_with_fake_target_and_local_provider(tmp_path: Path) -> None:
    provider = LocalSandboxProvider(config=LocalSandboxProviderConfig(workspace_root=tmp_path))
    result = await CapabilitySuiteRunner(
        manifest=_manifest(cases=(_case(),)),
        target=FakeCapabilityTarget(responses=[_text_message()]),
        request_options_factory=FakeRequestOptionsFactory(),
        sandbox_provider_registry=_provider_registry(provider),
    ).run_async()
    assert result.attempts[0].outcome_kind is AttemptOutcomeKind.SUCCESS
    assert not tuple(tmp_path.glob("session-*"))
