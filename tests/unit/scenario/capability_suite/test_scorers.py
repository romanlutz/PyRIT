# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from pyrit.executor.capability import (
    CapabilityOutcome,
    CapabilityTaskResult,
    CapabilityTerminationReason,
    ToolExecutionEvidence,
    ToolExecutionStatus,
)
from pyrit.models import MessagePiece, Score, TargetIdentifier
from pyrit.sandbox import (
    SandboxEnvironment,
    SandboxExecResult,
    SandboxOperationStatus,
    SandboxReadResult,
)
from pyrit.sandbox.contracts import SandboxSession
from pyrit.scenario.capability_suite.scorers import (
    ResultOnlyScorerAdapter,
    SandboxCommandScorer,
    SandboxFileScorer,
    SandboxStateMatchMode,
    SandboxStateMatchScorer,
    TextMatchMode,
    TextMatchScorer,
    ToolEvidenceScorer,
)

pytestmark = pytest.mark.usefixtures("patch_central_database")


def _target_identifier() -> TargetIdentifier:
    return TargetIdentifier(class_name="FakeTarget", class_module="tests.unit.scenario.capability_suite")


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _persist_final_message(*, text: str) -> tuple[uuid.UUID, ...]:
    from pyrit.memory import CentralMemory

    piece = MessagePiece(original_value=text, role="assistant", conversation_id=str(uuid.uuid4()))
    memory = CentralMemory.get_memory_instance()
    memory.add_message_pieces_to_memory(message_pieces=[piece])
    return (piece.id,)


def _result(*, final_message_piece_ids: tuple[uuid.UUID, ...] = (), evidence: tuple = (), scores: tuple = ()):
    return CapabilityTaskResult(
        case_id=uuid.uuid4(),
        conversation_id=str(uuid.uuid4()),
        target_identifier=_target_identifier(),
        outcome=CapabilityOutcome.COMPLETED,
        termination_reason=CapabilityTerminationReason.COMPLETION,
        final_message_piece_ids=final_message_piece_ids,
        evidence=evidence,
        scores=scores,
        started_at=_now(),
        ended_at=_now(),
    )


class _FakeEnvironment(SandboxEnvironment):
    def __init__(self, *, read_result: SandboxReadResult | None = None, exec_result: SandboxExecResult | None = None):
        self._read_result = read_result
        self._exec_result = exec_result
        self.exec_requests: list[object] = []

    @property
    def name(self) -> str:
        return "default"

    @property
    def connection_info(self):  # pragma: no cover - not exercised by these tests
        raise NotImplementedError

    async def start_process_async(self, *, request, operation_context=None):  # pragma: no cover - unused
        raise NotImplementedError

    async def exec_async(self, *, request, cancellation_event=None, operation_context=None) -> SandboxExecResult:
        self.exec_requests.append(request)
        assert self._exec_result is not None
        return self._exec_result

    async def read_file_async(self, *, path, max_bytes=None, operation_context=None) -> SandboxReadResult:
        assert self._read_result is not None
        return self._read_result

    async def write_file_async(self, *, path, data, operation_context=None):  # pragma: no cover - unused
        raise NotImplementedError


class _FakeSession(SandboxSession):
    def __init__(self, *, environment: _FakeEnvironment):
        self._environment = environment

    @property
    def environments(self) -> tuple[SandboxEnvironment, ...]:
        return (self._environment,)

    async def _initialize_async(self) -> None:  # pragma: no cover - unused
        return None

    async def _close_async(self) -> None:  # pragma: no cover - unused
        return None


def _session_with_environment(environment: _FakeEnvironment) -> SandboxSession:
    from pyrit.sandbox import SandboxSessionSpec

    session = _FakeSession.__new__(_FakeSession)
    SandboxSession.__init__(session, provider_name="fake", spec=SandboxSessionSpec())
    session._environment = environment
    return session


class ExistingScorerStub:
    """A minimal stand-in for a ``CapabilityResultScorer`` (e.g. ``MessageScorerAdapter``)."""

    def __init__(self, *, scores: list[Score]):
        self._scores = scores
        self.calls: list[tuple[CapabilityTaskResult, str]] = []

    async def score_result_async(self, *, result: CapabilityTaskResult, objective: str) -> list[Score]:
        self.calls.append((result, objective))
        return self._scores


async def test_text_match_scorer_exact_match_true() -> None:
    ids = _persist_final_message(text="done")
    scorer = TextMatchScorer(expected_value="done", mode=TextMatchMode.EXACT)
    scores = await scorer.score_result_async(result=_result(final_message_piece_ids=ids), objective="finish")
    assert len(scores) == 1
    assert scores[0].score_value == "True"
    assert scores[0].score_type == "true_false"


async def test_text_match_scorer_exact_match_false() -> None:
    ids = _persist_final_message(text="not quite")
    scorer = TextMatchScorer(expected_value="done", mode=TextMatchMode.EXACT)
    scores = await scorer.score_result_async(result=_result(final_message_piece_ids=ids), objective="finish")
    assert scores[0].score_value == "False"


async def test_text_match_scorer_substring_match_case_insensitive() -> None:
    ids = _persist_final_message(text="The task is DONE now.")
    scorer = TextMatchScorer(expected_value="done", mode=TextMatchMode.SUBSTRING, case_sensitive=False)
    scores = await scorer.score_result_async(result=_result(final_message_piece_ids=ids), objective="finish")
    assert scores[0].score_value == "True"


def test_text_match_scorer_from_config_rejects_missing_expected_value() -> None:
    with pytest.raises(ValueError, match="expected_value"):
        TextMatchScorer.from_config({})


def test_text_match_scorer_from_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        TextMatchScorer.from_config({"expected_value": "x", "mode": "fuzzy"})


def test_native_scorer_configs_reject_unknown_fields() -> None:
    with pytest.raises(ValueError):
        TextMatchScorer.from_config({"expected_value": "x", "unexpected": True})


async def test_tool_evidence_scorer_true_when_status_matches() -> None:
    evidence = ToolExecutionEvidence(
        call_id="call-1",
        request_piece_id=uuid.uuid4(),
        attempt_id=uuid.uuid4(),
        attempt_number=1,
        tool_name="lookup",
        status=ToolExecutionStatus.SUCCEEDED,
        started_at=_now(),
        ended_at=_now(),
    )
    scorer = ToolEvidenceScorer(tool_name="lookup")
    scores = await scorer.score_result_async(result=_result(evidence=(evidence,)), objective="finish")
    assert scores[0].score_value == "True"


async def test_tool_evidence_scorer_false_when_tool_absent() -> None:
    scorer = ToolEvidenceScorer(tool_name="lookup")
    scores = await scorer.score_result_async(result=_result(evidence=()), objective="finish")
    assert scores[0].score_value == "False"


def test_tool_evidence_scorer_from_config_rejects_invalid_status() -> None:
    with pytest.raises(ValueError, match="expected_status"):
        ToolEvidenceScorer.from_config({"tool_name": "lookup", "expected_status": "not-a-status"})


async def test_sandbox_file_scorer_matches_hash_and_content() -> None:
    data = b"expected contents"
    import hashlib

    read_result = SandboxReadResult(
        status=SandboxOperationStatus.SUCCEEDED, data=data, sha256=hashlib.sha256(data).hexdigest()
    )
    environment = _FakeEnvironment(read_result=read_result)
    session = _session_with_environment(environment)
    scorer = SandboxFileScorer(path="out.txt", expected_sha256=read_result.sha256, expected_content=data)
    scores = await scorer.score_async(result=_result(), objective="finish", session=session)
    assert scores[0].score_value == "True"


async def test_sandbox_file_scorer_false_when_read_fails() -> None:
    read_result = SandboxReadResult(status=SandboxOperationStatus.NOT_FOUND)
    environment = _FakeEnvironment(read_result=read_result)
    session = _session_with_environment(environment)
    scorer = SandboxFileScorer(path="missing.txt")
    scores = await scorer.score_async(result=_result(), objective="finish", session=session)
    assert scores[0].score_value == "False"


def test_sandbox_file_scorer_from_config_requires_path() -> None:
    with pytest.raises(ValueError, match="path"):
        SandboxFileScorer.from_config({})


async def test_sandbox_command_scorer_true_on_expected_exit_code() -> None:
    exec_result = SandboxExecResult(status=SandboxOperationStatus.SUCCEEDED, exit_code=0)
    environment = _FakeEnvironment(exec_result=exec_result)
    session = _session_with_environment(environment)
    scorer = SandboxCommandScorer(argv=("echo", "hi"), expected_exit_code=0)
    scores = await scorer.score_async(result=_result(), objective="finish", session=session)
    assert scores[0].score_value == "True"
    assert len(environment.exec_requests) == 1


async def test_sandbox_command_scorer_false_on_mismatched_exit_code() -> None:
    exec_result = SandboxExecResult(status=SandboxOperationStatus.SUCCEEDED, exit_code=1)
    environment = _FakeEnvironment(exec_result=exec_result)
    session = _session_with_environment(environment)
    scorer = SandboxCommandScorer(argv=("false",), expected_exit_code=0)
    scores = await scorer.score_async(result=_result(), objective="finish", session=session)
    assert scores[0].score_value == "False"


async def test_sandbox_state_match_scorer_compares_live_state_with_final_response() -> None:
    ids = _persist_final_message(text="Recovered flag: picoCTF{native-evidence}")
    exec_result = SandboxExecResult(
        status=SandboxOperationStatus.SUCCEEDED,
        exit_code=0,
        stdout=b"picoCTF{native-evidence}\n",
    )
    environment = _FakeEnvironment(exec_result=exec_result)
    scorer = SandboxStateMatchScorer(
        shell_script="cat /flag.txt",
        mode=SandboxStateMatchMode.STATE_IN_RESPONSE,
    )
    scores = await scorer.score_async(
        result=_result(final_message_piece_ids=ids),
        objective="find flag",
        session=_session_with_environment(environment),
    )
    assert scores[0].score_value == "True"
    assert len(environment.exec_requests) == 1


async def test_sandbox_state_match_scorer_requires_successful_state_read() -> None:
    ids = _persist_final_message(text="picoCTF{native-evidence}")
    exec_result = SandboxExecResult(
        status=SandboxOperationStatus.FAILED,
        exit_code=1,
        stdout=b"picoCTF{native-evidence}",
    )
    scorer = SandboxStateMatchScorer(shell_script="cat /flag.txt")
    scores = await scorer.score_async(
        result=_result(final_message_piece_ids=ids),
        objective="find flag",
        session=_session_with_environment(_FakeEnvironment(exec_result=exec_result)),
    )
    assert scores[0].score_value == "False"


def test_sandbox_state_match_scorer_from_config_validates_mode() -> None:
    scorer = SandboxStateMatchScorer.from_config(
        {"argv": ["cat", "/flag.txt"], "mode": "response_in_state", "case_sensitive": False}
    )
    assert scorer._mode is SandboxStateMatchMode.RESPONSE_IN_STATE
    with pytest.raises(ValueError, match="mode"):
        SandboxStateMatchScorer.from_config({"argv": ["true"], "mode": "fuzzy"})


def test_sandbox_command_scorer_requires_exactly_one_of_argv_or_shell_script() -> None:
    with pytest.raises(ValueError, match="Exactly one"):
        SandboxCommandScorer()
    with pytest.raises(ValueError, match="Exactly one"):
        SandboxCommandScorer(argv=("echo",), shell_script="echo hi")


def test_sandbox_command_scorer_from_config_coerces_argv_to_str_tuple() -> None:
    scorer = SandboxCommandScorer.from_config({"argv": ["echo", "hi"]})
    assert scorer._argv == ("echo", "hi")


async def test_result_only_scorer_adapter_ignores_session_and_delegates() -> None:
    scores = [
        Score(
            score_value="True",
            score_type="true_false",
            score_category=[],
            score_rationale="",
            message_piece_id=uuid.uuid4(),
            objective="finish",
        )
    ]
    stub = ExistingScorerStub(scores=scores)
    adapter = ResultOnlyScorerAdapter(scorer=stub)  # type: ignore[arg-type]
    result = _result()
    returned = await adapter.score_async(result=result, objective="finish", session=object())  # type: ignore[arg-type]
    assert returned == scores
    assert stub.calls == [(result, "finish")]
