# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark._inspect_native_generate import InspectNativeGenerate
from pyrit.executor.benchmark.inspect_native_task import InspectNativeTaskBridge
from pyrit.memory import CentralMemory
from pyrit.models import ScoreStatus
from tests.unit.executor.benchmark.test_inspect_native_task import OfflineFixtureBinding, _run_fixture_async

pytest.importorskip("inspect_ai")
pytestmark = pytest.mark.usefixtures("patch_central_database")


@pytest.mark.parametrize("task_options", [{"message_limit": 4}, {"turn_limit": 1}, {"token_limit": 1}])
async def test_native_limits_stop_before_second_provider_and_retain_last_tool_async(
    tmp_path: Path, task_options: dict[str, Any]
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    result, requests, _, native_score = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "b", None], task_options=task_options
    )
    assert len(requests) == binding.evaluator.await_count == 1
    assert result.score.get_value() == 0.75
    assert result.report.termination_reason.value == "budget"
    assert result.report.calls[-1].feedback == "OFFLINE grade 0.75\nexact native feedback"
    native_score.assert_awaited_once()
    assert binding.state.output.usage.total_tokens == 12


@pytest.mark.parametrize("task_options", [{"turn_limit": 0}, {"message_limit": 2}, {"token_limit": 0}])
async def test_native_zero_available_budget_never_calls_provider_or_tools_async(
    tmp_path: Path, task_options: dict[str, Any]
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75})
    result, requests, _, _ = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a"], task_options=task_options
    )
    assert requests == []
    binding.evaluator.assert_not_awaited()
    assert result.report.status.value == "no_submission"
    assert result.score.status is ScoreStatus.UNDETERMINED
    assert result.report.termination_reason.value == "budget"


async def test_native_output_token_budget_uses_observed_output_not_total_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    result, requests, _, _ = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "b", None], task_options={"token_limit": "output:3"}
    )
    assert len(requests) == binding.evaluator.await_count == 2
    assert result.score.get_value() == 0.25
    assert binding.state.output.usage.output_tokens == 2


@pytest.mark.parametrize(
    "task_options", [{"token_limit": "(input+output):1"}, {"cost_limit": 1.0}, {"working_limit": 1}]
)
async def test_unenforceable_native_limit_rejected_before_provider_or_evaluator_async(
    tmp_path: Path, task_options: dict[str, Any]
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75})
    with pytest.raises(NotImplementedError, match="External"):
        await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a"], task_options=task_options)
    binding.evaluator.assert_not_awaited()
    assert binding.bridge is not None and binding.bridge._generator.requests == 0


async def test_missing_native_token_usage_stops_before_further_dispatch_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75})
    result, requests, _, native_score = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "a"], task_options={"token_limit": 10}, include_usage=False
    )
    assert len(requests) == 1
    binding.evaluator.assert_not_awaited()
    native_score.assert_not_awaited()
    assert binding.state.output.usage is None
    assert result.report.status.value == "incomplete" and result.score.status is ScoreStatus.UNDETERMINED


async def test_tool_budget_blocks_an_extra_model_generation_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    result, requests, _, _ = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a", "b", None], max_tool_calls=1
    )
    assert len(requests) == binding.evaluator.await_count == 1
    assert result.score.get_value() == 0.75
    assert result.report.termination_reason.value == "budget"


@pytest.mark.parametrize(
    "boundary", ["native_generation_request", "native_generation_response", "native_tool_dispatch"]
)
async def test_stop_after_dispatch_preparation_prevents_irreversible_followup_async(
    tmp_path: Path, boundary: str
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    original = InspectRunArtifacts.append_async

    async def stop_async(self: InspectRunArtifacts, *, event: str, data: dict[str, Any]) -> None:
        await original(self, event=event, data=data)
        if event == boundary:
            assert binding.bridge is not None
            binding.bridge._generator.stop()

    with patch.object(InspectRunArtifacts, "append_async", new=stop_async):
        result, requests, _, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", None])
    assert len(requests) == (0 if boundary == "native_generation_request" else 1)
    binding.evaluator.assert_not_awaited()
    assert result.score.status is ScoreStatus.UNDETERMINED


async def test_task_completion_after_tool_blocks_an_extra_generation_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    original = InspectRunArtifacts.append_async

    async def complete_async(self: InspectRunArtifacts, *, event: str, data: dict[str, Any]) -> None:
        await original(self, event=event, data=data)
        if event == "native_tool_return":
            binding.state.completed = True

    with patch.object(InspectRunArtifacts, "append_async", new=complete_async):
        result, requests, _, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", None])
    assert len(requests) == binding.evaluator.await_count == 1
    assert result.score.get_value() == 0.75


@pytest.mark.parametrize("boundary", ["native_generation_request", "native_tool_dispatch"])
async def test_completion_during_awaited_dispatch_preparation_retains_prior_grade_async(
    tmp_path: Path, boundary: str
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    original = InspectRunArtifacts.append_async

    async def complete_async(self: InspectRunArtifacts, *, event: str, data: dict[str, Any]) -> None:
        await original(self, event=event, data=data)
        second = data.get("ordinal") == 2 or data.get("provider_call_id") == "offline-call-2"
        if event == boundary and second:
            binding.state.completed = True

    with patch.object(InspectRunArtifacts, "append_async", new=complete_async):
        result, requests, artifacts, _ = await _run_fixture_async(
            tmp_path=tmp_path, binding=binding, commands=["a", "b", None]
        )
    assert binding.evaluator.await_count == 1
    assert len(requests) == (1 if boundary == "native_generation_request" else 2)
    assert result.score.get_value() == 0.75
    assert len(result.report.calls) == 1
    if boundary == "native_tool_dispatch":
        assert artifacts.manifest["tool_calls"][-1]["status"] == "not_dispatched"


async def test_completion_inside_native_tool_execution_is_checked_before_callback_async(tmp_path: Path) -> None:
    import inspect_ai.model

    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    original = inspect_ai.model.execute_tools
    invocations = 0

    async def complete_before_callback_async(*args: Any, **kwargs: Any) -> Any:
        nonlocal invocations
        invocations += 1
        await asyncio.sleep(0)
        if invocations == 2:
            binding.state.completed = True
        return await original(*args, **kwargs)

    with patch.object(inspect_ai.model, "execute_tools", new=complete_before_callback_async):
        result, requests, _, _ = await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", None])
    assert len(requests) == 2
    assert binding.evaluator.await_count == 1
    assert result.score.get_value() == 0.75
    assert len(result.report.calls) == 1


@pytest.mark.parametrize("size", [4000, 8210])
async def test_feedback_is_exact_or_explicitly_incomplete_never_native_truncation_async(
    tmp_path: Path, size: int
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75})
    original_submit = binding.submit_async
    feedback = "OFFLINE exact\n" + ("x" * size)

    async def submit_async(artifact_ref: str) -> str:
        await original_submit(artifact_ref)
        binding.report["submissions"][-1]["feedback"] = feedback
        return feedback

    with patch.object(binding, "submit_async", new=submit_async):
        result, requests, artifacts, native_score = await _run_fixture_async(
            tmp_path=tmp_path, binding=binding, commands=["a", None], max_response_bytes=4096
        )
    binding.evaluator.assert_awaited_once()
    assert result.report.calls[0].feedback == feedback
    if size < 4096:
        assert len(requests) == 2
        assert requests[1]["input"][-1]["output"] == feedback
        assert result.score.get_value() == 0.75
    else:
        assert len(requests) == 1
        assert result.report.status.value == "incomplete"
        assert result.report.report.last_valid_grade == 0.75
        assert result.score.status is ScoreStatus.UNDETERMINED
        native_score.assert_not_awaited()
        assert artifacts.manifest["tool_calls"][0]["feedback_limit"]["forwarded_to_provider"] is False


async def test_large_utf8_feedback_below_explicit_bound_survives_native_default_cap_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "success": 1.0})
    original = binding.submit_async
    feedback = ("OFFLINE/SIMULATED long exact text\n" * 1024) + "\u03bb\n"

    async def submit_async(artifact_ref: str) -> str:
        result = await original(artifact_ref)
        if artifact_ref == "a":
            binding.report["submissions"][-1]["feedback"] = feedback
            return feedback
        return result

    with patch.object(binding, "submit_async", new=submit_async):
        result, requests, _, _ = await _run_fixture_async(
            tmp_path=tmp_path, binding=binding, commands=["a", "success", None], max_tool_output_bytes=65536
        )
    assert 16384 < len(feedback.encode("utf-8")) < 65536
    assert len(requests) == 2
    assert requests[1]["input"][-1]["output"] == feedback
    assert result.report.calls[0].feedback == feedback
    assert result.score.get_value() == 1.0


async def test_separate_tool_cap_does_not_change_provider_response_cap_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75})
    result, requests, _, _ = await _run_fixture_async(
        tmp_path=tmp_path, binding=binding, commands=["a", None], max_response_bytes=65536, max_tool_output_bytes=16
    )
    assert len(requests) == binding.evaluator.await_count == 1
    assert result.report.calls[0].feedback == "OFFLINE grade 0.75\nexact native feedback"
    assert result.score.status is ScoreStatus.UNDETERMINED


async def test_repeated_cancellation_joins_the_same_owned_message_write_before_return_async(tmp_path: Path) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"
    writing, joining, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = InspectNativeGenerate._persist_message_async
    original_finish = InspectNativeTaskBridge._finish_environment_async

    async def held_write_async(message: Any) -> None:
        if message.api_role == "tool" and json.loads(message.get_value())["call_id"] == "offline-call-2":
            writing.set()
            await release.wait()
        await original(message)

    async def observe_join_async(self: InspectNativeTaskBridge, **kwargs: Any) -> None:
        joining.set()
        await original_finish(self, **kwargs)

    with (
        patch.object(InspectNativeGenerate, "_persist_message_async", new=staticmethod(held_write_async)),
        patch.object(InspectNativeTaskBridge, "_finish_environment_async", new=observe_join_async),
    ):
        pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"]))
        try:
            await asyncio.wait_for(writing.wait(), timeout=20)
            await asyncio.wait_for(joining.wait(), timeout=20)
            for _ in range(3):
                pending.cancel("OFFLINE repeat cancellation while joining")
                try:
                    await asyncio.wait_for(asyncio.shield(pending), timeout=0.1)
                except TimeoutError:
                    pass
                except asyncio.CancelledError:
                    pytest.fail("The caller returned before its acquired tool message was durably retained.")
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError, match="callback-return boundary"):
            await pending
    assert binding.bridge is not None and binding.bridge.last_result is not None
    result = binding.bridge.last_result
    pieces = CentralMemory.get_memory_instance().get_message_pieces(conversation_id=result.report.conversation_id)
    outputs = [
        json.loads(piece.original_value) for piece in pieces if piece.original_value_data_type == "function_call_output"
    ]
    assert [item["call_id"] for item in outputs] == ["offline-call-1", "offline-call-2"]


async def test_cancel_result_recovers_identity_matched_finalized_native_log_async(tmp_path: Path) -> None:
    from inspect_ai.log import list_eval_logs_async, read_eval_log_async

    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"
    with pytest.raises(asyncio.CancelledError):
        await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"])
    assert binding.bridge is not None and binding.bridge.last_result is not None
    result = binding.bridge.last_result
    infos = await list_eval_logs_async(str(tmp_path / "run" / "native"), recursive=False)
    assert len(infos) == 1
    log = await read_eval_log_async(infos[0])
    assert log.stats.completed_at and log.status != "started"
    assert result.native_log == tmp_path / "run" / "native" / Path(log.location).name
    assert result.native_log.exists()
    assert result.native_eval_id == log.eval.eval_id
    assert result.native_sample_uuid == list(log.samples or [])[0].uuid


async def test_pending_write_deadline_records_uncertainty_without_resetting_on_repeat_cancel_async(
    tmp_path: Path,
) -> None:
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"
    held, release, written = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original = InspectNativeGenerate._persist_message_async

    async def hold_async(message: Any) -> None:
        if message.api_role == "tool" and json.loads(message.get_value())["call_id"] == "offline-call-2":
            held.set()
            await release.wait()
            try:
                await original(message)
            finally:
                written.set()
        else:
            await original(message)

    with (
        patch.object(InspectNativeGenerate, "_persist_message_async", new=staticmethod(hold_async)),
        patch.object(InspectNativeTaskBridge, "_CANCELLATION_SETTLE_SECONDS", 0.4),
    ):
        pending = asyncio.create_task(_run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"]))
        try:
            await asyncio.wait_for(held.wait(), timeout=20)
            started = asyncio.get_running_loop().time()
            for _ in range(3):
                await asyncio.sleep(0.08)
                pending.cancel("OFFLINE repeated cancellation")
            with pytest.raises(asyncio.CancelledError, match="callback-return boundary") as cancelled:
                await asyncio.wait_for(pending, timeout=2)
            assert asyncio.get_running_loop().time() - started < 1.5
            assert any("finalization" in note for note in cancelled.value.__notes__)
            manifest = json.loads((tmp_path / "run" / "manifest.json").read_text())
            assert manifest["canonical_messages_status"] == "unknown"
            assert manifest["pending_message_ids"]
            assert manifest["remote_stop_claim"] is False
        finally:
            release.set()
            await asyncio.wait_for(written.wait(), timeout=5)


@pytest.mark.parametrize("mismatch", ["run", "attempt", "sample", "epoch", "uuid", "unfinished", "unreadable"])
async def test_cancelled_log_recovery_never_adopts_unrelated_or_unfinished_evidence_async(
    tmp_path: Path, mismatch: str
) -> None:
    import inspect_ai.log

    original = inspect_ai.log.read_eval_log_async
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"

    async def changed_async(*args: Any, **kwargs: Any) -> Any:
        if mismatch == "unreadable":
            raise OSError("OFFLINE finalized log unavailable")
        log = await original(*args, **kwargs)
        if mismatch in ("run", "attempt"):
            assert log.eval.metadata is not None
            log.eval.metadata[f"pyrit_{mismatch}_id"] = "OFFLINE unrelated"
        elif mismatch == "unfinished":
            log.stats.completed_at = ""
        else:
            sample = list(log.samples or [])[0]
            if mismatch == "sample":
                sample.id = "another-case"
            elif mismatch == "epoch":
                sample.epoch = 2
            else:
                sample.uuid = "another-native-sample"
        return log

    with patch.object(inspect_ai.log, "read_eval_log_async", new=changed_async):
        with pytest.raises(asyncio.CancelledError, match="callback-return boundary"):
            await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"])
    assert binding.bridge is not None and binding.bridge.last_result is not None
    result = binding.bridge.last_result
    assert result.native_log is None and result.native_eval_id is None
    assert result.report.report.last_valid_grade == 0.25
    assert result.score.status is ScoreStatus.UNDETERMINED


async def test_cancelled_log_lookup_ignores_unrelated_finalized_log_beside_owned_match_async(tmp_path: Path) -> None:
    import inspect_ai.log

    original_list = inspect_ai.log.list_eval_logs_async
    original_read = inspect_ai.log.read_eval_log_async
    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    binding.cancel_after_return = "b"

    async def with_stale_async(*args: Any, **kwargs: Any) -> Any:
        infos = await original_list(*args, **kwargs)
        assert len(infos) == 1
        stale_path = tmp_path / "run" / "native" / "unrelated.eval"
        return [infos[0].model_copy(update={"name": str(stale_path)}), infos[0]]

    async def read_with_stale_async(path: Any, **kwargs: Any) -> Any:
        if Path(path).name != "unrelated.eval":
            return await original_read(path, **kwargs)
        real = await original_list(str(tmp_path / "run" / "native"), recursive=False)
        stale = await original_read(real[0], **kwargs)
        assert stale.eval.metadata is not None
        stale.eval.metadata["pyrit_attempt_id"] = "different-attempt"
        return stale

    with (
        patch.object(inspect_ai.log, "list_eval_logs_async", new=with_stale_async),
        patch.object(inspect_ai.log, "read_eval_log_async", new=read_with_stale_async),
    ):
        with pytest.raises(asyncio.CancelledError):
            await _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b", "b"])
    assert binding.bridge is not None and binding.bridge.last_result is not None
    result = binding.bridge.last_result
    assert result.native_log is not None and result.native_log.name != "unrelated.eval"
    assert result.native_log.exists()
    assert result.native_eval_id


def test_native_log_recovery_rejects_paths_outside_owned_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="outside"):
        InspectNativeTaskBridge._owned_log_path(
            location=str(tmp_path / "elsewhere.eval"), directory=tmp_path / "native"
        )
    with pytest.raises(ValueError, match="local owned"):
        InspectNativeTaskBridge._owned_log_path(location="https://example.invalid/log.eval", directory=tmp_path)
    with pytest.raises(ValueError, match="remote file host"):
        InspectNativeTaskBridge._owned_log_path(location="file://remote-host/share/log.eval", directory=tmp_path)


@pytest.mark.parametrize("boundary", ["owned_cleanup", "accept_log_read"])
async def test_first_late_cancellation_preserves_finalized_native_references_async(
    tmp_path: Path, boundary: str
) -> None:
    import inspect_ai.log

    binding = OfflineFixtureBinding({"a": 0.75, "b": 0.25})
    reached, release = asyncio.Event(), asyncio.Event()
    accepting = False
    original_read = inspect_ai.log.read_eval_log_async
    original_accept = InspectNativeTaskBridge._accept_async

    async def cleanup_async() -> None:
        if boundary == "owned_cleanup":
            reached.set()
            await release.wait()

    async def accept_async(self: InspectNativeTaskBridge, **kwargs: Any) -> Any:
        nonlocal accepting
        accepting = True
        return await original_accept(self, **kwargs)

    async def read_async(*args: Any, **kwargs: Any) -> Any:
        if boundary == "accept_log_read" and accepting and not reached.is_set():
            reached.set()
            await release.wait()
        return await original_read(*args, **kwargs)

    with (
        patch.object(InspectNativeTaskBridge, "_accept_async", new=accept_async),
        patch.object(inspect_ai.log, "read_eval_log_async", new=read_async),
    ):
        pending = asyncio.create_task(
            _run_fixture_async(tmp_path=tmp_path, binding=binding, commands=["a", "b"], cleanup_callback=cleanup_async)
        )
        try:
            await asyncio.wait_for(reached.wait(), timeout=30)
            pending.cancel(f"OFFLINE first cancellation during {boundary}")
            release.set()
            with pytest.raises(asyncio.CancelledError, match=boundary):
                await pending
        finally:
            release.set()
    assert binding.evaluator.await_count == 2
    assert binding.bridge is not None and binding.bridge.last_result is not None
    result = binding.bridge.last_result
    infos = await inspect_ai.log.list_eval_logs_async(str(tmp_path / "run" / "native"), recursive=False)
    assert len(infos) == 1
    log = await original_read(infos[0], resolve_attachments=True)
    sample = list(log.samples or [])[0]
    assert log.status == "success" and log.stats.completed_at and sample.id == "offline-case" and sample.epoch == 1
    assert result.native_log is not None and result.native_log.exists()
    assert result.native_log.name == Path(log.location).name
    assert result.native_eval_id == log.eval.eval_id and result.native_sample_uuid == sample.uuid
    assert result.report.report.last_valid_grade == 0.25
    assert result.report.status.value == "cancelled" and result.score.status is ScoreStatus.UNDETERMINED
