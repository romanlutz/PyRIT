# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline raw-forwarder fixtures through the real native workflow and isolated evidence store."""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager
from dataclasses import FrozenInstanceError, replace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from pyrit.executor.workflow.copilot_sdk_evidence import CopilotModelResponseOrigin
from pyrit.executor.workflow.copilot_sdk_transport import (
    CopilotSdkModelRequest,
    CopilotSdkModelResponse,
    CopilotSdkModelTransport,
)
from pyrit.executor.workflow.native_cyber_eval import NativeCyberEvaluation
from pyrit.models.native_cyber import NativeCyberRequest
from tests.unit.executor.workflow.test_copilot_sdk_evidence import (
    UnifiedBinding,
    raw_bytes_async,
)
from tests.unit.executor.workflow.test_copilot_sdk_evidence import (
    no_external_io_async as no_external_io_async,
)
from tests.unit.prompt_target.target.test_native_agent_target import EventFixture, event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator
    from pathlib import Path

    from pyrit.executor.workflow.copilot_sdk_evidence import CopilotSdkModelObserver
    from pyrit.executor.workflow.native_cyber_eval import NativeCyberRuntime
    from pyrit.memory import SQLiteMemory
    from pyrit.models.native_cyber import NativeAgentEvidence, NativeCyberJudgment
    from pyrit.models.native_cyber_evidence import NativeCyberRawWrite

pytestmark = pytest.mark.usefixtures("patch_central_database", "no_external_io_async")


async def assert_body_retained_async(*, memory: SQLiteMemory, request: CopilotSdkModelRequest, body: bytes) -> None:
    store = memory.native_cyber_evidence
    episode = await asyncio.to_thread(store.get_episode, run_id=request.identity.run_id)
    observed = episode.events[-1]
    assert observed.observed_event_id == request.request_id and observed.stream_offset is not None
    stream = next(value for value in episode.raw_streams if str(value.stream_id) == observed.observed_stream_id)
    chunks = await asyncio.to_thread(
        store.read_raw_chunks, run_id=request.identity.run_id, stream_id=stream.stream_id, allow_sensitive=True
    )
    raw = b"".join(chunk.data for chunk in chunks)
    assert raw[observed.stream_offset : observed.stream_offset + len(body)] == body


class PersistedForwarderFixture:
    """One fake already-admitted route; no network, SDK, credentials or real provider."""

    BODY = b'{ "id": "inert-provider", "output": "synthetic raw response", "extension": [1,2] }\n'
    HEADERS = (("Content-Type", "application/json"), ("X-Fixture", "metadata-only"), ("X-Fixture", "second"))

    def __init__(self, memory: SQLiteMemory) -> None:
        self.memory = memory
        self.calls: list[tuple[CopilotSdkModelRequest, float, int]] = []
        self.responses: list[CopilotSdkModelResponse] = []
        self.host_only_marker = "inert-host-credential-marker-never-captured"

    async def forward_async(
        self, *, request: CopilotSdkModelRequest, deadline: float, max_response_bytes: int
    ) -> CopilotSdkModelResponse:
        self.calls.append((request, deadline, max_response_bytes))
        episode = await asyncio.to_thread(self.memory.native_cyber_evidence.get_episode, run_id=request.identity.run_id)
        observed = episode.events[-1]
        assert observed.event_type == "ghcp.model.request" and observed.observed_event_id == request.request_id
        await assert_body_retained_async(memory=self.memory, request=request, body=request.body)
        assert not any(
            value.event_type == "ghcp.model.response" and value.observed_event_id == request.request_id
            for value in episode.events
        )
        response = CopilotSdkModelResponse(
            identity=request.identity,
            request_id=request.request_id,
            body=self.BODY,
            status_code=200,
            origin=CopilotModelResponseOrigin.PROVIDER,
            complete=True,
            headers=self.HEADERS,
        )
        self.responses.append(response)
        return response


class TransportBindingFixture(UnifiedBinding):
    """Reuse the existing SDK/tool/lifecycle fixtures, replacing only their host exchange."""

    def __init__(self, memory: SQLiteMemory) -> None:
        super().__init__(memory=memory, steps=True)
        self.forwarder = PersistedForwarderFixture(memory)
        self.transports: list[CopilotSdkModelTransport] = []
        self.requests: list[CopilotSdkModelRequest] = []
        self.results: list[CopilotSdkModelResponse] = []
        self.errors: list[BaseException] = []
        self.deadlines: list[float] = []

    def bind_model_observer(self, *, runtime: NativeCyberRuntime, observer: CopilotSdkModelObserver) -> None:
        super().bind_model_observer(runtime=runtime, observer=observer)
        self.transports.append(CopilotSdkModelTransport(observer=observer, forwarder=self.forwarder))

    async def exchange_async(self, *, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        return await self.transports[-1].forward_async(request=request, deadline=deadline)

    async def deliver_async(self, prompt: str, *, timeout: float) -> object:
        sdk = self.runtime.sdk
        sdk.prompts.append(prompt)
        ordinal = len(sdk.prompts)
        sdk.handler(EventFixture(event("user.message", {"content": prompt}, sessionId=sdk.session_id)))
        body = (
            b'{ "model":"synthetic", "prompt_cache_key":"keep-exactly", "extension":{"opaque":[1,true]}, '
            b'"stream":false, "input":' + json.dumps(prompt).encode() + b" }\n"
        )
        request = CopilotSdkModelRequest(
            identity=self.transports[-1].identity, request_id=f"wire-{ordinal}", body=body, streaming=False
        )
        self.requests.append(request)
        deadline = asyncio.get_running_loop().time() + timeout
        self.deadlines.append(deadline)
        try:
            response = await self.exchange_async(request=request, deadline=deadline)
        except (Exception, asyncio.CancelledError) as error:
            self.errors.append(error)
            raise
        episode = await asyncio.to_thread(sdk.memory.native_cyber_evidence.get_episode, run_id=request.identity.run_id)
        assert episode.events[-1].event_type.endswith("response")
        await assert_body_retained_async(memory=sdk.memory, request=request, body=response.body)
        self.results.append(response)
        for source in sdk.turns[ordinal - 1]:
            sdk.handler(EventFixture(source))
        return None


@pytest.fixture
def transport_binding(sqlite_instance: SQLiteMemory) -> Iterator[TransportBindingFixture]:
    binding = TransportBindingFixture(sqlite_instance)
    with patch.object(binding.runtime.sdk, "send_and_wait", side_effect=binding.deliver_async):
        yield binding


@pytest.mark.parametrize("turns", [1, 2])
async def test_transport_durable_original_bytes_to_one_score_after_owned_cleanup_async(
    *, tmp_path: Path, sqlite_instance: SQLiteMemory, transport_binding: TransportBindingFixture, turns: int
) -> None:
    binding = transport_binding
    run = NativeCyberEvaluation(
        binding=binding,
        request=NativeCyberRequest(instruction="fixture one", operator_steps=turns == 2),
        directory=tmp_path,
    )
    original_grade = binding.runtime.grade_async

    async def grade_async(*, evidence: NativeAgentEvidence) -> NativeCyberJudgment:
        episode = await asyncio.to_thread(run._evidence_store.get_episode, run_id=run.run_id)
        assert episode.report_content_id is None and episode.score_id is None
        assert all(turn.source_complete for turn in episode.turns)
        assert len(episode.turns) == turns and binding.runtime.sdk.disconnected and not binding.runtime.closed
        return await original_grade(evidence=evidence)

    with patch.object(binding.runtime, "grade_async", side_effect=grade_async):
        await run.start_async()
        if turns == 2:
            assert run.report is None and run.score is None
            await run.step_async("fixture two")
            await run.finish_async()
    assert run.status == "completed" and run.score.get_value() == 0.75
    assert binding.order == ["open", "grade", "close"] and binding.runtime.grade_count == 1
    assert len(binding.forwarder.calls) == len(binding.results) == len(binding.transports) == turns
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert episode.coverage_complete and episode.score_id == run.score.id
    assert episode.report_content_id == run.score.scorable.content_id
    assert len(episode.raw_streams) == 5 * turns and len(episode.events) == 9 * turns
    assert [item.sequence for item in episode.events] == list(range(1, 9 * turns + 1))
    for index, (request, deadline, limit) in enumerate(binding.forwarder.calls):
        assert request is binding.requests[index] and deadline == binding.deadlines[index]
        assert limit == run._evidence_store.MAX_APPEND_BYTES
        assert binding.results[index] is binding.forwarder.responses[index]
        assert binding.results[index].headers == PersistedForwarderFixture.HEADERS
        for source_id, body in (
            ("ghcp.model.requests", request.body),
            ("ghcp.model.responses", binding.results[index].body),
        ):
            assert (
                await raw_bytes_async(memory=sqlite_instance, run_id=run.run_id, source_id=source_id, turn=index + 1)
                == body
            )
        turn = episode.turns[index]
        assert turn.request_piece_ids and turn.response_piece_ids
        assert turn.tool_request_piece_ids and turn.tool_result_piece_ids
    assert len(await asyncio.to_thread(sqlite_instance.get_scores, score_type="float_scale")) == 1
    for stream in episode.raw_streams:
        chunks = await asyncio.to_thread(
            run._evidence_store.read_raw_chunks, run_id=run.run_id, stream_id=stream.stream_id, allow_sensitive=True
        )
        raw = b"".join(chunk.data for chunk in chunks)
        assert binding.forwarder.host_only_marker.encode() not in raw and b"metadata-only" not in raw
    public = run.report.model_dump_json()
    assert binding.forwarder.host_only_marker not in public and "keep-exactly" not in public
    assert "synthetic raw response" not in repr(binding.results[0])
    assert "metadata-only" not in repr(binding.results[0])
    assert "keep-exactly" not in repr(binding.requests[0])
    assert binding.forwarder.host_only_marker not in repr(binding.transports[0])
    for attribute in ("observer", "forwarder"):
        with pytest.raises(FrozenInstanceError):
            setattr(binding.transports[0], attribute, getattr(binding.transports[0], attribute))
    await run.finish_async()
    assert binding.runtime.grade_count == 1


@pytest.mark.parametrize(
    ("origin", "status", "complete"),
    [
        (CopilotModelResponseOrigin.HOST, 200, True),
        (CopilotModelResponseOrigin.SYNTHETIC, 200, True),
        (CopilotModelResponseOrigin.PROVIDER, 503, True),
        (CopilotModelResponseOrigin.PROVIDER, 200, False),
    ],
)
async def test_transport_never_upgrades_host_synthetic_error_or_incomplete_response_async(
    *,
    tmp_path: Path,
    transport_binding: TransportBindingFixture,
    origin: CopilotModelResponseOrigin,
    status: int,
    complete: bool,
) -> None:
    binding = transport_binding
    original = binding.forwarder.forward_async

    async def forward_async(**kwargs: object) -> CopilotSdkModelResponse:
        return replace(await original(**kwargs), origin=origin, status_code=status, complete=complete)

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding.forwarder, "forward_async", side_effect=forward_async):
        await run.start_async()
    assert binding.runtime.grade_count == 0 and binding.close_count == 1 and run.score.is_undetermined
    assert len(binding.forwarder.calls) == 1 and len(binding.results) == int(complete)
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    response = next(item for item in episode.events if item.event_type.endswith("response"))
    payload = run.report.agent.events[response.sequence - 1].payload
    assert payload["origin"] == origin.value and payload["complete"] == complete
    assert (response.source.value == "model") == (origin is CopilotModelResponseOrigin.PROVIDER)
    if not complete:
        assert isinstance(binding.errors[0], RuntimeError)


@pytest.mark.parametrize("origin", [CopilotModelResponseOrigin.HOST, CopilotModelResponseOrigin.SYNTHETIC])
async def test_explicit_caller_rejection_is_retained_without_forwarding_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, origin: CopilotModelResponseOrigin
) -> None:
    binding = transport_binding

    async def reject_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        response = CopilotSdkModelResponse(
            identity=request.identity,
            request_id=request.request_id,
            body=b'{"error":"inert caller rejection"}',
            status_code=403,
            origin=origin,
            complete=True,
        )
        return await binding.transports[-1].record_rejection_async(
            request=request, response=response, deadline=deadline
        )

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding, "exchange_async", side_effect=reject_async):
        await run.start_async()
    assert not binding.forwarder.calls and binding.results[0].status_code == 403
    assert run.score.is_undetermined and binding.runtime.grade_count == 0
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert not any(item.event_type in {"ghcp.model.request", "ghcp.model.response"} for item in episode.events)
    assert any(item.event_type == "ghcp.model.rejected_request" for item in episode.events)


@pytest.mark.parametrize("part", ["run_id", "session_id", "turn_id", "turn_index", "streaming"])
async def test_wrong_request_identity_or_streaming_never_invokes_forwarder_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, part: str
) -> None:
    binding = transport_binding
    original = binding.exchange_async

    async def exchange_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        changed = (
            replace(request, streaming=True)
            if part == "streaming"
            else replace(
                request, identity=replace(request.identity, **{part: 99 if part == "turn_index" else "foreign"})
            )
        )
        return await original(request=changed, deadline=deadline)

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding, "exchange_async", side_effect=exchange_async):
        await run.start_async()
    assert not binding.forwarder.calls and isinstance(binding.errors[0], ValueError)
    assert run.score.is_undetermined and binding.runtime.grade_count == 0 and binding.close_count == 1


@pytest.mark.parametrize("defect", ["request-id", "session", "oversize", "empty", "origin", "headers"])
async def test_invalid_original_response_cannot_return_or_reach_grading_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, defect: str
) -> None:
    binding = transport_binding
    original = binding.forwarder.forward_async

    async def forward_async(**kwargs: object) -> CopilotSdkModelResponse:
        response = await original(**kwargs)
        if defect == "request-id":
            return replace(response, request_id="foreign")
        if defect == "session":
            return replace(response, identity=replace(response.identity, session_id="foreign"))
        if defect == "origin":
            return replace(response, origin="provider")
        if defect == "headers":
            return replace(response, headers=[("X-Fixture", "mutable")])
        return replace(response, body=b"x" * (kwargs["max_response_bytes"] + 1) if defect == "oversize" else b"")

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding.forwarder, "forward_async", side_effect=forward_async):
        await run.start_async()
    assert len(binding.forwarder.calls) == 1 and not binding.results and isinstance(binding.errors[0], ValueError)
    assert binding.runtime.grade_count == 0 and run.score.is_undetermined


@pytest.mark.parametrize("part", ["request", "response"])
async def test_exact_capture_byte_limit_is_not_silently_lowered_or_rewritten_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, part: str
) -> None:
    binding = transport_binding
    original_exchange, original_forward = binding.exchange_async, binding.forwarder.forward_async

    async def exchange_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        body = b"x" * binding.transports[-1].observer.max_body_bytes if part == "request" else request.body
        return await original_exchange(request=replace(request, body=body), deadline=deadline)

    async def forward_async(**kwargs: object) -> CopilotSdkModelResponse:
        response = await original_forward(**kwargs)
        return replace(response, body=b"x" * kwargs["max_response_bytes"]) if part == "response" else response

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with (
        patch.object(binding, "exchange_async", side_effect=exchange_async),
        patch.object(binding.forwarder, "forward_async", side_effect=forward_async),
    ):
        await run.start_async()
    assert run.status == "completed" and binding.runtime.grade_count == 1
    body = binding.forwarder.calls[0][0].body if part == "request" else binding.results[0].body
    assert len(body) == binding.transports[0].observer.max_body_bytes


@pytest.mark.parametrize(
    ("origin", "status"), [(CopilotModelResponseOrigin.PROVIDER, 403), (CopilotModelResponseOrigin.HOST, 200)]
)
async def test_rejection_cannot_be_relabelled_as_provider_or_success_async(
    *,
    tmp_path: Path,
    transport_binding: TransportBindingFixture,
    origin: CopilotModelResponseOrigin,
    status: int,
) -> None:
    binding = transport_binding

    async def reject_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        return await binding.transports[-1].record_rejection_async(
            request=request,
            response=CopilotSdkModelResponse(
                identity=request.identity,
                request_id=request.request_id,
                body=b"caller rejection fixture",
                status_code=status,
                origin=origin,
                complete=True,
            ),
            deadline=deadline,
        )

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding, "exchange_async", side_effect=reject_async):
        await run.start_async()
    assert not binding.forwarder.calls and not binding.results and isinstance(binding.errors[0], ValueError)
    assert binding.runtime.grade_count == 0 and run.score.is_undetermined


@pytest.mark.parametrize("defect", ["oversize", "empty", "duplicate-id", "stale-observer"])
async def test_bad_request_or_prior_turn_reuse_blocks_dispatch_without_retries_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, defect: str
) -> None:
    binding = transport_binding
    repeated = defect in {"duplicate-id", "stale-observer"}
    run = NativeCyberEvaluation(
        binding=binding,
        request=NativeCyberRequest(instruction="first", operator_steps=repeated),
        directory=tmp_path,
    )
    if repeated:
        await run.start_async()
        assert len(binding.forwarder.calls) == 1
    original = binding.exchange_async

    async def exchange_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        if defect == "stale-observer":
            previous = binding.transports[0]
            return await previous.forward_async(request=replace(request, identity=previous.identity), deadline=deadline)
        changed = (
            replace(request, request_id=binding.requests[0].request_id)
            if repeated
            else replace(
                request,
                body=b"x" * (binding.transports[-1].observer.max_body_bytes + 1) if defect == "oversize" else b"",
            )
        )
        return await original(request=changed, deadline=deadline)

    with patch.object(binding, "exchange_async", side_effect=exchange_async):
        if repeated:
            await run.step_async("second")
        else:
            await run.start_async()
    assert len(binding.forwarder.calls) == int(repeated) and len(binding.results) == int(repeated)
    assert binding.runtime.grade_count == 0 and binding.close_count == 1 and run.score.is_undetermined
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert len(episode.turns) == 1 + int(repeated)
    assert [value.sequence for value in episode.events] == list(range(1, len(episode.events) + 1))


@pytest.mark.parametrize("exception_type", [OSError, TimeoutError, asyncio.CancelledError])
async def test_forwarder_exception_propagates_once_without_invented_response_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, exception_type: type[BaseException]
) -> None:
    binding = transport_binding
    failure = exception_type("Inert upstream failure")
    original = binding.forwarder.forward_async

    async def forward_async(**kwargs: object) -> CopilotSdkModelResponse:
        await original(**kwargs)
        raise failure

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding.forwarder, "forward_async", side_effect=forward_async):
        if exception_type is asyncio.CancelledError:
            with pytest.raises(asyncio.CancelledError):
                await run.start_async()
        else:
            await run.start_async()
    assert len(binding.forwarder.calls) == 1 and not binding.results
    assert isinstance(binding.errors[0], exception_type)
    if exception_type is not asyncio.CancelledError:
        assert binding.errors[0] is failure
    assert run.score.is_undetermined and binding.runtime.grade_count == 0 and binding.close_count == 1
    episode = await asyncio.to_thread(run._evidence_store.get_finalized_episode, run_id=run.run_id)
    assert not any(value.event_type.endswith("response") for value in episode.events)
    assert any("forwarding" in gap for gap in episode.gaps)


@pytest.mark.parametrize("part", ["request", "response", "response-receipt"])
async def test_capture_failure_blocks_forward_or_return_without_retry_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, part: str
) -> None:
    binding = transport_binding
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original = run._evidence_store.append_raw
    failed = 0

    def append_raw(**kwargs: object) -> NativeCyberRawWrite:
        nonlocal failed
        expected = binding.requests[0].body if part == "request" else PersistedForwarderFixture.BODY
        if kwargs["data"] != expected:
            return original(**kwargs)
        failed += 1
        if part == "response-receipt":
            return original(**kwargs).model_copy(update={"stored_bytes": 1})
        raise OSError("Inert evidence writer failed")

    with patch.object(run._evidence_store, "append_raw", side_effect=append_raw):
        await run.start_async()
    assert failed == 1 and len(binding.forwarder.calls) == int(part != "request") and not binding.results
    assert run.score.is_undetermined and binding.runtime.grade_count == 0 and binding.close_count == 1


@pytest.mark.parametrize("deadline", [0.0, float("inf"), float("nan"), True])
async def test_invalid_or_expired_caller_deadline_blocks_all_forwarding_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, deadline: float
) -> None:
    binding = transport_binding
    original = binding.exchange_async

    async def exchange_async(*, request: CopilotSdkModelRequest, **kwargs: object) -> CopilotSdkModelResponse:
        return await original(request=request, deadline=deadline)

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding, "exchange_async", side_effect=exchange_async):
        await run.start_async()
    assert not binding.forwarder.calls and not binding.results and binding.errors
    assert run.score.is_undetermined and binding.runtime.grade_count == 0


@pytest.mark.parametrize("part", ["request", "response"])
async def test_one_absolute_deadline_includes_durable_capture_and_joins_writes_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, part: str
) -> None:
    binding = transport_binding
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original_exchange, original_append = binding.exchange_async, run._evidence_store.append_raw
    writing, release, finished = threading.Event(), threading.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    caller_deadlines: list[float] = []

    async def exchange_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        deadline = min(deadline, loop.time() + 1.0)
        caller_deadlines.append(deadline)
        timer = loop.call_at(deadline, release.set)
        try:
            return await original_exchange(request=request, deadline=deadline)
        finally:
            timer.cancel()
            release.set()

    def append_raw(**kwargs: object) -> NativeCyberRawWrite:
        expected = binding.requests[0].body if part == "request" else PersistedForwarderFixture.BODY
        if kwargs["data"] != expected:
            return original_append(**kwargs)
        writing.set()
        assert release.wait(timeout=5)
        result = original_append(**kwargs)
        finished.set()
        return result

    with (
        patch.object(binding, "exchange_async", side_effect=exchange_async),
        patch.object(run._evidence_store, "append_raw", side_effect=append_raw),
    ):
        await run.start_async()
    assert writing.is_set() and finished.is_set() and isinstance(binding.errors[0], TimeoutError)
    assert len(binding.forwarder.calls) == int(part == "response") and not binding.results
    if binding.forwarder.calls:
        assert binding.forwarder.calls[0][1] == caller_deadlines[0]
    assert run.score.is_undetermined and binding.runtime.grade_count == 0 and binding.close_count == 1


@pytest.mark.parametrize("timeout", [False, True])
async def test_cancel_or_deadline_cancels_and_joins_owned_forwarding_task_before_cleanup_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, timeout: bool
) -> None:
    binding = transport_binding
    original_forward, original_exchange = binding.forwarder.forward_async, binding.exchange_async
    started, cleaning, release, finished = (asyncio.Event() for _ in range(4))
    workers: list[asyncio.Task[object]] = []

    async def forward_async(**kwargs: object) -> CopilotSdkModelResponse:
        await original_forward(**kwargs)
        worker = asyncio.current_task()
        assert worker is not None
        workers.append(worker)
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            finished.set()
        raise AssertionError("Inert forwarder must not complete normally")

    async def exchange_async(*, request: CopilotSdkModelRequest, deadline: float) -> CopilotSdkModelResponse:
        return await original_exchange(
            request=request, deadline=min(deadline, asyncio.get_running_loop().time() + 1) if timeout else deadline
        )

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with (
        patch.object(binding.forwarder, "forward_async", side_effect=forward_async),
        patch.object(binding, "exchange_async", side_effect=exchange_async),
    ):
        running = asyncio.create_task(run.start_async())
        try:
            await asyncio.wait_for(started.wait(), timeout=3)
            if not timeout:
                running.cancel()
            await asyncio.wait_for(cleaning.wait(), timeout=3)
            if not timeout:
                running.cancel()
            await asyncio.sleep(0)
            assert not running.done() and not workers[0].done()
            assert not binding.runtime.closed and run.score is None and run.report is None
        finally:
            release.set()
        if timeout:
            await running
            assert isinstance(binding.errors[0], TimeoutError)
        else:
            with pytest.raises(asyncio.CancelledError):
                await running
    assert finished.is_set() and workers[0].done()
    assert len(binding.forwarder.calls) == 1 and not binding.results
    assert run.score.is_undetermined and binding.runtime.grade_count == 0 and binding.close_count == 1


async def test_forwarder_suppressing_cancellation_cannot_return_a_complete_response_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture
) -> None:
    binding = transport_binding
    original = binding.forwarder.forward_async
    started = asyncio.Event()
    workers: list[asyncio.Task[object]] = []

    async def forward_async(**kwargs: object) -> CopilotSdkModelResponse:
        response = await original(**kwargs)
        worker = asyncio.current_task()
        assert worker is not None
        workers.append(worker)
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return response
        raise AssertionError("Inert forwarder should have been cancelled")

    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    with patch.object(binding.forwarder, "forward_async", side_effect=forward_async):
        running = asyncio.create_task(run.start_async())
        await asyncio.wait_for(started.wait(), timeout=3)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    assert workers[0].done() and not workers[0].cancelled()
    assert not binding.results and run.score.is_undetermined and binding.runtime.grade_count == 0
    assert binding.close_count == 1


@pytest.mark.parametrize("failure", ["quiescence", "cleanup"])
async def test_adapter_preserves_existing_quiescence_and_cleanup_score_gates_async(
    *, tmp_path: Path, transport_binding: TransportBindingFixture, failure: str
) -> None:
    binding = transport_binding
    run = NativeCyberEvaluation(binding=binding, request=NativeCyberRequest(instruction="fixture"), directory=tmp_path)
    original_context = binding.open_runtime

    @asynccontextmanager
    async def failing_context_async(*, run_id: str, request: NativeCyberRequest) -> AsyncIterator[NativeCyberRuntime]:
        async with original_context(run_id=run_id, request=request) as runtime:
            yield runtime
        raise OSError("Inert cleanup failure")

    if failure == "quiescence":
        with patch.object(
            binding.runtime.session, "quiesce_async", side_effect=RuntimeError("Inert quiescence failure")
        ):
            await run.start_async()
    else:
        with patch.object(binding, "open_runtime", side_effect=failing_context_async):
            await run.start_async()
    assert len(binding.forwarder.calls) == len(binding.results) == 1
    assert binding.runtime.grade_count == int(failure == "cleanup")
    assert binding.close_count == 1 and run.score.is_undetermined
