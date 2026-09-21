# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Manual batches exercised through real domain messages, normalizers, and SQLite."""

import asyncio
import threading
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from pyrit.backend.models.attacks import AddMessageRequest, ConverterConfigurationRequest, MessagePieceRequest
from pyrit.backend.models.message_batches import (
    MessageBatchBranchState,
    MessageBatchRequest,
    MessageBatchState,
    MessageBatchStatus,
    RequestConverterMode,
)
from pyrit.backend.routes import message_batches
from pyrit.backend.services.attack_service import AttackService
from pyrit.backend.services.converter_service import ConverterService
from pyrit.backend.services.manual_send_scheduler import (
    ManualSendConflictError,
    ManualSendQueueFullError,
    ManualSendScheduler,
)
from pyrit.backend.services.multi_send_service import (
    MessageBatchNotFoundError,
    MultiSendService,
    get_multi_send_service,
    shutdown_message_batches_async,
)
from pyrit.backend.services.target_service import TargetService
from pyrit.converter import Converter, ConverterResult
from pyrit.memory import SQLiteMemory
from pyrit.models import (
    AtomicAttackIdentifier,
    AttackIdentifier,
    AttackResult,
    Conversation,
    ConverterIdentifier,
    Message,
    MessagePiece,
    PromptDataType,
    construct_response_from_request,
)
from pyrit.prompt_normalizer import ConverterConfiguration
from unit.mocks import MockPromptTarget


def _request(
    *,
    conversation_id: str,
    count: int = 5,
    mode: RequestConverterMode = RequestConverterMode.SHARED,
    converter_ids: list[str] | None = None,
) -> MessageBatchRequest:
    return MessageBatchRequest(
        target_conversation_id=conversation_id,
        target_registry_name="target",
        pieces=[MessagePieceRequest(original_value="Next prompt")],
        count=count,
        submission_id=str(uuid.uuid4()),
        request_converter_mode=mode,
        request_converter_configurations=(
            [ConverterConfigurationRequest(converter_ids=converter_ids)] if converter_ids else None
        ),
    )


async def _wait_until_async(predicate: Callable[[], bool]) -> None:
    async with asyncio.timeout(10):
        while not predicate():
            await asyncio.sleep(0.001)


async def _wait_batch_async(*, service: MultiSendService, status: MessageBatchStatus) -> MessageBatchStatus:
    def is_finished() -> bool:
        current = service.get_status(attack_result_id=status.attack_result_id, batch_id=status.batch_id)
        return current.state in (MessageBatchState.COMPLETED, MessageBatchState.FAILED)

    await _wait_until_async(is_finished)
    return service.get_status(attack_result_id=status.attack_result_id, batch_id=status.batch_id)


@dataclass(kw_only=True)
class _Harness:
    memory: SQLiteMemory
    service: MultiSendService
    attack_service: AttackService
    scheduler: ManualSendScheduler
    attack: AttackResult
    target: MockPromptTarget
    send: AsyncMock
    target_service: MagicMock
    converter_service: MagicMock


@pytest.fixture
async def harness(sqlite_instance: SQLiteMemory, patch_central_database: MagicMock) -> AsyncIterator[_Harness]:
    target = MockPromptTarget()
    attack = AttackResult(
        conversation_id=str(uuid.uuid4()),
        objective="Manual test",
        atomic_attack_identifier=AtomicAttackIdentifier.build(
            attack_identifier=AttackIdentifier(
                class_name="ManualAttack", class_module="pyrit.backend", objective_target=target.get_identifier()
            )
        ),
    )
    await asyncio.to_thread(sqlite_instance.add_attack_results_to_memory, attack_results=[attack])
    await asyncio.to_thread(
        sqlite_instance.add_conversation_to_memory,
        conversation=Conversation(conversation_id=attack.conversation_id, target_identifier=target.get_identifier()),
    )
    target_service = MagicMock(spec=TargetService)
    target_service.get_target_object.return_value = target
    converter_service = MagicMock(spec=ConverterService)
    converter_service.get_converter_objects_for_ids.return_value = []
    scheduler = ManualSendScheduler(max_concurrency=2, max_operations=12)
    attack_service = AttackService()
    service = MultiSendService(memory=sqlite_instance, attack_service=attack_service, scheduler=scheduler)

    async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
        return [construct_response_from_request(request=message.get_piece(), response_text_pieces=["Reply"])]

    with (
        patch.object(target, "send_prompt_async", new_callable=AsyncMock, side_effect=send_async) as send,
        patch("pyrit.backend.services.multi_send_service.get_target_service", return_value=target_service),
        patch("pyrit.backend.services.attack_service.get_target_service", return_value=target_service),
        patch("pyrit.backend.services.attack_service.get_converter_service", return_value=converter_service),
        patch("pyrit.backend.services.attack_service.get_manual_send_scheduler", return_value=scheduler),
    ):
        yield _Harness(
            memory=sqlite_instance,
            service=service,
            attack_service=attack_service,
            scheduler=scheduler,
            attack=attack,
            target=target,
            send=send,
            target_service=target_service,
            converter_service=converter_service,
        )
        await service.shutdown_async()


def _converter(*, value: str, data_type: PromptDataType = "text") -> MagicMock:
    converter = MagicMock(spec=Converter)
    converter.get_identifier.return_value = ConverterIdentifier(
        class_name="TestConverter", class_module="tests.unit.backend"
    )
    converter.convert_tokens_async = AsyncMock(return_value=ConverterResult(output_text=value, output_type=data_type))
    return converter


@pytest.mark.parametrize("count", [0, 11, -1, True, False, 1.5, 5.0, "5", None])
def test_batch_count_is_a_strict_bounded_integer(count: object) -> None:
    data = _request(conversation_id="source").model_dump()
    data["count"] = count
    with pytest.raises(ValidationError):
        MessageBatchRequest.model_validate(data)


@pytest.mark.parametrize("count", [1, 5, 10])
def test_batch_count_boundaries_and_default_converter_mode(count: int) -> None:
    data = _request(conversation_id="source", count=count).model_dump()
    data.pop("request_converter_mode")
    request = MessageBatchRequest.model_validate(data)
    assert request.count == count
    assert request.request_converter_mode == RequestConverterMode.SHARED


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("role", "assistant"),
        ("role", "system"),
        ("send", False),
        ("pieces", []),
        ("submission_id", ""),
        ("submission_id", " "),
        ("submission_id", "a" * 129),
        ("target_registry_name", None),
        ("target_conversation_id", ""),
        ("request_converter_mode", "unknown"),
    ],
)
def test_batch_request_rejects_invalid_input(field: str, value: object) -> None:
    data = _request(conversation_id="source").model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        MessageBatchRequest.model_validate(data)


@pytest.mark.usefixtures("patch_central_database")
class TestMultiSendService:
    async def test_five_sends_register_four_clones_before_dispatch_and_keep_lineage(self, harness: _Harness) -> None:
        history = [
            MessagePiece(
                role="system", original_value="System", conversation_id=harness.attack.conversation_id, sequence=2
            ),
            MessagePiece(
                role="user", original_value="First", conversation_id=harness.attack.conversation_id, sequence=7
            ),
            MessagePiece(
                role="user", original_value="Second", conversation_id=harness.attack.conversation_id, sequence=7
            ),
            MessagePiece(
                role="simulated_assistant",
                original_value="Context",
                conversation_id=harness.attack.conversation_id,
                sequence=9,
            ),
        ]
        await asyncio.to_thread(harness.memory.add_message_pieces_to_memory, message_pieces=history)
        prefix = await asyncio.to_thread(
            harness.memory.get_conversation_messages, conversation_id=harness.attack.conversation_id
        )
        original_prefix = [message.model_dump() for message in prefix]
        registered_counts: list[int] = []

        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            results = await asyncio.to_thread(
                harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
            )
            registered_counts.append(len(results[0].get_active_conversation_ids()))
            message.get_piece().prompt_metadata["branch_only"] = message.conversation_id
            return [construct_response_from_request(request=message.get_piece(), response_text_pieces=["Reply"])]

        harness.send.side_effect = send_async
        request = _request(conversation_id=harness.attack.conversation_id)
        request.pieces.append(MessagePieceRequest(original_value="Other piece"))
        with patch.object(
            harness.memory, "get_conversation_messages", wraps=harness.memory.get_conversation_messages
        ) as read_snapshot:
            accepted = await harness.service.submit_async(
                attack_result_id=harness.attack.attack_result_id, request=request
            )
            result = await _wait_batch_async(service=harness.service, status=accepted)
            assert read_snapshot.call_count == 1
        assert result.state == MessageBatchState.COMPLETED
        assert harness.send.await_count == 5
        assert registered_counts == [5] * 5
        assert len({branch.conversation_id for branch in result.branches}) == 5
        original_ids: list[uuid.UUID] = []
        for branch in result.branches:
            messages = await asyncio.to_thread(
                harness.memory.get_conversation_messages, conversation_id=branch.conversation_id
            )
            assert [message.sequence for message in messages] == [2, 7, 9, 10, 11]
            assert [len(message.message_pieces) for message in messages] == [1, 2, 1, 2, 1]
            assert [[piece.original_prompt_id for piece in message.message_pieces] for message in messages[:3]] == [
                [piece.original_prompt_id for piece in message.message_pieces] for message in prefix
            ]
            source_metadata = await asyncio.to_thread(
                harness.memory._get_conversation, conversation_id=branch.conversation_id
            )
            assert source_metadata is not None
            assert source_metadata.target_identifier == harness.target.get_identifier()
            if branch.conversation_id == harness.attack.conversation_id:
                assert [message.model_dump() for message in messages[:3]] == original_prefix
                original_ids = [piece.id for piece in messages[3].message_pieces]
                assert [piece.original_prompt_id for piece in messages[3].message_pieces] == original_ids
            else:
                assert [piece.original_prompt_id for piece in messages[3].message_pieces] == original_ids
                assert all(piece.id not in original_ids for piece in messages[3].message_pieces)
            assert len(branch.new_message_piece_ids) == 3

    @pytest.mark.parametrize("system_only", [False, True])
    async def test_empty_and_system_only_histories(self, harness: _Harness, system_only: bool) -> None:
        if system_only:
            await asyncio.to_thread(
                harness.memory.add_message_to_memory,
                request=MessagePiece(
                    role="system", original_value="System", conversation_id=harness.attack.conversation_id
                ).to_message(),
            )
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        for branch in result.branches:
            messages = await asyncio.to_thread(
                harness.memory.get_conversation_messages, conversation_id=branch.conversation_id
            )
            assert [message.api_role for message in messages] == (
                ["system", "user", "assistant"] if system_only else ["user", "assistant"]
            )

    async def test_second_batch_on_a_pruned_branch_adds_only_two_copies(self, harness: _Harness) -> None:
        first = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id),
        )
        first = await _wait_batch_async(service=harness.service, status=first)
        source_id = first.branches[2].conversation_id
        second = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id, request=_request(conversation_id=source_id, count=3)
        )
        second = await _wait_batch_async(service=harness.service, status=second)
        attack = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        assert second.state == MessageBatchState.COMPLETED
        assert second.source_conversation_id == source_id
        assert len(attack[0].get_active_conversation_ids()) == 7
        assert harness.send.await_count == 8
        for branch in second.branches:
            messages = await asyncio.to_thread(
                harness.memory.get_conversation_messages, conversation_id=branch.conversation_id
            )
            assert len(messages) == 4

    @pytest.mark.parametrize(
        ("mode", "calls"), [(RequestConverterMode.SHARED, 1), (RequestConverterMode.PER_BRANCH, 3)]
    )
    async def test_request_conversion_preserves_text_to_media_type_and_identifiers(
        self, harness: _Harness, tmp_path: Path, mode: RequestConverterMode, calls: int
    ) -> None:
        image_path = tmp_path / "converted.png"
        await asyncio.to_thread(image_path.write_bytes, b"test image bytes")
        converter = _converter(value=str(image_path), data_type="image_path")
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(
                conversation_id=harness.attack.conversation_id, count=3, mode=mode, converter_ids=["converter"]
            ),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        assert converter.convert_tokens_async.await_count == calls
        for call in harness.send.await_args_list:
            piece = call.kwargs["message"].get_piece()
            assert piece.original_value == "Next prompt"
            assert piece.original_value_data_type == "text"
            assert piece.converted_value == str(image_path)
            assert piece.converted_value_data_type == "image_path"
            assert [identifier.hash for identifier in piece.converter_identifiers] == [converter.get_identifier().hash]
        assert len({id(call.kwargs["message"]) for call in harness.send.await_args_list}) == 3

    async def test_upload_is_persisted_once_and_client_snapshot_is_not_mutated(self, harness: _Harness) -> None:
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        request.pieces = [MessagePieceRequest(data_type="image_path", original_value="dGVzdA==", mime_type="image/png")]
        with patch.object(
            harness.attack_service,
            "_persist_base64_pieces_async",
            wraps=harness.attack_service._persist_base64_pieces_async,
        ) as persist_media:
            accepted = await harness.service.submit_async(
                attack_result_id=harness.attack.attack_result_id, request=request
            )
            result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        assert persist_media.await_count == 1
        assert request.pieces[0].original_value == "dGVzdA=="
        sent_paths = [call.kwargs["message"].get_piece().original_value for call in harness.send.await_args_list]
        assert len(set(sent_paths)) == 1
        assert sent_paths[0] != request.pieces[0].original_value

    async def test_preconverted_piece_does_not_execute_its_request_converter(self, harness: _Harness) -> None:
        converter = _converter(value="converted")
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        request = _request(conversation_id=harness.attack.conversation_id, count=3, converter_ids=["converter"])
        request.pieces[0].converted_value = "Already converted"
        request.pieces.append(MessagePieceRequest(original_value="Convert this"))
        accepted = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        assert converter.convert_tokens_async.await_count == 1
        for call in harness.send.await_args_list:
            assert call.kwargs["message"].get_values() == ["Already converted", "converted"]

    @pytest.mark.parametrize("mode", [RequestConverterMode.SHARED, RequestConverterMode.PER_BRANCH])
    async def test_multipart_conversion_keeps_pipeline_order_preconverted_values_and_existing_lineage(
        self, harness: _Harness, mode: RequestConverterMode
    ) -> None:
        first, second = _converter(value="unused"), _converter(value="unused")

        async def first_async(*, prompt: str, **kwargs: object) -> ConverterResult:
            return ConverterResult(output_text=f"{prompt}-first", output_type="text")

        async def second_async(*, prompt: str, **kwargs: object) -> ConverterResult:
            return ConverterResult(output_text=f"{prompt}-second", output_type="text")

        first.convert_tokens_async.side_effect = first_async
        second.convert_tokens_async.side_effect = second_async
        harness.converter_service.get_converter_objects_for_ids.side_effect = lambda *, converter_ids: (
            [first] if converter_ids == ["first"] else [second]
        )
        lineage_id = str(uuid.uuid4())
        request = _request(conversation_id=harness.attack.conversation_id, count=3, mode=mode)
        request.pieces = [
            MessagePieceRequest(
                original_value="Original",
                converted_value="Already converted",
                original_prompt_id=lineage_id,
            ),
            MessagePieceRequest(original_value="One"),
            MessagePieceRequest(original_value="Two"),
        ]
        request.request_converter_configurations = [
            ConverterConfigurationRequest(converter_ids=["first"]),
            ConverterConfigurationRequest(converter_ids=["second"], indexes_to_apply=[1, 2]),
        ]
        accepted = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        result = await _wait_batch_async(service=harness.service, status=accepted)

        assert result.state == MessageBatchState.COMPLETED
        assert first.convert_tokens_async.await_count == (2 if mode == RequestConverterMode.SHARED else 6)
        assert second.convert_tokens_async.await_count == first.convert_tokens_async.await_count
        for call in harness.send.await_args_list:
            message = call.kwargs["message"]
            assert message.get_values() == ["Already converted", "One-first-second", "Two-first-second"]
            assert str(message.message_pieces[0].original_prompt_id) == lineage_id

    async def test_count_one_keeps_the_source_without_creating_copies(self, harness: _Harness) -> None:
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=1),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)

        assert result.state == MessageBatchState.COMPLETED
        assert [branch.conversation_id for branch in result.branches] == [harness.attack.conversation_id]
        assert harness.send.await_count == 1
        attack = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        assert attack[0].related_conversations == set()
        response = await harness.attack_service.get_conversation_messages_async(
            attack_result_id=harness.attack.attack_result_id,
            conversation_id=harness.attack.conversation_id,
        )
        assert response is not None
        assert [message.role for message in response.messages] == ["user", "assistant"]

    async def test_responses_are_converted_independently(self, harness: _Harness) -> None:
        converter = _converter(value="Response conversion")
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        request.response_converter_configurations = [ConverterConfigurationRequest(converter_ids=["response"])]
        accepted = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        assert converter.convert_tokens_async.await_count == 3
        for branch in result.branches:
            pieces = await asyncio.to_thread(harness.memory.get_message_pieces, conversation_id=branch.conversation_id)
            assert pieces[-1].original_value == "Reply"
            assert pieces[-1].converted_value == "Response conversion"

    @pytest.mark.parametrize("failed_count", [1, 3])
    async def test_provider_failures_preserve_siblings_and_existing_error_pieces(
        self, harness: _Harness, failed_count: int
    ) -> None:
        calls = 0

        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            nonlocal calls
            calls += 1
            if calls <= failed_count:
                raise RuntimeError("provider-private diagnostic")
            return [construct_response_from_request(request=message.get_piece(), response_text_pieces=["Reply"])]

        harness.send.side_effect = send_async
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.FAILED
        assert calls == 3
        assert sum(branch.state == MessageBatchBranchState.FAILED for branch in result.branches) == failed_count
        assert "provider-private" not in result.model_dump_json()
        for branch in result.branches:
            pieces = await asyncio.to_thread(harness.memory.get_message_pieces, conversation_id=branch.conversation_id)
            assert len(pieces) == 2
            assert sum(piece.has_error() for piece in pieces) == (branch.state == MessageBatchBranchState.FAILED)

    async def test_per_branch_conversion_failure_is_saved_without_cancelling_siblings(self, harness: _Harness) -> None:
        converter = _converter(value="converted")
        converter.convert_tokens_async.side_effect = [
            ValueError("cannot convert"),
            ConverterResult(output_text="second", output_type="text"),
            ConverterResult(output_text="third", output_type="text"),
        ]
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(
                conversation_id=harness.attack.conversation_id,
                count=3,
                mode=RequestConverterMode.PER_BRANCH,
                converter_ids=["converter"],
            ),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert converter.convert_tokens_async.await_count == 3
        assert harness.send.await_count == 2
        assert [branch.state for branch in result.branches].count(MessageBatchBranchState.FAILED) == 1
        failed = next(branch for branch in result.branches if branch.state == MessageBatchBranchState.FAILED)
        pieces = await asyncio.to_thread(harness.memory.get_message_pieces, conversation_id=failed.conversation_id)
        assert len(pieces) == 2
        assert pieces[0].id == pieces[0].original_prompt_id
        assert pieces[1].response_error == "processing"

    async def test_shared_conversion_failure_does_not_create_clones_or_dispatch(self, harness: _Harness) -> None:
        converter = _converter(value="unused")
        converter.convert_tokens_async.side_effect = ValueError("cannot convert")
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3, converter_ids=["converter"]),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        attack = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        assert result.state == MessageBatchState.FAILED
        assert harness.send.await_count == 0
        assert len(attack[0].get_active_conversation_ids()) == 1
        assert len(result.branches) == 1
        assert result.branches[0].state == MessageBatchBranchState.FAILED

    async def test_invalid_converted_media_keeps_the_original_request_identity(
        self, harness: _Harness, tmp_path: Path
    ) -> None:
        converter = _converter(value=str(tmp_path / "missing.png"), data_type="image_path")
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3, converter_ids=["converter"]),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.FAILED
        assert harness.send.await_count == 0
        source = await asyncio.to_thread(
            harness.memory.get_message_pieces, conversation_id=harness.attack.conversation_id
        )
        assert source[0].id == source[0].original_prompt_id
        for branch in result.branches:
            pieces = await asyncio.to_thread(harness.memory.get_message_pieces, conversation_id=branch.conversation_id)
            assert pieces[0].original_prompt_id == source[0].id
            assert pieces[1].response_error == "processing"
        assert harness.scheduler._reserved == 0

    async def test_response_converter_failure_preserves_request_and_an_error(self, harness: _Harness) -> None:
        converter = _converter(value="unused")
        converter.convert_tokens_async.side_effect = ValueError("response conversion failed")
        harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        request.response_converter_configurations = [ConverterConfigurationRequest(converter_ids=["response"])]
        accepted = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.FAILED
        assert harness.send.await_count == 3
        assert converter.convert_tokens_async.await_count == 3
        for branch in result.branches:
            pieces = await asyncio.to_thread(harness.memory.get_message_pieces, conversation_id=branch.conversation_id)
            assert [piece.role for piece in pieces] == ["user", "assistant"]
            assert pieces[-1].response_error == "processing"
            assert len(branch.new_message_piece_ids) == 2

    async def test_write_only_target_is_successful(self, harness: _Harness) -> None:
        harness.send.side_effect = None
        harness.send.return_value = []
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        assert all(len(branch.new_message_piece_ids) == 1 for branch in result.branches)

    async def test_returned_error_message_uses_core_error_semantics_without_duplicate_errors(
        self, harness: _Harness
    ) -> None:
        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            return [
                construct_response_from_request(
                    request=message.get_piece(), response_text_pieces=["Stored error"], response_type="error"
                )
            ]

        harness.send.side_effect = send_async
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.FAILED
        assert all(branch.state == MessageBatchBranchState.FAILED for branch in result.branches)
        assert all(len(branch.new_message_piece_ids) == 2 for branch in result.branches)

    async def test_acceptance_is_prompt_and_progress_precedes_the_slowest_reply(self, harness: _Harness) -> None:
        release_preparation = asyncio.Event()
        preparation_started = asyncio.Event()
        release_slow_reply = asyncio.Event()
        prepare = harness.attack_service._prepare_message_async

        async def prepare_async(*, request: AddMessageRequest, conversation_id: str, sequence: int) -> Message:
            preparation_started.set()
            await release_preparation.wait()
            return await prepare(request=request, conversation_id=conversation_id, sequence=sequence)

        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            if message.conversation_id == harness.attack.conversation_id:
                await release_slow_reply.wait()
            return [construct_response_from_request(request=message.get_piece(), response_text_pieces=["Reply"])]

        harness.send.side_effect = send_async
        with patch.object(harness.attack_service, "_prepare_message_async", side_effect=prepare_async):
            request = _request(conversation_id=harness.attack.conversation_id, count=3)
            accepted = await asyncio.wait_for(
                harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request),
                timeout=3,
            )
            assert accepted.branches == []
            assert accepted.state == MessageBatchState.QUEUED
            await asyncio.wait_for(preparation_started.wait(), timeout=3)
            request.pieces[0].original_value = "Changed after acceptance"
            release_preparation.set()
            await _wait_until_async(
                lambda: any(
                    branch.state == MessageBatchBranchState.COMPLETED
                    for branch in harness.service.get_status(
                        attack_result_id=accepted.attack_result_id, batch_id=accepted.batch_id
                    ).branches
                )
            )
            progress = harness.service.get_status(
                attack_result_id=accepted.attack_result_id, batch_id=accepted.batch_id
            )
            release_slow_reply.set()
            result = await _wait_batch_async(service=harness.service, status=accepted)
        assert progress.state == MessageBatchState.RUNNING
        assert len(progress.branches) == 3
        assert progress.branches[0].state == MessageBatchBranchState.SENDING
        assert result.state == MessageBatchState.COMPLETED
        assert all(call.kwargs["message"].get_value() == "Next prompt" for call in harness.send.await_args_list)
        assert "Next prompt" not in result.model_dump_json()
        assert set(result.model_dump()) == {
            "batch_id",
            "attack_result_id",
            "source_conversation_id",
            "requested_count",
            "state",
            "branches",
            "error",
        }

    async def test_submission_identity_deduplicates_and_rejects_changed_payload(self, harness: _Harness) -> None:
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        first, duplicate = await asyncio.gather(
            harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request),
            harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request),
        )
        assert first.batch_id == duplicate.batch_id
        await _wait_batch_async(service=harness.service, status=first)
        same = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        assert same.batch_id == first.batch_id
        assert harness.send.await_count == 3
        request.pieces[0].original_value = "Different request"
        with pytest.raises(ManualSendConflictError, match="submission_id"):
            await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)

    @pytest.mark.parametrize("reordered_field", ["pieces", "converters", "configurations"])
    async def test_submission_identity_preserves_sequence_order(
        self, *, harness: _Harness, reordered_field: str
    ) -> None:
        request = _request(conversation_id=harness.attack.conversation_id, count=1)
        if reordered_field == "pieces":
            request.pieces = [
                MessagePieceRequest(original_value="First piece"),
                MessagePieceRequest(original_value="Second piece"),
            ]
        elif reordered_field == "converters":
            request.request_converter_configurations = [
                ConverterConfigurationRequest(converter_ids=["first", "second"])
            ]
            harness.converter_service.get_converter_objects_for_ids.return_value = [
                _converter(value="First conversion"),
                _converter(value="Second conversion"),
            ]
        else:
            request.request_converter_configurations = [
                ConverterConfigurationRequest(converter_ids=["first"]),
                ConverterConfigurationRequest(converter_ids=["second"]),
            ]
        accepted = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        finished = await _wait_batch_async(service=harness.service, status=accepted)
        assert finished.state == MessageBatchState.COMPLETED
        reordered = request.model_copy(deep=True)
        if reordered_field == "pieces":
            reordered.pieces.reverse()
        elif reordered_field == "converters":
            assert reordered.request_converter_configurations is not None
            reordered.request_converter_configurations[0].converter_ids.reverse()
        else:
            assert reordered.request_converter_configurations is not None
            reordered.request_converter_configurations.reverse()
        with pytest.raises(ManualSendConflictError, match="submission_id"):
            await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=reordered)

    @pytest.mark.parametrize(
        "invalid", ["missing_target", "mismatch", "unrelated", "adversarial", "converter", "piece"]
    )
    async def test_invalid_requests_have_no_persistent_side_effects(self, harness: _Harness, invalid: str) -> None:
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        if invalid == "missing_target":
            harness.target_service.get_target_object.return_value = None
        elif invalid == "mismatch":
            harness.target_service.get_target_object.return_value = MockPromptTarget(rpm=123)
        elif invalid in ("unrelated", "adversarial"):
            request.target_conversation_id = str(uuid.uuid4())
            if invalid == "adversarial":
                await asyncio.to_thread(
                    harness.memory.update_attack_result_by_id,
                    attack_result_id=harness.attack.attack_result_id,
                    update_fields={"adversarial_chat_conversation_ids": [request.target_conversation_id]},
                )
        elif invalid == "converter":
            request.request_converter_configurations = [ConverterConfigurationRequest(converter_ids=["missing"])]
            harness.converter_service.get_converter_objects_for_ids.side_effect = ValueError("Unknown converter")
        elif invalid == "piece":
            request.pieces[0].data_type = "not-a-data-type"
        with pytest.raises(ValueError):
            await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        attacks = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        assert len(attacks[0].get_active_conversation_ids()) == 1
        assert await asyncio.to_thread(harness.memory.get_message_pieces) == []
        assert harness.send.await_count == 0
        assert harness.scheduler._reserved == 0

    async def test_transient_handle_loss_does_not_remove_saved_conversations(self, harness: _Harness) -> None:
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        result = await _wait_batch_async(service=harness.service, status=accepted)
        restarted = MultiSendService(
            memory=harness.memory, attack_service=harness.attack_service, scheduler=harness.scheduler
        )
        with pytest.raises(MessageBatchNotFoundError, match="unavailable or expired"):
            restarted.get_status(attack_result_id=accepted.attack_result_id, batch_id=accepted.batch_id)
        for branch in result.branches:
            conversation = await harness.attack_service.get_conversation_messages_async(
                attack_result_id=accepted.attack_result_id, conversation_id=branch.conversation_id
            )
            assert conversation is not None
            assert len(conversation.messages) == 2
        assert harness.send.await_count == 3

    async def test_source_conversation_target_is_checked_even_when_attack_target_matches(
        self, harness: _Harness
    ) -> None:
        other_target = MockPromptTarget(rpm=42)
        other_conversation = Conversation(
            conversation_id=str(uuid.uuid4()), target_identifier=other_target.get_identifier()
        )
        await asyncio.to_thread(
            harness.memory.add_conversation_branches_to_attack,
            attack_result_id=harness.attack.attack_result_id,
            conversations=[other_conversation],
            message_pieces=[],
        )
        with pytest.raises(ValueError, match="source conversation's target"):
            await harness.service.submit_async(
                attack_result_id=harness.attack.attack_result_id,
                request=_request(conversation_id=other_conversation.conversation_id, count=3),
            )
        assert harness.send.await_count == 0
        assert harness.scheduler._reserved == 0
        attack = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        assert len(attack[0].get_active_conversation_ids()) == 2

    async def test_terminal_retention_is_bounded_and_expiration_is_explicit(self, harness: _Harness) -> None:
        with patch.object(harness.service, "MAX_TERMINAL_BATCHES", 1):
            first = await harness.service.submit_async(
                attack_result_id=harness.attack.attack_result_id,
                request=_request(conversation_id=harness.attack.conversation_id, count=1),
            )
            await _wait_batch_async(service=harness.service, status=first)
            second = await harness.service.submit_async(
                attack_result_id=harness.attack.attack_result_id,
                request=_request(conversation_id=harness.attack.conversation_id, count=1),
            )
            await _wait_batch_async(service=harness.service, status=second)
        assert len(harness.service._terminal) == 1
        with pytest.raises(MessageBatchNotFoundError):
            harness.service.get_status(attack_result_id=first.attack_result_id, batch_id=first.batch_id)
        with patch.object(harness.service, "TERMINAL_TTL_SECONDS", -1), pytest.raises(MessageBatchNotFoundError):
            harness.service.get_status(attack_result_id=second.attack_result_id, batch_id=second.batch_id)
        assert harness.service._submissions == {}

    async def test_shutdown_before_first_task_step_releases_admission(self, harness: _Harness) -> None:
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        await harness.service.shutdown_async()
        status = harness.service.get_status(attack_result_id=accepted.attack_result_id, batch_id=accepted.batch_id)
        assert status.state == MessageBatchState.FAILED
        assert harness.scheduler._reserved == 0
        assert not harness.scheduler._conversations
        assert harness.send.await_count == 0

    async def test_shutdown_during_send_saves_interruption_and_releases_permits(self, harness: _Harness) -> None:
        started = asyncio.Event()

        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            started.set()
            await asyncio.Event().wait()
            return []

        harness.send.side_effect = send_async
        accepted = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        await asyncio.wait_for(started.wait(), timeout=3)
        await harness.service.shutdown_async()
        status = harness.service.get_status(attack_result_id=accepted.attack_result_id, batch_id=accepted.batch_id)
        assert status.state == MessageBatchState.FAILED
        assert all(branch.state == MessageBatchBranchState.FAILED for branch in status.branches)
        assert all(branch.new_message_piece_ids for branch in status.branches)
        assert harness.scheduler._reserved == 0
        assert harness.scheduler._active == 0
        assert not harness.scheduler._conversations

    async def test_shutdown_waits_for_atomic_registration_before_releasing_source(self, harness: _Harness) -> None:
        registration_started = asyncio.Event()
        release_registration = threading.Event()
        loop = asyncio.get_running_loop()
        register = harness.memory.add_conversation_branches_to_attack

        def register_later(**kwargs: object) -> bool:
            loop.call_soon_threadsafe(registration_started.set)
            if not release_registration.wait(timeout=5):
                raise TimeoutError("Test did not release registration")
            return register(**kwargs)

        with patch.object(harness.memory, "add_conversation_branches_to_attack", side_effect=register_later):
            accepted = await harness.service.submit_async(
                attack_result_id=harness.attack.attack_result_id,
                request=_request(conversation_id=harness.attack.conversation_id, count=3),
            )
            await asyncio.wait_for(registration_started.wait(), timeout=3)
            shutdown = asyncio.create_task(harness.service.shutdown_async())
            await asyncio.sleep(0)
            assert not shutdown.done()
            assert harness.attack.conversation_id in harness.scheduler._conversations
            release_registration.set()
            await asyncio.wait_for(shutdown, timeout=5)
        result = harness.service.get_status(attack_result_id=accepted.attack_result_id, batch_id=accepted.batch_id)
        assert result.state == MessageBatchState.FAILED
        assert len(result.branches) == 3
        assert all(branch.state == MessageBatchBranchState.FAILED for branch in result.branches)
        assert all(branch.new_message_piece_ids for branch in result.branches)
        assert harness.scheduler._reserved == 0
        assert harness.send.await_count == 0

    async def test_concurrent_converter_tracking_keeps_both_pipelines(self, harness: _Harness) -> None:
        first, second = _converter(value="first"), _converter(value="second")
        second.get_identifier.return_value = ConverterIdentifier(
            class_name="OtherConverter", class_module="tests.unit.backend"
        )
        await asyncio.gather(
            *(
                harness.attack_service._update_attack_after_message_async(
                    attack_result_id=harness.attack.attack_result_id,
                    last_response_id=None,
                    request_converter_configurations=[ConverterConfiguration(converters=[converter])],
                    response_converter_configurations=[],
                )
                for converter in (first, second)
            )
        )
        attacks = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        identifier = attacks[0].get_attack_strategy_identifier()
        assert identifier is not None
        assert {item.hash for item in identifier.get_child_list("request_converters")} == {
            first.get_identifier().hash,
            second.get_identifier().hash,
        }

    async def test_concurrency_budget_is_shared_across_batches_and_single_send(self, harness: _Harness) -> None:
        extra_sources = [str(uuid.uuid4()), str(uuid.uuid4())]
        await asyncio.to_thread(
            harness.memory.add_conversation_branches_to_attack,
            attack_result_id=harness.attack.attack_result_id,
            conversations=[
                Conversation(conversation_id=item, target_identifier=harness.target.get_identifier())
                for item in extra_sources
            ],
            message_pieces=[],
        )
        release = asyncio.Event()
        active = 0
        peak = 0

        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            try:
                await release.wait()
                return [construct_response_from_request(request=message.get_piece(), response_text_pieces=["Reply"])]
            finally:
                active -= 1

        harness.send.side_effect = send_async
        first = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=harness.attack.conversation_id, count=3),
        )
        second = await harness.service.submit_async(
            attack_result_id=harness.attack.attack_result_id,
            request=_request(conversation_id=extra_sources[0], count=3),
        )
        single = asyncio.create_task(
            harness.attack_service.add_message_async(
                attack_result_id=harness.attack.attack_result_id,
                request=AddMessageRequest(
                    pieces=[MessagePieceRequest(original_value="Single")],
                    target_registry_name="target",
                    target_conversation_id=extra_sources[1],
                ),
            )
        )
        try:
            await _wait_until_async(lambda: active == 2)
            queued = harness.service.get_status(attack_result_id=second.attack_result_id, batch_id=second.batch_id)
        finally:
            release.set()
            single_result = await single
        first = await _wait_batch_async(service=harness.service, status=first)
        second = await _wait_batch_async(service=harness.service, status=second)
        assert first.state == second.state == MessageBatchState.COMPLETED
        assert peak == 2
        assert harness.send.await_count == 7
        assert queued.state == MessageBatchState.QUEUED or any(
            branch.state == MessageBatchBranchState.QUEUED for branch in queued.branches
        )
        assert len(single_result.messages.messages) == 2
        assert harness.scheduler._reserved == 0

    @pytest.mark.parametrize("use_converter", [False, True])
    async def test_rpm_and_converter_backed_work_are_serialized(self, harness: _Harness, use_converter: bool) -> None:
        active = 0
        peak = 0

        async def work_async() -> None:
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1

        async def send_async(*, message: Message, **kwargs: object) -> list[Message]:
            await work_async()
            return [construct_response_from_request(request=message.get_piece(), response_text_pieces=["Reply"])]

        harness.send.side_effect = send_async
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        if use_converter:
            converter = _converter(value="converted")

            async def convert_async(**kwargs: object) -> ConverterResult:
                await work_async()
                return ConverterResult(output_text="converted", output_type="text")

            converter.convert_tokens_async.side_effect = convert_async
            harness.converter_service.get_converter_objects_for_ids.return_value = [converter]
            request.request_converter_configurations = [ConverterConfigurationRequest(converter_ids=["converter"])]
            request.request_converter_mode = RequestConverterMode.PER_BRANCH
        else:
            harness.target._max_requests_per_minute = 100
        accepted = await harness.service.submit_async(attack_result_id=harness.attack.attack_result_id, request=request)
        result = await _wait_batch_async(service=harness.service, status=accepted)
        assert result.state == MessageBatchState.COMPLETED
        assert peak == 1
        assert harness.send.await_count == 3

    async def test_busy_conversation_and_full_queue_do_not_create_branches(self, harness: _Harness) -> None:
        claim = harness.scheduler.reserve(conversation_id=harness.attack.conversation_id)
        try:
            with pytest.raises(ManualSendConflictError):
                await harness.service.submit_async(
                    attack_result_id=harness.attack.attack_result_id,
                    request=_request(conversation_id=harness.attack.conversation_id),
                )
            with pytest.raises(ManualSendConflictError):
                await harness.attack_service.add_message_async(
                    attack_result_id=harness.attack.attack_result_id,
                    request=AddMessageRequest(
                        pieces=[MessagePieceRequest(original_value="Single")],
                        target_registry_name="target",
                        target_conversation_id=harness.attack.conversation_id,
                    ),
                )
        finally:
            claim.release()
        claim = harness.scheduler.reserve(conversation_id="another", count=12)
        try:
            with pytest.raises(ManualSendQueueFullError):
                await harness.service.submit_async(
                    attack_result_id=harness.attack.attack_result_id,
                    request=_request(conversation_id=harness.attack.conversation_id),
                )
        finally:
            claim.release()
        attack = await asyncio.to_thread(
            harness.memory.get_attack_results, attack_result_ids=[harness.attack.attack_result_id]
        )
        assert len(attack[0].get_active_conversation_ids()) == 1
        assert harness.scheduler._reserved == 0
        assert harness.send.await_count == 0

    async def test_routes_return_202_and_scoped_compact_status(self, harness: _Harness) -> None:
        app = FastAPI()
        app.include_router(message_batches.router, prefix="/api")
        request = _request(conversation_id=harness.attack.conversation_id, count=3)
        with patch.object(message_batches, "get_multi_send_service", return_value=harness.service):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                response = await client.post(
                    f"/api/attacks/{harness.attack.attack_result_id}/messages/batch",
                    json=request.model_dump(mode="json"),
                )
                assert response.status_code == 202
                batch_id = response.json()["batch_id"]
                poll = await client.get(f"/api/attacks/{harness.attack.attack_result_id}/message-batches/{batch_id}")
                assert poll.status_code == 200
                assert "messages" not in poll.json()
                wrong_scope = await client.get(f"/api/attacks/other/message-batches/{batch_id}")
                missing = await client.get(f"/api/attacks/{harness.attack.attack_result_id}/message-batches/missing")
                assert wrong_scope.status_code == 404
                assert missing.status_code == 404

    async def test_route_rejects_invalid_count_and_missing_attack(self, harness: _Harness) -> None:
        app = FastAPI()
        app.include_router(message_batches.router, prefix="/api")
        request = _request(conversation_id=harness.attack.conversation_id).model_dump(mode="json")
        with patch.object(message_batches, "get_multi_send_service", return_value=harness.service):
            async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
                for count in [True, 1.5, 0, 11]:
                    response = await client.post(
                        f"/api/attacks/{harness.attack.attack_result_id}/messages/batch",
                        json={**request, "count": count},
                    )
                    assert response.status_code == 422
                response = await client.post(f"/api/attacks/{uuid.uuid4()}/messages/batch", json=request)
                assert response.status_code == 404
        assert harness.send.await_count == 0


async def test_unused_service_shutdown_does_not_initialize_memory() -> None:
    get_multi_send_service.cache_clear()
    with patch("pyrit.backend.services.multi_send_service.CentralMemory") as memory:
        await shutdown_message_batches_async()
        memory.get_memory_instance.assert_not_called()
