# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Manual multi-send orchestration with transient progress and canonical attack membership."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

from pyrit.backend.mappers import request_to_pyrit_message
from pyrit.backend.models.message_batches import (
    MessageBatchBranch,
    MessageBatchBranchState,
    MessageBatchRequest,
    MessageBatchState,
    MessageBatchStatus,
    RequestConverterMode,
)
from pyrit.backend.services.attack_service import AttackService, get_attack_service
from pyrit.backend.services.manual_send_scheduler import (
    ManualSendConflictError,
    ManualSendReservation,
    ManualSendScheduler,
    get_manual_send_scheduler,
)
from pyrit.backend.services.target_service import get_target_service
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import Conversation, Message, MessagePiece, construct_response_from_request
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class MessageBatchNotFoundError(LookupError):
    """The attack or its transient batch handle is unavailable."""


@dataclass(kw_only=True)
class _ValidatedBatch:
    request: MessageBatchRequest
    target: PromptTarget
    source: Conversation
    request_configurations: list[ConverterConfiguration]
    response_configurations: list[ConverterConfiguration]
    source_message: Message | None = None


@dataclass(kw_only=True)
class _PreparedBranch:
    message: Message
    status: MessageBatchBranch
    last_response_id: str | None = None


@dataclass(kw_only=True)
class _Batch:
    status: MessageBatchStatus
    submission_id: str
    fingerprint: str
    reservation: ManualSendReservation
    task: asyncio.Task[None] | None = None
    finished_at: float | None = None
    last_response_id: str | None = None


class MultiSendService:
    """
    Accept bounded submissions and publish each branch as it settles.

    Handles and submission deduplication are local to this process and expire.
    They are not a durable ledger or an exactly-once guarantee at a provider.
    """

    TERMINAL_TTL_SECONDS = 600
    MAX_TERMINAL_BATCHES = 128
    SEND_ERROR = "Message send failed. Open the conversation for details and check server logs."
    PREPARATION_ERROR = "Batch preparation failed before target dispatch. Check the message and converter settings."
    INTERRUPTED_ERROR = (
        "Send interrupted. Provider delivery may be unknown; inspect saved conversations before sending again."
    )

    def __init__(
        self,
        *,
        memory: MemoryInterface | None = None,
        attack_service: AttackService | None = None,
        scheduler: ManualSendScheduler | None = None,
    ) -> None:
        """Initialize live bookkeeping with injected or application-wide services."""
        self._memory = memory if memory is not None else CentralMemory.get_memory_instance()
        self._attack_service = attack_service if attack_service is not None else get_attack_service()
        self._scheduler = scheduler if scheduler is not None else get_manual_send_scheduler()
        self._batches: dict[str, _Batch] = {}
        self._submissions: dict[tuple[str, str], str] = {}
        self._terminal: OrderedDict[str, float] = OrderedDict()
        self._accept_lock = asyncio.Lock()
        self._closing = False

    async def submit_async(self, *, attack_result_id: str, request: MessageBatchRequest) -> MessageBatchStatus:
        """
        Validate and accept a submission without awaiting history preparation or provider I/O.

        Returns:
            MessageBatchStatus: A detached, compact progress snapshot.

        Raises:
            MessageBatchNotFoundError: If the attack is missing.
            ManualSendConflictError: If the source is busy or an identity was reused with different input.
            ValueError: If the target, message, or converter configuration is invalid.
            RuntimeError: If shutdown has begun.
        """
        payload = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        async with self._accept_lock:
            if self._closing:
                raise RuntimeError("Message batches are shutting down")
            self._expire_terminal_batches()
            submission_key = (attack_result_id, request.submission_id)
            existing_id = self._submissions.get(submission_key)
            if existing_id is not None:
                existing = self._batches[existing_id]
                if existing.fingerprint != fingerprint:
                    raise ManualSendConflictError("submission_id was already used with a different batch request")
                return existing.status.model_copy(deep=True)
            reservation = self._scheduler.reserve(conversation_id=request.target_conversation_id, count=request.count)
            try:
                validated = await self._validate_async(
                    attack_result_id=attack_result_id, request=request.model_copy(deep=True)
                )
            except BaseException:
                reservation.release()
                raise
            batch = _Batch(
                status=MessageBatchStatus(
                    batch_id=str(uuid.uuid4()),
                    attack_result_id=attack_result_id,
                    source_conversation_id=request.target_conversation_id,
                    requested_count=request.count,
                ),
                submission_id=request.submission_id,
                fingerprint=fingerprint,
                reservation=reservation,
            )
            self._batches[batch.status.batch_id] = batch
            self._submissions[submission_key] = batch.status.batch_id
            batch.task = asyncio.create_task(self._run_batch_async(batch=batch, validated=validated))
            batch.task.add_done_callback(lambda task: self._on_batch_done(batch=batch, task=task))
            return batch.status.model_copy(deep=True)

    def get_status(self, *, attack_result_id: str, batch_id: str) -> MessageBatchStatus:
        """
        Read progress without fetching transcripts or reconstructing expired executions.

        Returns:
            MessageBatchStatus: A detached progress snapshot.

        Raises:
            MessageBatchNotFoundError: If the handle expired, was lost, or belongs to another attack.
        """
        self._expire_terminal_batches()
        batch = self._batches.get(batch_id)
        if batch is None or batch.status.attack_result_id != attack_result_id:
            raise MessageBatchNotFoundError(
                "Batch status is unavailable or expired. Reload saved conversations; do not automatically resend."
            )
        return batch.status.model_copy(deep=True)

    async def shutdown_async(self) -> None:
        """Stop accepting work and settle cancellations without retrying uncertain sends."""
        async with self._accept_lock:
            self._closing = True
            tasks = [batch.task for batch in self._batches.values() if batch.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _validate_async(self, *, attack_result_id: str, request: MessageBatchRequest) -> _ValidatedBatch:
        results = await asyncio.to_thread(self._memory.get_attack_results, attack_result_ids=[attack_result_id])
        if not results:
            raise MessageBatchNotFoundError(f"Attack '{attack_result_id}' not found")
        attack = results[0]
        if request.target_conversation_id not in attack.get_active_conversation_ids():
            raise ValueError("Source conversation is not an active objective conversation of this attack")
        assert request.target_registry_name is not None
        target = get_target_service().get_target_object(target_registry_name=request.target_registry_name)
        if target is None:
            raise ValueError("The selected target is not available")
        self._attack_service._validate_target_match(
            attack_identifier=attack.get_attack_strategy_identifier(), request=request
        )
        source = await asyncio.to_thread(self._memory._get_conversation, conversation_id=request.target_conversation_id)
        if (
            source is not None
            and source.target_identifier is not None
            and source.target_identifier.hash != target.get_identifier().hash
        ):
            raise ValueError("The selected target does not match the source conversation's target")
        source = source or Conversation(
            conversation_id=request.target_conversation_id, target_identifier=target.get_identifier()
        )
        # Mapping validates every piece, data type, and lineage UUID without writing media.
        request_to_pyrit_message(request=request, conversation_id=source.conversation_id, sequence=0)
        configurations = self._attack_service._resolve_request_converter_configs(request=request)
        configurations = self._attack_service._exclude_preconverted_piece_indexes(
            configurations=configurations,
            preconverted_indexes={
                index for index, piece in enumerate(request.pieces) if piece.converted_value is not None
            },
            piece_count=len(request.pieces),
        )
        return _ValidatedBatch(
            request=request,
            target=target,
            source=source,
            request_configurations=configurations,
            response_configurations=self._attack_service._resolve_converter_configs(
                configurations=request.response_converter_configurations
            ),
        )

    async def _run_batch_async(self, *, batch: _Batch, validated: _ValidatedBatch) -> None:
        branches: list[_PreparedBranch] = []
        try:
            exclusive = bool(
                validated.request.request_converter_mode == RequestConverterMode.SHARED
                and validated.request_configurations
            )
            async with self._scheduler.operation_async(exclusive=exclusive):
                batch.status.state = MessageBatchState.PREPARING
                branches = await self._prepare_async(batch=batch, validated=validated)
            batch.status.state = MessageBatchState.QUEUED
            await asyncio.gather(
                *(self._send_branch_async(batch=batch, validated=validated, branch=branch) for branch in branches)
            )
            await self._record_converter_usage_async(batch=batch, validated=validated)
            failed = sum(branch.status.state == MessageBatchBranchState.FAILED for branch in branches)
            batch.status.state = MessageBatchState.FAILED if failed else MessageBatchState.COMPLETED
            batch.status.error = f"{failed} of {len(branches)} sends failed." if failed else None
        except asyncio.CancelledError:
            batch.status.error = self.INTERRUPTED_ERROR
            await self._fail_unsettled_branches_async(batch=batch, branches=branches, error=self.INTERRUPTED_ERROR)
            batch.status.state = MessageBatchState.FAILED
        except Exception:
            logger.exception("Message batch '%s' failed", batch.status.batch_id)
            batch.status.error = self.PREPARATION_ERROR if not batch.status.branches else self.SEND_ERROR
            if not batch.status.branches and validated.source_message is not None:
                await self._record_preparation_failure_async(batch=batch, validated=validated)
            await self._fail_unsettled_branches_async(batch=batch, branches=branches, error=self.SEND_ERROR)
            batch.status.state = MessageBatchState.FAILED
        finally:
            self._finish_batch(batch)

    def _on_batch_done(self, *, batch: _Batch, task: asyncio.Task[None]) -> None:
        if batch.finished_at is not None:
            return
        # Cancellation before the coroutine's first step never enters its try/finally.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Message batch task failed", exc_info=(type(error), error, error.__traceback__))
        batch.status.state = MessageBatchState.FAILED
        batch.status.error = self.INTERRUPTED_ERROR
        self._finish_batch(batch)

    def _finish_batch(self, batch: _Batch) -> None:
        batch.reservation.release()
        batch.finished_at = time.monotonic()
        batch.task = None
        self._terminal[batch.status.batch_id] = batch.finished_at
        self._expire_terminal_batches()

    async def _prepare_async(self, *, batch: _Batch, validated: _ValidatedBatch) -> list[_PreparedBranch]:
        history = await asyncio.to_thread(
            self._memory.get_conversation_messages, conversation_id=validated.source.conversation_id
        )
        sequence = max((message.sequence for message in history), default=-1) + 1
        message = await self._attack_service._prepare_message_async(
            request=validated.request, conversation_id=validated.source.conversation_id, sequence=sequence
        )
        validated.source_message = message
        if validated.request.request_converter_mode == RequestConverterMode.SHARED:
            await PromptNormalizer().convert_values_async(
                converter_configurations=validated.request_configurations, message=message
            )
        conversations, pieces, branches = await asyncio.to_thread(
            self._prepare_copies,
            history=list(history),
            source=validated.source,
            message=message,
            count=validated.request.count,
        )
        batch.reservation.add_conversations([conversation.conversation_id for conversation in conversations])
        registration = asyncio.create_task(
            asyncio.to_thread(
                self._memory.add_conversation_branches_to_attack,
                attack_result_id=batch.status.attack_result_id,
                conversations=conversations,
                message_pieces=pieces,
                source_conversation=validated.source,
            )
        )
        try:
            stored = await asyncio.shield(registration)
        except asyncio.CancelledError:
            # A running database transaction cannot be cancelled by cancelling its thread await.
            stored = await registration
            if stored:
                batch.status.branches = [branch.status for branch in branches]
                await self._fail_unsettled_branches_async(batch=batch, branches=branches, error=self.INTERRUPTED_ERROR)
            raise
        if not stored:
            raise MessageBatchNotFoundError("Attack was removed before branch preparation committed")
        batch.status.branches = [branch.status for branch in branches]
        return branches

    def _prepare_copies(
        self, *, history: list[Message], source: Conversation, message: Message, count: int
    ) -> tuple[list[Conversation], list[MessagePiece], list[_PreparedBranch]]:
        conversations: list[Conversation] = []
        pieces: list[MessagePiece] = []
        branches = [_PreparedBranch(message=message, status=MessageBatchBranch(conversation_id=source.conversation_id))]
        for _ in range(count - 1):
            conversation_id, copied_pieces = self._memory.duplicate_messages(messages=history)
            conversations.append(source.model_copy(deep=True, update={"conversation_id": conversation_id}))
            pieces.extend(copied_pieces)
            branch_message = message.duplicate()
            for piece in branch_message.message_pieces:
                piece.conversation_id = conversation_id
            branches.append(
                _PreparedBranch(message=branch_message, status=MessageBatchBranch(conversation_id=conversation_id))
            )
        return conversations, pieces, branches

    async def _send_branch_async(self, *, batch: _Batch, validated: _ValidatedBatch, branch: _PreparedBranch) -> None:
        per_branch = validated.request.request_converter_mode == RequestConverterMode.PER_BRANCH
        exclusive = bool(
            validated.target._max_requests_per_minute
            or validated.response_configurations
            or (per_branch and validated.request_configurations)
        )
        try:
            async with self._scheduler.operation_async(exclusive=exclusive):
                branch.status.state = MessageBatchBranchState.SENDING
                batch.status.state = MessageBatchState.RUNNING
                normalizer = PromptNormalizer()
                error: str | None = None
                try:
                    if per_branch:
                        await normalizer.convert_values_async(
                            converter_configurations=validated.request_configurations, message=branch.message
                        )
                    await normalizer.send_prompt_async(
                        message=branch.message,
                        target=validated.target,
                        conversation_id=branch.status.conversation_id,
                        request_converter_configurations=[],
                        response_converter_configurations=validated.response_configurations,
                    )
                except Exception:
                    logger.exception(
                        "Send failed for batch '%s' branch '%s'",
                        batch.status.batch_id,
                        branch.status.conversation_id,
                    )
                    error = self.SEND_ERROR
                await self._settle_branch_async(branch=branch, error=error)
                if branch.last_response_id:
                    batch.last_response_id = branch.last_response_id
        except asyncio.CancelledError:
            await self._settle_branch_async(branch=branch, error=self.INTERRUPTED_ERROR)
            raise
        except Exception:
            logger.exception("Could not settle branch '%s'", branch.status.conversation_id)
            branch.status.state = MessageBatchBranchState.FAILED
            branch.status.error = "Send failed and its saved status could not be read. Check server logs."
        finally:
            batch.reservation.release_conversation(branch.status.conversation_id)

    async def _settle_branch_async(self, *, branch: _PreparedBranch, error: str | None) -> None:
        pieces = await self._read_new_pieces_async(branch)
        has_stored_error = any(piece.to_message().is_error() for piece in pieces)
        if error is None and has_stored_error:
            error = self.SEND_ERROR
        if error is None and not pieces:
            error = "The send did not produce saved message evidence. Check server logs before sending again."
            logger.error("No message evidence for branch '%s'", branch.status.conversation_id)
        if error is not None and not has_stored_error:
            try:
                await self._persist_failure_async(branch=branch, pieces=pieces, error=error)
                pieces = await self._read_new_pieces_async(branch)
            except Exception:
                logger.exception("Failed to save error evidence for branch '%s'", branch.status.conversation_id)
                error += " Failure details could not be saved."
        branch.status.new_message_piece_ids = [str(piece.id) for piece in pieces]
        branch.last_response_id = next((str(piece.id) for piece in reversed(pieces) if piece.role == "assistant"), None)
        branch.status.error = error
        branch.status.state = MessageBatchBranchState.FAILED if error else MessageBatchBranchState.COMPLETED

    async def _persist_failure_async(self, *, branch: _PreparedBranch, pieces: list[MessagePiece], error: str) -> None:
        normalizer = PromptNormalizer()
        stored_ids = {piece.id for piece in pieces}
        if not any(piece.id in stored_ids for piece in branch.message.message_pieces):
            try:
                await normalizer.hash_and_persist_message_async(message=branch.message)
            except (OSError, ValueError):
                # Invalid converted media must not leave siblings pointing to an unsaved original request.
                logger.exception("Could not hash failed request for branch '%s'", branch.status.conversation_id)
                await asyncio.to_thread(self._memory.add_message_to_memory, request=branch.message)
        error_response = construct_response_from_request(
            request=branch.message.message_pieces[0],
            response_text_pieces=[error],
            response_type="error",
            error="processing",
        )
        await normalizer.hash_and_persist_message_async(message=error_response)

    async def _read_new_pieces_async(self, branch: _PreparedBranch) -> list[MessagePiece]:
        pieces = await asyncio.to_thread(self._memory.get_message_pieces, conversation_id=branch.status.conversation_id)
        return [piece for piece in pieces if piece.sequence >= branch.message.sequence]

    async def _record_preparation_failure_async(self, *, batch: _Batch, validated: _ValidatedBatch) -> None:
        assert validated.source_message is not None
        branch = _PreparedBranch(
            message=validated.source_message,
            status=MessageBatchBranch(conversation_id=validated.source.conversation_id),
        )
        try:
            await asyncio.to_thread(self._memory.add_conversation_to_memory, conversation=validated.source)
            batch.status.branches = [branch.status]
            await self._settle_branch_async(branch=branch, error=self.PREPARATION_ERROR)
        except Exception:
            logger.exception("Could not persist preparation failure for batch '%s'", batch.status.batch_id)
            branch.status.state = MessageBatchBranchState.FAILED
            branch.status.error = "Batch preparation failed and its error details could not be saved."

    async def _fail_unsettled_branches_async(
        self, *, batch: _Batch, branches: list[_PreparedBranch], error: str
    ) -> None:
        for branch in branches:
            if branch.status.state in (MessageBatchBranchState.COMPLETED, MessageBatchBranchState.FAILED):
                continue
            try:
                await self._settle_branch_async(branch=branch, error=error)
            except Exception:
                logger.exception("Could not persist interruption for batch '%s'", batch.status.batch_id)
                branch.status.state = MessageBatchBranchState.FAILED
                branch.status.error = error + " Failure details could not be saved."

    async def _record_converter_usage_async(self, *, batch: _Batch, validated: _ValidatedBatch) -> None:
        await self._attack_service._update_attack_after_message_async(
            attack_result_id=batch.status.attack_result_id,
            last_response_id=batch.last_response_id,
            request_converter_configurations=validated.request_configurations,
            response_converter_configurations=validated.response_configurations,
        )

    def _expire_terminal_batches(self) -> None:
        oldest_allowed = time.monotonic() - self.TERMINAL_TTL_SECONDS
        while self._terminal:
            batch_id, finished_at = next(iter(self._terminal.items()))
            if finished_at > oldest_allowed and len(self._terminal) <= self.MAX_TERMINAL_BATCHES:
                break
            self._terminal.pop(batch_id)
            batch = self._batches.pop(batch_id)
            self._submissions.pop((batch.status.attack_result_id, batch.submission_id))


@lru_cache(maxsize=1)
def get_multi_send_service() -> MultiSendService:
    """Return the worker-local service without starting any background sends."""
    return MultiSendService()


async def shutdown_message_batches_async() -> None:
    """Shut down only an already-used service; do not initialize memory during shutdown."""
    if get_multi_send_service.cache_info().currsize:
        await get_multi_send_service().shutdown_async()
        get_multi_send_service.cache_clear()
