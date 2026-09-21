# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Shared execution for synchronous, single, and repeated manual sends."""

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, cast

from pyrit.backend.mappers import request_piece_to_pyrit_message_piece, request_to_pyrit_message
from pyrit.backend.models.attacks import AddMessageRequest, ConverterConfigurationRequest
from pyrit.backend.models.message_sends import (
    MessageSendBranch,
    MessageSendBranchState,
    MessageSendFailureStage,
    MessageSendRequest,
    MessageSendState,
    MessageSendStatus,
    RequestConverterMode,
)
from pyrit.backend.services.converter_service import get_converter_service
from pyrit.backend.services.manual_send_scheduler import (
    ManualSendConflictError,
    ManualSendReservation,
    ManualSendScheduler,
    get_manual_send_scheduler,
)
from pyrit.backend.services.media_persistence import persist_media_value_async
from pyrit.backend.services.target_service import get_target_service
from pyrit.common.deprecation import print_deprecation_message
from pyrit.memory import CentralMemory, MemoryInterface, data_serializer_factory
from pyrit.models import (
    AtomicAttackIdentifier,
    AttackIdentifier,
    AttackResult,
    AttackTechniqueIdentifier,
    ComponentIdentifier,
    Conversation,
    ConverterIdentifier,
    Message,
    MessagePiece,
    PromptDataType,
    construct_response_from_request,
)
from pyrit.prompt_normalizer import ConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class MessageSendNotFoundError(LookupError):
    """The attack or its transient operation handle is unavailable."""


@dataclass(kw_only=True)
class _ValidatedSend:
    request: MessageSendRequest
    target: PromptTarget
    source: Conversation
    request_configurations: list[ConverterConfiguration]
    response_configurations: list[ConverterConfiguration]
    source_message: Message | None = None


@dataclass(kw_only=True)
class _PreparedBranch:
    message: Message
    status: MessageSendBranch
    last_response_id: str | None = None


@dataclass(kw_only=True)
class _Send:
    status: MessageSendStatus
    submission_id: str
    fingerprint: str
    reservation: ManualSendReservation
    task: asyncio.Task[None] | None = None
    finished_at: float | None = None
    last_response_id: str | None = None
    failure: str | None = None


class MessageSendService:
    """
    Accept bounded submissions and publish each branch as it settles.

    Handles and submission deduplication are local to this process and expire.
    They are not a durable ledger or an exactly-once guarantee at a provider.
    """

    TERMINAL_TTL_SECONDS = 600
    MAX_TERMINAL_SENDS = 128
    SEND_ERROR = "Message send failed. Open the conversation for details and check server logs."
    PREPARATION_ERROR = "Send preparation failed before target dispatch. Check the message and converter settings."
    FINALIZATION_ERROR = (
        "Sends finished, but attack details could not be updated. Inspect saved conversations before sending again."
    )
    INTERRUPTED_ERROR = (
        "Send interrupted. Provider delivery may be unknown; inspect saved conversations before sending again."
    )

    def __init__(
        self,
        *,
        memory: MemoryInterface | None = None,
        scheduler: ManualSendScheduler | None = None,
    ) -> None:
        """Initialize live bookkeeping with injected or application-wide services."""
        self._memory = memory if memory is not None else CentralMemory.get_memory_instance()
        self._scheduler = scheduler if scheduler is not None else get_manual_send_scheduler()
        self._sends: dict[str, _Send] = {}
        self._submissions: dict[tuple[str, str], str] = {}
        self._terminal: OrderedDict[str, float] = OrderedDict()
        self._accept_lock = asyncio.Lock()
        self._message_metadata_lock = asyncio.Lock()
        self._closing = False

    async def send_and_wait_async(self, *, attack_result_id: str, request: AddMessageRequest) -> None:
        """
        Await a count-one operation for the legacy synchronous API.

        Raises:
            RuntimeError: If preparation, finalization, or saving failure evidence failed.
        """
        status = await self.submit_async(
            attack_result_id=attack_result_id,
            request=MessageSendRequest(**request.model_dump(), submission_id=str(uuid.uuid4())),
        )
        operation = self._sends[status.send_id]
        if operation.task is not None:
            await asyncio.shield(operation.task)
        if operation.failure is not None:
            raise RuntimeError(operation.failure)

    async def store_message_async(self, *, attack_result_id: str, request: AddMessageRequest) -> None:
        """
        Append context without dispatching, while respecting active conversation ownership.

        Raises:
            ValueError: If sending was requested or the conversation does not belong to the attack.
        """
        if request.send:
            raise ValueError("store_message_async requires send=False")
        reservation = self._scheduler.reserve(conversation_id=request.target_conversation_id)
        try:
            await self._get_attack_async(
                attack_result_id=attack_result_id, conversation_id=request.target_conversation_id
            )
            self._resolve_request_converter_configs(request=request)
            pieces = await asyncio.to_thread(
                self._memory.get_message_pieces, conversation_id=request.target_conversation_id
            )
            source = await asyncio.to_thread(
                self._memory._get_conversation, conversation_id=request.target_conversation_id
            )
            await self._store_message_only_async(
                conversation_id=request.target_conversation_id,
                request=request.model_copy(deep=True),
                sequence=max((piece.sequence for piece in pieces), default=-1) + 1,
                target_identifier=source.target_identifier if source else None,
            )
            await self._update_attack_after_message_async(
                attack_result_id=attack_result_id,
                last_response_id=None,
                request_converter_configurations=[],
                response_converter_configurations=[],
            )
        finally:
            reservation.release()

    async def submit_async(self, *, attack_result_id: str, request: MessageSendRequest) -> MessageSendStatus:
        """
        Validate and accept a submission without awaiting history preparation or provider I/O.

        Returns:
            MessageSendStatus: A detached, compact progress snapshot.

        Raises:
            MessageSendNotFoundError: If the attack is missing.
            ManualSendConflictError: If the source is busy or an identity was reused with different input.
            ValueError: If the target, message, or converter configuration is invalid.
            RuntimeError: If shutdown has begun.
        """
        payload = json.dumps(request.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        async with self._accept_lock:
            if self._closing:
                raise RuntimeError("Message sends are shutting down")
            self._expire_terminal_sends()
            submission_key = (attack_result_id, request.submission_id)
            existing_id = self._submissions.get(submission_key)
            if existing_id is not None:
                existing = self._sends[existing_id]
                if existing.fingerprint != fingerprint:
                    raise ManualSendConflictError("submission_id was already used with a different operation request")
                return existing.status.model_copy(deep=True)
            reservation = self._scheduler.reserve(conversation_id=request.target_conversation_id, count=request.count)
            try:
                validated = await self._validate_async(
                    attack_result_id=attack_result_id, request=request.model_copy(deep=True)
                )
            except BaseException:
                reservation.release()
                raise
            operation = _Send(
                status=MessageSendStatus(
                    send_id=str(uuid.uuid4()),
                    attack_result_id=attack_result_id,
                    source_conversation_id=request.target_conversation_id,
                    requested_count=request.count,
                ),
                submission_id=request.submission_id,
                fingerprint=fingerprint,
                reservation=reservation,
            )
            self._sends[operation.status.send_id] = operation
            self._submissions[submission_key] = operation.status.send_id
            operation.task = asyncio.create_task(self._run_send_async(operation=operation, validated=validated))
            operation.task.add_done_callback(lambda task: self._on_send_done(operation=operation, task=task))
            return operation.status.model_copy(deep=True)

    def get_status(self, *, attack_result_id: str, send_id: str) -> MessageSendStatus:
        """
        Read progress without fetching transcripts or reconstructing expired executions.

        Returns:
            MessageSendStatus: A detached progress snapshot.

        Raises:
            MessageSendNotFoundError: If the handle expired, was lost, or belongs to another attack.
        """
        self._expire_terminal_sends()
        operation = self._sends.get(send_id)
        if operation is None or operation.status.attack_result_id != attack_result_id:
            raise MessageSendNotFoundError(
                "Send status is unavailable or expired. Reload saved conversations; do not automatically resend."
            )
        return operation.status.model_copy(deep=True)

    async def get_status_async(self, *, attack_result_id: str, send_id: str, wait_ms: int = 0) -> MessageSendStatus:
        """
        Wait briefly for completion so fast sends do not pay a fixed polling delay.

        Returns:
            MessageSendStatus: The latest snapshot, including unfinished branches.

        Raises:
            ValueError: If the requested wait is outside the supported range.
            MessageSendNotFoundError: If the handle is missing, expired, or belongs to another attack.
        """
        if not 0 <= wait_ms <= 1000:
            raise ValueError("wait_ms must be between 0 and 1000")
        status = self.get_status(attack_result_id=attack_result_id, send_id=send_id)
        task = self._sends[send_id].task
        if task is not None and wait_ms:
            # Cancelling this read or timing it out must not cancel the accepted send.
            await asyncio.wait({task}, timeout=wait_ms / 1000)
            return self.get_status(attack_result_id=attack_result_id, send_id=send_id)
        return status

    async def shutdown_async(self) -> None:
        """Stop accepting work and settle cancellations without retrying uncertain sends."""
        async with self._accept_lock:
            self._closing = True
            tasks = [operation.task for operation in self._sends.values() if operation.task is not None]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _get_attack_async(self, *, attack_result_id: str, conversation_id: str) -> AttackResult:
        results = await asyncio.to_thread(self._memory.get_attack_results, attack_result_ids=[attack_result_id])
        if not results:
            raise MessageSendNotFoundError(f"Attack '{attack_result_id}' not found")
        attack = results[0]
        if conversation_id not in attack.get_active_conversation_ids():
            raise ValueError(f"Conversation '{conversation_id}' is not part of attack '{attack_result_id}'")
        return attack

    async def _validate_async(self, *, attack_result_id: str, request: MessageSendRequest) -> _ValidatedSend:
        attack = await self._get_attack_async(
            attack_result_id=attack_result_id, conversation_id=request.target_conversation_id
        )
        assert request.target_registry_name is not None
        target = get_target_service().get_target_object(target_registry_name=request.target_registry_name)
        if target is None:
            raise MessageSendNotFoundError(f"Target object for '{request.target_registry_name}' not found")
        self._validate_target_match(attack_identifier=attack.get_attack_strategy_identifier(), request=request)
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
        configurations = self._resolve_request_converter_configs(request=request)
        configurations = self._exclude_preconverted_piece_indexes(
            configurations=configurations,
            preconverted_indexes={
                index for index, piece in enumerate(request.pieces) if piece.converted_value is not None
            },
            piece_count=len(request.pieces),
        )
        return _ValidatedSend(
            request=request,
            target=target,
            source=source,
            request_configurations=configurations,
            response_configurations=self._resolve_converter_configs(
                configurations=request.response_converter_configurations
            ),
        )

    async def _run_send_async(self, *, operation: _Send, validated: _ValidatedSend) -> None:
        branches: list[_PreparedBranch] = []
        phase = MessageSendFailureStage.PREPARATION
        try:
            exclusive = bool(
                validated.request.request_converter_mode == RequestConverterMode.SHARED
                and validated.request_configurations
            )
            async with self._scheduler.operation_async(exclusive=exclusive):
                operation.status.state = MessageSendState.PREPARING
                branches = await self._prepare_async(operation=operation, validated=validated)
            operation.status.state = MessageSendState.QUEUED
            phase = MessageSendFailureStage.SENDING
            await asyncio.gather(
                *(
                    self._send_branch_async(operation=operation, validated=validated, branch=branch)
                    for branch in branches
                )
            )
            phase = MessageSendFailureStage.FINALIZATION
            await self._update_attack_after_message_async(
                attack_result_id=operation.status.attack_result_id,
                last_response_id=operation.last_response_id,
                request_converter_configurations=validated.request_configurations,
                response_converter_configurations=validated.response_configurations,
            )
            failed = sum(branch.status.state == MessageSendBranchState.FAILED for branch in branches)
            operation.status.state = MessageSendState.FAILED if failed else MessageSendState.COMPLETED
            operation.status.error = f"{failed} of {len(branches)} sends failed." if failed else None
            operation.status.failure_stage = MessageSendFailureStage.SENDING if failed else None
        except asyncio.CancelledError:
            operation.status.error = self.INTERRUPTED_ERROR
            operation.status.failure_stage = MessageSendFailureStage.INTERRUPTED
            operation.failure = self.INTERRUPTED_ERROR
            await self._fail_unsettled_branches_async(
                operation=operation, branches=branches, error=self.INTERRUPTED_ERROR
            )
            operation.status.state = MessageSendState.FAILED
        except Exception as exc:
            logger.exception("Message operation '%s' failed", operation.status.send_id)
            operation.status.failure_stage = phase
            operation.failure = str(exc)
            operation.status.error = (
                self.PREPARATION_ERROR
                if phase == MessageSendFailureStage.PREPARATION
                else self.FINALIZATION_ERROR
                if phase == MessageSendFailureStage.FINALIZATION
                else self.SEND_ERROR
            )
            if phase == MessageSendFailureStage.PREPARATION and validated.source_message is not None:
                await self._record_preparation_failure_async(operation=operation, validated=validated)
            await self._fail_unsettled_branches_async(operation=operation, branches=branches, error=self.SEND_ERROR)
            operation.status.state = MessageSendState.FAILED
        finally:
            self._finish_send(operation)

    def _on_send_done(self, *, operation: _Send, task: asyncio.Task[None]) -> None:
        if operation.finished_at is not None:
            return
        # Cancellation before the coroutine's first step never enters its try/finally.
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error("Message operation task failed", exc_info=(type(error), error, error.__traceback__))
        operation.status.state = MessageSendState.FAILED
        operation.status.error = self.INTERRUPTED_ERROR
        operation.status.failure_stage = MessageSendFailureStage.INTERRUPTED
        operation.failure = self.INTERRUPTED_ERROR
        self._finish_send(operation)

    def _finish_send(self, operation: _Send) -> None:
        operation.reservation.release()
        operation.finished_at = time.monotonic()
        operation.task = None
        self._terminal[operation.status.send_id] = operation.finished_at
        self._expire_terminal_sends()

    async def _prepare_async(self, *, operation: _Send, validated: _ValidatedSend) -> list[_PreparedBranch]:
        history: list[Message] = []
        if validated.request.count > 1:
            history = list(
                await asyncio.to_thread(
                    self._memory.get_conversation_messages, conversation_id=validated.source.conversation_id
                )
            )
            sequence = max((message.sequence for message in history), default=-1) + 1
        else:
            existing = await asyncio.to_thread(
                self._memory.get_message_pieces, conversation_id=validated.source.conversation_id
            )
            sequence = max((piece.sequence for piece in existing), default=-1) + 1
        message = await self._prepare_message_async(
            request=validated.request, conversation_id=validated.source.conversation_id, sequence=sequence
        )
        validated.source_message = message
        if validated.request.request_converter_mode == RequestConverterMode.SHARED:
            await PromptNormalizer().convert_values_async(
                converter_configurations=validated.request_configurations, message=message
            )
        if validated.request.count == 1:
            branch = _PreparedBranch(
                message=message, status=MessageSendBranch(conversation_id=validated.source.conversation_id)
            )
            operation.status.branches = [branch.status]
            return [branch]
        conversations, pieces, branches = await asyncio.to_thread(
            self._prepare_copies,
            history=list(history),
            source=validated.source,
            message=message,
            count=validated.request.count,
        )
        operation.reservation.add_conversations([conversation.conversation_id for conversation in conversations])
        registration = asyncio.create_task(
            asyncio.to_thread(
                self._memory.add_conversation_branches_to_attack,
                attack_result_id=operation.status.attack_result_id,
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
                operation.status.branches = [branch.status for branch in branches]
                await self._fail_unsettled_branches_async(
                    operation=operation, branches=branches, error=self.INTERRUPTED_ERROR
                )
            raise
        if not stored:
            raise MessageSendNotFoundError("Attack was removed before branch preparation committed")
        operation.status.branches = [branch.status for branch in branches]
        return branches

    def _prepare_copies(
        self, *, history: list[Message], source: Conversation, message: Message, count: int
    ) -> tuple[list[Conversation], list[MessagePiece], list[_PreparedBranch]]:
        conversations: list[Conversation] = []
        pieces: list[MessagePiece] = []
        branches = [_PreparedBranch(message=message, status=MessageSendBranch(conversation_id=source.conversation_id))]
        for _ in range(count - 1):
            conversation_id, copied_pieces = self._memory.duplicate_messages(messages=history)
            conversations.append(source.model_copy(deep=True, update={"conversation_id": conversation_id}))
            pieces.extend(copied_pieces)
            branch_message = message.duplicate()
            for piece in branch_message.message_pieces:
                piece.conversation_id = conversation_id
            branches.append(
                _PreparedBranch(message=branch_message, status=MessageSendBranch(conversation_id=conversation_id))
            )
        return conversations, pieces, branches

    async def _send_branch_async(self, *, operation: _Send, validated: _ValidatedSend, branch: _PreparedBranch) -> None:
        per_branch = validated.request.request_converter_mode == RequestConverterMode.PER_BRANCH
        exclusive = bool(
            validated.target._max_requests_per_minute
            or validated.response_configurations
            or (per_branch and validated.request_configurations)
        )
        try:
            async with self._scheduler.operation_async(exclusive=exclusive):
                branch.status.state = MessageSendBranchState.SENDING
                operation.status.state = MessageSendState.RUNNING
                normalizer = PromptNormalizer()
                try:
                    if per_branch:
                        await normalizer.convert_values_async(
                            converter_configurations=validated.request_configurations, message=branch.message
                        )
                    response = await normalizer.send_prompt_async(
                        message=branch.message,
                        target=validated.target,
                        conversation_id=branch.status.conversation_id,
                        request_converter_configurations=[],
                        response_converter_configurations=validated.response_configurations,
                    )
                    branch.status.state = (
                        MessageSendBranchState.FAILED if response.is_error() else MessageSendBranchState.COMPLETED
                    )
                    branch.status.error = self.SEND_ERROR if response.is_error() else None
                    branch.last_response_id = next(
                        (str(piece.id) for piece in reversed(response.message_pieces) if piece.role == "assistant"),
                        None,
                    )
                except Exception as exc:
                    logger.exception(
                        "Send failed for operation '%s' branch '%s'",
                        operation.status.send_id,
                        branch.status.conversation_id,
                    )
                    if not await self._settle_branch_async(branch=branch, error=self.SEND_ERROR):
                        operation.failure = str(exc)
                if branch.last_response_id:
                    operation.last_response_id = branch.last_response_id
        except asyncio.CancelledError:
            await self._settle_branch_async(branch=branch, error=self.INTERRUPTED_ERROR)
            raise
        except Exception as exc:
            logger.exception("Could not settle branch '%s'", branch.status.conversation_id)
            operation.failure = str(exc)
            branch.status.state = MessageSendBranchState.FAILED
            branch.status.error = "Send failed and its saved status could not be read. Check server logs."
        finally:
            operation.reservation.release_conversation(branch.status.conversation_id)

    async def _settle_branch_async(self, *, branch: _PreparedBranch, error: str) -> bool:
        pieces = await self._read_new_pieces_async(branch)
        has_stored_error = any(piece.to_message().is_error() for piece in pieces)
        if not has_stored_error:
            try:
                await self._persist_failure_async(branch=branch, pieces=pieces, error=error)
                pieces = await self._read_new_pieces_async(branch)
            except Exception:
                logger.exception("Failed to save error evidence for branch '%s'", branch.status.conversation_id)
                error += " Failure details could not be saved."
        branch.last_response_id = next((str(piece.id) for piece in reversed(pieces) if piece.role == "assistant"), None)
        branch.status.error = error
        branch.status.state = MessageSendBranchState.FAILED
        return any(piece.to_message().is_error() for piece in pieces)

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

    async def _record_preparation_failure_async(self, *, operation: _Send, validated: _ValidatedSend) -> None:
        assert validated.source_message is not None
        branch = _PreparedBranch(
            message=validated.source_message,
            status=MessageSendBranch(conversation_id=validated.source.conversation_id),
        )
        try:
            await asyncio.to_thread(self._memory.add_conversation_to_memory, conversation=validated.source)
            operation.status.branches = [branch.status]
            await self._settle_branch_async(branch=branch, error=self.PREPARATION_ERROR)
        except Exception:
            logger.exception("Could not persist preparation failure for operation '%s'", operation.status.send_id)
            branch.status.state = MessageSendBranchState.FAILED
            branch.status.error = "Send preparation failed and its error details could not be saved."

    async def _fail_unsettled_branches_async(
        self, *, operation: _Send, branches: list[_PreparedBranch], error: str
    ) -> None:
        for branch in branches:
            if branch.status.state in (MessageSendBranchState.COMPLETED, MessageSendBranchState.FAILED):
                continue
            try:
                await self._settle_branch_async(branch=branch, error=error)
            except Exception:
                logger.exception("Could not persist interruption for operation '%s'", operation.status.send_id)
                branch.status.state = MessageSendBranchState.FAILED
                branch.status.error = error + " Failure details could not be saved."

    def _expire_terminal_sends(self) -> None:
        oldest_allowed = time.monotonic() - self.TERMINAL_TTL_SECONDS
        while self._terminal:
            send_id, finished_at = next(iter(self._terminal.items()))
            if finished_at > oldest_allowed and len(self._terminal) <= self.MAX_TERMINAL_SENDS:
                break
            self._terminal.pop(send_id)
            operation = self._sends.pop(send_id)
            self._submissions.pop((operation.status.attack_result_id, operation.submission_id))

    def _validate_target_match(
        self, *, attack_identifier: ComponentIdentifier | None, request: AddMessageRequest
    ) -> None:
        """
        Validate that the request target matches the attack's stored target.

        Raises:
            ValueError: If the target in the request doesn't match the attack's target.
        """
        if not request.send or not request.target_registry_name:
            return

        stored_target_id = attack_identifier.get_child("objective_target") if attack_identifier else None
        if not stored_target_id:
            return

        target_service = get_target_service()
        request_target_obj = target_service.get_target_object(target_registry_name=request.target_registry_name)
        if not request_target_obj:
            return

        request_target_id = request_target_obj.get_identifier()
        if stored_target_id.hash != request_target_id.hash:
            raise ValueError(
                f"Target mismatch: attack was created with {stored_target_id.unique_name} "
                f"but request uses {request_target_id.unique_name}. "
                f"Create a new attack to use a different target."
            )

    async def _update_attack_after_message_async(
        self,
        *,
        attack_result_id: str,
        last_response_id: str | None,
        request_converter_configurations: list[ConverterConfiguration],
        response_converter_configurations: list[ConverterConfiguration],
    ) -> None:
        """
        Update attack recency and converter tracking after a message is added.

        Bumps the attack's ``timestamp`` column (the single indexed recency key) so the edited
        conversation re-floats to the top of the History view.

        Args:
            attack_result_id: The attack result to update.
            last_response_id: The latest target response piece ID, if one was stored.
            request_converter_configurations: Resolved request converter configurations used for this message.
            response_converter_configurations: Resolved response converter configurations used for this message.

        Raises:
            ValueError: If the attack disappeared before its metadata could be updated.
        """
        async with self._message_metadata_lock:
            results = await asyncio.to_thread(self._memory.get_attack_results, attack_result_ids=[attack_result_id])
            if not results:
                raise ValueError(f"Attack '{attack_result_id}' not found after message send")
            update_fields = self._build_message_update_fields(
                ar=results[0],
                last_response_id=last_response_id,
                request_converter_configurations=request_converter_configurations,
                response_converter_configurations=response_converter_configurations,
            )
            stored = await asyncio.to_thread(
                self._memory.update_attack_result_by_id,
                attack_result_id=attack_result_id,
                update_fields=update_fields,
            )
            if not stored:
                raise ValueError(f"Attack '{attack_result_id}' not found after message send")

    def _build_message_update_fields(
        self,
        *,
        ar: AttackResult,
        last_response_id: str | None,
        request_converter_configurations: list[ConverterConfiguration],
        response_converter_configurations: list[ConverterConfiguration],
    ) -> dict[str, Any]:
        """
        Merge converter usage into the latest attack metadata without changing its membership.

        Returns:
            dict[str, Any]: Recency, response, and converter fields to update.
        """
        update_fields: dict[str, Any] = {"timestamp": datetime.now(UTC)}
        if last_response_id:
            update_fields["last_response_id"] = last_response_id

        request_converter_ids = self._get_converter_identifiers(configurations=request_converter_configurations)
        response_converter_ids = self._get_converter_identifiers(configurations=response_converter_configurations)
        if request_converter_ids or response_converter_ids:
            attack_strategy_identifier = ar.get_attack_strategy_identifier()
            if attack_strategy_identifier and ar.atomic_attack_identifier:
                attack_id = AttackIdentifier.from_component_identifier(attack_strategy_identifier)
                merged_request_converters = self._merge_attack_result_converter_identifiers(
                    existing=attack_id.request_converters,
                    additions=request_converter_ids,
                )
                merged_response_converters = self._merge_attack_result_converter_identifiers(
                    existing=attack_id.response_converters,
                    additions=response_converter_ids,
                )
                if (
                    merged_request_converters != attack_id.request_converters
                    or merged_response_converters != attack_id.response_converters
                ):
                    new_attack_id = self._replace_converter_pipelines(
                        attack_id,
                        request_converters=merged_request_converters,
                        response_converters=merged_response_converters,
                    )
                    new_atomic = self._replace_attack_in_atomic(
                        AtomicAttackIdentifier.from_component_identifier(ar.atomic_attack_identifier),
                        attack=new_attack_id,
                    )
                    update_fields["atomic_attack_identifier"] = new_atomic.model_dump()

        return update_fields

    @staticmethod
    def _replace_converter_pipelines(
        attack_id: AttackIdentifier,
        *,
        request_converters: list[ConverterIdentifier],
        response_converters: list[ConverterIdentifier],
    ) -> AttackIdentifier:
        """
        Return a copy of ``attack_id`` with its converter pipelines replaced.

        Reconstructed through the constructor (not ``model_copy``) so the
        after-validator re-mirrors the typed converters into ``children`` and
        recomputes the content hash. All other params/children/attributes are
        preserved, so the identifier hashes identically apart from the converters.

        Returns:
            AttackIdentifier: A new identifier with the given converter pipelines.
        """
        return AttackIdentifier(
            class_name=attack_id.class_name,
            class_module=attack_id.class_module,
            params=dict(attack_id.params),
            children=dict(attack_id.children),
            attributes=dict(attack_id.attributes),
            request_converters=request_converters,
            response_converters=response_converters,
        )

    @staticmethod
    def _merge_attack_result_converter_identifiers(
        *,
        existing: list[ConverterIdentifier],
        additions: list[ConverterIdentifier],
    ) -> list[ConverterIdentifier]:
        """
        Merge converter usage into the aggregate attack result metadata.

        Attack result converter lists record which converters the attack used, not
        the exact converter pipeline for each message. Keep the first occurrence of
        each identifier across messages while preserving first-use order.

        Args:
            existing: Converter identifiers already recorded on the attack result.
            additions: Converter identifiers used by the new message.

        Returns:
            list[ConverterIdentifier]: Aggregate converter identifiers in first-use order.
        """
        merged = list(existing)
        existing_hashes = {converter.hash for converter in existing}
        for converter in additions:
            if converter.hash not in existing_hashes:
                merged.append(converter)
                existing_hashes.add(converter.hash)
        return merged

    @staticmethod
    def _replace_attack_in_atomic(
        atomic: AtomicAttackIdentifier, *, attack: AttackIdentifier
    ) -> AtomicAttackIdentifier:
        """
        Return a copy of ``atomic`` with its nested attack strategy replaced.

        Handles both the current nested shape (``atomic -> attack_technique ->
        attack``) and the legacy flat shape (``atomic -> attack``). Everything
        else is preserved so the composite identifier hashes identically apart
        from the swapped attack node.

        Returns:
            AtomicAttackIdentifier: A new composite identifier wrapping ``attack``.
        """
        technique = atomic.attack_technique
        if technique is not None:
            new_technique = AttackTechniqueIdentifier(
                class_name=technique.class_name,
                class_module=technique.class_module,
                params=dict(technique.params),
                children=dict(technique.children),
                attributes=dict(technique.attributes),
                attack=attack,
            )
            return AtomicAttackIdentifier(
                class_name=atomic.class_name,
                class_module=atomic.class_module,
                params=dict(atomic.params),
                children=dict(atomic.children),
                attributes=dict(atomic.attributes),
                attack_technique=new_technique,
            )
        # Legacy flat shape: the attack strategy lives in children["attack"].
        atomic_children = dict(atomic.children)
        atomic_children["attack"] = attack
        return AtomicAttackIdentifier(
            class_name=atomic.class_name,
            class_module=atomic.class_module,
            params=dict(atomic.params),
            children=atomic_children,
            attributes=dict(atomic.attributes),
        )

    @staticmethod
    async def _persist_base64_pieces_async(request: AddMessageRequest) -> None:
        """
        Persist base64-encoded non-text pieces to disk, updating values in-place.

        The frontend sends binary media (images, audio, etc.) as base64 strings
        with a ``*_path`` data_type.  The PyRIT target layer expects ``*_path``
        values to be **file paths**, so we decode the base64 data, write it to
        the results store, and replace the request values with the resulting
        file path before the message is built.

        If the value is already an HTTP(S) URL (e.g. an Azure Blob Storage URL
        from a remixed/copied message), it is kept as-is since the file already
        exists in storage.
        """
        for piece in request.pieces:
            # Only persist *_path types (image_path, audio_path, video_path, binary_path).
            # Other non-text types (url, reasoning, function_call, tool_call, etc.)
            # are text-like and must not be base64-decoded.
            if not piece.data_type.endswith("_path"):
                continue

            result = await persist_media_value_async(
                value=piece.original_value,
                data_type=cast("PromptDataType", piece.data_type),
                mime_type=piece.mime_type,
                serializer_factory=data_serializer_factory,
            )
            if result.resolved:
                piece.original_value = result.value
                if piece.converted_value is None:
                    piece.converted_value = result.value

    async def _prepare_message_async(
        self, *, request: AddMessageRequest, conversation_id: str, sequence: int
    ) -> Message:
        """
        Resolve uploaded media once, then map the request through the canonical mapper.

        Returns:
            Message: A fresh request ready for conversion.
        """
        await self._persist_base64_pieces_async(request)
        await asyncio.to_thread(self._resolve_video_remix_metadata, request)
        return request_to_pyrit_message(request=request, conversation_id=conversation_id, sequence=sequence)

    async def _store_message_only_async(
        self,
        *,
        conversation_id: str,
        request: AddMessageRequest,
        sequence: int,
        target_identifier: ComponentIdentifier | None = None,
    ) -> None:
        """Store message without sending (send=False)."""
        await self._persist_base64_pieces_async(request)
        await asyncio.to_thread(
            self._memory.add_conversation_to_memory,
            conversation=Conversation(conversation_id=conversation_id, target_identifier=target_identifier),
        )
        for p in request.pieces:
            piece = request_piece_to_pyrit_message_piece(
                piece=p,
                role=request.role,
                conversation_id=conversation_id,
                sequence=sequence,
            )
            await asyncio.to_thread(self._memory.add_message_pieces_to_memory, message_pieces=[piece])

    def _resolve_video_remix_metadata(self, request: AddMessageRequest) -> None:
        """
        Auto-resolve video_id metadata for remix mode.

        When a video_path piece is carried over from a previous conversation
        (via original_prompt_id) alongside a text piece, the video target
        requires video_id in the text piece's prompt_metadata. This method
        looks up the original piece's metadata and propagates the video_id.
        """
        video_pieces = [p for p in request.pieces if p.data_type == "video_path"]
        if not video_pieces:
            return

        text_piece = next((p for p in request.pieces if p.data_type == "text"), None)
        if not text_piece:
            return

        # Already has video_id — nothing to resolve
        if text_piece.prompt_metadata and text_piece.prompt_metadata.get("video_id"):
            return

        # Try to resolve video_id from the original prompt piece
        for vp in video_pieces:
            if not vp.original_prompt_id:
                continue
            original_pieces = self._memory.get_message_pieces(prompt_ids=[vp.original_prompt_id])
            if not original_pieces:
                continue
            video_id = (original_pieces[0].prompt_metadata or {}).get("video_id")
            if video_id:
                if text_piece.prompt_metadata is None:
                    text_piece.prompt_metadata = {}
                text_piece.prompt_metadata["video_id"] = video_id
                # Also set video_id on the video piece itself
                if vp.prompt_metadata is None:
                    vp.prompt_metadata = {}
                vp.prompt_metadata["video_id"] = video_id
                return

    def _resolve_request_converter_configs(self, *, request: AddMessageRequest) -> list[ConverterConfiguration]:
        """
        Resolve legacy or structured request converter configurations.

        Returns:
            list[ConverterConfiguration]: Resolved request configurations.
        """
        if request.converter_ids is not None:
            print_deprecation_message(
                old_item="AddMessageRequest.converter_ids",
                new_item="AddMessageRequest.request_converter_configurations",
                removed_in="1.3.0",
            )
        if request.converter_ids:
            converters = get_converter_service().get_converter_objects_for_ids(converter_ids=request.converter_ids)
            return ConverterConfiguration.from_converters(converters=converters)

        return self._resolve_converter_configs(configurations=request.request_converter_configurations)

    def _resolve_converter_configs(
        self,
        *,
        configurations: list[ConverterConfigurationRequest] | None,
    ) -> list[ConverterConfiguration]:
        """
        Resolve registry-backed converter configurations.

        Returns:
            list[ConverterConfiguration]: Resolved configurations in request order.
        """
        if not configurations:
            return []
        converter_service = get_converter_service()
        return [
            ConverterConfiguration(
                converters=converter_service.get_converter_objects_for_ids(converter_ids=configuration.converter_ids),
                indexes_to_apply=configuration.indexes_to_apply,
                prompt_data_types_to_apply=configuration.prompt_data_types_to_apply,
            )
            for configuration in configurations or []
        ]

    @staticmethod
    def _exclude_preconverted_piece_indexes(
        *,
        configurations: list[ConverterConfiguration],
        preconverted_indexes: set[int],
        piece_count: int,
    ) -> list[ConverterConfiguration]:
        """
        Exclude client-preconverted pieces from request converter configurations.

        Returns:
            list[ConverterConfiguration]: Configurations that still apply to at least one piece.
        """
        if not preconverted_indexes:
            return configurations

        filtered_configurations: list[ConverterConfiguration] = []
        for configuration in configurations:
            configured_indexes = configuration.indexes_to_apply
            candidate_indexes = configured_indexes if configured_indexes else range(piece_count)
            eligible_indexes = [index for index in candidate_indexes if index not in preconverted_indexes]
            if not eligible_indexes:
                continue
            filtered_configurations.append(
                ConverterConfiguration(
                    converters=configuration.converters,
                    indexes_to_apply=eligible_indexes,
                    prompt_data_types_to_apply=configuration.prompt_data_types_to_apply,
                )
            )
        return filtered_configurations

    @staticmethod
    def _get_converter_identifiers(*, configurations: list[ConverterConfiguration]) -> list[ConverterIdentifier]:
        """
        Flatten resolved converter identifiers in configuration order.

        Returns:
            list[ConverterIdentifier]: The converter identifiers.
        """
        return [
            ConverterIdentifier.from_component_identifier(converter.get_identifier())
            for configuration in configurations
            for converter in configuration.converters
        ]


@lru_cache(maxsize=1)
def get_message_send_service() -> MessageSendService:
    """Return the worker-local service without starting any background sends."""
    return MessageSendService()


async def shutdown_message_sends_async() -> None:
    """Shut down only an already-used service; do not initialize memory during shutdown."""
    if get_message_send_service.cache_info().currsize:
        await get_message_send_service().shutdown_async()
        get_message_send_service.cache_clear()
        get_manual_send_scheduler.cache_clear()
