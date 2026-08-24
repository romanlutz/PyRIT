# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import io
from collections.abc import Awaitable, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions import RateLimitException, pyrit_target_retry
from pyrit.executor.attack.component.prepended_history_send_context import PrependedHistorySendContext
from pyrit.memory import MemoryInterface
from pyrit.models import Message
from pyrit.prompt_target import (
    PromptTarget,
    ProviderAttempt,
    TargetCapabilities,
    TargetConfiguration,
    TextTarget,
    limit_requests_per_minute,
)
from pyrit.prompt_target.common.provider_attempt import _get_provider_attempt_or_legacy_noop


class _LegacyTarget(PromptTarget):
    def __init__(self, *, rpm: int | None = None) -> None:
        super().__init__(max_requests_per_minute=rpm)
        self.calls = 0

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        self.calls += 1
        return []


class _LegacyDecoratedTarget(_LegacyTarget):
    @limit_requests_per_minute
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)


class _TokenAwareTarget(PromptTarget):
    def __init__(
        self,
        *,
        operation: Callable[[], Awaitable[None]],
        setup: Callable[[], Awaitable[None]] | None = None,
        rpm: int | None = None,
    ) -> None:
        super().__init__(max_requests_per_minute=rpm)
        self._operation = operation
        self._setup = setup
        self.calls = 0

    async def _send_prompt_to_target_async(
        self,
        *,
        normalized_conversation: list[Message],
        provider_attempt: ProviderAttempt,
    ) -> list[Message]:
        self.calls += 1
        if self._setup:
            await self._setup()
        await provider_attempt.run_async(operation=self._operation)
        return []


class _KwargsTokenTarget(PromptTarget):
    async def _send_prompt_to_target_async(
        self,
        *,
        normalized_conversation: list[Message],
        **kwargs: object,
    ) -> list[Message]:
        provider_attempt = kwargs["provider_attempt"]
        assert isinstance(provider_attempt, ProviderAttempt)
        await provider_attempt.start_async()
        return []


class _MigratedTarget(PromptTarget):
    def __init__(self, *, rpm: int | None = None) -> None:
        super().__init__(max_requests_per_minute=rpm)
        self.calls = 0

    async def _send_prompt_to_target_async(
        self,
        *,
        normalized_conversation: list[Message],
        provider_attempt: ProviderAttempt | None = None,
    ) -> list[Message]:
        provider_attempt = _get_provider_attempt_or_legacy_noop(provider_attempt=provider_attempt)

        async def send_async() -> list[Message]:
            self.calls += 1
            return []

        return await provider_attempt.run_async(operation=send_async)


class _LegacyMigratedOverride(_MigratedTarget):
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        self.calls += 1
        return []


class _LegacySuperOverride(_MigratedTarget):
    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)


def _message(*, conversation_id: str = "conversation") -> Message:
    message = Message.from_prompt(prompt="request", role="user")
    message.get_piece().conversation_id = conversation_id
    return message


def _context(*, target: PromptTarget, conversation_id: str = "conversation") -> PrependedHistorySendContext:
    seed = _message(conversation_id=conversation_id)
    memory = MagicMock(spec=MemoryInterface)
    memory.get_conversation_messages.return_value = [seed]
    target._memory = memory
    target._configuration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )
    return PrependedHistorySendContext(
        conversation_id=conversation_id,
        seed_message_ids=(seed.get_piece().id,),
        replay_seed_each_send=False,
    )


@pytest.mark.usefixtures("patch_central_database")
async def test_legacy_target_starts_attempt_and_rate_limits_at_method_entry() -> None:
    target = _LegacyTarget(rpm=30)
    context = _context(target=target)

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        await target.send_prompt_async(message=_message(), send_context=context)

    sleep.assert_awaited_once_with(2.0)
    assert target.calls == 1
    assert context.provider_attempt_count == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_legacy_decorated_target_does_not_wait_twice() -> None:
    target = _LegacyDecoratedTarget(rpm=30)
    context = _context(target=target)

    with (
        pytest.warns(DeprecationWarning, match="limit_requests_per_minute"),
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        await target.send_prompt_async(message=_message(), send_context=context)

    sleep.assert_awaited_once_with(2.0)
    assert context.provider_attempt_count == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_token_aware_target_setup_failure_does_not_start_attempt() -> None:
    async def fail_setup_async() -> None:
        raise RuntimeError("setup failed")

    target = _TokenAwareTarget(operation=AsyncMock(), setup=fail_setup_async, rpm=30)
    context = _context(target=target)

    with (
        pytest.raises(RuntimeError, match="setup failed"),
        patch("asyncio.sleep", new_callable=AsyncMock) as sleep,
    ):
        await target.send_prompt_async(message=_message(), send_context=context)

    sleep.assert_not_awaited()
    assert context.provider_attempt_count == 0
    assert not context.is_seed_consumed


@pytest.mark.usefixtures("patch_central_database")
async def test_kwargs_target_receives_provider_attempt() -> None:
    target = _KwargsTokenTarget()
    context = _context(target=target)

    await target.send_prompt_async(message=_message(), send_context=context)

    assert context.provider_attempt_count == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_cancellation_during_rate_limit_does_not_start_attempt() -> None:
    wait_started = asyncio.Event()

    async def wait_for_rate_limit_async() -> None:
        wait_started.set()
        await asyncio.Event().wait()

    target = _TokenAwareTarget(operation=AsyncMock())
    target._wait_for_rate_limit_async = wait_for_rate_limit_async  # type: ignore[method-assign]
    context = _context(target=target)

    task = asyncio.create_task(target.send_prompt_async(message=_message(), send_context=context))
    await wait_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.provider_attempt_count == 0
    assert not context.is_seed_consumed
    context.begin_send()
    context.finish_send()


@pytest.mark.usefixtures("patch_central_database")
async def test_cancellation_after_token_start_is_attempted() -> None:
    operation_started = asyncio.Event()

    async def operation_async() -> None:
        operation_started.set()
        await asyncio.Event().wait()

    target = _TokenAwareTarget(operation=operation_async)
    context = _context(target=target)

    task = asyncio.create_task(target.send_prompt_async(message=_message(), send_context=context))
    await operation_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert context.provider_attempt_count == 1
    assert context.is_seed_consumed


@pytest.mark.usefixtures("patch_central_database")
async def test_concurrent_token_starts_wait_and_mark_once() -> None:
    async def start_twice_async() -> None:
        return None

    class ConcurrentStartTarget(PromptTarget):
        async def _send_prompt_to_target_async(
            self,
            *,
            normalized_conversation: list[Message],
            provider_attempt: ProviderAttempt,
        ) -> list[Message]:
            await asyncio.gather(provider_attempt.start_async(), provider_attempt.start_async())
            await start_twice_async()
            return []

    target = ConcurrentStartTarget(max_requests_per_minute=30)
    context = _context(target=target)

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        await target.send_prompt_async(message=_message(), send_context=context)

    sleep.assert_awaited_once_with(2.0)
    assert context.provider_attempt_count == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_target_retries_share_one_wait_and_attempt() -> None:
    class RetryingTarget(PromptTarget):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        @pyrit_target_retry
        async def _send_prompt_to_target_async(
            self,
            *,
            normalized_conversation: list[Message],
            provider_attempt: ProviderAttempt,
        ) -> list[Message]:
            await provider_attempt.start_async()
            self.calls += 1
            if self.calls == 1:
                raise RateLimitException(message="retry")
            return []

    target = RetryingTarget()
    context = _context(target=target)
    target._wait_for_rate_limit_async = AsyncMock()  # type: ignore[method-assign]

    with (
        patch("pyrit.exceptions.exception_classes._get_retry_wait_min_seconds", return_value=0),
        patch("pyrit.exceptions.exception_classes._get_retry_wait_max_seconds", return_value=0),
        patch("pyrit.exceptions.exception_classes.get_retry_max_num_attempts", return_value=2),
    ):
        await target.send_prompt_async(message=_message(), send_context=context)

    target._wait_for_rate_limit_async.assert_awaited_once_with()
    assert target.calls == 2
    assert context.provider_attempt_count == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_child_task_token_start_is_visible_to_parent_send() -> None:
    async def operation_async() -> None:
        return None

    class ChildTaskTarget(PromptTarget):
        async def _send_prompt_to_target_async(
            self,
            *,
            normalized_conversation: list[Message],
            provider_attempt: ProviderAttempt,
        ) -> list[Message]:
            await asyncio.create_task(provider_attempt.run_async(operation=operation_async))
            raise RuntimeError("provider failed")

    target = ChildTaskTarget()
    context = _context(target=target)

    with pytest.raises(RuntimeError, match="provider failed"):
        await target.send_prompt_async(message=_message(), send_context=context)

    assert context.provider_attempt_count == 1
    assert context.is_seed_consumed


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("target_type", [_LegacyMigratedOverride, _LegacySuperOverride])
async def test_legacy_override_of_migrated_target_uses_single_compatibility_attempt(
    target_type: type[_MigratedTarget],
) -> None:
    target = target_type(rpm=30)
    context = _context(target=target)

    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        await target.send_prompt_async(message=_message(), send_context=context)

    sleep.assert_awaited_once_with(2.0)
    assert target.calls == 1
    assert context.provider_attempt_count == 1


@pytest.mark.usefixtures("patch_central_database")
async def test_distinct_history_contexts_can_send_concurrently() -> None:
    entered = 0
    both_entered = asyncio.Event()
    release = asyncio.Event()

    async def operation_async() -> None:
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await release.wait()

    target = _TokenAwareTarget(operation=operation_async)
    target._configuration = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_system_prompt=True,
            supports_editable_history=True,
        )
    )
    seeds = {
        conversation_id: _message(conversation_id=conversation_id)
        for conversation_id in ("conversation-a", "conversation-b")
    }
    memory = MagicMock(spec=MemoryInterface)
    memory.get_conversation_messages.side_effect = lambda conversation_id: [seeds[conversation_id]]
    target._memory = memory
    contexts = {
        conversation_id: PrependedHistorySendContext(
            conversation_id=conversation_id,
            seed_message_ids=(seed.get_piece().id,),
            replay_seed_each_send=False,
        )
        for conversation_id, seed in seeds.items()
    }

    tasks = [
        asyncio.create_task(
            target.send_prompt_async(
                message=_message(conversation_id=conversation_id),
                send_context=contexts[conversation_id],
            )
        )
        for conversation_id in contexts
    ]
    await asyncio.wait_for(both_entered.wait(), timeout=1)
    release.set()
    await asyncio.gather(*tasks)

    assert [context.provider_attempt_count for context in contexts.values()] == [1, 1]


@pytest.mark.usefixtures("patch_central_database")
async def test_text_target_does_not_start_provider_attempt() -> None:
    target = TextTarget(text_stream=io.StringIO())
    context = _context(target=target)

    await target.send_prompt_async(message=_message(), send_context=context)

    assert context.provider_attempt_count == 0
    assert not context.is_seed_consumed


@pytest.mark.usefixtures("patch_central_database")
async def test_target_type_error_is_not_retried_as_legacy_signature() -> None:
    async def raise_type_error_async() -> None:
        raise TypeError("provider bug")

    target = _TokenAwareTarget(operation=raise_type_error_async)
    context = _context(target=target)

    with pytest.raises(TypeError, match="provider bug"):
        await target.send_prompt_async(message=_message(), send_context=context)

    assert target.calls == 1
    assert context.provider_attempt_count == 1
