# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Constructor-injected trusted source capture, not target-owned scoring or conversation policy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from pyrit.models import Message
    from pyrit.models.native_cyber import NativeAgentEvidence
    from pyrit.prompt_target.inspect_ghcp_target import InspectGhcpTurn
    from pyrit.prompt_target.native_agent_target import NativeAgentSession


class EvaluationFeedbackCapture(Protocol):
    """Reviewed prepared-send and source-capture seams installed only by the trusted runtime."""

    @property
    def policy_sha256(self) -> str:
        """The installed feedback policy identity."""
        ...

    def bind_native(self, *, session: NativeAgentSession) -> None:
        """Bind the native evidence probe without creating an SDK session or provider."""
        ...

    def bind_inspect(
        self, *, source_probe: Callable[[], NativeAgentEvidence], max_turns: int, operator_steps: bool = False
    ) -> None:
        """Bind an owning harness's live SDK capture; retained transport frames alone are insufficient."""
        ...

    async def before_send_async(self, *, request: Message) -> int:
        """Require the attack's exact ready/reserved input before transport delivery."""
        ...

    async def capture_native_async(
        self,
        *,
        request: Message,
        responses: Sequence[Message],
        evidence: NativeAgentEvidence,
    ) -> None:
        """Retain source identities and actual response-row mappings before normalizer writes."""
        ...

    async def capture_inspect_async(self, *, request: Message, response: Message, turn: InspectGhcpTurn) -> None:
        """Retain the reviewed source frame, without claiming original Mode 1 or token coverage."""
        ...
