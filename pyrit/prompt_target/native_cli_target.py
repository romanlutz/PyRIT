# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""One-shot PyRIT target for sandbox-owned native coding CLI JSONL runs."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrit.models import Message, MessagePiece, construct_response_from_request
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.prompt_target.common.target_capabilities import (
    CapabilityHandlingPolicy,
    CapabilityName,
    TargetCapabilities,
    UnsupportedCapabilityBehavior,
)
from pyrit.prompt_target.common.target_configuration import TargetConfiguration
from pyrit.prompt_target.native_cli_models import NativeCliEventKind
from pyrit.prompt_target.native_cli_transport import NativeCliRunner

if TYPE_CHECKING:
    from pyrit.models import ComponentIdentifier
    from pyrit.prompt_target.native_cli_models import (
        NativeCliEvent,
        NativeCliObservation,
        NativeCliRawChunk,
        NativeCliRunConfig,
        NativeCliRunOutcome,
    )
    from pyrit.prompt_target.native_cli_transport import NativeCliEvidenceSink, SandboxProcessLauncher


@dataclass(frozen=True, kw_only=True)
class NativeCliTargetRun:
    """The actual CLI outcome and response sources, linked to caller-owned evidence."""

    outcome: NativeCliRunOutcome
    response_sources: tuple[NativeCliObservation, ...]
    evidence_sink: NativeCliEvidenceSink
    conversation_id: str | None


class NativeCliCoverageError(RuntimeError):
    """A completed process without fully observed provider/tool evidence."""

    def __init__(self, *, run: NativeCliTargetRun) -> None:
        """Retain the incomplete run for explicit caller inspection."""
        self.run = run
        super().__init__(
            f"Native CLI evidence is incomplete (exit code {run.outcome.exit_code}, "
            f"{len(run.outcome.gaps)} coverage gaps)."
        )


class _ObservedResponseSink:
    """Forward evidence durably before retaining only observed assistant text."""

    def __init__(self, *, sink: NativeCliEvidenceSink) -> None:
        """Attach the caller-owned raw/event sink."""
        self._sink = sink
        self._assistant: list[NativeCliObservation] = []
        self._final: NativeCliObservation | None = None

    async def record_raw_async(self, *, chunk: NativeCliRawChunk) -> None:
        """Forward unchanged bytes before any normalization."""
        await self._sink.record_raw_async(chunk=chunk)

    async def record_event_async(self, *, event: NativeCliEvent) -> None:
        """Retain observed text only after the caller's event sink accepts it."""
        await self._sink.record_event_async(event=event)
        observation = event.observation
        if observation.kind is NativeCliEventKind.MODEL_MESSAGE and observation.text:
            self._assistant.append(observation)
        elif observation.kind is NativeCliEventKind.RUN_FINISHED and observation.text:
            self._final = observation

    def response_sources(self) -> tuple[NativeCliObservation, ...]:
        """
        Select actual assistant blocks, or the observed final result when no block exists.

        Returns:
            tuple[NativeCliObservation, ...]: Nonempty observed text sources, if any.
        """
        if self._assistant:
            return tuple(self._assistant)
        return (self._final,) if self._final is not None else ()


class NativeCliTarget(PromptTarget):
    """Send one prepared user instruction to a caller-owned native CLI sandbox."""

    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=False,
            supports_multi_message_pieces=False,
            supports_system_prompt=False,
        ),
        policy=CapabilityHandlingPolicy(
            behaviors={
                CapabilityName.MULTI_TURN: UnsupportedCapabilityBehavior.RAISE,
                CapabilityName.SYSTEM_PROMPT: UnsupportedCapabilityBehavior.RAISE,
                CapabilityName.JSON_SCHEMA: UnsupportedCapabilityBehavior.RAISE,
            }
        ),
    )

    def __init__(
        self,
        *,
        run_config: NativeCliRunConfig,
        launcher: SandboxProcessLauncher,
        evidence_sink: NativeCliEvidenceSink,
    ) -> None:
        """Bind fixed text-only capabilities to an injected sandbox and durable sink."""
        super().__init__()
        self._run_config = run_config
        self._launcher = launcher
        self._evidence_sink = evidence_sink
        self._invoked = False
        self._send_lock = asyncio.Lock()
        self._last_run: NativeCliTargetRun | None = None

    @property
    def last_run(self) -> NativeCliTargetRun | None:
        """The completed run handle, including incomplete coverage when observed."""
        return self._last_run

    @property
    def evidence_sink(self) -> NativeCliEvidenceSink:
        """The caller-owned recorder, including evidence from an interrupted run."""
        return self._evidence_sink

    def _build_identifier(self) -> ComponentIdentifier:
        return self._create_identifier(
            params={
                "adapter": "native_cli_jsonl",
                "protocol": self._run_config.protocol.value,
                "cli_version": self._run_config.cli_version,
                "cli_profile": self._run_config.cli_profile,
                "agent_workdir": str(self._run_config.agent_workdir),
                "model_gateway_endpoint": self._run_config.model_gateway_endpoint,
                "max_steps": self._run_config.max_steps,
            }
        )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        """
        Return only observed text from a complete native run, or no response for a write-only task.

        Returns:
            list[Message]: Actual assistant text messages; empty when no text was observed.

        Raises:
            ValueError: If the prepared request or capabilities are unsupported.
            RuntimeError: If this target was already invoked.
            NativeCliCoverageError: If the process finished without full evidence coverage.
        """
        request = self._validate_one_shot_request(normalized_conversation=normalized_conversation)
        async with self._send_lock:
            if self._invoked:
                raise RuntimeError("Native CLI targets are one-shot; create a new target and sandbox for another run.")
            self._invoked = True
        observed_sink = _ObservedResponseSink(sink=self._evidence_sink)
        outcome = await NativeCliRunner(launcher=self._launcher, sink=observed_sink).run_async(
            config=self._run_config, prompt=request.get_piece().converted_value
        )
        sources = observed_sink.response_sources()
        self._last_run = NativeCliTargetRun(
            outcome=outcome,
            response_sources=sources,
            evidence_sink=self._evidence_sink,
            conversation_id=request.get_piece().conversation_id,
        )
        if not outcome.coverage_complete:
            raise NativeCliCoverageError(run=self._last_run)
        return [self._to_message(request_piece=request.get_piece(), observation=source) for source in sources]

    def _validate_one_shot_request(self, *, normalized_conversation: list[Message]) -> Message:
        if self.configuration is not type(self)._DEFAULT_CONFIGURATION:
            raise ValueError("Native CLI target capabilities cannot be overridden without proven CLI support.")
        if len(normalized_conversation) != 1:
            raise ValueError("Native CLI targets cannot replay history or manage a multi-turn conversation.")
        request = normalized_conversation[0]
        if request.api_role != "user" or len(request.message_pieces) != 1:
            raise ValueError("Native CLI targets require exactly one prepared user text piece.")
        piece = request.get_piece()
        if piece.converted_value_data_type != "text" or not piece.converted_value.strip():
            raise ValueError("Native CLI targets require nonempty prepared text, not a tool call or media.")
        if piece.prompt_metadata.get("response_format") == "json" or "json_schema" in piece.prompt_metadata:
            raise ValueError("Native CLI targets do not support structured response formats.")
        return request

    @staticmethod
    def _to_message(*, request_piece: MessagePiece, observation: NativeCliObservation) -> Message:
        text = observation.text
        if not text:
            raise ValueError("An observed native CLI response source must contain text.")
        metadata: dict[str, str | int] = {"native_cli_source_kind": observation.kind.value}
        metadata.update(
            {
                key: value
                for key, value in (
                    ("native_cli_source_event_id", observation.source_event_id),
                    ("native_cli_source_message_id", observation.source_message_id),
                    ("native_cli_source_session_id", observation.source_session_id),
                    ("native_cli_parent_tool_use_id", observation.parent_tool_use_id),
                )
                if value is not None
            }
        )
        return construct_response_from_request(
            request=request_piece,
            response_text_pieces=[text],
            prompt_metadata=metadata,
        )
