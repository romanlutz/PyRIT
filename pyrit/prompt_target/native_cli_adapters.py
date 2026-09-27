# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Conservative decoders for documented Codex and Claude Code JSONL profiles."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Protocol

from pyrit.prompt_target.native_cli_models import NativeCliEventKind as Kind
from pyrit.prompt_target.native_cli_models import NativeCliEventStatus as Status
from pyrit.prompt_target.native_cli_models import NativeCliObservation

if TYPE_CHECKING:
    from pydantic import JsonValue


def _text(value: JsonValue) -> str | None:
    return value if isinstance(value, str) and value else None


def _partial(*, detail: str, source_event_id: str | None = None) -> NativeCliObservation:
    return NativeCliObservation(kind=Kind.PARTIAL, detail=detail, source_event_id=source_event_id)


class NativeCliAdapter(Protocol):
    """Decode supported provider frames without launching a CLI or executing tools."""

    def decode(self, *, payload: dict[str, JsonValue]) -> tuple[NativeCliObservation, ...]:
        """Map one JSONL object to actual observations or explicit partial coverage."""
        ...


class CodexExecJsonAdapter:
    """Decode the documented ``codex exec --json`` item lifecycle."""

    _TOOL_TYPES: ClassVar[frozenset[str]] = frozenset(
        {"command_execution", "file_change", "mcp_tool_call", "web_search"}
    )
    _OTHER_ITEM_TYPES: ClassVar[frozenset[str]] = frozenset({"reasoning", "plan_update"})

    def decode(self, *, payload: dict[str, JsonValue]) -> tuple[NativeCliObservation, ...]:
        """
        Decode thread, turn, item, and error frames in the exec JSONL profile.

        Returns:
            tuple[NativeCliObservation, ...]: Observed actions or explicit coverage gaps.
        """
        kind = payload.get("type")
        if not isinstance(kind, str):
            return (_partial(detail="Codex exec JSONL frame is missing a string type."),)
        if kind == "thread.started":
            thread_id = _text(payload.get("thread_id"))
            if thread_id is None:
                return (_partial(detail="Codex thread.started is missing thread_id."),)
            return (
                NativeCliObservation(kind=Kind.SESSION_STARTED, source_session_id=thread_id, source_event_id=thread_id),
            )
        if kind == "turn.started":
            return (NativeCliObservation(kind=Kind.TURN_STARTED, status=Status.RUNNING),)
        if kind == "turn.completed":
            return (NativeCliObservation(kind=Kind.TURN_COMPLETED, status=Status.COMPLETED),)
        if kind in {"turn.failed", "error"}:
            return (
                NativeCliObservation(
                    kind=Kind.ERROR,
                    status=Status.FAILED,
                    detail=f"Codex {kind} was observed.",
                    text=_text(payload.get("message")),
                    result=payload.get("error"),
                ),
            )
        if kind in {"item.started", "item.updated", "item.completed"}:
            return self._decode_item(payload=payload, phase=kind)
        return (_partial(detail="Unrecognized Codex exec JSONL event type."),)

    def _decode_item(self, *, payload: dict[str, JsonValue], phase: str) -> tuple[NativeCliObservation, ...]:
        item = payload.get("item")
        if not isinstance(item, dict):
            return (_partial(detail=f"Codex {phase} is missing an item object."),)
        item_id, item_type = _text(item.get("id")), _text(item.get("type"))
        if item_id is None or item_type is None:
            return (_partial(detail=f"Codex {phase} item is missing its id or type."),)
        if item_type == "agent_message":
            if phase != "item.completed":
                return (NativeCliObservation(kind=Kind.PROGRESS, source_event_id=item_id),)
            text = item.get("text")
            if not isinstance(text, str):
                return (_partial(detail="Completed Codex agent_message has no text.", source_event_id=item_id),)
            return (NativeCliObservation(kind=Kind.MODEL_MESSAGE, source_event_id=item_id, text=text),)
        if item_type in self._OTHER_ITEM_TYPES:
            return (NativeCliObservation(kind=Kind.AUXILIARY, source_event_id=item_id, name=item_type),)
        if item_type in self._TOOL_TYPES:
            return self._decode_tool(item=item, item_id=item_id, item_type=item_type, phase=phase)
        return (_partial(detail=f"Unrecognized Codex item type: {item_type}.", source_event_id=item_id),)

    def _decode_tool(
        self, *, item: dict[str, JsonValue], item_id: str, item_type: str, phase: str
    ) -> tuple[NativeCliObservation, ...]:
        native_status = _text(item.get("status"))
        exit_code = item.get("exit_code")
        if exit_code is not None and type(exit_code) is not int:
            return (_partial(detail=f"Codex tool {item_id} has a non-integer exit_code.", source_event_id=item_id),)
        arguments = self._tool_arguments(item=item, item_type=item_type)
        base = NativeCliObservation(
            kind=Kind.PROGRESS,
            source_event_id=item_id,
            source_tool_id=item_id,
            source_status=native_status,
            name=item_type,
            arguments=arguments,
            exit_code=exit_code if type(exit_code) is int else None,
        )
        if phase == "item.updated":
            status = Status.RUNNING if native_status == "in_progress" else Status.UNKNOWN
            observation = replace(base, status=status)
            if status is Status.UNKNOWN:
                return (observation, _partial(detail=f"Codex tool {item_id} has an unknown update status."))
            return (observation,)
        if phase == "item.started":
            status = Status.RUNNING if native_status == "in_progress" else Status.UNKNOWN
            observation = replace(base, kind=Kind.TOOL_STARTED, status=status)
            if status is Status.UNKNOWN:
                return (observation, _partial(detail=f"Codex tool {item_id} has an unknown start status."))
            return (observation,)
        if native_status == "failed" or (type(exit_code) is int and exit_code != 0):
            status = Status.FAILED
        elif native_status == "completed":
            status = Status.COMPLETED
        else:
            status = Status.UNKNOWN
        result, has_result = self._tool_result(item=item, item_type=item_type)
        observations = [replace(base, kind=Kind.TOOL_COMPLETED, status=status, result=result)]
        if has_result:
            observations.append(replace(base, kind=Kind.TOOL_RESULT, status=status, result=result))
        if native_status not in {"completed", "failed"} or (item_type == "command_execution" and exit_code is None):
            observations.append(
                _partial(detail=f"Codex tool {item_id} lacks a documented completion status or exit_code.")
            )
        if item_type == "command_execution" and not isinstance(arguments, str):
            observations.append(_partial(detail=f"Codex command_execution {item_id} has no command."))
        return tuple(observations)

    @staticmethod
    def _tool_arguments(*, item: dict[str, JsonValue], item_type: str) -> JsonValue:
        if item_type == "command_execution":
            return item.get("command")
        if item_type == "file_change":
            return item.get("changes")
        if item_type == "mcp_tool_call":
            return item.get("arguments")
        return item.get("query")

    @staticmethod
    def _tool_result(*, item: dict[str, JsonValue], item_type: str) -> tuple[JsonValue, bool]:
        if item_type == "command_execution":
            output = item.get("aggregated_output")
            return (output, True) if isinstance(output, str) else (None, False)
        if item_type == "file_change":
            changes = item.get("changes")
            return (changes, True) if isinstance(changes, list) else (None, False)
        fields = ("result", "error") if item_type == "mcp_tool_call" else ("result", "results")
        for field in fields:
            if field in item and item[field] is not None:
                return item[field], True
        return None, False


class ClaudePrintStreamJsonAdapter:
    """Decode complete messages in ``claude -p --output-format stream-json --verbose``."""

    _SYSTEM_EVENTS: ClassVar[frozenset[str]] = frozenset(
        {"api_retry", "compact_boundary", "plugin_install", "hook_started", "hook_progress", "hook_response"}
    )
    _ERROR_RESULTS: ClassVar[frozenset[str]] = frozenset(
        {"error_during_execution", "error_max_turns", "error_max_budget_usd", "error_max_structured_output_retries"}
    )

    def decode(self, *, payload: dict[str, JsonValue]) -> tuple[NativeCliObservation, ...]:
        """
        Decode complete Claude messages; token deltas are never tool receipts.

        Returns:
            tuple[NativeCliObservation, ...]: Observed actions or explicit coverage gaps.
        """
        kind = payload.get("type")
        if not isinstance(kind, str):
            return (_partial(detail="Claude print stream-json frame is missing a string type."),)
        if kind == "system":
            return self._decode_system(payload=payload)
        if kind in {"assistant", "user"}:
            return self._decode_message(payload=payload, role=kind)
        if kind == "result":
            return self._decode_result(payload=payload)
        if kind == "stream_event":
            return (_partial(detail="Claude stream_event is not a complete message in this profile."),)
        return (_partial(detail="Unrecognized Claude print stream-json event type."),)

    def _decode_system(self, *, payload: dict[str, JsonValue]) -> tuple[NativeCliObservation, ...]:
        subtype = payload.get("subtype")
        if subtype == "init":
            session_id = _text(payload.get("session_id"))
            if session_id is None:
                return (_partial(detail="Claude system/init is missing session_id."),)
            return (
                NativeCliObservation(
                    kind=Kind.SESSION_STARTED,
                    source_session_id=session_id,
                    source_event_id=_text(payload.get("uuid")),
                ),
            )
        if isinstance(subtype, str) and subtype in self._SYSTEM_EVENTS:
            return (
                NativeCliObservation(
                    kind=Kind.AUXILIARY,
                    source_event_id=_text(payload.get("uuid")),
                    source_session_id=_text(payload.get("session_id")),
                    name=subtype,
                ),
            )
        if subtype == "permission_denied":
            return (NativeCliObservation(kind=Kind.ERROR, status=Status.FAILED, detail="Claude permission denied."),)
        return (_partial(detail="Unrecognized Claude system subtype."),)

    def _decode_message(self, *, payload: dict[str, JsonValue], role: str) -> tuple[NativeCliObservation, ...]:
        session_id, uuid = _text(payload.get("session_id")), _text(payload.get("uuid"))
        parent = payload.get("parent_tool_use_id")
        if session_id is None or uuid is None or (parent is not None and _text(parent) is None):
            return (_partial(detail=f"Claude {role} is missing a valid session, uuid, or parent ID."),)
        message = payload.get("message")
        if not isinstance(message, dict) or message.get("role") != role:
            return (_partial(detail=f"Claude {role} is missing a matching message role.", source_event_id=uuid),)
        message_id = _text(message.get("id"))
        if role == "assistant" and message_id is None:
            return (_partial(detail="Claude assistant message is missing its ID.", source_event_id=uuid),)
        content = message.get("content")
        if role == "user" and isinstance(content, str):
            return (
                NativeCliObservation(
                    kind=Kind.AUXILIARY,
                    source_event_id=uuid,
                    source_session_id=session_id,
                    source_message_id=message_id,
                    parent_tool_use_id=_text(parent),
                ),
            )
        if not isinstance(content, list):
            return (_partial(detail=f"Claude {role} content is not a block list.", source_event_id=uuid),)
        observations = [
            self._decode_block(
                block=block,
                role=role,
                uuid=uuid,
                session_id=session_id,
                message_id=message_id,
                parent_id=_text(parent),
            )
            for block in content
        ]
        return tuple(observation for group in observations for observation in group) or (
            NativeCliObservation(kind=Kind.AUXILIARY, source_event_id=uuid, source_session_id=session_id),
        )

    def _decode_block(
        self,
        *,
        block: JsonValue,
        role: str,
        uuid: str,
        session_id: str,
        message_id: str | None,
        parent_id: str | None,
    ) -> tuple[NativeCliObservation, ...]:
        if not isinstance(block, dict):
            return (_partial(detail=f"Claude {role} content block is not an object.", source_event_id=uuid),)
        base = NativeCliObservation(
            kind=Kind.AUXILIARY,
            source_event_id=uuid,
            source_session_id=session_id,
            source_message_id=message_id,
            parent_tool_use_id=parent_id,
        )
        block_type = block.get("type")
        if role == "assistant" and block_type == "text" and isinstance(block.get("text"), str):
            return (replace(base, kind=Kind.MODEL_MESSAGE, text=block["text"]),)
        if role == "assistant" and block_type == "tool_use":
            tool_id, name = _text(block.get("id")), _text(block.get("name"))
            if tool_id is None or name is None or "input" not in block:
                return (_partial(detail="Claude tool_use lacks id, name, or input.", source_event_id=uuid),)
            return (
                replace(
                    base,
                    kind=Kind.TOOL_REQUESTED,
                    status=Status.REQUESTED,
                    source_tool_id=tool_id,
                    name=name,
                    arguments=block["input"],
                ),
            )
        if role == "user" and block_type == "tool_result":
            return self._decode_tool_result(block=block, uuid=uuid, base=base)
        if (role == "assistant" and block_type in {"thinking", "redacted_thinking"}) or (
            role == "user" and block_type == "text"
        ):
            return (base,)
        return (_partial(detail=f"Unrecognized Claude {role} content block.", source_event_id=uuid),)

    @staticmethod
    def _decode_tool_result(
        *, block: dict[str, JsonValue], uuid: str, base: NativeCliObservation
    ) -> tuple[NativeCliObservation, ...]:
        tool_id = _text(block.get("tool_use_id"))
        if tool_id is None:
            return (_partial(detail="Claude tool_result is missing tool_use_id.", source_event_id=uuid),)
        is_error = block.get("is_error")
        status = Status.FAILED if is_error is True else Status.COMPLETED if is_error is False else Status.UNKNOWN
        result = replace(
            base, kind=Kind.TOOL_RESULT, status=status, source_tool_id=tool_id, result=block.get("content")
        )
        if status is Status.UNKNOWN or "content" not in block:
            return (result, _partial(detail=f"Claude tool_result {tool_id} lacks explicit result or error status."))
        return (result,)

    def _decode_result(self, *, payload: dict[str, JsonValue]) -> tuple[NativeCliObservation, ...]:
        session_id, uuid = _text(payload.get("session_id")), _text(payload.get("uuid"))
        if session_id is None or uuid is None or payload.get("parent_tool_use_id") is not None:
            return (_partial(detail="Claude result lacks root session identity or uuid."),)
        subtype, is_error, result = payload.get("subtype"), payload.get("is_error"), payload.get("result")
        if subtype == "success" and is_error is False and isinstance(result, str):
            return (
                NativeCliObservation(
                    kind=Kind.RUN_FINISHED,
                    status=Status.COMPLETED,
                    source_event_id=uuid,
                    source_session_id=session_id,
                    source_status="success",
                    text=result,
                ),
            )
        if isinstance(subtype, str) and subtype in self._ERROR_RESULTS and is_error is True:
            return (
                NativeCliObservation(
                    kind=Kind.ERROR,
                    status=Status.FAILED,
                    source_event_id=uuid,
                    source_session_id=session_id,
                    source_status=subtype,
                    detail=f"Claude result: {subtype}.",
                    text=_text(payload.get("result")),
                ),
            )
        return (_partial(detail="Unrecognized or contradictory Claude result status.", source_event_id=uuid),)
