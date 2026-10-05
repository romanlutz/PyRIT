# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

from pyrit.models import Message
from pyrit.models.native_cyber import NativeAgentCapabilities
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import NativeAgentTarget
from pyrit.prompt_target.native_agent_target import CopilotSdkAgentSession

if TYPE_CHECKING:
    from collections.abc import Callable

    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


class EventFixture:
    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.event)


class SdkSessionFixture:
    """Inert SDK event source; no provider, CLI or resource is started."""

    def __init__(self, turns: list[list[dict[str, Any]]]) -> None:
        self.session_id = str(uuid4())
        self.turns = turns
        self.prompts: list[str] = []
        self.disconnected = False
        self.handler: Callable[[Any], None] | None = None

    def on(self, handler: Callable[[Any], None]) -> Callable[[], None]:
        self.handler = handler
        return self._unsubscribe

    def _unsubscribe(self) -> None:
        self.handler = None

    async def send_and_wait(self, prompt: str, *, timeout: float) -> object:
        self.prompts.append(prompt)
        for event in self.turns[len(self.prompts) - 1]:
            assert self.handler is not None
            self.handler(EventFixture(event))
        return None

    async def disconnect(self) -> None:
        self.disconnected = True


def event(kind: str, data: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {"id": str(uuid4()), "type": kind, "data": data, "timestamp": "2026-09-25T00:00:00Z", **extra}


def tool_turn(
    *, call_id: str = "call-1", output: str = "OFFLINE/SIMULATED", final: str | None = "Done"
) -> list[dict[str, Any]]:
    arguments = {"command": "printf fixture"}
    values = [
        event(
            "assistant.message",
            {"content": "", "toolRequests": [{"toolCallId": call_id, "name": "bash", "arguments": arguments}]},
        ),
        event("tool.execution_start", {"toolCallId": call_id, "toolName": "bash", "arguments": arguments}),
        event(
            "tool.execution_complete",
            {
                "toolCallId": call_id,
                "success": True,
                "result": {"content": output, "detailedContent": "separate detailed result"},
                "shellExecution": {"exitCode": 0},
            },
        ),
    ]
    if final is not None:
        values.append(event("assistant.message", {"content": final}))
    values.append(event("assistant.usage", {"inputTokens": 3, "outputTokens": 2}, ephemeral=True))
    values.append(event("session.idle", {}, ephemeral=True))
    return values


def agent_session(sdk: SdkSessionFixture, *, steps: bool = False) -> CopilotSdkAgentSession:
    return CopilotSdkAgentSession(
        session=sdk,
        environment_id=str(uuid4()),
        simulated=True,
        capabilities=NativeAgentCapabilities(retained_session=steps, operator_steps=steps, max_turns=3 if steps else 1),
        provenance={"fixture": "OFFLINE/SIMULATED", "transport": "inert_sdk_event_fixture"},
    )


async def test_real_target_normalizer_preserve_native_tool_correlation_async(sqlite_instance: SQLiteMemory) -> None:
    sdk = SdkSessionFixture([tool_turn()])
    session = agent_session(sdk)
    target = NativeAgentTarget(session=session)
    conversation_id = str(uuid4())
    response = await PromptNormalizer().send_prompt_async(
        message=Message.from_prompt(prompt="An exact native instruction", role="user"),
        target=target,
        conversation_id=conversation_id,
    )
    assert response.get_value() == "Done"
    assert sdk.prompts == ["An exact native instruction"]
    evidence = session.evidence()
    assert evidence.coverage_complete and evidence.idle
    assert evidence.tools[0].call_id == evidence.tool_requests[0].call_id == "call-1"
    assert evidence.tools[0].model_visible_output == "OFFLINE/SIMULATED"
    assert evidence.tools[0].detailed_output == "separate detailed result"
    assert evidence.tools[0].result != evidence.tools[0].model_visible_output
    assert evidence.events[-1].event_type == "session.idle"
    assert evidence.events[-2].payload["ephemeral"] is True
    messages = sqlite_instance.get_conversation_messages(conversation_id=conversation_id)
    assert [message.api_role for message in messages] == ["user", "assistant", "tool", "assistant"]
    assert json_value(messages[1].get_value())["data"]["toolCallId"] == "call-1"
    await session.quiesce_async()
    assert sdk.disconnected and session.evidence().idle


def json_value(value: str) -> dict[str, Any]:
    import json

    return json.loads(value)


async def test_tool_only_turn_is_not_a_fabricated_assistant_receipt_async() -> None:
    session = agent_session(SdkSessionFixture([tool_turn(final=None)]))
    response = await PromptNormalizer().send_prompt_async(
        message=Message.from_prompt(prompt="Inspect fixture", role="user"),
        target=NativeAgentTarget(session=session),
    )
    assert response.api_role == "tool"
    assert response.get_piece().original_value_data_type == "function_call_output"


async def test_evidence_snapshot_does_not_alias_the_retained_native_stream_async() -> None:
    session = agent_session(SdkSessionFixture([tool_turn()]))
    await session.send_async(prompt="fixture", timeout_seconds=1)
    first = session.evidence()
    first.events[0].payload["data"] = {"content": "modified outside"}
    first.tools[0].result["content"] = "modified outside"
    second = session.evidence()
    assert second.events[0].payload["data"]["content"] == ""
    assert second.tools[0].model_visible_output == "OFFLINE/SIMULATED"
    assert second.tools[0].result["content"] == "OFFLINE/SIMULATED"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_complete",
        "wrong_arguments",
        "duplicate_id",
        "missing_start",
        "session_error",
        "malformed_requests",
        "invalid_model_output",
        "invalid_detailed_output",
    ],
)
async def test_trace_gaps_are_explicit_not_success_async(defect: str) -> None:
    events = tool_turn()
    if defect == "missing_complete":
        events.pop(2)
    elif defect == "wrong_arguments":
        events[1]["data"]["arguments"] = {"command": "different"}
    elif defect == "duplicate_id":
        events[2]["id"] = events[1]["id"]
    elif defect == "missing_start":
        events.pop(1)
    elif defect == "malformed_requests":
        events[0]["data"]["toolRequests"] = "not a structured list"
    elif defect == "invalid_model_output":
        events[2]["data"]["result"]["content"] = {"not": "text"}
    elif defect == "invalid_detailed_output":
        events[2]["data"]["result"]["detailedContent"] = ["not", "text"]
    else:
        events.insert(3, event("session.error", {"message": "inert failure", "errorType": "fixture"}))
    session = agent_session(SdkSessionFixture([events]))
    await session.send_async(prompt="fixture", timeout_seconds=1)
    assert not session.evidence().coverage_complete
    assert session.evidence().gaps


async def test_start_before_model_request_is_a_permanent_causality_gap_async() -> None:
    events = tool_turn(final=None)
    events[0], events[1] = events[1], events[0]
    session = agent_session(SdkSessionFixture([events]))
    await session.send_async(prompt="fixture", timeout_seconds=1)
    evidence = session.evidence()
    assert evidence.idle and not evidence.coverage_complete
    assert any("started before its model tool request" in gap for gap in evidence.gaps)
    assert evidence.tools[0].call_id == evidence.tool_requests[0].call_id == "call-1"
    assert evidence.tools[0].start_sequence == 1
    assert evidence.tool_requests[0].request_sequence == 2
    assert evidence.tools[0].completion_sequence == 3
    assert evidence.tools[0].request_sequence is None
    assert len(evidence.events) == len(events)
    await session.quiesce_async()
    assert not session.evidence().coverage_complete


async def test_session_must_observe_root_idle_and_cannot_clone_workspace_async() -> None:
    session = agent_session(SdkSessionFixture([[event("session.idle", {}, agentId="child")]]), steps=True)
    with pytest.raises(RuntimeError, match="root session.idle"):
        await session.send_async(prompt="fixture", timeout_seconds=1)
    with pytest.raises(RuntimeError, match="still be running"):
        await session.quiesce_async()


async def test_same_retained_session_and_new_conversation_rejection_async() -> None:
    sdk = SdkSessionFixture([tool_turn(call_id="first"), tool_turn(call_id="second")])
    session = agent_session(sdk, steps=True)
    target = NativeAgentTarget(session=session)
    normalizer = PromptNormalizer()
    conversation_id = str(uuid4())
    for instruction in ("First", "Next"):
        await normalizer.send_prompt_async(
            message=Message.from_prompt(prompt=instruction, role="user"), target=target, conversation_id=conversation_id
        )
    assert sdk.prompts == ["First", "Next"]
    with pytest.raises(Exception, match="Error sending prompt"):
        await normalizer.send_prompt_async(
            message=Message.from_prompt(prompt="Not a clone", role="user"), target=target, conversation_id=str(uuid4())
        )
    assert len(sdk.prompts) == 2


def test_docker_stdio_prefix_contains_no_shell_auth_environment_or_mount() -> None:
    arguments = CopilotSdkAgentSession.docker_stdio_arguments(container_id="a" * 64, cli_path="/opt/copilot/copilot")
    assert arguments == ("exec", "-i", "a" * 64, "/opt/copilot/copilot")
    assert not any(item in arguments for item in ("-e", "--env", "--mount", "--privileged", "sh", "bash"))


@pytest.mark.parametrize(
    ("container_id", "cli_path"),
    [
        ("friendly-name", "/opt/copilot/copilot"),
        ("a" * 64, "../copilot"),
        ("a" * 64, "/opt/../copilot"),
        ("a" * 64, "C:\\copilot.exe"),
        ("--privileged", "/copilot"),
    ],
)
def test_docker_stdio_rejects_ambiguous_container_or_executable(*, container_id: str, cli_path: str) -> None:
    with pytest.raises(ValueError):
        CopilotSdkAgentSession.docker_stdio_arguments(container_id=container_id, cli_path=cli_path)
