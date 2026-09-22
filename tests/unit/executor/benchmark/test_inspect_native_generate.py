# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark._inspect_native_generate import InspectNativeGenerate
from pyrit.memory import CentralMemory
from pyrit.prompt_target import OpenAIResponseTarget
from tests.unit.mocks import openai_response_json_dict

pytest.importorskip("inspect_ai")


def _response(*, call: str | None, argument: str = "") -> dict[str, Any]:
    result = openai_response_json_dict()
    if call is None:
        result["output"][0]["content"][0]["text"] = "OFFLINE/SIMULATED final presentation"
    else:
        result["output"] = [
            {
                "type": "function_call",
                "call_id": call,
                "id": f"function-{call}",
                "name": "submit_fixture",
                "arguments": json.dumps({"artifact_ref": argument}),
                "status": "completed",
            }
        ]
    return result


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("scenario", ["lower_later", "full_success", "missing_then_success", "no_tool_then_success"])
async def test_native_generate_preserves_setup_string_feedback_and_stop_async(tmp_path: Path, scenario: str) -> None:
    from inspect_ai import Task, eval_async
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageSystem
    from inspect_ai.scorer import Score, Target, scorer
    from inspect_ai.solver import Generate, Solver, TaskState, solver
    from inspect_ai.tool import Tool, ToolError, tool
    from inspect_ai.util import store

    calls: list[str] = []
    requests: list[dict[str, Any]] = []
    full_success = False
    script = (
        [_response(call="call-1", argument="a"), _response(call="call-2", argument="b"), _response(call=None)]
        if scenario == "lower_later"
        else [_response(call="call-1", argument="missing"), _response(call="call-2", argument="success")]
        if scenario == "missing_then_success"
        else [_response(call=None), _response(call="call-2", argument="success")]
        if scenario == "no_tool_then_success"
        else [_response(call="call-1", argument="success")]
    )

    @tool
    def submit_fixture() -> Tool:
        async def submit_async(artifact_ref: str) -> str:
            """Submit an inert offline artifact reference.

            Args:
                artifact_ref: The inert fixture reference.
            """
            nonlocal full_success
            assert store().get("fixture_setup") == "OFFLINE/SIMULATED"
            calls.append(artifact_ref)
            if artifact_ref == "missing":
                raise ToolError("OFFLINE missing artifact")
            grade = {"a": 0.75, "b": 0.25, "success": 1.0}[artifact_ref]
            store().set("fixture_grade", grade)
            full_success = grade == 1.0
            return f"OFFLINE feedback {grade}\nexact"

        return submit_async

    @solver
    def native_solver() -> Solver:
        async def solve_async(state: TaskState, generate: Generate) -> TaskState:
            state.store.set("fixture_setup", "OFFLINE/SIMULATED")
            state.messages.insert(0, ChatMessageSystem(content="Task-owned OFFLINE/SIMULATED system"))
            state.tools = [submit_fixture()]
            while not state.completed:
                result = await generate(state, tool_calls="loop")
                assert result is state
                if scenario == "lower_later":
                    break
            state.store.set("native_solver_returned", True)
            return state

        return solve_async

    @scorer(metrics=[])
    def native_scorer() -> Any:
        async def score_async(state: TaskState, target: Target) -> Score:
            assert state.store.get("native_solver_returned") is True
            return Score(value=state.store.get("fixture_grade"))

        return score_async

    def provider(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        assert len(requests) <= len(script), "Unexpected follow-up provider call"
        return httpx.Response(200, json=script[len(requests) - 1])

    async def after_tool_async() -> bool:
        return full_success

    artifacts = InspectRunArtifacts(directory=tmp_path / "run", provenance={"mode": "OFFLINE/SIMULATED"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as client:

        def target_factory(schemas: list[dict[str, Any]]) -> OpenAIResponseTarget:
            return OpenAIResponseTarget(
                endpoint="https://fixture.invalid/v1",
                api_key="offline",
                model_name="offline-fixture",
                auto_execute_tools=False,
                extra_body_parameters={"tools": schemas, "parallel_tool_calls": False, "store": False},
                httpx_client_kwargs={"http_client": client, "max_retries": 0},
            )

        bridge = InspectNativeGenerate(
            target_factory=target_factory,
            model_name="offline-fixture",
            artifacts=artifacts,
            after_tool_async=after_tool_async,
            max_requests=5,
            max_tool_calls=4,
            max_tool_output_bytes=16384,
        )
        original = native_solver()

        @solver
        def bridge_solver() -> Solver:
            async def solve_async(state: TaskState, generate: Generate) -> TaskState:
                return await original(state, bridge.generate_async)

            return solve_async

        with (
            patch.dict("os.environ", {"RETRY_MAX_NUM_ATTEMPTS": "1"}),
            patch(
                "asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=AssertionError("No subprocess")
            ),
            patch("socket.socket.connect", side_effect=AssertionError("No socket")),
        ):
            logs = await eval_async(
                Task(
                    dataset=[Sample(id="fixture", input="Task-owned inert input")],
                    solver=bridge_solver(),
                    scorer=native_scorer(),
                ),
                model=None,
                log_dir=str(tmp_path / "native"),
                log_realtime=False,
                ctl_server=False,
                acp_server=False,
                retry_on_error=0,
            )
    assert logs[0].status == "success", logs[0].error.message if logs[0].error else None
    assert len(requests) == len(script)
    native_sample = list(logs[0].samples or [])[0]
    assert native_sample.scores["native_scorer"].value == (0.25 if scenario == "lower_later" else 1.0)
    assert requests[0]["input"] == [
        {"role": "developer", "content": [{"type": "input_text", "text": "Task-owned OFFLINE/SIMULATED system"}]},
        {"role": "user", "content": [{"type": "input_text", "text": "Task-owned inert input"}]},
    ]
    if scenario == "lower_later":
        assert requests[1]["input"][-1]["output"] == "OFFLINE feedback 0.75\nexact"
        assert requests[2]["input"][-1]["output"] == "OFFLINE feedback 0.25\nexact"
        assert calls == ["a", "b"]
    elif scenario == "missing_then_success":
        assert requests[1]["input"][-1]["output"] == "Error: OFFLINE missing artifact"
        assert calls == ["missing", "success"]
    elif scenario == "no_tool_then_success":
        assert requests[1]["input"] == [
            *requests[0]["input"],
            {"role": "assistant", "content": [{"type": "output_text", "text": "OFFLINE/SIMULATED final presentation"}]},
        ]
    memory = CentralMemory.get_memory_instance()
    pieces = list(memory.get_message_pieces(conversation_id=bridge.conversation_id))
    outputs = [
        json.loads(piece.original_value) for piece in pieces if piece.original_value_data_type == "function_call_output"
    ]
    assert len(outputs) == len(calls)
    if scenario != "lower_later":
        assert pieces[-1].role == "tool"
        assert outputs[-1]["output"] == "OFFLINE feedback 1.0\nexact"
        assert not native_sample.output.completion
