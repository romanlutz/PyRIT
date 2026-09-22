# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any

import pytest

from pyrit.executor.benchmark.submission.hooks import SubmissionHooks, SubmissionLimits, SubmissionTool


async def _callback_async(**kwargs: Any) -> str:
    return "OFFLINE/SIMULATED"


def _tool(name: str = "fixture") -> SubmissionTool:
    return SubmissionTool(
        name=name, description="Inert fixture", parameters={"type": "object"}, callback_async=_callback_async
    )


def test_duplicate_tools_and_overlapping_exception_policies_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        SubmissionHooks(
            tools=(_tool(), _tool()), read_report=dict, recoverable_errors=(), terminal_errors=(), error_feedback=str
        )
    with pytest.raises(ValueError, match="overlap"):
        SubmissionHooks(
            tools=(_tool(),),
            read_report=dict,
            recoverable_errors=(Exception,),
            terminal_errors=(RuntimeError,),
            error_feedback=str,
        )


@pytest.mark.parametrize(
    "kwargs", [{"max_requests": 0}, {"max_tool_calls": 0}, {"episode_timeout_seconds": 0}, {"max_response_bytes": 0}]
)
def test_invalid_limits_rejected(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        SubmissionLimits(**kwargs)


def test_tool_schema_preserved_without_private_or_default_tool_names() -> None:
    tool = _tool("caller_selected_tool")
    assert tool.response_definition() == {
        "type": "function",
        "name": "caller_selected_tool",
        "description": "Inert fixture",
        "parameters": {"type": "object"},
    }
