# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A public, no-sandbox Inspect ReAct Task with an explicitly authored continuation seam."""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.agent import AgentState, as_solver, react
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, tool
from inspect_ai.util import store

from pyrit.executor.benchmark.inspect_mode2_runner import active_mode2_controller


@solver
def prepare_mode2_sample() -> Solver:
    """Initialize the original Task's per-Sample state."""

    async def prepare_async(state: TaskState, generate: Generate) -> TaskState:
        state.store.set("mode2_sample_id", str(state.sample_id))
        state.store.set("lifecycle", ["setup"])
        return state

    return prepare_async


@tool
def harmless_lookup() -> Tool:
    """Return a fixed local result without accessing a sandbox or network."""

    async def lookup_async(value: str) -> str:
        """
        Look up the harmless fixture value.

        Args:
            value (str): The fixed benign lookup key.

        Returns:
            str: A local, constant result.
        """
        if value != "fixture" or store().get("lifecycle") != ["setup"]:
            raise ValueError("The inert tool ran outside its approved Sample setup.")
        store().set("lifecycle", ["setup", "tool"])
        return "fixture result"

    return lookup_async


async def mode2_continue_async(state: AgentState) -> bool | AgentState:
    """Delegate the authored ReAct continuation boundary to the active PyRIT attack."""
    return await active_mode2_controller().continue_async(state=state)


@scorer(metrics=[mean()])
def original_mode2_scorer() -> Scorer:
    """Preserve Inspect's intermediate attempt feedback and final scorer."""

    async def score_async(state: TaskState, target: Target) -> Score:
        lifecycle = state.store.get("lifecycle")
        if lifecycle == ["setup", "tool"] and state.output.completion.endswith("wrong"):
            state.store.set("lifecycle", [*lifecycle, "attempt_score"])
            return Score(value=0.0)
        if lifecycle == ["setup", "tool", "attempt_score"] and state.output.completion == "fixture result":
            state.store.set("lifecycle", [*lifecycle, "final_score"])
            return Score(value=1.0)
        raise RuntimeError("The original Inspect scorer ran outside its attempt/final schedule.")

    return score_async


async def cleanup_async(state: TaskState) -> None:
    """Check the original scorer ran before the Task's own cleanup."""
    lifecycle = state.store.get("lifecycle")
    if lifecycle != ["setup", "tool", "attempt_score", "final_score"] or not state.scores:
        raise RuntimeError("The original Inspect cleanup ran before its final scorer.")
    state.store.set("lifecycle", [*lifecycle, "cleanup"])


def mode2_inert_task() -> Task:
    """
    Construct the one SHA-pinned, in-process ReAct continuation fixture.

    Returns:
        Task: An original Inspect setup, ReAct agent, scorer, and cleanup.
    """
    outputs = [
        ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="harmless_lookup",
            tool_arguments={"value": "fixture"},
            tool_call_id="mode2-lookup",
        ),
        ModelOutput.for_tool_call(
            model="mockllm/model",
            tool_name="submit",
            tool_arguments={"answer": "wrong"},
            tool_call_id="mode2-wrong",
        ),
        ModelOutput.from_content(model="mockllm/model", content="fixture result"),
    ]
    for output in outputs:
        output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
    model = get_model("mockllm/model", custom_outputs=outputs)
    agent = react(
        prompt=None,
        tools=[harmless_lookup()],
        attempts=2,
        on_continue=mode2_continue_async,
    )
    return Task(
        dataset=[Sample(id="mode2-inert-1", input="harmless local fixture", target="fixture result")],
        name="inspect_mode2_inert",
        version=1,
        setup=prepare_mode2_sample(),
        solver=as_solver(agent),
        scorer=original_mode2_scorer(),
        cleanup=cleanup_async,
        model=model,
    )
