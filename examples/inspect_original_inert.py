# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""A public, model-free original Inspect Task for the Mode 1 import contract."""

from __future__ import annotations

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver


@solver
def prepare_inert_sample() -> Solver:
    """Prepare an in-memory marker in the original Task setup."""

    async def prepare_async(state: TaskState, generate: Generate) -> TaskState:
        state.store.set("lifecycle", ["setup"])
        return state

    return prepare_async


@solver
def original_inert_solver() -> Solver:
    """Answer the public fixture without invoking a model, tool, or sandbox."""

    async def solve_async(state: TaskState, generate: Generate) -> TaskState:
        lifecycle = state.store.get("lifecycle")
        if lifecycle != ["setup"]:
            raise RuntimeError("Original Inspect setup did not precede the solver.")
        state.store.set("lifecycle", [*lifecycle, "solver"])
        state.messages.append(ChatMessageAssistant(content="inert response"))
        state.output = ModelOutput.from_content(model="mockllm/model", content="inert response")
        return state

    return solve_async


@scorer(metrics=[mean()])
def original_inert_scorer() -> Scorer:
    """Grade the original solver response after its own setup."""

    async def score_async(state: TaskState, target: Target) -> Score:
        lifecycle = state.store.get("lifecycle")
        if lifecycle != ["setup", "solver"]:
            raise RuntimeError("Original Inspect scorer ran outside the Task lifecycle.")
        state.store.set("lifecycle", [*lifecycle, "score"])
        return Score(value=1.0 if state.output.completion == "inert response" else 0.0)

    return score_async


async def cleanup_async(state: TaskState) -> None:
    """Confirm the original scorer completed before Task cleanup."""
    lifecycle = state.store.get("lifecycle")
    if lifecycle != ["setup", "solver", "score"] or not state.scores:
        raise RuntimeError("Original Inspect cleanup ran before the original scorer.")
    state.store.set("lifecycle", [*lifecycle, "cleanup"])


def original_inert_task() -> Task:
    """
    Construct exactly one approved Task without a sandbox or model invocation.

    Returns:
        Task: An inert original-solver, setup, scorer, and cleanup lifecycle.
    """
    return Task(
        dataset=[Sample(id="original-inert-1", input="harmless fixture", target="inert response")],
        name="inspect_original_inert",
        version=1,
        setup=prepare_inert_sample(),
        solver=original_inert_solver(),
        scorer=original_inert_scorer(),
        cleanup=cleanup_async,
        model="mockllm/model",
    )
