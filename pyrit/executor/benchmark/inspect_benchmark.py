# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pyrit.memory import CentralMemory
from pyrit.models import Message, MessageScorable, ScoringExpectation
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.score import InspectEvalScorer

if TYPE_CHECKING:
    from inspect_ai import Task
    from inspect_ai.log import EvalLog, EvalSample
    from inspect_ai.solver import Solver, TaskState

    from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
    from pyrit.executor.benchmark._inspect_response_trace import InspectResponseTrace
    from pyrit.executor.benchmark.inspect_sandbox import InspectDockerProfile, InspectSandboxTools
    from pyrit.models import MessagePiece, Score
    from pyrit.prompt_target import PromptTarget


@dataclass(frozen=True, kw_only=True)
class InspectTaskBinding:
    """Keep the native task, candidate, system instructions, and grading criterion separate."""

    task: Task
    sample_id: int | str
    candidate: Message
    system_prompt: str
    native_scorer: str
    expectation: ScoringExpectation


@dataclass(frozen=True, kw_only=True)
class InspectRunResult:
    """References to the real final response and its persisted native-grade projection."""

    response: Message
    score: Score
    conversation_id: str
    native_eval_id: str
    native_sample_uuid: str


class InspectBenchmark:
    """Run one bound text sample through Inspect's lifecycle and an existing PyRIT target."""

    def __init__(
        self,
        *,
        objective_target: PromptTarget,
        model_name: str,
        artifacts: InspectRunArtifacts,
        tools: InspectSandboxTools,
        trace: InspectResponseTrace,
        docker_profile: InspectDockerProfile,
        episode_timeout_seconds: int,
    ) -> None:
        """
        Bind prepared dependencies without constructing prompts, targets, or environments.

        Raises:
            ValueError: If the episode timeout is not positive.
        """
        if episode_timeout_seconds < 1:
            raise ValueError("episode_timeout_seconds must be positive.")
        self._target = objective_target
        self._model_name = model_name
        self._artifacts = artifacts
        self._tools = tools
        self._trace = trace
        self._docker_profile = docker_profile
        self._episode_timeout_seconds = episode_timeout_seconds
        self._normalizer = PromptNormalizer()
        self._conversation_id = str(uuid4())
        self._response: Message | None = None
        self._binding: InspectTaskBinding | None = None
        self._has_run = False

    async def execute_async(self, *, binding: InspectTaskBinding) -> InspectRunResult:
        """
        Execute once, reconcile the checked native log, and verify owned-resource cleanup.

        Returns:
            InspectRunResult: The retained final answer, score, and cross-system identities.

        Raises:
            RuntimeError: If this instance has already executed.
        """
        from inspect_ai import eval_async

        if self._has_run:
            raise RuntimeError("An InspectBenchmark instance represents one independent attempt; create a fresh run.")
        self._has_run = True
        input_sha256 = self._validate_binding(binding)
        self._binding = binding
        self._artifacts.manifest.update(
            harness_status="running",
            conversation_id=self._conversation_id,
            input_sha256=input_sha256,
            candidate_sha256=hashlib.sha256(binding.candidate.get_value().encode("utf-8")).hexdigest(),
            system_sha256=hashlib.sha256(binding.system_prompt.encode("utf-8")).hexdigest(),
            target=self._target.get_identifier().model_dump(mode="json"),
        )
        await self._artifacts.save_async()
        try:
            async with asyncio.timeout(self._episode_timeout_seconds):
                logs = await eval_async(
                    tasks=binding.task,
                    model=None,
                    solver=self.create_solver(),
                    sample_id=binding.sample_id,
                    epochs=1,
                    retry_on_error=0,
                    task_retry_attempts=0,
                    score_on_error=False,
                    fail_on_error=True,
                    sandbox_cleanup=True,
                    max_samples=1,
                    max_tasks=1,
                    max_sandboxes=1,
                    time_limit=self._episode_timeout_seconds,
                    log_dir=str(self._artifacts.directory / "native"),
                    log_format="eval",
                    log_samples=True,
                    log_realtime=False,
                    log_model_api=False,
                    ctl_server=False,
                    acp_server=False,
                    metadata={
                        "pyrit_run_id": self._artifacts.run_id,
                        "pyrit_attempt_id": self._artifacts.attempt_id,
                        "external_model": self._model_name,
                        "harness_variant": "PyRIT final-answer-only; no Inspect model provider",
                    },
                )
            return await self._accept_log_async(logs=logs, binding=binding, input_sha256=input_sha256)
        except BaseException as error:
            self._artifacts.manifest.update(harness_status="error", error_type=type(error).__name__, error=str(error))
            await self._artifacts.append_async(
                event="episode_error", data={"error_type": type(error).__name__, "error": str(error)}
            )
            raise
        finally:
            self._artifacts.manifest.update(
                provider=self._trace.summary(),
                tool_executions=self._tools.executions,
                tool_dispatch_attempts=len(self._tools.executions),
                tool_executions_confirmed=sum(event["confirmed_started"] for event in self._tools.executions),
                container=self._docker_profile.container_evidence,
            )
            await self._artifacts.save_async()
            await self._docker_profile.verify_cleanup_async()

    def create_solver(self) -> Solver:
        """
        Build a public Inspect custom solver without a provider or manufactured model events.

        Returns:
            Solver: A solver that calls only the configured PyRIT target.
        """
        from inspect_ai.solver import Generate, solver

        @solver("pyrit_external_target")
        def external_target() -> Solver:
            # Inspect's Solver protocol calls these two parameters positionally.
            async def solve_async(state: TaskState, generate: Generate) -> TaskState:
                if self._binding is None:
                    raise RuntimeError("Call execute_async with a prepared binding before invoking this solver.")
                return await self._solve_async(state=state, binding=self._binding)

            return solve_async

        return external_target()

    async def _solve_async(self, *, state: TaskState, binding: InspectTaskBinding) -> TaskState:
        from inspect_ai.model import ChatMessageAssistant, ModelOutput

        if state.sample_id != binding.sample_id or state.epoch != 1 or self._response is not None:
            raise ValueError("Unexpected sample, epoch, or second solver invocation.")
        await self._docker_profile.verify_running_async()
        await asyncio.to_thread(
            self._target.set_system_prompt, system_prompt=binding.system_prompt, conversation_id=self._conversation_id
        )
        response = await self._normalizer.send_prompt_async(
            message=binding.candidate, target=self._target, conversation_id=self._conversation_id
        )
        piece = self._final_piece(response)
        model_executions = [event for event in self._tools.executions if event["origin"] == "model"]
        if not model_executions or any(
            event["status"] != "completed" or not event["confirmed_started"] for event in model_executions
        ):
            raise ValueError("Acceptance requires at least one completed real model-requested sandbox tool execution.")
        self._response = Message(message_pieces=[piece])
        state.output = ModelOutput.from_content(model=self._model_name, content=piece.converted_value)
        state.messages.append(ChatMessageAssistant(content=piece.converted_value, model=self._model_name))
        state.metadata.update(
            pyrit_conversation_id=self._conversation_id,
            pyrit_message_piece_id=str(piece.id),
            pyrit_run_id=self._artifacts.run_id,
        )
        state.completed = True
        await self._artifacts.append_async(
            event="final_answer",
            data={"message_piece_id": str(piece.id), "answer": piece.converted_value, "native_sample_uuid": state.uuid},
        )
        return state

    async def _accept_log_async(
        self, *, logs: list[EvalLog], binding: InspectTaskBinding, input_sha256: str
    ) -> InspectRunResult:
        from inspect_ai.log import read_eval_log_async

        if len(logs) != 1 or not logs[0].location:
            raise ValueError("Inspect did not return exactly one durable native log.")
        self._artifacts.manifest["native_log"] = logs[0].location
        log = await read_eval_log_async(logs[0].location, resolve_attachments=True)
        sample = await asyncio.to_thread(self._materialize_sample, log)
        self._artifacts.manifest.update(
            native_log=log.location,
            native_eval_id=log.eval.eval_id,
            native_run_id=log.eval.run_id,
            native_task_id=log.eval.task_id,
            native_task=log.eval.task,
            native_sample_uuid=sample.uuid,
            native_status=log.status,
            native_grade={name: grade.model_dump(mode="json") for name, grade in (sample.scores or {}).items()},
        )
        await self._artifacts.append_async(event="native_sample", data=sample.model_dump(mode="json"))
        if self._response is None or sample.uuid is None:
            raise ValueError("No authentic final response or native sample identity; no score will be fabricated.")
        scorer = InspectEvalScorer(
            log_path=Path(log.location),
            eval_id=log.eval.eval_id,
            sample_id=binding.sample_id,
            sample_uuid=sample.uuid,
            run_id=self._artifacts.run_id,
            attempt_id=self._artifacts.attempt_id,
            message_piece_id=self._response.get_piece().id,
            input_sha256=input_sha256,
            native_scorer=binding.native_scorer,
        )
        scores = await scorer.score_async(
            scorable=MessageScorable.from_message(self._response), expectation=binding.expectation
        )
        if len(scores) != 1:
            raise ValueError("Native grade projection did not persist exactly one score.")
        stored_scores = await asyncio.to_thread(
            CentralMemory.get_memory_instance().get_scores, score_ids=[str(scores[0].id)]
        )
        if len(stored_scores) != 1 or stored_scores[0].score_value != scores[0].score_value:
            raise ValueError("Projected score did not round-trip through canonical PyRIT memory.")
        messages = await asyncio.to_thread(
            CentralMemory.get_memory_instance().get_conversation_messages, conversation_id=self._conversation_id
        )
        self._artifacts.manifest.update(
            harness_status="completed",
            grade_status="complete",
            evidence_status="complete",
            final_answer=self._response.get_value(),
            score=scores[0].model_dump(mode="json"),
            canonical_message_piece_ids=[str(piece.id) for message in messages for piece in message.message_pieces],
        )
        return InspectRunResult(
            response=self._response,
            score=scores[0],
            conversation_id=self._conversation_id,
            native_eval_id=log.eval.eval_id,
            native_sample_uuid=sample.uuid,
        )

    @staticmethod
    def _materialize_sample(log: EvalLog) -> EvalSample:
        from inspect_ai.log import EvalSample

        samples = list(log.samples or [])
        if len(samples) != 1:
            raise ValueError("Inspect must retain exactly one sample, including error evidence.")
        # Detach nested lazy fields and resolved attachments while their backing log still exists.
        return EvalSample.model_validate(samples[0].model_dump(mode="json"))

    @staticmethod
    def _final_piece(response: Message) -> MessagePiece:
        if any(piece.has_error() or piece.is_truncated for piece in response.message_pieces):
            raise ValueError("The target response is errored or truncated, not a completed answer.")
        pieces = response.get_pieces_by_type(data_type="text")
        if len(pieces) != 1:
            raise ValueError("This adapter requires one final assistant text piece.")
        piece = pieces[0]
        if piece.api_role != "assistant" or piece.is_simulated or not piece.converted_value.strip():
            raise ValueError("A request echo, simulated message, or empty output is not an answer.")
        return piece

    @staticmethod
    def _validate_binding(binding: InspectTaskBinding) -> str:
        if len(binding.task.dataset) != 1 or binding.task.dataset[0].id != binding.sample_id:
            raise ValueError("Bind exactly one native task sample before evaluation.")
        sample = binding.task.dataset[0]
        if not isinstance(sample.input, str):
            raise TypeError("This prototype accepts text task inputs only.")
        if len(binding.candidate.message_pieces) != 1 or binding.candidate.api_role != "user":
            raise ValueError("Prepare exactly one user candidate message separately from the native task binding.")
        if binding.candidate.get_piece().converted_value_data_type != "text" or not binding.system_prompt:
            raise ValueError("The candidate must be text and system instructions must be supplied by the caller.")
        return hashlib.sha256(sample.input.encode("utf-8")).hexdigest()
