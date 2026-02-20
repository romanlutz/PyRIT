# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from pyrit.common.logger import logger as pyrit_logger
from pyrit.executor.workflow.core import (
    WorkflowContext,
    WorkflowResult,
    WorkflowStrategy,
)

logger = logging.getLogger(__name__)

_DEFAULT_CONTROL_INIT: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"


class GCGStatus(Enum):
    """Enumeration of possible GCG attack result statuses."""

    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class GCGContext(WorkflowContext):
    """
    Context for GCG adversarial suffix generation workflow.

    Contains per-execution parameters for training configuration and hyperparameters.
    Model infrastructure (paths, devices) is configured on the workflow instance.
    """

    # Training data
    train_data: str = ""
    n_train_data: int = 50
    test_data: str = ""
    n_test_data: int = 0

    # Optimization parameters
    control_init: str = _DEFAULT_CONTROL_INIT
    n_steps: int = 500
    test_steps: int = 50
    batch_size: int = 512
    learning_rate: float = 0.01
    topk: int = 256
    temp: int = 1

    # Loss weights
    target_weight: float = 1.0
    control_weight: float = 0.0

    # Training strategy flags
    transfer: bool = False
    progressive_goals: bool = False
    progressive_models: bool = False
    anneal: bool = False
    incr_control: bool = False
    stop_on_success: bool = False

    # Candidate filtering
    allow_non_ascii: bool = False
    filter_cand: bool = True
    gbda_deterministic: bool = True

    # Output
    result_prefix: str = ""
    logfile: str = ""
    verbose: bool = True
    random_seed: int = 42


@dataclass
class GCGResult(WorkflowResult):
    """
    Result of GCG adversarial suffix generation workflow.

    Contains the discovered adversarial suffix and associated metrics.
    """

    control_str: str
    n_steps: int
    loss: Optional[float] = None

    @property
    def success(self) -> bool:
        """
        Determine if the attack found a viable adversarial suffix.

        Returns:
            bool: True if a non-empty control string was produced.
        """
        return bool(self.control_str and self.control_str != _DEFAULT_CONTROL_INIT)

    @property
    def status(self) -> GCGStatus:
        """
        Get the status of the attack result.

        Returns:
            GCGStatus: The status of the attack result.
        """
        if self.loss is None:
            return GCGStatus.UNKNOWN
        return GCGStatus.SUCCESS if self.success else GCGStatus.FAILURE


class GCGWorkflow(WorkflowStrategy["GCGContext", "GCGResult"]):
    """
    Workflow for generating adversarial suffixes using the GCG algorithm.

    Wraps the existing GreedyCoordinateGradientAdversarialSuffixGenerator
    in the PyRIT WorkflowStrategy lifecycle (validate → setup → perform → teardown).

    Model infrastructure is configured at init time. Per-execution parameters
    (training data, hyperparameters) are provided via GCGContext.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_paths: list[str],
        tokenizer_paths: list[str],
        conversation_templates: list[str],
        token: str,
        num_train_models: int = 1,
        devices: Optional[list[str]] = None,
        model_kwargs: Optional[list[dict[str, Any]]] = None,
        tokenizer_kwargs: Optional[list[dict[str, Any]]] = None,
        workflow_logger: logging.Logger = pyrit_logger,
    ) -> None:
        """
        Initialize the GCG suffix generation workflow.

        Args:
            model_name (str): Name identifier for the model.
            model_paths (list[str]): Paths to model weights.
            tokenizer_paths (list[str]): Paths to tokenizer models.
            conversation_templates (list[str]): Conversation template names per model.
            token (str): HuggingFace authentication token.
            num_train_models (int): Number of models for training. Defaults to 1.
            devices (Optional[list[str]]): CUDA devices to use.
            model_kwargs (Optional[list[dict[str, Any]]]): Additional kwargs per model.
            tokenizer_kwargs (Optional[list[dict[str, Any]]]): Additional kwargs per tokenizer.
            workflow_logger (logging.Logger): Logger instance.
        """
        super().__init__(context_type=GCGContext, logger=workflow_logger)
        self._model_name = model_name
        self._model_paths = model_paths
        self._tokenizer_paths = tokenizer_paths
        self._conversation_templates = conversation_templates
        self._token = token
        self._num_train_models = num_train_models
        self._devices = devices or ["cuda:0"]
        self._model_kwargs = model_kwargs or [{"low_cpu_mem_usage": True, "use_cache": False}]
        self._tokenizer_kwargs = tokenizer_kwargs or [{"use_fast": False}]

        # Mutable state managed across lifecycle
        self._workers: list[Any] = []
        self._test_workers: list[Any] = []
        self._attack: Optional[Any] = None
        self._params: Optional[Any] = None

    def _validate_context(self, *, context: GCGContext) -> None:
        """
        Validate the GCG context before execution.

        Args:
            context (GCGContext): The context to validate.

        Raises:
            ValueError: If required parameters are missing or invalid.
        """
        if not self._model_paths:
            raise ValueError("model_paths must be provided and non-empty")
        if not self._tokenizer_paths:
            raise ValueError("tokenizer_paths must be provided and non-empty")
        if len(self._model_paths) != len(self._tokenizer_paths):
            raise ValueError(
                f"model_paths ({len(self._model_paths)}) and "
                f"tokenizer_paths ({len(self._tokenizer_paths)}) must have the same length"
            )
        if not self._conversation_templates:
            raise ValueError("conversation_templates must be provided and non-empty")
        if not self._token:
            raise ValueError("token (HuggingFace) must be provided")
        if not context.train_data:
            raise ValueError("train_data must be provided in context")
        if context.n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {context.n_steps}")
        if context.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {context.batch_size}")

    async def _setup_async(self, *, context: GCGContext) -> None:
        """
        Set up the GCG attack by loading models, goals/targets, and creating the attack.

        Args:
            context (GCGContext): The execution context with training parameters.
        """
        from pyrit.auxiliary_attacks.gcg.attack.base.attack_manager import get_goals_and_targets, get_workers
        from pyrit.auxiliary_attacks.gcg.attack.gcg.gcg_attack import (
            GCGAttackPrompt,
            GCGMultiPromptAttack,
            GCGPromptManager,
        )
        from pyrit.auxiliary_attacks.gcg.experiments.log import log_gpu_memory, log_params, log_train_goals
        from pyrit.auxiliary_attacks.gcg.experiments.train import (
            GreedyCoordinateGradientAdversarialSuffixGenerator,
        )

        self._params = GreedyCoordinateGradientAdversarialSuffixGenerator._build_params(
            token=self._token,
            tokenizer_paths=self._tokenizer_paths,
            model_name=self._model_name,
            model_paths=self._model_paths,
            conversation_templates=self._conversation_templates,
            result_prefix=context.result_prefix,
            train_data=context.train_data,
            control_init=context.control_init,
            n_train_data=context.n_train_data,
            n_steps=context.n_steps,
            test_steps=context.test_steps,
            batch_size=context.batch_size,
            transfer=context.transfer,
            target_weight=context.target_weight,
            control_weight=context.control_weight,
            progressive_goals=context.progressive_goals,
            progressive_models=context.progressive_models,
            anneal=context.anneal,
            incr_control=context.incr_control,
            stop_on_success=context.stop_on_success,
            verbose=context.verbose,
            allow_non_ascii=context.allow_non_ascii,
            num_train_models=self._num_train_models,
            devices=self._devices,
            model_kwargs=self._model_kwargs,
            tokenizer_kwargs=self._tokenizer_kwargs,
            n_test_data=context.n_test_data,
            test_data=context.test_data,
            learning_rate=context.learning_rate,
            topk=context.topk,
            temp=context.temp,
            filter_cand=context.filter_cand,
            gbda_deterministic=context.gbda_deterministic,
            logfile=context.logfile,
            random_seed=context.random_seed,
        )
        self._logger.info(f"GCG parameters: {self._params}")

        import mlflow

        mlflow.start_run()
        log_gpu_memory(step=0)
        log_params(params=self._params)

        train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(self._params)
        log_train_goals(train_goals=train_goals)

        train_targets, test_targets = GreedyCoordinateGradientAdversarialSuffixGenerator._apply_target_augmentation(
            train_targets=train_targets,
            test_targets=test_targets,
        )

        self._workers, self._test_workers = get_workers(self._params)

        managers = {
            "AP": GCGAttackPrompt,
            "PM": GCGPromptManager,
            "MPA": GCGMultiPromptAttack,
        }

        self._attack = GreedyCoordinateGradientAdversarialSuffixGenerator._create_attack(
            params=self._params,
            managers=managers,
            train_goals=train_goals,
            train_targets=train_targets,
            test_goals=test_goals,
            test_targets=test_targets,
            workers=self._workers,
            test_workers=self._test_workers,
        )

    async def _perform_async(self, *, context: GCGContext) -> GCGResult:
        """
        Execute the GCG optimization loop.

        Runs the attack in a thread pool to avoid blocking the event loop,
        since the GCG optimization is CPU/GPU-bound.

        Args:
            context (GCGContext): The execution context.

        Returns:
            GCGResult: The result containing the adversarial suffix and metrics.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._run_attack, context)
        return result

    def _run_attack(self, context: GCGContext) -> GCGResult:
        """
        Execute the attack synchronously.

        Args:
            context (GCGContext): The execution context.

        Returns:
            GCGResult: The result of the attack.
        """
        run_result = self._attack.run(
            n_steps=self._params.n_steps,
            batch_size=self._params.batch_size,
            topk=self._params.topk,
            temp=self._params.temp,
            target_weight=self._params.target_weight,
            control_weight=self._params.control_weight,
            test_steps=getattr(self._params, "test_steps", 1),
            anneal=self._params.anneal,
            incr_control=self._params.incr_control,
            stop_on_success=self._params.stop_on_success,
            verbose=self._params.verbose,
            filter_cand=self._params.filter_cand,
            allow_non_ascii=self._params.allow_non_ascii,
        )

        # attack.run returns (control_str, loss, steps) or (control_str, steps)
        if len(run_result) == 3:
            control_str, loss, n_steps = run_result
        else:
            control_str, n_steps = run_result
            loss = None

        return GCGResult(
            control_str=control_str,
            loss=loss,
            n_steps=n_steps,
        )

    async def _teardown_async(self, *, context: GCGContext) -> None:
        """
        Clean up workers and MLflow after execution.

        Args:
            context (GCGContext): The execution context.
        """
        for worker in self._workers + self._test_workers:
            try:
                worker.stop()
            except Exception as e:
                self._logger.warning(f"Error stopping worker: {e}")

        try:
            import mlflow

            mlflow.end_run()
        except Exception as e:
            self._logger.warning(f"Error ending MLflow run: {e}")

        self._workers = []
        self._test_workers = []
        self._attack = None
        self._params = None
