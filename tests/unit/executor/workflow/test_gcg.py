# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.executor.workflow.gcg import (
    _DEFAULT_CONTROL_INIT,
    GCGContext,
    GCGResult,
    GCGStatus,
    GCGWorkflow,
)

# --- Fixtures ---


@pytest.fixture
def model_paths() -> list[str]:
    return ["/models/llama-2"]


@pytest.fixture
def tokenizer_paths() -> list[str]:
    return ["/tokenizers/llama-2"]


@pytest.fixture
def conversation_templates() -> list[str]:
    return ["llama-2"]


@pytest.fixture
def token() -> str:
    return "hf_test_token_12345"


@pytest.fixture
def workflow(
    model_paths: list[str],
    tokenizer_paths: list[str],
    conversation_templates: list[str],
    token: str,
) -> GCGWorkflow:
    """Create a GCGWorkflow instance for testing."""
    return GCGWorkflow(
        model_name="llama-2",
        model_paths=model_paths,
        tokenizer_paths=tokenizer_paths,
        conversation_templates=conversation_templates,
        token=token,
    )


@pytest.fixture
def valid_context() -> GCGContext:
    """Create a valid GCG context for testing."""
    return GCGContext(
        train_data="https://example.com/data.csv",
        n_train_data=10,
        n_steps=100,
        batch_size=256,
    )


@pytest.fixture
def minimal_context() -> GCGContext:
    """Create a minimal context with just train_data."""
    return GCGContext(train_data="data.csv")


# --- GCGContext Tests ---


class TestGCGContext:
    """Tests for GCGContext dataclass."""

    def test_default_values(self) -> None:
        """Test that GCGContext has correct default values."""
        ctx = GCGContext()
        assert ctx.train_data == ""
        assert ctx.n_train_data == 50
        assert ctx.n_steps == 500
        assert ctx.test_steps == 50
        assert ctx.batch_size == 512
        assert ctx.learning_rate == 0.01
        assert ctx.topk == 256
        assert ctx.temp == 1
        assert ctx.target_weight == 1.0
        assert ctx.control_weight == 0.0
        assert ctx.transfer is False
        assert ctx.progressive_goals is False
        assert ctx.progressive_models is False
        assert ctx.anneal is False
        assert ctx.incr_control is False
        assert ctx.stop_on_success is False
        assert ctx.allow_non_ascii is False
        assert ctx.filter_cand is True
        assert ctx.gbda_deterministic is True
        assert ctx.verbose is True
        assert ctx.random_seed == 42
        assert ctx.control_init == _DEFAULT_CONTROL_INIT

    def test_custom_values(self) -> None:
        """Test GCGContext with custom values."""
        ctx = GCGContext(
            train_data="data.csv",
            n_train_data=100,
            n_steps=1000,
            batch_size=1024,
            learning_rate=0.05,
            transfer=True,
            stop_on_success=True,
        )
        assert ctx.train_data == "data.csv"
        assert ctx.n_train_data == 100
        assert ctx.n_steps == 1000
        assert ctx.batch_size == 1024
        assert ctx.learning_rate == 0.05
        assert ctx.transfer is True
        assert ctx.stop_on_success is True

    def test_duplicate_creates_independent_copy(self) -> None:
        """Test that duplicate creates a deep copy."""
        ctx = GCGContext(train_data="data.csv", n_steps=100)
        ctx_copy = ctx.duplicate()
        ctx_copy.n_steps = 999
        assert ctx.n_steps == 100
        assert ctx_copy.n_steps == 999


# --- GCGResult Tests ---


class TestGCGResult:
    """Tests for GCGResult dataclass."""

    def test_success_with_non_default_control(self) -> None:
        """Test success is True when control_str differs from default."""
        result = GCGResult(control_str="adversarial suffix here", n_steps=100, loss=0.5)
        assert result.success is True

    def test_success_false_with_default_control(self) -> None:
        """Test success is False when control_str matches default."""
        result = GCGResult(control_str=_DEFAULT_CONTROL_INIT, n_steps=100, loss=1.0)
        assert result.success is False

    def test_success_false_with_empty_control(self) -> None:
        """Test success is False when control_str is empty."""
        result = GCGResult(control_str="", n_steps=100, loss=2.0)
        assert result.success is False

    def test_status_success(self) -> None:
        """Test status is SUCCESS when attack succeeded."""
        result = GCGResult(control_str="found suffix", n_steps=100, loss=0.1)
        assert result.status == GCGStatus.SUCCESS

    def test_status_failure(self) -> None:
        """Test status is FAILURE when attack did not find a suffix."""
        result = GCGResult(control_str=_DEFAULT_CONTROL_INIT, n_steps=100, loss=5.0)
        assert result.status == GCGStatus.FAILURE

    def test_status_unknown_no_loss(self) -> None:
        """Test status is UNKNOWN when loss is None."""
        result = GCGResult(control_str="some suffix", n_steps=50, loss=None)
        assert result.status == GCGStatus.UNKNOWN

    def test_loss_optional(self) -> None:
        """Test that loss defaults to None."""
        result = GCGResult(control_str="suffix", n_steps=10)
        assert result.loss is None
        assert result.n_steps == 10


# --- GCGStatus Tests ---


class TestGCGStatus:
    """Tests for GCGStatus enum."""

    def test_enum_values(self) -> None:
        """Test that all status values exist."""
        assert GCGStatus.SUCCESS.value == "success"
        assert GCGStatus.FAILURE.value == "failure"
        assert GCGStatus.UNKNOWN.value == "unknown"


# --- GCGWorkflow Initialization Tests ---


class TestGCGWorkflowInit:
    """Tests for GCGWorkflow initialization."""

    def test_init_with_required_params(
        self,
        model_paths: list[str],
        tokenizer_paths: list[str],
        conversation_templates: list[str],
        token: str,
    ) -> None:
        """Test initialization with required parameters only."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=model_paths,
            tokenizer_paths=tokenizer_paths,
            conversation_templates=conversation_templates,
            token=token,
        )
        assert wf._model_name == "llama-2"
        assert wf._model_paths == model_paths
        assert wf._tokenizer_paths == tokenizer_paths
        assert wf._conversation_templates == conversation_templates
        assert wf._token == token
        assert wf._num_train_models == 1
        assert wf._devices == ["cuda:0"]
        assert wf._workers == []
        assert wf._test_workers == []
        assert wf._attack is None

    def test_init_with_all_optional_params(
        self,
        model_paths: list[str],
        tokenizer_paths: list[str],
        conversation_templates: list[str],
        token: str,
    ) -> None:
        """Test initialization with all optional parameters."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=model_paths,
            tokenizer_paths=tokenizer_paths,
            conversation_templates=conversation_templates,
            token=token,
            num_train_models=2,
            devices=["cuda:0", "cuda:1"],
            model_kwargs=[{"low_cpu_mem_usage": True}],
            tokenizer_kwargs=[{"use_fast": True}],
        )
        assert wf._num_train_models == 2
        assert wf._devices == ["cuda:0", "cuda:1"]
        assert wf._model_kwargs == [{"low_cpu_mem_usage": True}]
        assert wf._tokenizer_kwargs == [{"use_fast": True}]

    def test_init_defaults_devices(self, workflow: GCGWorkflow) -> None:
        """Test that devices defaults to ['cuda:0']."""
        assert workflow._devices == ["cuda:0"]

    def test_init_defaults_model_kwargs(self, workflow: GCGWorkflow) -> None:
        """Test that model_kwargs defaults to standard config."""
        assert workflow._model_kwargs == [{"low_cpu_mem_usage": True, "use_cache": False}]

    def test_init_defaults_tokenizer_kwargs(self, workflow: GCGWorkflow) -> None:
        """Test that tokenizer_kwargs defaults to standard config."""
        assert workflow._tokenizer_kwargs == [{"use_fast": False}]


# --- GCGWorkflow Validation Tests ---


class TestGCGWorkflowValidation:
    """Tests for GCGWorkflow context validation."""

    def test_validate_context_with_valid_context(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that validation passes with valid context."""
        workflow._validate_context(context=valid_context)

    def test_validate_empty_model_paths_raises(
        self,
        tokenizer_paths: list[str],
        conversation_templates: list[str],
        token: str,
        valid_context: GCGContext,
    ) -> None:
        """Test that empty model_paths raises ValueError."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=[],
            tokenizer_paths=tokenizer_paths,
            conversation_templates=conversation_templates,
            token=token,
        )
        with pytest.raises(ValueError, match="model_paths must be provided"):
            wf._validate_context(context=valid_context)

    def test_validate_empty_tokenizer_paths_raises(
        self,
        model_paths: list[str],
        conversation_templates: list[str],
        token: str,
        valid_context: GCGContext,
    ) -> None:
        """Test that empty tokenizer_paths raises ValueError."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=model_paths,
            tokenizer_paths=[],
            conversation_templates=conversation_templates,
            token=token,
        )
        with pytest.raises(ValueError, match="tokenizer_paths must be provided"):
            wf._validate_context(context=valid_context)

    def test_validate_mismatched_model_tokenizer_paths_raises(
        self,
        conversation_templates: list[str],
        token: str,
        valid_context: GCGContext,
    ) -> None:
        """Test that mismatched model/tokenizer path lengths raise ValueError."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=["/model1", "/model2"],
            tokenizer_paths=["/tok1"],
            conversation_templates=conversation_templates,
            token=token,
        )
        with pytest.raises(ValueError, match="must have the same length"):
            wf._validate_context(context=valid_context)

    def test_validate_empty_conversation_templates_raises(
        self,
        model_paths: list[str],
        tokenizer_paths: list[str],
        token: str,
        valid_context: GCGContext,
    ) -> None:
        """Test that empty conversation_templates raises ValueError."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=model_paths,
            tokenizer_paths=tokenizer_paths,
            conversation_templates=[],
            token=token,
        )
        with pytest.raises(ValueError, match="conversation_templates must be provided"):
            wf._validate_context(context=valid_context)

    def test_validate_empty_token_raises(
        self,
        model_paths: list[str],
        tokenizer_paths: list[str],
        conversation_templates: list[str],
        valid_context: GCGContext,
    ) -> None:
        """Test that empty token raises ValueError."""
        wf = GCGWorkflow(
            model_name="llama-2",
            model_paths=model_paths,
            tokenizer_paths=tokenizer_paths,
            conversation_templates=conversation_templates,
            token="",
        )
        with pytest.raises(ValueError, match="token.*must be provided"):
            wf._validate_context(context=valid_context)

    def test_validate_empty_train_data_raises(self, workflow: GCGWorkflow) -> None:
        """Test that empty train_data raises ValueError."""
        ctx = GCGContext(train_data="")
        with pytest.raises(ValueError, match="train_data must be provided"):
            workflow._validate_context(context=ctx)

    def test_validate_zero_n_steps_raises(self, workflow: GCGWorkflow) -> None:
        """Test that n_steps=0 raises ValueError."""
        ctx = GCGContext(train_data="data.csv", n_steps=0)
        with pytest.raises(ValueError, match="n_steps must be positive"):
            workflow._validate_context(context=ctx)

    def test_validate_negative_n_steps_raises(self, workflow: GCGWorkflow) -> None:
        """Test that negative n_steps raises ValueError."""
        ctx = GCGContext(train_data="data.csv", n_steps=-1)
        with pytest.raises(ValueError, match="n_steps must be positive"):
            workflow._validate_context(context=ctx)

    def test_validate_zero_batch_size_raises(self, workflow: GCGWorkflow) -> None:
        """Test that batch_size=0 raises ValueError."""
        ctx = GCGContext(train_data="data.csv", batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            workflow._validate_context(context=ctx)


# --- GCGWorkflow Setup Tests ---


@pytest.mark.usefixtures("patch_central_database")
class TestGCGWorkflowSetup:
    """Tests for GCGWorkflow setup method."""

    @pytest.mark.asyncio
    async def test_setup_builds_params_and_creates_attack(
        self, workflow: GCGWorkflow, valid_context: GCGContext
    ) -> None:
        """Test that _setup_async builds params, loads data, and creates attack."""
        mock_workers = [MagicMock(), MagicMock()]
        mock_test_workers = [MagicMock()]
        mock_attack = MagicMock()

        mock_build = MagicMock()
        mock_params = MagicMock()
        mock_build.return_value = mock_params

        mock_goals = MagicMock(return_value=(["goal1"], ["target1"], ["tgoal"], ["ttarget"]))
        mock_augment = MagicMock(return_value=(["target1_aug"], ["ttarget_aug"]))
        mock_get_workers = MagicMock(return_value=(mock_workers, mock_test_workers))
        mock_create = MagicMock(return_value=mock_attack)

        # Mock all GCG dependencies that are lazily imported
        mock_attack_lib = MagicMock()
        mock_train_module = MagicMock()
        mock_train_module.GreedyCoordinateGradientAdversarialSuffixGenerator._build_params = mock_build
        mock_train_module.GreedyCoordinateGradientAdversarialSuffixGenerator._apply_target_augmentation = mock_augment
        mock_train_module.GreedyCoordinateGradientAdversarialSuffixGenerator._create_attack = mock_create
        mock_attack_manager = MagicMock()
        mock_attack_manager.get_goals_and_targets = mock_goals
        mock_attack_manager.get_workers = mock_get_workers
        mock_log_module = MagicMock()
        mock_mlflow = MagicMock()

        mock_gcg_attack_module = MagicMock()
        mock_gcg_attack_module.GCGAttackPrompt = mock_attack_lib.GCGAttackPrompt
        mock_gcg_attack_module.GCGPromptManager = mock_attack_lib.GCGPromptManager
        mock_gcg_attack_module.GCGMultiPromptAttack = mock_attack_lib.GCGMultiPromptAttack

        with patch.dict(
            "sys.modules",
            {
                "mlflow": mock_mlflow,
                "pyrit.auxiliary_attacks.gcg.attack.gcg": mock_attack_lib,
                "pyrit.auxiliary_attacks.gcg.attack.gcg.gcg_attack": mock_gcg_attack_module,
                "pyrit.auxiliary_attacks.gcg.attack": MagicMock(gcg=mock_attack_lib),
                "pyrit.auxiliary_attacks.gcg.experiments.train": mock_train_module,
                "pyrit.auxiliary_attacks.gcg.experiments.log": mock_log_module,
                "pyrit.auxiliary_attacks.gcg.attack.base.attack_manager": mock_attack_manager,
            },
        ):
            await workflow._setup_async(context=valid_context)

            # Verify params were built
            mock_build.assert_called_once()
            build_kwargs = mock_build.call_args[1]
            assert build_kwargs["train_data"] == valid_context.train_data
            assert build_kwargs["n_steps"] == valid_context.n_steps
            assert build_kwargs["token"] == workflow._token

            # Verify mlflow started
            mock_mlflow.start_run.assert_called_once()

            # Verify goals loaded and augmented
            mock_goals.assert_called_once_with(mock_params)
            mock_augment.assert_called_once()

            # Verify workers created
            mock_get_workers.assert_called_once_with(mock_params)

            # Verify attack created
            mock_create.assert_called_once()
            assert workflow._attack is mock_attack
            assert workflow._workers is mock_workers
            assert workflow._test_workers is mock_test_workers
            assert workflow._params is mock_params


# --- GCGWorkflow Perform Tests ---


@pytest.mark.usefixtures("patch_central_database")
class TestGCGWorkflowPerform:
    """Tests for GCGWorkflow perform method."""

    @pytest.mark.asyncio
    async def test_perform_returns_gcg_result_three_tuple(
        self, workflow: GCGWorkflow, valid_context: GCGContext
    ) -> None:
        """Test _perform_async with 3-element return (control, loss, steps)."""
        mock_attack = MagicMock()
        mock_attack.run.return_value = ("adversarial suffix", 0.25, 100)
        workflow._attack = mock_attack
        workflow._params = MagicMock()

        result = await workflow._perform_async(context=valid_context)

        assert isinstance(result, GCGResult)
        assert result.control_str == "adversarial suffix"
        assert result.loss == 0.25
        assert result.n_steps == 100
        mock_attack.run.assert_called_once()

    @pytest.mark.asyncio
    async def test_perform_returns_gcg_result_two_tuple(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test _perform_async with 2-element return (control, steps)."""
        mock_attack = MagicMock()
        mock_attack.run.return_value = ("found suffix", 50)
        workflow._attack = mock_attack
        workflow._params = MagicMock()

        result = await workflow._perform_async(context=valid_context)

        assert isinstance(result, GCGResult)
        assert result.control_str == "found suffix"
        assert result.loss is None
        assert result.n_steps == 50

    @pytest.mark.asyncio
    async def test_perform_passes_correct_params_to_run(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that _perform_async passes params correctly to attack.run."""
        mock_attack = MagicMock()
        mock_attack.run.return_value = ("suffix", 0.1, 100)
        workflow._attack = mock_attack

        mock_params = MagicMock()
        mock_params.n_steps = 200
        mock_params.batch_size = 512
        mock_params.topk = 128
        mock_params.temp = 1
        mock_params.target_weight = 1.0
        mock_params.control_weight = 0.0
        mock_params.test_steps = 50
        mock_params.anneal = False
        mock_params.incr_control = False
        mock_params.stop_on_success = True
        mock_params.verbose = True
        mock_params.filter_cand = True
        mock_params.allow_non_ascii = False
        workflow._params = mock_params

        await workflow._perform_async(context=valid_context)

        call_kwargs = mock_attack.run.call_args[1]
        assert call_kwargs["n_steps"] == 200
        assert call_kwargs["batch_size"] == 512
        assert call_kwargs["topk"] == 128
        assert call_kwargs["stop_on_success"] is True


# --- GCGWorkflow Teardown Tests ---


@pytest.mark.usefixtures("patch_central_database")
class TestGCGWorkflowTeardown:
    """Tests for GCGWorkflow teardown method."""

    @pytest.mark.asyncio
    async def test_teardown_stops_all_workers(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that teardown stops all workers."""
        worker1 = MagicMock()
        worker2 = MagicMock()
        test_worker = MagicMock()
        workflow._workers = [worker1, worker2]
        workflow._test_workers = [test_worker]

        with patch.dict("sys.modules", {"mlflow": MagicMock()}):
            await workflow._teardown_async(context=valid_context)

        worker1.stop.assert_called_once()
        worker2.stop.assert_called_once()
        test_worker.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_teardown_ends_mlflow_run(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that teardown ends MLflow run."""
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            await workflow._teardown_async(context=valid_context)
            mock_mlflow.end_run.assert_called_once()

    @pytest.mark.asyncio
    async def test_teardown_clears_state(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that teardown clears internal state."""
        workflow._workers = [MagicMock()]
        workflow._test_workers = [MagicMock()]
        workflow._attack = MagicMock()
        workflow._params = MagicMock()

        with patch.dict("sys.modules", {"mlflow": MagicMock()}):
            await workflow._teardown_async(context=valid_context)

        assert workflow._workers == []
        assert workflow._test_workers == []
        assert workflow._attack is None
        assert workflow._params is None

    @pytest.mark.asyncio
    async def test_teardown_handles_worker_stop_errors(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that teardown handles errors in worker.stop gracefully."""
        failing_worker = MagicMock()
        failing_worker.stop.side_effect = RuntimeError("GPU error")
        ok_worker = MagicMock()
        workflow._workers = [failing_worker, ok_worker]

        with patch.dict("sys.modules", {"mlflow": MagicMock()}):
            await workflow._teardown_async(context=valid_context)

        # Both workers should have been attempted
        failing_worker.stop.assert_called_once()
        ok_worker.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_teardown_handles_mlflow_error(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that teardown handles MLflow errors gracefully."""
        mock_mlflow = MagicMock()
        mock_mlflow.end_run.side_effect = Exception("MLflow connection error")
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            # Should not raise
            await workflow._teardown_async(context=valid_context)

    @pytest.mark.asyncio
    async def test_teardown_handles_missing_mlflow(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test that teardown works when mlflow is not installed."""
        # Without mlflow in sys.modules, import will fail but should be caught
        await workflow._teardown_async(context=valid_context)


# --- GCGWorkflow Execute Tests ---


@pytest.mark.usefixtures("patch_central_database")
class TestGCGWorkflowExecute:
    """Tests for GCGWorkflow execute_async end-to-end."""

    @pytest.mark.asyncio
    async def test_execute_async_full_lifecycle(self, workflow: GCGWorkflow) -> None:
        """Test that execute_async runs the full lifecycle."""
        with (
            patch.object(workflow, "_validate_context") as mock_validate,
            patch.object(workflow, "_setup_async", new_callable=AsyncMock) as mock_setup,
            patch.object(workflow, "_perform_async", new_callable=AsyncMock) as mock_perform,
            patch.object(workflow, "_teardown_async", new_callable=AsyncMock) as mock_teardown,
        ):
            expected_result = GCGResult(control_str="found", n_steps=100, loss=0.1)
            mock_perform.return_value = expected_result

            result = await workflow.execute_async(
                train_data="data.csv",
                n_steps=100,
                batch_size=256,
            )

            assert result == expected_result
            mock_validate.assert_called_once()
            mock_setup.assert_called_once()
            mock_perform.assert_called_once()
            mock_teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_teardown_runs_on_perform_error(self, workflow: GCGWorkflow) -> None:
        """Test that teardown runs even if perform raises an exception."""
        with (
            patch.object(workflow, "_validate_context"),
            patch.object(workflow, "_setup_async", new_callable=AsyncMock),
            patch.object(workflow, "_perform_async", new_callable=AsyncMock) as mock_perform,
            patch.object(workflow, "_teardown_async", new_callable=AsyncMock) as mock_teardown,
        ):
            mock_perform.side_effect = RuntimeError("GPU out of memory")

            with pytest.raises(RuntimeError):
                await workflow.execute_async(train_data="data.csv")

            # Teardown should still be called
            mock_teardown.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_async_with_context_object(self, workflow: GCGWorkflow) -> None:
        """Test execute_with_context_async with a GCGContext object."""
        ctx = GCGContext(train_data="data.csv", n_steps=50)

        with (
            patch.object(workflow, "_validate_context"),
            patch.object(workflow, "_setup_async", new_callable=AsyncMock),
            patch.object(workflow, "_perform_async", new_callable=AsyncMock) as mock_perform,
            patch.object(workflow, "_teardown_async", new_callable=AsyncMock),
        ):
            expected = GCGResult(control_str="suffix", n_steps=50, loss=0.5)
            mock_perform.return_value = expected

            result = await workflow.execute_with_context_async(context=ctx)
            assert result == expected


# --- GCGWorkflow Run Attack Tests ---


class TestGCGWorkflowRunAttack:
    """Tests for the synchronous _run_attack helper."""

    def test_run_attack_three_tuple_result(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test _run_attack with 3-element tuple (MultiPromptAttack.run)."""
        mock_attack = MagicMock()
        mock_attack.run.return_value = ("control string", 0.3, 200)
        workflow._attack = mock_attack
        workflow._params = MagicMock()

        result = workflow._run_attack(valid_context)

        assert result.control_str == "control string"
        assert result.loss == 0.3
        assert result.n_steps == 200

    def test_run_attack_two_tuple_result(self, workflow: GCGWorkflow, valid_context: GCGContext) -> None:
        """Test _run_attack with 2-element tuple (IndividualPromptAttack.run)."""
        mock_attack = MagicMock()
        mock_attack.run.return_value = ("individual suffix", 150)
        workflow._attack = mock_attack
        workflow._params = MagicMock()

        result = workflow._run_attack(valid_context)

        assert result.control_str == "individual suffix"
        assert result.loss is None
        assert result.n_steps == 150
