# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Integration tests for the GCG workflow.

These tests require the ``gcg`` extra to be installed (torch, mlflow, etc.).
They validate that the workflow correctly wires together the GCG attack
components without actually running a full optimization (which would require
a GPU and take minutes/hours).
"""

from unittest.mock import MagicMock, patch

import pytest

# Skip entire module if GCG dependencies are not installed
torch = pytest.importorskip("torch", reason="GCG integration tests require torch (install with gcg extra)")
mlflow = pytest.importorskip("mlflow", reason="GCG integration tests require mlflow (install with gcg extra)")
pytest.importorskip("ml_collections", reason="GCG integration tests require ml_collections (install with gcg extra)")

from pyrit.auxiliary_attacks.gcg.experiments.train import (  # noqa: E402
    GreedyCoordinateGradientAdversarialSuffixGenerator,
)
from pyrit.executor.workflow.gcg import (  # noqa: E402
    _DEFAULT_CONTROL_INIT,
    GCGContext,
    GCGResult,
    GCGStatus,
    GCGWorkflow,
)


@pytest.fixture
def gcg_workflow() -> GCGWorkflow:
    """Create a GCGWorkflow with realistic but fake model config."""
    return GCGWorkflow(
        model_name="test-model",
        model_paths=["/fake/model/path"],
        tokenizer_paths=["/fake/tokenizer/path"],
        conversation_templates=["vicuna_v1.1"],
        token="hf_fake_token",
        devices=["cpu"],
    )


class TestGCGWorkflowIntegrationSetup:
    """Integration tests for GCG workflow setup with real GCG imports."""

    @pytest.mark.asyncio
    async def test_setup_imports_gcg_modules(self, gcg_workflow: GCGWorkflow) -> None:
        """Test that _setup_async correctly imports real GCG modules."""
        ctx = GCGContext(train_data="https://example.com/data.csv", n_steps=10, batch_size=64)

        # Mock the heavy operations (model loading, data fetching) but use real imports
        mock_params = MagicMock()
        mock_workers = ([MagicMock()], [MagicMock()])

        with (
            patch.object(
                GreedyCoordinateGradientAdversarialSuffixGenerator,
                "_build_params",
                return_value=mock_params,
            ),
            patch(
                "pyrit.auxiliary_attacks.gcg.attack.base.attack_manager.get_goals_and_targets",
                return_value=(["goal"], ["target"], [], []),
            ),
            patch(
                "pyrit.auxiliary_attacks.gcg.attack.base.attack_manager.get_workers",
                return_value=mock_workers,
            ),
            patch.object(
                GreedyCoordinateGradientAdversarialSuffixGenerator,
                "_apply_target_augmentation",
                return_value=(["target_aug"], []),
            ),
            patch.object(
                GreedyCoordinateGradientAdversarialSuffixGenerator,
                "_create_attack",
                return_value=MagicMock(),
            ),
            patch("pyrit.auxiliary_attacks.gcg.experiments.log.mlflow"),
            patch("mlflow.start_run"),
        ):
            await gcg_workflow._setup_async(context=ctx)

            assert gcg_workflow._params is mock_params
            assert gcg_workflow._attack is not None
            assert len(gcg_workflow._workers) == 1
            assert len(gcg_workflow._test_workers) == 1

    @pytest.mark.asyncio
    async def test_teardown_with_real_mlflow(self, gcg_workflow: GCGWorkflow) -> None:
        """Test teardown with real mlflow module available."""
        ctx = GCGContext(train_data="data.csv")
        mock_worker = MagicMock()
        gcg_workflow._workers = [mock_worker]

        with patch("mlflow.end_run"):
            await gcg_workflow._teardown_async(context=ctx)

        mock_worker.stop.assert_called_once()
        assert gcg_workflow._workers == []


class TestGCGWorkflowBuildParams:
    """Integration tests verifying parameter building with real GCG code."""

    def test_build_params_creates_config_dict(self) -> None:
        """Test that _build_params returns a ConfigDict with expected attributes."""
        params = GreedyCoordinateGradientAdversarialSuffixGenerator._build_params(
            token="hf_test",
            model_name="llama-2",
            model_paths=["/model"],
            tokenizer_paths=["/tokenizer"],
            train_data="data.csv",
            n_steps=100,
            batch_size=256,
            learning_rate=0.01,
        )
        assert params.token == "hf_test"
        assert params.model_name == "llama-2"
        assert params.n_steps == 100
        assert params.batch_size == 256
        assert params.learning_rate == 0.01

    def test_build_params_preserves_all_context_fields(self) -> None:
        """Test that all GCGContext fields map correctly to params."""
        ctx = GCGContext(
            train_data="train.csv",
            n_train_data=25,
            n_steps=200,
            batch_size=128,
            learning_rate=0.05,
            topk=512,
            temp=2,
            transfer=True,
            progressive_goals=True,
            stop_on_success=True,
            random_seed=99,
        )
        params = GreedyCoordinateGradientAdversarialSuffixGenerator._build_params(
            token="tok",
            model_name="m",
            model_paths=["/m"],
            tokenizer_paths=["/t"],
            conversation_templates=["v"],
            train_data=ctx.train_data,
            n_train_data=ctx.n_train_data,
            n_steps=ctx.n_steps,
            batch_size=ctx.batch_size,
            learning_rate=ctx.learning_rate,
            topk=ctx.topk,
            temp=ctx.temp,
            transfer=ctx.transfer,
            progressive_goals=ctx.progressive_goals,
            stop_on_success=ctx.stop_on_success,
            random_seed=ctx.random_seed,
        )
        assert params.train_data == "train.csv"
        assert params.n_train_data == 25
        assert params.n_steps == 200
        assert params.batch_size == 128
        assert params.learning_rate == 0.05
        assert params.topk == 512
        assert params.transfer is True
        assert params.progressive_goals is True
        assert params.stop_on_success is True
        assert params.random_seed == 99


class TestGCGWorkflowApplyTargetAugmentation:
    """Integration tests for target augmentation with real GCG code."""

    def test_augmentation_processes_targets(self) -> None:
        """Test that _apply_target_augmentation modifies targets."""
        train_targets = ["Sure, here is a plan"]
        test_targets = ["Sure, here is some info"]

        result_train, result_test = GreedyCoordinateGradientAdversarialSuffixGenerator._apply_target_augmentation(
            train_targets=train_targets,
            test_targets=test_targets,
        )

        # Results should be lists of same length
        assert len(result_train) == len(train_targets)
        assert len(result_test) == len(test_targets)

    def test_augmentation_with_empty_lists(self) -> None:
        """Test augmentation with empty target lists."""
        result_train, result_test = GreedyCoordinateGradientAdversarialSuffixGenerator._apply_target_augmentation(
            train_targets=[],
            test_targets=[],
        )
        assert result_train == []
        assert result_test == []


class TestGCGWorkflowResultIntegration:
    """Integration tests for GCGResult with real workflow context."""

    def test_result_from_successful_attack(self) -> None:
        """Test creating a result from a successful attack outcome."""
        result = GCGResult(
            control_str="embedding adversarial tokens here !!",
            loss=0.15,
            n_steps=500,
        )
        assert result.success is True
        assert result.status == GCGStatus.SUCCESS
        assert result.loss == 0.15

    def test_result_from_failed_attack(self) -> None:
        """Test creating a result from a failed attack (default control unchanged)."""
        result = GCGResult(
            control_str=_DEFAULT_CONTROL_INIT,
            loss=5.0,
            n_steps=1000,
        )
        assert result.success is False
        assert result.status == GCGStatus.FAILURE

    def test_result_round_trip_through_workflow(self) -> None:
        """Test that a GCGResult can be constructed from typical attack.run output."""
        # Simulate MultiPromptAttack.run return
        run_output = ("found adversarial suffix", 0.3, 250)
        control_str, loss, n_steps = run_output

        result = GCGResult(control_str=control_str, loss=loss, n_steps=n_steps)
        assert result.control_str == "found adversarial suffix"
        assert result.loss == 0.3
        assert result.n_steps == 250
        assert result.success is True

    def test_result_from_individual_attack_output(self) -> None:
        """Test result from IndividualPromptAttack.run (2-tuple)."""
        run_output = ("individual suffix", 150)
        control_str, n_steps = run_output

        result = GCGResult(control_str=control_str, n_steps=n_steps)
        assert result.control_str == "individual suffix"
        assert result.loss is None
        assert result.n_steps == 150
        assert result.status == GCGStatus.UNKNOWN


class TestGCGWorkflowEndToEnd:
    """End-to-end integration tests for the full workflow lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_mocked_attack(self) -> None:
        """Test the complete workflow lifecycle with real GCG imports but mocked execution."""
        workflow = GCGWorkflow(
            model_name="test-model",
            model_paths=["/fake/model"],
            tokenizer_paths=["/fake/tokenizer"],
            conversation_templates=["vicuna_v1.1"],
            token="hf_fake_token",
            devices=["cpu"],
        )

        mock_attack = MagicMock()
        mock_attack.run.return_value = ("adversarial result", 0.2, 100)

        mock_params = MagicMock()
        mock_params.n_steps = 100
        mock_params.batch_size = 64
        mock_params.topk = 256
        mock_params.temp = 1
        mock_params.target_weight = 1.0
        mock_params.control_weight = 0.0
        mock_params.test_steps = 10
        mock_params.anneal = False
        mock_params.incr_control = False
        mock_params.stop_on_success = False
        mock_params.verbose = False
        mock_params.filter_cand = True
        mock_params.allow_non_ascii = False

        with (
            patch.object(
                GreedyCoordinateGradientAdversarialSuffixGenerator,
                "_build_params",
                return_value=mock_params,
            ),
            patch(
                "pyrit.auxiliary_attacks.gcg.attack.base.attack_manager.get_goals_and_targets",
                return_value=(["goal"], ["target"], [], []),
            ),
            patch(
                "pyrit.auxiliary_attacks.gcg.attack.base.attack_manager.get_workers",
                return_value=([MagicMock()], []),
            ),
            patch.object(
                GreedyCoordinateGradientAdversarialSuffixGenerator,
                "_apply_target_augmentation",
                return_value=(["target"], []),
            ),
            patch.object(
                GreedyCoordinateGradientAdversarialSuffixGenerator,
                "_create_attack",
                return_value=mock_attack,
            ),
            patch("pyrit.auxiliary_attacks.gcg.experiments.log.mlflow"),
            patch("mlflow.start_run"),
            patch("mlflow.end_run"),
        ):
            result = await workflow.execute_with_context_async(
                context=GCGContext(
                    train_data="data.csv",
                    n_steps=100,
                    batch_size=64,
                )
            )

            assert isinstance(result, GCGResult)
            assert result.control_str == "adversarial result"
            assert result.loss == 0.2
            assert result.n_steps == 100
            assert result.success is True
            assert result.status == GCGStatus.SUCCESS
            mock_attack.run.assert_called_once()
