# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# ruff: noqa: E402

from unittest.mock import MagicMock

import pytest

pytest.importorskip(
    "pyrit.executor.promptgen.gcg.attack.base.attack_manager",
    reason="GCG optional dependencies (torch, etc.) not installed",
)
torch = pytest.importorskip("torch", reason="torch not installed")

from pyrit.executor.promptgen.gcg.attack.base.attack_manager import ModelWorkerOperation
from pyrit.executor.promptgen.gcg.attack.gcg.candidate_proposer import (
    CandidateProposalBatch,
    GCGCandidateProposer,
)
from pyrit.executor.promptgen.gcg.extension_protocols import CandidateFilter, SamplingStrategy


def _create_mock_worker(gradient_shape: tuple[int, ...]) -> MagicMock:
    """Create a mock ModelWorker returning a gradient of the specified shape."""
    worker = MagicMock()
    worker.tokenizer = MagicMock()
    worker.tokenizer.decode.side_effect = lambda ids, **kwargs: f"tok_{ids[0]}"
    worker.results = MagicMock()
    # Return a tensor of ones for the specified gradient shape
    worker.results.get.side_effect = lambda: torch.ones(gradient_shape, dtype=torch.float32)
    return worker


def _create_mock_prompt_manager() -> MagicMock:
    """Create a mock PromptManager holding control tokens."""
    prompt_manager = MagicMock()
    prompt_manager.control_toks = torch.tensor([1, 2, 3], dtype=torch.long)
    prompt_manager.disallowed_toks = None
    return prompt_manager


class TestCandidateProposalBatch:
    """Unit tests for the CandidateProposalBatch dataclass."""

    def test_num_groups_property(self) -> None:
        batch = CandidateProposalBatch(
            control_candidates_by_group=[["cand1", "cand2"], ["cand3"]],
            group_worker_indices=[0, 1],
        )
        assert batch.num_groups == 2
        assert batch.control_candidates_by_group == [["cand1", "cand2"], ["cand3"]]
        assert batch.group_worker_indices == [0, 1]


class TestGCGCandidateProposer:
    """Unit tests for the GCGCandidateProposer component."""

    def test_empty_workers_raises_value_error(self) -> None:
        sampling = MagicMock(spec=SamplingStrategy)
        candidate_filter = MagicMock(spec=CandidateFilter)
        with pytest.raises(ValueError, match="at least one worker"):
            GCGCandidateProposer(
                workers=[],
                prompts=[],
                sampling=sampling,
                candidate_filter=candidate_filter,
                main_device=torch.device("cpu"),
            )

    def test_worker_and_prompt_count_mismatch_raises_value_error(self) -> None:
        worker = _create_mock_worker((1, 10, 20))
        sampling = MagicMock(spec=SamplingStrategy)
        candidate_filter = MagicMock(spec=CandidateFilter)
        with pytest.raises(ValueError, match="count mismatch"):
            GCGCandidateProposer(
                workers=[worker],
                prompts=[],
                sampling=sampling,
                candidate_filter=candidate_filter,
                main_device=torch.device("cpu"),
            )

    def test_single_worker_gradient_dispatch_and_normalization(self) -> None:
        worker = _create_mock_worker((1, 3, 4))
        prompt_manager = _create_mock_prompt_manager()
        sampling = MagicMock(spec=SamplingStrategy)
        sampling.sample_candidates.return_value = torch.tensor([[10], [20]])
        candidate_filter = MagicMock(spec=CandidateFilter)
        candidate_filter.filter_candidates.return_value = ["filtered_1", "filtered_2"]

        proposer = GCGCandidateProposer(
            workers=[worker],
            prompts=[prompt_manager],
            sampling=sampling,
            candidate_filter=candidate_filter,
            main_device=torch.device("cpu"),
        )

        batch = proposer.propose_candidates(
            batch_size=2,
            topk=5,
            temp=1.0,
            allow_non_ascii=True,
            filter_cand=True,
            current_control_str="! ! !",
        )

        # Worker was dispatched with GRAD operation
        worker.assert_called_once_with(prompt_manager, ModelWorkerOperation.GRAD)
        # Results queue was queried
        worker.results.get.assert_called_once()
        # Sampling received normalized gradient (norm along dim=-1 should be ~1.0)
        call_kwargs = sampling.sample_candidates.call_args.kwargs
        grad_arg = call_kwargs["gradient"]
        norms = grad_arg.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms))
        # Candidate filter received sampled tensor and tokenizer
        candidate_filter.filter_candidates.assert_called_once_with(
            candidate_tokens=sampling.sample_candidates.return_value,
            tokenizer=worker.tokenizer,
            current_control="! ! !",
        )
        assert batch.num_groups == 1
        assert batch.control_candidates_by_group == [["filtered_1", "filtered_2"]]
        assert batch.group_worker_indices == [0]

    def test_same_shape_multiple_workers_accumulate_into_single_group(self) -> None:
        worker1 = _create_mock_worker((1, 3, 4))
        worker2 = _create_mock_worker((1, 3, 4))
        pm1 = _create_mock_prompt_manager()
        pm2 = _create_mock_prompt_manager()
        sampling = MagicMock(spec=SamplingStrategy)
        sampling.sample_candidates.return_value = torch.tensor([[10]])
        candidate_filter = MagicMock(spec=CandidateFilter)
        candidate_filter.filter_candidates.return_value = ["res"]

        proposer = GCGCandidateProposer(
            workers=[worker1, worker2],
            prompts=[pm1, pm2],
            sampling=sampling,
            candidate_filter=candidate_filter,
            main_device=torch.device("cpu"),
        )

        batch = proposer.propose_candidates(
            batch_size=1,
            topk=2,
            temp=1.0,
            allow_non_ascii=True,
            filter_cand=True,
            current_control_str="test",
        )

        assert batch.num_groups == 1
        # Accumulated gradient should have norm 2.0 along dim=-1 (sum of 2 unit-norm gradients)
        call_kwargs = sampling.sample_candidates.call_args.kwargs
        grad_arg = call_kwargs["gradient"]
        norms = grad_arg.norm(dim=-1)
        assert torch.allclose(norms, torch.full_like(norms, 2.0))
        # Sampling should only have been called once for the unified group
        assert sampling.sample_candidates.call_count == 1
        assert batch.group_worker_indices == [1]

    def test_mismatched_shapes_creates_multiple_candidate_groups(self) -> None:
        worker1 = _create_mock_worker((1, 3, 4))
        worker2 = _create_mock_worker((1, 3, 8))  # Different vocab/embedding dimension
        pm1 = _create_mock_prompt_manager()
        pm2 = _create_mock_prompt_manager()
        sampling = MagicMock(spec=SamplingStrategy)
        sampling.sample_candidates.side_effect = [
            torch.tensor([[10], [11]]),
            torch.tensor([[20], [21]]),
        ]
        candidate_filter = MagicMock(spec=CandidateFilter)
        candidate_filter.filter_candidates.side_effect = [
            ["g1_c1", "g1_c2"],
            ["g2_c1", "g2_c2"],
        ]

        proposer = GCGCandidateProposer(
            workers=[worker1, worker2],
            prompts=[pm1, pm2],
            sampling=sampling,
            candidate_filter=candidate_filter,
            main_device=torch.device("cpu"),
        )

        batch = proposer.propose_candidates(
            batch_size=2,
            topk=5,
            temp=1.0,
            allow_non_ascii=True,
            filter_cand=True,
            current_control_str="test",
        )

        assert batch.num_groups == 2
        assert batch.control_candidates_by_group == [["g1_c1", "g1_c2"], ["g2_c1", "g2_c2"]]
        assert batch.group_worker_indices == [0, 1]
        assert sampling.sample_candidates.call_count == 2
        assert candidate_filter.filter_candidates.call_count == 2
