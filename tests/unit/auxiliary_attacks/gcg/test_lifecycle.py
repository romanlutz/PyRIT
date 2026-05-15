# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the deprecated ``generate_suffix`` shim.

The new lifecycle behaviour (worker spawning, teardown on success and failure)
is exercised in detail in ``test_generator.py``. These tests just verify that
the shim still translates its kwargs correctly into a ``GCGGenerator`` +
``execute_async`` call and emits the deprecation warning.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

train_mod = pytest.importorskip(
    "pyrit.auxiliary_attacks.gcg.experiments.train",
    reason="GCG train module not available",
)
Generator = train_mod.GreedyCoordinateGradientAdversarialSuffixGenerator


@patch("pyrit.auxiliary_attacks.gcg.experiments.train.load_goals_and_targets")
def test_generate_suffix_emits_deprecation_warning(mock_loader: MagicMock) -> None:
    mock_loader.return_value = (["goal"], ["target"], [], [])

    with patch.object(train_mod.GCGGenerator, "execute_async", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = MagicMock()
        with pytest.warns(DeprecationWarning, match="generate_suffix"):
            Generator().generate_suffix(
                tokenizer_paths=["test/path"],
                model_paths=["test/path"],
                train_data="",
                n_steps=1,
            )

    mock_execute.assert_awaited_once()


@patch("pyrit.auxiliary_attacks.gcg.experiments.train.load_goals_and_targets")
def test_generate_suffix_passes_loaded_goals_to_execute_async(mock_loader: MagicMock) -> None:
    mock_loader.return_value = (["g1", "g2"], ["t1", "t2"], ["tg"], ["tt"])

    with patch.object(train_mod.GCGGenerator, "execute_async", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = MagicMock()
        with pytest.warns(DeprecationWarning):
            Generator().generate_suffix(
                tokenizer_paths=["test/path"],
                model_paths=["test/path"],
                train_data="some-csv-path",
                n_train_data=2,
                n_test_data=1,
                n_steps=1,
            )

    call_kwargs = mock_execute.await_args.kwargs
    assert call_kwargs["goals"] == ["g1", "g2"]
    assert call_kwargs["targets"] == ["t1", "t2"]
    assert call_kwargs["test_goals"] == ["tg"]
    assert call_kwargs["test_targets"] == ["tt"]


def test_generate_suffix_requires_model_paths() -> None:
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="model_paths must be a non-empty"):
            Generator().generate_suffix(
                tokenizer_paths=["test/path"],
                model_paths=None,
                train_data="",
                n_steps=1,
            )
