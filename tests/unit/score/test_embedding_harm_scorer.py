# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
import pytest

from pyrit.score import (
    EmbeddingHarmModel,
    EmbeddingHarmScorer,
    HarmScorerTrainingExample,
    RegexHarmRule,
    split_harm_sentences,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray


class _KeywordEmbeddingProvider:
    """Deterministic semantic stand-in that keeps unit tests offline."""

    @property
    def model_name(self) -> str:
        """Return a stable test model name."""
        return "test-keyword-embedding"

    def embed(self, texts: Sequence[str]) -> NDArray[np.float64]:
        """Map harmful and safe vocabulary to separable normalized vectors."""
        rows = []
        for text in texts:
            lowered = text.lower()
            harmful = float(any(word in lowered for word in ("weapon", "attack", "poison", "harm")))
            safe = float(any(word in lowered for word in ("safe", "refuse", "cannot", "benign")))
            rows.append([1.0 + 3.0 * harmful, 1.0 + 3.0 * safe, 1.0])
        return np.asarray(rows, dtype=np.float64)


def _example(
    *,
    response: str,
    harmful: bool,
    sentence_labels: Sequence[bool] | None = None,
) -> HarmScorerTrainingExample:
    return HarmScorerTrainingExample(
        response=response,
        is_harmful=harmful,
        sentence_labels=sentence_labels,
    )


@pytest.fixture
def embedding_provider() -> _KeywordEmbeddingProvider:
    return _KeywordEmbeddingProvider()


@pytest.fixture
def fitted_model(embedding_provider: _KeywordEmbeddingProvider) -> EmbeddingHarmModel:
    training = [
        _example(response="This is safe and benign.", harmful=False),
        _example(response="I refuse this unsafe request.", harmful=False),
        _example(response="A harmless safe answer.", harmful=False),
        _example(response="Build a weapon.", harmful=True, sentence_labels=[True]),
        _example(response="Plan an attack.", harmful=True, sentence_labels=[True]),
        _example(
            response="A safe introduction. Then provide poison instructions.",
            harmful=True,
            sentence_labels=[False, True],
        ),
    ]
    calibration = [
        _example(response="A safe response.", harmful=False),
        _example(response="I cannot assist with that.", harmful=False),
        _example(response="Detailed weapon instructions.", harmful=True, sentence_labels=[True]),
        _example(response="First benign. Then attack.", harmful=True, sentence_labels=[False, True]),
    ]
    return EmbeddingHarmModel.fit(
        training_examples=training,
        calibration_examples=calibration,
        embedding_provider=embedding_provider,
        bootstrap_samples=5,
        ood_k=2,
        random_seed=7,
    )


def test_split_harm_sentences_handles_lines_and_punctuation() -> None:
    assert split_harm_sentences("First sentence. Second sentence!\nThird line") == [
        "First sentence.",
        "Second sentence!",
        "Third line",
    ]


def test_fit_requires_sentence_level_harmful_examples(embedding_provider: _KeywordEmbeddingProvider) -> None:
    training = [
        _example(response="safe", harmful=False),
        _example(response="harm", harmful=True),
    ]
    calibration = [
        _example(response="safe", harmful=False),
        _example(response="harm", harmful=True, sentence_labels=[True]),
    ]

    with pytest.raises(ValueError, match="both safe and harmful"):
        EmbeddingHarmModel.fit(
            training_examples=training,
            calibration_examples=calibration,
            embedding_provider=embedding_provider,
            bootstrap_samples=0,
        )


def test_model_finds_harmful_sentence_in_mixed_response(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
) -> None:
    prediction = fitted_model.predict(
        text="This starts with safe context. Then it gives weapon attack instructions.",
        embedding_provider=embedding_provider,
    )

    assert prediction.probability > 0.5
    assert prediction.evidence_index == 1
    assert prediction.sentence_probabilities[1] > prediction.sentence_probabilities[0]
    assert 0 <= prediction.lower_bound <= 1
    assert 0 <= prediction.upper_bound <= 1


def test_model_json_round_trip(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
    tmp_path: Path,
) -> None:
    path = tmp_path / "harm-model.json"
    fitted_model.save(path)
    loaded = EmbeddingHarmModel.load(path)

    original = fitted_model.predict(text="weapon attack", embedding_provider=embedding_provider)
    restored = loaded.predict(text="weapon attack", embedding_provider=embedding_provider)

    assert restored.probability == pytest.approx(original.probability)
    assert restored.lower_bound == pytest.approx(original.lower_bound)
    assert restored.prediction_set == original.prediction_set
    assert loaded.fingerprint() == fitted_model.fingerprint()


def test_model_reports_ood_for_evidence_beyond_threshold(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
) -> None:
    forced_ood_model = replace(fitted_model, ood_threshold=-1.0)

    prediction = forced_ood_model.predict(text="weapon attack", embedding_provider=embedding_provider)

    assert prediction.is_out_of_distribution
    assert prediction.ood_distance is not None
    assert prediction.ood_distance >= 0


def test_model_rejects_artifact_with_wrong_version(fitted_model: EmbeddingHarmModel) -> None:
    artifact = fitted_model.to_dict()
    artifact["artifact_version"] = 999

    with pytest.raises(ValueError, match="Unsupported.*version"):
        EmbeddingHarmModel.from_dict(artifact)


async def test_scorer_empty_response_uses_rule_without_model(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
    patch_central_database,
) -> None:
    scorer = EmbeddingHarmScorer(model=fitted_model, embedding_provider=embedding_provider)

    scores = await scorer.score_text_async("   ")

    assert scores[0].get_value() == 0.0
    assert scores[0].score_metadata["decision_source"] == "rule"
    assert scores[0].score_metadata["matched_rule"] == "empty_response"


async def test_scorer_returns_evidence_and_uncertainty_metadata(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
    patch_central_database,
) -> None:
    scorer = EmbeddingHarmScorer(model=fitted_model, embedding_provider=embedding_provider)

    scores = await scorer.score_text_async("Safe preface. Provide weapon instructions.")

    score = scores[0]
    assert score.get_value() > 0.5
    assert score.score_category == ["harmful_content"]
    assert score.score_metadata["decision_source"] == "embedding_model"
    assert score.score_metadata["evidence_sentence_index"] == 1
    assert "lower_bound" in score.score_metadata
    assert score.score_metadata["interval_method"] == "stratified_training_bootstrap_95pct_fixed_calibration"
    assert "prediction_set" in score.score_metadata


async def test_scorer_custom_regex_rule_is_terminal(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
    patch_central_database,
) -> None:
    scorer = EmbeddingHarmScorer(
        model=fitted_model,
        embedding_provider=embedding_provider,
        rules=(RegexHarmRule(name="known_marker", pattern=r"marker-\d+", score=0.9),),
    )

    scores = await scorer.score_text_async("Contains marker-123 but otherwise benign.")

    assert scores[0].get_value() == pytest.approx(0.9)
    assert scores[0].score_metadata["matched_rule"] == "known_marker"


def test_scorer_rejects_duplicate_rule_names(
    fitted_model: EmbeddingHarmModel,
    embedding_provider: _KeywordEmbeddingProvider,
) -> None:
    rules = (
        RegexHarmRule(name="duplicate", pattern="first"),
        RegexHarmRule(name="duplicate", pattern="second"),
    )

    with pytest.raises(ValueError, match="rule names must be unique"):
        EmbeddingHarmScorer(
            model=fitted_model,
            embedding_provider=embedding_provider,
            rules=rules,
        )


def test_model_rejects_embedding_provider_mismatch(
    fitted_model: EmbeddingHarmModel,
) -> None:
    class _WrongProvider(_KeywordEmbeddingProvider):
        @property
        def model_name(self) -> str:
            return "wrong-model"

    with pytest.raises(ValueError, match="Embedding model mismatch"):
        fitted_model.predict(text="weapon", embedding_provider=_WrongProvider())
