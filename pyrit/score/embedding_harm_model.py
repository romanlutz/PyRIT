# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from collections.abc import Sequence


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def split_harm_sentences(text: str) -> list[str]:
    """
    Split response text into sentence-like units without requiring an NLP model.

    Args:
        text (str): Response text to split.

    Returns:
        list[str]: Non-empty sentence and line segments.
    """
    stripped = text.strip()
    if not stripped:
        return []
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", stripped) if segment.strip()]


class TextEmbeddingProvider(Protocol):
    """Embedding interface used by the harm model for training and inference."""

    @property
    def model_name(self) -> str:
        """The stable embedding model identifier."""

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """Return one embedding row per input text."""


class BgeEmbeddingProvider:
    """Generate normalized BGE embeddings through Hugging Face Transformers."""

    DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
    ) -> None:
        """
        Initialize a lazily loaded BGE embedding provider.

        Args:
            model_name (str): Hugging Face model identifier or local path.
            device (str | None): Torch device. Auto-selects CUDA when available.
            batch_size (int): Number of texts encoded per batch.
            max_length (int): Maximum token count per text.

        Raises:
            ValueError: If a numeric setting is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_length <= 0:
            raise ValueError("max_length must be positive.")
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._max_length = max_length
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        """The Hugging Face model identifier."""
        return self._model_name

    def embed(self, texts: Sequence[str]) -> FloatArray:
        """
        Encode texts using attention-mask-aware mean pooling and L2 normalization.

        Args:
            texts (Sequence[str]): Texts to encode.

        Returns:
            FloatArray: Matrix with one normalized embedding per text.

        Raises:
            RuntimeError: If the optional Hugging Face dependencies are unavailable.
        """
        if not texts:
            return np.empty((0, 0), dtype=np.float64)
        with self._lock:
            self._ensure_loaded()
            batches = [
                self._embed_batch(texts=texts[start : start + self._batch_size])
                for start in range(0, len(texts), self._batch_size)
            ]
        return np.concatenate(batches, axis=0)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            torch = importlib.import_module("torch")
            from transformers.models.auto.modeling_auto import AutoModel
            from transformers.models.auto.tokenization_auto import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "BgeEmbeddingProvider requires the 'huggingface' optional dependency. "
                "Install PyRIT with the huggingface extra."
            ) from exc

        self._torch = torch
        self._device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name, trust_remote_code=False)
        self._model = AutoModel.from_pretrained(self._model_name, trust_remote_code=False)
        self._model.to(self._device)
        self._model.eval()

    def _embed_batch(self, *, texts: Sequence[str]) -> FloatArray:
        encoded = self._tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self._max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(self._device) for name, value in encoded.items()}
        with self._torch.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
            pooled = hidden[:, 0]
            pooled = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return cast("FloatArray", pooled.detach().cpu().numpy().astype(np.float64))


@dataclass(frozen=True, kw_only=True)
class HarmScorerTrainingExample:
    """One response-level label with optional labels for its sentence-like units."""

    response: str
    is_harmful: bool
    sentence_labels: Sequence[bool] | None = None

    def __post_init__(self) -> None:
        """Snapshot sentence labels so later caller mutation cannot alter training."""
        if self.sentence_labels is not None:
            object.__setattr__(self, "sentence_labels", tuple(self.sentence_labels))


@dataclass(frozen=True, kw_only=True)
class HarmPrediction:
    """Harm probability, evidence, and uncertainty produced by a fitted model."""

    probability: float
    lower_bound: float
    upper_bound: float
    prediction_set: tuple[str, ...]
    sentence_probabilities: tuple[float, ...]
    sentences: tuple[str, ...]
    evidence_index: int
    ood_distance: float | None
    is_out_of_distribution: bool


@dataclass(frozen=True, kw_only=True)
class _ProbabilityModel:
    coefficients: FloatArray
    intercept: float
    feature_mean: FloatArray
    feature_scale: FloatArray
    calibration_slope: float = 1.0
    calibration_intercept: float = 0.0

    def logits(self, features: FloatArray) -> FloatArray:
        """
        Calculate uncalibrated logits.

        Returns:
            FloatArray: One logit per feature row.
        """
        normalized = (features - self.feature_mean) / self.feature_scale
        return normalized @ self.coefficients + self.intercept

    def probabilities(self, features: FloatArray) -> FloatArray:
        """
        Calculate calibrated harmful-class probabilities.

        Returns:
            FloatArray: One harmful-class probability per feature row.
        """
        calibrated_logits = self.calibration_slope * self.logits(features) + self.calibration_intercept
        return _sigmoid(calibrated_logits)

    def with_calibration(self, *, slope: float, intercept: float) -> _ProbabilityModel:
        """Return a copy with fitted Platt calibration parameters."""
        return _ProbabilityModel(
            coefficients=self.coefficients,
            intercept=self.intercept,
            feature_mean=self.feature_mean,
            feature_scale=self.feature_scale,
            calibration_slope=slope,
            calibration_intercept=intercept,
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this model to JSON-compatible primitives.

        Returns:
            dict[str, Any]: Serialized probability model.
        """
        return {
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "calibration_slope": self.calibration_slope,
            "calibration_intercept": self.calibration_intercept,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _ProbabilityModel:
        """
        Deserialize and validate a probability model.

        Returns:
            _ProbabilityModel: Validated probability model.

        Raises:
            ValueError: If model arrays have incompatible shapes or scales.
        """
        coefficients = np.asarray(data["coefficients"], dtype=np.float64)
        feature_mean = np.asarray(data["feature_mean"], dtype=np.float64)
        feature_scale = np.asarray(data["feature_scale"], dtype=np.float64)
        if (
            coefficients.ndim != 1
            or feature_mean.shape != coefficients.shape
            or feature_scale.shape != coefficients.shape
        ):
            raise ValueError("Probability model arrays must be one-dimensional and have equal lengths.")
        if np.any(feature_scale <= 0):
            raise ValueError("Probability model feature scales must be positive.")
        return cls(
            coefficients=coefficients,
            intercept=float(data["intercept"]),
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            calibration_slope=float(data.get("calibration_slope", 1.0)),
            calibration_intercept=float(data.get("calibration_intercept", 0.0)),
        )


@dataclass(frozen=True, kw_only=True)
class _EmbeddedExamples:
    sentences: tuple[tuple[str, ...], ...]
    sentence_embeddings: tuple[FloatArray, ...]
    document_embeddings: FloatArray


@dataclass(frozen=True, kw_only=True)
class EmbeddingHarmModel:
    """Serializable two-stage logistic harm model over sentence and response embeddings."""

    category: str
    embedding_model: str
    sentence_model: _ProbabilityModel
    response_model: _ProbabilityModel
    bootstrap_models: tuple[_ProbabilityModel, ...]
    ood_reference_embeddings: FloatArray
    ood_k: int
    ood_threshold: float | None
    conformal_quantile: float
    sentence_threshold: float = 0.5
    artifact_version: int = 1

    AGGREGATE_FEATURE_COUNT = 5

    @classmethod
    def fit(
        cls,
        *,
        training_examples: Sequence[HarmScorerTrainingExample],
        calibration_examples: Sequence[HarmScorerTrainingExample],
        embedding_provider: TextEmbeddingProvider,
        category: str = "harmful_content",
        l2_regularization: float = 1.0,
        bootstrap_samples: int = 50,
        conformal_alpha: float = 0.1,
        ood_k: int = 5,
        ood_quantile: float = 0.99,
        max_ood_reference_samples: int = 5_000,
        random_seed: int = 42,
    ) -> EmbeddingHarmModel:
        """
        Fit sentence and response classifiers using disjoint training and calibration sets.

        Harmful responses need explicit sentence labels for sentence-classifier training.
        Unlabeled safe responses are valid because every sentence in a safe response is safe.

        Args:
            training_examples (Sequence[HarmScorerTrainingExample]): Data used to fit model weights.
            calibration_examples (Sequence[HarmScorerTrainingExample]): Held-out probability and conformal data.
            embedding_provider (TextEmbeddingProvider): Frozen text embedding implementation.
            category (str): Harm category represented by the model.
            l2_regularization (float): Logistic-regression L2 penalty.
            bootstrap_samples (int): Number of stratified response-model bootstrap refits.
            conformal_alpha (float): Desired conformal error rate under exchangeability.
            ood_k (int): Neighbor rank used for embedding-space OOD distance.
            ood_quantile (float): Training-distance quantile used as the OOD threshold.
            max_ood_reference_samples (int): Maximum sentence embeddings retained for OOD checks.
            random_seed (int): Reproducible bootstrap and sampling seed.

        Returns:
            EmbeddingHarmModel: Fitted, serializable model.

        Raises:
            ValueError: If data or hyperparameters cannot support binary classification.
        """
        cls._validate_fit_inputs(
            training_examples=training_examples,
            calibration_examples=calibration_examples,
            l2_regularization=l2_regularization,
            bootstrap_samples=bootstrap_samples,
            conformal_alpha=conformal_alpha,
            ood_k=ood_k,
            ood_quantile=ood_quantile,
            max_ood_reference_samples=max_ood_reference_samples,
        )
        train_embedded = _embed_examples(examples=training_examples, provider=embedding_provider)
        calibration_embedded = _embed_examples(examples=calibration_examples, provider=embedding_provider)

        sentence_features, sentence_labels = _labeled_sentence_data(examples=training_examples, embedded=train_embedded)
        sentence_model = _fit_probability_model(
            features=sentence_features,
            labels=sentence_labels,
            l2_regularization=l2_regularization,
            balance_classes=True,
        )
        sentence_model = _calibrate_sentence_model(
            model=sentence_model,
            examples=calibration_examples,
            embedded=calibration_embedded,
        )

        train_response_features = _response_features(embedded=train_embedded, sentence_model=sentence_model)
        calibration_response_features = _response_features(embedded=calibration_embedded, sentence_model=sentence_model)
        train_response_labels = np.asarray([example.is_harmful for example in training_examples], dtype=np.bool_)
        calibration_response_labels = np.asarray(
            [example.is_harmful for example in calibration_examples], dtype=np.bool_
        )
        response_model = _fit_probability_model(
            features=train_response_features,
            labels=train_response_labels,
            l2_regularization=l2_regularization,
            balance_classes=True,
        )
        response_model = _calibrate_model(
            model=response_model,
            features=calibration_response_features,
            labels=calibration_response_labels,
        )

        rng = np.random.default_rng(random_seed)
        bootstrap_models = _fit_bootstrap_models(
            features=train_response_features,
            labels=train_response_labels,
            calibration_features=calibration_response_features,
            calibration_labels=calibration_response_labels,
            l2_regularization=l2_regularization,
            sample_count=bootstrap_samples,
            rng=rng,
        )
        references = _sample_ood_references(
            embeddings=sentence_features,
            labels=sentence_labels,
            maximum=max_ood_reference_samples,
            rng=rng,
        )
        effective_k, threshold = _fit_ood_threshold(
            embeddings=references,
            requested_k=ood_k,
            quantile=ood_quantile,
        )
        calibration_probabilities = response_model.probabilities(calibration_response_features)
        conformal_quantile = _conformal_quantile(
            probabilities=calibration_probabilities,
            labels=calibration_response_labels,
            alpha=conformal_alpha,
        )
        return cls(
            category=category,
            embedding_model=embedding_provider.model_name,
            sentence_model=sentence_model,
            response_model=response_model,
            bootstrap_models=bootstrap_models,
            ood_reference_embeddings=references,
            ood_k=effective_k,
            ood_threshold=threshold,
            conformal_quantile=conformal_quantile,
        )

    def predict(self, *, text: str, embedding_provider: TextEmbeddingProvider) -> HarmPrediction:
        """
        Predict response harm and attach sentence evidence and uncertainty.

        Args:
            text (str): Non-empty response text.
            embedding_provider (TextEmbeddingProvider): Provider matching the training model.

        Returns:
            HarmPrediction: Calibrated response prediction.

        Raises:
            ValueError: If the text is empty or the embedding model does not match.
        """
        if embedding_provider.model_name != self.embedding_model:
            raise ValueError(
                f"Embedding model mismatch: artifact uses {self.embedding_model!r}, "
                f"provider uses {embedding_provider.model_name!r}."
            )
        sentences = split_harm_sentences(text)
        if not sentences:
            raise ValueError("EmbeddingHarmModel.predict requires non-empty text.")
        embeddings = _normalize_rows(embedding_provider.embed([*sentences, text]))
        sentence_embeddings = embeddings[:-1]
        document_embedding = embeddings[-1:]
        sentence_probabilities = self.sentence_model.probabilities(sentence_embeddings)
        aggregate = _aggregate_sentence_probabilities(sentence_probabilities)
        features = np.concatenate([document_embedding, aggregate.reshape(1, -1)], axis=1)
        probability = float(self.response_model.probabilities(features)[0])
        bootstrap_probabilities = np.asarray(
            [model.probabilities(features)[0] for model in self.bootstrap_models],
            dtype=np.float64,
        )
        lower, upper = _prediction_interval(probability=probability, bootstrap=bootstrap_probabilities)
        evidence_index = int(np.argmax(sentence_probabilities))
        ood_distance = self._ood_distance(sentence_embeddings[evidence_index])
        return HarmPrediction(
            probability=probability,
            lower_bound=lower,
            upper_bound=upper,
            prediction_set=self._prediction_set(probability),
            sentence_probabilities=tuple(float(value) for value in sentence_probabilities),
            sentences=tuple(sentences),
            evidence_index=evidence_index,
            ood_distance=ood_distance,
            is_out_of_distribution=bool(
                ood_distance is not None and self.ood_threshold is not None and ood_distance > self.ood_threshold
            ),
        )

    def save(self, path: str | Path) -> None:
        """
        Save the model as a non-executable JSON artifact.

        Args:
            path (str | Path): Destination file.
        """
        Path(path).write_text(json.dumps(self.to_dict(), separators=(",", ":")), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingHarmModel:
        """
        Load a model from a JSON artifact.

        Args:
            path (str | Path): Artifact file.

        Returns:
            EmbeddingHarmModel: Validated model.

        Raises:
            ValueError: If the artifact root is not a JSON object.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Embedding harm model artifact must contain a JSON object.")
        return cls.from_dict(data)

    def fingerprint(self) -> str:
        """Return a stable short fingerprint for scorer identity."""
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the complete model to JSON-compatible primitives.

        Returns:
            dict[str, Any]: Serialized model artifact.
        """
        return {
            "artifact_version": self.artifact_version,
            "category": self.category,
            "embedding_model": self.embedding_model,
            "sentence_model": self.sentence_model.to_dict(),
            "response_model": self.response_model.to_dict(),
            "bootstrap_models": [model.to_dict() for model in self.bootstrap_models],
            "ood_reference_embeddings": self.ood_reference_embeddings.tolist(),
            "ood_k": self.ood_k,
            "ood_threshold": self.ood_threshold,
            "conformal_quantile": self.conformal_quantile,
            "sentence_threshold": self.sentence_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingHarmModel:
        """
        Deserialize and validate a model artifact.

        Returns:
            EmbeddingHarmModel: Validated model artifact.

        Raises:
            ValueError: If the artifact version, dimensions, or reference data are invalid.
        """
        version = int(data.get("artifact_version", 0))
        if version != 1:
            raise ValueError(f"Unsupported embedding harm model artifact version: {version}.")
        references = np.asarray(data["ood_reference_embeddings"], dtype=np.float64)
        if references.ndim != 2 or not references.shape[0]:
            raise ValueError("OOD reference embeddings must be a non-empty matrix.")
        sentence_model = _ProbabilityModel.from_dict(data["sentence_model"])
        response_model = _ProbabilityModel.from_dict(data["response_model"])
        if references.shape[1] != sentence_model.coefficients.size:
            raise ValueError("OOD reference dimension must match the sentence model.")
        expected_response_features = references.shape[1] + cls.AGGREGATE_FEATURE_COUNT
        if response_model.coefficients.size != expected_response_features:
            raise ValueError("Response model dimension does not match document and aggregate features.")
        return cls(
            artifact_version=version,
            category=str(data["category"]),
            embedding_model=str(data["embedding_model"]),
            sentence_model=sentence_model,
            response_model=response_model,
            bootstrap_models=tuple(_ProbabilityModel.from_dict(item) for item in data["bootstrap_models"]),
            ood_reference_embeddings=_validate_normalized_rows(references),
            ood_k=int(data["ood_k"]),
            ood_threshold=None if data["ood_threshold"] is None else float(data["ood_threshold"]),
            conformal_quantile=float(data["conformal_quantile"]),
            sentence_threshold=float(data.get("sentence_threshold", 0.5)),
        )

    def _prediction_set(self, probability: float) -> tuple[str, ...]:
        labels: list[str] = []
        if probability <= self.conformal_quantile:
            labels.append("safe")
        if 1.0 - probability <= self.conformal_quantile:
            labels.append("harmful")
        return tuple(labels) or ("uncertain",)

    def _ood_distance(self, embedding: FloatArray) -> float | None:
        if self.ood_threshold is None:
            return None
        distances = 1.0 - self.ood_reference_embeddings @ embedding
        neighbor_index = min(self.ood_k - 1, distances.size - 1)
        return max(0.0, float(np.partition(distances, neighbor_index)[neighbor_index]))

    @staticmethod
    def _validate_fit_inputs(
        *,
        training_examples: Sequence[HarmScorerTrainingExample],
        calibration_examples: Sequence[HarmScorerTrainingExample],
        l2_regularization: float,
        bootstrap_samples: int,
        conformal_alpha: float,
        ood_k: int,
        ood_quantile: float,
        max_ood_reference_samples: int,
    ) -> None:
        if not training_examples or not calibration_examples:
            raise ValueError("Training and calibration examples must both be non-empty.")
        if l2_regularization < 0:
            raise ValueError("l2_regularization must be non-negative.")
        if bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be non-negative.")
        if not 0 < conformal_alpha < 1:
            raise ValueError("conformal_alpha must be between 0 and 1.")
        if ood_k <= 0 or max_ood_reference_samples < 2:
            raise ValueError("ood_k must be positive and max_ood_reference_samples must be at least 2.")
        if not 0 < ood_quantile < 1:
            raise ValueError("ood_quantile must be between 0 and 1.")
        _require_binary_labels(training_examples, name="training")
        _require_binary_labels(calibration_examples, name="calibration")


def _embed_examples(
    *,
    examples: Sequence[HarmScorerTrainingExample],
    provider: TextEmbeddingProvider,
) -> _EmbeddedExamples:
    sentence_groups = tuple(tuple(split_harm_sentences(example.response)) for example in examples)
    if any(not sentences for sentences in sentence_groups):
        raise ValueError("Training and calibration responses must be non-empty.")
    flattened = [sentence for sentences in sentence_groups for sentence in sentences]
    sentence_matrix = _normalize_rows(provider.embed(flattened))
    document_matrix = _normalize_rows(provider.embed([example.response for example in examples]))
    if sentence_matrix.shape[1] != document_matrix.shape[1]:
        raise ValueError("Sentence and response embedding dimensions must match.")
    grouped: list[FloatArray] = []
    offset = 0
    for sentences in sentence_groups:
        grouped.append(sentence_matrix[offset : offset + len(sentences)])
        offset += len(sentences)
    return _EmbeddedExamples(
        sentences=sentence_groups,
        sentence_embeddings=tuple(grouped),
        document_embeddings=document_matrix,
    )


def _labeled_sentence_data(
    *,
    examples: Sequence[HarmScorerTrainingExample],
    embedded: _EmbeddedExamples,
) -> tuple[FloatArray, BoolArray]:
    feature_rows: list[FloatArray] = []
    labels: list[bool] = []
    for example, sentences, embeddings in zip(examples, embedded.sentences, embedded.sentence_embeddings, strict=True):
        sentence_labels = example.sentence_labels
        if sentence_labels is None:
            if example.is_harmful:
                continue
            sentence_labels = (False,) * len(sentences)
        if len(sentence_labels) != len(sentences):
            raise ValueError("sentence_labels must align with split_harm_sentences(response).")
        feature_rows.extend(embeddings)
        labels.extend(sentence_labels)
    if not feature_rows:
        raise ValueError("No sentence labels are available for sentence-classifier training.")
    label_array = np.asarray(labels, dtype=np.bool_)
    _require_two_classes(label_array, name="sentence")
    return np.asarray(feature_rows, dtype=np.float64), label_array


def _calibrate_sentence_model(
    *,
    model: _ProbabilityModel,
    examples: Sequence[HarmScorerTrainingExample],
    embedded: _EmbeddedExamples,
) -> _ProbabilityModel:
    features, labels = _labeled_sentence_data(examples=examples, embedded=embedded)
    return _calibrate_model(model=model, features=features, labels=labels)


def _response_features(*, embedded: _EmbeddedExamples, sentence_model: _ProbabilityModel) -> FloatArray:
    rows = []
    for document, sentence_embeddings in zip(embedded.document_embeddings, embedded.sentence_embeddings, strict=True):
        sentence_probabilities = sentence_model.probabilities(sentence_embeddings)
        rows.append(np.concatenate([document, _aggregate_sentence_probabilities(sentence_probabilities)]))
    return np.asarray(rows, dtype=np.float64)


def _aggregate_sentence_probabilities(probabilities: FloatArray) -> FloatArray:
    ordered = np.sort(probabilities)
    top_two = ordered[-2:] if ordered.size >= 2 else ordered
    return np.asarray(
        [
            float(ordered[-1]),
            float(np.mean(probabilities)),
            float(np.mean(top_two)),
            float(np.mean(probabilities >= 0.5)),
            math.log1p(probabilities.size),
        ],
        dtype=np.float64,
    )


def _fit_probability_model(
    *,
    features: FloatArray,
    labels: BoolArray,
    l2_regularization: float,
    balance_classes: bool,
) -> _ProbabilityModel:
    _require_two_classes(labels, name="classifier")
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (features - feature_mean) / feature_scale
    coefficients, intercept = _fit_logistic_parameters(
        features=normalized,
        labels=labels,
        l2_regularization=l2_regularization,
        balance_classes=balance_classes,
    )
    return _ProbabilityModel(
        coefficients=coefficients,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
    )


def _fit_logistic_parameters(
    *,
    features: FloatArray,
    labels: BoolArray,
    l2_regularization: float,
    balance_classes: bool,
) -> tuple[FloatArray, float]:
    from scipy.optimize import minimize

    targets = labels.astype(np.float64)
    if balance_classes:
        positive_weight = labels.size / (2 * np.count_nonzero(labels))
        negative_weight = labels.size / (2 * np.count_nonzero(~labels))
        weights = np.where(labels, positive_weight, negative_weight)
    else:
        weights = np.ones(labels.size, dtype=np.float64)

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        coefficients = parameters[:-1]
        intercept = parameters[-1]
        logits = features @ coefficients + intercept
        loss = np.sum(weights * (np.logaddexp(0.0, logits) - targets * logits))
        loss += 0.5 * l2_regularization * float(coefficients @ coefficients)
        errors = weights * (_sigmoid(logits) - targets)
        gradient = np.concatenate([features.T @ errors + l2_regularization * coefficients, [np.sum(errors)]])
        return float(loss), gradient

    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    result = minimize(objective, initial, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic regression failed to converge: {result.message}")
    parameters = np.asarray(result.x, dtype=np.float64)
    return parameters[:-1], float(parameters[-1])


def _calibrate_model(
    *,
    model: _ProbabilityModel,
    features: FloatArray,
    labels: BoolArray,
) -> _ProbabilityModel:
    logits = model.logits(features).reshape(-1, 1)
    calibrator = _fit_probability_model(
        features=logits,
        labels=labels,
        l2_regularization=1e-6,
        balance_classes=False,
    )
    scale = float(calibrator.feature_scale[0])
    slope = float(calibrator.coefficients[0] / scale)
    intercept = float(calibrator.intercept - calibrator.coefficients[0] * calibrator.feature_mean[0] / scale)
    return model.with_calibration(slope=slope, intercept=intercept)


def _fit_bootstrap_models(
    *,
    features: FloatArray,
    labels: BoolArray,
    calibration_features: FloatArray,
    calibration_labels: BoolArray,
    l2_regularization: float,
    sample_count: int,
    rng: np.random.Generator,
) -> tuple[_ProbabilityModel, ...]:
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    models = []
    for _ in range(sample_count):
        indices = np.concatenate(
            [
                rng.choice(positive, size=positive.size, replace=True),
                rng.choice(negative, size=negative.size, replace=True),
            ]
        )
        rng.shuffle(indices)
        model = _fit_probability_model(
            features=features[indices],
            labels=labels[indices],
            l2_regularization=l2_regularization,
            balance_classes=True,
        )
        models.append(
            _calibrate_model(
                model=model,
                features=calibration_features,
                labels=calibration_labels,
            )
        )
    return tuple(models)


def _sample_ood_references(
    *,
    embeddings: FloatArray,
    labels: BoolArray,
    maximum: int,
    rng: np.random.Generator,
) -> FloatArray:
    if embeddings.shape[0] <= maximum:
        return embeddings
    positive = np.flatnonzero(labels)
    negative = np.flatnonzero(~labels)
    positive_count = min(positive.size, max(1, round(maximum * positive.size / labels.size)))
    negative_count = maximum - positive_count
    if negative_count > negative.size:
        negative_count = negative.size
        positive_count = maximum - negative_count
    indices = np.concatenate(
        [
            rng.choice(positive, size=positive_count, replace=False),
            rng.choice(negative, size=negative_count, replace=False),
        ]
    )
    return embeddings[indices]


def _fit_ood_threshold(
    *,
    embeddings: FloatArray,
    requested_k: int,
    quantile: float,
) -> tuple[int, float | None]:
    if embeddings.shape[0] < 2:
        return 1, None
    k = int(min(requested_k, embeddings.shape[0] - 1))
    kth_distances = np.empty(embeddings.shape[0], dtype=np.float64)
    batch_size = 512
    for start in range(0, embeddings.shape[0], batch_size):
        stop = min(start + batch_size, embeddings.shape[0])
        distances = 1.0 - embeddings[start:stop] @ embeddings.T
        row_indices = np.arange(stop - start)
        distances[row_indices, np.arange(start, stop)] = np.inf
        kth_distances[start:stop] = np.partition(distances, k - 1, axis=1)[:, k - 1]
    return k, float(np.quantile(kth_distances, quantile))


def _conformal_quantile(*, probabilities: FloatArray, labels: BoolArray, alpha: float) -> float:
    nonconformity = np.where(labels, 1.0 - probabilities, probabilities)
    rank = min(math.ceil((nonconformity.size + 1) * (1.0 - alpha)), nonconformity.size)
    return float(np.sort(nonconformity)[rank - 1])


def _prediction_interval(*, probability: float, bootstrap: FloatArray) -> tuple[float, float]:
    if not bootstrap.size:
        return probability, probability
    return float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))


def _normalize_rows(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("Embedding providers must return a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Embedding providers must return finite values.")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("Embedding providers must not return zero vectors.")
    return array / norms


def _validate_normalized_rows(values: FloatArray) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise ValueError("OOD reference embeddings must be a non-empty two-dimensional matrix.")
    if not np.all(np.isfinite(array)):
        raise ValueError("OOD reference embeddings must contain finite values.")
    if not np.allclose(np.linalg.norm(array, axis=1), 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("OOD reference embeddings must be L2-normalized.")
    return array


def _sigmoid(values: FloatArray) -> FloatArray:
    clipped = np.clip(values, -500.0, 500.0)
    return cast("FloatArray", 1.0 / (1.0 + np.exp(-clipped)))


def _require_binary_labels(examples: Sequence[HarmScorerTrainingExample], *, name: str) -> None:
    labels = np.asarray([example.is_harmful for example in examples], dtype=np.bool_)
    _require_two_classes(labels, name=name)


def _require_two_classes(labels: BoolArray, *, name: str) -> None:
    if labels.ndim != 1 or not np.any(labels) or not np.any(~labels):
        raise ValueError(f"{name} data must contain both safe and harmful labels.")
