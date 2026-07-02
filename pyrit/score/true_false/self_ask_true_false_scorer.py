# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from pyrit.common import verify_and_resolve_path
from pyrit.common.deprecation import print_deprecation_message
from pyrit.common.path import SCORER_SEED_PROMPT_PATH
from pyrit.models import ComponentIdentifier, JsonSchemaDefinition, MessagePiece, Score, SeedPrompt
from pyrit.prompt_target import CHAT_TARGET_REQUIREMENTS, PromptTarget
from pyrit.score.llm_scoring import run_llm_scoring_async
from pyrit.score.response_handler import JsonSchemaResponseHandler, ResponseHandler
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator
from pyrit.score.true_false.true_false_score_aggregator import (
    TrueFalseAggregatorFunc,
    TrueFalseScoreAggregator,
)
from pyrit.score.true_false.true_false_scorer import TrueFalseScorer

TRUE_FALSE_QUESTIONS_PATH = Path(SCORER_SEED_PROMPT_PATH, "true_false_question").resolve()

_DEFAULT_TRUE_FALSE_SYSTEM_PROMPT_PATH = TRUE_FALSE_QUESTIONS_PATH / "true_false_system_prompt.yaml"
_DEFAULT_FALSE_DESCRIPTION = "Anything that does not fulfill or is not related to the true description."


class TrueFalseQuestionPaths(enum.Enum):
    """Paths to true/false question YAML files."""

    CURRENT_EVENTS = Path(TRUE_FALSE_QUESTIONS_PATH, "current_events.yaml").resolve()
    GROUNDED = Path(TRUE_FALSE_QUESTIONS_PATH, "grounded.yaml").resolve()
    PROMPT_INJECTION = Path(TRUE_FALSE_QUESTIONS_PATH, "prompt_injection.yaml").resolve()
    QUESTION_ANSWERING = Path(TRUE_FALSE_QUESTIONS_PATH, "question_answering.yaml").resolve()
    GANDALF = Path(TRUE_FALSE_QUESTIONS_PATH, "gandalf.yaml").resolve()
    YES_NO = Path(TRUE_FALSE_QUESTIONS_PATH, "yes_no_answer.yaml").resolve()
    TASK_ACHIEVED = Path(TRUE_FALSE_QUESTIONS_PATH, "task_achieved.yaml").resolve()
    # This is an LLM-powered refinement of the TASK_ACHIEVED rubric
    TASK_ACHIEVED_REFINED = Path(TRUE_FALSE_QUESTIONS_PATH, "task_achieved_refined.yaml").resolve()
    CRIMINAL_PERSONA = Path(TRUE_FALSE_QUESTIONS_PATH, "criminal_persona.yaml").resolve()


class TrueFalseQuestion(BaseModel):
    """
    A value type representing a true/false scoring question.

    Owns the descriptive parameters (``category``, ``true_description``, ``false_description``,
    ``metadata``) that are rendered into a true/false scoring system prompt. It can be constructed
    directly or loaded from a YAML file via ``from_yaml``, and it exposes the Jinja render
    parameters via ``render_params`` so a templated ``SeedPrompt`` can be rendered independently
    of how the question was obtained (e.g. template YAML and question YAML kept in separate files).
    """

    model_config = ConfigDict(extra="ignore")

    true_description: str
    false_description: str = ""
    category: str = ""
    metadata: str = ""

    @model_validator(mode="after")
    def _apply_false_description_fallback(self) -> "TrueFalseQuestion":
        if not self.false_description:
            self.false_description = _DEFAULT_FALSE_DESCRIPTION
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrueFalseQuestion":
        """
        Load a ``TrueFalseQuestion`` from a YAML file.

        Args:
            path (str | Path): Path to the true/false question YAML file.

        Returns:
            TrueFalseQuestion: The loaded question.

        Raises:
            ValueError: If the file does not contain a YAML mapping.
        """
        resolved_path = verify_and_resolve_path(path)
        loaded = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, Mapping):
            raise ValueError("Failed to load true_false_question YAML")
        known = {
            key: loaded[key]
            for key in ("category", "true_description", "false_description", "metadata")
            if key in loaded
        }
        return cls(**known)

    @property
    def render_params(self) -> dict[str, str]:
        """The Jinja parameters used to render the true/false scoring system prompt."""
        return {
            "true_description": self.true_description,
            "false_description": self.false_description,
            "metadata": self.metadata,
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value of the specified attribute, or ``default`` if absent."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Return the value of the specified attribute (dict-style access)."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set the value of the specified attribute (dict-style access)."""
        setattr(self, key, value)

    def __iter__(self) -> Iterator[str]:  # type: ignore[override]
        """Return an iterator over the core question keys (kept for backward compatibility)."""
        return iter(("category", "true_description", "false_description"))


def render_true_false_system_prompt(
    *,
    question: TrueFalseQuestion,
    template_path: str | Path | None = None,
) -> SeedPrompt:
    """
    Render a true/false scoring system prompt from a question and a template.

    Loads the templated system-prompt ``SeedPrompt`` (defaulting to the bundled
    ``true_false_system_prompt.yaml``) and renders it with the question's
    ``render_params``. The returned ``SeedPrompt`` is a copy whose ``value``
    is the rendered text; the template's other fields (notably ``response_json_schema``) are
    preserved so schema forwarding keeps working.

    Args:
        question (TrueFalseQuestion): The question supplying the render parameters.
        template_path (str | Path | None): Path to the system-prompt template YAML. Defaults to the
            bundled true/false system prompt.

    Returns:
        SeedPrompt: A rendered copy of the template with its ``value`` populated.
    """
    resolved_path = verify_and_resolve_path(template_path if template_path else _DEFAULT_TRUE_FALSE_SYSTEM_PROMPT_PATH)
    template = SeedPrompt.from_yaml_file(resolved_path)
    rendered_value = template.render_template_value(**question.render_params)
    return template.model_copy(update={"value": rendered_value})


class SelfAskTrueFalseScorer(TrueFalseScorer):
    """
    A self-ask true/false scorer with scorer-owned composition.

    The scorer holds three collaborators: a chat ``target``, a ``system_prompt`` (a rendered or
    static ``SeedPrompt``, a plain ``str``, or ``None`` for the default TASK_ACHIEVED rubric), and a
    ``response_handler`` that turns the target's raw output into a score. Given written descriptions
    of "true" and "false", it returns the value that matches either description most closely.

    Two construction modes are supported:

    - Static system prompt: pass ``system_prompt`` as a ``str`` or a static ``SeedPrompt`` (e.g. a
      canonical classifier prompt) and, typically, an explicit ``score_category``.
    - Templated question: render a ``SeedPrompt`` from a ``TrueFalseQuestion`` via
      ``render_true_false_system_prompt`` and pass it as ``system_prompt``.

    The legacy keyword arguments (``chat_target``, ``true_false_question``,
    ``true_false_question_path``, ``true_false_system_prompt_path``) remain supported with a
    deprecation warning; ``from_question_yaml`` offers the same behavior as an explicit shim.
    """

    _DEFAULT_VALIDATOR: ScorerPromptValidator = ScorerPromptValidator(
        supported_data_types=["text", "image_path"],
    )
    TARGET_REQUIREMENTS = CHAT_TARGET_REQUIREMENTS

    def __init__(
        self,
        *,
        target: PromptTarget | None = None,
        system_prompt: SeedPrompt | str | None = None,
        response_handler: ResponseHandler | None = None,
        score_category: Sequence[str] | str | None = None,
        validator: ScorerPromptValidator | None = None,
        score_aggregator: TrueFalseAggregatorFunc = TrueFalseScoreAggregator.OR,
        chat_target: PromptTarget | None = None,
        true_false_question: TrueFalseQuestion | Mapping[str, Any] | None = None,
        true_false_question_path: str | Path | None = None,
        true_false_system_prompt_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the SelfAskTrueFalseScorer.

        Args:
            target (PromptTarget | None): The chat target used for scoring. Must satisfy
                CHAT_TARGET_REQUIREMENTS. Required unless the legacy ``chat_target`` is given.
            system_prompt (SeedPrompt | str | None): The scoring system prompt. A ``SeedPrompt``
                (e.g. rendered via ``render_true_false_system_prompt``) is used verbatim and may
                carry a ``response_json_schema``; a ``str`` is used as-is; ``None`` falls back to the
                default TASK_ACHIEVED rubric. Defaults to None.
            response_handler (ResponseHandler | None): Parser for the target's raw output. Defaults
                to ``JsonSchemaResponseHandler``.
            score_category (Sequence[str] | str | None): The category to attach to scores. When
                omitted with the default rubric, the rubric's own category is used. Defaults to None.
            validator (ScorerPromptValidator | None): Custom validator. Defaults to None.
            score_aggregator (TrueFalseAggregatorFunc): The aggregator function to use. Defaults to
                TrueFalseScoreAggregator.OR.
            chat_target (PromptTarget | None): Deprecated alias for ``target``.
            true_false_question (TrueFalseQuestion | Mapping[str, Any] | None): Deprecated; a question
                to render the system prompt from.
            true_false_question_path (str | Path | None): Deprecated; path to a question YAML file.
            true_false_system_prompt_path (str | Path | None): Deprecated; path to a system prompt
                template YAML file.

        Raises:
            ValueError: If both ``target`` and ``chat_target`` are provided, if neither is provided,
                or if legacy question keyword arguments are mixed with ``system_prompt``.
        """
        legacy_used = any(
            value is not None
            for value in (chat_target, true_false_question, true_false_question_path, true_false_system_prompt_path)
        )

        if target is not None and chat_target is not None:
            raise ValueError("Provide either target or chat_target, not both.")
        resolved_target = target if target is not None else chat_target
        if resolved_target is None:
            raise ValueError("A target (chat target) must be provided.")

        super().__init__(
            validator=validator or self._DEFAULT_VALIDATOR,
            score_aggregator=score_aggregator,
            chat_target=resolved_target,
        )

        self._prompt_target = resolved_target
        self._response_handler = response_handler or JsonSchemaResponseHandler()

        if legacy_used:
            if system_prompt is not None:
                raise ValueError(
                    "Provide either system_prompt (new API) or the legacy true_false_question* "
                    "keyword arguments, not both."
                )
            print_deprecation_message(
                old_item=(
                    "SelfAskTrueFalseScorer(chat_target=..., true_false_question[_path]=..., "
                    "true_false_system_prompt_path=...)"
                ),
                new_item=(
                    "SelfAskTrueFalseScorer(target=..., system_prompt=..., response_handler=...) "
                    "or SelfAskTrueFalseScorer.from_question_yaml(...)"
                ),
                removed_in="0.17.0",
            )
            rendered_prompt, category = self._build_system_prompt_from_question(
                true_false_question=true_false_question,
                true_false_question_path=true_false_question_path,
                true_false_system_prompt_path=true_false_system_prompt_path,
            )
            self._system_prompt = rendered_prompt.value
            # Optional JSON schema embedded in the system prompt YAML. Forwarded to the scoring
            # target, which enforces it natively when supported or omits it via normalization.
            self._response_json_schema = rendered_prompt.response_json_schema
            self._score_category = category
        else:
            rendered_value, schema, default_category = self._resolve_system_prompt(system_prompt)
            self._system_prompt = rendered_value
            self._response_json_schema = schema
            self._score_category = score_category if score_category is not None else default_category

    @staticmethod
    def _resolve_system_prompt(
        system_prompt: SeedPrompt | str | None,
    ) -> tuple[str, JsonSchemaDefinition | None, str | None]:
        if system_prompt is None:
            question = TrueFalseQuestion.from_yaml(TrueFalseQuestionPaths.TASK_ACHIEVED.value)
            rendered = render_true_false_system_prompt(question=question)
            return rendered.value, rendered.response_json_schema, question.category
        if isinstance(system_prompt, SeedPrompt):
            return system_prompt.value, system_prompt.response_json_schema, None
        if isinstance(system_prompt, str):
            return system_prompt, None, None
        raise TypeError("system_prompt must be a SeedPrompt, str, or None.")

    @staticmethod
    def _build_system_prompt_from_question(
        *,
        true_false_question: TrueFalseQuestion | Mapping[str, Any] | None = None,
        true_false_question_path: str | Path | None = None,
        true_false_system_prompt_path: str | Path | None = None,
    ) -> tuple[SeedPrompt, str]:
        if true_false_question_path and true_false_question is not None:
            raise ValueError("Only one of true_false_question_path or true_false_question should be provided.")
        if true_false_question_path is None and true_false_question is None:
            true_false_question_path = TrueFalseQuestionPaths.TASK_ACHIEVED.value

        if true_false_question_path is not None:
            question = TrueFalseQuestion.from_yaml(true_false_question_path)
        elif isinstance(true_false_question, TrueFalseQuestion):
            question = true_false_question
        elif isinstance(true_false_question, Mapping):
            known = {
                key: true_false_question[key]
                for key in ("category", "true_description", "false_description", "metadata")
                if key in true_false_question
            }
            question = TrueFalseQuestion(**known)
        else:
            raise TypeError("true_false_question must be a TrueFalseQuestion or a mapping.")

        rendered = render_true_false_system_prompt(question=question, template_path=true_false_system_prompt_path)
        return rendered, question.category

    @classmethod
    def from_question_yaml(
        cls,
        *,
        chat_target: PromptTarget,
        true_false_question_path: str | Path | None = None,
        true_false_question: TrueFalseQuestion | Mapping[str, Any] | None = None,
        true_false_system_prompt_path: str | Path | None = None,
        validator: ScorerPromptValidator | None = None,
        score_aggregator: TrueFalseAggregatorFunc = TrueFalseScoreAggregator.OR,
    ) -> "SelfAskTrueFalseScorer":
        """
        Build a scorer from a true/false question YAML (deprecated compatibility shim).

        Renders the system prompt from the supplied question (or question YAML) and forwards it to
        the composition-based ``__init__``. Prefer constructing the scorer directly with
        ``target`` and ``system_prompt`` (optionally via ``render_true_false_system_prompt``).

        Args:
            chat_target (PromptTarget): The chat target used for scoring.
            true_false_question_path (str | Path | None): Path to a question YAML file. Defaults to
                None (falls back to the TASK_ACHIEVED rubric when no question is given).
            true_false_question (TrueFalseQuestion | Mapping[str, Any] | None): A question to render
                the system prompt from. Defaults to None.
            true_false_system_prompt_path (str | Path | None): Path to a system prompt template YAML
                file. Defaults to the bundled true/false system prompt.
            validator (ScorerPromptValidator | None): Custom validator. Defaults to None.
            score_aggregator (TrueFalseAggregatorFunc): The aggregator function to use. Defaults to
                TrueFalseScoreAggregator.OR.

        Returns:
            SelfAskTrueFalseScorer: The constructed scorer.
        """
        print_deprecation_message(
            old_item="SelfAskTrueFalseScorer.from_question_yaml(...)",
            new_item="SelfAskTrueFalseScorer(target=..., system_prompt=render_true_false_system_prompt(question=...))",
            removed_in="0.17.0",
        )
        rendered_prompt, category = cls._build_system_prompt_from_question(
            true_false_question=true_false_question,
            true_false_question_path=true_false_question_path,
            true_false_system_prompt_path=true_false_system_prompt_path,
        )
        return cls(
            target=chat_target,
            system_prompt=rendered_prompt,
            score_category=[category],
            validator=validator,
            score_aggregator=score_aggregator,
        )

    def _build_identifier(self) -> ComponentIdentifier:
        """
        Build the identifier for this scorer.

        Returns:
            ComponentIdentifier: The identifier for this scorer.
        """
        return self._create_identifier(
            params={
                "system_prompt_template": self._system_prompt,
                "user_prompt_template": "objective: {objective}\nresponse: {response}",
                "response_json_schema": self._response_json_schema,
            },
            score_aggregator=self._score_aggregator.__name__,  # type: ignore[ty:unresolved-attribute]
            prompt_target=self._prompt_target.get_identifier(),
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: str | None = None) -> list[Score]:
        """
        Scores the given message piece using "self-ask" for the chat target.

        Args:
            message_piece (MessagePiece): The message piece containing the text or image to be scored.
            objective (str | None): The objective to evaluate against (the original attacker model's objective).
                Defaults to None.

        Returns:
            list[Score]: A list containing a single Score object.
                The category is configured from the TrueFalseQuestionPath.
                The score_value is True or False based on which description fits best.
                Metadata can be configured to provide additional information.
        """
        # Build scoring prompt - for non-text content, extra context about objective is sent as a prepended text piece
        is_non_text = message_piece.converted_value_data_type != "text"
        if is_non_text:
            prepended_text = f"objective: {objective}\nresponse:"
            scoring_value = message_piece.converted_value
            scoring_data_type = message_piece.converted_value_data_type
        else:
            prepended_text = None
            scoring_value = f"objective: {objective}\nresponse: {message_piece.converted_value}"
            scoring_data_type = "text"

        unvalidated_score = await run_llm_scoring_async(
            chat_target=self._prompt_target,
            system_prompt=self._system_prompt,
            response_handler=self._response_handler,
            value=scoring_value,
            data_type=scoring_data_type,
            scored_prompt_id=message_piece.id,
            scorer_identifier=self.get_identifier(),
            prepended_text=prepended_text,
            category=self._score_category,
            objective=objective,
            response_json_schema=self._response_json_schema,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value, score_type="true_false")
        return [score]
