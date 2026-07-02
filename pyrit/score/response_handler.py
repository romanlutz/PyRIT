# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import json
from abc import abstractmethod
from typing import TYPE_CHECKING

from pyrit.exceptions import InvalidJsonException, remove_markdown_json
from pyrit.models import UnvalidatedScore

if TYPE_CHECKING:
    import uuid
    from collections.abc import Sequence

    from pyrit.models import ComponentIdentifier


class ResponseHandler(abc.ABC):
    """
    Turns the raw text a scoring target returned into an ``UnvalidatedScore``.

    A ResponseHandler owns response parsing and nothing else: given the text produced by a
    scoring LLM, it produces the unvalidated score object the scorer expects. It does not
    perform the LLM round-trip, build the system prompt, or decide how the resulting score
    branches. Different handlers implement different wire formats (e.g. JSON today).
    """

    @abstractmethod
    def parse(
        self,
        *,
        response_text: str,
        scorer_identifier: ComponentIdentifier,
        scored_prompt_id: str | uuid.UUID,
        category: Sequence[str] | str | None = None,
        objective: str | None = None,
    ) -> UnvalidatedScore:
        """
        Parse raw target output into an ``UnvalidatedScore``.

        Args:
            response_text (str): The raw text returned by the scoring target.
            scorer_identifier (ComponentIdentifier): Identifier of the scorer that produced the
                request, stored on the resulting score.
            scored_prompt_id (str | uuid.UUID): The ID of the message piece being scored.
            category (Sequence[str] | str | None): The category of the score. May instead be parsed
                from the response; supplying both is an error. Defaults to None.
            objective (str | None): The objective associated with the score, used for
                contextualizing the result. Defaults to None.

        Returns:
            UnvalidatedScore: The parsed score, whose ``raw_score_value`` still needs to be
                normalized and validated by the caller.
        """
        ...


class JsonSchemaResponseHandler(ResponseHandler):
    """
    Default ResponseHandler that parses JSON scoring responses.

    Reproduces PyRIT's historical scoring-response parsing: strip any markdown code fences,
    ``json.loads`` the text, then read the score value, rationale, optional description,
    category, and metadata from configurable keys.
    """

    def __init__(
        self,
        *,
        score_value_output_key: str = "score_value",
        rationale_output_key: str = "rationale",
        description_output_key: str = "description",
        metadata_output_key: str = "metadata",
        category_output_key: str = "category",
    ) -> None:
        """
        Initialize the handler with the JSON keys to read from the response.

        Args:
            score_value_output_key (str): Key holding the score value. Defaults to "score_value".
            rationale_output_key (str): Key holding the rationale. Defaults to "rationale".
            description_output_key (str): Key holding the description. Defaults to "description".
            metadata_output_key (str): Key holding the metadata. Defaults to "metadata".
            category_output_key (str): Key holding the category. Defaults to "category".
        """
        self._score_value_output_key = score_value_output_key
        self._rationale_output_key = rationale_output_key
        self._description_output_key = description_output_key
        self._metadata_output_key = metadata_output_key
        self._category_output_key = category_output_key

    def parse(
        self,
        *,
        response_text: str,
        scorer_identifier: ComponentIdentifier,
        scored_prompt_id: str | uuid.UUID,
        category: Sequence[str] | str | None = None,
        objective: str | None = None,
    ) -> UnvalidatedScore:
        """
        Parse a JSON scoring response into an ``UnvalidatedScore``.

        Args:
            response_text (str): The raw text returned by the scoring target.
            scorer_identifier (ComponentIdentifier): Identifier of the scorer that produced the
                request, stored on the resulting score.
            scored_prompt_id (str | uuid.UUID): The ID of the message piece being scored.
            category (Sequence[str] | str | None): The category of the score. May instead be parsed
                from the response; supplying both is an error. Defaults to None.
            objective (str | None): The objective associated with the score, used for
                contextualizing the result. Defaults to None.

        Returns:
            UnvalidatedScore: The parsed score, whose ``raw_score_value`` still needs to be
                normalized and validated by the caller.

        Raises:
            ValueError: If a category is present in both the response and the argument, or the
                parsed category is not a string or a list of strings.
            InvalidJsonException: If the response is not valid JSON or is missing a required key.
        """
        response_json = remove_markdown_json(response_text)
        try:
            parsed_response = json.loads(response_json)
            category_response = parsed_response.get(self._category_output_key)

            if category_response and category:
                raise ValueError("Category is present in the response and an argument")

            # Validate and normalize category to a list of strings
            cat_val = category_response if category_response is not None else category
            normalized_category: list[str] | None
            if cat_val is None:
                normalized_category = None
            elif isinstance(cat_val, str):
                normalized_category = [cat_val]
            elif isinstance(cat_val, list):
                if not all(isinstance(x, str) for x in cat_val):
                    raise ValueError("'category' must be a string or a list of strings")
                normalized_category = cat_val  # type: ignore[ty:invalid-assignment]
            else:
                # JSON must yield either a string or a list of strings
                raise ValueError("'category' must be a string or a list of strings")

            # Normalize metadata to a dictionary with string keys and string/int/float values
            raw_md = parsed_response.get(self._metadata_output_key)
            normalized_md: dict[str, str | int | float] | None
            if raw_md is None:
                normalized_md = None
            elif isinstance(raw_md, dict):
                # Coerce keys to str and filter to str/int/float values only
                normalized_md = {str(k): v for k, v in raw_md.items() if isinstance(v, (str, int, float))}
                # If dictionary becomes empty after filtering, keep as empty dict
            elif isinstance(raw_md, (str, int, float)):
                # Wrap primitive metadata into a namespaced field
                normalized_md = {"metadata": raw_md}
            else:
                # Unrecognized metadata shape; drop to avoid downstream errors
                normalized_md = None

            score = UnvalidatedScore(
                raw_score_value=str(parsed_response[self._score_value_output_key]),
                score_value_description=parsed_response.get(self._description_output_key),
                score_category=normalized_category,
                score_rationale=parsed_response[self._rationale_output_key],
                scorer_class_identifier=scorer_identifier,
                score_metadata=normalized_md,
                message_piece_id=scored_prompt_id,
                objective=objective,
            )

        except json.JSONDecodeError:
            raise InvalidJsonException(message=f"Invalid JSON response: {response_json}") from None

        except KeyError:
            raise InvalidJsonException(message=f"Invalid JSON response, missing Key: {response_json}") from None

        return score
