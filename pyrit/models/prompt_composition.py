# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from enum import Enum

from pydantic import BaseModel, Field

PROMPT_COMPOSITION_METADATA_KEY = "prompt_composition"


class PromptCompositionStrategy(str, Enum):
    """Known strategies used to assemble a generated prompt."""

    MANY_SHOT = "many_shot"


class PromptObjectivePlacement(str, Enum):
    """Where the objective appears within a generated prompt."""

    APPENDED = "appended"


class PromptComposition(BaseModel):
    """Structured provenance for a generated prompt."""

    strategy: PromptCompositionStrategy
    example_count: int = Field(ge=0)
    objective_placement: PromptObjectivePlacement
    character_count: int = Field(ge=0)
