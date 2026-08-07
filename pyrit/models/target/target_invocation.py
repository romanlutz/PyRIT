# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from pyrit.models.score import ComponentIdentifierField  # noqa: TC001 (runtime-required by Pydantic)


class TargetInvocation(BaseModel):
    """The target and fully resolved request options used for one invocation."""

    METADATA_KEY: ClassVar[str] = "target_invocation"

    model_config = ConfigDict(frozen=True, extra="forbid")

    target_identifier: ComponentIdentifierField
    effective_options: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        """Return the JSON-compatible prompt metadata representation."""
        return self.model_dump(mode="json")

    @classmethod
    def from_metadata(cls, *, metadata: dict[str, Any]) -> TargetInvocation | None:
        """
        Build an invocation from prompt metadata when present.

        Returns:
            The parsed invocation, or None when the metadata has no invocation.
        """
        value = metadata.get(cls.METADATA_KEY)
        return cls.model_validate(value) if value is not None else None
