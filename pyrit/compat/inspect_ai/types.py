# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inspect-shaped construction types used only at the compatibility boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, overload


@dataclass(frozen=True, kw_only=True)
class ContentText:
    """Text content in an Inspect-shaped chat message."""

    text: str
    type: Literal["text"] = "text"


@dataclass(frozen=True, kw_only=True)
class ContentImage:
    """Image reference in an Inspect-shaped chat message."""

    image: str
    detail: str | None = None
    type: Literal["image"] = "image"


MessageContent = str | list[ContentText | ContentImage]


@dataclass(frozen=True, kw_only=True)
class ChatMessage:
    """An Inspect-shaped chat message."""

    content: MessageContent
    role: str
    name: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ChatMessageUser(ChatMessage):
    """A user chat message."""

    role: str = "user"


@dataclass(frozen=True, kw_only=True)
class ChatMessageAssistant(ChatMessage):
    """An assistant chat message."""

    role: str = "assistant"


@dataclass(frozen=True, kw_only=True)
class ChatMessageSystem(ChatMessage):
    """A system chat message."""

    role: str = "system"


@dataclass(frozen=True, kw_only=True)
class ChatMessageTool(ChatMessage):
    """A tool chat message."""

    role: str = "tool"


@dataclass(frozen=True, kw_only=True)
class Sample:
    """One Inspect-shaped dataset sample."""

    input: str | list[ChatMessage]
    target: str | list[str] = ""
    choices: list[str] | None = None
    id: str | int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    sandbox: SandboxSpec | tuple[str, str] | str | None = None
    setup: str | None = None
    files: dict[str, str] | None = None


class Dataset(Sequence[Sample]):
    """A materialized, deterministic Inspect-shaped dataset."""

    def __init__(
        self,
        samples: Iterable[Sample],
        *,
        name: str | None = None,
        location: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a materialized dataset."""
        self._samples = tuple(samples)
        self.name = name
        self.location = location
        self.metadata = metadata or {}

    @overload
    def __getitem__(self, index: int) -> Sample: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Sample]: ...

    def __getitem__(self, index: int | slice) -> Sample | Sequence[Sample]:
        """Return a sample or sample slice."""
        return self._samples[index]

    def __iter__(self) -> Iterator[Sample]:
        """
        Iterate over materialized samples.

        Returns:
            Iterator[Sample]: The sample iterator.
        """
        return iter(self._samples)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self._samples)

    def filter(self, predicate: Callable[[Sample], bool]) -> Dataset:
        """Return a dataset containing samples accepted by ``predicate``."""
        return Dataset(
            (sample for sample in self._samples if predicate(sample)),
            name=self.name,
            location=self.location,
            metadata=dict(self.metadata),
        )


class MemoryDataset(Dataset):
    """An in-memory Inspect-shaped dataset."""


@dataclass(frozen=True, kw_only=True)
class SolverSpec:
    """A declarative Inspect solver graph node."""

    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ScorerSpec:
    """A declarative Inspect scorer graph node."""

    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ToolSpec:
    """A declarative Inspect tool graph node."""

    name: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class AgentPrompt:
    """Prompt fragments used to construct a ReAct agent."""

    instructions: str | None = None
    handoff_prompt: str | None = None
    assistant_prompt: str | None = None
    submit_prompt: str | None = None


@dataclass(frozen=True, kw_only=True)
class AgentSubmit:
    """Submission-tool behavior used by a ReAct agent."""

    name: str | None = None
    description: str | None = None
    answer_only: bool = False
    answer_delimiter: str = "\n\n"
    keep_in_messages: bool = False


@dataclass(frozen=True, kw_only=True)
class SandboxSpec:
    """A declarative Inspect sandbox reference."""

    type: str
    config: str | dict[str, Any] | None = None


@dataclass(frozen=True)
class Epochs:
    """Inspect-shaped epoch count and reducer reference."""

    epochs: int
    reducer: str | list[str] | None = None

    def __post_init__(self) -> None:
        """
        Validate the positive epoch count.

        Raises:
            ValueError: If the epoch count is not positive.
        """
        if self.epochs <= 0:
            raise ValueError("Epochs must be greater than zero.")


@dataclass(frozen=True, kw_only=True)
class GenerateConfig:
    """Inspect-shaped model generation configuration."""

    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    stop_seqs: list[str] | None = None
    max_connections: int | None = None
    system_message: str | None = None


@dataclass(frozen=True, kw_only=True)
class ModelName:
    """An Inspect-shaped model name reference."""

    model: str
    api: str | None = None


@dataclass(frozen=True, kw_only=True)
class Model:
    """A non-executing model reference."""

    name: str | ModelName
    config: GenerateConfig | None = None


@dataclass(frozen=True, kw_only=True)
class Score:
    """An Inspect-shaped score primitive."""

    value: bool | int | float | str
    answer: str | None = None
    explanation: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Target(list[str]):
    """A list-like Inspect scoring target."""

    def __init__(self, value: str | Iterable[str] = ()) -> None:
        """Initialize a target from one or more accepted values."""
        super().__init__([value] if isinstance(value, str) else value)

    @property
    def text(self) -> str:
        """The first accepted target value."""
        return self[0] if self else ""


@dataclass(frozen=True, kw_only=True)
class Task:
    """An Inspect-shaped task construction graph."""

    dataset: Dataset | Sequence[Sample]
    solver: SolverSpec | Sequence[SolverSpec]
    scorer: ScorerSpec | Sequence[ScorerSpec]
    sandbox: SandboxSpec | tuple[str, str] | str | None = None
    config: GenerateConfig | None = None
    epochs: int | Epochs | None = None
    fail_on_error: bool | float | None = None
    message_limit: int | None = None
    token_limit: int | None = None
    time_limit: int | None = None
    working_limit: int | None = None
    version: int | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


Solver = SolverSpec
Scorer = ScorerSpec
Tool = ToolSpec
Agent = SolverSpec
Generate = Callable[..., object]


class TaskState:
    """Construction-only type marker for pinned callback annotations."""
