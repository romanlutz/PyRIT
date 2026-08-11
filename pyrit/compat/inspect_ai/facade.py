# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dynamic module facade exposed temporarily as ``inspect_ai``."""

# The facade mirrors an external construction API. Its concise docstrings intentionally
# avoid duplicating the upstream API reference for every trivial constructor.
# ruff: noqa: DOC201, DOC501

from __future__ import annotations

import csv
import inspect
import json
import math
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast
from urllib.parse import urlparse

from pyrit.compat.inspect_ai.profile import InspectCompatibilityProfile, UnsupportedInspectFeatureError
from pyrit.compat.inspect_ai.types import (
    AgentPrompt,
    AgentSubmit,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
    Dataset,
    Epochs,
    FieldSpec,
    GenerateConfig,
    MemoryDataset,
    MetricSpec,
    Model,
    ModelName,
    ReducerSpec,
    Sample,
    SandboxSpec,
    Score,
    Scorer,
    ScorerSpec,
    Solver,
    SolverSpec,
    Target,
    Task,
    TaskState,
    Tool,
    ToolSpec,
    dataset_records_provenance,
)

DatasetLoader = Callable[..., object]
RecordMapper = Callable[[dict[str, Any]], Sample | list[Sample]]
TaskFactory = Callable[..., Task]
F = TypeVar("F", bound=Callable[..., Any])

_ACTIVE_TASK_REGISTRY: ContextVar[dict[str, TaskFactory] | None] = ContextVar(
    "inspect_compat_task_registry",
    default=None,
)
_ACTIVE_DATASET_LOADER: ContextVar[DatasetLoader | None] = ContextVar(
    "inspect_compat_dataset_loader",
    default=None,
)
_ACTIVE_ALLOW_NETWORK: ContextVar[bool] = ContextVar("inspect_compat_allow_network", default=False)
_ACTIVE_SOURCE_ROOT: ContextVar[Path | None] = ContextVar("inspect_compat_source_root", default=None)
_ACTIVE_DATA_ROOT: ContextVar[Path | None] = ContextVar("inspect_compat_data_root", default=None)
_ACTIVE_PROFILE: ContextVar[InspectCompatibilityProfile | None] = ContextVar(
    "inspect_compat_profile",
    default=None,
)


@dataclass(frozen=True)
class FacadeContextTokens:
    """Typed reset tokens for one facade activation."""

    task_registry: Token[dict[str, TaskFactory] | None]
    dataset_loader: Token[DatasetLoader | None]
    allow_network: Token[bool]
    source_root: Token[Path | None]
    data_root: Token[Path | None]
    profile: Token[InspectCompatibilityProfile | None]


class CompatibilityModule(ModuleType):
    """A module that fails precisely for undeclared compatibility symbols."""

    def __init__(self, *, name: str, profile: InspectCompatibilityProfile, is_package: bool = False) -> None:
        """Initialize a strict compatibility module."""
        super().__init__(name)
        self.__dict__["_compat_profile"] = profile
        if is_package:
            self.__dict__["__path__"] = []

    def __getattr__(self, name: str) -> object:
        """Reject unknown attributes instead of returning a success-shaped stub."""
        profile = cast("InspectCompatibilityProfile", self.__dict__["_compat_profile"])
        raise UnsupportedInspectFeatureError(
            symbol=f"{self.__name__}.{name}",
            source_profile=profile.profile_id,
        )


def activate_facade_context(
    *,
    task_registry: dict[str, TaskFactory],
    dataset_loader: DatasetLoader | None,
    allow_network: bool,
    source_root: Path,
    data_root: Path | None,
    profile: InspectCompatibilityProfile,
) -> FacadeContextTokens:
    """Activate per-load facade state and return reset tokens."""
    return FacadeContextTokens(
        task_registry=_ACTIVE_TASK_REGISTRY.set(task_registry),
        dataset_loader=_ACTIVE_DATASET_LOADER.set(dataset_loader),
        allow_network=_ACTIVE_ALLOW_NETWORK.set(allow_network),
        source_root=_ACTIVE_SOURCE_ROOT.set(source_root),
        data_root=_ACTIVE_DATA_ROOT.set(data_root),
        profile=_ACTIVE_PROFILE.set(profile),
    )


def deactivate_facade_context(tokens: FacadeContextTokens) -> None:
    """Reset per-load facade state."""
    _ACTIVE_TASK_REGISTRY.reset(tokens.task_registry)
    _ACTIVE_DATASET_LOADER.reset(tokens.dataset_loader)
    _ACTIVE_ALLOW_NETWORK.reset(tokens.allow_network)
    _ACTIVE_SOURCE_ROOT.reset(tokens.source_root)
    _ACTIVE_DATA_ROOT.reset(tokens.data_root)
    _ACTIVE_PROFILE.reset(tokens.profile)


def task(
    func: F | None = None,
    *,
    name: str | None = None,
    attribs: Mapping[str, Any] | None = None,
) -> F | Callable[[F], F]:
    """Register an Inspect-shaped task factory in the active isolated load."""
    del attribs

    def _decorate(factory: F) -> F:
        registry = _ACTIVE_TASK_REGISTRY.get()
        if registry is None:
            raise RuntimeError("The Inspect task decorator is only available inside the PyRIT compatibility loader.")
        task_name = name or _factory_name(factory)
        if task_name in registry:
            raise ValueError(f"Inspect task '{task_name}' was registered more than once.")
        registry[task_name] = cast("TaskFactory", factory)
        return factory

    return _decorate(func) if func is not None else _decorate


def registry_info(registry_type: str = "task") -> tuple[str, ...]:
    """Return registered component names for the active source load."""
    if registry_type != "task":
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_ai._util.registry.registry_info({registry_type})",
            source_profile=_active_profile().profile_id,
        )
    registry = _ACTIVE_TASK_REGISTRY.get()
    return tuple(sorted(registry or {}))


def registry_find(name: str, *, registry_type: str = "task") -> TaskFactory:
    """Look up one registered task factory in the active source load."""
    if registry_type != "task":
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_ai._util.registry.registry_find({registry_type})",
            source_profile=_active_profile().profile_id,
        )
    registry = _ACTIVE_TASK_REGISTRY.get()
    if registry is None or name not in registry:
        raise KeyError(f"Inspect {registry_type} registry has no entry named '{name}'.")
    return registry[name]


def _active_profile() -> InspectCompatibilityProfile:
    profile = _ACTIVE_PROFILE.get()
    if profile is None:
        raise RuntimeError("Inspect registry access is only available inside the compatibility loader.")
    return profile


def solver(func: F | None = None, *, name: str | None = None) -> F | Callable[[F], F]:
    """Capture a custom solver factory as a declarative graph node."""
    return _component_decorator(func=func, kind="solver", name=name)


def scorer(func: F | None = None, *, name: str | None = None, metrics: object = None) -> F | Callable[[F], F]:
    """Capture a custom scorer factory as a declarative graph node."""
    return _component_decorator(func=func, kind="scorer", name=name, metrics=metrics)


def tool(func: F | None = None, *, name: str | None = None) -> F | Callable[[F], F]:
    """Mark a custom tool factory without executing it."""
    return _component_decorator(func=func, kind="tool", name=name)


def _component_decorator(
    *,
    func: F | None,
    kind: str,
    name: str | None,
    metrics: object = None,
) -> F | Callable[[F], F]:
    def _decorate(factory: F) -> F:
        component_name = name or _factory_name(factory)
        source_module = getattr(factory, "__module__", None)
        source_qualname = getattr(factory, "__qualname__", None)
        if not isinstance(source_module, str) or not isinstance(source_qualname, str):
            raise TypeError("Inspect component factories must have string source identity attributes.")

        def _capture(*args: Any, **kwargs: Any) -> SolverSpec | ScorerSpec | ToolSpec:
            bound = inspect.signature(factory).bind(*args, **kwargs)
            bound.apply_defaults()
            config = {
                "factory_arguments": _json_compatible(dict(bound.arguments)),
                "source_identity": f"{source_module}.{source_qualname}",
            }
            if kind == "scorer":
                metric_specs = _metric_sequence(metrics)
                return ScorerSpec(name=component_name, config=config, metrics=metric_specs)
            spec_type = {"solver": SolverSpec, "tool": ToolSpec}[kind]
            return spec_type(name=component_name, config=config)

        _capture.__name__ = _factory_name(factory)
        _capture.__qualname__ = source_qualname
        _capture.__module__ = source_module
        return cast("F", _capture)

    return _decorate(func) if func is not None else _decorate


def _factory_name(factory: Callable[..., object]) -> str:
    value = getattr(factory, "__name__", None)
    if not isinstance(value, str):
        raise TypeError("Inspect component factories must have a string '__name__'.")
    return value


def multiple_choice(
    *,
    template: str | None = None,
    cot: bool = False,
    multiple_correct: bool = False,
    max_tokens: int | None = None,
    shuffle: bool | int | None = None,
) -> SolverSpec:
    """Construct the supported multiple-choice solver specification."""
    return SolverSpec(
        name="multiple_choice",
        config={
            "template": template,
            "cot": cot,
            "multiple_correct": multiple_correct,
            "max_tokens": max_tokens,
            "shuffle": shuffle,
        },
    )


def generate(
    tool_calls: str = "loop",
    *,
    max_retries: int | None = None,
    timeout: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    stop_seqs: list[str] | None = None,
    seed: int | None = None,
    cache: bool | None = False,
    **kwargs: Any,
) -> SolverSpec:
    """Construct a generate solver graph node."""
    return SolverSpec(
        name="generate",
        config={
            "tool_calls": tool_calls,
            "max_retries": max_retries,
            "timeout": timeout,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop_seqs": stop_seqs,
            "seed": seed,
            "cache": cache,
            **kwargs,
        },
    )


def chain(*solvers: SolverSpec | Iterable[SolverSpec]) -> SolverSpec:
    """Construct an ordered declarative solver chain."""
    steps: list[SolverSpec] = []
    for value in solvers:
        if isinstance(value, SolverSpec):
            steps.append(value)
        else:
            steps.extend(value)
    if not steps:
        raise ValueError("chain requires at least one solver.")
    if not all(isinstance(step, SolverSpec) for step in steps):
        raise TypeError("chain accepts only SolverSpec nodes.")
    return SolverSpec(name="chain", steps=tuple(steps))


def system_message(template: str, **params: Any) -> SolverSpec:
    """Construct a solver node that prepends or replaces the system message."""
    return SolverSpec(
        name="system_message",
        config={"template": _resolve_template_resource(template), "params": _json_compatible(params)},
    )


def prompt_template(template: str, **params: Any) -> SolverSpec:
    """Construct a solver node that templates the active user prompt."""
    return SolverSpec(
        name="prompt_template",
        config={"template": _resolve_template_resource(template), "params": _json_compatible(params)},
    )


def user_message(template: str, **params: Any) -> SolverSpec:
    """Construct a solver node that appends a user message."""
    return SolverSpec(
        name="user_message",
        config={"template": _resolve_template_resource(template), "params": _json_compatible(params)},
    )


def assistant_message(template: str, **params: Any) -> SolverSpec:
    """Construct a solver node that appends an assistant message."""
    return SolverSpec(
        name="assistant_message",
        config={"template": _resolve_template_resource(template), "params": _json_compatible(params)},
    )


def chain_of_thought(*, template: str | None = None) -> SolverSpec:
    """Construct a chain-of-thought solver graph node."""
    return SolverSpec(name="chain_of_thought", config={"template": template})


Generate = Callable[..., object]
Agent = SolverSpec
CORRECT = "C"
INCORRECT = "I"


def react(
    *,
    name: str | None = None,
    description: str | None = None,
    prompt: str | AgentPrompt | None = None,
    tools: Iterable[ToolSpec] | None = None,
    model: object = None,
    attempts: int = 1,
    submit: AgentSubmit | bool | None = None,
    on_continue: str | None = None,
    retry_refusals: int | None = None,
    compaction: object = None,
    truncation: str = "disabled",
    approval: object = None,
) -> SolverSpec:
    """Construct the pinned ReAct agent graph consumed by the CTF tasks."""
    unsupported = {
        "name": name,
        "description": description,
        "model": model,
        "on_continue": on_continue,
        "retry_refusals": retry_refusals,
        "compaction": compaction,
        "approval": approval,
    }
    requested = next((key for key, value in unsupported.items() if value is not None), None)
    if requested is not None:
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_ai.agent.react({requested}=...)",
            source_profile=_active_profile().profile_id,
        )
    if truncation != "disabled":
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.agent.react(truncation=...)",
            source_profile=_active_profile().profile_id,
        )
    if attempts <= 0:
        raise ValueError("react attempts must be greater than zero.")
    prompt_value = prompt if isinstance(prompt, AgentPrompt) else AgentPrompt(instructions=prompt)
    submit_value = AgentSubmit() if submit in (None, True) else submit
    if submit_value is False:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.agent.react(submit=False)",
            source_profile=_active_profile().profile_id,
        )
    return SolverSpec(
        name="react",
        config={
            "prompt": _json_compatible(prompt_value),
            "tools": [_json_compatible(tool) for tool in tools or ()],
            "attempts": attempts,
            "submit": _json_compatible(submit_value),
        },
    )


def as_solver(agent: SolverSpec) -> SolverSpec:
    """Return an agent graph as its solver graph."""
    return agent


def bash(
    *,
    timeout: float | None = None,
    user: str | None = None,
    sandbox: str | None = None,
    background: bool = False,
) -> ToolSpec:
    """Construct the pinned standard bash tool."""
    if background:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.tool.bash(background=True)",
            source_profile=_active_profile().profile_id,
        )
    return ToolSpec(name="bash", config={"timeout": timeout, "user": user, "sandbox": sandbox})


def python(
    *,
    timeout: float | None = None,
    user: str | None = None,
    sandbox: str | None = None,
) -> ToolSpec:
    """Construct the pinned standard Python tool."""
    return ToolSpec(name="python", config={"timeout": timeout, "user": user, "sandbox": sandbox})


@contextmanager
def message_limit(limit: int) -> Iterator[None]:
    """Provide the construction-time context shape used by the pinned solver callback."""
    if limit <= 0:
        raise ValueError("message_limit must be greater than zero.")
    yield


def sandbox(name: str | None = None) -> object:
    """Reject direct sandbox use in the construction worker."""
    raise UnsupportedInspectFeatureError(
        symbol=f"inspect_ai.util.sandbox({name or 'default'}) runtime",
        source_profile=_active_profile().profile_id,
        remediation="Use the native bounded compatibility callback proxy during scoring.",
    )


def choice() -> ScorerSpec:
    """Construct the supported choice scorer specification."""
    return ScorerSpec(
        name="choice",
        config={},
        metrics=(MetricSpec(name="accuracy"), MetricSpec(name="stderr", config={"cluster": None})),
    )


def match(*, location: str = "end", ignore_case: bool = True, numeric: bool = False) -> ScorerSpec:
    """Construct an exact-match scorer graph node."""
    return ScorerSpec(
        name="match",
        config={"location": location, "ignore_case": ignore_case, "numeric": numeric},
        metrics=(MetricSpec(name="accuracy"), MetricSpec(name="stderr", config={"cluster": None})),
    )


def includes(*, ignore_case: bool = True) -> ScorerSpec:
    """Construct an includes scorer graph node."""
    return ScorerSpec(
        name="includes",
        config={"ignore_case": ignore_case},
        metrics=(MetricSpec(name="accuracy"), MetricSpec(name="stderr", config={"cluster": None})),
    )


def accuracy(to_float: object = None) -> MetricSpec:
    """Construct an accuracy metric reference."""
    if to_float is not None:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.scorer.accuracy(to_float=...)",
            source_profile=_active_profile().profile_id,
        )
    return MetricSpec(name="accuracy")


def mean(to_float: object = None) -> MetricSpec:
    """Construct an arithmetic-mean metric reference."""
    if to_float is not None:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.scorer.mean(to_float=...)",
            source_profile=_active_profile().profile_id,
        )
    return MetricSpec(name="mean")


def stderr(to_float: object = None, cluster: str | None = None) -> MetricSpec:
    """Construct a standard-error metric reference."""
    if to_float is not None:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.scorer.stderr(to_float=...)",
            source_profile=_active_profile().profile_id,
        )
    return MetricSpec(name="stderr", config={"cluster": cluster})


def grouped(
    metric: MetricSpec,
    group_key: str,
    *,
    all: str | bool = "samples",  # noqa: A002 - Mirrors the pinned Inspect API.
    all_label: str = "all",
    name_template: str = "{group_name}",
    value_to_float: object = None,
) -> MetricSpec:
    """Construct a grouped metric over one sample metadata key."""
    if value_to_float is not None:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.scorer.grouped(value_to_float=...)",
            source_profile=_active_profile().profile_id,
        )
    if all not in ("samples", "groups", False):
        raise ValueError("grouped(all=...) must be 'samples', 'groups', or False.")
    return MetricSpec(
        name="grouped",
        config={
            "metric": _json_compatible(metric),
            "group_key": group_key,
            "all": all,
            "all_label": all_label,
            "name_template": name_template,
        },
    )


def mean_score() -> ReducerSpec:
    """Construct the default arithmetic-mean score reducer."""
    return ReducerSpec(name="mean")


def at_least(k: int, value: float = 1.0) -> ReducerSpec:
    """Construct a threshold-count score reducer."""
    return ReducerSpec(name="at_least", config={"k": k, "value": value})


def pass_at(k: int, value: float = 1.0) -> ReducerSpec:
    """Construct the Chen pass-at-k estimator."""
    return ReducerSpec(name="pass_at", config={"k": k, "value": value})


def pass_k(k: int, value: float = 1.0) -> ReducerSpec:
    """Construct the all-k-correct reliability estimator."""
    return ReducerSpec(name="pass_k", config={"k": k, "value": value})


def get_model(
    model: str | ModelName | None = None,
    *,
    config: GenerateConfig | None = None,
) -> Model:
    """Construct a model reference; compatibility never invokes a provider."""
    return Model(name=model or "injected-pyrit-target", config=config)


def hf_dataset(
    *,
    path: str,
    name: str | None = None,
    split: str | None = None,
    sample_fields: RecordMapper | FieldSpec | None = None,
    revision: str | None = None,
    data_dir: str | None = None,
    auto_id: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_choices: bool | int | None = None,
    limit: int | None = None,
    trust: bool = False,
    cached: bool = True,
    retry: bool = True,
    **kwargs: Any,
) -> Dataset:
    """Materialize a pinned Hugging Face dataset through the injected loader."""
    if not revision:
        raise TypeError("hf_dataset() requires a pinned 'revision'.")
    if trust:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.dataset.hf_dataset(trust=True)",
            source_profile=_active_profile().profile_id,
            remediation="Remote dataset code is never executed by the compatibility loader.",
        )
    if not cached or not retry:
        option = "cached" if not cached else "retry"
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_ai.dataset.hf_dataset({option}=...)",
            source_profile=_active_profile().profile_id,
        )
    if kwargs:
        option = sorted(kwargs)[0]
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_ai.dataset.hf_dataset({option}=...)",
            source_profile=_active_profile().profile_id,
            remediation="Materialize this dataset option outside the compatibility loader and inject records.",
        )
    loader = _ACTIVE_DATASET_LOADER.get()
    if loader is None:
        if not _ACTIVE_ALLOW_NETWORK.get():
            raise ValueError(
                "Inspect compatibility dataset loading is offline by default; inject dataset_loader or set "
                "allow_network=True."
            )
        from datasets import load_dataset

        loader = load_dataset
    raw = loader(
        path,
        name,
        split=split,
        revision=revision,
        data_dir=data_dir,
        trust_remote_code=False,
    )
    dataset = _map_records(
        raw=raw,
        sample_fields=sample_fields,
        name=name,
        location=path,
        metadata={
            "source_type": "huggingface",
            "path": path,
            "name": name,
            "split": split,
            "revision": revision,
            "injected_records": _ACTIVE_DATASET_LOADER.get() is not None,
            **({"data_dir": data_dir} if data_dir is not None else {}),
        },
    )
    return _apply_dataset_options(
        dataset=dataset,
        auto_id=auto_id,
        shuffle=shuffle,
        seed=seed,
        shuffle_choices=shuffle_choices,
        limit=limit,
    )


def json_dataset(
    json_file: str | None = None,
    *,
    file_path: str | None = None,
    sample_fields: RecordMapper | FieldSpec | None = None,
    auto_id: bool = False,
    name: str | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_choices: bool | int | None = None,
    limit: int | None = None,
    encoding: str = "utf-8",
    fs_options: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dataset:
    """Materialize a source-contained JSON or JSONL dataset."""
    if kwargs:
        option = sorted(kwargs)[0]
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_ai.dataset.json_dataset({option}=...)",
            source_profile=_active_profile().profile_id,
        )
    source = file_path or json_file
    if source is None:
        raise TypeError("json_dataset() requires 'json_file' or 'file_path'.")
    if fs_options:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.dataset.json_dataset(fs_options=...)",
            source_profile=_active_profile().profile_id,
        )
    path = _resolve_source_path(source)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding=encoding).splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding=encoding))
    dataset = _map_records(
        raw=records,
        sample_fields=sample_fields,
        name=name,
        location=source,
        metadata={"source_type": "json", "path": source},
    )
    return _apply_dataset_options(
        dataset=dataset,
        auto_id=auto_id,
        shuffle=shuffle,
        seed=seed,
        shuffle_choices=shuffle_choices,
        limit=limit,
    )


def load_json_dataset(
    file_path: str,
    eval_name: str,
    sample_fields: RecordMapper,
    *,
    shuffle: bool = False,
    **kwargs: Any,
) -> Dataset:
    """Provide the pinned inspect-evals local JSON helper over the strict facade loader."""
    unsupported = sorted(set(kwargs).difference({"cache_tag", "encoding", "fs_options", "refresh"}))
    if unsupported:
        raise UnsupportedInspectFeatureError(
            symbol=f"inspect_evals.utils.load_json_dataset({unsupported[0]}=...)",
            source_profile=_active_profile().profile_id,
        )
    return json_dataset(
        file_path=file_path,
        sample_fields=sample_fields,
        name=eval_name,
        shuffle=shuffle,
    )


def download_and_verify(*args: Any, **kwargs: Any) -> None:
    """Reject construction-time downloads; compatibility requires a pinned local cache."""
    del args, kwargs
    raise UnsupportedInspectFeatureError(
        symbol="inspect_evals.utils.download_and_verify runtime",
        source_profile=_active_profile().profile_id,
        remediation="Provide a pre-populated pinned INSPECT_EVALS_CACHE_DIR.",
    )


def csv_dataset(
    csv_file: str,
    *,
    sample_fields: RecordMapper | FieldSpec | None = None,
    auto_id: bool = False,
    shuffle: bool = False,
    seed: int | None = None,
    shuffle_choices: bool | int | None = None,
    limit: int | None = None,
    dialect: str = "unix",
    encoding: str = "utf-8",
    name: str | None = None,
    fs_options: dict[str, Any] | None = None,
    fieldnames: list[str] | None = None,
    delimiter: str = ",",
) -> Dataset:
    """Materialize a source-contained CSV dataset."""
    if fs_options:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.dataset.csv_dataset(fs_options=...)",
            source_profile=_active_profile().profile_id,
        )
    path = _resolve_source_path(csv_file)
    with path.open(encoding=encoding, newline="") as handle:
        records = list(
            csv.DictReader(
                handle,
                fieldnames=fieldnames,
                dialect=dialect,
                delimiter=delimiter,
            )
        )
    dataset = _map_records(
        raw=records,
        sample_fields=sample_fields,
        name=name,
        location=csv_file,
        metadata={"source_type": "csv", "path": csv_file},
    )
    return _apply_dataset_options(
        dataset=dataset,
        auto_id=auto_id,
        shuffle=shuffle,
        seed=seed,
        shuffle_choices=shuffle_choices,
        limit=limit,
    )


def _map_records(
    *,
    raw: object,
    sample_fields: RecordMapper | FieldSpec | None,
    name: str | None,
    location: str,
    metadata: dict[str, Any],
) -> Dataset:
    if isinstance(raw, Mapping):
        raise TypeError("Dataset loader returned a mapping; pass a concrete split or iterable of records.")
    if not isinstance(raw, Iterable):
        raise TypeError(f"Dataset loader returned unsupported type '{type(raw).__name__}'.")
    samples: list[Sample] = []
    for record in raw:
        if isinstance(record, Sample):
            mapped: Sample | list[Sample] = record
        elif isinstance(record, Mapping):
            item = dict(record)
            mapped = _map_record(item=item, sample_fields=sample_fields)
        else:
            raise TypeError(f"Dataset record has unsupported type '{type(record).__name__}'.")
        record_samples = cast("list[Sample]", mapped) if isinstance(mapped, list) else [mapped]
        if not all(isinstance(sample, Sample) for sample in record_samples):
            raise TypeError("Dataset record mapper must return inspect_ai.dataset.Sample or a list of samples.")
        samples.extend(record_samples)
    return _dataset_with_provenance(samples=samples, name=name, location=location, metadata=metadata)


def _map_record(
    *,
    item: dict[str, Any],
    sample_fields: RecordMapper | FieldSpec | None,
) -> Sample | list[Sample]:
    if sample_fields is None:
        sample_fields = FieldSpec()
    if not isinstance(sample_fields, FieldSpec):
        sample = sample_fields(item)
        if not isinstance(sample, Sample) and (
            not isinstance(sample, list) or not all(isinstance(item, Sample) for item in sample)
        ):
            raise TypeError("Dataset record mapper must return inspect_ai.dataset.Sample or a list of samples.")
        return sample
    if sample_fields.metadata:
        metadata = {field_name: item.get(field_name) for field_name in sample_fields.metadata}
    else:
        metadata_value = item.get("metadata")
        if isinstance(metadata_value, str):
            metadata_value = json.loads(metadata_value)
        if metadata_value is not None and not isinstance(metadata_value, dict):
            raise ValueError(f"Unexpected type for 'metadata' field: {type(metadata_value).__name__}.")
        metadata = dict(metadata_value or {})
    values: dict[str, Any] = {
        "input": _read_record_input(item.get(sample_fields.input)),
        "target": _read_record_target(item.get(sample_fields.target)) if sample_fields.target is not None else "",
        "choices": _read_record_choices(item.get(sample_fields.choices)) if sample_fields.choices is not None else None,
        "id": item.get(sample_fields.id) if sample_fields.id is not None else None,
        "metadata": metadata,
    }
    if sample_fields.sandbox is not None:
        values["sandbox"] = _read_record_sandbox(item.get(sample_fields.sandbox))
    if sample_fields.files is not None:
        values["files"] = _read_record_files(item.get(sample_fields.files))
    if sample_fields.setup is not None:
        setup = item.get(sample_fields.setup)
        values["setup"] = str(setup) if setup is not None else None
    return Sample(**values)


def _read_record_input(value: object) -> str | list[ChatMessage]:
    if not value:
        raise ValueError("No input in dataset.")
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise TypeError(f"Dataset input must be text or a message list, not '{type(value).__name__}'.")
    messages = []
    for index, item in enumerate(value):
        if isinstance(item, ChatMessage):
            messages.append(item)
            continue
        if not isinstance(item, Mapping):
            raise TypeError(f"Dataset input message {index} must be a mapping.")
        role = item.get("role")
        content = _read_record_content(item.get("content"), path=f"input[{index}].content")
        message_type = {
            "system": ChatMessageSystem,
            "user": ChatMessageUser,
            "assistant": ChatMessageAssistant,
        }.get(role)
        if message_type is None:
            raise UnsupportedInspectFeatureError(
                symbol=f"inspect_ai.dataset.FieldSpec.input[{index}].role={role!r}",
                source_profile=_active_profile().profile_id,
            )
        unsupported = next(
            (name for name in ("tool_calls", "tool_call_id", "function", "error") if item.get(name) is not None),
            None,
        )
        if unsupported is not None:
            raise UnsupportedInspectFeatureError(
                symbol=f"inspect_ai.dataset.FieldSpec.input[{index}].{unsupported}",
                source_profile=_active_profile().profile_id,
            )
        messages.append(message_type(content=content, source="input"))
    return messages


def _read_record_content(value: object, *, path: str) -> str | list[ContentText | ContentImage]:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise ValueError(f"{path} must contain text or a non-empty content-part list.")
    parts = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise TypeError(f"{path}[{index}] must be a mapping.")
        content_type = item.get("type")
        text = item.get("text")
        image = item.get("image")
        if content_type == "text" and isinstance(text, str):
            parts.append(ContentText(text=text))
        elif content_type == "image" and isinstance(image, str):
            detail = item.get("detail")
            if detail is not None and not isinstance(detail, str):
                raise TypeError(f"{path}[{index}].detail must be a string or None.")
            parts.append(ContentImage(image=image, detail=detail))
        else:
            raise UnsupportedInspectFeatureError(
                symbol=f"inspect_ai.dataset.FieldSpec.{path}[{index}].type={content_type!r}",
                source_profile=_active_profile().profile_id,
            )
    return parts


def _read_record_target(value: object) -> str | list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, list):
        return [str(item) for item in value]
    return str(value)


def _read_record_choices(value: object) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        choices = value.split(",")
        if len(choices) == 1:
            choices = value.split()
        return [choice.strip() for choice in choices]
    return [str(value)]


def _read_record_sandbox(value: object) -> SandboxSpec | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip().startswith("["):
            return SandboxSpec(type=value)
        value = json.loads(value)
    if isinstance(value, list) and len(value) == 2:
        return SandboxSpec(type=str(value[0]), config=str(value[1]))
    raise ValueError("Dataset sandbox must be a string or two-item list.")


def _read_record_files(value: object) -> dict[str, str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return cast("dict[str, str]", value)
    raise ValueError("Dataset files must be a string-to-string mapping or its JSON representation.")


def _apply_dataset_options(
    *,
    dataset: Dataset,
    auto_id: bool,
    shuffle: bool,
    seed: int | None,
    shuffle_choices: bool | int | None,
    limit: int | None,
) -> Dataset:
    if limit is not None and limit < 0:
        raise ValueError("Dataset limit must be non-negative.")
    effective_seed = seed if seed is not None else 0
    samples = [replace(sample, id=index + 1) if auto_id else sample for index, sample in enumerate(dataset)]
    shuffle_choices_enabled = shuffle_choices is True or (
        isinstance(shuffle_choices, int) and not isinstance(shuffle_choices, bool)
    )
    choice_seed = None
    if shuffle_choices_enabled:
        choice_seed = (
            shuffle_choices if isinstance(shuffle_choices, int) and not isinstance(shuffle_choices, bool) else None
        )
        rng = random.Random(choice_seed if choice_seed is not None else effective_seed)
        samples = [_shuffle_sample_choices(sample=sample, rng=rng) for sample in samples]
    if shuffle:
        random.Random(effective_seed).shuffle(samples)
    if limit is not None:
        samples = samples[:limit]
    return _dataset_with_provenance(
        samples=samples,
        name=dataset.name,
        location=dataset.location or "",
        metadata={
            **dataset.metadata,
        },
        shuffled=shuffle,
        provenance={
            **dataset.provenance,
            "selection": {
                "auto_id": auto_id,
                "shuffle": shuffle,
                "seed": effective_seed if shuffle or shuffle_choices_enabled else seed,
                "shuffle_choices": shuffle_choices,
                **(
                    {"choice_seed": choice_seed if choice_seed is not None else effective_seed}
                    if shuffle_choices_enabled
                    else {}
                ),
                "limit": limit,
            },
        },
    )


def _shuffle_sample_choices(*, sample: Sample, rng: random.Random) -> Sample:
    if not sample.choices:
        return sample
    indexes = list(range(len(sample.choices)))
    rng.shuffle(indexes)
    inverse = {original_index: new_index for new_index, original_index in enumerate(indexes)}
    choices = [sample.choices[index] for index in indexes]

    def _target(value: str) -> str:
        normalized = value.strip().upper()
        if len(normalized) != 1 or not normalized.isalpha():
            raise UnsupportedInspectFeatureError(
                symbol="inspect_ai.dataset.shuffle_choices(non-letter target)",
                source_profile=_active_profile().profile_id,
            )
        original_index = ord(normalized) - ord("A")
        if original_index not in inverse:
            raise ValueError(f"Choice target '{value}' is outside the sample's choices.")
        return chr(ord("A") + inverse[original_index])

    target = [_target(value) for value in sample.target] if isinstance(sample.target, list) else _target(sample.target)
    return replace(sample, choices=choices, target=target)


def _dataset_with_provenance(
    *,
    samples: Iterable[Sample],
    name: str | None,
    location: str,
    metadata: dict[str, Any],
    shuffled: bool = False,
    provenance: dict[str, Any] | None = None,
) -> Dataset:
    materialized = tuple(samples)
    return Dataset(
        materialized,
        name=name,
        location=location,
        shuffled=shuffled,
        metadata=metadata,
        provenance={
            **(provenance or {}),
            **dataset_records_provenance(materialized),
        },
    )


def _resolve_template_resource(template: str) -> str:
    root = _ACTIVE_SOURCE_ROOT.get()
    if root is None:
        raise RuntimeError("Inspect template resources can only be resolved inside the compatibility loader.")
    parsed = urlparse(template)
    if parsed.scheme and "://" in template:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.util.resource(remote template)",
            source_profile=_active_profile().profile_id,
            remediation="Template resources must be source-contained local files or literal strings.",
        )
    try:
        candidate = Path(template)
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        exists = candidate.exists()
    except (OSError, ValueError):
        return template
    if not exists:
        return template
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"Template resource path is outside the trusted source root: '{template}'.")
    if not candidate.is_file():
        raise ValueError(f"Template resource path is not a file: '{template}'.")
    return candidate.read_text(encoding="utf-8")


def _resolve_source_path(value: str) -> Path:
    root = _ACTIVE_SOURCE_ROOT.get()
    if root is None:
        raise RuntimeError("Source-contained datasets can only be loaded inside the compatibility loader.")
    candidate = Path(value)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        data_root = _ACTIVE_DATA_ROOT.get()
        allowed_roots = tuple(root for root in (root, data_root) if root is not None)
        if not any(resolved == allowed or allowed in resolved.parents for allowed in allowed_roots):
            raise ValueError(f"Dataset path is outside the trusted source/data roots: '{value}'.")
        return resolved
    resolved = (root / candidate).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"Dataset path '{value}' escapes the trusted source root.")
    return resolved


def build_compatibility_modules(*, profile: InspectCompatibilityProfile) -> dict[str, ModuleType]:
    """Build the temporary, strict ``inspect_ai`` module tree."""
    modules: dict[str, ModuleType] = {
        name: CompatibilityModule(name=name, profile=profile, is_package=True)
        for name in (
            "inspect_ai",
            "inspect_ai.agent",
            "inspect_ai._util",
            "inspect_ai._util.registry",
            "inspect_ai.dataset",
            "inspect_ai.model",
            "inspect_ai.sandbox",
            "inspect_ai.scorer",
            "inspect_ai.solver",
            "inspect_ai.tool",
            "inspect_ai.util",
        )
    }
    _export(modules["inspect_ai"], {"Epochs": Epochs, "Task": Task, "task": task})
    _export(
        modules["inspect_ai.agent"],
        {
            "Agent": Agent,
            "AgentPrompt": AgentPrompt,
            "AgentSubmit": AgentSubmit,
            "as_solver": as_solver,
            "react": react,
        },
    )
    _export(
        modules["inspect_ai._util.registry"],
        {
            "registry_find": registry_find,
            "registry_info": registry_info,
        },
    )
    _export(
        modules["inspect_ai.dataset"],
        {
            "Dataset": Dataset,
            "FieldSpec": FieldSpec,
            "MemoryDataset": MemoryDataset,
            "Sample": Sample,
            "csv_dataset": csv_dataset,
            "hf_dataset": hf_dataset,
            "json_dataset": json_dataset,
        },
    )
    _export(
        modules["inspect_ai.model"],
        {
            "ChatMessage": ChatMessage,
            "ChatMessageAssistant": ChatMessageAssistant,
            "ChatMessageSystem": ChatMessageSystem,
            "ChatMessageTool": ChatMessageTool,
            "ChatMessageUser": ChatMessageUser,
            "ContentImage": ContentImage,
            "ContentText": ContentText,
            "GenerateConfig": GenerateConfig,
            "Model": Model,
            "ModelName": ModelName,
            "get_model": get_model,
        },
    )
    _export(modules["inspect_ai.sandbox"], {"SandboxSpec": SandboxSpec})
    _export(
        modules["inspect_ai.scorer"],
        {
            "CORRECT": CORRECT,
            "INCORRECT": INCORRECT,
            "Score": Score,
            "Scorer": Scorer,
            "ScorerSpec": ScorerSpec,
            "Target": Target,
            "accuracy": accuracy,
            "at_least": at_least,
            "choice": choice,
            "grouped": grouped,
            "includes": includes,
            "match": match,
            "mean": mean,
            "mean_score": mean_score,
            "pass_at": pass_at,
            "pass_k": pass_k,
            "scorer": scorer,
            "stderr": stderr,
        },
    )
    _export(
        modules["inspect_ai.solver"],
        {
            "Generate": Generate,
            "Solver": Solver,
            "SolverSpec": SolverSpec,
            "TaskState": TaskState,
            "assistant_message": assistant_message,
            "chain": chain,
            "chain_of_thought": chain_of_thought,
            "generate": generate,
            "multiple_choice": multiple_choice,
            "prompt_template": prompt_template,
            "solver": solver,
            "system_message": system_message,
            "user_message": user_message,
        },
    )
    _export(
        modules["inspect_ai.tool"],
        {"Tool": Tool, "ToolSpec": ToolSpec, "bash": bash, "python": python, "tool": tool},
    )
    _export(
        modules["inspect_ai.util"],
        {
            "SandboxEnvironmentSpec": SandboxSpec,
            "message_limit": message_limit,
            "sandbox": sandbox,
        },
    )
    for name, module in modules.items():
        if "." not in name:
            continue
        parent_name, child_name = name.rsplit(".", 1)
        setattr(modules[parent_name], child_name, module)
    return modules


def _export(module: ModuleType, symbols: Mapping[str, object]) -> None:
    module.__dict__.update(symbols)
    module.__dict__["__all__"] = tuple(sorted(symbols))


def _json_compatible(value: object) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        field_names = cast("Mapping[str, object]", value.__dataclass_fields__)
        return {name: _json_compatible(getattr(value, name)) for name in field_names}
    raise TypeError(f"Inspect compatibility value '{type(value).__name__}' is not JSON serializable.")


def _metric_sequence(value: object) -> tuple[MetricSpec, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, (list, tuple)) else (value,)
    if not all(isinstance(item, MetricSpec) for item in values):
        raise TypeError("Inspect scorer metrics must contain only declarative MetricSpec nodes.")
    return tuple(cast("Iterable[MetricSpec]", values))
