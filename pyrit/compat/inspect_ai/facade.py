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
import random
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

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
    GenerateConfig,
    MemoryDataset,
    Model,
    ModelName,
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
)

DatasetLoader = Callable[..., object]
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
            if metrics is not None:
                config["metrics"] = _json_compatible(metrics)
            spec_type = {"solver": SolverSpec, "scorer": ScorerSpec, "tool": ToolSpec}[kind]
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


def generate(*, cache: bool = False, tool_calls: str = "loop") -> SolverSpec:
    """Construct a generate solver graph node."""
    return SolverSpec(name="generate", config={"cache": cache, "tool_calls": tool_calls})


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
    return ScorerSpec(name="choice", config={})


def match(*, location: str = "end", ignore_case: bool = True) -> ScorerSpec:
    """Construct an exact-match scorer graph node."""
    return ScorerSpec(name="match", config={"location": location, "ignore_case": ignore_case})


def includes(*, ignore_case: bool = True) -> ScorerSpec:
    """Construct an includes scorer graph node."""
    return ScorerSpec(name="includes", config={"ignore_case": ignore_case})


def accuracy() -> str:
    """Construct an accuracy metric reference."""
    return "accuracy"


def stderr() -> str:
    """Construct a standard-error metric reference."""
    return "stderr"


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
    sample_fields: Callable[[dict[str, Any]], Sample] | None = None,
    revision: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """Materialize a pinned Hugging Face dataset through the injected loader."""
    if not revision:
        raise TypeError("hf_dataset() requires a pinned 'revision'.")
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
    raw = loader(path, name, split=split, revision=revision, trust_remote_code=False)
    return _map_records(
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
        },
    )


def json_dataset(
    *,
    json_file: str | None = None,
    file_path: str | None = None,
    sample_fields: Callable[[dict[str, Any]], Sample] | None = None,
    name: str | None = None,
    shuffle: bool = False,
    seed: int | None = None,
    limit: int | bool | None = None,
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
    path = _resolve_source_path(source)
    if path.suffix.lower() == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = json.loads(path.read_text(encoding="utf-8"))
    dataset = _map_records(
        raw=records,
        sample_fields=sample_fields,
        name=name,
        location=source,
        metadata={"source_type": "json", "path": source},
    )
    samples = list(dataset)
    if shuffle:
        random.Random(seed).shuffle(samples)
    if isinstance(limit, int) and not isinstance(limit, bool):
        samples = samples[:limit]
    return Dataset(samples, name=dataset.name, location=dataset.location, metadata=dataset.metadata)


def load_json_dataset(
    file_path: str,
    eval_name: str,
    sample_fields: Callable[[dict[str, Any]], Sample],
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
    *,
    csv_file: str,
    sample_fields: Callable[[dict[str, Any]], Sample] | None = None,
    name: str | None = None,
) -> Dataset:
    """Materialize a source-contained CSV dataset."""
    path = _resolve_source_path(csv_file)
    with path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    return _map_records(
        raw=records,
        sample_fields=sample_fields,
        name=name,
        location=csv_file,
        metadata={"source_type": "csv", "path": csv_file},
    )


def _map_records(
    *,
    raw: object,
    sample_fields: Callable[[dict[str, Any]], Sample] | None,
    name: str | None,
    location: str,
    metadata: dict[str, Any],
) -> Dataset:
    if isinstance(raw, Mapping):
        raise TypeError("Dataset loader returned a mapping; pass a concrete split or iterable of records.")
    if not isinstance(raw, Iterable):
        raise TypeError(f"Dataset loader returned unsupported type '{type(raw).__name__}'.")
    samples = []
    for record in raw:
        if isinstance(record, Sample):
            sample = record
        elif isinstance(record, Mapping):
            item = dict(record)
            sample = sample_fields(item) if sample_fields else Sample(**item)
        else:
            raise TypeError(f"Dataset record has unsupported type '{type(record).__name__}'.")
        if not isinstance(sample, Sample):
            raise TypeError("Dataset record mapper must return inspect_ai.dataset.Sample.")
        samples.append(sample)
    return Dataset(samples, name=name, location=location, metadata=metadata)


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
            "choice": choice,
            "includes": includes,
            "match": match,
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
            "chain_of_thought": chain_of_thought,
            "generate": generate,
            "multiple_choice": multiple_choice,
            "solver": solver,
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
