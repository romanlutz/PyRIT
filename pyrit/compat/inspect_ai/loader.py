# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Contained source loading and ARC graph conversion for the Inspect facade."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from pyrit.compat.inspect_ai.facade import (
    DatasetLoader,
    TaskFactory,
    activate_facade_context,
    build_compatibility_modules,
    deactivate_facade_context,
    hf_dataset,
)
from pyrit.compat.inspect_ai.inventory import InspectApiInventory, inventory_inspect_api_usage
from pyrit.compat.inspect_ai.profile import (
    PINNED_INSPECT_EVALS_PROFILE,
    InspectCompatibilityProfile,
    InspectProfileMismatchError,
    UnsupportedInspectFeatureError,
    resolve_profile,
)
from pyrit.compat.inspect_ai.types import (
    ChatMessage,
    ContentImage,
    ContentText,
    Dataset,
    Sample,
    ScorerSpec,
    SolverSpec,
    Task,
)
from pyrit.executor.capability import CapabilityLimits, CapabilitySource
from pyrit.prompt_target import TargetRequestOptions
from pyrit.scenario.capability_suite.manifest import (
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CaseMessageManifest,
    CaseScorerManifest,
    LocalSandboxProviderManifestConfig,
    RunPolicyManifest,
    SuiteProvenance,
)
from pyrit.scenario.capability_suite.serialization import load_manifest_json

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from pyrit.executor.capability import (
        CapabilityRequestOptionsFactory,
        ToolDeclaration,
        ToolExecutionPolicy,
    )
    from pyrit.prompt_target import PromptTarget
    from pyrit.scenario.capability_suite import CapabilitySuiteRunResult


_IMPORT_LOCK = threading.RLock()
_MODULE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_TASK_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class InspectCompatibilityReport:
    """Compatibility facts produced while loading one pinned task."""

    profile_id: str
    inspect_api_profile: str
    source_revision: str
    detected_source_revision: str | None
    source_revision_verified: bool
    task_name: str
    task_parameters: dict[str, object]
    api_inventory: InspectApiInventory
    capabilities: dict[str, bool]
    limitations: tuple[str, ...]


@dataclass(frozen=True)
class LoadedInspectEval:
    """A loaded compatibility task and its safe native suite."""

    task: Task
    suite: CapabilitySuiteManifest
    report: InspectCompatibilityReport


@dataclass(frozen=True)
class InspectEvalRun:
    """A loaded compatibility task plus its native execution result."""

    loaded: LoadedInspectEval
    result: CapabilitySuiteRunResult


class _UnsupportedInspectModuleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, *, known_modules: frozenset[str], profile: InspectCompatibilityProfile) -> None:
        self._known_modules = known_modules
        self._profile = profile

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname.startswith("inspect_ai.") and fullname not in self._known_modules:
            raise UnsupportedInspectFeatureError(symbol=fullname, source_profile=self._profile.profile_id)
        return None


class _ContainedSourceFinder(importlib.abc.MetaPathFinder):
    def __init__(self, *, package_root: Path, profile: InspectCompatibilityProfile) -> None:
        self._package_root = package_root
        self._profile = profile

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del target
        if fullname != "inspect_evals" and not fullname.startswith("inspect_evals."):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            return spec
        origin = Path(spec.origin).resolve()
        if origin != self._package_root and self._package_root not in origin.parents:
            raise ValueError(f"Imported source module '{fullname}' escapes source root: '{origin}'.")
        if origin.suffix == ".py":
            inventory = inventory_inspect_api_usage(
                source_root=self._package_root.parent,
                profile=self._profile,
                source_files=(origin,),
            )
            if inventory.unsupported_symbols:
                raise UnsupportedInspectFeatureError(
                    symbol=inventory.unsupported_symbols[0],
                    source_profile=self._profile.profile_id,
                    remediation="Add only explicitly implemented symbols to the pinned compatibility profile.",
                )
        return spec


class _NoToolRequestOptionsFactory:
    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        del execution_policy
        if declarations:
            raise UnsupportedInspectFeatureError(
                symbol="inspect_ai.tool runtime",
                source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
            )
        return TargetRequestOptions()


def load_inspect_eval(
    *,
    source_root: Path,
    task_spec: str,
    task_parameters: Mapping[str, object] | None = None,
    profile_id: str = PINNED_INSPECT_EVALS_PROFILE.profile_id,
    dataset_loader: DatasetLoader | None = None,
    allow_network: bool = False,
    verify_source_revision: bool = True,
    worker_timeout_seconds: float = 300.0,
    source_verification_timeout_seconds: float = 120.0,
    case_timeout_seconds: float = 300.0,
) -> LoadedInspectEval:
    """
    Execute one trusted pinned task factory and convert its graph to a native suite.

    The source is arbitrary Python and executes with the current process identity.
    Only source/profile/import containment is provided; this is not a sandbox.

    Returns:
        LoadedInspectEval: The compatibility graph, report, and native suite.

    Raises:
        InspectProfileMismatchError: If the profile or checkout revision is not pinned.
        UnsupportedInspectFeatureError: If source requests an unsupported Inspect API.
    """
    profile = resolve_profile(profile_id)
    _validate_positive_timeout(
        name="source_verification_timeout_seconds",
        value=source_verification_timeout_seconds,
    )
    _validate_positive_timeout(name="case_timeout_seconds", value=case_timeout_seconds)
    resolved_root, _, package_root = _resolve_source_layout(source_root)
    module_name, task_name, module_path = _resolve_task_spec(
        task_spec=task_spec,
        package_root=package_root,
    )
    inventory = inventory_inspect_api_usage(
        source_root=resolved_root,
        profile=profile,
        source_files=(module_path,),
    )
    if inventory.unsupported_symbols:
        raise UnsupportedInspectFeatureError(
            symbol=inventory.unsupported_symbols[0],
            source_profile=profile.profile_id,
            remediation="Run inventory_inspect_api_usage() in CI and add only explicitly implemented symbols.",
        )
    checked_revision = _git_revision(
        resolved_root,
        timeout_seconds=source_verification_timeout_seconds,
    )
    revision_verified = False
    if verify_source_revision:
        if checked_revision != profile.inspect_evals_revision:
            raise InspectProfileMismatchError(
                f"Source checkout revision '{checked_revision or 'unverified'}' does not match pinned revision "
                f"'{profile.inspect_evals_revision}'."
            )
        _verify_clean_checkout(
            resolved_root,
            timeout_seconds=source_verification_timeout_seconds,
        )
        revision_verified = True
    records = _materialize_injected_arc_records(
        task_name=task_name,
        dataset_loader=dataset_loader,
    )
    return _run_worker(
        source_root=resolved_root,
        task_spec=task_spec,
        task_parameters=dict(task_parameters or {}),
        profile=profile,
        allow_network=allow_network,
        verify_source_revision=verify_source_revision,
        records=records,
        inventory=inventory,
        revision_verified=revision_verified,
        worker_timeout_seconds=worker_timeout_seconds,
        source_verification_timeout_seconds=source_verification_timeout_seconds,
        case_timeout_seconds=case_timeout_seconds,
    )


def _load_inspect_eval_in_process(
    *,
    source_root: Path,
    task_spec: str,
    task_parameters: Mapping[str, object] | None = None,
    profile_id: str = PINNED_INSPECT_EVALS_PROFILE.profile_id,
    dataset_records: list[dict[str, Any]] | None = None,
    allow_network: bool = False,
    verify_source_revision: bool = True,
    source_verification_timeout_seconds: float = 120.0,
    case_timeout_seconds: float = 300.0,
) -> LoadedInspectEval:
    profile = resolve_profile(profile_id)
    _validate_positive_timeout(
        name="source_verification_timeout_seconds",
        value=source_verification_timeout_seconds,
    )
    _validate_positive_timeout(name="case_timeout_seconds", value=case_timeout_seconds)
    resolved_root, package_parent, package_root = _resolve_source_layout(source_root)
    module_name, task_name, module_path = _resolve_task_spec(task_spec=task_spec, package_root=package_root)
    inventory = inventory_inspect_api_usage(
        source_root=resolved_root,
        profile=profile,
        source_files=(module_path,),
    )
    if inventory.unsupported_symbols:
        raise UnsupportedInspectFeatureError(
            symbol=inventory.unsupported_symbols[0],
            source_profile=profile.profile_id,
            remediation="Run inventory_inspect_api_usage() in CI and add only explicitly implemented symbols.",
        )
    checked_revision = _git_revision(
        resolved_root,
        timeout_seconds=source_verification_timeout_seconds,
    )
    revision_verified = False
    if verify_source_revision:
        if checked_revision != profile.inspect_evals_revision:
            raise InspectProfileMismatchError(
                f"Source checkout revision '{checked_revision or 'unverified'}' does not match pinned revision "
                f"'{profile.inspect_evals_revision}'."
            )
        _verify_clean_checkout(
            resolved_root,
            timeout_seconds=source_verification_timeout_seconds,
        )
        revision_verified = True
    parameters = dict(task_parameters or {})
    loader = (lambda *args, **kwargs: dataset_records) if dataset_records is not None else None
    task_graph = _execute_task_factory(
        package_parent=package_parent,
        package_root=package_root,
        module_name=module_name,
        task_name=task_name,
        parameters=parameters,
        dataset_loader=loader,
        allow_network=allow_network,
        profile=profile,
    )
    suite = _compile_arc_suite(
        task=task_graph,
        task_name=task_name,
        parameters=parameters,
        profile=profile,
        checked_revision=checked_revision,
        revision_verified=revision_verified,
        case_timeout_seconds=case_timeout_seconds,
    )
    report = InspectCompatibilityReport(
        profile_id=profile.profile_id,
        inspect_api_profile=profile.inspect_api_profile,
        source_revision=profile.inspect_evals_revision,
        detected_source_revision=checked_revision,
        source_revision_verified=revision_verified,
        task_name=task_name,
        task_parameters=parameters,
        api_inventory=inventory,
        capabilities=profile.capability_report(),
        limitations=(
            "Source loading executes arbitrary trusted Python in a dedicated compatibility worker process.",
            "Only ARC multiple_choice plus choice execution is implemented.",
            "Model calls route only through the injected PyRIT PromptTarget.",
            "AWS, Bedrock, SageMaker, EC2, GCP, Modal, Daytona, and other non-Azure providers are excluded.",
            "Agents, sandbox tools, custom scorer execution, stores, hooks, and EvalLog parity are excluded.",
            f"Native case execution is bounded to {case_timeout_seconds:g} seconds.",
        ),
    )
    return LoadedInspectEval(task=task_graph, suite=suite, report=report)


async def run_inspect_eval_async(
    *,
    source_root: Path,
    task_spec: str,
    target: PromptTarget,
    task_parameters: Mapping[str, object] | None = None,
    profile_id: str = PINNED_INSPECT_EVALS_PROFILE.profile_id,
    dataset_loader: DatasetLoader | None = None,
    allow_network: bool = False,
    verify_source_revision: bool = True,
    request_options_factory: CapabilityRequestOptionsFactory | None = None,
    worker_timeout_seconds: float = 300.0,
    source_verification_timeout_seconds: float = 120.0,
    case_timeout_seconds: float = 300.0,
) -> InspectEvalRun:
    """
    Load a pinned task and run its native suite through an injected PyRIT target.

    Returns:
        InspectEvalRun: The loaded suite and native execution result.
    """
    import asyncio

    from pyrit.compat.inspect_ai.scorer import InspectChoiceScorer
    from pyrit.scenario.capability_suite import (
        CapabilitySuiteRunner,
        ResultOnlyScorerAdapter,
        build_default_sandbox_provider_registry,
        build_default_scorer_registry,
    )

    loaded = await asyncio.to_thread(
        load_inspect_eval,
        source_root=source_root,
        task_spec=task_spec,
        task_parameters=task_parameters,
        profile_id=profile_id,
        dataset_loader=dataset_loader,
        allow_network=allow_network,
        verify_source_revision=verify_source_revision,
        worker_timeout_seconds=worker_timeout_seconds,
        source_verification_timeout_seconds=source_verification_timeout_seconds,
        case_timeout_seconds=case_timeout_seconds,
    )
    scorer_registry = build_default_scorer_registry()
    scorer_registry.register(
        kind="inspect_choice",
        factory=lambda config: ResultOnlyScorerAdapter(scorer=InspectChoiceScorer.from_config(config)),
    )
    result = await CapabilitySuiteRunner(
        manifest=loaded.suite,
        target=target,
        request_options_factory=request_options_factory or _NoToolRequestOptionsFactory(),
        sandbox_provider_registry=build_default_sandbox_provider_registry(),
        scorer_registry=scorer_registry,
    ).run_async()
    return InspectEvalRun(loaded=loaded, result=result)


def _resolve_source_layout(source_root: Path) -> tuple[Path, Path, Path]:
    root = source_root.resolve()
    candidates = (root / "src" / "inspect_evals", root / "inspect_evals")
    candidate = next((item for item in candidates if item.is_dir()), None)
    if candidate is None:
        raise ValueError(f"No source-contained 'inspect_evals' package exists beneath '{root}'.")
    package_root = candidate.resolve()
    if package_root != root and root not in package_root.parents:
        raise ValueError(f"The 'inspect_evals' package escapes trusted source root '{root}'.")
    return root, package_root.parent, package_root


def _resolve_task_spec(*, task_spec: str, package_root: Path) -> tuple[str, str, Path]:
    if "@" in task_spec:
        module_part, task_name = task_spec.rsplit("@", 1)
    elif ":" in task_spec:
        module_part, task_name = task_spec.rsplit(":", 1)
    else:
        raise ValueError("Task spec must use '<module-or-relative-file>@<task-name>'.")
    if not _TASK_PATTERN.fullmatch(task_name):
        raise ValueError(f"Unsafe Inspect task name '{task_name}'.")
    normalized = module_part.replace("\\", "/")
    if normalized.endswith(".py"):
        relative = Path(normalized)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe Inspect task module path '{module_part}'.")
        module_path = (package_root / relative).resolve()
        module_suffix = ".".join(relative.with_suffix("").parts)
    else:
        module_suffix = normalized.removeprefix("inspect_evals.").replace("/", ".")
        if not _MODULE_PATTERN.fullmatch(module_suffix):
            raise ValueError(f"Unsafe Inspect task module '{module_part}'.")
        module_path = (package_root / Path(*module_suffix.split("."))).with_suffix(".py").resolve()
    if module_path != package_root and package_root not in module_path.parents:
        raise ValueError(f"Inspect task module '{module_part}' escapes the source package.")
    if not module_path.is_file():
        raise ValueError(f"Inspect task module '{module_path}' does not exist.")
    return f"inspect_evals.{module_suffix}", task_name, module_path


def _execute_task_factory(
    *,
    package_parent: Path,
    package_root: Path,
    module_name: str,
    task_name: str,
    parameters: dict[str, object],
    dataset_loader: DatasetLoader | None,
    allow_network: bool,
    profile: InspectCompatibilityProfile,
) -> Task:
    registry: dict[str, TaskFactory] = {}
    with _isolated_import_scope(
        package_parent=package_parent,
        package_root=package_root,
        task_registry=registry,
        dataset_loader=dataset_loader,
        allow_network=allow_network,
        profile=profile,
    ):
        importlib.import_module(module_name)
        factory = registry.get(task_name)
        if factory is None:
            available = ", ".join(sorted(registry)) or "<none>"
            raise ValueError(f"Inspect task '{task_name}' was not registered. Available tasks: {available}.")
        try:
            inspect.signature(factory).bind(**parameters)
        except TypeError as error:
            raise ValueError(f"Invalid parameters for Inspect task '{task_name}': {error}") from error
        task_graph = factory(**parameters)
    if not isinstance(task_graph, Task):
        raise TypeError(f"Inspect task '{task_name}' returned '{type(task_graph).__name__}', expected Task.")
    return task_graph


def _materialize_injected_arc_records(
    *,
    task_name: str,
    dataset_loader: DatasetLoader | None,
) -> list[dict[str, Any]] | None:
    if dataset_loader is None:
        return None
    dataset_names = {"arc_easy": "ARC-Easy", "arc_challenge": "ARC-Challenge"}
    dataset_name = dataset_names.get(task_name)
    if dataset_name is None:
        raise UnsupportedInspectFeatureError(
            symbol=f"injected dataset loading for task '{task_name}'",
            source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
        )
    raw = dataset_loader(
        "allenai/ai2_arc",
        dataset_name,
        split="test",
        revision="210d026faf9955653af8916fad021475a3f00453",
        trust_remote_code=False,
    )
    if isinstance(raw, Mapping):
        raise TypeError("Injected ARC dataset loader must return an iterable of records, not a mapping.")
    if isinstance(raw, (str, bytes, Mapping)) or not isinstance(raw, Iterable):
        raise TypeError("Injected ARC dataset loader must return an iterable of mapping records.")
    records = list(raw)
    if not all(isinstance(record, Mapping) for record in records):
        raise TypeError("Injected ARC dataset loader returned a non-mapping record.")
    return [dict(cast("Mapping[Any, Any]", record)) for record in records]


def _run_worker(
    *,
    source_root: Path,
    task_spec: str,
    task_parameters: dict[str, object],
    profile: InspectCompatibilityProfile,
    allow_network: bool,
    verify_source_revision: bool,
    records: list[dict[str, Any]] | None,
    inventory: InspectApiInventory,
    revision_verified: bool,
    worker_timeout_seconds: float,
    source_verification_timeout_seconds: float,
    case_timeout_seconds: float,
) -> LoadedInspectEval:
    if worker_timeout_seconds <= 0:
        raise ValueError("worker_timeout_seconds must be greater than zero.")
    request = {
        "source_root": str(source_root),
        "task_spec": task_spec,
        "task_parameters": task_parameters,
        "profile_id": profile.profile_id,
        "dataset_records": records,
        "allow_network": allow_network,
        "verify_source_revision": verify_source_revision,
        "source_verification_timeout_seconds": source_verification_timeout_seconds,
        "case_timeout_seconds": case_timeout_seconds,
    }
    with tempfile.TemporaryDirectory(prefix="pyrit-inspect-compat-") as directory:
        request_path = Path(directory) / "request.json"
        response_path = Path(directory) / "response.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        process_environment = os.environ.copy()
        process_environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pyrit.compat.inspect_ai.worker",
                    str(request_path),
                    str(response_path),
                ],
                capture_output=True,
                check=False,
                encoding="utf-8",
                env=process_environment,
                timeout=worker_timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"Inspect compatibility worker exceeded {worker_timeout_seconds:g} seconds.") from error
        if not response_path.is_file():
            raise RuntimeError(
                "Inspect compatibility worker failed without a response: "
                f"{completed.stderr.strip() or completed.stdout.strip() or completed.returncode}"
            )
        response = json.loads(response_path.read_text(encoding="utf-8"))
    if not response["ok"]:
        _raise_worker_error(cast("dict[str, Any]", response["error"]))
    loaded = _deserialize_loaded(cast("dict[str, Any]", response["loaded"]))
    report = InspectCompatibilityReport(
        profile_id=loaded.report.profile_id,
        inspect_api_profile=loaded.report.inspect_api_profile,
        source_revision=loaded.report.source_revision,
        detected_source_revision=loaded.report.detected_source_revision,
        source_revision_verified=revision_verified,
        task_name=loaded.report.task_name,
        task_parameters=loaded.report.task_parameters,
        api_inventory=inventory,
        capabilities=loaded.report.capabilities,
        limitations=loaded.report.limitations,
    )
    return LoadedInspectEval(task=loaded.task, suite=loaded.suite, report=report)


def _raise_worker_error(error: dict[str, Any]) -> None:
    error_type = error["type"]
    if error_type == "UnsupportedInspectFeatureError":
        raise UnsupportedInspectFeatureError(
            symbol=error["symbol"],
            source_profile=error["source_profile"],
            remediation=error["remediation"],
        )
    if error_type == "InspectProfileMismatchError":
        raise InspectProfileMismatchError(error["message"])
    if error_type == "ValueError":
        raise ValueError(error["message"])
    if error_type == "TypeError":
        raise TypeError(error["message"])
    if error_type == "RuntimeError":
        raise RuntimeError(error["message"])
    raise RuntimeError(f"Inspect compatibility worker raised {error_type}: {error['message']}")


def _serialize_loaded(loaded: LoadedInspectEval) -> dict[str, Any]:
    return {
        "task": _serialize_task(loaded.task),
        "suite": loaded.suite.model_dump(mode="json"),
        "report": {
            "profile_id": loaded.report.profile_id,
            "inspect_api_profile": loaded.report.inspect_api_profile,
            "source_revision": loaded.report.source_revision,
            "detected_source_revision": loaded.report.detected_source_revision,
            "source_revision_verified": loaded.report.source_revision_verified,
            "task_name": loaded.report.task_name,
            "task_parameters": loaded.report.task_parameters,
            "api_inventory": loaded.report.api_inventory.to_dict(),
            "capabilities": loaded.report.capabilities,
            "limitations": list(loaded.report.limitations),
        },
    }


def _deserialize_loaded(data: dict[str, Any]) -> LoadedInspectEval:
    report_data = data["report"]
    inventory_data = report_data["api_inventory"]
    from pyrit.compat.inspect_ai.inventory import InspectApiUsage

    inventory = InspectApiInventory(
        source_root=inventory_data["source_root"],
        profile_id=inventory_data["profile_id"],
        usages=tuple(InspectApiUsage(**usage) for usage in inventory_data["usages"]),
        unsupported_symbols=tuple(inventory_data["unsupported_symbols"]),
    )
    report = InspectCompatibilityReport(
        profile_id=report_data["profile_id"],
        inspect_api_profile=report_data["inspect_api_profile"],
        source_revision=report_data["source_revision"],
        detected_source_revision=report_data["detected_source_revision"],
        source_revision_verified=report_data["source_revision_verified"],
        task_name=report_data["task_name"],
        task_parameters=report_data["task_parameters"],
        api_inventory=inventory,
        capabilities=report_data["capabilities"],
        limitations=tuple(report_data["limitations"]),
    )
    return LoadedInspectEval(
        task=_deserialize_task(data["task"]),
        suite=load_manifest_json(data["suite"]),
        report=report,
    )


def _serialize_task(task: Task) -> dict[str, Any]:
    dataset = task.dataset if isinstance(task.dataset, Dataset) else Dataset(task.dataset)
    return {
        "dataset": {
            "samples": [_serialize_sample(sample) for sample in dataset],
            "name": dataset.name,
            "location": dataset.location,
            "metadata": dataset.metadata,
        },
        "solver": [_serialize_spec(spec) for spec in _spec_sequence(task.solver, expected_type=SolverSpec)],
        "scorer": [_serialize_spec(spec) for spec in _spec_sequence(task.scorer, expected_type=ScorerSpec)],
        "config": vars(task.config) if task.config else None,
        "epochs": task.epochs,
        "fail_on_error": task.fail_on_error,
        "message_limit": task.message_limit,
        "token_limit": task.token_limit,
        "time_limit": task.time_limit,
        "working_limit": task.working_limit,
        "version": task.version,
        "metadata": task.metadata,
    }


def _deserialize_task(data: dict[str, Any]) -> Task:
    from pyrit.compat.inspect_ai.types import GenerateConfig

    dataset_data = data["dataset"]
    dataset = Dataset(
        (_deserialize_sample(sample) for sample in dataset_data["samples"]),
        name=dataset_data["name"],
        location=dataset_data["location"],
        metadata=dataset_data["metadata"],
    )
    solvers = tuple(SolverSpec(**spec) for spec in data["solver"])
    scorers = tuple(ScorerSpec(**spec) for spec in data["scorer"])
    return Task(
        dataset=dataset,
        solver=solvers[0] if len(solvers) == 1 else solvers,
        scorer=scorers[0] if len(scorers) == 1 else scorers,
        config=GenerateConfig(**data["config"]) if data["config"] else None,
        epochs=data["epochs"],
        fail_on_error=data["fail_on_error"],
        message_limit=data["message_limit"],
        token_limit=data["token_limit"],
        time_limit=data["time_limit"],
        working_limit=data["working_limit"],
        version=data["version"],
        metadata=data["metadata"],
    )


def _serialize_sample(sample: Sample) -> dict[str, Any]:
    input_data = (
        sample.input
        if isinstance(sample.input, str)
        else [
            {
                "role": message.role,
                "content": _message_content(message),
                "name": message.name,
                "source": message.source,
                "metadata": message.metadata,
            }
            for message in sample.input
        ]
    )
    return {
        "input": input_data,
        "target": sample.target,
        "choices": sample.choices,
        "id": sample.id,
        "metadata": sample.metadata,
        "sandbox": sample.sandbox,
        "setup": sample.setup,
        "files": sample.files,
    }


def _deserialize_sample(data: dict[str, Any]) -> Sample:
    input_data = data["input"]
    if isinstance(input_data, list):
        input_data = [ChatMessage(**message) for message in input_data]
    return Sample(
        input=input_data,
        target=data["target"],
        choices=data["choices"],
        id=data["id"],
        metadata=data["metadata"],
        sandbox=data["sandbox"],
        setup=data["setup"],
        files=data["files"],
    )


def _serialize_spec(spec: SolverSpec | ScorerSpec) -> dict[str, Any]:
    return {"name": spec.name, "config": spec.config}


@contextmanager
def _isolated_import_scope(
    *,
    package_parent: Path,
    package_root: Path,
    task_registry: dict[str, TaskFactory],
    dataset_loader: DatasetLoader | None,
    allow_network: bool,
    profile: InspectCompatibilityProfile,
) -> Iterator[None]:
    with _IMPORT_LOCK:
        compat_modules = build_compatibility_modules(profile=profile)
        source_shims = _build_pinned_source_shims()
        saved_compat = _take_modules("inspect_ai")
        saved_source = _take_modules("inspect_evals")
        finder = _UnsupportedInspectModuleFinder(
            known_modules=frozenset(compat_modules),
            profile=profile,
        )
        source_finder = _ContainedSourceFinder(package_root=package_root, profile=profile)
        path_value = str(package_parent)
        sys.path.insert(0, path_value)
        sys.meta_path.insert(0, finder)
        sys.meta_path.insert(0, source_finder)
        sys.modules.update(compat_modules)
        sys.modules.update(source_shims)
        tokens = activate_facade_context(
            task_registry=task_registry,
            dataset_loader=dataset_loader,
            allow_network=allow_network,
            source_root=package_root,
            profile=profile,
        )
        try:
            yield
            _validate_source_modules(package_root=package_root)
        finally:
            deactivate_facade_context(tokens)
            if finder in sys.meta_path:
                sys.meta_path.remove(finder)
            if source_finder in sys.meta_path:
                sys.meta_path.remove(source_finder)
            if path_value in sys.path:
                sys.path.remove(path_value)
            _drop_modules("inspect_ai")
            _drop_modules("inspect_evals")
            sys.modules.update(saved_compat)
            sys.modules.update(saved_source)


def _build_pinned_source_shims() -> dict[str, ModuleType]:
    utils = ModuleType("inspect_evals.utils")
    utils.__path__ = []
    huggingface = ModuleType("inspect_evals.utils.huggingface")
    huggingface.__dict__["hf_dataset"] = hf_dataset
    utils.__dict__["huggingface"] = huggingface
    return {
        "inspect_evals.utils": utils,
        "inspect_evals.utils.huggingface": huggingface,
    }


def _take_modules(prefix: str) -> dict[str, ModuleType]:
    saved = {
        name: module for name, module in tuple(sys.modules.items()) if name == prefix or name.startswith(f"{prefix}.")
    }
    for name in saved:
        del sys.modules[name]
    return saved


def _drop_modules(prefix: str) -> None:
    for name in tuple(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            del sys.modules[name]


def _validate_source_modules(*, package_root: Path) -> None:
    for name, module in tuple(sys.modules.items()):
        if name != "inspect_evals" and not name.startswith("inspect_evals."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved = Path(module_file).resolve()
        if resolved != package_root and package_root not in resolved.parents:
            raise ValueError(f"Imported source module '{name}' escaped source root: '{resolved}'.")


def _validate_positive_timeout(*, name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")


def _git_revision(source_root: Path, *, timeout_seconds: float) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout.strip() if completed.returncode == 0 else None


def _verify_clean_checkout(source_root: Path, *, timeout_seconds: float) -> None:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(source_root),
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--ignored=matching",
            ],
            capture_output=True,
            check=False,
            encoding="utf-8",
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise InspectProfileMismatchError(f"Unable to verify checkout cleanliness for '{source_root}'.") from error
    if completed.returncode != 0:
        raise InspectProfileMismatchError(f"Unable to verify checkout cleanliness for '{source_root}'.")
    changes = completed.stdout.strip()
    if changes:
        raise InspectProfileMismatchError(
            "Pinned Inspect-evals checkout contains modified or untracked files and cannot be revision-verified: "
            + "; ".join(changes.splitlines())
        )


def _compile_arc_suite(
    *,
    task: Task,
    task_name: str,
    parameters: dict[str, object],
    profile: InspectCompatibilityProfile,
    checked_revision: str | None,
    revision_verified: bool,
    case_timeout_seconds: float,
) -> CapabilitySuiteManifest:
    _reject_unsupported_task_options(task=task, profile=profile)
    solvers = _spec_sequence(task.solver, expected_type=SolverSpec)
    scorers = _spec_sequence(task.scorer, expected_type=ScorerSpec)
    if len(solvers) != 1 or solvers[0].name != "multiple_choice":
        symbol = f"inspect_ai.solver.{solvers[0].name}" if solvers else "inspect_ai.solver.<empty>"
        raise UnsupportedInspectFeatureError(symbol=symbol, source_profile=profile.profile_id)
    if len(scorers) != 1 or scorers[0].name != "choice":
        symbol = f"inspect_ai.scorer.{scorers[0].name}" if scorers else "inspect_ai.scorer.<empty>"
        raise UnsupportedInspectFeatureError(symbol=symbol, source_profile=profile.profile_id)
    if task.sandbox is not None:
        raise UnsupportedInspectFeatureError(symbol="inspect_ai.sandbox runtime", source_profile=profile.profile_id)
    dataset = task.dataset if isinstance(task.dataset, Dataset) else Dataset(task.dataset)
    cases = tuple(
        _compile_arc_sample(
            sample=sample,
            index=index,
            task=task,
            solver=solvers[0],
            dataset_metadata=dataset.metadata,
            case_timeout_seconds=case_timeout_seconds,
        )
        for index, sample in enumerate(dataset)
    )
    if not cases:
        raise ValueError(f"Inspect task '{task_name}' produced an empty dataset.")
    return CapabilitySuiteManifest(
        suite_id=_safe_identifier(f"inspect-compat-{task_name}"),
        name=f"Inspect compatibility: {task_name}",
        description="Unchanged pinned Inspect-evals task factory compiled to the native PyRIT capability runtime.",
        provenance=SuiteProvenance(
            source="UKGovernmentBEIS/inspect_evals compatibility loader",
            source_id=task_name,
            repository="https://github.com/UKGovernmentBEIS/inspect_evals",
            revision=checked_revision,
            license="MIT; dataset licenses remain upstream-specific",
            metadata={
                "compatibility_profile": profile.profile_id,
                "inspect_api_profile": profile.inspect_api_profile,
                "expected_revision": profile.inspect_evals_revision,
                "detected_revision": checked_revision,
                "source_revision_verified": revision_verified,
                "case_timeout_seconds": case_timeout_seconds,
                "task_parameters": _json_mapping(parameters),
                "dataset": _json_mapping(dataset.metadata),
            },
        ),
        sandbox_provider=LocalSandboxProviderManifestConfig(),
        run_policy=RunPolicyManifest(epochs=task.epochs or 1),
        cases=cases,
        tags=("inspect-evals", "compatibility", "arc", "native"),
        metadata={
            "compatibility_profile": profile.profile_id,
            "expected_revision": profile.inspect_evals_revision,
            "detected_revision": checked_revision,
            "source_revision_verified": revision_verified,
            "case_timeout_seconds": case_timeout_seconds,
            "task_version": _json_scalar(task.version),
            "task_metadata": _json_mapping(task.metadata),
            "solver": {"name": solvers[0].name, "config": _json_mapping(solvers[0].config)},
            "scorer": {"name": scorers[0].name, "config": _json_mapping(scorers[0].config)},
            "dataset": _json_mapping(dataset.metadata),
        },
    )


def _reject_unsupported_task_options(*, task: Task, profile: InspectCompatibilityProfile) -> None:
    if task.config is not None:
        configured_field = next((name for name, value in vars(task.config).items() if value is not None), None)
        if configured_field is not None:
            raise UnsupportedInspectFeatureError(
                symbol=f"inspect_ai.Task.config.{configured_field}",
                source_profile=profile.profile_id,
            )
    for field_name in ("fail_on_error", "message_limit", "token_limit", "time_limit", "working_limit"):
        if getattr(task, field_name) is not None:
            raise UnsupportedInspectFeatureError(
                symbol=f"inspect_ai.Task.{field_name}",
                source_profile=profile.profile_id,
            )


def _compile_arc_sample(
    *,
    sample: Sample,
    index: int,
    task: Task,
    solver: SolverSpec,
    dataset_metadata: Mapping[str, object],
    case_timeout_seconds: float,
) -> CapabilityCaseManifest:
    for field_name, value in (
        ("sandbox", sample.sandbox),
        ("setup", sample.setup),
        ("files", sample.files),
    ):
        if value not in (None, {}, []):
            raise UnsupportedInspectFeatureError(
                symbol=f"inspect_ai.dataset.Sample.{field_name}",
                source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
            )
    if not sample.choices:
        raise ValueError(f"ARC sample '{sample.id or index}' must declare choices.")
    if isinstance(sample.target, list):
        if len(sample.target) != 1:
            raise UnsupportedInspectFeatureError(
                symbol="inspect_ai.scorer.choice multiple targets",
                source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
            )
        target = sample.target[0]
    else:
        target = sample.target
    labels = [_answer_character(offset) for offset in range(len(sample.choices))]
    messages, objective = _sample_messages(sample=sample, labels=labels, solver=solver)
    max_tokens = solver.config.get("max_tokens")
    if max_tokens is not None:
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.solver.multiple_choice(max_tokens=...)",
            source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
        )
    limits = CapabilityLimits(
        max_wall_clock_seconds=case_timeout_seconds,
    )
    case_id = _safe_identifier(str(sample.id if sample.id is not None else f"sample-{index + 1}"))
    return CapabilityCaseManifest(
        case_id=case_id,
        objective=objective,
        messages=messages,
        scorers=(
            CaseScorerManifest(
                kind="inspect_choice",
                config={
                    "expected_value": str(target).upper(),
                    "allowed_options": labels,
                },
            ),
        ),
        limits=limits,
        source=CapabilitySource(
            source_type="inspect_evals",
            source_id=str(sample.id) if sample.id is not None else None,
            metadata={
                "sample_metadata": _json_mapping(sample.metadata),
                "dataset": _json_mapping(dataset_metadata),
            },
        ),
        tags=("reasoning", "multiple-choice", "native"),
        metadata={
            "answer_label": target,
            "choices": sample.choices,
            "sample_metadata": _json_mapping(sample.metadata),
        },
    )


def _answer_character(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index - 25)


def _sample_messages(
    *,
    sample: Sample,
    labels: list[str],
    solver: SolverSpec,
) -> tuple[tuple[CaseMessageManifest, ...], str]:
    if solver.config.get("multiple_correct", False):
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.solver.multiple_choice(multiple_correct=True)",
            source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
        )
    if solver.config.get("shuffle") not in (None, False):
        raise UnsupportedInspectFeatureError(
            symbol="inspect_ai.solver.multiple_choice(shuffle=True)",
            source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
        )
    choice_text = "\n".join(f"{label}) {choice}" for label, choice in zip(labels, sample.choices or [], strict=True))
    letters = ",".join(labels)
    template = solver.config.get("template")
    if template is not None and not isinstance(template, str):
        raise TypeError("multiple_choice template must be a string or None.")
    if template is not None and ("{question}" not in template or "{choices}" not in template):
        raise ValueError("The template must contain '{question}' and '{choices}' placeholders for string substitution.")
    if template is None:
        if solver.config.get("cot", False):
            template = (
                "Answer the following multiple choice question. The last line of your response should be of the "
                "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. "
                "Think step by step before answering.\n\n{question}\n\n{choices}"
            )
        else:
            template = (
                "Answer the following multiple choice question. The entire content of your response should be of "
                "the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}.\n\n"
                "{question}\n\n{choices}"
            )

    def render(question: str) -> str:
        return template.format(question=question, choices=choice_text, letters=letters)

    if isinstance(sample.input, str):
        return (CaseMessageManifest(role="user", content=render(sample.input)),), sample.input
    if not sample.input:
        raise ValueError("Inspect sample input messages must not be empty.")
    user_indexes = [index for index, message in enumerate(sample.input) if message.role == "user"]
    if not user_indexes:
        raise ValueError("Inspect multiple_choice requires at least one user message.")
    user_prompt_index = user_indexes[-1]
    objective = _message_content(sample.input[user_prompt_index])
    messages = []
    for index, message in enumerate(sample.input):
        if message.name is not None:
            raise UnsupportedInspectFeatureError(
                symbol="inspect_ai.model.ChatMessage.name",
                source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
            )
        if message.source is not None:
            raise UnsupportedInspectFeatureError(
                symbol="inspect_ai.model.ChatMessage.source",
                source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
            )
        content = _message_content(message)
        if index == user_prompt_index:
            content = render(content)
        role = cast("Any", message.role)
        messages.append(CaseMessageManifest(role=role, content=content, metadata=_json_mapping(message.metadata)))
    return tuple(messages), objective


def _message_content(message: ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    parts = []
    for content in message.content:
        if isinstance(content, ContentText):
            parts.append(content.text)
        elif isinstance(content, ContentImage):
            raise UnsupportedInspectFeatureError(
                symbol="inspect_ai.model.ContentImage execution",
                source_profile=PINNED_INSPECT_EVALS_PROFILE.profile_id,
            )
    return "\n".join(parts)


def _spec_sequence(value: object, *, expected_type: type[Any]) -> tuple[Any, ...]:
    values = tuple(value) if isinstance(value, (list, tuple)) else (value,)
    if not all(isinstance(item, expected_type) for item in values):
        raise TypeError(f"Expected only {expected_type.__name__} graph nodes.")
    return values


def _safe_identifier(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    if not normalized:
        raise ValueError(f"Value '{value}' cannot be normalized to a safe identifier.")
    return normalized


def _json_mapping(value: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _json_value(item) for key, item in value.items()}


def _json_value(value: object) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return _json_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"Compatibility metadata value '{type(value).__name__}' is not JSON serializable.")


def _json_scalar(value: object) -> Any:
    return _json_value(value)
