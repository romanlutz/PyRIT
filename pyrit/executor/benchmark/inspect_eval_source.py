# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Trusted, harness-neutral source resolution for the bounded Inspect Eval pilot."""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import CodeType
from typing import TYPE_CHECKING, Literal, TypeAlias
from uuid import uuid4

from inspect_ai import Task
from inspect_ai._util.registry import is_registry_object, registry_info, registry_params, registry_unqualified_name
from inspect_ai.dataset import MemoryDataset
from inspect_ai.util import ComposeConfig, SandboxEnvironmentSpec
from pydantic import BaseModel, ConfigDict, Field

from pyrit.models import EvalCaseRef, EvalPackageRef, EvalSourceKind, SeedObjective, config_hash

if TYPE_CHECKING:
    from collections.abc import Callable

    from inspect_ai.dataset import Sample


class _EvalManifest(BaseModel):
    """Only source identity and allowed input surfaces, never credentials or model routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[1] = 1
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    task_file: str
    factory: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    files_sha256: dict[str, str] = Field(min_length=1, max_length=32)
    task_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    task_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    sample_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
    scorer_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_fingerprints: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    cleanup_name: str | None = None
    health_command: tuple[str, ...] = Field(min_length=1, max_length=8)
    allow_initial_input_override: bool = False


class _PinnedSample(BaseModel):
    """One original text Sample in an exact, ordered Task inventory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    sample_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sample_target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _PinnedTask(BaseModel):
    """The original scorer, setup, cleanup and all selected Samples for one Task."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    task_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    task_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    scorer_name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,127}$")
    scorer_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_fingerprints: tuple[str, ...] = Field(default_factory=tuple, max_length=8)
    cleanup_name: str | None = None
    samples: tuple[_PinnedSample, ...] = Field(min_length=1, max_length=16)


class _EvalManifestV2(BaseModel):
    """Opt-in finite trusted-local inventory; schema-1 named source bytes stay unchanged."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal[2]
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
    task_file: str
    factory: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    files_sha256: dict[str, str] = Field(min_length=1, max_length=32)
    health_command: tuple[str, ...] = Field(min_length=1, max_length=8)
    tasks: tuple[_PinnedTask, ...] = Field(min_length=1, max_length=8)


_SupportedManifest: TypeAlias = _EvalManifest | _EvalManifestV2


@dataclass(frozen=True, kw_only=True)
class ResolvedInspectEvalCase:
    """A real authored Task plus its source-only case identity and derived objective."""

    case: EvalCaseRef
    task: Task
    objective: SeedObjective
    original_input_sha256: str
    original_target_sha256: str
    pinned_samples: tuple[_PinnedSample, ...]
    strict_inventory: bool
    scorer_name: str
    scorer_fingerprint: str
    setup_fingerprints: tuple[str, ...]
    cleanup_name: str | None
    health_command: tuple[str, ...]
    source_files: tuple[tuple[Path, str, str], ...]
    source_manifest_path: Path | None
    source_manifest_sha256: str | None
    sandbox_sha256: str
    input_override_sha256: str | None
    approved_image_ids: dict[str, str]
    approved_images: dict[str, str]

    def verify_unchanged(self) -> None:
        """
        Recheck code/assets and in-memory Task before this case can start.

        Raises:
            ValueError: If its trusted source, Task, or effective sandbox drifted.
        """
        if self.source_manifest_path is not None and (
            _sha256_file(self.source_manifest_path) != self.source_manifest_sha256
        ):
            raise ValueError("Trusted Eval manifest changed after source resolution.")
        for path, relative, expected in self.source_files:
            if not _trusted_file(path=path, root=path.parents[len(PurePosixPath(relative).parts) - 1]):
                raise ValueError("Trusted Eval source path changed after resolution.")
            if _sha256_file(path) != expected:
                raise ValueError("Trusted Eval source asset changed after resolution.")
        _validate_task(
            task=self.task,
            expected_name=self.case.task_name,
            expected_version=self.case.task_version,
            selected_sample_id=self.case.sample_id,
            samples=self.pinned_samples,
            strict_inventory=self.strict_inventory,
            source_files=self.source_files,
            scorer_name=self.scorer_name,
            scorer_fingerprint=self.scorer_fingerprint,
            setup_fingerprints=self.setup_fingerprints,
            cleanup_name=self.cleanup_name,
            sandbox_sha256=self.sandbox_sha256,
            approved_image_ids=self.approved_image_ids,
            approved_images=self.approved_images,
        )


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError("A trusted Eval source file is missing or a symlink.")
    with path.open("rb") as contents:
        return hashlib.file_digest(contents, "sha256").hexdigest()


def _relative_file(*, root: Path, name: str) -> Path:
    parts = PurePosixPath(name).parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in name.split("/"))
        or "\\" in name
        or ":" in name
        or name.startswith("/")
    ):
        raise ValueError("Trusted Eval source paths must be canonical relative files.")
    candidate = root.joinpath(*parts)
    if not _trusted_file(path=candidate, root=root):
        raise ValueError("Trusted Eval source files must be regular, inside the approved directory, and not symlinks.")
    return candidate


def _trusted_file(*, path: Path, root: Path) -> bool:
    return (
        root.is_dir()
        and not root.is_symlink()
        and path.is_file()
        and not path.is_symlink()
        and path.resolve().is_relative_to(root.resolve())
        and all(not parent.is_symlink() for parent in path.parents if parent != root and parent.is_relative_to(root))
    )


def _source_files(*, root: Path, manifest: _SupportedManifest) -> tuple[tuple[Path, str, str], ...]:
    if manifest.task_file not in manifest.files_sha256 or not manifest.task_file.endswith(".py"):
        raise ValueError("The trusted Eval factory must be one pinned Python source file.")
    checked: list[tuple[Path, str, str]] = []
    for name, digest in sorted(manifest.files_sha256.items()):
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("Trusted Eval source assets need exact SHA256 digests.")
        path = _relative_file(root=root, name=name)
        if _sha256_file(path) != digest:
            raise ValueError("Trusted Eval source asset digest changed.")
        checked.append((path, name, digest))
    return tuple(checked)


def _registered_fingerprint(*, component: object) -> str:
    if not is_registry_object(component):
        raise ValueError("Inspect Eval V1 requires a registered original setup/solver/scorer.")
    name = registry_unqualified_name(registry_info(component).name)
    try:
        params = json.loads(json.dumps(registry_params(component), sort_keys=True, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("Inspect Eval V1 cannot fingerprint non-JSON authored solver/scorer settings.") from error
    return config_hash({"name": name, "params": params})


def _validate_sample(*, sample: Sample, pinned: _PinnedSample) -> None:
    if (
        str(sample.id) != pinned.sample_id
        or not isinstance(sample.input, str)
        or not sample.input.strip()
        or len(sample.input.encode("utf-8")) > 32768
        or hashlib.sha256(sample.input.encode("utf-8")).hexdigest() != pinned.sample_input_sha256
        or not isinstance(sample.target, str)
        or hashlib.sha256(sample.target.encode("utf-8")).hexdigest() != pinned.sample_target_sha256
        or sample.choices is not None
        or sample.metadata is not None
    ):
        raise ValueError("Materialized Inspect Task/Sample identity or supported input changed.")
    if (
        sample.sandbox is not None
        or sample.files is not None
        or sample.setup is not None
        or sample.checkpoint is not None
    ):
        raise ValueError("Inspect Eval V1 does not qualify per-Sample sandbox, files, or setup overrides.")


def _validate_task(
    *,
    task: Task,
    expected_name: str,
    expected_version: str,
    selected_sample_id: str,
    samples: tuple[_PinnedSample, ...],
    strict_inventory: bool,
    source_files: tuple[tuple[Path, str, str], ...],
    scorer_name: str,
    scorer_fingerprint: str,
    setup_fingerprints: tuple[str, ...],
    cleanup_name: str | None,
    approved_image_ids: dict[str, str],
    approved_images: dict[str, str],
    sandbox_sha256: str | None = None,
) -> str:
    if task.sample_source is not None or len(task.dataset) != len(samples):
        if len(samples) == 1 and not strict_inventory:
            raise ValueError("Inspect Eval V1 requires exactly one materialized Task and Sample.")
        raise ValueError("Inspect Eval requires its exact declared materialized Task/Sample inventory.")
    if (
        task.name != expected_name
        or str(task.version) != expected_version
        or selected_sample_id not in {sample.sample_id for sample in samples}
        or tuple(str(sample.id) for sample in task.dataset) != tuple(sample.sample_id for sample in samples)
    ):
        raise ValueError("Materialized Inspect Task/Sample identity or supported input changed.")
    for sample, pinned in zip(task.dataset, samples, strict=True):
        _validate_sample(sample=sample, pinned=pinned)
    if strict_inventory:
        _validate_strict_task_inventory(task=task)
        _validate_callback_sources(task=task, source_files=source_files)
    if not isinstance(task.sandbox, SandboxEnvironmentSpec) or not isinstance(task.sandbox.config, ComposeConfig):
        raise ValueError("Inspect Eval V1 requires one reviewed Docker Compose Task sandbox.")
    if task.sandbox.type != "docker":
        raise ValueError("Inspect Eval V1 supports only a Docker Compose Task sandbox.")
    services = task.sandbox.config.services
    if (
        set(services) != set(approved_image_ids)
        or set(services) != set(approved_images)
        or any(
            service.pull_policy != "never"
            or not service.image
            or re.fullmatch(r"sha256:[0-9a-f]{64}", approved_image_ids[name]) is None
            or service.image != approved_images[name]
            for name, service in services.items()
        )
    ):
        raise ValueError("Every prebuilt Eval image needs its approved tag and pinned local image ID.")
    scorer = task.scorer or []
    if (
        len(scorer) != 1
        or not is_registry_object(scorer[0])
        or registry_unqualified_name(registry_info(scorer[0]).name) != scorer_name
        or _registered_fingerprint(component=scorer[0]) != scorer_fingerprint
    ):
        raise ValueError("Materialized Inspect Task changed its one original scorer or settings.")
    if registry_params(scorer[0]):
        raise ValueError("Inspect Eval V1 cannot qualify parameterized scorers or external grading assets.")
    if not is_registry_object(task.solver) or registry_info(task.solver).name != "inspect_ai/generate":
        raise ValueError("Inspect Eval V1 cannot discard authored solver initialization.")
    if registry_params(task.solver):
        raise ValueError("Inspect Eval V1 does not qualify customized authored solver settings.")
    setup_steps = task.setup if isinstance(task.setup, list) else [task.setup] if task.setup is not None else []
    if tuple(_registered_fingerprint(component=step) for step in setup_steps) != setup_fingerprints:
        raise ValueError("The original Inspect Task.setup steps or settings changed.")
    if any(registry_params(step) for step in setup_steps):
        raise ValueError("Inspect Eval V1 cannot qualify parameterized Task.setup or external preparation assets.")
    if (getattr(task.cleanup, "__qualname__", None) if task.cleanup else None) != cleanup_name:
        raise ValueError("The original Inspect Task cleanup callback changed.")
    if (
        task.model is not None
        or task.model_roles is not None
        or task.metadata is not None
        or task.epochs not in (None, 1)
        or task.checkpoint is not None
        or task.on_checkpoint is not None
        or task.on_resume is not None
        or task.config.model_dump(exclude_defaults=True)
    ):
        raise ValueError("Inspect Eval V1 does not qualify extra authored Task model, metadata, or checkpoint policy.")
    digest = hashlib.sha256(task.sandbox.config.model_dump_json(by_alias=True).encode("utf-8")).hexdigest()
    if sandbox_sha256 is not None and digest != sandbox_sha256:
        raise ValueError("The effective per-Sample Compose sandbox changed before execution.")
    return digest


def _validate_strict_task_inventory(*, task: Task) -> None:
    if (
        not isinstance(task.dataset, MemoryDataset)
        or task.dataset.name is not None
        or task.dataset.location is not None
        or task.dataset.shuffled
        or any(
            getattr(task, field) is not None
            for field in (
                "metrics",
                "approval",
                "review",
                "epochs_reducer",
                "fail_on_error",
                "continue_on_fail",
                "score_on_error",
                "message_limit",
                "token_limit",
                "token_limit_type",
                "turn_limit",
                "time_limit",
                "working_limit",
                "cost_limit",
                "early_stopping",
                "tags",
                "viewer",
                "headline_metric",
                "_display_name",
            )
        )
    ):
        raise ValueError("Inspect Eval V2 cannot qualify additional dataset or Task execution settings.")


def _validate_callback_sources(*, task: Task, source_files: tuple[tuple[Path, str, str], ...]) -> None:
    approved_paths = {path.resolve() for path, _, _ in source_files}
    setup = task.setup if isinstance(task.setup, list) else [task.setup] if task.setup is not None else []
    callbacks = [*(task.scorer or []), *setup, *([task.cleanup] if task.cleanup is not None else [])]
    for callback in callbacks:
        code = getattr(callback, "__code__", None)
        if not isinstance(code, CodeType) or Path(code.co_filename).resolve() not in approved_paths:
            raise ValueError("Original Inspect scorer, setup and cleanup must originate from pinned Eval source files.")


def _load_task_factory(*, name: str, path: Path, kind: EvalSourceKind) -> Callable[..., object]:
    if kind is EvalSourceKind.NAMED:
        module = importlib.import_module("examples.inspect_ghcp_protocol_smoke")
    else:
        spec = importlib.util.spec_from_file_location(f"trusted_inspect_eval_{uuid4().hex}", path)
        if spec is None or spec.loader is None:
            raise ValueError("The trusted local Eval Task module could not be loaded.")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    factory = getattr(module, name, None)
    if not callable(factory):
        raise ValueError("Trusted Eval Task factory was not found.")
    return factory


def _pinned_tasks(*, manifest: _SupportedManifest) -> tuple[_PinnedTask, ...]:
    if isinstance(manifest, _EvalManifest):
        return (
            _PinnedTask(
                task_name=manifest.task_name,
                task_version=manifest.task_version,
                scorer_name=manifest.scorer_name,
                scorer_fingerprint=manifest.scorer_fingerprint,
                setup_fingerprints=manifest.setup_fingerprints,
                cleanup_name=manifest.cleanup_name,
                samples=(
                    _PinnedSample(
                        sample_id=manifest.sample_id,
                        sample_input_sha256=manifest.sample_input_sha256,
                        sample_target_sha256=manifest.sample_target_sha256,
                    ),
                ),
            ),
        )
    if not 2 <= sum(len(task.samples) for task in manifest.tasks) <= 32:
        raise ValueError("Trusted Eval V2 must pin 2 to 32 original Task/Sample cases.")
    task_ids = [(task.task_name, task.task_version) for task in manifest.tasks]
    if len(task_ids) != len(set(task_ids)) or any(
        len({sample.sample_id for sample in task.samples}) != len(task.samples) for task in manifest.tasks
    ):
        raise ValueError("Trusted Eval V2 contains duplicate Task or per-Task Sample identities.")
    return manifest.tasks


def _materialize_tasks(
    *,
    root: Path,
    manifest: _SupportedManifest,
    kind: EvalSourceKind,
    source_files: tuple[tuple[Path, str, str], ...],
    agent_image: str,
    target_image: str,
) -> tuple[Task, ...]:
    task_path = next(path for path, name, _ in source_files if name == manifest.task_file)
    try:
        factory = _load_task_factory(name=manifest.factory, path=task_path, kind=kind)
        if _source_files(root=root, manifest=manifest) != source_files:
            raise ValueError("Trusted Eval source changed while its factory was imported.")
        constructed = factory(agent_image=agent_image, target_image=target_image)
    except (FileNotFoundError, ImportError) as error:
        raise ValueError("Trusted Eval Task factory could not load a declared dependency.") from error
    if _source_files(root=root, manifest=manifest) != source_files:
        raise ValueError("Trusted Eval source changed while its Task was materialized.")
    if isinstance(manifest, _EvalManifest):
        if not isinstance(constructed, Task):
            raise ValueError("Trusted Eval factory returned a non-Task or multiple Tasks.")
        return (constructed,)
    if isinstance(constructed, Task):
        constructed = (constructed,)
    if not isinstance(constructed, (list, tuple)) or len(constructed) != len(manifest.tasks):
        raise ValueError("Trusted Eval factory did not return the exact declared Task inventory.")
    tasks: list[Task] = []
    for task in constructed:
        if not isinstance(task, Task):
            raise ValueError("Trusted Eval factory returned a non-Task in its declared inventory.")
        tasks.append(task)
    return tuple(tasks)


class EvalSourceFactory:
    """Resolve a pinned named or explicitly approved local source before Task launch."""

    MANIFEST_FILE = "inspect-eval.json"
    NAMED_SOURCE_SHA256 = "0d7175315b4eca1ff3a2221b6d8e3a3c60b4a8ae48abcf96031847e73f3a44e7"
    NAMED_SAMPLE_SHA256 = "701f3fb4e19e8fa42ef2184c6ec33ac564b535283959e1db14760e3b5b2e7ccb"
    NAMED_TARGET_SHA256 = "da49b56729c23790a2b3c7c9b4a840112f819df24bad1ef93d0f52ac73b4497c"

    @classmethod
    def resolve(
        cls,
        *,
        family: str | None,
        trusted_dir: Path | None,
        trusted_local: bool,
        revision_sha256: str | None,
        input_override: str | None,
        agent_image: str,
        target_image: str,
        approved_image_ids: dict[str, str],
    ) -> ResolvedInspectEvalCase:
        """
        Preserve the schema-1 single-case API without silently choosing a sweep's first case.

        Returns:
            ResolvedInspectEvalCase: One complete Task/Sample source case.

        Raises:
            ValueError: If the trusted source contains multiple cases.
        """
        cases = cls.resolve_cases(
            family=family,
            trusted_dir=trusted_dir,
            trusted_local=trusted_local,
            revision_sha256=revision_sha256,
            input_override=input_override,
            agent_image=agent_image,
            target_image=target_image,
            approved_image_ids=approved_image_ids,
        )
        if len(cases) != 1:
            raise ValueError("A multi-case Eval requires resolve_cases; selecting its first Sample is forbidden.")
        return cases[0]

    @classmethod
    def resolve_cases(
        cls,
        *,
        family: str | None,
        trusted_dir: Path | None,
        trusted_local: bool,
        revision_sha256: str | None,
        input_override: str | None,
        agent_image: str,
        target_image: str,
        approved_image_ids: dict[str, str],
    ) -> tuple[ResolvedInspectEvalCase, ...]:
        """
        Pin and validate every original Task and Sample before building case work.

        Returns:
            tuple[ResolvedInspectEvalCase, ...]: Cases in declared Task/Sample order.

        Raises:
            ValueError: If source, exact inventory, images, input, or sandbox is unqualified.
        """
        if (family is None) == (trusted_dir is None):
            raise ValueError("Select exactly one named Eval family or explicitly trusted local directory.")
        if family is not None:
            if trusted_local:
                raise ValueError("Local-code trust cannot be combined with a named Eval family.")
            root, manifest, digest = cls._named_source(family=family)
            kind = EvalSourceKind.NAMED
            manifest_path = None
        else:
            assert trusted_dir is not None
            root, manifest, digest, manifest_path = cls._local_source(
                trusted_dir=trusted_dir, trusted_local=trusted_local, revision_sha256=revision_sha256
            )
            kind = EvalSourceKind.TRUSTED_LOCAL
        if revision_sha256 is not None and revision_sha256 != digest:
            raise ValueError("The selected Eval revision differs from its pinned source digest.")
        source_files = _source_files(root=root, manifest=manifest)
        pinned_tasks = _pinned_tasks(manifest=manifest)
        if input_override is not None and not isinstance(manifest, _EvalManifest):
            raise ValueError("A GUI initial-input variant cannot sweep multiple Eval cases.")
        if not manifest.health_command[0].startswith("/") or any(
            not value or len(value) > 128 or "\n" in value for value in manifest.health_command
        ):
            raise ValueError("An Eval Task must name a fixed, bounded target-side health command.")
        tasks = _materialize_tasks(
            root=root,
            manifest=manifest,
            kind=kind,
            source_files=source_files,
            agent_image=agent_image,
            target_image=target_image,
        )
        approved_images = {"agent": agent_image, "model-bridge": agent_image, "target": target_image}
        sandbox_hashes = tuple(
            _validate_task(
                task=task,
                expected_name=pinned.task_name,
                expected_version=pinned.task_version,
                selected_sample_id=pinned.samples[0].sample_id,
                samples=pinned.samples,
                strict_inventory=isinstance(manifest, _EvalManifestV2),
                source_files=source_files,
                scorer_name=pinned.scorer_name,
                scorer_fingerprint=pinned.scorer_fingerprint,
                setup_fingerprints=pinned.setup_fingerprints,
                cleanup_name=pinned.cleanup_name,
                approved_image_ids=approved_image_ids,
                approved_images=approved_images,
            )
            for task, pinned in zip(tasks, pinned_tasks, strict=True)
        )
        if len(set(sandbox_hashes)) != 1:
            raise ValueError("One Eval run cannot mix different effective per-Sample Compose sandbox profiles.")
        package = EvalPackageRef(kind=kind, name=manifest.name, source_sha256=digest)
        if input_override is not None:
            assert isinstance(manifest, _EvalManifest)
            if not manifest.allow_initial_input_override:
                raise ValueError("This Eval source did not approve initial Sample.input edits.")
            if not input_override.strip() or len(input_override.encode("utf-8")) > 32768:
                raise ValueError("The initial input edit must contain at most 32768 nonempty UTF-8 bytes.")
        cases: list[ResolvedInspectEvalCase] = []
        for task, pinned, sandbox_sha in zip(tasks, pinned_tasks, sandbox_hashes, strict=True):
            for sample, declared in zip(task.dataset, pinned.samples, strict=True):
                effective_task = task
                effective_sample = sample
                pinned_samples = pinned.samples
                override_sha = None
                if input_override is not None:
                    override_sha = hashlib.sha256(input_override.encode("utf-8")).hexdigest()
                    effective_sample = sample.model_copy(update={"input": input_override})
                    effective_task = copy.copy(task)
                    effective_task.dataset = MemoryDataset(
                        samples=[effective_sample],
                        name=task.dataset.name,
                        location=task.dataset.location,
                        shuffled=task.dataset.shuffled,
                    )
                    pinned_samples = (declared.model_copy(update={"sample_input_sha256": override_sha}),)
                    _validate_task(
                        task=effective_task,
                        expected_name=pinned.task_name,
                        expected_version=pinned.task_version,
                        selected_sample_id=declared.sample_id,
                        samples=pinned_samples,
                        strict_inventory=False,
                        source_files=source_files,
                        scorer_name=pinned.scorer_name,
                        scorer_fingerprint=pinned.scorer_fingerprint,
                        setup_fingerprints=pinned.setup_fingerprints,
                        cleanup_name=pinned.cleanup_name,
                        approved_image_ids=approved_image_ids,
                        approved_images=approved_images,
                        sandbox_sha256=sandbox_sha,
                    )
                if not isinstance(effective_sample.input, str):
                    raise ValueError("Inspect Eval accepts only supported text Sample.input.")
                case = EvalCaseRef(
                    package=package,
                    task_name=pinned.task_name,
                    task_version=pinned.task_version,
                    sample_id=declared.sample_id,
                    epoch=1,
                )
                cases.append(
                    ResolvedInspectEvalCase(
                        case=case,
                        task=effective_task,
                        objective=SeedObjective(value=effective_sample.input),
                        original_input_sha256=declared.sample_input_sha256,
                        original_target_sha256=declared.sample_target_sha256,
                        pinned_samples=pinned_samples,
                        strict_inventory=isinstance(manifest, _EvalManifestV2),
                        scorer_name=pinned.scorer_name,
                        scorer_fingerprint=pinned.scorer_fingerprint,
                        setup_fingerprints=pinned.setup_fingerprints,
                        cleanup_name=pinned.cleanup_name,
                        health_command=manifest.health_command,
                        source_files=source_files,
                        source_manifest_path=manifest_path,
                        source_manifest_sha256=digest if manifest_path is not None else None,
                        sandbox_sha256=sandbox_sha,
                        input_override_sha256=override_sha,
                        approved_image_ids=dict(approved_image_ids),
                        approved_images=dict(approved_images),
                    )
                )
        return tuple(cases)

    @classmethod
    def _named_source(cls, *, family: str) -> tuple[Path, _EvalManifest, str]:
        if family != "benign_protocol":
            raise ValueError("Unknown or unqualified named Inspect Eval family.")
        root = Path(__file__).resolve().parents[3]
        manifest = _EvalManifest(
            name=family,
            task_file="examples/inspect_ghcp_protocol_smoke.py",
            factory="original_benign_task",
            files_sha256={"examples/inspect_ghcp_protocol_smoke.py": cls.NAMED_SOURCE_SHA256},
            task_name="inspect_ghcp_benign_protocol_smoke",
            task_version="1",
            sample_id="benign-ghcp-protocol-1",
            sample_input_sha256=cls.NAMED_SAMPLE_SHA256,
            sample_target_sha256=cls.NAMED_TARGET_SHA256,
            scorer_name="original_target_marker_scorer",
            scorer_fingerprint=config_hash({"name": "original_target_marker_scorer", "params": {}}),
            setup_fingerprints=(config_hash({"name": "prepare_target_marker", "params": {}}),),
            cleanup_name="cleanup_async",
            health_command=("/bin/test", "-f", "/tmp/inspect-marker"),
            allow_initial_input_override=True,
        )
        return root, manifest, hashlib.sha256(manifest.model_dump_json(exclude_none=True).encode("utf-8")).hexdigest()

    @classmethod
    def _local_source(
        cls, *, trusted_dir: Path, trusted_local: bool, revision_sha256: str | None
    ) -> tuple[Path, _SupportedManifest, str, Path]:
        if not trusted_local or revision_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", revision_sha256) is None:
            raise ValueError("Local Eval code requires explicit trust and its exact manifest SHA256.")
        if not trusted_dir.is_absolute() or trusted_dir.is_symlink() or not trusted_dir.is_dir():
            raise ValueError("Trusted Eval directory must be an existing absolute, non-symlink directory.")
        root = trusted_dir.resolve(strict=True)
        manifest_path = _relative_file(root=root, name=cls.MANIFEST_FILE)
        if manifest_path.stat().st_size > 65_536:
            raise ValueError("Trusted Eval manifest exceeds its bounded size.")
        content = manifest_path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        if digest != revision_sha256:
            raise ValueError("Local Eval manifest changed from the explicitly approved revision.")
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("Trusted Eval manifest must declare a supported schema and exact Task inventory.")
        version = data.get("schema_version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError("Trusted Eval manifest has an unsupported schema version.")
        manifest = (_EvalManifest if version == 1 else _EvalManifestV2).model_validate(data)
        return root, manifest, digest, manifest_path
