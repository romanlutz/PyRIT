# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Static compilers that produce a ``CapabilitySuiteManifest`` from a concrete source.

Every compiler here is static: it reads declarative data (local JSON/JSONL/CSV files,
an already-fetched ``SeedDataset``, or files inside a checked-out evaluation
repository) and maps it into manifest models. No compiler in this module ever
imports or executes third-party Python, and none ever construct or evaluate
Jinja/format-string expressions supplied by the source data itself -- prompt
"templates" here are plain ``string.Template`` (``$name``/``${name}``) substitutions,
which can only look up values, never execute code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from string import Template
from typing import TYPE_CHECKING, Protocol, cast

from pyrit.scenario.capability_suite.manifest import (
    AssetMode,
    BuildContextAssetKind,
    BuildContextAssetManifest,
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CaseAssetManifest,
    CaseMessageManifest,
    RunPolicyManifest,
    SuiteProvenance,
    validate_safe_relative_path,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from pyrit.models import JSONValue
    from pyrit.models.literals import ChatMessageRole
    from pyrit.models.seeds.seed_dataset import SeedDataset
    from pyrit.scenario.capability_suite.manifest import SandboxProviderManifest

    _JSONRecord = Mapping[str, JSONValue]


class ManifestCompiler(Protocol):
    """A static compiler that produces one immutable ``CapabilitySuiteManifest``."""

    def compile(self) -> CapabilitySuiteManifest:
        """Compile and return the manifest."""


def _render_template(*, template: str, record: Mapping[str, object]) -> str:
    """
    Render a safe ``string.Template`` substitution -- lookup only, never executed code.

    Returns:
        str: The template with placeholders substituted from ``record``.

    Raises:
        ValueError: If the template references an unknown field or is malformed.
    """
    flat = {key: ("" if value is None else str(value)) for key, value in record.items()}
    try:
        return Template(template).substitute(**flat)
    except (KeyError, ValueError) as error:
        raise ValueError(f"Prompt template could not be resolved safely: {error}") from error


@dataclass(frozen=True)
class RecordAssetMapping:
    """Maps one per-record local file reference into a staged, hashed manifest asset."""

    source_field: str
    destination_template: str
    asset_id_template: str = "asset-{index}-{asset_index}"
    environment: str | None = None
    mode: AssetMode = AssetMode.READ_ONLY


@dataclass(frozen=True)
class BuildContextAssetSource:
    """A contained Dockerfile, Compose file, or build-context source to hash."""

    kind: BuildContextAssetKind
    source: str


@dataclass(frozen=True)
class RecordFieldMapping:
    """A deterministic, explicit mapping from one flat record to one capability case."""

    objective_template: str
    message_content_template: str
    case_id_field: str | None = None
    message_role: ChatMessageRole = "user"
    tags_fields: tuple[str, ...] = ()
    metadata_fields: tuple[str, ...] = ()
    assets: tuple[RecordAssetMapping, ...] = ()


@dataclass(frozen=True)
class RecordSplit:
    """A deterministic slice applied to filtered records, preserving source order."""

    offset: int = 0
    limit: int | None = None

    def apply(self, records: Sequence[_JSONRecord]) -> Sequence[_JSONRecord]:
        """Return the deterministic slice of ``records`` selected by this split."""
        end = None if self.limit is None else self.offset + self.limit
        return records[self.offset : end]


class UnsafeAssetPathError(ValueError):
    """Raised when a record references a local asset outside of its containment root."""


def compile_build_context_assets(
    *,
    root: Path,
    sources: tuple[BuildContextAssetSource, ...],
) -> tuple[BuildContextAssetManifest, ...]:
    """
    Validate and hash Dockerfile/Compose/build-context files beneath ``root``.

    Returns:
        tuple[BuildContextAssetManifest, ...]: Immutable asset integrity records.

    Raises:
        UnsafeAssetPathError: If a source escapes ``root``.
    """
    resolved_root = root.resolve()
    assets: list[BuildContextAssetManifest] = []
    for source in sources:
        validate_safe_relative_path(source.source)
        resolved = (resolved_root / source.source).resolve()
        if resolved != resolved_root and resolved_root not in resolved.parents:
            raise UnsafeAssetPathError(f"Build-context source '{source.source}' escapes root '{resolved_root}'.")
        assets.append(
            BuildContextAssetManifest(
                kind=source.kind,
                source=source.source,
                sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
            )
        )
    return tuple(assets)


class _LocalRecordSuiteCompiler:
    """Shared field-mapping/templating/filter/split/asset-hash logic for local records."""

    def __init__(
        self,
        *,
        suite_id: str,
        name: str,
        provenance: SuiteProvenance,
        sandbox_provider: SandboxProviderManifest,
        field_mapping: RecordFieldMapping,
        path: Path | None = None,
        description: str | None = None,
        run_policy: RunPolicyManifest | None = None,
        record_filter: Callable[[Mapping[str, JSONValue]], bool] | None = None,
        split: RecordSplit | None = None,
        assets_root: Path | None = None,
        tags: tuple[str, ...] = (),
    ) -> None:
        """Initialize shared static-compiler state."""
        self._suite_id = suite_id
        self._name = name
        self._description = description
        self._provenance = provenance
        self._sandbox_provider = sandbox_provider
        self._run_policy = run_policy or RunPolicyManifest()
        self._field_mapping = field_mapping
        self._path = path
        self._record_filter = record_filter
        self._split = split
        self._assets_root = assets_root.resolve() if assets_root is not None else None
        self._tags = tags

    def _read_records(self) -> list[Mapping[str, JSONValue]]:
        """Read raw flat records from the compiler's source. Overridden by subclasses."""
        raise NotImplementedError

    def compile(self) -> CapabilitySuiteManifest:
        """
        Compile the configured source into an immutable capability-suite manifest.

        Returns:
            CapabilitySuiteManifest: The compiled, strictly-validated manifest.

        Raises:
            UnsafeAssetPathError: If a record references an asset outside ``assets_root``.
            ValueError: If field mapping produces an invalid case (delegated to Pydantic).
        """
        records = self._read_records()
        if self._record_filter is not None:
            records = [record for record in records if self._record_filter(record)]
        if self._split is not None:
            records = list(self._split.apply(records))

        cases = tuple(self._build_case(index=index, record=record) for index, record in enumerate(records))
        return CapabilitySuiteManifest(
            suite_id=self._suite_id,
            name=self._name,
            description=self._description,
            provenance=self._provenance,
            sandbox_provider=self._sandbox_provider,
            run_policy=self._run_policy,
            cases=cases,
            tags=self._tags,
        )

    def _build_case(self, *, index: int, record: Mapping[str, JSONValue]) -> CapabilityCaseManifest:
        mapping = self._field_mapping
        case_id = str(record[mapping.case_id_field]) if mapping.case_id_field else _content_case_id(record)
        objective = _render_template(template=mapping.objective_template, record=record)
        content = _render_template(template=mapping.message_content_template, record=record)
        tags = tuple(str(record[name]) for name in mapping.tags_fields if record.get(name) is not None)
        metadata = {name: record[name] for name in mapping.metadata_fields if name in record}
        assets = tuple(
            self._build_asset(
                index=index,
                asset_index=asset_index,
                record=record,
                asset_mapping=asset_mapping,
            )
            for asset_index, asset_mapping in enumerate(mapping.assets)
        )
        return CapabilityCaseManifest(
            case_id=case_id,
            objective=objective,
            messages=(CaseMessageManifest(role=mapping.message_role, content=content),),
            assets=assets,
            tags=tags,
            metadata=metadata,
        )

    def _build_asset(
        self,
        *,
        index: int,
        asset_index: int,
        record: Mapping[str, JSONValue],
        asset_mapping: RecordAssetMapping,
    ) -> CaseAssetManifest:
        if self._assets_root is None:
            raise ValueError("A record declares an asset but no 'assets_root' was configured on the compiler.")
        source = str(record[asset_mapping.source_field])
        validate_safe_relative_path(source)
        resolved = (self._assets_root / source).resolve()
        if self._assets_root not in resolved.parents and resolved != self._assets_root:
            raise UnsafeAssetPathError(f"Asset source '{source}' escapes assets_root '{self._assets_root}'.")
        sha256 = hashlib.sha256(resolved.read_bytes()).hexdigest()
        destination = _render_template(template=asset_mapping.destination_template, record=record)
        asset_id = asset_mapping.asset_id_template.format(index=index, asset_index=asset_index)
        return CaseAssetManifest(
            asset_id=asset_id,
            source=source,
            sha256=sha256,
            destination=destination,
            environment=asset_mapping.environment,
            mode=asset_mapping.mode,
        )


class LocalJsonSuiteCompiler(_LocalRecordSuiteCompiler):
    """Compile a suite from a local JSON file containing an array of flat records."""

    def _read_records(self) -> list[Mapping[str, JSONValue]]:
        if self._path is None:
            raise ValueError("LocalJsonSuiteCompiler requires a source path.")
        data = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array of records in '{self._path}', got {type(data).__name__}.")
        return data


class LocalJsonlSuiteCompiler(_LocalRecordSuiteCompiler):
    """Compile a suite from a local newline-delimited JSON (JSONL) file."""

    def _read_records(self) -> list[Mapping[str, JSONValue]]:
        if self._path is None:
            raise ValueError("LocalJsonlSuiteCompiler requires a source path.")
        records: list[Mapping[str, JSONValue]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
        return records


class LocalCsvSuiteCompiler(_LocalRecordSuiteCompiler):
    """Compile a suite from a local CSV file, one row per case."""

    def _read_records(self) -> list[Mapping[str, JSONValue]]:
        if self._path is None:
            raise ValueError("LocalCsvSuiteCompiler requires a source path.")
        with self._path.open(mode="r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))


class HuggingFaceDatasetSuiteCompiler(_LocalRecordSuiteCompiler):
    """Compile records loaded through the existing ``datasets.load_dataset`` facility."""

    def __init__(
        self,
        *,
        dataset_name: str,
        split_name: str,
        suite_id: str,
        name: str,
        provenance: SuiteProvenance,
        sandbox_provider: SandboxProviderManifest,
        field_mapping: RecordFieldMapping,
        dataset_config: str | None = None,
        revision: str | None = None,
        token: str | None = None,
        dataset_loader: Callable[..., object] | None = None,
        description: str | None = None,
        run_policy: RunPolicyManifest | None = None,
        record_filter: Callable[[Mapping[str, JSONValue]], bool] | None = None,
        split: RecordSplit | None = None,
        tags: tuple[str, ...] = (),
    ) -> None:
        """Initialize a deterministic Hugging Face record compiler."""
        super().__init__(
            suite_id=suite_id,
            name=name,
            provenance=provenance,
            sandbox_provider=sandbox_provider,
            field_mapping=field_mapping,
            description=description,
            run_policy=run_policy,
            record_filter=record_filter,
            split=split,
            tags=tags,
        )
        self._dataset_name = dataset_name
        self._split_name = split_name
        self._dataset_config = dataset_config
        self._revision = revision
        self._token = token
        self._dataset_loader = dataset_loader

    def _read_records(self) -> list[Mapping[str, JSONValue]]:
        loader = self._dataset_loader
        if loader is None:
            from datasets import load_dataset

            loader = load_dataset
        dataset = loader(
            self._dataset_name,
            self._dataset_config,
            split=self._split_name,
            revision=self._revision,
            token=self._token,
            trust_remote_code=False,
        )
        if not isinstance(dataset, Iterable):
            raise ValueError("Hugging Face dataset loader must return an iterable of rows.")
        records: list[Mapping[str, JSONValue]] = []
        for record in dataset:
            if not isinstance(record, Mapping):
                raise ValueError("Hugging Face dataset rows must be mappings.")
            try:
                normalized = json.loads(json.dumps(dict(record)))
            except (TypeError, ValueError) as error:
                raise ValueError("Hugging Face dataset rows must contain only JSON values.") from error
            records.append(cast("dict[str, JSONValue]", normalized))
        return records


class SeedDatasetCompiler:
    """Compile a suite from an already-fetched PyRIT ``SeedDataset``."""

    def __init__(
        self,
        *,
        seed_dataset: SeedDataset,
        suite_id: str,
        name: str,
        provenance: SuiteProvenance,
        sandbox_provider: SandboxProviderManifest,
        description: str | None = None,
        run_policy: RunPolicyManifest | None = None,
        message_role: ChatMessageRole = "user",
        tags: tuple[str, ...] = (),
    ) -> None:
        """
        Initialize the seed-dataset compiler.

        Callers must fetch the dataset themselves (e.g. via
        ``SeedDatasetProvider.fetch_dataset_async``) -- this compiler never makes
        network calls, keeping it static and deterministic given its input.
        """
        self._seed_dataset = seed_dataset
        self._suite_id = suite_id
        self._name = name
        self._description = description
        self._provenance = provenance
        self._sandbox_provider = sandbox_provider
        self._run_policy = run_policy or RunPolicyManifest()
        self._message_role = message_role
        self._tags = tags

    def compile(self) -> CapabilitySuiteManifest:
        """
        Compile every seed in the dataset into one capability case each.

        Returns:
            CapabilitySuiteManifest: The compiled, strictly-validated manifest.
        """
        cases = []
        for index, seed in enumerate(self._seed_dataset.seeds):
            case_id = str(seed.id) if seed.id is not None else f"seed-{index}"
            harm_tags = tuple(seed.harm_categories or ())
            cases.append(
                CapabilityCaseManifest(
                    case_id=case_id,
                    objective=seed.value,
                    messages=(CaseMessageManifest(role=self._message_role, content=seed.value, data_type="text"),),
                    tags=harm_tags,
                    metadata={"dataset_name": seed.dataset_name} if seed.dataset_name else {},
                )
            )
        return CapabilitySuiteManifest(
            suite_id=self._suite_id,
            name=self._name,
            description=self._description,
            provenance=self._provenance,
            sandbox_provider=self._sandbox_provider,
            run_policy=self._run_policy,
            cases=tuple(cases),
            tags=self._tags,
        )


_EXECUTABLE_METHODOLOGY_PATTERNS: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"inspect_ai"),
    re.compile(rb"inspect_evals"),
    re.compile(rb"@task\b"),
    re.compile(rb"@solver\b"),
    re.compile(rb"@scorer\b"),
)
_STATIC_DATA_SUFFIXES = (".json", ".jsonl", ".csv")


@dataclass(frozen=True)
class CompatibilityReport:
    """The result of statically scanning a checked-out eval repository."""

    repo_path: str
    static_data_files: tuple[str, ...] = ()
    executable_indicator_files: tuple[str, ...] = ()

    @property
    def is_statically_compilable(self) -> bool:
        """Whether static data exists without executable methodology."""
        return bool(self.static_data_files) and not self.executable_indicator_files

    @property
    def has_executable_methodology(self) -> bool:
        """Whether executable task, solver, or scorer code was detected."""
        return bool(self.executable_indicator_files)


class UnsupportedExecutableMethodologyError(Exception):
    """Raised when a checked-out repo only exposes an executable (non-static) methodology."""

    def __init__(self, *, report: CompatibilityReport) -> None:
        """Initialize the error with the compatibility report that triggered it."""
        message = (
            f"Checked-out eval repository at '{report.repo_path}' has no statically-compilable data "
            f"files, but {len(report.executable_indicator_files)} file(s) indicate an executable "
            "methodology (e.g. Inspect AI tasks/solvers/scorers). PyRIT's capability-suite compiler "
            "never imports or executes third-party Python, so only a static, data-file-based suite "
            "can be compiled from this repository."
        )
        super().__init__(message)
        self.report = report


def scan_checked_out_eval_repo(*, repo_path: Path) -> CompatibilityReport:
    """
    Statically scan a checked-out evaluation repository for data files vs. executable code.

    Never imports or executes any file; this only inspects file suffixes and does a
    text-pattern scan (never ``exec``/``eval``/``importlib``) for known executable-
    methodology indicators (Inspect AI decorators/imports).

    Returns:
        CompatibilityReport: The discovered static data files and executable indicators.
    """
    repo_path = repo_path.resolve()
    static_data_files: list[str] = []
    executable_indicator_files: list[str] = []
    ignored_directories = {".git", ".venv", "node_modules", "__pycache__"}
    for candidate in sorted(repo_path.rglob("*")):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(repo_path).as_posix()
        if any(part in ignored_directories for part in candidate.relative_to(repo_path).parts):
            continue
        if candidate.suffix in _STATIC_DATA_SUFFIXES:
            static_data_files.append(relative)
        elif candidate.suffix == ".py":
            content = candidate.read_bytes()
            if any(pattern.search(content) for pattern in _EXECUTABLE_METHODOLOGY_PATTERNS):
                executable_indicator_files.append(relative)
    return CompatibilityReport(
        repo_path=str(repo_path),
        static_data_files=tuple(static_data_files),
        executable_indicator_files=tuple(executable_indicator_files),
    )


class CheckedOutEvalRepoCompiler:
    """
    Gate compilation of a checked-out eval repository on static-only compatibility.

    Scans the repository for statically-compilable data files vs. executable-
    methodology indicators, raising ``UnsupportedExecutableMethodologyError`` if no
    static data file is found. When static data files exist, the caller-supplied
    ``build_static_compiler`` factory is used to actually compile them -- this class
    never parses/maps record fields itself, keeping that logic in one place (the
    local record compilers above).
    """

    def __init__(
        self,
        *,
        repo_path: Path,
        build_static_compiler: Callable[[CompatibilityReport], ManifestCompiler],
    ) -> None:
        """Initialize the adapter around a repo path and an injected static-compiler factory."""
        self._repo_path = repo_path
        self._build_static_compiler = build_static_compiler

    def compatibility_report(self) -> CompatibilityReport:
        """
        Return the static-vs-executable compatibility report for the configured repo.

        Returns:
            CompatibilityReport: The scan result.
        """
        return scan_checked_out_eval_repo(repo_path=self._repo_path)

    def compile(self) -> CapabilitySuiteManifest:
        """
        Compile the checked-out repository's static data files into a manifest.

        Returns:
            CapabilitySuiteManifest: The compiled, strictly-validated manifest.

        Raises:
            UnsupportedExecutableMethodologyError: If no static data file is found.
        """
        report = self.compatibility_report()
        if not report.is_statically_compilable:
            raise UnsupportedExecutableMethodologyError(report=report)
        return self._build_static_compiler(report).compile()


def _content_case_id(record: Mapping[str, JSONValue]) -> str:
    encoded = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return f"case-{hashlib.sha256(encoded).hexdigest()[:16]}"
