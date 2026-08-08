# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import csv
import hashlib
import json
from typing import TYPE_CHECKING

import pytest

from pyrit.models.seeds.seed_dataset import SeedDataset
from pyrit.models.seeds.seed_prompt import SeedPrompt
from pyrit.scenario.capability_suite.compiler import (
    BuildContextAssetSource,
    CheckedOutEvalRepoCompiler,
    CompatibilityReport,
    HuggingFaceDatasetSuiteCompiler,
    LocalCsvSuiteCompiler,
    LocalJsonlSuiteCompiler,
    LocalJsonSuiteCompiler,
    RecordAssetMapping,
    RecordFieldMapping,
    RecordSplit,
    SeedDatasetCompiler,
    UnsafeAssetPathError,
    UnsupportedExecutableMethodologyError,
    compile_build_context_assets,
    scan_checked_out_eval_repo,
)
from pyrit.scenario.capability_suite.manifest import (
    BuildContextAssetKind,
    LocalSandboxProviderManifestConfig,
    SuiteProvenance,
)
from pyrit.scenario.capability_suite.serialization import manifest_hash

if TYPE_CHECKING:
    from pathlib import Path


def _provenance() -> SuiteProvenance:
    return SuiteProvenance(source="unit-test", repository="example/repo", revision="abc123", license="MIT")


def _sandbox_provider() -> LocalSandboxProviderManifestConfig:
    return LocalSandboxProviderManifestConfig()


def _field_mapping(**overrides: object) -> RecordFieldMapping:
    defaults: dict[str, object] = {
        "objective_template": "Complete: $objective",
        "message_content_template": "$objective",
        "case_id_field": "id",
    }
    defaults.update(overrides)
    return RecordFieldMapping(**defaults)


_RECORDS = [
    {"id": "rec-1", "objective": "do the first thing"},
    {"id": "rec-2", "objective": "do the second thing"},
]


def test_local_json_suite_compiler_compiles_records(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    source.write_text(json.dumps(_RECORDS), encoding="utf-8")

    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        path=source,
    )
    manifest = compiler.compile()

    assert [case.case_id for case in manifest.cases] == ["rec-1", "rec-2"]
    assert manifest.cases[0].objective == "Complete: do the first thing"
    assert manifest.cases[0].messages[0].content == "do the first thing"


def test_local_json_suite_compiler_rejects_non_array(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    source.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        path=source,
    )
    with pytest.raises(ValueError, match="Expected a JSON array"):
        compiler.compile()


def test_local_jsonl_suite_compiler_compiles_records(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text("\n".join(json.dumps(record) for record in _RECORDS) + "\n", encoding="utf-8")

    compiler = LocalJsonlSuiteCompiler(
        suite_id="suite-jsonl",
        name="JSONL suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        path=source,
    )
    manifest = compiler.compile()

    assert [case.case_id for case in manifest.cases] == ["rec-1", "rec-2"]


def test_local_jsonl_suite_compiler_skips_blank_lines(tmp_path: Path) -> None:
    source = tmp_path / "records.jsonl"
    source.write_text(f"\n{json.dumps(_RECORDS[0])}\n\n   \n{json.dumps(_RECORDS[1])}\n", encoding="utf-8")

    compiler = LocalJsonlSuiteCompiler(
        suite_id="suite-jsonl",
        name="JSONL suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        path=source,
    )
    manifest = compiler.compile()

    assert len(manifest.cases) == 2


def test_local_csv_suite_compiler_compiles_records(tmp_path: Path) -> None:
    source = tmp_path / "records.csv"
    with source.open(mode="w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "objective"])
        writer.writeheader()
        writer.writerows(_RECORDS)

    compiler = LocalCsvSuiteCompiler(
        suite_id="suite-csv",
        name="CSV suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        path=source,
    )
    manifest = compiler.compile()

    assert [case.case_id for case in manifest.cases] == ["rec-1", "rec-2"]


def test_local_json_suite_compiler_hash_is_deterministic_across_compiles(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    source.write_text(json.dumps(_RECORDS), encoding="utf-8")

    def _compile() -> str:
        compiler = LocalJsonSuiteCompiler(
            suite_id="suite-json",
            name="JSON suite",
            provenance=_provenance(),
            sandbox_provider=_sandbox_provider(),
            field_mapping=_field_mapping(),
            path=source,
        )
        return manifest_hash(compiler.compile())

    assert _compile() == _compile()


def test_local_json_suite_compiler_defaults_case_id_when_no_field_configured(tmp_path: Path) -> None:
    source = tmp_path / "records.json"
    source.write_text(json.dumps(_RECORDS), encoding="utf-8")

    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(case_id_field=None),
        path=source,
    )
    manifest = compiler.compile()

    assert [case.case_id for case in manifest.cases] == [
        "case-114be7fefc525afc",
        "case-66916422754b2ea0",
    ]


def test_record_filter_and_split_are_applied_deterministically(tmp_path: Path) -> None:
    records = [{"id": f"rec-{i}", "objective": f"task {i}"} for i in range(5)]
    source = tmp_path / "records.json"
    source.write_text(json.dumps(records), encoding="utf-8")

    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        path=source,
        record_filter=lambda record: int(record["id"].split("-")[1]) % 2 == 0,
        split=RecordSplit(offset=1, limit=1),
    )
    manifest = compiler.compile()

    # Filter keeps rec-0, rec-2, rec-4 (in source order); split(offset=1, limit=1) keeps rec-2 only.
    assert [case.case_id for case in manifest.cases] == ["rec-2"]


def test_field_mapping_tags_and_metadata_fields(tmp_path: Path) -> None:
    records = [{"id": "rec-1", "objective": "do it", "category": "cat-a", "note": "hello"}]
    source = tmp_path / "records.json"
    source.write_text(json.dumps(records), encoding="utf-8")

    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(tags_fields=("category",), metadata_fields=("note",)),
        path=source,
    )
    manifest = compiler.compile()

    case = manifest.cases[0]
    assert case.tags == ("cat-a",)
    assert case.metadata == {"note": "hello"}


def test_asset_mapping_hashes_and_stages_local_file(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    asset_file = assets_root / "payload.bin"
    asset_file.write_bytes(b"hello world")
    expected_sha256 = hashlib.sha256(b"hello world").hexdigest()

    records = [{"id": "rec-1", "objective": "do it", "asset_path": "payload.bin"}]
    source = tmp_path / "records.json"
    source.write_text(json.dumps(records), encoding="utf-8")

    mapping = _field_mapping(
        assets=(
            RecordAssetMapping(
                source_field="asset_path",
                destination_template="workspace/$asset_path",
            ),
        )
    )
    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=mapping,
        path=source,
        assets_root=assets_root,
    )
    manifest = compiler.compile()

    asset = manifest.cases[0].assets[0]
    assert asset.source == "payload.bin"
    assert asset.sha256 == expected_sha256
    assert asset.destination == "workspace/payload.bin"
    assert asset.asset_id == "asset-0-0"


def test_asset_mapping_assigns_unique_ids_to_multiple_assets(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    (assets_root / "a.bin").write_bytes(b"a")
    (assets_root / "b.bin").write_bytes(b"b")
    source = tmp_path / "records.json"
    source.write_text(
        json.dumps([{"id": "rec-1", "objective": "do it", "a": "a.bin", "b": "b.bin"}]),
        encoding="utf-8",
    )
    mapping = _field_mapping(
        assets=(
            RecordAssetMapping(source_field="a", destination_template="workspace/$a"),
            RecordAssetMapping(source_field="b", destination_template="workspace/$b"),
        )
    )
    manifest = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=mapping,
        path=source,
        assets_root=assets_root,
    ).compile()
    assert [asset.asset_id for asset in manifest.cases[0].assets] == ["asset-0-0", "asset-0-1"]


def test_compile_build_context_assets_hashes_dockerfile_and_compose(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (tmp_path / "compose.yaml").write_text("services: {}\n", encoding="utf-8")
    assets = compile_build_context_assets(
        root=tmp_path,
        sources=(
            BuildContextAssetSource(kind=BuildContextAssetKind.DOCKERFILE, source="Dockerfile"),
            BuildContextAssetSource(kind=BuildContextAssetKind.COMPOSE_FILE, source="compose.yaml"),
        ),
    )
    assert [asset.kind for asset in assets] == [
        BuildContextAssetKind.DOCKERFILE,
        BuildContextAssetKind.COMPOSE_FILE,
    ]
    assert all(len(asset.sha256) == 64 for asset in assets)


def test_asset_mapping_rejects_path_escaping_assets_root(tmp_path: Path) -> None:
    assets_root = tmp_path / "assets"
    assets_root.mkdir()
    outside_file = tmp_path / "outside.bin"
    outside_file.write_bytes(b"secret")

    records = [{"id": "rec-1", "objective": "do it", "asset_path": "../outside.bin"}]
    source = tmp_path / "records.json"
    source.write_text(json.dumps(records), encoding="utf-8")

    mapping = _field_mapping(
        assets=(RecordAssetMapping(source_field="asset_path", destination_template="/workspace/x"),)
    )
    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=mapping,
        path=source,
        assets_root=assets_root,
    )
    with pytest.raises(ValueError, match="traversal"):
        compiler.compile()


def test_asset_mapping_requires_assets_root_configured(tmp_path: Path) -> None:
    records = [{"id": "rec-1", "objective": "do it", "asset_path": "payload.bin"}]
    source = tmp_path / "records.json"
    source.write_text(json.dumps(records), encoding="utf-8")

    mapping = _field_mapping(
        assets=(RecordAssetMapping(source_field="asset_path", destination_template="/workspace/x"),)
    )
    compiler = LocalJsonSuiteCompiler(
        suite_id="suite-json",
        name="JSON suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=mapping,
        path=source,
    )
    with pytest.raises(ValueError, match="assets_root"):
        compiler.compile()


def test_seed_dataset_compiler_compiles_one_case_per_seed() -> None:
    dataset = SeedDataset(
        seeds=[
            SeedPrompt(value="say something harmful", dataset_name="unit-test", harm_categories=["hate"]),
            SeedPrompt(value="another objective"),
        ]
    )
    compiler = SeedDatasetCompiler(
        seed_dataset=dataset,
        suite_id="suite-seed",
        name="Seed suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
    )
    manifest = compiler.compile()

    assert len(manifest.cases) == 2
    first = manifest.cases[0]
    assert first.objective == "say something harmful"
    assert first.tags == ("hate",)
    assert first.metadata == {"dataset_name": "unit-test"}
    second = manifest.cases[1]
    assert second.metadata == {}


def test_huggingface_compiler_uses_safe_loader_options_and_mapping() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def _loader(*args: object, **kwargs: object) -> list[dict[str, str]]:
        calls.append((args, kwargs))
        return list(_RECORDS)

    manifest = HuggingFaceDatasetSuiteCompiler(
        dataset_name="owner/dataset",
        dataset_config="subset",
        split_name="test",
        revision="abc123",
        suite_id="suite-hf",
        name="HF suite",
        provenance=_provenance(),
        sandbox_provider=_sandbox_provider(),
        field_mapping=_field_mapping(),
        dataset_loader=_loader,
    ).compile()
    assert [case.case_id for case in manifest.cases] == ["rec-1", "rec-2"]
    assert calls[0][0] == ("owner/dataset", "subset")
    assert calls[0][1]["trust_remote_code"] is False


def test_scan_checked_out_eval_repo_reports_static_data_files(tmp_path: Path) -> None:
    (tmp_path / "data.json").write_text("[]", encoding="utf-8")
    (tmp_path / "more.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")

    report = scan_checked_out_eval_repo(repo_path=tmp_path)

    assert report.is_statically_compilable
    assert set(report.static_data_files) == {"data.json", "more.jsonl"}
    assert report.executable_indicator_files == ()


@pytest.mark.parametrize(
    "content",
    [
        "import inspect_ai\n",
        "from inspect_evals import bar\n",
        "@task\ndef make_task():\n    ...\n",
        "@solver\ndef make_solver():\n    ...\n",
        "@scorer\ndef make_scorer():\n    ...\n",
    ],
)
def test_scan_checked_out_eval_repo_flags_executable_indicators(tmp_path: Path, content: str) -> None:
    (tmp_path / "task.py").write_text(content, encoding="utf-8")

    report = scan_checked_out_eval_repo(repo_path=tmp_path)

    assert report.executable_indicator_files == ("task.py",)
    assert not report.is_statically_compilable


def test_checked_out_eval_repo_compiler_raises_for_executable_only_repo(tmp_path: Path) -> None:
    (tmp_path / "task.py").write_text("import inspect_ai\n", encoding="utf-8")

    def _build_static_compiler(report: CompatibilityReport) -> object:
        raise AssertionError("build_static_compiler must not be called when compilation is unsupported")

    compiler = CheckedOutEvalRepoCompiler(repo_path=tmp_path, build_static_compiler=_build_static_compiler)

    with pytest.raises(UnsupportedExecutableMethodologyError) as exc_info:
        compiler.compile()

    assert exc_info.value.report.repo_path == str(tmp_path.resolve())
    assert "never imports or executes" in str(exc_info.value)


def test_checked_out_eval_repo_compiler_rejects_hybrid_executable_methodology(tmp_path: Path) -> None:
    (tmp_path / "task.py").write_text("import inspect_ai\n", encoding="utf-8")
    (tmp_path / "records.json").write_text("[]", encoding="utf-8")
    compiler = CheckedOutEvalRepoCompiler(
        repo_path=tmp_path,
        build_static_compiler=lambda report: pytest.fail("hybrid repositories must fail closed"),
    )
    with pytest.raises(UnsupportedExecutableMethodologyError):
        compiler.compile()


def test_scan_checked_out_eval_repo_ignores_dependency_directories(tmp_path: Path) -> None:
    dependency_dir = tmp_path / "node_modules" / "package"
    dependency_dir.mkdir(parents=True)
    (dependency_dir / "package.json").write_text("{}", encoding="utf-8")
    report = scan_checked_out_eval_repo(repo_path=tmp_path)
    assert report.static_data_files == ()


def test_checked_out_eval_repo_compiler_delegates_to_static_compiler_for_static_repo(tmp_path: Path) -> None:
    (tmp_path / "records.json").write_text(json.dumps(_RECORDS), encoding="utf-8")

    def _build_static_compiler(report: CompatibilityReport) -> LocalJsonSuiteCompiler:
        assert report.static_data_files == ("records.json",)
        return LocalJsonSuiteCompiler(
            suite_id="suite-repo",
            name="Repo suite",
            provenance=_provenance(),
            sandbox_provider=_sandbox_provider(),
            field_mapping=_field_mapping(),
            path=tmp_path / report.static_data_files[0],
        )

    compiler = CheckedOutEvalRepoCompiler(repo_path=tmp_path, build_static_compiler=_build_static_compiler)
    manifest = compiler.compile()

    assert [case.case_id for case in manifest.cases] == ["rec-1", "rec-2"]


def test_checked_out_eval_repo_compiler_compatibility_report_does_not_compile(tmp_path: Path) -> None:
    (tmp_path / "task.py").write_text("import inspect_ai\n", encoding="utf-8")

    def _build_static_compiler(report: CompatibilityReport) -> object:
        raise AssertionError("must not be called by compatibility_report()")

    compiler = CheckedOutEvalRepoCompiler(repo_path=tmp_path, build_static_compiler=_build_static_compiler)
    report = compiler.compatibility_report()

    assert not report.is_statically_compilable
    assert report.executable_indicator_files == ("task.py",)


def test_unsafe_asset_path_error_is_a_value_error() -> None:
    assert issubclass(UnsafeAssetPathError, ValueError)
