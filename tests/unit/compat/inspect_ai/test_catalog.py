# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING

import pytest

from pyrit.cli import pyrit_capability_suite
from pyrit.compat.inspect_ai import cli as inspect_cli
from pyrit.compat.inspect_ai.catalog import (
    InspectCatalogRegressionError,
    InspectCatalogReport,
    _contains_classification_term,
    check_inspect_catalog_regression,
)
from pyrit.compat.inspect_ai.profile import PINNED_INSPECT_EVALS_PROFILE
from pyrit.compat.inspect_ai.source import InspectSourceVerification

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def catalog_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    arc = root / "src" / "inspect_evals" / "arc"
    arc.mkdir(parents=True)
    (arc / "eval.yaml").write_text(
        "title: ARC\n"
        "description: fixture\n"
        "group: Reasoning\n"
        "version: 1-A\n"
        "tasks:\n"
        "  - name: arc_easy\n"
        "    dataset_samples: 1\n",
        encoding="utf-8",
    )
    (arc / "arc.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.scorer import choice\n"
        "@task\n"
        "def arc_easy():\n"
        "    return Task(dataset=[], scorer=choice())\n",
        encoding="utf-8",
    )
    cloud = root / "src" / "inspect_evals" / "cloud_fixture"
    cloud.mkdir()
    (cloud / "eval.yaml").write_text(
        "title: Cloud fixture\n"
        "description: Requires Modal runtime\n"
        "group: Agentic\n"
        "version: 1-A\n"
        "tasks:\n"
        "  - name: cloud_fixture\n"
        "    dataset_samples: 1\n",
        encoding="utf-8",
    )
    (cloud / "cloud_fixture.py").write_text(
        "from inspect_ai import Task, task\n@task\ndef cloud_fixture():\n    return Task(dataset=[])\n",
        encoding="utf-8",
    )
    return root


def test_cloud_classification_uses_term_boundaries() -> None:
    assert _contains_classification_term(text="provider: modal", term="modal")
    assert not _contains_classification_term(text="supported modalities", term="modal")
    assert _contains_classification_term(text="google_cloud provider", term="google_cloud")


def test_catalog_regression_rejects_unreviewed_api_inventory() -> None:
    report = InspectCatalogReport(
        profile_id=PINNED_INSPECT_EVALS_PROFILE.profile_id,
        inspect_evals_revision=PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision,
        source_root="fixture",
        source_revision_verified=True,
        api_symbols_sha256="4515d2aba8bedf78de2c0ee866f44de6167ab92aa4eccf9d8fc321b2e789cd34",
        api_symbol_count=262,
        task_factory_count=249,
        families=(),
        excluded_cloud_surfaces=(),
        compatibility_claims=(),
    )

    with pytest.raises(InspectCatalogRegressionError, match="classify every added/removed API"):
        check_inspect_catalog_regression(report=dataclasses.replace(report, api_symbols_sha256="new-api"))


def test_build_catalog_classifies_supported_and_excluded_cloud_families(catalog_source_root: Path) -> None:
    from pyrit.compat.inspect_ai.catalog import build_inspect_catalog

    report = build_inspect_catalog(source_root=catalog_source_root, verify_source=False)
    families = {family.family: family for family in report.families}

    assert families["arc"].compatibility_status == "supported"
    assert families["cloud_fixture"].compatibility_status == "unsupported"
    assert any(blocker.startswith("modal provider/") for blocker in families["cloud_fixture"].blockers)
    modal = next(surface for surface in report.excluded_cloud_surfaces if surface.surface == "modal")
    assert modal.found_in_families == ("cloud_fixture",)


def test_catalog_cli_commands_emit_stable_outputs(
    catalog_source_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_path = tmp_path / "tasks.json"
    report_path = tmp_path / "report.json"
    catalog_path = tmp_path / "catalog.json"
    check_calls = []
    monkeypatch.setattr(
        inspect_cli,
        "check_inspect_catalog_regression",
        lambda *, report: check_calls.append(report),
    )

    assert (
        pyrit_capability_suite.main(
            [
                "inspect-evals",
                "tasks",
                "--source",
                str(catalog_source_root),
                "--no-verify-source",
                "--format",
                "json",
                "--output",
                str(tasks_path),
            ]
        )
        == 0
    )
    assert (
        pyrit_capability_suite.main(
            [
                "inspect-evals",
                "report",
                "--source",
                str(catalog_source_root),
                "--no-verify-source",
                "--format",
                "json",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert (
        pyrit_capability_suite.main(
            [
                "inspect-evals",
                "catalog",
                "--source",
                str(catalog_source_root),
                "--no-verify-source",
                "--check",
                "--format",
                "json",
                "--output",
                str(catalog_path),
            ]
        )
        == 0
    )

    tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
    assert [task["task_spec"] for task in tasks["tasks"]] == [
        "arc/arc.py@arc_easy",
        "cloud_fixture/cloud_fixture.py@cloud_fixture",
    ]
    assert json.loads(report_path.read_text(encoding="utf-8"))["families"][0]["family"] == "arc"
    assert json.loads(catalog_path.read_text(encoding="utf-8"))["task_factory_count"] == 2
    assert len(check_calls) == 1


@pytest.mark.parametrize(("action", "extra_args"), (("prepare", ("--offline",)), ("validate", ("--source", "fixture"))))
def test_source_cli_commands_emit_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    action: str,
    extra_args: tuple[str, ...],
) -> None:
    verification = InspectSourceVerification(
        source_root="fixture",
        repository="https://example.invalid/inspect_evals",
        revision=PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision,
        tree_hash="tree",
        license="MIT",
        license_sha256="license",
        clean=True,
    )
    monkeypatch.setattr(inspect_cli, "prepare_inspect_source", lambda **kwargs: verification)
    monkeypatch.setattr(inspect_cli, "validate_inspect_source", lambda **kwargs: verification)
    output = tmp_path / f"{action}.json"

    assert pyrit_capability_suite.main(["inspect-evals", "source", action, *extra_args, "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["revision"] == verification.revision
