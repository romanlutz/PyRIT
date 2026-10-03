# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from build_scripts.pyrit_wrapped.models import ItemKind, TaxonomyConfig, WorkItem
from build_scripts.pyrit_wrapped.taxonomy import Taxonomy


@pytest.mark.parametrize(
    ("path", "topic", "artifact", "surface"),
    [
        ("tests/unit/converter/test_example.py", "Converters", "Tests", "Python framework"),
        ("frontend/src/Chat.test.tsx", "Frontend", "Tests", "GUI"),
        ("pyrit/backend/routes/chat.py", "GUI backend", "Product code", "GUI"),
        ("doc/code/converters/example.ipynb", "Converters", "Documentation/examples", "Python framework"),
        ("pyrit/datasets/seed_datasets/test.yaml", "Datasets", "Dataset content", "Python framework"),
        ("pyrit/datasets/seed_datasets/test_prompts.yaml", "Datasets", "Dataset content", "Python framework"),
        ("pyrit/datasets/seed_datasets/provider.py", "Datasets", "Product code", "Python framework"),
        ("tests/unit/converter/fixture.md", "Converters", "Tests", "Python framework"),
        (".github/workflows/python.yml", "Build and CI", "Configuration/CI", "Tooling and infrastructure"),
    ],
)
def test_path_dimensions(
    *, path: str, topic: str, artifact: str, surface: str, item: WorkItem, taxonomy_config: TaxonomyConfig
) -> None:
    value = Taxonomy(taxonomy_config).classify(item.model_copy(update={"paths": [path]}))
    assert value.primary_topic == topic
    assert value.primary_artifact == artifact
    assert value.surface == surface
    assert not value.inferred


@pytest.mark.parametrize(
    ("title", "intent"),
    [
        ("FIX title", "FIX"),
        ("FIX: title", "FIX"),
        ("feat(scope): title", "FEAT"),
        ("[FEAT] title", "FEAT"),
        ("MAINT title", "MAINT"),
        ("fix(scope)!: breaking", "FIX"),
        ("not a known prefix", "Unknown"),
    ],
)
def test_intent_prefixes(*, title: str, intent: str, item: WorkItem, taxonomy_config: TaxonomyConfig) -> None:
    assert Taxonomy(taxonomy_config).classify(item.model_copy(update={"title": title})).intent == intent


def test_lockfiles_do_not_dominate(*, item: WorkItem, taxonomy_config: TaxonomyConfig) -> None:
    paths = ["pyrit/converter/example.py", "uv.lock", "frontend/package-lock.json"]
    value = Taxonomy(taxonomy_config).classify(item.model_copy(update={"paths": paths, "changed_files": 3}))
    assert value.primary_topic == "Converters"
    assert value.primary_artifact == "Product code"
    assert "Generated/lock files" in value.artifacts


def test_mixed_primary_and_overlapping_topics(*, item: WorkItem, taxonomy_config: TaxonomyConfig) -> None:
    value = Taxonomy(taxonomy_config).classify(
        item.model_copy(update={"paths": ["pyrit/converter/example.py", "pyrit/score/example.py"], "changed_files": 2})
    )
    assert value.primary_topic == "Mixed"
    assert value.topics == ["Converters", "Scorers"]


def test_incomplete_files_are_inferred(*, item: WorkItem, taxonomy_config: TaxonomyConfig) -> None:
    value = Taxonomy(taxonomy_config).classify(
        item.model_copy(update={"files_complete": False, "title": "GUI feature"})
    )
    assert value.inferred
    assert value.primary_artifact == "Unknown"
    assert value.languages == []


def test_issues_use_labels_and_inference(*, item: WorkItem, taxonomy_config: TaxonomyConfig) -> None:
    value = Taxonomy(taxonomy_config).classify(
        item.model_copy(update={"kind": ItemKind.ISSUE, "title": "Need a new scorer", "labels": ["GUI"]})
    )
    assert value.inferred
    assert value.topics == ["Frontend", "Scorers"]
    assert value.primary_topic == "Mixed"
