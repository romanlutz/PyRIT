# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import fnmatch
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def publication_patterns() -> list[str]:
    template = yaml.safe_load((REPO_ROOT / ".azuredevops" / "test-job-template.yml").read_text(encoding="utf-8"))
    publishers = [step for step in template["jobs"][0]["steps"] if step.get("task") == "PublishTestResults@2"]
    assert len(publishers) == 1
    publisher = publishers[0]
    assert publisher["condition"] == "always()"
    assert publisher["inputs"]["testResultsFormat"] == "JUnit"
    assert publisher["inputs"]["mergeTestResults"] is True
    patterns: object = publisher["inputs"]["testResultsFiles"]
    assert isinstance(patterns, str)
    return patterns.splitlines()


def test_partner_makefile_output_is_published(publication_patterns: list[str]) -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    target = re.search(r"(?m)^partner-integration-test:\n\t.*--junitxml=(\S+)", makefile)
    assert target is not None
    output = target.group(1)
    assert any(fnmatch.fnmatchcase(output, pattern) for pattern in publication_patterns)


@pytest.mark.parametrize(
    ("path", "published"),
    [
        ("junit/test-results.xml", True),
        ("junit/test-results-1.xml", True),
        ("junit/test-results-12.xml", True),
        ("junit/other-results.xml", False),
        ("junit/partner-test-results.txt", False),
        ("other/partner-test-results.xml", False),
    ],
)
def test_publication_preserves_existing_filename_scope(
    path: str, published: bool, publication_patterns: list[str]
) -> None:
    assert any(fnmatch.fnmatchcase(path, pattern) for pattern in publication_patterns) is published
