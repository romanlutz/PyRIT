# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _dev_uv_pin() -> str:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pins: list[str] = [dep for dep in pyproject["dependency-groups"]["dev"] if dep.startswith("uv==")]
    assert len(pins) == 1
    return pins[0].removeprefix("uv==")


def _setup_uv_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        for job_name, job in workflow["jobs"].items():
            for index, step in enumerate(job.get("steps", [])):
                if step.get("uses", "").startswith("astral-sh/setup-uv@"):
                    versions[f"{path.name} {job_name} step {index}"] = step.get("with", {}).get("version", "")
    return versions


def test_dev_uv_pin_matches_every_setup_uv_version() -> None:
    # pre-commit-uv builds the hook environments with the uv from the dev dependencies, and under
    # `uv run` it is also the uv that hooks call, so it has to stay on the version CI installs.
    pin = _dev_uv_pin()
    versions = _setup_uv_versions()
    assert versions
    assert versions == dict.fromkeys(versions, pin)
