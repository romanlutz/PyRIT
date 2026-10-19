# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tomllib
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def _locked_uv_version() -> str:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    versions = {package["version"] for package in lock["package"] if package["name"] == "uv"}
    assert len(versions) == 1, versions
    return versions.pop()


def _setup_uv_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for path in sorted((REPO_ROOT / ".github" / "workflows").glob("*.yml")):
        workflow = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
        for job_name, job in workflow["jobs"].items():
            for index, step in enumerate(job.get("steps", [])):
                if step.get("uses", "").startswith("astral-sh/setup-uv@"):
                    versions[f"{path.name} {job_name} step {index}"] = step.get("with", {}).get("version", "")
    return versions


def test_locked_uv_matches_every_setup_uv_version() -> None:
    # pre-commit-uv builds the hook environments with the uv that uv.lock selects for the dev
    # dependencies, and under `uv run` it is also the uv that hooks call. When an update moves
    # uv in uv.lock, the setup-uv versions have to move with it.
    locked = _locked_uv_version()
    versions = _setup_uv_versions()
    assert versions
    assert versions == dict.fromkeys(versions, locked)
