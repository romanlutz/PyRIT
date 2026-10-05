# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline coverage for opt-in cohost image construction."""

import importlib.util
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def image_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("pyrit_image_builder", REPO_ROOT / "docker" / "build_pyrit_docker.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("cohost_original", [False, True])
def test_local_image_build_preserves_default_and_explicit_cohost_argument(
    *, image_builder: ModuleType, cohost_original: bool
) -> None:
    with (
        patch.object(image_builder, "build_devcontainer", return_value=True),
        patch.object(image_builder, "get_git_info", return_value=("a" * 40, False)),
        patch.object(image_builder.subprocess, "run", autospec=True) as run,
    ):
        run.return_value.returncode = 0
        image_builder.build_image(source="local", cohost_original=cohost_original)

    command = run.call_args.args[0]
    assert ("PYRIT_COHOST_ORIGINAL=true" in command) == cohost_original
    assert "PYRIT_SOURCE=local" in command
    assert "GIT_COMMIT=" + "a" * 40 in command
    assert "GIT_MODIFIED=false" in command
    tags = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-t"]
    assert tags == (["pyrit:cohost-" + "a" * 40] if cohost_original else ["pyrit:" + "a" * 40, "pyrit:latest"])
    assert ("--builder" in command) == cohost_original
    if cohost_original:
        assert command[command.index("--builder") + 1] == "default"


def test_cohost_pypi_is_rejected_before_any_docker_command(image_builder: ModuleType) -> None:
    with patch.object(image_builder, "build_devcontainer", autospec=True) as build_base:
        with pytest.raises(ValueError, match="matching uv.lock"):
            image_builder.build_image(source="pypi", version="1.2.0", cohost_original=True)
    build_base.assert_not_called()


def test_cohost_dirty_source_is_rejected_before_any_docker_command(image_builder: ModuleType) -> None:
    with (
        patch.object(image_builder, "get_git_info", return_value=("a" * 40, True)),
        patch.object(image_builder, "build_devcontainer", autospec=True) as build_base,
        patch.object(image_builder.subprocess, "run", autospec=True) as run,
    ):
        with pytest.raises(ValueError, match="clean committed source"):
            image_builder.build_image(source="local", cohost_original=True)
    build_base.assert_not_called()
    run.assert_not_called()


@pytest.mark.parametrize("changed", [("b" * 40, False), ("a" * 40, True)])
def test_cohost_source_change_during_base_build_denies_production(
    *, image_builder: ModuleType, changed: tuple[str, bool]
) -> None:
    with (
        patch.object(image_builder, "get_git_info", side_effect=[("a" * 40, False), changed]),
        patch.object(image_builder, "build_devcontainer", return_value=True),
        patch.object(image_builder.subprocess, "run", autospec=True) as run,
    ):
        with pytest.raises(ValueError, match="base-image build"):
            image_builder.build_image(source="local", cohost_original=True)
    run.assert_not_called()


def test_cohost_source_change_during_production_is_not_reported_as_success(
    image_builder: ModuleType, capsys: pytest.CaptureFixture[str]
) -> None:
    with (
        patch.object(
            image_builder,
            "get_git_info",
            side_effect=[("a" * 40, False), ("a" * 40, False), ("b" * 40, False)],
        ),
        patch.object(image_builder, "build_devcontainer", return_value=True),
        patch.object(image_builder.subprocess, "run", autospec=True) as run,
    ):
        run.return_value.returncode = 0
        with pytest.raises(ValueError, match="not qualified"):
            image_builder.build_image(source="local", cohost_original=True)
    assert run.call_count == 1
    assert "image built successfully" not in capsys.readouterr().out


def test_default_dirty_source_tags_are_unchanged(image_builder: ModuleType) -> None:
    with (
        patch.object(image_builder, "get_git_info", return_value=("a" * 40, True)),
        patch.object(image_builder, "build_devcontainer", return_value=True),
        patch.object(image_builder.subprocess, "run", autospec=True) as run,
    ):
        run.return_value.returncode = 0
        image_builder.build_image(source="local")
    command = run.call_args.args[0]
    tags = [command[index + 1] for index, value in enumerate(command[:-1]) if value == "-t"]
    assert tags == ["pyrit:" + "a" * 40 + "-modified", "pyrit:latest"]
    assert "GIT_MODIFIED=true" in command


def test_cohost_image_uses_separate_pinned_tooling_and_locked_public_dependencies() -> None:
    dockerfile = (REPO_ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    installer = (REPO_ROOT / "docker" / "install_cohost_tooling.sh").read_text(encoding="utf-8")
    startup = (REPO_ROOT / "docker" / "start.sh").read_text(encoding="utf-8")
    assert "ARG PYRIT_COHOST_ORIGINAL=false" in dockerfile
    assert "uv sync --locked --python /opt/venv/bin/python --extra inspect --extra cohost-original" in dockerfile
    assert "COPY --chown=vscode:vscode pyproject.toml uv.lock" in dockerfile
    assert "UV_PYTHON_DOWNLOADS=never" in dockerfile
    assert "cpython-3.12.11-linux-x86_64-gnu" in installer
    assert "741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf" in installer
    assert "sha256sum --check --status" in installer
    assert "/opt/pyrit-cohost" in installer
    assert "install_cohost_tooling" not in startup
    assert "python install" not in startup
