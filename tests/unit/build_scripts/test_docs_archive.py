# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Offline archive integrity, identity, recovery and publication contracts."""

from __future__ import annotations

import copy
import io
import json
import stat
import subprocess
import tarfile
import zipfile
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

from build_scripts import compose_docs_dist, docs_archive, docs_archive_release, inject_version_picker

if TYPE_CHECKING:
    from pathlib import Path


def _metadata() -> dict[str, Any]:
    return {
        "schema": 1,
        "slug": "1.1.0",
        "source_sha": "a" * 40,
        "base": "/PyRIT",
        "kind": "build",
        "repository": "microsoft/PyRIT",
        "workflow_sha": "b" * 40,
        "run_id": 123,
        "run_attempt": 1,
    }


def _site(path: Path) -> Path:
    path.mkdir()
    (path / "index.html").write_text("<html><head></head><body>original</body></html>", encoding="utf-8")
    (path / ".hidden").mkdir()
    (path / ".hidden" / "asset.bin").write_bytes(b"\x00\xff")
    (path / "static").mkdir()
    (path / "static" / "image.png").write_bytes(b"image")
    return path


def _archive(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    archive = tmp_path / "docs.tar.gz"
    pin = docs_archive.pack(site=_site(tmp_path / "site"), output=archive, metadata=_metadata())
    return archive, {"asset_id": 42, **pin}


def _config(pin: dict[str, Any]) -> dict[str, Any]:
    return {
        "archive_release": {"repository": "microsoft/PyRIT", "tag": "docs-archives"},
        "default": "1.1.0",
        "stable": "1.1.0",
        "versions": [
            {"slug": "latest", "name": "latest", "ref": "main"},
            {"slug": "1.1.0", "name": "1.1.0", "ref": "releases/v1.1.0", "archive": pin},
        ],
    }


def test_pack_roundtrip_preserves_hidden_and_static_assets(tmp_path: Path) -> None:
    archive, pin = _archive(tmp_path)
    target = tmp_path / "restored"
    docs_archive.unpack(archive_path=archive, target=target, pin=pin)
    assert (target / ".hidden" / "asset.bin").read_bytes() == b"\x00\xff"
    assert (target / "static" / "image.png").read_bytes() == b"image"
    assert (target / "index.html").read_bytes() == (tmp_path / "site" / "index.html").read_bytes()
    with tarfile.open(archive) as source:
        assert json.load(source.extractfile("provenance.json")) == pin["provenance"]


def test_checksum_mismatch_never_creates_site(tmp_path: Path) -> None:
    archive, pin = _archive(tmp_path)
    archive.write_bytes(archive.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        docs_archive.unpack(archive_path=archive, target=tmp_path / "restored", pin=pin)
    assert not (tmp_path / "restored").exists()


def test_provenance_mismatch_never_creates_site(tmp_path: Path) -> None:
    archive, pin = _archive(tmp_path)
    pin["provenance"]["source_sha"] = "c" * 40
    with pytest.raises(ValueError, match="provenance"):
        docs_archive.unpack(archive_path=archive, target=tmp_path / "restored", pin=pin)
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize(
    "name",
    ["../escaped", "/absolute", "html/../../escaped", "html\\escaped", "html/C:/file", "html/NUL", "html/a./b"],
)
def test_archive_rejects_unsafe_paths(*, tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        member = tarfile.TarInfo(name)
        member.size = 1
        source.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe archive path"):
        docs_archive.unpack(
            archive_path=archive, target=tmp_path / "target", pin={"sha256": docs_archive.digest(archive)}
        )
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE, tarfile.CHRTYPE])
def test_archive_rejects_links_and_special_files(*, tmp_path: Path, kind: bytes) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        member = tarfile.TarInfo("html/link")
        member.type = kind
        member.linkname = "../../outside"
        source.addfile(member)
    with pytest.raises(ValueError, match="non-file"):
        docs_archive.unpack(
            archive_path=archive, target=tmp_path / "target", pin={"sha256": docs_archive.digest(archive)}
        )
    assert not (tmp_path / "target").exists()


@pytest.mark.parametrize("names", [("html/index.html", "html/index.html"), ("html/a", "html/A")])
def test_archive_rejects_duplicate_members(*, tmp_path: Path, names: tuple[str, str]) -> None:
    archive = tmp_path / "duplicate.tar.gz"
    with tarfile.open(archive, "w:gz") as source:
        for name in names:
            member = tarfile.TarInfo(name)
            member.size = 1
            source.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="duplicate"):
        docs_archive.unpack(
            archive_path=archive, target=tmp_path / "target", pin={"sha256": docs_archive.digest(archive)}
        )


def test_pack_rejects_composed_html_and_existing_archives(tmp_path: Path) -> None:
    archive, _ = _archive(tmp_path)
    site = tmp_path / "site"
    with pytest.raises(FileExistsError, match="overwrite"):
        docs_archive.pack(site=site, output=archive, metadata=_metadata())
    (site / "index.html").write_text(inject_version_picker.INJECT_MARKER, encoding="utf-8")
    with pytest.raises(ValueError, match="pre-picker"):
        docs_archive.pack(site=site, output=tmp_path / "new.tar.gz", metadata=_metadata())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("asset_id", 0),
        ("sha256", "bad"),
        ("source_sha", "main"),
        ("provenance", {}),
    ],
)
def test_pin_rejects_missing_or_mutable_identity(*, tmp_path: Path, field: str, value: Any) -> None:
    _, pin = _archive(tmp_path)
    pin[field] = value
    with pytest.raises(ValueError):
        docs_archive.validate_pins(cfg=_config(pin), base="/PyRIT")


def test_pin_checks_slug_base_source_and_build_provenance(tmp_path: Path) -> None:
    _, pin = _archive(tmp_path)
    for field in ("slug", "base", "source_sha", "run_id", "run_attempt", "workflow_sha", "kind", "repository"):
        wrong = copy.deepcopy(pin)
        wrong["provenance"][field] = "wrong"
        with pytest.raises(ValueError):
            docs_archive.validate_pins(cfg=_config(wrong), base="/PyRIT")


def test_unbootstrapped_configuration_fails_closed(tmp_path: Path) -> None:
    _, pin = _archive(tmp_path)
    cfg = _config(pin)
    del cfg["versions"][1]["archive"]
    with pytest.raises(ValueError, match="bootstrap required"):
        docs_archive.fetch_versions(cfg=cfg, artifacts_dir=tmp_path / "artifacts", base="/PyRIT")
    del cfg["archive_release"]
    with pytest.raises(ValueError, match="bootstrap first"):
        docs_archive.validate_pins(cfg=cfg, base="/PyRIT")


def test_fetch_checks_release_membership_and_missing_assets(tmp_path: Path) -> None:
    _, pin = _archive(tmp_path)
    with (
        patch.object(docs_archive, "_gh_json", return_value={"id": 5}),
        patch.object(docs_archive, "release_assets", return_value=[]),
    ):
        with pytest.raises(ValueError, match="missing from release"):
            docs_archive.fetch_versions(cfg=_config(pin), artifacts_dir=tmp_path / "artifacts", base="/PyRIT")


def test_fetch_corruption_aborts_before_composition(tmp_path: Path) -> None:
    archive, pin = _archive(tmp_path)

    def download(*, endpoint: str, target: Path) -> None:
        assert endpoint.endswith("/assets/42")
        target.write_bytes(b"corrupt")

    with (
        patch.object(docs_archive, "_gh_json", return_value={"id": 5}),
        patch.object(docs_archive, "release_assets", return_value=[{"id": 42, "state": "uploaded"}]),
        patch.object(docs_archive, "_download", side_effect=download),
    ):
        with pytest.raises(ValueError, match="checksum mismatch"):
            docs_archive.fetch_versions(cfg=_config(pin), artifacts_dir=tmp_path / "artifacts", base="/PyRIT")
    assert archive.is_file()
    assert not (tmp_path / "artifacts" / "site-1.1.0").exists()


def test_fetch_frozen_source_and_compose_current_picker(tmp_path: Path) -> None:
    archive, pin = _archive(tmp_path)
    cfg = _config(pin)
    cfg["versions"][1]["ref"] = "releases/branch-moved"

    def download(*, endpoint: str, target: Path) -> None:
        assert endpoint == "repos/microsoft/PyRIT/releases/assets/42"
        target.write_bytes(archive.read_bytes())

    artifacts = tmp_path / "artifacts"
    with (
        patch.object(docs_archive, "_gh_json", return_value={"id": 5}),
        patch.object(docs_archive, "release_assets", return_value=[{"id": 42, "state": "uploaded"}]),
        patch.object(docs_archive, "_download", side_effect=download),
    ):
        docs_archive.fetch_versions(cfg=cfg, artifacts_dir=artifacts, base="/PyRIT")
    _site(artifacts / "site-latest")
    config = tmp_path / "versions.json"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    dist = tmp_path / "dist"
    assert (
        compose_docs_dist.main(
            ["--config", str(config), "--artifacts-dir", str(artifacts), "--dist-dir", str(dist), "--base", "/PyRIT"]
        )
        == 0
    )
    for _ in range(2):
        assert inject_version_picker.main(["--site-dir", str(dist), "--base", "/PyRIT"]) == 0
    assert (dist / "1.1.0" / "index.html").read_text(encoding="utf-8").count(inject_version_picker.INJECT_MARKER) == 1
    assert (dist / "1.1.0" / ".hidden" / "asset.bin").read_bytes() == b"\x00\xff"
    assert (dist / "1.1.0" / "pages.json").is_file()
    assert "url=/PyRIT/1.1.0/" in (dist / "index.html").read_text(encoding="utf-8")
    assert "url=/PyRIT/1.1.0/" in (dist / "stable" / "index.html").read_text(encoding="utf-8")


@pytest.mark.parametrize("slugs", [["latest"], ["latest", "1.1.0", "extra"]])
def test_composition_requires_exact_version_set(*, tmp_path: Path, slugs: list[str]) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for slug in slugs:
        _site(artifacts / f"site-{slug}")
    config = tmp_path / "versions.json"
    config.write_text(json.dumps(_config({})), encoding="utf-8")
    dist = tmp_path / "dist"
    assert (
        compose_docs_dist.main(
            ["--config", str(config), "--artifacts-dir", str(artifacts), "--dist-dir", str(dist), "--base", "/PyRIT"]
        )
        == 1
    )
    assert not (dist / "versions.json").exists()


@pytest.mark.parametrize("name", ["../outside", "/absolute", "C:\\outside"])
def test_recovery_zip_rejects_traversal(*, tmp_path: Path, name: str) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(name, "bad")
    with pytest.raises(ValueError, match="unsafe"):
        docs_archive_release._extract_artifact(archive=archive, site=tmp_path / "site")
    assert not (tmp_path / "site").exists()


def test_recovery_zip_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    member = zipfile.ZipInfo("link")
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive, "w") as source:
        source.writestr(member, "../outside")
    with pytest.raises(ValueError, match="link"):
        docs_archive_release._extract_artifact(archive=archive, site=tmp_path / "site")


@pytest.mark.parametrize(
    "change",
    [
        {"event": "pull_request"},
        {"conclusion": "failure"},
        {"head_branch": "feature"},
        {"path": ".github/workflows/other.yml"},
        {"head_repository": {"full_name": "fork/PyRIT"}},
    ],
)
def test_recovery_requires_trusted_successful_main_run(*, tmp_path: Path, change: dict[str, Any]) -> None:
    run = {
        "event": "push",
        "conclusion": "success",
        "head_branch": "main",
        "path": ".github/workflows/docs.yml",
        "repository": {"full_name": "microsoft/PyRIT"},
        "head_repository": {"full_name": "microsoft/PyRIT"},
        **change,
    }
    with patch.object(docs_archive_release, "_gh_json", return_value=run):
        with pytest.raises(ValueError, match="trusted main"):
            docs_archive_release.recover(
                repository="microsoft/PyRIT", run_id=123, slug="1.1.0", base="/PyRIT", output_dir=tmp_path
            )


def test_publication_never_clobbers_or_creates_releases(tmp_path: Path) -> None:
    archive, pin = _archive(tmp_path)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")
    release = {"draft": False, "tag_name": "docs-archives", "assets": []}
    with (
        patch.object(docs_archive_release, "_gh_json", return_value=release),
        patch.object(
            docs_archive_release.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "upload")
        ) as run,
    ):
        with pytest.raises(subprocess.CalledProcessError):
            docs_archive_release.publish(
                repository="microsoft/PyRIT", tag="docs-archives", archive=archive, pin_path=pin_path
            )
    command = run.call_args.args[0]
    assert command == ["gh", "release", "upload", "docs-archives", str(archive), "--repo", "microsoft/PyRIT"]
    assert "--clobber" not in command
    with pytest.raises(ValueError, match="dedicated"):
        docs_archive_release.publish(repository="microsoft/PyRIT", tag="v1.1.0", archive=archive, pin_path=pin_path)


def test_release_assets_reads_all_pages() -> None:
    with patch.object(docs_archive, "_gh_json", return_value=[[{"id": 1}], [{"id": 2}]]) as request:
        assert docs_archive.release_assets(repository="microsoft/PyRIT", release_id=5) == [{"id": 1}, {"id": 2}]
    assert request.call_args.args[0] == [
        "--paginate",
        "--slurp",
        "repos/microsoft/PyRIT/releases/5/assets?per_page=100",
    ]


@pytest.mark.parametrize("corrupt_upload", [False, True])
def test_publication_verifies_new_asset_and_returns_actual_id(*, tmp_path: Path, corrupt_upload: bool) -> None:
    archive, pin = _archive(tmp_path)
    pin_path = tmp_path / "pin.json"
    pin_path.write_text(json.dumps(pin), encoding="utf-8")

    def download(*, endpoint: str, target: Path) -> None:
        assert endpoint.endswith("/assets/99")
        target.write_bytes(b"bad" if corrupt_upload else archive.read_bytes())

    with (
        patch.object(
            docs_archive_release, "_gh_json", return_value={"id": 5, "draft": False, "tag_name": "docs-archives"}
        ),
        patch.object(docs_archive_release, "release_assets", return_value=[{"id": 99, "name": archive.name}]),
        patch.object(docs_archive_release, "_download", side_effect=download),
        patch.object(docs_archive_release.subprocess, "run"),
    ):
        if corrupt_upload:
            with pytest.raises(ValueError, match="uploaded release asset checksum"):
                docs_archive_release.publish(
                    repository="microsoft/PyRIT", tag="docs-archives", archive=archive, pin_path=pin_path
                )
        else:
            result = docs_archive_release.publish(
                repository="microsoft/PyRIT", tag="docs-archives", archive=archive, pin_path=pin_path
            )
            assert result["asset_id"] == 99
            assert result["sha256"] == pin["sha256"]


@pytest.mark.parametrize("failure", ["none", "expired", "missing", "corrupt", "missing-sha", "wrong-base"])
def test_recovery_verifies_original_output_and_checkout(*, tmp_path: Path, failure: str) -> None:
    original = tmp_path / "original.zip"
    with zipfile.ZipFile(original, "w") as archive:
        archive.writestr("index.html", "<html><head></head></html>")
        archive.writestr(".hidden/data", b"\xff")
    run = {
        "conclusion": "success",
        "event": "push",
        "head_branch": "main",
        "path": ".github/workflows/docs.yml",
        "repository": {"full_name": "microsoft/PyRIT"},
        "head_repository": {"full_name": "microsoft/PyRIT"},
        "head_sha": "b" * 40,
        "run_attempt": 1,
    }
    jobs = {"jobs": [{"id": 456, "name": "Build 1.1.0", "conclusion": "success"}]}
    artifacts = {
        "artifacts": []
        if failure == "missing"
        else [
            {
                "id": 789,
                "name": "site-1.1.0",
                "expired": failure == "expired",
                "digest": "sha256:" + docs_archive.digest(original),
            }
        ]
    }
    logs = "time BASE_URL: /PyRIT/1.1.0\n"
    if failure != "missing-sha":
        logs += "time git log -1 --format=%H\ntime " + "a" * 40 + "\n"
    if failure == "wrong-base":
        logs = logs.replace("/PyRIT/1.1.0", "/Other/1.1.0")

    def download(*, endpoint: str, target: Path, release_asset: bool) -> None:
        assert not release_asset
        assert endpoint.endswith("/artifacts/789/zip")
        target.write_bytes(b"corruption" if failure == "corrupt" else original.read_bytes())

    with (
        patch.object(docs_archive_release, "_gh_json", side_effect=[run, jobs, artifacts]),
        patch.object(docs_archive_release.subprocess, "check_output", return_value=logs),
        patch.object(docs_archive_release, "_download", side_effect=download),
    ):
        if failure != "none":
            with pytest.raises(ValueError):
                docs_archive_release.recover(
                    repository="microsoft/PyRIT", run_id=123, slug="1.1.0", base="/PyRIT", output_dir=tmp_path / "out"
                )
            assert not list(tmp_path.rglob("*.tar.gz"))
        else:
            archive = docs_archive_release.recover(
                repository="microsoft/PyRIT", run_id=123, slug="1.1.0", base="/PyRIT", output_dir=tmp_path / "out"
            )
            pin = json.loads(archive.with_suffix(".json").read_text(encoding="utf-8"))
            assert pin["source_sha"] == "a" * 40
            assert pin["provenance"]["workflow_sha"] == "b" * 40
            assert pin["provenance"]["artifact_id"] == 789
            docs_archive.unpack(archive_path=archive, target=tmp_path / "restored", pin=pin)
            assert (tmp_path / "restored" / ".hidden" / "data").read_bytes() == b"\xff"


def test_composition_rejects_stale_unconfigured_dist_entries(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    _site(artifacts / "site-latest")
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "old-version").mkdir()
    with pytest.raises(ValueError, match="unconfigured"):
        compose_docs_dist.stage_versions(artifacts_dir=artifacts, dist_dir=dist, expected={"latest"})
    assert not (dist / "latest").exists()
