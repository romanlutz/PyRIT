# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Pack raw HTML once and retrieve pinned documentation archives without rebuilding."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from build_scripts.resolve_docs_matrix import load_config


def _gh_json(args: list[str]) -> Any:
    return json.loads(subprocess.check_output(["gh", "api", *args], text=True))


def release_assets(*, repository: str, release_id: int) -> list[dict[str, Any]]:
    pages = _gh_json(["--paginate", "--slurp", f"repos/{repository}/releases/{release_id}/assets?per_page=100"])
    return [asset for page in pages for asset in page]


def _download(*, endpoint: str, target: Path, release_asset: bool = True) -> None:
    headers = ["-H", "Accept: application/octet-stream"] if release_asset else []
    with target.open("wb") as stream:
        subprocess.run(["gh", "api", *headers, endpoint], stdout=stream, check=True)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_identity(*, slug: str, source_sha: str, base: str) -> None:
    if not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", slug):
        raise ValueError(f"historical slug must be a release version: {slug!r}")
    if not re.fullmatch(r"[0-9a-f]{40}", source_sha):
        raise ValueError("source_sha must be a full immutable commit SHA")
    if not re.fullmatch(r"/[A-Za-z0-9_.-]+", base):
        raise ValueError("base must be a repository URL path, e.g. /PyRIT")


def _safe_name(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or path.is_absolute()
        or "\\" in name
        or ":" in name
        or any(part in ("", ".", "..") or part.endswith((".", " ")) for part in name.split("/"))
        or any(ord(char) < 32 for char in name)
        or any(re.fullmatch(r"(?i)(con|prn|aux|nul|com[0-9]|lpt[0-9])(\..*)?", part) for part in path.parts)
    ):
        raise ValueError(f"unsafe archive path: {name!r}")
    return path


def validate_site(site: Path) -> None:
    if site.is_symlink() or not (site / "index.html").is_file():
        raise ValueError(f"missing HTML site index: {site}")
    for path in site.rglob("*"):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError(f"links and special files are not allowed: {path}")
        _safe_name(path.relative_to(site).as_posix())
        if path.suffix == ".html" and b"<!-- pyrit-version-picker -->" in path.read_bytes():
            raise ValueError(f"archive must contain pre-picker HTML: {path}")


def pack(*, site: Path, output: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    validate_identity(slug=metadata["slug"], source_sha=metadata["source_sha"], base=metadata["base"])
    validate_provenance(metadata)
    if metadata.get("schema") != 1:
        raise ValueError("unsupported archive provenance schema")
    validate_site(site)
    if output.resolve().is_relative_to(site.resolve()):
        raise ValueError("archive output must be outside the raw HTML site")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite archive: {output}")
    with tarfile.open(output, "w:gz") as archive:
        data = json.dumps(metadata, sort_keys=True).encode("utf-8")
        info = tarfile.TarInfo("provenance.json")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        for path in sorted(site.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=f"html/{path.relative_to(site).as_posix()}", recursive=False)
    return {"sha256": digest(output), "source_sha": metadata["source_sha"], "provenance": metadata}


def unpack(*, archive_path: Path, target: Path, pin: dict[str, Any]) -> None:
    if digest(archive_path) != pin["sha256"]:
        raise ValueError(f"archive checksum mismatch: {archive_path}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite site: {target}")
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        seen: set[str] = set()
        for member in members:
            path = _safe_name(member.name)
            canonical = member.name.casefold()
            if canonical in seen or not member.isfile() or member.size > 2 * 1024**3:
                raise ValueError(f"duplicate, non-file or oversized archive member: {member.name}")
            if member.name == "provenance.json" and member.size > 1024**2:
                raise ValueError("archive provenance exceeds size limit")
            seen.add(canonical)
            if path.as_posix() != "provenance.json" and (path.parts[0] != "html" or len(path.parts) < 2):
                raise ValueError(f"unexpected archive member: {member.name}")
        if sum(member.size for member in members) > 8 * 1024**3:
            raise ValueError("archive exceeds the uncompressed size limit")
        provenance = archive.extractfile("provenance.json")
        if provenance is None or json.load(provenance) != pin["provenance"]:
            raise ValueError("archive provenance does not match the reviewed pin")
        # All paths and types are checked before writing. Never use extractall.
        with tempfile.TemporaryDirectory(dir=target.parent) as temporary:
            site = Path(temporary) / "html"
            for member in members:
                if member.name == "provenance.json":
                    continue
                relative = PurePosixPath(member.name).relative_to("html")
                path = site.joinpath(*relative.parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError(f"missing archive member: {member.name}")
                with source, path.open("xb") as destination:
                    shutil.copyfileobj(source, destination)
            validate_site(site)
            site.rename(target)


def validate_pins(*, cfg: dict[str, Any], base: str) -> None:
    store = cfg.get("archive_release", {})
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", store.get("repository", "")):
        raise ValueError("archive_release.repository is not configured; complete the documented bootstrap first")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", store.get("tag", "")):
        raise ValueError("archive_release.tag is not configured")
    for entry in cfg["versions"]:
        if entry["slug"] == "latest":
            continue
        pin = entry.get("archive")
        if not isinstance(pin, dict):
            raise ValueError(f"{entry['slug']}: missing archive pin; bootstrap required (no rebuild fallback)")
        if not isinstance(pin.get("asset_id"), int) or isinstance(pin["asset_id"], bool) or pin["asset_id"] <= 0:
            raise ValueError(f"{entry['slug']}: invalid release asset ID")
        if not re.fullmatch(r"[0-9a-f]{64}", pin.get("sha256", "")):
            raise ValueError(f"{entry['slug']}: invalid archive SHA-256")
        validate_identity(slug=entry["slug"], source_sha=pin.get("source_sha", ""), base=base)
        expected = {"schema": 1, "slug": entry["slug"], "source_sha": pin["source_sha"], "base": base}
        provenance = pin.get("provenance", {})
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{entry['slug']}: inconsistent archive identity/provenance")
        validate_provenance(provenance)


def validate_provenance(metadata: dict[str, Any]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", metadata.get("repository", "")):
        raise ValueError("provenance requires the source repository")
    for key in ("run_id", "run_attempt"):
        if not isinstance(metadata.get(key), int) or isinstance(metadata[key], bool) or metadata[key] <= 0:
            raise ValueError(f"provenance requires {key}")
    if not re.fullmatch(r"[0-9a-f]{40}", metadata.get("workflow_sha", "")):
        raise ValueError("provenance requires the immutable workflow SHA")
    if metadata.get("kind") not in ("build", "recovered"):
        raise ValueError("provenance kind must be build or recovered")
    if metadata["kind"] == "recovered":
        if not isinstance(metadata.get("artifact_id"), int) or metadata["artifact_id"] <= 0:
            raise ValueError("recovered provenance requires the original artifact ID")
        if not re.fullmatch(r"[0-9a-f]{64}", metadata.get("artifact_sha256", "")):
            raise ValueError("recovered provenance requires the original artifact digest")


def fetch_versions(*, cfg: dict[str, Any], artifacts_dir: Path, base: str) -> None:
    validate_pins(cfg=cfg, base=base)
    store = cfg["archive_release"]
    release = _gh_json([f"repos/{store['repository']}/releases/tags/{store['tag']}"])
    assets = {asset["id"]: asset for asset in release_assets(repository=store["repository"], release_id=release["id"])}
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        for entry in cfg["versions"]:
            if entry["slug"] == "latest":
                continue
            pin = entry["archive"]
            asset = assets.get(pin["asset_id"])
            if asset is None or asset["state"] != "uploaded":
                raise ValueError(f"{entry['slug']}: pinned archive asset missing from release")
            archive = Path(temporary) / f"{entry['slug']}.tar.gz"
            _download(
                endpoint=f"repos/{store['repository']}/releases/assets/{pin['asset_id']}",
                target=archive,
            )
            unpack(archive_path=archive, target=artifacts_dir / f"site-{entry['slug']}", pin=pin)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "fetch"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", type=Path, required=True)
        sub.add_argument("--base", required=True)
        if command == "fetch":
            sub.add_argument("--artifacts-dir", type=Path, required=True)
    sub = commands.add_parser("pack")
    sub.add_argument("--site", type=Path, required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "pack":
        metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
        validate_provenance(metadata)
        print(json.dumps(pack(site=args.site, output=args.output, metadata=metadata), indent=2))
    else:
        cfg = load_config(args.config)
        if args.command == "validate":
            validate_pins(cfg=cfg, base=args.base)
        else:
            fetch_versions(cfg=cfg, artifacts_dir=args.artifacts_dir, base=args.base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
