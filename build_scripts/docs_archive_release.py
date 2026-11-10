# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Explicit, trusted recovery/publication of one immutable documentation archive."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from build_scripts.docs_archive import (
    _download,
    _gh_json,
    _safe_name,
    digest,
    pack,
    release_assets,
    unpack,
    validate_identity,
    validate_provenance,
)


def _extract_artifact(*, archive: Path, site: Path) -> None:
    with zipfile.ZipFile(archive) as source:
        members = source.infolist()
        seen: set[str] = set()
        if sum(member.file_size for member in members) > 8 * 1024**3:
            raise ValueError("artifact exceeds the uncompressed size limit")
        for member in members:
            _safe_name(member.filename.rstrip("/") if member.is_dir() else member.filename)
            mode = member.external_attr >> 16
            kind = stat.S_IFMT(mode)
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR) or member.filename.casefold() in seen:
                raise ValueError(f"duplicate, link or special artifact member: {member.filename}")
            seen.add(member.filename.casefold())
        site.mkdir()
        for member in members:
            path = site.joinpath(*_safe_name(member.filename.rstrip("/")).parts)
            if member.is_dir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as content, path.open("xb") as target:
                    shutil.copyfileobj(content, target)


def recover(*, repository: str, run_id: int, slug: str, base: str, output_dir: Path) -> Path:
    validate_identity(slug=slug, source_sha="0" * 40, base=base)
    run = _gh_json([f"repos/{repository}/actions/runs/{run_id}"])
    if (
        run["conclusion"] != "success"
        or run["event"] not in ("push", "workflow_dispatch")
        or run["head_branch"] != "main"
        or run["path"] != ".github/workflows/docs.yml"
        or run["repository"]["full_name"] != repository
        or run["head_repository"]["full_name"] != repository
    ):
        raise ValueError("recovery requires a successful trusted main docs run, not a PR/fork run")
    attempt = run["run_attempt"]
    pages = _gh_json([f"repos/{repository}/actions/runs/{run_id}/attempts/{attempt}/jobs?per_page=100"])
    jobs = [job for job in pages["jobs"] if job["name"] == f"Build {slug}" and job["conclusion"] == "success"]
    if len(jobs) != 1:
        raise ValueError(f"cannot identify successful checkout job for {slug}")
    logs = subprocess.check_output(
        ["gh", "api", "--allow-escape-sequences", f"repos/{repository}/actions/jobs/{jobs[0]['id']}/logs"], text=True
    )
    logs = re.sub(r"\x1b\[[0-9;]*m", "", logs)
    if not re.search(rf"\bBASE_URL: {re.escape(base)}/{re.escape(slug)}\s*(?:\r?\n|$)", logs):
        raise ValueError(f"cannot verify original base URL for {slug}")
    matches = re.findall(r"git log -1 --format=%H\r?\n[^\n]*? ([0-9a-f]{40})\s*(?:\r?\n|$)", logs)
    if len(matches) != 1:
        raise ValueError(f"cannot establish exact historical checkout SHA from {slug} job logs")
    source_sha = matches[0]
    validate_identity(slug=slug, source_sha=source_sha, base=base)
    pages = _gh_json([f"repos/{repository}/actions/runs/{run_id}/artifacts?per_page=100"])
    artifacts = [artifact for artifact in pages["artifacts"] if artifact["name"] == f"site-{slug}"]
    if len(artifacts) != 1 or artifacts[0]["expired"]:
        raise ValueError(f"complete site-{slug} artifact is missing or expired; explicit rebuild required")
    artifact = artifacts[0]
    expected_digest = artifact.get("digest", "").removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
        raise ValueError("original Actions artifact has no SHA-256 digest")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"docs-{slug}-{source_sha}-run{run_id}-{attempt}.tar.gz"
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"recovery output already exists: {output}")
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "artifact.zip"
        _download(
            endpoint=f"repos/{repository}/actions/artifacts/{artifact['id']}/zip", target=archive, release_asset=False
        )
        if digest(archive) != expected_digest:
            raise ValueError("downloaded Actions artifact checksum mismatch")
        site = Path(temporary) / "site"
        _extract_artifact(archive=archive, site=site)
        metadata = {
            "schema": 1,
            "slug": slug,
            "source_sha": source_sha,
            "base": base,
            "kind": "recovered",
            "repository": repository,
            "workflow_sha": run["head_sha"],
            "run_id": run_id,
            "run_attempt": attempt,
            "artifact_id": artifact["id"],
            "artifact_sha256": expected_digest,
        }
        pin = pack(site=site, output=output, metadata=metadata)
    output.with_suffix(".json").write_text(json.dumps(pin, indent=2) + "\n", encoding="utf-8")
    return output


def publish(*, repository: str, tag: str, archive: Path, pin_path: Path) -> dict[str, Any]:
    if not re.fullmatch(r"docs-archives(?:-[A-Za-z0-9_.-]+)?", tag):
        raise ValueError("only a dedicated docs-archives storage release may receive archives")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    validate_provenance(pin["provenance"])
    if pin["provenance"]["repository"] != repository:
        raise ValueError("archive source must be in the destination repository")
    validate_identity(slug=pin["provenance"]["slug"], source_sha=pin["source_sha"], base=pin["provenance"]["base"])
    if pin["provenance"]["source_sha"] != pin["source_sha"]:
        raise ValueError("source SHA does not match provenance")
    with tempfile.TemporaryDirectory() as temporary:
        unpack(archive_path=archive, target=Path(temporary) / "site", pin=pin)
    # The storage release must already exist. Never create/move a tag here.
    release = _gh_json([f"repos/{repository}/releases/tags/{tag}"])
    if release["draft"] or release["tag_name"] != tag:
        raise ValueError("archive storage release must be published")
    subprocess.run(
        ["gh", "release", "upload", tag, str(archive), "--repo", repository],
        check=True,
        stdout=sys.stderr,
    )
    assets = [
        asset
        for asset in release_assets(repository=repository, release_id=release["id"])
        if asset["name"] == archive.name
    ]
    if len(assets) != 1:
        raise ValueError("cannot identify newly uploaded archive")
    with tempfile.TemporaryDirectory() as temporary:
        downloaded = Path(temporary) / "uploaded.tar.gz"
        _download(endpoint=f"repos/{repository}/releases/assets/{assets[0]['id']}", target=downloaded)
        if digest(downloaded) != pin["sha256"]:
            raise ValueError("uploaded release asset checksum mismatch; do not activate this pin")
    return {**pin, "asset_id": assets[0]["id"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sub = commands.add_parser("recover")
    sub.add_argument("--repository", required=True)
    sub.add_argument("--run-id", type=int, required=True)
    sub.add_argument("--slug", required=True)
    sub.add_argument("--base", required=True)
    sub.add_argument("--output-dir", type=Path, required=True)
    sub = commands.add_parser("publish")
    sub.add_argument("--repository", required=True)
    sub.add_argument("--tag", required=True)
    sub.add_argument("--archive", type=Path, required=True)
    sub.add_argument("--pin", type=Path, required=True)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", args.repository):
        raise ValueError("invalid repository")
    if args.command == "recover":
        print(
            recover(
                repository=args.repository,
                run_id=args.run_id,
                slug=args.slug,
                base=args.base,
                output_dir=args.output_dir,
            )
        )
    else:
        pin = publish(repository=args.repository, tag=args.tag, archive=args.archive, pin_path=args.pin)
        print(json.dumps(pin, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
