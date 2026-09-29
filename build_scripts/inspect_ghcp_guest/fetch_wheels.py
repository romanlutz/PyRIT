# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Stage only uv.lock-pinned Linux CPython 3.12 wheels for an offline guest build."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import tomllib
from itertools import chain
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import urlopen

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


class LinuxWheelhouse:
    """Verify public wheel URLs, ABI compatibility and exact uv.lock SHA256s."""

    MAX_WHEEL_BYTES = 52_428_800

    def __init__(self, *, lock: Path, output: Path) -> None:
        """Keep generated artifacts under this worktree's ignored virtual environment."""
        workspace = Path.cwd().resolve()
        self._output = output.resolve()
        if not self._output.is_relative_to(workspace) or ".venv" not in self._output.parts:
            raise ValueError("Guest wheelhouse must be staged within this worktree's ignored .venv.")
        self._lock = lock
        platforms = [f"manylinux_2_{version}_x86_64" for version in range(39, 16, -1)]
        platforms.extend(["manylinux2014_x86_64", "linux_x86_64"])
        ordered = chain(
            cpython_tags(python_version=(3, 12), abis=["cp312"], platforms=platforms),
            compatible_tags(python_version=(3, 12), interpreter="cp312", platforms=platforms),
        )
        self._priority = {tag: index for index, tag in enumerate(ordered)}

    def select(self, *, package: dict[str, Any]) -> dict[str, Any]:
        """
        Pick the most compatible pinned Linux wheel, never an sdist or other ABI.

        Returns:
            dict[str, Any]: The exact wheel URL, length, and locked SHA256.

        Raises:
            ValueError: If no pinned wheel is compatible with CPython3.12 on Ubuntu x86_64.
        """
        candidates: list[tuple[int, dict[str, Any]]] = []
        for wheel in package.get("wheels", []):
            url = wheel.get("url")
            if not isinstance(url, str):
                continue
            filename = Path(urlsplit(url).path).name
            try:
                name, version, _, tags = parse_wheel_filename(filename)
            except ValueError:
                continue
            if canonicalize_name(name) != canonicalize_name(package["name"]) or str(version) != package["version"]:
                continue
            rank = min((self._priority[tag] for tag in tags if tag in self._priority), default=None)
            if rank is not None:
                candidates.append((rank, wheel))
        if not candidates:
            raise ValueError(f"No Linux CPython3.12 wheel is pinned for {package['name']}.")
        return min(candidates, key=lambda candidate: candidate[0])[1]

    def stage(self) -> list[dict[str, Any]]:
        """
        Download and verify every locked dependency without installing any package.

        Returns:
            list[dict[str, Any]]: Exact public wheel filenames, hashes and byte counts.
        """
        lock = tomllib.loads(self._lock.read_text(encoding="utf-8"))
        self._output.mkdir(parents=True, exist_ok=True)
        manifest: list[dict[str, Any]] = []
        for package in lock["package"]:
            if package["name"] == "pyrit-inspect-ghcp-guest":
                continue
            wheel = self.select(package=package)
            url = wheel["url"]
            parsed = urlsplit(url)
            checksum = wheel.get("hash", "")
            size = wheel.get("size")
            if (
                parsed.scheme != "https"
                or parsed.hostname != "files.pythonhosted.org"
                or parsed.username is not None
                or parsed.query
                or parsed.fragment
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", checksum)
                or type(size) is not int
                or not 0 < size <= self.MAX_WHEEL_BYTES
            ):
                raise ValueError(f"Unsafe or unpinned public wheel for {package['name']}.")
            filename = Path(parsed.path).name
            destination = self._output / filename
            if destination.exists():
                self._check(path=destination, expected_hash=checksum[7:], expected_size=size)
            else:
                self._download(url=url, destination=destination, expected_hash=checksum[7:], expected_size=size)
            manifest.append(
                {
                    "package": package["name"],
                    "version": package["version"],
                    "filename": filename,
                    "sha256": checksum[7:],
                    "size": size,
                    "url": url,
                }
            )
        (self._output / "manifest.json").write_text(
            json.dumps({"platform": "cp312-manylinux_x86_64", "wheels": manifest}, indent=2) + "\n",
            encoding="utf-8",
        )
        return manifest

    @staticmethod
    def _check(*, path: Path, expected_hash: str, expected_size: int) -> None:
        digest = hashlib.sha256()
        length = 0
        with path.open("rb") as stream:
            while chunk := stream.read(65_536):
                length += len(chunk)
                digest.update(chunk)
        if length != expected_size or digest.hexdigest() != expected_hash:
            raise ValueError(f"Wheel {path.name} differs from its exact uv.lock hash or size.")

    @classmethod
    def _download(cls, *, url: str, destination: Path, expected_hash: str, expected_size: int) -> None:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix="wheel-", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with urlopen(url, timeout=30) as source, temporary_path.open("wb") as target:
                while chunk := source.read(65_536):
                    target.write(chunk)
                    if target.tell() > cls.MAX_WHEEL_BYTES:
                        raise ValueError("Public wheel exceeded the approved size cap.")
            cls._check(path=temporary_path, expected_hash=expected_hash, expected_size=expected_size)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)


def main() -> None:
    """Stage public Linux wheels into the ignored worktree image-build context."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = Path(__file__).with_name("uv.lock")
    records = LinuxWheelhouse(lock=lock, output=args.output).stage()
    print(json.dumps({"wheel_count": len(records), "wheel_sha256": [row["sha256"] for row in records]}))


if __name__ == "__main__":
    main()
