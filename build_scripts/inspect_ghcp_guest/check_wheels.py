# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Verify the exact source-locked public wheels before local-only uv re-locking."""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


class OfflineWheelCheck:
    """Reject any wheel or package differing from the committed guest dependency lock."""

    def __init__(self, *, directory: Path, wheelhouse: Path) -> None:
        """Bind Docker's staged, no-egress dependency inputs."""
        self._directory = directory
        self._local = Path("/opt/pyrit/guest-local")
        self._wheelhouse = wheelhouse
        self._expected_path = directory / "original-packages.json"

    def before(self) -> None:
        """Verify every staged wheel against the original public PyPI-source uv.lock."""
        packages = self._packages(directory=self._directory)
        manifest = json.loads((self._wheelhouse / "manifest.json").read_text(encoding="utf-8"))
        wheels = manifest.get("wheels") if isinstance(manifest, dict) else None
        if not isinstance(wheels, list) or manifest.get("platform") != "cp312-manylinux_x86_64":
            raise ValueError("Guest wheel manifest must identify exact Linux CPython3.12 dependencies.")
        original = {entry["name"]: entry for entry in packages if entry["name"] != "pyrit-inspect-ghcp-guest"}
        if len(original) != len(wheels):
            raise ValueError("Wheelhouse differs from the locked guest dependency count.")
        verified: dict[str, str] = {}
        for item in wheels:
            if not isinstance(item, dict) or item.get("package") not in original:
                raise ValueError("Wheelhouse contains an unknown guest package.")
            name = item["package"]
            filename = item.get("filename")
            expected_hash = item.get("sha256")
            expected_size = item.get("size")
            if (
                name in verified
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or type(expected_size) is not int
                or original[name]["version"] != item.get("version")
                or not any(
                    wheel.get("url", "").endswith("/" + filename)
                    and wheel.get("hash") == f"sha256:{expected_hash}"
                    and wheel.get("size") == expected_size
                    for wheel in original[name].get("wheels", [])
                )
            ):
                raise ValueError(f"Guest wheel {name} differs from the original uv.lock.")
            path = self._wheelhouse / filename
            if not path.is_file() or path.stat().st_size != expected_size:
                raise ValueError(f"Guest wheel {name} is missing or has a different byte length.")
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != expected_hash:
                    raise ValueError(f"Guest wheel {name} does not match its uv.lock SHA256.")
            verified[name] = original[name]["version"]
        self._expected_path.write_text(json.dumps(verified, sort_keys=True), encoding="utf-8")

    def after(self) -> None:
        """Reject changed dependency versions after uv switches only to local wheel sources."""
        self.before()
        expected = json.loads(self._expected_path.read_text(encoding="utf-8"))
        actual = {
            entry["name"]: entry["version"]
            for entry in self._packages(directory=self._local)
            if entry["name"] != "pyrit-inspect-ghcp-guest"
        }
        if actual != expected:
            raise ValueError("Offline uv re-locking changed the approved guest dependency set.")

    @staticmethod
    def _packages(*, directory: Path) -> list[dict[str, Any]]:
        lock = tomllib.loads((directory / "uv.lock").read_text(encoding="utf-8"))
        return lock["package"]


def main() -> None:
    """Check the staged wheelhouse and offline lock against one fixed build path."""
    if sys.argv[1:] not in (["before"], ["after"]):
        raise ValueError("Use exactly one offline guest wheel check phase.")
    verifier = OfflineWheelCheck(
        directory=Path("/opt/pyrit/guest-deps"),
        wheelhouse=Path("/opt/pyrit/wheelhouse"),
    )
    if sys.argv[1] == "before":
        verifier.before()
    else:
        verifier.after()


if __name__ == "__main__":
    main()
