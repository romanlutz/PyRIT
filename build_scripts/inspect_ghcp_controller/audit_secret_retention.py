# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Audit a single Inspect/GHCP run's private artifacts without revealing its token."""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import codecs
import hashlib
import json
import re
import sys
import zipfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, TYPE_CHECKING
from urllib.parse import unquote_to_bytes

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface
    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot


class TokenAbsenceScanner:
    """Compare common literal/reversible token windows by SHA256 only."""

    TOKEN_WINDOW = re.compile(rb"(?=([A-Za-z0-9_-]{43}))")
    BASE64_WINDOW = re.compile(rb"(?=([A-Za-z0-9+/_-]{58}(?:==)?))")
    HEX_WINDOW = re.compile(rb"(?=([0-9a-fA-F]{86}))")
    UTF16_LE_WINDOW = re.compile(rb"(?=((?:[A-Za-z0-9_-]\x00){43}))")
    UTF16_BE_WINDOW = re.compile(rb"(?=((?:\x00[A-Za-z0-9_-]){43}))")
    JSON_ESCAPE_WINDOW = re.compile(rb"(?=((?:\\u00[0-7][0-9a-fA-F]){43}))")
    URL_ESCAPE_WINDOW = re.compile(rb"(?=((?:%[0-9a-fA-F]{2}){43}))")
    MAX_ARTIFACT_BYTES = 64 * 1024 * 1024

    def __init__(self, *, token_sha256: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", token_sha256):
            raise ValueError("The private controller receipt has no valid token fingerprint.")
        self._digest = bytes.fromhex(token_sha256)
        self.files_checked = 0
        self.archive_entries_checked = 0
        self.bytes_checked = 0

    def assert_absent(self, *, data: bytes, source: str) -> None:
        """
        Reject an exact token value or common reversible encodings.

        Raises:
            RuntimeError: If an artifact contains the scoped token; its value is never included.
        """
        for candidate in self.TOKEN_WINDOW.finditer(data):
            if hashlib.sha256(candidate[1]).digest() == self._digest:
                raise RuntimeError(f"Run token was retained in {source}; content withheld.")
        for candidate in self.BASE64_WINDOW.finditer(data):
            try:
                decoded = base64.b64decode(candidate[1].rstrip(b"=") + b"==", altchars=b"-_", validate=True)
            except (binascii.Error, ValueError):
                continue
            if hashlib.sha256(decoded).digest() == self._digest:
                raise RuntimeError(f"Encoded run token was retained in {source}; content withheld.")
        for pattern, decoder in (
            (self.HEX_WINDOW, lambda value: bytes.fromhex(value.decode("ascii"))),
            (self.UTF16_LE_WINDOW, lambda value: value[::2]),
            (self.UTF16_BE_WINDOW, lambda value: value[1::2]),
            (self.JSON_ESCAPE_WINDOW, lambda value: codecs.decode(value, "unicode_escape").encode("ascii")),
            (self.URL_ESCAPE_WINDOW, unquote_to_bytes),
        ):
            for candidate in pattern.finditer(data):
                decoded = decoder(candidate[1])
                if hashlib.sha256(decoded).digest() == self._digest:
                    raise RuntimeError(f"Encoded run token was retained in {source}; content withheld.")

    def scan_file(self, *, path: Path, source: str) -> None:
        """
        Scan bounded file bytes, including windows that cross read boundaries.

        Raises:
            ValueError: If a private artifact exceeds the approved audit bound.
        """
        with path.open("rb") as stream:
            self._scan_stream(stream=stream, source=source)
        self.files_checked += 1

    def scan_archive(self, *, path: Path) -> None:
        """Decompress and scan every Inspect EvalLog attachment without extracting it."""
        if not zipfile.is_zipfile(path):
            return
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                if entry.file_size > self.MAX_ARTIFACT_BYTES:
                    raise ValueError("An Inspect EvalLog attachment exceeds the audit byte bound.")
                with archive.open(entry) as stream:
                    self._scan_stream(stream=stream, source="Inspect EvalLog attachment")
                self.archive_entries_checked += 1

    def _scan_stream(self, *, stream: IO[bytes], source: str) -> None:
        total = 0
        tail = b""
        while chunk := stream.read(65_536):
            total += len(chunk)
            if total > self.MAX_ARTIFACT_BYTES:
                raise ValueError(f"{source} exceeds the approved audit byte bound.")
            self.assert_absent(data=tail + chunk, source=source)
            tail = (tail + chunk)[-257:]
        self.bytes_checked += total


def _fingerprint(*, run_dir: Path, run_id: str) -> str:
    stage = run_dir / "controller-stage.jsonl"
    rows = [json.loads(line) for line in stage.read_bytes().splitlines()]
    if not all(isinstance(row, dict) and row.get("run_id") == run_id for row in rows):
        raise ValueError("Controller stage receipts do not belong to this exact run.")
    hashes = [row.get("control_token_sha256") for row in rows if row.get("stage") == "episode_created"]
    if len(hashes) != 1 or not isinstance(hashes[0], str):
        raise ValueError("The controller did not retain one token-free pre-delivery fingerprint.")
    return hashes[0]


def _scan_sources(
    *, memory: MemoryInterface, run_id: str, scanner: TokenAbsenceScanner
) -> tuple[NativeCyberEpisodeSnapshot, dict[str, bytes]]:
    capture = memory.native_cyber_evidence
    snapshot = capture.get_episode(run_id=run_id)
    sources: dict[str, bytes] = {}
    for stream in snapshot.raw_streams:
        content = bytearray()
        cursor = 0
        while page := capture.read_raw_chunks(
            run_id=run_id, stream_id=stream.stream_id, allow_sensitive=True, after_sequence=cursor, limit=16
        ):
            content.extend(b"".join(chunk.data for chunk in page))
            cursor = page[-1].sequence
            if len(content) > TokenAbsenceScanner.MAX_ARTIFACT_BYTES:
                raise ValueError("A PyRIT source exceeded the bounded private audit.")
        if len(content) != stream.stored_bytes or (
            stream.stored_sha256 is not None and hashlib.sha256(content).hexdigest() != stream.stored_sha256
        ):
            raise ValueError("An Inspect source disagrees with its retained byte ledger or SHA256.")
        scanner.assert_absent(data=bytes(content), source="PyRIT run-scoped raw stream")
        scanner.bytes_checked += len(content)
        sources[stream.key.observed_source_id] = bytes(content)
    return snapshot, sources


def _verify_publication(
    *,
    memory: MemoryInterface,
    snapshot: NativeCyberEpisodeSnapshot,
    sources: dict[str, bytes],
    fingerprint: str,
    scanner: TokenAbsenceScanner,
) -> dict[str, object]:
    from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
    from pyrit.memory.memory_models import ScoreEntry
    from pyrit.models import ScoreStatus
    from pyrit.models.inspect_ghcp import InspectGhcpReport, InspectGhcpStatus

    control = sources.get(InspectGhcpEvidenceStore.CONTROL_KEY.observed_source_id, b"")
    rows = [json.loads(line) for line in control.splitlines()]
    if (
        not all(isinstance(row, dict) for row in rows)
        or [row.get("service") for row in rows] != ["model-bridge", "agent"]
        or any(row.get("frame_sha256") != fingerprint or row.get("frame_size_bytes") != 43 for row in rows)
    ):
        raise ValueError("Two token-free Inspect handoff receipts do not match the pre-delivery fingerprint.")
    if snapshot.finalized_at is None or snapshot.report_content_id is None or snapshot.score_id is None:
        raise RuntimeError("Inspect did not atomically publish a report and UND Score for this run.")
    stored = memory.get_scorable_content(content_ids=[snapshot.report_content_id])
    canonical = stored[snapshot.report_content_id].value
    scanner.assert_absent(data=canonical.encode("utf-8"), source="PyRIT canonical report")
    report = InspectGhcpReport.model_validate_json(canonical)
    if (
        report.schema_version != 3
        or report.status is not InspectGhcpStatus.INCOMPLETE
        or not report.token_files_absent_before_turn
        or report.control_receipt_sha256 != hashlib.sha256(control).hexdigest()
        or snapshot.report_sha256 != report.sha256()
        or snapshot.score_status is not ScoreStatus.UNDETERMINED
        or report.benchmark_verified
        or not report.provider_verified
        or report.turn_count != 2
        or report.model_http_200_count < 2
        or report.adversarial_http_200_count < 1
        or report.successful_tool_execution_count < 1
        or report.judgment is None
        or not report.stopped_before_scoring
        or not report.gateway_alive_before_scoring
        or not report.gateway_alive_after_scoring
        or not report.target_alive_before_scoring
        or not report.target_alive_after_scoring
        or not report.original_cleanup_called
        or not report.original_cleanup_succeeded
        or not report.sandbox_cleanup_observed
        or report.required_gaps != ("Original cyber task/scorer/target image were not independently verified.",)
    ):
        raise ValueError("The run did not publish one verified schema-3 benign/UND source outcome.")
    with closing(memory.get_session()) as session:
        score = session.get(ScoreEntry, snapshot.score_id)
        if (
            score is None
            or score.status != ScoreStatus.UNDETERMINED.value
            or score.score_value is not None
            or score.scorable_content_id != snapshot.report_content_id
        ):
            raise ValueError("The one linked PyRIT Score is not UND with a NULL numeric value.")
    return {
        "score_id": str(snapshot.score_id),
        "score_status": snapshot.score_status.value,
        "score_value_is_null": True,
        "report_sha256": report.sha256(),
        "control_receipts": len(rows),
        "sdk_events": report.sdk_event_count,
        "pyrit_turns": report.turn_count,
        "successful_tools": report.successful_tool_execution_count,
        "model_http_200": report.model_http_200_count,
        "adversarial_http_200": report.adversarial_http_200_count,
        "original_scorer_observed": report.judgment is not None,
        "required_gaps": list(report.required_gaps),
    }


def _audit(*, run_id: str, stdout: Path, stderr: Path, database: str = "protocol-smoke.db") -> dict[str, object]:
    from pyrit.memory import CentralMemory
    from pyrit.setup import SQLITE, initialize_pyrit_async

    if not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", run_id):
        raise ValueError("Inspect source audit requires one canonical run UUID.")
    if database not in {"protocol-smoke.db", "oneclick-protocol.db"}:
        raise ValueError("Inspect source audit requires one approved private SQLite file name.")
    root = (Path.cwd() / ".venv" / "inspect-ghcp").resolve(strict=True)
    run_dir = root / run_id
    if not run_dir.is_dir() or run_dir.is_symlink() or run_dir.resolve() != root / run_id:
        raise ValueError("The Inspect source directory is not the expected private run folder.")
    for path in (stdout, stderr):
        if not path.is_file() or path.is_symlink() or path.resolve().parent != root:
            raise ValueError("Controller stdout/stderr captures must be private files in this worktree.")
    fingerprint = _fingerprint(run_dir=run_dir, run_id=run_id)
    scanner = TokenAbsenceScanner(token_sha256=fingerprint)
    db = root / database
    if not db.is_file() or db.is_symlink():
        raise ValueError("The private PyRIT source database is missing or not a regular file.")
    asyncio.run(
        initialize_pyrit_async(memory_db_type=SQLITE, db_path=db, env_files=[], load_defaults=False, silent=True)
    )
    snapshot, sources = _scan_sources(memory=CentralMemory.get_memory_instance(), run_id=run_id, scanner=scanner)
    eval_files = []
    for artifact in sorted(run_dir.rglob("*")):
        if artifact.is_symlink():
            raise ValueError("A private Inspect artifact unexpectedly points outside the run folder.")
        if artifact.is_file():
            scanner.scan_file(path=artifact, source="Inspect/controller run artifact")
            if artifact.suffix == ".eval":
                eval_files.append(artifact)
                scanner.scan_archive(path=artifact)
    if len(eval_files) != 1:
        raise ValueError("The original Inspect run has no single auditable EvalLog archive.")
    for capture in (stdout, stderr):
        scanner.scan_file(path=capture, source="Controller stdout/stderr or error")
    for suffix in ("", "-wal", "-shm"):
        file = Path(str(db) + suffix)
        if file.exists():
            scanner.scan_file(path=file, source="PyRIT SQLite database or journal")
    publication = _verify_publication(
        memory=CentralMemory.get_memory_instance(),
        snapshot=snapshot,
        sources=sources,
        fingerprint=fingerprint,
        scanner=scanner,
    )
    return {
        "run_id": run_id,
        "at": datetime.now(UTC).isoformat(),
        "token_value_matches": 0,
        "files_checked": scanner.files_checked,
        "eval_archive_entries_checked": scanner.archive_entries_checked,
        "raw_streams_checked": len(snapshot.raw_streams),
        "bytes_checked": scanner.bytes_checked,
        **publication,
    }


def main() -> None:
    """Audit exactly one run without exposing the secret or its SHA in output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument(
        "--database", default="protocol-smoke.db", choices=["protocol-smoke.db", "oneclick-protocol.db"]
    )
    args = parser.parse_args()
    proof = _audit(run_id=args.run_id, stdout=args.stdout, stderr=args.stderr, database=args.database)
    output = Path.cwd() / ".venv" / "inspect-ghcp" / args.run_id / "token-absence-audit.json"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(proof, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps(proof, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"Inspect token audit failed ({type(error).__name__}); details withheld.", file=sys.stderr)
        raise SystemExit(1) from None
