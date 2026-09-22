# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiofiles

from pyrit.models.submission_v2 import SubmissionProvenanceV2

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.models.submission_v2 import RetainedSubmissionReportV2, StrictSubmissionReportV2


class SubmissionEvidenceWriterV2:
    """Retain v2 candidates with the binding's validated, immutable provenance."""

    def __init__(self, *, directory: Path, provenance: SubmissionProvenanceV2) -> None:
        """Initialize a fresh owned run writer without touching its directory."""
        self.directory = directory
        self.provenance = provenance
        self._sequence = 0
        self._lock = asyncio.Lock()
        self._candidate: Path | None = None
        self._published = False

    async def initialize_async(self) -> None:
        """Create a fresh run directory and label its first event truthfully."""
        await asyncio.to_thread(self.directory.mkdir, parents=True, exist_ok=False)
        await self.append_async(event="run_started", data={})

    async def append_async(self, *, event: str, data: dict[str, Any]) -> None:
        """Flush one event without permitting payload fields to override provenance."""
        async with self._lock:
            self._sequence += 1
            record = {
                **data,
                "schema_version": 2,
                "mode": self.provenance.mode.value,
                "simulated": self.provenance.simulated,
                "evidence_label": self.provenance.evidence_label,
                "event_sequence": self._sequence,
                "time_utc": datetime.now(UTC).isoformat(),
                "event": event,
            }
            async with aiofiles.open(self.directory / "events.jsonl", "a", encoding="utf-8", newline="\n") as stream:
                await stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
                await stream.flush()

    async def snapshot_async(self, report: StrictSubmissionReportV2) -> None:
        """
        Retain an acquired binding projection without relabeling it.

        Raises:
            ValueError: If the binding changed provenance.
        """
        if report.provenance() != self.provenance:
            raise ValueError("A v2 run cannot change provenance after initialization.")
        await self.append_async(event="binding_report", data={"report": report.model_dump(mode="json")})

    async def retain_async(self, report: RetainedSubmissionReportV2) -> Path:
        """
        Write an unscored candidate before the final SQLite score transaction.

        Returns:
            Path: The owned candidate, not a commit receipt.

        Raises:
            RuntimeError: If a report was already published.
            ValueError: If report provenance changed.
        """
        if self._published:
            raise RuntimeError("Published v2 evidence cannot be replaced.")
        if SubmissionProvenanceV2(mode=report.mode, simulated=report.simulated) != self.provenance:
            raise ValueError("A v2 candidate cannot change provenance.")
        path = self.directory / f"{report.sha256()}.json"
        if path == self._candidate:
            return path
        async with aiofiles.open(path, "x", encoding="utf-8", newline="\n") as stream:
            await stream.write(report.canonical_json())
            await stream.flush()
        previous = self._candidate
        self._candidate = path
        if previous is not None:
            await asyncio.to_thread(previous.unlink)
        await self.append_async(
            event="report_prepared",
            data={"report_sha256": report.sha256(), "report_file": path.name, "publication": "unscored_candidate"},
        )
        return path

    def mark_published(self, path: Path) -> None:
        """
        Seal the exact owned candidate after its durable score commit.

        Raises:
            ValueError: If the path is not this run's candidate.
        """
        if path != self._candidate:
            raise ValueError("Only the owned v2 candidate may be published.")
        self._published = True
