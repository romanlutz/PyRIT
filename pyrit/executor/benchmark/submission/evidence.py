# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiofiles

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.models.submission import RetainedSubmissionReport, StrictSubmissionReport


class SubmissionEvidenceWriter:
    """Append offline events and retain exact, immutable report content."""

    def __init__(self, *, directory: Path) -> None:
        """Initialize a writer for a fresh run directory."""
        self.directory = directory
        self._lock = asyncio.Lock()
        self._sequence = 0
        self._candidate_path: Path | None = None
        self._published = False

    async def initialize_async(self) -> None:
        """Create a new owned evidence directory, never overwriting an earlier run."""
        await asyncio.to_thread(self.directory.mkdir, parents=True, exist_ok=False)
        await self.append_async(event="run_started", data={})

    async def append_async(self, *, event: str, data: dict[str, Any]) -> None:
        """Flush one labeled journal entry before the caller proceeds."""
        async with self._lock:
            self._sequence += 1
            entry = {
                "evidence_label": "OFFLINE/SIMULATED",
                "event_sequence": self._sequence,
                "time_utc": datetime.now(UTC).isoformat(),
                "event": event,
                **data,
            }
            async with aiofiles.open(self.directory / "events.jsonl", "a", encoding="utf-8", newline="\n") as stream:
                await stream.write(json.dumps(entry, ensure_ascii=True, allow_nan=False) + "\n")
                await stream.flush()

    async def snapshot_async(self, report: StrictSubmissionReport) -> None:
        """Retain a binding snapshot without changing the binding's selection."""
        await self.append_async(event="binding_report", data={"report": report.model_dump(mode="json")})

    async def retain_async(self, report: RetainedSubmissionReport) -> Path:
        """
        Prepare an unscored report candidate under its content digest.

        Returns:
            Path: The exact candidate file, published only when its score is committed.

        Raises:
            RuntimeError: If this writer already published a report.
        """
        if self._published:
            raise RuntimeError("Published report evidence cannot be replaced.")
        path = self.directory / f"{report.sha256()}.json"
        if path == self._candidate_path:
            return path
        async with aiofiles.open(path, "x", encoding="utf-8", newline="\n") as stream:
            await stream.write(report.canonical_json())
            await stream.flush()
        previous_candidate = self._candidate_path
        self._candidate_path = path
        if previous_candidate is not None:
            await asyncio.to_thread(previous_candidate.unlink)
        await self.append_async(
            event="report_prepared",
            data={"report_sha256": report.sha256(), "report_file": path.name, "publication": "unscored_candidate"},
        )
        return path

    def mark_published(self, path: Path) -> None:
        """
        Seal the candidate immediately after the public score transaction commits.

        Raises:
            ValueError: If the published path does not name this writer's candidate.
        """
        if path != self._candidate_path:
            raise ValueError("Only this writer's owned candidate may be published.")
        self._published = True
