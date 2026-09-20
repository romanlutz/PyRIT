# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiofiles

if TYPE_CHECKING:
    from pathlib import Path


class InspectRunArtifacts:
    """Retain adapter evidence separately from canonical PyRIT messages and scores."""

    def __init__(self, *, directory: Path, provenance: dict[str, Any]) -> None:
        """Create a new, exclusively owned run directory."""
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory.resolve()
        self.run_id = str(uuid4())
        self.attempt_id = str(uuid4())
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "created_at": datetime.now(UTC).isoformat(),
            "provenance": provenance,
            "harness_status": "not_started",
            "grade_status": "not_available",
            "evidence_status": "partial",
            "cleanup_status": "not_checked",
        }
        self._lock = asyncio.Lock()

    async def append_async(self, *, event: str, data: dict[str, Any]) -> None:
        """Append and flush one real observation, without request headers or credentials."""
        record = {
            "time": datetime.now(UTC).isoformat(),
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "event": event,
            "data": data,
        }
        async with self._lock:
            async with aiofiles.open(self.directory / "events.jsonl", "a", encoding="utf-8") as stream:
                await stream.write(json.dumps(record, ensure_ascii=True) + "\n")
                await stream.flush()
                await asyncio.to_thread(os.fsync, stream.fileno())

    async def save_async(self) -> None:
        """Atomically replace the manifest after flushing its contents."""
        async with self._lock:
            temporary = self.directory / "manifest.json.tmp"
            async with aiofiles.open(temporary, "w", encoding="utf-8") as stream:
                await stream.write(json.dumps(self.manifest, indent=2, ensure_ascii=True) + "\n")
                await stream.flush()
                await asyncio.to_thread(os.fsync, stream.fileno())
            await asyncio.to_thread(os.replace, temporary, self.directory / "manifest.json")
