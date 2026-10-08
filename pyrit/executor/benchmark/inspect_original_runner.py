# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run one allowlisted original Inspect Task unchanged and import its original log."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from inspect_ai import eval_async

from pyrit.common.async_compatibility import run_legacy_sync_async
from pyrit.executor.benchmark.inspect_eval_projection import InspectProjectionVersion
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory
from pyrit.executor.benchmark.inspect_original_eval import InspectOriginalEvalImporter, InspectOriginalImport
from pyrit.models import EvalCaseRef, EvalRunRef
from pyrit.models.native_cyber_evidence import NativeCyberEpisodeStart

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.memory import MemoryInterface


@dataclass(frozen=True, kw_only=True)
class InspectOriginalRun:
    """The original Task's source identity and its ungraded, committed log import."""

    case: EvalCaseRef
    run: EvalRunRef
    imported: InspectOriginalImport
    archive_path: Path


async def run_original_inert_eval_async(
    *,
    memory: MemoryInterface,
    log_dir: Path,
    family: str = "inspect_original_inert",
    run_instance_id: uuid.UUID | None = None,
) -> InspectOriginalRun:
    """
    Run the sole approved in-process original Inspect Task without substituting any brick.

    Args:
        memory (MemoryInterface): The initialized PyRIT evidence store.
        log_dir (Path): An existing private local directory for the original `.eval`.
        family (str): The one approved named Task.
        run_instance_id (uuid.UUID | None): A caller-owned unique run ID, if a Scenario allocated one.

    Returns:
        InspectOriginalRun: Case identity and sealed Inspect-only sample evidence, never a PyRIT Score.

    Raises:
        ValueError: If the source or local log directory is not approved.
        RuntimeError: If Inspect fails before returning a final original `.eval`.
        asyncio.CancelledError: If the caller cancels the original Inspect run.
    """
    source = await asyncio.to_thread(EvalSourceFactory.resolve_original_inert, family=family)
    if str(log_dir).startswith(("\\\\", "//")):
        raise ValueError("Original Inspect log directory must be local, not a network share.")
    directory_valid = await asyncio.to_thread(lambda: log_dir.is_dir() and not log_dir.is_symlink())
    if not directory_valid:
        raise ValueError("Original Inspect logs require an existing, non-symlink local directory.")
    await asyncio.to_thread(source.verify_unchanged)
    run = EvalRunRef(spec=source.spec, run_instance_id=run_instance_id or uuid.uuid4())
    episode_id = f"inspect-run-{run.run_instance_id.hex}"
    importer = InspectOriginalEvalImporter(memory=memory)
    capture = memory.native_cyber_evidence
    await run_legacy_sync_async(
        capture.create_episode,
        start=NativeCyberEpisodeStart(
            run_id=episode_id,
            binding_name="inspect-original",
            binding_version=InspectProjectionVersion.TOOL_CALLS.binding_version,
            task_id=source.case.task_name,
            task_version=source.case.task_version,
            started_at=datetime.now(UTC),
            simulated=True,
            required_raw_streams=(importer.ARCHIVE_KEY, importer.RESOLVED_KEY),
            raw_byte_limit=importer.RAW_QUOTA,
        ),
    )

    # Importing this module is intentionally delayed: offline `.eval` imports never register a hook.
    from pyrit.executor.benchmark.inspect_original_hooks import InspectOriginalLiveCapture, active_original_capture

    live = await InspectOriginalLiveCapture.begin_async(memory=memory, episode_id=episode_id)
    try:
        with active_original_capture(capture=live):
            logs = await eval_async(tasks=source.task, log_dir=str(log_dir), log_format="eval")
    except asyncio.CancelledError:
        await run_legacy_sync_async(
            capture.mark_capture_gap, run_id=episode_id, reason="Original Inspect run was cancelled before log import."
        )
        raise
    except Exception as error:
        await run_legacy_sync_async(
            capture.mark_capture_gap, run_id=episode_id, reason="Original Inspect Task failed before log import."
        )
        raise RuntimeError(
            f"Original Inspect Task failed; pending capture {episode_id} needs reconciliation."
        ) from error
    finally:
        await live.close_async()

    if len(logs) != 1 or not logs[0].location:
        await run_legacy_sync_async(
            capture.mark_capture_gap, run_id=episode_id, reason="Original Inspect Task returned no unique `.eval`."
        )
        raise RuntimeError(f"Original Inspect Task returned no unique log; pending capture {episode_id} retained.")
    location = await asyncio.to_thread(_approved_log_location, path=logs[0].location, log_dir=log_dir)
    try:
        await asyncio.to_thread(source.verify_unchanged)
    except ValueError as error:
        await run_legacy_sync_async(
            capture.mark_capture_gap, run_id=episode_id, reason="Original Inspect Task source drifted after execution."
        )
        await importer._import_async(
            path=location, cases=None, run=None, live_observer=live, require_no_model_calls=True
        )
        raise RuntimeError(
            f"Original Inspect Task source drifted; ungraded archive {episode_id} was retained."
        ) from error
    imported = await importer._import_async(
        path=location,
        cases=(source.case,),
        run=run,
        live_observer=live,
        require_no_model_calls=True,
    )
    if imported.inspect_run_id != logs[0].eval.run_id or not imported.episode.coverage_complete:
        raise RuntimeError(
            f"Original Inspect run {episode_id} has unqualified or incomplete imported evidence; "
            "do not publish a benchmark grade."
        )
    return InspectOriginalRun(case=source.case, run=run, imported=imported, archive_path=location)


def _approved_log_location(*, path: str, log_dir: Path) -> Path:
    """
    Check the original log is in this invocation's local directory.

    Returns:
        Path: The single resolved original `.eval` path.

    Raises:
        ValueError: If Inspect returned a foreign or non-Eval log file.
    """
    from pathlib import Path
    from urllib.parse import urlsplit
    from urllib.request import url2pathname  # ty: ignore[deprecated]  # needed on Python 3.11-3.12

    uri = urlsplit(path)
    if uri.scheme == "file":
        if uri.netloc not in {"", "localhost"} or uri.query or uri.fragment:
            raise ValueError("Inspect original file URI must identify a local, unmodified file path.")
        candidate = Path(url2pathname(uri.path))  # ty: ignore[deprecated]
    elif uri.scheme and not (len(uri.scheme) == 1 and len(path) > 2 and path[1] == ":" and path[2] in {"/", "\\"}):
        raise ValueError("Inspect original log URI must use only a local file path.")
    else:
        candidate = Path(path)
    if str(candidate).startswith(("\\\\", "//")):
        raise ValueError("Inspect original log cannot come from a network share.")
    candidate = candidate.resolve(strict=True)
    if candidate.suffix != ".eval" or not candidate.is_relative_to(log_dir.resolve(strict=True)):
        raise ValueError("Inspect returned an original log outside the approved local directory.")
    return candidate
