# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Opt-in local queue proof using only the unchanged public, model-free Inspect Task."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from pyrit.common.async_compatibility import run_legacy_sync_async
from pyrit.memory import SQLiteMemory
from pyrit.models.evaluation_job import EvaluationJobRequest, EvaluationJobState


def _validate_root(root: Path) -> None:
    if not root.is_absolute() or root.resolve() != root or root.is_symlink():
        raise ValueError("Example root must be an explicit absolute local non-symlink directory.")


async def main_async(*, root: Path) -> None:
    """Persist exact original evidence and canonical SQLite results, without a model or cloud."""
    await asyncio.to_thread(_validate_root, root)
    await run_legacy_sync_async(root.mkdir, parents=True, exist_ok=False)
    memory = SQLiteMemory.__new__(SQLiteMemory)
    memory.__init__(db_path=str(root / "canonical.sqlite"), _defer_initialization=True, silent=True)
    memory.results_path = str(root)
    memory.disable_embedding()
    try:
        await memory.initialize_async()
        await _run_example_async(root=root, memory=memory)
    finally:
        await memory.dispose_engine_async()


async def _run_example_async(*, root: Path, memory: SQLiteMemory) -> None:
    from pyrit.executor.jobs.inspect import create_public_original_job_port_async

    port = await create_public_original_job_port_async(
        root=root / "queue", memory=memory, allowed_actor_ids=frozenset({"local-example"})
    )
    try:
        await port.startup_async()
        registration = port.registry.registrations[0]

        request = EvaluationJobRequest(
            job_id=uuid4(),
            run_id=uuid4(),
            attempt_id=uuid4(),
            runtime=registration.runtime,
            source=registration.source,
            case_id=registration.case_id,
            execution_profile_sha256=registration.execution_profile_sha256,
        )
        submission = await port.submit_async(request=request, actor_id="local-example")
        accepted = await port.status_async(job_id=request.job_id, actor_id="local-example")
        assert accepted.state is EvaluationJobState.QUEUED and accepted.canonical is None
        port.start_consumer()
        terminal = await port.wait_async(job_id=request.job_id, actor_id="local-example")
        if terminal.state is not EvaluationJobState.SUCCEEDED or terminal.canonical is None:
            raise RuntimeError(f"Local harmless job did not succeed: {terminal.reason}")
        scores = await memory.get_scores_async(
            score_ids=[str(identifier) for identifier in terminal.canonical.score_ids]
        )
        results = await memory.get_attack_results_async(
            attack_result_ids=[str(identifier) for identifier in terminal.canonical.attack_result_ids]
        )
        print(
            json.dumps(
                {
                    "submission": submission.model_dump(mode="json"),
                    "state": terminal.state.value,
                    "evidence": terminal.evidence.value,
                    "cleanup": terminal.cleanup.value,
                    "canonical": terminal.canonical.model_dump(mode="json"),
                    "source_grades": [score.score_value for score in scores],
                    "attack_outcomes": [result.outcome.value for result in results],
                    "events": [event.kind.value for event in terminal.events],
                },
                indent=2,
            )
        )
    finally:
        await port.shutdown_async()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="New absolute directory for local evidence.")
    args = parser.parse_args()
    asyncio.run(main_async(root=args.root))
