# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Test-only isolated process for the SHA-pinned public inert original Task."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

_PROOF_PREFIX = "ORIGINAL_FIXTURE_PROOF:"


async def _run_public_fixture_async(*, root: Path) -> dict[str, Any]:
    """Execute the unchanged public Task with its own SQLite, then read its typed projection."""
    from pyrit.backend.services.scenario_run_service import ScenarioRunService
    from pyrit.memory import CentralMemory, SQLiteMemory
    from pyrit.models import SCENARIO_RUN_PLAN_METADATA_KEY, ScenarioRunPlan
    from pyrit.models.catalog.scenario import OriginalInspectImportSummary
    from pyrit.registry import ScenarioRegistry
    from pyrit.scenario.scenarios.benchmark.inspect_original_inert import InspectOriginalInertScenario

    memory = SQLiteMemory(db_path=root / "original-worker.sqlite")
    memory.results_path = str(root / "results")
    memory.disable_embedding()
    memory.reset_database()
    CentralMemory.set_memory_instance(memory)
    try:
        registry = ScenarioRegistry.get_registry_singleton()
        registry.register_class(InspectOriginalInertScenario, name="benchmark.inspect_original_inert")
        scenario = await registry.create_and_initialize_async(
            "benchmark.inspect_original_inert",
            scenario_params={},
            max_concurrency=1,
            max_retries=0,
            include_baseline=False,
        )
        finished = await scenario.run_async()
        [stored] = memory.get_scenario_results(scenario_result_ids=[str(finished.id)])
        plan = ScenarioRunPlan.model_validate(stored.metadata[SCENARIO_RUN_PLAN_METADATA_KEY])
        imported = ScenarioRunService.verify_original_inspect_import(memory=memory, scenario_result=stored, plan=plan)
        assert isinstance(imported, OriginalInspectImportSummary)
        [score] = memory.get_scores(score_ids=[str(imported.score_id)])
        assert score.score_metadata is not None
        return {
            "original_score": imported.score_value,
            "archive_sha256": imported.archive_sha256,
            "source_score_id": str(imported.score_id),
            "source_attack_result_id": str(imported.attack_result_id),
            "final_score_event_id": str(score.score_metadata["inspect_final_score_event_id"]),
            "final_score_event_sha256": str(score.score_metadata["inspect_final_score_event_sha256"]),
            "pyrit_score_status": score.status.value,
            "pyrit_outcome": imported.outcome.value,
            "source_coverage_complete": True,
        }
    finally:
        memory.dispose_engine()


def main() -> None:
    """Set owner-only roots before importing PyRIT or Inspect in this process."""
    with tempfile.TemporaryDirectory(prefix="pyrit-public-worker-") as workspace:
        root = Path(workspace)
        for child in ("appdata", "localappdata", "cache", "home", "tmp", "results"):
            (root / child).mkdir()
        os.environ["USERPROFILE"] = str(root / "home")
        os.environ["HOME"] = str(root / "home")
        os.environ["APPDATA"] = str(root / "appdata")
        os.environ["LOCALAPPDATA"] = str(root / "localappdata")
        os.environ["XDG_CACHE_HOME"] = str(root / "cache")
        os.environ["TMP"] = str(root / "tmp")
        os.environ["TEMP"] = str(root / "tmp")
        tempfile.tempdir = None
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from platformdirs import windows as platformdirs_windows

        platformdirs_windows.get_win_folder = platformdirs_windows.get_win_folder_from_env_vars
        try:
            proof = asyncio.run(_run_public_fixture_async(root=root))
            sys.stdout.write(f"{_PROOF_PREFIX}{json.dumps(proof, separators=(',', ':'))}\n")
        finally:
            logging.shutdown()


if __name__ == "__main__":
    main()
