# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Run one approved, benign Inspect Eval through its registered PyRIT Scenario."""

from __future__ import annotations

import asyncio
import json
from contextlib import closing
from pathlib import Path
from typing import TYPE_CHECKING

from sqlalchemy import select

from examples.inspect_ghcp_protocol_smoke import _docker_host, _require_protocol_proof
from pyrit.memory import CentralMemory
from pyrit.memory.memory_models import ScoreEntry
from pyrit.models import EvalScoreProvenance, EvalScoreRole, ScoreStatus
from pyrit.models.inspect_ghcp import InspectGhcpReport
from pyrit.scenario.core import TaskOwnedAtomicAttack
from pyrit.scenario.scenarios.benchmark.inspect_eval import InspectEvalScenario
from pyrit.setup import SQLITE, initialize_pyrit_async

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface
    from pyrit.models.native_cyber_evidence import NativeCyberEpisodeSnapshot


def _count_unlabeled_control_scores(*, memory: MemoryInterface, episode: NativeCyberEpisodeSnapshot) -> int:
    """
    Count only the original two false canary Scores, never a second original grade.

    Returns:
        int: Two MessagePiece-anchored, unlabeled SubStringScorer control rows.

    Raises:
        RuntimeError: If an unknown, manually stopped or relabeled Score appears.
    """
    response_ids = {piece for turn in episode.turns for piece in turn.response_piece_ids}
    if not response_ids or episode.report_content_id is None or episode.score_id is None:
        raise RuntimeError("The Inspect protocol has no retained response or original report Score.")
    response_id_text = {str(piece) for piece in response_ids}
    with closing(memory.get_session()) as session:
        original = list(
            session.scalars(select(ScoreEntry).where(ScoreEntry.scorable_content_id == episode.report_content_id))
        )
        controls = list(
            session.scalars(select(ScoreEntry).where(ScoreEntry.prompt_request_response_id.in_(response_ids)))
        )
    if len(original) != 1 or original[0].id != episode.score_id or len(controls) != 2:
        raise RuntimeError("The Inspect protocol changed its one original or two control Score links.")
    records = memory.get_scores(score_ids=[str(row.id) for row in controls])
    if len(records) != 2 or any(
        record.status is not ScoreStatus.COMPLETE
        or record.score_type != "true_false"
        or record.score_value != "false"
        or record.scorer_class_identifier is None
        or record.scorer_class_identifier.class_name != "SubStringScorer"
        or record.score_metadata
        or str(record.message_piece_id) not in response_id_text
        for record in records
    ):
        raise RuntimeError("The benign canary controls are not two unlabeled, false PyRIT Score rows.")
    return len(records)


async def run_async() -> None:
    """
    Execute only the pinned, non-benchmark family on approved local resources.

    Raises:
        RuntimeError: If the original Score or Scenario linkage is incomplete.
    """
    _docker_host()
    database = Path.cwd() / ".venv" / "inspect-ghcp" / "oneclick-protocol.db"
    await initialize_pyrit_async(
        memory_db_type=SQLITE, db_path=database, env_files=[], load_defaults=False, silent=True
    )
    scenario = InspectEvalScenario()
    scenario.set_params_from_args(args={"eval_family": "benign_protocol"})
    await scenario.initialize_async()
    result = await scenario.run_async()
    [work] = scenario._atomic_attacks
    if not isinstance(work, TaskOwnedAtomicAttack):
        raise TypeError("The Inspect Eval Scenario did not create task-owned case work.")
    groups = result.get_display_groups()
    attacks = groups.get(work.display_group)
    if attacks is None or len(attacks) != 1:
        raise RuntimeError("The benign Inspect Eval Scenario did not retain exactly one case result.")
    attack = attacks[0]
    score = attack.automated_score
    if score is None or score.score_metadata is None or score.status is not ScoreStatus.UNDETERMINED:
        raise RuntimeError("The Inspect Eval Scenario did not link one original UND Score.")
    provenance = EvalScoreProvenance.from_metadata(metadata=score.score_metadata)
    if provenance.role is not EvalScoreRole.BENCHMARK_ORIGINAL or provenance.case_run_id != work.case_run_id:
        raise RuntimeError("The Inspect Eval Scenario linked a foreign original Score.")
    run_id = score.score_metadata.get("run_id")
    if not isinstance(run_id, str):
        raise RuntimeError("The original Inspect source episode ID was not retained.")
    memory = CentralMemory.get_memory_instance()
    episode = memory.native_cyber_evidence.get_finalized_episode(run_id=run_id)
    if episode.score_id != score.id or episode.report_content_id is None:
        raise RuntimeError("The Scenario Score differs from the original atomic Inspect report.")
    content = memory.get_scorable_content(content_ids=[episode.report_content_id])
    report = InspectGhcpReport.model_validate_json(content[episode.report_content_id].value)
    _require_protocol_proof(report=report, score=score)
    control_count = await asyncio.to_thread(_count_unlabeled_control_scores, memory=memory, episode=episode)
    print(
        json.dumps(
            {
                "scenario_result_id": str(result.id),
                "case_run_id": work.case_run_id,
                "source_sha256": work.case.package.source_sha256,
                "run_id": report.run_id,
                "inspect_log_id": report.inspect_log_id,
                "original_marker_lifecycle_value": report.judgment.numeric_value if report.judgment else None,
                "score_id": str(score.id),
                "score_status": score.status.value,
                "numeric_score_is_null": score.score_value is None,
                "linked_original_und_score_count": 1,
                "unlabeled_false_canary_control_scores": control_count,
                "turns": report.turn_count,
                "successful_tools": report.successful_tool_execution_count,
                "required_gaps": list(report.required_gaps),
                "raw_bytes": episode.stored_raw_bytes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    asyncio.run(run_async())
