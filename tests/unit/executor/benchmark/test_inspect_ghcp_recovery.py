# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Real SQLite interrupted-episode recovery without replaying model or scorer."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from pyrit.executor.benchmark.inspect_ghcp_recovery import recover_interrupted_inspect_run_async
from pyrit.memory.inspect_ghcp_evidence import InspectGhcpEvidenceStore
from pyrit.models import ScoreStatus
from pyrit.models.inspect_ghcp import InspectGhcpReport, InspectGhcpStatus
from pyrit.models.native_cyber_evidence import NativeCyberEpisodeStart

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory


@pytest.mark.usefixtures("patch_central_database")
class TestInspectGhcpRecovery:
    async def test_original_numeric_judgment_is_retained_without_complete_score(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        run_id = str(uuid4())
        store = InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=run_id,
            task_name="original_inspect_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=100_000,
        )
        original = {"value": 1.0, "explanation": "Original lifecycle readback"}
        content = json.dumps(
            {
                "status": "success",
                "eval": {"task": "original_inspect_task", "run_id": "original-eval-run"},
                "samples": [
                    {
                        "id": "original-sample",
                        "epoch": 1,
                        "scores": {"original_scorer": original},
                        "events": [
                            {
                                "event": "score",
                                "scorer": "original_scorer",
                                "uuid": "source-score-1",
                                "score": original,
                            }
                        ],
                    }
                ],
            },
            sort_keys=True,
        ).encode("utf-8")
        log_sha = store.record_inspect_log(content=content)
        pending = sqlite_instance.native_cyber_evidence.get_episode(run_id=run_id)
        assert pending.finalized_at is None and pending.score_id is None
        saved = await recover_interrupted_inspect_run_async(
            memory=sqlite_instance,
            run_id=run_id,
            sample_id="original-sample",
            cli_sha256="a" * 64,
            model_id="local-alias",
            wire_model="local-model",
            cleanup_confirmed=True,
        )
        repeated = await recover_interrupted_inspect_run_async(
            memory=sqlite_instance,
            run_id=run_id,
            sample_id="original-sample",
            cli_sha256="a" * 64,
            model_id="local-alias",
            wire_model="local-model",
            cleanup_confirmed=True,
        )
        stored = sqlite_instance.get_scorable_content(content_ids=[saved.report_content_id])
        report = InspectGhcpReport.model_validate_json(stored[saved.report_content_id].value)
        assert report.schema_version == 3
        assert report.control_receipt_sha256 is None
        assert not report.token_files_absent_before_turn
        assert repeated.score_id == saved.score_id
        assert repeated.stored_raw_bytes == pending.stored_raw_bytes
        assert saved.score_status is ScoreStatus.UNDETERMINED
        assert report.status is InspectGhcpStatus.ERROR
        assert report.judgment is not None and report.judgment.numeric_value == 1.0
        assert report.judgment.source_event_id == "source-score-1"
        assert report.judgment.normalization_version == 1
        assert report.inspect_log_sha256 == log_sha == hashlib.sha256(content).hexdigest()
        assert report.model_request_count == 0

    async def test_requires_cleanup_then_publishes_exactly_one_undetermined_score(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        run_id = str(uuid4())
        InspectGhcpEvidenceStore(
            memory=sqlite_instance,
            run_id=run_id,
            task_name="original_inspect_task",
            task_version="1",
            started_at=datetime.now(UTC),
            raw_byte_limit=10_000,
        )
        options = {
            "memory": sqlite_instance,
            "run_id": run_id,
            "sample_id": "original-sample",
            "cli_sha256": "a" * 64,
            "model_id": "qwen3-local",
            "wire_model": "qwen3:1.7b",
        }
        with pytest.raises(RuntimeError, match="cleanup"):
            await recover_interrupted_inspect_run_async(**options, cleanup_confirmed=False)
        pending = sqlite_instance.native_cyber_evidence.get_episode(run_id=run_id)
        assert pending.finalized_at is None
        assert pending.score_id is None

        first = await recover_interrupted_inspect_run_async(**options, cleanup_confirmed=True)
        second = await recover_interrupted_inspect_run_async(**options, cleanup_confirmed=True)
        assert first.finalized_at is not None
        assert not first.coverage_complete
        assert first.score_status is ScoreStatus.UNDETERMINED
        assert first.score_id == second.score_id
        assert first.report_sha256 == second.report_sha256
        assert first.stored_raw_bytes == second.stored_raw_bytes
        assert len([gap for gap in first.gaps if "before a complete atomic" in gap]) == 1

    async def test_legacy_binding_recovery_replays_without_schema3_receipts(
        self, sqlite_instance: SQLiteMemory
    ) -> None:
        run_id = str(uuid4())
        sqlite_instance.native_cyber_evidence.create_episode(
            start=NativeCyberEpisodeStart(
                run_id=run_id,
                binding_name="inspect-ghcp",
                binding_version="2",
                task_id="original_inspect_task",
                task_version="1",
                started_at=datetime.now(UTC),
                simulated=False,
                required_raw_streams=(
                    InspectGhcpEvidenceStore.SDK_KEY,
                    InspectGhcpEvidenceStore.MODEL_KEY,
                    InspectGhcpEvidenceStore.HOST_KEY,
                    InspectGhcpEvidenceStore.ADVERSARIAL_KEY,
                ),
                raw_byte_limit=10_000,
            )
        )
        saved = await recover_interrupted_inspect_run_async(
            memory=sqlite_instance,
            run_id=run_id,
            sample_id="original-sample",
            cli_sha256="a" * 64,
            model_id="local-alias",
            wire_model="local-model",
            cleanup_confirmed=True,
        )
        stored = sqlite_instance.get_scorable_content(content_ids=[saved.report_content_id])
        canonical = stored[saved.report_content_id].value
        report = InspectGhcpReport.model_validate_json(canonical)
        assert report.schema_version == 2
        assert "control_receipt_sha256" not in canonical
        assert "token_files_absent_before_turn" not in canonical
        assert report.canonical_json() == canonical
        assert saved.score_status is ScoreStatus.UNDETERMINED
