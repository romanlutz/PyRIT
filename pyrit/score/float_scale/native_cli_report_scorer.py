# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Build an unpersisted content-anchored score from an acquired CLI report."""

from __future__ import annotations

from pyrit.models import ContentScorable, Score, ScorerIdentifier, ScoreStatus, ScoringExpectation
from pyrit.models.native_cli_report import NativeCliReportStatus, NativeCliRunReport


class NativeCliReportScoreBuilder:
    """Pure projection only: unlike ``Scorer.score_async``, never writes memory."""

    def build_score(self, *, report: NativeCliRunReport, expectation: ScoringExpectation | None = None) -> Score:
        """
        Build a provisional float score anchored to the exact canonical report.

        Args:
            report (NativeCliRunReport): Already assembled original CLI/grader evidence.
            expectation (ScoringExpectation | None): Caller-provided grading context.

        Returns:
            Score: Unpersisted ``ContentScorable`` score for a caller-owned finalizer.

        Raises:
            TypeError: If the report or expectation has an unsupported type.
            ValueError: If report validation fails on a copied/modified instance.
        """
        if not isinstance(report, NativeCliRunReport):
            raise TypeError("A native CLI report is required, not an assistant sentence or GHCP event.")
        ScoringExpectation.validate_type(expectation)
        validated = NativeCliRunReport.model_validate(report.model_dump(mode="json"))
        digest = validated.sha256()
        completed = validated.status is NativeCliReportStatus.COMPLETED
        value = validated.judgment.value if completed and validated.judgment is not None else None
        return Score(
            score_type="float_scale",
            score_value=str(value) if value is not None else None,
            status=ScoreStatus.COMPLETE if value is not None else ScoreStatus.UNDETERMINED,
            score_rationale=(
                validated.judgment.rationale
                if completed and validated.judgment is not None
                else f"No final original-grader verdict: the native CLI run is {validated.status.value}."
            ),
            scored_expectation=expectation,
            scorable=ContentScorable(value=validated.canonical_json(), data_type="text"),
            scorer_class_identifier=ScorerIdentifier.of(
                self,
                scorer_type="float_scale",
                params={"contract": "native-cli-run-v1", "report_sha256": digest},
            ),
            score_metadata={
                "contract_version": "native-cli-run-v1",
                "report_sha256": digest,
                "task_id": validated.task_id,
                "task_version": validated.task_version,
                "run_id": validated.run_id,
                "turn_id": validated.turn_id,
                "turn_index": validated.turn_index,
                "run_status": validated.status.value,
                "cli_protocol": validated.protocol.value,
                "cli_version": validated.cli_version,
                "cli_profile": validated.cli_profile,
                "cleanup": validated.cleanup.value,
                "coverage_complete": int(validated.evidence.coverage_complete),
                "observed_steps": (
                    validated.evidence.observed_steps if validated.evidence.observed_steps is not None else "unobserved"
                ),
                "exit_code": validated.evidence.exit_code if validated.evidence.exit_code is not None else "unobserved",
                "gap_count": len(validated.evidence.gaps),
                "simulated": (str(validated.simulated).lower() if validated.simulated is not None else "unknown"),
                "publication_state": "unpersisted_candidate",
            },
        )


def build_native_cli_report_score(
    *, report: NativeCliRunReport, expectation: ScoringExpectation | None = None
) -> Score:
    """
    Project existing CLI and grader evidence without calling a provider or writing memory.

    Args:
        report (NativeCliRunReport): Caller-assembled CLI run report.
        expectation (ScoringExpectation | None): Original scoring context.

    Returns:
        Score: A provisional, unpersisted content-anchored score.

    Raises:
        TypeError: If the input is not a CLI report or the expectation is invalid.
        ValueError: If report evidence is invalid or the original grade contradicts its status.
    """
    return NativeCliReportScoreBuilder().build_score(report=report, expectation=expectation)
