# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import aiofiles
import httpx

from pyrit.executor.benchmark.submission.hooks import SubmissionHooks, SubmissionTool
from tests.unit.mocks import openai_response_json_dict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path


class FixtureRejectionError(Exception):
    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


class FixtureFailureError(Exception):
    pass


@dataclass(frozen=True, kw_only=True)
class FrozenFixture:
    submission_id: str
    content: bytes
    sha256: str
    sequence: int


def completed_observation(
    grade: object,
    *,
    feedback: str = "OFFLINE/SIMULATED feedback",
    completeness: str = "complete",
    cleanup: str = "not_required",
) -> dict[str, Any]:
    return {
        "grade": grade,
        "feedback": feedback,
        "evidence_label": "OFFLINE/SIMULATED",
        "behavior_outcome": "pass" if grade == 1 else "fail" if grade == 0 else "partial",
        "evidence_completeness": completeness,
        "cleanup_status": cleanup,
    }


def empty_report() -> dict[str, Any]:
    return {
        "contract_version": "strict-submission-v1",
        "mode": "offline",
        "simulated": True,
        "status": "no_submission",
        "last_valid_grade": None,
        "selected_submission_id": None,
        "full_success": False,
        "submissions": [],
        "capabilities": {"remote_query": False, "remote_cancel": False, "idempotency": False},
    }


class InertSubmissionBinding:
    """Locally authored test policy with inert bytes and a fake evaluator only."""

    def __init__(self, *, directory: Path, evaluator: Callable[[FrozenFixture], Awaitable[dict[str, Any]]]) -> None:
        self.directory = directory
        self.evaluator = evaluator
        self.report = empty_report()
        self.artifacts = {"fixture": b"OFFLINE/SIMULATED inert bytes\n"}
        self.dispatches: list[FrozenFixture] = []
        self.frozen_paths: list[Path] = []

    def read_report(self) -> dict[str, Any]:
        return copy.deepcopy(self.report)

    def hooks(self) -> SubmissionHooks:
        return SubmissionHooks(
            tools=(
                SubmissionTool(
                    name="submit_fixture",
                    description="Submit an inert offline fixture.",
                    parameters={
                        "type": "object",
                        "properties": {"artifact_ref": {"type": "string"}},
                        "required": ["artifact_ref"],
                        "additionalProperties": False,
                    },
                    callback_async=self.submit_async,
                ),
            ),
            read_report=self.read_report,
            recoverable_errors=(FixtureRejectionError,),
            terminal_errors=(FixtureFailureError,),
            error_feedback=lambda error: f"Error: {error}",
        )

    async def submit_async(self, *, artifact_ref: str) -> str:
        record: dict[str, Any] = {
            "submission_id": str(uuid4()),
            "sequence": len(self.report["submissions"]) + 1,
            "artifact_sha256": None,
            "artifact_size_bytes": None,
            "dispatch_state": "not_dispatched",
            "status": "rejected",
            "grade": None,
            "behavior_outcome": None,
            "feedback": "",
            "error_code": None,
            "raw_evidence": {"evidence_label": "OFFLINE/SIMULATED"},
            "evidence_completeness": "complete",
            "acceptance": "not_dispatched",
            "observed_request_id": None,
            "observed_receipt_id": None,
            "remote_disposition": "not_dispatched",
            "cleanup_status": "not_required",
        }
        self.report["submissions"].append(record)
        if self.report["full_success"]:
            record.update(status="guarded", feedback="Already complete.")
            return record["feedback"]
        if artifact_ref not in self.artifacts:
            record.update(feedback="Fixture not found.\nChoose an existing fixture.", error_code="missing_fixture")
            raise FixtureRejectionError(record["feedback"])
        frozen = FrozenFixture(
            submission_id=record["submission_id"],
            content=bytes(self.artifacts[artifact_ref]),
            sha256=hashlib.sha256(self.artifacts[artifact_ref]).hexdigest(),
            sequence=record["sequence"],
        )
        await asyncio.to_thread(self.directory.mkdir, parents=True, exist_ok=True)
        path = self.directory / f"{frozen.submission_id}.bin"
        async with aiofiles.open(path, "xb") as stream:
            await stream.write(frozen.content)
        self.frozen_paths.append(path)
        record.update(
            artifact_sha256=frozen.sha256,
            artifact_size_bytes=len(frozen.content),
            dispatch_state="dispatched",
            acceptance="unknown",
            remote_disposition="unknown",
            evidence_completeness="unknown",
        )
        async with aiofiles.open(self.directory / "dispatches.jsonl", "a", encoding="utf-8") as stream:
            await stream.write(json.dumps({"evidence_label": "OFFLINE/SIMULATED", **record}) + "\n")
            await stream.flush()
        self.dispatches.append(frozen)
        try:
            observation = await self.evaluator(frozen)
        except asyncio.CancelledError:
            record.update(status="cancelled", feedback="Local await cancelled.", error_code="local_cancel")
            self.report["status"] = "cancelled"
            raise
        except ConnectionError as error:
            record.update(status="unknown", feedback=str(error), error_code="unknown_after_dispatch")
            self.report["status"] = "unknown"
            raise FixtureFailureError(str(error)) from error
        except (ImportError, RuntimeError) as error:
            record.update(status="error", feedback=str(error), error_code="fixture_infrastructure")
            self.report["status"] = "error"
            raise FixtureFailureError(str(error)) from error
        record["dispatch_state"] = "returned"
        grade = observation.get("grade")
        if (
            type(grade) not in (int, float)
            or not isinstance(grade, (int, float))
            or not math.isfinite(grade)
            or not 0 <= grade <= 1
            or not isinstance(observation.get("feedback"), str)
        ):
            record.update(
                status="error",
                feedback="Malformed fixture result.",
                error_code="invalid_result",
                raw_evidence={"evidence_label": "OFFLINE/SIMULATED", "invalid_result": repr(observation)},
            )
            self.report["status"] = "error"
            raise FixtureFailureError(record["feedback"])
        record.update(
            status="completed",
            grade=grade,
            behavior_outcome=observation["behavior_outcome"],
            feedback=observation["feedback"],
            raw_evidence=copy.deepcopy(observation),
            acceptance="accepted",
            remote_disposition="completed",
            evidence_completeness=observation["evidence_completeness"],
            cleanup_status=observation["cleanup_status"],
        )
        self.report.update(
            status="completed"
            if observation["evidence_completeness"] == "complete"
            and observation["cleanup_status"] in {"not_required", "complete"}
            else "incomplete",
            last_valid_grade=grade,
            selected_submission_id=frozen.submission_id,
            full_success=grade == 1 and observation["behavior_outcome"] == "pass",
        )
        feedback = observation["feedback"]
        assert isinstance(feedback, str)
        return feedback


def tool_turn(*, call_id: str, artifact_ref: str = "fixture") -> dict[str, Any]:
    return {
        "type": "function_call",
        "name": "submit_fixture",
        "call_id": call_id,
        "id": f"function-{call_id}",
        "arguments": json.dumps({"artifact_ref": artifact_ref}),
        "status": "completed",
    }


def text_turn(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "id": "offline-final",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


class OfflineProvider:
    def __init__(self, turns: list[dict[str, Any]]) -> None:
        self.turns = turns
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        if len(self.requests) > len(self.turns):
            raise AssertionError("Unexpected additional OFFLINE/SIMULATED model request.")
        response = openai_response_json_dict()
        response["output"] = [self.turns[len(self.requests) - 1]]
        return httpx.Response(200, json=response)
