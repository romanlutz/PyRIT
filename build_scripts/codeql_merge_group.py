# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Publish prepared CodeQL SARIF or cancel a positively verified obsolete merge group."""

from __future__ import annotations

import argparse
import base64
import gzip
import http.client
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn
from urllib.parse import quote, urlsplit

if TYPE_CHECKING:
    from collections.abc import Mapping


class GitHubAPIError(RuntimeError):
    def __init__(self, *, status: int, message: str, errors: Any = None) -> None:
        super().__init__(f"GitHub API returned HTTP {status}: {message}")
        self.status = status
        self.message = message
        self.errors = errors


class WorkflowCancelledError(RuntimeError):
    pass


class GitHubAPI:
    TIMEOUT_SECONDS = 30

    def __init__(self, *, api_url: str, token: str) -> None:
        parsed = urlsplit(api_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or not token
        ):
            raise ValueError("A HTTPS GitHub API URL and a nonempty Actions token are required")
        self._host = parsed.netloc
        self._prefix = parsed.path.rstrip("/")
        self._token = token

    def request(self, *, method: str, path: str, expected_status: int, payload: dict[str, Any] | None = None) -> Any:
        connection = http.client.HTTPSConnection(self._host, timeout=self.TIMEOUT_SECONDS)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
            "User-Agent": "PyRIT-CodeQL-merge-group",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        try:
            connection.request(
                method, self._prefix + path, body=json.dumps(payload) if payload else None, headers=headers
            )
            response = connection.getresponse()
            status, raw = response.status, response.read()
        finally:
            connection.close()
        try:
            body = json.loads(raw) if raw else {}
        except json.JSONDecodeError as error:
            raise GitHubAPIError(status=status, message="Invalid JSON response") from error
        if status != expected_status:
            message = body.get("message") if isinstance(body, dict) else None
            raise GitHubAPIError(
                status=status,
                message=message if isinstance(message, str) else "Unexpected response",
                errors=body.get("errors") if isinstance(body, dict) else None,
            )
        return body


@dataclass(frozen=True, kw_only=True)
class MergeGroup:
    repository: str
    ref: str
    sha: str
    run_id: int
    run_attempt: int

    @property
    def api_path(self) -> str:
        return f"/repos/{self.repository}"


def load_merge_group(environment: Mapping[str, str]) -> MergeGroup:
    if environment["GITHUB_EVENT_NAME"] != "merge_group":
        raise ValueError("This publisher only supports merge_group events")
    event = json.loads(Path(environment["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
    if not isinstance(event, dict) or not isinstance(event.get("merge_group"), dict):
        raise ValueError("Expected a merge-group event object")
    group = event["merge_group"]
    ref, sha, repository = environment["GITHUB_REF"], environment["GITHUB_SHA"], environment["GITHUB_REPOSITORY"]
    base_ref = group["base_ref"]
    if not isinstance(base_ref, str) or not base_ref.startswith("refs/heads/"):
        raise ValueError("The merge group's base must be a branch ref")
    prefix = f"refs/heads/gh-readonly-queue/{base_ref.removeprefix('refs/heads/')}/"
    if event["action"] != "checks_requested" or not ref.startswith(prefix) or len(ref) <= len(prefix):
        raise ValueError("Unsupported merge-group action or queue ref")
    if group["head_ref"] != ref or group["head_sha"] != sha or event["repository"]["full_name"] != repository:
        raise ValueError("The merge-group event does not match the tested repository, ref and SHA")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not re.fullmatch(r"[\w.-]+/[\w.-]+", repository, flags=re.ASCII):
        raise ValueError("A full tested SHA and owner/repository name are required")
    run_id, run_attempt = environment["GITHUB_RUN_ID"], environment["GITHUB_RUN_ATTEMPT"]
    if not re.fullmatch(r"[1-9][0-9]*", run_id) or not re.fullmatch(r"[1-9][0-9]*", run_attempt):
        raise ValueError("The current workflow run ID and attempt must be positive integers")
    return MergeGroup(repository=repository, ref=ref, sha=sha, run_id=int(run_id), run_attempt=int(run_attempt))


def _current_sha(*, api: GitHubAPI, group: MergeGroup) -> str | None:
    ref = quote(group.ref.removeprefix("refs/"), safe="/")
    refs = api.request(method="GET", path=f"{group.api_path}/git/matching-refs/{ref}", expected_status=200)
    if not isinstance(refs, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("ref"), str) for item in refs
    ):
        raise ValueError("Invalid matching-refs response")
    matches = [item for item in refs if item["ref"] == group.ref]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Matching-refs returned more than one exact queue ref")
    target = matches[0].get("object")
    if not isinstance(target, dict):
        raise ValueError("The queue ref has no valid commit object")
    sha = target.get("sha")
    if target.get("type") != "commit" or not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("The queue ref does not point to a valid commit")
    return sha


def _verify_run(*, api: GitHubAPI, group: MergeGroup) -> None:
    run = api.request(method="GET", path=f"{group.api_path}/actions/runs/{group.run_id}", expected_status=200)
    expected = {
        "id": group.run_id,
        "run_attempt": group.run_attempt,
        "event": "merge_group",
        "head_sha": group.sha,
        "head_branch": group.ref.removeprefix("refs/heads/"),
        "path": ".github/workflows/codeql.yml",
    }
    if not isinstance(run, dict) or any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Refusing to cancel a different workflow run or attempt")
    if run.get("status") == "completed" and run.get("conclusion") == "cancelled":
        raise WorkflowCancelledError("The verified CodeQL workflow run is already cancelled")
    if run.get("status") != "in_progress" or run.get("conclusion") is not None:
        raise ValueError("Refusing to cancel a workflow run that is not in progress")


def _verify_no_failed_jobs(*, api: GitHubAPI, group: MergeGroup) -> None:
    path = f"{group.api_path}/actions/runs/{group.run_id}/attempts/{group.run_attempt}/jobs?per_page=100"
    result = api.request(method="GET", path=path, expected_status=200)
    if not isinstance(result, dict) or not isinstance(result.get("jobs"), list):
        raise ValueError("Invalid workflow jobs response")
    jobs = result["jobs"]
    if not jobs or result.get("total_count") != len(jobs):
        raise ValueError("Cannot verify all CodeQL matrix jobs before cancellation")
    for job in jobs:
        if (
            not isinstance(job, dict)
            or not isinstance(job.get("steps"), list)
            or any(not isinstance(step, dict) for step in job["steps"])
        ):
            raise ValueError("Invalid workflow job steps")
        conclusions = [job.get("conclusion"), *(step.get("conclusion") for step in job["steps"])]
        if any(value not in (None, "success", "skipped", "cancelled") for value in conclusions):
            raise RuntimeError("A CodeQL matrix job or step failed; refusing to replace failure with cancellation")


def _retire(*, api: GitHubAPI, group: MergeGroup, current_sha: str | None) -> NoReturn:
    _verify_run(api=api, group=group)
    _verify_no_failed_jobs(api=api, group=group)
    reason = "is absent" if current_sha is None else f"now points to {current_sha}"
    print(
        f"::notice::Verified obsolete merge candidate: {group.ref} {reason}; cancelling run {group.run_id}", flush=True
    )
    try:
        api.request(method="POST", path=f"{group.api_path}/actions/runs/{group.run_id}/cancel", expected_status=202)
    except GitHubAPIError as error:
        if error.status == 409:
            _verify_run(api=api, group=group)
        raise
    # Stay in a cancellable step until the runner receives cancellation. Never return a green check.
    time.sleep(60)
    raise TimeoutError("Cancellation was accepted, but the runner did not stop within 60 seconds")


def _ensure_live(*, api: GitHubAPI, group: MergeGroup) -> None:
    current_sha = _current_sha(api=api, group=group)
    if current_sha != group.sha:
        _retire(api=api, group=group, current_sha=current_sha)


def _missing_ref_message(group: MergeGroup) -> str:
    return f"ref '{group.ref}' not found in this repository"


def _wait_for_processing(*, api: GitHubAPI, group: MergeGroup, sarif_id: str) -> None:
    for delay in (5, 10, 20, 40, 80):
        time.sleep(delay)
        path = f"{group.api_path}/code-scanning/sarifs/{quote(sarif_id, safe='')}"
        result = api.request(method="GET", path=path, expected_status=200)
        if not isinstance(result, dict):
            raise ValueError("Invalid SARIF processing response")
        status = result.get("processing_status")
        if status == "complete":
            return
        if status == "failed":
            errors = result.get("errors")
            if errors == [_missing_ref_message(group)]:
                _ensure_live(api=api, group=group)
            raise RuntimeError(f"SARIF processing failed: {errors}")
        if status != "pending":
            raise ValueError(f"Unsupported SARIF processing status: {status}")
        _ensure_live(api=api, group=group)
    raise TimeoutError("SARIF processing did not complete within five status checks")


def prepared_sarif(*, path: Path, category: str) -> bytes:
    raw = path.read_bytes()
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("Expected a prepared SARIF object")
    runs = document.get("runs")
    if document.get("version") != "2.1.0" or not isinstance(runs, list) or not runs:
        raise ValueError("Expected a prepared SARIF 2.1.0 file with CodeQL runs")
    for run in runs:
        if (
            not isinstance(run, dict)
            or not isinstance(run.get("tool"), dict)
            or not isinstance(run["tool"].get("driver"), dict)
            or not isinstance(run.get("automationDetails"), dict)
            or run["tool"]["driver"].get("name") != "CodeQL"
            or run["automationDetails"].get("id") != f"{category}/"
        ):
            raise ValueError("Prepared CodeQL SARIF does not match the matrix language category")
    return raw


def publish_merge_group(*, api: GitHubAPI, group: MergeGroup, sarif: bytes, checkout: Path) -> str:
    _ensure_live(api=api, group=group)
    payload = {
        "commit_sha": group.sha,
        "ref": group.ref,
        "sarif": base64.b64encode(gzip.compress(sarif, mtime=0)).decode("ascii"),
        "checkout_uri": checkout.resolve().as_uri(),
        "tool_name": "CodeQL",
    }
    try:
        result = api.request(
            method="POST", path=f"{group.api_path}/code-scanning/sarifs", expected_status=202, payload=payload
        )
    except GitHubAPIError as error:
        if error.status in (400, 404) and error.message == _missing_ref_message(group) and error.errors in (None, []):
            _ensure_live(api=api, group=group)
        raise
    if not isinstance(result, dict):
        raise ValueError("SARIF upload was accepted without a valid upload ID")
    sarif_id = result.get("id")
    if not isinstance(sarif_id, str) or not sarif_id:
        raise ValueError("SARIF upload was accepted without a valid upload ID")
    _wait_for_processing(api=api, group=group, sarif_id=sarif_id)
    _ensure_live(api=api, group=group)
    print(f"CodeQL SARIF {sarif_id} processed for {group.ref} at {group.sha}")
    return sarif_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sarif-file", required=True, type=Path)
    parser.add_argument(
        "--category",
        required=True,
        choices=("/language:actions", "/language:javascript-typescript", "/language:python"),
    )
    parser.add_argument("--analysis-outcome", required=True)
    args = parser.parse_args(argv)
    try:
        if args.analysis_outcome != "success":
            raise ValueError("Publishing requires successful CodeQL analysis")
        group = load_merge_group(os.environ)
        sarif = prepared_sarif(path=args.sarif_file, category=args.category)
        api = GitHubAPI(api_url=os.environ["GITHUB_API_URL"], token=os.environ["GITHUB_TOKEN"])
        publish_merge_group(api=api, group=group, sarif=sarif, checkout=Path(os.environ["GITHUB_WORKSPACE"]))
    except (WorkflowCancelledError, KeyboardInterrupt) as error:
        print(f"CodeQL merge-group publication cancelled: {error}", file=sys.stderr)
        return 130
    except (ValueError, KeyError, RuntimeError, OSError, http.client.HTTPException) as error:
        message = str(error).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
        print(f"::error::{message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
