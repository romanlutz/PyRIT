# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Offline contracts for publishing live CodeQL candidates and explicitly retiring obsolete ones."""

from __future__ import annotations

import base64
import gzip
import http.client
import json
import os
import re
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from build_scripts import codeql_merge_group as publisher

REPO_ROOT = Path(__file__).resolve().parents[3]
SHA = "a" * 40
REF = f"refs/heads/gh-readonly-queue/main/pr-42-{SHA}"
REPOSITORY = "owner/repository"
API_PATH = f"/repos/{REPOSITORY}"
REF_PATH = f"{API_PATH}/git/matching-refs/{REF.removeprefix('refs/')}"
RUN_PATH = f"{API_PATH}/actions/runs/123"
JOBS_PATH = f"{RUN_PATH}/attempts/1/jobs?per_page=100"
UPLOAD_PATH = f"{API_PATH}/code-scanning/sarifs"
UPLOAD_ID = "sarif-upload-id"
CATEGORY = "/language:python"


@pytest.fixture
def group() -> publisher.MergeGroup:
    return publisher.MergeGroup(repository=REPOSITORY, ref=REF, sha=SHA, run_id=123, run_attempt=1)


@pytest.fixture
def api() -> MagicMock:
    return MagicMock(spec=publisher.GitHubAPI)


@pytest.fixture
def sarif_file(tmp_path: Path) -> Path:
    path = tmp_path / "upload.sarif"
    path.write_text(
        json.dumps(
            {
                "version": "2.1.0",
                "runs": [
                    {
                        "tool": {"driver": {"name": "CodeQL"}},
                        "automationDetails": {"id": CATEGORY + "/"},
                        "results": [{"partialFingerprints": {"primaryLocationLineHash": "existing-fingerprint"}}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def environment(tmp_path: Path) -> dict[str, str]:
    event = tmp_path / "event.json"
    event.write_text(
        json.dumps(
            {
                "action": "checks_requested",
                "merge_group": {"base_ref": "refs/heads/main", "head_ref": REF, "head_sha": SHA},
                "repository": {"full_name": REPOSITORY},
            }
        ),
        encoding="utf-8",
    )
    return {
        "GITHUB_EVENT_NAME": "merge_group",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_REPOSITORY": REPOSITORY,
        "GITHUB_REF": REF,
        "GITHUB_SHA": SHA,
        "GITHUB_RUN_ID": "123",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_API_URL": "https://api.github.com",
        "GITHUB_TOKEN": "offline-test-token",
        "GITHUB_WORKSPACE": str(tmp_path),
    }


def _refs(sha: str = SHA) -> list[dict[str, Any]]:
    return [{"ref": REF, "object": {"type": "commit", "sha": sha}}]


def _run(**changes: Any) -> dict[str, Any]:
    return {
        "id": 123,
        "run_attempt": 1,
        "event": "merge_group",
        "head_sha": SHA,
        "head_branch": REF.removeprefix("refs/heads/"),
        "path": ".github/workflows/codeql.yml",
        "status": "in_progress",
        "conclusion": None,
        **changes,
    }


def _jobs() -> dict[str, Any]:
    return {"total_count": 3, "jobs": [{"conclusion": None, "steps": [{"conclusion": "success"}]} for _ in range(3)]}


def _request_paths(api: MagicMock) -> list[tuple[str, str]]:
    return [(call.kwargs["method"], call.kwargs["path"]) for call in api.request.call_args_list]


def _publish(*, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path) -> str:
    return publisher.publish_merge_group(
        api=api,
        group=group,
        sarif=publisher.prepared_sarif(path=sarif_file, category=CATEGORY),
        checkout=sarif_file.parent,
    )


def _main(*, environment: dict[str, str], sarif_file: Path, outcome: str = "success") -> int:
    with patch.dict(os.environ, environment, clear=True):
        return publisher.main(["--sarif-file", str(sarif_file), "--category", CATEGORY, "--analysis-outcome", outcome])


def test_live_matching_ref_uploads_exact_prepared_sarif(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path
) -> None:
    api.request.side_effect = [_refs(), {"id": UPLOAD_ID}, {"processing_status": "complete"}, _refs()]
    with patch.object(publisher.time, "sleep") as sleep:
        assert _publish(api=api, group=group, sarif_file=sarif_file) == UPLOAD_ID
    assert _request_paths(api) == [
        ("GET", REF_PATH),
        ("POST", UPLOAD_PATH),
        ("GET", f"{UPLOAD_PATH}/{UPLOAD_ID}"),
        ("GET", REF_PATH),
    ]
    upload = api.request.call_args_list[1].kwargs
    assert upload["expected_status"] == 202
    payload = upload["payload"]
    assert payload["commit_sha"] == SHA
    assert payload["ref"] == REF
    assert payload["tool_name"] == "CodeQL"
    assert payload["checkout_uri"] == sarif_file.parent.resolve().as_uri()
    assert gzip.decompress(base64.b64decode(payload["sarif"])) == sarif_file.read_bytes()
    sleep.assert_called_once_with(5)


@pytest.mark.parametrize("refs", [[], _refs("b" * 40), [{"ref": REF + "-suffix", "object": {"sha": SHA}}]])
def test_verified_absent_or_changed_ref_requests_cancellation_without_upload(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, refs: list[dict[str, Any]]
) -> None:
    api.request.side_effect = [refs, _run(), _jobs(), {}]
    with patch.object(publisher.time, "sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api) == [
        ("GET", REF_PATH),
        ("GET", RUN_PATH),
        ("GET", JOBS_PATH),
        ("POST", RUN_PATH + "/cancel"),
    ]
    assert api.request.call_args.kwargs["expected_status"] == 202


@pytest.mark.parametrize("status", [400, 404])
@pytest.mark.parametrize("current_refs", [[], _refs("b" * 40)])
def test_disappearance_between_precheck_and_upload_is_rechecked_then_cancelled(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, status: int, current_refs: list[dict[str, Any]]
) -> None:
    rejection = publisher.GitHubAPIError(status=status, message=f"ref '{REF}' not found in this repository")
    api.request.side_effect = [_refs(), rejection, current_refs, _run(), _jobs(), {}]
    with patch.object(publisher.time, "sleep", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api) == [
        ("GET", REF_PATH),
        ("POST", UPLOAD_PATH),
        ("GET", REF_PATH),
        ("GET", RUN_PATH),
        ("GET", JOBS_PATH),
        ("POST", RUN_PATH + "/cancel"),
    ]


def test_missing_ref_rejection_with_still_live_candidate_remains_failure(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path
) -> None:
    rejection = publisher.GitHubAPIError(status=404, message=f"ref '{REF}' not found in this repository")
    api.request.side_effect = [_refs(), rejection, _refs()]
    with pytest.raises(publisher.GitHubAPIError) as raised:
        _publish(api=api, group=group, sarif_file=sarif_file)
    assert raised.value is rejection
    assert _request_paths(api) == [("GET", REF_PATH), ("POST", UPLOAD_PATH), ("GET", REF_PATH)]


@pytest.mark.parametrize("stage", ["precheck", "upload", "processing", "recheck", "cancel"])
@pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 502, 503])
def test_api_errors_never_become_obsolete_candidate_success(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, stage: str, status: int
) -> None:
    error = publisher.GitHubAPIError(status=status, message="authentication, rate limit or service failure")
    prefixes: dict[str, list[Any]] = {
        "precheck": [],
        "upload": [_refs()],
        "processing": [_refs(), {"id": UPLOAD_ID}],
        "recheck": [
            _refs(),
            publisher.GitHubAPIError(status=404, message=f"ref '{REF}' not found in this repository"),
        ],
        "cancel": [[], _run(), _jobs()],
    }
    api.request.side_effect = [*prefixes[stage], error]
    with patch.object(publisher.time, "sleep"):
        with pytest.raises(publisher.GitHubAPIError) as raised:
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert raised.value is error
    cancellations = [path for method, path in _request_paths(api) if method == "POST" and path.endswith("/cancel")]
    assert cancellations == ([RUN_PATH + "/cancel"] if stage == "cancel" else [])


@pytest.mark.parametrize("stage", ["precheck", "upload", "processing", "cancel"])
@pytest.mark.parametrize("error", [OSError("network unavailable"), http.client.HTTPException("connection failed")])
def test_network_errors_remain_failures(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, stage: str, error: Exception
) -> None:
    prefixes: dict[str, list[Any]] = {
        "precheck": [],
        "upload": [_refs()],
        "processing": [_refs(), {"id": UPLOAD_ID}],
        "cancel": [[], _run(), _jobs()],
    }
    api.request.side_effect = [*prefixes[stage], error]
    with patch.object(publisher.time, "sleep"):
        with pytest.raises(type(error)) as raised:
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert raised.value is error


@pytest.mark.parametrize(
    "error",
    [
        publisher.GitHubAPIError(status=400, message="Invalid SARIF"),
        publisher.GitHubAPIError(status=404, message="Not Found"),
        publisher.GitHubAPIError(status=422, message=f"ref '{REF}' not found in this repository"),
        publisher.GitHubAPIError(
            status=400, message=f"ref '{REF}' not found in this repository", errors=["Invalid SARIF"]
        ),
    ],
)
def test_genuine_or_unsupported_upload_rejection_does_not_consult_expiry(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, error: publisher.GitHubAPIError
) -> None:
    api.request.side_effect = [_refs(), error]
    with pytest.raises(publisher.GitHubAPIError) as raised:
        _publish(api=api, group=group, sarif_file=sarif_file)
    assert raised.value is error
    assert _request_paths(api) == [("GET", REF_PATH), ("POST", UPLOAD_PATH)]


@pytest.mark.parametrize("status", ["pending", "complete", "failed"])
def test_expiry_after_accepted_upload_is_not_reported_as_success(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, status: str
) -> None:
    response: dict[str, Any] = {"processing_status": status}
    if status == "failed":
        response["errors"] = [f"ref '{REF}' not found in this repository"]
    api.request.side_effect = [_refs(), {"id": UPLOAD_ID}, response, [], _run(), _jobs(), {}]
    with patch.object(publisher.time, "sleep", side_effect=[None, KeyboardInterrupt]):
        with pytest.raises(KeyboardInterrupt):
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api)[-4:] == [
        ("GET", REF_PATH),
        ("GET", RUN_PATH),
        ("GET", JOBS_PATH),
        ("POST", RUN_PATH + "/cancel"),
    ]


@pytest.mark.parametrize("errors", [["Invalid SARIF"], [f"ref '{REF}' not found in this repository", "Invalid SARIF"]])
def test_processing_rejection_remains_failure_without_expiry_fallback(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, errors: list[str]
) -> None:
    api.request.side_effect = [_refs(), {"id": UPLOAD_ID}, {"processing_status": "failed", "errors": errors}]
    with patch.object(publisher.time, "sleep"):
        with pytest.raises(RuntimeError, match="SARIF processing failed"):
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api) == [("GET", REF_PATH), ("POST", UPLOAD_PATH), ("GET", f"{UPLOAD_PATH}/{UPLOAD_ID}")]


def test_processing_timeout_is_bounded_and_fails(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path
) -> None:
    api.request.side_effect = [
        _refs(),
        {"id": UPLOAD_ID},
        *[item for _ in range(5) for item in ({"processing_status": "pending"}, _refs())],
    ]
    with patch.object(publisher.time, "sleep") as sleep:
        with pytest.raises(TimeoutError, match="five status checks"):
            _publish(api=api, group=group, sarif_file=sarif_file)
    assert [call.args[0] for call in sleep.call_args_list] == [5, 10, 20, 40, 80]
    assert len(_request_paths(api)) == 12
    assert not any(path.endswith("/cancel") for _, path in _request_paths(api))


def test_accepted_cancellation_never_returns_a_successful_check(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path
) -> None:
    api.request.side_effect = [[], _run(), _jobs(), {}]
    with patch.object(publisher.time, "sleep") as sleep:
        with pytest.raises(TimeoutError, match="runner did not stop within 60 seconds"):
            _publish(api=api, group=group, sarif_file=sarif_file)
    sleep.assert_called_once_with(60)
    assert _request_paths(api)[-1] == ("POST", RUN_PATH + "/cancel")


@pytest.mark.parametrize(
    "changes",
    [
        {"id": 999},
        {"run_attempt": 2},
        {"event": "push"},
        {"head_sha": "b" * 40},
        {"head_branch": "main"},
        {"path": ".github/workflows/other.yml"},
        {"status": "completed", "conclusion": "failure"},
        {"status": "completed", "conclusion": "success"},
    ],
)
def test_retirement_cannot_target_another_run_or_override_completed_outcomes(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, changes: dict[str, Any]
) -> None:
    api.request.side_effect = [[], _run(**changes)]
    with pytest.raises(ValueError, match="Refusing to cancel"):
        _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api) == [("GET", REF_PATH), ("GET", RUN_PATH)]


@pytest.mark.parametrize("conclusion", ["failure", "timed_out", "startup_failure", "action_required"])
@pytest.mark.parametrize("location", ["job", "step"])
def test_sibling_analysis_failures_block_retirement(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, conclusion: str, location: str
) -> None:
    jobs = _jobs()
    if location == "job":
        jobs["jobs"][0]["conclusion"] = conclusion
    else:
        jobs["jobs"][0]["steps"][0]["conclusion"] = conclusion
    api.request.side_effect = [[], _run(), jobs]
    with pytest.raises(RuntimeError, match="refusing to replace failure with cancellation"):
        _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api) == [("GET", REF_PATH), ("GET", RUN_PATH), ("GET", JOBS_PATH)]


def test_incomplete_job_listing_cannot_authorize_cancellation(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path
) -> None:
    jobs = _jobs()
    jobs["total_count"] = 101
    api.request.side_effect = [[], _run(), jobs]
    with pytest.raises(ValueError, match="Cannot verify all"):
        _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api)[-1] == ("GET", JOBS_PATH)


@pytest.mark.parametrize("conflict", [False, True])
def test_already_cancelled_workflow_has_nonzero_cli_outcome(
    *, api: MagicMock, environment: dict[str, str], sarif_file: Path, conflict: bool
) -> None:
    cancelled = _run(status="completed", conclusion="cancelled")
    responses: list[Any] = [[], cancelled]
    if conflict:
        responses = [[], _run(), _jobs(), publisher.GitHubAPIError(status=409, message="Conflict"), cancelled]
    api.request.side_effect = responses
    with patch.object(publisher, "GitHubAPI", return_value=api):
        assert _main(environment=environment, sarif_file=sarif_file) == 130
    assert _request_paths(api)[-1] == ("GET", RUN_PATH)


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "skipped", ""])
def test_analyzer_failure_or_nonexecution_cannot_publish_or_cancel(
    *, environment: dict[str, str], sarif_file: Path, outcome: str
) -> None:
    with patch.object(publisher, "GitHubAPI") as constructor:
        assert _main(environment=environment, sarif_file=sarif_file, outcome=outcome) == 1
    constructor.assert_not_called()


def test_runner_cancellation_is_nonzero_and_does_not_continue_publishing(
    *, api: MagicMock, environment: dict[str, str], sarif_file: Path
) -> None:
    api.request.side_effect = [[], _run(), _jobs(), {}]
    with (
        patch.object(publisher, "GitHubAPI", return_value=api),
        patch.object(publisher.time, "sleep", side_effect=KeyboardInterrupt),
    ):
        assert _main(environment=environment, sarif_file=sarif_file) == 130
    assert _request_paths(api)[-1] == ("POST", RUN_PATH + "/cancel")


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("GITHUB_EVENT_NAME", "push"),
        ("GITHUB_REF", "refs/heads/main"),
        ("GITHUB_SHA", "b" * 40),
        ("GITHUB_REPOSITORY", "other/repository"),
        ("GITHUB_RUN_ID", "0"),
        ("GITHUB_RUN_ATTEMPT", "-1"),
    ],
)
def test_invalid_event_identity_fails_before_api_access(
    *, environment: dict[str, str], sarif_file: Path, key: str, value: str
) -> None:
    environment[key] = value
    with patch.object(publisher, "GitHubAPI") as constructor:
        assert _main(environment=environment, sarif_file=sarif_file) == 1
    constructor.assert_not_called()


def test_event_identifies_exact_tested_merge_group(environment: dict[str, str]) -> None:
    assert publisher.load_merge_group(environment) == publisher.MergeGroup(
        repository=REPOSITORY, ref=REF, sha=SHA, run_id=123, run_attempt=1
    )


def test_queue_branch_suffix_is_not_used_as_the_tested_head_sha(environment: dict[str, str]) -> None:
    environment["GITHUB_SHA"] = "b" * 40
    event_path = Path(environment["GITHUB_EVENT_PATH"])
    event = json.loads(event_path.read_text(encoding="utf-8"))
    event["merge_group"]["head_sha"] = environment["GITHUB_SHA"]
    event_path.write_text(json.dumps(event), encoding="utf-8")
    group = publisher.load_merge_group(environment)
    assert group.ref == REF
    assert group.sha == "b" * 40


def test_wrong_sarif_category_cannot_publish_or_retire(*, environment: dict[str, str], sarif_file: Path) -> None:
    sarif_file.write_text(
        sarif_file.read_text(encoding="utf-8").replace(CATEGORY, "/language:actions"), encoding="utf-8"
    )
    with patch.object(publisher, "GitHubAPI") as constructor:
        assert _main(environment=environment, sarif_file=sarif_file) == 1
    constructor.assert_not_called()


@pytest.mark.parametrize("status", [301, 302, 400, 401, 403, 404, 429, 500, 503])
def test_http_transport_retains_status_and_never_follows_redirects(status: int) -> None:
    connection = MagicMock(spec=http.client.HTTPSConnection)
    response = MagicMock(spec=http.client.HTTPResponse)
    response.status = status
    response.read.return_value = json.dumps({"message": "Request rejected"}).encode()
    connection.getresponse.return_value = response
    with patch.object(publisher.http.client, "HTTPSConnection", return_value=connection) as constructor:
        client = publisher.GitHubAPI(api_url="https://github.example/api/v3", token="offline-test-token")
        with pytest.raises(publisher.GitHubAPIError) as raised:
            client.request(method="GET", path=REF_PATH, expected_status=200)
    assert raised.value.status == status
    assert raised.value.message == "Request rejected"
    constructor.assert_called_once_with("github.example", timeout=30)
    assert connection.request.call_args.args[:2] == ("GET", "/api/v3" + REF_PATH)
    connection.request.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize(
    "api_url", ["http://github.example", "https://token@github.example", "https://github.example?x=1"]
)
def test_http_transport_rejects_insecure_or_credential_bearing_urls(api_url: str) -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        publisher.GitHubAPI(api_url=api_url, token="offline-test-token")


def test_workflow_preserves_required_names_languages_events_and_native_diagnostics() -> None:
    workflow = yaml.load(
        (REPO_ROOT / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    assert set(workflow["on"]) == {"push", "pull_request", "merge_group", "workflow_dispatch", "schedule"}
    assert workflow["on"]["push"]["branches"] == ["main", "releases/v*"]
    assert workflow["on"]["pull_request"]["branches"] == ["main", "releases/**"]
    assert workflow["on"]["schedule"] == [{"cron": "23 2 * * 1"}]
    assert workflow["permissions"] == {}
    assert set(workflow["jobs"]) == {"analyze"}
    job = workflow["jobs"]["analyze"]
    assert job["name"] == "CodeQL (${{ matrix.language }})"
    assert job["strategy"] == {
        "fail-fast": "false",
        "matrix": {"language": ["actions", "javascript-typescript", "python"]},
    }
    assert job["permissions"] == {"actions": "write", "contents": "read", "security-events": "write"}
    assert "if" not in job
    assert "continue-on-error" not in job
    assert all("continue-on-error" not in step for step in job["steps"])
    steps = {step["name"]: step for step in job["steps"]}
    assert steps["Checkout"]["with"] == {"ref": "${{ github.sha }}", "persist-credentials": "false"}
    analyze = steps["Perform CodeQL Analysis"]
    assert "if" not in analyze
    assert analyze["uses"] == "github/codeql-action/analyze@2892aa5e19bbd11bc0cff5427e3b750a04d9e3c2"
    assert analyze["with"]["category"] == "/language:${{ matrix.language }}"
    assert analyze["with"]["upload"] == "failure-only"
    assert analyze["with"]["post-processed-sarif-path"] == "${{ runner.temp }}/codeql-upload"
    native = steps["Publish CodeQL results"]
    assert native["uses"] == "github/codeql-action/upload-sarif@2892aa5e19bbd11bc0cff5427e3b750a04d9e3c2"
    assert native["with"] == {
        "sarif_file": "${{ runner.temp }}/codeql-upload/upload.sarif",
        "category": analyze["with"]["category"],
    }
    assert native["if"] == "github.event_name != 'merge_group'"
    publish = steps["Publish merge-group CodeQL results"]
    assert publish["if"] == "github.event_name == 'merge_group'"
    assert steps["Set up merge-group upload helper"]["if"] == publish["if"]
    assert publish["env"]["ANALYSIS_OUTCOME"] == "${{ steps.analysis.outcome }}"
    assert publish["env"]["SARIF_FILE"] == "${{ runner.temp }}/codeql-upload/upload.sarif"
    assert publish["env"]["SARIF_CATEGORY"] == analyze["with"]["category"]
    assert publish["timeout-minutes"] == "8"
    assert "uv run --no-project python -m build_scripts.codeql_merge_group" in publish["run"]


@pytest.mark.parametrize(
    ("event_name", "fork"),
    [
        ("push", False),
        ("pull_request", False),
        ("pull_request", True),
        ("schedule", False),
        ("workflow_dispatch", False),
        ("merge_group", False),
    ],
)
def test_normal_and_fork_events_select_native_upload_and_never_retirement(*, event_name: str, fork: bool) -> None:
    workflow = yaml.load(
        (REPO_ROOT / ".github" / "workflows" / "codeql.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader
    )
    steps = {step["name"]: step for step in workflow["jobs"]["analyze"]["steps"]}
    selected = []
    for name in ("Publish CodeQL results", "Set up merge-group upload helper", "Publish merge-group CodeQL results"):
        condition = steps[name]["if"]
        comparison = re.fullmatch(r"github\.event_name (==|!=) '([^']+)'", condition)
        assert comparison is not None
        matches = event_name == comparison[2]
        if matches == (comparison[1] == "=="):
            selected.append(name)
    if event_name == "merge_group":
        assert selected == ["Set up merge-group upload helper", "Publish merge-group CodeQL results"]
    else:
        assert selected == ["Publish CodeQL results"]
    if fork:
        assert "pull_request_target" not in workflow["on"]
        assert not {"ref", "sha", "token"} & steps["Publish CodeQL results"]["with"].keys()


@pytest.mark.parametrize(
    "error",
    [
        publisher.GitHubAPIError(status=403, message="Forbidden"),
        publisher.GitHubAPIError(status=429, message="Rate limited"),
        publisher.GitHubAPIError(status=503, message="Unavailable"),
        OSError("Network failure"),
    ],
)
def test_cli_reports_api_and_network_failures_as_nonzero(
    *,
    api: MagicMock,
    environment: dict[str, str],
    sarif_file: Path,
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    api.request.side_effect = error
    with patch.object(publisher, "GitHubAPI", return_value=api):
        assert _main(environment=environment, sarif_file=sarif_file) == 1
    assert "::error::" in capsys.readouterr().err
    assert _request_paths(api) == [("GET", REF_PATH)]


def test_cli_only_succeeds_after_analysis_and_live_upload_processing(
    *, api: MagicMock, environment: dict[str, str], sarif_file: Path
) -> None:
    api.request.side_effect = [_refs(), {"id": UPLOAD_ID}, {"processing_status": "complete"}, _refs()]
    with patch.object(publisher, "GitHubAPI", return_value=api), patch.object(publisher.time, "sleep"):
        assert _main(environment=environment, sarif_file=sarif_file) == 0
    assert _request_paths(api)[-1] == ("GET", REF_PATH)


@pytest.mark.parametrize(
    "response",
    [None, {}, [{"ref": REF, "object": None}], _refs() * 2, [{"ref": REF, "object": {"type": "tag", "sha": SHA}}]],
)
def test_invalid_ref_response_cannot_authorize_upload_or_retirement(
    *, api: MagicMock, group: publisher.MergeGroup, sarif_file: Path, response: Any
) -> None:
    api.request.return_value = response
    with pytest.raises(ValueError):
        _publish(api=api, group=group, sarif_file=sarif_file)
    assert _request_paths(api) == [("GET", REF_PATH)]


@pytest.mark.parametrize(
    "document", [[], {"version": "2.1.0", "runs": []}, {"version": "2.1.0", "runs": [{"tool": None}]}]
)
def test_invalid_prepared_sarif_cannot_authorize_retirement(
    *, environment: dict[str, str], sarif_file: Path, document: Any
) -> None:
    sarif_file.write_text(json.dumps(document), encoding="utf-8")
    with patch.object(publisher, "GitHubAPI") as constructor:
        assert _main(environment=environment, sarif_file=sarif_file) == 1
    constructor.assert_not_called()


@pytest.mark.parametrize("state", ["live", "obsolete", "upload-race"])
def test_cli_exercises_real_http_adapter_with_only_mocked_connections(
    *, environment: dict[str, str], sarif_file: Path, state: str
) -> None:
    responses: list[tuple[int, Any]] = [
        (200, _refs()),
        (202, {"id": UPLOAD_ID}),
        (200, {"processing_status": "complete"}),
        (200, _refs()),
    ]
    if state != "live":
        responses = [(200, []), (200, _run()), (200, _jobs()), (202, None)]
        if state == "upload-race":
            responses = [(200, _refs()), (404, {"message": f"ref '{REF}' not found in this repository"}), *responses]
    connections = []
    for status, body in responses:
        connection = MagicMock(spec=http.client.HTTPSConnection)
        response = MagicMock(spec=http.client.HTTPResponse)
        response.status = status
        response.read.return_value = json.dumps(body).encode() if body is not None else b""
        connection.getresponse.return_value = response
        connections.append(connection)
    sleeps = None if state == "live" else KeyboardInterrupt
    with (
        patch.object(publisher.http.client, "HTTPSConnection", side_effect=connections),
        patch.object(publisher.time, "sleep", side_effect=sleeps),
    ):
        assert _main(environment=environment, sarif_file=sarif_file) == (0 if state == "live" else 130)
    for connection in connections:
        connection.request.assert_called_once()
        connection.close.assert_called_once()
    last_request = connections[-1].request.call_args
    assert last_request.args[:2] == (("GET", REF_PATH) if state == "live" else ("POST", RUN_PATH + "/cancel"))
    if state == "live":
        payload = json.loads(connections[1].request.call_args.kwargs["body"])
        assert set(payload) == {"commit_sha", "ref", "sarif", "checkout_uri", "tool_name"}
        assert gzip.decompress(base64.b64decode(payload["sarif"])) == sarif_file.read_bytes()
