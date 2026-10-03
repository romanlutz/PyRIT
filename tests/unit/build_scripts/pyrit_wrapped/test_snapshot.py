# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

from build_scripts.pyrit_wrapped.github_client import GitHubClient, Response
from build_scripts.pyrit_wrapped.metrics import Metrics
from build_scripts.pyrit_wrapped.models import Activity, Period, Snapshot, TaxonomyConfig, WorkItem, WrappedError
from build_scripts.pyrit_wrapped.snapshot import ApiReview, Collector


def test_review_mapping_does_not_need_body_text() -> None:
    value = ApiReview.model_validate(
        {
            "id": 1,
            "user": {"node_id": "U_owner", "login": "owner"},
            "state": "APPROVED",
            "submitted_at": "2026-01-02T00:00:00Z",
            "html_url": "https://github.com/microsoft/PyRIT/pull/1#pullrequestreview-1",
            "_wrapped_has_body": True,
        }
    ).to_review(1)
    assert value.has_body
    assert "body" not in value.model_dump()


async def test_file_ceiling_is_visible(item: WorkItem) -> None:
    client = MagicMock(spec=GitHubClient)
    client.list_async = AsyncMock(return_value=[{"filename": f"file{number}.py"} for number in range(3000)])
    collector = Collector(client=client, progress=lambda message: None)
    item = item.model_copy(update={"changed_files": 3001})
    warnings = await collector._load_files_async({item.number: item})
    assert not item.files_complete
    assert len(warnings) == 1
    assert "3000 of 3001" in warnings[0]


async def test_collector_runs_with_typed_records(
    *, period: Period, taxonomy_config: TaxonomyConfig, snapshot: Snapshot
) -> None:
    owner = {"node_id": "U_owner", "login": "owner", "type": "User"}
    other = {"node_id": "U_other", "login": "other", "type": "User"}
    base = {
        "created_at": "2026-01-02T00:00:00Z",
        "updated_at": "2026-01-03T00:00:00Z",
        "state": "closed",
        "labels": [],
        "changed_files": 1,
    }
    items = {
        1: {
            **base,
            "node_id": "PR_1",
            "number": 1,
            "title": "FIX converter",
            "user": owner,
            "html_url": "https://github.com/microsoft/PyRIT/pull/1",
            "merged_by": owner,
            "merged_at": "2026-01-03T00:00:00Z",
        },
        2: {
            **base,
            "node_id": "PR_2",
            "number": 2,
            "title": "FEAT target",
            "user": other,
            "html_url": "https://github.com/microsoft/PyRIT/pull/2",
            "merged_by": owner,
            "merged_at": "2026-01-03T00:00:00Z",
        },
        3: {
            **base,
            "node_id": "I_3",
            "number": 3,
            "title": "New scorer",
            "user": owner,
            "html_url": "https://github.com/microsoft/PyRIT/issues/3",
        },
    }

    async def get_async(*, path: str, params: dict[str, str] | None = None) -> Response:
        if path == "users/owner":
            data = owner
        elif path == "repos/microsoft/PyRIT":
            data = {"full_name": "microsoft/PyRIT", "private": False, "created_at": "2024-01-01T00:00:00Z"}
        else:
            data = items[int(path.rsplit("/", 1)[1])]
        return Response.model_validate({"fetched_at": period.cutoff, "next_page": None, "data": data})

    async def list_async(
        *, path: str, params: dict[str, str] | None = None, stop_after: datetime | None = None
    ) -> list[dict[str, object]]:
        if path.endswith("/reviews"):
            return [
                {
                    "id": 100,
                    "user": owner,
                    "state": "COMMENTED",
                    "submitted_at": "2026-01-04T00:00:00Z",
                    "html_url": "https://github.com/microsoft/PyRIT/pull/2#pullrequestreview-100",
                    "_wrapped_has_body": True,
                }
            ]
        if path.endswith("/files"):
            return [{"filename": "pyrit/prompt_target/example.py"}]
        return []

    client = MagicMock(spec=GitHubClient)
    client.response_times = [period.cutoff]
    client.get_async = AsyncMock(side_effect=get_async)
    client.list_async = AsyncMock(side_effect=list_async)
    client.search_async = AsyncMock(side_effect=[{1}, {3}, {1, 2}, {2}])
    result = await Collector(client=client, progress=lambda message: None).collect_async(
        login="owner", period=period, taxonomy=taxonomy_config
    )
    replay = Snapshot.model_validate_json(result.model_dump_json())
    stats = Metrics(replay).calculate()
    assert stats.counts[Activity.AUTHORED] == stats.counts[Activity.LANDED] == stats.counts[Activity.ISSUES] == 1
    assert stats.counts[Activity.MERGED] == 2
    assert stats.counts[Activity.REVIEWED] == stats.counts[Activity.REVIEW_BODIES] == 1
    assert stats.own_prs_merged == stats.other_prs_merged == 1


async def test_private_repository_is_rejected(*, period: Period, taxonomy_config: TaxonomyConfig) -> None:
    client = MagicMock(spec=GitHubClient)
    client.get_async = AsyncMock(
        side_effect=[
            Response(fetched_at=period.cutoff, next_page=None, data={"node_id": "U_owner", "login": "owner"}),
            Response(
                fetched_at=period.cutoff,
                next_page=None,
                data={"full_name": "microsoft/PyRIT", "private": True, "created_at": "2024-01-01T00:00:00Z"},
            ),
        ]
    )
    with pytest.raises(WrappedError, match="public"):
        await Collector(client=client, progress=lambda message: None).collect_async(
            login="owner", period=period, taxonomy=taxonomy_config
        )
