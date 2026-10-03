# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from build_scripts.pyrit_wrapped.github_client import GitHubClient, GitHubTransportError, Response
from build_scripts.pyrit_wrapped.models import WrappedError


@pytest.fixture
def client(tmp_path: Path) -> GitHubClient:
    return GitHubClient(cache_dir=tmp_path / "cache", progress=lambda message: None)


def _response(*, data: object, next_page: int | None = None) -> Response:
    return Response.model_validate({"fetched_at": datetime.now(UTC), "next_page": next_page, "data": data})


async def test_cache_replays_and_discards_bodies(*, client: GitHubClient, tmp_path: Path) -> None:
    output = (
        'HTTP/2.0 200 OK\r\nContent-Type: application/json\r\n\r\n{"body": "not retained", "email": "not retained"}'
    )
    with patch.object(client, "_run_async", new_callable=AsyncMock) as run:
        run.return_value = (0, output)
        first = await client.get_async(path="users/owner")
        second = await client.get_async(path="users/owner")
        assert first == second
        assert run.call_count == 1
    assert first.data == {"_wrapped_has_body": True}
    assert all("not retained" not in path.read_text(encoding="utf-8") for path in (tmp_path / "cache").glob("*.json"))


async def test_corrupt_cache_is_explicit(client: GitHubClient) -> None:
    with patch.object(client, "_run_async", new_callable=AsyncMock) as run:
        run.return_value = (0, "HTTP/2.0 200 OK\n\n{}")
        await client.get_async(path="users/owner")
    next(client.cache_dir.glob("*.json")).write_text("{", encoding="utf-8")
    with pytest.raises(WrappedError, match="Invalid request cache"):
        await client.get_async(path="users/owner")


async def test_http_rate_limit_retries(client: GitHubClient) -> None:
    limited = 'HTTP/2.0 429 Too Many Requests\nRetry-After: 0\n\n{"message": "rate limit"}'
    with patch.object(client, "_run_async", new_callable=AsyncMock) as run:
        with patch.object(client, "_pace_async", new_callable=AsyncMock):
            run.side_effect = [(1, limited), (0, "HTTP/2.0 200 OK\n\n{}")]
            assert (await client.get_async(path="users/owner")).data == {}
            assert run.call_count == 2


async def test_permission_errors_do_not_become_zero(client: GitHubClient) -> None:
    with patch.object(client, "_run_async", new_callable=AsyncMock) as run:
        run.return_value = (1, 'HTTP/2.0 403 Forbidden\n\n{"message": "Resource not accessible"}')
        with pytest.raises(WrappedError, match="HTTP 403"):
            await client.get_async(path="users/owner")
        assert run.call_count == 1
    assert not client.cache_dir.exists()


async def test_endpoint_allowlist(client: GitHubClient) -> None:
    with pytest.raises(WrappedError, match="unexpected"):
        await client.get_async(path="https://evil.example/token")


async def test_missing_gh_explains_authentication(client: GitHubClient) -> None:
    with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock, side_effect=FileNotFoundError):
        with pytest.raises(WrappedError, match="GitHub CLI"):
            await client._run_async(path="users/owner", params={})


async def test_pagination_follows_links(client: GitHubClient) -> None:
    with patch.object(client, "get_async", new_callable=AsyncMock) as get:
        get.side_effect = [_response(data=[{"id": 1}], next_page=2), _response(data=[{"id": 2}])]
        assert await client.list_async(path="repos/microsoft/PyRIT/issues/comments") == [{"id": 1}, {"id": 2}]
        assert get.call_args_list[1].kwargs["params"]["page"] == "2"


async def test_pagination_rejects_loops(client: GitHubClient) -> None:
    with patch.object(client, "get_async", new_callable=AsyncMock, return_value=_response(data=[], next_page=1)):
        with pytest.raises(WrappedError, match="Nonadvancing"):
            await client.list_async(path="repos/microsoft/PyRIT/issues/comments")


async def test_comment_cutoff_uses_created_order(client: GitHubClient) -> None:
    with patch.object(client, "get_async", new_callable=AsyncMock) as get:
        get.return_value = _response(data=[{"created_at": "2027-01-01T00:00:00Z"}], next_page=2)
        await client.list_async(
            path="repos/microsoft/PyRIT/issues/comments", stop_after=datetime(2027, 1, 1, tzinfo=UTC)
        )
        assert get.call_count == 1


@pytest.mark.parametrize("total", [1000, 1001])
async def test_search_shards_above_pagination_limit(*, client: GitHubClient, total: int) -> None:
    root = _response(data={"total_count": total, "incomplete_results": False, "items": []})
    responses = [root]
    for first, last in ((1, 501), (501, total + 1)):
        numbers = list(range(first, last))
        for offset in range(0, len(numbers), 100):
            page = offset // 100 + 1
            responses.append(
                _response(
                    data={
                        "total_count": len(numbers),
                        "incomplete_results": False,
                        "items": [{"number": number} for number in numbers[offset : offset + 100]],
                    },
                    next_page=page + 1 if offset + 100 < len(numbers) else None,
                )
            )
    with patch.object(client, "get_async", new_callable=AsyncMock, side_effect=responses) as get:
        values = await client._search_range_async(
            query="is:pr reviewed-by:owner",
            date_field="created",
            start=datetime(2026, 1, 1, tzinfo=UTC),
            end=datetime(2026, 1, 3, tzinfo=UTC),
        )
        assert values == set(range(1, total + 1))
        assert get.call_count == len(responses)


async def test_search_incomplete_one_second_fails(client: GitHubClient) -> None:
    response = _response(data={"total_count": 1, "incomplete_results": True, "items": [{"number": 1}]})
    start = datetime(2026, 1, 1, tzinfo=UTC)
    with patch.object(client, "get_async", new_callable=AsyncMock, return_value=response):
        with pytest.raises(WrappedError, match="one-second"):
            await client._search_range_async(
                query="is:pr", date_field="created", start=start, end=start + timedelta(seconds=1)
            )


async def test_search_deduplication_catches_truncation(client: GitHubClient) -> None:
    response = _response(data={"total_count": 2, "incomplete_results": False, "items": [{"number": 1}, {"number": 1}]})
    with patch.object(client, "get_async", new_callable=AsyncMock, return_value=response):
        with pytest.raises(WrappedError, match="fewer unique"):
            await client.search_async(
                query="is:pr",
                date_field="created",
                start=datetime(2026, 1, 1, tzinfo=UTC),
                end=datetime(2026, 2, 1, tzinfo=UTC),
            )


def test_pagination_parser() -> None:
    link = '<https://api.github.com/repos/microsoft/PyRIT/pulls?page=2>; rel="next"'
    assert GitHubClient._next_page(link) == 2
    with pytest.raises(WrappedError, match="host"):
        GitHubClient._next_page('<https://evil.example/?page=2>; rel="next"')


async def test_recognized_transport_failures_retry(client: GitHubClient) -> None:
    with patch.object(client, "_run_async", new_callable=AsyncMock) as run:
        with patch.object(client, "_pace_async", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                run.side_effect = [GitHubTransportError("dial tcp timeout"), (0, "HTTP/2.0 200 OK\n\n{}")]
                assert (await client.get_async(path="users/owner")).data == {}
                assert run.call_count == 2


async def test_transport_retry_budget_is_bounded(client: GitHubClient) -> None:
    with patch.object(
        client, "_run_async", new_callable=AsyncMock, side_effect=GitHubTransportError("dial tcp timeout")
    ) as run:
        with patch.object(client, "_pace_async", new_callable=AsyncMock):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(GitHubTransportError):
                    await client.get_async(path="users/owner")
                assert run.call_count == 3
