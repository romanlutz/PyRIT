# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

from pydantic import AwareDatetime, Field, JsonValue, TypeAdapter, ValidationError

from build_scripts.pyrit_wrapped.models import Model, WrappedError

if TYPE_CHECKING:
    from collections.abc import Callable


class Response(Model):
    version: int = 1
    fetched_at: AwareDatetime
    next_page: int | None
    data: JsonValue


class SearchItem(Model):
    number: int = Field(gt=0)

    model_config = {"extra": "ignore"}


class SearchPage(Model):
    total_count: int = Field(ge=0)
    incomplete_results: bool
    items: list[SearchItem]

    model_config = {"extra": "ignore"}


class GitHubTransportError(WrappedError):
    """A recognized transient connection or request timeout."""


class GitHubHttpError(WrappedError):
    def __init__(self, *, status: int, path: str, message: str) -> None:
        self.status = status
        self.path = path
        super().__init__(message)


class GitHubClient:
    _MAX_ATTEMPTS = 3
    _MAX_WAIT_SECONDS = 120
    _TIMEOUT_SECONDS = 90
    _CACHE_LIFETIME = timedelta(hours=24)
    _SENSITIVE_FIELDS = {
        "body_html",
        "body_text",
        "diff_hunk",
        "patch",
        "email",
        "gravatar_id",
        "permissions",
        "security_and_analysis",
    }
    _TRANSIENT_MESSAGES = (
        "dial tcp",
        "connectex:",
        "i/o timeout",
        "tls handshake timeout",
        "connection reset",
        "connection refused",
        "unexpected eof",
    )

    def __init__(
        self,
        *,
        cache_dir: Path,
        refresh: bool = False,
        progress: Callable[[str], None],
    ) -> None:
        self.cache_dir = cache_dir
        self.refresh = refresh
        self.progress = progress
        self.response_times: list[datetime] = []
        self._semaphore = asyncio.Semaphore(6)
        self._clock_lock = asyncio.Lock()
        self._next_request = 0.0
        self._next_search = 0.0

    async def get_async(self, *, path: str, params: dict[str, str] | None = None) -> Response:
        if not re.fullmatch(r"(?:users/[A-Za-z0-9-]+|repos/microsoft/PyRIT(?:/[\w/-]+)?|search/issues)", path):
            raise WrappedError(f"Refusing an unexpected GitHub API endpoint: {path}")
        params = params or {}
        key = hashlib.sha256(json.dumps([path, sorted(params.items())]).encode()).hexdigest()
        cache_file = self.cache_dir / f"{key}.json"
        cached = await asyncio.to_thread(self._read_cache, cache_file)
        if cached is not None and not self.refresh:
            self.response_times.append(cached.fetched_at)
            return cached
        async with self._semaphore:
            response = await self._request_async(path=path, params=params)
            await asyncio.to_thread(write_json_atomic, path=cache_file, content=response.model_dump_json())
        self.response_times.append(response.fetched_at)
        return response

    async def list_async(
        self,
        *,
        path: str,
        params: dict[str, str] | None = None,
        stop_after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        page: int | None = 1
        while page is not None:
            response = await self.get_async(path=path, params={**(params or {}), "per_page": "100", "page": str(page)})
            if not isinstance(response.data, list) or any(not isinstance(item, dict) for item in response.data):
                raise WrappedError(f"Expected a paginated object list from {path}.")
            batch = [object_data(item) for item in response.data]
            records.extend(batch)
            if stop_after is not None and self._past_cutoff(batch=batch, cutoff=stop_after):
                break
            if response.next_page is not None and response.next_page <= page:
                raise WrappedError(f"Nonadvancing pagination from {path}.")
            page = response.next_page
        return records

    async def search_async(self, *, query: str, date_field: str, start: datetime, end: datetime) -> set[int]:
        if end <= start:
            return set()
        if date_field not in {"created", "merged"}:
            raise WrappedError(f"Unsupported search date field: {date_field}")
        start = start.replace(microsecond=0)
        # Search timestamps have second precision. Local event filtering preserves the exact cutoff.
        end = end.replace(microsecond=0) + timedelta(seconds=1)
        return await self._search_range_async(query=query, date_field=date_field, start=start, end=end)

    async def _search_range_async(self, *, query: str, date_field: str, start: datetime, end: datetime) -> set[int]:
        qualifier = f"{date_field}:{iso_time(start)}..{iso_time(end - timedelta(seconds=1))}"
        params = {"q": f"repo:microsoft/PyRIT {query} {qualifier}", "per_page": "100", "page": "1"}
        response = await self.get_async(path="search/issues", params=params)
        first = SearchPage.model_validate(response.data)
        if first.total_count >= 1000 or first.incomplete_results:
            if end - start <= timedelta(seconds=1):
                raise WrappedError("GitHub search cannot be completely retrieved in a one-second shard.")
            seconds = int((end - start).total_seconds()) // 2
            middle = start + timedelta(seconds=max(1, seconds))
            left = await self._search_range_async(query=query, date_field=date_field, start=start, end=middle)
            right = await self._search_range_async(query=query, date_field=date_field, start=middle, end=end)
            return left | right
        numbers = {item.number for item in first.items}
        page = response.next_page
        while page is not None:
            response = await self.get_async(path="search/issues", params={**params, "page": str(page)})
            batch = SearchPage.model_validate(response.data)
            if batch.incomplete_results or batch.total_count != first.total_count:
                raise WrappedError(
                    "GitHub search changed or became incomplete during pagination; retry with --refresh."
                )
            numbers.update(item.number for item in batch.items)
            next_page = response.next_page
            if next_page is not None and next_page <= page:
                raise WrappedError("GitHub search returned nonadvancing pagination.")
            page = next_page
        if len(numbers) != first.total_count:
            raise WrappedError("GitHub search returned fewer unique records than its total; retry with --refresh.")
        return numbers

    async def _request_async(self, *, path: str, params: dict[str, str]) -> Response:
        for attempt in range(self._MAX_ATTEMPTS):
            await self._pace_async(search=path == "search/issues")
            try:
                code, output = await self._run_async(path=path, params=params)
            except GitHubTransportError:
                if attempt == self._MAX_ATTEMPTS - 1:
                    raise
                delay = 2 ** (attempt + 1)
                self.progress(f"Temporary GitHub connection failure; retrying after {delay} seconds.")
                await asyncio.sleep(delay)
                continue
            status, headers, data = self._parse_response(output)
            if code == 0 and 200 <= status < 300:
                return Response(
                    fetched_at=datetime.now(UTC),
                    next_page=self._next_page(headers.get("link", "")),
                    data=self._sanitize(data),
                )
            message = object_data(data).get("message", "GitHub request failed") if isinstance(data, dict) else ""
            delay = self._retry_delay(status=status, headers=headers, message=str(message), attempt=attempt)
            if delay is None or attempt == self._MAX_ATTEMPTS - 1 or delay > self._MAX_WAIT_SECONDS:
                raise GitHubHttpError(
                    status=status,
                    path=path,
                    message=f"GitHub {path} failed (HTTP {status}). {message}. "
                    "Check gh authentication/access or retry later; completed requests are cached.",
                )
            self.progress(f"GitHub HTTP {status}; retrying after {delay:.0f} seconds.")
            await asyncio.sleep(delay)
        raise WrappedError("GitHub retry budget exhausted.")

    async def _run_async(self, *, path: str, params: dict[str, str]) -> tuple[int, str]:
        args = ["gh", "api", "--hostname", "github.com", "--method", "GET", "--include", path]
        args.extend(["-H", "Accept: application/vnd.github+json", "-H", "X-GitHub-Api-Version: 2022-11-28"])
        for key, value in sorted(params.items()):
            args.extend(["-f", f"{key}={value}"])
        try:
            process = await asyncio.create_subprocess_exec(
                *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
        except FileNotFoundError as error:
            raise WrappedError("Install GitHub CLI and run gh auth login before collecting a report.") from error
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self._TIMEOUT_SECONDS)
        except TimeoutError as error:
            raise GitHubTransportError(f"GitHub request timed out: {path}. Completed requests are cached.") from error
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
        output = stdout.decode("utf-8")
        if not output.startswith("HTTP/"):
            detail = stderr.decode("utf-8").strip()
            if any(message in detail.lower() for message in self._TRANSIENT_MESSAGES):
                raise GitHubTransportError(f"Temporary GitHub connection failure for {path}: {detail}")
            raise WrappedError(f"GitHub CLI could not request {path}: {detail}")
        return process.returncode or 0, output

    async def _pace_async(self, *, search: bool) -> None:
        async with self._clock_lock:
            now = time.monotonic()
            target = max(now, self._next_request, self._next_search if search else now)
            await asyncio.sleep(target - now)
            self._next_request = target + 0.12
            if search:
                self._next_search = target + 2.1

    def _read_cache(self, path: Path) -> Response | None:
        if self.refresh or not path.exists():
            return None
        try:
            cached = Response.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise WrappedError(f"Invalid request cache {path}; retry with --refresh.") from error
        if cached.version != 1:
            raise WrappedError(f"Unsupported request cache version in {path}; retry with --refresh.")
        age = datetime.now(UTC) - cached.fetched_at
        if age < timedelta(0):
            raise WrappedError(f"Request cache has a future timestamp: {path}")
        return cached if age < self._CACHE_LIFETIME else None

    @classmethod
    def _sanitize(cls, value: JsonValue) -> JsonValue:
        if isinstance(value, list):
            return [cls._sanitize(item) for item in value]
        if isinstance(value, dict):
            result = {
                key: cls._sanitize(item) for key, item in value.items() if key not in cls._SENSITIVE_FIELDS | {"body"}
            }
            if "body" in value:
                body = value["body"]
                result["_wrapped_has_body"] = isinstance(body, str) and bool(body.strip())
            return result
        return value

    @staticmethod
    def _parse_response(output: str) -> tuple[int, dict[str, str], JsonValue]:
        head, separator, body = output.replace("\r\n", "\n").partition("\n\n")
        if not separator:
            raise WrappedError("GitHub CLI returned no HTTP response body.")
        lines = head.splitlines()
        status = int(lines[0].split()[1])
        headers = dict(line.lower().split(": ", 1) for line in lines[1:] if ": " in line)
        try:
            return status, headers, TypeAdapter[JsonValue](JsonValue).validate_json(body)
        except ValidationError as error:
            raise WrappedError("GitHub CLI returned malformed JSON.") from error

    @staticmethod
    def _next_page(link: str) -> int | None:
        for part in link.split(","):
            if 'rel="next"' not in part:
                continue
            match = re.search(r"<([^>]+)>", part)
            if match is None:
                raise WrappedError("Malformed GitHub pagination link.")
            parsed = urlparse(match[1])
            if parsed.scheme != "https" or parsed.netloc != "api.github.com":
                raise WrappedError("Unexpected GitHub pagination host.")
            page = parse_qs(parsed.query).get("page", [])
            if len(page) != 1 or not page[0].isdigit() or int(page[0]) < 1:
                raise WrappedError("Missing or invalid GitHub pagination page.")
            return int(page[0])
        return None

    @staticmethod
    def _retry_delay(*, status: int, headers: dict[str, str], message: str, attempt: int) -> float | None:
        if status not in {403, 429, 500, 502, 503, 504}:
            return None
        if "retry-after" in headers:
            return max(0, float(headers["retry-after"]))
        if headers.get("x-ratelimit-remaining") == "0":
            return max(1, float(headers["x-ratelimit-reset"]) - time.time() + 1)
        if status in {403, 429}:
            return 60 if status == 429 or "rate limit" in message.lower() else None
        return float(2 ** (attempt + 1))

    @staticmethod
    def _past_cutoff(*, batch: list[dict[str, Any]], cutoff: datetime) -> bool:
        values = [item["created_at"] for item in batch if "created_at" in item]
        return bool(values) and max(datetime.fromisoformat(value.replace("Z", "+00:00")) for value in values) >= cutoff


def object_data(value: JsonValue) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WrappedError("GitHub returned an object with an unexpected shape.")
    return value


def iso_time(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def write_json_atomic(*, path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".wrapped-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
