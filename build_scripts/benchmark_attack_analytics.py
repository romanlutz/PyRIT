# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Measure real analytics routes on an isolated, synthetic SQLite database."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import random
import sqlite3
import sys
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import FastAPI, Request
from sqlalchemy import event, insert, update

from pyrit.analytics import AttackResultAnalytics
from pyrit.backend.middleware.auth import AuthenticatedUser
from pyrit.backend.middleware.error_handlers import register_error_handlers
from pyrit.backend.routes.analytics import router
from pyrit.backend.services.analytics_service import get_analytics_service
from pyrit.memory import SQLiteMemory
from pyrit.memory.memory_models import AttackResultEntry, Base
from pyrit.models import AtomicAttackIdentifier, AttackIdentifier, ConverterIdentifier, TargetIdentifier

if TYPE_CHECKING:
    from collections.abc import Sequence

    from starlette.middleware.base import RequestResponseEndpoint
    from starlette.responses import Response


class AnalyticsBenchmark:
    """Exercise the production route/SDK/query stack without production data."""

    def __init__(self, *, directory: Path, count: int, journal_mode: str) -> None:
        """Create only a new scratch database in the supplied temporary directory."""
        self.path = directory / "analytics.sqlite"
        memory = SQLiteMemory(db_path=self.path, skip_schema_migration=True)
        if not isinstance(memory, SQLiteMemory):
            raise TypeError("Expected a SQLite memory instance")
        self.memory = memory
        engine = memory.engine
        if engine is None:
            raise RuntimeError("The benchmark database engine was not initialized")
        self.engine = engine
        self.memory.results_path = str(directory)
        Base.metadata.create_all(self.engine)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(f"PRAGMA journal_mode={journal_mode}")
        self.count = count
        self.journal_mode = journal_mode
        self.service = AttackResultAnalytics(memory=self.memory)
        self.app = self._app()
        self._stop_writer = threading.Event()
        self._write_samples: list[float] = []
        self._failures: list[dict[str, Any]] = []
        self._statement_count = 0
        self._statement_lock = threading.Lock()
        self._writer_pool = ThreadPoolExecutor(max_workers=1)
        self._environment = {
            "python": platform.python_version(),
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        }

    def seed(self) -> None:
        """
        Populate real identifier projections and skewed synthetic metadata.

        Identifiers are shared across many result rows, while case labels are
        high-cardinality. Two IDs deliberately share each main conversation to
        detect accidental conversation-based deduplication. Legacy/missing
        identifier variants exercise the same paths as stored historical results.
        """
        identifiers = self._identifiers()
        with closing(self.memory.get_session()) as session:
            for identifier in identifiers:
                self.memory._persist_identifier(session=session, identifier=identifier)
            session.commit()
        serialized = [(identifier.hash, identifier.model_dump()) for identifier in identifiers]
        generator = random.Random(20260914)
        start = datetime(2026, 1, 1, tzinfo=UTC)
        outcomes = ("success", "failure", "error", "undetermined")
        with self.engine.begin() as connection:
            for batch_start in range(0, self.count, 2500):
                rows = []
                for index in range(batch_start, min(self.count, batch_start + 2500)):
                    identifier_hash, identifier = serialized[generator.randrange(len(serialized))]
                    category = generator.randrange(40)
                    rows.append(
                        {
                            "id": uuid.UUID(int=index + 1),
                            "conversation_id": str(uuid.UUID(int=index // 2 + 1)),
                            "objective": f"Synthetic objective {index}. " + "Safe evaluation metadata. " * 6,
                            "outcome": outcomes[generator.randrange(4)],
                            "timestamp": start + timedelta(seconds=index),
                            "operation": (
                                "operation-000" if index < self.count * 0.6 else f"operation-{index % 100:03}"
                            ),
                            "operator": f"operator-{index % 50:03}",
                            "labels": {"team": f"team-{index % 10}", "case": f"case-{index:06}"},
                            "targeted_harm_categories": [
                                f"category-{(category + offset) % 40:02}" for offset in range(3)
                            ],
                            "atomic_attack_identifier_hash": identifier_hash if index % 10 else None,
                            "atomic_attack_identifier": identifier if index % 100 else None,
                            "executed_turns": 0,
                            "execution_time_ms": 0,
                        }
                    )
                connection.execute(insert(AttackResultEntry), rows)

    async def run_async(self, *, analysts: int, iterations: int) -> dict[str, Any]:
        """
        Measure successful HTTP/ASGI requests with concurrent result writes.

        Returns:
            dict[str, Any]: Reproducible workload metadata and observed latency.
        """
        event.listen(self.engine, "before_cursor_execute", self._record_statement)
        samples: dict[str, list[float]] = {"report": [], "facet": [], "page": [], "continuation": []}
        cold: dict[str, float] = {}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://benchmark"
        ) as client:
            cursor = None
            for kind in samples:
                cold[kind], cursor = await self._request_async(client=client, kind=kind, iteration=0, cursor=cursor)
        print(f"Cold request timings: {cold}", file=sys.stderr, flush=True)
        started = time.perf_counter()
        writer = self._writer_pool.submit(self._write)
        try:
            measurements = await asyncio.gather(
                *[self._analyst_async(index=index, iterations=iterations) for index in range(analysts)]
            )
            for measurement in measurements:
                for kind, latency in measurement:
                    samples[kind].append(latency)
        finally:
            self._stop_writer.set()
            await asyncio.wrap_future(writer)
        elapsed = time.perf_counter() - started
        event.remove(self.engine, "before_cursor_execute", self._record_statement)
        latencies = {kind: self._percentile(values) for kind, values in samples.items()}
        database_stat = await asyncio.to_thread(self.path.stat)
        return {
            "transport": "ASGI with real routes/SDK/SQL/serialization; network and auth IO excluded",
            **self._environment,
            "journal_mode": self.journal_mode,
            "cache_state": "cold_seconds is the first request after seeding; OS filesystem caches are not flushed",
            "database_bytes": database_stat.st_size,
            "initial_results": self.count,
            "analysts": analysts,
            "iterations_per_analyst": iterations,
            "coalescing": "distinct synthetic user scopes; no cross-analyst coalescing",
            "pagination": "page measures first pages; continuation follows each analyst's preceding next_cursor",
            "data_shape": {
                "operations": 100,
                "largest_operation_share": 0.6,
                "operators": 50,
                "harm_categories": 40,
                "categories_per_result": 3,
                "converter_types": 16,
                "request_converters_per_result": 2,
                "identifier_configurations": 64,
                "legacy_json_fallback_share": 0.09,
                "missing_identifier_share": 0.01,
                "unique_case_labels": self.count,
                "results_per_main_conversation": 2,
            },
            "cold_seconds": cold,
            "p95_seconds": latencies,
            "maximum_seconds": {kind: round(max(values), 4) for kind, values in samples.items()},
            "successful_requests": sum(len(values) for values in samples.values()) - len(self._failures),
            "failures": self._failures,
            "elapsed_seconds": round(elapsed, 4),
            "database_statements": self._statement_count,
            "write_transactions": len(self._write_samples),
            "write_p95_seconds": self._percentile(self._write_samples),
            "meets_latency_targets": not self._failures
            and latencies["report"] <= 2
            and latencies["facet"] <= 0.5
            and latencies["page"] <= 0.5
            and latencies["continuation"] <= 0.5,
        }

    def close(self) -> None:
        """Stop owned workers and release the scratch database."""
        self.service.shutdown()
        self._writer_pool.shutdown()
        self.memory.dispose_engine()

    def _app(self) -> FastAPI:
        """
        Mount real analytics routes with an isolated SDK dependency and synthetic identities.

        The transport bypasses external authentication only inside this disposable
        benchmark app. Distinct analyst identities prevent cross-user coalescing
        from making the measured workload appear cheaper.

        Returns:
            FastAPI: The in-process app used by each synthetic analyst.
        """
        app = FastAPI()
        register_error_handlers(app)
        app.include_router(router, prefix="/api")
        app.dependency_overrides[get_analytics_service] = lambda: self.service

        @app.middleware("http")
        async def benchmark_identity_async(request: Request, call_next: RequestResponseEndpoint) -> Response:
            """
            Assign the benchmark-only identity that partitions request reuse.

            Returns:
                Response: The real route response after synthetic identity assignment.
            """
            identity = request.headers.get("X-Benchmark-Analyst", "cold")
            request.state.user = AuthenticatedUser(
                oid=identity, name="Synthetic analyst", email="benchmark@example.invalid", groups=[]
            )
            return await call_next(request)

        return app

    async def _analyst_async(self, *, index: int, iterations: int) -> list[tuple[str, float]]:
        """
        Issue one analyst's sequential report, facet, first-page and continuation requests.

        Analysts run concurrently, but each request is timed from admission through
        response decoding. Failed requests stay in the latency sample and are also
        recorded explicitly; quick failures cannot produce a passing benchmark.

        Returns:
            list[tuple[str, float]]: Operation labels and measured end-to-end seconds.
        """
        samples = []
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app),
            base_url="http://benchmark",
            headers={"X-Benchmark-Analyst": str(index)},
        ) as client:
            for iteration in range(iterations):
                cursor = None
                for kind in ("report", "facet", "page", "continuation"):
                    if kind == "continuation" and cursor is None:
                        continue
                    started = time.perf_counter()
                    try:
                        latency, cursor = await self._request_async(
                            client=client, kind=kind, iteration=iteration, cursor=cursor
                        )
                    except httpx.HTTPStatusError as error:
                        latency = time.perf_counter() - started
                        self._failures.append(
                            {
                                "kind": kind,
                                "analyst": index,
                                "iteration": iteration,
                                "status": error.response.status_code,
                            }
                        )
                    samples.append((kind, latency))
        return samples

    async def _request_async(
        self, *, client: httpx.AsyncClient, kind: str, iteration: int, cursor: str | None = None
    ) -> tuple[float, str | None]:
        """
        Exercise a real endpoint and verify its basic population/output bounds.

        Reports alternate the two categorical matrix paths. Facets search a
        high-cardinality label within the largest operation. A continuation uses
        the preceding first page's opaque cursor, exercising actual keyset seeking.

        Returns:
            tuple[float, str | None]: End-to-end seconds and the next result cursor.

        Raises:
            httpx.HTTPStatusError: If the backend rejected or failed the request.
            AssertionError: If a successful response has inconsistent counts or bounds.
        """
        if kind == "report":
            body: dict[str, Any] = {
                "group_by": {"name": "targeted_harm_category"},
                "compare_by": {"name": "converter_type" if iteration % 2 else "attack_type"},
            }
            endpoint = "query"
        elif kind == "facet":
            body = {
                "dimension": {"name": "label", "label_key": "case"},
                "search": "case-05",
                "filters": {
                    "dimensions": [{"dimension": {"name": "operation"}, "values": [{"value": "operation-000"}]}]
                },
            }
            endpoint = "facets"
        else:
            body = {"limit": 25}
            if kind == "continuation":
                if cursor is None:
                    raise AssertionError("A continuation workload needs the preceding result cursor")
                body["cursor"] = cursor
            endpoint = "results"
        started = time.perf_counter()
        response = await client.post(f"/api/analytics/attacks/{endpoint}", json=body)
        response.raise_for_status()
        data = response.json()
        if kind == "report":
            summary = data["summary"]
            total = sum(summary[name] for name in ("successes", "failures", "undetermined", "errors"))
            if total != summary["total_results"] or total < self.count:
                raise AssertionError("Report totals do not count the saved result IDs")
            if len(data["cells"]) > 400 or len(data["results"]["items"]) != 25:
                raise AssertionError("Report output bounds do not match the request")
        elif kind == "facet" and not data["items"]:
            raise AssertionError("Expected matching facet values")
        elif kind in {"page", "continuation"} and len(data["items"]) != 25:
            raise AssertionError("Expected a complete result page")
        next_cursor: str | None = None
        if endpoint == "results":
            cursor_value = data.get("next_cursor")
            if not isinstance(cursor_value, str):
                raise AssertionError("The benchmark result page should provide a continuation cursor")
            next_cursor = cursor_value
        return time.perf_counter() - started, next_cursor

    def _write(self) -> None:
        """
        Add background write pressure without changing the reader's filters.

        Each transaction updates an existing outcome and every second transaction
        inserts another unique result ID. New IDs lie above the seeded range.
        Both activity and commit latency are measured until the readers finish.
        """
        index = 1
        while not self._stop_writer.is_set():
            started = time.perf_counter()
            now = datetime.now(tz=UTC)
            with self.engine.begin() as connection:
                connection.execute(
                    update(AttackResultEntry)
                    .where(AttackResultEntry.id == uuid.UUID(int=index))
                    .values(outcome="success" if index % 2 else "failure", timestamp=now)
                )
                if index % 2 == 0:
                    connection.execute(
                        insert(AttackResultEntry),
                        {
                            "id": uuid.UUID(int=self.count + index),
                            "conversation_id": str(uuid.UUID(int=self.count + index)),
                            "objective": "Synthetic concurrent insert",
                            "outcome": "undetermined",
                            "timestamp": now,
                            "executed_turns": 0,
                            "execution_time_ms": 0,
                        },
                    )
            self._write_samples.append(time.perf_counter() - started)
            index += 1
            self._stop_writer.wait(0.1)

    def _record_statement(
        self, connection: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
    ) -> None:
        """Count driver statements across concurrent readers and the writer, including setup reads."""
        with self._statement_lock:
            self._statement_count += 1

    @staticmethod
    def _identifiers() -> list[AtomicAttackIdentifier]:
        """
        Build reusable configurations with overlapping converter memberships.

        Returns:
            list[AtomicAttackIdentifier]: Sixty-four configurations spanning 16 attack
            types, 16 converter types and eight target models.
        """
        return [
            AtomicAttackIdentifier.build(
                attack_identifier=AttackIdentifier(
                    class_name=f"BenchmarkAttack{index % 16:02}",
                    class_module="benchmark",
                    params={"configuration": index},
                    objective_target=TargetIdentifier(
                        class_name="MockTarget", class_module="benchmark", model_name=f"model-{index % 8}"
                    ),
                    request_converters=[
                        ConverterIdentifier(class_name=f"BenchmarkConverter{value % 16:02}", class_module="benchmark")
                        for value in (index, index + 1)
                    ],
                )
            )
            for index in range(64)
        ]

    @staticmethod
    def _percentile(values: Sequence[float]) -> float:
        """
        Select the nearest-rank p95 without rounding away threshold failures.

        Returns:
            float: The observed sample at the 95th-percentile rank.

        Raises:
            AssertionError: If a required workload produced no measurements.
        """
        if not values:
            raise AssertionError("A benchmark workload produced no measurements")
        return sorted(values)[math.ceil(len(values) * 0.95) - 1]


def main() -> None:
    """Run a reproducible benchmark without touching an existing database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=int, default=100_000)
    parser.add_argument("--analysts", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--journal-mode", choices=["delete", "wal"], default="wal")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.results < 60_000 or args.analysts < 1 or args.iterations < 1:
        parser.error("Use at least 60000 results and positive analyst/iteration counts")
    cache_directory = Path.cwd() / ".pytest_cache"
    cache_directory.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="analytics-benchmark-", dir=cache_directory) as temporary:
        benchmark = AnalyticsBenchmark(directory=Path(temporary), count=args.results, journal_mode=args.journal_mode)
        try:
            benchmark.seed()
            result = asyncio.run(benchmark.run_async(analysts=args.analysts, iterations=args.iterations))
        finally:
            benchmark.close()
        serialized = json.dumps(result, indent=2)
        if args.output is not None:
            args.output.write_text(serialized, encoding="utf-8")
        print(serialized)
        if not result["meets_latency_targets"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
