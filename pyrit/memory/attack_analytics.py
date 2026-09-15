# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded raw outcome aggregates and lightweight result projections."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from pyrit.common.pagination import decode_keyset_cursor, encode_keyset_cursor, fingerprint_filters
from pyrit.exceptions.analytics_exception import AnalyticsDataException, AnalyticsTimeoutException
from pyrit.memory.attack_analytics_query import AttackAnalyticsQueryCompiler
from pyrit.models import (
    AttackAnalyticsFacetQuery,
    AttackAnalyticsFilters,
    AttackAnalyticsQuery,
    AttackAnalyticsResultRow,
    AttackAnalyticsResults,
    AttackAnalyticsResultsQuery,
    AttackAnalyticsValue,
    AttackAnalyticsValueKind,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sqlalchemy.engine import RowMapping
    from sqlalchemy.orm import Session

    from pyrit.memory.memory_interface import MemoryInterface
    from pyrit.memory.query_control import QueryControl


@dataclass
class RawAnalyticsOption:
    """A stored metadata key and its optional display name."""

    key: AttackAnalyticsValue
    label: str | None


@dataclass
class RawAnalyticsGroup:
    """A grouped raw outcome count without metric calculations."""

    option: RawAnalyticsOption
    counts: dict[str, int]
    column: RawAnalyticsOption | None = None


@dataclass
class RawAnalyticsReport:
    """The projections read in one database snapshot."""

    counts: dict[str, int]
    groups: list[RawAnalyticsGroup]
    rows: list[RawAnalyticsOption]
    columns: list[RawAnalyticsOption]
    cells: list[RawAnalyticsGroup]
    has_more_groups: bool
    axes_truncated: bool
    results: AttackAnalyticsResults
    warnings: list[str]
    profiles: list[dict[str, Any]] | None = None


@dataclass
class RawAnalyticsFacets:
    """A bounded list of stored metadata values."""

    items: list[RawAnalyticsOption]
    has_more: bool
    computed_at: datetime


class AttackAnalyticsReader:
    """Read metadata through memory without hydrating AttackResult object graphs."""

    # Frequent Python callbacks serialize concurrent SQLite readers on the GIL.
    SQLITE_PROGRESS_STEPS: ClassVar[int] = 100_000
    MAX_COMPACT_PROFILES: ClassVar[int] = 4096
    MAX_COMPACT_VALUE_LENGTH: ClassVar[int] = 4096
    MAX_COMPACT_TOTAL_LENGTH: ClassVar[int] = 1_000_000

    def __init__(self, *, memory: MemoryInterface) -> None:
        """Initialize the memory-backed reader."""
        self._memory = memory

    def report(
        self, *, query: AttackAnalyticsQuery, control: QueryControl, use_compact_profiles: bool = False
    ) -> RawAnalyticsReport:
        """
        Read the report and its first result page in a short consistent view.

        Returns:
            RawAnalyticsReport: Raw counts and bounded projections.
        """
        with self._session(control=control, consistent=True) as (session, dialect, warnings):
            compiler = AttackAnalyticsQueryCompiler(dialect=dialect, filters=query.filters)
            counts = dict(session.execute(compiler.totals()).tuples().all())
            groups: list[RawAnalyticsGroup] = []
            rows: list[RawAnalyticsOption] = []
            columns: list[RawAnalyticsOption] = []
            cells: list[RawAnalyticsGroup] = []
            has_more = False
            truncated = False
            statement = (
                compiler.compact_profiles(
                    query=query,
                    limit=self.MAX_COMPACT_PROFILES + 1,
                    max_value_length=self.MAX_COMPACT_VALUE_LENGTH,
                )
                if use_compact_profiles
                else None
            )
            profiles = None
            if statement is not None:
                records = session.execute(statement).mappings().all()
                if (
                    len(records) <= self.MAX_COMPACT_PROFILES
                    and not any(record["oversized"] for record in records)
                    and sum(len(value) for record in records for value in record.values() if isinstance(value, str))
                    <= self.MAX_COMPACT_TOTAL_LENGTH
                ):
                    profiles = [dict(record) for record in records]
            if profiles is not None:
                pass
            elif query.compare_by is None:
                records = session.execute(compiler.groups(query)).mappings().all()
                has_more = len(records) > query.group_limit
                groups = [self._group(record) for record in records[: query.group_limit]]
            else:
                for record in session.execute(compiler.matrix(query)).mappings():
                    truncated = bool(record["truncated"])
                    if record["record"] == "row":
                        rows.append(self._option(record, index=0))
                    elif record["record"] == "column":
                        columns.append(self._option(record, index=1))
                    else:
                        cells.append(self._group(record, matrix=True))
            results = self._results(
                session=session,
                dialect=dialect,
                query=AttackAnalyticsResultsQuery(filters=query.filters, limit=query.result_limit),
            )
            control.check()
            return RawAnalyticsReport(
                counts=counts,
                groups=groups,
                rows=rows,
                columns=columns,
                cells=cells,
                has_more_groups=has_more,
                axes_truncated=truncated,
                results=results,
                warnings=warnings,
                profiles=profiles,
            )

    def results(self, *, query: AttackAnalyticsResultsQuery, control: QueryControl) -> AttackAnalyticsResults:
        """
        Read only a result page, without any report/count query.

        Returns:
            AttackAnalyticsResults: A page with its own freshness timestamp.
        """
        with self._session(control=control) as (session, dialect, _):
            result = self._results(session=session, dialect=dialect, query=query)
            control.check()
            return result

    def facets(self, *, query: AttackAnalyticsFacetQuery, control: QueryControl) -> RawAnalyticsFacets:
        """
        Read one requested facet, excluding its own predicates to allow alternatives.

        Returns:
            RawAnalyticsFacets: The bounded facet page.
        """
        filters = query.filters.model_copy(
            update={
                "dimensions": [
                    predicate for predicate in query.filters.dimensions if predicate.dimension != query.dimension
                ]
            }
        )
        with self._session(control=control) as (session, dialect, _):
            compiler = AttackAnalyticsQueryCompiler(dialect=dialect, filters=filters)
            records = session.execute(compiler.facet(query)).mappings().all()
            control.check()
            return RawAnalyticsFacets(
                items=[self._option(record, index=0) for record in records[: query.limit]],
                has_more=len(records) > query.limit,
                computed_at=datetime.now(tz=UTC),
            )

    @contextmanager
    def _session(self, *, control: QueryControl, consistent: bool = False) -> Iterator[tuple[Session, str, list[str]]]:
        control.check()
        try:
            session = self._memory.get_session(timeout=control.remaining)
        except TimeoutError as error:
            raise AnalyticsTimeoutException from error
        with closing(session):
            dialect = session.get_bind().dialect.name
            options = {"isolation_level": "SNAPSHOT"} if consistent and dialect == "mssql" else {}
            connection = session.connection(execution_options=options)
            driver = connection.connection.driver_connection
            if driver is None:
                raise AnalyticsDataException("The analytics database connection is closed.")
            warnings: list[str] = []
            old_timeout: int | None = None

            def before_execute(
                conn: Any, cursor: Any, statement: str, parameters: Any, context: Any, executemany: bool
            ) -> None:
                control.check()
                if dialect == "mssql":
                    driver.timeout = max(1, math.ceil(control.remaining))

            event.listen(connection, "before_cursor_execute", before_execute)
            try:
                control.check()
                if dialect == "sqlite":
                    if not isinstance(driver, sqlite3.Connection):
                        raise NotImplementedError("SQLite analytics requires a sqlite3 connection")
                    driver.set_progress_handler(lambda: int(control.expired), self.SQLITE_PROGRESS_STEPS)
                    if consistent:
                        mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()
                        if mode not in {"wal", "memory"}:
                            warnings.append(
                                "SQLite rollback journaling can slow analytics during concurrent writes. "
                                "The documented performance workload uses explicitly configured WAL mode. "
                                "Analytics has not changed database settings."
                            )
                        if not driver.in_transaction:
                            connection.exec_driver_sql("BEGIN")
                elif dialect == "mssql":
                    old_timeout = getattr(driver, "timeout", None)
                    if not isinstance(old_timeout, int):
                        raise NotImplementedError("Azure SQL analytics requires an ODBC query-timeout property")
                    driver.timeout = max(1, math.ceil(control.remaining))
                yield session, dialect, warnings
            except OperationalError as error:
                if control.expired:
                    raise AnalyticsTimeoutException from error
                raise
            finally:
                event.remove(connection, "before_cursor_execute", before_execute)
                if isinstance(driver, sqlite3.Connection):
                    driver.set_progress_handler(None, 0)
                if old_timeout is not None and not connection.invalidated and not connection.closed:
                    driver.timeout = old_timeout

    def _results(self, *, session: Session, dialect: str, query: AttackAnalyticsResultsQuery) -> AttackAnalyticsResults:
        fingerprint = self._fingerprint(query.filters)
        after = decode_keyset_cursor(cursor=query.cursor, fingerprint=fingerprint)
        if query.cursor is not None and after is None:
            raise ValueError("Invalid or stale analytics cursor. Reload results with the current filters.")
        compiler = AttackAnalyticsQueryCompiler(dialect=dialect, filters=query.filters)
        records = session.execute(compiler.results(limit=query.limit, after=after)).mappings().all()
        items = [self._result_row(record) for record in records[: query.limit]]
        has_more = len(records) > query.limit
        cursor = (
            encode_keyset_cursor(
                timestamp=items[-1].updated_at,
                identifier=items[-1].attack_result_id,
                fingerprint=fingerprint,
            )
            if has_more and items
            else None
        )
        return AttackAnalyticsResults(
            items=items,
            has_more=has_more,
            next_cursor=cursor,
            computed_at=datetime.now(tz=UTC),
        )

    @staticmethod
    def _fingerprint(filters: AttackAnalyticsFilters) -> str:
        return fingerprint_filters(
            filters={"filters": filters.model_dump(mode="json"), "result_selection": "all_results"}
        )

    @staticmethod
    def _option(record: RowMapping, *, index: int) -> RawAnalyticsOption:
        try:
            kind = AttackAnalyticsValueKind(record[f"kind{index}"])
            key = AttackAnalyticsValue(
                kind=kind,
                value=record[f"value{index}"] if kind is AttackAnalyticsValueKind.VALUE else None,
            )
        except (ValueError, ValidationError) as error:
            raise AnalyticsDataException("Stored attack metadata contains an invalid dimension value.") from error
        return RawAnalyticsOption(key=key, label=record[f"label{index}"])

    @classmethod
    def _group(cls, record: RowMapping, *, matrix: bool = False) -> RawAnalyticsGroup:
        try:
            counts = json.loads(record["counts"])
        except (json.JSONDecodeError, TypeError) as error:
            raise AnalyticsDataException("Stored attack outcomes could not be aggregated.") from error
        if not isinstance(counts, dict) or any(
            not isinstance(key, str) or type(count) is not int or count < 0 for key, count in counts.items()
        ):
            raise AnalyticsDataException("Stored attack outcomes contain invalid counts.")
        return RawAnalyticsGroup(
            option=cls._option(record, index=0),
            counts=counts,
            column=cls._option(record, index=1) if matrix else None,
        )

    @classmethod
    def _result_row(cls, record: RowMapping) -> AttackAnalyticsResultRow:
        data = dict(record)
        data["attack_result_id"] = str(data["attack_result_id"])
        if data["scenario_result_id"] is not None:
            data["scenario_result_id"] = str(data["scenario_result_id"])
        if data["targeted_harm_categories"] is None:
            data["targeted_harm_categories"] = []
        if data["labels"] is None:
            data["labels"] = {}
        data["request_converters"] = cls._converter_names(data["request_converters"])
        data["response_converters"] = cls._converter_names(data["response_converters"])
        try:
            return AttackAnalyticsResultRow.model_validate(data)
        except ValidationError as error:
            raise AnalyticsDataException("A stored result contains invalid analytics metadata.") from error

    @staticmethod
    def _converter_names(raw: str | None) -> list[str]:
        if raw is None:
            return []
        try:
            values = json.loads(raw)
        except json.JSONDecodeError as error:
            raise AnalyticsDataException("Stored converter metadata is not valid JSON.") from error
        if values is None:
            return []
        if not isinstance(values, list):
            raise AnalyticsDataException("Stored converter metadata is not a list.")
        names: set[str] = set()
        for value in values:
            name = value.get("class_name") if isinstance(value, dict) else value
            if name is None:
                continue
            if not isinstance(name, str):
                raise AnalyticsDataException("A recorded converter type is not a string.")
            names.add(name)
        return sorted(names)
