# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""SDK-owned outcome analytics over stored AttackResult IDs."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, ClassVar, TypeVar
from weakref import WeakKeyDictionary

from pydantic import BaseModel

from pyrit.analytics._profile_aggregation import ProfileAggregation
from pyrit.analytics.result_analysis import _compute_stats
from pyrit.common.pagination import fingerprint_filters
from pyrit.exceptions.analytics_exception import AnalyticsDataException
from pyrit.memory import CentralMemory
from pyrit.memory.attack_analytics import AttackAnalyticsReader, RawAnalyticsOption, RawAnalyticsReport
from pyrit.models import (
    AttackAnalyticsCell,
    AttackAnalyticsDimension,
    AttackAnalyticsDimensionName,
    AttackAnalyticsFacetQuery,
    AttackAnalyticsFacets,
    AttackAnalyticsFilter,
    AttackAnalyticsGroup,
    AttackAnalyticsOption,
    AttackAnalyticsQuery,
    AttackAnalyticsReport,
    AttackAnalyticsResults,
    AttackAnalyticsResultsQuery,
    AttackAnalyticsStatistics,
    AttackAnalyticsValue,
    AttackAnalyticsValueKind,
    AttackOutcome,
)

if TYPE_CHECKING:
    from pyrit.analytics._execution import AnalyticsExecution
    from pyrit.memory.memory_interface import MemoryInterface
    from pyrit.memory.query_control import QueryControl

QueryT = TypeVar("QueryT", bound=BaseModel)


class AttackResultAnalytics:
    """
    Query saved outcomes without inspecting conversations or individual scores.

    Instances sharing a memory backend share bounded execution workers. The optional
    access scope partitions in-flight request reuse; it is not an authorization policy.
    """

    _EXECUTIONS: ClassVar[WeakKeyDictionary[MemoryInterface, AnalyticsExecution]] = WeakKeyDictionary()
    _EXECUTION_LOCK: ClassVar[threading.Lock] = threading.Lock()
    _MULTIVALUED: ClassVar[frozenset[AttackAnalyticsDimensionName]] = frozenset(
        {AttackAnalyticsDimensionName.TARGETED_HARM_CATEGORY, AttackAnalyticsDimensionName.CONVERTER_TYPE}
    )

    def __init__(self, *, memory: MemoryInterface | None = None) -> None:
        """Initialize with a memory backend, defaulting to CentralMemory."""
        self._memory = memory if memory is not None else CentralMemory.get_memory_instance()
        self._reader = AttackAnalyticsReader(memory=self._memory)

    def query(self, *, query: AttackAnalyticsQuery | None = None, access_scope: str = "") -> AttackAnalyticsReport:
        """
        Compute a coherent outcome report and initial result page.

        Returns:
            AttackAnalyticsReport: SDK-calculated statistics and exact drill-down predicates.
        """
        request = self._copy(query if query is not None else AttackAnalyticsQuery())
        result = self._execution().run(
            report=True,
            key=self._key(operation="report", query=request, access_scope=access_scope),
            task=lambda control: self._report(query=request, control=control),
        )
        return result.model_copy(deep=True)

    async def query_async(
        self, *, query: AttackAnalyticsQuery | None = None, access_scope: str = ""
    ) -> AttackAnalyticsReport:
        """
        Compute a report without blocking the caller's event loop.

        Returns:
            AttackAnalyticsReport: The same contract as the synchronous query.
        """
        request = self._copy(query if query is not None else AttackAnalyticsQuery())
        result = await self._execution().run_async(
            report=True,
            key=self._key(operation="report", query=request, access_scope=access_scope),
            task=lambda control: self._report(query=request, control=control),
        )
        return result.model_copy(deep=True)

    def results(
        self, *, query: AttackAnalyticsResultsQuery | None = None, access_scope: str = ""
    ) -> AttackAnalyticsResults:
        """
        Fetch a result page without recalculating report aggregates.

        Returns:
            AttackAnalyticsResults: A fresh lightweight page.
        """
        request = self._copy(query if query is not None else AttackAnalyticsResultsQuery())
        result = self._execution().run(
            report=False,
            key=self._key(operation="results", query=request, access_scope=access_scope),
            task=lambda control: self._reader.results(query=request, control=control),
        )
        return result.model_copy(deep=True)

    async def results_async(
        self, *, query: AttackAnalyticsResultsQuery | None = None, access_scope: str = ""
    ) -> AttackAnalyticsResults:
        """
        Fetch a result page asynchronously without running a report.

        Returns:
            AttackAnalyticsResults: A fresh lightweight page.
        """
        request = self._copy(query if query is not None else AttackAnalyticsResultsQuery())
        result = await self._execution().run_async(
            report=False,
            key=self._key(operation="results", query=request, access_scope=access_scope),
            task=lambda control: self._reader.results(query=request, control=control),
        )
        return result.model_copy(deep=True)

    def facets(self, *, query: AttackAnalyticsFacetQuery, access_scope: str = "") -> AttackAnalyticsFacets:
        """
        Look up one bounded page of options for an opened facet.

        Returns:
            AttackAnalyticsFacets: Stored values narrowed by the other active filters.
        """
        request = self._copy(query)
        result = self._execution().run(
            report=False,
            key=self._key(operation="facets", query=request, access_scope=access_scope),
            task=lambda control: self._facets(query=request, control=control),
        )
        return result.model_copy(deep=True)

    async def facets_async(self, *, query: AttackAnalyticsFacetQuery, access_scope: str = "") -> AttackAnalyticsFacets:
        """
        Look up a facet without blocking the event loop.

        Returns:
            AttackAnalyticsFacets: The same options as the synchronous lookup.
        """
        request = self._copy(query)
        result = await self._execution().run_async(
            report=False,
            key=self._key(operation="facets", query=request, access_scope=access_scope),
            task=lambda control: self._facets(query=request, control=control),
        )
        return result.model_copy(deep=True)

    def shutdown(self) -> None:
        """Stop analytics workers shared by this memory backend."""
        with self._EXECUTION_LOCK:
            execution = self._EXECUTIONS.pop(self._memory, None)
        if execution is not None:
            execution.shutdown()

    def _execution(self) -> AnalyticsExecution:
        with self._EXECUTION_LOCK:
            execution = self._EXECUTIONS.get(self._memory)
            if execution is None:
                from pyrit.analytics._execution import AnalyticsExecution

                execution = AnalyticsExecution()
                self._EXECUTIONS[self._memory] = execution
            return execution

    def _report(self, *, query: AttackAnalyticsQuery, control: QueryControl) -> AttackAnalyticsReport:
        raw = self._reader.report(query=query, control=control, use_compact_profiles=True)
        ProfileAggregation.populate(report=raw, query=query, control=control)
        report = AttackAnalyticsReport(
            filters=query.filters,
            group_by=query.group_by,
            compare_by=query.compare_by,
            summary=self._statistics(raw.counts),
            outcome_filter_applied=bool(query.filters.outcomes),
            groups_overlap=query.group_by.name in self._MULTIVALUED
            or (query.compare_by is not None and query.compare_by.name in self._MULTIVALUED),
            groups=[
                AttackAnalyticsGroup(
                    **self._option(raw=group.option, dimension=query.group_by).model_dump(),
                    statistics=self._statistics(group.counts),
                    drilldown_filters=[self._drilldown(dimension=query.group_by, key=group.option.key)],
                )
                for group in raw.groups
            ],
            has_more_groups=raw.has_more_groups,
            next_group_offset=query.group_offset + len(raw.groups) if raw.has_more_groups else None,
            rows=[self._option(raw=option, dimension=query.group_by) for option in raw.rows],
            columns=[self._option(raw=option, dimension=query.compare_by) for option in raw.columns]
            if query.compare_by is not None
            else [],
            cells=self._cells(query=query, raw=raw),
            axes_truncated=raw.axes_truncated,
            results=raw.results,
            computed_at=raw.results.computed_at,
            warnings=raw.warnings,
        )
        control.check()
        return report

    def _facets(self, *, query: AttackAnalyticsFacetQuery, control: QueryControl) -> AttackAnalyticsFacets:
        raw = self._reader.facets(query=query, control=control)
        return AttackAnalyticsFacets(
            items=[self._option(raw=option, dimension=query.dimension) for option in raw.items],
            has_more=raw.has_more,
            next_offset=query.offset + len(raw.items) if raw.has_more else None,
            computed_at=raw.computed_at,
        )

    @classmethod
    def _cells(cls, *, query: AttackAnalyticsQuery, raw: RawAnalyticsReport) -> list[AttackAnalyticsCell]:
        if query.compare_by is None:
            return []
        counts = {
            (cls._value_key(cell.option.key), cls._value_key(cell.column.key)): cell.counts
            for cell in raw.cells
            if cell.column is not None
        }
        return [
            AttackAnalyticsCell(
                row=row.key,
                column=column.key,
                statistics=cls._statistics(counts.get((cls._value_key(row.key), cls._value_key(column.key)), {})),
                drilldown_filters=[
                    cls._drilldown(dimension=query.group_by, key=row.key),
                    cls._drilldown(dimension=query.compare_by, key=column.key),
                ],
            )
            for row in raw.rows
            for column in raw.columns
        ]

    @staticmethod
    def _statistics(counts: dict[str, int]) -> AttackAnalyticsStatistics:
        if set(counts) - {outcome.value for outcome in AttackOutcome}:
            raise AnalyticsDataException("Stored results contain an unsupported attack outcome.")
        if any(type(count) is not int or count < 0 for count in counts.values()):
            raise AnalyticsDataException("Stored results contain invalid outcome counts.")
        stats = _compute_stats(
            successes=counts.get(AttackOutcome.SUCCESS.value, 0),
            failures=counts.get(AttackOutcome.FAILURE.value, 0),
            undetermined=counts.get(AttackOutcome.UNDETERMINED.value, 0),
            errors=counts.get(AttackOutcome.ERROR.value, 0),
        )
        total = stats.total_decided + stats.undetermined + stats.errors
        return AttackAnalyticsStatistics(
            success_rate=stats.success_rate,
            total_decided=stats.total_decided,
            successes=stats.successes,
            failures=stats.failures,
            undetermined=stats.undetermined,
            errors=stats.errors,
            total_results=total,
            decided_share=stats.total_decided / total if total else None,
            outcome_shares={
                outcome: counts.get(outcome.value, 0) / total if total else 0.0 for outcome in AttackOutcome
            },
        )

    @staticmethod
    def _option(*, raw: RawAnalyticsOption, dimension: AttackAnalyticsDimension) -> AttackAnalyticsOption:
        if raw.key.kind is AttackAnalyticsValueKind.MISSING:
            label = "Not recorded"
        elif raw.key.kind is AttackAnalyticsValueKind.NO_CONVERTERS:
            label = "No converters"
        else:
            label = raw.label if raw.label is not None else raw.key.value or ""
            if not label.strip():
                label = "(Blank)"
            elif (
                dimension.name
                in {
                    AttackAnalyticsDimensionName.OBJECTIVE_TARGET,
                    AttackAnalyticsDimensionName.SCENARIO,
                }
                and raw.key.value is not None
                and label != raw.key.value
            ):
                label = f"{label} ({raw.key.value[-8:]})"
        return AttackAnalyticsOption(key=raw.key, label=label)

    @staticmethod
    def _drilldown(*, dimension: AttackAnalyticsDimension, key: AttackAnalyticsValue) -> AttackAnalyticsFilter:
        return AttackAnalyticsFilter(dimension=dimension, values=[key])

    @staticmethod
    def _value_key(value: AttackAnalyticsValue) -> tuple[AttackAnalyticsValueKind, str | None]:
        return value.kind, value.value

    @staticmethod
    def _copy(query: QueryT) -> QueryT:
        return type(query).model_validate(query.model_dump())

    @staticmethod
    def _key(*, operation: str, query: BaseModel, access_scope: str) -> str:
        return fingerprint_filters(
            filters={"operation": operation, "query": query.model_dump(mode="json"), "access_scope": access_scope},
            length=64,
        )
