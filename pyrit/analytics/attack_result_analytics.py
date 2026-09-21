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
    AttackAnalyticsFilters,
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
        """
        Bind analytics to saved results without starting threads or querying the database.

        Args:
            memory (MemoryInterface | None): The initialized backend to read. None
                resolves the currently configured CentralMemory instance.
        """
        self._memory = memory if memory is not None else CentralMemory.get_memory_instance()
        self._reader = AttackAnalyticsReader(memory=self._memory)

    def query(self, *, query: AttackAnalyticsQuery | None = None, access_scope: str = "") -> AttackAnalyticsReport:
        """
        Compute a coherent outcome report and initial result page.

        Args:
            query (AttackAnalyticsQuery | None): Filters and requested grouping.
                None selects all saved results grouped by operation.
            access_scope (str): Partition for in-flight reuse, not row-level authorization.
                Callers enforcing distinct access policies must use distinct scopes.

        Returns:
            AttackAnalyticsReport: Independently owned statistics and drill-down
            predicates. The report does not retain a snapshot for subsequent requests.

        Raises:
            AnalyticsBusyException: If the bounded execution queue cannot admit the query.
            AnalyticsTimeoutException: If execution exceeds its deadline.
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

        Args:
            query (AttackAnalyticsQuery | None): The same request as ``query``.
            access_scope (str): The caller's partition for in-flight reuse.

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

        Args:
            query (AttackAnalyticsResultsQuery | None): Cohort and opaque page cursor.
                None requests the first page of all saved results.
            access_scope (str): The caller's partition for in-flight reuse.

        Returns:
            AttackAnalyticsResults: A fresh page with its own read timestamp. Its
            recency cursor avoids offset drift but cannot freeze mutable results.

        Raises:
            ValueError: If the cursor is malformed or belongs to different filters.
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

        Args:
            query (AttackAnalyticsResultsQuery | None): The same request as ``results``.
            access_scope (str): The caller's partition for in-flight reuse.

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

        Args:
            query (AttackAnalyticsFacetQuery): Dimension, search, pagination and cohort.
                The selected dimension's own predicates are omitted to expose alternatives.
            access_scope (str): The caller's partition for in-flight reuse.

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

        Args:
            query (AttackAnalyticsFacetQuery): The same request as ``facets``.
            access_scope (str): The caller's partition for in-flight reuse.

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
        """
        Stop workers shared by every analytics instance using this memory backend.

        This waits for running tasks to finish cooperative cancellation, so call it
        outside an event loop. The closing controller remains registered until its
        workers exit; new requests cannot open a second pool during that interval.
        A later query can create a fresh execution controller.
        """
        with self._EXECUTION_LOCK:
            execution = self._EXECUTIONS.get(self._memory)
        if execution is not None:
            execution.shutdown()
            with self._EXECUTION_LOCK:
                if self._EXECUTIONS.get(self._memory) is execution:
                    del self._EXECUTIONS[self._memory]

    def _execution(self) -> AnalyticsExecution:
        """
        Lazily obtain the backend's shared admission and execution limits.

        The weak-key map avoids retaining discarded memory objects, while the lock
        prevents concurrent first requests from creating separate worker pools.

        Returns:
            AnalyticsExecution: The bounded controller shared by this backend.
        """
        with self._EXECUTION_LOCK:
            execution = self._EXECUTIONS.get(self._memory)
            if execution is None:
                from pyrit.analytics._execution import AnalyticsExecution

                execution = AnalyticsExecution()
                self._EXECUTIONS[self._memory] = execution
            return execution

    def _report(self, *, query: AttackAnalyticsQuery, control: QueryControl) -> AttackAnalyticsReport:
        """
        Interpret raw memory projections into the shared analytics contract.

        Memory chooses a consistent read view and returns counts, not success-rate
        policy. This layer finishes any bounded profile aggregation, defines rates,
        adds overlap/ASR annotations, and builds exact additional drill-down predicates.

        Returns:
            AttackAnalyticsReport: The requested chart, overall statistics and first page.
        """
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
            drilldown_unavailable_reason=self._drilldown_unavailable_reason(query),
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
        """
        Add display labels and a next-page offset to memory's single-facet projection.

        Returns:
            AttackAnalyticsFacets: Bounded options with distinct typed absence keys.
        """
        raw = self._reader.facets(query=query, control=control)
        return AttackAnalyticsFacets(
            items=[self._option(raw=option, dimension=query.dimension) for option in raw.items],
            has_more=raw.has_more,
            next_offset=query.offset + len(raw.items) if raw.has_more else None,
            computed_at=raw.computed_at,
        )

    @classmethod
    def _cells(cls, *, query: AttackAnalyticsQuery, raw: RawAnalyticsReport) -> list[AttackAnalyticsCell]:
        """
        Fill the displayed matrix, including absent combinations as empty cells.

        An empty cell has zero results and unavailable ASR, not a zero-success
        verdict. Its drill-down appends both axis predicates to the current cohort.

        Returns:
            list[AttackAnalyticsCell]: Cells for the bounded row/column cross product.
        """
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
        """
        Apply outcome policy to raw counts without averaging subgroup percentages.

        ASR excludes errors/undetermined from its denominator; the decided share
        and outcome composition use the entire cohort. Missing outcome keys mean
        zero observations, while unknown outcomes or invalid counts are data errors.

        Returns:
            AttackAnalyticsStatistics: Counts and their correctly scoped proportions.

        Raises:
            AnalyticsDataException: If stored outcomes or aggregate counts are invalid.
        """
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
        """
        Label a group while keeping its exact identity separate from visible text.

        Missing and empty-pipeline buckets get explanatory labels. Target/scenario
        identities retain a short distinguishing suffix when display names collide.

        Returns:
            AttackAnalyticsOption: Display text paired with the unchanged filter key.
        """
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
    def _drilldown_unavailable_reason(query: AttackAnalyticsQuery) -> str | None:
        """
        Mark a terminal drill-down before its additional predicates exceed query limits.

        A final click may reach the limit, so rejecting every full-budget report
        would merely move the failure to that next report. Keep that report valid
        and expose the limit instead of dropping predicates or broadening its cohort.

        Returns:
            str | None: A user-facing reason when another group/cell click cannot fit.
        """
        additional = 2 if query.compare_by is not None else 1
        if len(query.filters.dimensions) + additional > AttackAnalyticsFilters.MAX_PREDICATES:
            return (
                "Remove a dimension filter before drilling down further "
                f"(maximum {AttackAnalyticsFilters.MAX_PREDICATES} predicates)."
            )
        if (
            sum(len(predicate.values) for predicate in query.filters.dimensions) + additional
            > AttackAnalyticsFilters.MAX_VALUES
        ):
            return (
                "Select fewer dimension values before drilling down further "
                f"(maximum {AttackAnalyticsFilters.MAX_VALUES} values)."
            )
        return None

    @staticmethod
    def _drilldown(*, dimension: AttackAnalyticsDimension, key: AttackAnalyticsValue) -> AttackAnalyticsFilter:
        """
        Select one membership as an additional predicate, not a replacement filter.

        Returns:
            AttackAnalyticsFilter: A typed equality/membership constraint for this group.
        """
        return AttackAnalyticsFilter(dimension=dimension, values=[key])

    @staticmethod
    def _value_key(value: AttackAnalyticsValue) -> tuple[AttackAnalyticsValueKind, str | None]:
        """
        Key cells by both bucket kind and value so absence cannot collide with real text.

        Returns:
            tuple[AttackAnalyticsValueKind, str | None]: The immutable lookup identity.
        """
        return value.kind, value.value

    @staticmethod
    def _copy(query: QueryT) -> QueryT:
        """
        Snapshot and revalidate a caller-owned query before sharing or queuing it.

        Returns:
            QueryT: An independent, normalized request unaffected by later caller mutation.
        """
        return type(query).model_validate(query.model_dump())

    @staticmethod
    def _key(*, operation: str, query: BaseModel, access_scope: str) -> str:
        """
        Identify interchangeable work within an already backend-specific controller.

        Operation and caller scope prevent unlike result types or different users
        from sharing a flight. This fingerprint is neither authorization nor a cache key
        for completed data; it exists only for in-flight coalescing.

        Returns:
            str: A full SHA256 fingerprint of the normalized work identity.
        """
        return fingerprint_filters(
            filters={"operation": operation, "query": query.model_dump(mode="json"), "access_scope": access_scope},
            length=64,
        )
