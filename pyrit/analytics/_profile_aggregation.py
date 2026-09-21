# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Finish SQLite's bounded categorical aggregation without hydrating saved results.

Memory has already applied every cohort predicate and counted result IDs into
profiles. This module only expands their metadata memberships and sums weights;
it does not fetch records, refilter the cohort, or recompute saved outcomes.
The same typed keys, display-label choice, ordering, and truncation rules must
match the general SQL grouping path.
"""

from __future__ import annotations

import json
import string
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pyrit.exceptions.analytics_exception import AnalyticsDataException
from pyrit.memory.attack_analytics import RawAnalyticsGroup, RawAnalyticsOption
from pyrit.models import AttackAnalyticsDimensionName, AttackAnalyticsValue, AttackAnalyticsValueKind

if TYPE_CHECKING:
    from pyrit.memory.attack_analytics import RawAnalyticsReport
    from pyrit.memory.query_control import QueryControl
    from pyrit.models import AttackAnalyticsDimension, AttackAnalyticsQuery

ValueKey = tuple[str, str]


@dataclass
class _Profile:
    """
    One pre-counted outcome bucket with unique memberships for each requested axis.

    ``weight`` is the number of saved result IDs represented, not a membership
    count. Each values dictionary maps a (kind, text) key to its display option,
    removing duplicate/case-equivalent array members before any weights are added.
    """

    values: list[dict[ValueKey, RawAnalyticsOption]]
    outcome: str
    weight: int


class ProfileAggregation:
    """
    Aggregate the reader-approved profile set with SQLite's exact key semantics.

    Only attack types, harm categories, and converter types enter this path.
    Row, per-value, and total-text limits are enforced by memory before decoding.
    These methods do not provide a general unbounded in-memory analytics fallback.
    """

    _ASCII_LOWER: ClassVar[dict[int, int]] = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

    @classmethod
    def populate(cls, *, report: RawAnalyticsReport, query: AttackAnalyticsQuery, control: QueryControl) -> None:
        """
        Populate raw groups or heatmap fields in place, leaving cohort totals untouched.

        The per-dimension option cache avoids repeatedly decoding identical
        metadata across outcomes/profiles. It lives only for this call: it is not
        a saved-result cache, and no stale query results survive between requests.
        ``profiles=None`` means SQL already populated the chart; [] means a valid
        empty profile result.

        Args:
            report (RawAnalyticsReport): Reader output whose profiles already passed all caps.
            query (AttackAnalyticsQuery): The same axes and limits used to select profiles.
            control (QueryControl): Shared operation budget, checked during decoding and
                after aggregation before the report can be returned.
        """
        if report.profiles is None:
            return
        dimensions = [query.group_by] + ([query.compare_by] if query.compare_by is not None else [])
        profiles = []
        axes: list[dict[ValueKey, RawAnalyticsOption]] = [{} for _ in dimensions]
        option_cache: list[dict[str | None, dict[ValueKey, RawAnalyticsOption]]] = [{} for _ in dimensions]
        for index, record in enumerate(report.profiles):
            if index % 128 == 0:
                control.check()
            profile = cls._profile(record=record, dimensions=dimensions, option_cache=option_cache)
            profiles.append(profile)
            for axis, values in zip(axes, profile.values, strict=True):
                for key, option in values.items():
                    cls._add_option(axis=axis, key=key, option=option)
        if query.compare_by is None:
            cls._groups(report=report, query=query, profiles=profiles, axis=axes[0])
        else:
            cls._matrix(report=report, query=query, profiles=profiles, axes=axes)
        control.check()

    @classmethod
    def _profile(
        cls,
        *,
        record: dict[str, Any],
        dimensions: list[AttackAnalyticsDimension],
        option_cache: list[dict[str | None, dict[ValueKey, RawAnalyticsOption]]],
    ) -> _Profile:
        """
        Validate a SQL profile and reuse its decoded membership maps within this request.

        Args:
            record (dict[str, Any]): Raw source strings/NULLs, saved outcome, and result-ID weight.
            dimensions (list[AttackAnalyticsDimension]): Axis semantics in SQL source-column order.
            option_cache (list[dict[str | None, dict[ValueKey, RawAnalyticsOption]]]): Independent
                raw-value caches per axis; request and response pipelines cannot overwrite one another.

        Returns:
            _Profile: Unique memberships and the original, nonnegative integer weight.

        Raises:
            AnalyticsDataException: If the stored count, outcome, or source representation is invalid.
        """
        weight, outcome = record["weight"], record["outcome"]
        if type(weight) is not int or weight < 0 or not isinstance(outcome, str):
            raise AnalyticsDataException("Stored categorical profiles contain invalid counts.")
        values = []
        for index, dimension in enumerate(dimensions):
            raw = record[f"source{index}"]
            if raw is not None and not isinstance(raw, str):
                raise AnalyticsDataException("A compact metadata profile is not a string.")
            cached = option_cache[index]
            if raw not in cached:
                cached[raw] = cls._values(raw=raw, dimension=dimension)
            values.append(cached[raw])
        return _Profile(values=values, outcome=outcome, weight=weight)

    @classmethod
    def _values(cls, *, raw: str | None, dimension: AttackAnalyticsDimension) -> dict[ValueKey, RawAnalyticsOption]:
        """
        Decode a single source without collapsing recorded emptiness into missing metadata.

        Attack types are scalar text, not JSON. Other supported sources are arrays:
        absent/JSON-null values and null members are missing; [] is no-converters
        only for a converter axis and is missing for harm categories. Historical
        converter objects contribute their class_name, not a serialized object label.

        Returns:
            dict[ValueKey, RawAnalyticsOption]: Unique typed keys with representative labels.

        Raises:
            AnalyticsDataException: If array metadata is invalid JSON, has the wrong
                shape, or contains unsupported member values.
        """
        if dimension.name is AttackAnalyticsDimensionName.ATTACK_TYPE:
            return cls._options([raw])
        try:
            values = json.loads(raw) if raw is not None else None
        except json.JSONDecodeError as error:
            raise AnalyticsDataException("Stored category/converter metadata is invalid JSON.") from error
        if values is None:
            return cls._options([None])
        if not isinstance(values, list):
            raise AnalyticsDataException("Stored category/converter metadata is not an array.")
        if not values:
            kind = (
                AttackAnalyticsValueKind.NO_CONVERTERS
                if dimension.name is AttackAnalyticsDimensionName.CONVERTER_TYPE
                else AttackAnalyticsValueKind.MISSING
            )
            return {(kind.value, ""): RawAnalyticsOption(key=AttackAnalyticsValue(kind=kind), label=None)}
        if dimension.name is AttackAnalyticsDimensionName.CONVERTER_TYPE:
            values = [value.get("class_name") if isinstance(value, dict) else value for value in values]
        return cls._options(values)

    @classmethod
    def _options(cls, values: list[Any]) -> dict[ValueKey, RawAnalyticsOption]:
        """
        Form deduplicated typed keys using SQLite's ASCII-only LOWER behavior.

        Python's Unicode lower/casefold would merge keys SQLite keeps distinct,
        making the resulting chart's drill-down predicates disagree with its counts.
        Missing keys use empty text internally, but their kind distinguishes them
        from a real blank string; literal ``Unknown`` is ordinary metadata.

        Returns:
            dict[ValueKey, RawAnalyticsOption]: One option per (kind, folded text) pair.

        Raises:
            AnalyticsDataException: If a member is neither a string nor missing.
        """
        options: dict[ValueKey, RawAnalyticsOption] = {}
        for label in values:
            if label is None:
                key = AttackAnalyticsValue(kind=AttackAnalyticsValueKind.MISSING)
            elif isinstance(label, str):
                key = AttackAnalyticsValue(value=label.translate(cls._ASCII_LOWER))
            else:
                raise AnalyticsDataException("Stored category/converter metadata contains a non-string value.")
            cls._add_option(
                axis=options,
                key=(key.kind.value, key.value or ""),
                option=RawAnalyticsOption(key=key, label=label),
            )
        return options

    @staticmethod
    def _add_option(*, axis: dict[ValueKey, RawAnalyticsOption], key: ValueKey, option: RawAnalyticsOption) -> None:
        """Keep the binary-smallest original label for a key, matching SQL MIN rather than arrival order."""
        current = axis.get(key)
        if current is None or (option.label or "") < (current.label or ""):
            axis[key] = option

    @staticmethod
    def _groups(
        *,
        report: RawAnalyticsReport,
        query: AttackAnalyticsQuery,
        profiles: list[_Profile],
        axis: dict[ValueKey, RawAnalyticsOption],
    ) -> None:
        """
        Sum each profile's weight once per group, then apply count/key ordering and pagination.

        A multi-valued profile contributes to multiple groups but never twice to
        the same key. Limits affect displayed groups and has_more, not the counts
        calculated for a group or the independent overall cohort counts.
        """
        counts: dict[ValueKey, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for profile in profiles:
            for key in profile.values[0]:
                counts[key][profile.outcome] += profile.weight
        ordered = sorted(counts, key=lambda key: (-sum(counts[key].values()), key))
        end = query.group_offset + query.group_limit
        report.has_more_groups = len(ordered) > end
        report.groups = [
            RawAnalyticsGroup(option=axis[key], counts=dict(counts[key])) for key in ordered[query.group_offset : end]
        ]

    @staticmethod
    def _matrix(
        *,
        report: RawAnalyticsReport,
        query: AttackAnalyticsQuery,
        profiles: list[_Profile],
        axes: list[dict[ValueKey, RawAnalyticsOption]],
    ) -> None:
        """
        Bound each key-ordered axis before accumulating weighted membership pairs.

        Only visible row/column combinations are expanded in Python. Within a
        profile the memberships are already sets, so a result contributes once to
        each cell even when both original arrays contain repeats. Axis truncation
        depends on all keys, not on which cells happen to be nonempty.
        """
        selected = [set(sorted(axis)[: query.axis_limit]) for axis in axes]
        report.axes_truncated = any(len(axis) > query.axis_limit for axis in axes)
        report.rows = [axes[0][key] for key in sorted(selected[0])]
        report.columns = [axes[1][key] for key in sorted(selected[1])]
        counts: dict[tuple[ValueKey, ValueKey], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for profile in profiles:
            for row in profile.values[0].keys() & selected[0]:
                for column in profile.values[1].keys() & selected[1]:
                    counts[row, column][profile.outcome] += profile.weight
        report.cells = [
            RawAnalyticsGroup(option=axes[0][row], column=axes[1][column], counts=dict(outcomes))
            for (row, column), outcomes in sorted(counts.items())
        ]
