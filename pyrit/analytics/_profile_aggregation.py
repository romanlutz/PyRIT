# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded SDK aggregation of SQLite's pre-counted categorical profiles."""

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
    values: list[dict[ValueKey, RawAnalyticsOption]]
    outcome: str
    weight: int


class ProfileAggregation:
    """Aggregate a bounded profile set while preserving SQL filter/key semantics."""

    _ASCII_LOWER: ClassVar[dict[int, int]] = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

    @classmethod
    def populate(cls, *, report: RawAnalyticsReport, query: AttackAnalyticsQuery, control: QueryControl) -> None:
        """Populate raw chart counts from the bounded profile projection."""
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
    def _values(cls, *, raw: Any, dimension: AttackAnalyticsDimension) -> dict[ValueKey, RawAnalyticsOption]:
        if dimension.name is AttackAnalyticsDimensionName.ATTACK_TYPE:
            return cls._options([raw])
        try:
            values = json.loads(raw) if isinstance(raw, str) else raw
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
        options: dict[ValueKey, RawAnalyticsOption] = {}
        for label in values:
            if label is None:
                key = AttackAnalyticsValue(kind=AttackAnalyticsValueKind.MISSING)
            elif isinstance(label, str):
                # SQLite LOWER folds ASCII only; Unicode lower() would break exact drill-downs.
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
