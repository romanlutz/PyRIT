# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Portable, projection-only SQL for attack-result analytics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING, Any, ClassVar

from sqlalchemy import (
    Unicode,
    UnicodeText,
    and_,
    case,
    cast,
    column,
    func,
    literal,
    null,
    or_,
    select,
    true,
    union_all,
)

from pyrit.memory.analytics_sql import (
    JsonArrayAggregate,
    JsonArrayEmpty,
    JsonArrayItems,
    JsonArrayJoin,
    JsonContainer,
    JsonObjectAggregate,
    JsonScalar,
)
from pyrit.memory.memory_models import (
    AtomicAttackIdentifierEntry,
    AttackIdentifierEntry,
    AttackRequestConverterIdentifierEntry,
    AttackResponseConverterIdentifierEntry,
    AttackResultEntry,
    AttackTechniqueIdentifierEntry,
    ConverterIdentifierEntry,
    ScenarioResultEntry,
    TargetIdentifierEntry,
)
from pyrit.models import (
    AttackAnalyticsConverterDirection,
    AttackAnalyticsDimension,
    AttackAnalyticsDimensionName,
    AttackAnalyticsFacetQuery,
    AttackAnalyticsFilter,
    AttackAnalyticsFilters,
    AttackAnalyticsMatchMode,
    AttackAnalyticsQuery,
    AttackAnalyticsValue,
    AttackAnalyticsValueKind,
)

if TYPE_CHECKING:
    from sqlalchemy.sql import ColumnElement, Select
    from sqlalchemy.sql.selectable import CTE, CompoundSelect, FromClause

    from pyrit.common.pagination import DecodedKeysetCursor


@dataclass
class _Source:
    value: ColumnElement[Any]
    label: ColumnElement[Any]
    array: bool = False
    converters: bool = False
    insensitive: bool = False


class AttackAnalyticsQueryCompiler:
    """Build count and metadata queries without loading ORM result graphs."""

    _IDENTIFIER_DIMENSIONS: ClassVar[frozenset[AttackAnalyticsDimensionName]] = frozenset(
        {
            AttackAnalyticsDimensionName.ATTACK_TYPE,
            AttackAnalyticsDimensionName.CONVERTER_TYPE,
            AttackAnalyticsDimensionName.OBJECTIVE_TARGET,
            AttackAnalyticsDimensionName.MODEL,
        }
    )
    _COMPACT_MATRIX_DIMENSIONS: ClassVar[frozenset[AttackAnalyticsDimensionName]] = frozenset(
        {
            AttackAnalyticsDimensionName.TARGETED_HARM_CATEGORY,
            AttackAnalyticsDimensionName.ATTACK_TYPE,
            AttackAnalyticsDimensionName.CONVERTER_TYPE,
        }
    )

    def __init__(
        self,
        *,
        dialect: str,
        filters: AttackAnalyticsFilters,
        root: CTE | None = None,
        label_columns: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize an alias-safe result-ID query.

        Raises:
            NotImplementedError: If the memory dialect is unsupported.
        """
        if dialect not in {"sqlite", "mssql"}:
            raise NotImplementedError(f"Attack analytics does not support the {dialect!r} dialect")
        self.dialect = dialect
        self.filters = filters
        self.root = root if root is not None else AttackResultEntry.__table__.alias("analytics_results")
        self._atomic_hash = (
            self.root.c.atomic_attack_identifier_hash
            if root is not None
            else self.root.c.resolved_atomic_attack_identifier_hash
        )
        self._label_columns = label_columns or {}
        self._from: FromClause = self.root
        self._sources: dict[str, _Source] = {}
        self._identifiers_ready = False
        self.atomic = AtomicAttackIdentifierEntry.__table__.alias("analytics_atomic")
        self.technique = AttackTechniqueIdentifierEntry.__table__.alias("analytics_technique")
        self.attack = AttackIdentifierEntry.__table__.alias("analytics_attack")
        self.target = TargetIdentifierEntry.__table__.alias("analytics_target")
        self._converter_arrays: dict[AttackAnalyticsConverterDirection, ColumnElement[Any]] = {}

    def totals(self) -> Select[Any]:
        """
        Select raw counts by persisted outcome over distinct result IDs.

        Returns:
            Select[Any]: A bounded count projection.
        """
        conditions = self._conditions()
        return (
            select(self.root.c.outcome, func.count().label("count"))
            .select_from(self._from)
            .where(*conditions)
            .group_by(self.root.c.outcome)
        )

    def compact_profiles(self, *, query: AttackAnalyticsQuery, limit: int, max_value_length: int) -> Select[Any] | None:
        """
        Select bounded, pre-counted categorical profiles for SDK aggregation.

        Returns:
            Select[Any] | None: A SQLite profile query, or None for the general SQL path.
        """
        dimensions = [query.group_by] + ([query.compare_by] if query.compare_by is not None else [])
        if self.dialect != "sqlite" or not all(
            dimension.name in self._COMPACT_MATRIX_DIMENSIONS for dimension in dimensions
        ):
            return None
        compiler = self._group_compiler(dimensions)
        profiles = compiler._profiles([compiler._source(dimension) for dimension in dimensions])
        oversized = or_(
            *[func.length(profiles.c[f"source{index}"]) > max_value_length for index in range(len(dimensions))]
        )
        return select(
            *[
                (
                    case((func.length(value) > max_value_length, None), else_=value).label(key)
                    if key.startswith(("source", "display"))
                    else value
                )
                for key, value in profiles.c.items()
            ],
            oversized.label("oversized"),
        ).limit(limit)

    def groups(self, query: AttackAnalyticsQuery) -> Select[Any]:
        """
        Select one bounded page of grouped raw outcome counts.

        Returns:
            Select[Any]: Groups with JSON outcome/count dictionaries.
        """
        grouped = self._group_compiler([query.group_by])
        if grouped is not self:
            return grouped.groups(query)
        sources = [self._source(query.group_by)]
        profiles = self._profiles(sources)
        members = self._memberships(profiles=profiles, sources=sources, dimensions=[0], name="group_members")
        grouped = self._aggregate(members, dimensions=[0])
        return (
            select(grouped)
            .order_by(grouped.c.total.desc(), grouped.c.kind0, grouped.c.value0)
            .offset(query.group_offset)
            .limit(query.group_limit + 1)
        )

    def matrix(self, query: AttackAnalyticsQuery) -> CompoundSelect[Any]:
        """
        Select bounded axes and cells in one statement sharing compacted profiles.

        Returns:
            CompoundSelect[Any]: Tagged axis and cell rows.

        Raises:
            ValueError: If a comparison dimension is absent.
        """
        if query.compare_by is None:
            raise ValueError("A matrix requires compare_by")
        grouped = self._group_compiler([query.group_by, query.compare_by])
        if grouped is not self:
            return grouped.matrix(query)
        sources = [self._source(query.group_by), self._source(query.compare_by)]
        profiles = self._profiles(sources)
        if all(dimension.name in self._COMPACT_MATRIX_DIMENSIONS for dimension in (query.group_by, query.compare_by)):
            return self._compact_matrix(profiles=profiles, sources=sources, limit=query.axis_limit)
        candidates = [
            self._axis(profiles=profiles, sources=sources, index=index, limit=query.axis_limit + 1)
            for index in range(2)
        ]
        visible = [
            select(axis)
            .order_by(axis.c[f"kind{index}"], axis.c[f"value{index}"])
            .limit(query.axis_limit)
            .cte(f"visible_axis{index}")
            for index, axis in enumerate(candidates)
        ]
        members = self._memberships(profiles=profiles, sources=sources, dimensions=[0, 1], name="cell_members")
        selected = members.join(visible[0], self._same_key(members, visible[0], index=0)).join(
            visible[1], self._same_key(members, visible[1], index=1)
        )
        bounded = select(members).select_from(selected).cte("bounded_cell_members")
        cells = self._aggregate(bounded, dimensions=[0, 1])
        truncated = case(
            (
                or_(
                    *[
                        select(func.count()).select_from(axis).scalar_subquery() > query.axis_limit
                        for axis in candidates
                    ]
                ),
                True,
            ),
            else_=False,
        ).label("truncated")
        return self._matrix_output(axes=visible, cells=cells, truncated=truncated)

    def _compact_matrix(self, *, profiles: CTE, sources: list[_Source], limit: int) -> CompoundSelect[Any]:
        members = self._memberships(profiles=profiles, sources=sources, dimensions=[0, 1], name="compact_cell_members")
        cells = self._aggregate(members, dimensions=[0, 1])
        ranked = select(
            cells,
            func.dense_rank().over(order_by=[cells.c.kind0, cells.c.value0]).label("rank0"),
            func.dense_rank().over(order_by=[cells.c.kind1, cells.c.value1]).label("rank1"),
        ).cte("ranked_analytics_cells")
        axes = [
            select(
                ranked.c[f"kind{index}"],
                ranked.c[f"value{index}"],
                func.min(ranked.c[f"label{index}"]).label(f"label{index}"),
            )
            .where(ranked.c[f"rank{index}"] <= limit)
            .group_by(ranked.c[f"kind{index}"], ranked.c[f"value{index}"])
            .cte(f"compact_axis{index}")
            for index in range(2)
        ]
        truncated = case(
            (
                or_(*[select(func.max(ranked.c[f"rank{index}"])).scalar_subquery() > limit for index in range(2)]),
                True,
            ),
            else_=False,
        ).label("truncated")
        bounded = select(ranked).where(ranked.c.rank0 <= limit, ranked.c.rank1 <= limit).cte("visible_cells")
        return self._matrix_output(axes=axes, cells=bounded, truncated=truncated)

    @classmethod
    def _matrix_output(cls, *, axes: list[CTE], cells: CTE, truncated: ColumnElement[Any]) -> CompoundSelect[Any]:
        records = [cls._axis_record(axes[index], index=index, truncated=truncated) for index in range(2)]
        records.append(
            select(
                literal("cell").label("record"),
                cells.c.kind0,
                cells.c.value0,
                cells.c.label0,
                cells.c.kind1,
                cells.c.value1,
                cells.c.label1,
                cells.c.counts,
                truncated,
            )
        )
        return union_all(*records).order_by("record", "kind0", "value0", "kind1", "value1")

    def facet(self, query: AttackAnalyticsFacetQuery) -> Select[Any]:
        """
        Select only the opened facet's keys, without computing outcome aggregates.

        Returns:
            Select[Any]: A bounded, searchable facet page.
        """
        sources = [self._source(query.dimension)]
        if not sources[0].array:
            return self._scalar_facet(query=query, source=sources[0])
        profiles = self._profiles(sources, counts=False)
        members = self._memberships(profiles=profiles, sources=sources, dimensions=[0], name="facet_members")
        statement = select(
            members.c.kind0,
            members.c.value0,
            func.min(members.c.label0).label("label0"),
        )
        if query.search:
            escaped = query.search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            statement = statement.where(
                func.lower(members.c.label0).like(func.lower(literal(f"%{escaped}%")), escape="\\")
            )
        return (
            statement.group_by(members.c.kind0, members.c.value0)
            .order_by(members.c.kind0, members.c.value0)
            .offset(query.offset)
            .limit(query.limit + 1)
        )

    def _scalar_facet(self, *, query: AttackAnalyticsFacetQuery, source: _Source) -> Select[Any]:
        conditions = self._conditions()
        kind, value, label = self._scalar_key(source.value, source.label, insensitive=source.insensitive)
        if query.search:
            escaped = query.search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            conditions.append(func.lower(label).like(func.lower(literal(f"%{escaped}%")), escape="\\"))
        return (
            select(kind.label("kind0"), value.label("value0"), func.min(label).label("label0"))
            .select_from(self._from)
            .where(*conditions)
            .group_by(kind, value)
            .order_by(kind, value)
            .offset(query.offset)
            .limit(query.limit + 1)
        )

    def results(self, *, limit: int, after: DecodedKeysetCursor | None = None) -> Select[Any]:
        """
        Select a page of metadata, never responses or scores.

        Returns:
            Select[Any]: A recency-ordered metadata projection.
        """
        attack = self._source(AttackAnalyticsDimension(name=AttackAnalyticsDimensionName.ATTACK_TYPE))
        target = self._source(AttackAnalyticsDimension(name=AttackAnalyticsDimensionName.OBJECTIVE_TARGET))
        model = self._source(AttackAnalyticsDimension(name=AttackAnalyticsDimensionName.MODEL))
        request = self._converter_source(AttackAnalyticsConverterDirection.REQUEST)
        response = self._converter_source(AttackAnalyticsConverterDirection.RESPONSE)
        conditions = self._conditions()
        if after is not None:
            timestamp = after.timestamp.astimezone(UTC).replace(tzinfo=None)
            conditions.append(
                or_(
                    self.root.c.timestamp < timestamp,
                    and_(self.root.c.timestamp == timestamp, self.root.c.id < after.identifier),
                )
            )
        return (
            select(
                self.root.c.id.label("attack_result_id"),
                func.substr(self.root.c.objective, 1, 200).label("objective_preview")
                if self.dialect == "sqlite"
                else func.substring(self.root.c.objective, 1, 200).label("objective_preview"),
                self.root.c.outcome,
                self.root.c.timestamp.label("updated_at"),
                self.root.c.operation,
                self.root.c.operator,
                attack.label.label("attack_type"),
                model.label.label("target_model"),
                target.value.label("target_identifier_hash"),
                self.root.c.attribution_parent_id.label("scenario_result_id"),
                self.root.c.targeted_harm_categories,
                request.value.label("request_converters"),
                response.value.label("response_converters"),
                self.root.c.labels,
            )
            .select_from(self._from)
            .where(*conditions)
            .order_by(self.root.c.timestamp.desc(), self.root.c.id.desc())
            .limit(limit + 1)
        )

    def _source(self, dimension: AttackAnalyticsDimension) -> _Source:
        key = dimension.model_dump_json()
        if key not in self._sources:
            self._sources[key] = self._create_source(dimension)
        return self._sources[key]

    def _create_source(self, dimension: AttackAnalyticsDimension) -> _Source:
        name = dimension.name
        if name in {AttackAnalyticsDimensionName.OPERATION, AttackAnalyticsDimensionName.OPERATOR}:
            value = self.root.c[name.value]
            return _Source(value=value, label=value)
        if name is AttackAnalyticsDimensionName.LABEL:
            projected = self._label_columns.get(dimension.label_key or "")
            value = (
                self.root.c[projected]
                if projected is not None
                else JsonScalar(self.root.c.labels, f"$.{json.dumps(dimension.label_key)}")
            )
            return _Source(value=value, label=value)
        if name is AttackAnalyticsDimensionName.TARGETED_HARM_CATEGORY:
            value = cast(self.root.c.targeted_harm_categories, UnicodeText())
            return _Source(value=value, label=value, array=True, insensitive=True)
        if name is AttackAnalyticsDimensionName.SCENARIO:
            scenario = ScenarioResultEntry.__table__.alias("analytics_scenario")
            self._from = self._from.outerjoin(scenario, scenario.c.id == self.root.c.attribution_parent_id)
            value = func.lower(cast(self.root.c.attribution_parent_id, Unicode(36)))
            return _Source(value=value, label=func.coalesce(scenario.c.scenario_name, value))
        self._ensure_identifiers()
        if name is AttackAnalyticsDimensionName.CONVERTER_TYPE:
            return self._converter_source(dimension.converter_direction)
        if name is AttackAnalyticsDimensionName.ATTACK_TYPE:
            value = func.coalesce(self.attack.c.class_name, self._attack_property("class_name"))
            return _Source(value=value, label=value, insensitive=True)
        if name is AttackAnalyticsDimensionName.OBJECTIVE_TARGET:
            value = func.coalesce(self.target.c.hash, self._target_property("hash"))
            label = func.coalesce(
                self.target.c.model_name,
                self._target_property("params.model_name"),
                self.target.c.class_name,
                self._target_property("class_name"),
                value,
            )
            return _Source(value=value, label=label)
        value = func.coalesce(
            self.target.c.model_name,
            self.target.c.underlying_model_name,
            self._target_property("params.model_name"),
            self._target_property("params.underlying_model_name"),
        )
        return _Source(value=value, label=value)

    def _ensure_identifiers(self) -> None:
        if self._identifiers_ready:
            return
        self._from = (
            self._from.outerjoin(self.atomic, self.atomic.c.hash == self._atomic_hash)
            .outerjoin(self.technique, self.technique.c.hash == self.atomic.c.attack_technique_identifier_hash)
            .outerjoin(self.attack, self.attack.c.hash == self.technique.c.attack_identifier_hash)
            .outerjoin(self.target, self.target.c.hash == self.attack.c.objective_target_hash)
        )
        self._identifiers_ready = True

    def _attack_property(self, path: str) -> ColumnElement[Any]:
        return func.coalesce(
            JsonScalar(self.attack.c.identifier_json, f"$.{path}"),
            JsonScalar(self.root.c.atomic_attack_identifier, f"$.children.attack_technique.children.attack.{path}"),
            JsonScalar(self.root.c.atomic_attack_identifier, f"$.children.attack.{path}"),
        )

    def _target_property(self, path: str) -> ColumnElement[Any]:
        canonical_path = path.removeprefix("params.")
        return func.coalesce(
            JsonScalar(self.target.c.identifier_json, f"$.{canonical_path}"),
            JsonScalar(self.target.c.identifier_json, f"$.{path}"),
            self._attack_property(f"children.objective_target.{canonical_path}"),
            self._attack_property(f"children.objective_target.{path}"),
        )

    def _converter_source(self, direction: AttackAnalyticsConverterDirection) -> _Source:
        self._ensure_identifiers()
        if direction not in self._converter_arrays:
            table = (
                AttackRequestConverterIdentifierEntry.__table__
                if direction is AttackAnalyticsConverterDirection.REQUEST
                else AttackResponseConverterIdentifierEntry.__table__
            )
            edge = table.alias(f"analytics_{direction.value}_edge")
            converter = ConverterIdentifierEntry.__table__.alias(f"analytics_{direction.value}_converter")
            names = (
                select(
                    edge.c.attack_identifier_hash,
                    func.coalesce(
                        converter.c.class_name, JsonScalar(converter.c.identifier_json, "$.class_name")
                    ).label("class_name"),
                )
                .select_from(edge.outerjoin(converter, converter.c.hash == edge.c.converter_identifier_hash))
                .distinct()
                .cte(f"{self.root.name}_{direction.value}_converter_names")
            )
            arrays = (
                select(
                    names.c.attack_identifier_hash,
                    JsonArrayAggregate(names.c.class_name).label("names"),
                )
                .group_by(names.c.attack_identifier_hash)
                .cte(f"{self.root.name}_{direction.value}_converter_arrays")
            )
            self._from = self._from.outerjoin(arrays, arrays.c.attack_identifier_hash == self.attack.c.hash)
            path = f"children.{direction.value}_converters"
            self._converter_arrays[direction] = func.coalesce(
                arrays.c.names,
                JsonContainer(self.attack.c.identifier_json, f"$.{path}"),
                JsonContainer(
                    self.root.c.atomic_attack_identifier,
                    f"$.children.attack_technique.children.attack.{path}",
                ),
                JsonContainer(self.root.c.atomic_attack_identifier, f"$.children.attack.{path}"),
            )
        value = self._converter_arrays[direction]
        return _Source(value=value, label=value, array=True, converters=True, insensitive=True)

    def _conditions(self) -> list[ColumnElement[bool]]:
        conditions: list[ColumnElement[bool]] = []
        if self.filters.outcomes:
            conditions.append(self.root.c.outcome.in_([value.value for value in self.filters.outcomes]))
        if self.filters.updated_after is not None:
            conditions.append(self.root.c.timestamp >= self.filters.updated_after.astimezone(UTC).replace(tzinfo=None))
        if self.filters.updated_before is not None:
            conditions.append(self.root.c.timestamp < self.filters.updated_before.astimezone(UTC).replace(tzinfo=None))
        for index, predicate in enumerate(self.filters.dimensions):
            conditions.append(self._predicate(predicate, index=index))
        return conditions

    def _predicate(self, predicate: AttackAnalyticsFilter, *, index: int) -> ColumnElement[bool]:
        source = self._source(predicate.dimension)
        if not source.array:
            value = func.lower(source.value) if source.insensitive else source.value
            matches = []
            for option in predicate.values:
                if option.kind is AttackAnalyticsValueKind.MISSING:
                    matches.append(source.value.is_(None))
                else:
                    expected = literal(option.value, UnicodeText())
                    if source.insensitive:
                        expected = func.lower(expected)
                    matches.append(self._collate(value) == self._collate(expected))
            return or_(*matches)
        items = JsonArrayItems(source.value).table_valued(column("key"), column("value"), column("type"))
        items = items.alias(f"filter_items_{index}")
        kind, value, _ = self._array_key(source, source.value, items)
        matches = []
        for option in predicate.values:
            if option.kind is AttackAnalyticsValueKind.NO_CONVERTERS:
                matches.append(JsonArrayEmpty(source.value))
            elif option.kind is AttackAnalyticsValueKind.MISSING:
                matches.append(
                    or_(
                        source.value.is_(None),
                        source.value == "null",
                        JsonArrayEmpty(source.value) if not source.converters else literal(False),
                        select(1).select_from(items).where(kind == "missing").exists(),
                    )
                )
            else:
                matches.append(
                    select(1).select_from(items).where(self._key_condition(kind, value, option, source)).exists()
                )
        return and_(*matches) if predicate.match_mode is AttackAnalyticsMatchMode.ALL else or_(*matches)

    def _key_condition(
        self,
        kind: ColumnElement[Any],
        value: ColumnElement[Any],
        option: AttackAnalyticsValue,
        source: _Source,
    ) -> ColumnElement[bool]:
        if option.kind is not AttackAnalyticsValueKind.VALUE:
            return kind == option.kind.value
        expected = literal(option.value, UnicodeText())
        if source.insensitive:
            expected = func.lower(expected)
        return and_(kind == "value", self._collate(value) == self._collate(expected))

    def _collate(self, value: ColumnElement[Any]) -> ColumnElement[Any]:
        return value.collate("BINARY" if self.dialect == "sqlite" else "Latin1_General_100_BIN2")

    def _group_compiler(self, dimensions: list[AttackAnalyticsDimension]) -> AttackAnalyticsQueryCompiler:
        if "analytics_weight" in self.root.c or not any(
            dimension.name in self._IDENTIFIER_DIMENSIONS for dimension in dimensions
        ):
            return self
        base = AttackAnalyticsQueryCompiler(dialect=self.dialect, filters=self.filters)
        conditions = base._conditions()
        valid_atomic = (
            select(AtomicAttackIdentifierEntry.hash)
            .join(
                AttackTechniqueIdentifierEntry,
                AttackTechniqueIdentifierEntry.hash == AtomicAttackIdentifierEntry.attack_technique_identifier_hash,
            )
            .join(
                AttackIdentifierEntry,
                AttackIdentifierEntry.hash == AttackTechniqueIdentifierEntry.attack_identifier_hash,
            )
            .where(AttackIdentifierEntry.identifier_json.isnot(None))
        )
        if any(
            dimension.name
            in {
                AttackAnalyticsDimensionName.OBJECTIVE_TARGET,
                AttackAnalyticsDimensionName.MODEL,
            }
            for dimension in dimensions
        ):
            valid_atomic = valid_atomic.join(
                TargetIdentifierEntry,
                TargetIdentifierEntry.hash == AttackIdentifierEntry.objective_target_hash,
            ).where(TargetIdentifierEntry.identifier_json.isnot(None))
        values: dict[str, ColumnElement[Any]] = {
            "atomic_attack_identifier_hash": base._atomic_hash,
            "outcome": base.root.c.outcome,
            "atomic_attack_identifier": cast(null(), UnicodeText()),
        }
        label_columns: dict[str, str] = {}
        for index, dimension in enumerate(dimensions):
            if dimension.name in {
                AttackAnalyticsDimensionName.OPERATION,
                AttackAnalyticsDimensionName.OPERATOR,
            }:
                values[dimension.name.value] = base.root.c[dimension.name.value]
            elif dimension.name is AttackAnalyticsDimensionName.TARGETED_HARM_CATEGORY:
                values["targeted_harm_categories"] = base.root.c.targeted_harm_categories
            elif dimension.name is AttackAnalyticsDimensionName.SCENARIO:
                values["attribution_parent_id"] = base.root.c.attribution_parent_id
            elif dimension.name is AttackAnalyticsDimensionName.LABEL and dimension.label_key is not None:
                name = f"analytics_label_{index}"
                values[name] = JsonScalar(base.root.c.labels, f"$.{json.dumps(dimension.label_key)}")
                label_columns[dimension.label_key] = name
        complete = base._atomic_hash.in_(valid_atomic)
        # Keep modern facts index-only; only the legacy branch needs the full identifier JSON.
        modern = (
            select(
                *[expression.label(name) for name, expression in values.items()],
                func.count().label("analytics_weight"),
            )
            .select_from(base._from)
            .where(*conditions, complete)
            .group_by(*[value for name, value in values.items() if name != "atomic_attack_identifier"])
        )
        values["atomic_attack_identifier"] = cast(base.root.c.atomic_attack_identifier, UnicodeText())
        legacy = (
            select(
                *[expression.label(name) for name, expression in values.items()],
                func.count().label("analytics_weight"),
            )
            .select_from(base._from)
            .where(*conditions)
            .group_by(*values.values())
        )
        facts = union_all(
            modern,
            legacy.where(base._atomic_hash.is_(None)),
            legacy.where(base._atomic_hash.isnot(None), ~complete),
        ).cte("analytics_facts")
        return AttackAnalyticsQueryCompiler(
            dialect=self.dialect, filters=AttackAnalyticsFilters(), root=facts, label_columns=label_columns
        )

    def _profiles(self, sources: list[_Source], *, counts: bool = True) -> CTE:
        conditions = self._conditions()
        values: list[ColumnElement[Any]] = []
        for index, source in enumerate(sources):
            values.append(cast(source.value, UnicodeText()).label(f"source{index}"))
            if not source.array:
                values.append(cast(source.label, UnicodeText()).label(f"display{index}"))
        if not counts:
            return select(*values).select_from(self._from).where(*conditions).distinct().cte("analytics_profiles")
        weight = func.sum(self.root.c.analytics_weight) if "analytics_weight" in self.root.c else func.count()
        return (
            select(*values, self.root.c.outcome, weight.label("weight"))
            .select_from(self._from)
            .where(*conditions)
            .group_by(*values, self.root.c.outcome)
            .cte("analytics_profiles")
        )

    def _memberships(self, *, profiles: CTE, sources: list[_Source], dimensions: list[int], name: str) -> CTE:
        origin: FromClause = profiles
        values = list(profiles.c)
        keys: list[ColumnElement[Any]] = list(profiles.c)
        overlapping = any(sources[index].array for index in dimensions)
        for index in dimensions:
            source = sources[index]
            raw = profiles.c[f"source{index}"]
            if source.array:
                items = JsonArrayItems(raw).table_valued(column("key"), column("value"), column("type"))
                items = items.alias(f"{name}_items{index}")
                origin = JsonArrayJoin(origin, items, true(), isouter=True)
                kind, value, label = self._array_key(source, raw, items)
            else:
                kind, value, label = self._scalar_key(
                    raw, profiles.c[f"display{index}"], insensitive=source.insensitive
                )
            keys.extend([kind, value])
            values.extend(
                [
                    kind.label(f"kind{index}"),
                    value.label(f"value{index}"),
                    (func.min(label) if overlapping else label).label(f"label{index}"),
                ]
            )
        statement = select(*values).select_from(origin)
        if overlapping:
            statement = statement.group_by(*keys)
        return statement.cte(name)

    def _scalar_key(
        self, value: ColumnElement[Any], label: ColumnElement[Any], *, insensitive: bool = False
    ) -> tuple[ColumnElement[Any], ColumnElement[Any], ColumnElement[Any]]:
        key = func.lower(value) if insensitive else value
        return case((value.is_(None), "missing"), else_="value"), self._collate(func.coalesce(key, "")), label

    def _array_key(
        self, source: _Source, raw: ColumnElement[Any], items: FromClause
    ) -> tuple[ColumnElement[Any], ColumnElement[Any], ColumnElement[Any]]:
        object_type = "object" if self.dialect == "sqlite" else 5
        string_type = "text" if self.dialect == "sqlite" else 1
        label = cast(items.c.value, UnicodeText())
        if source.converters:
            document = case((items.c.type == object_type, items.c.value), else_="{}")
            label = case(
                (items.c.type == object_type, JsonScalar(document, "$.class_name")),
                else_=label,
            )
        valid_types = [string_type, object_type] if source.converters else [string_type]
        absent = or_(raw.is_(None), raw == "null")
        kind = case(
            (absent, "missing"),
            (JsonArrayEmpty(raw), "no_converters" if source.converters else "missing"),
            (
                func.json_type(raw) != "array" if self.dialect == "sqlite" else ~func.ltrim(raw).startswith("["),
                "invalid",
            ),
            (label.is_(None), "missing"),
            (~items.c.type.in_(valid_types), "invalid"),
            else_="value",
        )
        value = func.lower(label) if source.insensitive else label
        value = func.coalesce(value, "")
        return kind, self._collate(value), label

    def _axis(self, *, profiles: CTE, sources: list[_Source], index: int, limit: int) -> CTE:
        members = self._memberships(profiles=profiles, sources=sources, dimensions=[index], name=f"axis{index}_members")
        return (
            select(
                members.c[f"kind{index}"],
                members.c[f"value{index}"],
                func.min(members.c[f"label{index}"]).label(f"label{index}"),
            )
            .group_by(members.c[f"kind{index}"], members.c[f"value{index}"])
            .order_by(members.c[f"kind{index}"], members.c[f"value{index}"])
            .limit(limit)
            .cte(f"axis{index}_candidates")
        )

    @staticmethod
    def _aggregate(members: CTE, *, dimensions: list[int]) -> CTE:
        keys = [members.c[f"{field}{index}"] for index in dimensions for field in ("kind", "value")]
        labels = [func.min(members.c[f"label{index}"]).label(f"label{index}") for index in dimensions]
        buckets = (
            select(*keys, *labels, members.c.outcome, func.sum(members.c.weight).label("count"))
            .group_by(*keys, members.c.outcome)
            .cte(f"{members.name}_outcomes")
        )
        keys = [buckets.c[f"{field}{index}"] for index in dimensions for field in ("kind", "value")]
        return (
            select(
                *keys,
                *[func.min(buckets.c[f"label{index}"]).label(f"label{index}") for index in dimensions],
                JsonObjectAggregate(buckets.c.outcome, buckets.c.count).label("counts"),
                func.sum(buckets.c.count).label("total"),
            )
            .group_by(*keys)
            .cte(f"{members.name}_counts")
        )

    @staticmethod
    def _same_key(left: FromClause, right: FromClause, *, index: int) -> ColumnElement[bool]:
        # Kind separates missing values from real empty strings, keeping both join keys indexable.
        return and_(
            left.c[f"kind{index}"] == right.c[f"kind{index}"],
            left.c[f"value{index}"] == right.c[f"value{index}"],
        )

    @staticmethod
    def _axis_record(axis: CTE, *, index: int, truncated: ColumnElement[Any]) -> Select[Any]:
        empty = cast(null(), UnicodeText())
        return select(
            literal("row" if index == 0 else "column").label("record"),
            *[
                (axis.c[f"{field}{dimension}"] if dimension == index else empty).label(f"{field}{dimension}")
                for dimension in range(2)
                for field in ("kind", "value", "label")
            ],
            empty.label("counts"),
            truncated,
        )
