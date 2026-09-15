# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Alias-safe JSON expressions used by SQLite and Azure SQL analytics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import Boolean, String, UnicodeText
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement
from sqlalchemy.sql.selectable import Join

if TYPE_CHECKING:
    from sqlalchemy.sql.compiler import SQLCompiler


class JsonScalar(FunctionElement[str]):
    """A scalar JSON property, addressed by a bound path."""

    type = UnicodeText()
    inherit_cache = True


class ResolvedAttackIdentifierHash(FunctionElement[str]):
    """The persisted identifier hash, with the canonical JSON hash as a legacy fallback."""

    type = String(64)
    inherit_cache = True


@compiles(ResolvedAttackIdentifierHash, "sqlite")
def _sqlite_resolved_hash(element: ResolvedAttackIdentifierHash, compiler: Any, **kwargs: Any) -> str:
    reference, document = _arguments(element, compiler, **kwargs)
    return f"coalesce({reference}, json_extract({document}, '$.hash'))"


@compiles(ResolvedAttackIdentifierHash, "mssql")
def _mssql_resolved_hash(element: ResolvedAttackIdentifierHash, compiler: Any, **kwargs: Any) -> str:
    reference, document = _arguments(element, compiler, **kwargs)
    return f"CONVERT(varchar(64), coalesce({reference}, JSON_VALUE({document}, '$.hash')))"


class JsonContainer(FunctionElement[str]):
    """An array or object JSON property, addressed by a bound path."""

    type = UnicodeText()
    inherit_cache = True


class JsonArrayItems(FunctionElement[str]):
    """A table-valued JSON array expansion."""

    type = UnicodeText()
    inherit_cache = True


class JsonArrayEmpty(FunctionElement[bool]):
    """Whether a JSON value is a recorded empty array, independent of whitespace."""

    type = Boolean()
    inherit_cache = True


class JsonArrayAggregate(FunctionElement[str]):
    """Aggregate scalar strings as a JSON array, including null elements."""

    type = UnicodeText()
    inherit_cache = True


class JsonObjectAggregate(FunctionElement[str]):
    """Aggregate outcome/count pairs into a JSON object."""

    type = UnicodeText()
    inherit_cache = True


class JsonArrayJoin(Join):
    """An outer lateral array expansion preserving empty/missing metadata."""

    inherit_cache = False


def _arguments(element: FunctionElement[Any], compiler: Any, **kwargs: Any) -> list[str]:
    return [compiler.process(argument, **kwargs) for argument in element.clauses]


@compiles(JsonScalar, "sqlite")
@compiles(JsonContainer, "sqlite")
def _sqlite_json_property(element: FunctionElement[str], compiler: Any, **kwargs: Any) -> str:
    return f"json_extract({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonScalar, "mssql")
def _mssql_json_scalar(element: JsonScalar, compiler: Any, **kwargs: Any) -> str:
    return f"JSON_VALUE({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonContainer, "mssql")
def _mssql_json_container(element: JsonContainer, compiler: Any, **kwargs: Any) -> str:
    return f"JSON_QUERY({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonArrayItems, "sqlite")
def _sqlite_json_items(element: JsonArrayItems, compiler: Any, **kwargs: Any) -> str:
    return f"json_each({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonArrayItems, "mssql")
def _mssql_json_items(element: JsonArrayItems, compiler: Any, **kwargs: Any) -> str:
    return f"OPENJSON({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonArrayEmpty, "sqlite")
def _sqlite_array_empty(element: JsonArrayEmpty, compiler: Any, **kwargs: Any) -> str:
    value = _arguments(element, compiler, **kwargs)[0]
    return f"CASE WHEN json_type({value}) = 'array' AND json_array_length({value}) = 0 THEN 1 ELSE 0 END"


@compiles(JsonArrayEmpty, "mssql")
def _mssql_array_empty(element: JsonArrayEmpty, compiler: Any, **kwargs: Any) -> str:
    value = _arguments(element, compiler, **kwargs)[0]
    return (
        f"CASE WHEN LEFT(LTRIM({value}), 1) = N'[' AND NOT EXISTS (SELECT 1 FROM OPENJSON({value})) THEN 1 ELSE 0 END"
    )


@compiles(JsonArrayAggregate, "sqlite")
def _sqlite_json_array(element: JsonArrayAggregate, compiler: Any, **kwargs: Any) -> str:
    return f"json_group_array({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonArrayAggregate, "mssql")
def _mssql_json_array(element: JsonArrayAggregate, compiler: Any, **kwargs: Any) -> str:
    value = _arguments(element, compiler, **kwargs)[0]
    item = (
        f"CASE WHEN {value} IS NULL THEN N'null' "
        f"ELSE CONCAT(N'\"', STRING_ESCAPE(CAST({value} AS nvarchar(max)), 'json'), N'\"') END"
    )
    return f"CONCAT(N'[', STRING_AGG(CAST({item} AS nvarchar(max)), N','), N']')"


@compiles(JsonObjectAggregate, "sqlite")
def _sqlite_json_object(element: JsonObjectAggregate, compiler: Any, **kwargs: Any) -> str:
    return f"json_group_object({', '.join(_arguments(element, compiler, **kwargs))})"


@compiles(JsonObjectAggregate, "mssql")
def _mssql_json_object(element: JsonObjectAggregate, compiler: Any, **kwargs: Any) -> str:
    key, value = _arguments(element, compiler, **kwargs)
    pair = f"CONCAT(N'\"', STRING_ESCAPE(CAST({key} AS nvarchar(max)), 'json'), N'\":', CAST({value} AS nvarchar(max)))"
    return f"CONCAT(N'{{', STRING_AGG({pair}, N','), N'}}')"


@compiles(JsonArrayJoin, "sqlite")
def _sqlite_array_join(element: JsonArrayJoin, compiler: SQLCompiler, **kwargs: Any) -> str:
    sql = compiler.visit_join(element, **kwargs)
    if not isinstance(sql, str):
        raise TypeError("The SQLite join compiler did not return SQL text")
    return sql


@compiles(JsonArrayJoin, "mssql")
def _mssql_array_join(element: JsonArrayJoin, compiler: Any, **kwargs: Any) -> str:
    options = {**kwargs, "asfrom": True}
    left = compiler.process(element.left, **options)
    right = compiler.process(element.right, **options)
    return f"{left} OUTER APPLY {right}"
