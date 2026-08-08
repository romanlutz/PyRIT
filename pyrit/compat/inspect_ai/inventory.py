# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Deterministic static inventory of Inspect API usage."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pyrit.compat.inspect_ai.profile import InspectCompatibilityProfile


@dataclass(frozen=True, order=True)
class InspectApiUsage:
    """One statically observed Inspect API reference."""

    symbol: str
    source_file: str
    line: int


@dataclass(frozen=True)
class InspectApiInventory:
    """A deterministic compatibility inventory for one source tree."""

    source_root: str
    profile_id: str
    usages: tuple[InspectApiUsage, ...]
    unsupported_symbols: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible inventory data."""
        return {
            "source_root": self.source_root,
            "profile_id": self.profile_id,
            "usages": [asdict(usage) for usage in self.usages],
            "unsupported_symbols": list(self.unsupported_symbols),
        }


class _InspectUsageVisitor(ast.NodeVisitor):
    def __init__(self, *, source_file: str) -> None:
        self.source_file = source_file
        self.usages: set[InspectApiUsage] = set()
        self.aliases: dict[str, str] = {}

    def visit_Import(self, node: ast.Import) -> None:
        """Record imports rooted at ``inspect_ai``."""
        for alias in node.names:
            if alias.name != "inspect_ai" and not alias.name.startswith("inspect_ai."):
                continue
            self._record(symbol=alias.name, line=node.lineno)
            local_name = alias.asname or alias.name.split(".", 1)[0]
            self.aliases[local_name] = alias.name if alias.asname else "inspect_ai"
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Record symbols imported from ``inspect_ai`` modules."""
        if node.module != "inspect_ai" and not (node.module or "").startswith("inspect_ai."):
            return
        module = node.module or "inspect_ai"
        self._record(symbol=module, line=node.lineno)
        for alias in node.names:
            symbol = f"{module}.{alias.name}"
            self._record(symbol=symbol, line=node.lineno)
            self.aliases[alias.asname or alias.name] = symbol
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Record qualified attribute access through a known Inspect alias."""
        symbol = self._qualified_name(node)
        if symbol is not None and (symbol == "inspect_ai" or symbol.startswith("inspect_ai.")):
            self._record(symbol=symbol, line=node.lineno)
        self.generic_visit(node)

    def _qualified_name(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = self._qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None

    def _record(self, *, symbol: str, line: int) -> None:
        self.usages.add(InspectApiUsage(symbol=symbol, source_file=self.source_file, line=line))


def inventory_inspect_api_usage(
    *,
    source_root: Path,
    profile: InspectCompatibilityProfile,
    source_files: tuple[Path, ...] | None = None,
) -> InspectApiInventory:
    """
    Inventory Inspect symbols without importing or executing source.

    Returns:
        InspectApiInventory: Stable symbol/file/line inventory and unknown symbols.

    Raises:
        ValueError: If any requested source file escapes ``source_root``.
    """
    root = source_root.resolve()
    paths = source_files or tuple(sorted(root.rglob("*.py")))
    usages: set[InspectApiUsage] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"Inventory source '{path}' escapes source root '{root}'.")
        tree = ast.parse(resolved.read_text(encoding="utf-8"), filename=str(resolved))
        visitor = _InspectUsageVisitor(source_file=resolved.relative_to(root).as_posix())
        visitor.visit(tree)
        usages.update(visitor.usages)
    ordered = tuple(sorted(usages))
    unsupported = tuple(sorted({usage.symbol for usage in ordered if usage.symbol not in profile.supported_symbols}))
    return InspectApiInventory(
        source_root=str(root),
        profile_id=profile.profile_id,
        usages=ordered,
        unsupported_symbols=unsupported,
    )
