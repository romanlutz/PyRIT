# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared import-time helpers for the experimental promptgen subpackages.

Both ``gcg`` and ``pgd`` emit an ``ExperimentalWarning`` on import and expose
their torch-dependent public symbols lazily via a PEP 562 module ``__getattr__``,
so the cheap config / data / manifest helpers keep importing without ``torch``.
These helpers factor out that otherwise-duplicated boilerplate.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

from pyrit.exceptions import ExperimentalWarning

if TYPE_CHECKING:
    from collections.abc import Callable


def warn_experimental(*, module_name: str) -> None:
    """
    Emit the standard experimental-API warning for a promptgen subpackage.

    Args:
        module_name (str): The importing package name (pass ``__name__``), echoed
            into the warning so it points at the specific experimental module.
    """
    warnings.warn(
        f"{module_name} is experimental: APIs may change in any release without a "
        "deprecation cycle. Pin pyrit to a specific version if you depend on this "
        "module. To silence: "
        "warnings.filterwarnings('ignore', category=pyrit.exceptions.ExperimentalWarning).",
        ExperimentalWarning,
        stacklevel=3,
    )


def build_lazy_exports(
    *,
    module_globals: dict[str, Any],
    lazy_imports: dict[str, tuple[str, str]],
) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """
    Build PEP 562 ``__getattr__`` / ``__dir__`` hooks for a package's lazy exports.

    Torch-dependent public symbols are imported on first attribute access so that
    importing the package root stays torch-free. The returned pair is bound at
    module scope: ``__getattr__, __dir__ = build_lazy_exports(...)``.

    Args:
        module_globals (dict[str, Any]): The importing module's ``globals()``.
            Resolved attributes are cached here on first access.
        lazy_imports (dict[str, tuple[str, str]]): Maps each public name to the
            ``(source_module, attribute)`` to import on first access.

    Returns:
        tuple[Callable[[str], Any], Callable[[], list[str]]]: The ``__getattr__``
        and ``__dir__`` module hooks.
    """
    module_name = module_globals.get("__name__", "?")

    def lazy_getattr(name: str) -> Any:
        if name in lazy_imports:
            import importlib

            source_module, attr = lazy_imports[name]
            value = getattr(importlib.import_module(source_module), attr)
            module_globals[name] = value
            return value
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    def lazy_dir() -> list[str]:
        return sorted(set(list(module_globals.keys()) + list(lazy_imports.keys())))

    return lazy_getattr, lazy_dir
