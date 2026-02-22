# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import inspect
import warnings

_PYRIT_PACKAGE_PREFIX = __name__.split(".")[0] + "."


def _is_internal_import() -> bool:
    """Check if the import originates from within the pyrit package."""
    for frame_info in inspect.stack():
        filename = frame_info.filename
        if filename == __file__:
            continue
        # Check if any frame in the call stack is a pyrit module (but not this file)
        module = inspect.getmodule(frame_info.frame)
        if module and module.__name__.startswith(_PYRIT_PACKAGE_PREFIX):
            return True
    return False


if not _is_internal_import():
    warnings.warn(
        "pyrit.auxiliary_attacks is deprecated and will be removed in release 0.12.0. "
        "Use pyrit.executor.workflow.GCGWorkflow instead. "
        "See doc/code/executor/workflow/3_gcg.ipynb for usage examples.",
        FutureWarning,
        stacklevel=2,
    )
