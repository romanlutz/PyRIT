# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""The multimodal_pgd package must emit an ExperimentalWarning on import."""

from __future__ import annotations

import importlib
import warnings

import pyrit.executor.promptgen.multimodal_pgd
from pyrit.exceptions import ExperimentalWarning


def test_importing_multimodal_pgd_emits_experimental_warning() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(pyrit.executor.promptgen.multimodal_pgd)

    experimental = [w for w in caught if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental) == 1
    assert "pyrit.executor.promptgen.multimodal_pgd is experimental" in str(experimental[0].message)


def test_experimental_warning_can_be_silenced() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", category=ExperimentalWarning)
        importlib.reload(pyrit.executor.promptgen.multimodal_pgd)

    assert [w for w in caught if issubclass(w.category, ExperimentalWarning)] == []
