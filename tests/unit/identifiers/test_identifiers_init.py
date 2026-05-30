# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import importlib

import pytest

from pyrit.identifiers import ComponentIdentifier


def test_scorer_identifier_alias_returns_component_identifier():
    """Accessing the deprecated ``ScorerIdentifier`` alias resolves to ``ComponentIdentifier``."""
    module = importlib.import_module("pyrit.identifiers")
    with pytest.warns(DeprecationWarning, match=r"pyrit\.identifiers\.ScorerIdentifier is deprecated"):
        alias = module.ScorerIdentifier
    assert alias is ComponentIdentifier


def test_scorer_identifier_alias_warning_mentions_removal_version():
    """The deprecation warning includes the planned removal version (0.16.0)."""
    module = importlib.import_module("pyrit.identifiers")
    with pytest.warns(DeprecationWarning, match=r"removed in 0\.16\.0"):
        _ = module.ScorerIdentifier


def test_unknown_attribute_raises_attribute_error():
    """Accessing a name that is neither a real export nor a deprecated alias raises ``AttributeError``."""
    module = importlib.import_module("pyrit.identifiers")
    with pytest.raises(AttributeError, match=r"module 'pyrit\.identifiers' has no attribute 'DoesNotExist'"):
        _ = module.DoesNotExist
