# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from datetime import timedelta
from pathlib import Path

import pytest

from build_scripts.pyrit_wrapped.cli import _collection_session
from build_scripts.pyrit_wrapped.github_client import GitHubClient
from build_scripts.pyrit_wrapped.models import Period, TaxonomyConfig, WrappedError


def test_cache_does_not_retain_source_patches() -> None:
    assert GitHubClient._sanitize({"filename": "example.py", "patch": "unneeded source content"}) == {
        "filename": "example.py"
    }


def test_interrupted_session_preserves_cutoff(
    *, period: Period, taxonomy_config: TaxonomyConfig, tmp_path: Path
) -> None:
    path = tmp_path / "session.json"
    first = _collection_session(path=path, period=period, taxonomy=taxonomy_config, refresh=False)
    newer = period.model_copy(update={"cutoff": period.cutoff + timedelta(minutes=5)})
    resumed = _collection_session(path=path, period=newer, taxonomy=taxonomy_config, refresh=False)
    assert resumed.identifier == first.identifier
    assert resumed.period.cutoff == first.period.cutoff
    assert not resumed.complete


def test_refresh_uses_new_cutoff_and_request_namespace(
    *, period: Period, taxonomy_config: TaxonomyConfig, tmp_path: Path
) -> None:
    path = tmp_path / "session.json"
    first = _collection_session(path=path, period=period, taxonomy=taxonomy_config, refresh=False)
    newer = period.model_copy(update={"cutoff": period.cutoff + timedelta(minutes=5)})
    refreshed = _collection_session(path=path, period=newer, taxonomy=taxonomy_config, refresh=True)
    assert refreshed.identifier != first.identifier
    assert refreshed.period.cutoff == newer.cutoff


def test_changed_taxonomy_does_not_reuse_old_collection(
    *, period: Period, taxonomy_config: TaxonomyConfig, tmp_path: Path
) -> None:
    path = tmp_path / "session.json"
    first = _collection_session(path=path, period=period, taxonomy=taxonomy_config, refresh=False)
    changed = taxonomy_config.model_copy(update={"version": 2})
    second = _collection_session(path=path, period=period, taxonomy=changed, refresh=False)
    assert first.identifier != second.identifier


def test_session_year_mismatch_is_explicit(*, period: Period, taxonomy_config: TaxonomyConfig, tmp_path: Path) -> None:
    path = tmp_path / "session.json"
    _collection_session(path=path, period=period, taxonomy=taxonomy_config, refresh=False)
    earlier = Period.for_year(year=2025, now=period.cutoff)
    with pytest.raises(WrappedError, match="different reporting year"):
        _collection_session(path=path, period=earlier, taxonomy=taxonomy_config, refresh=False)
