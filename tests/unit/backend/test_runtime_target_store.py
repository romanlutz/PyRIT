# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the runtime targets persistence store."""

import asyncio
import json
import logging
import os
import stat
import sys
from pathlib import Path

import pytest

from pyrit.backend.persistence.runtime_targets import (
    RuntimeTargetEntry,
    RuntimeTargetStore,
    sanitize_params,
)


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "runtime_targets.json"


@pytest.fixture
def store(store_path: Path) -> RuntimeTargetStore:
    return RuntimeTargetStore(path=store_path)


class TestRuntimeTargetEntry:
    """RuntimeTargetEntry construction-time guarantees."""

    def test_strips_api_key_from_params(self) -> None:
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="OpenAIChatTarget",
            auth_mode="api_key",
            params={"endpoint": "https://x", "api_key": "super-secret"},
        )

        assert "api_key" not in entry.params
        assert entry.params == {"endpoint": "https://x"}

    def test_sets_created_at_when_missing(self) -> None:
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="TextTarget",
            auth_mode="api_key",
        )

        assert entry.created_at != ""

    def test_preserves_existing_created_at(self) -> None:
        original_ts = "2024-01-01T00:00:00+00:00"
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="TextTarget",
            auth_mode="api_key",
            created_at=original_ts,
        )

        assert entry.created_at == original_ts


class TestSanitizeParams:
    """sanitize_params helper."""

    def test_removes_api_key_only(self) -> None:
        result = sanitize_params({"endpoint": "https://x", "model_name": "gpt", "api_key": "k"})
        assert result == {"endpoint": "https://x", "model_name": "gpt"}

    def test_returns_shallow_copy(self) -> None:
        original = {"endpoint": "https://x"}
        result = sanitize_params(original)
        result["endpoint"] = "https://changed"
        assert original["endpoint"] == "https://x"


class TestLoad:
    """RuntimeTargetStore.load_async."""

    async def test_load_returns_empty_when_file_missing(self, store: RuntimeTargetStore) -> None:
        assert await store.load_async() == []

    async def test_load_returns_empty_when_file_is_empty(self, store_path: Path) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("", encoding="utf-8")
        store = RuntimeTargetStore(path=store_path)

        assert await store.load_async() == []

    async def test_load_returns_empty_and_warns_on_malformed_json(
        self,
        store_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text("{not json", encoding="utf-8")
        store = RuntimeTargetStore(path=store_path)

        with caplog.at_level(logging.WARNING):
            entries = await store.load_async()

        assert entries == []
        assert any("malformed" in record.getMessage() for record in caplog.records)

    async def test_load_returns_empty_on_wrong_schema_version(
        self,
        store_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(
            json.dumps({"version": 999, "targets": [{"target_registry_name": "t", "type": "TextTarget"}]}),
            encoding="utf-8",
        )
        store = RuntimeTargetStore(path=store_path)

        with caplog.at_level(logging.WARNING):
            entries = await store.load_async()

        assert entries == []
        assert any("schema version" in record.getMessage() for record in caplog.records)

    async def test_load_returns_empty_when_root_is_not_object(
        self,
        store_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(json.dumps([{"foo": "bar"}]), encoding="utf-8")
        store = RuntimeTargetStore(path=store_path)

        with caplog.at_level(logging.WARNING):
            entries = await store.load_async()

        assert entries == []
        assert any("unexpected shape" in record.getMessage() for record in caplog.records)

    async def test_load_skips_invalid_entries_with_warning(
        self,
        store_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "targets": [
                        {"target_registry_name": "good", "type": "TextTarget", "auth_mode": "api_key"},
                        {"type": "TextTarget"},  # missing name
                        "not an object",
                        {"target_registry_name": "bad-auth", "type": "TextTarget", "auth_mode": "magic"},
                    ],
                }
            ),
            encoding="utf-8",
        )
        store = RuntimeTargetStore(path=store_path)

        with caplog.at_level(logging.WARNING):
            entries = await store.load_async()

        assert [e.target_registry_name for e in entries] == ["good"]


class TestAppendAndRemove:
    """RuntimeTargetStore.append_async and remove_async."""

    async def test_append_round_trips(self, store: RuntimeTargetStore) -> None:
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="OpenAIChatTarget",
            auth_mode="api_key",
            params={"endpoint": "https://x", "model_name": "gpt-4o"},
        )

        await store.append_async(entry)
        loaded = await store.load_async()

        assert len(loaded) == 1
        assert loaded[0].target_registry_name == "t-1"
        assert loaded[0].params == {"endpoint": "https://x", "model_name": "gpt-4o"}

    async def test_append_never_writes_api_key(self, store: RuntimeTargetStore, store_path: Path) -> None:
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="OpenAIChatTarget",
            auth_mode="api_key",
            params={"endpoint": "https://x", "api_key": "should-not-persist"},
        )

        await store.append_async(entry)

        raw = store_path.read_text(encoding="utf-8")
        payload = json.loads(raw)
        params_persisted = payload["targets"][0]["params"]
        assert "should-not-persist" not in raw
        assert "api_key" not in params_persisted

    async def test_append_replaces_entry_with_same_name(self, store: RuntimeTargetStore) -> None:
        first = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="OpenAIChatTarget",
            auth_mode="api_key",
            params={"endpoint": "https://old"},
        )
        second = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="OpenAIChatTarget",
            auth_mode="entra",
            params={"endpoint": "https://new"},
        )

        await store.append_async(first)
        await store.append_async(second)
        loaded = await store.load_async()

        assert len(loaded) == 1
        assert loaded[0].auth_mode == "entra"
        assert loaded[0].params == {"endpoint": "https://new"}

    async def test_remove_unknown_is_noop(self, store: RuntimeTargetStore) -> None:
        result = await store.remove_async("never-existed")
        assert result is False
        assert await store.load_async() == []

    async def test_remove_known_returns_true_and_drops_entry(self, store: RuntimeTargetStore) -> None:
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="TextTarget",
            auth_mode="api_key",
        )
        await store.append_async(entry)

        result = await store.remove_async("t-1")

        assert result is True
        assert await store.load_async() == []

    async def test_append_creates_parent_directory(self, tmp_path: Path) -> None:
        path = tmp_path / "deeply" / "nested" / "runtime_targets.json"
        store = RuntimeTargetStore(path=path)

        await store.append_async(
            RuntimeTargetEntry(
                target_registry_name="t-1",
                type="TextTarget",
                auth_mode="api_key",
            )
        )

        assert path.exists()

    async def test_writes_are_atomic_via_temp_file(self, store: RuntimeTargetStore, store_path: Path) -> None:
        entry = RuntimeTargetEntry(
            target_registry_name="t-1",
            type="TextTarget",
            auth_mode="api_key",
        )
        await store.append_async(entry)

        # The temp file must not linger after a successful write.
        assert not store_path.with_suffix(store_path.suffix + ".tmp").exists()

    async def test_concurrent_appends_are_serialized(self, store: RuntimeTargetStore) -> None:
        entries = [
            RuntimeTargetEntry(
                target_registry_name=f"t-{i}",
                type="TextTarget",
                auth_mode="api_key",
            )
            for i in range(10)
        ]

        await asyncio.gather(*(store.append_async(e) for e in entries))

        loaded = await store.load_async()
        assert sorted(e.target_registry_name for e in loaded) == [f"t-{i}" for i in range(10)]


@pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX permissions not enforced on Windows")
class TestPosixPermissions:
    """File permission policy on POSIX systems."""

    async def test_file_is_written_with_owner_only_permissions(
        self,
        store: RuntimeTargetStore,
        store_path: Path,
    ) -> None:
        await store.append_async(
            RuntimeTargetEntry(
                target_registry_name="t-1",
                type="TextTarget",
                auth_mode="api_key",
            )
        )

        mode = stat.S_IMODE(os.stat(store_path).st_mode)
        assert mode == 0o600
