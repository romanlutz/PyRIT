# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""RED contract tests for the paginated seed browsing API (#2748).

These tests deliberately target the approved public HTTP contract.  The route,
models, and narrow memory read helpers are not present until the feature is
implemented; consequently the current expected failure is a missing-route
response, rather than a test-time production stub.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from pyrit.backend.main import app
from pyrit.models import SeedObjective, SeedPrompt, SeedSimulatedConversation

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface


DATASET = "browse-contract"
NAMED_KEY = f"dataset:named:{DATASET}"
UNNAMED_KEY = "dataset:unnamed"


@pytest.fixture
def client(patch_central_database) -> TestClient:
    """Use the real SQLite memory fixture behind the API application."""
    return TestClient(app)


async def _add(memory: MemoryInterface, *seeds: SeedPrompt | SeedObjective) -> None:
    await memory.add_seeds_to_memory_async(seeds=list(seeds), added_by="2748-test")


def _list(client: TestClient, selection_key: str = NAMED_KEY, **params: object):
    return client.get(f"/api/datasets/{selection_key}/seeds", params=params)


def _detail(client: TestClient, example_id: str, selection_key: str = NAMED_KEY):
    return client.get(f"/api/datasets/{selection_key}/seeds/{example_id}")


def _items(response):
    assert response.status_code == 200, response.text
    return response.json()["items"]


class TestEmptyAndDatasetSelection:
    async def test_empty_dataset_is_a_valid_empty_page(self, client, sqlite_instance: MemoryInterface):
        response = _list(client)
        assert response.status_code == 200
        body = response.json()
        assert body["items"] == []
        assert body["pagination"]["has_more"] is False
        assert body["pagination"]["next_cursor"] is None

    async def test_empty_filter_result_is_not_an_error(self, client, sqlite_instance: MemoryInterface):
        await _add(sqlite_instance, SeedPrompt(value="ordinary", dataset_name=DATASET))
        response = _list(client, search="absent")
        assert response.status_code == 200
        assert response.json()["items"] == []

    async def test_named_selection_key_is_not_display_name(self, client, sqlite_instance: MemoryInterface):
        await _add(sqlite_instance, SeedPrompt(value="named", dataset_name=DATASET))
        assert len(_items(_list(client, NAMED_KEY))) == 1
        assert _list(client, DATASET).status_code in {400, 404, 422}

    async def test_unnamed_selection_and_invalid_selection(self, client, sqlite_instance: MemoryInterface):
        await _add(sqlite_instance, SeedPrompt(value="unnamed"))
        assert len(_items(_list(client, UNNAMED_KEY))) == 1
        assert _list(client, "dataset:named:not-loaded").status_code in {400, 404, 422}

    async def test_named_unnamed_namespace_is_distinct(self, client, sqlite_instance: MemoryInterface):
        await _add(sqlite_instance, SeedPrompt(value="literal", dataset_name="__unnamed__"), SeedPrompt(value="none"))
        assert len(_items(_list(client, "dataset:named:__unnamed__"))) == 1
        assert len(_items(_list(client, UNNAMED_KEY))) == 1


class TestPaginationAndIdentity:
    async def test_one_page_has_existing_pagination_shape(self, client, sqlite_instance: MemoryInterface):
        await _add(sqlite_instance, SeedPrompt(value="one", dataset_name=DATASET))
        page = _list(client, limit=1)
        assert page.status_code == 200
        assert set(page.json()["pagination"]) >= {"limit", "has_more", "next_cursor", "prev_cursor"}

    async def test_multiple_pages_have_no_duplicates_or_omissions(self, client, sqlite_instance: MemoryInterface):
        seeds = [SeedPrompt(value=f"prompt-{i}", dataset_name=DATASET) for i in range(5)]
        await _add(sqlite_instance, *seeds)
        first = _list(client, limit=2)
        first_items = _items(first)
        second = _list(client, limit=2, cursor=first.json()["pagination"]["next_cursor"])
        all_items = first_items + _items(second)
        while second.json()["pagination"]["has_more"]:
            second = _list(client, limit=2, cursor=second.json()["pagination"]["next_cursor"])
            all_items += _items(second)
        ids = [item["example_id"] for item in all_items]
        assert len(ids) == len(set(ids)) == 5

    @pytest.mark.parametrize("limit", [0, -1, 101])
    async def test_page_size_is_validated(self, client, sqlite_instance: MemoryInterface, limit: int):
        assert _list(client, limit=limit).status_code in {400, 422}

    async def test_group_identity_preserves_ids_and_does_not_hash_merge(self, client, sqlite_instance: MemoryInterface):
        group_id = uuid4()
        first = SeedPrompt(value="same", dataset_name=DATASET, prompt_group_id=group_id)
        second = SeedPrompt(value="same", dataset_name=DATASET, prompt_group_id=group_id)
        ungrouped = SeedPrompt(value="same", dataset_name=DATASET)
        await _add(sqlite_instance, first, second, ungrouped)
        items = _items(_list(client))
        assert len(items) == 2
        assert {str(first.id), str(second.id), str(ungrouped.id)} == set(items[0]["seed_ids"] + items[1]["seed_ids"])
        assert sorted(len(item["seed_ids"]) for item in items) == [1, 2]
        assert all(item["example_id"] in {str(group_id), str(ungrouped.id)} for item in items)
        assert all("generated" not in item["example_id"] for item in items)

    async def test_order_is_complete_group_date_then_id(self, client, sqlite_instance: MemoryInterface):
        tied = datetime(2024, 1, 1, tzinfo=UTC)
        old_group = uuid4()
        new_group = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(value="old", dataset_name=DATASET, prompt_group_id=old_group, date_added=tied),
            SeedPrompt(
                value="new", dataset_name=DATASET, prompt_group_id=new_group, date_added=tied + timedelta(days=1)
            ),
            SeedPrompt(
                value="late member",
                dataset_name=DATASET,
                prompt_group_id=old_group,
                date_added=tied + timedelta(days=2),
            ),
        )
        items = _items(_list(client))
        assert [item["example_id"] for item in items] == [str(new_group), str(old_group)]

    async def test_tied_dates_use_descending_logical_id_tie_breaker(self, client, sqlite_instance: MemoryInterface):
        date_added = datetime(2024, 1, 1, tzinfo=UTC)
        lower = UUID("00000000-0000-0000-0000-000000000001")
        higher = UUID("00000000-0000-0000-0000-000000000002")
        await _add(
            sqlite_instance,
            SeedPrompt(value="lower", dataset_name=DATASET, prompt_group_id=lower, date_added=date_added),
            SeedPrompt(value="higher", dataset_name=DATASET, prompt_group_id=higher, date_added=date_added),
        )
        assert [item["example_id"] for item in _items(_list(client))] == [str(higher), str(lower)]

    async def test_group_is_never_split_and_detail_preserves_role_sequence(
        self, client, sqlite_instance: MemoryInterface, tmp_path
    ):
        group_id = uuid4()
        image_path = tmp_path / "group-image.png"
        image_path.write_bytes(b"local test image")
        await _add(
            sqlite_instance,
            SeedPrompt(
                value=str(image_path),
                dataset_name=DATASET,
                prompt_group_id=group_id,
                data_type="image_path",
                sequence=0,
            ),
            SeedPrompt(value="text", dataset_name=DATASET, prompt_group_id=group_id, data_type="text", sequence=1),
        )
        page = _list(client, limit=1)
        assert len(_items(page)) == 1
        detail = _detail(client, str(group_id))
        members = detail.json()["members"] if detail.status_code == 200 else []
        assert [member["sequence"] for member in members] == [0, 1]
        assert all("role" in member for member in members)


class TestFilters:
    async def test_modality_is_or_and_matching_member_returns_complete_group(
        self, client, sqlite_instance: MemoryInterface, tmp_path
    ):
        group_id = uuid4()
        image_path = tmp_path / "group-image.png"
        audio_path = tmp_path / "standalone-audio.wav"
        image_path.write_bytes(b"local test image")
        audio_path.write_bytes(b"local test audio")
        await _add(
            sqlite_instance,
            SeedPrompt(value=str(image_path), dataset_name=DATASET, prompt_group_id=group_id, data_type="image_path"),
            SeedPrompt(value="text", dataset_name=DATASET, prompt_group_id=group_id, data_type="text"),
            SeedPrompt(value=str(audio_path), dataset_name=DATASET, data_type="audio_path"),
        )
        items = _items(_list(client, modality=["image_path", "audio_path"]))
        assert {item["example_id"] for item in items} == {str(group_id)} | {
            str(next(seed.id for seed in sqlite_instance.get_seeds(data_types=["audio_path"])))
        }
        assert len(_detail(client, str(group_id)).json()["members"]) == 2

    @pytest.mark.parametrize("category, expected", [("Violence", True), ("vio", False), ("missing", False)])
    async def test_harm_category_is_case_insensitive_whole_value_and_missing_is_unlabeled(
        self, client, sqlite_instance: MemoryInterface, category: str, expected: bool
    ):
        seed = SeedPrompt(value="harm", dataset_name=DATASET, harm_categories=["VIOLENCE"])
        unlabeled = SeedPrompt(value="unlabeled", dataset_name=DATASET, harm_categories=[])
        await _add(sqlite_instance, seed, unlabeled)
        items = _items(_list(client, harm_category=category))
        assert (len(items) == 1 and str(seed.id) in items[0]["seed_ids"]) is expected
        all_items = _items(_list(client))
        unlabeled_item = next(item for item in all_items if str(unlabeled.id) in item["seed_ids"])
        assert unlabeled_item["has_unlabeled_harm"] is True

    async def test_multiple_harm_categories_are_or_not_existing_get_seeds_all_semantics(self, client, sqlite_instance):
        await _add(
            sqlite_instance,
            SeedPrompt(value="hate", dataset_name=DATASET, harm_categories=["hate"]),
            SeedPrompt(value="violence", dataset_name=DATASET, harm_categories=["violence"]),
            SeedPrompt(value="both", dataset_name=DATASET, harm_categories=["hate", "violence"]),
        )
        items = _items(_list(client, harm_category=["hate", "violence"]))
        assert len(items) == 3
        assert {item["seed_ids"][0] for item in items} == {
            str(seed.id) for seed in sqlite_instance.get_seeds(dataset_name=DATASET)
        }

    async def test_seed_type_filter_is_or_and_returns_complete_group(self, client, sqlite_instance):
        group_id = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(value="prompt", dataset_name=DATASET, prompt_group_id=group_id),
            SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=group_id),
        )
        items = _items(_list(client, seed_type=["objective"]))
        assert len(items) == 1 and items[0]["piece_count"] == 2

    async def test_filters_are_and_across_filters_but_match_at_example_level(self, client, sqlite_instance, tmp_path):
        group_id = uuid4()
        image_path = tmp_path / "filter-image.png"
        image_path.write_bytes(b"local test image")
        await _add(
            sqlite_instance,
            SeedPrompt(value=str(image_path), dataset_name=DATASET, prompt_group_id=group_id, data_type="image_path"),
            SeedPrompt(value="violence", dataset_name=DATASET, prompt_group_id=group_id, harm_categories=["violence"]),
            SeedPrompt(value=str(image_path), dataset_name=DATASET, data_type="image_path"),
        )
        items = _items(_list(client, modality="image_path", harm_category="violence"))
        assert len(items) == 1 and items[0]["example_id"] == str(group_id)


class TestTextSearchAndSafety:
    async def test_text_search_is_literal_case_insensitive_and_text_only(self, client, sqlite_instance, tmp_path):
        image_path = tmp_path / "media-path.png"
        image_path.write_bytes(b"local test image")
        await _add(
            sqlite_instance,
            SeedPrompt(value="Need 100% literal_value", dataset_name=DATASET),
            SeedObjective(value="OBJECTIVE text", dataset_name=DATASET),
            SeedPrompt(value=str(image_path), dataset_name=DATASET, data_type="image_path"),
            SeedPrompt(value="metadata-only", dataset_name=DATASET, metadata={"secret": "literal_value"}),
        )
        assert len(_items(_list(client, search="100% literal_value"))) == 1
        assert len(_items(_list(client, search="literal_value"))) == 1
        assert len(_items(_list(client, search="objective"))) == 1
        assert _items(_list(client, search="image_path")) == []

    async def test_cursor_is_opaque_bound_to_dataset_and_effective_filters(self, client, sqlite_instance):
        await _add(sqlite_instance, *(SeedPrompt(value=str(i), dataset_name=DATASET) for i in range(3)))
        cursor = _list(client, limit=1).json()["pagination"]["next_cursor"]
        assert cursor and not cursor.startswith("1")
        assert _list(client, limit=1, cursor="not-a-cursor").status_code in {400, 422}
        assert _list(client, UNNAMED_KEY, limit=1, cursor=cursor).status_code in {400, 422}
        assert _list(client, limit=1, search="different", cursor=cursor).status_code in {400, 422}

    async def test_missing_detail_example_is_not_silently_empty(self, client, sqlite_instance):
        response = _detail(client, str(uuid4()))
        assert response.status_code in {404, 422}

    async def test_template_is_not_rendered_or_loaded(self, client, sqlite_instance):
        template = SeedPrompt(
            value="{{ dangerous }}", dataset_name=DATASET, is_jinja_template=True, parameters=["dangerous"]
        )
        await _add(sqlite_instance, template)
        with patch.object(SeedPrompt, "render_template_value", side_effect=AssertionError("rendered")) as render:
            with patch("pathlib.Path.read_text", side_effect=AssertionError("loaded")) as load:
                response = _list(client, search="dangerous")
        assert response.status_code == 200
        assert render.call_count == load.call_count == 0
        item = response.json()["items"][0]
        assert item["is_template"] is True
        assert item["parameters"] == ["dangerous"]

    async def test_simulated_configuration_is_returned_without_generation_or_target_call(self, client, sqlite_instance):
        config = SeedSimulatedConversation(
            dataset_name=DATASET,
            adversarial_chat_system_prompt=SeedPrompt(value="{{ objective }}"),
            simulated_target_system_prompt=SeedPrompt(value="{{ objective }}"),
            num_turns=2,
        )
        await _add(sqlite_instance, config)
        with patch(
            "pyrit.executor.attack.multi_turn.simulated_conversation.generate_simulated_conversation_async"
        ) as generate:
            response = _list(client, search="num_turns")
        assert response.status_code == 200
        assert generate.call_count == 0
        assert response.json()["items"][0]["seed_types"] == ["simulated_conversation"]


class TestPreviewDetailAndCounts:
    async def test_preview_uses_100_character_convention_and_hides_full_content(self, client, sqlite_instance):
        short = "x" * 100
        long = "y" * 101
        await _add(
            sqlite_instance, SeedPrompt(value=short, dataset_name=DATASET), SeedPrompt(value=long, dataset_name=DATASET)
        )
        items = _items(_list(client))
        assert any(item["preview"] == short and item["preview_truncated"] is False for item in items)
        long_item = next(item for item in items if item["preview"].startswith("y"))
        assert len(long_item["preview"]) <= 103 and long_item["preview_truncated"] is True
        assert long not in long_item["preview"]

    async def test_media_preview_is_label_only_and_never_bytes_path_or_credentials(
        self, client, sqlite_instance, tmp_path
    ):
        image_path = tmp_path / "image.png"
        image_path.write_bytes(b"local test image")
        await _add(sqlite_instance, SeedPrompt(value=str(image_path), dataset_name=DATASET, data_type="image_path"))
        item = _items(_list(client))[0]
        assert "image" in item["preview"].lower()
        assert str(tmp_path) not in item["preview"] and "sig=" not in item["preview"]
        assert "bytes" not in item and "content" not in item

    async def test_detail_returns_all_persisted_fields_without_new_ids_or_rendering(self, client, sqlite_instance):
        seed_id = uuid4()
        group_id = uuid4()
        seed = SeedPrompt(
            id=seed_id,
            value="full text",
            dataset_name=DATASET,
            prompt_group_id=group_id,
            role="user",
            sequence=4,
            source="source",
            authors=["author"],
            groups=["group"],
            metadata={"persisted": "yes"},
        )
        await _add(sqlite_instance, seed)
        response = _detail(client, str(group_id))
        assert response.status_code == 200
        member = response.json()["members"][0]
        assert member["id"] == str(seed_id)
        assert member["prompt_group_id"] == str(group_id)
        assert member["value"] == "full text"
        for field in (
            "role",
            "sequence",
            "value_sha256",
            "dataset_name",
            "source",
            "authors",
            "groups",
            "date_added",
            "added_by",
            "metadata",
            "data_type",
        ):
            assert field in member

    async def test_counts_are_logical_examples_and_use_same_predicates(self, client, sqlite_instance, tmp_path):
        group_id = uuid4()
        image_path = tmp_path / "count-image.png"
        image_path.write_bytes(b"local test image")
        await _add(
            sqlite_instance,
            SeedPrompt(value=str(image_path), dataset_name=DATASET, prompt_group_id=group_id, data_type="image_path"),
            SeedObjective(value="two", dataset_name=DATASET, prompt_group_id=group_id),
            SeedPrompt(value="three", dataset_name=DATASET, data_type="text"),
        )
        response = _list(client, modality="image_path")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 1
        assert body["items"][0]["piece_count"] == 2
        assert body["items"][0]["objective_count"] == 1


class TestDatabaseBoundsAndSideEffects:
    async def test_page_query_is_bounded_and_does_not_n_plus_one(self, client, sqlite_instance):
        await _add(sqlite_instance, *(SeedPrompt(value=f"p{i}", dataset_name=DATASET) for i in range(250)))
        statements = []
        from sqlalchemy import event

        def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement.lower())

        event.listen(sqlite_instance.engine, "before_cursor_execute", capture)
        try:
            response = _list(client, limit=2)
        finally:
            event.remove(sqlite_instance.engine, "before_cursor_execute", capture)
        assert response.status_code == 200
        assert len(response.json()["items"]) == 2
        assert len(statements) < 12
        assert not any("select" in statement and "250" in statement for statement in statements)

    async def test_browsing_is_read_only_and_does_not_fetch_provider_or_write(self, client, sqlite_instance):
        with (
            patch(
                "pyrit.datasets.SeedDatasetProvider.get_all_dataset_names_async",
                side_effect=AssertionError("provider fetch"),
            ),
            patch.object(sqlite_instance, "add_seeds_to_memory_async", side_effect=AssertionError("write")) as write,
        ):
            response = _list(client)
        assert response.status_code == 200
        assert write.call_count == 0


class TestCompatibility:
    async def test_existing_dataset_list_route_remains_unchanged(self, client):
        response = client.get("/api/datasets")
        assert response.status_code == 200
        assert "items" in response.json()

    async def test_existing_get_seeds_harm_semantics_remain_all_categories(self, sqlite_instance: MemoryInterface):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", harm_categories=["hate"]),
            SeedPrompt(value="two", harm_categories=["hate", "violence"]),
        )
        assert len(sqlite_instance.get_seeds(harm_categories=["hate", "violence"])) == 1
