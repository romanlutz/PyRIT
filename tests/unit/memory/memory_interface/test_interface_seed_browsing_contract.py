# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""RED contract tests for the portable #2748 logical-example read helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event

from pyrit.models import SeedObjective, SeedPrompt

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface


DATASET = "memory-browse-contract"


def _field(value: Any, name: str) -> Any:
    """Read a contract field from either a Pydantic result or a mapping."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name)


def _page(memory: MemoryInterface, **kwargs: Any) -> Any:
    """Call the proposed narrow, database-backed logical-example page helper."""
    helper = getattr(memory, "get_seed_example_page", None)
    assert helper is not None, "RED: MemoryInterface.get_seed_example_page is not implemented"
    return helper(**kwargs)


async def _add(memory: MemoryInterface, *seeds: SeedPrompt | SeedObjective) -> None:
    await memory.add_seeds_to_memory_async(seeds=list(seeds), added_by="2748-memory-test")


class TestSeedBrowsingMemoryContract:
    async def test_logical_identity_uses_group_id_else_seed_id(self, sqlite_instance: MemoryInterface):
        group_id = uuid4()
        grouped = SeedPrompt(value="grouped", dataset_name=DATASET, prompt_group_id=group_id)
        ungrouped = SeedPrompt(value="ungrouped", dataset_name=DATASET)
        await _add(sqlite_instance, grouped, ungrouped)

        page = _page(sqlite_instance, dataset_name=DATASET, limit=10)
        examples = _field(page, "items")
        assert {str(_field(item, "example_id")) for item in examples} == {str(group_id), str(ungrouped.id)}
        assert {str(_field(item, "seed_ids")[0]) for item in examples} == {str(grouped.id), str(ungrouped.id)}

    async def test_order_is_earliest_complete_date_then_id(self, sqlite_instance: MemoryInterface):
        old_group = UUID("00000000-0000-0000-0000-000000000001")
        new_group = UUID("00000000-0000-0000-0000-000000000002")
        start = datetime(2024, 1, 1, tzinfo=UTC)
        await _add(
            sqlite_instance,
            SeedPrompt(value="old", dataset_name=DATASET, prompt_group_id=old_group, date_added=start),
            SeedPrompt(
                value="new", dataset_name=DATASET, prompt_group_id=new_group, date_added=start + timedelta(days=1)
            ),
            SeedPrompt(
                value="late member",
                dataset_name=DATASET,
                prompt_group_id=old_group,
                date_added=start + timedelta(days=3),
            ),
        )
        ids = [
            _field(item, "example_id")
            for item in _field(_page(sqlite_instance, dataset_name=DATASET, limit=10), "items")
        ]
        assert [str(value) for value in ids] == [str(new_group), str(old_group)]

    async def test_tied_timestamps_use_descending_logical_id(self, sqlite_instance: MemoryInterface):
        timestamp = datetime(2024, 1, 1, tzinfo=UTC)
        lower = UUID("00000000-0000-0000-0000-000000000001")
        higher = UUID("00000000-0000-0000-0000-000000000002")
        await _add(
            sqlite_instance,
            SeedPrompt(value="lower", dataset_name=DATASET, prompt_group_id=lower, date_added=timestamp),
            SeedPrompt(value="higher", dataset_name=DATASET, prompt_group_id=higher, date_added=timestamp),
        )
        ids = [
            _field(item, "example_id")
            for item in _field(_page(sqlite_instance, dataset_name=DATASET, limit=10), "items")
        ]
        assert [str(value) for value in ids] == [str(higher), str(lower)]

    async def test_cursor_continuation_pages_logical_examples_not_seed_rows(self, sqlite_instance: MemoryInterface):
        group_id = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(value="first member", dataset_name=DATASET, prompt_group_id=group_id),
            SeedPrompt(value="second member", dataset_name=DATASET, prompt_group_id=group_id),
            *(SeedPrompt(value=f"single-{index}", dataset_name=DATASET) for index in range(3)),
        )
        first = _page(sqlite_instance, dataset_name=DATASET, limit=1)
        second = _page(sqlite_instance, dataset_name=DATASET, limit=1, cursor=_field(first, "next_cursor"))
        assert len(_field(first, "items")) == len(_field(second, "items")) == 1
        assert _field(first, "items")[0] != _field(second, "items")[0]
        assert len(_field(_field(first, "items")[0], "members")) == 2

        seen = _field(first, "items") + _field(second, "items")
        cursor = _field(second, "next_cursor")
        while cursor:
            page = _page(sqlite_instance, dataset_name=DATASET, limit=1, cursor=cursor)
            seen += _field(page, "items")
            cursor = _field(page, "next_cursor")
        assert len({_field(item, "example_id") for item in seen}) == 4

    async def test_filters_are_member_or_and_example_and(self, sqlite_instance: MemoryInterface):
        group_id = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(value="image", dataset_name=DATASET, prompt_group_id=group_id, data_type="text"),
            SeedPrompt(value="violence", dataset_name=DATASET, prompt_group_id=group_id, harm_categories=["violence"]),
            SeedPrompt(value="only image", dataset_name=DATASET, data_type="text"),
        )
        page = _page(
            sqlite_instance,
            dataset_name=DATASET,
            data_types=["text"],
            harm_categories=["violence"],
            limit=10,
        )
        assert len(_field(page, "items")) == 1
        item = _field(page, "items")[0]
        expected_ids = {str(seed.id) for seed in sqlite_instance.get_seeds(prompt_group_ids=[group_id])}
        assert expected_ids == {str(seed_id) for seed_id in _field(item, "seed_ids")}
        assert len(_field(item, "members")) == 2

    async def test_modality_harm_and_seed_type_values_are_or(self, sqlite_instance: MemoryInterface):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", dataset_name=DATASET, data_type="text", harm_categories=["hate"]),
            SeedPrompt(value="two", dataset_name=DATASET, data_type="url", harm_categories=["violence"]),
            SeedObjective(value="three", dataset_name=DATASET),
        )
        page = _page(
            sqlite_instance,
            dataset_name=DATASET,
            data_types=["text", "url"],
            harm_categories=["hate", "violence"],
            seed_types=["prompt", "objective"],
            limit=10,
        )
        assert len(_field(page, "items")) == 3

    async def test_harm_matching_is_case_insensitive_whole_value_and_missing_is_unlabeled(
        self, sqlite_instance: MemoryInterface
    ):
        labeled = SeedPrompt(value="labeled", dataset_name=DATASET, harm_categories=["VIOLENCE"])
        unlabeled = SeedPrompt(value="unlabeled", dataset_name=DATASET, harm_categories=[])
        await _add(sqlite_instance, labeled, unlabeled)
        exact = _page(sqlite_instance, dataset_name=DATASET, harm_categories=["violence"], limit=10)
        substring = _page(sqlite_instance, dataset_name=DATASET, harm_categories=["vio"], limit=10)
        assert len(_field(exact, "items")) == 1
        assert str(_field(exact, "items")[0]["seed_ids"][0]) == str(labeled.id)
        assert len(_field(substring, "items")) == 0
        all_items = _field(_page(sqlite_instance, dataset_name=DATASET, limit=10), "items")
        unlabeled_item = next(
            item for item in all_items if str(unlabeled.id) in [str(i) for i in _field(item, "seed_ids")]
        )
        assert _field(unlabeled_item, "has_unlabeled_harm") is True

    @pytest.mark.parametrize("text", ["100% literal", "literal_value"])
    async def test_text_search_is_case_insensitive_literal_and_text_only(
        self, sqlite_instance: MemoryInterface, text: str, tmp_path
    ):
        media = tmp_path / "media-path.png"
        media.write_bytes(b"local image")
        await _add(
            sqlite_instance,
            SeedPrompt(value="Need 100% literal_value", dataset_name=DATASET),
            SeedPrompt(value=str(media), dataset_name=DATASET, data_type="image_path"),
            SeedPrompt(value="metadata", dataset_name=DATASET, metadata={"search": "metadata_only"}),
        )
        page = _page(sqlite_instance, dataset_name=DATASET, value_search=text.upper(), limit=10)
        assert len(_field(page, "items")) == 1
        assert (
            len(_field(_page(sqlite_instance, dataset_name=DATASET, value_search="metadata_only", limit=10), "items"))
            == 0
        )
        assert (
            len(_field(_page(sqlite_instance, dataset_name=DATASET, value_search="media-path", limit=10), "items")) == 0
        )

    async def test_count_uses_same_logical_predicates_as_page(self, sqlite_instance: MemoryInterface):
        group_id = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(value="matching", dataset_name=DATASET, prompt_group_id=group_id, data_type="text"),
            SeedObjective(value="related", dataset_name=DATASET, prompt_group_id=group_id),
            SeedPrompt(value="other", dataset_name=DATASET, data_type="text"),
        )
        page = _page(sqlite_instance, dataset_name=DATASET, data_types=["text"], limit=10)
        assert _field(page, "total") == 2
        grouped = next(item for item in _field(page, "items") if _field(item, "piece_count") == 2)
        assert str(_field(grouped, "example_id")) == str(group_id)

    async def test_only_selected_examples_are_expanded_to_all_members(self, sqlite_instance: MemoryInterface):
        selected = uuid4()
        not_selected = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(value="selected", dataset_name=DATASET, prompt_group_id=selected, harm_categories=["hate"]),
            SeedPrompt(value="selected related", dataset_name=DATASET, prompt_group_id=selected),
            SeedPrompt(value="unselected", dataset_name=DATASET, prompt_group_id=not_selected),
        )
        page = _page(sqlite_instance, dataset_name=DATASET, harm_categories=["hate"], limit=10)
        assert len(_field(page, "items")) == 1
        assert len(_field(_field(page, "items")[0], "members")) == 2

    async def test_query_is_database_bounded_and_not_n_plus_one(self, sqlite_instance: MemoryInterface):
        await _add(sqlite_instance, *(SeedPrompt(value=f"seed-{index}", dataset_name=DATASET) for index in range(250)))
        statements: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement.lower())

        event.listen(sqlite_instance.engine, "before_cursor_execute", capture)
        try:
            page = _page(sqlite_instance, dataset_name=DATASET, limit=2)
        finally:
            event.remove(sqlite_instance.engine, "before_cursor_execute", capture)
        assert len(_field(page, "items")) == 2
        assert len(statements) < 12
        assert any(" limit " in f" {statement} " for statement in statements)

    async def test_malformed_cursor_and_filter_mismatch_are_rejected(self, sqlite_instance: MemoryInterface):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", dataset_name=DATASET),
            SeedPrompt(value="two", dataset_name=DATASET, harm_categories=["violence"]),
        )
        with pytest.raises(ValueError):
            _page(sqlite_instance, dataset_name=DATASET, limit=1, cursor="malformed")
        first = _page(sqlite_instance, dataset_name=DATASET, limit=1)
        with pytest.raises(ValueError):
            _page(
                sqlite_instance,
                dataset_name=DATASET,
                harm_categories=["violence"],
                limit=1,
                cursor=_field(first, "next_cursor"),
            )

    async def test_existing_get_seeds_harm_semantics_remain_all_categories(self, sqlite_instance: MemoryInterface):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", harm_categories=["hate"]),
            SeedPrompt(value="two", harm_categories=["hate", "violence"]),
        )
        assert len(sqlite_instance.get_seeds(harm_categories=["hate", "violence"])) == 1
