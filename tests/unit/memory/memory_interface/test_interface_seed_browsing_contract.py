# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""RED contract tests for the portable #2748 logical-example read helper."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event

from pyrit.models import SeedObjective, SeedPrompt, SeedSimulatedConversation

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


async def _add(memory: MemoryInterface, *seeds: SeedPrompt | SeedObjective | SeedSimulatedConversation) -> None:
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
        group_id = UUID("00000000-0000-0000-0000-000000000003")
        second_group_id = UUID("00000000-0000-0000-0000-000000000002")
        first_group_id = UUID("00000000-0000-0000-0000-000000000001")
        ungrouped_id = UUID("00000000-0000-0000-0000-000000000004")
        first_timestamp = datetime(2024, 1, 1, tzinfo=UTC)
        tied_timestamp = datetime(2024, 1, 2, tzinfo=UTC)
        group_three_first = SeedPrompt(
            value="group three first", dataset_name=DATASET, prompt_group_id=group_id, date_added=tied_timestamp
        )
        group_three_second = SeedPrompt(
            value="group three second", dataset_name=DATASET, prompt_group_id=group_id, date_added=tied_timestamp
        )
        group_two = SeedPrompt(
            value="group two", dataset_name=DATASET, prompt_group_id=second_group_id, date_added=tied_timestamp
        )
        group_one_early = SeedPrompt(
            value="group one early", dataset_name=DATASET, prompt_group_id=first_group_id, date_added=first_timestamp
        )
        group_one_late = SeedPrompt(
            value="group one late",
            dataset_name=DATASET,
            prompt_group_id=first_group_id,
            date_added=datetime(2024, 1, 3, tzinfo=UTC),
        )
        ungrouped = SeedPrompt(value="ungrouped", dataset_name=DATASET, id=ungrouped_id, date_added=first_timestamp)
        await _add(
            sqlite_instance,
            group_three_first,
            group_three_second,
            group_two,
            group_one_early,
            group_one_late,
            ungrouped,
        )
        pages: list[Any] = []
        cursor = None
        while True:
            page = _page(sqlite_instance, dataset_name=DATASET, limit=1, cursor=cursor)
            pages.extend(_field(page, "items"))
            cursor = _field(page, "next_cursor")
            if cursor is None:
                break

        assert [str(_field(item, "example_id")) for item in pages] == [
            str(group_id),
            str(second_group_id),
            str(ungrouped_id),
            str(first_group_id),
        ]
        assert [len(_field(item, "members")) for item in pages] == [2, 1, 1, 2]
        assert [{str(seed_id) for seed_id in _field(item, "seed_ids")} for item in pages] == [
            {str(group_three_first.id), str(group_three_second.id)},
            {str(group_two.id)},
            {str(ungrouped.id)},
            {str(group_one_early.id), str(group_one_late.id)},
        ]
        assert len({_field(item, "example_id") for item in pages}) == len(pages) == 4

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

    async def test_modality_harm_and_seed_type_values_are_or(self, sqlite_instance: MemoryInterface, tmp_path):
        modality_only = SeedPrompt(value="https://example.com/modality-only", dataset_name=DATASET, data_type="url")
        modality_only_second = SeedPrompt(
            value=str(tmp_path / "modality-only.png"), dataset_name=DATASET, data_type="image_path"
        )
        (tmp_path / "modality-only.png").write_bytes(b"image")
        harm_only = SeedPrompt(value="harm only", dataset_name=DATASET, data_type="reasoning", harm_categories=["hate"])
        seed_type_only = SeedSimulatedConversation(
            dataset_name=DATASET,
            adversarial_chat_system_prompt=SeedPrompt(value="adversarial"),
            simulated_target_system_prompt=SeedPrompt(value="target"),
        )
        all_filters_group = uuid4()
        all_filters_prompt = SeedPrompt(
            value="https://example.com/all-filters",
            dataset_name=DATASET,
            prompt_group_id=all_filters_group,
            data_type="url",
            harm_categories=["violence"],
        )
        all_filters_objective = SeedObjective(
            value="all filters objective", dataset_name=DATASET, prompt_group_id=all_filters_group
        )
        no_filters = SeedPrompt(
            value="no filters", dataset_name=DATASET, data_type="reasoning", harm_categories=["other"]
        )
        await _add(
            sqlite_instance,
            modality_only,
            modality_only_second,
            harm_only,
            seed_type_only,
            all_filters_prompt,
            all_filters_objective,
            no_filters,
        )
        modality_page = _page(sqlite_instance, dataset_name=DATASET, data_types=["url", "image_path"], limit=10)
        harm_page = _page(sqlite_instance, dataset_name=DATASET, harm_categories=["hate", "violence"], limit=10)
        seed_type_page = _page(
            sqlite_instance,
            dataset_name=DATASET,
            seed_types=["objective", "simulated_conversation"],
            limit=10,
        )
        combined_page = _page(
            sqlite_instance,
            dataset_name=DATASET,
            data_types=["url", "image_path"],
            harm_categories=["hate", "violence"],
            seed_types=["objective", "simulated_conversation"],
            limit=10,
        )

        assert {str(_field(item, "example_id")) for item in _field(modality_page, "items")} == {
            str(modality_only.id),
            str(modality_only_second.id),
            str(all_filters_group),
        }
        assert {str(_field(item, "example_id")) for item in _field(harm_page, "items")} == {
            str(harm_only.id),
            str(all_filters_group),
        }
        assert {str(_field(item, "example_id")) for item in _field(seed_type_page, "items")} == {
            str(seed_type_only.id),
            str(all_filters_group),
        }
        assert [str(_field(item, "example_id")) for item in _field(combined_page, "items")] == [str(all_filters_group)]
        assert _field(combined_page, "total") == 1

    async def test_filtered_order_uses_earliest_member_not_earliest_matching_member(
        self, sqlite_instance: MemoryInterface
    ):
        group_a = uuid4()
        group_b = uuid4()
        t1 = datetime(2024, 1, 1, tzinfo=UTC)
        t2 = datetime(2024, 1, 2, tzinfo=UTC)
        t3 = datetime(2024, 1, 3, tzinfo=UTC)
        await _add(
            sqlite_instance,
            SeedPrompt(value="a earliest", dataset_name=DATASET, prompt_group_id=group_a, date_added=t1),
            SeedPrompt(
                value="https://example.com/a-matching-later",
                dataset_name=DATASET,
                prompt_group_id=group_a,
                date_added=t3,
                data_type="url",
            ),
            SeedPrompt(
                value="https://example.com/b-matching",
                dataset_name=DATASET,
                prompt_group_id=group_b,
                date_added=t2,
                data_type="url",
            ),
        )
        page = _page(sqlite_instance, dataset_name=DATASET, data_types=["url"], limit=10)
        assert [str(_field(item, "example_id")) for item in _field(page, "items")] == [
            str(group_b),
            str(group_a),
        ]

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

    async def test_percent_search_is_literal(self, sqlite_instance: MemoryInterface):
        literal_match = SeedPrompt(value="contains 100%", dataset_name=DATASET)
        wildcard_only = SeedPrompt(value="contains 1000", dataset_name=DATASET)
        await _add(sqlite_instance, literal_match, wildcard_only)
        page = _page(sqlite_instance, dataset_name=DATASET, value_search="100%", limit=10)
        assert [str(_field(item, "seed_ids")[0]) for item in _field(page, "items")] == [str(literal_match.id)]

    async def test_underscore_search_is_literal(self, sqlite_instance: MemoryInterface):
        literal_match = SeedPrompt(value="contains a_b", dataset_name=DATASET)
        wildcard_only = SeedPrompt(value="contains acb", dataset_name=DATASET)
        await _add(sqlite_instance, literal_match, wildcard_only)
        page = _page(sqlite_instance, dataset_name=DATASET, value_search="a_b", limit=10)
        assert [str(_field(item, "seed_ids")[0]) for item in _field(page, "items")] == [str(literal_match.id)]

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
        groups = [uuid4() for _ in range(20)]
        await _add(
            sqlite_instance,
            *(
                SeedPrompt(value=f"seed-{index}", dataset_name=DATASET, prompt_group_id=group_id)
                for index, group_id in enumerate(groups)
            ),
            *(SeedPrompt(value=f"extra-{index}", dataset_name=DATASET) for index in range(250)),
        )
        statements: list[str] = []

        def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
            statements.append(statement.lower())

        event.listen(sqlite_instance.engine, "before_cursor_execute", capture)
        try:
            page = _page(sqlite_instance, dataset_name=DATASET, limit=20)
        finally:
            event.remove(sqlite_instance.engine, "before_cursor_execute", capture)
        assert len(_field(page, "items")) == 20
        select_statements = [statement for statement in statements if statement.lstrip().startswith("select")]
        grouped_page_queries = [
            statement for statement in select_statements if "group by" in statement and "order by" in statement
        ]
        assert grouped_page_queries
        assert all(" limit " in f" {statement} " for statement in grouped_page_queries)
        assert len(select_statements) <= 5

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

    @pytest.mark.parametrize(
        ("changed_filters", "changed_dataset"),
        [
            ({"value_search": "changed"}, None),
            ({"data_types": ["url"]}, None),
            ({"harm_categories": ["violence"]}, None),
            ({"seed_types": ["objective"]}, None),
            ({}, "another-memory-dataset"),
        ],
    )
    async def test_cursor_is_bound_to_every_effective_filter(
        self,
        sqlite_instance: MemoryInterface,
        changed_filters: dict[str, object],
        changed_dataset: str | None,
    ):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", dataset_name=DATASET),
            SeedPrompt(
                value="https://example.com/two", dataset_name=DATASET, data_type="url", harm_categories=["violence"]
            ),
        )
        first = _page(sqlite_instance, dataset_name=DATASET, limit=1)
        with pytest.raises(ValueError):
            _page(
                sqlite_instance,
                dataset_name=changed_dataset or DATASET,
                limit=1,
                cursor=_field(first, "next_cursor"),
                **changed_filters,
            )

    async def test_existing_get_seeds_harm_semantics_remain_all_categories(self, sqlite_instance: MemoryInterface):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", harm_categories=["hate"]),
            SeedPrompt(value="two", harm_categories=["hate", "violence"]),
        )
        assert len(sqlite_instance.get_seeds(harm_categories=["hate", "violence"])) == 1
