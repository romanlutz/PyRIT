# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Azure SQL execution contract for the #2748 seed browsing seam."""

from __future__ import annotations

from contextlib import closing
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from pyrit.memory import AzureSQLMemory, SeedExampleDatasetScope
from pyrit.memory.memory_models import SeedEntry
from pyrit.models import SeedObjective, SeedPrompt


@pytest.mark.run_only_if_all_tests
async def test_seed_browsing_contract_on_azure_sql(azuresql_instance: AzureSQLMemory):
    test_id = str(uuid4())
    dataset = f"2748-azure-{test_id}"
    shared_group = uuid4()
    newer_group = uuid4()
    older_group = uuid4()
    base_time = datetime(2024, 1, 1, tzinfo=UTC)
    seeds = [
        SeedPrompt(
            value="https://example.com/violence",
            dataset_name=dataset,
            prompt_group_id=shared_group,
            data_type="url",
            harm_categories=["violence", "hate", 'special_%_"_é'],
            date_added=base_time,
            added_by=test_id,
        ),
        SeedObjective(
            value="objective",
            dataset_name=dataset,
            prompt_group_id=shared_group,
            date_added=base_time + timedelta(seconds=1),
            added_by=test_id,
        ),
        SeedPrompt(
            value="literal 100% a_b",
            dataset_name=dataset,
            prompt_group_id=newer_group,
            date_added=base_time + timedelta(days=1),
            harm_categories=["nonviolence", "violence-extra"],
            added_by=test_id,
        ),
        SeedPrompt(
            value="older",
            dataset_name=dataset,
            prompt_group_id=older_group,
            date_added=base_time,
            harm_categories=[],
            added_by=test_id,
        ),
        SeedPrompt(
            value="unnamed null",
            prompt_group_id=shared_group,
            harm_categories=None,
            date_added=base_time,
            added_by=test_id,
        ),
        SeedPrompt(
            value="unnamed empty",
            dataset_name="",
            prompt_group_id=shared_group,
            harm_categories=[],
            date_added=base_time + timedelta(seconds=1),
            added_by=test_id,
        ),
    ]
    await azuresql_instance.add_seeds_to_memory_async(seeds=seeds, added_by=test_id)

    try:
        named = SeedExampleDatasetScope.named(dataset)
        unnamed = SeedExampleDatasetScope.unnamed()

        named_page = azuresql_instance.get_seed_example_page(dataset_scope=named, limit=100)
        named_ids = {item.example_id for item in named_page.items}
        assert named_ids == {shared_group, newer_group, older_group}
        assert [item.example_id for item in named_page.items] == [
            newer_group,
            *sorted((shared_group, older_group), reverse=True),
        ]
        shared = next(item for item in named_page.items if item.example_id == shared_group)
        assert {member.id for member in shared.members} == {seeds[0].id, seeds[1].id}
        assert shared.objective_count == 1

        assert {
            item.example_id for item in azuresql_instance.get_seed_example_page(dataset_scope=unnamed, limit=100).items
        } == {shared_group}

        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, data_types=["url"], limit=100
            ).items
        } == {shared_group}
        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, seed_types=["objective"], limit=100
            ).items
        } == {shared_group}
        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, harm_categories=["VIOLENCE"], limit=100
            ).items
        } == {shared_group}
        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, harm_categories=["missing", "HATE"], limit=100
            ).items
        } == {shared_group}
        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, harm_categories=['special_%_"_É'], limit=100
            ).items
        } == {shared_group}
        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, harm_categories=["nonviolence"], limit=100
            ).items
        } == {newer_group}

        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, value_search="100%", limit=100
            ).items
        } == {newer_group}
        assert {
            item.example_id
            for item in azuresql_instance.get_seed_example_page(
                dataset_scope=named, value_search="a_b", limit=100
            ).items
        } == {newer_group}

        unlabeled = next(item for item in named_page.items if item.example_id == older_group)
        assert unlabeled.has_unlabeled_harm is True

        first = azuresql_instance.get_seed_example_page(dataset_scope=named, limit=2)
        assert first.next_cursor is not None
        second = azuresql_instance.get_seed_example_page(dataset_scope=named, limit=100, cursor=first.next_cursor)
        paged_ids = [item.example_id for item in first.items + second.items]
        assert len(paged_ids) == len(set(paged_ids)) == 3
        assert set(paged_ids) == named_ids
    finally:
        with closing(azuresql_instance.get_session()) as session:
            session.query(SeedEntry).filter(SeedEntry.added_by == test_id).delete(synchronize_session=False)
            session.commit()
