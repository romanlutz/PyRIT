# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import SeedEntry
from pyrit.models import Seed, SeedObjective, SeedPrompt, SeedSimulatedConversation

DATASET = "browse"
T0 = datetime(2024, 1, 1, tzinfo=UTC)


async def _add(memory: MemoryInterface, *seeds: Seed) -> None:
    await memory.add_seeds_to_memory_async(seeds=list(seeds), added_by="test")


async def _store(memory: MemoryInterface, entry: SeedEntry) -> None:
    async with await memory.get_session_async() as session:
        session.add(entry)
        await session.commit()


async def _ids(memory: MemoryInterface, dataset_name: str | None = DATASET, **filters) -> list[UUID]:
    examples, _, _ = await memory.get_seed_examples_async(dataset_name=dataset_name, limit=100, **filters)
    return list(examples)


async def test_get_seed_examples_orders_by_first_date_then_id_and_seeks(sqlite_instance: MemoryInterface):
    tied_low, tied_high = sorted([uuid4(), uuid4()])
    newest, oldest = uuid4(), uuid4()
    await _add(
        sqlite_instance,
        SeedPrompt(value="newest late", dataset_name=DATASET, prompt_group_id=newest, date_added=T0 + timedelta(3)),
        SeedPrompt(value="newest", dataset_name=DATASET, prompt_group_id=newest, date_added=T0 + timedelta(2)),
        SeedPrompt(value="tied low", dataset_name=DATASET, prompt_group_id=tied_low, date_added=T0 + timedelta(1)),
        SeedPrompt(value="tied high", dataset_name=DATASET, prompt_group_id=tied_high, date_added=T0 + timedelta(1)),
        SeedPrompt(value="oldest", dataset_name=DATASET, id=oldest, date_added=T0),
    )

    first, first_total, after = await sqlite_instance.get_seed_examples_async(dataset_name=DATASET, limit=2)
    second, second_total, last = await sqlite_instance.get_seed_examples_async(
        dataset_name=DATASET, limit=2, after=after
    )

    assert [*first, *second] == [newest, tied_high, tied_low, oldest]
    assert after is not None
    assert (after.timestamp, after.identifier) == (T0 + timedelta(1), str(tied_high))
    assert last is None
    assert first_total == second_total == 4


async def test_get_seed_examples_orders_filtered_examples_by_first_date_of_all_members(
    sqlite_instance: MemoryInterface,
):
    group, single = uuid4(), uuid4()
    await _add(
        sqlite_instance,
        SeedPrompt(value="old", dataset_name=DATASET, prompt_group_id=group, date_added=T0),
        SeedPrompt(value="needle", dataset_name=DATASET, prompt_group_id=group, date_added=T0 + timedelta(3)),
        SeedPrompt(value="needle", dataset_name=DATASET, id=single, date_added=T0 + timedelta(1)),
    )

    assert await _ids(sqlite_instance, value_search="needle") == [single, group]


async def test_get_seed_examples_returns_complete_groups_objective_first(sqlite_instance: MemoryInterface):
    group = uuid4()
    objective = SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=group)
    second = SeedPrompt(value="second", dataset_name=DATASET, prompt_group_id=group, sequence=1)
    first = SeedPrompt(value="first", dataset_name=DATASET, prompt_group_id=group, sequence=0)
    await _add(sqlite_instance, second, first, objective)

    examples, total, _ = await sqlite_instance.get_seed_examples_async(
        dataset_name=DATASET, limit=10, value_search="second"
    )

    assert total == 1
    assert [seed.id for seed in examples[group]] == [objective.id, first.id, second.id]
    assert isinstance(examples[group][0], SeedObjective)


async def test_get_seed_examples_filters_match_across_members(sqlite_instance: MemoryInterface):
    both, harm_only = uuid4(), uuid4()
    await _add(
        sqlite_instance,
        SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=both, harm_categories=["Violence"]),
        SeedPrompt(value="say hello", dataset_name=DATASET, prompt_group_id=both),
        SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=harm_only, harm_categories=["Violence"]),
        SeedPrompt(value="goodbye", dataset_name=DATASET, prompt_group_id=harm_only),
    )

    assert set(await _ids(sqlite_instance, harm_categories=["missing", "VIOLENCE"])) == {both, harm_only}
    assert await _ids(sqlite_instance, harm_categories=["violence"], value_search="HELLO") == [both]
    assert await _ids(sqlite_instance, seed_types=["objective"], data_types=["image_path"]) == []


async def test_get_seed_examples_keeps_named_and_unnamed_scopes_apart(sqlite_instance: MemoryInterface):
    group = uuid4()
    named = SeedPrompt(value="named", dataset_name=DATASET, prompt_group_id=group)
    null_name = SeedPrompt(value="null name", prompt_group_id=group)
    empty_name = SeedPrompt(value="empty name", dataset_name="", prompt_group_id=group)
    await _add(sqlite_instance, named, null_name, empty_name)

    named_examples, _, _ = await sqlite_instance.get_seed_examples_async(dataset_name=DATASET, limit=10)
    unnamed_examples, _, _ = await sqlite_instance.get_seed_examples_async(dataset_name=None, limit=10)

    assert [seed.id for seed in named_examples[group]] == [named.id]
    assert {seed.id for seed in unnamed_examples[group]} == {null_name.id, empty_name.id}
    assert await _ids(sqlite_instance, dataset_name="other") == []


@pytest.mark.parametrize(
    ("search", "expected"),
    [("%", ["50% off"]), ("_", ["snake_case"]), ("\\", ["a\\b"]), ("[x]", ["[x]"]), ("0_", []), ("x%", [])],
)
async def test_get_seed_examples_search_is_literal(sqlite_instance: MemoryInterface, search: str, expected: list[str]):
    await _add(
        sqlite_instance,
        *(SeedPrompt(value=value, dataset_name=DATASET) for value in ["50% off", "snake_case", "a\\b", "[x]"]),
    )

    examples, _, _ = await sqlite_instance.get_seed_examples_async(dataset_name=DATASET, limit=10, value_search=search)

    assert [seeds[0].value for seeds in examples.values()] == expected


async def test_get_seed_examples_search_ignores_media_values(sqlite_instance: MemoryInterface):
    media = SeedPrompt(value="/data/needle.png", data_type="image_path", dataset_name=DATASET, added_by="test")
    await _store(sqlite_instance, SeedEntry(entry=media))

    assert await _ids(sqlite_instance, value_search="needle") == []


async def test_get_seed_examples_search_ignores_simulated_conversation_json(sqlite_instance: MemoryInterface):
    group = uuid4()
    standalone = SeedSimulatedConversation(
        num_turns=2, adversarial_chat_system_prompt=SeedPrompt(value="needle"), dataset_name=DATASET
    )
    grouped = SeedSimulatedConversation(
        num_turns=2,
        adversarial_chat_system_prompt=SeedPrompt(value="needle"),
        dataset_name=DATASET,
        prompt_group_id=group,
    )
    objective = SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=group)
    await _add(sqlite_instance, standalone, grouped, objective)

    assert await _ids(sqlite_instance, value_search="needle") == []
    assert await _ids(sqlite_instance, value_search="num_turns") == []
    assert await _ids(sqlite_instance, value_search="objective") == [group]
    assert set(await _ids(sqlite_instance, seed_types=["simulated_conversation"])) == {standalone.id, group}


def _legacy_conversation_entry(*, missing_file: Path, prompt_group_id: UUID) -> SeedEntry:
    entry = SeedEntry(
        entry=SeedSimulatedConversation(
            num_turns=2,
            adversarial_chat_system_prompt=SeedPrompt(value="placeholder"),
            dataset_name=DATASET,
            prompt_group_id=prompt_group_id,
            added_by="test",
        )
    )
    entry.value = json.dumps(
        {"num_turns": 2, "sequence": 0, "adversarial_chat_system_prompt_path": str(missing_file)},
        sort_keys=True,
        separators=(",", ":"),
    )
    return entry


async def test_get_seed_examples_skips_seeds_that_cannot_be_read(
    sqlite_instance: MemoryInterface, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    mixed, broken = uuid4(), uuid4()
    prompt = SeedPrompt(value="readable", dataset_name=DATASET, prompt_group_id=mixed)
    await _add(sqlite_instance, prompt)
    mixed_entry = _legacy_conversation_entry(missing_file=tmp_path / "gone.yaml", prompt_group_id=mixed)
    broken_entry = _legacy_conversation_entry(missing_file=tmp_path / "gone.yaml", prompt_group_id=broken)
    skipped_ids = [str(mixed_entry.id), str(broken_entry.id)]
    await _store(sqlite_instance, mixed_entry)
    await _store(sqlite_instance, broken_entry)

    with (
        caplog.at_level(logging.WARNING, logger="pyrit.memory.memory_interface"),
        pytest.warns(DeprecationWarning, match="adversarial_chat_system_prompt_path"),
    ):
        examples, total, _ = await sqlite_instance.get_seed_examples_async(dataset_name=DATASET, limit=10)

    assert [seed.id for seed in examples[mixed]] == [prompt.id]
    assert broken not in examples
    assert total == 2
    messages = [record.getMessage() for record in caplog.records]
    assert all(any(seed_id in message for message in messages) for seed_id in skipped_ids)


async def test_get_seed_examples_returns_domain_simulated_conversation(sqlite_instance: MemoryInterface):
    conversation = SeedSimulatedConversation(
        num_turns=2,
        adversarial_chat_system_prompt=SeedPrompt(value="adversarial", parameters=["objective"]),
        simulated_target_system_prompt=SeedPrompt(value="target", parameters=["objective", "num_turns"]),
        dataset_name=DATASET,
    )
    await _add(sqlite_instance, conversation)

    seeds = await sqlite_instance.get_seed_example_async(dataset_name=DATASET, example_id=conversation.id)

    assert len(seeds) == 1
    assert isinstance(seeds[0], SeedSimulatedConversation)
    assert seeds[0].adversarial_chat_system_prompt.value == "adversarial"


async def test_get_seed_example_returns_empty_outside_its_dataset(sqlite_instance: MemoryInterface):
    group = uuid4()
    objective = SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=group, date_added=T0)
    await _add(sqlite_instance, objective, SeedPrompt(value="prompt", dataset_name=DATASET, prompt_group_id=group))

    seeds = await sqlite_instance.get_seed_example_async(dataset_name=DATASET, example_id=group)

    assert seeds[0].id == objective.id
    assert len(seeds) == 2
    assert await sqlite_instance.get_seed_example_async(dataset_name="other", example_id=group) == []
    assert await sqlite_instance.get_seed_example_async(dataset_name=None, example_id=group) == []
