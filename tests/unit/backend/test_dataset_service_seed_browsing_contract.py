# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""RED contract tests for DatasetService seed browsing policy (#2748)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import patch
from uuid import uuid4

import pytest

from pyrit.backend.models.common import PaginationInfo
from pyrit.backend.services.dataset_service import DatasetService
from pyrit.models import SeedObjective, SeedPrompt

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface


DATASET = "service-browse-contract"
SELECTION_KEY = f"dataset:named:{DATASET}"


def _field(value: Any, name: str) -> Any:
    """Read a contract field from either a Pydantic result or a mapping."""
    return value.get(name) if isinstance(value, dict) else getattr(value, name)


def _service_method(service: DatasetService, name: str):
    method = getattr(service, name, None)
    assert method is not None, f"RED: DatasetService.{name} is not implemented"
    return method


async def _add(memory: MemoryInterface, *seeds: SeedPrompt | SeedObjective) -> None:
    await memory.add_seeds_to_memory_async(seeds=list(seeds), added_by="2748-service-test")


@pytest.fixture
def dataset_service(sqlite_instance: MemoryInterface) -> DatasetService:
    with patch(
        "pyrit.backend.services.dataset_service.CentralMemory.get_memory_instance", return_value=sqlite_instance
    ):
        yield DatasetService()


class TestDatasetServiceSeedBrowsingContract:
    async def test_list_resolves_selection_key_and_returns_pagination_info(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        await _add(sqlite_instance, SeedPrompt(value="prompt", dataset_name=DATASET))
        response = await _service_method(dataset_service, "list_seed_examples_async")(
            selection_key=SELECTION_KEY, limit=10
        )
        assert isinstance(_field(response, "pagination"), PaginationInfo)
        assert _field(_field(response, "pagination"), "limit") == 10
        assert len(_field(response, "items")) == 1

    async def test_unnamed_selection_key_is_supported_without_display_name_resolution(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        await _add(sqlite_instance, SeedPrompt(value="unnamed"))
        response = await _service_method(dataset_service, "list_seed_examples_async")(
            selection_key="dataset:unnamed", limit=10
        )
        assert len(_field(response, "items")) == 1

    async def test_cursor_is_bound_to_selection_and_effective_filters(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        await _add(sqlite_instance, *(SeedPrompt(value=str(index), dataset_name=DATASET) for index in range(3)))
        first = await _service_method(dataset_service, "list_seed_examples_async")(selection_key=SELECTION_KEY, limit=1)
        cursor = _field(_field(first, "pagination"), "next_cursor")
        assert cursor
        with pytest.raises(ValueError):
            await _service_method(dataset_service, "list_seed_examples_async")(
                selection_key="dataset:unnamed", limit=1, cursor=cursor
            )
        with pytest.raises(ValueError):
            await _service_method(dataset_service, "list_seed_examples_async")(
                selection_key=SELECTION_KEY, limit=1, cursor=cursor, search="changed"
            )

    @pytest.mark.parametrize(
        "changed_filters",
        [
            {"search": "changed"},
            {"data_types": ["url"]},
            {"harm_categories": ["violence"]},
            {"seed_types": ["objective"]},
        ],
    )
    async def test_cursor_is_bound_to_each_effective_filter(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface, changed_filters: dict[str, object]
    ):
        await _add(
            sqlite_instance,
            SeedPrompt(value="one", dataset_name=DATASET),
            SeedPrompt(
                value="https://example.com/two", dataset_name=DATASET, data_type="url", harm_categories=["violence"]
            ),
        )
        first = await _service_method(dataset_service, "list_seed_examples_async")(selection_key=SELECTION_KEY, limit=1)
        cursor = _field(_field(first, "pagination"), "next_cursor")
        with pytest.raises(ValueError):
            await _service_method(dataset_service, "list_seed_examples_async")(
                selection_key=SELECTION_KEY, limit=1, cursor=cursor, **changed_filters
            )

    async def test_malformed_cursor_and_invalid_selection_are_rejected(self, dataset_service: DatasetService):
        with pytest.raises(ValueError):
            await _service_method(dataset_service, "list_seed_examples_async")(
                selection_key=SELECTION_KEY, limit=10, cursor="malformed"
            )
        with pytest.raises(ValueError):
            await _service_method(dataset_service, "list_seed_examples_async")(
                selection_key="display-name-not-selection-key", limit=10
            )

    async def test_list_formats_group_preview_types_modalities_counts_and_harm_summary(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        group_id = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(
                value="prompt",
                dataset_name=DATASET,
                prompt_group_id=group_id,
                data_type="text",
                harm_categories=["violence"],
            ),
            SeedObjective(value="objective", dataset_name=DATASET, prompt_group_id=group_id),
            SeedPrompt(value="".join("x" for _ in range(101)), dataset_name=DATASET, prompt_group_id=group_id),
        )
        response = await _service_method(dataset_service, "list_seed_examples_async")(
            selection_key=SELECTION_KEY, limit=10
        )
        item = _field(response, "items")[0]
        assert _field(item, "preview_truncated") is True
        assert _field(item, "preview") == ("x" * 100) + "..."
        assert _field(item, "piece_count") == 3
        assert _field(item, "objective_count") == 1
        assert "text" in _field(item, "modalities")
        assert "violence" in _field(item, "harm_categories")
        assert "objective" in _field(item, "seed_types")

    async def test_detail_returns_complete_group_and_persisted_provenance(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        group_id = uuid4()
        seed = SeedPrompt(
            value="full text",
            dataset_name=DATASET,
            prompt_group_id=group_id,
            role="user",
            sequence=4,
            source="source",
            authors=["author"],
            groups=["group"],
            metadata={"persisted": True},
            parameters=["name"],
        )
        await _add(
            sqlite_instance,
            seed,
            SeedObjective(
                value="condition",
                dataset_name=DATASET,
                prompt_group_id=group_id,
                metadata={"conditions": "must hold"},
            ),
        )
        detail = await _service_method(dataset_service, "get_seed_example_async")(
            selection_key=SELECTION_KEY, example_id=str(group_id)
        )
        members = _field(detail, "members")
        assert len(members) == 2
        prompt = next(member for member in members if _field(member, "id") == seed.id)
        objective = next(member for member in members if _field(member, "seed_type") == "objective")
        assert _field(prompt, "prompt_group_id") == group_id
        assert _field(prompt, "name") == seed.name
        assert _field(prompt, "value") == seed.value
        assert _field(prompt, "role") == "user"
        assert _field(prompt, "sequence") == 4
        stored_prompt = next(
            stored for stored in sqlite_instance.get_seeds(prompt_group_ids=[group_id]) if stored.id == seed.id
        )
        assert _field(prompt, "value_sha256") == stored_prompt.value_sha256
        assert _field(prompt, "dataset_name") == DATASET
        assert _field(prompt, "source") == "source"
        assert _field(prompt, "authors") == ["author"]
        assert _field(prompt, "groups") == ["group"]
        assert _field(prompt, "date_added") == stored_prompt.date_added
        assert _field(prompt, "added_by") == "2748-service-test"
        assert _field(prompt, "metadata") == {"persisted": True}
        assert _field(prompt, "data_type") == "text"
        assert _field(objective, "value") == "condition"
        assert _field(objective, "metadata") == {"conditions": "must hold"}

    async def test_detail_preserves_template_parameters_and_objective_conditions(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        group_id = uuid4()
        await _add(
            sqlite_instance,
            SeedPrompt(
                value="{{ name }}",
                dataset_name=DATASET,
                prompt_group_id=group_id,
                is_jinja_template=True,
                parameters=["name"],
            ),
            SeedObjective(value="condition", dataset_name=DATASET, prompt_group_id=group_id),
        )
        detail = await _service_method(dataset_service, "get_seed_example_async")(
            selection_key=SELECTION_KEY, example_id=str(group_id)
        )
        assert _field(detail, "members")
        template = next(member for member in _field(detail, "members") if _field(member, "seed_type") == "prompt")
        assert _field(template, "value") == "{{ name }}"
        assert _field(template, "is_jinja_template") is True
        assert _field(template, "parameters") == ["name"]
        assert any(_field(member, "seed_type") == "objective" for member in _field(detail, "members"))

    async def test_invalid_detail_does_not_generate_group_identity(self, dataset_service: DatasetService):
        with pytest.raises(ValueError):
            await _service_method(dataset_service, "get_seed_example_async")(
                selection_key=SELECTION_KEY, example_id=str(uuid4())
            )

    async def test_browsing_has_no_template_or_generation_side_effects(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface
    ):
        await _add(sqlite_instance, SeedPrompt(value="{{ dangerous }}", dataset_name=DATASET, is_jinja_template=True))
        with patch("pyrit.models.SeedPrompt.render_template_value", side_effect=AssertionError("rendered")) as render:
            response = await _service_method(dataset_service, "list_seed_examples_async")(
                selection_key=SELECTION_KEY, limit=10
            )
        assert _field(response, "items")
        assert render.call_count == 0

    async def test_media_preview_is_type_label_without_bytes_or_path_leak(
        self, dataset_service: DatasetService, sqlite_instance: MemoryInterface, tmp_path
    ):
        media = tmp_path / "private-image.png"
        media.write_bytes(b"local image")
        await _add(sqlite_instance, SeedPrompt(value=str(media), dataset_name=DATASET, data_type="image_path"))
        response = await _service_method(dataset_service, "list_seed_examples_async")(
            selection_key=SELECTION_KEY, limit=10
        )
        item = _field(response, "items")[0]
        assert "image" in _field(item, "preview").lower()
        assert str(tmp_path) not in _field(item, "preview")
        assert "bytes" not in (item if isinstance(item, dict) else item.model_dump())
