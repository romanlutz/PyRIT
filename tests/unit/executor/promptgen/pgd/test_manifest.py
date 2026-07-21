# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the PGD manifest schema + JSONL I/O (torch-free)."""

from __future__ import annotations

import json
import logging

from pyrit.executor.promptgen.pgd.manifest import (
    SCHEMA_VERSION,
    PGDManifestEntry,
    append_manifest_entry,
    read_manifest,
    write_manifest,
)


def _make_entry(**overrides) -> PGDManifestEntry:
    defaults = {
        "id": "pgd_llava_advbench_001_eps_bounded",
        "behavior_id": "advbench_001",
        "behavior_text": "Describe how to ...",
        "target_text": "Sure, here is how to ...",
        "image_path": "/cache/pgd_llava_advbench_001_eps_bounded.png",
        "vlm_id": "llava-hf/llava-1.5-7b-hf",
        "variant": "eps_bounded",
        "num_steps_run": 487,
        "final_loss": 0.034,
        "epsilon": 16 / 255,
        "step_size": 2 / 255,
        "stop_loss": 0.05,
        "succeeded_stop_criterion": True,
        "seed_image_path": "/data/cat.png",
    }
    defaults.update(overrides)
    return PGDManifestEntry(**defaults)


def test_entry_stamps_defaults() -> None:
    entry = _make_entry()
    assert entry.schema_version == SCHEMA_VERSION
    assert entry.transfer_eval_results is None
    assert entry.generated_at.endswith("Z")
    assert entry.pyrit_version


def test_json_line_round_trip() -> None:
    entry = _make_entry()
    line = entry.to_json_line()
    assert "\n" not in line
    restored = PGDManifestEntry.from_dict(json.loads(line))
    assert restored == entry


def test_from_dict_ignores_unknown_keys() -> None:
    data = _make_entry().to_dict()
    data["future_field"] = "ignored"
    restored = PGDManifestEntry.from_dict(data)
    assert restored.behavior_id == "advbench_001"


def test_write_and_read_manifest(tmp_path) -> None:
    entries = [_make_entry(id="a", behavior_id="a"), _make_entry(id="b", behavior_id="b")]
    path = tmp_path / "manifest.jsonl"
    write_manifest(entries=entries, path=path)

    read_back = read_manifest(path=path)
    assert [e.behavior_id for e in read_back] == ["a", "b"]


def test_append_manifest_entry_creates_and_grows(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    append_manifest_entry(entry=_make_entry(id="1", behavior_id="1"), path=path)
    append_manifest_entry(entry=_make_entry(id="2", behavior_id="2"), path=path)

    assert len(read_manifest(path=path)) == 2


def test_read_manifest_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(_make_entry().to_json_line() + "\n\n", encoding="utf-8")
    assert len(read_manifest(path=path)) == 1


def test_read_manifest_warns_on_schema_drift(tmp_path, caplog) -> None:
    path = tmp_path / "manifest.jsonl"
    data = _make_entry().to_dict()
    data["schema_version"] = 999
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        entries = read_manifest(path=path)

    assert len(entries) == 1
    assert any("schema_version" in record.message for record in caplog.records)


def test_verification_fields_default_to_none() -> None:
    entry = _make_entry()
    assert entry.model_response is None
    assert entry.target_emitted is None


def test_verification_fields_round_trip() -> None:
    entry = _make_entry(model_response="Sure, here is the plan", target_emitted=True)
    restored = PGDManifestEntry.from_dict(json.loads(entry.to_json_line()))
    assert restored.model_response == "Sure, here is the plan"
    assert restored.target_emitted is True
