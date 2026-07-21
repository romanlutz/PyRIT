# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit tests for the PGD CSV behavior loader (torch-free)."""

from __future__ import annotations

import pytest

from pyrit.executor.promptgen.pgd.config import PGDDataConfig
from pyrit.executor.promptgen.pgd.data import BehaviorRow, load_behaviors


def _write_csv(path, rows: str) -> str:
    path.write_text(rows, encoding="utf-8")
    return str(path)


def test_load_behaviors_full_columns(tmp_path) -> None:
    csv_path = _write_csv(
        tmp_path / "b.csv",
        "behavior_id,behavior,target,seed_image_path\n"
        "adv_1,Do a bad thing,Sure here it is,/img/1.png\n"
        "adv_2,Do another,Sure here again,/img/2.png\n",
    )
    rows = load_behaviors(data=PGDDataConfig(behaviors_csv=csv_path))
    assert rows == [
        BehaviorRow("adv_1", "Do a bad thing", "Sure here it is", "/img/1.png"),
        BehaviorRow("adv_2", "Do another", "Sure here again", "/img/2.png"),
    ]


def test_load_behaviors_synthesizes_missing_target_and_id(tmp_path) -> None:
    csv_path = _write_csv(tmp_path / "b.csv", "behavior\nBuild something dangerous\n")
    rows = load_behaviors(data=PGDDataConfig(behaviors_csv=csv_path))
    assert rows[0].behavior_id == "behavior_0000"
    assert rows[0].target_text.startswith("Sure, here is")
    assert rows[0].seed_image_path == ""


def test_load_behaviors_accepts_column_aliases(tmp_path) -> None:
    csv_path = _write_csv(tmp_path / "b.csv", "goal,target_text,image_path\nHarmful goal,Sure thing,/i.png\n")
    rows = load_behaviors(data=PGDDataConfig(behaviors_csv=csv_path))
    assert rows[0].behavior == "Harmful goal"
    assert rows[0].target_text == "Sure thing"
    assert rows[0].seed_image_path == "/i.png"


def test_load_behaviors_respects_n_behaviors_cap(tmp_path) -> None:
    csv_path = _write_csv(tmp_path / "b.csv", "behavior\na\nb\nc\n")
    rows = load_behaviors(data=PGDDataConfig(behaviors_csv=csv_path, n_behaviors=2))
    assert len(rows) == 2


def test_load_behaviors_requires_csv_path() -> None:
    with pytest.raises(ValueError, match="behaviors_csv"):
        load_behaviors(data=PGDDataConfig(behaviors_csv=""))


def test_load_behaviors_raises_on_missing_behavior_text(tmp_path) -> None:
    csv_path = _write_csv(tmp_path / "b.csv", "behavior,target\n,Sure thing\n")
    with pytest.raises(ValueError, match="no behavior text"):
        load_behaviors(data=PGDDataConfig(behaviors_csv=csv_path))
