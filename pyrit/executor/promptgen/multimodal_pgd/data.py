# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
CSV -> behavior rows loader for the Multimodal PGD generator.

Decoupled from ``MultiModalPGDGenerator`` so callers with behaviors already in
memory can build ``BehaviorRow`` objects and iterate ``execute_async`` directly.
Uses the standard-library ``csv`` module (no ``pandas`` runtime dependency) and is
free of ``torch`` / ``transformers`` imports.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pyrit.executor.promptgen.multimodal_pgd.targets import default_affirmative_target

if TYPE_CHECKING:
    from pyrit.executor.promptgen.multimodal_pgd.config import MultiModalPGDDataConfig

_BEHAVIOR_COLUMNS = ("behavior", "behavior_text", "goal")
_TARGET_COLUMNS = ("target", "target_text")
_SEED_IMAGE_COLUMNS = ("seed_image_path", "image_path", "image")
_ID_COLUMNS = ("behavior_id", "id")


@dataclass
class BehaviorRow:
    """
    One behavior to attack.

    Attributes:
        behavior_id (str): Stable identifier for the behavior.
        behavior (str): The carrier behavior / instruction text.
        target_text (str): The affirmative target string to optimize for.
        seed_image_path (str): Path to the seed image (empty for the blank-image
            variant, which starts from random noise).
    """

    behavior_id: str
    behavior: str
    target_text: str
    seed_image_path: str = ""


def _first_present(row: dict[str, str], columns: tuple[str, ...]) -> str:
    """Return the first non-empty value among ``columns`` in ``row`` (else "")."""
    for column in columns:
        value = row.get(column)
        if value:
            return value.strip()
    return ""


def _open_csv(behaviors_csv: str) -> io.StringIO:
    """
    Open a CSV source as an in-memory text stream, supporting http(s) URLs.

    Local paths are read from disk; ``http://`` / ``https://`` sources are fetched
    into memory (stdlib only, so this module stays torch- and pandas-free).

    Returns:
        io.StringIO: An in-memory text stream over the CSV content.
    """
    if behaviors_csv.startswith(("http://", "https://")):
        with urllib.request.urlopen(behaviors_csv) as response:  # noqa: S310 - explicit http(s) guard above
            return io.StringIO(response.read().decode("utf-8"))
    with open(behaviors_csv, newline="", encoding="utf-8") as f:
        return io.StringIO(f.read())


def load_behaviors(*, data: MultiModalPGDDataConfig) -> list[BehaviorRow]:
    """
    Load behavior rows from the CSV configured by ``data``.

    The CSV must have a header with a behavior column (one of ``behavior`` /
    ``behavior_text`` / ``goal``). Target (``target`` / ``target_text``) and seed
    image (``seed_image_path`` / ``image_path`` / ``image``) columns are optional; a
    missing target falls back to ``default_affirmative_target``. A missing
    ``behavior_id`` / ``id`` column is filled with the zero-padded row index.

    Args:
        data (MultiModalPGDDataConfig): CSV path or URL plus optional row cap.

    Returns:
        list[BehaviorRow]: The parsed rows (capped at ``data.n_behaviors`` when > 0).

    Raises:
        ValueError: If ``behaviors_csv`` is empty or a row has no behavior text.
    """
    if not data.behaviors_csv:
        raise ValueError("load_behaviors: MultiModalPGDDataConfig.behaviors_csv must be set.")

    rows: list[BehaviorRow] = []
    reader = csv.DictReader(_open_csv(data.behaviors_csv))
    for index, raw in enumerate(reader):
        behavior = _first_present(raw, _BEHAVIOR_COLUMNS)
        if not behavior:
            raise ValueError(f"load_behaviors: row {index} has no behavior text in {data.behaviors_csv}.")
        target_text = _first_present(raw, _TARGET_COLUMNS) or default_affirmative_target(behavior=behavior)
        behavior_id = _first_present(raw, _ID_COLUMNS) or f"behavior_{index:04d}"
        rows.append(
            BehaviorRow(
                behavior_id=behavior_id,
                behavior=behavior,
                target_text=target_text,
                seed_image_path=_first_present(raw, _SEED_IMAGE_COLUMNS),
            )
        )

    if data.n_behaviors > 0:
        rows = rows[: data.n_behaviors]
    return rows


__all__ = ["BehaviorRow", "load_behaviors"]
