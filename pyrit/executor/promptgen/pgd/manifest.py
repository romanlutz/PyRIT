# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Manifest schema and JSONL I/O for PGD — the Phase 1 hand-off contract.

The PGD generator writes a JSONL manifest (one ``PGDManifestEntry`` per
line) alongside the perturbed PNGs it caches under ``seed-prompt-entries``. A
downstream ``VisualPromptInjection`` scenario reads this manifest via a
``_PGDPrecomputedDataset`` loader (which lives in the scenario package, not here)
and maps each row to an image+text seed prompt.

Manifest schema (``schema_version`` 3), one JSON object per line::

    {
      "id": "pgd_llava_1_5_7b_advbench_001_eps_bounded",
      "behavior_id": "advbench_001",
      "behavior_text": "Describe how to ...",
      "target_text": "Sure, here is how to ...",
      "image_path": ".../seed-prompt-entries/pgd_..._eps_bounded.png",
      "vlm_id": "llava-hf/llava-1.5-7b-hf",
      "variant": "eps_bounded",              # one of eps_bounded | blank_image | patch
      "num_steps_run": 487,
      "final_loss": 0.034,
      "deployed_loss": 0.036,                # loss recomputed on the reloaded 8-bit PNG
      "epsilon": 0.0627,
      "step_size": 0.00784,
      "stop_loss": 0.05,
      "succeeded_stop_criterion": true,
      "seed_image_path": ".../cats/cat_001.png",
      "model_response": "Sure, here is how to ...",  # null when verification skipped
      "target_emitted": true,                # null when verification skipped
      "generated_at": "2026-05-30T18:00:00Z",
      "pyrit_version": "0.14.0",
      "schema_version": 3,
      "transfer_eval_results": null           # reserved for a future cross-VLM ASR runner
    }

Notes:
- All paths are absolute on the writer machine; a reader is expected to rewrite them
  relative to its local cache root.
- ``final_loss`` is measured during optimization on the deployed 8-bit image (the loop
  scores a straight-through quantized tensor). ``deployed_loss`` re-measures the loss on
  the PNG after it is rendered and re-preprocessed from scratch; the two should agree
  closely, and a large gap flags a remaining optimize-vs-deploy mismatch. ``deployed_loss``
  is ``null`` when the recomputation was skipped or failed.
- ``model_response`` / ``target_emitted`` record a *functional* check: the crafted image
  is fed back through the VLM and the reply is compared to ``target_text``. Both are
  ``null`` when verification was disabled or the target cannot generate responses;
  ``succeeded_stop_criterion`` still reflects only the optimization loss.
- ``transfer_eval_results`` is reserved as ``null`` in v1; future versions can fill it
  with ``[{target_vlm_id, attack_success_rate, scorer_id}, ...]``.
- ``read_manifest`` warns (rather than fails) on unknown ``schema_version`` values so
  that a newer manifest remains loosely readable by an older reader.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 3


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a trailing ``Z``."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _pyrit_version() -> str:
    """Return the installed pyrit version string (looked up lazily)."""
    import pyrit

    return pyrit.__version__


@dataclass
class PGDManifestEntry:
    """
    One row of a PGD manifest.

    See the module docstring for the full field-by-field schema. Prefer the
    ``create`` classmethod for new rows so ``generated_at``, ``pyrit_version``, and
    ``schema_version`` are stamped consistently.

    Attributes:
        id (str): Deterministic unique row id ``pgd_<vlm_slug>_<behavior_id>_<variant>``.
        behavior_id (str): Stable identifier for the behavior (e.g. an AdvBench id).
        behavior_text (str): The benign carrier prompt paired with the image.
        target_text (str): The affirmative target string the attack optimized for.
        image_path (str): Absolute path to the cached perturbed PNG.
        vlm_id (str): HuggingFace id of the VLM the image was crafted against.
        variant (str): PGD variant value (``eps_bounded`` | ``blank_image`` | ``patch``).
        num_steps_run (int): Number of optimization steps actually executed.
        final_loss (float): Loss at the final step, scored on the deployed 8-bit image.
        epsilon (float): Epsilon bound used (normalized model space).
        step_size (float): Per-step gradient-sign step size used.
        stop_loss (float): Early-stop threshold that was configured.
        succeeded_stop_criterion (bool): Whether the run hit ``final_loss <= stop_loss``.
        deployed_loss (float | None): Loss recomputed on the PNG after it is rendered and
            re-preprocessed from scratch; should closely track ``final_loss``. ``None`` when
            the recomputation was skipped or failed.
        seed_image_path (str): Absolute path to the seed image (empty for blank-image).
        model_response (str | None): The VLM's decoded reply when the crafted image is
            fed back through it. ``None`` when verification was skipped.
        target_emitted (bool | None): Whether ``model_response`` begins with
            ``target_text``. ``None`` when verification was skipped.
        generated_at (str): ISO 8601 UTC timestamp of when the row was produced.
        pyrit_version (str): PyRIT version that produced the row.
        schema_version (int): Manifest schema version. Currently ``3``.
        transfer_eval_results (list[dict[str, Any]] | None): Reserved for future
            cross-VLM transfer-attack evaluation results. ``None`` in v1.
    """

    id: str
    behavior_id: str
    behavior_text: str
    target_text: str
    image_path: str
    vlm_id: str
    variant: str
    num_steps_run: int
    final_loss: float
    epsilon: float
    step_size: float
    stop_loss: float
    succeeded_stop_criterion: bool
    deployed_loss: float | None = None
    seed_image_path: str = ""
    model_response: str | None = None
    target_emitted: bool | None = None
    generated_at: str = field(default_factory=_utc_now_iso)
    pyrit_version: str = field(default_factory=_pyrit_version)
    schema_version: int = SCHEMA_VERSION
    transfer_eval_results: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return this entry as a plain JSON-serializable dict."""
        return asdict(self)

    def to_json_line(self) -> str:
        """Return this entry as a single-line JSON string (no trailing newline)."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PGDManifestEntry:
        """
        Reconstruct an entry from a manifest dict, ignoring unknown keys.

        Args:
            data (dict[str, Any]): A manifest row.

        Returns:
            PGDManifestEntry: The reconstructed entry.

        Raises:
            ValueError: If a required field is missing.
        """
        known = {f for f in cls.__dataclass_fields__}  # noqa: C416 - explicit set for clarity
        filtered = {k: v for k, v in data.items() if k in known}
        try:
            return cls(**filtered)
        except TypeError as e:
            raise ValueError(f"PGDManifestEntry.from_dict: malformed manifest row: {e}") from e


def write_manifest(*, entries: list[PGDManifestEntry], path: str | Path) -> None:
    """
    Write ``entries`` to ``path`` as JSONL (overwriting any existing file).

    Args:
        entries (list[PGDManifestEntry]): The rows to write.
        path (str | Path): Destination JSONL path.
    """
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(entry.to_json_line())
            f.write("\n")


def append_manifest_entry(*, entry: PGDManifestEntry, path: str | Path) -> None:
    """
    Append a single ``entry`` to the JSONL manifest at ``path``.

    Args:
        entry (PGDManifestEntry): The row to append.
        path (str | Path): Destination JSONL path (created if absent).
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(entry.to_json_line())
        f.write("\n")


def read_manifest(*, path: str | Path) -> list[PGDManifestEntry]:
    """
    Read a JSONL manifest, skipping blank lines and warning on version drift.

    Args:
        path (str | Path): Path to a JSONL manifest.

    Returns:
        list[PGDManifestEntry]: The parsed rows.

    Raises:
        ValueError: If a non-blank line is not valid JSON or is a malformed row.
    """
    entries: list[PGDManifestEntry] = []
    with open(path, encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"read_manifest: line {line_number} is not valid JSON: {e}") from e
            version = data.get("schema_version")
            if version != SCHEMA_VERSION:
                logger.warning(
                    "read_manifest: line %d has schema_version %r (reader supports %d); reading best-effort.",
                    line_number,
                    version,
                    SCHEMA_VERSION,
                )
            entries.append(PGDManifestEntry.from_dict(data))
    return entries


__all__ = [
    "PGDManifestEntry",
    "SCHEMA_VERSION",
    "append_manifest_entry",
    "read_manifest",
    "write_manifest",
]
