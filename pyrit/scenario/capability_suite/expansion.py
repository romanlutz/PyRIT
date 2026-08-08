# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Deterministic case-by-epoch expansion for capability-suite runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyrit.scenario.capability_suite.manifest import CapabilityCaseManifest, CapabilitySuiteManifest


@dataclass(frozen=True)
class CaseRunUnit:
    """One deterministic unit of scheduled work: one case/epoch/repetition."""

    unit_key: str
    case: CapabilityCaseManifest
    epoch: int
    repetition: int


def expand_suite(manifest: CapabilitySuiteManifest) -> tuple[CaseRunUnit, ...]:
    """
    Deterministically expand every case into epochs and independent attempts.

    Retries are a runtime recovery mechanism and are intentionally not part of this
    pure expansion. Each expanded repetition is an independently measured run.

    Returns:
        tuple[CaseRunUnit, ...]: Units in stable case, epoch, repetition order.
    """
    units: list[CaseRunUnit] = []
    for case in manifest.cases:
        for epoch in range(1, manifest.run_policy.epochs + 1):
            units.extend(
                CaseRunUnit(
                    unit_key=f"{manifest.suite_id}:{case.case_id}:epoch{epoch}:run{repetition}",
                    case=case,
                    epoch=epoch,
                    repetition=repetition,
                )
                for repetition in range(1, manifest.run_policy.attempts + 1)
            )
    return tuple(units)
