# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pyrit.scenario.capability_suite.expansion import expand_suite
from pyrit.scenario.capability_suite.manifest import (
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CaseMessageManifest,
    LocalSandboxProviderManifestConfig,
    RunPolicyManifest,
    SuiteProvenance,
)


def _case(case_id: str) -> CapabilityCaseManifest:
    return CapabilityCaseManifest(
        case_id=case_id,
        objective="finish",
        messages=(CaseMessageManifest(role="user", content="hi"),),
    )


def _manifest(*, epochs: int, attempts: int = 1) -> CapabilitySuiteManifest:
    return CapabilitySuiteManifest(
        suite_id="suite-1",
        name="Example suite",
        provenance=SuiteProvenance(source="unit-test"),
        sandbox_provider=LocalSandboxProviderManifestConfig(),
        run_policy=RunPolicyManifest(epochs=epochs, attempts=attempts),
        cases=(_case("case-a"), _case("case-b")),
    )


def test_expand_suite_produces_cases_times_epochs_times_attempts_units() -> None:
    manifest = _manifest(epochs=2, attempts=3)
    units = expand_suite(manifest)
    assert len(units) == 12


def test_expand_suite_orders_by_case_then_epoch_then_repetition() -> None:
    manifest = _manifest(epochs=2, attempts=2)
    units = expand_suite(manifest)
    assert [(unit.case.case_id, unit.epoch, unit.repetition) for unit in units] == [
        ("case-a", 1, 1),
        ("case-a", 1, 2),
        ("case-a", 2, 1),
        ("case-a", 2, 2),
        ("case-b", 1, 1),
        ("case-b", 1, 2),
        ("case-b", 2, 1),
        ("case-b", 2, 2),
    ]


def test_expand_suite_unit_keys_are_unique_and_deterministic() -> None:
    manifest = _manifest(epochs=2)
    first = expand_suite(manifest)
    second = expand_suite(manifest)
    assert [unit.unit_key for unit in first] == [unit.unit_key for unit in second]
    assert len({unit.unit_key for unit in first}) == len(first)
    assert first[0].unit_key == "suite-1:case-a:epoch1:run1"


def test_expand_suite_single_epoch_default() -> None:
    manifest = _manifest(epochs=1)
    units = expand_suite(manifest)
    assert len(units) == 2
    assert all(unit.epoch == 1 for unit in units)
