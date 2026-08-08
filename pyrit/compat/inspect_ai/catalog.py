# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Catalog-wide static compatibility diagnostics for pinned Inspect-evals source."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from pyrit.compat.inspect_ai.inventory import inventory_inspect_api_usage
from pyrit.compat.inspect_ai.profile import PINNED_INSPECT_EVALS_PROFILE
from pyrit.compat.inspect_ai.source import validate_inspect_source
from pyrit.scenario.capability_suite.inspect_evals import (
    FidelityClassification,
    InspectEvalFamilyReport,
    analyze_inspect_evals_source_tree,
)

if TYPE_CHECKING:
    from pathlib import Path

_GOLDEN_API_SYMBOLS_SHA256 = "4515d2aba8bedf78de2c0ee866f44de6167ab92aa4eccf9d8fc321b2e789cd34"
_GOLDEN_API_SYMBOL_COUNT = 262
_GOLDEN_FAMILY_COUNT = 129
_GOLDEN_TASK_COUNT = 249
_SUPPORTED_TASKS = {
    "arc": ("arc_challenge", "arc_easy"),
    "gdm_in_house_ctf": ("gdm_in_house_ctf",),
    "gdm_intercode_ctf": ("gdm_intercode_ctf",),
}
_CLOUD_SURFACES = {
    "aws": ("aws", "amazon web services"),
    "bedrock": ("bedrock",),
    "sagemaker": ("sagemaker",),
    "ec2": ("ec2",),
    "gcp": ("gcp", "google cloud", "google_cloud"),
    "modal": ("modal",),
    "daytona": ("daytona",),
    "kubernetes": ("kubernetes", "k8s"),
}


class InspectCatalogRegressionError(RuntimeError):
    """Raised when the pinned catalog no longer matches reviewed compatibility metadata."""


@dataclass(frozen=True)
class InspectCloudSurfaceStatus:
    """One explicitly classified external cloud surface."""

    surface: str
    status: str
    found_in_families: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class InspectCatalogFamily:
    """Stable static compatibility diagnostics for one task family."""

    family: str
    source_directory: str
    task_factories: tuple[dict[str, object], ...]
    compatibility_status: str
    fidelity: str
    inspect_api_symbols: tuple[str, ...]
    unsupported_inspect_api_symbols: tuple[str, ...]
    external_data: tuple[str, ...]
    assets: tuple[str, ...]
    required_providers: tuple[str, ...]
    blockers: tuple[str, ...]
    profile_coverage: dict[str, int | str]


@dataclass(frozen=True)
class InspectCatalogReport:
    """Stable catalog-wide inventory suitable for users and CI."""

    profile_id: str
    inspect_evals_revision: str
    source_root: str
    source_revision_verified: bool
    api_symbols_sha256: str
    api_symbol_count: int
    task_factory_count: int
    families: tuple[InspectCatalogFamily, ...]
    excluded_cloud_surfaces: tuple[InspectCloudSurfaceStatus, ...]
    compatibility_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return deterministic JSON-compatible catalog data."""
        return {
            "profile_id": self.profile_id,
            "inspect_evals_revision": self.inspect_evals_revision,
            "source_root": self.source_root,
            "source_revision_verified": self.source_revision_verified,
            "api_symbols_sha256": self.api_symbols_sha256,
            "api_symbol_count": self.api_symbol_count,
            "task_factory_count": self.task_factory_count,
            "families": [asdict(family) for family in self.families],
            "excluded_cloud_surfaces": [asdict(surface) for surface in self.excluded_cloud_surfaces],
            "compatibility_claims": list(self.compatibility_claims),
        }

    def to_text(self) -> str:
        """
        Render a concise, deterministic human-readable catalog report.

        Returns:
            str: Human-readable catalog summary.
        """
        supported = [family for family in self.families if family.compatibility_status == "supported"]
        partial = [family for family in self.families if family.compatibility_status == "partial"]
        unsupported = [family for family in self.families if family.compatibility_status == "unsupported"]
        lines = [
            f"Inspect-evals revision: {self.inspect_evals_revision}",
            f"Compatibility profile: {self.profile_id}",
            f"Source revision verified: {str(self.source_revision_verified).lower()}",
            (
                f"Families: {len(self.families)} (supported={len(supported)}, partial={len(partial)}, "
                f"unsupported={len(unsupported)})"
            ),
            f"Task factories: {self.task_factory_count}",
            f"Referenced inspect_ai APIs: {self.api_symbol_count} ({self.api_symbols_sha256})",
            "",
            "Supported unchanged tasks:",
        ]
        for family in supported:
            tasks = ", ".join(str(task["name"]) for task in family.task_factories)
            lines.append(f"  {family.family}: {tasks}")
        lines.extend(("", "Excluded cloud surfaces:"))
        lines.extend(f"  {surface.surface}: {surface.status}" for surface in self.excluded_cloud_surfaces)
        return "\n".join(lines)


def build_inspect_catalog(
    *,
    source_root: Path,
    verify_source: bool = True,
) -> InspectCatalogReport:
    """
    Scan every pinned task module without importing or executing upstream source.

    Returns:
        InspectCatalogReport: Stable family, API, provider, asset, and blocker diagnostics.
    """
    root = source_root.resolve()
    source_verified = False
    if verify_source:
        validate_inspect_source(source_root=root)
        source_verified = True
    source_report = analyze_inspect_evals_source_tree(source_root=root)
    inventory = inventory_inspect_api_usage(source_root=root, profile=PINNED_INSPECT_EVALS_PROFILE)
    symbols = tuple(sorted({usage.symbol for usage in inventory.usages}))
    digest = _symbols_digest(symbols)
    usages_by_file: dict[str, set[str]] = {}
    for usage in inventory.usages:
        usages_by_file.setdefault(usage.source_file, set()).add(usage.symbol)
    families = tuple(
        _catalog_family(
            report=family,
            usages_by_file=usages_by_file,
        )
        for family in source_report.families
    )
    cloud_surfaces = _cloud_surface_statuses(families=families)
    return InspectCatalogReport(
        profile_id=PINNED_INSPECT_EVALS_PROFILE.profile_id,
        inspect_evals_revision=PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision,
        source_root=str(root),
        source_revision_verified=source_verified and source_report.revision_verified,
        api_symbols_sha256=digest,
        api_symbol_count=len(symbols),
        task_factory_count=sum(len(family.task_factories) for family in families),
        families=families,
        excluded_cloud_surfaces=cloud_surfaces,
        compatibility_claims=(
            "arc/arc.py@arc_easy",
            "arc/arc.py@arc_challenge",
            "gdm_intercode_ctf/gdm_intercode_ctf.py@gdm_intercode_ctf",
            "gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf",
        ),
    )


def check_inspect_catalog_regression(*, report: InspectCatalogReport) -> None:
    """
    Fail if pinned APIs or reviewed supported task claims change.

    Raises:
        InspectCatalogRegressionError: If the golden inventory or supported tasks regress.
    """
    errors: list[str] = []
    if report.inspect_evals_revision != PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision:
        errors.append(f"unexpected revision {report.inspect_evals_revision}")
    if report.api_symbols_sha256 != _GOLDEN_API_SYMBOLS_SHA256:
        errors.append(
            "Inspect API inventory changed; classify every added/removed API before updating the golden digest "
            f"(expected {_GOLDEN_API_SYMBOLS_SHA256}, got {report.api_symbols_sha256})"
        )
    if report.api_symbol_count != _GOLDEN_API_SYMBOL_COUNT:
        errors.append(f"expected {_GOLDEN_API_SYMBOL_COUNT} API symbols, got {report.api_symbol_count}")
    if len(report.families) != _GOLDEN_FAMILY_COUNT:
        errors.append(f"expected {_GOLDEN_FAMILY_COUNT} families, got {len(report.families)}")
    if report.task_factory_count != _GOLDEN_TASK_COUNT:
        errors.append(f"expected {_GOLDEN_TASK_COUNT} task factories, got {report.task_factory_count}")
    observed_supported = {
        family.family: tuple(sorted(str(task["name"]) for task in family.task_factories))
        for family in report.families
        if family.compatibility_status == "supported"
    }
    if observed_supported != _SUPPORTED_TASKS:
        errors.append(f"supported task claims changed: expected {_SUPPORTED_TASKS}, got {observed_supported}")
    if errors:
        raise InspectCatalogRegressionError("; ".join(errors))


def _catalog_family(
    *,
    report: InspectEvalFamilyReport,
    usages_by_file: dict[str, set[str]],
) -> InspectCatalogFamily:
    symbols = tuple(
        sorted(
            {
                symbol
                for source_file, file_symbols in usages_by_file.items()
                if source_file == report.source_directory or source_file.startswith(f"{report.source_directory}/")
                for symbol in file_symbols
            }
        )
    )
    supported = PINNED_INSPECT_EVALS_PROFILE.supported_symbols
    unsupported = tuple(symbol for symbol in symbols if symbol not in supported)
    status = _compatibility_status(report)
    providers = set(report.sandboxes)
    providers.update("docker" for item in report.executable_setup if "compose" in item or "container" in item)
    blockers = list(report.portability_blockers)
    blockers.extend(_family_cloud_blockers(report=report))
    return InspectCatalogFamily(
        family=report.family,
        source_directory=report.source_directory,
        task_factories=tuple(asdict(task) for task in report.tasks),
        compatibility_status=status,
        fidelity=report.fidelity.value,
        inspect_api_symbols=symbols,
        unsupported_inspect_api_symbols=unsupported,
        external_data=report.datasets,
        assets=report.assets,
        required_providers=tuple(sorted(providers)),
        blockers=tuple(dict.fromkeys(blockers)),
        profile_coverage={
            "profile_id": PINNED_INSPECT_EVALS_PROFILE.profile_id,
            "referenced_symbols": len(symbols),
            "supported_symbols": len(symbols) - len(unsupported),
            "unsupported_symbols": len(unsupported),
        },
    )


def _compatibility_status(report: InspectEvalFamilyReport) -> str:
    if report.family in _SUPPORTED_TASKS:
        return "supported"
    if report.fidelity is FidelityClassification.PARTIAL:
        return "partial"
    return "unsupported"


def _family_cloud_blockers(*, report: InspectEvalFamilyReport) -> tuple[str, ...]:
    serialized = json.dumps(report.to_dict(), sort_keys=True).lower()
    blockers = []
    for surface, patterns in _CLOUD_SURFACES.items():
        if any(_contains_classification_term(text=serialized, term=pattern) for pattern in patterns):
            blockers.append(f"{surface} provider/runtime surfaces are excluded from this compatibility profile.")
    return tuple(blockers)


def _cloud_surface_statuses(
    *,
    families: tuple[InspectCatalogFamily, ...],
) -> tuple[InspectCloudSurfaceStatus, ...]:
    statuses = []
    for surface in _CLOUD_SURFACES:
        found = tuple(
            family.family
            for family in families
            if any(blocker.startswith(f"{surface} provider/") for blocker in family.blockers)
        )
        statuses.append(
            InspectCloudSurfaceStatus(
                surface=surface,
                status="excluded",
                found_in_families=found,
                reason="Non-Azure cloud provider/runtime integration is outside the pinned compatibility profile.",
            )
        )
    statuses.append(
        InspectCloudSurfaceStatus(
            surface="other_non_azure_clouds",
            status="excluded",
            found_in_families=(),
            reason="No unlisted non-Azure cloud provider is implicitly supported.",
        )
    )
    return tuple(statuses)


def _symbols_digest(symbols: tuple[str, ...]) -> str:
    payload = ("\n".join(symbols) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _contains_classification_term(*, text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None
