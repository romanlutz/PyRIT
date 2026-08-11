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
from pyrit.compat.inspect_ai.static_mapping_audit import reviewed_static_mapping_audit
from pyrit.scenario.capability_suite.inspect_evals import (
    FidelityClassification,
    InspectEvalFamilyReport,
    InspectEvalTaskFactory,
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
    "bbh": ("bbh",),
    "bbq": ("bbq",),
    "boolq": ("boolq",),
    "commonsense_qa": ("commonsense_qa",),
    "gdm_in_house_ctf": ("gdm_in_house_ctf",),
    "gdm_intercode_ctf": ("gdm_intercode_ctf",),
    "hellaswag": ("hellaswag",),
    "humaneval": ("humaneval",),
    "mbpp": ("mbpp",),
    "musr": ("musr",),
    "onet": ("onet_m6",),
    "paws": ("paws",),
    "pre_flight": ("pre_flight",),
    "pubmedqa": ("pubmedqa",),
    "race_h": ("race_h",),
    "sec_qa": ("sec_qa_v1", "sec_qa_v1_5_shot", "sec_qa_v2", "sec_qa_v2_5_shot"),
    "wmdp": ("wmdp_bio", "wmdp_chem", "wmdp_cyber"),
}
_CYBERMETRIC_BLOCKER = (
    "The exact commit and SHA256 are pinned, but the upstream dataset declares no license. Supply only locally "
    "authorized exact bytes; the native factory remains partial until dataset reuse terms are established."
)
_VSTAR_BLOCKER = (
    "The exact Hugging Face revision has no dataset or image license/provenance statement. Native ordered "
    "text/image compilation requires an authorized per-file-checksummed staged snapshot and an image-capable "
    "target."
)
_FACTORY_BLOCKERS = {
    ("apps", "apps"): (
        (
            "The default and pass_at_k reducer variants compile to the native Python code-evaluation scorer, but the "
            "unchanged factory also exposes median/mode epoch reducers that are not yet represented by the shared "
            "typed reducer framework. Those variants fail during compilation before model or sandbox execution."
        ),
    ),
    ("bigcodebench", "bigcodebench"): (
        (
            "The pinned factory selects a mutable :latest execution image with an incompletely pinned dependency "
            "closure (including TensorFlow). Provide a content-addressed prebuilt image with a verified dependency "
            "lock before native execution can be enabled."
        ),
    ),
    ("class_eval", "class_eval"): (
        (
            "The exact dataset revision is CC-BY-NC-4.0 and the pinned factory selects a mutable :latest image with "
            "unpinned dependencies. Supply authorized exact-revision records and a content-addressed locked runtime."
        ),
    ),
    ("cybermetric", "cybermetric_80"): (_CYBERMETRIC_BLOCKER,),
    ("cybermetric", "cybermetric_500"): (_CYBERMETRIC_BLOCKER,),
    ("cybermetric", "cybermetric_2000"): (_CYBERMETRIC_BLOCKER,),
    ("cybermetric", "cybermetric_10000"): (_CYBERMETRIC_BLOCKER,),
    ("gpqa", "gpqa_diamond"): (
        (
            "The CSV is SHA256-pinned, but license coverage for the exact OpenAI mirror is not established and the "
            "pinned factory shuffles choices without a seed. Use an authorized verified cache; full reproducibility "
            "and source trust remain blocked."
        ),
    ),
    ("medqa", "medqa"): (
        (
            "The pinned bigbio/med_qa source declares its dataset license as UNKNOWN. The unchanged factory compiles "
            "and runs with content-verified cached records, but full acquisition support requires the user to confirm "
            "upstream terms and pass locally authorized records with --data."
        ),
    ),
    ("piqa", "piqa"): (
        (
            "The pinned dataset builder resolves floating external Google Storage assets without source-declared "
            "content hashes. Cache content-verified PIQA records and pass them with --data; full source acquisition "
            "remains unsupported."
        ),
    ),
    ("usaco", "usaco"): (
        (
            "The source pins the Google Drive ZIP SHA256 but declares no dataset license. Native file/stdin mechanics "
            "also require an authorized exact-byte cache and a content-addressed runtime that enforces the requested "
            "Linux resource limits."
        ),
    ),
    ("vstar_bench", "vstar_bench_attribute_recognition"): (_VSTAR_BLOCKER,),
    ("vstar_bench", "vstar_bench_spatial_relationship_reasoning"): (_VSTAR_BLOCKER,),
    ("vimgolf_challenges", "vimgolf_single_turn"): (
        (
            "The exact Hugging Face revision declares no dataset license, and the verifier depends on Vim plus mutable "
            "base/apt layers. Supply authorized content-verified rows and a content-addressed Vim runtime before "
            "enabling native editor-state verification."
        ),
    ),
    ("winogrande", "winogrande"): (
        (
            "The exact Hugging Face revision labels licensing as 'More Information Needed', and optional evaluation "
            "shuffle=True is unseeded. Supply authorized exact-revision records; full source trust remains blocked."
        ),
    ),
}
_STATIC_MAPPING_FAMILIES = frozenset(
    {
        "bbh",
        "bbq",
        "apps",
        "bigcodebench",
        "boolq",
        "class_eval",
        "commonsense_qa",
        "cybermetric",
        "gpqa",
        "hellaswag",
        "humaneval",
        "mbpp",
        "medqa",
        "musr",
        "onet",
        "paws",
        "piqa",
        "pre_flight",
        "pubmedqa",
        "race_h",
        "sec_qa",
        "usaco",
        "vstar_bench",
        "vimgolf_challenges",
        "winogrande",
        "wmdp",
    }
)
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
            tasks = ", ".join(
                str(task["name"]) for task in family.task_factories if task["compatibility_status"] == "supported"
            )
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
        compatibility_claims=tuple(
            _task_claim(source_directory=family.source_directory, task=task)
            for family in families
            for task in family.task_factories
            if task["compatibility_status"] == "supported"
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
        family.family: tuple(
            sorted(str(task["name"]) for task in family.task_factories if task["compatibility_status"] == "supported")
        )
        for family in report.families
        if any(task["compatibility_status"] == "supported" for task in family.task_factories)
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
    task_factories = tuple(_task_factory_record(report=report, task=task) for task in report.tasks)
    status = _compatibility_status(report=report, task_factories=task_factories)
    reviewed_status = (
        report.family in _STATIC_MAPPING_FAMILIES
        and status in {"supported", "partial"}
        and any(str(task["compatibility_status"]) in {"supported", "partial"} for task in task_factories)
    )
    providers = set(report.sandboxes)
    providers.update("docker" for item in report.executable_setup if "compose" in item or "container" in item)
    blockers = [] if reviewed_status else list(report.portability_blockers)
    blockers.extend(_family_cloud_blockers(report=report))
    blockers.extend(blocker for task in task_factories for blocker in _factory_blockers(task))
    return InspectCatalogFamily(
        family=report.family,
        source_directory=report.source_directory,
        task_factories=task_factories,
        compatibility_status=status,
        fidelity=(
            FidelityClassification.NATIVE.value
            if status == "supported"
            else FidelityClassification.PARTIAL.value
            if reviewed_status
            else report.fidelity.value
        ),
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


def _compatibility_status(
    *,
    report: InspectEvalFamilyReport,
    task_factories: tuple[dict[str, object], ...],
) -> str:
    statuses = {str(task["compatibility_status"]) for task in task_factories}
    if statuses == {"supported"}:
        return "supported"
    if "supported" in statuses or "partial" in statuses:
        return "partial"
    if report.fidelity is FidelityClassification.PARTIAL:
        return "partial"
    return "unsupported"


def _task_factory_record(*, report: InspectEvalFamilyReport, task: InspectEvalTaskFactory) -> dict[str, object]:
    record = asdict(task)
    name = str(record["name"])
    audit = reviewed_static_mapping_audit(family=report.family, factory=name)
    blockers = _FACTORY_BLOCKERS.get((report.family, name), ())
    if name in _SUPPORTED_TASKS.get(report.family, ()):
        status = "supported"
    elif blockers:
        status = "partial"
    else:
        status = "unsupported"
        blockers = tuple(report.portability_blockers) or (
            "This unchanged factory has not been validated through the pinned native compatibility compiler.",
        )
    record.update(
        {
            "compatibility_status": status,
            "blockers": blockers,
            "dataset_policy": (
                "Pinned public source revision; remote dataset code disabled; cacheable and offline-safe after "
                "content acquisition."
                if status == "supported"
                else None
            ),
            **({"reviewed_static_mapping_audit": audit} if audit is not None else {}),
        }
    )
    return record


def _factory_blockers(task: dict[str, object]) -> tuple[str, ...]:
    blockers = task.get("blockers")
    if not isinstance(blockers, tuple) or not all(isinstance(blocker, str) for blocker in blockers):
        return ()
    return tuple(str(blocker) for blocker in blockers)


def _task_claim(*, source_directory: str, task: dict[str, object]) -> str:
    source_file = str(task["source_file"])
    marker = "/inspect_evals/"
    relative = source_file.split(marker, maxsplit=1)[-1] if marker in source_file else source_file
    if relative == source_directory:
        relative = relative.rsplit("/", maxsplit=1)[-1]
    return f"{relative}@{task['name']}"


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
