# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from enum import Enum
from typing import Any

from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import (
    _RemoteDatasetLoader,
)
from pyrit.models import SeedDataset, SeedPrompt

logger = logging.getLogger(__name__)

HF_DATASET_NAME: str = "ScaleAI/fortress_public"
HF_REVISION: str = "0c096becbc75bb12065c8059a53960c7f0d4d35c"
HF_SOURCE_URL: str = f"https://huggingface.co/datasets/{HF_DATASET_NAME}"

# The risk_domain field in HF stores CBRNE's full descriptive name. Other two
# domains are stored verbatim as below. Inspect Evals applies the same expansion.
_CBRNE_FULL: str = "Chemical, Biological, Radiological, Nuclear, and Explosive (CBRNE)"


class FortressRiskDomain(Enum):
    """
    Filter for which Fortress risk domain to load.

    Values match the canonical short labels from the upstream paper. CBRNE is stored
    in the HF dataset under its long descriptive name and is expanded internally
    when filtering, so users always pass the short ``"CBRNE"`` form.
    """

    CBRNE = "CBRNE"
    POLITICAL_VIOLENCE_AND_TERRORISM = "Political Violence & Terrorism"
    CRIMINAL_AND_FINANCIAL_ILLICIT_ACTIVITIES = "Criminal & Financial Illicit Activities"


class FortressRiskSubdomain(Enum):
    """
    Filter for which Fortress risk subdomain to load.

    Values are the short title prefix used by the upstream dataset; the stored
    ``risk_subdomain`` field carries a longer descriptive suffix after a colon
    (e.g. ``"Illegal Weapons: 3d-printed weapons, ..."``). Filtering uses a
    prefix match against this short form so the descriptive suffix is preserved
    in metadata for downstream consumers.

    Note: the upstream README documents that some rows have a subdomain
    assignment that does not match the canonical domain mapping
    (e.g. a ``Terrorism`` subdomain row under the ``CBRNE`` domain). Filters
    respect the stored values, not the documented mapping, so such rows are
    surfaced exactly as upstream emits them.
    """

    CHEMICAL = "Chemical"
    BIOLOGICAL = "Biological"
    RADIOLOGICAL_NUCLEAR_WMD = "Radiological and Nuclear (WMD)"
    EXPLOSIVES = "Explosives"
    TERRORISM = "Terrorism"
    POLITICAL_VIOLENCE = "Political Violence"
    ILLEGAL_WEAPONS = "Illegal Weapons"
    COORDINATION_OF_ILLICIT_ACTIVITIES = "Coordination of Illicit Activities"
    FRAUD = "Fraud"
    PRIVACY_SCAMS = "Privacy/Scams"


_AUTHORS: list[str] = [
    "Christina Q. Knight",
    "Kaustubh Deshpande",
    "Ved Sirdeshmukh",
    "Meher Mankikar",
    "Scale Red Team",
    "SEAL Research Team",
    "Julian Michael",
]

_GROUPS: list[str] = ["Scale AI"]


class _FortressBaseDataset(_RemoteDatasetLoader):
    """
    Base loader for the FORTRESS (Frontier Risk Evaluation for National Security
    and Public Safety) dataset from Scale AI.

    FORTRESS is 500 expert-crafted adversarial prompts, each paired with a benign
    rephrasing on the same topic and an instance-specific 4-7 binary-question rubric.
    The dataset is designed for the paired (ARS, ORS) safety-vs-utility evaluation
    described in the paper. Three sibling loaders are exposed:

    - ``_FortressAdversarialDataset`` — the 500 adversarial prompts only (each row
      carries its rubric and ``num_dim`` in metadata so the per-row rubric scorer
      can grade against it).
    - ``_FortressBenignDataset`` — the 500 benign twin prompts only.
    - ``_FortressPairedDataset`` — both halves, with cross-link metadata.

    Use restriction: This dataset should be used for safety evaluations only, and
    it is prohibited to use these data for any adversarial training or research.

    References:
        - Paper: [@knight2025fortress] (arXiv:2506.14922).
        - HuggingFace: https://huggingface.co/datasets/ScaleAI/fortress_public
        - Project page: https://scale.com/research/fortress
        - Inspect Evals port (canonical execution harness):
          https://github.com/UKGovernmentBEIS/inspect_evals/tree/main/src/inspect_evals/fortress

    License: CC BY 4.0 with the Scale AI use restriction quoted above.
    """

    HF_DATASET_NAME: str = HF_DATASET_NAME
    HF_REVISION: str = HF_REVISION

    modalities: list[str] = ["text"]
    harm_categories: list[str] = [
        "biological",
        "chemical",
        "coordination of illicit activities",
        "explosives",
        "fraud",
        "illegal weapons",
        "political violence",
        "privacy/scams",
        "radiological and nuclear (wmd)",
        "terrorism",
    ]

    def __init__(
        self,
        *,
        risk_domain: FortressRiskDomain | None = None,
        risk_subdomain: FortressRiskSubdomain | None = None,
    ) -> None:
        """
        Initialize the Fortress dataset loader.

        Args:
            risk_domain (FortressRiskDomain | None): If set, only return rows whose
                ``risk_domain`` field matches this domain. Defaults to None (all domains).
            risk_subdomain (FortressRiskSubdomain | None): If set, only return rows whose
                ``risk_subdomain`` field starts with the given subdomain prefix.
                Defaults to None (all subdomains).

        Raises:
            ValueError: If ``risk_domain`` / ``risk_subdomain`` is not the expected enum type.
        """
        if risk_domain is not None:
            self._validate_enum(value=risk_domain, enum_cls=FortressRiskDomain, label="risk_domain")
        if risk_subdomain is not None:
            self._validate_enum(value=risk_subdomain, enum_cls=FortressRiskSubdomain, label="risk_subdomain")

        self.risk_domain = risk_domain
        self.risk_subdomain = risk_subdomain

    async def _fetch_filtered_rows(self, *, cache: bool) -> list[dict[str, Any]]:
        """
        Fetch the (single) Fortress train split from HF and apply enum filters.

        Args:
            cache (bool): Whether to cache the fetched dataset.

        Returns:
            list[dict[str, Any]]: The filtered raw HF rows in upstream order.

        Raises:
            ValueError: If no rows remain after filtering.
        """
        data = await self._fetch_from_huggingface(
            dataset_name=self.HF_DATASET_NAME,
            split="train",
            cache=cache,
            revision=self.HF_REVISION,
        )

        domain_filter = self._resolved_domain_filter_value()
        subdomain_prefix = self.risk_subdomain.value if self.risk_subdomain is not None else None

        filtered: list[dict[str, Any]] = []
        for row in data:
            if domain_filter is not None and row.get("risk_domain") != domain_filter:
                continue
            if subdomain_prefix is not None and not str(row.get("risk_subdomain", "")).startswith(subdomain_prefix):
                continue
            filtered.append(dict(row))

        if not filtered:
            domain_label = self.risk_domain.value if self.risk_domain is not None else "(any)"
            subdomain_label = self.risk_subdomain.value if self.risk_subdomain is not None else "(any)"
            raise ValueError(
                "SeedDataset cannot be empty. Check your filter criteria. "
                f"Fortress filter: risk_domain={domain_label!r}, risk_subdomain={subdomain_label!r}."
            )

        return filtered

    def _resolved_domain_filter_value(self) -> str | None:
        """
        Resolve the value used to filter the HF ``risk_domain`` column.

        CBRNE is stored under its long descriptive form in the HF data, so the
        short enum value must be expanded before comparison. The other two
        domains use their enum value verbatim.

        Returns:
            str | None: The string to match against ``row["risk_domain"]``, or
                None if no domain filter is set.
        """
        if self.risk_domain is None:
            return None
        if self.risk_domain == FortressRiskDomain.CBRNE:
            return _CBRNE_FULL
        return self.risk_domain.value

    def _build_seed_prompt(
        self,
        *,
        row: dict[str, Any],
        adversarial_or_benign: str,
        description: str,
    ) -> SeedPrompt:
        """
        Build a single SeedPrompt for either the adversarial or benign half of a row.

        Args:
            row (dict[str, Any]): A raw Fortress HF row.
            adversarial_or_benign (str): ``"adversarial"`` or ``"benign"``.
            description (str): Per-sibling description to attach to the SeedPrompt.

        Returns:
            SeedPrompt: The constructed seed prompt.

        Raises:
            ValueError: If ``adversarial_or_benign`` is not one of the two valid values.
        """
        if adversarial_or_benign not in ("adversarial", "benign"):
            raise ValueError(f"adversarial_or_benign must be 'adversarial' or 'benign'; got {adversarial_or_benign!r}.")

        is_adversarial = adversarial_or_benign == "adversarial"
        value = row["adversarial_prompt"] if is_adversarial else row["benign_prompt"]
        paired_value = row["benign_prompt"] if is_adversarial else row["adversarial_prompt"]
        subdomain_full = str(row.get("risk_subdomain", ""))
        subdomain_short = _shorten_risk_subdomain(subdomain_full)

        metadata: dict[str, str | int] = {
            "fortress_id": int(row["ID"]),
            "risk_domain": str(row.get("risk_domain", "")),
            "risk_subdomain": subdomain_short,
            "risk_subdomain_full": subdomain_full,
            "adversarial_or_benign": adversarial_or_benign,
            "paired_prompt": str(paired_value),
            "use_restriction": "no_adversarial_training",
        }

        if is_adversarial:
            rubric_list = list(row.get("rubric") or [])
            metadata["rubric"] = "\n".join(str(c) for c in rubric_list)
            metadata["num_dim"] = len(rubric_list)

        return SeedPrompt(
            value=str(value),
            data_type="text",
            dataset_name=self.dataset_name,
            harm_categories=[subdomain_short] if subdomain_short else [],
            description=description,
            source=HF_SOURCE_URL,
            authors=_AUTHORS,
            groups=_GROUPS,
            metadata=metadata,
        )


class _FortressAdversarialDataset(_FortressBaseDataset):
    """
    Loader for the adversarial half of FORTRESS (500 expert-crafted prompts).

    Each row carries the per-prompt 4-7-question rubric in
    ``metadata["rubric"]`` (newline-separated) and ``metadata["num_dim"]``
    so ``FortressRubricScorer`` can grade responses against the prompt's own rubric.

    Use restriction: This dataset should be used for safety evaluations only, and
    it is prohibited to use these data for any adversarial training or research.

    Reference: [@knight2025fortress]
    """

    size: str = "medium"
    tags: set[str] = {"safety", "jailbreak", "calibration", "national_security"}

    _DESCRIPTION: str = (
        "Adversarial half of FORTRESS: 500 expert-crafted national-security prompts "
        "spanning CBRNE, Political Violence & Terrorism, and Criminal & Financial "
        "Illicit Activities. Each prompt carries a per-row 4-7-question binary rubric "
        "in metadata for use with the FortressRubricScorer."
    )

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "fortress_adversarial"

    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch the adversarial Fortress prompts as a SeedDataset.

        Args:
            cache (bool): Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: The adversarial prompts, optionally filtered by domain/subdomain.

        Raises:
            ValueError: If no rows remain after filtering.
        """
        logger.info(
            f"Loading Fortress adversarial dataset (risk_domain={self.risk_domain}, "
            f"risk_subdomain={self.risk_subdomain})"
        )

        rows = await self._fetch_filtered_rows(cache=cache)
        seeds = [
            self._build_seed_prompt(row=row, adversarial_or_benign="adversarial", description=self._DESCRIPTION)
            for row in rows
        ]

        logger.info(f"Successfully loaded {len(seeds)} adversarial prompts from Fortress dataset")
        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)


class _FortressBenignDataset(_FortressBaseDataset):
    """
    Loader for the benign half of FORTRESS (500 prompts paired with the adversarial split).

    Each benign prompt is a same-topic rephrasing of an adversarial prompt, used to
    measure over-refusal. The paired adversarial prompt text is preserved in
    ``metadata["paired_prompt"]`` for downstream consumers that want to compute
    the paired (ARS, ORS) metric.

    Use restriction: This dataset should be used for safety evaluations only, and
    it is prohibited to use these data for any adversarial training or research.

    Reference: [@knight2025fortress]
    """

    size: str = "medium"
    tags: set[str] = {"safety", "over_refusal", "calibration", "national_security"}

    _DESCRIPTION: str = (
        "Benign half of FORTRESS: 500 same-topic rephrasings of the adversarial prompts, "
        "used to measure over-refusal. The paired adversarial prompt is preserved in "
        "SeedPrompt metadata."
    )

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "fortress_benign"

    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch the benign Fortress prompts as a SeedDataset.

        Args:
            cache (bool): Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: The benign prompts, optionally filtered by domain/subdomain.

        Raises:
            ValueError: If no rows remain after filtering.
        """
        logger.info(
            f"Loading Fortress benign dataset (risk_domain={self.risk_domain}, risk_subdomain={self.risk_subdomain})"
        )

        rows = await self._fetch_filtered_rows(cache=cache)
        seeds = [
            self._build_seed_prompt(row=row, adversarial_or_benign="benign", description=self._DESCRIPTION)
            for row in rows
        ]

        logger.info(f"Successfully loaded {len(seeds)} benign prompts from Fortress dataset")
        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)


class _FortressPairedDataset(_FortressBaseDataset):
    """
    Loader for the full FORTRESS dataset with both adversarial and benign prompts.

    For each upstream row, two SeedPrompts are emitted (adversarial first, then benign)
    sharing a ``fortress_id`` in metadata. Adversarial prompts carry the per-row rubric
    in ``metadata["rubric"]`` / ``metadata["num_dim"]``; both halves carry the partner
    prompt's text in ``metadata["paired_prompt"]``. This is the loader to use when the
    consumer wants both halves available in a single SeedDataset.

    Use restriction: This dataset should be used for safety evaluations only, and
    it is prohibited to use these data for any adversarial training or research.

    Reference: [@knight2025fortress]
    """

    size: str = "large"
    tags: set[str] = {"safety", "jailbreak", "over_refusal", "calibration", "national_security"}

    _DESCRIPTION: str = (
        "Paired FORTRESS dataset: each adversarial prompt is followed by its benign "
        "twin, sharing a fortress_id in metadata. Adversarial prompts carry per-row "
        "rubric metadata; both halves carry the partner prompt for downstream "
        "(ARS, ORS) trade-off analysis."
    )

    @property
    def dataset_name(self) -> str:
        """Return the dataset name."""
        return "fortress_paired"

    async def fetch_dataset_async(self, *, cache: bool = True) -> SeedDataset:
        """
        Fetch both adversarial and benign Fortress prompts as a single SeedDataset.

        Each upstream row yields two SeedPrompts (adversarial then benign) sharing
        a ``fortress_id`` in metadata.

        Args:
            cache (bool): Whether to cache the fetched dataset. Defaults to True.

        Returns:
            SeedDataset: All adversarial + benign prompts after any domain/subdomain filtering.

        Raises:
            ValueError: If no rows remain after filtering.
        """
        logger.info(
            f"Loading Fortress paired dataset (risk_domain={self.risk_domain}, risk_subdomain={self.risk_subdomain})"
        )

        rows = await self._fetch_filtered_rows(cache=cache)
        seeds: list[SeedPrompt] = []
        for row in rows:
            seeds.append(
                self._build_seed_prompt(row=row, adversarial_or_benign="adversarial", description=self._DESCRIPTION)
            )
            seeds.append(
                self._build_seed_prompt(row=row, adversarial_or_benign="benign", description=self._DESCRIPTION)
            )

        logger.info(f"Successfully loaded {len(seeds)} prompts ({len(seeds) // 2} paired rows) from Fortress dataset")
        return SeedDataset(seeds=seeds, dataset_name=self.dataset_name)


def _shorten_risk_subdomain(long_subdomain: str) -> str:
    """
    Shorten the stored ``risk_subdomain`` to its canonical title prefix.

    HF stores the subdomain as ``"<Title>: <descriptive suffix>"``. We match against
    the known short prefixes from :class:`FortressRiskSubdomain` and return the
    matching short form. If no match (unexpected upstream value), the raw value
    before the first colon is returned as a best-effort fallback.

    Args:
        long_subdomain (str): The stored ``risk_subdomain`` value.

    Returns:
        str: The short title form (e.g. ``"Illegal Weapons"``), or an empty string
            when the input is empty.
    """
    if not long_subdomain:
        return ""
    for subdomain in FortressRiskSubdomain:
        if long_subdomain.startswith(subdomain.value):
            return subdomain.value
    # Best-effort fallback for unexpected upstream values.
    return long_subdomain.split(":", 1)[0].strip()
