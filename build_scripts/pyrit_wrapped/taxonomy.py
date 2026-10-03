# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import re
from collections import Counter
from pathlib import PurePosixPath

from build_scripts.pyrit_wrapped.models import Classification, ItemKind, TaxonomyConfig, WorkItem


class Taxonomy:
    _GENERATED = {"uv.lock", "package-lock.json", "yarn.lock", "poetry.lock", "pnpm-lock.yaml"}
    _LANGUAGES = {".py": "Python", ".ts": "TypeScript", ".tsx": "TypeScript", ".js": "JavaScript", ".jsx": "JavaScript"}
    _DOC_ALIASES = {"targets": "prompt_target", "converters": "converter", "scoring": "score", "scenarios": "scenario"}
    _INTENT_PREFIX = re.compile(r"^\[?([A-Za-z]+)\]?(?:\([^)]*\))?!?(?::|\s|$)")

    def __init__(self, config: TaxonomyConfig) -> None:
        self.config = config

    def classify(self, item: WorkItem) -> Classification:
        intent, intent_evidence = self._intent(item)
        if item.kind == ItemKind.ISSUE or not item.files_complete or not item.paths:
            return self._metadata_classification(item=item, intent=intent, evidence=intent_evidence)
        files = [self._classify_path(path) for path in item.paths]
        substantive = [entry for entry in files if entry[2] != "Generated/lock files"] or files
        return Classification(
            topics=sorted({entry[0] for entry in substantive}),
            primary_topic=self._primary([entry[0] for entry in substantive]),
            surface=self._primary([entry[1] for entry in substantive]),
            artifacts=sorted({entry[2] for entry in files}),
            primary_artifact=self._primary([entry[2] for entry in substantive]),
            languages=sorted({entry[3] for entry in files if entry[3]}),
            intent=intent,
            evidence=intent_evidence + [f"path: {path}" for path in sorted(item.paths)],
        )

    def _intent(self, item: WorkItem) -> tuple[str, list[str]]:
        match = self._INTENT_PREFIX.match(item.title)
        if match and match[1].upper() in self.config.intent_aliases:
            return self.config.intent_aliases[match[1].upper()], [f"title prefix: {match[1]}"]
        labels = {label.lower() for label in item.labels}
        intents = {self.config.intent_labels[label] for label in labels if label in self.config.intent_labels}
        return (next(iter(intents)) if len(intents) == 1 else "Unknown"), [
            f"label: {label}" for label in sorted(labels)
        ]

    def _metadata_classification(self, *, item: WorkItem, intent: str, evidence: list[str]) -> Classification:
        topics = {
            self.config.topic_labels[label.lower()]
            for label in item.labels
            if label.lower() in self.config.topic_labels
        }
        for topic, keywords in self.config.topic_keywords.items():
            if any(re.search(rf"\b{re.escape(word)}\b", item.title, re.IGNORECASE) for word in keywords):
                topics.add(topic)
                evidence = [*evidence, f"inferred title topic: {topic}"]
        return Classification(
            topics=sorted(topics) or ["Unknown"],
            primary_topic=next(iter(topics)) if len(topics) == 1 else ("Mixed" if topics else "Unknown"),
            surface="Unknown",
            artifacts=["Unknown"],
            primary_artifact="Unknown",
            languages=[],
            intent=intent,
            evidence=[*evidence, *[f"topic label: {label}" for label in sorted(item.labels)]],
            inferred=True,
        )

    def _classify_path(self, path: str) -> tuple[str, str, str, str]:
        normalized = path.replace("\\", "/").lower()
        artifact = self._artifact(normalized)
        logical = self._logical_path(normalized)
        rule = next((rule for rule in self.config.path_rules if logical.startswith(rule.prefix.lower())), None)
        topic, surface = (rule.topic, rule.surface) if rule else ("Unknown", "Unknown")
        if rule is None and artifact == "Documentation/examples":
            topic, surface = "Documentation", "Documentation"
        if rule is None and artifact in {"Configuration/CI", "Generated/lock files"}:
            topic, surface = "Build and CI", "Tooling and infrastructure"
        return topic, surface, artifact, self._LANGUAGES.get(PurePosixPath(normalized).suffix, "")

    def _artifact(self, path: str) -> str:
        parsed = PurePosixPath(path)
        if parsed.name in self._GENERATED or "generated" in parsed.parts or "__pycache__" in parsed.parts:
            return "Generated/lock files"
        if path.startswith("doc/"):
            return "Documentation/examples"
        if path.startswith("tests/") or any(part in {"__tests__", "e2e"} for part in parsed.parts):
            return "Tests"
        if "seed_datasets" in parsed.parts and parsed.suffix != ".py":
            return "Dataset content"
        if parsed.suffix in {".md", ".rst", ".ipynb"}:
            return "Documentation/examples"
        if re.search(r"\.(?:test|spec)\.[^/]+$", path) or parsed.name.startswith("test_"):
            return "Tests"
        if path.startswith((".github/", ".azuredevops/", "infra/", "docker/")):
            return "Configuration/CI"
        if parsed.suffix in {".yaml", ".yml", ".toml", ".ini", ".cfg", ".json"} or parsed.name == "makefile":
            return "Configuration/CI"
        if parsed.suffix in self._LANGUAGES or parsed.suffix in {".css", ".html", ".sh", ".ps1"}:
            return "Product code"
        return "Unknown"

    def _logical_path(self, path: str) -> str:
        for prefix in ("tests/unit/", "tests/integration/", "tests/end_to_end/"):
            if path.startswith(prefix):
                remainder = path.removeprefix(prefix)
                if remainder.startswith("build_scripts/"):
                    return remainder
                return "pyrit/" + remainder
        if path.startswith("doc/code/"):
            remainder = path.removeprefix("doc/code/")
            component, _, rest = remainder.partition("/")
            return "pyrit/" + self._DOC_ALIASES.get(component, component) + "/" + rest
        return path

    @staticmethod
    def _primary(values: list[str]) -> str:
        counts = Counter(values)
        category, count = counts.most_common(1)[0]
        return category if count > len(values) / 2 else "Mixed"
