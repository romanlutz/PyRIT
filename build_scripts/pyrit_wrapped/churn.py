# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from pathlib import PurePosixPath

from build_scripts.pyrit_wrapped.models import FileChange, LineTotals, LocReport, LocScope, WrappedError


def language_of(path: str) -> str:
    file = PurePosixPath(path.lower())
    if file.name in {"uv.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}:
        return "Lockfiles"
    return {
        ".py": "Python",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".md": "Markdown",
        ".rst": "Documentation",
        ".json": "JSON",
        ".ipynb": "Notebook JSON",
    }.get(file.suffix, "Other")


def summarize_churn(
    *, files: list[FileChange], scope: LocScope, complete: bool, reason: str | None = None
) -> LocReport:
    languages: dict[str, LineTotals] = defaultdict(LineTotals)
    for language in ("TypeScript", "Python", "YAML"):
        languages[language]
    for file in files:
        if file.binary is not True:
            value = languages[language_of(file.path)]
            value.additions += file.additions
            languages[language_of(file.previous_path or file.path)].deletions += file.deletions
    totals = LineTotals(
        additions=sum(file.additions for file in files if file.binary is not True),
        deletions=sum(file.deletions for file in files if file.binary is not True),
    )
    return LocReport(
        scope=scope,
        complete=complete,
        file_count=len(files),
        binary_files=sum(file.binary is True for file in files)
        if all(file.binary is not None for file in files)
        else None,
        totals=totals if complete else None,
        by_language=dict(sorted(languages.items())) if complete else None,
        reason=reason,
    )


def parse_numstat(data: bytes) -> list[FileChange]:
    fields = data.decode("utf-8", errors="surrogateescape").split("\0")
    result: list[FileChange] = []
    index = 0
    while index < len(fields) and fields[index]:
        parts = fields[index].split("\t", 2)
        index += 1
        if len(parts) != 3:
            raise WrappedError("Git numstat returned a malformed record.")
        additions, deletions, path = parts
        previous = None
        if not path:
            if index + 1 >= len(fields) or not fields[index] or not fields[index + 1]:
                raise WrappedError("Git numstat returned an incomplete rename.")
            previous, path = fields[index : index + 2]
            index += 2
        binary = additions == deletions == "-"
        if not binary and (not additions.isdigit() or not deletions.isdigit()):
            raise WrappedError("Git numstat returned invalid line counts.")
        result.append(
            FileChange(
                path=path,
                previous_path=previous,
                binary=binary,
                additions=0 if binary else int(additions),
                deletions=0 if binary else int(deletions),
            )
        )
    return result
