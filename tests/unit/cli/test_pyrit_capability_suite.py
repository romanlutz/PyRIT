# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

from pyrit.cli import pyrit_capability_suite


def test_inspect_evals_command_analyzes_and_compiles_without_server(tmp_path):
    family_root = tmp_path / "src" / "inspect_evals" / "arc"
    family_root.mkdir(parents=True)
    (family_root / "eval.yaml").write_text(
        "title: ARC\n"
        "description: fixture\n"
        "group: Reasoning\n"
        "version: 1-A\n"
        "tasks:\n"
        "  - name: arc_easy\n"
        "    dataset_samples: 1\n"
        "metadata:\n"
        "  fast: true\n",
        encoding="utf-8",
    )
    (family_root / "arc.py").write_text("@task\ndef arc_easy():\n    return multiple_choice()\n", encoding="utf-8")
    data = tmp_path / "arc.json"
    data.write_text(
        json.dumps(
            [
                {
                    "id": "arc-1",
                    "question": "2 + 2?",
                    "choices": {"label": ["A", "B"], "text": ["3", "4"]},
                    "answerKey": "B",
                }
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"

    result = pyrit_capability_suite.main(
        [
            "inspect-evals",
            "--source",
            str(tmp_path),
            "--family",
            "arc",
            "--data",
            str(data),
            "--report",
            str(report_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert result == 0
    assert json.loads(report_path.read_text(encoding="utf-8"))["families"][0]["fidelity"] == "native"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["suite_id"].startswith("inspect-evals-arc")
