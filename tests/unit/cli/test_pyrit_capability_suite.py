# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from unittest.mock import AsyncMock

import pytest

from pyrit.cli import pyrit_capability_suite
from pyrit.compat.inspect_ai import cli as inspect_cli
from pyrit.compat.inspect_ai.profile import PINNED_INSPECT_EVALS_PROFILE
from pyrit.models import Message, TargetResponseMetadata
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration

pytestmark = pytest.mark.usefixtures("patch_central_database")


def test_default_profile_matches_pinned_compatibility_contract() -> None:
    assert PINNED_INSPECT_EVALS_PROFILE.profile_id == pyrit_capability_suite._DEFAULT_INSPECT_PROFILE_ID


class _ArcTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset({frozenset({"text"})}),
        )
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        response = Message.from_prompt(prompt="ANSWER: B", role="assistant")
        for piece in response.message_pieces:
            piece.conversation_id = normalized_conversation[-1].conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="arc-cli",
                stop_reason="completed",
                provider_stop_reason="completed",
            )
        )
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        del normalized_conversation


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


def test_unchanged_arc_compile_and_run_commands_need_no_embedded_credentials(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    source_root = tmp_path / "source"
    package = source_root / "src" / "inspect_evals"
    arc = package / "arc"
    utils = package / "utils"
    arc.mkdir(parents=True)
    utils.mkdir()
    for path in (package / "__init__.py", arc / "__init__.py", utils / "__init__.py"):
        path.write_text("", encoding="utf-8")
    (package / "metadata.py").write_text(
        "class Version:\n"
        "    comparability_version = 1\n"
        "    def to_metadata(self): return {'full_task_version': '1-A'}\n"
        "class Metadata: version = Version()\n"
        "def load_eval_metadata(name): return Metadata()\n",
        encoding="utf-8",
    )
    (utils / "huggingface.py").write_text(
        "import inspect_ai.dataset\n"
        "def hf_dataset(*args, **kwargs): return inspect_ai.dataset.hf_dataset(*args, **kwargs)\n",
        encoding="utf-8",
    )
    (arc / "arc.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import Sample\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "from inspect_evals.metadata import load_eval_metadata\n"
        "from inspect_evals.utils.huggingface import hf_dataset\n"
        "def convert(record):\n"
        "    return Sample(input=record['question'], choices=record['choices']['text'], target='B', id=record['id'])\n"
        "@task\n"
        "def arc_challenge():\n"
        "    return Task(dataset=hf_dataset(path='fixture', sample_fields=convert, revision='pinned'), "
        "solver=multiple_choice(), scorer=choice(), version=1)\n",
        encoding="utf-8",
    )
    data_path = tmp_path / "arc.json"
    data_path.write_text(
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
    manifest_path = tmp_path / "manifest.json"

    assert (
        pyrit_capability_suite.main(
            [
                "inspect-evals",
                "dry-run",
                "--source",
                str(source_root),
                "--task",
                "arc/arc.py@arc_challenge",
                "--data",
                str(data_path),
                "--no-verify-source",
                "--manifest",
                str(manifest_path),
            ]
        )
        == 0
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["cases"][0]["case_id"] == "arc-1"

    monkeypatch.setattr(inspect_cli, "_resolve_target_async", AsyncMock(return_value=_ArcTarget()))
    result_path = tmp_path / "result.json"
    assert (
        pyrit_capability_suite.main(
            [
                "inspect-evals",
                "run",
                "--source",
                str(source_root),
                "--task",
                "arc/arc.py@arc_challenge",
                "--data",
                str(data_path),
                "--no-verify-source",
                "--target",
                "fixture",
                "--result",
                str(result_path),
            ]
        )
        == 0
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["result"]["aggregate"]["outcome_counts"]["success"] == 1
