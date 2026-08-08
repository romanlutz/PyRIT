# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

import pyrit.compat.inspect_ai.loader as inspect_loader
from pyrit.compat.inspect_ai import (
    PINNED_INSPECT_EVALS_PROFILE,
    InspectProfileMismatchError,
    UnsupportedInspectFeatureError,
    inventory_inspect_api_usage,
    load_inspect_eval,
    run_inspect_eval_async,
)
from pyrit.compat.inspect_ai.scorer import parse_inspect_choice_answer
from pyrit.models import Message, TargetResponseMetadata
from pyrit.prompt_target import PromptTarget, TargetCapabilities, TargetConfiguration

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.usefixtures("patch_central_database")


def test_worker_process_uses_isolated_python_and_minimal_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("PYTHONPATH", "untrusted-import-root")
    monkeypatch.setenv("PATH", "worker-path")

    environment = inspect_loader._worker_environment()
    command = inspect_loader._worker_command(
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
    )

    assert environment["PATH"] == "worker-path"
    assert "OPENAI_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert command[1:4] == ("-I", "-B", "-m")


_ARC_SOURCE = """
from typing import Any, Literal

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice

from inspect_evals.metadata import load_eval_metadata
from inspect_evals.utils.huggingface import hf_dataset

DATASET_PATH = "allenai/ai2_arc"
ARC_DATASET_REVISION = "210d026faf9955653af8916fad021475a3f00453"
EVAL_VERSION = load_eval_metadata("arc").version

def arc_task(dataset_name: Literal["ARC-Easy", "ARC-Challenge"]) -> Task:
    return Task(
        dataset=hf_dataset(
            path=DATASET_PATH,
            name=dataset_name,
            split="test",
            sample_fields=record_to_sample,
            revision=ARC_DATASET_REVISION,
        ),
        solver=multiple_choice(),
        scorer=choice(),
        version=EVAL_VERSION.comparability_version,
        metadata=EVAL_VERSION.to_metadata(),
    )

@task
def arc_easy() -> Task:
    return arc_task("ARC-Easy")

@task
def arc_challenge() -> Task:
    return arc_task("ARC-Challenge")

def record_to_sample(record: dict[str, Any]) -> Sample:
    choices = record["choices"]
    choices = dict(zip(choices["label"], choices["text"]))
    answerKey = record["answerKey"]
    target_index = list(choices.keys()).index(answerKey)
    target = chr(ord("A") + target_index)
    return Sample(
        input=record["question"],
        choices=list(choices.values()),
        target=target,
        id=record["id"],
    )
"""

_METADATA_SOURCE = """
class TaskVersion:
    comparability_version = 2

    def to_metadata(self):
        return {
            "full_task_version": "2-A",
            "task_interface_version": "A",
            "task_comparability_version": 2,
        }

class EvalMetadata:
    version = TaskVersion()

def load_eval_metadata(name):
    assert name == "arc"
    return EvalMetadata()
"""

_HUGGINGFACE_SOURCE = """
import inspect_ai.dataset

def hf_dataset(*args, **kwargs):
    return inspect_ai.dataset.hf_dataset(*args, **kwargs)
"""


class _ScriptedTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            input_modalities=frozenset({frozenset({"text"})}),
        )
    )

    def __init__(self, *, response: str) -> None:
        super().__init__()
        self._response = response
        self.received: list[list[Message]] = []

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        self.received.append(normalized_conversation)
        response = Message.from_prompt(prompt=self._response, role="assistant")
        conversation_id = normalized_conversation[-1].conversation_id
        for piece in response.message_pieces:
            piece.conversation_id = conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="inspect-compat-fixture",
                stop_reason="completed",
                provider_stop_reason="completed",
            )
        )
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        return


@pytest.fixture
def arc_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "inspect_evals_checkout"
    package = root / "src" / "inspect_evals"
    arc = package / "arc"
    utils = package / "utils"
    arc.mkdir(parents=True)
    utils.mkdir()
    for path in (package / "__init__.py", arc / "__init__.py", utils / "__init__.py"):
        path.write_text("", encoding="utf-8")
    (arc / "arc.py").write_text(_ARC_SOURCE, encoding="utf-8")
    (package / "metadata.py").write_text(_METADATA_SOURCE, encoding="utf-8")
    (utils / "huggingface.py").write_text(_HUGGINGFACE_SOURCE, encoding="utf-8")
    return root


@pytest.fixture
def arc_records() -> list[dict[str, Any]]:
    return [
        {
            "id": "Mercury_7175875",
            "question": "Which object is attracted to a magnet?",
            "choices": {
                "label": ["A", "B", "C", "D"],
                "text": ["iron nail", "wood block", "glass bead", "rubber band"],
            },
            "answerKey": "A",
        }
    ]


def _load_arc(*, source_root: Path, records: list[dict[str, Any]], task_name: str = "arc_challenge"):
    return load_inspect_eval(
        source_root=source_root,
        task_spec=f"arc/arc.py@{task_name}",
        dataset_loader=lambda *args, **kwargs: records,
        verify_source_revision=False,
    )


def test_load_inspect_eval_executes_unchanged_arc_factory_and_mapper(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    loaded = _load_arc(source_root=arc_source_root, records=arc_records)

    assert loaded.report.profile_id == PINNED_INSPECT_EVALS_PROFILE.profile_id
    assert loaded.report.source_revision_verified is False
    assert loaded.task.metadata["full_task_version"] == "2-A"
    assert loaded.task.dataset[0].target == "A"
    case = loaded.suite.cases[0]
    assert case.case_id == "Mercury_7175875"
    assert case.messages[0].content == (
        "Answer the following multiple choice question. The entire content of your response should be of the "
        "following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of A,B,C,D.\n\n"
        "Which object is attracted to a magnet?\n\n"
        "A) iron nail\nB) wood block\nC) glass bead\nD) rubber band"
    )
    assert case.scorers[0].config == {
        "expected_value": "A",
        "allowed_options": ["A", "B", "C", "D"],
    }
    assert loaded.suite.metadata["source_revision_verified"] is False
    assert loaded.suite.metadata["detected_revision"] is None
    assert loaded.suite.metadata["case_timeout_seconds"] == 300.0
    assert "Native case execution is bounded to 300 seconds." in loaded.report.limitations
    assert loaded.suite.metadata["solver"]["name"] == "multiple_choice"
    assert loaded.suite.metadata["scorer"]["name"] == "choice"
    assert loaded.suite.metadata["dataset"] == {
        "source_type": "huggingface",
        "path": "allenai/ai2_arc",
        "name": "ARC-Challenge",
        "split": "test",
        "revision": "210d026faf9955653af8916fad021475a3f00453",
        "injected_records": True,
    }
    assert loaded.suite.cases[0].source is not None
    assert loaded.suite.cases[0].source.metadata["dataset"]["revision"] == ("210d026faf9955653af8916fad021475a3f00453")
    json.dumps(loaded.suite.model_dump(mode="json"))


def test_cli_and_worker_run_from_outside_source_cwd(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
    tmp_path: Path,
) -> None:
    data_path = tmp_path / "arc-records.json"
    data_path.write_text(json.dumps(arc_records), encoding="utf-8")
    manifest_path = tmp_path / "arc-manifest.json"
    outside_cwd = tmp_path / "outside"
    outside_cwd.mkdir()
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyrit.cli.pyrit_capability_suite",
            "inspect-evals",
            "compile",
            "--source",
            str(arc_source_root),
            "--task",
            "arc/arc.py@arc_challenge",
            "--data",
            str(data_path),
            "--no-verify-source",
            "--manifest",
            str(manifest_path),
        ],
        cwd=outside_cwd,
        env=environment,
        capture_output=True,
        check=False,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["cases"][0]["case_id"] == "Mercury_7175875"


async def test_async_cancellation_terminates_source_worker(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "slow.py").write_text(
        "import time\n"
        "time.sleep(30)\n"
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def slow():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='x', choices=['x'], target='A')]), "
        "solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )
    cancellation_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    loop.call_later(0.1, cancellation_event.set)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(
            run_inspect_eval_async(
                source_root=root,
                task_spec="slow.py@slow",
                verify_source_revision=False,
                target=_ScriptedTarget(response="ANSWER: A"),
                cancellation_event=cancellation_event,
            ),
            timeout=5,
        )


def test_loader_cleans_aliases_after_success(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    before = {name: module for name, module in sys.modules.items() if name.startswith(("inspect_ai", "inspect_evals"))}

    _load_arc(source_root=arc_source_root, records=arc_records)

    after = {name: module for name, module in sys.modules.items() if name.startswith(("inspect_ai", "inspect_evals"))}
    assert after == before


def test_loader_restores_preexisting_real_install_modules(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    real_root = ModuleType("inspect_ai")
    real_dataset = ModuleType("inspect_ai.dataset")
    real_root.dataset = real_dataset

    with patch.dict(sys.modules, {"inspect_ai": real_root, "inspect_ai.dataset": real_dataset}):
        loaded = _load_arc(source_root=arc_source_root, records=arc_records)
        assert loaded.suite.cases
        assert sys.modules["inspect_ai"] is real_root
        assert sys.modules["inspect_ai.dataset"] is real_dataset


def test_loader_cleans_aliases_after_source_failure(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals" / "broken"
    package.mkdir(parents=True)
    (root / "src" / "inspect_evals" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "task.py").write_text(
        "from inspect_ai import Task, task\nraise RuntimeError('fixture failure')\n",
        encoding="utf-8",
    )
    before = {name: module for name, module in sys.modules.items() if name.startswith(("inspect_ai", "inspect_evals"))}

    with pytest.raises(RuntimeError, match="fixture failure"):
        load_inspect_eval(
            source_root=root,
            task_spec="broken/task.py@broken",
            verify_source_revision=False,
        )

    after = {name: module for name, module in sys.modules.items() if name.startswith(("inspect_ai", "inspect_evals"))}
    assert after == before


def test_loader_serializes_concurrent_imports_without_cross_talk(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    barrier = threading.Barrier(2)

    def _worker(task_name: str) -> str:
        barrier.wait()
        loaded = _load_arc(source_root=arc_source_root, records=arc_records, task_name=task_name)
        return loaded.report.task_name

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(_worker, ("arc_easy", "arc_challenge")))

    assert set(results) == {"arc_easy", "arc_challenge"}
    assert "inspect_ai" not in sys.modules


def test_loader_is_reentrant(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    common_imports = (
        "from inspect_ai import Task, task\n"
        "from inspect_ai._util.registry import registry_find, registry_info\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
    )
    (package / "inner.py").write_text(
        common_imports + "@task\n"
        "def inner():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='inner', choices=['x'], target='A')]), "
        "solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )
    (package / "outer.py").write_text(
        "from pathlib import Path\n"
        "from pyrit.compat.inspect_ai import load_inspect_eval\n" + common_imports + "@task\n"
        "def outer():\n"
        "    nested = load_inspect_eval(source_root=Path(__file__).parents[2], task_spec='inner.py@inner', "
        "verify_source_revision=False)\n"
        "    assert nested.suite.cases[0].objective == 'inner'\n"
        "    return Task(dataset=MemoryDataset([Sample(input='outer', choices=['x'], target='A')]), "
        "solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="outer.py@outer",
        verify_source_revision=False,
    )

    assert loaded.suite.cases[0].objective == "outer"
    assert "inspect_ai" not in sys.modules


def test_task_decorator_validates_factory_parameters(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "parameterized.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai._util.registry import registry_find, registry_info\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task(name='variant')\n"
        "def build_variant(*, prefix='default'):\n"
        "    return Task(dataset=MemoryDataset([Sample(input=prefix, choices=['x'], target='A')]), "
        "solver=multiple_choice(), scorer=choice())\n"
        "assert registry_info() == ('variant',)\n"
        "assert registry_find('variant') is build_variant\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="parameterized.py@variant",
        task_parameters={"prefix": "custom"},
        verify_source_revision=False,
    )

    assert loaded.report.task_parameters == {"prefix": "custom"}
    assert loaded.suite.cases[0].objective == "custom"
    with pytest.raises(ValueError, match="Invalid parameters"):
        load_inspect_eval(
            source_root=root,
            task_spec="parameterized.py@variant",
            task_parameters={"unknown": True},
            verify_source_revision=False,
        )


def test_inventory_is_deterministic_and_reports_unknown_symbol(arc_source_root: Path) -> None:
    profile = PINNED_INSPECT_EVALS_PROFILE
    first = inventory_inspect_api_usage(source_root=arc_source_root, profile=profile)
    second = inventory_inspect_api_usage(source_root=arc_source_root, profile=profile)
    assert first.to_dict() == second.to_dict()
    assert first.unsupported_symbols == ()

    source = arc_source_root / "src" / "inspect_evals" / "arc" / "unknown.py"
    source.write_text("from inspect_ai.agent import handoff\n", encoding="utf-8")
    inventory = inventory_inspect_api_usage(
        source_root=arc_source_root,
        profile=profile,
        source_files=(source,),
    )
    assert "inspect_ai.agent.handoff" in inventory.unsupported_symbols


def test_loader_rejects_unknown_symbol_in_imported_source_module(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "helper.py").write_text("from inspect_ai.agent import handoff\n", encoding="utf-8")
    (package / "task.py").write_text(
        "from inspect_ai import Task, task\nfrom inspect_evals import helper\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError) as error:
        load_inspect_eval(
            source_root=root,
            task_spec="task.py@fixture",
            verify_source_revision=False,
        )

    assert error.value.symbol == "inspect_ai.agent.handoff"


def test_loader_rejects_unknown_symbols_before_execution(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    sentinel = tmp_path / "executed"
    (package / "unknown.py").write_text(
        f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('executed')\n"
        "from inspect_ai.agent import handoff\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError) as error:
        load_inspect_eval(
            source_root=root,
            task_spec="unknown.py@unknown",
            verify_source_revision=False,
        )

    assert error.value.symbol == "inspect_ai.agent.handoff"
    assert error.value.source_profile == PINNED_INSPECT_EVALS_PROFILE.profile_id
    assert "Remediation:" in str(error.value)
    assert not sentinel.exists()


def test_loader_rejects_profile_and_revision_mismatches(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    with pytest.raises(InspectProfileMismatchError, match="Unknown Inspect compatibility profile"):
        load_inspect_eval(
            source_root=arc_source_root,
            task_spec="arc/arc.py@arc_easy",
            profile_id="latest",
            dataset_loader=lambda *args, **kwargs: arc_records,
            verify_source_revision=False,
        )
    with pytest.raises(InspectProfileMismatchError, match="unverified"):
        load_inspect_eval(
            source_root=arc_source_root,
            task_spec="arc/arc.py@arc_easy",
            dataset_loader=lambda *args, **kwargs: arc_records,
        )


@pytest.mark.parametrize(
    "task_spec",
    (
        "../outside.py@task",
        "arc/../../outside.py@task",
        "arc/arc.py@not-valid!",
        "arc.arc",
    ),
)
def test_loader_rejects_unsafe_or_malformed_task_specs(arc_source_root: Path, task_spec: str) -> None:
    with pytest.raises(ValueError):
        load_inspect_eval(
            source_root=arc_source_root,
            task_spec=task_spec,
            verify_source_revision=False,
        )


def test_loader_serializes_inspect_messages_and_task_metadata(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "messages.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import ChatMessageSystem, ChatMessageUser\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def messages():\n"
        "    sample = Sample(input=[ChatMessageSystem(content='Be concise.'), "
        "ChatMessageUser(content='Choose.')], choices=['first', 'second'], target='B', "
        "metadata={'difficulty': 'easy'})\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(shuffle=False), "
        "scorer=choice(), metadata={'category': 'fixture'})\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="messages.py@messages",
        verify_source_revision=False,
    )

    case = loaded.suite.cases[0]
    assert [message.role for message in case.messages] == ["system", "user"]
    assert case.messages[0].content == "Be concise."
    assert "B) second" in case.messages[1].content
    assert loaded.suite.metadata["task_metadata"] == {"category": "fixture"}
    assert case.metadata["sample_metadata"] == {"difficulty": "easy"}


async def test_run_inspect_eval_executes_full_arc_case_through_injected_target(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    target = _ScriptedTarget(response="ANSWER: A")

    execution = await run_inspect_eval_async(
        source_root=arc_source_root,
        task_spec="arc/arc.py@arc_challenge",
        target=target,
        dataset_loader=lambda *args, **kwargs: arc_records,
        verify_source_revision=False,
    )

    assert execution.result.attempts[0].task_result.scores[0].score_value == "True"
    assert target.received
    assert "Which object is attracted to a magnet?" in target.received[0][-1].message_pieces[0].converted_value


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("ANSWER: B", "B"),
        ("answer : b.", "B"),
        ("reasoning\nANSWER: C", "C"),
        ("ANSWER: A\nANSWER: D", "D"),
        ("B", None),
        ("ANSWER: E", None),
    ],
)
def test_choice_parser_matches_pinned_inspect_contract(completion: str, expected: str | None) -> None:
    assert (
        parse_inspect_choice_answer(
            completion,
            allowed_options=frozenset({"A", "B", "C", "D"}),
        )
        == expected
    )


@pytest.mark.parametrize(
    ("sample_field", "symbol"),
    [
        ("setup='echo unsupported'", "Sample.setup"),
        ("sandbox='docker'", "Sample.sandbox"),
        ("files={'input.txt': 'unsupported'}", "Sample.files"),
    ],
)
def test_loader_rejects_unsupported_sample_execution_fields(
    tmp_path: Path,
    sample_field: str,
    symbol: str,
) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def unsupported():\n"
        f"    sample = Sample(input='Question?', choices=['one', 'two'], target='A', {sample_field})\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError, match=symbol):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


@pytest.mark.parametrize(
    ("message_field", "symbol"),
    [("name='named'", "ChatMessage.name"), ("source='source'", "ChatMessage.source")],
)
def test_loader_rejects_unsupported_message_execution_fields(
    tmp_path: Path,
    message_field: str,
    symbol: str,
) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import ChatMessageUser\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def unsupported():\n"
        f"    message = ChatMessageUser(content='Question?', {message_field})\n"
        "    sample = Sample(input=[message], choices=['one', 'two'], target='A')\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError, match=symbol):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_loader_rejects_unsupported_task_generation_config(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import GenerateConfig\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def unsupported():\n"
        "    sample = Sample(input='Question?', choices=['one', 'two'], target='A')\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(), scorer=choice(), "
        "config=GenerateConfig(max_tokens=8))\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError, match=r"Task\.config\.max_tokens"):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_loader_templates_last_user_message_not_trailing_assistant(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "messages.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import ChatMessageAssistant, ChatMessageUser\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def messages():\n"
        "    sample = Sample(input=[ChatMessageUser(content='Question?'), "
        "ChatMessageAssistant(content='Earlier answer')], choices=['one', 'two'], target='A')\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="messages.py@messages",
        verify_source_revision=False,
    )

    assert "ANSWER: $LETTER" in loaded.suite.cases[0].messages[0].content
    assert loaded.suite.cases[0].messages[1].content == "Earlier answer"
    assert loaded.suite.cases[0].objective == "Question?"


def test_loader_uses_pinned_choice_labels_after_z(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "labels.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def labels():\n"
        "    sample = Sample(input='Question?', choices=[str(i) for i in range(28)], target='2')\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="labels.py@labels",
        verify_source_revision=False,
    )

    prompt = loaded.suite.cases[0].messages[0].content
    assert "Z) 25\n1) 26\n2) 27" in prompt
    assert loaded.suite.cases[0].scorers[0].config["allowed_options"][-3:] == ["Z", "1", "2"]


def test_loader_terminates_worker_after_timeout(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "slow.py").write_text(
        "import time\n"
        "time.sleep(5)\n"
        "from inspect_ai import Task, task\n"
        "@task\n"
        "def slow():\n"
        "    raise AssertionError('unreachable')\n",
        encoding="utf-8",
    )

    with pytest.raises(TimeoutError, match="worker exceeded"):
        load_inspect_eval(
            source_root=root,
            task_spec="slow.py@slow",
            verify_source_revision=False,
            worker_timeout_seconds=0.1,
        )


def test_revision_verification_rejects_ignored_source_files(
    arc_source_root: Path,
    arc_records: list[dict[str, Any]],
) -> None:
    (arc_source_root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    subprocess.run(["git", "init", str(arc_source_root)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(arc_source_root), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(arc_source_root),
            "-c",
            "user.name=PyRIT Test",
            "-c",
            "user.email=pyrit@example.invalid",
            "commit",
            "-m",
            "fixture",
        ],
        check=True,
        capture_output=True,
    )
    (arc_source_root / "ignored.py").write_text("raise RuntimeError('must not load')\n", encoding="utf-8")

    with (
        patch(
            "pyrit.compat.inspect_ai.loader._git_revision",
            return_value=PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision,
        ),
        pytest.raises(InspectProfileMismatchError, match="ignored.py"),
    ):
        load_inspect_eval(
            source_root=arc_source_root,
            task_spec="arc/arc.py@arc_challenge",
            dataset_loader=lambda *args, **kwargs: arc_records,
        )
