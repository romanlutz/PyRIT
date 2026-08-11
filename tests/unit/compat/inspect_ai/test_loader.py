# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
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

import pyrit.compat.inspect_ai.facade as inspect_facade
import pyrit.compat.inspect_ai.loader as inspect_loader
from pyrit.compat.inspect_ai import (
    PINNED_INSPECT_EVALS_PROFILE,
    InspectProfileMismatchError,
    UnsupportedInspectFeatureError,
    inventory_inspect_api_usage,
    load_inspect_eval,
    run_inspect_eval_async,
)
from pyrit.compat.inspect_ai.scorer import (
    InspectPatternScorer,
    InspectPatternScorerConfig,
    InspectTextScorer,
    InspectTextScorerConfig,
    normalize_inspect_score,
    parse_inspect_choice_answer,
)
from pyrit.compat.inspect_ai.types import Dataset, Sample, Target
from pyrit.compat.inspect_ai.types import Score as InspectScore
from pyrit.models import Message, TargetResponseMetadata
from pyrit.prompt_target import (
    OpenAIChatRequestOptions,
    PromptTarget,
    TargetCapabilities,
    TargetConfiguration,
)

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

    environment = inspect_loader._worker_environment(allow_network=False)
    network_environment = inspect_loader._worker_environment(allow_network=True)
    command = inspect_loader._worker_command(
        request_path=tmp_path / "request.json",
        response_path=tmp_path / "response.json",
    )

    assert environment["PATH"] == "worker-path"
    assert "OPENAI_API_KEY" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert "HF_DATASETS_OFFLINE" not in network_environment
    assert "HF_HUB_OFFLINE" not in network_environment
    assert command[1:4] == ("-I", "-B", "-m")


def test_code_evaluation_mapping_rejects_unknown_scorer_through_public_loader(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals" / "humaneval"
    package.mkdir(parents=True)
    (root / "src" / "inspect_evals" / "__init__.py").write_text("", encoding="utf-8")
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "humaneval.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import CORRECT, Score, accuracy, scorer\n"
        "from inspect_ai.solver import generate\n"
        "@scorer(metrics=[accuracy()])\n"
        "def custom_verify():\n"
        "    async def score(state, target):\n"
        "        return Score(value=CORRECT)\n"
        "    return score\n"
        "@task\n"
        "def humaneval():\n"
        "    sample = Sample(input='complete add', target='return a + b', id='case', "
        "metadata={'prompt': 'def add(a, b):\\n', 'test': 'def check(candidate):\\n"
        "    assert candidate(1, 2) == 3', 'entry_point': 'add'})\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=generate(), "
        "scorer=custom_verify(), sandbox='docker')\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError, match="custom_verify"):
        load_inspect_eval(
            source_root=root,
            task_spec="humaneval/humaneval.py@humaneval",
            verify_source_revision=False,
        )


def test_offline_worker_sets_huggingface_offline_before_source_import(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "offline.py").write_text(
        "import datasets.config\n"
        "import huggingface_hub.constants\n"
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import match\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def offline():\n"
        "    assert datasets.config.HF_DATASETS_OFFLINE is True\n"
        "    assert huggingface_hub.constants.HF_HUB_OFFLINE is True\n"
        "    return Task(dataset=MemoryDataset([Sample(input='Q', target='A')]), "
        "solver=generate(), scorer=match())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="offline.py@offline",
        verify_source_revision=False,
        allow_network=False,
    )

    assert len(loaded.suite.cases) == 1


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


class _ConfigurableScriptedTarget(_ScriptedTarget):
    def __init__(self, *, response: str) -> None:
        super().__init__(response=response)
        self.temperatures: list[float | None] = []
        self.max_completion_tokens: list[int | None] = []

    def _get_default_request_options(self) -> OpenAIChatRequestOptions:
        return OpenAIChatRequestOptions(
            temperature=None,
            top_p=None,
            max_completion_tokens=None,
            frequency_penalty=None,
            presence_penalty=None,
            seed=None,
            n=None,
            extra_body_parameters=None,
        )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        options = self._get_request_options(OpenAIChatRequestOptions)
        self.temperatures.append(options.temperature if isinstance(options.temperature, float) else None)
        self.max_completion_tokens.append(
            options.max_completion_tokens if isinstance(options.max_completion_tokens, int) else None
        )
        return await super()._send_prompt_to_target_async(normalized_conversation=normalized_conversation)


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


def test_loader_rejects_unsupported_message_name(
    tmp_path: Path,
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
        "    message = ChatMessageUser(content='Question?', name='named')\n"
        "    sample = Sample(input=[message], choices=['one', 'two'], target='A')\n"
        "    return Task(dataset=MemoryDataset([sample]), solver=multiple_choice(), scorer=choice())\n",
        encoding="utf-8",
    )

    with pytest.raises(UnsupportedInspectFeatureError, match=r"Sample\.input\[0\]\.name"):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_loader_preserves_multiple_choice_task_max_tokens(tmp_path: Path) -> None:
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

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="unsupported.py@unsupported",
        verify_source_revision=False,
    )

    assert loaded.suite.cases[0].limits.max_output_tokens == 8


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


def test_static_solver_composition_compiles_and_runs_through_public_seam(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "prompts").mkdir()
    (package / "prompts" / "system.txt").write_text("Rules for {topic}", encoding="utf-8")
    (package / "static.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import ChatMessageAssistant, ChatMessageUser\n"
        "from inspect_ai.scorer import accuracy, grouped, includes, match, pass_at\n"
        "from inspect_ai.solver import chain, generate, prompt_template, system_message\n"
        "@task\n"
        "def static():\n"
        "    sample = Sample(input=[ChatMessageUser(content='Example'), "
        "ChatMessageAssistant(content='Demonstration'), ChatMessageUser(content='finish')], "
        "target='done', metadata={'topic': 'science'})\n"
        "    other = Sample(input='finish', target='wrong', metadata={'topic': 'history'})\n"
        "    return Task(dataset=MemoryDataset([sample, other]), setup=system_message('prompts/system.txt'), "
        "solver=chain(prompt_template('Question: {prompt}'), generate('none', max_tokens=32)), "
        "scorer=[includes(), match(location='exact')], metrics=[grouped(accuracy(), 'topic')], "
        "epochs=2, epochs_reducer=pass_at(1))\n",
        encoding="utf-8",
    )
    target = _ConfigurableScriptedTarget(response="done")

    run = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="static.py@static",
            verify_source_revision=False,
            target=target,
        )
    )

    case = run.loaded.suite.cases[0]
    assert [message.role for message in case.messages] == ["system", "user", "assistant", "user"]
    assert [message.content for message in case.messages] == [
        "Rules for science",
        "Example",
        "Demonstration",
        "Question: finish",
    ]
    assert [scorer.kind for scorer in case.scorers] == ["inspect_text", "inspect_text"]
    assert all(scorer.metrics for scorer in case.scorers)
    assert all(scorer.metrics[0].group_aggregate == "samples" for scorer in case.scorers)
    assert all(scorer.reducers for scorer in case.scorers)
    assert len(target.received) == 4
    assert target.max_completion_tokens == [32, 32, 32, 32]
    assert set(run.result.aggregate.reducer_values.values()) == {0.5}
    grouped = next(iter(run.result.aggregate.grouped_metric_values.values()))
    assert grouped == {
        "metadata.topic=history": 0.0,
        "metadata.topic=science": 1.0,
    }


def test_pattern_scorer_runs_correct_incorrect_and_no_match_cases(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "pattern_eval.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import pattern\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def pattern_eval():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='q', target='Yes')]), "
        "solver=generate(), scorer=pattern(r'(Yes|No).?\\Z'))\n",
        encoding="utf-8",
    )

    correct = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="pattern_eval.py@pattern_eval",
            verify_source_revision=False,
            target=_ScriptedTarget(response="Yes."),
        )
    )
    incorrect = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="pattern_eval.py@pattern_eval",
            verify_source_revision=False,
            target=_ScriptedTarget(response="No"),
        )
    )
    no_match = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="pattern_eval.py@pattern_eval",
            verify_source_revision=False,
            target=_ScriptedTarget(response="Maybe"),
        )
    )

    assert correct.result.attempts[0].task_result.scores[0].score_value == "True"
    assert incorrect.result.attempts[0].task_result.scores[0].score_value == "False"
    assert no_match.result.attempts[0].task_result.scores[0].score_value == "False"
    assert correct.loaded.suite.cases[0].scorers[0].kind == "inspect_pattern"


def test_multiple_choice_setup_metrics_and_temperature_run_natively(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mc_eval.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import GenerateConfig\n"
        "from inspect_ai.scorer import accuracy, choice, stderr\n"
        "from inspect_ai.solver import multiple_choice, system_message\n"
        "@task\n"
        "def mc_eval():\n"
        "    return Task(dataset=MemoryDataset([Sample(id='case', input='q', target='A', "
        "choices=['right', 'wrong'], metadata={'cluster': 'one'})]), "
        "solver=[system_message('rules'), multiple_choice(cot=True)], scorer=choice(), "
        "metrics=[accuracy(), stderr(cluster='cluster')], config=GenerateConfig(temperature=0))\n",
        encoding="utf-8",
    )
    target = _ConfigurableScriptedTarget(response="ANSWER: A")

    run = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="mc_eval.py@mc_eval",
            verify_source_revision=False,
            target=target,
        )
    )

    case = run.loaded.suite.cases[0]
    assert [message.role for message in case.messages] == ["system", "user"]
    assert case.scorers[0].metrics[1].cluster_by == "cluster"
    assert target.received
    assert run.result.attempts[0].task_result.scores[0].score_value == "True"
    assert target.temperatures == [0.0]


def test_generation_option_preflight_fails_before_model_call(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "configured.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import GenerateConfig\n"
        "from inspect_ai.scorer import choice\n"
        "from inspect_ai.solver import multiple_choice\n"
        "@task\n"
        "def configured():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='q', target='A', choices=['a', 'b'])]), "
        "solver=multiple_choice(), scorer=choice(), config=GenerateConfig(temperature=0))\n",
        encoding="utf-8",
    )
    target = _ScriptedTarget(response="ANSWER: A")

    with pytest.raises(ValueError, match="cannot preserve Inspect generation option 'temperature'"):
        asyncio.run(
            run_inspect_eval_async(
                source_root=root,
                task_spec="configured.py@configured",
                verify_source_revision=False,
                target=target,
            )
        )

    assert target.received == []


def test_dataset_records_support_request_specific_multi_split_data(arc_source_root: Path) -> None:
    records = {
        "allenai/ai2_arc|ARC-Challenge|test|210d026faf9955653af8916fad021475a3f00453": [
            {
                "id": "case",
                "question": "q",
                "choices": {"label": ["A", "B"], "text": ["right", "wrong"]},
                "answerKey": "A",
            }
        ]
    }

    loaded = load_inspect_eval(
        source_root=arc_source_root,
        task_spec="arc/arc.py@arc_challenge",
        dataset_records=records,
        verify_source_revision=False,
    )

    assert loaded.suite.cases[0].case_id == "case"


def test_unkeyed_dataset_records_reject_distinct_requests() -> None:
    loader = inspect_loader._injected_records_loader([{"input": "fixture"}])

    assert loader("fixture/path", "first", split="test", revision="pinned")
    with pytest.raises(ValueError, match="Unkeyed injected dataset records"):
        loader("fixture/path", "second", split="test", revision="pinned")


def test_generic_dataset_key_rejects_distinct_requests() -> None:
    loader = inspect_loader._injected_records_loader({"fixture/path": [{"input": "fixture"}]})

    assert loader("fixture/path", "first", split="test", revision="pinned")
    with pytest.raises(ValueError, match="matches multiple distinct dataset requests"):
        loader("fixture/path", "second", split="test", revision="pinned")


def test_empty_dataset_records_do_not_fall_back_to_acquisition(arc_source_root: Path) -> None:
    with pytest.raises(ValueError, match="produced an empty dataset"):
        load_inspect_eval(
            source_root=arc_source_root,
            task_spec="arc/arc.py@arc_challenge",
            dataset_records=[],
            allow_network=True,
            verify_source_revision=False,
        )


def test_combined_dataset_recomputes_provenance_and_aggregates_source_metadata() -> None:
    first = Dataset(
        [Sample(id="one", input="q1", target="A")],
        name="first",
        location="fixture/combined",
        metadata={"path": "fixture/combined", "name": "first", "revision": "pinned"},
    )
    second = Dataset(
        [Sample(id="two", input="q2", target="B")],
        name="second",
        location="fixture/combined",
        metadata={"path": "fixture/combined", "name": "second", "revision": "pinned"},
    )

    combined = Dataset(
        [first[0], second[0]],
        name="first",
        location="fixture/combined",
    )

    assert combined.provenance["record_count"] == 2
    assert combined.metadata["source_type"] == "combined"
    assert [source["name"] for source in combined.metadata["sources"]] == ["first", "second"]


def test_pattern_group_matching_preserves_pinned_answer_semantics() -> None:
    scorer = InspectPatternScorer(
        config=InspectPatternScorerConfig(
            expected_values=("foo",),
            pattern="(foo)",
            ignore_case=True,
        )
    )
    match_all = InspectPatternScorer(
        config=InspectPatternScorerConfig(
            expected_values=("foo",),
            pattern="(foo) (foo)",
            ignore_case=True,
            match_all=True,
        )
    )

    assert scorer._match_groups(("FOO",)) == ("FOO", "C")
    assert scorer._match_groups(("bar",)) == ("bar", "I")
    assert match_all._match_groups(("foo", "foo")) == ("foo", "C")

    empty_target = InspectPatternScorer(
        config=InspectPatternScorerConfig(
            expected_values=("",),
            pattern="(.*)",
            ignore_case=True,
        )
    )
    empty_target_all = InspectPatternScorer(
        config=InspectPatternScorerConfig(
            expected_values=("",),
            pattern="(.*)",
            ignore_case=True,
            match_all=True,
        )
    )
    assert empty_target._match_groups(("",)) == ("", "I")
    assert empty_target_all._match_groups(("",)) == ("", "I")
    assert Target(["A", "B"]).text == "AB"
    assert Target().text == ""

    unicode_case = InspectPatternScorer(
        config=InspectPatternScorerConfig(
            expected_values=("SS",),
            pattern="(.*)",
            ignore_case=True,
        )
    )
    assert unicode_case._match_groups(("ß",)) == ("ß", "I")


def test_script_backed_dataset_requires_injected_records_before_network(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "scripted.py").write_text(
        "from pathlib import Path\n"
        "from inspect_ai import Task, task\n"
        "from inspect_ai.scorer import match\n"
        "from inspect_ai.solver import generate\n"
        "from inspect_evals.hf_dataset_script_helper import load_hf_dataset_with_script\n"
        "def map_record(record):\n"
        "    from inspect_ai.dataset import Sample\n"
        "    return Sample(input=record['input'], target=record['target'])\n"
        "@task\n"
        "def scripted():\n"
        "    dataset = load_hf_dataset_with_script(repo_id='fixture/scripted', "
        "record_to_sample=map_record, builder_cls=object, cache_dir_fp=Path('.'), "
        "split='test', revision='pinned')\n"
        "    return Task(dataset=dataset, solver=generate(), scorer=match())\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Script-backed Hugging Face datasets require locally materialized"):
        load_inspect_eval(
            source_root=root,
            task_spec="scripted.py@scripted",
            allow_network=True,
            verify_source_revision=False,
        )


def test_static_solver_compile_rejects_unknown_generate_option_before_model_execution(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import includes\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def unsupported():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='x', target='y')]), "
        "solver=generate('none', logprobs=True), scorer=includes())\n",
        encoding="utf-8",
    )

    with pytest.raises(
        UnsupportedInspectFeatureError,
        match=r"Task\.solver\[0\]\.generate\(logprobs=\.\.\.\)",
    ):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_static_solver_compile_reports_unsupported_setup_path(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import includes\n"
        "from inspect_ai.solver import generate, multiple_choice\n"
        "@task\n"
        "def unsupported():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='x', target='y')]), "
        "setup=multiple_choice(), solver=generate('none'), scorer=includes())\n",
        encoding="utf-8",
    )

    with pytest.raises(
        UnsupportedInspectFeatureError,
        match=r"Task\.setup\[0\]\.multiple_choice",
    ):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_static_scorer_compile_rejects_unknown_match_location(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import match\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def unsupported():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='x', target='y')]), "
        "solver=generate('none'), scorer=match(location='middle'))\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Task\.scorer\[0\]\.match\(location=\.\.\.\) has unknown value 'middle'",
    ):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_static_scorer_compile_rejects_dict_task_metrics_until_score_key_mapping_is_supported(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "unsupported.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import match, mean\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def unsupported():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='x', target='y')]), "
        "solver=generate('none'), scorer=match(), metrics={'accuracy': [mean()]})\n",
        encoding="utf-8",
    )

    with pytest.raises(
        UnsupportedInspectFeatureError,
        match=r"Task\.metrics dict routing for scorer\[0\]",
    ):
        load_inspect_eval(
            source_root=root,
            task_spec="unsupported.py@unsupported",
            verify_source_revision=False,
        )


def test_static_multimodal_messages_preserve_parts_and_fail_text_only_preflight(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (package / "vision.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import ChatMessageUser, ContentImage, ContentText\n"
        "from inspect_ai.scorer import includes\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def vision():\n"
        "    local = ChatMessageUser(content=[ContentText(text='describe'), "
        "ContentImage(image='image.png', detail='low')])\n"
        "    remote = ChatMessageUser(content=[ContentText(text='describe'), "
        "ContentImage(image='https://example.test/image.png')])\n"
        "    inline = ChatMessageUser(content=[ContentText(text='describe'), "
        "ContentImage(image='data:image/png;base64,iVBORw0KGgo=')])\n"
        "    return Task(dataset=MemoryDataset([Sample(input=[local], target='cat'), "
        "Sample(input=[remote], target='cat'), Sample(input=[inline], target='cat')]), "
        "solver=generate('none'), scorer=includes())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="vision.py@vision",
        verify_source_revision=False,
    )

    message = loaded.suite.cases[0].messages[0]
    assert message.parts[0].content == "describe"
    assert message.parts[1].data_type == "url"
    assert message.parts[1].content == "data:image/png;base64,iVBORw0KGgo="
    assert message.parts[1].metadata["source"] == "image.png"
    assert message.parts[1].metadata["media_type"] == "image/png"
    assert loaded.suite.cases[1].messages[0].parts[1].data_type == "url"
    assert loaded.suite.cases[2].messages[0].parts[1].data_type == "url"
    target = _ScriptedTarget(response="cat")
    with pytest.raises(ValueError, match=r"(?s)url.*text|text.*url"):
        asyncio.run(
            run_inspect_eval_async(
                source_root=root,
                task_spec="vision.py@vision",
                verify_source_revision=False,
                target=target,
            )
        )
    assert target.received == []


def test_json_field_mapping_shuffle_limit_has_deterministic_provenance(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "records.json").write_text(
        json.dumps(
            [
                {"qid": "a", "question": "A?", "answer": "A", "subject": "one"},
                {"qid": "b", "question": "B?", "answer": "B", "subject": "two"},
                {"qid": "c", "question": "C?", "answer": "C", "subject": "three"},
            ]
        ),
        encoding="utf-8",
    )
    (package / "records.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import FieldSpec, json_dataset\n"
        "from inspect_ai.scorer import includes\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def records():\n"
        "    dataset = json_dataset('records.json', "
        "sample_fields=FieldSpec(input='question', target='answer', id='qid', metadata=['subject']), "
        "shuffle=True, seed=9, limit=2)\n"
        "    return Task(dataset=dataset, solver=generate('none'), scorer=includes())\n",
        encoding="utf-8",
    )

    first = load_inspect_eval(
        source_root=root,
        task_spec="records.py@records",
        verify_source_revision=False,
    )
    second = load_inspect_eval(
        source_root=root,
        task_spec="records.py@records",
        verify_source_revision=False,
    )

    assert first.task.dataset.provenance == second.task.dataset.provenance
    assert first.task.dataset.provenance["record_count"] == 2
    assert first.task.dataset.provenance["selection"] == {
        "auto_id": False,
        "shuffle": True,
        "seed": 9,
        "shuffle_choices": None,
        "limit": 2,
    }
    assert [case.case_id for case in first.suite.cases] == [case.case_id for case in second.suite.cases]


def test_csv_field_mapping_and_auto_ids_compile_through_public_seam(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "records.csv").write_text("qid,question,target\nold-a,A?,1\nold-b,B?,2\n", encoding="utf-8")
    (package / "records.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import FieldSpec, csv_dataset\n"
        "from inspect_ai.scorer import includes\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def records():\n"
        "    dataset = csv_dataset('records.csv', "
        "sample_fields=FieldSpec(input='question', target='target', choices=None, id='qid'), auto_id=True)\n"
        "    return Task(dataset=dataset, solver=generate('none'), scorer=includes())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="records.py@records",
        verify_source_revision=False,
    )

    assert [sample.id for sample in loaded.task.dataset] == [1, 2]
    assert [case.case_id for case in loaded.suite.cases] == ["1", "2"]
    assert loaded.task.dataset.metadata["source_type"] == "csv"
    assert loaded.task.dataset.provenance["selection"]["auto_id"] is True


def test_field_spec_normalizes_messages_numeric_targets_and_serialized_choices(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (package / "records.json").write_text(
        json.dumps(
            [
                {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "choose"},
                                {"type": "image", "image": "image.png", "detail": "low"},
                            ],
                        }
                    ],
                    "target": 2,
                    "choices": "one, two, three",
                }
            ]
        ),
        encoding="utf-8",
    )
    (package / "records.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import FieldSpec, json_dataset\n"
        "from inspect_ai.scorer import includes\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def records():\n"
        "    return Task(dataset=json_dataset('records.json', sample_fields=FieldSpec()), "
        "solver=generate('none'), scorer=includes())\n",
        encoding="utf-8",
    )

    loaded = load_inspect_eval(
        source_root=root,
        task_spec="records.py@records",
        verify_source_revision=False,
    )

    sample = loaded.task.dataset[0]
    assert sample.target == "2"
    assert sample.choices == ["one", "two", "three"]
    assert not isinstance(sample.input, str)
    assert sample.input[0].role == "user"
    assert [part.type for part in sample.input[0].content] == ["text", "image"]


@pytest.mark.parametrize(
    ("location", "completion", "expected"),
    [
        ("begin", "  Cat, because it fits.  ", "cat"),
        ("end", "Answer: cat.", "cat"),
        ("exact", "  [Cat]!  ", "cat"),
    ],
)
def test_inspect_match_normalizes_surrounding_punctuation(
    location: str,
    completion: str,
    expected: str,
) -> None:
    scorer = InspectTextScorer(
        config=InspectTextScorerConfig(
            expected_values=(expected,),
            mode="match",
            location=location,
        )
    )

    assert scorer._matches(completion=completion, expected=expected)


def test_dataset_filter_recomputes_content_provenance() -> None:
    dataset = Dataset(
        (Sample(input="one", id=1), Sample(input="two", id=2)),
        provenance={"record_count": 2, "records_sha256": "before"},
    )

    filtered = dataset.filter(lambda sample: sample.id == 1)

    assert filtered.provenance["record_count"] == 1
    assert filtered.provenance["records_sha256"] != "before"
    assert filtered.provenance["filter_applied"] is True


def test_dataset_options_replace_existing_ids_and_honor_zero_choice_seed() -> None:
    selected = inspect_facade._apply_dataset_options(
        dataset=Dataset((Sample(input="choose", target="A", choices=["one", "two", "three"], id="old"),)),
        auto_id=True,
        shuffle=False,
        seed=None,
        shuffle_choices=0,
        limit=None,
    )

    assert selected[0].id == 1
    assert selected[0].choices == ["one", "three", "two"]
    assert selected[0].target == "A"
    assert selected.provenance["selection"]["seed"] == 0
    assert selected.provenance["selection"]["choice_seed"] == 0


def test_normalize_inspect_score_expands_dict_values_with_metadata() -> None:
    scores = normalize_inspect_score(
        score=InspectScore(
            value={"accuracy": True, "fluency": 0.75},
            answer="answer",
            explanation="two dimensions",
            metadata={"subject": "science"},
        ),
        message_piece_id="piece-1",
        objective="respond",
    )

    assert [(score.score_type, score.score_value) for score in scores] == [
        ("true_false", "True"),
        ("float_scale", "0.75"),
    ]
    assert [score.score_metadata["score_key"] for score in scores] == ["accuracy", "fluency"]
    assert all(score.score_metadata["subject"] == "science" for score in scores)


def test_normalize_inspect_score_rejects_unrepresentable_metadata() -> None:
    with pytest.raises(ValueError, match="metadata values"):
        normalize_inspect_score(
            score=InspectScore(value=True, metadata={"nested": {"not": "representable"}}),
            message_piece_id="piece-1",
            objective="respond",
        )


_MIXED_MAPPING_SOURCE = """
from enum import Enum, auto

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Score, Scorer, Target, accuracy, answer, choice, grouped, match, scorer, stderr
from inspect_ai.solver import Generate, Solver, TaskState, generate, multiple_choice, solver

class DatasetType(Enum):
    MULTIPLE_CHOICE = auto()
    BINARY_CHOICE = auto()
    EXACT_MATCH = auto()

@solver
def dispatch_solver() -> Solver:
    multiple_choice_solve = multiple_choice()
    generate_solve = generate()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        dataset_type = state.metadata["dataset_type"]
        if dataset_type == DatasetType.MULTIPLE_CHOICE:
            return await multiple_choice_solve(state, generate)
        else:
            return await generate_solve(state, generate)
    return solve

@scorer(metrics=[grouped(accuracy(), "dataset_name"), stderr()])
def dispatch_scorer() -> Scorer:
    choice_score = choice()
    answer_score = answer(pattern="word")
    match_score = match(location="end")

    async def score(state: TaskState, target: Target) -> Score | None:
        dataset_type = state.metadata["dataset_type"]
        if dataset_type == DatasetType.MULTIPLE_CHOICE:
            return await choice_score(state, target)
        elif dataset_type == DatasetType.BINARY_CHOICE:
            return await answer_score(state, target)
        else:
            return await match_score(state, target)
    return score

@task
def mixed_mapping() -> Task:
    return Task(
        dataset=MemoryDataset([
            Sample(
                id="mc",
                input="Pick the first.",
                choices=["first", "second"],
                target="A",
                metadata={"dataset_type": DatasetType.MULTIPLE_CHOICE, "dataset_name": "mc"},
            ),
            Sample(
                id="binary",
                input="Return true.",
                target="True",
                metadata={"dataset_type": DatasetType.BINARY_CHOICE, "dataset_name": "binary"},
            ),
            Sample(
                id="exact",
                input="Return final.",
                target="final",
                metadata={"dataset_type": DatasetType.EXACT_MATCH, "dataset_name": "exact"},
            ),
        ]),
        solver=dispatch_solver(),
        scorer=dispatch_scorer(),
        config=GenerateConfig(temperature=0),
    )
"""


def test_metadata_dispatched_mixed_mapping_compiles_and_runs_natively(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mixed_mapping.py").write_text(_MIXED_MAPPING_SOURCE, encoding="utf-8")
    target = _ConfigurableScriptedTarget(response="ANSWER: True")

    execution = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="mixed_mapping.py@mixed_mapping",
            verify_source_revision=False,
            target=target,
        )
    )

    assert [case.scorers[0].kind for case in execution.loaded.suite.cases] == [
        "inspect_choice",
        "inspect_pattern",
        "inspect_text",
    ]
    assert [attempt.task_result.scores[0].score_value for attempt in execution.result.attempts] == [
        "False",
        "True",
        "False",
    ]
    assert all(
        case.scorers[0].metrics[0].group_by == ("metadata.dataset_name",) for case in execution.loaded.suite.cases
    )
    grouped_values = {
        key: value
        for groups in execution.result.aggregate.grouped_metric_values.values()
        for key, value in groups.items()
    }
    assert grouped_values == {
        "metadata.dataset_name=binary": 1.0,
        "metadata.dataset_name=exact": 0.0,
        "metadata.dataset_name=mc": 0.0,
    }
    assert target.temperatures == [0.0, 0.0, 0.0]


def test_metadata_dispatch_rejects_nonliteral_component_options_without_executing_them(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = _MIXED_MAPPING_SOURCE.replace(
        "class DatasetType(Enum):",
        "def opaque_pattern():\n    raise RuntimeError('opaque callback executed')\n\nclass DatasetType(Enum):",
    ).replace('answer_score = answer(pattern="word")', "answer_score = answer(pattern=opaque_pattern())")
    (package / "mixed_mapping.py").write_text(source, encoding="utf-8")

    with pytest.raises(UnsupportedInspectFeatureError, match="metadata dispatch component literal"):
        load_inspect_eval(
            source_root=root,
            task_spec="mixed_mapping.py@mixed_mapping",
            verify_source_revision=False,
        )


@pytest.mark.parametrize(
    ("completion", "expected"),
    [
        ("ANSWER: false\nANSWER: True", "True"),
        ("reasoning\nANSWER: True.", "True"),
        ("ANSWER: Truth", "False"),
        ("True", "False"),
    ],
)
def test_answer_word_uses_last_pinned_answer_token(
    completion: str,
    expected: str,
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "answer_eval.py").write_text(
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.scorer import answer\n"
        "from inspect_ai.solver import generate\n"
        "@task\n"
        "def answer_eval():\n"
        "    return Task(dataset=MemoryDataset([Sample(input='answer', target='True')]), "
        "solver=generate(), scorer=answer(pattern='word'))\n",
        encoding="utf-8",
    )

    execution = asyncio.run(
        run_inspect_eval_async(
            source_root=root,
            task_spec="answer_eval.py@answer_eval",
            verify_source_revision=False,
            target=_ScriptedTarget(response=completion),
        )
    )

    assert execution.result.attempts[0].task_result.scores[0].score_value == expected


def test_verified_derived_json_cache_is_offline_safe_and_tamper_evident(tmp_path: Path) -> None:
    root = tmp_path / "source"
    package = root / "src" / "inspect_evals"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    cache_root = tmp_path / "cache"
    source_asset = cache_root / "fixture" / "records.json"
    source_asset.parent.mkdir(parents=True)
    source_bytes = b'{"questions":[{"question":"Q?","answers":{"A":"one","B":"two"},"solution":"A"}]}'
    source_asset.write_bytes(source_bytes)
    digest = hashlib.sha256(source_bytes).hexdigest()
    (package / "derived.py").write_text(
        "import json\n"
        "from inspect_ai import Task, task\n"
        "from inspect_ai.dataset import Sample\n"
        "from inspect_ai.scorer import match\n"
        "from inspect_ai.solver import generate\n"
        "from inspect_evals.constants import INSPECT_EVALS_CACHE_PATH\n"
        "from inspect_evals.utils import load_json_dataset\n"
        "from inspect_evals.utils.download import download_and_verify\n"
        f"EXPECTED = '{digest}'\n"
        "def record_to_sample(record):\n"
        "    return Sample(input=record['question'], target=record['solution'])\n"
        "@task\n"
        "def derived():\n"
        "    source = INSPECT_EVALS_CACHE_PATH / 'fixture' / 'records.json'\n"
        "    derived = source.with_suffix('.jsonl')\n"
        "    download_and_verify('https://example.test/records.json', EXPECTED, source)\n"
        "    if not derived.exists():\n"
        "        data = json.loads(source.read_text(encoding='utf-8'))\n"
        "        derived.write_text(''.join(json.dumps(row) + '\\n' for row in data['questions']), encoding='utf-8')\n"
        "    return Task(dataset=load_json_dataset(str(derived), 'fixture', record_to_sample, auto_id=True), "
        "solver=generate(), scorer=match())\n",
        encoding="utf-8",
    )

    first = load_inspect_eval(
        source_root=root,
        task_spec="derived.py@derived",
        verify_source_revision=False,
        inspect_evals_cache_dir=cache_root,
    )
    second = load_inspect_eval(
        source_root=root,
        task_spec="derived.py@derived",
        verify_source_revision=False,
        inspect_evals_cache_dir=cache_root,
    )

    assert first.task.dataset.provenance == second.task.dataset.provenance
    assert (cache_root / "fixture" / "records.jsonl.sha256").is_file()

    derived_path = cache_root / "fixture" / "records.jsonl"
    derived_path.write_text(
        '{"question":"Q?","answers":{"B":"two","A":"one"},"solution":"A"}\n',
        encoding="utf-8",
    )
    derived_path.with_name(f"{derived_path.name}.sha256").write_text(
        f"{hashlib.sha256(derived_path.read_bytes()).hexdigest()}\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="rows do not match"):
        load_inspect_eval(
            source_root=root,
            task_spec="derived.py@derived",
            verify_source_revision=False,
            inspect_evals_cache_dir=cache_root,
        )
