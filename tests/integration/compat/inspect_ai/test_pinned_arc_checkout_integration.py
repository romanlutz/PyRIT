# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from pyrit.compat.inspect_ai import PINNED_INSPECT_EVALS_PROFILE, load_inspect_eval, run_inspect_eval_async
from pyrit.compat.inspect_ai.catalog import build_inspect_catalog, check_inspect_catalog_regression
from pyrit.models import Message, MessagePiece, TargetResponseMetadata, ToolCallRequest
from pyrit.prompt_target import (
    OpenAIChatRequestOptions,
    PromptTarget,
    TargetCapabilities,
    TargetConfiguration,
    TargetRequestOptions,
)

if TYPE_CHECKING:
    from pyrit.executor.capability import ToolDeclaration, ToolExecutionPolicy


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        subprocess.run(["docker", "compose", "version"], check=True, capture_output=True, timeout=10)
        subprocess.run(["docker", "info"], check=True, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return True


class _RequestOptionsFactory:
    def build_request_options(
        self,
        *,
        declarations: tuple[ToolDeclaration, ...],
        execution_policy: ToolExecutionPolicy,
    ) -> TargetRequestOptions:
        del declarations, execution_policy
        return TargetRequestOptions()


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

    def __init__(self, *, response: str = "ANSWER: B") -> None:
        super().__init__()
        self._response = response
        self.received: list[list[Message]] = []
        self.temperatures: list[float | None] = []

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
        self.received.append(normalized_conversation)
        response = Message.from_prompt(prompt=self._response, role="assistant")
        conversation_id = normalized_conversation[-1].conversation_id
        for piece in response.message_pieces:
            piece.conversation_id = conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="pinned-arc-integration",
                stop_reason="completed",
                provider_stop_reason="completed",
            )
        )
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        return


class _SubmitTarget(PromptTarget):
    _DEFAULT_CONFIGURATION = TargetConfiguration(
        capabilities=TargetCapabilities(
            supports_multi_turn=True,
            supports_multi_message_pieces=True,
            supports_system_prompt=True,
            supports_editable_history=True,
            supports_external_tool_execution=True,
            input_modalities=frozenset({frozenset({"text"}), frozenset({"function_call_output"})}),
            output_modalities=frozenset({frozenset({"text"}), frozenset({"function_call"})}),
        )
    )

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        response = Message(
            message_pieces=[
                MessagePiece(
                    role="assistant",
                    original_value=ToolCallRequest(
                        call_id="submit-1",
                        name="submit",
                        arguments='{"answer":"not-the-flag"}',
                    ).to_json(),
                    original_value_data_type="function_call",
                    converted_value_data_type="function_call",
                )
            ]
        )
        for piece in response.message_pieces:
            piece.conversation_id = normalized_conversation[-1].conversation_id
        self._record_response_metadata(
            metadata=TargetResponseMetadata(
                provider_response_id="pinned-in-house-integration",
                stop_reason="tool_calls",
                provider_stop_reason="tool_calls",
            )
        )
        return [response]

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        del normalized_conversation


@pytest.mark.run_only_if_all_tests
async def test_user_supplied_pinned_checkout_runs_unchanged_arc_source(sqlite_instance) -> None:
    del sqlite_instance
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")
    records: list[dict[str, Any]] = [
        {
            "id": "integration-arc",
            "question": "Which number is even?",
            "choices": {"label": ["A", "B"], "text": ["3", "4"]},
            "answerKey": "B",
        }
    ]

    execution = await run_inspect_eval_async(
        source_root=Path(source_value),
        task_spec="arc/arc.py@arc_challenge",
        target=_ArcTarget(),
        dataset_loader=lambda *args, **kwargs: records,
    )

    assert execution.loaded.report.source_revision_verified is True
    assert execution.loaded.report.source_revision == PINNED_INSPECT_EVALS_PROFILE.inspect_evals_revision
    assert execution.loaded.suite.cases[0].scorers[0].config["expected_value"] == "B"
    task_result = execution.result.attempts[0].task_result
    assert task_result is not None
    assert task_result.scores[0].score_value == "True"


_BBH_MULTIPLE_CHOICE = (
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
)
_BBH_BINARY = (
    "boolean_expressions",
    "causal_judgement",
    "formal_fallacies",
    "navigate",
    "sports_understanding",
    "web_of_lies",
)
_BBH_EXACT = ("multistep_arithmetic_two", "object_counting", "word_sorting")
_BBH_DYCK = ("dyck_languages",)
_BBH_REVISION = "76eaa8c29ad448752cd44201a1246618e2454cac"


def _bbh_injected_records() -> dict[str, list[dict[str, object]]]:
    subsets = (*_BBH_MULTIPLE_CHOICE, *_BBH_BINARY, *_BBH_EXACT, *_BBH_DYCK)
    records: dict[str, list[dict[str, object]]] = {
        f"Joschka/big_bench_hard|few_shot_prompts|few_shot_prompts|{_BBH_REVISION}": [
            {
                "dataset_name": name,
                "answer_only_prompt": f"{name} answer-only example",
                "chain_of_thought_prompt": f"{name} chain-of-thought example",
            }
            for name in subsets
        ]
    }
    for name in subsets:
        record: dict[str, object] = {"question": f"Question for {name}"}
        if name in _BBH_MULTIPLE_CHOICE:
            record.update(
                {
                    "choices": {"label": ["A)", "B)"], "text": ["first", "second"]},
                    "target": "B",
                }
            )
        elif name in _BBH_BINARY:
            record["target"] = "True"
        elif name in _BBH_DYCK:
            record["target"] = ")"
        else:
            record["target"] = "final"
        records[f"Joschka/big_bench_hard|{name}|{name}|{_BBH_REVISION}"] = [record]
    return records


@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize("prompt_type", ["zero_shot", "answer_only", "chain_of_thought"])
def test_user_supplied_pinned_checkout_constructs_every_bbh_mapping(
    prompt_type: str,
) -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="bbh/bbh.py@bbh",
        task_parameters={"subset_name": None, "prompt_type": prompt_type},
        dataset_records=_bbh_injected_records(),
    )

    assert len(loaded.suite.cases) == 27
    assert len({case.case_id for case in loaded.suite.cases}) == 27
    scorer_kinds = [case.scorers[0].kind for case in loaded.suite.cases]
    assert scorer_kinds.count("inspect_choice") == 17
    assert scorer_kinds.count("inspect_pattern") == 6
    assert scorer_kinds.count("inspect_text") == 4
    assert all(case.source is not None and case.source.source_id for case in loaded.suite.cases)
    assert all(case.scorers[0].metrics[0].group_by == ("metadata.dataset_name",) for case in loaded.suite.cases)


@pytest.mark.run_only_if_all_tests
async def test_user_supplied_pinned_bbh_runs_correct_and_incorrect_outputs(sqlite_instance) -> None:
    del sqlite_instance
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")
    records = _bbh_injected_records()
    source_root = Path(source_value)

    correct = await run_inspect_eval_async(
        source_root=source_root,
        task_spec="bbh/bbh.py@bbh",
        task_parameters={"subset_name": "boolean_expressions", "prompt_type": "answer_only"},
        dataset_records=records,
        target=_ArcTarget(response="ANSWER: True"),
    )
    assert correct.result.attempts[0].task_result.scores[0].score_value == "True"

    incorrect = await run_inspect_eval_async(
        source_root=source_root,
        task_spec="bbh/bbh.py@bbh",
        task_parameters={"subset_name": "boolean_expressions", "prompt_type": "answer_only"},
        dataset_records=records,
        target=_ArcTarget(response="ANSWER: False"),
    )

    assert incorrect.result.attempts[0].task_result.scores[0].score_value == "False"


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_checkout_reuses_real_bbh_cache_offline() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    cache_value = os.getenv("PYRIT_INSPECT_EVALS_CACHE_DIR")
    if not source_value or not cache_value or os.getenv("PYRIT_INSPECT_EVALS_REAL_DATA") != "1":
        pytest.skip("Set the pinned source/cache variables and PYRIT_INSPECT_EVALS_REAL_DATA=1.")
    source_root = Path(source_value)
    cache_root = Path(cache_value)
    acquire = os.getenv("PYRIT_INSPECT_EVALS_ALLOW_NETWORK") == "1"

    first = load_inspect_eval(
        source_root=source_root,
        task_spec="bbh/bbh.py@bbh",
        task_parameters={"subset_name": None, "prompt_type": "answer_only"},
        inspect_evals_cache_dir=cache_root,
        allow_network=acquire,
    )
    offline = load_inspect_eval(
        source_root=source_root,
        task_spec="bbh/bbh.py@bbh",
        task_parameters={"subset_name": None, "prompt_type": "answer_only"},
        inspect_evals_cache_dir=cache_root,
        allow_network=False,
    )

    assert len(first.suite.cases) == 6509
    assert first.task.dataset.provenance == offline.task.dataset.provenance


@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize(
    "dataset_name",
    [
        "winogrande_debiased",
        "winogrande_l",
        "winogrande_m",
        "winogrande_s",
        "winogrande_xl",
        "winogrande_xs",
    ],
)
@pytest.mark.parametrize("fewshot", [0, 1])
def test_user_supplied_pinned_checkout_constructs_winogrande_variants(
    dataset_name: str,
    fewshot: int,
) -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")
    revision = "01e74176c63542e6b0bcb004dcdea22d94fb67b5"
    records = {
        f"allenai/winogrande|{dataset_name}|validation|{revision}": [
            {
                "sentence": "Sarah thanked Maria because _ helped.",
                "option1": "Sarah",
                "option2": "Maria",
                "answer": "2",
            }
        ],
        f"allenai/winogrande|{dataset_name}|train|{revision}": [
            {
                "sentence": "The dog chased the cat because _ ran.",
                "option1": "the dog",
                "option2": "the cat",
                "answer": "2",
            }
        ],
    }

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="winogrande/winogrande.py@winogrande",
        task_parameters={"dataset_name": dataset_name, "fewshot": fewshot},
        dataset_records=records,
    )

    case = loaded.suite.cases[0]
    assert case.limits.max_output_tokens == 64
    assert [message.role for message in case.messages] == (["system", "user"] if fewshot else ["user"])
    assert case.scorers[0].kind == "inspect_choice"
    assert "[BLANK]" in case.messages[-1].content


@pytest.mark.run_only_if_all_tests
async def test_user_supplied_pinned_vstar_preserves_and_preflights_image_content(
    tmp_path: Path,
    sqlite_instance,
) -> None:
    del sqlite_instance
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")
    revision = "d9ae62c903da0c98336e85c5ee89cd863b04b4da"
    cache_root = tmp_path / "cache"
    snapshot = cache_root / ".pyrit" / "inspect-compat" / "staged-snapshots" / "craigwu--vstar_bench" / revision
    image = snapshot / "direct_attributes" / "sample.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"\xff\xd8\xff\xd9")
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "repo_id": "craigwu/vstar_bench",
                "repo_type": "dataset",
                "revision": revision,
                "files": {
                    "direct_attributes/sample.jpg": hashlib.sha256(image.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {
            "image": "direct_attributes/sample.jpg",
            "text": (
                "What material is shown?\n(A) rubber\n(B) cotton\n"
                "Answer with the option's letter from the given choices directly."
            ),
            "category": "direct_attributes",
            "question_id": "vstar-1",
            "label": "A",
        }
    ]

    with pytest.raises(ValueError, match="per-file SHA256"):
        load_inspect_eval(
            source_root=Path(source_value),
            task_spec="vstar_bench/vstar_bench.py@vstar_bench_attribute_recognition",
            dataset_records=records,
            inspect_evals_cache_dir=tmp_path / "missing-cache",
        )

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="vstar_bench/vstar_bench.py@vstar_bench_attribute_recognition",
        dataset_records=records,
        inspect_evals_cache_dir=cache_root,
    )

    parts = loaded.suite.cases[0].messages[0].parts
    assert parts is not None
    assert [part.data_type for part in parts] == ["text", "url"]
    assert parts[1].content.startswith("data:image/jpeg;base64,")

    outside = cache_root / "outside.jpg"
    outside.write_bytes(b"\xff\xd8\xff\xd9")
    escaped_records = [{**records[0], "image": str(outside)}]
    with pytest.raises(ValueError, match="trusted staging root"):
        load_inspect_eval(
            source_root=Path(source_value),
            task_spec="vstar_bench/vstar_bench.py@vstar_bench_attribute_recognition",
            dataset_records=escaped_records,
            inspect_evals_cache_dir=cache_root,
        )

    target = _ArcTarget(response="ANSWER: A")
    with pytest.raises(ValueError, match=r"\{text, url\}"):
        await run_inspect_eval_async(
            source_root=Path(source_value),
            task_spec="vstar_bench/vstar_bench.py@vstar_bench_attribute_recognition",
            dataset_records=records,
            inspect_evals_cache_dir=cache_root,
            target=target,
        )
    assert target.received == []

    image.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_inspect_eval(
            source_root=Path(source_value),
            task_spec="vstar_bench/vstar_bench.py@vstar_bench_attribute_recognition",
            dataset_records=records,
            inspect_evals_cache_dir=cache_root,
        )


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_gpqa_cache_constructs_cot_and_answer_only() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    cache_value = os.getenv("PYRIT_INSPECT_EVALS_CACHE_DIR")
    if not source_value or not cache_value or os.getenv("PYRIT_INSPECT_EVALS_REAL_DATA") != "1":
        pytest.skip("Set the pinned source/cache variables and PYRIT_INSPECT_EVALS_REAL_DATA=1.")

    loaded = [
        load_inspect_eval(
            source_root=Path(source_value),
            task_spec="gpqa/gpqa.py@gpqa_diamond",
            task_parameters={"cot": cot},
            inspect_evals_cache_dir=Path(cache_value),
            allow_network=False,
        )
        for cot in (True, False)
    ]

    assert all(len(item.suite.cases) == 198 for item in loaded)
    assert all(item.suite.run_policy.epochs == 4 for item in loaded)
    assert "Think step by step" in loaded[0].suite.cases[0].messages[0].content
    assert "Think step by step" not in loaded[1].suite.cases[0].messages[0].content


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_catalog_classifies_advanced_static_mappings() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    report = build_inspect_catalog(source_root=Path(source_value))
    check_inspect_catalog_regression(report=report)
    families = {family.family: family for family in report.families}

    assert report.api_symbol_count == 262
    assert report.task_factory_count == 249
    assert len(report.families) == 129
    assert families["bbh"].compatibility_status == "supported"
    assert {
        family: families[family].compatibility_status for family in ("cybermetric", "gpqa", "vstar_bench", "winogrande")
    } == {
        "cybermetric": "partial",
        "gpqa": "partial",
        "vstar_bench": "partial",
        "winogrande": "partial",
    }
    assert all(
        "reviewed_static_mapping_audit" in task
        for family in ("bbh", "cybermetric", "gpqa", "vstar_bench", "winogrande")
        for task in families[family].task_factories
    )


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_checkout_constructs_static_multi_dataset_factories() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")
    source_root = Path(source_value)
    bbq_record = {
        "example_id": 0,
        "question_index": "1",
        "question_polarity": "neg",
        "context_condition": "ambig",
        "category": "Age",
        "answer_info": {"ans0": ["unknown", "unknown"], "ans1": ["old", "old"], "ans2": ["young", "young"]},
        "additional_metadata": {
            "subcategory": "fixture",
            "stereotyped_groups": ["old"],
            "version": "1",
            "source": "fixture",
        },
        "context": "An older and younger person arrived.",
        "question": "Who arrived?",
        "ans0": "Unknown",
        "ans1": "The older person",
        "ans2": "The younger person",
        "label": 0,
    }
    secqa_dev = {
        "Question": "Which protocol provides encrypted remote login?",
        "A": "SSH",
        "B": "FTP",
        "C": "HTTP",
        "D": "Telnet",
        "Answer": "A",
    }
    secqa_test = {
        "Question": "Which property prevents unauthorized disclosure?",
        "A": "Availability",
        "B": "Confidentiality",
        "C": "Integrity",
        "D": "Non-repudiation",
        "Answer": "B",
    }

    bbq = load_inspect_eval(
        source_root=source_root,
        task_spec="bbq/bbq.py@bbq",
        task_parameters={"subsets": "Age", "shuffle": False},
        dataset_records={
            "heegyu/bbq|Age|test|5d6faae52070aa5eb71b46d1c0723d3ba7930209": [bbq_record],
        },
    )
    secqa = load_inspect_eval(
        source_root=source_root,
        task_spec="sec_qa/sec_qa.py@sec_qa_v1_5_shot",
        dataset_records={
            "zefang-liu/secqa|secqa_v1|dev|d00a07484283be5602e2bae36dbefdaaf555a9fb": [secqa_dev],
            "zefang-liu/secqa|secqa_v1|test|d00a07484283be5602e2bae36dbefdaaf555a9fb": [secqa_test],
        },
    )

    assert bbq.report.source_revision_verified is True
    assert bbq.suite.cases[0].case_id == "Age_00000"
    assert bbq.suite.cases[0].source is not None
    assert bbq.suite.cases[0].source.metadata["dataset"]["name"] == "Age"
    assert secqa.report.source_revision_verified is True
    assert secqa.suite.cases[0].case_id.startswith("secqa_")
    assert secqa.suite.cases[0].messages[0].role == "system"
    assert secqa_dev["Question"] in secqa.suite.cases[0].messages[0].content
    assert secqa_test["Question"] in secqa.suite.cases[0].messages[-1].content


@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize(
    ("task_spec", "task_parameters", "records"),
    [
        (
            "bbq/bbq.py@bbq",
            {"subsets": ["Age", "Religion"], "shuffle": False},
            [
                {
                    "example_id": 0,
                    "question_index": "1",
                    "question_polarity": "neg",
                    "context_condition": "ambig",
                    "category": "Age",
                    "answer_info": {},
                    "additional_metadata": {},
                    "context": "context",
                    "question": "question",
                    "ans0": "unknown",
                    "ans1": "first",
                    "ans2": "second",
                    "label": 0,
                }
            ],
        ),
        (
            "sec_qa/sec_qa.py@sec_qa_v1_5_shot",
            {},
            [{"Question": "question", "A": "a", "B": "b", "C": "c", "D": "d", "Answer": "A"}],
        ),
    ],
)
def test_user_supplied_pinned_checkout_rejects_unkeyed_multi_dataset_records(
    task_spec: str,
    task_parameters: dict[str, object],
    records: list[dict[str, object]],
) -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    with pytest.raises(ValueError, match="Unkeyed injected dataset records"):
        load_inspect_eval(
            source_root=Path(source_value),
            task_spec=task_spec,
            task_parameters=task_parameters,
            dataset_records=records,
        )


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_checkout_constructs_unchanged_in_house_ctf_source() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf",
        task_parameters={"challenges": "ssh", "epochs": 1},
    )

    assert loaded.report.source_revision_verified is True
    assert loaded.suite.cases[0].case_id == "ssh"
    assert loaded.suite.cases[0].scorers[0].kind == "inspect_check_flag"


@pytest.mark.run_only_if_all_tests
def test_user_supplied_pinned_checkout_constructs_unchanged_intercode_ctf_source() -> None:
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    cache_value = os.getenv("PYRIT_INSPECT_EVALS_CACHE_DIR")
    if not source_value or not cache_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT and PYRIT_INSPECT_EVALS_CACHE_DIR to exact pinned inputs.")

    loaded = load_inspect_eval(
        source_root=Path(source_value),
        task_spec="gdm_intercode_ctf/gdm_intercode_ctf.py@gdm_intercode_ctf",
        task_parameters={"sample_ids": [2]},
        inspect_evals_cache_dir=Path(cache_value),
    )

    assert loaded.report.source_revision_verified is True
    assert loaded.suite.cases[0].source is not None
    assert loaded.suite.cases[0].source.source_id == "2"
    assert [tool.declaration.name for tool in loaded.suite.cases[0].tools] == ["bash", "python", "submit"]


@pytest.mark.docker
@pytest.mark.run_only_if_all_tests
@pytest.mark.skipif(not _docker_available(), reason="Docker CLI, Compose v2, and a running daemon are required.")
async def test_user_supplied_pinned_in_house_ctf_runs_real_docker(sqlite_instance) -> None:
    del sqlite_instance
    source_value = os.getenv("PYRIT_INSPECT_EVALS_SOURCE_ROOT")
    if not source_value:
        pytest.skip("Set PYRIT_INSPECT_EVALS_SOURCE_ROOT to an exact pinned inspect_evals checkout.")

    execution = await run_inspect_eval_async(
        source_root=Path(source_value),
        task_spec="gdm_in_house_ctf/gdm_in_house_ctf.py@gdm_in_house_ctf",
        task_parameters={"challenges": "ssh", "epochs": 1},
        target=_SubmitTarget(),
        request_options_factory=_RequestOptionsFactory(),
    )

    attempt = execution.result.attempts[0]
    assert attempt.task_result is not None, attempt.error
    assert attempt.task_result.tool_calls == 1
    assert attempt.task_result.scores[0].score_value == "False"
