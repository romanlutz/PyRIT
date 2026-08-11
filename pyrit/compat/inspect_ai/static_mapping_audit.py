# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Reviewed factory-level evidence for advanced static mapping compatibility."""

from __future__ import annotations

from typing import Any

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

_CYBERMETRIC_FILES = {
    "cybermetric_80": (
        "CyberMetric-80",
        "1624aeecc54761198bff4828442ce4a10de6cb87c9da3298088b34ae11e15ba0",
    ),
    "cybermetric_500": (
        "CyberMetric-500",
        "036747c989da9f38f39a6b33fa2d5ab14147c928df0274217bbecab20be88faa",
    ),
    "cybermetric_2000": (
        "CyberMetric-2000",
        "3ccc4d425bc4e74d27e0e9790d62369e4626e325d2b661d851c61b1648a0cd4a",
    ),
    "cybermetric_10000": (
        "CyberMetric-10000",
        "4e35bb62c73b60bd27e54512b2ace6f9286ff2921a45a3d5fe2da22401e04bbb",
    ),
}


def reviewed_static_mapping_audit(*, family: str, factory: str) -> dict[str, Any] | None:
    """Return reviewed evidence for one pinned advanced mapping factory."""
    if family == "bbh" and factory == "bbh":
        return _bbh_audit()
    if family == "cybermetric" and factory in _CYBERMETRIC_FILES:
        return _cybermetric_audit(factory=factory)
    if family == "gpqa" and factory == "gpqa_diamond":
        return _gpqa_audit()
    if family == "vstar_bench" and factory.startswith("vstar_bench_"):
        return _vstar_audit(factory=factory)
    if family == "winogrande" and factory == "winogrande":
        return _winogrande_audit()
    return None


def _bbh_audit() -> dict[str, Any]:
    return {
        "audit_version": 2,
        "factory_arguments": {
            "subset_name": [None, *_BBH_MULTIPLE_CHOICE, *_BBH_BINARY, *_BBH_EXACT, *_BBH_DYCK],
            "prompt_type": ["zero_shot", "answer_only", "chain_of_thought"],
            "canonical_combination_count": 84,
        },
        "dataset_sources": [
            {
                "kind": "huggingface",
                "repository": "Joschka/big_bench_hard",
                "revision": _BBH_REVISION,
                "configs_and_splits": [
                    *_BBH_MULTIPLE_CHOICE,
                    *_BBH_BINARY,
                    *_BBH_EXACT,
                    *_BBH_DYCK,
                    "few_shot_prompts",
                ],
                "license": "MIT",
                "gated": False,
                "representative_sha256": {
                    "date_understanding": "2d29ef88c0b1df515057c1a3eceb12d44ff329613c253d9759d10a3fb38835bb",
                    "few_shot_prompts": "fb6255bc800b9ebe86b630494d135a98150be6a025410a7897b3484a36b62316",
                },
            }
        ],
        "record_mapping": {
            "multiple_choice": {"input": "question", "choices": "choices.text", "target": "target"},
            "binary": {"input": "question", "target": "target.strip()"},
            "exact": {"input": "question", "target": "target"},
            "dyck": {"input": "question", "target": "'ANSWER: ' + target"},
            "id": "auto_id then '<subset>_<ordinal:03d>'",
            "metadata": ["dataset_type", "dataset_name"],
            "filtered_records": {"ruin_names": 2},
        },
        "solver_graph": {
            "dispatch_key": "dataset_type",
            "MULTIPLE_CHOICE": "multiple_choice",
            "default": "generate",
            "task_generate_config": {"temperature": 0},
        },
        "few_shot": {
            "source": "few_shot_prompts",
            "answer_only": "dataset-stored three-shot prompt",
            "chain_of_thought": "dataset-stored three-shot prompt",
            "zero_shot": "no examples",
        },
        "messages_and_content": {
            "roles": ["user"],
            "multiple_choice": "BBH prefix/query followed by canonical labeled choices",
            "generated": "format patch, optional examples, then QUESTION",
            "content_types": ["text"],
        },
        "staging": {"images": False, "files": False},
        "scorers": {
            "MULTIPLE_CHOICE": "choice",
            "BINARY_CHOICE": "answer(pattern='word')",
            "default": "match(location='end')",
        },
        "metrics_and_reducers": ["grouped(accuracy, dataset_name)", "stderr", "implicit epoch mean"],
        "external_requirements": ["exact-revision Hugging Face records", "offline validated dataset cache"],
    }


def _cybermetric_audit(*, factory: str) -> dict[str, Any]:
    dataset_name, digest = _CYBERMETRIC_FILES[factory]
    revision = "205262cdf5022ba890e792efd176fb19d42913fa"
    return {
        "audit_version": 2,
        "factory_arguments": {},
        "dataset_sources": [
            {
                "kind": "direct_url",
                "url": f"https://raw.githubusercontent.com/cybermetric/CyberMetric/{revision}/{dataset_name}-v1.json",
                "revision": revision,
                "sha256": digest,
                "license": None,
            }
        ],
        "record_mapping": {
            "input": "'Question: ' + question + formatted answer-key options",
            "choices": "answer keys in insertion order",
            "target": "solution",
            "id": "auto_id=True overrides the mapper MD5 with one-based ordinals",
        },
        "solver_graph": ["system_message", "multiple_choice"],
        "few_shot": None,
        "messages_and_content": {
            "roles": ["system", "user"],
            "system": "You are a security expert who answers questions.",
            "choice_semantics": "the answer-key choices intentionally duplicate the mapper's textual options",
            "content_types": ["text"],
        },
        "staging": {
            "images": False,
            "files": False,
            "derived_jsonl": "exact rows must match a same-construction SHA256-verified source JSON asset",
        },
        "scorers": ["choice"],
        "metrics_and_reducers": ["accuracy", "stderr", "implicit epoch mean"],
        "external_requirements": ["authorized exact bytes", "verified direct-asset cache"],
    }


def _gpqa_audit() -> dict[str, Any]:
    return {
        "audit_version": 2,
        "factory_arguments": {
            "cot": [True, False],
            "epochs": "positive integer",
            "high_level_domain": ["Biology", "Chemistry", "Physics"],
            "subdomain": [
                "Astrophysics",
                "Chemistry (general)",
                "Condensed Matter Physics",
                "Electromagnetism and Photonics",
                "Genetics",
                "High-energy particle physics",
                "Inorganic Chemistry",
                "Molecular Biology",
                "Optics and Acoustics",
                "Organic Chemistry",
                "Physics (general)",
                "Quantum Mechanics",
                "Relativistic Mechanics",
            ],
        },
        "dataset_sources": [
            {
                "kind": "direct_url",
                "url": "https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv",
                "sha256": "41d1213cd7a4998605a26c2798500652572007161b3a92817ba46b35befcd305",
                "rows": 198,
                "license": "not established for the exact mirrored bytes",
            }
        ],
        "record_mapping": {
            "input": "Question",
            "choices": ["Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2", "Incorrect Answer 3"],
            "target": "A before choice shuffle, then remapped",
            "id": "Record ID",
            "metadata": ["High-level domain", "Subdomain"],
        },
        "solver_graph": ["multiple_choice(cot=<factory argument>)"],
        "few_shot": None,
        "messages_and_content": {"roles": ["user"], "content_types": ["text"]},
        "staging": {"images": False, "files": False},
        "scorers": ["choice"],
        "metrics_and_reducers": ["accuracy", "stderr", "mean across epochs"],
        "external_requirements": ["authorized checksum-matching CSV", "offline verified direct-asset cache"],
        "reproducibility_note": "shuffle_choices=True uses an unseeded pinned factory path",
    }


def _vstar_audit(*, factory: str) -> dict[str, Any]:
    category, samples = (
        ("direct_attributes", 115) if factory == "vstar_bench_attribute_recognition" else ("relative_position", 76)
    )
    return {
        "audit_version": 2,
        "factory_arguments": {},
        "dataset_sources": [
            {
                "kind": "huggingface_snapshot",
                "repository": "craigwu/vstar_bench",
                "revision": "d9ae62c903da0c98336e85c5ee89cd863b04b4da",
                "split": "test",
                "category": category,
                "expected_samples": samples,
                "license": None,
            }
        ],
        "record_mapping": {
            "input": ["question text", "snapshot-local image"],
            "choices": "option lines stripped of '(X) ' prefixes",
            "target": "label",
            "id": "question_id",
            "metadata": ["category"],
        },
        "solver_graph": ["multiple_choice"],
        "few_shot": None,
        "messages_and_content": {
            "roles": ["user"],
            "ordered_content_types": ["text", "image"],
        },
        "staging": {
            "images": "trusted snapshot path embedded as a content-hashed data URI",
            "manifest": "staged snapshots require exact identity and per-file SHA256 values",
            "reject": ["traversal", "missing file", "unsupported MIME", "image-incompatible target"],
        },
        "scorers": ["choice"],
        "metrics_and_reducers": ["accuracy", "stderr", "implicit epoch mean"],
        "external_requirements": ["authorized image snapshot", "image-capable target"],
    }


def _winogrande_audit() -> dict[str, Any]:
    return {
        "audit_version": 2,
        "factory_arguments": {
            "dataset_name": [
                "winogrande_debiased",
                "winogrande_l",
                "winogrande_m",
                "winogrande_s",
                "winogrande_xl",
                "winogrande_xs",
            ],
            "fewshot": "non-negative integer",
            "fewshot_seed": "integer",
            "fewshot_shuffle": [True, False],
            "shuffle": [True, False],
        },
        "dataset_sources": [
            {
                "kind": "huggingface",
                "repository": "allenai/winogrande",
                "revision": "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
                "splits": ["validation", "train when fewshot > 0"],
                "configs": 6,
                "validation_sha256": "f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1",
                "license": "More Information Needed",
            }
        ],
        "record_mapping": {
            "input": "'Sentence: ' + sentence with '_' replaced by '[BLANK]'",
            "choices": ["option1", "option2"],
            "target": {"1": "A", "2": "B"},
            "id": "winogrande_<MD5(sentence)[:8]>",
            "duplicate_filter": "keep first stable ID after shuffle/limit",
        },
        "solver_graph": ["optional system_message", "multiple_choice(custom template, shuffle=False)"],
        "few_shot": {
            "source": "train split",
            "representation": "one canonical system message containing ordered rendered examples",
            "default": {"count": 5, "seed": 42, "shuffle": True},
        },
        "messages_and_content": {
            "roles": ["system when fewshot > 0", "user"],
            "content_types": ["text"],
        },
        "staging": {"images": False, "files": False},
        "scorers": ["choice"],
        "metrics_and_reducers": ["accuracy", "stderr", "implicit epoch mean"],
        "generation": {"max_tokens": 64},
        "external_requirements": ["authorized exact-revision validation records", "train records for few-shot mode"],
    }
