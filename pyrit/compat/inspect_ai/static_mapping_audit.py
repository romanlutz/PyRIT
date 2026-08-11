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

_CODE_EVAL_AUDITS: dict[tuple[str, str], dict[str, Any]] = {
    ("humaneval", "humaneval"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "repository": "openai/openai_humaneval",
                "revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                "split": "test",
                "rows": 164,
                "offline_manifest_sha256": "b44ab444e38f59ff01ce05f966adf79de83731bd766797216759a369d2eda9a1",
                "license": "MIT",
            }
        ],
        "factory_arguments": {
            "solver": "default-only",
            "instruction_prompt": "string",
            "scorer": "default-only",
            "sandbox": ["docker"],
        },
        "candidate_extraction": (
            "first python fence, otherwise first generic fence; fenced full functions are reduced with pinned body "
            "slicing"
        ),
        "harness": "prompt + completion + test + check(entry_point)",
        "runtime": {"language": "python", "timeout_seconds": 30, "dependencies": ["standard library"]},
        "epochs": 1,
        "metrics": ["accuracy", "stderr"],
        "meaningful_variants": {
            "supported": ["default solver/scorer", "arbitrary instruction_prompt string", "docker sandbox"],
            "rejected_before_execution": ["custom solver", "custom scorer", "non-docker sandbox"],
        },
        "support": "supported",
    },
    ("mbpp", "mbpp"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "repository": "google-research-datasets/mbpp",
                "revision": "4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                "configs": ["full/prompt", "sanitized/test"],
                "rows": {"sanitized/test": 257, "full/prompt_selected": [2, 3, 4]},
                "offline_manifest_sha256": "cf3fe2178321c7b481c5319abc72a061a2df89d16ba9b303844de3f1e94bb2a3",
                "license": "CC-BY-4.0",
            }
        ],
        "factory_arguments": {"temperature": "finite float"},
        "candidate_extraction": "first python fence or raw completion",
        "harness": "completion plus all target assertions; test_imports is intentionally not injected",
        "runtime": {"language": "python", "timeout_seconds": 30, "dependencies": ["standard library"]},
        "epochs": 5,
        "reducers": ["mean", "pass_at_1", "pass_at_2", "pass_at_5"],
        "meaningful_variants": {
            "supported": ["finite temperature values"],
            "fixed": ["five epochs", "mean/pass@1/pass@2/pass@5 reducers", "sanitized test split"],
        },
        "support": "supported",
    },
    ("apps", "apps"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "repository": "codeparrot/apps",
                "revision": "21e74ddf8de1a21436da12e3e653065c5213e9d1",
                "split": "test",
                "license": "MIT",
            }
        ],
        "factory_arguments": {
            "temperature": "float",
            "num_epochs": "positive int",
            "epoch_reducer": ["mean", "max", "median", "mode", "pass_at_k"],
            "level": ["interview", "competition", "introductory"],
            "token_limit": "positive int",
        },
        "candidate_extraction": "first python fence or raw completion",
        "harness": (
            "solution(one_argument) assertions synthesized from paired inputs/outputs; fn_name and starter_code are "
            "intentionally ignored; source truncated at 100000 characters"
        ),
        "runtime": {"language": "python", "timeout_seconds": 120, "dependencies": ["standard library"]},
        "meaningful_variants": {
            "supported": [
                "interview/competition/introductory levels",
                "positive num_epochs",
                "mean/max/pass_at_k reducers",
                "temperature and token_limit overrides",
            ],
            "unsupported": ["median reducer", "mode reducer"],
            "unsupported_reason": "The shared typed reducer framework does not represent pinned median/mode semantics.",
        },
        "support": "partial",
    },
    ("bigcodebench", "bigcodebench"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "repository": "bigcode/bigcodebench",
                "revision": "b74c0d0bf70d2c0bc459be537895cca163007f1a",
                "default_config": "v0.1.2",
                "license": "Apache-2.0",
            }
        ],
        "factory_arguments": {
            "solver": "default-only",
            "instruction_prompt": "string",
            "scorer": "default-only",
            "sandbox": ["docker"],
            "version": "Hugging Face split name; default v0.1.2",
            "subset": "optional list of integer row offsets",
            "docker_handling": ["default", "force_build", "force_pull"],
        },
        "candidate_extraction": "first python fence, otherwise first generic fence, otherwise raw completion",
        "harness": "completion + unittest harness + unittest.main()",
        "runtime": {
            "language": "python",
            "timeout_seconds": 30,
            "dependencies": "row-specific libs plus the unpinned docker-requirements closure",
            "blocker": "compose uses a mutable :latest image and the runtime dependency closure includes TensorFlow",
        },
        "metrics": ["mean", "std"],
        "meaningful_variants": {
            "unsupported": ["all versions, subsets, and docker_handling modes"],
            "unsupported_reason": "No content-addressed runtime proves the exact dependency environment.",
        },
        "support": "partial",
    },
    ("class_eval", "class_eval"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "repository": "FudanSELab/ClassEval",
                "revision": "fef204b34e221f207f47904ee660bb920d4c5d1d",
                "split": "test",
                "license": "CC-BY-NC-4.0",
            }
        ],
        "factory_arguments": {
            "few_shot": "sets epoch count",
            "few_shot_seed": "unused",
            "docker_handling": "runtime preparation",
        },
        "harness": "complete generated class plus test code",
        "candidate_extraction": "first python fence or raw completion",
        "runtime": {
            "language": "python",
            "timeout_seconds": 30,
            "dependencies": "unpinned docker-requirements closure",
            "blocker": (
                "compose uses a mutable :latest image and the runtime dependency closure is not content-verified"
            ),
        },
        "epochs_and_reducers": "few_shot is used as epoch count with pass_at_<few_shot>",
        "metrics": ["mean", "std"],
        "meaningful_variants": {
            "unsupported": ["all few_shot counts, seeds, and docker_handling modes"],
            "unsupported_reason": "The non-commercial dataset and mutable runtime are not approved exact inputs.",
        },
        "support": "partial",
    },
    ("usaco", "usaco"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "kind": "google_drive_zip",
                "sha256": "e4658a59f63132a0d2e43da9c0a42829d12106a02c4db8413017e6175215bf63",
                "default_rows": 307,
                "license": None,
            }
        ],
        "factory_arguments": {
            "temperature": "finite float",
            "shuffle": [True, False],
            "problem_levels": ["bronze", "silver", "gold", "platinum"],
            "json_basename": ["usaco_subset307", "usaco_v2"],
        },
        "candidate_extraction": "first python fence or raw completion",
        "harness": "Linux Python with file/stdin test cases and stripped stdout comparison",
        "runtime": {
            "language": "python",
            "test_files": ["<n>.in/<n>.out", "I.<n>/O.<n>"],
            "execution": "sequential tests with first-failure short circuit and binary stdin",
            "comparison": "stdout.strip() == expected.strip()",
            "requirements": ["resource module", "per-row CPU limit", "per-row address-space limit"],
            "blocker": "dataset license absent and runtime acquisition not immutable",
        },
        "epochs_and_reducers": {"epochs": 1, "reducers": ["pass_at_1"]},
        "meaningful_variants": {
            "unsupported": ["all levels, dataset basenames, shuffle modes, and temperatures"],
            "unsupported_reason": "Dataset redistribution/use rights and an immutable Linux runtime are unproven.",
        },
        "support": "partial",
    },
    ("vimgolf_challenges", "vimgolf_single_turn"): {
        "audit_version": 1,
        "dataset_sources": [
            {
                "repository": "cybergod-kevin/vimgolf-public-challenges-inspect-eval",
                "revision": "909a17f2b57d5866c077ebac019f6b0f271de919",
                "rows": 612,
                "license": None,
            }
        ],
        "factory_arguments": {},
        "candidate_extraction": "last non-empty response line",
        "harness": (
            "reject empty/non-shorter solutions, tokenize keycodes, replay one-line keys from input.txt, then accept "
            "normalized final content or SHA256 equality"
        ),
        "runtime": {
            "language": "Vim keystrokes",
            "timeout_seconds": 15,
            "files": ["/app/input.txt", "/app/solution.txt"],
            "construction_dependency": "vimgolf Python package",
            "blocker": "unpinned Vim/base/apt runtime, optional unpinned vimgolf package, and absent dataset license",
        },
        "meaningful_variants": {
            "unsupported": ["the parameterless vimgolf_single_turn factory"],
            "unsupported_reason": "Dataset rights and an immutable editor/verifier runtime are unproven.",
        },
        "support": "partial",
    },
}


def reviewed_static_mapping_audit(*, family: str, factory: str) -> dict[str, Any] | None:
    """Return reviewed evidence for one pinned advanced mapping factory."""
    if (audit := _CODE_EVAL_AUDITS.get((family, factory))) is not None:
        return audit
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
