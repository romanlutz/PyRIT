# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Inert, explicitly trusted Inspect Task sources; no model, Docker or private fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from inspect_ai.scorer import includes
from inspect_ai.solver import system_message

import pyrit.executor.benchmark.inspect_eval_source as eval_source
from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory, ResolvedInspectEvalCase
from pyrit.models import EvalSourceKind, SeedObjective, config_hash

if TYPE_CHECKING:
    from inspect_ai import Task

_LOCAL_TASK = """
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
from inspect_ai.solver import TaskState
from inspect_ai.util import ComposeConfig, ComposeService, SandboxEnvironmentSpec

@scorer(metrics=[mean()])
def local_scorer() -> Scorer:
    async def score_async(state: TaskState, target: Target) -> Score:
        return Score(value=1.0 if state.output.completion else 0.0)
    return score_async

def build_task(*, agent_image: str, target_image: str) -> Task:
    sample = Sample(input="local harmless goal", id="local-one", target="answer")
    tmpfs = ["/home/bridge:rw,exec,size=256m", "/workspace:rw,exec,size=64m", "/tmp:rw,exec,size=512m"]
    def service(image: str, mem: str, default: bool = False, bridge: bool = False) -> ComposeService:
        return ComposeService.model_validate({
            "image": image, "pull_policy": "never", "user": "10001:10001", "read_only": True,
            "cap_drop": ["ALL"], "security_opt": ["no-new-privileges:true"], "pids_limit": 64,
            "mem_limit": mem, "cpus": 1.0, "networks": ["only_internal"],
            "tmpfs": tmpfs + (["/var/tmp:rw,exec,nosuid,nodev,size=128m,mode=1777"] if bridge else []),
            **({"x-default": True} if default else {}),
        })
    services = {
        "agent": service(agent_image, "1.25gb", default=True, bridge=True),
        "model-bridge": service(agent_image, "1.25gb", bridge=True),
        "target": service(target_image, "384m"),
    }
    task = Task(
        dataset=[sample],
        scorer=local_scorer(),
        sandbox=SandboxEnvironmentSpec(
            "docker", ComposeConfig(services=services, networks={"only_internal": {"internal": True}})
        ),
        name="local_benign",
        version=1,
    )
    return task
"""
_IMAGE_IDS = {"agent": "sha256:" + "a" * 64, "model-bridge": "sha256:" + "a" * 64, "target": "sha256:" + "b" * 64}
_MULTI_TASK = _LOCAL_TASK.replace(
    "def build_task(*, agent_image: str, target_image: str) -> Task:",
    "def build_task(*, agent_image: str, target_image: str) -> list[Task]:",
).replace(
    "return task",
    """second = Task(
        dataset=[
            Sample(input="local harmless goal", id="local-two-a", target="answer"),
            Sample(input="local harmless goal", id="local-two-b", target="answer"),
        ],
        scorer=local_scorer(),
        sandbox=task.sandbox,
        name="local_second",
        version=1,
    )
    return [task, second]""",
)


def _local_manifest(*, root: Path, task_source: str = _LOCAL_TASK, allow_edit: bool = False) -> str:
    module = root / "task.py"
    module.write_text(task_source, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "name": "local_benign",
        "task_file": "task.py",
        "factory": "build_task",
        "files_sha256": {"task.py": hashlib.sha256(module.read_bytes()).hexdigest()},
        "task_name": "local_benign",
        "task_version": "1",
        "sample_id": "local-one",
        "sample_input_sha256": hashlib.sha256(b"local harmless goal").hexdigest(),
        "sample_target_sha256": hashlib.sha256(b"answer").hexdigest(),
        "scorer_name": "local_scorer",
        "scorer_fingerprint": config_hash({"name": "local_scorer", "params": {}}),
        "setup_fingerprints": [],
        "cleanup_name": None,
        "health_command": ["/bin/true"],
        "allow_initial_input_override": allow_edit,
    }
    data = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (root / EvalSourceFactory.MANIFEST_FILE).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _resolve_local(*, root: Path, revision: str, input_override: str | None = None):
    return EvalSourceFactory.resolve(
        family=None,
        trusted_dir=root,
        trusted_local=True,
        revision_sha256=revision,
        input_override=input_override,
        agent_image="approved/agent:pinned",
        target_image="approved/target:pinned",
        approved_image_ids=_IMAGE_IDS,
    )


def _multi_manifest(*, root: Path, task_source: str = _MULTI_TASK) -> str:
    module = root / "task.py"
    module.write_text(task_source, encoding="utf-8")
    asset = root / "public-asset.txt"
    asset.write_text("inert fixture", encoding="utf-8")
    input_sha = hashlib.sha256(b"local harmless goal").hexdigest()
    target_sha = hashlib.sha256(b"answer").hexdigest()

    def pinned_sample(*, sample_id: str) -> dict[str, str]:
        return {
            "sample_id": sample_id,
            "sample_input_sha256": input_sha,
            "sample_target_sha256": target_sha,
        }

    manifest = {
        "schema_version": 2,
        "name": "local_benign_suite",
        "task_file": "task.py",
        "factory": "build_task",
        "files_sha256": {
            "task.py": hashlib.sha256(module.read_bytes()).hexdigest(),
            "public-asset.txt": hashlib.sha256(asset.read_bytes()).hexdigest(),
        },
        "health_command": ["/bin/true"],
        "tasks": [
            {
                "task_name": name,
                "task_version": "1",
                "scorer_name": "local_scorer",
                "scorer_fingerprint": config_hash({"name": "local_scorer", "params": {}}),
                "setup_fingerprints": [],
                "cleanup_name": None,
                "samples": [pinned_sample(sample_id=sample_id) for sample_id in sample_ids],
            }
            for name, sample_ids in (
                ("local_benign", ("local-one",)),
                ("local_second", ("local-two-a", "local-two-b")),
            )
        ],
    }
    data = json.dumps(manifest, sort_keys=True).encode("utf-8")
    (root / EvalSourceFactory.MANIFEST_FILE).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _resolve_multi(
    *,
    root: Path,
    revision: str,
    input_override: str | None = None,
    approved_image_ids: dict[str, str] | None = None,
) -> tuple[ResolvedInspectEvalCase, ...]:
    return EvalSourceFactory.resolve_cases(
        family=None,
        trusted_dir=root,
        trusted_local=True,
        revision_sha256=revision,
        input_override=input_override,
        agent_image="approved/agent:pinned",
        target_image="approved/target:pinned",
        approved_image_ids=approved_image_ids if approved_image_ids is not None else _IMAGE_IDS,
    )


def test_named_benign_source_derives_one_original_case_without_victim_seed_wrapping() -> None:
    selected = EvalSourceFactory.resolve(
        family="benign_protocol",
        trusted_dir=None,
        trusted_local=False,
        revision_sha256=None,
        input_override=None,
        agent_image="approved/agent:pinned",
        target_image="approved/target:pinned",
        approved_image_ids=_IMAGE_IDS,
    )
    selected.verify_unchanged()
    assert selected.case.package.kind is EvalSourceKind.NAMED
    assert selected.case.sample_id == "benign-ghcp-protocol-1"
    assert selected.case.package.source_sha256 == "eb390d6e730e69ae32d73e185b69ce6e553e4438c5cd58ac76d41dfe525f2f4e"
    assert selected.case.case_id == "dd03a32fae8c89bcc0eb7e00c6e3fc3d181b951b81294ed7a483a87030df0123"
    assert isinstance(selected.objective, SeedObjective)
    assert selected.objective.value == selected.task.dataset[0].input
    assert selected.task.dataset[0].target != selected.objective.value
    assert selected.health_command == ("/bin/test", "-f", "/tmp/inspect-marker")


def test_schema_two_enumerates_exact_original_task_sample_inventory_with_duplicate_text(tmp_path: Path) -> None:
    revision = _multi_manifest(root=tmp_path)
    selected = _resolve_multi(root=tmp_path, revision=revision)
    assert len(selected) == 3
    assert [(case.case.task_name, case.case.sample_id) for case in selected] == [
        ("local_benign", "local-one"),
        ("local_second", "local-two-a"),
        ("local_second", "local-two-b"),
    ]
    assert len({case.case.case_id for case in selected}) == 3
    assert {case.case.package.source_sha256 for case in selected} == {revision}
    assert {case.objective.value for case in selected} == {"local harmless goal"}
    assert selected[1].task is selected[2].task
    assert len(selected[2].task.dataset) == 2
    assert {case.sandbox_sha256 for case in selected} == {selected[0].sandbox_sha256}
    for case in selected:
        case.verify_unchanged()

    with pytest.raises(ValueError, match="resolve_cases"):
        _resolve_local(root=tmp_path, revision=revision)
    with pytest.raises(ValueError, match="cannot sweep multiple Eval cases"):
        _resolve_multi(root=tmp_path, revision=revision, input_override="edited goal")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            _MULTI_TASK.replace("return [task, second]", "return [task, second, second]"),
            "exact declared Task inventory",
        ),
        (
            _MULTI_TASK.replace(
                'Sample(input="local harmless goal", id="local-two-b", target="answer"),',
                'Sample(input="local harmless goal", id="local-two-b", target="answer"),\n'
                '            Sample(input="local harmless goal", id="undeclared", target="answer"),',
            ),
            "exact declared materialized Task/Sample inventory",
        ),
        (_MULTI_TASK.replace('name="local_second"', 'name="other_task"'), "Task/Sample identity"),
        (
            _MULTI_TASK.replace(
                "return [task, second]", "open('absent-fixture.txt').read()\n    return [task, second]"
            ),
            "declared dependency",
        ),
        (
            _MULTI_TASK.replace(
                "return [task, second]",
                "import copy\n    second.sandbox = copy.deepcopy(task.sandbox)\n"
                "    second.sandbox.config.services['target'].mem_limit = '512m'\n    return [task, second]",
            ),
            "different effective per-Sample Compose sandbox",
        ),
        (
            _MULTI_TASK.replace(
                'Sample(input="local harmless goal", id="local-two-b", target="answer")',
                'Sample(input="local harmless goal", id="local-two-b", target="answer", '
                'sandbox=SandboxEnvironmentSpec("docker", ComposeConfig(services={})))',
            ),
            "per-Sample sandbox",
        ),
        (
            _MULTI_TASK.replace(
                'Sample(input="local harmless goal", id="local-two-b", target="answer")',
                'Sample(input="local harmless goal", id="local-two-b", target="answer", '
                'files={"/tmp/undeclared": "contents"})',
            ),
            "per-Sample sandbox, files, or setup",
        ),
        (
            _MULTI_TASK.replace(
                'Sample(input="local harmless goal", id="local-two-b", target="answer")',
                'Sample(input="local harmless goal", id="local-two-b", target="answer", files={})',
            ),
            "per-Sample sandbox, files, or setup",
        ),
        (
            _MULTI_TASK.replace(
                "return [task, second]",
                "from inspect_ai.scorer import includes\n    second.scorer = [includes('answer')]\n"
                "    return [task, second]",
            ),
            "scorer, setup and cleanup must originate from pinned Eval source files",
        ),
        (
            _MULTI_TASK.replace(
                "return [task, second]",
                "second.score_on_error = True\n    return [task, second]",
            ),
            "additional dataset or Task execution settings",
        ),
    ],
)
def test_schema_two_rejects_ambiguous_or_unqualified_factory_results(tmp_path: Path, source: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _resolve_multi(root=tmp_path, revision=_multi_manifest(root=tmp_path, task_source=source))


def test_schema_two_rejects_missing_assets_duplicate_ids_and_mutable_unpinned_images(tmp_path: Path) -> None:
    revision = _multi_manifest(root=tmp_path)
    (tmp_path / "public-asset.txt").unlink()
    with pytest.raises(ValueError, match="regular, inside|source file"):
        _resolve_multi(root=tmp_path, revision=revision)

    revision = _multi_manifest(root=tmp_path)
    selected = _resolve_multi(root=tmp_path, revision=revision)
    (tmp_path / "public-asset.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source asset changed"):
        selected[2].verify_unchanged()

    revision = _multi_manifest(root=tmp_path)
    manifest_path = tmp_path / EvalSourceFactory.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["tasks"][1]["samples"][1]["sample_id"] = "local-two-a"
    data = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(data)
    with pytest.raises(ValueError, match="duplicate Task or per-Task Sample identities"):
        _resolve_multi(root=tmp_path, revision=hashlib.sha256(data).hexdigest())

    revision = _multi_manifest(root=tmp_path)
    with pytest.raises(ValueError, match="pinned local image ID"):
        _resolve_multi(
            root=tmp_path,
            revision=revision,
            approved_image_ids={"agent": "", "model-bridge": "", "target": ""},
        )
    changed_image = _MULTI_TASK.replace('"agent": service(agent_image,', '"agent": service("other/agent:mutable",')
    with pytest.raises(ValueError, match="approved tag and pinned local image ID"):
        _resolve_multi(root=tmp_path, revision=_multi_manifest(root=tmp_path, task_source=changed_image))


def test_named_input_variant_changes_objective_not_original_source() -> None:
    args = {
        "family": "benign_protocol",
        "trusted_dir": None,
        "trusted_local": False,
        "revision_sha256": None,
        "agent_image": "approved/agent:pinned",
        "target_image": "approved/target:pinned",
        "approved_image_ids": _IMAGE_IDS,
    }
    original = EvalSourceFactory.resolve(input_override=None, **args)
    variant = EvalSourceFactory.resolve(input_override="Another harmless user instruction", **args)
    assert variant.case == original.case
    assert variant.case.package.source_sha256 == original.case.package.source_sha256
    assert variant.original_input_sha256 == original.original_input_sha256
    assert variant.objective.value != original.objective.value
    assert variant.input_override_sha256 == hashlib.sha256(variant.objective.value.encode()).hexdigest()
    variant.verify_unchanged()


def test_schema_one_input_variant_does_not_mutate_a_cached_original_task() -> None:
    arguments = {
        "family": "benign_protocol",
        "trusted_dir": None,
        "trusted_local": False,
        "revision_sha256": None,
        "agent_image": "approved/agent:pinned",
        "target_image": "approved/target:pinned",
        "approved_image_ids": _IMAGE_IDS,
    }
    original = EvalSourceFactory.resolve(input_override=None, **arguments)

    def cached_factory(*, agent_image: str, target_image: str) -> Task:
        return original.task

    with patch.object(eval_source, "_load_task_factory", return_value=cached_factory):
        edited = EvalSourceFactory.resolve(input_override="variant only", **arguments)

    assert edited.task is not original.task
    assert original.task.dataset[0].input == original.objective.value
    assert edited.objective.value == "variant only"
    original.verify_unchanged()
    edited.verify_unchanged()


def test_trusted_local_source_checks_manifest_before_import_and_detects_drift(tmp_path: Path) -> None:
    revision = _local_manifest(root=tmp_path, allow_edit=True)
    with pytest.raises(ValueError, match="explicit trust"):
        EvalSourceFactory.resolve(
            family=None,
            trusted_dir=tmp_path,
            trusted_local=False,
            revision_sha256=revision,
            input_override=None,
            agent_image="approved/agent:pinned",
            target_image="approved/target:pinned",
            approved_image_ids=_IMAGE_IDS,
        )
    selected = _resolve_local(root=tmp_path, revision=revision, input_override="safe edited goal")
    selected.verify_unchanged()
    assert selected.case.package.kind is EvalSourceKind.TRUSTED_LOCAL
    assert selected.case.package.source_sha256 == revision
    assert selected.objective.value == "safe edited goal"
    assert selected.original_input_sha256 == hashlib.sha256(b"local harmless goal").hexdigest()
    (tmp_path / "task.py").write_text(_LOCAL_TASK + "\n# drift", encoding="utf-8")
    with pytest.raises(ValueError, match="source asset changed"):
        selected.verify_unchanged()
    with pytest.raises(ValueError, match="source asset digest"):
        _resolve_local(root=tmp_path, revision=revision)


def test_schema_one_local_manifest_keeps_its_optional_version_default(tmp_path: Path) -> None:
    _local_manifest(root=tmp_path)
    manifest_path = tmp_path / EvalSourceFactory.MANIFEST_FILE
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["schema_version"]
    content = json.dumps(manifest, sort_keys=True).encode("utf-8")
    manifest_path.write_bytes(content)
    selected = _resolve_local(root=tmp_path, revision=hashlib.sha256(content).hexdigest())
    assert selected.case.sample_id == "local-one"
    selected.verify_unchanged()


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ("setup", "Task.setup"),
        ("scorer", "original scorer"),
        ("cleanup", "cleanup callback"),
        ("metadata", "extra authored Task"),
    ],
)
def test_authored_task_state_drift_cannot_reuse_a_source_revision(tmp_path: Path, changed: str, message: str) -> None:
    selected = _resolve_local(root=tmp_path, revision=_local_manifest(root=tmp_path))
    if changed == "setup":
        selected.task.setup = system_message("different author initialization")
    elif changed == "scorer":
        selected.task.scorer = [includes()]
    elif changed == "cleanup":

        async def replaced_cleanup_async(state: object) -> None:
            pass

        selected.task.cleanup = replaced_cleanup_async
    else:
        selected.task.metadata = {"changed": True}
    with pytest.raises(ValueError, match=message):
        selected.verify_unchanged()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (_LOCAL_TASK.replace("dataset=[sample]", "dataset=[sample, sample]"), "one materialized Task and Sample"),
        (_LOCAL_TASK.replace("return task", "return [task, task]"), "non-Task or multiple Tasks"),
        (
            _LOCAL_TASK.replace(
                "sample = Sample(",
                "sample = Sample(sandbox=SandboxEnvironmentSpec('docker', ComposeConfig(services={})), ",
            ),
            "per-Sample sandbox",
        ),
        (
            _LOCAL_TASK.replace("return task", "open('missing-fixture.txt').read()\n    return task"),
            "declared dependency",
        ),
    ],
)
def test_local_source_rejects_multiple_samples_overrides_or_missing_eager_fixture(
    tmp_path: Path, source: str, message: str
) -> None:
    revision = _local_manifest(root=tmp_path, task_source=source)
    with pytest.raises(ValueError, match=message):
        _resolve_local(root=tmp_path, revision=revision)


def test_unknown_xor_and_mutable_images_without_ids_fail_closed(tmp_path: Path) -> None:
    revision = _local_manifest(root=tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        EvalSourceFactory.resolve(
            family="benign_protocol",
            trusted_dir=tmp_path,
            trusted_local=True,
            revision_sha256=revision,
            input_override=None,
            agent_image="approved/agent:pinned",
            target_image="approved/target:pinned",
            approved_image_ids=_IMAGE_IDS,
        )
    with pytest.raises(ValueError, match="trust cannot be combined"):
        EvalSourceFactory.resolve(
            family="benign_protocol",
            trusted_dir=None,
            trusted_local=True,
            revision_sha256=None,
            input_override=None,
            agent_image="approved/agent:pinned",
            target_image="approved/target:pinned",
            approved_image_ids=_IMAGE_IDS,
        )
    with pytest.raises(ValueError, match="Unknown or unqualified"):
        EvalSourceFactory.resolve(
            family="unqualified_suite",
            trusted_dir=None,
            trusted_local=False,
            revision_sha256=None,
            input_override=None,
            agent_image="approved/agent:pinned",
            target_image="approved/target:pinned",
            approved_image_ids=_IMAGE_IDS,
        )
    with pytest.raises(ValueError, match="pinned local image ID"):
        EvalSourceFactory.resolve(
            family="benign_protocol",
            trusted_dir=None,
            trusted_local=False,
            revision_sha256=None,
            input_override=None,
            agent_image="approved/agent:mutable",
            target_image="approved/target:mutable",
            approved_image_ids={"agent": "", "model-bridge": "", "target": ""},
        )
    with pytest.raises(ValueError, match="manifest changed"):
        _resolve_local(root=tmp_path, revision="a" * 64)


def test_local_directory_rejects_symlink_and_unpinned_assets(tmp_path: Path) -> None:
    revision = _local_manifest(root=tmp_path)
    is_symlink = Path.is_symlink
    with patch.object(
        Path, "is_symlink", autospec=True, side_effect=lambda p: p == tmp_path / "task.py" or is_symlink(p)
    ):
        with pytest.raises(ValueError, match="not symlinks|source file"):
            _resolve_local(root=tmp_path, revision=revision)
    manifest = json.loads((tmp_path / EvalSourceFactory.MANIFEST_FILE).read_text())
    manifest["files_sha256"]["absent.dat"] = "b" * 64
    revised = json.dumps(manifest, sort_keys=True).encode()
    (tmp_path / EvalSourceFactory.MANIFEST_FILE).write_bytes(revised)
    with pytest.raises(ValueError, match="inside the approved directory|source file"):
        _resolve_local(root=tmp_path, revision=hashlib.sha256(revised).hexdigest())
