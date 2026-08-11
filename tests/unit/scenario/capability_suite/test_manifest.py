# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pyrit.executor.capability import CapabilityLimits
from pyrit.scenario.capability_suite.manifest import (
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    CaseAssetManifest,
    CaseMessageContentManifest,
    CaseMessageManifest,
    CaseSetupStepManifest,
    CaseToolManifest,
    LocalSandboxProviderManifestConfig,
    ScoreReducerManifest,
    SuiteProvenance,
    ToolDeclaration,
    ToolImplementationManifest,
    validate_safe_relative_path,
)


def _provenance() -> SuiteProvenance:
    return SuiteProvenance(source="unit-test", repository="example/repo", revision="abc123", license="MIT")


def _case(case_id: str = "case-1", **overrides: object) -> CapabilityCaseManifest:
    defaults: dict[str, object] = {
        "case_id": case_id,
        "objective": "finish the task",
        "messages": (CaseMessageManifest(role="user", content="hello"),),
    }
    defaults.update(overrides)
    return CapabilityCaseManifest(**defaults)


def _manifest(**overrides: object) -> CapabilitySuiteManifest:
    defaults: dict[str, object] = {
        "suite_id": "suite-1",
        "name": "Example suite",
        "provenance": _provenance(),
        "sandbox_provider": LocalSandboxProviderManifestConfig(),
        "cases": (_case(),),
    }
    defaults.update(overrides)
    return CapabilitySuiteManifest(**defaults)


def test_validate_safe_relative_path_accepts_plain_relative_path() -> None:
    assert validate_safe_relative_path("assets/data.json") == "assets/data.json"


def test_case_message_manifest_preserves_ordered_multimodal_parts() -> None:
    message = CaseMessageManifest(
        role="user",
        parts=(
            CaseMessageContentManifest(content="describe this", data_type="text"),
            CaseMessageContentManifest(content="image.png", data_type="image_path"),
        ),
    )

    assert [part.data_type for part in message.content_parts] == ["text", "image_path"]


def test_case_message_manifest_rejects_ambiguous_or_empty_content() -> None:
    with pytest.raises(ValidationError, match="requires either"):
        CaseMessageManifest(role="user")
    with pytest.raises(ValidationError, match="cannot declare both"):
        CaseMessageManifest(
            role="user",
            content="legacy",
            parts=(CaseMessageContentManifest(content="new"),),
        )


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "",
        "/etc/passwd",
        "~/secrets",
        "C:\\Windows\\System32",
        "//server/share",
        "../escape",
        "assets/../../escape",
        "assets\\data.json",
        "assets//data.json",
        "assets/./data.json",
        "./.",
    ],
)
def test_validate_safe_relative_path_rejects_unsafe_paths(unsafe_path: str) -> None:
    with pytest.raises(ValueError):
        validate_safe_relative_path(unsafe_path)


def test_case_asset_manifest_rejects_unsafe_source_or_destination() -> None:
    with pytest.raises(ValidationError):
        CaseAssetManifest(asset_id="a1", source="../escape", sha256="0" * 64, destination="dest.txt")
    with pytest.raises(ValidationError):
        CaseAssetManifest(asset_id="a1", source="source.txt", sha256="0" * 64, destination="/abs/dest.txt")


def test_manifest_rejects_unknown_top_level_field() -> None:
    data = _manifest().model_dump(mode="json")
    data["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        CapabilitySuiteManifest.model_validate(data)


def test_manifest_rejects_unknown_case_field() -> None:
    data = _manifest().model_dump(mode="json")
    data["cases"][0]["unexpected_field"] = "surprise"
    with pytest.raises(ValidationError):
        CapabilitySuiteManifest.model_validate(data)


def test_manifest_is_frozen() -> None:
    manifest = _manifest()
    with pytest.raises(ValidationError):
        manifest.name = "renamed"  # type: ignore[misc]


def test_manifest_requires_at_least_one_case() -> None:
    with pytest.raises(ValidationError):
        _manifest(cases=())


def test_manifest_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValidationError):
        _manifest(cases=(_case(case_id="dup"), _case(case_id="dup")))


def test_case_rejects_duplicate_asset_ids() -> None:
    asset = CaseAssetManifest(asset_id="a1", source="s.txt", sha256="0" * 64, destination="d.txt")
    with pytest.raises(ValidationError):
        _case(assets=(asset, asset))


def test_case_rejects_duplicate_tool_declaration_names() -> None:
    declaration = ToolDeclaration(name="lookup", description="Look up.", input_schema={"type": "object"})
    tool = CaseToolManifest(declaration=declaration, implementation=ToolImplementationManifest(kind="noop"))
    with pytest.raises(ValidationError):
        _case(tools=(tool, tool))


def test_case_rejects_custom_tool_collision_with_sandbox_tools() -> None:
    declaration = ToolDeclaration(name="sandbox_exec", input_schema={"type": "object"})
    tool = CaseToolManifest(declaration=declaration, implementation=ToolImplementationManifest(kind="noop"))
    with pytest.raises(ValidationError, match="collide"):
        _case(sandbox_tools_prefix="sandbox", tools=(tool,))


def test_case_requires_sandbox_default_environment_in_allowlist() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        _case(
            sandbox_tools_prefix="sandbox",
            sandbox_tools_default_environment="attacker",
        )
    with pytest.raises(ValidationError, match="default environment"):
        _case(
            sandbox_tools_prefix="sandbox",
            sandbox_tools_default_environment="attacker",
            sandbox_tools_allowed_environments=("victim",),
        )


def test_asset_hash_must_be_lowercase_sha256_hex() -> None:
    with pytest.raises(ValidationError):
        CaseAssetManifest(asset_id="a1", source="s.txt", sha256="g" * 64, destination="d.txt")


def test_setup_step_requires_exactly_one_of_argv_or_shell_script() -> None:
    with pytest.raises(ValidationError):
        CaseSetupStepManifest()
    with pytest.raises(ValidationError):
        CaseSetupStepManifest(argv=("echo", "hi"), shell_script="echo hi")
    step = CaseSetupStepManifest(argv=("echo", "hi"))
    assert step.argv == ("echo", "hi")


def test_case_limits_default_to_shared_capability_limits() -> None:
    case = _case()
    assert isinstance(case.limits, CapabilityLimits)
    assert case.limits.max_model_generations == CapabilityLimits().max_model_generations


def test_score_reducer_accepts_any_finite_threshold() -> None:
    reducer = ScoreReducerManifest(name="threshold", kind="mean", threshold=2.5)

    assert reducer.threshold == 2.5


def test_score_reducer_rejects_non_finite_threshold() -> None:
    with pytest.raises(ValidationError, match="finite"):
        ScoreReducerManifest(name="threshold", kind="mean", threshold=float("inf"))
