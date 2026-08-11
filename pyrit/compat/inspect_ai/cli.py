# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Implementation helpers for the Inspect-evals capability-suite CLI."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import signal
import uuid
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from pyrit.compat.inspect_ai.catalog import (
    build_inspect_catalog,
    check_inspect_catalog_regression,
)
from pyrit.compat.inspect_ai.loader import (
    LoadedInspectEval,
    load_inspect_eval,
    run_loaded_inspect_eval_async,
)
from pyrit.compat.inspect_ai.source import prepare_inspect_source, validate_inspect_source
from pyrit.scenario.capability_suite.manifest import (
    CapabilityCaseManifest,
    CapabilitySuiteManifest,
    DockerSandboxProviderManifestConfig,
    SandboxProviderManifest,
)

if TYPE_CHECKING:
    import argparse


def run_source_command(args: argparse.Namespace) -> int:
    """
    Acquire or validate pinned source and emit verification data.

    Returns:
        int: Process exit code.
    """
    if args.source_action == "prepare":
        verification = prepare_inspect_source(
            cache_dir=args.cache_dir,
            offline=args.offline,
            timeout_seconds=args.timeout,
        )
    else:
        verification = validate_inspect_source(
            source_root=args.source,
            require_clean=not args.allow_dirty,
            timeout_seconds=args.timeout,
        )
    _emit_json(verification.to_dict(), output=args.output)
    return 0


def run_tasks_command(args: argparse.Namespace) -> int:
    """
    List every statically discovered task factory.

    Returns:
        int: Process exit code.
    """
    catalog = build_inspect_catalog(source_root=args.source, verify_source=not args.no_verify_source)
    tasks = [
        {
            **task,
            "family": family.family,
            "task_spec": _task_spec(task=task, source_directory=family.source_directory),
        }
        for family in catalog.families
        if args.family is None or family.family == args.family
        for task in family.task_factories
    ]
    if args.format == "json":
        _emit_json({"profile_id": catalog.profile_id, "tasks": tasks}, output=args.output)
    else:
        lines = [
            (
                f"{task['task_spec']}\t{task['compatibility_status']}\t"
                f"parameters={_task_parameters_text(task)}\truntime={_task_runtime_text(task)}"
            )
            for task in tasks
        ]
        _emit_text("\n".join(lines), output=args.output)
    return 0


def run_report_command(args: argparse.Namespace) -> int:
    """
    Show profile and catalog compatibility diagnostics.

    Returns:
        int: Process exit code.
    """
    catalog = build_inspect_catalog(source_root=args.source, verify_source=not args.no_verify_source)
    if args.format == "json":
        _emit_json(catalog.to_dict(), output=args.output)
    else:
        _emit_text(catalog.to_text(), output=args.output)
    return 0


def run_catalog_command(args: argparse.Namespace) -> int:
    """
    Emit catalog diagnostics and optionally enforce golden regression metadata.

    Returns:
        int: Process exit code.
    """
    catalog = build_inspect_catalog(source_root=args.source, verify_source=not args.no_verify_source)
    if args.check:
        check_inspect_catalog_regression(report=catalog)
    if args.format == "json":
        _emit_json(catalog.to_dict(), output=args.output)
    else:
        text = catalog.to_text()
        if args.check:
            text += "\nCatalog regression check: passed"
        _emit_text(text, output=args.output)
    return 0


def run_compile_command(args: argparse.Namespace) -> int:
    """
    Compile or dry-run one unchanged supported task without model credentials.

    Returns:
        int: Process exit code.
    """
    loaded = _load_and_restrict(args)
    _write_loaded_outputs(loaded=loaded, manifest_path=args.manifest, report_path=args.report)
    if args.manifest is None and args.report is None:
        _emit_json(_loaded_summary(loaded), output=None)
    return 0


def run_execute_command(args: argparse.Namespace) -> int:
    """
    Execute one unchanged supported task through a registered PyRIT target.

    Returns:
        int: Process exit code.
    """
    loaded = _load_and_restrict(args)
    return asyncio.run(_run_execute_command_async(args=args, loaded=loaded))


async def _run_execute_command_async(*, args: argparse.Namespace, loaded: LoadedInspectEval) -> int:
    target = await _resolve_target_async(target_name=args.target, target_role=args.target_role, config_file=args.config)
    request_options_factory = _validate_target_and_request_options(
        target=target,
        manifest=loaded.suite,
        task=loaded.task,
    )
    cancellation_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    registered_signals: list[signal.Signals] = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, cancellation_event.set)
            registered_signals.append(signum)
        except (NotImplementedError, RuntimeError):
            continue
    try:
        result = await run_loaded_inspect_eval_async(
            loaded=loaded,
            target=target,
            inspect_evals_cache_dir=args.inspect_evals_cache_dir,
            request_options_factory=request_options_factory,
            resume_id=args.resume_id,
            cancellation_event=cancellation_event,
        )
    finally:
        for signum in registered_signals:
            loop.remove_signal_handler(signum)
    payload = {
        "profile_id": loaded.report.profile_id,
        "task": loaded.report.task_name,
        "target": args.target,
        "target_role": args.target_role,
        "compatibility_report": _loaded_report(loaded),
        "result": dataclasses.asdict(result),
    }
    _emit_json(payload, output=args.result)
    return 0


def _load_and_restrict(args: argparse.Namespace) -> LoadedInspectEval:
    parameters = _parse_task_parameters(args.task_param)
    _apply_named_task_parameters(args=args, parameters=parameters)
    dataset_records = _local_dataset_records(args.data) if args.data is not None else None
    loaded = load_inspect_eval(
        source_root=args.source,
        task_spec=args.task,
        task_parameters=parameters,
        profile_id=args.profile,
        dataset_records=dataset_records,
        inspect_evals_cache_dir=args.inspect_evals_cache_dir,
        allow_network=args.allow_network,
        verify_source_revision=not args.no_verify_source,
        worker_timeout_seconds=args.worker_timeout,
        source_verification_timeout_seconds=args.source_verification_timeout,
        case_timeout_seconds=args.case_timeout,
    )
    suite = _restrict_manifest(
        manifest=loaded.suite,
        case_ids=tuple(args.case_id),
        case_limit=args.limit,
        epochs=args.epochs,
        attempts=args.attempts,
        max_concurrency=args.concurrency,
        sandbox_provider=args.sandbox_provider,
        sandbox_config_path=args.sandbox_config,
        retain_sandboxes=args.retain_sandboxes,
    )
    return dataclasses.replace(loaded, suite=suite)


def _parse_task_parameters(values: list[str]) -> dict[str, object]:
    parameters: dict[str, object] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"Task parameter '{item}' must use NAME=JSON_VALUE.")
        name, raw_value = item.split("=", 1)
        if not name:
            raise ValueError("Task parameter names cannot be empty.")
        try:
            parameters[name] = json.loads(raw_value)
        except json.JSONDecodeError:
            parameters[name] = raw_value
    return parameters


def _apply_named_task_parameters(*, args: argparse.Namespace, parameters: dict[str, object]) -> None:
    task_name = args.task.rsplit("@", 1)[-1].rsplit(":", 1)[-1]
    if args.submission_attempts is not None:
        parameters["max_attempts"] = args.submission_attempts
    if args.max_messages is not None:
        parameters["max_messages"] = args.max_messages
    if task_name == "gdm_intercode_ctf" and args.case_id:
        try:
            parameters["sample_ids"] = [int(case_id) for case_id in args.case_id]
        except ValueError as error:
            raise ValueError("GDM InterCode --case-id values must be integer task IDs.") from error
    if task_name == "gdm_in_house_ctf" and args.case_id:
        parameters["challenges"] = list(args.case_id)
    if task_name == "gdm_in_house_ctf" and args.epochs is not None:
        parameters["epochs"] = args.epochs


def _local_dataset_records(path: Path) -> list[dict[str, object]] | dict[str, list[dict[str, object]]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(records, list) and all(isinstance(record, dict) for record in records):
        return records
    if isinstance(records, dict) and all(
        isinstance(key, str) and isinstance(value, list) and all(isinstance(record, dict) for record in value)
        for key, value in records.items()
    ):
        return records
    raise ValueError(
        f"Local dataset '{path}' must contain a JSON array of objects or an object mapping dataset keys to arrays."
    )


def _restrict_manifest(
    *,
    manifest: CapabilitySuiteManifest,
    case_ids: tuple[str, ...],
    case_limit: int | None,
    epochs: int | None,
    attempts: int | None,
    max_concurrency: int | None,
    sandbox_provider: str,
    sandbox_config_path: Path | None,
    retain_sandboxes: bool,
) -> CapabilitySuiteManifest:
    cases = manifest.cases
    if case_ids:
        selected = frozenset(case_ids)
        cases = tuple(
            case
            for case in cases
            if case.case_id in selected or (case.source is not None and case.source.source_id in selected)
        )
        if not cases:
            raise ValueError(f"No compiled cases matched --case-id values: {', '.join(case_ids)}.")
    if case_limit is not None:
        cases = cases[:case_limit]
    run_policy_updates = {
        key: value
        for key, value in {
            "epochs": epochs,
            "attempts": attempts,
            "max_concurrency": max_concurrency,
        }.items()
        if value is not None
    }
    run_policy = manifest.run_policy.model_copy(update=run_policy_updates)
    suite_provider, cases = _configure_sandboxes(
        suite_provider=manifest.sandbox_provider,
        cases=cases,
        sandbox_provider=sandbox_provider,
        sandbox_config_path=sandbox_config_path,
        retain_sandboxes=retain_sandboxes,
    )
    return manifest.model_copy(update={"cases": cases, "run_policy": run_policy, "sandbox_provider": suite_provider})


def _configure_sandboxes(
    *,
    suite_provider: SandboxProviderManifest,
    cases: tuple[CapabilityCaseManifest, ...],
    sandbox_provider: str,
    sandbox_config_path: Path | None,
    retain_sandboxes: bool,
) -> tuple[SandboxProviderManifest, tuple[CapabilityCaseManifest, ...]]:
    providers = {(case.sandbox_provider or suite_provider).provider_type for case in cases}
    if sandbox_provider == "hyperv" and "docker" in providers:
        raise ValueError(
            "Hyper-V cannot run GDM Compose task semantics. Use --sandbox-provider docker; "
            "no Hyper-V Compose compatibility is implemented."
        )
    if sandbox_provider not in {"auto", *providers}:
        raise ValueError(
            f"Compiled task requires {', '.join(sorted(providers))}; --sandbox-provider {sandbox_provider} "
            "cannot preserve its semantics."
        )
    overrides: dict[str, object] | None = None
    if sandbox_config_path is not None:
        overrides = json.loads(sandbox_config_path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("--sandbox-config must contain a JSON object.")
    configured_suite = (
        _configure_provider(
            provider=suite_provider,
            overrides=overrides,
            retain_sandboxes=retain_sandboxes,
        )
        if any(case.sandbox_provider is None for case in cases)
        else suite_provider
    )
    configured_cases = tuple(
        case.model_copy(
            update={
                "sandbox_provider": _configure_provider(
                    provider=case.sandbox_provider,
                    overrides=overrides,
                    retain_sandboxes=retain_sandboxes,
                )
            }
        )
        if case.sandbox_provider is not None
        else case
        for case in cases
    )
    return configured_suite, configured_cases


def _configure_provider(
    *,
    provider: SandboxProviderManifest,
    overrides: dict[str, object] | None,
    retain_sandboxes: bool,
) -> SandboxProviderManifest:
    if overrides is not None:
        if not isinstance(provider, DockerSandboxProviderManifestConfig):
            raise ValueError("--sandbox-config currently supports only Docker-backed unchanged tasks.")
        provider = provider.model_copy(update={"config": provider.config.model_copy(update=overrides)})
    if retain_sandboxes:
        if not isinstance(provider, DockerSandboxProviderManifestConfig):
            raise ValueError("--retain-sandboxes is supported only for Docker-backed unchanged tasks.")
        provider = provider.model_copy(
            update={"config": provider.config.model_copy(update={"retain_resources_on_close": True})}
        )
    return provider


async def _resolve_target_async(*, target_name: str | None, target_role: str, config_file: Path | None) -> Any:
    from pyrit.registry import TargetRegistry
    from pyrit.setup import ConfigurationLoader

    config = ConfigurationLoader.load_with_overrides(config_file=config_file)
    await config.initialize_pyrit_async()
    registry = TargetRegistry.get_registry_singleton().instances
    if target_name is not None:
        target = registry.get(target_name)
        if target is None:
            raise ValueError(
                f"Registered target '{target_name}' was not found. Available targets: "
                f"{', '.join(registry.get_names()) or '(none)'}."
            )
        return target
    matches = registry.get_by_tag(tag=target_role)
    if len(matches) != 1:
        names = ", ".join(entry.name for entry in matches) or "(none)"
        raise ValueError(
            f"Target role tag '{target_role}' must resolve to exactly one registered target; matched: {names}. "
            "Use --target for an exact registry name."
        )
    return matches[0].instance


def _validate_target_and_request_options(
    *,
    target: Any,
    manifest: CapabilitySuiteManifest,
    task: Any | None = None,
) -> Any:
    from pyrit.compat.inspect_ai.loader import _effective_generate_config, _NoToolRequestOptionsFactory
    from pyrit.executor.capability import (
        build_capability_request_options_factory,
        validate_capability_target,
    )
    from pyrit.scenario.capability_suite import get_capability_suite_target_requirements

    target_requirements = get_capability_suite_target_requirements(manifest=manifest)
    validate_capability_target(
        target=target,
        request_options_factory=None,
        requires_multi_turn=target_requirements.requires_multi_turn,
        requires_tools=target_requirements.requires_tools,
        requires_system_prompt=target_requirements.requires_system_prompt,
        required_input_modalities=target_requirements.required_input_modalities,
        required_output_modalities=target_requirements.required_output_modalities,
    )
    factory = (
        build_capability_request_options_factory(target=target)
        if target_requirements.requires_tools
        else _NoToolRequestOptionsFactory(
            request_options_type=target.request_options_type,
            generate_config=_effective_generate_config(task) if task is not None else None,
        )
    )
    validate_capability_target(
        target=target,
        request_options_factory=factory,
        requires_multi_turn=target_requirements.requires_multi_turn,
        requires_tools=target_requirements.requires_tools,
        requires_system_prompt=target_requirements.requires_system_prompt,
        required_input_modalities=target_requirements.required_input_modalities,
        required_output_modalities=target_requirements.required_output_modalities,
    )
    return factory


def _write_loaded_outputs(
    *,
    loaded: LoadedInspectEval,
    manifest_path: Path | None,
    report_path: Path | None,
) -> None:
    if manifest_path is not None:
        _emit_json(loaded.suite.model_dump(mode="json"), output=manifest_path)
    if report_path is not None:
        _emit_json(_loaded_report(loaded), output=report_path)


def _loaded_summary(loaded: LoadedInspectEval) -> dict[str, object]:
    return {
        "profile_id": loaded.report.profile_id,
        "task_name": loaded.report.task_name,
        "task_parameters": loaded.report.task_parameters,
        "case_count": len(loaded.suite.cases),
        "sandbox_providers": sorted(
            {(case.sandbox_provider or loaded.suite.sandbox_provider).provider_type for case in loaded.suite.cases}
        ),
        "runtime_requirements": loaded.suite.metadata.get("runtime_requirements", {}),
        "requires_model_credentials": False,
        "manifest": loaded.suite.model_dump(mode="json"),
        "compatibility_report": _loaded_report(loaded),
    }


def _loaded_report(loaded: LoadedInspectEval) -> dict[str, object]:
    suite_metadata = loaded.suite.metadata
    return {
        "profile_id": loaded.report.profile_id,
        "inspect_api_profile": loaded.report.inspect_api_profile,
        "source_revision": loaded.report.source_revision,
        "detected_source_revision": loaded.report.detected_source_revision,
        "source_revision_verified": loaded.report.source_revision_verified,
        "task_name": loaded.report.task_name,
        "task_parameters": loaded.report.task_parameters,
        "api_inventory": loaded.report.api_inventory.to_dict(),
        "capabilities": loaded.report.capabilities,
        "limitations": list(loaded.report.limitations),
        "construction_inventory": {
            "factory_parameters": loaded.report.task_parameters,
            "dataset": suite_metadata.get("dataset", {}),
            "dataset_provenance": suite_metadata.get("dataset_provenance", {}),
            "solver": suite_metadata.get("solver"),
            "scorer": suite_metadata.get("scorer"),
            "task_config": suite_metadata.get("task_config", {}),
            "runtime_requirements": suite_metadata.get("runtime_requirements", {}),
            "case_count": len(loaded.suite.cases),
            "case_ids": [case.case_id for case in loaded.suite.cases],
            "case_scorers": [scorer.model_dump(mode="json") for case in loaded.suite.cases for scorer in case.scorers],
        },
    }


def _task_spec(*, task: dict[str, object], source_directory: str) -> str:
    source_file = str(task["source_file"])
    marker = "src/inspect_evals/"
    relative = source_file.split(marker, 1)[-1] if marker in source_file else source_file
    if relative == source_directory:
        relative = Path(relative).name
    return f"{relative}@{task['name']}"


def _task_parameters_text(task: dict[str, object]) -> str:
    parameters = task.get("parameters")
    if not isinstance(parameters, (list, tuple)):
        return "-"
    return ",".join(str(parameter) for parameter in parameters) or "-"


def _task_runtime_text(task: dict[str, object]) -> str:
    audit = task.get("reviewed_static_mapping_audit")
    if not isinstance(audit, dict):
        return "-"
    runtime = audit.get("runtime")
    if not isinstance(runtime, dict):
        return "-"
    language = runtime.get("language")
    dependencies = runtime.get("dependencies")
    parts = [str(language)] if language is not None else []
    if isinstance(dependencies, list):
        parts.append("+".join(str(dependency) for dependency in dependencies))
    blocker = runtime.get("blocker")
    if blocker is not None:
        parts.append(f"blocked:{blocker}")
    return ",".join(parts) or "-"


def _emit_json(payload: object, *, output: Path | None) -> None:
    _emit_text(json.dumps(payload, default=_json_default, indent=2, sort_keys=True), output=output)


def _emit_text(text: str, *, output: Path | None) -> None:
    if output is None:
        print(text)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _json_default(value: object) -> object:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date, uuid.UUID, Path)):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
