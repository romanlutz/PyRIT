# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Offline capability-suite interoperability commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pyrit_capability_suite",
        description="Analyze external eval sources and compile reviewed native capability-suite manifests.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect-evals",
        help="Acquire, diagnose, compile, and run pinned unchanged inspect-evals tasks.",
    )
    inspect_parser.add_argument("--source", type=Path)
    inspect_parser.add_argument(
        "--family",
        choices=("arc", "gdm_intercode_ctf", "gdm_in_house_ctf", "swe_bench"),
    )
    inspect_parser.add_argument("--data", type=Path)
    inspect_parser.add_argument("--case-id", action="append", default=[])
    inspect_parser.add_argument("--manifest", type=Path)
    inspect_parser.add_argument("--report", type=Path)
    inspect_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Permit pinned Hugging Face retrieval when no local data file is supplied.",
    )
    inspect_commands = inspect_parser.add_subparsers(dest="inspect_command")
    _add_source_parser(inspect_commands)
    _add_tasks_parser(inspect_commands)
    _add_report_parser(inspect_commands)
    _add_catalog_parser(inspect_commands)
    _add_compile_parser(inspect_commands)
    _add_run_parser(inspect_commands)
    return parser


def _add_source_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("source", help="Acquire or validate the pinned source checkout.")
    actions = parser.add_subparsers(dest="source_action", required=True)
    prepare = actions.add_parser("prepare", help="Populate and verify the content-addressed source cache.")
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--offline", action="store_true")
    prepare.add_argument("--timeout", type=float, default=300.0)
    prepare.add_argument("--output", type=Path)
    validate = actions.add_parser("validate", help="Read-only validation of a supplied checkout.")
    validate.add_argument("--source", required=True, type=Path)
    validate.add_argument("--allow-dirty", action="store_true")
    validate.add_argument("--timeout", type=float, default=120.0)
    validate.add_argument("--output", type=Path)


def _add_tasks_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("tasks", help="List statically discovered task factories.")
    _add_catalog_source_args(parser)
    parser.add_argument("--family")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)


def _add_report_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("report", help="Show profile/API/family compatibility diagnostics.")
    _add_catalog_source_args(parser)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)


def _add_catalog_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("catalog", help="Emit stable catalog-wide diagnostics for users or CI.")
    _add_catalog_source_args(parser)
    parser.add_argument("--check", action="store_true", help="Fail if pinned golden compatibility metadata regresses.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path)


def _add_catalog_source_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--no-verify-source", action="store_true")


def _add_compile_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "compile",
        aliases=("dry-run",),
        help="Compile an unchanged supported task without model credentials.",
    )
    _add_task_args(parser)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--report", type=Path)


def _add_run_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("run", help="Run an unchanged supported task through a registered PyRIT target.")
    _add_task_args(parser)
    parser.add_argument("--config", type=Path, help="PyRIT configuration file that registers the target.")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--target", help="Exact target name from TargetRegistry.")
    target.add_argument(
        "--target-role",
        default="default_objective_target",
        help="Registry tag selecting exactly one target (default: default_objective_target).",
    )
    parser.add_argument("--result", type=Path)


def _add_task_args(parser: argparse.ArgumentParser) -> None:
    from pyrit.compat.inspect_ai.profile import PINNED_INSPECT_EVALS_PROFILE

    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--task",
        required=True,
        help="Relative module/file and factory, e.g. arc/arc.py@arc_challenge.",
    )
    parser.add_argument("--profile", default=PINNED_INSPECT_EVALS_PROFILE.profile_id)
    parser.add_argument("--task-param", action="append", default=[], metavar="NAME=JSON_VALUE")
    parser.add_argument("--data", type=Path, help="Local ARC JSON records; never executed.")
    parser.add_argument("--inspect-evals-cache-dir", type=Path, help="Prepared external task data/assets root.")
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--epochs", type=_positive_int)
    parser.add_argument("--attempts", type=_positive_int)
    parser.add_argument("--submission-attempts", type=_positive_int)
    parser.add_argument("--max-messages", type=_positive_int)
    parser.add_argument("--concurrency", type=_positive_int)
    parser.add_argument("--sandbox-provider", choices=("auto", "docker", "hyperv"), default="auto")
    parser.add_argument("--sandbox-config", type=Path, help="JSON overrides for the compiled Docker provider config.")
    parser.add_argument("--retain-sandboxes", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--no-verify-source", action="store_true")
    parser.add_argument("--worker-timeout", type=float, default=300.0)
    parser.add_argument("--source-verification-timeout", type=float, default=120.0)
    parser.add_argument("--case-timeout", type=float, default=300.0)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _run_inspect_evals(args: argparse.Namespace) -> int:
    if args.inspect_command is not None:
        from pyrit.compat.inspect_ai import cli

        handlers = {
            "source": cli.run_source_command,
            "tasks": cli.run_tasks_command,
            "report": cli.run_report_command,
            "catalog": cli.run_catalog_command,
            "compile": cli.run_compile_command,
            "dry-run": cli.run_compile_command,
            "run": cli.run_execute_command,
        }
        return handlers[args.inspect_command](args)
    if args.source is None:
        print("Error: --source is required.", file=sys.stderr)
        return 2
    from pyrit.scenario.capability_suite import (
        InspectEvalFamily,
        analyze_inspect_evals_source_tree,
        compile_inspect_eval_family,
        dump_manifest_json,
    )

    report = analyze_inspect_evals_source_tree(source_root=args.source)
    report_json = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.report is not None:
        args.report.write_text(report_json + "\n", encoding="utf-8")
    else:
        print(report_json)

    if args.family is None:
        return 0
    if args.manifest is None:
        print("Error: --manifest is required when --family is used.", file=sys.stderr)
        return 2
    manifest = compile_inspect_eval_family(
        family=InspectEvalFamily(args.family),
        source_root=args.source,
        data_path=args.data,
        case_ids=tuple(args.case_id),
        allow_network=args.allow_network,
    )
    manifest_json = json.dumps(dump_manifest_json(manifest), indent=2, sort_keys=True)
    args.manifest.write_text(manifest_json + "\n", encoding="utf-8")
    return 0


def main(args: list[str] | None = None) -> int:
    """
    Run the offline capability-suite CLI.

    Returns:
        int: Process exit code.
    """
    try:
        parsed = _build_parser().parse_args(args)
        if parsed.command == "inspect-evals":
            return _run_inspect_evals(parsed)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
