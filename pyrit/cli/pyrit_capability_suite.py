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
        help="Statically analyze a checked-out inspect-evals tree without importing or executing it.",
    )
    inspect_parser.add_argument("--source", required=True, type=Path)
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
    return parser


def _run_inspect_evals(args: argparse.Namespace) -> int:
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
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
