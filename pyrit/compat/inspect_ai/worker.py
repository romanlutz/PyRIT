# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Dedicated process entry point for trusted Inspect source construction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pyrit.compat.inspect_ai.loader import _load_inspect_eval_in_process, _serialize_loaded
from pyrit.compat.inspect_ai.profile import UnsupportedInspectFeatureError


def main() -> int:
    """
    Execute one compatibility construction request.

    Returns:
        int: Zero after writing a structured worker response.
    """
    request_path = Path(sys.argv[1])
    response_path = Path(sys.argv[2])
    request = json.loads(request_path.read_text(encoding="utf-8"))
    try:
        loaded = _load_inspect_eval_in_process(
            source_root=Path(request["source_root"]),
            task_spec=request["task_spec"],
            task_parameters=request["task_parameters"],
            profile_id=request["profile_id"],
            dataset_records=request["dataset_records"],
            inspect_evals_cache_dir=(
                Path(request["inspect_evals_cache_dir"]) if request["inspect_evals_cache_dir"] is not None else None
            ),
            allow_network=request["allow_network"],
            verify_source_revision=request["verify_source_revision"],
            source_verification_timeout_seconds=request["source_verification_timeout_seconds"],
            case_timeout_seconds=request["case_timeout_seconds"],
        )
        response: dict[str, Any] = {"ok": True, "loaded": _serialize_loaded(loaded)}
    except Exception as error:
        error_data = {
            "type": type(error).__name__,
            "message": str(error),
        }
        if isinstance(error, UnsupportedInspectFeatureError):
            error_data.update(
                {
                    "symbol": error.symbol,
                    "source_profile": error.source_profile,
                    "remediation": error.remediation,
                }
            )
        response = {"ok": False, "error": error_data}
    response_path.write_text(json.dumps(response), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
