# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Strict JSON decoding shared by the independent model wire transports."""

import json
import math
from typing import Any, NoReturn


def _reject_json_constant(value: str) -> NoReturn:
    raise ValueError("Non-finite JSON numbers are not supported")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Non-finite JSON numbers are not supported")
    return number


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON fields are not supported")
        result[key] = value
    return result


def strict_json_loads(*, value: bytes | str) -> Any:
    """
    Reject duplicate fields and non-finite numbers before validating wire shapes.

    Args:
        value (bytes | str): Original JSON bytes or SSE event data.

    Returns:
        Any: Parsed JSON value with finite numbers and unique object keys.
    """
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        parse_float=_finite_json_float,
        object_pairs_hook=_unique_json_object,
    )
