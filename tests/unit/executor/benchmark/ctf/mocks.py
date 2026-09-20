# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
import json
from collections.abc import Callable
from typing import Any

from pyrit.executor.benchmark.ctf.docker_environment import CommandResult


def runner_transport(
    *, execution_id: str, result: CommandResult, finished: bool = True, transport_exit: int = 0
) -> CommandResult:
    records: list[dict[str, Any]] = [{"event": "started", "execution_id": execution_id, "pid": 42}]
    for stream, value in (("stdout", result.stdout), ("stderr", result.stderr)):
        if value:
            records.append(
                {
                    "event": "output",
                    "execution_id": execution_id,
                    "stream": stream,
                    "data": base64.b64encode(value.encode("utf-8")).decode("ascii"),
                }
            )
    if finished:
        records.append(
            {
                "event": "finished",
                "execution_id": execution_id,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "truncated": result.truncated,
            }
        )
    return CommandResult(
        stdout="".join(json.dumps(record) + "\n" for record in records), stderr="", returncode=transport_exit
    )


def mock_docker_exec(result: CommandResult) -> Callable[..., CommandResult]:
    def run(*, arguments: list[str], timeout: float, output_limit: int) -> CommandResult:
        assert timeout <= 30 and output_limit == 65536
        return runner_transport(execution_id=arguments[-4], result=result)

    return run
