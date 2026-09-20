# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Bounded stream capture, executed only inside the Linux Inspect sandbox."""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import sys
import time
from typing import Any


def _decode(*, raw: bytearray, limit: int) -> tuple[str, bool]:
    encoded = raw.decode("utf-8", errors="replace").encode("utf-8")
    return encoded[:limit].decode("utf-8", errors="ignore"), len(encoded) > limit


def _kill_group(process: subprocess.Popen[bytes]) -> str | None:
    if sys.platform == "linux":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return None
        except OSError as error:
            return f"{type(error).__name__}: {error}"
    else:
        raise RuntimeError("Process-group cleanup requires the Linux task sandbox.")
    return None


def _frame(*, event: str, execution_id: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"protocol": "pyrit-inspect-tool-v1", "event": event, "execution_id": execution_id, **payload},
            ensure_ascii=True,
        ),
        flush=True,
    )


def _capture(*, command: list[str], timeout: float, limit: int, execution_id: str) -> tuple[dict[str, Any], str | None]:
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    truncated = False
    timed_out = False
    termination_error = None
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    _frame(event="started", execution_id=execution_id)
    try:
        if process.stdout is None or process.stderr is None:
            raise RuntimeError("Tool output pipes were not created.")
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    termination_error = _kill_group(process)
                    break
                for key, _ in selector.select(timeout=min(remaining, 0.1)):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffer = buffers[key.data]
                    available = max(0, limit - len(buffer))
                    buffer.extend(chunk[:available])
                    truncated = truncated or len(chunk) > available
            if not timed_out:
                try:
                    process.wait(timeout=max(0.001, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    termination_error = _kill_group(process)
            if timed_out:
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    termination_error = termination_error or "Process group did not stop within the capture grace."
        returncode = None if timed_out else process.returncode
    finally:
        process.stdout.close()
        process.stderr.close()
    stdout, stdout_rebounded = _decode(raw=buffers["stdout"], limit=limit)
    stderr, stderr_rebounded = _decode(raw=buffers["stderr"], limit=limit)
    truncated = truncated or stdout_rebounded or stderr_rebounded
    error = "tool_timeout" if timed_out else "output_truncated" if truncated else "nonzero_exit" if returncode else None
    return (
        {
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "timed_out": timed_out,
            "truncated": truncated,
            "error": error,
            "execution_id": execution_id,
        },
        termination_error,
    )


def main() -> None:
    """
    Run the supplied tool with finite memory and time, and emit its JSON envelope.

    Raises:
        RuntimeError: If this worker is accidentally invoked outside Linux.
    """
    if sys.platform != "linux":
        raise RuntimeError("This worker must execute inside the Linux task sandbox, never on the host.")
    options = json.load(sys.stdin)
    result, termination_error = _capture(
        command=options["command"],
        timeout=options["timeout"],
        limit=options["limit"],
        execution_id=options["execution_id"],
    )
    _frame(
        event="completed",
        execution_id=options["execution_id"],
        result=result,
        termination_error=termination_error,
    )


if __name__ == "__main__":
    main()
