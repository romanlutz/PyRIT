# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Standalone Linux-container runner; only its framed output is a control channel."""

from __future__ import annotations

import base64
import json
import os
import selectors
import subprocess
import sys
import time
from typing import Any


class _CommandRunner:
    _CHUNK_SIZE = 4096

    def __init__(self, *, execution_id: str, arguments: list[str], timeout: float, output_limit: int) -> None:
        self._execution_id = execution_id
        self._arguments = arguments
        self._timeout = timeout
        self._output_limit = output_limit
        self._buffers = {"stdout": bytearray(), "stderr": bytearray()}
        self._retained = {"stdout": 0, "stderr": 0}
        self._truncated = False

    def run(self) -> None:
        try:
            process = subprocess.Popen(
                self._arguments,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
            )
        except OSError as error:
            self._emit(event="launch_error", data={"message": str(error)})
            return
        assert process.stdout is not None and process.stderr is not None
        try:
            self._emit(event="started", data={"pid": process.pid})
            deadline = time.monotonic() + self._timeout
            timed_out = self._read_output(process=process, deadline=deadline)
            returncode = None
            if not timed_out:
                try:
                    returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
            for stream in self._buffers:
                self._flush(stream=stream, partial=True)
            self._emit(
                event="finished",
                data={"returncode": returncode, "timed_out": timed_out, "truncated": self._truncated},
            )
        finally:
            # The host journals a timeout before terminating the entire owned container.
            process.stdout.close()
            process.stderr.close()

    def _read_output(self, *, process: subprocess.Popen[bytes], deadline: float) -> bool:
        assert process.stdout is not None and process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return True
                for key, _ in selector.select(timeout=remaining):
                    chunk = os.read(key.fd, 8192)
                    stream = key.data
                    if not chunk:
                        selector.unregister(key.fileobj)
                        self._flush(stream=stream, partial=True)
                        continue
                    kept = chunk[: self._output_limit - self._retained[stream]]
                    self._truncated |= len(kept) < len(chunk)
                    self._retained[stream] += len(kept)
                    self._buffers[stream].extend(kept)
                    self._flush(stream=stream, partial=False)
        return False

    def _flush(self, *, stream: str, partial: bool) -> None:
        buffer = self._buffers[stream]
        while len(buffer) >= self._CHUNK_SIZE or (partial and buffer):
            chunk = bytes(buffer[: self._CHUNK_SIZE])
            del buffer[: self._CHUNK_SIZE]
            self._emit(event="output", data={"stream": stream, "data": base64.b64encode(chunk).decode("ascii")})

    def _emit(self, *, event: str, data: dict[str, Any]) -> None:
        print(json.dumps({"event": event, "execution_id": self._execution_id, **data}), flush=True)


def main() -> None:
    """Run the fixed container protocol, without importing PyRIT inside the container."""
    _CommandRunner(
        execution_id=sys.argv[1],
        arguments=json.loads(sys.argv[2]),
        timeout=float(sys.argv[3]),
        output_limit=int(sys.argv[4]),
    ).run()


if __name__ == "__main__":
    main()
