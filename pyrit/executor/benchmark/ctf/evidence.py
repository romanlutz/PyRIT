# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import aiofiles

from pyrit.executor.benchmark.ctf.docker_environment import DockerCommandError, DockerExecutionError

if TYPE_CHECKING:
    from pathlib import Path

    import httpx

    from pyrit.executor.benchmark.ctf.docker_environment import DockerCTFEnvironment


class RunEvidence:
    """A prototype-specific journal; never records authentication headers or tokens."""

    def __init__(self, *, directory: Path, run_id: str) -> None:
        """Initialize evidence in an already-created, private run directory."""
        self.directory = directory
        self.state: dict[str, Any] = {
            "run_id": run_id,
            "status": "preparing",
            "attempt": 1,
            "epoch": 1,
            "model_requests": 0,
            "model_responses": [],
            "tool_executions": [],
            "final_output": None,
            "raw_grade": None,
            "score_id": None,
            "episode_seconds": None,
        }
        self._pending_call: dict[str, Any] | None = None
        self._seen_call_ids: set[str] = set()
        self._lock = asyncio.Lock()

    async def record_async(self, *, event: str, data: dict[str, Any]) -> None:
        """Append an event and atomically checkpoint the current manifest."""
        async with self._lock:
            entry = {"event": event, "time_utc": datetime.now(UTC).isoformat(), **data}
            async with aiofiles.open(self.directory / "events.jsonl", "a", encoding="utf-8") as stream:
                await stream.write(json.dumps(entry, ensure_ascii=True) + "\n")
                await stream.flush()
            temporary = self.directory / "manifest.json.tmp"
            async with aiofiles.open(temporary, "w", encoding="utf-8") as stream:
                await stream.write(json.dumps(self.state, indent=2, ensure_ascii=True) + "\n")
            await asyncio.to_thread(temporary.replace, self.directory / "manifest.json")

    async def on_request_async(self, request: httpx.Request) -> None:
        """
        Retain the actual model request body, without credentials.

        Raises:
            RuntimeError: If the episode is terminal or the model request budget is exhausted.
        """
        if self.state.get("terminal_error"):
            raise RuntimeError("The episode is terminal; further model requests are prohibited.")
        if self.state["model_requests"] >= 9:
            raise RuntimeError("The episode exhausted its nine model-request budget.")
        self.state["model_requests"] += 1
        sequence = self.state["model_requests"]
        request.extensions["native_ctf_sequence"] = sequence
        body = json.loads(await request.aread())
        await self.record_async(event="model_request", data={"sequence": sequence, "body": body})

    async def on_response_async(self, response: httpx.Response) -> None:
        """
        Retain provider output before PyRIT's target-level tool loop can fail.

        Raises:
            RuntimeError: If the provider returns parallel calls despite the disabled setting.
        """
        payload = await response.aread()
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            parsed = {"non_json_body": payload[:65536].decode("utf-8", errors="replace")}
        body: dict[str, Any] = parsed if isinstance(parsed, dict) else {"unexpected_json_body": parsed}
        sequence = response.request.extensions["native_ctf_sequence"]
        summary = {
            "sequence": sequence,
            "http_status": response.status_code,
            "request_id": response.headers.get("x-request-id") or response.headers.get("apim-request-id"),
            "response_id": body.get("id"),
            "model": body.get("model"),
            "status": body.get("status"),
            "usage": body.get("usage"),
        }
        self.state["model_responses"].append(summary)
        await self.record_async(event="model_response", data={**summary, "body": body})
        calls = [
            item
            for item in body.get("output", []) or []
            if isinstance(item, dict) and item.get("type") == "function_call"
        ]
        if len(calls) > 1:
            raise RuntimeError("Parallel tool calls are unsupported by this target loop.")
        if calls:
            call_id = calls[0].get("call_id")
            if not isinstance(call_id, str) or not call_id or call_id in self._seen_call_ids:
                raise RuntimeError("A provider tool call ID is missing or reused.")
            self._seen_call_ids.add(call_id)
        self._pending_call = {**calls[0], "request_sequence": sequence} if calls else None

    def claim_tool_call(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Correlate the callback's arguments with the just-received provider call.

        Returns:
            dict[str, Any]: The original call, including its provider-generated call ID.

        Raises:
            RuntimeError: If the callback cannot be matched to real provider evidence.
        """
        call = self._pending_call
        if call is None or call["name"] != name or json.loads(call["arguments"]) != arguments:
            raise RuntimeError("Tool execution has no matching provider call evidence.")
        self._pending_call = None
        return call


class DockerToolHarness:
    """Bind public Responses-target callbacks to the owned container and journal."""

    def __init__(self, *, environment: DockerCTFEnvironment, evidence: RunEvidence) -> None:
        """Initialize the container-only tool binding."""
        self.environment = environment
        self.evidence = evidence

    async def bash_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Run the model-requested bash command inside Docker.

        Returns:
            dict[str, Any]: The shared tool-result envelope.
        """
        return await self._execute_async(name="bash", arguments=arguments)

    async def python_async(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """
        Run the model-requested Python code inside Docker.

        Returns:
            dict[str, Any]: The shared tool-result envelope.
        """
        return await self._execute_async(name="python", arguments=arguments)

    async def _execute_async(self, *, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        call = self.evidence.claim_tool_call(name=name, arguments=arguments)
        entry = {
            "call_id": call["call_id"],
            "request_sequence": call["request_sequence"],
            "name": name,
            "arguments": arguments,
            "status": "requested",
            "execution_started_confirmed": False,
            "container_id": self.environment.container_id,
        }
        self.evidence.state["tool_executions"].append(entry)
        await self.evidence.record_async(event="tool_requested", data=entry)
        try:
            result = await self.environment.execute_async(name=name, arguments=arguments)
        except DockerExecutionError as error:
            await self._record_execution_failure_async(entry=entry, error=error)
            raise
        entry.update(
            {
                "status": "timed_out" if result["timed_out"] else "completed",
                "execution_started_confirmed": True,
                "result": result,
                "execution_id": result["execution_id"],
            }
        )
        if result["timed_out"]:
            self.evidence.state["terminal_error"] = "tool_timeout"
        await self.evidence.record_async(event="tool_finished", data=entry)
        if result["timed_out"]:
            raise RuntimeError("Docker tool timed out; the episode is terminal and cannot continue or be graded.")
        return result

    async def _record_execution_failure_async(self, *, entry: dict[str, Any], error: DockerExecutionError) -> None:
        self.evidence.state["terminal_error"] = error.reason
        entry.update(
            status="timed_out" if error.reason == "tool_timeout" else "execution_failed",
            execution_id=error.execution_id,
            execution_started_confirmed=error.execution_started,
            execution_error=error.reason,
            diagnostics=error.diagnostics,
        )
        if error.result is not None:
            entry["result"] = error.result
        await self.evidence.record_async(event="tool_failed", data=entry)
        try:
            await self.environment.terminate_async()
        except (DockerCommandError, OSError) as termination_error:
            entry["termination_error"] = {
                "type": type(termination_error).__name__,
                "message": str(termination_error),
            }
            await self.evidence.record_async(event="termination_failed", data=entry)
            raise error from termination_error
        entry["termination_status"] = "killed"
        await self.evidence.record_async(event="container_terminated", data=entry)
