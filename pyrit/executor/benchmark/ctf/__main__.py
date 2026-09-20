# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from pyrit.executor.benchmark.ctf.docker_environment import DockerCTFEnvironment
    from pyrit.executor.benchmark.ctf.evidence import RunEvidence
    from pyrit.executor.benchmark.ctf.gdm_intercode import CTFTask
    from pyrit.memory import SQLiteMemory


def _error_details(error: BaseException | None) -> list[dict[str, str]]:
    details = []
    while error is not None:
        details.append({"type": type(error).__name__, "message": str(error)})
        error = error.__cause__
    return details


def _create_memory(directory: Path) -> SQLiteMemory:
    from pyrit.memory import SQLiteMemory

    memory = SQLiteMemory(db_path=directory / "pyrit.db", silent=True)
    if not isinstance(memory, SQLiteMemory):
        raise TypeError("The SQLite singleton did not return a SQLiteMemory instance.")
    memory.results_path = str(directory)
    memory.disable_embedding()
    return memory


def _implementation_hashes() -> dict[str, str]:
    root = Path(__file__).parent
    files = [
        *root.glob("*.py"),
        root.parents[2] / "score" / "true_false" / "includes_scorer.py",
        root.parents[2] / "auth" / "azure_auth.py",
    ]
    return {str(path.relative_to(root.parents[3])): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


async def _export_memory_async(*, memory: SQLiteMemory | None, evidence: RunEvidence, run_id: str) -> None:
    if memory is None:
        evidence.state["memory_export_status"] = "not_initialized"
        return
    try:
        messages = await asyncio.to_thread(memory.get_conversation_messages, conversation_id=run_id)
        evidence.state["memory_messages"] = [message.model_dump(mode="json") for message in messages]
        evidence.state["memory_export_status"] = "exported"
    finally:
        await asyncio.to_thread(memory.dispose_engine)


async def _run_async(*, endpoint: str, model: str, directory: Path, run_id: str) -> dict[str, Any]:
    from pyrit.executor.benchmark.ctf.docker_environment import DockerCTFEnvironment
    from pyrit.executor.benchmark.ctf.evidence import RunEvidence
    from pyrit.executor.benchmark.ctf.gdm_intercode import GDMIntercodeTask4
    from pyrit.memory import CentralMemory

    evidence = RunEvidence(directory=directory, run_id=run_id)
    environment = DockerCTFEnvironment(image=GDMIntercodeTask4.IMAGE, run_id=run_id)
    memory: SQLiteMemory | None = None
    evidence.state.update(
        {
            "endpoint": endpoint,
            "requested_model": model,
            "auth": "pyrit.auth.get_azure_openai_auth",
            "versions": {package: version(package) for package in ("pyrit", "openai", "httpx", "azure-identity")},
            "python_version": sys.version,
            "memory_path": str(directory / "pyrit.db"),
            "limits": {
                "max_output_tokens": 2048,
                "tool_executions": 8,
                "tool_seconds": 30,
                "episode_seconds": 180,
                "cpus": 1,
                "memory_bytes": 536870912,
                "pids": 128,
            },
            "tool_schemas": GDMIntercodeTask4.tool_definitions(),
            "system_prompt": GDMIntercodeTask4.SYSTEM_PROMPT,
            "source_commit": await _git_commit_async(),
            "implementation_sha256": await asyncio.to_thread(_implementation_hashes),
        }
    )
    await evidence.record_async(event="run_started", data={"run_id": run_id})
    try:
        task = await GDMIntercodeTask4.load_async(directory=directory)
        evidence.state["source"] = task.provenance
        await environment.acquire_image_async()
        evidence.state["image"] = environment.image_identity
        started = time.monotonic()
        evidence.state["status"] = "running"
        try:
            async with asyncio.timeout(180):
                memory = await asyncio.to_thread(_create_memory, directory)
                CentralMemory.set_memory_instance(memory)
                await environment.start_async(files=task.files)
                evidence.state["container"] = environment.container_identity
                await evidence.record_async(event="container_started", data={"container_id": environment.container_id})
                await _evaluate_async(
                    endpoint=endpoint, model=model, task=task, environment=environment, evidence=evidence
                )
        finally:
            evidence.state["episode_seconds"] = time.monotonic() - started
    finally:
        error = sys.exception()
        if error is not None:
            evidence.state.update(status="error", errors=_error_details(error))
        evidence.state["container_id"] = environment.container_id
        evidence.state["container_name"] = environment.name
        evidence.state["tool_execution_attempts"] = environment.execution_attempts
        evidence.state["confirmed_tool_executions"] = environment.executions
        try:
            await environment.cleanup_async()
        finally:
            cleanup_error = sys.exception()
            evidence.state["cleanup_status"] = environment.cleanup_status
            if cleanup_error is not None and cleanup_error is not error:
                evidence.state.update(status="error", cleanup_errors=_error_details(cleanup_error))
            try:
                await _export_memory_async(memory=memory, evidence=evidence, run_id=run_id)
            finally:
                export_error = sys.exception()
                if export_error is not None and export_error is not cleanup_error:
                    evidence.state.update(status="error", memory_export_errors=_error_details(export_error))
                await evidence.record_async(event="run_finished", data={"status": evidence.state["status"]})
    return evidence.state


async def _evaluate_async(
    *,
    endpoint: str,
    model: str,
    task: CTFTask,
    environment: DockerCTFEnvironment,
    evidence: RunEvidence,
) -> None:
    import httpx
    from azure.identity.aio import DefaultAzureCredential

    from pyrit.auth import get_azure_openai_auth
    from pyrit.executor.benchmark.ctf.evidence import DockerToolHarness
    from pyrit.executor.benchmark.ctf.gdm_intercode import GDMIntercodeTask4
    from pyrit.executor.benchmark.ctf.native import NativeCTFBenchmark
    from pyrit.prompt_target import OpenAIResponseTarget
    from pyrit.score import IncludesScorer

    harness = DockerToolHarness(environment=environment, evidence=evidence)
    async with (
        DefaultAzureCredential(
            exclude_interactive_browser_credential=True, exclude_broker_credential=True
        ) as credential,
        httpx.AsyncClient(
            timeout=180,
            trust_env=False,
            event_hooks={"request": [evidence.on_request_async], "response": [evidence.on_response_async]},
        ) as client,
    ):
        target = OpenAIResponseTarget(
            endpoint=endpoint,
            model_name=model,
            api_key=get_azure_openai_auth(endpoint, credential=credential),
            headers="{}",
            max_output_tokens=2048,
            fail_on_missing_function=True,
            custom_functions={"bash": harness.bash_async, "python": harness.python_async},
            extra_body_parameters={
                "tools": GDMIntercodeTask4.tool_definitions(),
                "parallel_tool_calls": False,
                "store": False,
            },
            httpx_client_kwargs={"http_client": client, "max_retries": 0},
        )
        evidence.state["target_identifier"] = target.get_identifier().model_dump(mode="json")
        benchmark = NativeCTFBenchmark(
            objective_target=target, scorer=IncludesScorer(expected=task.expected, categories=["ctf_flag"])
        )
        run_id = evidence.state["run_id"]
        response = await benchmark.send_async(
            seed=task.seed,
            system_prompt=GDMIntercodeTask4.SYSTEM_PROMPT,
            conversation_id=run_id,
            run_id=run_id,
        )
        if environment.terminal_error or evidence.state.get("terminal_error"):
            raise RuntimeError("A terminal Docker execution failure prevents a final answer or grade.")
        if not evidence.state["tool_executions"] or not environment.executions:
            raise RuntimeError("No model-requested Docker tool execution occurred; this episode is not valid.")
        if environment.timed_out:
            raise RuntimeError("A Docker tool timed out; the episode is incomplete and will not be graded.")
        evidence.state["final_output"] = response.get_pieces_by_type(data_type="text")[0].converted_value
        score = await benchmark.score_async(response)
        evidence.state.update(
            status="completed",
            score_id=str(score.id),
            score=score.model_dump(mode="json"),
            raw_grade=score.score_metadata,
        )


async def _git_commit_async() -> str:
    from pyrit.executor.benchmark.ctf.docker_environment import run_process_async

    result = await run_process_async(arguments=["git", "rev-parse", "HEAD"], timeout=10)
    if result.returncode != 0:
        raise RuntimeError("Cannot establish the source commit for this prototype run.")
    return result.stdout.strip()


def main() -> None:
    """Run one explicitly configured native task-4 episode, with no retry."""
    parser = argparse.ArgumentParser(description="Native PyRIT task-4 CTF, minimal Docker/final-answer-only variant.")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("results") / "native-ctf")
    args = parser.parse_args()
    from pyrit.auth import is_azure_openai_endpoint

    if not args.endpoint.startswith("https://") or not is_azure_openai_endpoint(args.endpoint):
        parser.error("This operator binding requires an HTTPS Azure OpenAI endpoint.")
    os.environ["RETRY_MAX_NUM_ATTEMPTS"] = "1"
    run_id = str(uuid4())
    directory = (args.output_root / run_id).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Run evidence: {directory}", flush=True)
    state = asyncio.run(_run_async(endpoint=args.endpoint, model=args.model, directory=directory, run_id=run_id))
    print(
        json.dumps(
            {key: state[key] for key in ("run_id", "status", "final_output", "raw_grade", "cleanup_status")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
