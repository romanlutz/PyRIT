# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from azure.identity.aio import DefaultAzureCredential

from examples.inspect_ctf import _run_live_async
from pyrit.executor.benchmark import InspectBenchmark
from pyrit.executor.benchmark._inspect_artifacts import InspectRunArtifacts
from pyrit.executor.benchmark.inspect_sandbox import InspectDockerProfile

pytest.importorskip("inspect_ai")


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize("error_type", [None, RuntimeError, asyncio.CancelledError])
async def test_operator_closes_credential_and_client_on_every_exit_async(
    tmp_path: Path, error_type: type[BaseException] | None
) -> None:
    from inspect_ai import Task
    from inspect_ai.dataset import Sample

    artifacts = InspectRunArtifacts(directory=tmp_path / "run", provenance={})
    profile = InspectDockerProfile(artifacts=artifacts, image="python:test@sha256:123")
    task = Task(dataset=[Sample(id=4, input="fixture task", target="fixture")])
    credential = DefaultAzureCredential(exclude_interactive_browser_credential=True, exclude_broker_credential=True)
    benchmark = MagicMock(spec=InspectBenchmark)
    benchmark.execute_async = AsyncMock(side_effect=error_type("fixture") if error_type else None)
    clients: list[httpx.AsyncClient] = []

    class TrackingClient(httpx.AsyncClient):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            clients.append(self)

    with (
        patch("pyrit.setup.initialize_pyrit_async", new_callable=AsyncMock),
        patch("azure.identity.aio.DefaultAzureCredential", return_value=credential) as credential_factory,
        patch.object(credential, "close", new_callable=AsyncMock, wraps=credential.close) as close_credential,
        patch("httpx.AsyncClient", new=TrackingClient),
        patch("pyrit.auth.get_azure_openai_auth", return_value=AsyncMock(return_value="fixture-token")) as auth,
        patch("pyrit.executor.benchmark.InspectBenchmark", return_value=benchmark),
        patch("examples.inspect_ctf._prepare_task", return_value=task),
    ):
        if error_type:
            with pytest.raises(error_type, match="fixture"):
                await _run_live_async(
                    artifacts=artifacts,
                    profile=profile,
                    endpoint="https://fixture.openai.azure.com/openai/v1",
                    model="fixture",
                )
        else:
            await _run_live_async(
                artifacts=artifacts,
                profile=profile,
                endpoint="https://fixture.openai.azure.com/openai/v1",
                model="fixture",
            )
    assert credential_factory.call_args.kwargs == {
        "exclude_interactive_browser_credential": True,
        "exclude_broker_credential": True,
    }
    assert auth.call_args.kwargs["credential"] is credential
    close_credential.assert_awaited_once()
    assert clients and all(client.is_closed for client in clients)
