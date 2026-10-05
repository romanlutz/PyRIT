# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Backend compatibility fixtures independent of a packaged workspace stamp."""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID

import pytest

from pyrit import _compatibility
from pyrit.backend.main import app
from pyrit.backend.services.attack_service import get_attack_service
from pyrit.backend.services.manual_send_scheduler import get_manual_send_scheduler
from pyrit.backend.services.message_send_service import get_message_send_service
from pyrit.backend.services.scenario_run_service import reset_scenario_run_service_async
from pyrit.backend.services.scenario_service import get_scenario_service

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from pyrit.backend.models.original_worker import CohostBackendConfig


@pytest.fixture(autouse=True)
async def isolated_backend_services_async() -> AsyncIterator[None]:
    """Keep cached service owners bound to this test's memory and event loop."""
    factories = (
        get_attack_service,
        get_message_send_service,
        get_manual_send_scheduler,
        get_scenario_service,
    )
    await reset_scenario_run_service_async()
    for factory in factories:
        factory.cache_clear()
    try:
        yield
    finally:
        await reset_scenario_run_service_async()
        for factory in factories:
            factory.cache_clear()


@pytest.fixture(autouse=True)
def compatibility_id() -> Iterator[str]:
    """Supply startup provenance and initialize clients that deliberately skip lifespan."""
    identity = "0.14.0+g" + "a" * 40
    with (
        patch.object(_compatibility, "get_compatibility_id", return_value=identity),
        patch.object(app.state, "compatibility_id", identity, create=True),
    ):
        yield identity


@pytest.fixture
def compatibility_headers(compatibility_id: str) -> dict[str, str]:
    """Provide the marker required by business API requests."""
    return {_compatibility.COMPATIBILITY_HEADER: compatibility_id}


@pytest.fixture
def worker_config(*, tmp_path: Path, patch_central_database: None) -> CohostBackendConfig:
    """Stage only the public no-model child in an isolated, backend-owned fixture policy."""
    from pyrit.backend.models.original_worker import CohostBackendConfig, CohostRelayConfig, CohostSourcePolicy
    from pyrit.executor.benchmark.inspect_eval_source import EvalSourceFactory

    source = EvalSourceFactory.resolve_original_inert(family="inspect_original_inert")
    staged = tmp_path / "staged"
    staged.mkdir()
    entry = staged / "cohost_original_worker.py"
    shutil.copyfile(Path(__file__).with_name(entry.name), entry)
    worker_python = Path(sys.executable).resolve()
    environment = {"PUBLIC_COHOST_FIXTURE_REPO": str(Path.cwd())}
    if sys.platform == "win32":
        # The Windows venv redirector starts another PID; exercise a real owned interpreter instead.
        worker_python = Path(sys.base_prefix) / "python.exe"
        environment["PYTHONPATH"] = os.pathsep.join((str(Path(sys.prefix) / "Lib" / "site-packages"), str(Path.cwd())))
    return CohostBackendConfig(
        profile_ref="public-cohost",
        source_alias="public-inert",
        public_commit=_compatibility.get_compatibility_id().rsplit("+g", 1)[-1],
        worker_python=worker_python,
        worker_python_version=sys.version_info[:2],
        worker_inspect_version=version("inspect_ai"),
        worker_entrypoint=entry,
        worker_entrypoint_sha256=hashlib.sha256(entry.read_bytes()).hexdigest(),
        source_root=staged,
        jobs_root=tmp_path / "jobs",
        source=CohostSourcePolicy(
            contract_sha256="a" * 64,
            spec=source.spec,
            cases=(source.case,),
            primary_scorer="original_inert_scorer",
            display_values=frozenset({"1.0"}),
        ),
        relay=CohostRelayConfig(endpoint="https://pyrit-github-pipeline.openai.azure.com/"),
        allowed_operator_oids=frozenset({"00000000-0000-4000-8000-000000000003"}),
        allowed_group_ids=frozenset({"00000000-0000-4000-8000-000000000004"}),
        managed_identity_client_id=UUID("00000000-0000-4000-8000-000000000005"),
        worker_environment=environment,
        active_timeout_seconds=120,
        min_free_bytes=67_108_864,
        expected_database_name="public-fixture-only",
        result_container_url="",
        local_test=True,
    )
