# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Fail-closed native relay, exact startup authority and durable validation-scope controls."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import os
from datetime import UTC, datetime, timedelta
from time import monotonic
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import httpx
import pytest
from azure.core import MatchConditions
from azure.core.exceptions import AzureError, ResourceNotFoundError
from pydantic import JsonValue, TypeAdapter, ValidationError
from starlette.requests import Request

from pyrit.backend.models.original_worker import CohostBackendConfig, CohostRelayConfig, CohostValidationScope
from pyrit.backend.routes import original_model
from pyrit.backend.services.original_model_relay import OriginalModelRelay, OriginalRelayError
from pyrit.backend.services.original_worker_preflight import CohostPreflight, CohostPreflightError
from pyrit.backend.services.original_worker_state import CohostValidationState, bootstrap_validation_state
from pyrit.setup.configuration_loader import ConfigurationLoader

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from starlette.types import Message


@pytest.fixture(name="relay")
async def relay_async(worker_config: CohostBackendConfig) -> AsyncIterator[OriginalModelRelay]:
    await asyncio.to_thread(worker_config.jobs_root.mkdir)
    payload = {
        "id": "public-response",
        "choices": [],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    }
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    service = OriginalModelRelay(
        config=worker_config.relay, preflight=CohostPreflight(config=worker_config, environment={}), client=client
    )
    with patch.object(service, "_model_token_async", AsyncMock(return_value=("not-a-bearer-token", {}))):
        yield service
    await service.shutdown_async()


def _admit(*, relay: OriginalModelRelay, root: Path) -> tuple[UUID, str]:
    root.mkdir()
    job = uuid4()
    capability = relay.admit(job_ref=job, run_id=uuid4(), run_root=root)
    return job, capability


async def _forward_async(
    *, relay: OriginalModelRelay, job: UUID, capability: str, identifier: str = "public-inspect-one"
) -> None:
    await relay.forward_async(
        job_ref=job,
        capability=capability,
        content=b'{"model":"gpt-4-32","messages":[{"role":"user","content":"Public synthetic"}]}',
        inspect_request_id=identifier,
    )


async def test_relay_preserves_incoming_ids_exact_request_response_and_authentic_usage_async(
    *, relay: OriginalModelRelay, tmp_path: Path
) -> None:
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    await _forward_async(relay=relay, job=job, capability=capability)
    [row] = relay._jobs[job].records
    assert row["inspect_request_id"] == "public-inspect-one"
    assert row["request_id"] != row["inspect_request_id"]
    assert row["response_id"] == "public-response"
    assert row["prompt_tokens"] == 8 and row["completion_tokens"] == 2
    assert row["total_tokens"] == row["upstream_usage_total_tokens"] == 10
    assert len(row["request_payload_sha256"]) == len(row["response_sha256"]) == 64
    receipt = await relay.close_async(job_ref=job, capability=capability)
    assert receipt["request_count"] == 1
    assert receipt["upstream_usage_total_tokens"] == 10
    assert receipt["usage_complete"] is True and receipt["upstream_drained"] is True


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b'{"messages":[]}',
        b'{"messages":[{}],"stream":true}',
        b'{"model":"foreign","messages":[{}]}',
        b'{"messages":[{}],"n":2}',
        b'{"messages":[{}],"max_tokens":4097}',
        b'{"messages":[{}],"max_tokens":true}',
        b'{"messages":[{}],"url":"https://example.test"}',
        pytest.param(b"x" * 524_289, id="oversized-request"),
    ],
)
async def test_invalid_payload_permanently_closes_before_credentials_or_dispatch_async(
    *, relay: OriginalModelRelay, tmp_path: Path, payload: bytes
) -> None:
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    with pytest.raises(OriginalRelayError):
        await relay.forward_async(
            job_ref=job, capability=capability, content=payload, inspect_request_id="public-inspect-one"
        )
    relay._model_token_async.assert_not_awaited()
    assert relay._request_count == 0
    with pytest.raises(OriginalRelayError, match="relay_capability_invalid"):
        await _forward_async(relay=relay, job=job, capability=capability)


async def test_qualification_is_first_once_and_included_in_normal_budgets_async(
    *, relay: OriginalModelRelay, tmp_path: Path
) -> None:
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    qualification = b'{"messages":[{"role":"user","content":"Reply only OK"}],"max_tokens":8}'
    await relay.forward_async(job_ref=job, capability=capability, content=qualification, request_kind="qualification")
    with pytest.raises(OriginalRelayError, match="qualification_not_admitted"):
        await relay.forward_async(
            job_ref=job, capability=capability, content=qualification, request_kind="qualification"
        )
    closed = await relay.close_async(job_ref=job)
    assert closed["request_count"] == 1 and closed["upstream_usage_total_tokens"] == 10
    assert relay._request_count == 1 and relay._observed_tokens == 10
    assert relay._jobs[job].records[0]["request_kind"] == "qualification"


async def test_duplicate_or_missing_inspect_id_is_not_a_second_original_post_async(
    *, relay: OriginalModelRelay, tmp_path: Path
) -> None:
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    await _forward_async(relay=relay, job=job, capability=capability)
    with pytest.raises(OriginalRelayError, match="inspect_request_id_invalid"):
        await _forward_async(relay=relay, job=job, capability=capability)
    assert relay._request_count == 1


async def test_caller_cancellation_does_not_cancel_actual_upstream_or_claim_drain_async(
    *, relay: OriginalModelRelay, tmp_path: Path
) -> None:
    entered, finish = asyncio.Event(), asyncio.Event()

    async def respond_async(request: httpx.Request) -> httpx.Response:
        entered.set()
        await finish.wait()
        return httpx.Response(
            200,
            json={"id": "public-response", "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond_async))
    await relay._client.aclose()
    relay._client = client
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    caller = asyncio.create_task(_forward_async(relay=relay, job=job, capability=capability))
    await asyncio.wait_for(entered.wait(), 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    assert relay.has_active_work()
    closing = asyncio.create_task(relay.close_async(job_ref=job))
    await asyncio.sleep(0)
    assert not closing.done()
    finish.set()
    receipt = await closing
    assert receipt["upstream_drained"] is True and receipt["request_count"] == 1
    assert receipt["upstream_usage_total_tokens"] == 2


@pytest.mark.parametrize("response", [httpx.Response(500), httpx.Response(200, json={"id": "missing-usage"})])
async def test_unknown_unmetered_dispatch_blocks_later_admissions_and_restores_uncertainty_async(
    *, relay: OriginalModelRelay, tmp_path: Path, response: httpx.Response
) -> None:
    await relay._client.aclose()
    relay._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: response))
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    with pytest.raises(OriginalRelayError):
        await _forward_async(relay=relay, job=job, capability=capability)
    assert not relay.allows_new_run()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200)))
    restored = OriginalModelRelay(config=relay.config, preflight=relay.preflight, client=client)
    await restored.startup_async()
    assert not restored.allows_new_run()
    assert restored._request_count == 1
    await restored.shutdown_async()


def _scope(worker_config: CohostBackendConfig) -> tuple[CohostValidationState, dict[str, JsonValue]]:
    scope = CohostValidationScope(
        instance_id=uuid4(),
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        state_container_url="https://publicfixture.blob.core.windows.net/validation-state",
    )
    initial = worker_config.model_copy(update={"validation_scope": scope})
    bound, content = bootstrap_validation_state(config=initial)
    service = CohostValidationState(preflight=CohostPreflight(config=bound, environment={}))
    value = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
    service._key = service._validate_value(value)
    service._value = value
    service._lease_deadline = monotonic() + 60
    service._available = True
    service._etag = "owned-etag"
    service._lease = MagicMock()
    service._blob = MagicMock()
    service._blob.upload_blob = AsyncMock(return_value={"etag": "next-etag"})
    return service, value


async def test_validation_pair_consumes_before_use_never_recycles_and_restores_closed_counter_async(
    worker_config: CohostBackendConfig,
) -> None:
    scope, _ = _scope(worker_config)
    for _ in range(2):
        reservation = uuid4()
        await scope.reserve_async(
            reservation_id=reservation,
            operator_oid=next(iter(worker_config.allowed_operator_oids)),
            profile_ref=worker_config.profile_ref,
        )
        assert not scope.can_reserve()
        await scope.bind_job_async(reservation_id=reservation, app_run_id=uuid4(), job_ref=uuid4(), control_id=uuid4())
        await scope.finish_job_async(reservation_id=reservation, proved=True)
    assert not scope.can_reserve()
    with pytest.raises(CohostPreflightError, match="cannot admit"):
        await scope.reserve_async(
            reservation_id=uuid4(),
            operator_oid=next(iter(worker_config.allowed_operator_oids)),
            profile_ref=worker_config.profile_ref,
        )
    restored = CohostValidationState(preflight=scope.preflight)
    restored._key = restored._validate_value(scope._value)
    restored._value = copy.deepcopy(scope._value)
    restored._available, restored._lease_deadline = True, monotonic() + 60
    assert not restored.can_reserve()
    assert scope._blob.upload_blob.call_args.kwargs["match_condition"] is MatchConditions.IfNotModified
    assert scope._blob.upload_blob.call_args.kwargs["lease"] is scope._lease


async def test_unknown_prior_job_state_loss_or_failed_remote_commit_never_resets_admission_async(
    worker_config: CohostBackendConfig,
) -> None:
    scope, initial = _scope(worker_config)
    await scope.reserve_async(
        reservation_id=uuid4(),
        operator_oid=next(iter(worker_config.allowed_operator_oids)),
        profile_ref=worker_config.profile_ref,
    )
    restored = CohostValidationState(preflight=scope.preflight)
    restored._key = restored._validate_value(scope._value)
    restored._value = copy.deepcopy(scope._value)
    restored._available, restored._lease_deadline = True, monotonic() + 60
    assert restored.has_unknown_job() and not restored.can_reserve()
    with pytest.raises(CohostPreflightError):
        restored._validate_value({})
    tampered = copy.deepcopy(initial)
    tampered["jobs"] = []
    tampered["model"]["requests"] = 1
    with pytest.raises(CohostPreflightError, match="modified"):
        restored._validate_value(tampered)
    other, _ = _scope(worker_config)
    other._blob.upload_blob = AsyncMock(side_effect=AzureError("public mocked storage outage"))
    with pytest.raises(CohostPreflightError, match="commit is uncertain"):
        await other.reserve_async(
            reservation_id=uuid4(),
            operator_oid=next(iter(worker_config.allowed_operator_oids)),
            profile_ref=worker_config.profile_ref,
        )
    assert not other.can_reserve()
    assert other._value["jobs"] == []


async def test_model_remote_dispatch_intent_is_committed_before_http_and_failed_barrier_sends_nothing_async(
    *, worker_config: CohostBackendConfig, tmp_path: Path
) -> None:
    scope, _ = _scope(worker_config)
    await asyncio.to_thread(worker_config.jobs_root.mkdir)
    sent: list[httpx.Request] = []
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request: sent.append(request)))
    relay = OriginalModelRelay(
        config=worker_config.relay, preflight=scope.preflight, client=client, validation_state=scope
    )
    await relay.startup_async()
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    scope._blob.upload_blob = AsyncMock(side_effect=AzureError("public mocked outage"))
    with patch.object(relay, "_model_token_async", AsyncMock(return_value=("not-a-token", {}))):
        with pytest.raises(OriginalRelayError):
            await _forward_async(relay=relay, job=job, capability=capability)
    assert sent == []
    assert not scope.is_available() and not relay.allows_new_run()
    await relay.shutdown_async()


@pytest.mark.parametrize(
    "changed", ["enable_live_reinitialization", "allow_custom_initializers", "max_concurrent_scenario_runs"]
)
def test_preview_rejects_mutable_initialization_and_parallel_scenarios(
    *, worker_config: CohostBackendConfig, changed: str
) -> None:
    loader = ConfigurationLoader(
        memory_db_type="sqlite", env_files=[], initialization_scripts=[], max_concurrent_scenario_runs=1
    )
    setattr(loader, changed, 2 if changed == "max_concurrent_scenario_runs" else True)
    with pytest.raises(CohostPreflightError, match="immutable"):
        CohostPreflight(config=worker_config, environment={}).validate_configuration(loader=loader, environment={})


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://pyrit-github-pipeline.openai.azure.com",
        "https://example.test",
        "https://pyrit-github-pipeline.openai.azure.com/other",
        "https://pyrit-github-pipeline.openai.azure.com/?key=not-real",
    ],
)
def test_fixed_route_rejects_arbitrary_url_secret_or_profile(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        CohostRelayConfig(endpoint=endpoint)


async def test_descriptor_hash_dev_gate_and_installed_child_pin_fail_before_launch_async(
    *, worker_config: CohostBackendConfig, tmp_path: Path
) -> None:
    path = tmp_path / "descriptor.json"
    await asyncio.to_thread(path.write_text, worker_config.model_dump_json(), encoding="utf-8")
    digest = hashlib.sha256(await asyncio.to_thread(path.read_bytes)).hexdigest()
    with patch.dict(
        os.environ,
        {
            CohostPreflight.CONFIG_ENV: str(path),
            CohostPreflight.CONFIG_SHA_ENV: "0" * 64,
            "PYRIT_DEV_MODE": "true",
        },
    ):
        with pytest.raises(CohostPreflightError, match="digest differs"):
            await CohostPreflight.from_environment_async()
    with patch.dict(
        os.environ,
        {
            CohostPreflight.CONFIG_ENV: str(path),
            CohostPreflight.CONFIG_SHA_ENV: digest,
            "PYRIT_DEV_MODE": "false",
        },
    ):
        with pytest.raises(CohostPreflightError, match="Local SQLite fixtures"):
            await CohostPreflight.from_environment_async()
    preflight = CohostPreflight(config=worker_config, environment={})
    with patch.object(
        preflight,
        "_probe_worker_async",
        AsyncMock(
            return_value={
                "python": list(worker_config.worker_python_version),
                "inspect": "not-qualified",
                "compatibility_id": "not-qualified",
            }
        ),
    ):
        with pytest.raises(CohostPreflightError, match="runtime differs"):
            await preflight.verify_staging_async()


@pytest.mark.parametrize("stage", ["credential", "upstream"])
async def test_native_deadline_includes_credentials_and_unknown_dispatch_stays_blocked_async(
    *, relay: OriginalModelRelay, tmp_path: Path, stage: str
) -> None:
    relay.config = relay.config.model_copy(update={"request_timeout_seconds": 1})
    never = asyncio.Event()

    async def token_async() -> tuple[str, dict[str, JsonValue]]:
        await never.wait()
        raise AssertionError("The bounded native credential should time out.")

    async def respond_async(request: httpx.Request) -> httpx.Response:
        await never.wait()
        raise AssertionError("The bounded native upstream should time out.")

    if stage == "credential":
        relay._model_token_async = token_async
    else:
        await relay._client.aclose()
        relay._client = httpx.AsyncClient(transport=httpx.MockTransport(respond_async))
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    with pytest.raises(OriginalRelayError, match="upstream_unverified"):
        await _forward_async(relay=relay, job=job, capability=capability)
    assert not relay.has_active_work()
    assert relay._request_count == int(stage == "upstream")
    assert relay.allows_new_run() is (stage == "credential")
    closed = await relay.close_async(job_ref=job)
    assert closed["upstream_drained"] is (stage == "credential")
    assert closed["usage_complete"] is (stage == "credential")


async def test_oversized_native_response_never_publishes_guessed_usage_async(
    *, relay: OriginalModelRelay, tmp_path: Path
) -> None:
    relay.config = relay.config.model_copy(update={"max_response_bytes": 1024})
    await relay._client.aclose()
    relay._client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 1025))
    )
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    with pytest.raises(OriginalRelayError, match="response_too_large"):
        await _forward_async(relay=relay, job=job, capability=capability)
    assert not relay.allows_new_run() and relay._observed_tokens == 0
    closed = await relay.close_async(job_ref=job)
    assert closed["upstream_drained"] is False and closed["usage_complete"] is False


async def test_authentic_final_response_may_overshoot_observed_stop_without_rewriting_it_async(
    *, relay: OriginalModelRelay, tmp_path: Path
) -> None:
    relay.config = relay.config.model_copy(update={"max_observed_tokens": 1})
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    content, _ = await relay.forward_async(
        job_ref=job,
        capability=capability,
        content=b'{"messages":[{"role":"user","content":"Public synthetic"}]}',
        inspect_request_id="public-one",
    )
    assert TypeAdapter(dict[str, JsonValue]).validate_json(content)["usage"]["total_tokens"] == 10
    assert relay._jobs[job].observed_tokens == 10
    with pytest.raises(OriginalRelayError, match="capability_invalid"):
        await _forward_async(relay=relay, job=job, capability=capability, identifier="public-two")


@pytest.mark.parametrize("kind", ["request", "close"])
@pytest.mark.parametrize("mode,status", [("partial", 408), ("disconnect", 400), ("overflow", 413)])
async def test_actual_route_bounds_body_before_credential_or_dispatch_and_revokes_async(
    *, relay: OriginalModelRelay, tmp_path: Path, kind: str, mode: str, status: int
) -> None:
    job, capability = await asyncio.to_thread(_admit, relay=relay, root=tmp_path / "role")
    first = True

    async def receive_async() -> Message:
        nonlocal first
        if mode == "disconnect":
            return {"type": "http.disconnect"}
        if mode == "overflow":
            return {"type": "http.request", "body": b"x" * (relay.config.max_request_bytes + 1), "more_body": False}
        if first:
            first = False
            return {"type": "http.request", "body": b"{", "more_body": True}
        if kind == "close":
            assert not relay._jobs[job].accepting
        await asyncio.Event().wait()
        raise AssertionError("Partial body must time out.")

    request = Request(
        {"type": "http", "headers": [(b"authorization", ("Bearer " + capability).encode())]}, receive=receive_async
    )
    route = (
        original_model.original_model_completion_async
        if kind == "request"
        else original_model.original_model_close_async
    )
    with (
        patch.object(original_model, "_relay", return_value=relay),
        patch.object(relay, "REQUEST_BODY_TIMEOUT_SECONDS", 0.02),
    ):
        response = await route(job_ref=job, request=request)
    assert response.status_code == status
    relay._model_token_async.assert_not_awaited()
    assert not relay._jobs[job].accepting and relay._request_count == 0


@pytest.mark.parametrize("missing", [False, True])
async def test_remote_state_startup_uses_existing_exact_blob_lease_and_never_initializes_async(
    *, worker_config: CohostBackendConfig, missing: bool
) -> None:
    from azure.identity.aio import DefaultAzureCredential
    from azure.storage.blob.aio import BlobClient, BlobLeaseClient

    seed, value = _scope(worker_config)
    state = CohostValidationState(preflight=seed.preflight)
    credential = MagicMock(spec=DefaultAzureCredential)
    credential.close = AsyncMock()
    blob, lease = MagicMock(spec=BlobClient), MagicMock(spec=BlobLeaseClient)
    blob.close, lease.acquire, lease.release, lease.renew = AsyncMock(), AsyncMock(), AsyncMock(), AsyncMock()

    async def chunks_async() -> AsyncIterator[bytes]:
        content = CohostValidationState._content(value)
        yield content[:10]
        yield content[10:]

    stream = SimpleNamespace(properties=SimpleNamespace(etag="owned-etag"), chunks=chunks_async)
    blob.download_blob = AsyncMock(
        side_effect=ResourceNotFoundError("Public missing state") if missing else None,
        return_value=stream,
    )
    with (
        patch("azure.identity.aio.DefaultAzureCredential", return_value=credential),
        patch.object(BlobClient, "from_blob_url", return_value=blob),
        patch("azure.storage.blob.aio.BlobLeaseClient", return_value=lease),
    ):
        if missing:
            with pytest.raises(CohostPreflightError, match="unavailable or unverified"):
                await state.startup_async()
            assert not state.is_available()
        else:
            await state.startup_async()
            assert state.is_available() and state.can_reserve()
            assert state._etag == "owned-etag"
            lease.acquire.assert_awaited_once()
            assert lease.acquire.call_args.kwargs["lease_duration"] == 60
            assert blob.download_blob.call_args.kwargs["lease"] is lease
            await state.shutdown_async()
    blob.upload_blob.assert_not_called()
    credential.close.assert_awaited_once()
    lease.release.assert_awaited_once()


@pytest.mark.parametrize("cause", ["renew_error", "expired", "missing_etag"])
async def test_remote_authority_loss_fails_closed_without_recycling_admission_async(
    *, worker_config: CohostBackendConfig, cause: str
) -> None:
    state, _ = _scope(worker_config)
    if cause == "missing_etag":
        state._blob.upload_blob = AsyncMock(return_value={})
        with pytest.raises(CohostPreflightError, match="uncertain"):
            await state.reserve_async(
                reservation_id=uuid4(),
                operator_oid=next(iter(worker_config.allowed_operator_oids)),
                profile_ref=worker_config.profile_ref,
            )
        assert state._value["jobs"] == []
    else:
        state._lease.renew = AsyncMock(side_effect=AzureError("Public lease loss"))
        if cause == "expired":
            state._lease_deadline = monotonic() - 1
        with patch("pyrit.backend.services.original_worker_state.asyncio.sleep", new_callable=AsyncMock):
            await state._renew_async()
        if cause == "expired":
            state._lease.renew.assert_not_awaited()
    assert not state.is_available() and not state.can_reserve()
