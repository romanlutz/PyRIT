# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Signed policy identities across independent interpreters and unchanged semantic boundaries."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import uuid4

import pytest
from pydantic import JsonValue, TypeAdapter

from pyrit.backend.models.original_worker import CohostBackendConfig, CohostValidationScope
from pyrit.backend.services.original_worker_preflight import CohostPreflight, CohostPreflightError
from pyrit.backend.services.original_worker_state import CohostValidationState, bootstrap_validation_state
from pyrit.models import config_hash

_POLICY_PROBE = """
import asyncio
import base64
import hashlib
import json
import sys
from collections.abc import AsyncIterator
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from azure.identity.aio import DefaultAzureCredential
from azure.storage.blob.aio import BlobClient, BlobLeaseClient, StorageStreamDownloader
from pydantic import JsonValue, TypeAdapter
from pyrit.backend.models.original_worker import CohostBackendConfig
from pyrit.backend.services.original_worker_preflight import CohostPreflight, CohostPreflightError
from pyrit.backend.services.original_worker_state import CohostValidationState, bootstrap_validation_state
from pyrit.models import config_hash

async def restore_async(*, config: CohostBackendConfig, content: bytes) -> dict[str, object]:
    async def chunks_async() -> AsyncIterator[bytes]:
        yield content

    service = CohostValidationState(preflight=CohostPreflight(config=config, environment={}))
    credential = MagicMock(spec=DefaultAzureCredential)
    credential.close = AsyncMock()
    credential.get_token = AsyncMock(side_effect=AssertionError("No credential calls are admitted."))
    stream = MagicMock(spec=StorageStreamDownloader)
    stream.properties = SimpleNamespace(etag="public-hashseed-etag")
    stream.chunks = chunks_async
    blob = MagicMock(spec=BlobClient)
    blob.download_blob = AsyncMock(return_value=stream)
    blob.close = AsyncMock()
    lease = MagicMock(spec=BlobLeaseClient)
    lease.acquire = AsyncMock()
    lease.release = AsyncMock()
    with (
        patch("azure.identity.aio.DefaultAzureCredential", return_value=credential),
        patch.object(BlobClient, "from_blob_url", return_value=blob),
        patch("azure.storage.blob.aio.BlobLeaseClient", return_value=lease),
    ):
        try:
            await service.startup_async()
            result = {
                "accepted": True,
                "can_reserve": service.can_reserve(),
                "budget": service.budget.model_dump(mode="json"),
                "jobs": service._value["jobs"],
            }
        except CohostPreflightError as error:
            result = {"accepted": False, "error": str(error), "cause": str(error.__cause__)}
        finally:
            await service.shutdown_async()
        credential.get_token.assert_not_called()
        return result

value = json.loads(sys.stdin.buffer.read())
config = CohostBackendConfig.model_validate_json(value["config"], strict=True)
if value["action"] == "bootstrap":
    with patch("pyrit.backend.services.original_worker_state.secrets.token_bytes", return_value=b"B" * 32):
        config, content = bootstrap_validation_state(
            config=config, created_at=datetime.fromisoformat(value["created_at"])
        )
    result = {"config": config.model_dump_json(), "state_b64": base64.b64encode(content).decode()}
elif value["action"] == "restore":
    result = asyncio.run(restore_async(config=config, content=base64.b64decode(value["state_b64"], validate=True)))
else:
    raise AssertionError("Unsupported public policy probe.")
result.update(
    policy_sha256=CohostValidationState.policy_sha256(config),
    retained_policy_sha256=config_hash({
        "source": config.source.model_dump(mode="json"),
        "profile_ref": config.profile_ref,
        "database": config.expected_database_name,
        "result_container": config.result_container_url,
    }),
    source=config.source.model_dump(mode="json"),
)
print(json.dumps(result, sort_keys=True))
"""


@pytest.fixture
def policy_config(worker_config: CohostBackendConfig) -> CohostBackendConfig:
    """Use multiple values in every unordered field, never private configuration or authority."""
    config = worker_config.model_copy(
        update={
            "source": worker_config.source.model_copy(
                update={"display_values": frozenset({"0.0", "0.5", "1.0", "UND"})}
            ),
            "allowed_operator_oids": frozenset({"public-operator-a", "public-operator-b", "public-operator-c"}),
            "allowed_group_ids": frozenset({"public-group-a", "public-group-b", "public-group-c"}),
            "relay": worker_config.relay.model_copy(
                update={"max_completion_tokens": 8192, "request_timeout_seconds": 60}
            ),
            "validation_scope": CohostValidationScope(
                instance_id=uuid4(),
                expires_at=datetime.now(UTC) + timedelta(hours=1),
                state_container_url="https://publicfixture.blob.core.windows.net/validation-state",
            ),
        }
    )
    return CohostBackendConfig.model_validate_json(config.model_dump_json(), strict=True)


async def _policy_probe_async(*, seed: int, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    environment = {key: item for key, item in os.environ.items() if key.upper() in {"PATH", "SYSTEMROOT", "WINDIR"}}
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _POLICY_PROBE,
        env={**environment, "PYTHONHASHSEED": str(seed), "PYTHONUTF8": "1"},
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, errors = await asyncio.wait_for(
            process.communicate(TypeAdapter(dict[str, JsonValue]).dump_json(value)), timeout=45
        )
        assert process.returncode == 0, errors.decode(errors="replace")
        return TypeAdapter(dict[str, JsonValue]).validate_json(output, strict=True)
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


def _bootstrap(config: CohostBackendConfig) -> tuple[CohostBackendConfig, bytes]:
    with patch("pyrit.backend.services.original_worker_state.secrets.token_bytes", return_value=b"B" * 32):
        return bootstrap_validation_state(config=config)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4, 5, 42, 123])
async def test_signed_bootstrap_restores_across_fresh_hashseed_interpreters_async(
    *, policy_config: CohostBackendConfig, seed: int
) -> None:
    bound, content = _bootstrap(policy_config)
    state = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
    produced = await _policy_probe_async(
        seed=seed,
        value={
            "action": "bootstrap",
            "config": policy_config.model_dump_json(),
            "created_at": state["created_at"],
        },
    )
    restored = await _policy_probe_async(
        seed=(seed + 1) % 124,
        value={"action": "restore", "config": produced["config"], "state_b64": produced["state_b64"]},
    )
    assert restored["accepted"], {"producer_seed": seed, "restorer_seed": (seed + 1) % 124, "restore": restored}
    assert restored["can_reserve"] is True
    assert produced["policy_sha256"] == restored["policy_sha256"] == state["policy_sha256"]
    assert produced["config"] == bound.model_dump_json()
    assert produced["state_b64"] == base64.b64encode(content).decode()
    assert produced["retained_policy_sha256"] == restored["retained_policy_sha256"]
    assert restored["source"] == bound.source.model_dump(mode="json")
    assert restored["budget"] == {"requests": 0, "observed_tokens": 0, "unresolved": False}


async def test_one_signed_packet_restores_in_all_eight_hashseed_readers_and_rejects_foreign_policy_async(
    *, policy_config: CohostBackendConfig, request: pytest.FixtureRequest
) -> None:
    produced = await _policy_probe_async(
        seed=123,
        value={
            "action": "bootstrap",
            "config": policy_config.model_dump_json(),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )
    seeds = [0, 1, 2, 3, 4, 5, 42, 123]
    results = [
        await _policy_probe_async(
            seed=seed,
            value={"action": "restore", "config": produced["config"], "state_b64": produced["state_b64"]},
        )
        for seed in seeds
    ]
    assert all(item["accepted"] is True and item["can_reserve"] is True for item in results), results
    assert {item["policy_sha256"] for item in results} == {produced["policy_sha256"]}
    assert {item["retained_policy_sha256"] for item in results} == {produced["retained_policy_sha256"]}
    assert isinstance(produced["config"], str) and isinstance(produced["state_b64"], str)
    bound = CohostBackendConfig.model_validate_json(produced["config"], strict=True)
    foreign = bound.model_copy(update={"source": bound.source.model_copy(update={"primary_scorer": "foreign_scorer"})})
    rejected = await _policy_probe_async(
        seed=42,
        value={"action": "restore", "config": foreign.model_dump_json(), "state_b64": produced["state_b64"]},
    )
    assert rejected["accepted"] is False
    assert rejected["cause"] == "Validation identity/lifetime/counter state is inconsistent."
    request.node.user_properties.append(
        (
            "public_hashseed_receipt",
            json.dumps(
                {
                    "schema": "public-synthetic-policy-cross-process/v1",
                    "producer_seed": 123,
                    "reader_seeds": seeds,
                    "accepted_fresh_readers": len(results),
                    "distinct_policy_fingerprints": 1,
                    "distinct_retained_policy_fingerprints": 1,
                    "policy_sha256": produced["policy_sha256"],
                    "retained_policy_sha256": produced["retained_policy_sha256"],
                    "same_config_sha256": hashlib.sha256(produced["config"].encode()).hexdigest(),
                    "same_packet_sha256": hashlib.sha256(
                        base64.b64decode(produced["state_b64"], validate=True)
                    ).hexdigest(),
                    "foreign_scorer_state_rejected_on_startup": True,
                    "only_fake_authority_key_used": True,
                    "credentials_network_original_or_model_calls": False,
                },
                sort_keys=True,
            ),
        )
    )


def test_unordered_json_serialization_preserves_typed_sets_and_source_case_order(
    policy_config: CohostBackendConfig,
) -> None:
    value = policy_config.model_dump(mode="json")
    assert value["source"]["display_values"] == sorted(policy_config.source.display_values)
    assert value["allowed_operator_oids"] == sorted(policy_config.allowed_operator_oids)
    assert value["allowed_group_ids"] == sorted(policy_config.allowed_group_ids)
    assert value["source"]["cases"] == [case.model_dump(mode="json") for case in policy_config.source.cases]
    typed = policy_config.model_dump(mode="python")
    assert isinstance(typed["source"]["display_values"], frozenset)
    assert isinstance(typed["allowed_operator_oids"], frozenset)
    assert isinstance(typed["allowed_group_ids"], frozenset)
    assert isinstance(typed["source"]["cases"], tuple)
    assert CohostBackendConfig.model_validate_json(policy_config.model_dump_json(), strict=True) == policy_config


@pytest.mark.parametrize(
    "path,replacement",
    [
        (("source", "display_values"), ["0.0", "0.5", "1.0"]),
        (("source", "primary_scorer"), "foreign_scorer"),
        (("source", "contract_sha256"), "b" * 64),
        (("source", "spec", "harness", "config_sha256"), "c" * 64),
        (("source", "spec", "model_route", "config_sha256"), "d" * 64),
        (("source", "cases", 0, "task_version"), "foreign-version"),
        (("allowed_operator_oids",), ["foreign-operator"]),
        (("allowed_group_ids",), ["foreign-group"]),
        (("managed_identity_client_id",), "00000000-0000-4000-8000-000000000009"),
        (("expected_database_name",), "foreign-database"),
        (("result_container_url",), "https://publicfixture.blob.core.windows.net/foreign-results"),
        (("profile_ref",), "foreign-profile"),
        (("source_alias",), "foreign-source"),
        (("validation_scope", "instance_id"), "00000000-0000-4000-8000-000000000010"),
        (("validation_scope", "state_container_url"), "https://publicfixture.blob.core.windows.net/foreign-state"),
        (("relay", "request_timeout_seconds"), 61),
        (("relay", "max_requests"), 49),
        (("relay", "max_completion_tokens"), 4096),
    ],
)
def test_authentic_signed_state_rejects_changed_semantic_policy(
    *, policy_config: CohostBackendConfig, path: tuple[str | int, ...], replacement: JsonValue
) -> None:
    bound, content = _bootstrap(policy_config)
    changed = bound.model_dump(mode="json")
    target = changed
    for field in path[:-1]:
        target = target[field]
    target[path[-1]] = replacement
    foreign = CohostBackendConfig.model_validate_json(json.dumps(changed), strict=True)
    assert CohostValidationState.policy_sha256(foreign) != CohostValidationState.policy_sha256(bound)
    service = CohostValidationState(preflight=CohostPreflight(config=foreign, environment={}))
    with pytest.raises(CohostPreflightError, match="identity/lifetime/counter state is inconsistent"):
        service._validate_value(TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True))
    assert not service.is_available() and not service.can_reserve()


async def test_signed_ordered_job_history_remains_ordered_on_fresh_restore_async(
    policy_config: CohostBackendConfig,
) -> None:
    bound, content = _bootstrap(policy_config)
    state = TypeAdapter(dict[str, JsonValue]).validate_json(content, strict=True)
    jobs: list[JsonValue] = [
        {
            "reservation_id": str(uuid4()),
            "operator_oid": "public-operator-b",
            "profile_ref": bound.profile_ref,
            "phase": "closed",
        },
        {
            "reservation_id": str(uuid4()),
            "operator_oid": "public-operator-a",
            "profile_ref": bound.profile_ref,
            "phase": "closed",
        },
    ]
    assert CohostValidationState._content({"jobs": jobs}) != CohostValidationState._content({"jobs": jobs[::-1]})
    assert config_hash({"ordered_steps": ["setup", "solver", "scorer", "cleanup"]}) != config_hash(
        {"ordered_steps": ["cleanup", "scorer", "solver", "setup"]}
    )
    state["jobs"] = jobs
    state.pop("signature")
    state["signature"] = hmac.new(b"B" * 32, CohostValidationState._content(state), hashlib.sha256).hexdigest()
    restored = await _policy_probe_async(
        seed=42,
        value={
            "action": "restore",
            "config": bound.model_dump_json(),
            "state_b64": base64.b64encode(CohostValidationState._content(state)).decode(),
        },
    )
    assert restored["accepted"] is True and restored["jobs"] == jobs
    assert restored["can_reserve"] is False
