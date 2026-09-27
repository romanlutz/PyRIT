# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import json
import tomllib
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import httpx
import pytest
from pydantic import SecretStr

from pyrit.executor.workflow.docker_agent import DockerSandboxLauncher, DockerStopOnlyAgentLease
from pyrit.executor.workflow.docker_engine import DockerEngineError
from pyrit.executor.workflow.docker_guest_auth import DockerGuestAuth, codex_gateway_config
from pyrit.models import Message
from pyrit.models.native_cyber_evidence import NativeCyberEvidenceSource
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import NativeCliTarget
from pyrit.prompt_target.gateway.claude_messages import create_claude_messages_app
from pyrit.prompt_target.gateway.codex_responses import create_codex_responses_app
from pyrit.prompt_target.gateway.httpx_messages_backend import HttpxMessagesBackend
from pyrit.prompt_target.gateway.httpx_responses_backend import HttpxResponsesBackend
from pyrit.prompt_target.gateway.messages_contract import MessagesCapabilities
from pyrit.prompt_target.gateway.responses_contract import BackendCapabilities, GatewayLimits, GatewayRoute
from pyrit.prompt_target.native_cli_models import NativeCliProtocol
from tests.unit.executor.workflow.test_docker_agent import config, make_agent
from tests.unit.executor.workflow.test_docker_engine import CONTAINER_ID, FakeEngine, wire_frame
from tests.unit.executor.workflow.test_native_cli_evidence import _start_sink_async
from tests.unit.prompt_target.gateway.messages_mocks import FakeMessagesBackend, request_body
from tests.unit.prompt_target.gateway.test_codex_responses import FakeModelOnlyBackend
from tests.unit.prompt_target.target.test_native_cli_target import _claude, _codex

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def guest_route(*, protocol: NativeCliProtocol, run_id: str = "guest-run") -> GatewayRoute:
    return GatewayRoute(
        run_id=run_id,
        model="codex-fixture" if protocol is NativeCliProtocol.CODEX_EXEC_JSON else "claude-offline-model",
        guest_token="inert-guest-only-" + "a" * 40,
    )


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
async def test_exec_receives_only_protocol_guest_token_and_run_env_without_retaining_it_async(
    protocol: NativeCliProtocol,
) -> None:
    route = guest_route(protocol=protocol)
    auth = DockerGuestAuth.from_route(route=route, protocol=protocol)
    fake = FakeEngine()
    engine = fake.client()
    handle = await engine.create_exec_async(
        container_id=CONTAINER_ID,
        argv=("/opt/pinned-cli", "--", "fixture"),
        user="1000:1000",
        working_directory="/tmp/work",
        guest_auth=auth,
    )
    expected = (
        [f"PYRIT_GUEST_MODEL_TOKEN={route.guest_token}", f"PYRIT_RUN_ID={route.run_id}"]
        if protocol is NativeCliProtocol.CODEX_EXEC_JSON
        else [f"ANTHROPIC_AUTH_TOKEN={route.guest_token}", f"ANTHROPIC_CUSTOM_HEADERS=X-PyRIT-Run-ID: {route.run_id}"]
    )
    assert fake.created["Env"] == expected
    assert route.guest_token not in repr(auth) and route.guest_token not in repr(asdict(auth))
    assert route.guest_token not in repr(handle) and route.guest_token not in repr(engine)
    assert not any(route.guest_token in arg for arg in fake.created["Cmd"])
    inspection = await engine.inspect_exec_async(handle)
    assert "Env" not in inspection and "Env" not in inspection["ProcessConfig"]
    assert not hasattr(handle, "guest_auth") and not hasattr(handle, "environment")
    await engine.close_async()


@pytest.mark.parametrize("token", ["", "short", "a" * 32 + "\r\nInjected: header", "a" * 257])
def test_invalid_guest_token_fails_without_including_it_in_errors(token: str) -> None:
    with pytest.raises(ValueError) as error:
        DockerGuestAuth(run_id="run", model="model", protocol=NativeCliProtocol.CODEX_EXEC_JSON, token=SecretStr(token))
    if token:
        assert token not in str(error.value)


@pytest.mark.parametrize("run_id", ["", "run\r\nOther: header", "run with spaces", "x" * 129])
def test_guest_run_header_cannot_be_missing_or_injected(run_id: str) -> None:
    with pytest.raises(ValueError):
        DockerGuestAuth(
            run_id=run_id,
            model="model",
            protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE,
            token=SecretStr("a" * 40),
        )


def test_codex_user_config_uses_only_dedicated_guest_env_auth() -> None:
    document = codex_gateway_config(model="codex-fixture", base_url="http://gateway.sandbox/v1")
    parsed = tomllib.loads(document)
    assert parsed["model_provider"] == "pyrit_gateway" and parsed["model"] == "codex-fixture"
    assert parsed["model_providers"]["pyrit_gateway"] == {
        "name": "PyRIT run gateway",
        "base_url": "http://gateway.sandbox/v1",
        "wire_api": "responses",
        "env_key": "PYRIT_GUEST_MODEL_TOKEN",
        "requires_openai_auth": False,
        "env_http_headers": {"X-PyRIT-Run-ID": "PYRIT_RUN_ID"},
    }
    assert "OPENAI_API_KEY" not in document and "Authorization" not in document


@pytest.mark.parametrize("mismatch", ["run", "protocol", "missing"])
async def test_guest_credentials_cannot_be_attached_to_another_run_or_profile_async(mismatch: str) -> None:
    lease, command, fake, engine, _ = make_agent()
    auth = lease._guest_auth
    if mismatch == "run":
        auth = replace(auth, run_id="another-run")
    elif mismatch == "protocol":
        auth = replace(auth, protocol=NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE)
    else:
        auth = None
    with pytest.raises(ValueError, match="exact run"):
        DockerStopOnlyAgentLease(
            run_id=lease.run_id,
            spec=lease._spec,
            runner=command,
            engine=engine,
            profile=lease._profile,
            guest_auth=auth,
        )
    assert not fake.requests and not command.calls
    await engine.close_async()


async def test_guest_token_is_not_permitted_in_prompt_argv_async() -> None:
    lease, command, fake, engine, approved = make_agent()
    with pytest.raises(ValueError, match="prompt/argv") as error:
        await DockerSandboxLauncher(lease=lease).launch_async(
            config=approved, prompt=lease._guest_auth.token.get_secret_value()
        )
    assert lease._guest_auth.token.get_secret_value() not in str(error.value)
    assert not fake.requests and not command.calls
    await engine.close_async()


@pytest.mark.parametrize("missing", ["profile-pin", "image-pin", "container-pin", "path", "model"])
async def test_unqualified_codex_user_config_blocks_before_exec_async(missing: str) -> None:
    lease, command, fake, engine, approved = make_agent()
    if missing == "profile-pin":
        with pytest.raises(ValueError, match="user-level"):
            replace(lease._profile, codex_config_sha256=None)
        await engine.close_async()
        return
    if missing == "model":
        lease._guest_auth = replace(lease._guest_auth, model="wrong-model")
    else:
        populate = command.mutate_up
        image_change = command.mutate_image

        def mutate_image(image: dict[str, Any]) -> None:
            assert image_change is not None
            image_change(image)
            if missing == "image-pin":
                image["Config"]["Labels"].pop("org.pyrit.native.codex-user-config-sha256")

        def mutate_container(runner: Any) -> None:
            assert populate is not None
            populate(runner)
            for container in runner.containers.values():
                if missing == "container-pin":
                    container["Config"]["Labels"].pop("org.pyrit.native.codex-user-config-sha256")
                if missing == "path":
                    container["Config"]["Labels"]["org.pyrit.native.codex-user-config-path"] = (
                        "/workspace/.codex/config.toml"
                    )

        command.mutate_image, command.mutate_up = mutate_image, mutate_container
    with pytest.raises(DockerEngineError, match="Codex"):
        await lease.acquire_async()
    assert fake.created is None and not command.containers
    await engine.close_async()


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
@pytest.mark.parametrize(
    "failure", [None, "missing-token", "foreign-token", "missing-run", "foreign-run", "no-request"]
)
async def test_engine_guest_env_authenticates_actual_gateway_and_db_capture_stays_controller_owned_async(
    *,
    sqlite_instance: SQLiteMemory,
    protocol: NativeCliProtocol,
    failure: str | None,
) -> None:
    sink = await _start_sink_async(memory=sqlite_instance, protocol=protocol, include_model_gateway=True)
    route = guest_route(protocol=protocol, run_id=sink.run_id)
    approved = config(timeout=10, protocol=protocol)
    if protocol is NativeCliProtocol.CLAUDE_PRINT_STREAM_JSON_VERBOSE:
        approved = replace(approved, model_gateway_endpoint="http://gateway.sandbox")
    lease, command, fake, engine, _ = make_agent(run_config=approved, route=route)
    limits = GatewayLimits(max_output_tokens_per_request=64, max_requests=1)
    if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
        backend = FakeModelOnlyBackend()
        app = create_codex_responses_app(
            route=route, limits=limits, backend=backend, observation_callback=sink.record_gateway_observation_async
        )
        model_body = {"model": route.model, "input": "inert model input", "store": False}
        model_path = "/v1/responses"
        fake.wire = (wire_frame(1, _codex()),)
    else:
        backend = FakeMessagesBackend()
        app = create_claude_messages_app(
            route=route, limits=limits, backend=backend, observation_callback=sink.record_messages_observation_async
        )
        model_body = request_body()
        model_path = "/v1/messages?beta=true"
        fake.wire = (wire_frame(1, _claude(assistant_text="inert answer", result_text="inert result")),)
    responses: list[int] = []

    async def request_from_env_async() -> None:
        if failure == "no-request":
            return
        env = dict(item.split("=", 1) for item in fake.created["Env"])
        headers = {"Content-Type": "application/json"}
        if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
            staged = tomllib.loads(codex_gateway_config(model=route.model, base_url=approved.model_gateway_endpoint))
            provider = staged["model_providers"][staged["model_provider"]]
            headers["Authorization"] = "Bearer " + env[provider["env_key"]]
            headers.update({name: env[key] for name, key in provider["env_http_headers"].items()})
        else:
            headers["Authorization"] = "Bearer " + env["ANTHROPIC_AUTH_TOKEN"]
            name, value = env["ANTHROPIC_CUSTOM_HEADERS"].split(": ", 1)
            headers[name] = value
            headers["anthropic-version"] = "2023-06-01"
        if failure == "missing-token":
            headers.pop("Authorization")
        elif failure == "foreign-token":
            headers["Authorization"] = "Bearer " + "b" * 40
        elif failure == "missing-run":
            headers.pop("X-PyRIT-Run-ID")
        elif failure == "foreign-run":
            headers["X-PyRIT-Run-ID"] = "another-run"
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway.sandbox"
        ) as guest_client:
            reply = await guest_client.post(model_path, json=model_body, headers=headers)
            responses.append(reply.status_code)
            if failure is None:
                blocked = await guest_client.post(model_path, json=model_body, headers=headers)
                assert blocked.status_code == 429

    fake.on_start = request_from_env_async
    await lease.acquire_async()
    target = NativeCliTarget(run_config=approved, launcher=DockerSandboxLauncher(lease=lease), evidence_sink=sink)
    request = Message.from_prompt(prompt="inert task", role="user")
    response = await PromptNormalizer().send_prompt_async(message=request, target=target)
    assert target.last_run is not None
    outcome = target.last_run.outcome
    assert outcome.exit_code == 0 and outcome.coverage_complete
    assert lease.agent_stop.stopped
    await sink.finish_async(
        outcome=outcome, request_piece_ids=(request.get_piece().id,), response_piece_ids=(response.get_piece().id,)
    )
    episode = await asyncio.to_thread(sqlite_instance.native_cyber_evidence.get_episode, run_id=sink.run_id)
    model_requests = [
        event
        for event in episode.events
        if event.source is NativeCyberEvidenceSource.MODEL and event.event_type.endswith(".request")
    ]
    if failure is None:
        assert responses == [200] and len(backend.requests) == len(model_requests) == 1
        assert backend.requests[0].run_id == sink.run_id
        assert episode.turns[0].source_complete
        streams = [stream for stream in episode.raw_streams if stream.key.source is NativeCyberEvidenceSource.MODEL]
        assert streams and all(stream.source_complete and stream.stored_bytes > 0 for stream in streams)
        for stream in streams:
            chunks = await asyncio.to_thread(
                sqlite_instance.native_cyber_evidence.read_raw_chunks,
                run_id=sink.run_id,
                stream_id=stream.stream_id,
                allow_sensitive=True,
            )
            assert route.guest_token.encode() not in b"".join(chunk.data for chunk in chunks)
    else:
        expected = [] if failure == "no-request" else [401 if failure.endswith("token") else 403]
        assert responses == expected and not model_requests and not backend.requests
        assert not episode.turns[0].source_complete
        assert any("Model gateway" in gap for gap in episode.turns[0].gaps)
        assert not episode.coverage_complete and episode.score_status.value == "undetermined"
    assert route.guest_token not in json.dumps(command.documents)
    assert route.guest_token not in repr(lease._profile) and route.guest_token not in repr(approved)
    await lease.close_async()
    await engine.close_async()


@pytest.mark.parametrize("protocol", list(NativeCliProtocol))
async def test_primary_credential_inequality_stays_at_host_backend_boundary_async(protocol: NativeCliProtocol) -> None:
    route = guest_route(protocol=protocol)
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(500))) as client:
        with pytest.raises(ValueError, match="distinct"):
            if protocol is NativeCliProtocol.CODEX_EXEC_JSON:
                HttpxResponsesBackend(
                    route=route,
                    endpoint="https://model.invalid/v1/responses",
                    auth_token=route.guest_token,
                    client=client,
                    limits=GatewayLimits(),
                    capabilities=BackendCapabilities(),
                )
            else:
                HttpxMessagesBackend(
                    route=route,
                    endpoint="https://model.invalid/v1/messages",
                    host_api_key=route.guest_token,
                    client=client,
                    limits=GatewayLimits(),
                    capabilities=MessagesCapabilities(),
                )
    assert "host" not in " ".join(DockerGuestAuth.__dataclass_fields__)
