# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Unit-check the exact guest CLI environment; live proof uses a real sandbox."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.executor.benchmark import inspect_ghcp_guest


async def test_sdk_child_env_is_run_private_and_does_not_inherit_host_credentials() -> None:
    pinned_path = "/opt/pyrit/guest-local/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    home = Path("/tmp/pyrit-inspect-home-11111111-1111-1111-1111-111111111111")
    token = "scoped-run-token-" + "x" * 32
    config = {
        "run_id": "11111111-1111-1111-1111-111111111111",
        "token_file": "/tmp/pyrit-inspect-token-11111111-1111-1111-1111-111111111111",
        "gateway_url": "http://model-bridge:18181",
        "model_id": "qwen3-local",
        "wire_model": "qwen3:1.7b",
        "cli_path": "/opt/pyrit/copilot",
        "cli_sha256": "a" * 64,
        "max_turns": 2,
        "timeout_seconds": 60,
        "max_model_bytes": 65536,
        "max_prompt_tokens": 8192,
        "max_output_tokens": 1024,
        "allowed_tools": ["bash"],
    }
    identity = {"worker_pid": 123, "cli_pid": 124, "uid": 10001, "net_namespace": "net:[1]"}
    session = MagicMock()
    session.session_id = "real-session-id"
    with (
        patch.dict("os.environ", {"PATH": pinned_path, "GITHUB_TOKEN": "must-not-inherit"}),
        patch.object(inspect_ghcp_guest, "_prepare_guest_home", return_value=(home, home / "copilot-state")),
        patch.object(inspect_ghcp_guest, "read_scoped_token", return_value=token),
        patch.object(inspect_ghcp_guest, "_cli_identity", return_value=identity),
        patch.object(inspect_ghcp_guest, "_write_frame") as write_frame,
        patch.object(inspect_ghcp_guest, "CopilotClient", autospec=True) as client_type,
    ):
        client = client_type.return_value
        client.start = AsyncMock()
        client.create_session = AsyncMock(return_value=session)
        worker = inspect_ghcp_guest._GuestSession(config=config)
        try:
            await worker.start_async()
        finally:
            assert worker._handler is not None
            await worker._handler.close_async()
        environment = client_type.call_args.kwargs["env"]
        assert environment == {
            "PATH": pinned_path,
            "HOME": str(home),
            "COPILOT_HOME": str(home / "copilot-state"),
            "COPILOT_PROVIDER_BASE_URL": "http://model-bridge:18181/v1",
            "COPILOT_MODEL": "qwen3-local",
            "COPILOT_SKIP_CLI_DOWNLOAD": "1",
        }
        assert client_type.call_args.kwargs["use_logged_in_user"] is False
        assert client.create_session.call_args.kwargs["provider"]["bearer_token"] == token
        assert write_frame.call_count == 1
