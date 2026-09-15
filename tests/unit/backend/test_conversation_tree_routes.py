# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import io
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from pyrit.backend.middleware.error_handlers import register_error_handlers
from pyrit.backend.models.conversation_tree import (
    ConversationTreePreviewRequest,
    TreeMessageReference,
    TreePreviewLevel,
)
from pyrit.backend.routes.conversation_tree import router
from pyrit.backend.services.conversation_tree_media import ConversationTreeMedia
from pyrit.backend.services.conversation_tree_service import ConversationTreeService, get_conversation_tree_service
from unit.memory.conversation_tree_test_helpers import (
    make_tree_message,
    store_tree_attack,
    store_tree_messages,
)

if TYPE_CHECKING:
    from pyrit.memory.sqlite_memory import SQLiteMemory
    from pyrit.models import Message

pytestmark = pytest.mark.usefixtures("patch_central_database")


@pytest.fixture
def tree_service(sqlite_instance: SQLiteMemory) -> ConversationTreeService:
    return ConversationTreeService(memory=sqlite_instance)


@pytest.fixture
def tree_client(tree_service: ConversationTreeService) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    register_error_handlers(app)
    app.dependency_overrides[get_conversation_tree_service] = lambda: tree_service
    return TestClient(app)


def store_image_message(*, memory: SQLiteMemory, conversation_id: str, size: tuple[int, int] = (800, 400)) -> Message:
    folder = Path(memory.results_path) / "prompt-memory-entries"
    folder.mkdir(exist_ok=True)
    path = folder / f"{uuid4()}.png"
    Image.new("RGB", size, color="red").save(path)
    message = make_tree_message(conversation_id=conversation_id, values=[("text", "image"), ("image_path", str(path))])
    store_tree_messages(memory=memory, messages=[message])
    return message


def test_tree_route_serializes_exact_frozen_shape(*, sqlite_instance: SQLiteMemory, tree_client: TestClient) -> None:
    main = str(uuid4())
    message = make_tree_message(conversation_id=main)
    store_tree_messages(memory=sqlite_instance, messages=[message])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    response = tree_client.get(f"/api/attacks/{attack.attack_result_id}/conversation-tree")
    assert response.status_code == 200
    page = response.json()
    assert set(page) == {
        "attack_result_id",
        "main_conversation_id",
        "revision",
        "nodes",
        "conversations",
        "processed_conversations",
        "total_conversations",
        "next_cursor",
        "complete",
    }
    assert page["complete"] is True
    assert page["next_cursor"] is None
    assert page["nodes"][0]["message"] == {"conversation_id": main, "sequence": 0}
    assert page["nodes"][0]["created_at"].endswith("Z")
    assert page["conversations"] == [{"conversation_id": main, "node_id": page["nodes"][0]["node_id"]}]
    assert set(page["nodes"][0]) == {
        "node_id",
        "parent_node_id",
        "message",
        "role",
        "piece_types",
        "piece_count",
        "preview_key",
        "created_at",
    }


@pytest.mark.parametrize("limit", ["0", "-1", "101", "1.5", "true"])
def test_route_rejects_invalid_page_limits(*, tree_client: TestClient, limit: str) -> None:
    response = tree_client.get(f"/api/attacks/{uuid4()}/conversation-tree", params={"limit": limit})
    assert response.status_code == 422


def test_route_scope_and_cursor_errors_are_explicit(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient, tree_service: ConversationTreeService
) -> None:
    main, adversarial = str(uuid4()), str(uuid4())
    messages = [make_tree_message(conversation_id=main, sequence=index) for index in range(3)]
    store_tree_messages(memory=sqlite_instance, messages=messages)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, adversarial=[adversarial])
    url = f"/api/attacks/{attack.attack_result_id}/conversation-tree"
    assert tree_client.get(url, params={"prioritize_conversation_id": adversarial}).status_code == 403
    assert tree_client.get(f"/api/attacks/{uuid4()}/conversation-tree").status_code == 404
    assert tree_client.get(url, params={"cursor": "malformed"}).status_code == 409
    cursor = tree_client.get(url, params={"limit": 1}).json()["next_cursor"]
    assert tree_client.get(url, params={"cursor": cursor, "limit": 2}).status_code == 409
    with patch.object(tree_service, "_CACHE_TTL_SECONDS", 0):
        response = tree_client.get(url, params={"cursor": cursor, "limit": 1})
    assert response.status_code == 410
    assert "expired" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"messages": [], "level": "text"},
        {"messages": [{"conversation_id": "conversation", "sequence": 0}] * 65, "level": "text"},
        {"messages": [{"conversation_id": "conversation", "sequence": True}], "level": "text"},
        {"messages": [{"conversation_id": "conversation", "sequence": 1.5}], "level": "text"},
        {"messages": [{"conversation_id": "conversation", "sequence": "1"}], "level": "text"},
        {"messages": [{"conversation_id": "", "sequence": 0}], "level": "text"},
        {"messages": [{"conversation_id": "conversation", "sequence": 0}], "level": "unknown"},
        {"messages": [{"conversation_id": "conversation", "sequence": 0}], "level": "text", "extra": True},
    ],
)
def test_preview_size_and_shape_validation(*, tree_client: TestClient, body: dict[str, object]) -> None:
    response = tree_client.post(f"/api/attacks/{uuid4()}/conversation-tree/previews", json=body)
    assert response.status_code == 422


@pytest.mark.parametrize("level", ["text", "thumbnail", "full"])
def test_preview_and_thumbnail_routes_exclude_foreign_and_adversarial_pieces(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient, level: str
) -> None:
    main, adversarial, foreign = (str(uuid4()) for _ in range(3))
    adversarial_message = store_image_message(memory=sqlite_instance, conversation_id=adversarial)
    foreign_message = store_image_message(memory=sqlite_instance, conversation_id=foreign)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, adversarial=[adversarial])
    store_tree_attack(memory=sqlite_instance, main_conversation_id=foreign)
    for message in (adversarial_message, foreign_message):
        response = tree_client.post(
            f"/api/attacks/{attack.attack_result_id}/conversation-tree/previews",
            json={"messages": [{"conversation_id": message.conversation_id, "sequence": 0}], "level": level},
        )
        assert response.status_code == 403
        thumbnail = tree_client.get(
            f"/api/attacks/{attack.attack_result_id}/conversation-tree/pieces/{message.message_pieces[-1].id}/thumbnail"
        )
        assert thumbnail.status_code == 403


def test_text_phase_never_signs_urls_resolves_paths_or_decodes_assets(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient
) -> None:
    main = str(uuid4())
    store_image_message(memory=sqlite_instance, conversation_id=main)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    with (
        patch("pyrit.backend.mappers.attack_mappers._sign_blob_url_async", new_callable=AsyncMock) as sign,
        patch("pyrit.backend.mappers.attack_mappers._resolve_media_url", side_effect=AssertionError("No media URLs")),
        patch.object(Image, "open", side_effect=AssertionError("No decoding")),
    ):
        response = tree_client.post(
            f"/api/attacks/{attack.attack_result_id}/conversation-tree/previews",
            json={"messages": [{"conversation_id": main, "sequence": 0}] * 2, "level": "text"},
        )
    assert response.status_code == 200
    sign.assert_not_called()
    previews = response.json()["previews"]
    assert len(previews) == 1
    assert len(previews[0]["pieces"]) == 2
    assert set(previews[0]["pieces"][1]) == {
        "piece_id",
        "data_type",
        "text",
        "truncated",
        "media_url",
        "thumbnail_url",
        "mime_type",
        "filename",
        "response_error",
    }
    assert previews[0]["pieces"][1]["media_url"] is None
    assert previews[0]["pieces"][1]["filename"] is None


def test_thumbnail_route_serves_small_real_image_without_signing_or_full_originals(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient
) -> None:
    main = str(uuid4())
    message = store_image_message(memory=sqlite_instance, conversation_id=main)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    with patch("pyrit.backend.mappers.attack_mappers._sign_blob_url_async", new_callable=AsyncMock) as sign:
        response = tree_client.post(
            f"/api/attacks/{attack.attack_result_id}/conversation-tree/previews",
            json={"messages": [{"conversation_id": main, "sequence": 0}], "level": "thumbnail"},
        )
        piece = response.json()["previews"][0]["pieces"][-1]
        assert piece["media_url"] is None
        assert piece["thumbnail_url"] is not None
        thumbnail = tree_client.get(piece["thumbnail_url"])
    sign.assert_not_called()
    assert thumbnail.status_code == 200
    assert thumbnail.headers["content-type"] == "image/png"
    with Image.open(io.BytesIO(thumbnail.content)) as image:
        assert image.size == (256, 128)
    assert thumbnail.content != Path(message.message_pieces[-1].converted_value).read_bytes()
    assert len(thumbnail.content) < 256 * 256 * 4


def test_full_preview_uses_existing_safe_media_mapping_only_on_request(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient
) -> None:
    main = str(uuid4())
    store_image_message(memory=sqlite_instance, conversation_id=main)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    with patch.object(Image, "open", side_effect=AssertionError("Full preview must not decode media")):
        response = tree_client.post(
            f"/api/attacks/{attack.attack_result_id}/conversation-tree/previews",
            json={"messages": [{"conversation_id": main, "sequence": 0}], "level": "full"},
        )
    assert response.status_code == 200
    piece = response.json()["previews"][0]["pieces"][-1]
    assert piece["media_url"].startswith("/api/media?path=")
    assert piece["mime_type"] == "image/png"
    assert piece["filename"].startswith("image_")


def test_remote_media_gets_placeholder_thumbnail_and_only_full_phase_signs(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient
) -> None:
    main = str(uuid4())
    message = make_tree_message(
        conversation_id=main,
        values=[
            ("image_path", "https://test.blob.core.windows.net/container/image.png"),
            ("video_path", "https://example.com/video.mp4"),
            ("audio_path", "https://example.com/audio.wav"),
        ],
    )
    store_tree_messages(memory=sqlite_instance, messages=[message])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    request = {"messages": [{"conversation_id": main, "sequence": 0}], "level": "thumbnail"}
    url = f"/api/attacks/{attack.attack_result_id}/conversation-tree/previews"
    with (
        patch("pyrit.backend.mappers.attack_mappers._sign_blob_url_async", new_callable=AsyncMock) as sign,
        patch.object(Image, "open", side_effect=AssertionError("Never fetch/decode remote media")),
    ):
        response = tree_client.post(url, json=request)
        sign.assert_not_called()
        assert all(
            piece["thumbnail_url"] is None and piece["media_url"] is None
            for piece in response.json()["previews"][0]["pieces"]
        )
        sign.side_effect = lambda *, blob_url: blob_url + "?signed"
        request["level"] = "full"
        full = tree_client.post(url, json=request)
    assert full.status_code == 200
    assert sign.call_count == 3
    assert all(piece["media_url"].endswith("?signed") for piece in full.json()["previews"][0]["pieces"])
    remote_thumbnail = tree_client.get(
        f"/api/attacks/{attack.attack_result_id}/conversation-tree/pieces/{message.message_pieces[0].id}/thumbnail"
    )
    assert remote_thumbnail.status_code == 404


@pytest.mark.parametrize("unsafe_value", ["outside.png", r"..\outside.png", r"\\example.test\share\image.png"])
def test_tree_media_reuses_storage_path_safety(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient, unsafe_value: str
) -> None:
    main = str(uuid4())
    message = make_tree_message(conversation_id=main, values=[("image_path", unsafe_value)])
    store_tree_messages(memory=sqlite_instance, messages=[message])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    response = tree_client.post(
        f"/api/attacks/{attack.attack_result_id}/conversation-tree/previews",
        json={"messages": [{"conversation_id": main, "sequence": 0}], "level": "full"},
    )
    assert response.status_code == 403
    thumbnail = tree_client.get(
        f"/api/attacks/{attack.attack_result_id}/conversation-tree/pieces/{message.message_pieces[0].id}/thumbnail"
    )
    assert thumbnail.status_code == 403


def test_thumbnail_decode_and_file_size_budgets(*, sqlite_instance: SQLiteMemory, tree_client: TestClient) -> None:
    main = str(uuid4())
    message = store_image_message(memory=sqlite_instance, conversation_id=main)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    url = f"/api/attacks/{attack.attack_result_id}/conversation-tree/pieces/{message.message_pieces[-1].id}/thumbnail"
    with patch.object(ConversationTreeMedia, "MAX_SOURCE_BYTES", 32):
        assert tree_client.get(url).status_code == 404
    with patch.object(ConversationTreeMedia, "MAX_DECODE_PIXELS", 10):
        assert tree_client.get(url).status_code == 404
    Path(message.message_pieces[-1].converted_value).write_bytes(b"invalid image")
    assert tree_client.get(url).status_code == 404


def test_thumbnail_preserves_image_orientation_and_restricts_decoders(
    *, sqlite_instance: SQLiteMemory, tree_client: TestClient
) -> None:
    main = str(uuid4())
    message = store_image_message(memory=sqlite_instance, conversation_id=main)
    path = Path(message.message_pieces[-1].converted_value)
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (800, 400), color="red").save(path, exif=exif)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    url = f"/api/attacks/{attack.attack_result_id}/conversation-tree/pieces/{message.message_pieces[-1].id}/thumbnail"
    with patch.object(Image, "open", wraps=Image.open) as image_open:
        response = tree_client.get(url)
    assert response.status_code == 200
    assert image_open.call_args.kwargs["formats"] == ConversationTreeMedia._IMAGE_FORMATS
    assert "EPS" not in ConversationTreeMedia._IMAGE_FORMATS
    with Image.open(io.BytesIO(response.content)) as thumbnail:
        assert thumbnail.size == (128, 256)


async def test_held_media_work_does_not_block_structure_or_text(
    *, sqlite_instance: SQLiteMemory, tree_service: ConversationTreeService
) -> None:
    main = str(uuid4())
    message = store_image_message(memory=sqlite_instance, conversation_id=main)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    running = 0
    event_loop_thread = threading.get_ident()
    worker_threads: set[int] = set()

    def hold_media(_piece: object) -> bytes:
        nonlocal running
        with lock:
            running += 1
            worker_threads.add(threading.get_ident())
            if running == 2:
                entered.set()
        assert release.wait(timeout=10)
        return b"thumbnail"

    with patch.object(ConversationTreeMedia, "render_thumbnail", side_effect=hold_media):
        tasks = [
            asyncio.create_task(
                tree_service.get_thumbnail_async(
                    attack_result_id=attack.attack_result_id, piece_id=message.message_pieces[-1].id
                )
            )
            for _ in range(2)
        ]
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            page = await asyncio.wait_for(
                tree_service.get_page_async(attack_result_id=attack.attack_result_id), timeout=5
            )
            previews = await asyncio.wait_for(
                tree_service.get_previews_async(
                    attack_result_id=attack.attack_result_id,
                    request=ConversationTreePreviewRequest(
                        messages=[TreeMessageReference(conversation_id=main, sequence=0)], level=TreePreviewLevel.TEXT
                    ),
                ),
                timeout=5,
            )
            assert page.complete
            assert previews.previews[0].pieces[0].text == "image"
            assert not any(task.done() for task in tasks)
            assert event_loop_thread not in worker_threads
        finally:
            release.set()
            await asyncio.gather(*tasks)


async def test_cancelled_http_waiters_keep_media_capacity_until_workers_finish(
    *, sqlite_instance: SQLiteMemory, tree_service: ConversationTreeService
) -> None:
    main = str(uuid4())
    message = store_image_message(memory=sqlite_instance, conversation_id=main)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    count = 0

    def hold_media(_piece: object) -> bytes:
        nonlocal count
        with lock:
            count += 1
            if count == 2:
                entered.set()
        assert release.wait(timeout=10)
        return b"thumbnail"

    async def request_thumbnail_async() -> bytes:
        return await tree_service.get_thumbnail_async(
            attack_result_id=attack.attack_result_id, piece_id=message.message_pieces[-1].id
        )

    with patch.object(ConversationTreeMedia, "render_thumbnail", side_effect=hold_media):
        requests = [asyncio.create_task(request_thumbnail_async()) for _ in range(2)]
        third: asyncio.Task[bytes] | None = None
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            for request in requests:
                request.cancel()
            outcomes = await asyncio.gather(*requests, return_exceptions=True)
            assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes)
            attempted = asyncio.Event()

            async def third_request_async() -> bytes:
                attempted.set()
                return await request_thumbnail_async()

            third = asyncio.create_task(third_request_async())
            await attempted.wait()
            assert len(tree_service._active_reads) == 2
            assert count == 2
            assert not third.done()
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)
            if third is not None:
                await third
            await asyncio.gather(*tuple(tree_service._active_reads))
    assert not tree_service._active_reads
