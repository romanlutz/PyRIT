# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import asyncio
import hashlib
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import UUID, uuid4

import pytest
from sqlalchemy import update

from pyrit.backend.models.conversation_tree import (
    ConversationTreePage,
    ConversationTreePreviewRequest,
    TreeMessageReference,
    TreePreviewLevel,
)
from pyrit.backend.services.conversation_tree_media import ConversationTreeMedia
from pyrit.backend.services.conversation_tree_service import (
    ConversationTreeService,
    TreeCursorExpiredError,
    TreeCursorMismatchError,
)
from pyrit.memory.conversation_tree import TreeSnapshotChangedError
from pyrit.memory.memory_models import AttackResultEntry
from unit.memory.conversation_tree_test_helpers import (
    capture_tree_reads,
    clone_tree_history,
    make_tree_message,
    store_tree_attack,
    store_tree_messages,
)

if TYPE_CHECKING:
    from pyrit.backend.models.conversation_tree import ConversationTreeNode
    from pyrit.memory.sqlite_memory import SQLiteMemory
    from pyrit.models import Message

pytestmark = pytest.mark.usefixtures("patch_central_database")


async def collect_tree_async(
    *, service: ConversationTreeService, first: ConversationTreePage, limit: int
) -> tuple[list[ConversationTreeNode], dict[str, str | None]]:
    nodes = list(first.nodes)
    endpoints = {endpoint.conversation_id: endpoint.node_id for endpoint in first.conversations}
    page = first
    seen = {node.node_id for node in nodes}
    while page.next_cursor is not None:
        prior_count = page.processed_conversations
        page = await service.get_page_async(
            attack_result_id=first.attack_result_id, limit=limit, cursor=page.next_cursor
        )
        assert page.revision == first.revision
        assert page.main_conversation_id == first.main_conversation_id
        assert prior_count <= page.processed_conversations <= page.total_conversations
        assert not seen.intersection(node.node_id for node in page.nodes)
        for node in page.nodes:
            assert node.parent_node_id is None or node.parent_node_id in seen
            seen.add(node.node_id)
            nodes.append(node)
        for endpoint in page.conversations:
            assert endpoint.conversation_id not in endpoints
            assert endpoint.node_id is None or endpoint.node_id in seen
            endpoints[endpoint.conversation_id] = endpoint.node_id
    assert page.complete
    assert len(endpoints) == page.processed_conversations == page.total_conversations
    return nodes, endpoints


async def test_nested_cloning_empty_and_exact_prefix_endpoints(sqlite_instance: SQLiteMemory) -> None:
    main, empty, adversarial, foreign = (str(uuid4()) for _ in range(4))
    history = [
        make_tree_message(conversation_id=main, role="system", values=[("text", "setup")]),
        make_tree_message(conversation_id=main, sequence=1, role="simulated_assistant"),
        make_tree_message(
            conversation_id=main, sequence=2, role="user", values=[("text", "prompt"), ("image_path", "image.png")]
        ),
        make_tree_message(conversation_id=main, sequence=3, role="tool", values=[("tool_call", "tool result")]),
    ]
    branch, branch_history = clone_tree_history(memory=sqlite_instance, messages=history[:2])
    branch_history.extend(
        [
            make_tree_message(conversation_id=branch, sequence=2, role="developer", values=[("text", "branch")]),
            make_tree_message(conversation_id=branch, sequence=3, role="assistant", values=[("error", "branch error")]),
        ]
    )
    branch_history[-1].message_pieces[0].response_error = "processing"
    nested, nested_history = clone_tree_history(memory=sqlite_instance, messages=branch_history[:3])
    nested_history.append(make_tree_message(conversation_id=nested, sequence=3, role="assistant"))
    prefix, prefix_history = clone_tree_history(memory=sqlite_instance, messages=history[:2])
    excluded = [make_tree_message(conversation_id=adversarial), make_tree_message(conversation_id=foreign)]
    store_tree_messages(
        memory=sqlite_instance, messages=[*history, *branch_history, *nested_history, *prefix_history, *excluded]
    )
    attack = store_tree_attack(
        memory=sqlite_instance,
        main_conversation_id=main,
        pruned=[branch, nested, prefix, empty],
        adversarial=[adversarial],
    )
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    assert len(first.nodes) == 1
    assert not first.complete
    assert first.processed_conversations == 0
    assert first.conversations == []
    nodes, endpoints = await collect_tree_async(service=service, first=first, limit=1)
    assert len(nodes) == 7
    assert len([node for node in nodes if node.parent_node_id is None]) == 1
    assert endpoints[empty] is None
    assert endpoints[prefix] == next(node.node_id for node in nodes if node.role == "simulated_assistant")
    assert endpoints[main] != endpoints[branch] != endpoints[nested]
    assert set(endpoints) == attack.get_active_conversation_ids()
    assert {node.role for node in nodes} == {"system", "simulated_assistant", "user", "tool", "developer", "assistant"}
    assert any(node.piece_types == ["text", "image_path"] and node.piece_count == 2 for node in nodes)
    for node in nodes:
        assert node.message.conversation_id in attack.get_active_conversation_ids()
        assert set(node.model_dump()) == {
            "node_id",
            "parent_node_id",
            "message",
            "role",
            "piece_types",
            "piece_count",
            "preview_key",
            "created_at",
        }


@pytest.mark.parametrize(
    "difference", ["independent-reply", "second-lineage", "converted-value", "role", "error", "sequence"]
)
async def test_lineage_and_actual_representation_prevent_false_merges(
    *, sqlite_instance: SQLiteMemory, difference: str
) -> None:
    main = str(uuid4())
    prompt = make_tree_message(conversation_id=main)
    response = make_tree_message(
        conversation_id=main, sequence=1, role="assistant", values=[("text", "same"), ("text", "second")]
    )
    branch, copied = clone_tree_history(memory=sqlite_instance, messages=[prompt, response])
    changed = copied[-1]
    if difference == "independent-reply":
        copied[-1] = make_tree_message(
            conversation_id=branch, sequence=1, role="assistant", values=[("text", "same"), ("text", "second")]
        )
    elif difference == "second-lineage":
        changed.message_pieces[1].original_prompt_id = uuid4()
    elif difference == "converted-value":
        changed.message_pieces[1].converted_value = "changed"
        changed.message_pieces[1].converted_value_sha256 = hashlib.sha256(b"changed").hexdigest()
    elif difference == "role":
        for piece in changed.message_pieces:
            piece.role = "tool"
    elif difference == "sequence":
        for piece in changed.message_pieces:
            piece.sequence = 2
    else:
        changed.message_pieces[1].response_error = "blocked"
    store_tree_messages(memory=sqlite_instance, messages=[prompt, response, *copied])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id)
    nodes, endpoints = await collect_tree_async(service=service, first=first, limit=100)
    assert len(nodes) == 3
    assert endpoints[main] != endpoints[branch]
    assert len({node.parent_node_id for node in nodes if node.parent_node_id is not None}) == 1


async def test_prefix_scope_and_parent_independent_preview_key(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    first_root = make_tree_message(conversation_id=main, values=[("text", "first root")])
    shared = make_tree_message(conversation_id=main, sequence=1, role="assistant")
    branch, copied = clone_tree_history(memory=sqlite_instance, messages=[shared])
    other_root = make_tree_message(conversation_id=branch, values=[("text", "different root")])
    store_tree_messages(memory=sqlite_instance, messages=[first_root, shared, other_root, *copied])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    service = ConversationTreeService(memory=sqlite_instance)
    page = await service.get_page_async(attack_result_id=attack.attack_result_id)
    replies = [node for node in page.nodes if node.role == "assistant"]
    assert len(page.nodes) == 4
    assert len(replies) == 2
    assert replies[0].node_id != replies[1].node_id
    assert replies[0].preview_key == replies[1].preview_key


async def test_priority_and_main_promotion_preserve_node_identities(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    history = [
        make_tree_message(conversation_id=main, sequence=index, values=[("text", str(index)), ("text", "piece")])
        for index in range(3)
    ]
    branch, copied = clone_tree_history(memory=sqlite_instance, messages=history)
    copied.append(make_tree_message(conversation_id=branch, sequence=3))
    store_tree_messages(memory=sqlite_instance, messages=[*history, *copied])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(
        attack_result_id=attack.attack_result_id, limit=2, prioritize_conversation_id=branch
    )
    assert all(node.message.conversation_id == branch for node in first.nodes)
    with sqlite_instance.get_session() as session:
        session.execute(
            update(AttackResultEntry)
            .where(AttackResultEntry.id == UUID(attack.attack_result_id))
            .values(conversation_id=branch, pruned_conversation_ids=[main])
        )
        session.commit()
    original_nodes, original_endpoints = await collect_tree_async(service=service, first=first, limit=2)
    refreshed = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=2)
    nodes, endpoints = await collect_tree_async(service=service, first=refreshed, limit=2)
    assert refreshed.main_conversation_id == branch
    assert refreshed.revision != first.revision
    assert {node.node_id for node in nodes} == {node.node_id for node in original_nodes}
    assert endpoints == original_endpoints


async def test_append_and_new_membership_do_not_change_an_open_snapshot(sqlite_instance: SQLiteMemory) -> None:
    main, empty, new_branch = (str(uuid4()) for _ in range(3))
    history = [make_tree_message(conversation_id=main, sequence=index) for index in range(3)]
    store_tree_messages(memory=sqlite_instance, messages=history)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[empty])
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    store_tree_messages(
        memory=sqlite_instance,
        messages=[
            make_tree_message(conversation_id=main, sequence=3),
            make_tree_message(conversation_id=empty),
            make_tree_message(conversation_id=new_branch),
        ],
    )
    with sqlite_instance.get_session() as session:
        session.execute(
            update(AttackResultEntry)
            .where(AttackResultEntry.id == UUID(attack.attack_result_id))
            .values(pruned_conversation_ids=[empty, new_branch])
        )
        session.commit()
    old_nodes, old_endpoints = await collect_tree_async(service=service, first=first, limit=1)
    assert len(old_nodes) == 3
    assert old_endpoints[empty] is None
    assert new_branch not in old_endpoints
    refreshed = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    nodes, endpoints = await collect_tree_async(service=service, first=refreshed, limit=1)
    assert len(nodes) == 6
    assert set(endpoints) == {main, empty, new_branch}
    assert {node.node_id for node in old_nodes} <= {node.node_id for node in nodes}


async def test_retrying_the_same_cursor_is_idempotent_even_concurrently(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    store_tree_messages(
        memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=index) for index in range(4)]
    )
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    with patch.object(service._reader, "read_message_page", wraps=service._reader.read_message_page) as read_page:
        pages = await asyncio.gather(
            *(
                service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor)
                for _ in range(2)
            )
        )
    assert pages[0] == pages[1]
    assert pages[0].nodes[0].message.sequence == 1
    read_page.assert_called_once()
    next_page = await service.get_page_async(
        attack_result_id=attack.attack_result_id, limit=1, cursor=pages[0].next_cursor
    )
    assert next_page.nodes[0].message.sequence == 2


async def test_cursor_scope_limits_expiration_eviction_and_cache_loss(sqlite_instance: SQLiteMemory) -> None:
    main, other = str(uuid4()), str(uuid4())
    store_tree_messages(
        memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=index) for index in range(3)]
    )
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[other])
    foreign = store_tree_attack(memory=sqlite_instance, main_conversation_id=other)
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    assert first.next_cursor is not None
    with pytest.raises(TreeCursorMismatchError, match="attack and limit"):
        await service.get_page_async(attack_result_id=foreign.attack_result_id, limit=1, cursor=first.next_cursor)
    with pytest.raises(TreeCursorMismatchError, match="attack and limit"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=2, cursor=first.next_cursor)
    with pytest.raises(TreeCursorMismatchError, match="priority"):
        await service.get_page_async(
            attack_result_id=attack.attack_result_id,
            limit=1,
            cursor=first.next_cursor,
            prioritize_conversation_id=other,
        )
    with pytest.raises(TreeCursorMismatchError, match="Invalid"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor="invalid")
    with pytest.raises(TreeCursorExpiredError, match="evicted"):
        await ConversationTreeService(memory=sqlite_instance).get_page_async(
            attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor
        )
    with patch.object(service, "_CACHE_TTL_SECONDS", 0), pytest.raises(TreeCursorExpiredError, match="expired"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor)
    fresh = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    with patch.object(service, "_CACHE_MAX_SNAPSHOTS", 1):
        await service.get_page_async(attack_result_id=foreign.attack_result_id, limit=1)
    with pytest.raises(TreeCursorExpiredError, match="evicted"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=fresh.next_cursor)
    restored = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    assert restored.nodes[0].node_id == first.nodes[0].node_id


async def test_removing_membership_invalidates_cached_pages(sqlite_instance: SQLiteMemory) -> None:
    main, branch = str(uuid4()), str(uuid4())
    store_tree_messages(
        memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=index) for index in range(3)]
    )
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    with sqlite_instance.get_session() as session:
        session.execute(
            update(AttackResultEntry)
            .where(AttackResultEntry.id == UUID(attack.attack_result_id))
            .values(pruned_conversation_ids=[])
        )
        session.commit()
    with pytest.raises(TreeSnapshotChangedError, match="membership"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor)


async def test_page_failure_does_not_advance_cursor(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    store_tree_messages(
        memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=index) for index in range(3)]
    )
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    with patch.object(service._reader, "read_message_page", side_effect=RuntimeError("Read failed")):
        with pytest.raises(RuntimeError, match="Read failed"):
            await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor)
    recovered = await service.get_page_async(
        attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor
    )
    assert recovered.nodes[0].message.sequence == 1


async def test_text_budget_is_per_message_preserves_every_piece_and_deduplicates(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    message = make_tree_message(
        conversation_id=main,
        values=[
            ("text", "a" * 180),
            ("image_path", "never-sign-or-open-this.png"),
            ("text", "b" * 1000),
            ("audio_path", "never-sign-or-open-this.wav"),
            ("tool_call", "c" * 20),
            ("text", ""),
        ],
    )
    message.message_pieces[-2].response_error = "processing"
    store_tree_messages(memory=sqlite_instance, messages=[message])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    service = ConversationTreeService(memory=sqlite_instance)
    reference = TreeMessageReference(conversation_id=main, sequence=0)
    with (
        patch.object(ConversationTreeMedia, "describe", side_effect=AssertionError("Text must not resolve media")),
        patch.object(sqlite_instance, "get_message_pieces", side_effect=AssertionError("No full messages")),
        patch.object(sqlite_instance, "get_attack_results", side_effect=AssertionError("No full attacks")),
        patch.object(sqlite_instance, "get_scores", side_effect=AssertionError("No scores")),
        capture_tree_reads(sqlite_instance) as reads,
    ):
        response = await service.get_previews_async(
            attack_result_id=attack.attack_result_id,
            request=ConversationTreePreviewRequest(messages=[reference, reference], level=TreePreviewLevel.TEXT),
        )
    assert len(response.previews) == 1
    pieces = response.previews[0].pieces
    assert len(pieces) == 6
    assert [piece.text for piece in pieces] == ["a" * 180, None, "b" * 20, None, "", ""]
    assert [piece.truncated for piece in pieces] == [False, False, True, False, True, False]
    assert sum(len(piece.text or "") for piece in pieces) == service.TEXT_PREVIEW_CHAR_LIMIT == 200
    assert all(piece.media_url is None and piece.thumbnail_url is None for piece in pieces)
    assert pieces[-2].response_error == "processing"
    assert "never-sign" not in response.model_dump_json()
    assert len(reads) == 2


async def test_more_than_1000_nodes_and_duplicate_histories_keep_first_read_bounded(
    sqlite_instance: SQLiteMemory,
) -> None:
    main, empty = str(uuid4()), str(uuid4())
    history = [
        make_tree_message(
            conversation_id=main,
            sequence=index,
            role="user" if index % 2 == 0 else "assistant",
            values=[("text", f"message-{index}"), ("image_path", f"never-read-{index}.png")]
            if index % 20 == 0
            else [("text", f"message-{index}")],
        )
        for index in range(1050)
    ]
    all_messages: list[Message] = list(history)
    branches = []
    for _ in range(5):
        branch, copied = clone_tree_history(memory=sqlite_instance, messages=history)
        copied.append(
            make_tree_message(conversation_id=branch, sequence=1050, role="assistant", values=[("text", "same reply")])
        )
        branches.append(branch)
        all_messages.extend(copied)
    prefix, copied_prefix = clone_tree_history(memory=sqlite_instance, messages=history[:300])
    all_messages.extend(copied_prefix)
    store_tree_messages(memory=sqlite_instance, messages=all_messages)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[*branches, prefix, empty])
    service = ConversationTreeService(memory=sqlite_instance)
    with (
        patch.object(service._reader, "read_message_page", wraps=service._reader.read_message_page) as read_page,
        capture_tree_reads(sqlite_instance) as reads,
    ):
        first = await service.get_page_async(attack_result_id=attack.attack_result_id)
    read_page.assert_called_once()
    assert len(first.nodes) == 100
    assert first.processed_conversations == 0
    assert first.total_conversations == 8
    assert not first.complete
    assert all(node.message.conversation_id == main for node in first.nodes)
    assert sum(node.piece_count for node in first.nodes) == 105
    assert len(reads) == 4
    assert all(read.parameter_count < 999 for read in reads)
    assert all("original_value," not in read.statement and "ScoreEntries" not in read.statement for read in reads)
    with capture_tree_reads(sqlite_instance) as later_reads:
        nodes, endpoints = await collect_tree_async(service=service, first=first, limit=100)
    assert len(nodes) == 1055
    assert endpoints[empty] is None
    assert len(endpoints) == 8
    assert len({endpoints[branch] for branch in branches}) == 5
    assert all(read.parameter_count < 999 for read in later_reads)
    assert len(all_messages) > 6000


async def test_empty_histories_have_bounded_endpoint_pages_and_no_synthetic_root(
    sqlite_instance: SQLiteMemory,
) -> None:
    conversations = [str(uuid4()) for _ in range(5)]
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=conversations[0], pruned=conversations[1:])
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=2)
    assert first.nodes == []
    assert len(first.conversations) == first.processed_conversations == 2
    assert not first.complete
    nodes, endpoints = await collect_tree_async(service=service, first=first, limit=2)
    assert nodes == []
    assert endpoints == dict.fromkeys(conversations)


async def test_text_budget_counts_characters_preserves_spaces_and_resets_for_each_message(
    sqlite_instance: SQLiteMemory,
) -> None:
    main = str(uuid4())
    messages = [
        make_tree_message(conversation_id=main, values=[("text", "\U0001f600" * 199 + " " + "extra")]),
        make_tree_message(conversation_id=main, sequence=1, values=[("text", " " * 200)]),
    ]
    store_tree_messages(memory=sqlite_instance, messages=messages)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    service = ConversationTreeService(memory=sqlite_instance)
    response = await service.get_previews_async(
        attack_result_id=attack.attack_result_id,
        request=ConversationTreePreviewRequest(
            messages=[
                TreeMessageReference(conversation_id=main, sequence=1),
                TreeMessageReference(conversation_id=main, sequence=0),
            ],
            level=TreePreviewLevel.TEXT,
        ),
    )
    assert [preview.message.sequence for preview in response.previews] == [1, 0]
    assert [len(preview.pieces[0].text or "") for preview in response.previews] == [200, 200]
    assert response.previews[0].pieces[0].text == " " * 200
    assert not response.previews[0].pieces[0].truncated
    assert response.previews[1].pieces[0].text == "\U0001f600" * 199 + " "
    assert response.previews[1].pieces[0].truncated


async def test_cache_retention_is_bounded_by_records_and_old_cursors_expire(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    store_tree_messages(
        memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=index) for index in range(3)]
    )
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    service = ConversationTreeService(memory=sqlite_instance)
    with patch.object(service, "_CACHE_MAX_RECORDS", 4):
        first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
        for _ in range(3):
            await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
        assert sum(snapshot.weight for snapshot in service._cache.values()) <= 4
        with pytest.raises(TreeCursorExpiredError, match="evicted"):
            await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor)


async def test_replay_retention_and_modified_cursor_are_explicitly_rejected(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    store_tree_messages(
        memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=index) for index in range(9)]
    )
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    service = ConversationTreeService(memory=sqlite_instance)
    first = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1)
    assert first.next_cursor is not None
    changed = first.next_cursor[:-1] + ("a" if first.next_cursor[-1] != "a" else "b")
    with pytest.raises(TreeCursorExpiredError, match="invalid"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=changed)
    page = first
    for _ in range(service._REPLAY_PAGES + 1):
        page = await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=page.next_cursor)
    with pytest.raises(TreeCursorExpiredError, match="retry window"):
        await service.get_page_async(attack_result_id=attack.attack_result_id, limit=1, cursor=first.next_cursor)


async def test_legacy_text_hashes_still_merge_real_cloned_prefixes(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    original = make_tree_message(conversation_id=main, values=[("text", "first"), ("text", "second")])
    branch, copied = clone_tree_history(memory=sqlite_instance, messages=[original])
    for piece in copied[0].message_pieces:
        piece.original_value_sha256 = None
        piece.converted_value_sha256 = None
    store_tree_messages(memory=sqlite_instance, messages=[original, *copied])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    service = ConversationTreeService(memory=sqlite_instance)
    page = await service.get_page_async(attack_result_id=attack.attack_result_id)
    assert page.complete
    assert len(page.nodes) == 1
    assert len(page.conversations) == 2
    assert len({endpoint.node_id for endpoint in page.conversations}) == 1


async def test_independent_equal_media_messages_never_merge(sqlite_instance: SQLiteMemory) -> None:
    main, branch = str(uuid4()), str(uuid4())
    messages = [
        make_tree_message(
            conversation_id=conversation_id, values=[("text", "same description"), ("image_path", "same-image.png")]
        )
        for conversation_id in (main, branch)
    ]
    store_tree_messages(memory=sqlite_instance, messages=messages)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    service = ConversationTreeService(memory=sqlite_instance)
    page = await service.get_page_async(attack_result_id=attack.attack_result_id)
    assert page.complete
    assert len(page.nodes) == 2
    assert len({node.node_id for node in page.nodes}) == 2
    assert len({node.preview_key for node in page.nodes}) == 1
    assert all(node.parent_node_id is None for node in page.nodes)
    assert len({endpoint.node_id for endpoint in page.conversations}) == 2
