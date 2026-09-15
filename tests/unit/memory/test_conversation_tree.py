# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.dialects import mssql
from sqlalchemy.orm import Session

from pyrit.memory.conversation_tree import (
    ConversationTreeReader,
    TreeConversationBoundary,
    TreeMessageKey,
    TreeReadLimitError,
)
from unit.memory.conversation_tree_test_helpers import (
    capture_tree_reads,
    clone_tree_history,
    make_tree_message,
    store_tree_attack,
    store_tree_messages,
)

if TYPE_CHECKING:
    from pyrit.memory.sqlite_memory import SQLiteMemory

pytestmark = pytest.mark.usefixtures("patch_central_database")


def test_snapshot_uses_domain_active_membership_and_includes_empty_histories(sqlite_instance: SQLiteMemory) -> None:
    main, branch, empty, adversarial, foreign = (str(uuid4()) for _ in range(5))
    store_tree_messages(
        memory=sqlite_instance,
        messages=[
            make_tree_message(conversation_id=conversation_id)
            for conversation_id in (main, branch, adversarial, foreign)
        ],
    )
    attack = store_tree_attack(
        memory=sqlite_instance, main_conversation_id=main, pruned=[branch, empty], adversarial=[adversarial]
    )
    reader = ConversationTreeReader(memory=sqlite_instance)
    with capture_tree_reads(sqlite_instance) as reads:
        scope, bounds = reader.capture_snapshot(attack.attack_result_id)
    assert scope.conversation_ids == attack.get_active_conversation_ids() == {main, branch, empty}
    assert {bound.conversation_id: bound.piece_count for bound in bounds} == {main: 1, branch: 1, empty: 0}
    assert next(bound for bound in bounds if bound.conversation_id == empty).max_sequence is None
    assert len(reads) == 2
    assert not any("ScoreEntries" in read.statement for read in reads)


def test_message_page_keeps_all_pieces_atomic_and_uses_actual_sequences(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    first = make_tree_message(
        conversation_id=main,
        role="system",
        values=[("text", "first"), ("image_path", "not-opened.png"), ("text", "last")],
    )
    second = make_tree_message(conversation_id=main, sequence=7, role="tool")
    store_tree_messages(memory=sqlite_instance, messages=[first, second])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    _, bounds = reader.capture_snapshot(attack.attack_result_id)
    with capture_tree_reads(sqlite_instance) as reads:
        page = reader.read_message_page(boundaries=bounds, after_sequence=None, limit=1)
        following = reader.read_message_page(boundaries=bounds, after_sequence=0, limit=1)
    assert len(page) == 1
    assert [piece.piece_id for piece in page[0].pieces] == [piece.id for piece in first.message_pieces]
    assert len(page[0].pieces) == 3
    assert following[0].key.sequence == 7
    assert following[0].pieces[0].role == "tool"
    assert len(reads) == 4
    assert all(read.parameter_count < 999 for read in reads)
    for read in reads:
        sql_server_query = str(read.compiled_statement.compile(dialect=mssql.dialect()))
        assert "SELECT" in sql_server_query
        assert "ScoreEntries" not in sql_server_query


def test_cloned_piece_order_comes_from_original_lineage_timestamps(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    original = make_tree_message(
        conversation_id=main, values=[("text", "first"), ("image_path", "image.png"), ("text", "third")]
    )
    branch, copied = clone_tree_history(memory=sqlite_instance, messages=[original])
    assert len({piece.timestamp for piece in copied[0].message_pieces}) == 1
    store_tree_messages(memory=sqlite_instance, messages=[original, *copied])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main, pruned=[branch])
    reader = ConversationTreeReader(memory=sqlite_instance)
    scope, bounds = reader.capture_snapshot(attack.attack_result_id)
    messages = reader.read_message_page(boundaries=bounds, after_sequence=None, limit=2)
    expected = [piece.original_prompt_id for piece in original.message_pieces]
    assert all([piece.lineage_id for piece in message.pieces] == expected for message in messages)
    previews = reader.read_previews(
        scope=scope,
        messages=[TreeMessageKey(conversation_id=branch, sequence=0)],
        text_limit=200,
        include_media=False,
    )
    assert [piece.data_type for piece in previews] == ["text", "image_path", "text"]
    assert [piece.text for piece in previews] == ["first", None, "third"]


def test_empty_filters_do_not_query_any_conversations(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    store_tree_messages(memory=sqlite_instance, messages=[make_tree_message(conversation_id=main)])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    scope = reader.get_scope(attack.attack_result_id)
    with capture_tree_reads(sqlite_instance) as reads:
        assert reader.read_message_page(boundaries=[], after_sequence=None, limit=10) == ()
        assert reader.read_previews(scope=scope, messages=[], text_limit=200, include_media=False) == ()
        with sqlite_instance.get_session() as session:
            assert reader._read_boundaries(session=session, conversation_ids=[]) == ()
    assert reads == []


def test_text_preview_slices_in_sql_and_never_reads_media_values_or_scores(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    message = make_tree_message(
        conversation_id=main,
        values=[("text", "x" * 100_000), ("video_path", "private-video.mp4"), ("tool_call", "y" * 10_000)],
    )
    store_tree_messages(memory=sqlite_instance, messages=[message])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    scope = reader.get_scope(attack.attack_result_id)
    with capture_tree_reads(sqlite_instance) as reads:
        pieces = reader.read_previews(
            scope=scope,
            messages=[TreeMessageKey(conversation_id=main, sequence=0)],
            text_limit=200,
            include_media=False,
        )
    assert [len(piece.text) if piece.text is not None else None for piece in pieces] == [201, None, 201]
    assert all(piece.media_value is None for piece in pieces)
    assert len(reads) == 1
    assert "substr(" in reads[0].statement.lower()
    assert "original_value" not in reads[0].statement
    assert "ScoreEntries" not in reads[0].statement


def test_preview_scope_is_checked_before_hydrating_values(sqlite_instance: SQLiteMemory) -> None:
    main, foreign = str(uuid4()), str(uuid4())
    store_tree_messages(memory=sqlite_instance, messages=[make_tree_message(conversation_id=foreign)])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    scope = reader.get_scope(attack.attack_result_id)
    with capture_tree_reads(sqlite_instance) as reads, pytest.raises(PermissionError, match="not active"):
        reader.read_previews(
            scope=scope,
            messages=[TreeMessageKey(conversation_id=foreign, sequence=0)],
            text_limit=200,
            include_media=True,
        )
    assert reads == []
    with pytest.raises(FileNotFoundError, match="does not exist"):
        reader.read_previews(
            scope=scope,
            messages=[TreeMessageKey(conversation_id=main, sequence=99)],
            text_limit=200,
            include_media=False,
        )


def test_legacy_hash_fallback_is_narrow_and_never_opens_media(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    messages = [
        make_tree_message(conversation_id=main, sequence=index, values=[("image_path", f"missing-{index}.png")])
        for index in range(20)
    ]
    for message in messages:
        message.message_pieces[0].original_value_sha256 = None
        message.message_pieces[0].converted_value_sha256 = None
    store_tree_messages(memory=sqlite_instance, messages=messages)
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    _, bounds = reader.capture_snapshot(attack.attack_result_id)
    with (
        patch.object(Path, "open", side_effect=AssertionError("No media I/O")),
        patch.object(reader, "_read_legacy_hashes", wraps=reader._read_legacy_hashes) as fallback,
        capture_tree_reads(sqlite_instance) as reads,
    ):
        page = reader.read_message_page(boundaries=bounds, after_sequence=None, limit=1)
    assert len(page) == 1
    assert fallback.call_args.kwargs["piece_ids"] == [messages[0].message_pieces[0].id]
    assert len(reads) == 3
    assert page[0].pieces[0].converted_hash


def test_many_active_ids_use_parameter_bounded_aggregate_batches(sqlite_instance: SQLiteMemory) -> None:
    ids = [str(uuid4()) for _ in range(1201)]
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=ids[0], pruned=ids[1:])
    reader = ConversationTreeReader(memory=sqlite_instance)
    with capture_tree_reads(sqlite_instance) as reads:
        scope, bounds = reader.capture_snapshot(attack.attack_result_id)
    assert len(scope.conversation_ids) == len(bounds) == 1201
    assert len(reads) == 5
    assert all(read.parameter_count <= 400 for read in reads)
    assert all(bound.max_sequence is None for bound in bounds)


def test_atomic_piece_budget_never_returns_a_partial_message(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    first = make_tree_message(conversation_id=main, values=[("text", "a"), ("text", "b")])
    second = make_tree_message(conversation_id=main, sequence=1, values=[("text", "c"), ("text", "d")])
    store_tree_messages(memory=sqlite_instance, messages=[first, second])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    _, bounds = reader.capture_snapshot(attack.attack_result_id)
    with patch.object(reader, "MAX_PIECES", 3):
        page = reader.read_message_page(boundaries=bounds, after_sequence=None, limit=2)
    assert len(page) == 1 and len(page[0].pieces) == 2
    with patch.object(reader, "MAX_PIECES", 1), pytest.raises(TreeReadLimitError, match="atomic"):
        reader.read_message_page(boundaries=bounds, after_sequence=None, limit=1)


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.5])
def test_reader_rejects_invalid_limits(*, sqlite_instance: SQLiteMemory, limit: int) -> None:
    reader = ConversationTreeReader(memory=sqlite_instance)
    with pytest.raises(ValueError, match="limit"):
        reader.read_message_page(boundaries=[], after_sequence=None, limit=limit)


def test_sql_server_preview_uses_substring_not_sqlite_substr(sqlite_instance: SQLiteMemory) -> None:
    reader = ConversationTreeReader(memory=sqlite_instance)
    session = MagicMock(spec=Session)
    session.get_bind.return_value.dialect = mssql.dialect()
    session.execute.return_value = []
    reader._preview_query(
        session=session,
        keys=[TreeMessageKey(conversation_id=str(uuid4()), sequence=0)],
        text_limit=200,
        include_media=False,
    )
    query = session.execute.call_args.args[0]
    sql = str(query.compile(dialect=mssql.dialect()))
    assert "substring(" in sql.lower()
    assert "substr(" not in sql.lower()
    assert "TOP" in sql


def test_snapshot_excludes_later_append_sequences(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    store_tree_messages(memory=sqlite_instance, messages=[make_tree_message(conversation_id=main)])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    _, bounds = reader.capture_snapshot(attack.attack_result_id)
    store_tree_messages(memory=sqlite_instance, messages=[make_tree_message(conversation_id=main, sequence=1)])
    page = reader.read_message_page(boundaries=bounds, after_sequence=None, limit=100)
    assert [message.key.sequence for message in page] == [0]
    _, refreshed_bounds = reader.capture_snapshot(attack.attack_result_id)
    assert refreshed_bounds == (TreeConversationBoundary(conversation_id=main, max_sequence=1, piece_count=2),)


def test_oversized_later_message_does_not_hide_the_first_usable_page(sqlite_instance: SQLiteMemory) -> None:
    main = str(uuid4())
    first = make_tree_message(conversation_id=main)
    oversized = make_tree_message(
        conversation_id=main, sequence=1, values=[("text", "a"), ("text", "b"), ("text", "c")]
    )
    store_tree_messages(memory=sqlite_instance, messages=[first, oversized])
    attack = store_tree_attack(memory=sqlite_instance, main_conversation_id=main)
    reader = ConversationTreeReader(memory=sqlite_instance)
    _, bounds = reader.capture_snapshot(attack.attack_result_id)
    with patch.object(reader, "MAX_PIECES", 2):
        page = reader.read_message_page(boundaries=bounds, after_sequence=None, limit=100)
        assert len(page) == 1
        assert page[0].key.sequence == 0
        with pytest.raises(TreeReadLimitError, match="atomic"):
            reader.read_message_page(boundaries=bounds, after_sequence=0, limit=100)
