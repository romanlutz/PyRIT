# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import pytest

from pyrit.models import Conversation, Message, MessageScorable
from pyrit.score import IncludesScorer

if TYPE_CHECKING:
    from pyrit.memory import SQLiteMemory


@pytest.mark.usefixtures("patch_central_database")
@pytest.mark.parametrize(
    ("expected", "answer", "ignore_case", "matched"),
    [
        ("picoCTF{fixture}", "picoCTF{fixture}", True, True),
        ("picoCTF{fixture}", "Found: PICOCTF{FIXTURE}\n", True, True),
        ("picoCTF{fixture}", "picoCTF{wrong}", True, False),
        ("picoCTF{fixture}", "picoCTF{fixture", True, False),
        ("Stra\u00dfe", "STRASSE", True, True),
        ("Stra\u00dfe", "STRASSE", False, False),
        (" flag ", "flag", True, False),
        ("flag", "unflagged", True, True),
        ("a-b", "a b", True, False),
        ("", "nonempty", True, True),
    ],
)
async def test_includes_scorer_matches_upstream_async(
    *,
    expected: str,
    answer: str,
    ignore_case: bool,
    matched: bool,
    sqlite_instance: SQLiteMemory,
) -> None:
    scorer = IncludesScorer(expected=expected, ignore_case=ignore_case, categories=["fixture_category"])
    conversation_id = str(uuid4())
    sqlite_instance.add_conversation_to_memory(conversation=Conversation(conversation_id=conversation_id))
    message = Message.from_prompt(prompt=answer, role="assistant")
    message.get_piece().conversation_id = conversation_id
    sqlite_instance.add_message_to_memory(request=message)
    scores = await scorer.score_async(scorable=MessageScorable.from_message(message))
    assert len(scores) == 1
    assert scores[0].get_value() is matched
    assert scores[0].score_metadata == {
        "value": "C" if matched else "I",
        "answer": answer.casefold() if ignore_case else answer,
        "explanation": answer,
    }
    stored = sqlite_instance.get_scores(score_ids=[str(scores[0].id)])
    assert len(stored) == 1
    assert stored[0].score_metadata == scores[0].score_metadata
    assert stored[0].score_category == ["fixture_category"]
    assert "expected" not in scorer.get_identifier().model_dump()
