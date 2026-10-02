"""Conversation memory, scoped to (tenant, user) so one user can never load another user's history."""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict


class ConversationStore:
    def __init__(self, max_conversations: int = 10_000, max_turns: int = 20) -> None:
        self._data: OrderedDict[tuple[str, str, str], list[dict[str, str]]] = OrderedDict()
        self._lock = threading.Lock()
        self.max_conversations = max_conversations
        self.max_turns = max_turns

    def new_id(self) -> str:
        return uuid.uuid4().hex

    def history(self, tenant: str, user: str, conversation_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._data.get((tenant, user, conversation_id), []))

    def append(self, tenant: str, user: str, conversation_id: str, question: str, answer: str) -> None:
        key = (tenant, user, conversation_id)
        with self._lock:
            turns = self._data.setdefault(key, [])
            turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
            del turns[: max(0, len(turns) - 2 * self.max_turns)]
            self._data.move_to_end(key)
            while len(self._data) > self.max_conversations:
                self._data.popitem(last=False)
