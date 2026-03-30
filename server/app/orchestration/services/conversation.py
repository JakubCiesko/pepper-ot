import asyncio
from dataclasses import dataclass
import time
from uuid import uuid4


@dataclass
class ConversationMessage:
    id: str
    role: str
    text: str
    timestamp: float


@dataclass
class ConversationState:
    chat_id: str
    created_at: float
    updated_at: float
    messages: list[ConversationMessage]


class ConversationService:
    def __init__(self, max_messages: int = 10):
        self.max_messages = max(2, int(max_messages))
        self._conversations: dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def ensure_conversation(
        self, chat_id: str | None = None
    ) -> ConversationState:
        async with self._lock:
            if chat_id and chat_id in self._conversations:
                return self._conversations[chat_id]
            now = time.time()
            cid = chat_id or str(uuid4())
            state = ConversationState(
                chat_id=cid,
                created_at=now,
                updated_at=now,
                messages=[],
            )
            self._conversations[cid] = state
            return state

    async def add_message(
        self, chat_id: str, role: str, text: str
    ) -> ConversationMessage:
        role_norm = role.strip().lower()
        if role_norm not in {"user", "assistant"}:
            raise ValueError("role must be one of: user, assistant")

        state = await self.ensure_conversation(chat_id)
        msg = ConversationMessage(
            id=str(uuid4()),
            role=role_norm,
            text=text.strip(),
            timestamp=time.time(),
        )
        async with self._lock:
            state.messages.append(msg)
            if len(state.messages) > self.max_messages:
                state.messages = state.messages[-self.max_messages :]
            state.updated_at = time.time()
        return msg

    async def prompt_history(
        self,
        chat_id: str,
        *,
        include_last_user: bool = False,
    ) -> list[tuple[str, str]]:
        state = await self.ensure_conversation(chat_id)
        async with self._lock:
            messages = list(state.messages)
        if not include_last_user and messages and messages[-1].role == "user":
            messages = messages[:-1]
        return [(msg.role, msg.text) for msg in messages]

    async def get_conversation(self, chat_id: str) -> ConversationState | None:
        async with self._lock:
            return self._conversations.get(chat_id)

    async def list_conversations(self, limit: int = 20) -> list[dict]:
        async with self._lock:
            conversations = sorted(
                self._conversations.values(),
                key=lambda item: item.updated_at,
                reverse=True,
            )
        out = []
        for state in conversations[: max(1, int(limit))]:
            last = state.messages[-1] if state.messages else None
            out.append(
                {
                    "chat_id": state.chat_id,
                    "created_at": state.created_at,
                    "updated_at": state.updated_at,
                    "message_count": len(state.messages),
                    "last_message": (
                        {
                            "role": last.role,
                            "text": last.text,
                            "timestamp": last.timestamp,
                        }
                        if last is not None
                        else None
                    ),
                }
            )
        return out

    async def delete_conversation(self, chat_id: str) -> bool:
        async with self._lock:
            return self._conversations.pop(chat_id, None) is not None

    async def reset_conversation(self, chat_id: str) -> bool:
        async with self._lock:
            state = self._conversations.get(chat_id)
            if state is None:
                return False
            state.messages = []
            state.updated_at = time.time()
            return True

    @staticmethod
    def serialize_message(msg: ConversationMessage) -> dict:
        return {
            "id": msg.id,
            "role": msg.role,
            "text": msg.text,
            "timestamp": msg.timestamp,
        }

    @staticmethod
    def serialize_conversation(state: ConversationState) -> dict:
        return {
            "chat_id": state.chat_id,
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "messages": [
                ConversationService.serialize_message(msg) for msg in state.messages
            ],
        }
