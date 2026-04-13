import asyncio
from dataclasses import dataclass
import time
from uuid import uuid4


@dataclass
class ConversationMessage:
    id: str
    role: str
    text_original: str
    text_model: str
    language_original: str | None
    language_model: str | None
    translation_applied: bool
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
        self,
        chat_id: str,
        role: str,
        text_original: str,
        text_model: str,
        *,
        language_original: str | None = None,
        language_model: str | None = None,
        translation_applied: bool = False,
    ) -> ConversationMessage:
        role_norm = role.strip().lower()
        if role_norm not in {"user", "assistant"}:
            raise ValueError("role must be one of: user, assistant")
        original = text_original.strip()
        model = text_model.strip()
        if not original:
            raise ValueError("text_original must not be empty")
        if not model:
            raise ValueError("text_model must not be empty")

        state = await self.ensure_conversation(chat_id)
        msg = ConversationMessage(
            id=str(uuid4()),
            role=role_norm,
            text_original=original,
            text_model=model,
            language_original=language_original,
            language_model=language_model,
            translation_applied=translation_applied,
            timestamp=time.time(),
        )
        async with self._lock:
            state.messages.append(msg)
            if len(state.messages) > self.max_messages:
                state.messages = state.messages[-self.max_messages :]
            state.updated_at = time.time()
        return msg

    async def _history_for_field(
        self,
        chat_id: str,
        *,
        include_last_user: bool = False,
        field: str,
    ) -> list[tuple[str, str]]:
        state = await self.ensure_conversation(chat_id)
        async with self._lock:
            messages = list(state.messages)
        if not include_last_user and messages and messages[-1].role == "user":
            messages = messages[:-1]
        return [(msg.role, getattr(msg, field)) for msg in messages]

    async def prompt_history_model(
        self,
        chat_id: str,
        *,
        include_last_user: bool = False,
    ) -> list[tuple[str, str]]:
        return await self._history_for_field(
            chat_id,
            include_last_user=include_last_user,
            field="text_model",
        )

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
                            "text": last.text_original,
                            "text_original": last.text_original,
                            "text_model": last.text_model,
                            "language_original": last.language_original,
                            "language_model": last.language_model,
                            "translation_applied": last.translation_applied,
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
            "text": msg.text_original,
            "text_original": msg.text_original,
            "text_model": msg.text_model,
            "language_original": msg.language_original,
            "language_model": msg.language_model,
            "translation_applied": msg.translation_applied,
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
