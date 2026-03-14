(function () {
    const conversationContainer = document.getElementById("sentences-content");
    const conversationInput = document.getElementById("conversation-input");
    const conversationSend = document.getElementById("conversation-send");
    const conversationNew = document.getElementById("conversation-new");
    const conversationStatus = document.getElementById("conversation-status");
    const conversationChatId = document.getElementById("conversation-chat-id");

    let activeChatId = null;

    function setConversationStatus(text, ok = true) {
        if (!conversationStatus) return;
        conversationStatus.textContent = text || "";
        conversationStatus.classList.toggle("text-red-500", !ok);
    }

    function updateChatIdLabel() {
        if (!conversationChatId) return;
        conversationChatId.textContent = activeChatId ? `chat_id: ${activeChatId}` : "";
    }

    function appendConversationMessage(message) {
        if (!message || !conversationContainer) return;
        if (
            conversationContainer.children.length === 1 &&
            conversationContainer.children[0].textContent.includes("No messages")
        ) {
            conversationContainer.innerHTML = "";
        }
        const isUser = message.role === "user";
        const row = document.createElement("div");
        row.className = `mb-2 flex ${isUser ? "justify-end" : "justify-start"}`;
        const bubble = document.createElement("div");
        bubble.className = isUser
            ? "max-w-[85%] bg-sky-900 border border-sky-700 p-2 rounded text-slate-100 whitespace-pre-wrap break-all"
            : "max-w-[85%] bg-slate-950 border border-slate-800 p-2 rounded text-slate-300 whitespace-pre-wrap break-all";
        const rolePrefix = isUser ? "You" : "Pepper";
        bubble.textContent = `${rolePrefix}: ${message.text || ""}`;
        row.appendChild(bubble);
        conversationContainer.appendChild(row);
        conversationContainer.scrollTop = conversationContainer.scrollHeight;
    }

    function replaceConversation(messages, chatId = null) {
        if (!conversationContainer) return;
        conversationContainer.innerHTML = "";
        if (chatId) activeChatId = chatId;
        updateChatIdLabel();
        if (!Array.isArray(messages) || messages.length === 0) {
            conversationContainer.innerHTML = `<p class="panel-muted">No messages yet...</p>`;
            return;
        }
        messages.forEach(appendConversationMessage);
    }

    async function sendConversationMessage() {
        if (!conversationInput || !conversationSend) return;
        const query = String(conversationInput.value || "").trim();
        if (!query) return;
        conversationSend.disabled = true;
        const original = conversationSend.textContent;
        conversationSend.textContent = "Sending...";
        try {
            const res = await fetch("/api/v1/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query,
                    chat_id: activeChatId,
                }),
            });
            if (!res.ok) {
                let detail = "Failed to send message";
                try {
                    const body = await res.json();
                    detail = body.detail || detail;
                } catch {
                    // ignore
                }
                throw new Error(detail);
            }
            const payload = await res.json();
            activeChatId = payload.chat_id || activeChatId;
            updateChatIdLabel();
            conversationInput.value = "";
            setConversationStatus("Message sent");
        } catch (err) {
            setConversationStatus(err.message || "Failed to send message", false);
        } finally {
            conversationSend.disabled = false;
            conversationSend.textContent = original;
        }
    }

    function startNewConversation() {
        activeChatId = null;
        replaceConversation([], null);
        setConversationStatus("Started new conversation");
    }

    async function loadLatestConversation() {
        try {
            const listRes = await fetch("/api/v1/chat/conversations?limit=1");
            if (!listRes.ok) return;
            const listPayload = await listRes.json();
            const items = Array.isArray(listPayload.items) ? listPayload.items : [];
            if (!items.length) return;
            const chatId = items[0].chat_id;
            if (!chatId) return;
            const convoRes = await fetch(`/api/v1/chat/conversations/${chatId}`);
            if (!convoRes.ok) return;
            const convo = await convoRes.json();
            replaceConversation(convo.messages || [], convo.chat_id || chatId);
        } catch (err) {
            console.error("Failed to load latest conversation", err);
        }
    }

    function handleChatMessageEvent(payload) {
        const chatId = payload?.chat_id || null;
        if (activeChatId === null && chatId) {
            activeChatId = chatId;
            updateChatIdLabel();
        }
        if (chatId && activeChatId && chatId !== activeChatId) {
            return;
        }
        appendConversationMessage(payload?.message || {});
    }

    function init() {
        if (conversationSend) {
            conversationSend.addEventListener("click", () => {
                sendConversationMessage();
            });
        }
        if (conversationInput) {
            conversationInput.addEventListener("keydown", event => {
                if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendConversationMessage();
                }
            });
        }
        if (conversationNew) {
            conversationNew.addEventListener("click", () => {
                startNewConversation();
            });
        }
        updateChatIdLabel();
    }

    window.PepperConversationPanel = {
        init,
        loadLatestConversation,
        handleChatMessageEvent,
    };
})();
