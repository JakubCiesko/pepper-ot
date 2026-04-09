import { showStatusMessage } from "../../core/notifications.js";

let activeChatId = null;
let getActiveFrameSnapshot = () => null;

const dom = {
	container: document.getElementById("sentences-content"),
	input: document.getElementById("conversation-input"),
	send: document.getElementById("conversation-send"),
	newButton: document.getElementById("conversation-new"),
	status: document.getElementById("conversation-status"),
	chatId: document.getElementById("conversation-chat-id"),
	routeSelect: document.getElementById("conversation-route-select"),
};

function setConversationStatus(text, ok = true) {
	if (!dom.status) return;
	dom.status.textContent = text || "";
	dom.status.classList.toggle("text-red-500", !ok);
}

function updateChatIdLabel() {
	if (!dom.chatId) return;
	dom.chatId.textContent = activeChatId ? `chat_id: ${activeChatId}` : "";
}

function selectedRoute() {
	const value = dom.routeSelect?.value || "chat";
	return value === "vision_chat" ? "vision_chat" : "chat";
}

function base64ToBlob(base64, mimeType = "image/jpeg") {
	const binary = atob(base64);
	const len = binary.length;
	const bytes = new Uint8Array(len);
	for (let i = 0; i < len; i += 1) {
		bytes[i] = binary.charCodeAt(i);
	}
	return new Blob([bytes], { type: mimeType });
}

function appendConversationMessage(message) {
	if (!message || !dom.container) return;
	if (
		dom.container.children.length === 1 &&
		dom.container.children[0].textContent.includes("No messages")
	) {
		dom.container.innerHTML = "";
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
	dom.container.appendChild(row);
	dom.container.scrollTop = dom.container.scrollHeight;
}

function replaceConversation(messages, chatId = null) {
	if (!dom.container) return;
	dom.container.innerHTML = "";
	if (chatId) activeChatId = chatId;
	updateChatIdLabel();
	if (!Array.isArray(messages) || messages.length === 0) {
		dom.container.innerHTML = `<p class="panel-muted">No messages yet...</p>`;
		return;
	}
	messages.forEach(appendConversationMessage);
}

async function sendChatMessage(query) {
	const res = await fetch("/api/v1/chat", {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ query, chat_id: activeChatId }),
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
	setConversationStatus("Message sent via /api/v1/chat");
}

async function sendVisionChatMessage(query) {
	const frame = getActiveFrameSnapshot?.() || null;
	if (!frame?.image) {
		throw new Error("No active frame image available. Run detect first.");
	}
	appendConversationMessage({ role: "user", text: query });
	const imageBlob = base64ToBlob(frame.image, "image/jpeg");
	const form = new FormData();
	form.append("file", imageBlob, "live_frame.jpg");
	form.append("query", query);
	const res = await fetch("/api/v1/vision_chat", {
		method: "POST",
		body: form,
	});
	if (!res.ok) {
		let detail = "Failed to send vision chat message";
		try {
			const body = await res.json();
			detail = body.detail || detail;
		} catch {
			// ignore
		}
		throw new Error(detail);
	}
	const payload = await res.json();
	appendConversationMessage({ role: "assistant", text: payload.answer || "" });
	setConversationStatus("Message sent via /api/v1/vision_chat");
}

async function sendConversationMessage() {
	if (!dom.input || !dom.send) return;
	const query = String(dom.input.value || "").trim();
	if (!query) return;
	dom.send.disabled = true;
	const original = dom.send.textContent;
	dom.send.textContent = "Sending...";
	try {
		const route = selectedRoute();
		if (route === "chat") {
			await sendChatMessage(query);
		} else {
			await sendVisionChatMessage(query);
		}
		dom.input.value = "";
	} catch (err) {
		setConversationStatus(err.message || "Failed to send message", false);
		showStatusMessage(err.message || "Failed to send message", false);
	} finally {
		dom.send.disabled = false;
		dom.send.textContent = original;
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

export function handleConversationWsMessage(payload) {
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

export function initConversationPanel(options = {}) {
	getActiveFrameSnapshot =
		options.getActiveFrameSnapshot || getActiveFrameSnapshot;

	if (dom.send) {
		dom.send.addEventListener("click", () => {
			sendConversationMessage();
		});
	}
	if (dom.input) {
		dom.input.addEventListener("keydown", (event) => {
			if (event.key === "Enter" && !event.shiftKey) {
				event.preventDefault();
				sendConversationMessage();
			}
		});
	}
	if (dom.newButton) {
		dom.newButton.addEventListener("click", () => {
			startNewConversation();
		});
	}
	if (dom.routeSelect) {
		dom.routeSelect.addEventListener("change", () => {
			const route = selectedRoute();
			if (route === "vision_chat") {
				setConversationStatus(
					"Vision chat mode enabled. Replies come from /api/v1/vision_chat.",
				);
			} else {
				setConversationStatus(
					"Chat mode enabled. Replies come from /api/v1/chat.",
				);
			}
		});
	}
	updateChatIdLabel();
	loadLatestConversation();
}
