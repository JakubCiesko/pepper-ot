export function showStatusMessage(msg, success = true, timeoutMs = 3000) {
	const el = document.getElementById("status-message");
	if (!el) return;
	el.textContent = msg;
	el.classList.remove("hidden");
	el.classList.toggle("text-emerald-400", success);
	el.classList.toggle("text-rose-400", !success);
	setTimeout(() => el.classList.add("hidden"), timeoutMs);
}
