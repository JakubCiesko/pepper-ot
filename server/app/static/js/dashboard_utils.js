// collapsable utility
//function toggleSection(id) {
//	const content = document.getElementById(id);
//	const arrow = document.getElementById(id.replace("-content", "-arrow"));
//	if (content.classList.contains("hidden")) {
//		content.classList.remove("hidden");
//		arrow.textContent = "▲";
//	} else {
//		content.classList.add("hidden");
//		arrow.textContent = "▼";
//	}
//}
function showStatusMessage(msg, success = true) {
	const el = document.getElementById("status-message");
	el.textContent = msg;
	el.classList.remove("hidden");
	el.classList.toggle("text-emerald-400", success);
	el.classList.toggle("text-rose-400", !success);
	setTimeout(() => el.classList.add("hidden"), 3000); // hide after 3s
}
window.showStatusMessage = showStatusMessage;
//const sentencesContainer = document.getElementById("sentences-content");

//function displayPepperSentence(text) {
//	if (!text || !sentencesContainer) return;
//
//	// Remove placeholder text
//	if (
//		sentencesContainer.children.length === 1 &&
//		sentencesContainer.children[0].textContent.includes("No messages")
//	) {
//		sentencesContainer.innerHTML = "";
//	}
//
//	// Clear any previous sentence → keep only the newest
//	sentencesContainer.innerHTML = "";
//
//	const div = document.createElement("div");
//	div.className =
//		"bg-slate-950 border border-slate-800 p-3 rounded shadow-sm text-slate-300 whitespace-pre-wrap break-all overflow-x-auto";
//	div.textContent = text;
//
//	sentencesContainer.appendChild(div);
//}

function initTabGroup(groupName) {
	const buttons = Array.from(
		document.querySelectorAll(
			`[data-tab-group="${groupName}"][data-tab-target]`,
		),
	);
	const panels = Array.from(
		document.querySelectorAll(`.tab-panel[data-tab-group="${groupName}"]`),
	);
	if (buttons.length === 0 || panels.length === 0) return;

	function activate(targetId) {
		panels.forEach((panel) => {
			panel.classList.toggle("hidden", panel.id !== targetId);
		});
		buttons.forEach((btn) => {
			const isActive = btn.getAttribute("data-tab-target") === targetId;
			btn.classList.toggle("active", isActive);
		});
	}

	buttons.forEach((btn) => {
		btn.addEventListener("click", () =>
			activate(btn.getAttribute("data-tab-target")),
		);
	});

	activate(buttons[0].getAttribute("data-tab-target"));
}

function initThemeToggle() {
	const toggles = [
		document.getElementById("theme-toggle"),
		document.getElementById("theme-toggle-mobile"),
	].filter(Boolean);
	if (toggles.length === 0) return;

	const stored = localStorage.getItem("pepper_theme");
	if (stored === "dark") {
		document.body.classList.remove("theme-light");
		document.body.classList.add("theme-dark");
	} else {
		document.body.classList.remove("theme-dark");
		document.body.classList.add("theme-light");
	}

	toggles.forEach((toggle) => {
		toggle.addEventListener("click", () => {
			document.body.classList.toggle("theme-dark");
			document.body.classList.toggle("theme-light");
			localStorage.setItem(
				"pepper_theme",
				document.body.classList.contains("theme-dark") ? "dark" : "light",
			);
		});
	});
}

function initPageNavigation() {
	const navItems = Array.from(
		document.querySelectorAll(".nav-item[data-page-target]"),
	);
	const pages = Array.from(document.querySelectorAll(".page-panel"));
	if (navItems.length === 0 || pages.length === 0) return;

	function activate(pageId) {
		pages.forEach((page) => {
			page.classList.toggle("hidden", page.id !== pageId);
		});
		navItems.forEach((btn) => {
			const isActive = btn.getAttribute("data-page-target") === pageId;
			btn.classList.toggle("active", isActive);
		});
	}

	navItems.forEach((btn) => {
		btn.addEventListener("click", () =>
			activate(btn.getAttribute("data-page-target")),
		);
	});

	activate(navItems[0].getAttribute("data-page-target"));
}

function initSidebarToggle() {
	const toggle = document.getElementById("sidebar-toggle");
	if (!toggle) return;
	const stored = localStorage.getItem("pepper_sidebar");
	if (stored === "collapsed") {
		document.body.classList.add("sidebar-collapsed");
	}
	toggle.addEventListener("click", () => {
		document.body.classList.toggle("sidebar-collapsed");
		localStorage.setItem(
			"pepper_sidebar",
			document.body.classList.contains("sidebar-collapsed")
				? "collapsed"
				: "expanded",
		);
	});
}

initTabGroup("live");
initTabGroup("settings");
initThemeToggle();
initPageNavigation();
initSidebarToggle();
