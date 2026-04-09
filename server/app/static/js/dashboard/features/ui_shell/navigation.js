export function initPageNavigation() {
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
