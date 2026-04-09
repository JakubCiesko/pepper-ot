export function initSidebarToggle() {
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
