const mermaidConfig = {
	startOnLoad: false,
	securityLevel: "loose",
	theme: "neutral",
};

function renderMermaid() {
	if (typeof mermaid === "undefined") return;
	mermaid.initialize(mermaidConfig);
	mermaid.run({ querySelector: ".mermaid" });
}

if (window.document$) {
	window.document$.subscribe(() => {
		renderMermaid();
	});
} else {
	window.addEventListener("DOMContentLoaded", () => {
		renderMermaid();
	});
}
