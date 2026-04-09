const dom = {
	sceneGraphContainer: document.getElementById("scene-graph-content"),
	sceneGraphRaw: document.getElementById("scene-graph-raw"),
	sceneGraphCanvas: document.getElementById("scene-graph-canvas"),
	sgZoomIn: document.getElementById("sg-zoom-in"),
	sgZoomOut: document.getElementById("sg-zoom-out"),
	sgFit: document.getElementById("sg-fit"),
};

let cy = null;

function labelColor(label) {
	if (!label) return "#22c55e";
	let hash = 0;
	for (let i = 0; i < label.length; i += 1) {
		hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
	}
	const hue = hash % 360;
	return `hsl(${hue}, 70%, 55%)`;
}

function buildElements(edges, memory) {
	const rels = Array.isArray(edges) ? edges : [];
	const memObjects = memory?.objects || [];
	const idToLabel = {};
	memObjects.forEach((obj) => {
		idToLabel[String(obj.id)] = obj.label;
	});

	const nodeIds = new Set();
	rels.forEach((e) => {
		nodeIds.add(String(e.sub));
		nodeIds.add(String(e.obj));
	});

	return [
		...Array.from(nodeIds).map((id) => {
			const lbl = idToLabel[id] || "";
			return {
				data: {
					id,
					label: lbl ? `${id}: ${lbl}` : id,
					color: labelColor(lbl),
				},
			};
		}),
		...rels.map((e, idx) => ({
			data: {
				id: `e${idx}-${e.sub}-${e.obj}-${e.rel}`,
				source: String(e.sub),
				target: String(e.obj),
				label: e.rel,
			},
		})),
	];
}

export function renderSceneGraph(edges, memory) {
	if (!dom.sceneGraphContainer) return;
	const rels = Array.isArray(edges) ? edges : [];
	dom.sceneGraphContainer.innerHTML = "";
	if (rels.length > 0) {
		rels.forEach((edge) => {
			const div = document.createElement("div");
			div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
			div.textContent = `${edge.sub} ${edge.rel} ${edge.obj}`;
			dom.sceneGraphContainer.appendChild(div);
		});
	} else {
		dom.sceneGraphContainer.innerHTML = `<p class="text-slate-500">No relations yet...</p>`;
	}

	if (dom.sceneGraphRaw) {
		dom.sceneGraphRaw.textContent = JSON.stringify(rels, null, 2);
	}
	if (!dom.sceneGraphCanvas || !window.cytoscape) return;

	if (cy) cy.destroy();
	const cs = getComputedStyle(document.body);
	const panelText = cs.getPropertyValue("--panel-text").trim();
	const panelMuted = cs.getPropertyValue("--panel-muted").trim();
	const panelBg = cs.getPropertyValue("--panel-bg").trim();

	cy = window.cytoscape({
		container: dom.sceneGraphCanvas,
		elements: buildElements(rels, memory || {}),
		style: [
			{
				selector: "node",
				style: {
					"background-color": "data(color)",
					label: "data(label)",
					color: panelText,
					"text-valign": "center",
					"text-halign": "center",
					"font-size": "10px",
				},
			},
			{
				selector: "edge",
				style: {
					"curve-style": "bezier",
					"target-arrow-shape": "triangle",
					"line-color": panelMuted,
					"target-arrow-color": panelMuted,
					label: "data(label)",
					"font-size": "9px",
					"text-background-color": panelBg,
					"text-background-opacity": 1,
					"text-background-padding": "2px",
					color: panelText,
				},
			},
		],
		layout: { name: "cose", fit: true, padding: 10 },
	});
}

export function initSceneGraphPanel() {
	if (dom.sgZoomIn) {
		dom.sgZoomIn.addEventListener("click", () => {
			if (!cy) return;
			cy.zoom({
				level: cy.zoom() * 1.2,
				renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
			});
		});
	}
	if (dom.sgZoomOut) {
		dom.sgZoomOut.addEventListener("click", () => {
			if (!cy) return;
			cy.zoom({
				level: cy.zoom() * 0.8,
				renderedPosition: { x: cy.width() / 2, y: cy.height() / 2 },
			});
		});
	}
	if (dom.sgFit) {
		dom.sgFit.addEventListener("click", () => {
			if (cy) cy.fit(undefined, 20);
		});
	}
}
