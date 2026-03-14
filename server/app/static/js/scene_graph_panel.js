(function () {
    const sceneGraphContainer = document.getElementById("scene-graph-content");
    const sceneGraphRaw = document.getElementById("scene-graph-raw");
    const sceneGraphCanvas = document.getElementById("scene-graph-canvas");
    const sgZoomIn = document.getElementById("sg-zoom-in");
    const sgZoomOut = document.getElementById("sg-zoom-out");
    const sgFit = document.getElementById("sg-fit");

    function labelColor(label) {
        if (!label) return "#22c55e";
        let hash = 0;
        for (let i = 0; i < label.length; i++) {
            hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
        }
        const hue = hash % 360;
        return `hsl(${hue}, 70%, 55%)`;
    }

    function renderSceneGraph(edges, memory) {
        if (!sceneGraphContainer) return;
        const rels = Array.isArray(edges) ? edges : [];

        sceneGraphContainer.innerHTML = "";
        if (rels.length > 0) {
            rels.forEach(edge => {
                const div = document.createElement("div");
                div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
                div.textContent = `${edge.sub} ${edge.rel} ${edge.obj}`;
                sceneGraphContainer.appendChild(div);
            });
        } else {
            sceneGraphContainer.innerHTML = `<p class="text-slate-500">No relations yet...</p>`;
        }

        if (sceneGraphRaw) {
            sceneGraphRaw.textContent = JSON.stringify(rels, null, 2);
        }
        if (!sceneGraphCanvas || !window.cytoscape) return;

        const mem = memory || {};
        const memObjects = mem.objects || [];
        const idToLabel = {};
        memObjects.forEach(obj => {
            idToLabel[String(obj.id)] = obj.label;
        });

        const nodeIds = new Set();
        rels.forEach(e => {
            nodeIds.add(String(e.sub));
            nodeIds.add(String(e.obj));
        });
        const elements = [
            ...Array.from(nodeIds).map(id => {
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

        if (window._sceneGraphCy) {
            window._sceneGraphCy.destroy();
        }
        const cs = getComputedStyle(document.body);
        const panelText = cs.getPropertyValue("--panel-text").trim();
        const panelMuted = cs.getPropertyValue("--panel-muted").trim();
        const panelBg = cs.getPropertyValue("--panel-bg").trim();
        window._sceneGraphCy = window.cytoscape({
            container: sceneGraphCanvas,
            elements,
            style: [
                {
                    selector: "node",
                    style: {
                        "background-color": "data(color)",
                        "label": "data(label)",
                        "color": panelText,
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
                        "label": "data(label)",
                        "font-size": "9px",
                        "text-background-color": panelBg,
                        "text-background-opacity": 1,
                        "text-background-padding": "2px",
                        "color": panelText,
                    },
                },
            ],
            layout: { name: "cose", fit: true, padding: 10 },
        });
    }

    function init() {
        if (sgZoomIn) {
            sgZoomIn.addEventListener("click", () => {
                if (window._sceneGraphCy) {
                    window._sceneGraphCy.zoom({
                        level: window._sceneGraphCy.zoom() * 1.2,
                        renderedPosition: {
                            x: window._sceneGraphCy.width() / 2,
                            y: window._sceneGraphCy.height() / 2,
                        },
                    });
                }
            });
        }

        if (sgZoomOut) {
            sgZoomOut.addEventListener("click", () => {
                if (window._sceneGraphCy) {
                    window._sceneGraphCy.zoom({
                        level: window._sceneGraphCy.zoom() * 0.8,
                        renderedPosition: {
                            x: window._sceneGraphCy.width() / 2,
                            y: window._sceneGraphCy.height() / 2,
                        },
                    });
                }
            });
        }

        if (sgFit) {
            sgFit.addEventListener("click", () => {
                if (window._sceneGraphCy) {
                    window._sceneGraphCy.fit(undefined, 20);
                }
            });
        }
    }

    window.PepperSceneGraphPanel = {
        init,
        renderSceneGraph,
    };
})();
