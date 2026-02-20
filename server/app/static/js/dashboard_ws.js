const ws = new WebSocket(`ws://${location.host}/dashboard/events`);

const detectionsContainer = document.getElementById("detections-content");
const annotatedImage = document.getElementById("annotated-image");
const memoryContainer = document.getElementById("memory-content");
const memoryRaw = document.getElementById("memory-raw");
const sceneGraphContainer = document.getElementById("scene-graph-content");
const sceneGraphRaw = document.getElementById("scene-graph-raw");
const sceneGraphCanvas = document.getElementById("scene-graph-canvas");
const sgZoomIn = document.getElementById("sg-zoom-in");
const sgZoomOut = document.getElementById("sg-zoom-out");
const sgFit = document.getElementById("sg-fit");

function secondsAgo(ts) {
    if (!ts) return "n/a";
    const now = Date.now() / 1000;
    const delta = now - ts;
    return `${delta.toFixed(1)}s ago`;
}

function labelColor(label) {
    if (!label) return "#22c55e";
    let hash = 0;
    for (let i = 0; i < label.length; i++) {
        hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
    }
    const hue = hash % 360;
    return `hsl(${hue}, 70%, 55%)`;
}

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "sentence") {
        displayPepperSentence(data.text);
        return;
    }

    // Clear previous data
    detectionsContainer.innerHTML = "";
    if (memoryContainer) memoryContainer.innerHTML = "";
    if (sceneGraphContainer) sceneGraphContainer.innerHTML = "";
    if (data.objects && data.objects.length > 0) {
    data.objects.forEach(obj => {
        const div = document.createElement("div");
        div.className = "obj mb-2 p-2 rounded";

        // Use color from backend colors dict
        const labelColor = data.colors && data.colors[obj.label]
            ? `rgb(${data.colors[obj.label].join(",")})`
            : "#ddd";

        div.style.backgroundColor = labelColor;
        div.style.color = "black";

        div.innerHTML = `<strong>${obj.label}</strong> (${obj.confidence.toFixed(2)}) - bbox: [${obj.bbox.map(x => x.toFixed(1)).join(", ")}]`;
        detectionsContainer.appendChild(div);
    });
    } else {
        detectionsContainer.innerHTML = `<p class="text-slate-500">No objects detected</p>`;
    }
    // image
    if (data.image) {
        annotatedImage.src = `data:image/jpeg;base64,${data.image}`;
        //annotatedImage.classList.remove("hidden");
    }
    //else {
    //    annotatedImage.classList.add("hidden");
    //}
    if (memoryContainer) {
        const mem = data.memory || {};
        const memObjects = mem.objects || [];
        if (memObjects.length > 0) {
            memObjects.forEach(obj => {
                const div = document.createElement("div");
                div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
                const attrs = (obj.attributes && obj.attributes.length > 0)
                    ? obj.attributes.join(", ")
                    : "no attributes";
                const bbox = obj.bbox ? `[${obj.bbox.map(n => n.toFixed(1)).join(", ")}]` : "n/a";
                div.innerHTML = `
                    <div class="flex items-center justify-between">
                        <strong>${obj.label}</strong> <span class="text-xs text-slate-400">#${obj.id}</span>
                    </div>
                    <div class="text-xs text-slate-400 mt-1">hits ${obj.hits} · first ${secondsAgo(obj.first_seen)} · last ${secondsAgo(obj.last_seen)}</div>
                    <div class="text-xs text-slate-300 mt-1">attrs: ${attrs}</div>
                    <div class="text-xs text-slate-300 mt-1">bbox: ${bbox}</div>
                `;
                memoryContainer.appendChild(div);
            });
        } else {
            memoryContainer.innerHTML = `<p class="text-slate-500">No tracked objects yet...</p>`;
        }
        if (memoryRaw) {
            memoryRaw.textContent = JSON.stringify(mem, null, 2);
        }
    }

    if (sceneGraphContainer) {
        const edges = data.scene_graph || [];
        if (edges.length > 0) {
            edges.forEach(edge => {
                const div = document.createElement("div");
                div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
                div.textContent = `${edge.sub} ${edge.rel} ${edge.obj}`;
                sceneGraphContainer.appendChild(div);
            });
        } else {
            sceneGraphContainer.innerHTML = `<p class="text-slate-500">No relations yet...</p>`;
        }
        if (sceneGraphRaw) {
            sceneGraphRaw.textContent = JSON.stringify(edges, null, 2);
        }
        if (sceneGraphCanvas) {
            const mem = data.memory || {};
            const memObjects = mem.objects || [];
            const idToLabel = {};
            memObjects.forEach(obj => {
                idToLabel[String(obj.id)] = obj.label;
            });

            const nodeIds = new Set();
            edges.forEach(e => {
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
                            color: labelColor(lbl)
                        }
                    };
                }),
                ...edges.map((e, idx) => ({
                    data: {
                        id: `e${idx}-${e.sub}-${e.obj}-${e.rel}`,
                        source: String(e.sub),
                        target: String(e.obj),
                        label: e.rel
                    }
                }))
            ];
            if (window.cytoscape) {
                if (window._sceneGraphCy) {
                    window._sceneGraphCy.destroy();
                }
                window._sceneGraphCy = window.cytoscape({
                    container: sceneGraphCanvas,
                    elements,
                    style: [
                        {
                            selector: "node",
                            style: {
                                "background-color": "data(color)",
                                "label": "data(label)",
                                "color": "#0f172a",
                                "text-valign": "center",
                                "text-halign": "center",
                                "font-size": "10px"
                            }
                        },
                        {
                            selector: "edge",
                            style: {
                                "curve-style": "bezier",
                                "target-arrow-shape": "triangle",
                                "line-color": "#64748b",
                                "target-arrow-color": "#64748b",
                                "label": "data(label)",
                                "font-size": "9px",
                                "text-background-color": "#0f172a",
                                "text-background-opacity": 1,
                                "text-background-padding": "2px",
                                "color": "#e2e8f0"
                            }
                        }
                    ],
                    layout: { name: "cose", fit: true, padding: 10 }
                });
            }
        }
    }
};

async function loadLastState() {
    try {
        const res = await fetch("/api/v1/state");
        if (!res.ok) return;
        const data = await res.json();
        if (data && (data.image || data.objects?.length || data.scene_graph?.length)) {
            ws.onmessage({ data: JSON.stringify(data) });
        }
    } catch (err) {
        console.error("Failed to load last state", err);
    }
}

loadLastState();

if (sgZoomIn) {
    sgZoomIn.addEventListener("click", () => {
        if (window._sceneGraphCy) {
            window._sceneGraphCy.zoom({
                level: window._sceneGraphCy.zoom() * 1.2,
                renderedPosition: {
                    x: window._sceneGraphCy.width() / 2,
                    y: window._sceneGraphCy.height() / 2
                }
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
                    y: window._sceneGraphCy.height() / 2
                }
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
