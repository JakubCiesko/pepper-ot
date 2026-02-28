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
const inferenceMetricsContainer = document.getElementById("inference-metrics");
const memoryEditorStatus = document.getElementById("memory-editor-status");
const memObjId = document.getElementById("mem-obj-id");
const memObjLabel = document.getElementById("mem-obj-label");
const memObjBbox = document.getElementById("mem-obj-bbox");
const memObjAttrs = document.getElementById("mem-obj-attrs");
const memObjCreate = document.getElementById("mem-obj-create");
const memObjUpdate = document.getElementById("mem-obj-update");
const memObjDelete = document.getElementById("mem-obj-delete");
const memRelSubject = document.getElementById("mem-rel-subject");
const memRelObject = document.getElementById("mem-rel-object");
const memRelPredicate = document.getElementById("mem-rel-predicate");
const memRelCount = document.getElementById("mem-rel-count");
const memRelCreate = document.getElementById("mem-rel-create");
const memRelUpdate = document.getElementById("mem-rel-update");
const memRelDelete = document.getElementById("mem-rel-delete");

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

function showMemoryEditorStatus(message, ok = true) {
    if (!memoryEditorStatus) return;
    memoryEditorStatus.textContent = message;
    memoryEditorStatus.classList.toggle("text-red-500", !ok);
}

function parseBboxCsv(value) {
    const parts = String(value || "")
        .split(",")
        .map(v => Number(v.trim()))
        .filter(v => Number.isFinite(v));
    return parts.length === 4 ? parts : null;
}

function parseCommaList(value) {
    return String(value || "")
        .split(",")
        .map(v => v.trim())
        .filter(Boolean);
}

async function refreshMemoryFromApi() {
    const res = await fetch("/api/v1/memory");
    if (!res.ok) {
        throw new Error("Failed to refresh memory");
    }
    const mem = await res.json();
    renderMemory(mem);
}

function renderMemory(mem) {
    if (!memoryContainer) return;
    memoryContainer.innerHTML = "";
    const memObjects = mem.objects || [];
    const memRelations = mem.relationships || [];
    if (memObjects.length > 0) {
        memObjects.forEach(obj => {
            const div = document.createElement("div");
            div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950";
            const attrs = (obj.attributes && obj.attributes.length > 0)
                ? obj.attributes.join(", ")
                : "no attributes";
            const bbox = obj.bbox ? `[${obj.bbox.map(n => n.toFixed(1)).join(", ")}]` : "n/a";
            div.innerHTML = `
                <div><strong>${obj.label} (#${obj.id})</strong></div>
                <div class="text-xs text-slate-400 mt-1">hits ${obj.hits} · first ${secondsAgo(obj.first_seen)} · last ${secondsAgo(obj.last_seen)}</div>
                <div class="text-xs text-slate-300 mt-1">attrs: ${attrs}</div>
                <div class="text-xs text-slate-300 mt-1">bbox: ${bbox}</div>
            `;
            memoryContainer.appendChild(div);
        });
    } else {
        memoryContainer.innerHTML = `<p class="text-slate-500">No tracked objects yet...</p>`;
    }
    const relHeader = document.createElement("div");
    relHeader.className = "mt-4 mb-2 text-xs uppercase tracking-widest panel-muted";
    relHeader.textContent = "Relations";
    memoryContainer.appendChild(relHeader);
    if (memRelations.length > 0) {
        memRelations.forEach(rel => {
            const div = document.createElement("div");
            div.className = "mb-2 p-2 rounded border border-slate-800 bg-slate-950 text-xs";
            div.textContent = `${rel.subject_id} ${rel.predicate} ${rel.object_id} (count: ${rel.count})`;
            memoryContainer.appendChild(div);
        });
    } else {
        const emptyRel = document.createElement("p");
        emptyRel.className = "text-slate-500 text-xs";
        emptyRel.textContent = "No relationships in memory yet.";
        memoryContainer.appendChild(emptyRel);
    }

    if (memoryRaw) {
        memoryRaw.textContent = JSON.stringify(mem, null, 2);
    }
    const summaryMemory = document.getElementById("summary-memory");
    if (summaryMemory) summaryMemory.textContent = mem.objects ? mem.objects.length : "—";
}

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "sentence") {
        displayPepperSentence(data.text);
        return;
    }
    if (data.type === "memory") {
        renderMemory(data.memory || {});
        return;
    }

    // Clear previous data
    detectionsContainer.innerHTML = "";
    if (memoryContainer) memoryContainer.innerHTML = "";
    if (sceneGraphContainer) sceneGraphContainer.innerHTML = "";
    if (inferenceMetricsContainer) inferenceMetricsContainer.innerHTML = "";
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

        const objectId = obj.object_id !== undefined && obj.object_id !== null ? `#${obj.object_id}` : "n/a";
        const conf = Number.isFinite(obj.confidence) ? obj.confidence.toFixed(2) : "n/a";
        const bbox = Array.isArray(obj.bbox) ? obj.bbox.map(x => Number(x).toFixed(1)).join(", ") : "n/a";
        div.innerHTML = `<strong>${obj.label}</strong> <span class="text-xs">(${objectId})</span> (${conf}) - bbox: [${bbox}]`;
        detectionsContainer.appendChild(div);
    });
    } else {
        detectionsContainer.innerHTML = `<p class="text-slate-500">No objects detected</p>`;
    }

    if (inferenceMetricsContainer) {
        const metrics = data.metrics || {};
        const keys = Object.keys(metrics);
        if (keys.length > 0) {
            keys.sort();
            keys.forEach(key => {
                const row = document.createElement("div");
                row.className = "flex items-center justify-between text-sm mb-1";
                const value = metrics[key];
                const formatted = Number.isFinite(value) ? `${value.toFixed(4)} s` : String(value);
                row.innerHTML = `<span class="panel-muted">${key}</span><span>${formatted}</span>`;
                inferenceMetricsContainer.appendChild(row);
            });
        } else {
            inferenceMetricsContainer.innerHTML = `<p class="panel-muted">No metrics recorded yet...</p>`;
        }
    }
    // image
    if (data.image) {
        annotatedImage.src = `data:image/jpeg;base64,${data.image}`;
        //annotatedImage.classList.remove("hidden");
    }
    //else {
    //    annotatedImage.classList.add("hidden");
    //}
    renderMemory(data.memory || {});

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
                                "font-size": "10px"
                            }
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
                                "color": panelText
                            }
                        }
                    ],
                    layout: { name: "cose", fit: true, padding: 10 }
                });
            }
        }
    }

    const summaryObjects = document.getElementById("summary-objects");
    const summaryMemory = document.getElementById("summary-memory");
    const summaryRelations = document.getElementById("summary-relations");
    if (summaryObjects) summaryObjects.textContent = data.objects ? data.objects.length : "—";
    if (summaryMemory) summaryMemory.textContent = data.memory?.objects ? data.memory.objects.length : "—";
    if (summaryRelations) summaryRelations.textContent = data.scene_graph ? data.scene_graph.length : "—";
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

async function doMemoryRequest(url, options = {}) {
    const res = await fetch(url, options);
    if (!res.ok) {
        let detail = "Request failed";
        try {
            const body = await res.json();
            detail = body.detail || detail;
        } catch {
            // ignore
        }
        throw new Error(detail);
    }
    try {
        return await res.json();
    } catch {
        return { ok: true };
    }
}

if (memObjCreate) {
    memObjCreate.addEventListener("click", async () => {
        try {
            const bbox = parseBboxCsv(memObjBbox.value);
            if (!memObjLabel.value.trim()) throw new Error("Object label is required");
            if (!bbox) throw new Error("bbox must be x1,y1,x2,y2");
            const payload = {
                label: memObjLabel.value.trim(),
                bbox,
                attributes: parseCommaList(memObjAttrs.value)
            };
            const parsedId = Number(memObjId.value);
            if (Number.isInteger(parsedId) && parsedId > 0) payload.id = parsedId;
            await doMemoryRequest("/api/v1/memory/object", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            await refreshMemoryFromApi();
            showMemoryEditorStatus("Object created");
        } catch (err) {
            showMemoryEditorStatus(err.message || "Failed to create object", false);
        }
    });
}

if (memObjUpdate) {
    memObjUpdate.addEventListener("click", async () => {
        try {
            const objectId = Number(memObjId.value);
            if (!Number.isInteger(objectId) || objectId <= 0) {
                throw new Error("Valid object id is required for update");
            }
            const payload = {};
            if (memObjLabel.value.trim()) payload.label = memObjLabel.value.trim();
            if (memObjBbox.value.trim()) {
                const bbox = parseBboxCsv(memObjBbox.value);
                if (!bbox) throw new Error("bbox must be x1,y1,x2,y2");
                payload.bbox = bbox;
            }
            if (memObjAttrs.value.trim()) payload.attributes = parseCommaList(memObjAttrs.value);
            if (Object.keys(payload).length === 0) throw new Error("No update fields set");
            await doMemoryRequest(`/api/v1/memory/object/${objectId}`, {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            await refreshMemoryFromApi();
            showMemoryEditorStatus("Object updated");
        } catch (err) {
            showMemoryEditorStatus(err.message || "Failed to update object", false);
        }
    });
}

if (memObjDelete) {
    memObjDelete.addEventListener("click", async () => {
        try {
            const objectId = Number(memObjId.value);
            if (!Number.isInteger(objectId) || objectId <= 0) {
                throw new Error("Valid object id is required for delete");
            }
            await doMemoryRequest(`/api/v1/memory/object/${objectId}?cascade_relations=true`, {
                method: "DELETE"
            });
            await refreshMemoryFromApi();
            showMemoryEditorStatus("Object deleted");
        } catch (err) {
            showMemoryEditorStatus(err.message || "Failed to delete object", false);
        }
    });
}

if (memRelCreate) {
    memRelCreate.addEventListener("click", async () => {
        try {
            const subjectId = Number(memRelSubject.value);
            const objectId = Number(memRelObject.value);
            const predicate = memRelPredicate.value.trim();
            if (!Number.isInteger(subjectId) || subjectId <= 0) throw new Error("Valid subject id required");
            if (!Number.isInteger(objectId) || objectId <= 0) throw new Error("Valid object id required");
            if (!predicate) throw new Error("Predicate is required");
            const payload = { subject_id: subjectId, object_id: objectId, predicate };
            const count = Number(memRelCount.value);
            if (Number.isInteger(count) && count > 0) payload.count = count;
            await doMemoryRequest("/api/v1/memory/relation", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            await refreshMemoryFromApi();
            showMemoryEditorStatus("Relation created");
        } catch (err) {
            showMemoryEditorStatus(err.message || "Failed to create relation", false);
        }
    });
}

if (memRelUpdate) {
    memRelUpdate.addEventListener("click", async () => {
        try {
            const subjectId = Number(memRelSubject.value);
            const objectId = Number(memRelObject.value);
            const predicate = memRelPredicate.value.trim();
            if (!Number.isInteger(subjectId) || subjectId <= 0) throw new Error("Valid subject id required");
            if (!Number.isInteger(objectId) || objectId <= 0) throw new Error("Valid object id required");
            if (!predicate) throw new Error("Predicate is required");
            const payload = { subject_id: subjectId, object_id: objectId, predicate };
            const count = Number(memRelCount.value);
            if (Number.isInteger(count) && count > 0) payload.count = count;
            if (!payload.count) throw new Error("Set count to update relation");
            await doMemoryRequest("/api/v1/memory/relation", {
                method: "PATCH",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            await refreshMemoryFromApi();
            showMemoryEditorStatus("Relation updated");
        } catch (err) {
            showMemoryEditorStatus(err.message || "Failed to update relation", false);
        }
    });
}

if (memRelDelete) {
    memRelDelete.addEventListener("click", async () => {
        try {
            const subjectId = Number(memRelSubject.value);
            const objectId = Number(memRelObject.value);
            const predicate = memRelPredicate.value.trim();
            if (!Number.isInteger(subjectId) || subjectId <= 0) throw new Error("Valid subject id required");
            if (!Number.isInteger(objectId) || objectId <= 0) throw new Error("Valid object id required");
            if (!predicate) throw new Error("Predicate is required");
            const qs = new URLSearchParams({
                subject_id: String(subjectId),
                predicate,
                object_id: String(objectId)
            });
            await doMemoryRequest(`/api/v1/memory/relation?${qs.toString()}`, {
                method: "DELETE"
            });
            await refreshMemoryFromApi();
            showMemoryEditorStatus("Relation deleted");
        } catch (err) {
            showMemoryEditorStatus(err.message || "Failed to delete relation", false);
        }
    });
}

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
