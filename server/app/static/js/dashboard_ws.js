const ws = new WebSocket(`wss://${location.host}/dashboard/events`);

const detectionsContainer = document.getElementById("detections-content");
const annotatedImage = document.getElementById("annotated-image");
const inferenceMetricsContainer = document.getElementById("inference-metrics");
const captionContainer = document.getElementById("caption-content");

function renderCaption(text) {
    if (!text || !captionContainer) return;
    if (
        captionContainer.children.length === 1 &&
        captionContainer.children[0].textContent.includes("No caption")
    ) {
        captionContainer.innerHTML = "";
    }
    captionContainer.innerHTML = "";
    const div = document.createElement("div");
    div.className =
        "bg-slate-950 border border-slate-800 p-3 rounded shadow-sm text-slate-300 whitespace-pre-wrap break-all overflow-x-auto";
    div.textContent = text;
    captionContainer.appendChild(div);
}

function renderDetections(objects, colors) {
    if (!detectionsContainer) return;
    detectionsContainer.innerHTML = "";
    if (!Array.isArray(objects) || objects.length === 0) {
        detectionsContainer.innerHTML = `<p class="text-slate-500">No objects detected</p>`;
        return;
    }

    objects.forEach(obj => {
        const div = document.createElement("div");
        div.className = "obj mb-2 p-2 rounded";
        const labelColor = colors && colors[obj.label]
            ? `rgb(${colors[obj.label].join(",")})`
            : "#ddd";
        div.style.backgroundColor = labelColor;
        div.style.color = "black";
        const objectId = obj.object_id !== undefined && obj.object_id !== null ? `#${obj.object_id}` : "n/a";
        const conf = Number.isFinite(obj.confidence) ? obj.confidence.toFixed(2) : "n/a";
        const bbox = Array.isArray(obj.bbox) ? obj.bbox.map(x => Number(x).toFixed(1)).join(", ") : "n/a";
        div.innerHTML = `<strong>${obj.label}</strong> <span class="text-xs">(${objectId})</span> (${conf}) - bbox: [${bbox}]`;
        detectionsContainer.appendChild(div);
    });
}

function renderMetrics(metrics) {
    if (!inferenceMetricsContainer) return;
    inferenceMetricsContainer.innerHTML = "";
    const payload = metrics || {};
    const keys = Object.keys(payload);
    if (keys.length === 0) {
        inferenceMetricsContainer.innerHTML = `<p class="panel-muted">No metrics recorded yet...</p>`;
        return;
    }
    keys.sort();
    keys.forEach(key => {
        const row = document.createElement("div");
        row.className = "flex items-center justify-between text-sm mb-1";
        const value = payload[key];
        const formatted = Number.isFinite(value) ? `${value.toFixed(4)} s` : String(value);
        row.innerHTML = `<span class="panel-muted">${key}</span><span>${formatted}</span>`;
        inferenceMetricsContainer.appendChild(row);
    });
}

function updateSummaries(payload) {
    const summaryObjects = document.getElementById("summary-objects");
    const summaryMemory = document.getElementById("summary-memory");
    const summaryRelations = document.getElementById("summary-relations");
    if (summaryObjects) summaryObjects.textContent = payload.objects ? payload.objects.length : "—";
    if (summaryMemory) summaryMemory.textContent = payload.memory?.objects ? payload.memory.objects.length : "—";
    if (summaryRelations) summaryRelations.textContent = payload.scene_graph ? payload.scene_graph.length : "—";
}

function renderDetectionPayload(data) {
    renderDetections(data.objects, data.colors);
    renderMetrics(data.metrics);
    if (data.image && annotatedImage) {
        annotatedImage.src = `data:image/jpeg;base64,${data.image}`;
    }
    if (window.PepperMemoryPanel) {
        window.PepperMemoryPanel.renderMemory(data.memory || {});
    }
    if (window.PepperSceneGraphPanel) {
        window.PepperSceneGraphPanel.renderSceneGraph(
            data.scene_graph || [],
            data.memory || {}
        );
    }
    updateSummaries(data);
}

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);

    if (data.type === "chat_message") {
        if (window.PepperConversationPanel) {
            window.PepperConversationPanel.handleChatMessageEvent(data);
        }
        return;
    }
    if (data.type === "caption") {
        renderCaption(data.text);
        return;
    }
    if (data.type === "memory") {
        if (window.PepperMemoryPanel) {
            window.PepperMemoryPanel.renderMemory(data.memory || {});
        }
        return;
    }

    renderDetectionPayload(data);
};

async function loadLastState() {
    try {
        const res = await fetch("/api/v1/state");
        if (!res.ok) return;
        const data = await res.json();
        if (data && (data.image || data.objects?.length || data.scene_graph?.length)) {
            renderDetectionPayload(data);
        }
    } catch (err) {
        console.error("Failed to load last state", err);
    }
}

if (window.PepperMemoryPanel) {
    window.PepperMemoryPanel.init();
}
if (window.PepperSceneGraphPanel) {
    window.PepperSceneGraphPanel.init();
}
if (window.PepperConversationPanel) {
    window.PepperConversationPanel.init();
}

loadLastState();
if (window.PepperConversationPanel) {
    window.PepperConversationPanel.loadLatestConversation();
}
