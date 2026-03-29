const ws = new WebSocket(`wss://${location.host}/dashboard/events`);

const detectionsContainer = document.getElementById("detections-content");
const annotatedImage = document.getElementById("annotated-image");
const inferenceMetricsContainer = document.getElementById("inference-metrics");
const captionContainer = document.getElementById("caption-content");

const liveCarouselTrack = document.getElementById("live-carousel-track");
const liveCarouselPrev = document.getElementById("live-carousel-prev");
const liveCarouselNext = document.getElementById("live-carousel-next");
const liveCarouselIndex = document.getElementById("live-carousel-index");

const sgCarouselTrack = document.getElementById("sg-carousel-track");
const sgCarouselPrev = document.getElementById("sg-carousel-prev");
const sgCarouselNext = document.getElementById("sg-carousel-next");
const sgCarouselIndex = document.getElementById("sg-carousel-index");
const processImageInput = document.getElementById("process-image-input");
const processImageBtn = document.getElementById("process-image-btn");
const processImageName = document.getElementById("process-image-name");

const MAX_LIVE_FRAMES = 10;
const liveFrames = [];
let activeLiveFrameIndex = -1;

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

function formatFrameTimestamp(ts) {
    if (!Number.isFinite(ts)) return "Unknown time";
    return new Date(ts * 1000).toLocaleTimeString();
}

function updateCarouselIndexLabels() {
    const text = liveFrames.length === 0 || activeLiveFrameIndex < 0
        ? "0 / 0"
        : `${activeLiveFrameIndex + 1} / ${liveFrames.length}`;
    if (liveCarouselIndex) liveCarouselIndex.textContent = text;
    if (sgCarouselIndex) sgCarouselIndex.textContent = text;
}

function summarizeSceneGraph(sceneGraph) {
    const rels = Array.isArray(sceneGraph) ? sceneGraph : [];
    if (rels.length === 0) return "No relations";
    const sample = rels
        .slice(0, 2)
        .map(edge => `${edge.sub} ${edge.rel} ${edge.obj}`)
        .join(" | ");
    return rels.length > 2 ? `${sample} ...` : sample;
}

function renderLiveCarousel() {
    if (!liveCarouselTrack) return;
    liveCarouselTrack.innerHTML = "";
    liveFrames.forEach((frame, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `live-carousel-item ${idx === activeLiveFrameIndex ? "active" : ""}`;
        btn.setAttribute("aria-label", `Frame ${idx + 1}`);
        btn.addEventListener("click", () => setActiveLiveFrame(idx));

        const img = document.createElement("img");
        img.className = "live-carousel-thumb";
        img.src = `data:image/jpeg;base64,${frame.image}`;
        img.alt = `Recent frame ${idx + 1}`;
        btn.appendChild(img);

        const meta = document.createElement("div");
        meta.className = "live-carousel-meta";
        const hasCaption =
            typeof frame.caption === "string" && frame.caption.trim().length > 0;
        meta.textContent = `${formatFrameTimestamp(frame.timestamp)} • caption ${hasCaption ? "yes" : "no"}`;
        btn.appendChild(meta);

        liveCarouselTrack.appendChild(btn);
    });
}

function renderSceneGraphCarousel() {
    if (!sgCarouselTrack) return;
    sgCarouselTrack.innerHTML = "";
    liveFrames.forEach((frame, idx) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `sg-carousel-item ${idx === activeLiveFrameIndex ? "active" : ""}`;
        btn.setAttribute("aria-label", `Scene graph ${idx + 1}`);
        btn.addEventListener("click", () => setActiveLiveFrame(idx));

        const title = document.createElement("div");
        title.className = "sg-carousel-title";
        title.textContent = `${Array.isArray(frame.scene_graph) ? frame.scene_graph.length : 0} relations`;
        btn.appendChild(title);

        const meta = document.createElement("div");
        meta.className = "sg-carousel-meta";
        meta.textContent = formatFrameTimestamp(frame.timestamp);
        btn.appendChild(meta);

        const preview = document.createElement("div");
        preview.className = "sg-carousel-preview";
        preview.textContent = summarizeSceneGraph(frame.scene_graph);
        btn.appendChild(preview);

        sgCarouselTrack.appendChild(btn);
    });
}

function renderActiveFrameSnapshot() {
    if (activeLiveFrameIndex < 0 || activeLiveFrameIndex >= liveFrames.length) return;
    const frame = liveFrames[activeLiveFrameIndex];

    if (annotatedImage && frame.image) {
        annotatedImage.src = `data:image/jpeg;base64,${frame.image}`;
    }
    renderCaption(frame.caption || "No caption for this frame.");
    renderDetections(frame.objects, frame.colors);
    renderMetrics(frame.metrics);

    if (window.PepperMemoryPanel) {
        window.PepperMemoryPanel.renderMemory(frame.memory || {});
    }
    if (window.PepperSceneGraphPanel) {
        window.PepperSceneGraphPanel.renderSceneGraph(
            frame.scene_graph || [],
            frame.memory || {}
        );
    }
    updateSummaries(frame);
}

function setActiveLiveFrame(index) {
    if (!Number.isInteger(index)) return;
    if (index < 0 || index >= liveFrames.length) return;
    activeLiveFrameIndex = index;
    renderActiveFrameSnapshot();
    renderLiveCarousel();
    renderSceneGraphCarousel();
    updateCarouselIndexLabels();
}

function snapshotFromPayload(data) {
    return {
        id: data.id || `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
        timestamp: Number.isFinite(data.timestamp) ? data.timestamp : Date.now() / 1000,
        image: data.image || "",
        objects: Array.isArray(data.objects) ? data.objects : [],
        colors: data.colors && typeof data.colors === "object" ? data.colors : {},
        metrics: data.metrics && typeof data.metrics === "object" ? data.metrics : {},
        memory: data.memory && typeof data.memory === "object" ? data.memory : {},
        scene_graph: Array.isArray(data.scene_graph) ? data.scene_graph : [],
        caption: typeof data.caption === "string" ? data.caption : "",
        caption_provider:
            typeof data.caption_provider === "string" ? data.caption_provider : "",
        caption_model_id:
            typeof data.caption_model_id === "string" ? data.caption_model_id : "",
    };
}

function isSameAsLatest(frame) {
    if (liveFrames.length === 0) return false;
    const latest = liveFrames[0];
    if (frame.id && latest.id && frame.id === latest.id) return true;
    return latest.image === frame.image && latest.timestamp === frame.timestamp;
}

function pushFrameSnapshot(data) {
    const frame = snapshotFromPayload(data);
    if (!frame.image) return;

    if (isSameAsLatest(frame)) {
        liveFrames[0] = frame;
        setActiveLiveFrame(0);
        return;
    }

    liveFrames.unshift(frame);
    if (liveFrames.length > MAX_LIVE_FRAMES) {
        liveFrames.length = MAX_LIVE_FRAMES;
    }
    setActiveLiveFrame(0);
}

function renderDetectionPayload(data) {
    pushFrameSnapshot(data);
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

function goToPrevFrame() {
    if (liveFrames.length === 0) return;
    const next = activeLiveFrameIndex <= 0
        ? liveFrames.length - 1
        : activeLiveFrameIndex - 1;
    setActiveLiveFrame(next);
}

function goToNextFrame() {
    if (liveFrames.length === 0) return;
    const next = activeLiveFrameIndex >= liveFrames.length - 1
        ? 0
        : activeLiveFrameIndex + 1;
    setActiveLiveFrame(next);
}

function getActiveFrameSnapshot() {
    if (activeLiveFrameIndex < 0 || activeLiveFrameIndex >= liveFrames.length) {
        return null;
    }
    const frame = liveFrames[activeLiveFrameIndex];
    return frame ? { ...frame } : null;
}

window.PepperLiveFeed = {
    getActiveFrameSnapshot,
};

async function processUploadedImage() {
    if (!processImageInput || !processImageBtn) return;
    const file = processImageInput.files?.[0];
    if (!file) {
        if (typeof showStatusMessage === "function") {
            showStatusMessage("Select an image first.", false);
        }
        return;
    }

    processImageBtn.disabled = true;
    const originalText = processImageBtn.textContent;
    processImageBtn.textContent = "Processing...";
    try {
        const form = new FormData();
        form.append("file", file);
        form.append("publish", "true");
        const res = await fetch("/api/v1/detect", {
            method: "POST",
            body: form,
        });
        if (!res.ok) {
            let detail = "Failed to process image";
            try {
                const body = await res.json();
                detail = body.detail || detail;
            } catch {
                // ignore
            }
            throw new Error(detail);
        }
        if (typeof showStatusMessage === "function") {
            showStatusMessage("Image submitted to /api/v1/detect");
        }
        processImageInput.value = "";
        if (processImageName) processImageName.textContent = "";
    } catch (err) {
        if (typeof showStatusMessage === "function") {
            showStatusMessage(err.message || "Failed to process image", false);
        }
    } finally {
        processImageBtn.disabled = false;
        processImageBtn.textContent = originalText;
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

if (liveCarouselPrev) {
    liveCarouselPrev.addEventListener("click", goToPrevFrame);
}
if (liveCarouselNext) {
    liveCarouselNext.addEventListener("click", goToNextFrame);
}
if (sgCarouselPrev) {
    sgCarouselPrev.addEventListener("click", goToPrevFrame);
}
if (sgCarouselNext) {
    sgCarouselNext.addEventListener("click", goToNextFrame);
}
if (processImageInput) {
    processImageInput.addEventListener("change", () => {
        const file = processImageInput.files?.[0];
        if (processImageName) {
            processImageName.textContent = file ? file.name : "";
        }
    });
}
if (processImageBtn) {
    processImageBtn.addEventListener("click", processUploadedImage);
}
updateCarouselIndexLabels();

loadLastState();
if (window.PepperConversationPanel) {
    window.PepperConversationPanel.loadLatestConversation();
}
