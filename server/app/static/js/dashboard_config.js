const slider = document.getElementById("threshold-slider");
const input = document.getElementById("threshold-input");
const backendSelect = document.getElementById("backend-select");
const languageSelect = document.getElementById("language-select");

const visBbox = document.getElementById("vis-bbox");
const visMask = document.getElementById("vis-mask");
const visPolygon = document.getElementById("vis-polygon");
const visLabels = document.getElementById("vis-labels");
const visLine = document.getElementById("vis-line");
const visOpacity = document.getElementById("vis-opacity");
const visColor = document.getElementById("vis-color");
const storagePersist = document.getElementById("storage-persist");
const storageImage = document.getElementById("storage-image");
const storagePath = document.getElementById("storage-path");

const vlmSystem = document.getElementById("vlm-system");
const vlmUser = document.getElementById("vlm-user");
const vlmPredicates = document.getElementById("vlm-predicates");
const vlmObjects = document.getElementById("vlm-objects");

const chatSystem = document.getElementById("chat-system");
const chatContext = document.getElementById("chat-context");

const applyBtn = document.getElementById("apply-config");
const saveBtn = document.getElementById("save-config");
const reloadBtn = document.getElementById("reload-config");
const reloadAllBtn = document.getElementById("reload-all");
const downloadBtn = document.getElementById("download-config");
const downloadSavedBtn = document.getElementById("download-config-saved");
const uploadInput = document.getElementById("upload-config");
const reloadWarning = document.getElementById("reload-warning");
const uploadProgress = document.getElementById("upload-progress");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const uploadProgressText = document.getElementById("upload-progress-text");

async function fetchModels() {
    try {
        const res = await fetch("/dashboard/config/get_models");
        const data = await res.json();
        backendSelect.innerHTML = "";
        data.models.forEach(model => {
            const opt = document.createElement("option");
            opt.value = model;
            opt.textContent = model;
            backendSelect.appendChild(opt);
        });
    } catch (err) {
        console.error("Failed to fetch models:", err);
        backendSelect.innerHTML = "<option>Error loading</option>";
    }
}

function setThreshold(value) {
    slider.value = value;
    input.value = value;
}

function parseLines(text) {
    return text
        .split("\n")
        .map(l => l.trim())
        .filter(Boolean);
}

function parseObjects(text) {
    const obj = {};
    parseLines(text).forEach(line => {
        const idx = line.indexOf(":");
        if (idx === -1) return;
        const key = line.slice(0, idx).trim();
        const val = line.slice(idx + 1).trim();
        if (key && val) obj[key] = val;
    });
    return obj;
}

function objectsToText(objects) {
    if (!objects) return "";
    return Object.entries(objects).map(([k, v]) => `${k}: ${v}`).join("\n");
}

async function loadConfig() {
    const res = await fetch("/api/v1/config");
    const data = await res.json();
    const active = data.active;
    const resolved = data.active_resolved || {};

    setThreshold(active.detection.confidence_threshold ?? 0.5);
    backendSelect.value = active.detection.backend || "rt_detr";
    languageSelect.value = active.system.language || "en";

    visBbox.checked = !!active.visualization.show_bbox;
    visMask.checked = !!active.visualization.show_mask;
    visPolygon.checked = !!active.visualization.show_polygon;
    visLabels.checked = !!active.visualization.show_labels;
    visLine.value = active.visualization.line_thickness ?? 2;
    visOpacity.value = active.visualization.mask_opacity ?? 0.5;
    visColor.value = active.visualization.color_lookup || "index";

    storagePersist.checked = !!active.storage?.persist_last_state;
    storageImage.checked = !!active.storage?.store_image;
    storagePath.value = active.storage?.last_state_path || "state/last_state.json";

    const resolvedUnderstanding = resolved.understanding || {};
    vlmSystem.value = resolvedUnderstanding.resolved_system_prompt || "";
    vlmUser.value = resolvedUnderstanding.resolved_user_prompt || "";
    vlmPredicates.value = (resolvedUnderstanding.resolved_ontology?.predicates || []).join("\n");
    vlmObjects.value = objectsToText(resolvedUnderstanding.resolved_ontology?.objects);

    const resolvedChat = resolved.chat || {};
    chatSystem.value = resolvedChat.resolved_system_prompt || "";
    chatContext.value = resolvedChat.resolved_context_template || "";
}

function buildPatch() {
    return {
        system: {
            language: languageSelect.value
        },
        detection: {
            backend: backendSelect.value,
            confidence_threshold: parseFloat(input.value)
        },
        visualization: {
            show_bbox: visBbox.checked,
            show_mask: visMask.checked,
            show_polygon: visPolygon.checked,
            show_labels: visLabels.checked,
            line_thickness: parseInt(visLine.value, 10),
            mask_opacity: parseFloat(visOpacity.value),
            color_lookup: visColor.value
        },
        storage: {
            persist_last_state: storagePersist.checked,
            store_image: storageImage.checked,
            last_state_path: storagePath.value
        },
        understanding: {
            system_prompt: { text: vlmSystem.value },
            user_prompt: { text: vlmUser.value },
            ontology: {
                predicates: parseLines(vlmPredicates.value),
                objects: parseObjects(vlmObjects.value)
            }
        },
        chat: {
            system_prompt: { text: chatSystem.value },
            context_template: { text: chatContext.value }
        }
    };
}

async function applyConfig() {
    const patch = buildPatch();
    const res = await fetch("/api/v1/config", {
        method: "PATCH",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(patch)
    });
    if (res.ok) {
        const data = await res.json();
        showStatusMessage(data.reloaded ? "Applied (hard reload)" : "Applied hot config");
        if (data.requires_reload && data.requires_reload.length > 0) {
            reloadWarning.textContent = `Hard changes detected: ${data.requires_reload.join(", ")}`;
            reloadWarning.classList.remove("hidden");
        } else {
            reloadWarning.classList.add("hidden");
        }
    } else {
        showStatusMessage("Failed to apply config", false);
    }
}

async function saveConfig() {
    const res = await fetch("/api/v1/config/save", { method: "POST" });
    if (res.ok) {
        showStatusMessage("Saved to server/config.yaml");
    } else {
        showStatusMessage("Failed to save config", false);
    }
}

async function reloadConfig() {
    const res = await fetch("/api/v1/config/reload", { method: "POST" });
    if (res.ok) {
        await loadConfig();
        showStatusMessage("Reloaded saved config");
    } else {
        showStatusMessage("Failed to reload config", false);
    }
}

function downloadConfig(source) {
    const url = source === "saved"
        ? "/api/v1/config/download?source=saved"
        : "/api/v1/config/download";
    window.location.href = url;
}

async function uploadConfig(file) {
    uploadProgress.classList.remove("hidden");
    uploadProgressBar.style.width = "0%";
    uploadProgressText.textContent = "0%";

    const form = new FormData();
    form.append("file", file);

    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/v1/config/upload", true);

    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            uploadProgressBar.style.width = `${percent}%`;
            uploadProgressText.textContent = `${percent}%`;
        }
    };

    xhr.onreadystatechange = async () => {
        if (xhr.readyState === 4) {
            if (xhr.status >= 200 && xhr.status < 300) {
                await loadConfig();
                showStatusMessage("Uploaded config applied (in-memory)");
            } else {
                showStatusMessage("Failed to upload config", false);
            }
            setTimeout(() => uploadProgress.classList.add("hidden"), 500);
        }
    };

    xhr.send(form);
}

slider.addEventListener("input", () => {
    input.value = slider.value;
});

input.addEventListener("change", () => {
    let val = parseFloat(input.value);
    if (isNaN(val)) val = 0;
    if (val < 0) val = 0;
    if (val > 1) val = 1;
    input.value = val.toFixed(2);
    slider.value = val.toFixed(2);
});

applyBtn.addEventListener("click", applyConfig);
saveBtn.addEventListener("click", saveConfig);
reloadBtn.addEventListener("click", reloadConfig);
reloadAllBtn.addEventListener("click", async () => {
    await applyConfig();
    await reloadConfig();
});
downloadBtn.addEventListener("click", () => downloadConfig("active"));
downloadSavedBtn.addEventListener("click", () => downloadConfig("saved"));

uploadInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) uploadConfig(file);
});

fetchModels();
loadConfig();
