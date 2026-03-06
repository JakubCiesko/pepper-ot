const slider = document.getElementById("threshold-slider");
const input = document.getElementById("threshold-input");
const backendSelect = document.getElementById("backend-select");
const detectionDevice = document.getElementById("detection-device");
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
const vlmDevice = document.getElementById("vlm-device");
const vlmProvider = document.getElementById("vlm-provider");
const vlmModelId = document.getElementById("vlm-model-id");
const vlmBaseUrl = document.getElementById("vlm-base-url");
const vlmApiKeyEnv = document.getElementById("vlm-api-key-env");
const vlmStructuredMode = document.getElementById("vlm-structured-strategy");
const vlmClientInitKwargs = document.getElementById("vlm-client-init-kwargs");
const vlmCallKwargs = document.getElementById("vlm-call-kwargs");
const sggMode = document.getElementById("sgg-mode");
const sggRulesJson = document.getElementById("sgg-rules-json");

const memoryMaxDormantFrames = document.getElementById("memory-max-dormant-frames");
const memoryAssocVisualWeight = document.getElementById("memory-assoc-visual-weight");
const memoryAssocGeometryWeight = document.getElementById("memory-assoc-geometry-weight");
const memoryAssocMatchThreshold = document.getElementById("memory-assoc-match-threshold");
const memoryFeatReidModel = document.getElementById("memory-feat-reid-model");
const memoryFeatDevice = document.getElementById("memory-feat-device");
const memoryFeatTargetSize = document.getElementById("memory-feat-target-size");
const memoryFeatResampling = document.getElementById("memory-feat-resampling");
const memoryMaxAgeSeconds = document.getElementById("memory-max-age-seconds");
const memoryMaxObjects = document.getElementById("memory-max-objects");
const memoryMaxRelations = document.getElementById("memory-max-relations");

const chatSystem = document.getElementById("chat-system");
const chatDevice = document.getElementById("chat-device");
const chatContext = document.getElementById("chat-context");
const chatProvider = document.getElementById("chat-provider");
const chatModelId = document.getElementById("chat-model-id");
const chatBaseUrl = document.getElementById("chat-base-url");
const chatApiKeyEnv = document.getElementById("chat-api-key-env");
const chatStructuredMode = document.getElementById("chat-structured-strategy");
const chatClientInitKwargs = document.getElementById("chat-client-init-kwargs");
const chatCallKwargs = document.getElementById("chat-call-kwargs");

const applyBtn = document.getElementById("apply-config");
const saveBtn = document.getElementById("save-config");
const reloadBtn = document.getElementById("reload-config");
const downloadBtn = document.getElementById("download-config");
const downloadSavedBtn = document.getElementById("download-config-saved");
const uploadInput = document.getElementById("upload-config");
const reloadWarning = document.getElementById("reload-warning");
const uploadProgress = document.getElementById("upload-progress");
const uploadProgressBar = document.getElementById("upload-progress-bar");
const uploadProgressText = document.getElementById("upload-progress-text");
let isApplyingConfig = false;
let lastApplyAt = 0;
const APPLY_DEBOUNCE_MS = 500;

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

function parseOntologyList(text) {
    return text
        .split(/[\n,]/)
        .map(v => v.trim())
        .filter(Boolean);
}

function parseTargetSize(text) {
    const parts = String(text || "")
        .split(",")
        .map(v => Number(v.trim()))
        .filter(v => Number.isFinite(v) && v > 0);
    return parts.length === 2 ? [Math.round(parts[0]), Math.round(parts[1])] : null;
}

function parseJsonObject(text, fieldName) {
    const value = String(text || "").trim();
    if (!value) return {};
    let parsed;
    try {
        parsed = JSON.parse(value);
    } catch (err) {
        throw new Error(`Invalid JSON in ${fieldName}`);
    }
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error(`${fieldName} must be a JSON object`);
    }
    return parsed;
}

async function loadConfig() {
    const res = await fetch("/api/v1/config");
    const data = await res.json();
    const active = data.active;
    const resolved = data.active_resolved || {};

    setThreshold(active.detection.confidence_threshold ?? 0.5);
    backendSelect.value = active.detection.backend || "rt_detr";
    detectionDevice.value = active.detection.device || "cuda";
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

    const resolvedSceneGraph = resolved.scene_graph || {};
    const resolvedVlm = resolvedSceneGraph.vlm || {};
    vlmSystem.value = resolvedVlm.resolved_system_prompt || "";
    vlmUser.value = resolvedVlm.resolved_user_prompt || "";
    vlmPredicates.value = (resolvedVlm.resolved_ontology?.predicates || []).join("\n");
    vlmObjects.value = (active.detection?.ontology || resolved.detection?.resolved_ontology || []).join("\n");
    vlmDevice.value = active.scene_graph?.vlm?.device || "";
    vlmProvider.value = active.scene_graph?.vlm?.provider || "openai";
    vlmModelId.value = active.scene_graph?.vlm?.model_id || "";
    vlmBaseUrl.value = active.scene_graph?.vlm?.base_url || "";
    vlmApiKeyEnv.value = active.scene_graph?.vlm?.api_key_env || "";
    vlmStructuredMode.value = active.scene_graph?.vlm?.structured_output?.mode || "parse_output";
    vlmClientInitKwargs.value = JSON.stringify(active.scene_graph?.vlm?.client_init_kwargs || {}, null, 2);
    vlmCallKwargs.value = JSON.stringify(active.scene_graph?.vlm?.call_kwargs || {}, null, 2);

    const resolvedChat = resolved.chat || {};
    chatSystem.value = resolvedChat.resolved_system_prompt || "";
    chatDevice.value = active.chat?.device || "";
    chatContext.value = resolvedChat.resolved_context_template || "";
    chatProvider.value = active.chat?.provider || "openai";
    chatModelId.value = active.chat?.model_id || "";
    chatBaseUrl.value = active.chat?.base_url || "";
    chatApiKeyEnv.value = active.chat?.api_key_env || "";
    chatStructuredMode.value = active.chat?.structured_output?.mode || "parse_output";
    chatClientInitKwargs.value = JSON.stringify(active.chat?.client_init_kwargs || {}, null, 2);
    chatCallKwargs.value = JSON.stringify(active.chat?.call_kwargs || {}, null, 2);

    sggMode.value = active.scene_graph?.mode || "hybrid";
    sggRulesJson.value = JSON.stringify(active.scene_graph?.rules?.rule_list || [], null, 2);

    const tracking = active.tracking || {};
    const assoc = tracking.association || {};
    const feat = tracking.feature_extraction || {};
    memoryMaxDormantFrames.value = tracking.max_dormant_frames ?? 30;
    memoryAssocVisualWeight.value = assoc.visual_weight ?? 0.8;
    memoryAssocGeometryWeight.value = assoc.geometry_weight ?? 0.2;
    memoryAssocMatchThreshold.value = assoc.match_threshold ?? 0.4;
    memoryFeatReidModel.value = feat.reid_model || "";
    memoryFeatDevice.value = feat.device || "";
    memoryFeatTargetSize.value = Array.isArray(feat.target_size) ? feat.target_size.join(",") : "";
    memoryFeatResampling.value = feat.resampling_method || "";
    memoryMaxAgeSeconds.value = tracking.memory_max_age_seconds ?? 60;
    memoryMaxObjects.value = tracking.memory_max_objects ?? 200;
    memoryMaxRelations.value = tracking.memory_max_relations ?? 500;
}

function buildPatch() {
    let rules = [];
    try {
        rules = sggRulesJson.value.trim() ? JSON.parse(sggRulesJson.value) : [];
    } catch (err) {
        throw new Error("Invalid JSON in SGG rules");
    }
    const parsedVlmClientInitKwargs = parseJsonObject(vlmClientInitKwargs.value, "VLM Client Init Kwargs");
    const parsedVlmCallKwargs = parseJsonObject(vlmCallKwargs.value, "VLM Call Kwargs");
    const parsedChatClientInitKwargs = parseJsonObject(chatClientInitKwargs.value, "LLM Client Init Kwargs");
    const parsedChatCallKwargs = parseJsonObject(chatCallKwargs.value, "LLM Call Kwargs");
    return {
        system: {
            language: languageSelect.value
        },
        detection: {
            backend: backendSelect.value,
            confidence_threshold: parseFloat(input.value),
            device: detectionDevice.value.trim() || "cuda",
            ontology: parseOntologyList(vlmObjects.value)
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
        tracking: {
            max_dormant_frames: parseInt(memoryMaxDormantFrames.value, 10) || 30,
            memory_max_age_seconds: parseInt(memoryMaxAgeSeconds.value, 10) || 60,
            memory_max_objects: parseInt(memoryMaxObjects.value, 10) || 200,
            memory_max_relations: parseInt(memoryMaxRelations.value, 10) || 500,
            association: {
                visual_weight: parseFloat(memoryAssocVisualWeight.value) || 0.8,
                geometry_weight: parseFloat(memoryAssocGeometryWeight.value) || 0.2,
                match_threshold: parseFloat(memoryAssocMatchThreshold.value) || 0.4
            },
            feature_extraction: {
                reid_model: memoryFeatReidModel.value.trim() || null,
                device: memoryFeatDevice.value.trim() || null,
                target_size: parseTargetSize(memoryFeatTargetSize.value),
                resampling_method: memoryFeatResampling.value.trim() || null
            }
        },
        scene_graph: {
            mode: sggMode.value,
            vlm: {
                provider: vlmProvider.value,
                model_id: vlmModelId.value.trim(),
                device: vlmDevice.value.trim() || null,
                base_url: vlmBaseUrl.value.trim() || null,
                api_key_env: vlmApiKeyEnv.value.trim() || null,
                client_init_kwargs: parsedVlmClientInitKwargs,
                call_kwargs: parsedVlmCallKwargs,
                structured_output: {
                    mode: vlmStructuredMode.value
                },
                system_prompt: { text: vlmSystem.value },
                user_prompt: { text: vlmUser.value },
                ontology: {
                    predicates: parseLines(vlmPredicates.value)
                }
            },
            rules: {
                enabled: true,
                rule_list: rules
            }
        },
        chat: {
            provider: chatProvider.value,
            model_id: chatModelId.value.trim(),
            device: chatDevice.value.trim() || null,
            base_url: chatBaseUrl.value.trim() || null,
            api_key_env: chatApiKeyEnv.value.trim() || null,
            client_init_kwargs: parsedChatClientInitKwargs,
            call_kwargs: parsedChatCallKwargs,
            structured_output: {
                mode: chatStructuredMode.value
            },
            system_prompt: { text: chatSystem.value },
            context_template: { text: chatContext.value }
        }
    };
}

async function applyConfig() {
    const now = Date.now();
    if (isApplyingConfig || now - lastApplyAt < APPLY_DEBOUNCE_MS) {
        return;
    }

    let patch;
    try {
        patch = buildPatch();
    } catch (err) {
        showStatusMessage(err.message, false);
        return;
    }

    isApplyingConfig = true;
    lastApplyAt = now;
    const prevLabel = applyBtn.textContent;
    applyBtn.disabled = true;
    applyBtn.classList.add("opacity-60", "cursor-not-allowed");
    applyBtn.textContent = "Applying...";

    try {
        const res = await fetch("/api/v1/config", {
            method: "PATCH",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify(patch)
        });
        if (res.ok) {
            const data = await res.json();
            showStatusMessage(data.reloaded ? "Applied changes (hard reload)" : "Applied changes");
            if (data.requires_reload && data.requires_reload.length > 0) {
                reloadWarning.textContent = `Hard changes detected: ${data.requires_reload.join(", ")}`;
                reloadWarning.classList.remove("hidden");
            } else {
                reloadWarning.classList.add("hidden");
            }
        } else {
            showStatusMessage("Failed to apply config", false);
        }
    } catch (err) {
        console.error("Failed to apply config:", err);
        showStatusMessage("Failed to apply config", false);
    } finally {
        isApplyingConfig = false;
        applyBtn.disabled = false;
        applyBtn.classList.remove("opacity-60", "cursor-not-allowed");
        applyBtn.textContent = prevLabel;
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
downloadBtn.addEventListener("click", () => downloadConfig("active"));
downloadSavedBtn.addEventListener("click", () => downloadConfig("saved"));

uploadInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) uploadConfig(file);
});

fetchModels();
loadConfig();
