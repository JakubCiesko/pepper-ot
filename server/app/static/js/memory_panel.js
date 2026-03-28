(function () {
    const memoryContainer = document.getElementById("memory-content");
    const memoryRaw = document.getElementById("memory-raw");
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
    const memRefresh = document.getElementById("mem-refresh");
    const memReset = document.getElementById("mem-reset");

    function secondsAgo(ts) {
        if (!ts) return "n/a";
        const now = Date.now() / 1000;
        const delta = now - ts;
        return `${delta.toFixed(1)}s ago`;
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

    function prefillObjectEditor(obj) {
        if (memObjId) memObjId.value = obj.id ?? "";
        if (memObjLabel) memObjLabel.value = obj.label ?? "";
        if (memObjBbox) {
            memObjBbox.value = Array.isArray(obj.bbox) ? obj.bbox.join(",") : "";
        }
        if (memObjAttrs) {
            memObjAttrs.value = Array.isArray(obj.attributes)
                ? obj.attributes.join(",")
                : "";
        }
    }

    function prefillRelationEditor(rel) {
        if (memRelSubject) memRelSubject.value = rel.subject_id ?? "";
        if (memRelObject) memRelObject.value = rel.object_id ?? "";
        if (memRelPredicate) memRelPredicate.value = rel.predicate ?? "";
        if (memRelCount) memRelCount.value = rel.count ?? "";
    }

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
                const bbox = obj.bbox
                    ? `[${obj.bbox.map(n => n.toFixed(1)).join(", ")}]`
                    : "n/a";
                div.innerHTML = `
                    <div class="flex items-start justify-between gap-2">
                        <div><strong>${obj.label} (#${obj.id})</strong></div>
                        <button class="memory-prefill-object px-2 py-1 text-xs rounded border panel-border panel-alt" type="button">Prefill</button>
                    </div>
                    <div class="text-xs text-slate-400 mt-1">hits ${obj.hits} · first ${secondsAgo(obj.first_seen)} · last ${secondsAgo(obj.last_seen)}</div>
                    <div class="text-xs text-slate-300 mt-1">attrs: ${attrs}</div>
                    <div class="text-xs text-slate-300 mt-1">bbox: ${bbox}</div>
                `;
                const prefillBtn = div.querySelector(".memory-prefill-object");
                if (prefillBtn) {
                    prefillBtn.addEventListener("click", () => {
                        prefillObjectEditor(obj);
                        showMemoryEditorStatus(`Prefilled object #${obj.id}`);
                    });
                }
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
                div.innerHTML = `
                    <div class="flex items-start justify-between gap-2">
                        <span>${rel.subject_id} ${rel.predicate} ${rel.object_id} (count: ${rel.count})</span>
                        <button class="memory-prefill-relation px-2 py-1 text-xs rounded border panel-border panel-alt" type="button">Prefill</button>
                    </div>
                `;
                const prefillBtn = div.querySelector(".memory-prefill-relation");
                if (prefillBtn) {
                    prefillBtn.addEventListener("click", () => {
                        prefillRelationEditor(rel);
                        showMemoryEditorStatus(
                            `Prefilled relation ${rel.subject_id} ${rel.predicate} ${rel.object_id}`
                        );
                    });
                }
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

    function bindMemoryCrud() {
        if (memRefresh) {
            memRefresh.addEventListener("click", async () => {
                try {
                    await refreshMemoryFromApi();
                    showMemoryEditorStatus("Memory refreshed");
                } catch (err) {
                    showMemoryEditorStatus(
                        err.message || "Failed to refresh memory",
                        false
                    );
                }
            });
        }

        if (memReset) {
            memReset.addEventListener("click", async () => {
                const ok = window.confirm(
                    "Reset scene memory? This removes all tracked objects and relationships."
                );
                if (!ok) return;
                try {
                    await doMemoryRequest("/api/v1/memory/reset?confirm=true", {
                        method: "POST",
                    });
                    await refreshMemoryFromApi();
                    showMemoryEditorStatus("Memory reset");
                } catch (err) {
                    showMemoryEditorStatus(
                        err.message || "Failed to reset memory",
                        false
                    );
                }
            });
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
                        attributes: parseCommaList(memObjAttrs.value),
                    };
                    const parsedId = Number(memObjId.value);
                    if (Number.isInteger(parsedId) && parsedId > 0) payload.id = parsedId;
                    await doMemoryRequest("/api/v1/memory/object", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify(payload),
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
                        body: JSON.stringify(payload),
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
                        method: "DELETE",
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
                        body: JSON.stringify(payload),
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
                        body: JSON.stringify(payload),
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
                        object_id: String(objectId),
                    });
                    await doMemoryRequest(`/api/v1/memory/relation?${qs.toString()}`, {
                        method: "DELETE",
                    });
                    await refreshMemoryFromApi();
                    showMemoryEditorStatus("Relation deleted");
                } catch (err) {
                    showMemoryEditorStatus(err.message || "Failed to delete relation", false);
                }
            });
        }
    }

    window.PepperMemoryPanel = {
        init: bindMemoryCrud,
        renderMemory,
    };
})();
