(function () {
  const bundle = JSON.parse(document.getElementById("bundle-data").textContent);
  const storageKey = `scene-annotation:${bundle.bundle_id}`;

  const els = {
    bundleTitle: document.getElementById("bundle-title"),
    bundleMeta: document.getElementById("bundle-meta"),
    progress: document.getElementById("progress"),
    imageList: document.getElementById("image-list"),
    imageTitle: document.getElementById("image-title"),
    imageStatus: document.getElementById("image-status"),
    viewLabel: document.getElementById("view-label"),
    sceneImage: document.getElementById("scene-image"),
    captionText: document.getElementById("caption-text"),
    objectList: document.getElementById("object-list"),
    predicateList: document.getElementById("predicate-list"),
    attributeList: document.getElementById("attribute-list"),
    vocabCounts: document.getElementById("vocab-counts"),
    relationFilter: document.getElementById("relation-filter"),
    notes: document.getElementById("notes"),
    tripleBody: document.getElementById("triple-body"),
    editorStatus: document.getElementById("editor-status"),
    prevBtn: document.getElementById("prev-btn"),
    nextBtn: document.getElementById("next-btn"),
    toggleViewBtn: document.getElementById("toggle-view-btn"),
    saveBtn: document.getElementById("save-btn"),
    copyBtn: document.getElementById("copy-btn"),
    markDoneBtn: document.getElementById("mark-done-btn"),
    addRelationBtn: document.getElementById("add-relation-btn"),
    addAttributeBtn: document.getElementById("add-attribute-btn"),
  };

  const state = loadState() || {
    currentIndex: 0,
    showSom: true,
    items: bundle.items.map((item) => ({
      rows: (item.draft_relationships || []).map(normalizeRow),
      done: false,
      notes: "",
    })),
  };

  let activeRowIndex = null;

  function normalizeRow(row) {
    return {
      sub: row?.sub != null ? String(row.sub) : "",
      rel: row?.rel != null ? String(row.rel) : "",
      obj: row?.obj != null ? String(row.obj) : "",
    };
  }

  function loadState() {
    try {
      const raw = localStorage.getItem(storageKey);
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      if (!parsed || !Array.isArray(parsed.items)) return null;
      const items = bundle.items.map((item, idx) => {
        const stored = parsed.items[idx] || {};
        return {
          rows: Array.isArray(stored.rows)
            ? stored.rows.map(normalizeRow)
            : (item.draft_relationships || []).map(normalizeRow),
          done: Boolean(stored.done),
          notes: String(stored.notes || ""),
        };
      });
      return {
        currentIndex: Number.isInteger(parsed.currentIndex)
          ? Math.min(Math.max(parsed.currentIndex, 0), Math.max(0, bundle.items.length - 1))
          : 0,
        showSom: parsed.showSom !== false,
        items,
      };
    } catch {
      return null;
    }
  }

  function saveState() {
    localStorage.setItem(storageKey, JSON.stringify(state));
  }

  function currentItem() {
    return bundle.items[state.currentIndex];
  }

  function currentStateItem() {
    return state.items[state.currentIndex];
  }

  function objectOptions() {
    const items = currentItem().objects || [];
    return items
      .map((obj) => {
        const label = obj.label ? ` ${obj.label}` : "";
        return `<option value="${escapeHtml(String(obj.id))}">${escapeHtml(
          `${obj.id}${label}`
        )}</option>`;
      })
      .join("");
  }

  function vocabRelations() {
    const query = els.relationFilter.value.trim().toLowerCase();
    const preds = bundle.items[state.currentIndex].vocabulary?.predicates || [];
    const attrs = bundle.items[state.currentIndex].vocabulary?.attributes || [];

    const filter = (value) => !query || value.toLowerCase().includes(query);
    return {
      predicates: preds.filter(filter),
      attributes: attrs.filter(filter),
    };
  }

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function render() {
    const item = currentItem();
    const current = currentStateItem();
    const rows = current.rows;
    const objects = item.objects || [];
    const vocab = vocabRelations();

    els.bundleTitle.textContent = bundle.bundle_id;
    els.bundleMeta.textContent = `${bundle.items.length} images | ${bundle.generated_utc}`;
    els.progress.textContent = `${state.currentIndex + 1}/${bundle.items.length}`;
    els.imageTitle.textContent = `Image ${state.currentIndex + 1}`;
    els.imageStatus.textContent = current.done ? "done" : "in progress";
    els.viewLabel.textContent = state.showSom && item.som_image_uri ? "SoM image" : "raw image";
    els.sceneImage.src = state.showSom && item.som_image_uri ? item.som_image_uri : item.raw_image_uri;
    els.captionText.textContent = item.caption || "(no caption)";
    els.notes.value = current.notes || "";
    els.vocabCounts.textContent = `${(item.vocabulary?.predicates || []).length} predicates, ${(item.vocabulary?.attributes || []).length} attributes`;

    renderImageList();
    renderObjects(objects);
    renderVocabulary(vocab);
    renderRows(rows, objects, vocab);
    saveState();
  }

  function renderImageList() {
    els.imageList.innerHTML = bundle.items
      .map((item, idx) => {
        const active = idx === state.currentIndex ? "current" : "";
        const done = state.items[idx]?.done ? "done" : "";
        const relCount = (state.items[idx]?.rows || []).filter(
          (row) => row.sub && row.rel && row.obj
        ).length;
        return `
          <div class="image-row ${active} ${done}" data-index="${idx}">
            <div class="image-row-header">
              <strong class="image-path">${escapeHtml(item.image_path || item.key)}</strong>
              <span>${relCount}</span>
            </div>
          </div>
        `;
      })
      .join("");
    els.imageList.querySelectorAll(".image-row").forEach((node) => {
      node.addEventListener("click", () => {
        state.currentIndex = Number(node.dataset.index);
        activeRowIndex = null;
        render();
      });
    });
  }

  function renderObjects(objects) {
    if (!objects.length) {
      els.objectList.innerHTML = `<div class="empty-state">No objects available.</div>`;
      return;
    }
    els.objectList.innerHTML = objects
      .map((obj) => {
        const bbox = Array.isArray(obj.bbox) ? obj.bbox.map((x) => Math.round(Number(x))).join(", ") : "";
        return `
          <div class="object-row">
            <div class="object-id">#${escapeHtml(obj.id)}</div>
            <div>
              <div class="object-label">${escapeHtml(obj.label || "object")}</div>
              <div class="subtle">${escapeHtml(bbox)}</div>
            </div>
            <div class="object-actions">
              <button class="mini-btn" data-action="subject" data-id="${escapeHtml(obj.id)}">S</button>
              <button class="mini-btn" data-action="object" data-id="${escapeHtml(obj.id)}">O</button>
            </div>
          </div>
        `;
      })
      .join("");
    els.objectList.querySelectorAll("button[data-action]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const id = String(btn.dataset.id);
        if (activeRowIndex == null) {
          addRow(btn.dataset.action === "subject" ? { sub: id, obj: id } : { sub: id, obj: id }, btn.dataset.action);
          return;
        }
        const row = currentStateItem().rows[activeRowIndex];
        if (!row) return;
        if (btn.dataset.action === "subject") {
          row.sub = id;
          if (isAttributeRow(row)) row.obj = id;
        } else {
          row.obj = id;
          if (isAttributeRow(row)) row.sub = id;
        }
        render();
      });
    });
  }

  function renderVocabulary(vocab) {
    const renderTags = (values, kind) =>
      values.length
        ? values.map((value) => `<span class="tag ${kind}">${escapeHtml(value)}</span>`).join("")
        : `<div class="empty-state">None</div>`;
    els.predicateList.innerHTML = renderTags(vocab.predicates, "pred");
    els.attributeList.innerHTML = renderTags(vocab.attributes, "attr");
  }

  function relationOptions(includeAttributes = true) {
    const vocab = vocabRelations();
    const subjectOptions = objectOptions();
    const renderOptions = (values, kind) =>
      values
        .map(
          (value) =>
            `<option value="${escapeHtml(value)}">${escapeHtml(value)} (${kind})</option>`
        )
        .join("");
    return {
      objectOptions: subjectOptions,
      relationOptions: `
        ${vocab.predicates.length ? `<optgroup label="Predicates">${renderOptions(vocab.predicates, "rel")}</optgroup>` : ""}
        ${includeAttributes && vocab.attributes.length ? `<optgroup label="Attributes">${renderOptions(vocab.attributes, "attr")}</optgroup>` : ""}
      `,
    };
  }

  function renderRows(rows, objects) {
    const { objectOptions: objOptions, relationOptions: relOptions } = relationOptions(true);
    if (!rows.length) {
      els.tripleBody.innerHTML = `<tr><td colspan="6"><div class="empty-state">No triples yet. Add one from the buttons above.</div></td></tr>`;
      els.editorStatus.textContent = "0 rows";
      return;
    }
    els.tripleBody.innerHTML = rows
      .map((row, idx) => {
        const type = isAttributeRow(row) ? "attr" : "rel";
        const active = idx === activeRowIndex ? "row-active" : "";
        return `
          <tr class="${active}" data-row-index="${idx}">
            <td>${idx + 1}</td>
            <td>
              <select class="row-select" data-field="sub" data-row-index="${idx}">
                ${objOptions}
              </select>
            </td>
            <td>
              <select class="row-select" data-field="rel" data-row-index="${idx}">
                <option value="">Select relation</option>
                ${relOptions}
              </select>
            </td>
            <td>
              <select class="row-select" data-field="obj" data-row-index="${idx}">
                ${objOptions}
              </select>
            </td>
            <td><span class="type-badge ${type}">${type === "attr" ? "attribute" : "relation"}</span></td>
            <td>
              <div class="row-tools">
                <button class="mini-btn" data-action="clone" data-row-index="${idx}">Dup</button>
                <button class="mini-btn" data-action="delete" data-row-index="${idx}">Del</button>
              </div>
            </td>
          </tr>
        `;
      })
      .join("");

    const selects = els.tripleBody.querySelectorAll("select.row-select");
    selects.forEach((select) => {
      const row = rows[Number(select.dataset.rowIndex)];
      if (!row) return;
      const field = select.dataset.field;
      select.value = row[field] || "";
      select.addEventListener("change", () => {
        const currentRows = currentStateItem().rows;
        const target = currentRows[Number(select.dataset.rowIndex)];
        if (!target) return;
        target[field] = select.value;
        if (field === "sub" || field === "obj") {
          if (isAttributeRow(target)) {
            target.obj = target.sub || target.obj;
          }
        }
        syncCurrentRowType(Number(select.dataset.rowIndex));
        render();
      });
      select.addEventListener("focus", () => {
        activeRowIndex = Number(select.dataset.rowIndex);
        renderRowActive();
      });
    });

    els.tripleBody.querySelectorAll("tr[data-row-index]").forEach((rowEl) => {
      rowEl.addEventListener("click", (event) => {
        if (event.target.closest("button") || event.target.closest("select")) return;
        activeRowIndex = Number(rowEl.dataset.rowIndex);
        renderRowActive();
      });
    });

    els.tripleBody.querySelectorAll("button[data-action]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const rowIndex = Number(btn.dataset.rowIndex);
        activeRowIndex = rowIndex;
        if (btn.dataset.action === "delete") deleteRow(rowIndex);
        if (btn.dataset.action === "clone") cloneRow(rowIndex);
      });
    });

    els.editorStatus.textContent = `${rows.length} rows`;
  }

  function renderRowActive() {
    els.tripleBody.querySelectorAll("tr[data-row-index]").forEach((rowEl) => {
      rowEl.classList.toggle("row-active", Number(rowEl.dataset.rowIndex) === activeRowIndex);
    });
  }

  function isAttributeRow(row) {
    return row && String(row.sub || "") === String(row.obj || "");
  }

  function syncCurrentRowType(rowIndex) {
    const row = currentStateItem().rows[rowIndex];
    if (!row) return;
    if (isAttributeRow(row) && row.sub && !row.obj) row.obj = row.sub;
  }

  function addRow(seed = null, mode = "relation") {
    const objects = currentItem().objects || [];
    const firstId = objects[0] ? String(objects[0].id) : "";
    const secondId = objects[1] ? String(objects[1].id) : firstId;
    const vocab = currentItem().vocabulary || { predicates: [], attributes: [] };
    const row = normalizeRow(
      seed || {
        sub: mode === "attribute" ? firstId : firstId,
        rel:
          mode === "attribute"
            ? vocab.attributes[0] || vocab.predicates[0] || ""
            : vocab.predicates[0] || vocab.attributes[0] || "",
        obj: mode === "attribute" ? firstId : secondId,
      }
    );
    currentStateItem().rows.push(row);
    activeRowIndex = currentStateItem().rows.length - 1;
    render();
  }

  function deleteRow(index) {
    const rows = currentStateItem().rows;
    rows.splice(index, 1);
    if (activeRowIndex != null) {
      activeRowIndex = Math.min(activeRowIndex, rows.length - 1);
    }
    render();
  }

  function cloneRow(index) {
    const row = currentStateItem().rows[index];
    if (!row) return;
    currentStateItem().rows.splice(index + 1, 0, normalizeRow(row));
    activeRowIndex = index + 1;
    render();
  }

  function nextImage(delta) {
    state.currentIndex = (state.currentIndex + delta + bundle.items.length) % bundle.items.length;
    activeRowIndex = null;
    render();
  }

  function toggleDone() {
    currentStateItem().done = !currentStateItem().done;
    render();
  }

  function buildExportPayload() {
    const out = {};
    bundle.items.forEach((item, idx) => {
      const rows = (state.items[idx]?.rows || [])
        .map(normalizeRow)
        .filter((row) => row.sub && row.rel && row.obj);
      out[item.image_path] = {
        relationships: rows,
      };
    });
    return out;
  }

  function downloadJSON() {
    const payload = buildExportPayload();
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ground_truth_scene_graph.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  }

  async function copyJSON() {
    const text = JSON.stringify(buildExportPayload(), null, 2);
    try {
      await navigator.clipboard.writeText(text);
      els.editorStatus.textContent = "Copied JSON to clipboard";
    } catch {
      els.editorStatus.textContent = "Clipboard copy failed";
    }
  }

  function loadCurrentNotes() {
    els.notes.value = currentStateItem().notes || "";
  }

  function bind() {
    els.prevBtn.addEventListener("click", () => nextImage(-1));
    els.nextBtn.addEventListener("click", () => nextImage(1));
    els.toggleViewBtn.addEventListener("click", () => {
      state.showSom = !state.showSom;
      render();
    });
    els.saveBtn.addEventListener("click", downloadJSON);
    els.copyBtn.addEventListener("click", copyJSON);
    els.markDoneBtn.addEventListener("click", toggleDone);
    els.addRelationBtn.addEventListener("click", () => addRow(null, "relation"));
    els.addAttributeBtn.addEventListener("click", () => addRow(null, "attribute"));
    els.relationFilter.addEventListener("input", () => render());
    els.notes.addEventListener("input", () => {
      currentStateItem().notes = els.notes.value;
      saveState();
    });
    document.addEventListener("keydown", (event) => {
      if (event.target && ["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName)) {
        if (!(event.ctrlKey || event.metaKey)) return;
      }
      if (event.key === "ArrowLeft" || event.key === "p") {
        event.preventDefault();
        nextImage(-1);
      }
      if (event.key === "ArrowRight" || event.key === "n") {
        event.preventDefault();
        nextImage(1);
      }
      if (event.key === "a" && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        addRow(null, "relation");
      }
      if (event.key === "d" && !event.ctrlKey && !event.metaKey) {
        event.preventDefault();
        if (activeRowIndex != null) deleteRow(activeRowIndex);
      }
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s") {
        event.preventDefault();
        downloadJSON();
      }
    });
  }

  function init() {
    bind();
    loadCurrentNotes();
    render();
  }

  init();
})();
