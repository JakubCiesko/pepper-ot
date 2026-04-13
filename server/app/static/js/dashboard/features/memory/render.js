import { secondsAgo } from './parsers.js';

let memoryCy = null;

function nodeColor(label) {
  if (!label) return '#22c55e';
  let hash = 0;
  for (let i = 0; i < label.length; i += 1) {
    hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  return `hsl(${hue}, 65%, 52%)`;
}

function buildNodeLabel(obj) {
  const lines = [`${obj.label} #${obj.id}`];
  const attrs = Array.isArray(obj.attributes) ? obj.attributes : [];
  if (attrs.length > 0) {
    const preview = attrs.slice(0, 2).join(', ');
    lines.push(attrs.length > 2 ? `${preview} +${attrs.length - 2}` : preview);
  }
  return lines.join('\n');
}

function buildGraphElements(mem, graphObjectIds, cropMap) {
  const ids = Array.isArray(graphObjectIds)
    ? graphObjectIds.map((id) => String(id))
    : [];
  const selected = new Set(ids);
  const objects = Array.isArray(mem?.objects) ? mem.objects : [];
  const relationships = Array.isArray(mem?.relationships)
    ? mem.relationships
    : [];
  const nodes = objects
    .filter((obj) => selected.has(String(obj.id)))
    .map((obj) => {
      const imageB64 = cropMap?.[obj.id];
      const image = imageB64 ? `data:image/jpeg;base64,${imageB64}` : '';
      return {
        data: {
          id: String(obj.id),
          label: buildNodeLabel(obj),
          color: nodeColor(obj.label),
          image,
          hasImage: image ? 'yes' : 'no',
        },
      };
    });
  const edges = relationships
    .filter(
      (rel) =>
        selected.has(String(rel.subject_id)) &&
        selected.has(String(rel.object_id)),
    )
    .map((rel, idx) => ({
      data: {
        id: `mem-e-${idx}-${rel.subject_id}-${rel.predicate}-${rel.object_id}`,
        source: String(rel.subject_id),
        target: String(rel.object_id),
        label: rel.predicate,
      },
    }));
  return [...nodes, ...edges];
}

function destroyMemoryGraph() {
  if (memoryCy) {
    memoryCy.destroy();
    memoryCy = null;
  }
}

function renderMemoryGraph(dom, mem) {
  if (!dom.memoryGraph) return;
  const hasGraphData =
    Object.prototype.hasOwnProperty.call(mem || {}, 'graphObjectIds') ||
    Object.prototype.hasOwnProperty.call(mem || {}, 'cropMap');
  if (!hasGraphData) return;

  const graphObjectIds = Array.isArray(mem?.graphObjectIds)
    ? mem.graphObjectIds
    : [];
  const cropMap =
    mem?.cropMap && typeof mem.cropMap === 'object' ? mem.cropMap : {};

  destroyMemoryGraph();

  if (!graphObjectIds.length) {
    dom.memoryGraph.innerHTML =
      '<p class="text-slate-500">No memory graph yet...</p>';
    return;
  }

  if (!window.cytoscape) {
    dom.memoryGraph.innerHTML =
      '<p class="text-slate-500">Cytoscape is not available.</p>';
    return;
  }

  dom.memoryGraph.innerHTML = '';
  const cs = getComputedStyle(document.body);
  const panelText = cs.getPropertyValue('--panel-text').trim() || '#e5e7eb';
  const panelMuted = cs.getPropertyValue('--panel-muted').trim() || '#94a3b8';
  const panelBg = cs.getPropertyValue('--panel-bg').trim() || '#0f172a';

  memoryCy = window.cytoscape({
    container: dom.memoryGraph,
    elements: buildGraphElements(mem || {}, graphObjectIds, cropMap),
    style: [
      {
        selector: 'node',
        style: {
          width: 150,
          height: 110,
          shape: 'round-rectangle',
          'background-color': 'data(color)',
          'background-image': 'data(image)',
          'background-fit': 'cover',
          'background-width': '100%',
          'background-height': '78%',
          'background-position-y': '18%',
          'border-width': 2,
          'border-color': panelMuted,
          label: 'data(label)',
          color: panelText,
          'text-wrap': 'wrap',
          'text-max-width': 136,
          'font-size': '10px',
          'font-weight': 600,
          'text-valign': 'bottom',
          'text-halign': 'center',
          'text-margin-y': 2,
          'text-background-color': panelBg,
          'text-background-opacity': 0.9,
          'text-background-padding': '3px',
        },
      },
      {
        selector: 'node[hasImage = "no"]',
        style: {
          'background-image': 'none',
        },
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          width: 2,
          'line-color': panelMuted,
          'target-arrow-color': panelMuted,
          label: 'data(label)',
          'font-size': '9px',
          'text-background-color': panelBg,
          'text-background-opacity': 0.95,
          'text-background-padding': '2px',
          color: panelText,
        },
      },
    ],
    layout: {
      name: 'cose',
      fit: true,
      padding: 18,
      animate: false,
    },
    wheelSensitivity: 0.2,
  });
}

export function showMemoryEditorStatus(dom, message, ok = true) {
  if (!dom.memoryEditorStatus) return;
  dom.memoryEditorStatus.textContent = message;
  dom.memoryEditorStatus.classList.toggle('text-red-500', !ok);
}

export function renderPregeneratedQa(dom, payload) {
  if (!dom.memPregeneratedQaResults) return;
  const items = Array.isArray(payload?.pregenerated_qa)
    ? payload.pregenerated_qa
    : [];
  if (!items.length) {
    dom.memPregeneratedQaResults.innerHTML =
      '<p class="panel-muted">No pregenerated Q/A yet.</p>';
    return;
  }

  const metadata =
    payload?.metadata && typeof payload.metadata === 'object'
      ? payload.metadata
      : {};
  const provider = metadata.provider ? String(metadata.provider) : '';
  const modelId = metadata.model_id ? String(metadata.model_id) : '';
  dom.memPregeneratedQaResults.innerHTML = '';

  if (provider || modelId) {
    const meta = document.createElement('p');
    meta.className = 'text-xs panel-muted mb-2';
    meta.textContent = `Generated by ${provider}${provider && modelId ? ' / ' : ''}${modelId}`;
    dom.memPregeneratedQaResults.appendChild(meta);
  }

  items.forEach((item, idx) => {
    const question = String(item?.question || '').trim();
    const answer = String(item?.answer || '').trim();
    if (!question || !answer) return;

    const card = document.createElement('div');
    card.className = 'rounded border border-slate-800 bg-slate-950 p-3';

    const label = document.createElement('div');
    label.className = 'text-xs uppercase tracking-widest panel-muted mb-1';
    label.textContent = `Pair ${idx + 1}`;

    const questionEl = document.createElement('div');
    questionEl.className = 'text-sm font-semibold';
    questionEl.style.color = 'var(--panel-text)';
    questionEl.textContent = question;

    const answerEl = document.createElement('div');
    answerEl.className = 'text-sm mt-2';
    answerEl.style.color = 'var(--panel-text)';
    answerEl.textContent = answer;

    card.appendChild(label);
    card.appendChild(questionEl);
    card.appendChild(answerEl);
    dom.memPregeneratedQaResults.appendChild(card);
  });
}

export function prefillObjectEditor(dom, obj) {
  if (dom.memObjId) dom.memObjId.value = obj.id ?? '';
  if (dom.memObjLabel) dom.memObjLabel.value = obj.label ?? '';
  if (dom.memObjBbox) {
    dom.memObjBbox.value = Array.isArray(obj.bbox) ? obj.bbox.join(',') : '';
  }
  if (dom.memObjAttrs) {
    dom.memObjAttrs.value = Array.isArray(obj.attributes)
      ? obj.attributes.join(',')
      : '';
  }
}

export function prefillRelationEditor(dom, rel) {
  if (dom.memRelSubject) dom.memRelSubject.value = rel.subject_id ?? '';
  if (dom.memRelObject) dom.memRelObject.value = rel.object_id ?? '';
  if (dom.memRelPredicate) dom.memRelPredicate.value = rel.predicate ?? '';
  if (dom.memRelCount) dom.memRelCount.value = rel.count ?? '';
}

export function renderMemory(dom, mem) {
  renderMemoryGraph(dom, mem);
  if (!dom.memoryContainer) return;
  dom.memoryContainer.innerHTML = '';
  const memObjects = mem.objects || [];
  const memRelations = mem.relationships || [];
  if (memObjects.length > 0) {
    memObjects.forEach((obj) => {
      const div = document.createElement('div');
      div.className = 'mb-2 p-2 rounded border border-slate-800 bg-slate-950';
      const attrs =
        obj.attributes && obj.attributes.length > 0
          ? obj.attributes.join(', ')
          : 'no attributes';
      const bbox = obj.bbox
        ? `[${obj.bbox.map((n) => n.toFixed(1)).join(', ')}]`
        : 'n/a';
      div.innerHTML = `
                    <div class="flex items-start justify-between gap-2">
                        <div><strong>${obj.label} (#${obj.id})</strong></div>
                        <button class="memory-prefill-object px-2 py-1 text-xs rounded border panel-border panel-alt" type="button">Prefill</button>
                    </div>
                    <div class="text-xs text-slate-400 mt-1">hits ${obj.hits} · first ${secondsAgo(obj.first_seen)} · last ${secondsAgo(obj.last_seen)}</div>
                    <div class="text-xs text-slate-300 mt-1">attrs: ${attrs}</div>
                    <div class="text-xs text-slate-300 mt-1">bbox: ${bbox}</div>
                `;
      const prefillBtn = div.querySelector('.memory-prefill-object');
      if (prefillBtn) {
        prefillBtn.addEventListener('click', () => {
          prefillObjectEditor(dom, obj);
          showMemoryEditorStatus(dom, `Prefilled object #${obj.id}`);
        });
      }
      dom.memoryContainer.appendChild(div);
    });
  } else {
    dom.memoryContainer.innerHTML = `<p class="text-slate-500">No tracked objects yet...</p>`;
  }
  const relHeader = document.createElement('div');
  relHeader.className =
    'mt-4 mb-2 text-xs uppercase tracking-widest panel-muted';
  relHeader.textContent = 'Relations';
  dom.memoryContainer.appendChild(relHeader);
  if (memRelations.length > 0) {
    memRelations.forEach((rel) => {
      const div = document.createElement('div');
      div.className =
        'mb-2 p-2 rounded border border-slate-800 bg-slate-950 text-xs';
      div.innerHTML = `
                    <div class="flex items-start justify-between gap-2">
                        <span>${rel.subject_id} ${rel.predicate} ${rel.object_id} (count: ${rel.count})</span>
                        <button class="memory-prefill-relation px-2 py-1 text-xs rounded border panel-border panel-alt" type="button">Prefill</button>
                    </div>
                `;
      const prefillBtn = div.querySelector('.memory-prefill-relation');
      if (prefillBtn) {
        prefillBtn.addEventListener('click', () => {
          prefillRelationEditor(dom, rel);
          showMemoryEditorStatus(
            dom,
            `Prefilled relation ${rel.subject_id} ${rel.predicate} ${rel.object_id}`,
          );
        });
      }
      dom.memoryContainer.appendChild(div);
    });
  } else {
    const emptyRel = document.createElement('p');
    emptyRel.className = 'text-slate-500 text-xs';
    emptyRel.textContent = 'No relationships in memory yet.';
    dom.memoryContainer.appendChild(emptyRel);
  }

  if (dom.memoryRaw) {
    const rawPayload = { ...(mem || {}) };
    delete rawPayload.cropMap;
    delete rawPayload.graphObjectIds;
    delete rawPayload.summary;
    dom.memoryRaw.textContent = JSON.stringify(rawPayload, null, 2);
  }
  const summaryMemory = document.getElementById('summary-memory');
  if (summaryMemory)
    summaryMemory.textContent = mem.objects ? mem.objects.length : '—';
}
