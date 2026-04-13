import { secondsAgo } from './parsers.js';

export function showMemoryEditorStatus(dom, message, ok = true) {
  if (!dom.memoryEditorStatus) return;
  dom.memoryEditorStatus.textContent = message;
  dom.memoryEditorStatus.classList.toggle('text-red-500', !ok);
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
  const hasSummary = Object.prototype.hasOwnProperty.call(mem || {}, 'summary');
  const summary = mem?.summary || null;
  if (dom.memoryGraph && hasSummary) {
    const svg = typeof summary?.graph_svg === 'string' ? summary.graph_svg : '';
    if (svg.trim()) {
      dom.memoryGraph.innerHTML = svg;
    } else {
      dom.memoryGraph.innerHTML = `<p class="text-slate-500">No memory graph yet...</p>`;
    }
  }
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
    dom.memoryRaw.textContent = JSON.stringify(mem, null, 2);
  }
  const summaryMemory = document.getElementById('summary-memory');
  if (summaryMemory)
    summaryMemory.textContent = mem.objects ? mem.objects.length : '—';
}
