import { doMemoryRequest, fetchMemory, fetchMemorySummary } from './api.js';
import { parseBboxCsv, parseCommaList } from './parsers.js';
import { renderMemory, showMemoryEditorStatus } from './render.js';

function getRenderLimit(dom) {
  const raw = Number(dom?.memRenderLimit?.value);
  if (!Number.isInteger(raw)) return 5;
  return Math.max(1, Math.min(raw, 6));
}

export async function refreshMemory(dom) {
  const renderLimit = getRenderLimit(dom);
  if (dom?.memRenderLimit) {
    dom.memRenderLimit.value = String(renderLimit);
  }
  const [mem, summary] = await Promise.all([
    fetchMemory(),
    fetchMemorySummary(renderLimit),
  ]);
  const payload = {
    ...(mem || {}),
    summary: summary || null,
  };
  renderMemory(dom, payload);
  return payload;
}

export function bindMemoryCrud(dom) {
  if (dom.memRenderLimit) {
    dom.memRenderLimit.addEventListener('change', async () => {
      try {
        await refreshMemory(dom);
        showMemoryEditorStatus(
          dom,
          `Memory graph render limit set to ${getRenderLimit(dom)}`,
        );
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to refresh memory graph',
          false,
        );
      }
    });
  }
  if (dom.memRefresh) {
    dom.memRefresh.addEventListener('click', async () => {
      try {
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Memory refreshed');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to refresh memory',
          false,
        );
      }
    });
  }
  if (dom.memReset) {
    dom.memReset.addEventListener('click', async () => {
      const ok = window.confirm(
        'Reset scene memory? This removes all tracked objects and relationships.',
      );
      if (!ok) return;
      try {
        await doMemoryRequest('/api/v1/memory/reset?confirm=true', {
          method: 'POST',
        });
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Memory reset');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to reset memory',
          false,
        );
      }
    });
  }

  if (dom.memObjCreate) {
    dom.memObjCreate.addEventListener('click', async () => {
      try {
        const bbox = parseBboxCsv(dom.memObjBbox.value);
        if (!dom.memObjLabel.value.trim())
          throw new Error('Object label is required');
        if (!bbox) throw new Error('bbox must be x1,y1,x2,y2');
        const payload = {
          label: dom.memObjLabel.value.trim(),
          bbox,
          attributes: parseCommaList(dom.memObjAttrs.value),
        };
        const parsedId = Number(dom.memObjId.value);
        if (Number.isInteger(parsedId) && parsedId > 0) payload.id = parsedId;
        await doMemoryRequest('/api/v1/memory/object', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Object created');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to create object',
          false,
        );
      }
    });
  }

  if (dom.memObjUpdate) {
    dom.memObjUpdate.addEventListener('click', async () => {
      try {
        const objectId = Number(dom.memObjId.value);
        if (!Number.isInteger(objectId) || objectId <= 0)
          throw new Error('Valid object id is required for update');
        const payload = {};
        if (dom.memObjLabel.value.trim())
          payload.label = dom.memObjLabel.value.trim();
        if (dom.memObjBbox.value.trim()) {
          const bbox = parseBboxCsv(dom.memObjBbox.value);
          if (!bbox) throw new Error('bbox must be x1,y1,x2,y2');
          payload.bbox = bbox;
        }
        if (dom.memObjAttrs.value.trim())
          payload.attributes = parseCommaList(dom.memObjAttrs.value);
        if (Object.keys(payload).length === 0)
          throw new Error('No update fields set');
        await doMemoryRequest(`/api/v1/memory/object/${objectId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Object updated');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to update object',
          false,
        );
      }
    });
  }

  if (dom.memObjDelete) {
    dom.memObjDelete.addEventListener('click', async () => {
      try {
        const objectId = Number(dom.memObjId.value);
        if (!Number.isInteger(objectId) || objectId <= 0)
          throw new Error('Valid object id is required for delete');
        await doMemoryRequest(
          `/api/v1/memory/object/${objectId}?cascade_relations=true`,
          { method: 'DELETE' },
        );
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Object deleted');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to delete object',
          false,
        );
      }
    });
  }

  if (dom.memRelCreate) {
    dom.memRelCreate.addEventListener('click', async () => {
      try {
        const subjectId = Number(dom.memRelSubject.value);
        const objectId = Number(dom.memRelObject.value);
        const predicate = dom.memRelPredicate.value.trim();
        if (!Number.isInteger(subjectId) || subjectId <= 0)
          throw new Error('Valid subject id required');
        if (!Number.isInteger(objectId) || objectId <= 0)
          throw new Error('Valid object id required');
        if (!predicate) throw new Error('Predicate is required');
        const payload = {
          subject_id: subjectId,
          object_id: objectId,
          predicate,
        };
        const count = Number(dom.memRelCount.value);
        if (Number.isInteger(count) && count > 0) payload.count = count;
        await doMemoryRequest('/api/v1/memory/relation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Relation created');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to create relation',
          false,
        );
      }
    });
  }

  if (dom.memRelUpdate) {
    dom.memRelUpdate.addEventListener('click', async () => {
      try {
        const subjectId = Number(dom.memRelSubject.value);
        const objectId = Number(dom.memRelObject.value);
        const predicate = dom.memRelPredicate.value.trim();
        if (!Number.isInteger(subjectId) || subjectId <= 0)
          throw new Error('Valid subject id required');
        if (!Number.isInteger(objectId) || objectId <= 0)
          throw new Error('Valid object id required');
        if (!predicate) throw new Error('Predicate is required');
        const payload = {
          subject_id: subjectId,
          object_id: objectId,
          predicate,
        };
        const count = Number(dom.memRelCount.value);
        if (Number.isInteger(count) && count > 0) payload.count = count;
        if (!payload.count) throw new Error('Set count to update relation');
        await doMemoryRequest('/api/v1/memory/relation', {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Relation updated');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to update relation',
          false,
        );
      }
    });
  }

  if (dom.memRelDelete) {
    dom.memRelDelete.addEventListener('click', async () => {
      try {
        const subjectId = Number(dom.memRelSubject.value);
        const objectId = Number(dom.memRelObject.value);
        const predicate = dom.memRelPredicate.value.trim();
        if (!Number.isInteger(subjectId) || subjectId <= 0)
          throw new Error('Valid subject id required');
        if (!Number.isInteger(objectId) || objectId <= 0)
          throw new Error('Valid object id required');
        if (!predicate) throw new Error('Predicate is required');
        const qs = new URLSearchParams({
          subject_id: String(subjectId),
          predicate,
          object_id: String(objectId),
        });
        await doMemoryRequest(`/api/v1/memory/relation?${qs.toString()}`, {
          method: 'DELETE',
        });
        await refreshMemory(dom);
        showMemoryEditorStatus(dom, 'Relation deleted');
      } catch (err) {
        showMemoryEditorStatus(
          dom,
          err.message || 'Failed to delete relation',
          false,
        );
      }
    });
  }
}
