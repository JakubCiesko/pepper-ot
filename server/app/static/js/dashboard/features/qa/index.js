import { showStatusMessage } from '../../core/notifications.js';

const dom = {
  refreshBtn: document.getElementById('qa-pool-refresh'),
  saveBtn: document.getElementById('qa-pool-save'),
  resetEditorBtn: document.getElementById('qa-pool-reset-editor'),
  forceGenerateBtn: document.getElementById('qa-pool-force-generate'),
  status: document.getElementById('qa-pool-status'),
  textarea: document.getElementById('qa-pool-json'),
};

let lastSnapshot = [];

function setStatus(message, ok = true) {
  if (!dom.status) return;
  dom.status.textContent = message || '';
  dom.status.classList.toggle('text-red-500', !ok);
}

async function fetchPool() {
  const res = await fetch('/api/v1/chat/pregenerated_qa_pool');
  if (!res.ok) {
    let detail = 'Failed to load QA pool';
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

function parseEditorItems() {
  const raw = String(dom.textarea?.value || '').trim();
  if (!raw) return [];
  const parsed = JSON.parse(raw);
  if (Array.isArray(parsed)) return parsed;
  if (parsed && typeof parsed === 'object' && Array.isArray(parsed.items)) {
    return parsed.items;
  }
  throw new Error('Editor JSON must be an array or { items: [...] }');
}

async function savePool(items) {
  const res = await fetch('/api/v1/chat/pregenerated_qa_pool', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ items }),
  });
  if (!res.ok) {
    let detail = 'Failed to save QA pool';
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

async function forceGenerateIfEmpty() {
  const res = await fetch('/api/v1/chat/pregenerate_qa', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ force_generation: true }),
  });
  if (!res.ok) {
    let detail = 'Failed to force-generate QA';
    try {
      const body = await res.json();
      if (body?.detail) detail = body.detail;
    } catch {
      // ignore
    }
    throw new Error(detail);
  }
  return res.json();
}

function renderPoolItems(items) {
  const payload = Array.isArray(items) ? items : [];
  lastSnapshot = payload;
  if (!dom.textarea) return;
  dom.textarea.value = JSON.stringify(payload, null, 2);
}

export async function refreshQAPoolPanel() {
  const payload = await fetchPool();
  renderPoolItems(payload?.items || []);
  const count = Array.isArray(payload?.items) ? payload.items.length : 0;
  setStatus(`Loaded ${count} QA pool items`);
  return payload;
}

export function initQAPoolPanel() {
  if (!dom.textarea) return;
  void refreshQAPoolPanel().catch((err) => {
    setStatus(err.message || 'Failed to load QA pool', false);
  });

  if (dom.refreshBtn) {
    dom.refreshBtn.addEventListener('click', async () => {
      try {
        await refreshQAPoolPanel();
      } catch (err) {
        setStatus(err.message || 'Failed to refresh QA pool', false);
      }
    });
  }

  if (dom.resetEditorBtn) {
    dom.resetEditorBtn.addEventListener('click', () => {
      renderPoolItems(lastSnapshot);
      setStatus('Editor reset to last loaded snapshot');
    });
  }

  if (dom.forceGenerateBtn) {
    dom.forceGenerateBtn.addEventListener('click', async () => {
      const previousLabel = dom.forceGenerateBtn.textContent;
      try {
        dom.forceGenerateBtn.disabled = true;
        dom.forceGenerateBtn.textContent = 'Generating...';
        await forceGenerateIfEmpty();
        await refreshQAPoolPanel();
        setStatus('Force generation completed');
      } catch (err) {
        setStatus(err.message || 'Failed to force-generate QA', false);
      } finally {
        dom.forceGenerateBtn.disabled = false;
        dom.forceGenerateBtn.textContent =
          previousLabel || 'Force Generate If Empty';
      }
    });
  }

  if (dom.saveBtn) {
    dom.saveBtn.addEventListener('click', async () => {
      const previousLabel = dom.saveBtn.textContent;
      try {
        dom.saveBtn.disabled = true;
        dom.saveBtn.textContent = 'Saving...';
        const items = parseEditorItems();
        const payload = await savePool(items);
        renderPoolItems(payload?.items || []);
        setStatus(
          `Saved ${Array.isArray(payload?.items) ? payload.items.length : 0} QA pool items`,
        );
        showStatusMessage('Saved QA pool');
      } catch (err) {
        setStatus(err.message || 'Failed to save QA pool', false);
        showStatusMessage(err.message || 'Failed to save QA pool', false);
      } finally {
        dom.saveBtn.disabled = false;
        dom.saveBtn.textContent = previousLabel || 'Save Pool';
      }
    });
  }
}
