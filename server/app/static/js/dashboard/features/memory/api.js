import { requestJson } from '../../core/http.js';

export async function doMemoryRequest(url, options = {}) {
  return requestJson(url, options, 'Request failed');
}

export async function fetchMemory() {
  return requestJson('/api/v1/memory', {}, 'Failed to refresh memory');
}

export async function fetchMemorySummary(renderLimit = 5) {
  return requestJson(
    `/api/v1/memory/summary?render_limit=${encodeURIComponent(renderLimit)}`,
    {},
    'Failed to refresh memory summary',
  );
}

export async function fetchMemoryObjectCrop(objectId) {
  return requestJson(
    `/api/v1/memory/object/${encodeURIComponent(objectId)}/crop`,
    {},
    'Failed to fetch object crop',
  );
}

export async function fetchPregeneratedQa() {
  return requestJson(
    '/api/v1/chat/pregenerate_qa',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    },
    'Failed to pregenerate scene Q/A',
  );
}
