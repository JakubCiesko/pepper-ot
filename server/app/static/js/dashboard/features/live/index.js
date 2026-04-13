import { showStatusMessage } from '../../core/notifications.js';
import { refreshMemoryPanel, renderMemoryPanel } from '../memory/index.js';
import { renderSceneGraph } from '../scene_graph/index.js';

const MAX_LIVE_FRAMES = 10;
const liveFrames = [];
let activeLiveFrameIndex = -1;

const dom = {
  detectionsContainer: document.getElementById('detections-content'),
  annotatedImage: document.getElementById('annotated-image'),
  inferenceMetricsContainer: document.getElementById('inference-metrics'),
  captionContainer: document.getElementById('caption-content'),
  liveCarouselTrack: document.getElementById('live-carousel-track'),
  liveCarouselPrev: document.getElementById('live-carousel-prev'),
  liveCarouselNext: document.getElementById('live-carousel-next'),
  liveCarouselIndex: document.getElementById('live-carousel-index'),
  sgCarouselTrack: document.getElementById('sg-carousel-track'),
  sgCarouselPrev: document.getElementById('sg-carousel-prev'),
  sgCarouselNext: document.getElementById('sg-carousel-next'),
  sgCarouselIndex: document.getElementById('sg-carousel-index'),
  processImageInput: document.getElementById('process-image-input'),
  processImageBtn: document.getElementById('process-image-btn'),
  processImageName: document.getElementById('process-image-name'),
};

function renderCaption(text) {
  if (!text || !dom.captionContainer) return;
  if (
    dom.captionContainer.children.length === 1 &&
    dom.captionContainer.children[0].textContent.includes('No caption')
  ) {
    dom.captionContainer.innerHTML = '';
  }
  dom.captionContainer.innerHTML = '';
  const div = document.createElement('div');
  div.className =
    'bg-slate-950 border border-slate-800 p-3 rounded shadow-sm text-slate-300 whitespace-pre-wrap break-all overflow-x-auto';
  div.textContent = text;
  dom.captionContainer.appendChild(div);
}

function renderDetections(objects, colors) {
  if (!dom.detectionsContainer) return;
  dom.detectionsContainer.innerHTML = '';
  if (!Array.isArray(objects) || objects.length === 0) {
    dom.detectionsContainer.innerHTML = `<p class="text-slate-500">No objects detected</p>`;
    return;
  }

  objects.forEach((obj) => {
    const div = document.createElement('div');
    div.className = 'obj mb-2 p-2 rounded';
    const labelColor = colors?.[obj.label]
      ? `rgb(${colors[obj.label].join(',')})`
      : '#ddd';
    div.style.backgroundColor = labelColor;
    div.style.color = 'black';
    const objectId =
      obj.object_id !== undefined && obj.object_id !== null
        ? `#${obj.object_id}`
        : 'n/a';
    const conf = Number.isFinite(obj.confidence)
      ? obj.confidence.toFixed(2)
      : 'n/a';
    const bbox = Array.isArray(obj.bbox)
      ? obj.bbox.map((x) => Number(x).toFixed(1)).join(', ')
      : 'n/a';
    div.innerHTML = `<strong>${obj.label}</strong> <span class="text-xs">(${objectId})</span> (${conf}) - bbox: [${bbox}]`;
    dom.detectionsContainer.appendChild(div);
  });
}

function renderMetrics(metrics) {
  if (!dom.inferenceMetricsContainer) return;
  dom.inferenceMetricsContainer.innerHTML = '';
  const payload = metrics || {};
  const keys = Object.keys(payload);
  if (keys.length === 0) {
    dom.inferenceMetricsContainer.innerHTML = `<p class="panel-muted">No metrics recorded yet...</p>`;
    return;
  }
  keys.sort();
  keys.forEach((key) => {
    const row = document.createElement('div');
    row.className = 'flex items-center justify-between text-sm mb-1';
    const value = payload[key];
    const formatted = Number.isFinite(value)
      ? `${value.toFixed(4)} s`
      : String(value);
    row.innerHTML = `<span class="panel-muted">${key}</span><span>${formatted}</span>`;
    dom.inferenceMetricsContainer.appendChild(row);
  });
}

function updateSummaries(payload) {
  const summaryObjects = document.getElementById('summary-objects');
  const summaryMemory = document.getElementById('summary-memory');
  const summaryRelations = document.getElementById('summary-relations');
  if (summaryObjects)
    summaryObjects.textContent = payload.objects ? payload.objects.length : '—';
  if (summaryMemory)
    summaryMemory.textContent = payload.memory?.objects
      ? payload.memory.objects.length
      : '—';
  if (summaryRelations)
    summaryRelations.textContent = payload.scene_graph
      ? payload.scene_graph.length
      : '—';
}

function formatFrameTimestamp(ts) {
  if (!Number.isFinite(ts)) return 'Unknown time';
  return new Date(ts * 1000).toLocaleTimeString();
}

function summarizeSceneGraph(sceneGraph) {
  const rels = Array.isArray(sceneGraph) ? sceneGraph : [];
  if (rels.length === 0) return 'No relations';
  const sample = rels
    .slice(0, 2)
    .map((edge) => `${edge.sub} ${edge.rel} ${edge.obj}`)
    .join(' | ');
  return rels.length > 2 ? `${sample} ...` : sample;
}

function updateCarouselIndexLabels() {
  const text =
    liveFrames.length === 0 || activeLiveFrameIndex < 0
      ? '0 / 0'
      : `${activeLiveFrameIndex + 1} / ${liveFrames.length}`;
  if (dom.liveCarouselIndex) dom.liveCarouselIndex.textContent = text;
  if (dom.sgCarouselIndex) dom.sgCarouselIndex.textContent = text;
}

function renderLiveCarousel() {
  if (!dom.liveCarouselTrack) return;
  dom.liveCarouselTrack.innerHTML = '';
  liveFrames.forEach((frame, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `live-carousel-item ${idx === activeLiveFrameIndex ? 'active' : ''}`;
    btn.setAttribute('aria-label', `Frame ${idx + 1}`);
    btn.addEventListener('click', () => setActiveLiveFrame(idx));

    const img = document.createElement('img');
    img.className = 'live-carousel-thumb';
    img.src = `data:image/jpeg;base64,${frame.image}`;
    img.alt = `Recent frame ${idx + 1}`;
    btn.appendChild(img);

    const meta = document.createElement('div');
    meta.className = 'live-carousel-meta';
    const hasCaption =
      typeof frame.caption === 'string' && frame.caption.trim().length > 0;
    meta.textContent = `${formatFrameTimestamp(frame.timestamp)} • caption ${hasCaption ? 'yes' : 'no'}`;
    btn.appendChild(meta);

    dom.liveCarouselTrack.appendChild(btn);
  });
}

function renderSceneGraphCarousel() {
  if (!dom.sgCarouselTrack) return;
  dom.sgCarouselTrack.innerHTML = '';
  liveFrames.forEach((frame, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `sg-carousel-item ${idx === activeLiveFrameIndex ? 'active' : ''}`;
    btn.setAttribute('aria-label', `Scene graph ${idx + 1}`);
    btn.addEventListener('click', () => setActiveLiveFrame(idx));

    const title = document.createElement('div');
    title.className = 'sg-carousel-title';
    title.textContent = `${Array.isArray(frame.scene_graph) ? frame.scene_graph.length : 0} relations`;
    btn.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'sg-carousel-meta';
    meta.textContent = formatFrameTimestamp(frame.timestamp);
    btn.appendChild(meta);

    const preview = document.createElement('div');
    preview.className = 'sg-carousel-preview';
    preview.textContent = summarizeSceneGraph(frame.scene_graph);
    btn.appendChild(preview);

    dom.sgCarouselTrack.appendChild(btn);
  });
}

function renderActiveFrameSnapshot() {
  if (activeLiveFrameIndex < 0 || activeLiveFrameIndex >= liveFrames.length) {
    return;
  }
  const frame = liveFrames[activeLiveFrameIndex];
  if (dom.annotatedImage && frame.image) {
    dom.annotatedImage.src = `data:image/jpeg;base64,${frame.image}`;
  }
  renderCaption(frame.caption || 'No caption for this frame.');
  renderDetections(frame.objects, frame.colors);
  renderMetrics(frame.metrics);
  renderMemoryPanel(frame.memory || {});
  void refreshMemoryPanel().catch((err) => {
    console.error('Failed to refresh memory summary', err);
  });
  renderSceneGraph(frame.scene_graph || [], frame.memory || {});
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
    timestamp: Number.isFinite(data.timestamp)
      ? data.timestamp
      : Date.now() / 1000,
    image: data.image || '',
    objects: Array.isArray(data.objects) ? data.objects : [],
    colors: data.colors && typeof data.colors === 'object' ? data.colors : {},
    metrics:
      data.metrics && typeof data.metrics === 'object' ? data.metrics : {},
    memory: data.memory && typeof data.memory === 'object' ? data.memory : {},
    scene_graph: Array.isArray(data.scene_graph) ? data.scene_graph : [],
    caption: typeof data.caption === 'string' ? data.caption : '',
    caption_provider:
      typeof data.caption_provider === 'string' ? data.caption_provider : '',
    caption_model_id:
      typeof data.caption_model_id === 'string' ? data.caption_model_id : '',
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

async function loadLastState() {
  try {
    const res = await fetch('/api/v1/state');
    if (!res.ok) return;
    const data = await res.json();
    if (
      data &&
      (data.image || data.objects?.length || data.scene_graph?.length)
    ) {
      pushFrameSnapshot(data);
    }
  } catch (err) {
    console.error('Failed to load last state', err);
  }
}

function goToPrevFrame() {
  if (liveFrames.length === 0) return;
  const next =
    activeLiveFrameIndex <= 0
      ? liveFrames.length - 1
      : activeLiveFrameIndex - 1;
  setActiveLiveFrame(next);
}

function goToNextFrame() {
  if (liveFrames.length === 0) return;
  const next =
    activeLiveFrameIndex >= liveFrames.length - 1
      ? 0
      : activeLiveFrameIndex + 1;
  setActiveLiveFrame(next);
}

async function processUploadedImage() {
  if (!dom.processImageInput || !dom.processImageBtn) return;
  const file = dom.processImageInput.files?.[0];
  if (!file) {
    showStatusMessage('Select an image first.', false);
    return;
  }

  dom.processImageBtn.disabled = true;
  const originalText = dom.processImageBtn.textContent;
  dom.processImageBtn.textContent = 'Processing...';
  try {
    const form = new FormData();
    form.append('file', file);
    form.append('publish', 'true');
    const res = await fetch('/api/v1/detect', {
      method: 'POST',
      body: form,
    });
    if (!res.ok) {
      let detail = 'Failed to process image';
      try {
        const body = await res.json();
        detail = body.detail || detail;
      } catch {
        // ignore
      }
      throw new Error(detail);
    }
    showStatusMessage('Image submitted to /api/v1/detect');
    dom.processImageInput.value = '';
    if (dom.processImageName) dom.processImageName.textContent = '';
  } catch (err) {
    showStatusMessage(err.message || 'Failed to process image', false);
  } finally {
    dom.processImageBtn.disabled = false;
    dom.processImageBtn.textContent = originalText;
  }
}

export function getActiveFrameSnapshot() {
  if (activeLiveFrameIndex < 0 || activeLiveFrameIndex >= liveFrames.length) {
    return null;
  }
  const frame = liveFrames[activeLiveFrameIndex];
  return frame ? { ...frame } : null;
}

export function handleLiveWsMessage(data) {
  if (data?.type === 'caption') {
    renderCaption(data.text);
    return;
  }
  pushFrameSnapshot(data);
}

export function initLivePanel() {
  if (dom.liveCarouselPrev)
    dom.liveCarouselPrev.addEventListener('click', goToPrevFrame);
  if (dom.liveCarouselNext)
    dom.liveCarouselNext.addEventListener('click', goToNextFrame);
  if (dom.sgCarouselPrev)
    dom.sgCarouselPrev.addEventListener('click', goToPrevFrame);
  if (dom.sgCarouselNext)
    dom.sgCarouselNext.addEventListener('click', goToNextFrame);
  if (dom.processImageInput) {
    dom.processImageInput.addEventListener('change', () => {
      const file = dom.processImageInput.files?.[0];
      if (dom.processImageName)
        dom.processImageName.textContent = file ? file.name : '';
    });
  }
  if (dom.processImageBtn) {
    dom.processImageBtn.addEventListener('click', processUploadedImage);
  }
  updateCarouselIndexLabels();
  loadLastState();
}

export function handleMemoryWsMessage(data) {
  renderMemoryPanel(data?.memory || {});
  void refreshMemoryPanel().catch((err) => {
    console.error('Failed to refresh memory summary', err);
  });
}
