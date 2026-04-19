import { showStatusMessage } from '../../core/notifications.js';

const slider = document.getElementById('threshold-slider');
const input = document.getElementById('threshold-input');
const backendSelect = document.getElementById('backend-select');
const detectionDevice = document.getElementById('detection-device');
const outputLanguageSelect = document.getElementById('output-language-select');

const visBbox = document.getElementById('vis-bbox');
const visMask = document.getElementById('vis-mask');
const visPolygon = document.getElementById('vis-polygon');
const visLabels = document.getElementById('vis-labels');
const visLine = document.getElementById('vis-line');
const visOpacity = document.getElementById('vis-opacity');
const visColor = document.getElementById('vis-color');
const visMaskBackend = document.getElementById('vis-mask-backend');
const visDevice = document.getElementById('vis-device');
const storagePersist = document.getElementById('storage-persist');
const storageImage = document.getElementById('storage-image');
const storagePath = document.getElementById('storage-path');

const vlmSystem = document.getElementById('vlm-system');
const vlmUser = document.getElementById('vlm-user');
const vlmPredicates = document.getElementById('vlm-predicates');
const vlmObjects = document.getElementById('vlm-objects');
const vlmDevice = document.getElementById('vlm-device');
const vlmProvider = document.getElementById('vlm-provider');
const vlmModelId = document.getElementById('vlm-model-id');
const vlmBaseUrl = document.getElementById('vlm-base-url');
const vlmApiKeyEnv = document.getElementById('vlm-api-key-env');
const vlmStructuredMode = document.getElementById('vlm-structured-strategy');
const vlmStructuredStrict = document.getElementById('vlm-structured-strict');
const vlmStructuredSchema = document.getElementById('vlm-structured-schema');
const vlmLocalPromptStyle = document.getElementById('vlm-local-prompt-style');
const vlmLocalImageTokenStrategy = document.getElementById(
  'vlm-local-image-token-strategy',
);
const vlmStructuredCapability = document.getElementById(
  'vlm-structured-capability',
);
const vlmClientInitKwargs = document.getElementById('vlm-client-init-kwargs');
const vlmClientInitKwargsStatus = document.getElementById(
  'vlm-client-init-kwargs-status',
);
const vlmCallKwargs = document.getElementById('vlm-call-kwargs');
const vlmCallKwargsStatus = document.getElementById('vlm-call-kwargs-status');
const sggEnableVlm = document.getElementById('sgg-enable-vlm');
const sggEnableRules = document.getElementById('sgg-enable-rules');
const sggEnableReltr = document.getElementById('sgg-enable-reltr');
const sggRulesJson = document.getElementById('sgg-rules-json');
const reltrCheckpointPath = document.getElementById('reltr-checkpoint-path');
const reltrDevice = document.getElementById('reltr-device');
const reltrThreshold = document.getElementById('reltr-threshold');
const reltrTopk = document.getElementById('reltr-topk');
const reltrIouMatchThreshold = document.getElementById(
  'reltr-iou-match-threshold',
);

const pipelinePreset = document.getElementById('pipeline-preset');
const pipelineCaption = document.getElementById('pipeline-caption');
const pipelineDetect = document.getElementById('pipeline-detect');
const pipelineTrackMemory = document.getElementById('pipeline-track-memory');
const pipelinePaintSom = document.getElementById('pipeline-paint-som');
const pipelineSceneGraph = document.getElementById('pipeline-scene-graph');
const pipelineQaGeneration = document.getElementById('pipeline-qa-generation');
const pipelineUpdateSceneMemory = document.getElementById(
  'pipeline-update-scene-memory',
);
const pipelineSummary = document.getElementById('pipeline-summary');
const qaPairsPerUpdate = document.getElementById('qa-pairs-per-update');
const qaPoolMaxEntries = document.getElementById('qa-pool-max-entries');
const workerEnabled = document.getElementById('worker-enabled');
const workerHost = document.getElementById('worker-host');
const workerPort = document.getElementById('worker-port');
const workerIdleTimeout = document.getElementById('worker-idle-timeout');
const workerIdleCheckInterval = document.getElementById(
  'worker-idle-check-interval',
);
const workerStartupTimeout = document.getElementById('worker-startup-timeout');
const workerRequestTimeout = document.getElementById('worker-request-timeout');
const workerShutdownGrace = document.getElementById('worker-shutdown-grace');
const workerMaxStartupQueue = document.getElementById(
  'worker-max-startup-queue',
);
const workerHealthcheckInterval = document.getElementById(
  'worker-healthcheck-interval',
);
const workerRestartMaxAttempts = document.getElementById(
  'worker-restart-max-attempts',
);
const workerRestartWindow = document.getElementById('worker-restart-window');
const workerRestartBackoff = document.getElementById('worker-restart-backoff');
const workerCircuitBreakerCooldown = document.getElementById(
  'worker-circuit-breaker-cooldown',
);
const workerAutoWarmup = document.getElementById(
  'worker-auto-warmup-on-startup',
);

const memoryMaxDormantFrames = document.getElementById(
  'memory-max-dormant-frames',
);
const memoryAssocVisualWeight = document.getElementById(
  'memory-assoc-visual-weight',
);
const memoryAssocGeometryWeight = document.getElementById(
  'memory-assoc-geometry-weight',
);
const memoryAssocMatchThreshold = document.getElementById(
  'memory-assoc-match-threshold',
);
const memoryFeatReidModel = document.getElementById('memory-feat-reid-model');
const memoryFeatDevice = document.getElementById('memory-feat-device');
const memoryFeatTargetSize = document.getElementById('memory-feat-target-size');
const memoryFeatResampling = document.getElementById('memory-feat-resampling');
const memoryMaxAgeSeconds = document.getElementById('memory-max-age-seconds');
const memoryMaxObjects = document.getElementById('memory-max-objects');
const memoryMaxRelations = document.getElementById('memory-max-relations');

const fusionPersonBboxMatchThresholdPx = document.getElementById(
  'fusion-person-bbox-match-threshold-px',
);
const fusionEstimatedPersonBboxBasePx = document.getElementById(
  'fusion-estimated-person-bbox-base-px',
);
const fusionEstimatedPersonBboxMinPx = document.getElementById(
  'fusion-estimated-person-bbox-min-px',
);
const fusionEstimatedPersonBboxMaxPx = document.getElementById(
  'fusion-estimated-person-bbox-max-px',
);
const fusionAngularYawThresholdRad = document.getElementById(
  'fusion-angular-yaw-threshold-rad',
);
const fusionAngularPitchThresholdRad = document.getElementById(
  'fusion-angular-pitch-threshold-rad',
);
const fusionMatchedPersonMinConfidence = document.getElementById(
  'fusion-matched-person-min-confidence',
);
const fusionSyntheticPersonConfidence = document.getElementById(
  'fusion-synthetic-person-confidence',
);
const fusionPepperBindingMaxMisses = document.getElementById(
  'fusion-pepper-binding-max-misses',
);

const chatSystem = document.getElementById('chat-system');
const chatUser = document.getElementById('chat-user');
const chatObjectSystem = document.getElementById('chat-object-system');
const chatObjectUser = document.getElementById('chat-object-user');
const chatDevice = document.getElementById('chat-device');
const chatProvider = document.getElementById('chat-provider');
const chatModelId = document.getElementById('chat-model-id');
const chatBaseUrl = document.getElementById('chat-base-url');
const chatApiKeyEnv = document.getElementById('chat-api-key-env');
const chatClientInitKwargs = document.getElementById('chat-client-init-kwargs');
const chatClientInitKwargsStatus = document.getElementById(
  'chat-client-init-kwargs-status',
);
const chatCallKwargs = document.getElementById('chat-call-kwargs');
const chatCallKwargsStatus = document.getElementById('chat-call-kwargs-status');
const captionSystem = document.getElementById('caption-system');
const captionUser = document.getElementById('caption-user');
const captionMode = document.getElementById('caption-mode');
const captionMaxWords = document.getElementById('caption-max-words');
const captionDevice = document.getElementById('caption-device');
const captionProvider = document.getElementById('caption-provider');
const captionModelId = document.getElementById('caption-model-id');
const captionBaseUrl = document.getElementById('caption-base-url');
const captionApiKeyEnv = document.getElementById('caption-api-key-env');
const captionClientInitKwargs = document.getElementById(
  'caption-client-init-kwargs',
);
const captionClientInitKwargsStatus = document.getElementById(
  'caption-client-init-kwargs-status',
);
const captionCallKwargs = document.getElementById('caption-call-kwargs');
const captionCallKwargsStatus = document.getElementById(
  'caption-call-kwargs-status',
);
const translationsLabels = document.getElementById('translations-labels');
const translationsAttributes = document.getElementById(
  'translations-attributes',
);
const translationsRelations = document.getElementById('translations-relations');
const translationsLabelsStatus = document.getElementById(
  'translations-labels-status',
);
const translationsAttributesStatus = document.getElementById(
  'translations-attributes-status',
);
const translationsRelationsStatus = document.getElementById(
  'translations-relations-status',
);

const applyBtn = document.getElementById('apply-config');
const saveBtn = document.getElementById('save-config');
const reloadBtn = document.getElementById('reload-config');
const downloadBtn = document.getElementById('download-config');
const downloadSavedBtn = document.getElementById('download-config-saved');
const uploadInput = document.getElementById('upload-config');
const reloadWarning = document.getElementById('reload-warning');
const uploadProgress = document.getElementById('upload-progress');
const uploadProgressBar = document.getElementById('upload-progress-bar');
const uploadProgressText = document.getElementById('upload-progress-text');

let isApplyingConfig = false;
let lastApplyAt = 0;
const APPLY_DEBOUNCE_MS = 500;
let configContracts = {};

const PIPELINE_PRESETS_FALLBACK = {
  full: {
    caption: true,
    detect: true,
    track_memory: true,
    paint_som: true,
    scene_graph: true,
    qa_generation: true,
    update_scene_memory: true,
  },
  detect_only: {
    caption: false,
    detect: true,
    track_memory: false,
    paint_som: false,
    scene_graph: false,
    qa_generation: false,
    update_scene_memory: false,
  },
  caption_only: {
    caption: true,
    detect: false,
    track_memory: false,
    paint_som: false,
    scene_graph: false,
    qa_generation: false,
    update_scene_memory: false,
  },
  vlm_only: {
    caption: false,
    detect: false,
    track_memory: false,
    paint_som: false,
    scene_graph: true,
    qa_generation: false,
    update_scene_memory: false,
  },
  rules_only: {
    caption: false,
    detect: true,
    track_memory: true,
    paint_som: false,
    scene_graph: true,
    qa_generation: false,
    update_scene_memory: true,
  },
  minimal: {
    caption: false,
    detect: true,
    track_memory: false,
    paint_som: false,
    scene_graph: false,
    qa_generation: false,
    update_scene_memory: false,
  },
};

function setThreshold(value) {
  slider.value = value;
  input.value = value;
}

function parseLines(text) {
  return text
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);
}

function parseOntologyList(text) {
  return text
    .split(/[\n,]/)
    .map((v) => v.trim())
    .filter(Boolean);
}

function parseTargetSize(text) {
  const parts = String(text || '')
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && v > 0);
  return parts.length === 2
    ? [Math.round(parts[0]), Math.round(parts[1])]
    : null;
}

function parseNumberList(text) {
  return String(text || '')
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v) && v > 0);
}

function parseNumberValue(text, fallback) {
  const value = Number.parseFloat(String(text || '').trim());
  return Number.isFinite(value) ? value : fallback;
}

function parseNonNegativeNumberValue(text, fallback) {
  const value = parseNumberValue(text, fallback);
  return value >= 0 ? value : fallback;
}

function parseNonNegativeIntegerValue(text, fallback) {
  const value = Number.parseInt(String(text || '').trim(), 10);
  return Number.isInteger(value) && value >= 0 ? value : fallback;
}

function parseJsonObject(text, fieldName) {
  const value = String(text || '').trim();
  if (!value) return {};
  let parsed;
  try {
    parsed = JSON.parse(value);
  } catch (_err) {
    throw new Error(`Invalid JSON in ${fieldName}`, { cause: _err });
  }
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`${fieldName} must be a JSON object`);
  }
  return parsed;
}

function setJsonValidationStatus(el, statusEl, fieldLabel) {
  const text = String(el.value || '').trim();
  if (!text) {
    statusEl.textContent = 'Valid JSON object (empty defaults to {}).';
    statusEl.classList.remove('text-red-600');
    statusEl.classList.add('panel-muted');
    return true;
  }
  try {
    const parsed = JSON.parse(text);
    if (
      parsed === null ||
      typeof parsed !== 'object' ||
      Array.isArray(parsed)
    ) {
      throw new Error('must be JSON object');
    }
    statusEl.textContent = 'Valid JSON object.';
    statusEl.classList.remove('text-red-600');
    statusEl.classList.add('panel-muted');
    return true;
  } catch {
    statusEl.textContent = `${fieldLabel} must be a valid JSON object.`;
    statusEl.classList.remove('panel-muted');
    statusEl.classList.add('text-red-600');
    return false;
  }
}

function updateStructuredCapabilityHint(providerEl, modeEl, targetEl) {
  const provider = providerEl.value;
  const mode = modeEl.value;
  const matrix = configContracts.structured_output_support || {};
  const supports = matrix[provider]?.[mode];
  if (supports === false) {
    targetEl.textContent = `Provider ${provider} does not support ${mode}; backend will deterministically fall back to parse_output.`;
    targetEl.classList.add('text-amber-600');
    targetEl.classList.remove('panel-muted');
    return;
  }
  targetEl.textContent = `Structured mode ${mode} for provider ${provider}.`;
  targetEl.classList.remove('text-amber-600');
  targetEl.classList.add('panel-muted');
}

function pipelinePresetMap() {
  return configContracts.pipeline_presets || PIPELINE_PRESETS_FALLBACK;
}

function applyPipelinePresetSelection(selectedPreset) {
  const map = pipelinePresetMap();
  const presetValues = map[selectedPreset];
  if (!presetValues) return;
  pipelineCaption.checked = !!presetValues.caption;
  pipelineDetect.checked = !!presetValues.detect;
  pipelineTrackMemory.checked = !!presetValues.track_memory;
  pipelinePaintSom.checked = !!presetValues.paint_som;
  pipelineSceneGraph.checked = !!presetValues.scene_graph;
  pipelineQaGeneration.checked = !!presetValues.qa_generation;
  pipelineUpdateSceneMemory.checked = !!presetValues.update_scene_memory;
}

function derivePipelinePreset() {
  const map = pipelinePresetMap();
  const current = {
    caption: !!pipelineCaption.checked,
    detect: !!pipelineDetect.checked,
    track_memory: !!pipelineTrackMemory.checked,
    paint_som: !!pipelinePaintSom.checked,
    scene_graph: !!pipelineSceneGraph.checked,
    qa_generation: !!pipelineQaGeneration.checked,
    update_scene_memory: !!pipelineUpdateSceneMemory.checked,
  };
  const names = Object.keys(map);
  for (const name of names) {
    const cfg = map[name];
    if (
      !!cfg.caption === current.caption &&
      !!cfg.detect === current.detect &&
      !!cfg.track_memory === current.track_memory &&
      !!cfg.paint_som === current.paint_som &&
      !!cfg.scene_graph === current.scene_graph &&
      !!cfg.qa_generation === current.qa_generation &&
      !!cfg.update_scene_memory === current.update_scene_memory
    ) {
      return name;
    }
  }
  return 'custom';
}

function updatePipelineControlsUi() {
  if (!pipelineDetect.checked) {
    pipelineTrackMemory.checked = false;
    pipelinePaintSom.checked = false;
  }
  if (!pipelineSceneGraph.checked || !pipelineTrackMemory.checked) {
    pipelineUpdateSceneMemory.checked = false;
  }
  if (!pipelineSceneGraph.checked) {
    pipelineQaGeneration.checked = false;
  }

  pipelineTrackMemory.disabled = !pipelineDetect.checked;
  pipelinePaintSom.disabled = !pipelineDetect.checked;
  pipelineQaGeneration.disabled = !pipelineSceneGraph.checked;
  pipelineUpdateSceneMemory.disabled =
    !pipelineSceneGraph.checked || !pipelineTrackMemory.checked;

  const summary = [];
  if (
    !pipelineDetect.checked &&
    pipelineSceneGraph.checked &&
    !pipelinePaintSom.checked
  ) {
    summary.push('Direct-image scene graph route is active.');
  }
  if (
    pipelineSceneGraph.checked &&
    !sggEnableVlm.checked &&
    !sggEnableRules.checked &&
    !sggEnableReltr.checked
  ) {
    summary.push('Invalid: at least one scene-graph backend must be enabled.');
  }
  if (
    pipelineSceneGraph.checked &&
    (sggEnableRules.checked || sggEnableReltr.checked) &&
    !pipelineDetect.checked
  ) {
    summary.push('Invalid: Rules and RelTR require detect=true.');
  }
  if (!pipelineTrackMemory.checked) {
    summary.push('Tracking disabled: IDs are frame-local only.');
  }
  if (pipelineCaption.checked && !pipelineDetect.checked) {
    summary.push('Caption-only path active (fast frame description).');
  }
  if (!pipelinePaintSom.checked && pipelineSceneGraph.checked) {
    summary.push('Scene graph uses raw image (no SoM overlay).');
  }
  if (pipelineQaGeneration.checked && !pipelineSceneGraph.checked) {
    summary.push('Invalid: QA generation requires scene_graph=true.');
  }
  pipelineSummary.textContent = summary.length
    ? summary.join(' ')
    : 'Current stage configuration is valid.';

  const derived = derivePipelinePreset();
  if (pipelinePreset.value !== derived) {
    pipelinePreset.value = derived;
  }
}

async function fetchModels() {
  try {
    const res = await fetch('/dashboard/config/get_models');
    const data = await res.json();
    backendSelect.innerHTML = '';
    data.models.forEach((model) => {
      const opt = document.createElement('option');
      opt.value = model;
      opt.textContent = model;
      backendSelect.appendChild(opt);
    });
  } catch (err) {
    console.error('Failed to fetch models:', err);
    backendSelect.innerHTML = '<option>Error loading</option>';
  }
}

async function loadConfig() {
  const res = await fetch('/api/v1/config');
  const data = await res.json();
  const active = data.active;
  const resolved = data.active_resolved || {};
  configContracts = data.contracts || {};

  setThreshold(active.detection.confidence_threshold ?? 0.5);
  backendSelect.value = active.detection.backend || 'rt_detr';
  detectionDevice.value = active.detection.device || 'cuda';
  outputLanguageSelect.value = active.system.output_language || 'default';

  visBbox.checked = !!active.visualization.show_bbox;
  visMask.checked = !!active.visualization.show_mask;
  visPolygon.checked = !!active.visualization.show_polygon;
  visLabels.checked = !!active.visualization.show_labels;
  visLine.value = active.visualization.line_thickness ?? 2;
  visOpacity.value = active.visualization.mask_opacity ?? 0.5;
  visColor.value = active.visualization.color_lookup || 'index';
  visMaskBackend.value = active.visualization.mask_backend || 'grabcut';
  visDevice.value = active.visualization.device || 'cuda';

  storagePersist.checked = !!active.storage?.persist_last_state;
  storageImage.checked = !!active.storage?.store_image;
  storagePath.value =
    active.storage?.last_state_path || 'state/last_state.json';

  const resolvedSceneGraph = resolved.scene_graph || {};
  const resolvedVlm = resolvedSceneGraph.vlm || {};
  vlmSystem.value = resolvedVlm.resolved_system_prompt || '';
  vlmUser.value = resolvedVlm.resolved_user_prompt || '';
  vlmPredicates.value = (resolvedVlm.resolved_ontology?.predicates || []).join(
    '\n',
  );
  vlmObjects.value = (
    active.detection?.ontology ||
    resolved.detection?.resolved_ontology ||
    []
  ).join('\n');
  vlmDevice.value = active.scene_graph?.vlm?.device || '';
  vlmProvider.value = active.scene_graph?.vlm?.provider || 'openai';
  vlmModelId.value = active.scene_graph?.vlm?.model_id || '';
  vlmBaseUrl.value = active.scene_graph?.vlm?.base_url || '';
  vlmApiKeyEnv.value = active.scene_graph?.vlm?.api_key_env || '';
  vlmStructuredMode.value =
    active.scene_graph?.vlm?.structured_output?.mode || 'parse_output';
  vlmStructuredStrict.value = String(
    active.scene_graph?.vlm?.structured_output?.strict ?? true,
  );
  vlmStructuredSchema.value =
    active.scene_graph?.vlm?.structured_schema || 'scene_graph';
  vlmLocalPromptStyle.value =
    active.scene_graph?.vlm?.local_vlm_hints?.prompt_template_style || 'auto';
  vlmLocalImageTokenStrategy.value =
    active.scene_graph?.vlm?.local_vlm_hints?.image_token_strategy || 'auto';
  vlmClientInitKwargs.value = JSON.stringify(
    active.scene_graph?.vlm?.client_init_kwargs || {},
    null,
    2,
  );
  vlmCallKwargs.value = JSON.stringify(
    active.scene_graph?.vlm?.call_kwargs || {},
    null,
    2,
  );
  vlmProvider.dispatchEvent(new Event('change'));

  const resolvedChat = resolved.chat || {};
  chatSystem.value = resolvedChat.resolved_system_prompt || '';
  chatUser.value = resolvedChat.resolved_user_prompt || '';
  chatObjectSystem.value = resolvedChat.resolved_object_system_prompt || '';
  chatObjectUser.value = resolvedChat.resolved_object_user_prompt || '';
  chatDevice.value = active.chat?.device || '';
  chatProvider.value = active.chat?.provider || 'openai';
  chatModelId.value = active.chat?.model_id || '';
  chatBaseUrl.value = active.chat?.base_url || '';
  chatApiKeyEnv.value = active.chat?.api_key_env || '';
  chatClientInitKwargs.value = JSON.stringify(
    active.chat?.client_init_kwargs || {},
    null,
    2,
  );
  chatCallKwargs.value = JSON.stringify(
    active.chat?.call_kwargs || {},
    null,
    2,
  );
  const resolvedCaption = resolved.caption || {};
  captionSystem.value = resolvedCaption.resolved_system_prompt || '';
  captionUser.value = resolvedCaption.resolved_user_prompt || '';
  captionMode.value = active.caption?.mode || 'prompted';
  captionMaxWords.value = active.caption?.max_words ?? '';
  captionDevice.value = active.caption?.device || '';
  captionProvider.value = active.caption?.provider || 'local_hf';
  captionModelId.value = active.caption?.model_id || '';
  captionBaseUrl.value = active.caption?.base_url || '';
  captionApiKeyEnv.value = active.caption?.api_key_env || '';
  captionClientInitKwargs.value = JSON.stringify(
    active.caption?.client_init_kwargs || {},
    null,
    2,
  );
  captionCallKwargs.value = JSON.stringify(
    active.caption?.call_kwargs || {},
    null,
    2,
  );
  const translations = data.translations?.active || {};
  translationsLabels.value = JSON.stringify(
    translations.labels?.cs || {},
    null,
    2,
  );
  translationsAttributes.value = JSON.stringify(
    translations.attributes?.cs || {},
    null,
    2,
  );
  translationsRelations.value = JSON.stringify(
    translations.relations?.cs || {},
    null,
    2,
  );

  sggEnableVlm.checked = active.scene_graph?.vlm?.enabled ?? true;
  sggEnableRules.checked = active.scene_graph?.rules?.enabled ?? true;
  sggEnableReltr.checked = active.scene_graph?.reltr?.enabled ?? false;
  sggRulesJson.value = JSON.stringify(
    active.scene_graph?.rules?.rule_list || [],
    null,
    2,
  );
  reltrCheckpointPath.value = active.scene_graph?.reltr?.checkpoint_path || '';
  reltrDevice.value = active.scene_graph?.reltr?.device || 'cuda';
  reltrThreshold.value = active.scene_graph?.reltr?.threshold ?? 0.3;
  reltrTopk.value = active.scene_graph?.reltr?.topk ?? 100;
  reltrIouMatchThreshold.value =
    active.scene_graph?.reltr?.iou_match_threshold ?? 0.5;

  const controls = active.pipeline_controls || {};
  pipelinePreset.value = controls.preset || 'full';
  pipelineCaption.checked = controls.caption ?? true;
  pipelineDetect.checked = controls.detect ?? true;
  pipelineTrackMemory.checked = controls.track_memory ?? true;
  pipelinePaintSom.checked = controls.paint_som ?? true;
  pipelineSceneGraph.checked = controls.scene_graph ?? true;
  pipelineQaGeneration.checked = controls.qa_generation ?? false;
  pipelineUpdateSceneMemory.checked = controls.update_scene_memory ?? true;
  updatePipelineControlsUi();
  const qaConfig = active.qa_generation || {};
  qaPairsPerUpdate.value = qaConfig.pairs_per_update ?? 2;
  qaPoolMaxEntries.value = qaConfig.pool_max_entries ?? 200;
  const worker = active.worker || {};
  workerEnabled.checked = worker.enabled ?? true;
  workerHost.value = worker.host || '127.0.0.1';
  workerPort.value = worker.port ?? 8765;
  workerIdleTimeout.value = worker.idle_timeout_seconds ?? 600;
  workerIdleCheckInterval.value = worker.idle_check_interval_seconds ?? 2.0;
  workerStartupTimeout.value = worker.startup_timeout_seconds ?? 120.0;
  workerRequestTimeout.value = worker.request_timeout_seconds ?? 180.0;
  workerShutdownGrace.value = worker.shutdown_grace_seconds ?? 15.0;
  workerMaxStartupQueue.value = worker.max_startup_queue ?? 32;
  workerHealthcheckInterval.value = worker.healthcheck_interval_seconds ?? 2.0;
  workerRestartMaxAttempts.value = worker.restart_max_attempts ?? 3;
  workerRestartWindow.value = worker.restart_window_seconds ?? 60;
  workerRestartBackoff.value = (
    worker.restart_backoff_seconds || [1.0, 3.0, 10.0]
  ).join(', ');
  workerCircuitBreakerCooldown.value =
    worker.circuit_breaker_cooldown_seconds ?? 30;
  workerAutoWarmup.checked = !!worker.auto_warmup_on_startup;

  const tracking = active.tracking || {};
  const assoc = tracking.association || {};
  const feat = tracking.feature_extraction || {};
  memoryMaxDormantFrames.value = tracking.max_dormant_frames ?? 30;
  memoryAssocVisualWeight.value = assoc.visual_weight ?? 0.8;
  memoryAssocGeometryWeight.value = assoc.geometry_weight ?? 0.2;
  memoryAssocMatchThreshold.value = assoc.match_threshold ?? 0.4;
  memoryFeatReidModel.value = feat.reid_model || '';
  memoryFeatDevice.value = feat.device || '';
  memoryFeatTargetSize.value = Array.isArray(feat.target_size)
    ? feat.target_size.join(',')
    : '';
  memoryFeatResampling.value = feat.resampling_method || '';
  memoryMaxAgeSeconds.value = tracking.memory_max_age_seconds ?? 60;
  memoryMaxObjects.value = tracking.memory_max_objects ?? 200;
  memoryMaxRelations.value = tracking.memory_max_relations ?? 500;

  const fusion = active.fusion || {};
  fusionPersonBboxMatchThresholdPx.value =
    fusion.person_bbox_match_threshold_px ?? 10.0;
  fusionEstimatedPersonBboxBasePx.value =
    fusion.estimated_person_bbox_base_px ?? 80.0;
  fusionEstimatedPersonBboxMinPx.value =
    fusion.estimated_person_bbox_min_px ?? 40.0;
  fusionEstimatedPersonBboxMaxPx.value =
    fusion.estimated_person_bbox_max_px ?? 200.0;
  fusionAngularYawThresholdRad.value = fusion.angular_yaw_threshold_rad ?? 0.2;
  fusionAngularPitchThresholdRad.value =
    fusion.angular_pitch_threshold_rad ?? 0.15;
  fusionMatchedPersonMinConfidence.value =
    fusion.matched_person_min_confidence ?? 0.85;
  fusionSyntheticPersonConfidence.value =
    fusion.synthetic_person_confidence ?? 0.65;
  fusionPepperBindingMaxMisses.value = fusion.pepper_binding_max_misses ?? 4;

  updateStructuredCapabilityHint(
    vlmProvider,
    vlmStructuredMode,
    vlmStructuredCapability,
  );
  setJsonValidationStatus(
    vlmClientInitKwargs,
    vlmClientInitKwargsStatus,
    'VLM Client Init Kwargs',
  );
  setJsonValidationStatus(
    vlmCallKwargs,
    vlmCallKwargsStatus,
    'VLM Call Kwargs',
  );
  setJsonValidationStatus(
    chatClientInitKwargs,
    chatClientInitKwargsStatus,
    'LLM Client Init Kwargs',
  );
  setJsonValidationStatus(
    chatCallKwargs,
    chatCallKwargsStatus,
    'LLM Call Kwargs',
  );
  setJsonValidationStatus(
    captionClientInitKwargs,
    captionClientInitKwargsStatus,
    'Caption Client Init Kwargs',
  );
  setJsonValidationStatus(
    captionCallKwargs,
    captionCallKwargsStatus,
    'Caption Call Kwargs',
  );
  setJsonValidationStatus(
    translationsLabels,
    translationsLabelsStatus,
    'Translations Labels (cs)',
  );
  setJsonValidationStatus(
    translationsAttributes,
    translationsAttributesStatus,
    'Translations Attributes (cs)',
  );
  setJsonValidationStatus(
    translationsRelations,
    translationsRelationsStatus,
    'Translations Relations (cs)',
  );
}

function buildPatch() {
  const rules = (() => {
    try {
      return sggRulesJson.value.trim() ? JSON.parse(sggRulesJson.value) : [];
    } catch (err) {
      throw new Error('Invalid JSON in SGG rules', { cause: err });
    }
  })();
  if (
    !sggEnableVlm.checked &&
    !sggEnableRules.checked &&
    !sggEnableReltr.checked
  ) {
    throw new Error('At least one scene graph backend must be enabled');
  }

  const parsedVlmClientInitKwargs = parseJsonObject(
    vlmClientInitKwargs.value,
    'VLM Client Init Kwargs',
  );
  const parsedVlmCallKwargs = parseJsonObject(
    vlmCallKwargs.value,
    'VLM Call Kwargs',
  );
  const parsedChatClientInitKwargs = parseJsonObject(
    chatClientInitKwargs.value,
    'LLM Client Init Kwargs',
  );
  const parsedChatCallKwargs = parseJsonObject(
    chatCallKwargs.value,
    'LLM Call Kwargs',
  );
  const parsedCaptionClientInitKwargs = parseJsonObject(
    captionClientInitKwargs.value,
    'Caption Client Init Kwargs',
  );
  const parsedCaptionCallKwargs = parseJsonObject(
    captionCallKwargs.value,
    'Caption Call Kwargs',
  );
  const parsedTranslationLabels = parseJsonObject(
    translationsLabels.value,
    'Translations Labels (cs)',
  );
  const parsedTranslationAttributes = parseJsonObject(
    translationsAttributes.value,
    'Translations Attributes (cs)',
  );
  const parsedTranslationRelations = parseJsonObject(
    translationsRelations.value,
    'Translations Relations (cs)',
  );
  const parsedWorkerRestartBackoff = parseNumberList(
    workerRestartBackoff.value,
  );
  if (!parsedWorkerRestartBackoff.length) {
    throw new Error(
      'Worker Restart Backoff must contain at least one positive number',
    );
  }

  return {
    translations: {
      labels: { cs: parsedTranslationLabels },
      attributes: { cs: parsedTranslationAttributes },
      relations: { cs: parsedTranslationRelations },
    },
    system: {
      output_language: outputLanguageSelect.value,
    },
    detection: {
      backend: backendSelect.value,
      confidence_threshold: parseFloat(input.value),
      device: detectionDevice.value.trim() || 'cuda',
      ontology: parseOntologyList(vlmObjects.value),
    },
    visualization: {
      show_bbox: visBbox.checked,
      show_mask: visMask.checked,
      show_polygon: visPolygon.checked,
      show_labels: visLabels.checked,
      line_thickness: parseInt(visLine.value, 10),
      mask_opacity: parseFloat(visOpacity.value),
      color_lookup: visColor.value,
      mask_backend: visMaskBackend.value || 'grabcut',
      device: visDevice.value.trim() || 'cuda',
    },
    storage: {
      persist_last_state: storagePersist.checked,
      store_image: storageImage.checked,
      last_state_path: storagePath.value,
    },
    tracking: {
      max_dormant_frames: parseInt(memoryMaxDormantFrames.value, 10) || 30,
      memory_max_age_seconds: parseInt(memoryMaxAgeSeconds.value, 10) || 60,
      memory_max_objects: parseInt(memoryMaxObjects.value, 10) || 200,
      memory_max_relations: parseInt(memoryMaxRelations.value, 10) || 500,
      association: {
        visual_weight: parseFloat(memoryAssocVisualWeight.value) || 0.8,
        geometry_weight: parseFloat(memoryAssocGeometryWeight.value) || 0.2,
        match_threshold: parseFloat(memoryAssocMatchThreshold.value) || 0.4,
      },
      feature_extraction: {
        reid_model: memoryFeatReidModel.value.trim() || null,
        device: memoryFeatDevice.value.trim() || null,
        target_size: parseTargetSize(memoryFeatTargetSize.value),
        resampling_method: memoryFeatResampling.value.trim() || null,
      },
    },
    fusion: {
      person_bbox_match_threshold_px: parseNonNegativeNumberValue(
        fusionPersonBboxMatchThresholdPx.value,
        10.0,
      ),
      estimated_person_bbox_base_px: parseNonNegativeNumberValue(
        fusionEstimatedPersonBboxBasePx.value,
        80.0,
      ),
      estimated_person_bbox_min_px: parseNonNegativeNumberValue(
        fusionEstimatedPersonBboxMinPx.value,
        40.0,
      ),
      estimated_person_bbox_max_px: parseNonNegativeNumberValue(
        fusionEstimatedPersonBboxMaxPx.value,
        200.0,
      ),
      angular_yaw_threshold_rad: parseNonNegativeNumberValue(
        fusionAngularYawThresholdRad.value,
        0.2,
      ),
      angular_pitch_threshold_rad: parseNonNegativeNumberValue(
        fusionAngularPitchThresholdRad.value,
        0.15,
      ),
      matched_person_min_confidence: parseNonNegativeNumberValue(
        fusionMatchedPersonMinConfidence.value,
        0.85,
      ),
      synthetic_person_confidence: parseNonNegativeNumberValue(
        fusionSyntheticPersonConfidence.value,
        0.65,
      ),
      pepper_binding_max_misses: parseNonNegativeIntegerValue(
        fusionPepperBindingMaxMisses.value,
        4,
      ),
    },
    scene_graph: {
      vlm: {
        enabled: sggEnableVlm.checked,
        provider: vlmProvider.value,
        model_id: vlmModelId.value.trim(),
        device: vlmDevice.value.trim() || null,
        base_url: vlmBaseUrl.value.trim() || null,
        api_key_env: vlmApiKeyEnv.value.trim() || null,
        client_init_kwargs: parsedVlmClientInitKwargs,
        call_kwargs: parsedVlmCallKwargs,
        structured_output: {
          mode: vlmStructuredMode.value,
          strict: vlmStructuredStrict.value === 'true',
        },
        structured_schema: vlmStructuredSchema.value,
        local_vlm_hints: {
          prompt_template_style: vlmLocalPromptStyle.value,
          image_token_strategy: vlmLocalImageTokenStrategy.value,
        },
        system_prompt: { text: vlmSystem.value },
        user_prompt: { text: vlmUser.value },
        ontology: {
          predicates: parseLines(vlmPredicates.value),
        },
      },
      rules: {
        enabled: sggEnableRules.checked,
        rule_list: rules,
      },
      reltr: {
        enabled: sggEnableReltr.checked,
        checkpoint_path: reltrCheckpointPath.value.trim() || null,
        device: reltrDevice.value.trim() || 'cpu',
        threshold: parseFloat(reltrThreshold.value) || 0.3,
        topk: parseInt(reltrTopk.value, 10) || 100,
        iou_match_threshold: parseFloat(reltrIouMatchThreshold.value) || 0.5,
      },
    },
    chat: {
      provider: chatProvider.value,
      model_id: chatModelId.value.trim(),
      device: chatDevice.value.trim() || null,
      base_url: chatBaseUrl.value.trim() || null,
      api_key_env: chatApiKeyEnv.value.trim() || null,
      client_init_kwargs: parsedChatClientInitKwargs,
      call_kwargs: parsedChatCallKwargs,
      system_prompt: { text: chatSystem.value },
      user_prompt: chatUser.value.trim() ? { text: chatUser.value } : null,
      object_system_prompt: chatObjectSystem.value.trim()
        ? { text: chatObjectSystem.value }
        : null,
      object_user_prompt: chatObjectUser.value.trim()
        ? { text: chatObjectUser.value }
        : null,
    },
    caption: {
      provider: captionProvider.value,
      model_id: captionModelId.value.trim(),
      device: captionDevice.value.trim() || null,
      base_url: captionBaseUrl.value.trim() || null,
      api_key_env: captionApiKeyEnv.value.trim() || null,
      client_init_kwargs: parsedCaptionClientInitKwargs,
      call_kwargs: parsedCaptionCallKwargs,
      mode: captionMode.value,
      max_words: captionMaxWords.value
        ? parseInt(captionMaxWords.value, 10)
        : null,
      system_prompt: { text: captionSystem.value },
      user_prompt: { text: captionUser.value },
    },
    qa_generation: {
      pairs_per_update: parseInt(qaPairsPerUpdate.value, 10) || 2,
      pool_max_entries: parseInt(qaPoolMaxEntries.value, 10) || 200,
    },
    worker: {
      enabled: workerEnabled.checked,
      host: workerHost.value.trim() || '127.0.0.1',
      port: parseInt(workerPort.value, 10) || 8765,
      idle_timeout_seconds: parseInt(workerIdleTimeout.value, 10) || 600,
      idle_check_interval_seconds:
        parseFloat(workerIdleCheckInterval.value) || 2.0,
      startup_timeout_seconds: parseFloat(workerStartupTimeout.value) || 120.0,
      request_timeout_seconds: parseFloat(workerRequestTimeout.value) || 180.0,
      shutdown_grace_seconds: parseFloat(workerShutdownGrace.value) || 15.0,
      max_startup_queue: parseInt(workerMaxStartupQueue.value, 10) || 32,
      healthcheck_interval_seconds:
        parseFloat(workerHealthcheckInterval.value) || 2.0,
      restart_max_attempts: parseInt(workerRestartMaxAttempts.value, 10) || 3,
      restart_window_seconds: parseInt(workerRestartWindow.value, 10) || 60,
      restart_backoff_seconds: parsedWorkerRestartBackoff,
      circuit_breaker_cooldown_seconds:
        parseInt(workerCircuitBreakerCooldown.value, 10) || 30,
      auto_warmup_on_startup: workerAutoWarmup.checked,
    },
    pipeline_controls: {
      preset: pipelinePreset.value,
      caption: pipelineCaption.checked,
      detect: pipelineDetect.checked,
      track_memory: pipelineTrackMemory.checked,
      paint_som: pipelinePaintSom.checked,
      scene_graph: pipelineSceneGraph.checked,
      qa_generation: pipelineQaGeneration.checked,
      update_scene_memory: pipelineUpdateSceneMemory.checked,
    },
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
  applyBtn.classList.add('opacity-60', 'cursor-not-allowed');
  applyBtn.textContent = 'Applying...';

  try {
    const res = await fetch('/api/v1/config', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (res.ok) {
      const data = await res.json();
      showStatusMessage(
        data.reloaded ? 'Applied changes (hard reload)' : 'Applied changes',
      );
      if (data.requires_reload && data.requires_reload.length > 0) {
        reloadWarning.textContent = `Hard changes detected: ${data.requires_reload.join(', ')}`;
        reloadWarning.classList.remove('hidden');
      } else {
        reloadWarning.classList.add('hidden');
      }
    } else {
      let detail = 'Failed to apply config';
      try {
        const payload = await res.json();
        if (payload?.detail) detail = payload.detail;
      } catch (_err) {
        throw new Error('Error Applying config', { cause: _err });
      }
      showStatusMessage(detail, false);
    }
  } catch (err) {
    console.error('Failed to apply config:', err);
    showStatusMessage('Failed to apply config', false);
  } finally {
    isApplyingConfig = false;
    applyBtn.disabled = false;
    applyBtn.classList.remove('opacity-60', 'cursor-not-allowed');
    applyBtn.textContent = prevLabel;
  }
}

async function saveConfig() {
  const res = await fetch('/api/v1/config/save', { method: 'POST' });
  if (res.ok) {
    showStatusMessage('Saved to server/config.yaml');
  } else {
    showStatusMessage('Failed to save config', false);
  }
}

async function reloadConfig() {
  const res = await fetch('/api/v1/config/reload', { method: 'POST' });
  if (res.ok) {
    await loadConfig();
    showStatusMessage('Reloaded saved config');
  } else {
    showStatusMessage('Failed to reload config', false);
  }
}

function downloadConfig(source) {
  const url =
    source === 'saved'
      ? '/api/v1/config/download?source=saved'
      : '/api/v1/config/download';
  window.location.href = url;
}

async function uploadConfig(file) {
  uploadProgress.classList.remove('hidden');
  uploadProgressBar.style.width = '0%';
  uploadProgressText.textContent = '0%';

  const form = new FormData();
  form.append('file', file);

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/v1/config/upload', true);

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
        showStatusMessage('Uploaded config applied (in-memory)');
      } else {
        showStatusMessage('Failed to upload config', false);
      }
      setTimeout(() => uploadProgress.classList.add('hidden'), 500);
    }
  };

  xhr.send(form);
}

slider.addEventListener('input', () => {
  input.value = slider.value;
});

input.addEventListener('change', () => {
  let val = parseFloat(input.value);
  if (Number.isNaN(val)) val = 0;
  if (val < 0) val = 0;
  if (val > 1) val = 1;
  input.value = val.toFixed(2);
  slider.value = val.toFixed(2);
});

vlmProvider.addEventListener('change', () => {
  updateStructuredCapabilityHint(
    vlmProvider,
    vlmStructuredMode,
    vlmStructuredCapability,
  );
});
vlmStructuredMode.addEventListener('change', () => {
  updateStructuredCapabilityHint(
    vlmProvider,
    vlmStructuredMode,
    vlmStructuredCapability,
  );
});
vlmClientInitKwargs.addEventListener('input', () =>
  setJsonValidationStatus(
    vlmClientInitKwargs,
    vlmClientInitKwargsStatus,
    'VLM Client Init Kwargs',
  ),
);
vlmCallKwargs.addEventListener('input', () =>
  setJsonValidationStatus(
    vlmCallKwargs,
    vlmCallKwargsStatus,
    'VLM Call Kwargs',
  ),
);
chatClientInitKwargs.addEventListener('input', () =>
  setJsonValidationStatus(
    chatClientInitKwargs,
    chatClientInitKwargsStatus,
    'LLM Client Init Kwargs',
  ),
);
chatCallKwargs.addEventListener('input', () =>
  setJsonValidationStatus(
    chatCallKwargs,
    chatCallKwargsStatus,
    'LLM Call Kwargs',
  ),
);
captionClientInitKwargs.addEventListener('input', () =>
  setJsonValidationStatus(
    captionClientInitKwargs,
    captionClientInitKwargsStatus,
    'Caption Client Init Kwargs',
  ),
);
captionCallKwargs.addEventListener('input', () =>
  setJsonValidationStatus(
    captionCallKwargs,
    captionCallKwargsStatus,
    'Caption Call Kwargs',
  ),
);
translationsLabels.addEventListener('input', () =>
  setJsonValidationStatus(
    translationsLabels,
    translationsLabelsStatus,
    'Translations Labels (cs)',
  ),
);
translationsAttributes.addEventListener('input', () =>
  setJsonValidationStatus(
    translationsAttributes,
    translationsAttributesStatus,
    'Translations Attributes (cs)',
  ),
);
translationsRelations.addEventListener('input', () =>
  setJsonValidationStatus(
    translationsRelations,
    translationsRelationsStatus,
    'Translations Relations (cs)',
  ),
);
pipelinePreset.addEventListener('change', () => {
  if (pipelinePreset.value !== 'custom') {
    applyPipelinePresetSelection(pipelinePreset.value);
  }
  updatePipelineControlsUi();
});
pipelineCaption.addEventListener('change', updatePipelineControlsUi);
pipelineDetect.addEventListener('change', updatePipelineControlsUi);
pipelineTrackMemory.addEventListener('change', updatePipelineControlsUi);
pipelinePaintSom.addEventListener('change', updatePipelineControlsUi);
pipelineSceneGraph.addEventListener('change', updatePipelineControlsUi);
pipelineQaGeneration.addEventListener('change', updatePipelineControlsUi);
pipelineUpdateSceneMemory.addEventListener('change', updatePipelineControlsUi);
sggEnableVlm.addEventListener('change', updatePipelineControlsUi);
sggEnableRules.addEventListener('change', updatePipelineControlsUi);
sggEnableReltr.addEventListener('change', updatePipelineControlsUi);

applyBtn.addEventListener('click', applyConfig);
saveBtn.addEventListener('click', saveConfig);
reloadBtn.addEventListener('click', reloadConfig);
downloadBtn.addEventListener('click', () => downloadConfig('active'));
downloadSavedBtn.addEventListener('click', () => downloadConfig('saved'));

uploadInput.addEventListener('change', (event) => {
  const file = event.target.files[0];
  if (file) uploadConfig(file);
});

fetchModels();
loadConfig();
