import { createDashboardSocket } from './core/ws.js';
import {
  handleConversationWsMessage,
  initConversationPanel,
} from './features/conversation/index.js';
import {
  getActiveFrameSnapshot,
  handleLiveWsMessage,
  handleMemoryWsMessage,
  initLivePanel,
} from './features/live/index.js';
import { initMemoryPanel } from './features/memory/index.js';
import { initSceneGraphPanel } from './features/scene_graph/index.js';
import { initUiShell } from './features/ui_shell/index.js';
import './features/config/index.js';

function handleWsMessage(data) {
  if (!data) return;
  if (data.type === 'chat_message') {
    handleConversationWsMessage(data);
    return;
  }
  if (data.type === 'memory') {
    handleMemoryWsMessage(data);
    return;
  }
  handleLiveWsMessage(data);
}

function init() {
  initUiShell();
  initMemoryPanel();
  initSceneGraphPanel();
  initLivePanel();
  initConversationPanel({ getActiveFrameSnapshot });
  createDashboardSocket(handleWsMessage);
}

init();
