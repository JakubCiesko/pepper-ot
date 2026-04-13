import { bindMemoryCrud, refreshMemory } from './actions.js';
import { getMemoryDomRefs } from './dom_refs.js';
import { renderMemory } from './render.js';

const dom = getMemoryDomRefs();

export function initMemoryPanel() {
  bindMemoryCrud(dom);
  void refreshMemory(dom).catch((err) => {
    console.error('Failed to initialize memory panel', err);
  });
}

export function renderMemoryPanel(mem) {
  renderMemory(dom, mem);
}

export async function refreshMemoryPanel() {
  return refreshMemory(dom);
}
