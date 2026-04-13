import { bindMemoryCrud, refreshMemory } from './actions.js';
import { getMemoryDomRefs } from './dom_refs.js';
import { renderMemory } from './render.js';

const dom = getMemoryDomRefs();

export function initMemoryPanel() {
  bindMemoryCrud(dom);
}

export function renderMemoryPanel(mem) {
  renderMemory(dom, mem);
}

export async function refreshMemoryPanel() {
  return refreshMemory(dom);
}
