export function getMemoryDomRefs() {
  return {
    memoryContainer: document.getElementById('memory-content'),
    memoryRaw: document.getElementById('memory-raw'),
    memoryEditorStatus: document.getElementById('memory-editor-status'),
    memObjId: document.getElementById('mem-obj-id'),
    memObjLabel: document.getElementById('mem-obj-label'),
    memObjBbox: document.getElementById('mem-obj-bbox'),
    memObjAttrs: document.getElementById('mem-obj-attrs'),
    memObjCreate: document.getElementById('mem-obj-create'),
    memObjUpdate: document.getElementById('mem-obj-update'),
    memObjDelete: document.getElementById('mem-obj-delete'),
    memRelSubject: document.getElementById('mem-rel-subject'),
    memRelObject: document.getElementById('mem-rel-object'),
    memRelPredicate: document.getElementById('mem-rel-predicate'),
    memRelCount: document.getElementById('mem-rel-count'),
    memRelCreate: document.getElementById('mem-rel-create'),
    memRelUpdate: document.getElementById('mem-rel-update'),
    memRelDelete: document.getElementById('mem-rel-delete'),
    memRefresh: document.getElementById('mem-refresh'),
    memReset: document.getElementById('mem-reset'),
  };
}
