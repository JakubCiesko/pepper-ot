export function secondsAgo(ts) {
  if (!ts) return 'n/a';
  const now = Date.now() / 1000;
  const delta = now - ts;
  return `${delta.toFixed(1)}s ago`;
}

export function parseBboxCsv(value) {
  const parts = String(value || '')
    .split(',')
    .map((v) => Number(v.trim()))
    .filter((v) => Number.isFinite(v));
  return parts.length === 4 ? parts : null;
}

export function parseCommaList(value) {
  return String(value || '')
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
}
