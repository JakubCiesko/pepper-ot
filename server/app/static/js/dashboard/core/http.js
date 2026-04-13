export async function readJsonSafe(response, fallback = null) {
  try {
    return await response.json();
  } catch {
    return fallback;
  }
}

export async function requestJson(
  url,
  options = {},
  fallbackError = 'Request failed',
) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = fallbackError;
    const payload = await readJsonSafe(response, null);
    if (payload?.detail) detail = payload.detail;
    throw new Error(detail);
  }
  return readJsonSafe(response, {});
}
