export function createDashboardSocket(onMessage) {
  const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${location.host}/dashboard/events`);
  ws.onmessage = (event) => {
    let data = null;
    try {
      data = JSON.parse(event.data);
    } catch {
      return;
    }
    onMessage?.(data);
  };
  return ws;
}
