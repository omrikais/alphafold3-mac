/** WebSocket client with automatic reconnection. */

import type { WSMessage } from "./types";

export type WSCallback = (msg: WSMessage) => void;

export function connectJobProgress(
  jobId: string,
  onMessage: WSCallback,
  onClose?: () => void,
): { close: () => void } {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL || "";
  const apiKey = process.env.NEXT_PUBLIC_API_KEY || "";
  let url: string;
  if (apiUrl) {
    // Development: connect to explicit backend URL
    const wsBase = apiUrl.replace(/^http/, "ws");
    url = `${wsBase}/api/jobs/${jobId}/progress`;
  } else {
    // Production: same origin
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    url = `${protocol}//${window.location.host}/api/jobs/${jobId}/progress`;
  }
  // M-10: Pass API key as query param for WebSocket auth
  if (apiKey) {
    url += `?token=${encodeURIComponent(apiKey)}`;
  }

  let ws: WebSocket | null = null;
  let closed = false;
  let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;

  function connect() {
    if (closed) return;

    ws = new WebSocket(url);

    ws.onmessage = (event) => {
      try {
        const msg: WSMessage = JSON.parse(event.data);
        if (msg.type === "ping") return; // ignore keepalive
        onMessage(msg);

        // Stop reconnecting on terminal messages
        if (msg.type === "completed" || msg.type === "failed" || msg.type === "cancelled") {
          closed = true;
          ws?.close();
        }
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      if (!closed) {
        // Auto-reconnect after 2s
        reconnectTimeout = setTimeout(connect, 2000);
      } else {
        onClose?.();
      }
    };

    ws.onerror = () => {
      ws?.close();
    };
  }

  connect();

  return {
    close() {
      closed = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      ws?.close();
    },
  };
}
