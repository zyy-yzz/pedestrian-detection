import { useEffect, useRef, useCallback, useState } from 'react';
import { useStreamStore } from '../store/streamStore';
import type { StreamFrame } from '../types';

const MAX_RECONNECT_DELAY = 16000;

export function useStream() {
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);
  const [isReady, setIsReady] = useState(false);
  const setConnected = useStreamStore((s) => s.setConnected);
  const setLiveCount = useStreamStore((s) => s.setLiveCount);
  const setLatency = useStreamStore((s) => s.setLatency);

  const connect = useCallback(() => {
    const token = localStorage.getItem('access_token');
    if (!token) return;
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const base = import.meta.env.VITE_WS_URL || (import.meta.env.DEV
      ? 'ws://localhost:8000'
      : `${protocol}://${window.location.host}/ws`);
    const ws = new WebSocket(`${base}/stream?token=${token}`);

    ws.onopen = () => {
      reconnectAttemptRef.current = 0;
      setIsReady(true);
      setConnected(true);
    };

    ws.onclose = () => {
      setIsReady(false);
      setConnected(false);
      reconnect();
    };

    ws.onmessage = (event) => {
      try {
        const frame: StreamFrame = JSON.parse(event.data);
        setLiveCount(frame.pedestrian_count);
        setLatency(frame.inference_ms);
      } catch {
        // Malformed frame — ignore, let the stream continue
      }
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [setConnected, setLiveCount, setLatency]);

  const reconnect = useCallback(() => {
    if (reconnectTimerRef.current) return;
    const delay = Math.min(1000 * 2 ** reconnectAttemptRef.current, MAX_RECONNECT_DELAY);
    reconnectAttemptRef.current += 1;
    reconnectTimerRef.current = setTimeout(() => {
      reconnectTimerRef.current = null;
      connect();
    }, delay);
  }, [connect]);

  const sendFrame = useCallback((blob: Blob) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(blob);
    }
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.onclose = null;
      wsRef.current.close();
      wsRef.current = null;
    }
    setIsReady(false);
    setConnected(false);
  }, [setConnected]);

  useEffect(() => {
    return () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
      }
      if (wsRef.current) {
        wsRef.current.onclose = null;
        wsRef.current.close();
      }
    };
  }, []);

  return { connect, sendFrame, disconnect, isReady };
}