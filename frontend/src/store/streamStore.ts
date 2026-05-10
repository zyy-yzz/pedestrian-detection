import { create } from 'zustand';

interface StreamState {
  isConnected: boolean;
  liveCount: number;
  latency: number;
  setConnected: (connected: boolean) => void;
  setLiveCount: (count: number) => void;
  setLatency: (ms: number) => void;
}

export const useStreamStore = create<StreamState>((set) => ({
  isConnected: false,
  liveCount: 0,
  latency: 0,
  setConnected: (isConnected) => set({ isConnected }),
  setLiveCount: (liveCount) => set({ liveCount }),
  setLatency: (latency) => set({ latency }),
}));
