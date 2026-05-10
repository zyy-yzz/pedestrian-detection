import { create } from 'zustand';
import type { HistoryEntry } from '../types';

interface HistoryState {
  jobs: HistoryEntry[];
  page: number;
  total: number;
  setJobs: (jobs: HistoryEntry[]) => void;
  setPage: (page: number) => void;
  setTotal: (total: number) => void;
}

export const useHistoryStore = create<HistoryState>((set) => ({
  jobs: [],
  page: 1,
  total: 0,
  setJobs: (jobs) => set({ jobs }),
  setPage: (page) => set({ page }),
  setTotal: (total) => set({ total }),
}));
