import { create } from 'zustand';
import type { DetectionJob, CompareDetectionResponse, CompareResult, DetectionConfig, EnhancementOption } from '../types';

interface DetectionState {
  currentJob: DetectionJob | null;
  compareData: CompareDetectionResponse | null;
  isLoading: boolean;
  error: string | null;
  config: DetectionConfig;
  setJob: (job: DetectionJob) => void;
  setCompareData: (data: CompareDetectionResponse) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setConfig: (config: Partial<DetectionConfig>) => void;
  toggleEnhancement: (enh: EnhancementOption) => void;
  reset: () => void;
}

const DEFAULT_CONFIG: DetectionConfig = {
  confThreshold: 0.25,
  iouThreshold: 0.45,
  enhancements: ['clahe', 'denoise'],
  compare: false,
};

export const useDetectionStore = create<DetectionState>((set) => ({
  currentJob: null,
  compareData: null,
  isLoading: false,
  error: null,
  config: { ...DEFAULT_CONFIG },

  setJob: (job) => set({ currentJob: job, error: null }),
  setCompareData: (compareData) => set({ compareData, error: null }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
  setConfig: (partial) => set((s) => ({ config: { ...s.config, ...partial } })),
  toggleEnhancement: (enh) =>
    set((s) => {
      const current = s.config.enhancements;
      const next = current.includes(enh)
        ? current.filter((e) => e !== enh)
        : [...current, enh];
      return { config: { ...s.config, enhancements: next } };
    }),
  reset: () => set({
    currentJob: null,
    compareData: null,
    isLoading: false,
    error: null,
  }),
}));