import { useCallback } from 'react';
import { useDetectionStore } from '../store/detectionStore';
import apiClient from '../api/client';
import type { CompareDetectionResponse } from '../types';

export function useImageDetection() {
  const setJob = useDetectionStore((s) => s.setJob);
  const setCompareData = useDetectionStore((s) => s.setCompareData);
  const setLoading = useDetectionStore((s) => s.setLoading);
  const setError = useDetectionStore((s) => s.setError);
  const config = useDetectionStore((s) => s.config);

  const submitImage = useCallback(
    async (file: File) => {
      setLoading(true);
      setError(null);
      try {
        const form = new FormData();
        form.append('file', file);
        form.append('conf_threshold', String(config.confThreshold));
        form.append('iou_threshold', String(config.iouThreshold));
        form.append('enhancements', config.enhancements.join(','));
        form.append('compare', String(config.compare));

        const { data } = await apiClient.post('/detect/image', form);

        if (config.compare && data.comparison) {
          setCompareData(data as CompareDetectionResponse);
        } else {
          setJob(data);
        }
      } catch (err: unknown) {
        const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ?? 'Detection failed';
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [setJob, setCompareData, setLoading, setError, config]
  );

  return { submitImage };
}