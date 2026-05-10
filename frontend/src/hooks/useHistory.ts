import { useCallback } from 'react';
import { useHistoryStore } from '../store/historyStore';
import apiClient from '../api/client';

export function useHistory() {
  const setJobs = useHistoryStore((s) => s.setJobs);
  const setTotal = useHistoryStore((s) => s.setTotal);

  const fetchPage = useCallback(
    async (page: number = 1, limit: number = 10) => {
      const { data } = await apiClient.get('/history', { params: { page, page_size: limit } });
      setJobs(data.jobs);
      setTotal(data.total);
    },
    [setJobs, setTotal]
  );

  const deleteJob = useCallback(async (jobId: string) => {
    await apiClient.delete(`/history/${jobId}`);
  }, []);

  return { fetchPage, deleteJob };
}
