import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import apiClient from '../api/client';
import type { DetectionJob } from '../types';
import StatsPanel from '../components/detection/StatsPanel';
import ConfidenceBadge from '../components/ui/ConfidenceBadge';

export default function ResultDetail() {
  const { id } = useParams<{ id: string }>();
  const [job, setJob] = useState<DetectionJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    apiClient.get(`/results/${id}`)
      .then(({ data }) => { setJob(data); setError(null); })
      .catch((err) => {
        setJob(null);
        setError(err.response?.status === 404
          ? 'Job not found'
          : err.response?.data?.detail ?? err.message ?? 'Failed to load job');
      })
      .finally(() => setLoading(false));
  }, [id]);

  if (loading) {
    return (
      <div className="flex h-96 items-center justify-center">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-green-500 border-t-transparent" />
      </div>
    );
  }

  if (error || !job) {
    return (
      <div className="mx-auto max-w-3xl px-4 py-20 text-center">
        <h2 className="text-xl font-bold text-white">{error || 'Job not found'}</h2>
        <Link to="/history" className="mt-3 inline-block text-sm text-green-400 hover:text-green-300">
          &larr; Back to history
        </Link>
      </div>
    );
  }

  const detections = job.detections ?? [];
  const avgConf = detections.length
    ? detections.reduce((s, d) => s + d.confidence, 0) / detections.length
    : undefined;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <Link to="/history" className="mono text-xs text-slate-500 transition-colors hover:text-slate-300">
        &larr; History
      </Link>

      <div className="mt-4 mb-8">
        <h1 className="text-2xl font-bold tracking-tight text-white">
          Job <span className="text-green-400 mono text-lg">{job.job_id.slice(0, 8)}&hellip;</span>
        </h1>
        <p className="mt-1 text-sm text-slate-500 mono">
          {job.created_at ? new Date(job.created_at).toLocaleString() : ''}
        </p>
      </div>

      <div className="mb-8">
        <StatsPanel
          pedestrianCount={job.pedestrian_count ?? 0}
          processingTimeMs={job.processing_time_ms}
          avgConfidence={avgConf}
        />
      </div>

      {detections.length > 0 && (
        <div className="space-y-3 rounded-xl border border-[#1e1e3a] bg-[#0d0d1a] p-6">
          <h3 className="mono text-sm font-semibold uppercase tracking-wider text-slate-400">
            Detections ({detections.length})
          </h3>
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {detections.map((det, i) => (
              <div
                key={det.id ?? i}
                className="flex items-center justify-between rounded-lg bg-[#06060c] px-4 py-3"
              >
                <div className="flex items-center gap-3">
                  <span className="flex h-7 w-7 items-center justify-center rounded-md bg-[#1e1e3a] text-xs font-semibold text-slate-300 mono">
                    {i + 1}
                  </span>
                  <div>
                    <p className="text-sm font-medium text-slate-200">Pedestrian</p>
                    <p className="mono text-xs text-slate-500">
                      [{det.bbox.x1.toFixed(0)}, {det.bbox.y1.toFixed(0)}] — [{det.bbox.x2.toFixed(0)}, {det.bbox.y2.toFixed(0)}]
                    </p>
                  </div>
                </div>
                <ConfidenceBadge confidence={det.confidence} />
              </div>
            ))}
          </div>
        </div>
      )}

      {job.status === 'failed' && (
        <div className="mt-6 rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 mono text-sm text-red-400">
          {job.error_message || 'Job failed with unknown error'}
        </div>
      )}
    </div>
  );
}
