import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useHistory } from '../hooks/useHistory';
import { useHistoryStore } from '../store/historyStore';

export default function JobHistory() {
  const { fetchPage, deleteJob } = useHistory();
  const { jobs, total } = useHistoryStore();
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(false);
  const pageSize = 10;

  useEffect(() => {
    setLoading(true);
    fetchPage(page, pageSize).finally(() => setLoading(false));
  }, [page, fetchPage]);

  const totalPages = Math.max(1, Math.ceil(total / pageSize));

  const statusBadge = (status: string) => {
    const map: Record<string, string> = {
      completed: 'bg-green-500/10 text-green-400 border-green-500/20',
      processing: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
      queued: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
      failed: 'bg-red-500/10 text-red-400 border-red-500/20',
    };
    return map[status] || 'bg-slate-500/10 text-slate-400 border-slate-500/20';
  };

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight text-white">
          Job <span className="text-green-400">History</span>
        </h1>
        <p className="mt-1 text-sm text-slate-500 mono">
          {total} total detection {total === 1 ? 'job' : 'jobs'}
        </p>
      </div>

      {loading ? (
        <div className="flex h-40 items-center justify-center">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-green-500 border-t-transparent" />
        </div>
      ) : jobs.length === 0 ? (
        <div className="flex h-40 flex-col items-center justify-center rounded-xl border border-[#1e1e3a] bg-[#0d0d1a]">
          <p className="text-sm text-slate-500">No detection jobs yet.</p>
          <Link to="/detect" className="mt-2 text-sm font-medium text-green-400 hover:text-green-300">
            Start a new detection &rarr;
          </Link>
        </div>
      ) : (
        <>
          <div className="overflow-hidden rounded-xl border border-[#1e1e3a]">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[#1e1e3a] bg-[#0d0d1a]">
                  {['Job ID', 'Type', 'File', 'Status', 'Count', 'Date'].map((h) => (
                    <th key={h} className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-slate-500">
                      {h}
                    </th>
                  ))}
                  <th className="px-4 py-3" />
                </tr>
              </thead>
              <tbody>
                {jobs.map((job, i) => (
                  <tr
                    key={job.job_id}
                    className={`border-b border-[#1e1e3a]/50 transition-colors hover:bg-[#0d0d1a]/50 ${
                      i % 2 === 0 ? 'bg-[#06060c]' : 'bg-[#0a0a14]'
                    }`}
                  >
                    <td className="px-4 py-3">
                      <Link
                        to={`/results/${job.job_id}`}
                        className="mono text-xs text-green-400 hover:text-green-300"
                      >
                        {job.job_id.slice(0, 12)}&hellip;
                      </Link>
                    </td>
                    <td className="px-4 py-3 mono text-xs uppercase text-slate-400">{job.input_type}</td>
                    <td className="px-4 py-3 text-sm text-slate-300 max-w-[160px] truncate">
                      {job.original_filename || '—'}
                    </td>
                    <td className="px-4 py-3">
                      <span className={`inline-block rounded border px-2 py-0.5 text-xs font-medium mono ${statusBadge(job.status)}`}>
                        {job.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 mono text-sm text-slate-300">
                      {job.pedestrian_count ?? '—'}
                    </td>
                    <td className="px-4 py-3 mono text-xs text-slate-500">
                      {job.created_at ? new Date(job.created_at).toLocaleDateString() : '—'}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <button
                        onClick={async () => {
                          await deleteJob(job.job_id);
                          fetchPage(page, pageSize);
                        }}
                        className="mono text-xs text-slate-600 transition-colors hover:text-red-400"
                      >
                        DEL
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {totalPages > 1 && (
            <div className="mt-4 flex items-center justify-center gap-2">
              <button
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                className="rounded-lg border border-[#1e1e3a] px-3 py-1.5 text-sm text-slate-400 transition-colors hover:border-slate-500 disabled:opacity-30"
              >
                Prev
              </button>
              <span className="mono text-sm text-slate-400">
                {page} / {totalPages}
              </span>
              <button
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                className="rounded-lg border border-[#1e1e3a] px-3 py-1.5 text-sm text-slate-400 transition-colors hover:border-slate-500 disabled:opacity-30"
              >
                Next
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}
