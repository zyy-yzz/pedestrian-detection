import { useState } from 'react';
import DropZone from '../ui/DropZone';

export default function VideoDetector() {
  const [shownHint, setShownHint] = useState(false);

  return (
    <div className="animate-fade-in space-y-6">
      <DropZone
        onDrop={() => setShownHint(true)}
        accept={{ 'video/*': ['.mp4', '.avi', '.mov'] }}
        label="Drop video here or click to browse"
      />
      {shownHint && (
        <div className="rounded-lg border border-amber-500/20 bg-amber-500/10 px-4 py-2 mono text-xs text-amber-400">
          Video upload received. This feature will be available in the next release.
        </div>
      )}
      <div className="rounded-xl border border-[#1e1e3a] bg-[#0d0d1a] p-8 text-center">
        <div className="mb-4 flex justify-center">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#555578" strokeWidth="1.5" strokeLinecap="round">
            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18" />
            <line x1="7" y1="2" x2="7" y2="22" />
            <line x1="17" y1="2" x2="17" y2="22" />
            <line x1="2" y1="12" x2="22" y2="12" />
            <line x1="2" y1="7" x2="7" y2="7" />
            <line x1="2" y1="17" x2="7" y2="17" />
            <line x1="17" y1="7" x2="22" y2="7" />
            <line x1="17" y1="17" x2="22" y2="17" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-slate-300">Video Processing</h3>
        <p className="mt-1 text-sm text-slate-500 mono">
          Video upload and frame-by-frame detection will be available soon.
        </p>
      </div>
    </div>
  );
}
