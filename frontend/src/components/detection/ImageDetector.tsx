import { useState } from 'react';
import DropZone from '../ui/DropZone';
import EnhancementToolbar from './EnhancementToolbar';
import DetectionOverlay from './DetectionOverlay';
import CompareView from './CompareView';
import StatsPanel, { CompareStats } from './StatsPanel';
import ConfidenceBadge from '../ui/ConfidenceBadge';
import ExportButton from '../ui/ExportButton';
import { useImageDetection } from '../../hooks/useDetection';
import { useDetectionStore } from '../../store/detectionStore';

export default function ImageDetector() {
  const [preview, setPreview] = useState<string | null>(null);
  const { submitImage } = useImageDetection();
  const {
    currentJob, compareData, isLoading, error, config, reset: resetStore,
  } = useDetectionStore();

  const handleDrop = async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(file);
    await submitImage(file);
  };

  // Derive data for rendering
  const isCompare = config.compare && compareData != null;
  const detections = isCompare
    ? compareData.enhanced.detections
    : currentJob?.detections ?? [];
  const avgConf = detections.length
    ? detections.reduce((s, d) => s + d.confidence, 0) / detections.length
    : undefined;

  return (
    <div className="animate-fade-in space-y-6">
      {/* Toolbar */}
      <EnhancementToolbar />

      {/* Upload zone or results */}
      {preview == null ? (
        <DropZone
          onDrop={handleDrop}
          accept={{ 'image/*': ['.jpg', '.jpeg', '.png', '.bmp'] }}
        />
      ) : (
        <div className="space-y-6">
          {/* Loading state */}
          {isLoading && (
            <div className="flex h-80 items-center justify-center rounded-xl border border-[#1e1e3a] bg-[#0d0d1a]">
              <div className="flex flex-col items-center gap-3">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-green-500 border-t-transparent" />
                <p className="mono text-sm text-slate-400">
                  {config.compare ? 'Running comparison...' : 'Processing...'}
                </p>
              </div>
            </div>
          )}

          {/* Error outside results condition so it shows on first failure */}
          {!isLoading && error && !currentJob && !compareData && (
            <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 mono text-sm text-red-400">
              {error}
            </div>
          )}

          {/* Results */}
          {!isLoading && (currentJob || compareData) && (
            <>
              {/* Compare mode */}
              {isCompare && (
                <CompareView
                  imageUrl={preview}
                  enhanced={compareData.enhanced}
                  vanilla={compareData.vanilla}
                />
              )}

              {/* Single mode */}
              {!isCompare && currentJob && (
                <div className="grid gap-6 lg:grid-cols-5">
                  <div className="lg:col-span-3">
                    <DetectionOverlay
                      imageUrl={preview}
                      detections={detections.map((d) => ({
                        id: d.id,
                        bbox: d.bbox,
                        confidence: d.confidence,
                        class_name: d.class_name || 'pedestrian',
                      }))}
                    />
                  </div>
                  <div className="lg:col-span-2 space-y-4">
                    <StatsPanel
                      pedestrianCount={currentJob.pedestrian_count ?? 0}
                      processingTimeMs={currentJob.processing_time_ms}
                      avgConfidence={avgConf}
                    />
                    {detections.length > 0 && (
                      <div className="space-y-2 rounded-xl border border-[#1e1e3a] bg-[#0d0d1a] p-4">
                        <h3 className="mono text-xs font-semibold uppercase tracking-wider text-slate-400">
                          Detections ({detections.length})
                        </h3>
                        <div className="max-h-64 space-y-1.5 overflow-y-auto">
                          {detections.map((det, i) => (
                            <div key={i} className="flex items-center justify-between rounded-lg bg-[#06060c] px-3 py-2">
                              <span className="mono text-xs text-slate-300">Pedestrian #{i + 1}</span>
                              <ConfidenceBadge confidence={det.confidence} />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                    <ExportButton imageUrl={preview} detections={detections} />
                  </div>
                </div>
              )}

              {/* Compare stats + export */}
              {isCompare && (
                <div className="space-y-4">
                  <CompareStats
                    enhanced={{
                      pedestrian_count: compareData.enhanced.pedestrian_count,
                      processing_time_ms: compareData.enhanced.processing_time_ms,
                    }}
                    vanilla={{
                      pedestrian_count: compareData.vanilla.pedestrian_count,
                      processing_time_ms: compareData.vanilla.processing_time_ms,
                    }}
                    comparison={compareData.comparison}
                  />
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="mono text-xs text-slate-500 mb-1">Export Enhanced</p>
                      <ExportButton imageUrl={preview} detections={compareData.enhanced.detections} />
                    </div>
                    <div>
                      <p className="mono text-xs text-slate-500 mb-1">Export Vanilla</p>
                      <ExportButton imageUrl={preview} detections={compareData.vanilla.detections} />
                    </div>
                  </div>
                </div>
              )}

              {/* Error */}
              {error && (
                <div className="rounded-lg border border-red-500/20 bg-red-500/10 px-4 py-3 mono text-sm text-red-400">
                  {error}
                </div>
              )}

              {/* New Detection */}
              <button
                onClick={() => {
                  setPreview(null);
                  resetStore();
                }}
                className="w-full rounded-lg border border-[#1e1e3a] bg-[#0d0d1a] px-4 py-2.5 text-sm font-medium
                           text-slate-400 transition-colors hover:border-slate-500 hover:text-white"
              >
                New Detection
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}