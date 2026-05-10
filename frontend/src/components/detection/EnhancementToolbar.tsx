import { useDetectionStore } from '../../store/detectionStore';
import { ALL_ENHANCEMENTS, ENHANCEMENT_LABELS } from '../../types';
import type { EnhancementOption } from '../../types';

export default function EnhancementToolbar() {
  const config = useDetectionStore((s) => s.config);
  const setConfig = useDetectionStore((s) => s.setConfig);
  const toggleEnhancement = useDetectionStore((s) => s.toggleEnhancement);

  return (
    <div className="rounded-xl border border-[#1e1e3a] bg-[#0d0d1a] p-4 space-y-4">
      {/* Confidence & IoU sliders */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-1">
          <label className="mono text-xs text-slate-400 flex justify-between">
            <span>Confidence</span>
            <span className="text-green-400">{config.confThreshold.toFixed(2)}</span>
          </label>
          <input
            type="range" min="0.05" max="0.95" step="0.05"
            value={config.confThreshold}
            onChange={(e) => setConfig({ confThreshold: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-[#1e1e3a] rounded-lg appearance-none cursor-pointer
                       accent-green-500"
          />
        </div>
        <div className="space-y-1">
          <label className="mono text-xs text-slate-400 flex justify-between">
            <span>IoU</span>
            <span className="text-green-400">{config.iouThreshold.toFixed(2)}</span>
          </label>
          <input
            type="range" min="0.1" max="0.9" step="0.05"
            value={config.iouThreshold}
            onChange={(e) => setConfig({ iouThreshold: parseFloat(e.target.value) })}
            className="w-full h-1.5 bg-[#1e1e3a] rounded-lg appearance-none cursor-pointer
                       accent-green-500"
          />
        </div>
      </div>

      {/* Enhancement toggles */}
      <div>
        <p className="mono text-xs font-semibold uppercase tracking-wider text-slate-500 mb-2">
          Enhancement Methods
        </p>
        <div className="grid grid-cols-3 gap-1.5">
          {ALL_ENHANCEMENTS.map((enh: EnhancementOption) => (
            <button
              key={enh}
              onClick={() => toggleEnhancement(enh)}
              className={`px-2.5 py-1.5 rounded-md text-xs font-medium transition-all border ${
                config.enhancements.includes(enh)
                  ? 'bg-green-500/15 border-green-500/40 text-green-300'
                  : 'bg-[#06060c] border-[#1e1e3a] text-slate-500 hover:border-slate-600'
              }`}
            >
              {ENHANCEMENT_LABELS[enh].split(' ')[0]}
            </button>
          ))}
        </div>
      </div>

      {/* Compare toggle */}
      <div className="flex items-center justify-between pt-2 border-t border-[#1e1e3a]">
        <span className="mono text-xs font-semibold uppercase tracking-wider text-slate-400">
          Compare with Vanilla YOLOv8
        </span>
        <button
          onClick={() => setConfig({ compare: !config.compare })}
          className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
            config.compare ? 'bg-green-500' : 'bg-[#1e1e3a]'
          }`}
        >
          <span
            className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
              config.compare ? 'translate-x-6' : 'translate-x-1'
            }`}
          />
        </button>
      </div>
    </div>
  );
}