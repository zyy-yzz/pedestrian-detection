interface Props {
  pedestrianCount: number;
  processingTimeMs?: number;
  avgConfidence?: number;
  label?: string;
}

function _fmt(n: number | undefined | null, suffix: string = ''): string {
  if (n == null) return '---';
  if (suffix === '%') return `${(n * 100).toFixed(1)}%`;
  if (suffix === 'ms') return `${n}ms`;
  return String(n);
}

export default function StatsPanel({ pedestrianCount, processingTimeMs, avgConfidence, label }: Props) {
  return (
    <div className="rounded-xl border border-[#1e1e3a] bg-[#0d0d1a] p-4">
      {label && (
        <h3 className="mono text-xs font-semibold uppercase tracking-wider text-slate-400 mb-3">
          {label}
        </h3>
      )}
      <div className="grid grid-cols-3 gap-3">
        <div className="text-center">
          <div className="text-2xl font-bold text-green-400">{pedestrianCount}</div>
          <div className="mono text-[10px] text-slate-500 mt-0.5">Detected</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-amber-400">{_fmt(processingTimeMs, 'ms')}</div>
          <div className="mono text-[10px] text-slate-500 mt-0.5">Latency</div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-blue-400">{_fmt(avgConfidence, '%')}</div>
          <div className="mono text-[10px] text-slate-500 mt-0.5">Avg Conf</div>
        </div>
      </div>
    </div>
  );
}

export function CompareStats({
  enhanced,
  vanilla,
  comparison,
}: {
  enhanced: { pedestrian_count: number; processing_time_ms: number; avgConfidence?: number };
  vanilla: { pedestrian_count: number; processing_time_ms: number; avgConfidence?: number };
  comparison: { delta_count: number; delta_time_ms: number; avg_conf_enhanced: number; avg_conf_vanilla: number };
}) {
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-3">
        <StatsPanel
          label="Enhanced YOLOv8"
          pedestrianCount={enhanced.pedestrian_count}
          processingTimeMs={enhanced.processing_time_ms}
          avgConfidence={comparison.avg_conf_enhanced}
        />
        <StatsPanel
          label="Vanilla YOLOv8"
          pedestrianCount={vanilla.pedestrian_count}
          processingTimeMs={vanilla.processing_time_ms}
          avgConfidence={comparison.avg_conf_vanilla}
        />
      </div>

      {/* Delta indicators */}
      <div className="rounded-lg border border-[#1e1e3a] bg-[#0d0d1a] p-3">
        <div className="grid grid-cols-2 gap-3 mono text-xs">
          <div className="flex justify-between">
            <span className="text-slate-500">Detection Delta</span>
            <span className={comparison.delta_count >= 0 ? 'text-green-400' : 'text-red-400'}>
              {comparison.delta_count >= 0 ? '+' : ''}{comparison.delta_count}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-slate-500">Time Delta</span>
            <span className={comparison.delta_time_ms <= 0 ? 'text-green-400' : 'text-amber-400'}>
              {comparison.delta_time_ms >= 0 ? '+' : ''}{comparison.delta_time_ms}ms
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}