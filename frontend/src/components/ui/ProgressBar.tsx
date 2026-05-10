interface ProgressBarProps {
  progress: number;
  label?: string;
}

export default function ProgressBar({ progress, label }: ProgressBarProps) {
  return (
    <div className="w-full">
      {(label != null) && (
        <div className="mb-1.5 flex justify-between text-xs mono text-slate-400">
          <span>{label}</span>
          <span>{Math.round(progress)}%</span>
        </div>
      )}
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#1e1e3a]">
        <div
          className="h-full rounded-full bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)] transition-all duration-700 ease-out"
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>
    </div>
  );
}
