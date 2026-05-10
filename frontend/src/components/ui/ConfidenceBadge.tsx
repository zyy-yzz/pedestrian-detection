function confidenceColour(conf: number): string {
  if (conf >= 0.85) return 'bg-green-500 text-green-50';
  if (conf >= 0.65) return 'bg-amber-500 text-amber-50';
  return 'bg-red-500 text-red-50';
}

export default function ConfidenceBadge({ confidence }: { confidence: number }) {
  return (
    <span
      className={`inline-flex items-center rounded-md px-2 py-0.5 text-xs font-semibold mono tracking-wide ${confidenceColour(confidence)}`}
    >
      {(confidence * 100).toFixed(0)}%
    </span>
  );
}
