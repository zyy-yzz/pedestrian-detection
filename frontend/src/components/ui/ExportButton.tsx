import { useCallback } from 'react';

interface Props {
  imageUrl: string | null;
  detections: Array<{ bbox: { x1: number; y1: number; x2: number; y2: number }; confidence: number; class_name: string }>;
  label?: string;
  disabled?: boolean;
}

export default function ExportButton({ imageUrl, detections, label = 'Download', disabled }: Props) {
  const downloadPNG = useCallback(() => {
    if (!imageUrl) return;
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(img, 0, 0);
      for (const det of detections) {
        const { x1, y1, x2, y2 } = det.bbox;
        const color = det.confidence >= 0.85 ? '#22c55e' : det.confidence >= 0.65 ? '#f59e0b' : '#ef4444';
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
      }
      canvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = `detection-${Date.now()}.png`;
        link.click();
        URL.revokeObjectURL(url);
      }, 'image/png');
    };
    img.src = imageUrl;
  }, [imageUrl, detections]);

  const downloadJSON = useCallback(() => {
    const data = {
      exported_at: new Date().toISOString(),
      pedestrian_count: detections.length,
      detections: detections.map((d) => ({
        bbox: d.bbox,
        confidence: d.confidence,
        class_name: d.class_name,
      })),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `detection-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
  }, [detections]);

  return (
    <div className="flex gap-2">
      <button
        onClick={downloadPNG}
        disabled={disabled || !imageUrl}
        className="flex-1 rounded-lg border border-[#1e1e3a] bg-[#0d0d1a] px-4 py-2 text-sm font-medium
                   text-green-400 transition-colors hover:border-green-500/40 hover:bg-green-500/10
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Download PNG
      </button>
      <button
        onClick={downloadJSON}
        disabled={disabled || detections.length === 0}
        className="flex-1 rounded-lg border border-[#1e1e3a] bg-[#0d0d1a] px-4 py-2 text-sm font-medium
                   text-blue-400 transition-colors hover:border-blue-500/40 hover:bg-blue-500/10
                   disabled:opacity-40 disabled:cursor-not-allowed"
      >
        Export JSON
      </button>
    </div>
  );
}