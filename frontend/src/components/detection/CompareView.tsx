import { useEffect, useRef } from 'react';
import type { DetectionResponse } from '../../types';

interface Props {
  imageUrl: string;
  enhanced: DetectionResponse;
  vanilla: DetectionResponse;
}

function _drawOnCanvas(
  canvas: HTMLCanvasElement | null,
  imageUrl: string,
  detections: DetectionResponse['detections'],
) {
  if (!canvas) return;
  const img = new Image();
  img.onload = () => {
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.drawImage(img, 0, 0);

    for (const det of detections) {
      const { x1, y1, x2, y2 } = det.bbox;
      const conf = det.confidence;
      const color = conf >= 0.85 ? '#22c55e' : conf >= 0.65 ? '#f59e0b' : '#ef4444';

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      ctx.fillStyle = color;
      const label = `Pedestrian ${(conf * 100).toFixed(0)}%`;
      ctx.font = '12px monospace';
      const tm = ctx.measureText(label);
      ctx.fillRect(x1, y1 - 18, tm.width + 8, 16);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1 + 4, y1 - 5);
    }
  };
  img.src = imageUrl;
}

export default function CompareView({ imageUrl, enhanced, vanilla }: Props) {
  const leftRef = useRef<HTMLCanvasElement>(null);
  const rightRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    _drawOnCanvas(leftRef.current, imageUrl, enhanced.detections);
    _drawOnCanvas(rightRef.current, imageUrl, vanilla.detections);
  }, [imageUrl, enhanced, vanilla]);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      {/* Enhanced */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="mono text-xs font-semibold text-green-400">Enhanced YOLOv8</span>
          <span className="mono text-xs text-slate-500">
            {enhanced.pedestrian_count} pedestrians · {enhanced.processing_time_ms}ms
          </span>
        </div>
        <div className="rounded-lg border border-green-500/20 overflow-hidden bg-[#06060c]">
          <canvas ref={leftRef} className="w-full h-auto" />
        </div>
      </div>

      {/* Vanilla */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="mono text-xs font-semibold text-amber-400">Vanilla YOLOv8</span>
          <span className="mono text-xs text-slate-500">
            {vanilla.pedestrian_count} pedestrians · {vanilla.processing_time_ms}ms
          </span>
        </div>
        <div className="rounded-lg border border-amber-500/20 overflow-hidden bg-[#06060c]">
          <canvas ref={rightRef} className="w-full h-auto" />
        </div>
      </div>
    </div>
  );
}