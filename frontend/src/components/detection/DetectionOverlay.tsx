import { useRef, useEffect } from 'react';
import type { Detection } from '../../types';

interface Props {
  imageUrl: string;
  detections: Detection[];
  width?: number;
  height?: number;
}

function confidenceToHex(conf: number): string {
  if (conf >= 0.85) return '#22c55e';
  if (conf >= 0.65) return '#f59e0b';
  return '#ef4444';
}

export default function DetectionOverlay({ imageUrl, detections, width = 640, height = 640 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const loadedRef = useRef(false);

  useEffect(() => {
    const img = imgRef.current;
    const canvas = canvasRef.current;
    if (!img || !canvas) return;

    loadedRef.current = false;

    const draw = () => {
      canvas.width = img.naturalWidth || width;
      canvas.height = img.naturalHeight || height;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      for (const det of detections) {
        const { x1, y1, x2, y2 } = det.bbox;
        const x = x1, y = y1;
        const w = x2 - x1, h = y2 - y1;

        const colour = confidenceToHex(det.confidence);

        ctx.strokeStyle = colour;
        ctx.lineWidth = 2;
        ctx.shadowColor = colour;
        ctx.shadowBlur = 6;
        ctx.strokeRect(x, y, w, h);
        ctx.shadowBlur = 0;

        const label = `Pedestrian ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = 'bold 12px "JetBrains Mono", monospace';
        const textW = ctx.measureText(label).width + 8;

        ctx.fillStyle = colour;
        ctx.fillRect(x, Math.max(0, y - 22), textW, 22);

        ctx.fillStyle = colour === '#f59e0b' ? '#1a1a2e' : '#ffffff';
        ctx.fillText(label, x + 4, Math.max(15, y - 7));
      }
    };

    if (img.complete && img.naturalWidth > 0) {
      loadedRef.current = true;
      draw();
    } else {
      img.onload = () => { loadedRef.current = true; draw(); };
    }
  }, [imageUrl, detections, width, height]);

  return (
    <div className="scan-lines relative overflow-hidden rounded-xl border border-[#1e1e3a] bg-black">
      <img
        ref={imgRef}
        src={imageUrl}
        alt="Detection result"
        className="block w-full"
      />
      <canvas
        ref={canvasRef}
        className="absolute left-0 top-0 h-full w-full"
      />
    </div>
  );
}
