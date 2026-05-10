import { useRef, useEffect, useState, useCallback } from 'react';
import { useStream } from '../../hooks/useStream';
import { useStreamStore } from '../../store/streamStore';

export default function LiveDetector() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const { connect, sendFrame, disconnect } = useStream();
  const { isConnected, liveCount, latency } = useStreamStore();

  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      setIsStreaming(true);
      connect();
    } catch {
      alert('Camera access denied');
    }
  }, [connect]);

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setIsStreaming(false);
    disconnect();
  }, [disconnect]);

  useEffect(() => {
    if (isStreaming && isConnected) {
      intervalRef.current = setInterval(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas || video.readyState < 2) return;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        canvas.toBlob((blob) => {
          if (blob) sendFrame(blob);
        }, 'image/jpeg', 0.6);
      }, 1000 / 10);
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [isStreaming, isConnected, sendFrame]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  return (
    <div className="animate-fade-in space-y-6">
      <div className="relative overflow-hidden rounded-xl border border-[#1e1e3a] bg-black">
        {!isStreaming ? (
          <div className="flex h-80 flex-col items-center justify-center">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border-2 border-[#1e1e3a]">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#555578" strokeWidth="1.5" strokeLinecap="round">
                <path d="M23 7l-7 5 7 5V7z" />
                <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
              </svg>
            </div>
            <p className="text-sm font-medium text-slate-500">Camera feed will appear here</p>
          </div>
        ) : (
          <video ref={videoRef} autoPlay muted playsInline className="w-full" />
        )}
        <canvas ref={canvasRef} className="hidden" />

        {isStreaming && (
          <div className="absolute left-3 top-3 flex items-center gap-2 rounded-full bg-black/70 px-3 py-1 backdrop-blur-sm">
            <span className={`h-2 w-2 rounded-full ${isConnected ? 'animate-pulse bg-green-500' : 'bg-red-500'}`} />
            <span className="mono text-xs text-white">{isConnected ? 'LIVE' : 'DISCONNECTED'}</span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-xl border border-green-500/20 bg-green-500/5 p-4 text-center">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Pedestrians</p>
          <p className="mt-1 text-2xl font-bold text-green-400 mono">{liveCount}</p>
        </div>
        <div className="rounded-xl border border-[#1e1e3a] bg-[#0d0d1a]/70 p-4 text-center">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Latency</p>
          <p className="mt-1 text-2xl font-bold text-white mono">{latency}<span className="ml-0.5 text-sm font-normal text-slate-500">ms</span></p>
        </div>
        <div className="rounded-xl border border-[#1e1e3a] bg-[#0d0d1a]/70 p-4 text-center">
          <p className="text-xs font-medium uppercase tracking-wider text-slate-500">Status</p>
          <p className={`mt-1 text-lg font-bold ${isConnected ? 'text-green-400' : 'text-slate-500'} mono`}>
            {isConnected ? 'CONNECTED' : 'IDLE'}
          </p>
        </div>
      </div>

      <div className="flex gap-3">
        {!isStreaming ? (
          <button
            onClick={startCamera}
            className="flex items-center gap-2 rounded-lg bg-green-600 px-6 py-2.5 text-sm font-semibold text-white transition-colors hover:bg-green-500"
          >
            <span className="text-base">●</span> Start Stream
          </button>
        ) : (
          <button
            onClick={stopCamera}
            className="flex items-center gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-6 py-2.5 text-sm font-semibold text-red-400 transition-colors hover:bg-red-500/20"
          >
            <span className="text-base">■</span> Stop Stream
          </button>
        )}
      </div>
    </div>
  );
}
