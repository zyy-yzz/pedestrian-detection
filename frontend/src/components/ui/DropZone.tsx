import { useDropzone } from 'react-dropzone';

interface DropZoneProps {
  onDrop: (files: File[]) => void;
  accept?: Record<string, string[]>;
  label?: string;
}

export default function DropZone({ onDrop, accept, label }: DropZoneProps) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop, accept, maxFiles: 1 });

  return (
    <div
      {...getRootProps()}
      className={`group relative flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed p-10 transition-all duration-300 ${
        isDragActive
          ? 'border-green-400 bg-green-500/5 shadow-[0_0_30px_rgba(34,197,94,0.1)]'
          : 'border-[#1e1e3a] bg-[#0d0d1a]/50 hover:border-slate-500 hover:bg-[#0d0d1a]'
      }`}
    >
      <input {...getInputProps()} />
      <div className={`mb-3 transition-transform duration-300 ${isDragActive ? 'scale-110' : 'group-hover:scale-105'}`}>
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke={isDragActive ? '#22c55e' : '#555578'} strokeWidth="1.5" strokeLinecap="round">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
      </div>
      <p className={`text-sm font-medium mono ${isDragActive ? 'text-green-400' : 'text-slate-500'}`}>
        {isDragActive ? 'Drop to detect' : (label || 'Drop image here or click to browse')}
      </p>
      <p className="mt-1 text-xs text-slate-600 mono">JPG / PNG / BMP &middot; max 20MB</p>
    </div>
  );
}
