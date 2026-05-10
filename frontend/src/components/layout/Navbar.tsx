import { Link, useLocation } from 'react-router-dom';

export default function Navbar() {
  const location = useLocation();

  const isActive = (path: string) =>
    location.pathname === path ||
    (path === '/detect' && location.pathname === '/') ||
    location.pathname.startsWith(path + '/');

  const linkClass = (path: string) =>
    `relative px-4 py-2 text-sm font-medium tracking-wider uppercase transition-colors ${
      isActive(path)
        ? 'text-green-400'
        : 'text-slate-400 hover:text-slate-200'
    }`;

  return (
    <nav className="sticky top-0 z-50 border-b border-[#1e1e3a] bg-[#06060c]/90 backdrop-blur-md">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
        <Link to="/" className="flex items-center gap-3 group">
          <div className="relative flex h-9 w-9 items-center justify-center">
            <div className="absolute inset-0 animate-pulse-glow rounded-full bg-green-500/10" />
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#22c55e" strokeWidth="2" strokeLinecap="round">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
          <span className="hidden text-lg font-bold tracking-tight text-white sm:inline">
            Night<span className="text-green-400">Ped</span>
          </span>
        </Link>

        <div className="flex items-center gap-1 rounded-lg border border-[#1e1e3a] bg-[#0d0d1a] p-0.5">
          <Link to="/detect" className={linkClass('/detect')}>
            <span className="flex items-center gap-1.5">
              <span className="text-xs">●</span> Detect
            </span>
          </Link>
          <Link to="/history" className={linkClass('/history')}>
            <span className="flex items-center gap-1.5">
              <span className="text-xs">○</span> History
            </span>
          </Link>
        </div>
      </div>
    </nav>
  );
}
