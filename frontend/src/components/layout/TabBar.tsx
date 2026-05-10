interface TabBarProps {
  tabs: string[];
  active: string;
  onSelect: (tab: string) => void;
}

export default function TabBar({ tabs, active, onSelect }: TabBarProps) {
  return (
    <div className="flex gap-1 rounded-lg border border-[#1e1e3a] bg-[#0d0d1a] p-1">
      {tabs.map((tab) => (
        <button
          key={tab}
          onClick={() => onSelect(tab)}
          className={`rounded-md px-5 py-2 text-sm font-medium tracking-wide transition-all ${
            active === tab
              ? 'bg-green-500/10 text-green-400 shadow-[0_0_12px_rgba(34,197,94,0.15)]'
              : 'text-slate-400 hover:text-slate-200'
          }`}
        >
          {tab}
        </button>
      ))}
    </div>
  );
}
