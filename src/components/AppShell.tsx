import { useState, type ReactNode } from 'react';
import {
  Activity,
  BarChart3,
  BookOpen,
  Cpu,
  History,
  LayoutDashboard,
  Menu,
  Plus,
  ScanSearch,
  Settings,
  Shield,
  X,
} from 'lucide-react';
import type { AppView, HealthResponse } from '../types/forensics';

interface AppShellProps {
  activeView: AppView;
  onNavigate: (view: AppView) => void;
  onNewInvestigation: () => void;
  health: HealthResponse | null;
  children: ReactNode;
}

const NAV_ITEMS: Array<{
  id: AppView;
  label: string;
  title: string;
  Icon: typeof LayoutDashboard;
}> = [
  { id: 'dashboard', label: 'Dashboard', title: 'Operations Dashboard', Icon: LayoutDashboard },
  { id: 'investigation', label: 'New Investigation', title: 'New Image Investigation', Icon: ScanSearch },
  { id: 'history', label: 'Investigation History', title: 'Investigation History', Icon: History },
  { id: 'models', label: 'AI Models', title: 'AI Models & Checkpoints', Icon: Cpu },
  { id: 'analytics', label: 'Analytics', title: 'Forensic Analytics', Icon: BarChart3 },
  { id: 'methodology', label: 'Methodology', title: 'Detection Methodology & Limitations', Icon: BookOpen },
  { id: 'system', label: 'System Settings', title: 'System Settings & Diagnostics', Icon: Settings },
];

export function AppShell({
  activeView,
  onNavigate,
  onNewInvestigation,
  health,
  children,
}: AppShellProps) {
  const [mobileNavOpen, setMobileNavOpen] = useState(false);

  const currentNav = NAV_ITEMS.find((n) => n.id === activeView) ?? NAV_ITEMS[0];

  const handleSelectView = (view: AppView) => {
    onNavigate(view);
    setMobileNavOpen(false);
  };

  const renderNavLinks = () => (
    <nav aria-label="Primary Workspace Navigation" className="flex-1 space-y-1 px-3 py-4">
      {NAV_ITEMS.map((item) => {
        const isActive = activeView === item.id;
        const Icon = item.Icon;
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => handleSelectView(item.id)}
            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs font-medium transition-colors whitespace-nowrap cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 ${
              isActive
                ? 'bg-[#111827] text-cyan-300 border border-slate-800'
                : 'text-slate-400 hover:bg-[#111827]/60 hover:text-slate-100 border border-transparent'
            }`}
          >
            <Icon
              className={`h-4 w-4 shrink-0 ${isActive ? 'text-cyan-400' : 'text-slate-500'}`}
              aria-hidden="true"
            />
            <span className="truncate">{item.label}</span>
          </button>
        );
      })}
    </nav>
  );

  return (
    <div className="min-h-screen bg-[#07090E] text-slate-100 flex">
      {/* Desktop Left Sidebar */}
      <aside className="no-print hidden lg:flex lg:w-[252px] lg:shrink-0 lg:flex-col border-r border-slate-800 bg-[#0B0E17]">
        <div className="flex h-16 items-center gap-2.5 border-b border-slate-800 px-5">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-cyan-500/30 bg-cyan-950/30 text-cyan-400">
            <Shield className="h-4 w-4" aria-hidden="true" />
          </div>
          <div className="min-w-0">
            <div className="font-display text-base font-bold tracking-tight text-slate-100 whitespace-nowrap">
              MediaGuard
            </div>
            <div className="text-[11px] text-slate-400 whitespace-nowrap truncate">
              Verify Before You Trust
            </div>
          </div>
        </div>

        {renderNavLinks()}

        <div className="border-t border-slate-800 p-4 space-y-2 text-xs">
          <div className="flex items-center justify-between gap-2 text-slate-400">
            <span>Backend API</span>
            <span
              className={`font-mono font-medium ${
                health?.status === 'ok'
                  ? 'text-emerald-400'
                  : health
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }`}
            >
              {health?.status === 'ok' ? 'Ready' : health ? 'Degraded' : 'Offline'}
            </span>
          </div>
          <div className="flex items-center justify-between gap-2 text-slate-400">
            <span>Compute</span>
            <span className="font-mono text-slate-200 truncate max-w-[140px]">
              {health ? (health.cuda_available ? 'CUDA GPU' : 'CPU Mode') : 'Unavailable'}
            </span>
          </div>
        </div>
      </aside>

      {/* Mobile Drawer Overlay */}
      {mobileNavOpen && (
        <div className="no-print fixed inset-0 z-50 flex lg:hidden">
          <div
            className="fixed inset-0 bg-black/70"
            onClick={() => setMobileNavOpen(false)}
            aria-hidden="true"
          />
          <aside className="relative z-10 flex w-64 max-w-[82vw] flex-col border-r border-slate-800 bg-[#0B0E17]">
            <div className="flex h-16 items-center justify-between border-b border-slate-800 px-5">
              <span className="font-display text-base font-bold text-slate-100">MediaGuard</span>
              <button
                type="button"
                onClick={() => setMobileNavOpen(false)}
                aria-label="Close navigation menu"
                className="rounded-lg p-1.5 text-slate-400 hover:bg-slate-800 hover:text-slate-100 cursor-pointer"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>
            {renderNavLinks()}
          </aside>
        </div>
      )}

      {/* Main Content Column */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top Bar Contract */}
        <header className="no-print sticky top-0 z-30 flex h-16 items-center justify-between gap-6 border-b border-slate-800 bg-[#07090E]/95 px-6 backdrop-blur">
          <div className="flex items-center gap-3 min-w-0">
            <button
              type="button"
              onClick={() => setMobileNavOpen(true)}
              aria-label="Open navigation menu"
              className="rounded-lg border border-slate-800 bg-[#0B0E17] p-2 text-slate-300 hover:text-white lg:hidden cursor-pointer"
            >
              <Menu className="h-4 w-4" aria-hidden="true" />
            </button>
            <div className="flex items-center gap-2 text-sm min-w-0">
              <span className="hidden sm:inline text-slate-500 whitespace-nowrap">MediaGuard</span>
              <span className="hidden sm:inline text-slate-700" aria-hidden="true">/</span>
              <h1 className="font-display text-sm sm:text-base font-semibold text-slate-100 truncate">
                {currentNav.title}
              </h1>
            </div>
          </div>

          <div className="flex items-center gap-4 shrink-0">
            <div className="hidden md:flex items-center gap-2 text-xs text-slate-400 whitespace-nowrap">
              <Activity className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
              <span className="font-mono">
                {health
                  ? `${health.cuda_available ? 'CUDA' : 'CPU'} · ${health.status === 'ok' ? 'All Models Loaded' : 'Check Models'}`
                  : 'FastAPI Disconnected'}
              </span>
            </div>
            <button
              type="button"
              onClick={onNewInvestigation}
              className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-400"
            >
              <Plus className="h-3.5 w-3.5" aria-hidden="true" />
              <span>New Investigation</span>
            </button>
          </div>
        </header>

        <main className="flex-1">{children}</main>

        <footer className="no-print border-t border-slate-800/80 bg-[#07090E] px-6 py-4">
          <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 text-xs text-slate-500">
            <span>MediaGuard v1.1.0 · Verify Before You Trust</span>
            <span>Original images deleted after analysis · Raw model scores are experimental indicators, not proof</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
