import type { ModalityTab } from '../types/forensics';

interface ModalityBarProps {
  activeModality: ModalityTab;
  onSelectModality: (modality: ModalityTab) => void;
}

const MODALITIES: Array<{ id: ModalityTab; label: string; statusLabel: string }> = [
  { id: 'image', label: 'Image Analysis', statusLabel: 'Active v1' },
  { id: 'video', label: 'Video Pipeline', statusLabel: 'Queued' },
  { id: 'audio', label: 'Audio Pipeline', statusLabel: 'Queued' },
];

export function ModalityBar({ activeModality, onSelectModality }: ModalityBarProps) {
  return (
    <div className="no-print border-b border-slate-800 bg-[#0B0E17]">
      <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-6 py-2.5">
        <div
          role="tablist"
          aria-label="Analysis Modalities"
          className="flex items-center gap-1 rounded-lg bg-[#07090E] p-1 border border-slate-800"
        >
          {MODALITIES.map((mod) => {
            const isSelected = activeModality === mod.id;
            const isQueued = mod.id !== 'image';
            return (
              <button
                key={mod.id}
                type="button"
                role="tab"
                aria-selected={isSelected}
                onClick={() => onSelectModality(mod.id)}
                className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 ${
                  isSelected
                    ? 'bg-slate-800 text-slate-100'
                    : isQueued
                    ? 'text-slate-400 hover:text-slate-200'
                    : 'text-slate-300 hover:text-slate-100'
                }`}
              >
                <span>{mod.label}</span>
                <span aria-hidden="true" className="text-slate-600">
                  ·
                </span>
                <span
                  className={
                    mod.id === 'image' ? 'text-cyan-400' : 'text-amber-400/90'
                  }
                >
                  {mod.statusLabel}
                </span>
              </button>
            );
          })}
        </div>

        <p className="text-xs text-slate-400">
          AI-assisted media analysis · Experimental indicators are not definitive proof
        </p>
      </div>
    </div>
  );
}
