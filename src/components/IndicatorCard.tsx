import { AlertTriangle, CheckCircle2, ExternalLink, HelpCircle, Loader2, MinusCircle, XCircle } from 'lucide-react';
import type { TaskIndicator, TaskStatus } from '../types/forensics';

interface IndicatorCardProps {
  title: string;
  roleLabel?: string;
  scopeDescription: string;
  indicator: TaskIndicator | null;
  offlineMessage?: string | null;
}

const STATUS_META: Record<
  TaskStatus,
  { label: string; colorClass: string; Icon: typeof AlertTriangle }
> = {
  unavailable: {
    label: 'Unavailable',
    colorClass: 'text-amber-400',
    Icon: MinusCircle,
  },
  loading: {
    label: 'Loading Model',
    colorClass: 'text-cyan-400',
    Icon: Loader2,
  },
  no_face_detected: {
    label: 'No Face Detected',
    colorClass: 'text-slate-300',
    Icon: HelpCircle,
  },
  inconclusive: {
    label: 'Inconclusive',
    colorClass: 'text-amber-400',
    Icon: AlertTriangle,
  },
  evaluated: {
    label: 'Evaluated',
    colorClass: 'text-emerald-400',
    Icon: CheckCircle2,
  },
  error: {
    label: 'Analysis Failed',
    colorClass: 'text-rose-400',
    Icon: XCircle,
  },
};

export function IndicatorCard({
  title,
  roleLabel,
  scopeDescription,
  indicator,
  offlineMessage,
}: IndicatorCardProps) {
  const status: TaskStatus = indicator?.status ?? 'unavailable';
  const meta = STATUS_META[status];
  const StatusIcon = meta.Icon;

  const hasScore = indicator && indicator.score !== null && status === 'evaluated';
  const formattedScore = hasScore
    ? `${(indicator.score! * 100).toFixed(1)}%`
    : '—';

  const thresholdPct = indicator
    ? `${(indicator.decision_threshold * 100).toFixed(0)}%`
    : '50%';

  const highSignal =
    hasScore && indicator!.score! >= indicator!.decision_threshold;

  return (
    <div className="flex flex-col justify-between rounded-xl border border-slate-800 bg-[#0B0E17] p-6">
      <div>
        <div className="flex items-start justify-between gap-4">
          <div>
            {roleLabel && (
              <div className="mb-1 font-mono text-[11px] uppercase tracking-wider text-cyan-400">
                {roleLabel}
              </div>
            )}
            <h3 className="text-base font-semibold text-slate-100">{title}</h3>
            <p className="mt-1 text-xs leading-relaxed text-slate-400">{scopeDescription}</p>
          </div>
          <div className={`flex items-center gap-1.5 text-xs font-medium whitespace-nowrap shrink-0 ${meta.colorClass}`}>
            <StatusIcon
              className={`h-4 w-4 ${status === 'loading' ? 'animate-spin' : ''}`}
              aria-hidden="true"
            />
            <span>{meta.label}</span>
          </div>
        </div>

        {/* Metrics Strip */}
        <div className="mt-5 grid grid-cols-2 gap-4 border-t border-slate-800 pt-4 sm:grid-cols-4">
          <div>
            <div className="text-xs text-slate-400">Raw Score</div>
            <div
              className={`mt-1 font-mono text-2xl font-semibold tabular-nums ${
                hasScore
                  ? highSignal
                    ? 'text-rose-400'
                    : 'text-emerald-400'
                  : 'text-slate-100'
              }`}
            >
              {formattedScore}
            </div>
          </div>

          <div>
            <div className="text-xs text-slate-400">Threshold</div>
            <div className="mt-1.5 font-mono text-sm font-medium tabular-nums text-slate-200">
              {thresholdPct}
            </div>
          </div>

          <div>
            <div className="text-xs text-slate-400">Device &amp; Time</div>
            <div className="mt-1.5 font-mono text-xs tabular-nums text-slate-200 truncate" title={indicator?.execution_device || undefined}>
              {indicator?.execution_device || '—'}
            </div>
            {indicator?.processing_ms !== null && indicator?.processing_ms !== undefined && (
              <div className="mt-0.5 font-mono text-xs tabular-nums text-slate-400">
                {indicator.processing_ms} ms
              </div>
            )}
          </div>

          <div className="sm:text-right">
            <div className="text-xs text-slate-400">Calibration</div>
            <div className="mt-1.5 font-mono text-xs text-slate-300">
              {indicator?.calibrated ? 'Calibrated' : 'Uncalibrated'}
            </div>
            {indicator?.faces_detected !== null && indicator?.faces_detected !== undefined && (
              <div className="mt-0.5 font-mono text-xs tabular-nums text-cyan-400">
                {indicator.faces_detected} face{indicator.faces_detected === 1 ? '' : 's'}
              </div>
            )}
          </div>
        </div>

        {/* Detection Result Bar */}
        {indicator?.prediction_label && (
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-800 bg-[#111827] px-3.5 py-2.5 text-xs">
            <span className="text-slate-400">Detection Result</span>
            <span
              className={`font-semibold ${
                status === 'no_face_detected'
                  ? 'text-slate-200'
                  : highSignal
                  ? 'text-rose-300'
                  : 'text-emerald-300'
              }`}
            >
              {indicator.prediction_label}
            </span>
          </div>
        )}
      </div>

      {/* Model Provenance & Limitations */}
      <dl className="mt-5 space-y-2 border-t border-slate-800/80 pt-4 text-xs">
        <div className="flex items-start justify-between gap-3">
          <dt className="text-slate-500 shrink-0">Model Name</dt>
          <dd className="text-right font-medium text-slate-200">
            {indicator?.model_name || indicator?.model_id || 'Not assigned'}
          </dd>
        </div>
        {indicator?.architecture && (
          <div className="flex items-start justify-between gap-3">
            <dt className="text-slate-500 shrink-0">Architecture</dt>
            <dd className="text-right font-mono text-[11px] text-slate-300 max-w-[340px]">
              {indicator.architecture}
            </dd>
          </div>
        )}
        {indicator?.model_source && (
          <div className="flex items-start justify-between gap-3">
            <dt className="text-slate-500 shrink-0">Model Source</dt>
            <dd className="text-right">
              <a
                href={indicator.model_source}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 font-mono text-[11px] text-cyan-400 hover:text-cyan-300"
              >
                <span className="truncate max-w-[260px]">{indicator.model_source.replace(/^https?:\/\//, '')}</span>
                <ExternalLink className="h-3 w-3 shrink-0" aria-hidden="true" />
              </a>
            </dd>
          </div>
        )}
        <div className="pt-1 text-slate-300 leading-relaxed">
          {offlineMessage ||
            indicator?.reason ||
            'Upload an image and click Run Forensic Analysis to execute real pretrained model inference.'}
        </div>
        {indicator?.limitations && (
          <p className="leading-relaxed text-slate-500">
            Limitations: {indicator.limitations}
          </p>
        )}
      </dl>
    </div>
  );
}
