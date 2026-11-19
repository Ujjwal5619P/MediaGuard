import { useCallback, useEffect, useState } from 'react';
import {
  CheckCircle2,
  Download,
  FolderSearch,
  Play,
  RefreshCw,
  ShieldAlert,
  XCircle,
} from 'lucide-react';
import {
  ApiClientError,
  downloadEvaluationExport,
  fetchAnalytics,
  fetchEvaluationRun,
  fetchEvaluationRuns,
  inspectEvaluationFolder,
  runDatasetEvaluation,
} from '../services/api';
import type {
  AnalyticsResponse,
  AppView,
  DatasetEvaluationDetail,
  DatasetEvaluationSummary,
  DatasetInspectResponse,
  DatasetSplitRole,
  DetectorEvaluationMetrics,
} from '../types/forensics';

interface AnalyticsViewProps {
  apiBaseUrl: string;
  onNavigate: (view: AppView) => void;
}

const SCORE_BIN_LABELS = ['0–20%', '20–40%', '40–60%', '60–80%', '80–100%'];

function formatPct(val: number | null): string {
  if (val === null || val === undefined) return '—';
  return `${(val * 100).toFixed(2)}%`;
}

function DetectorMetricsPanel({
  roleTitle,
  metrics,
}: {
  roleTitle: string;
  metrics: DetectorEvaluationMetrics;
}) {
  const { confusion_matrix: cm } = metrics;

  return (
    <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-slate-800 pb-3">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-wider text-cyan-400">
            {roleTitle}
          </div>
          <h3 className="mt-0.5 text-base font-semibold text-slate-100">{metrics.model_name}</h3>
          <div className="font-mono text-xs text-slate-400">{metrics.model_id}</div>
        </div>
        <div className="text-right font-mono text-xs tabular-nums text-slate-300">
          <div>
            Evaluated: {metrics.evaluated_count}
            {metrics.error_count > 0 ? ` · Errors: ${metrics.error_count}` : ''}
          </div>
          <div className="text-slate-400">
            Mean Latency: {metrics.mean_inference_ms !== null ? `${metrics.mean_inference_ms} ms` : '—'}
            {metrics.peak_vram_mb !== null ? ` · Peak VRAM: ${metrics.peak_vram_mb} MB` : ''}
          </div>
        </div>
      </div>

      {/* Classification Metrics */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
          <div className="text-[11px] text-slate-400">Accuracy</div>
          <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-slate-100">
            {formatPct(metrics.accuracy)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
          <div className="text-[11px] text-slate-400">Precision</div>
          <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-slate-100">
            {formatPct(metrics.precision)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
          <div className="text-[11px] text-slate-400">Recall</div>
          <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-slate-100">
            {formatPct(metrics.recall)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
          <div className="text-[11px] text-slate-400">F1-Score</div>
          <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-cyan-400">
            {formatPct(metrics.f1_score)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
          <div className="text-[11px] text-slate-400">False Positive Rate (FPR)</div>
          <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-amber-300">
            {formatPct(metrics.false_positive_rate)}
          </div>
        </div>
        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
          <div className="text-[11px] text-slate-400">False Negative Rate (FNR)</div>
          <div className="mt-1 font-mono text-lg font-semibold tabular-nums text-rose-300">
            {formatPct(metrics.false_negative_rate)}
          </div>
        </div>
      </div>

      {/* 2x2 Confusion Matrix */}
      <div>
        <div className="mb-2 text-xs font-medium text-slate-300">
          Confusion Matrix (Positive Class: AI_GENERATED)
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-center font-mono text-xs tabular-nums">
            <thead>
              <tr className="border-b border-slate-800 text-slate-400">
                <th className="py-2 pr-3 text-left font-normal">Ground Truth \ Predicted</th>
                <th className="px-3 py-2 font-medium">Pred AI_GENERATED</th>
                <th className="px-3 py-2 font-medium">Pred REAL</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/80">
              <tr>
                <td className="py-2.5 pr-3 text-left text-slate-300">Actual AI_GENERATED</td>
                <td className="bg-emerald-950/20 px-3 py-2.5 text-emerald-300 font-semibold">
                  TP: {cm.tp}
                </td>
                <td className="bg-rose-950/20 px-3 py-2.5 text-rose-300 font-semibold">
                  FN: {cm.fn}
                </td>
              </tr>
              <tr>
                <td className="py-2.5 pr-3 text-left text-slate-300">Actual REAL</td>
                <td className="bg-amber-950/20 px-3 py-2.5 text-amber-300 font-semibold">
                  FP: {cm.fp}
                </td>
                <td className="bg-emerald-950/20 px-3 py-2.5 text-emerald-300 font-semibold">
                  TN: {cm.tn}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export function AnalyticsView({ apiBaseUrl, onNavigate }: AnalyticsViewProps) {
  const [subTab, setSubTab] = useState<'telemetry' | 'evaluation'>('telemetry');

  // Telemetry state
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Dataset Evaluation state
  const [datasetDir, setDatasetDir] = useState('dataset');
  const [splitRole, setSplitRole] = useState<DatasetSplitRole>('evaluation');
  const [thresholdInput, setThresholdInput] = useState('0.385');
  const [maxImagesInput, setMaxImagesInput] = useState('');
  const [includeUnivfd, setIncludeUnivfd] = useState(true);

  const [inspectResult, setInspectResult] = useState<DatasetInspectResponse | null>(null);
  const [inspecting, setInspecting] = useState(false);
  const [runningEval, setRunningEval] = useState(false);
  const [evalError, setEvalError] = useState<string | null>(null);
  const [savedRuns, setSavedRuns] = useState<DatasetEvaluationSummary[]>([]);
  const [activeEval, setActiveEval] = useState<DatasetEvaluationDetail | null>(null);

  const loadAnalytics = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await fetchAnalytics(apiBaseUrl);
      setAnalytics(data);
    } catch (err) {
      setAnalytics(null);
      setError(
        err instanceof ApiClientError
          ? err.message
          : 'Unable to load SQLite analytics.'
      );
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  const loadSavedRuns = useCallback(async () => {
    try {
      const runs = await fetchEvaluationRuns(apiBaseUrl, 20);
      setSavedRuns(runs);
    } catch {
      setSavedRuns([]);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadAnalytics();
    void loadSavedRuns();
  }, [loadAnalytics, loadSavedRuns]);

  const handleInspectFolder = async () => {
    setInspecting(true);
    setEvalError(null);
    try {
      const info = await inspectEvaluationFolder(apiBaseUrl, datasetDir);
      setInspectResult(info);
    } catch (err) {
      setInspectResult(null);
      setEvalError(
        err instanceof ApiClientError
          ? err.message
          : 'Unable to inspect dataset folders.'
      );
    } finally {
      setInspecting(false);
    }
  };

  const handleRunEvaluation = async () => {
    const parsedThresh = Number(thresholdInput);
    if (!Number.isFinite(parsedThresh) || parsedThresh <= 0 || parsedThresh >= 1) {
      setEvalError('Decision threshold must be strictly between 0.01 and 0.99.');
      return;
    }
    const parsedMax =
      maxImagesInput.trim() !== '' ? Number(maxImagesInput.trim()) : null;
    if (parsedMax !== null && (!Number.isInteger(parsedMax) || parsedMax < 1)) {
      setEvalError('Max images must be a positive whole number or left blank.');
      return;
    }

    setRunningEval(true);
    setEvalError(null);
    try {
      const detail = await runDatasetEvaluation(apiBaseUrl, {
        dataset_dir: datasetDir.trim() || null,
        split_role: splitRole,
        decision_threshold: parsedThresh,
        include_univfd: includeUnivfd,
        max_images: parsedMax,
      });
      setActiveEval(detail);
      await loadSavedRuns();
    } catch (err) {
      setEvalError(
        err instanceof ApiClientError
          ? `${err.code}: ${err.message}`
          : 'Dataset evaluation failed.'
      );
    } finally {
      setRunningEval(false);
    }
  };

  const handleSelectSavedRun = async (evaluationId: string) => {
    if (!evaluationId) return;
    setEvalError(null);
    try {
      const detail = await fetchEvaluationRun(apiBaseUrl, evaluationId);
      setActiveEval(detail);
    } catch (err) {
      setEvalError(
        err instanceof Error ? err.message : 'Unable to load evaluation record.'
      );
    }
  };

  const handleExportEval = async (fmt: 'csv' | 'json') => {
    if (!activeEval) return;
    try {
      await downloadEvaluationExport(apiBaseUrl, activeEval.evaluation_id, fmt);
    } catch (err) {
      setEvalError(
        err instanceof Error ? err.message : `Failed to export ${fmt.toUpperCase()}.`
      );
    }
  };

  const maxBin = Math.max(1, ...(analytics?.ai_score_buckets ?? [1]));
  const maxDaily = Math.max(
    1,
    ...(analytics?.daily_activity.map((d) => d.total) ?? [1])
  );

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-6 space-y-6">
      {/* Header + Segmented View Toggle */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">
            Forensic Detection Analytics &amp; Dataset Evaluation
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Live SQLite investigation metrics alongside local offline benchmark evaluation for Imager-1-Tiny and UnivFD comparison.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-[#0B0E17] p-1">
            <button
              type="button"
              onClick={() => setSubTab('telemetry')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                subTab === 'telemetry'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Investigation Telemetry
            </button>
            <button
              type="button"
              onClick={() => setSubTab('evaluation')}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                subTab === 'evaluation'
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Dataset Evaluation
            </button>
          </div>

          {subTab === 'telemetry' && (
            <button
              type="button"
              onClick={() => void loadAnalytics()}
              className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white whitespace-nowrap shrink-0 cursor-pointer"
            >
              <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
              <span>Refresh Metrics</span>
            </button>
          )}
        </div>
      </div>

      {subTab === 'telemetry' ? (
        error ? (
          <div className="rounded-xl border border-amber-900/60 bg-amber-950/20 p-6 text-center space-y-3">
            <div className="text-sm font-semibold text-amber-200">Analytics Store Unreachable</div>
            <p className="text-xs text-slate-300">{error}</p>
            <button
              type="button"
              onClick={() => onNavigate('system')}
              className="rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer"
            >
              Configure Backend Endpoint
            </button>
          </div>
        ) : !analytics ? (
          <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-8 text-center text-xs text-slate-400">
            Loading SQLite analytics...
          </div>
        ) : analytics.total_scans === 0 ? (
          <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-10 text-center space-y-3">
            <div className="text-sm font-semibold text-slate-200">No Recorded Scans Yet</div>
            <p className="text-xs text-slate-400">
              SQLite currently contains 0 investigations. Analyze images in New Investigation to populate real detection statistics, or switch to Dataset Evaluation to benchmark local folders.
            </p>
            <button
              type="button"
              onClick={() => onNavigate('investigation')}
              className="rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer"
            >
              Open New Investigation
            </button>
          </div>
        ) : (
          <>
            {/* Summary KPI Strip */}
            <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <div className="text-xs text-slate-400">Total Investigations</div>
                <div className="mt-2 font-mono text-2xl font-semibold tabular-nums text-slate-100">
                  {analytics.total_scans}
                </div>
                <div className="mt-1 text-xs text-slate-500 tabular-nums">
                  {analytics.originals_deleted_count} originals verified deleted
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <div className="text-xs text-slate-400">Imager-1-Tiny Signal Rate</div>
                <div className="mt-2 font-mono text-2xl font-semibold tabular-nums text-rose-400">
                  {analytics.ai_evaluated_count > 0
                    ? `${((analytics.ai_flagged_count / analytics.ai_evaluated_count) * 100).toFixed(1)}%`
                    : '—'}
                </div>
                <div className="mt-1 text-xs text-slate-500 tabular-nums">
                  {analytics.ai_flagged_count} flagged / {analytics.ai_natural_count} natural
                  {analytics.univfd_evaluated_count
                    ? ` · UnivFD ${analytics.univfd_flagged_count ?? 0}/${analytics.univfd_evaluated_count}`
                    : ''}
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <div className="text-xs text-slate-400">Face Manipulation Rate</div>
                <div className="mt-2 font-mono text-2xl font-semibold tabular-nums text-amber-400">
                  {analytics.face_evaluated_count > 0
                    ? `${((analytics.face_flagged_count / analytics.face_evaluated_count) * 100).toFixed(1)}%`
                    : '—'}
                </div>
                <div className="mt-1 text-xs text-slate-500 tabular-nums">
                  {analytics.face_flagged_count} flagged of {analytics.face_evaluated_count} with faces
                </div>
              </div>

              <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <div className="text-xs text-slate-400">Latency Range (Min / Avg / Max)</div>
                <div className="mt-2 font-mono text-lg font-semibold tabular-nums text-cyan-400">
                  {analytics.avg_inference_ms !== null
                    ? `${analytics.min_inference_ms} / ${Math.round(analytics.avg_inference_ms)} / ${analytics.max_inference_ms} ms`
                    : '—'}
                </div>
                <div className="mt-1 text-xs text-slate-500 tabular-nums">
                  {analytics.artifacts_retained_count} attribution overlays retained
                </div>
              </div>
            </section>

            {/* Charts Row */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <h3 className="text-sm font-semibold text-slate-100">
                  Imager-1-Tiny Raw P(ai) Score Distribution
                </h3>
                <p className="mt-0.5 text-xs text-slate-400">
                  Frequency across 20% probability buckets (decision threshold 50%).
                </p>

                <div className="mt-6 space-y-3">
                  {SCORE_BIN_LABELS.map((label, idx) => {
                    const count = analytics.ai_score_buckets[idx] ?? 0;
                    const pct = Math.round((count / maxBin) * 100);
                    const isHigh = idx >= 3;
                    return (
                      <div key={label} className="grid grid-cols-[72px_1fr_44px] items-center gap-3 text-xs font-mono tabular-nums">
                        <span className="text-slate-400">{label}</span>
                        <div className="h-5 w-full overflow-hidden rounded bg-[#07090E] border border-slate-800">
                          <div
                            style={{ width: `${pct}%` }}
                            className={`h-full transition-all ${
                              isHigh ? 'bg-rose-500/70' : idx === 2 ? 'bg-amber-500/70' : 'bg-emerald-500/70'
                            }`}
                          />
                        </div>
                        <span className="text-right text-slate-200">{count}</span>
                      </div>
                    );
                  })}
                </div>
              </section>

              <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <h3 className="text-sm font-semibold text-slate-100">
                  Daily Scan Volume
                </h3>
                <p className="mt-0.5 text-xs text-slate-400">
                  Last 14 active days in SQLite investigation history.
                </p>

                <div className="mt-6 flex h-44 items-end gap-3 border-b border-slate-800 pb-2">
                  {analytics.daily_activity.map((b) => {
                    const hPct = Math.max(8, Math.round((b.total / maxDaily) * 100));
                    return (
                      <div
                        key={b.date}
                        title={`${b.date}: ${b.total} scans`}
                        className="flex flex-1 flex-col items-center justify-end h-full min-w-[26px]"
                      >
                        <span className="mb-1 font-mono text-[11px] tabular-nums text-slate-300">
                          {b.total}
                        </span>
                        <div
                          style={{ height: `${hPct}%` }}
                          className="w-full max-w-[34px] rounded-t bg-cyan-500/40 border border-cyan-500/50"
                        />
                      </div>
                    );
                  })}
                </div>
                <div className="mt-2 flex items-center justify-between gap-2 font-mono text-[11px] tabular-nums text-slate-500">
                  {analytics.daily_activity.map((b) => (
                    <span key={b.date} className="flex-1 text-center">
                      {b.date.slice(5)}
                    </span>
                  ))}
                </div>
              </section>
            </div>

            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <h3 className="text-sm font-semibold text-slate-100">Execution Device Distribution</h3>
                <div className="mt-3 divide-y divide-slate-800 text-xs">
                  {Object.entries(analytics.device_breakdown).map(([dev, count]) => (
                    <div key={dev} className="flex items-center justify-between py-2.5 font-mono">
                      <span className="text-slate-300">{dev}</span>
                      <span className="tabular-nums text-slate-100">{count} scans</span>
                    </div>
                  ))}
                </div>
              </section>

              <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <h3 className="text-sm font-semibold text-slate-100">Verified Image Formats</h3>
                <div className="mt-3 divide-y divide-slate-800 text-xs">
                  {Object.entries(analytics.mime_breakdown).map(([mime, count]) => (
                    <div key={mime} className="flex items-center justify-between py-2.5 font-mono">
                      <span className="text-slate-300">{mime}</span>
                      <span className="tabular-nums text-slate-100">{count} files</span>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          </>
        )
      ) : (
        /* Local Dataset Evaluation Tab */
        <div className="space-y-6">
          {/* Scientific Protocol Guardrail Notice */}
          <div className="flex items-start gap-3 rounded-xl border border-slate-800 bg-[#0B0E17] p-4 text-xs text-slate-300">
            <ShieldAlert className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" aria-hidden="true" />
            <div className="space-y-1 leading-relaxed">
              <div className="font-semibold text-slate-100">
                Strict Local Evaluation Protocol (Zero Automatic Training or Threshold Tuning)
              </div>
              <p className="text-slate-400">
                Inspects local <span className="font-mono text-slate-200">ai_generated/</span> (`AI_GENERATED`) and <span className="font-mono text-slate-200">real/</span> (`REAL`) directories without uploading externally. No weights are modified, and decision thresholds are fixed per run—select thresholds on <span className="font-mono text-slate-200">validation</span> splits only and keep <span className="font-mono text-slate-200">final_test</span> untouched.
              </p>
            </div>
          </div>

          {/* Configuration & Run Controls */}
          <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-4">
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-5">
              <div className="lg:col-span-2">
                <label htmlFor="eval-dataset-dir" className="block text-xs text-slate-400">
                  Local Dataset Folder Path
                </label>
                <input
                  id="eval-dataset-dir"
                  type="text"
                  value={datasetDir}
                  onChange={(e) => setDatasetDir(e.target.value)}
                  placeholder="dataset"
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-[#07090E] px-3 py-2 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="eval-split-role" className="block text-xs text-slate-400">
                  Split Designation
                </label>
                <select
                  id="eval-split-role"
                  value={splitRole}
                  onChange={(e) => setSplitRole(e.target.value as DatasetSplitRole)}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-[#07090E] px-3 py-2 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
                >
                  <option value="validation">validation</option>
                  <option value="evaluation">evaluation</option>
                  <option value="final_test">final_test (untouched)</option>
                </select>
              </div>

              <div>
                <label htmlFor="eval-threshold" className="block text-xs text-slate-400">
                  Fixed Threshold (0–1)
                </label>
                <input
                  id="eval-threshold"
                  type="number"
                  min="0.05"
                  max="0.95"
                  step="0.05"
                  value={thresholdInput}
                  onChange={(e) => setThresholdInput(e.target.value)}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-[#07090E] px-3 py-2 font-mono text-xs tabular-nums text-slate-100 focus:border-cyan-500 focus:outline-none"
                />
              </div>

              <div>
                <label htmlFor="eval-max-images" className="block text-xs text-slate-400">
                  Max Images (optional)
                </label>
                <input
                  id="eval-max-images"
                  type="number"
                  min="1"
                  max="5000"
                  placeholder="All files"
                  value={maxImagesInput}
                  onChange={(e) => setMaxImagesInput(e.target.value)}
                  className="mt-1.5 w-full rounded-lg border border-slate-800 bg-[#07090E] px-3 py-2 font-mono text-xs tabular-nums text-slate-100 placeholder:text-slate-600 focus:border-cyan-500 focus:outline-none"
                />
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-800 pt-4">
              <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeUnivfd}
                  onChange={(e) => setIncludeUnivfd(e.target.checked)}
                  className="accent-cyan-500"
                />
                <span>Include UnivFD CLIP ViT-L/14 secondary comparison benchmark</span>
              </label>

              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={inspecting}
                  onClick={() => void handleInspectFolder()}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-[#111827] px-3.5 py-2 text-xs font-medium text-slate-200 hover:border-slate-600 hover:text-white whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <FolderSearch className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>{inspecting ? 'Scanning Folders...' : 'Inspect Local Folders'}</span>
                </button>
                <button
                  type="button"
                  disabled={runningEval}
                  onClick={() => void handleRunEvaluation()}
                  className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500 whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Play className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>{runningEval ? 'Evaluating Dataset...' : 'Run Local Evaluation'}</span>
                </button>
              </div>
            </div>

            {evalError && (
              <div
                role="alert"
                className="rounded-lg border border-rose-900/60 bg-rose-950/20 px-4 py-3 text-xs text-rose-200"
              >
                {evalError}
              </div>
            )}

            {inspectResult && (
              <div className="grid grid-cols-1 gap-3 border-t border-slate-800 pt-4 text-xs sm:grid-cols-3 font-mono tabular-nums">
                <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                  <div className="text-slate-400">ai_generated/ folder</div>
                  <div className="mt-1 text-slate-100">
                    {inspectResult.ai_generated_exists
                      ? `${inspectResult.ai_generated_count} supported images`
                      : 'Directory missing'}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                  <div className="text-slate-400">real/ folder</div>
                  <div className="mt-1 text-slate-100">
                    {inspectResult.real_exists
                      ? `${inspectResult.real_count} supported images`
                      : 'Directory missing'}
                  </div>
                </div>
                <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                  <div className="text-slate-400">Total Supported / Skipped</div>
                  <div className="mt-1 text-slate-100">
                    {inspectResult.total_supported_count} images · {inspectResult.unsupported_or_skipped.length} skipped
                  </div>
                </div>
              </div>
            )}
          </section>

          {/* Past SQLite Evaluation Runs Selector */}
          {savedRuns.length > 0 && (
            <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#0B0E17] px-4 py-3 text-xs">
              <div className="font-medium text-slate-300">
                Archived SQLite Evaluation Runs ({savedRuns.length})
              </div>
              <select
                aria-label="Load archived evaluation run"
                value={activeEval?.evaluation_id || ''}
                onChange={(e) => void handleSelectSavedRun(e.target.value)}
                className="rounded-lg border border-slate-800 bg-[#07090E] px-3 py-1.5 font-mono text-xs text-slate-200 focus:border-cyan-500 focus:outline-none"
              >
                <option value="">Select a saved evaluation run...</option>
                {savedRuns.map((run) => (
                  <option key={run.evaluation_id} value={run.evaluation_id}>
                    {run.created_at.slice(0, 19).replace('T', ' ')} · {run.split_role} · Acc{' '}
                    {formatPct(run.smaug_metrics.accuracy)} ({run.total_images} imgs)
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Active Evaluation Metrics & Confusion Matrices */}
          {activeEval && (
            <div className="space-y-6">
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-[#0B0E17] px-5 py-4 text-xs">
                <div className="space-y-1">
                  <div className="font-mono text-slate-200">
                    Run ID: {activeEval.evaluation_id} · Split: {activeEval.split_role} · Fixed Threshold: {activeEval.decision_threshold.toFixed(2)}
                  </div>
                  <div className="text-slate-400">
                    {activeEval.threshold_policy_note} · Total Images: {activeEval.total_images} ({activeEval.ai_generated_count} AI, {activeEval.real_count} Real) · Device: {activeEval.execution_device} · Duration: {activeEval.total_duration_ms} ms
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void handleExportEval('csv')}
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-[#111827] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:text-white whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>Export CSV</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => void handleExportEval('json')}
                    className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-[#111827] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:text-white whitespace-nowrap shrink-0 cursor-pointer"
                  >
                    <Download className="h-3.5 w-3.5" aria-hidden="true" />
                    <span>Export JSON</span>
                  </button>
                </div>
              </div>

              <div
                className={`grid grid-cols-1 gap-6 ${
                  activeEval.univfd_metrics ? 'lg:grid-cols-2' : ''
                }`}
              >
                <DetectorMetricsPanel
                  roleTitle="Primary AI-Generated Detector"
                  metrics={activeEval.smaug_metrics}
                />
                {activeEval.univfd_metrics && (
                  <DetectorMetricsPanel
                    roleTitle="Secondary Comparison Detector"
                    metrics={activeEval.univfd_metrics}
                  />
                )}
              </div>

              {/* Per-Image Results Table */}
              <section className="overflow-hidden rounded-xl border border-slate-800 bg-[#0B0E17]">
                <div className="border-b border-slate-800 px-5 py-3.5">
                  <h3 className="text-sm font-semibold text-slate-100">
                    Per-Image Evaluation Predictions ({activeEval.items.length})
                  </h3>
                </div>
                <div className="overflow-x-auto max-h-[460px]">
                  <table className="w-full border-collapse text-left text-xs">
                    <thead className="sticky top-0 bg-[#07090E] text-slate-400">
                      <tr className="border-b border-slate-800">
                        <th className="px-4 py-3 font-medium">Relative Path</th>
                        <th className="px-4 py-3 font-medium">Ground Truth</th>
                        <th className="px-4 py-3 font-medium">Primary Score</th>
                        <th className="px-4 py-3 font-medium">Primary Pred</th>
                        <th className="px-4 py-3 font-medium">Verdict</th>
                        {activeEval.include_univfd && (
                          <>
                            <th className="px-4 py-3 font-medium">UnivFD Score</th>
                            <th className="px-4 py-3 font-medium">UnivFD Pred</th>
                          </>
                        )}
                        <th className="px-4 py-3 font-medium text-right">Latency</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/80 font-mono tabular-nums">
                      {activeEval.items.map((item) => (
                        <tr key={item.relative_path} className="hover:bg-[#111827]/70">
                          <td className="px-4 py-2.5 text-slate-200 max-w-[240px] truncate" title={item.relative_path}>
                            {item.relative_path}
                          </td>
                          <td className="px-4 py-2.5 text-slate-300">{item.ground_truth}</td>
                          <td className="px-4 py-2.5 text-slate-200">
                            {item.smaug_score !== null ? `${(item.smaug_score * 100).toFixed(2)}%` : item.smaug_status}
                          </td>
                          <td className="px-4 py-2.5 text-slate-300">
                            {item.smaug_prediction || '—'}
                          </td>
                          <td className="px-4 py-2.5">
                            {item.smaug_correct === true ? (
                              <span className="inline-flex items-center gap-1 text-emerald-400">
                                <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
                                Correct
                              </span>
                            ) : item.smaug_correct === false ? (
                              <span className="inline-flex items-center gap-1 text-rose-400">
                                <XCircle className="h-3.5 w-3.5" aria-hidden="true" />
                                Error
                              </span>
                            ) : (
                              <span className="text-amber-400">Skipped</span>
                            )}
                          </td>
                          {activeEval.include_univfd && (
                            <>
                              <td className="px-4 py-2.5 text-slate-300">
                                {item.univfd_score !== null ? `${(item.univfd_score * 100).toFixed(2)}%` : item.univfd_status || '—'}
                              </td>
                              <td className="px-4 py-2.5 text-slate-400">
                                {item.univfd_prediction || '—'}
                              </td>
                            </>
                          )}
                          <td className="px-4 py-2.5 text-right text-slate-400">
                            {item.smaug_inference_ms !== null ? `${item.smaug_inference_ms} ms` : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
