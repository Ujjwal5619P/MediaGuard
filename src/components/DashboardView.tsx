import { useCallback, useEffect, useState } from 'react';
import { ArrowRight, Cpu, Eye, RefreshCw, ScanSearch, Server } from 'lucide-react';
import { ApiClientError, fetchAnalytics, fetchScans } from '../services/api';
import type {
  AnalyticsResponse,
  AppView,
  HealthResponse,
  ScanSummary,
} from '../types/forensics';

interface DashboardViewProps {
  apiBaseUrl: string;
  health: HealthResponse | null;
  healthError: string | null;
  onNavigate: (view: AppView) => void;
  onInspectScan: (scanId: string) => Promise<void>;
  onStartNewInvestigation: () => void;
}

export function DashboardView({
  apiBaseUrl,
  health,
  healthError,
  onNavigate,
  onInspectScan,
  onStartNewInvestigation,
}: DashboardViewProps) {
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null);
  const [recentScans, setRecentScans] = useState<ScanSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);

  const loadDashboardData = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    try {
      const [stats, scansResp] = await Promise.all([
        fetchAnalytics(apiBaseUrl),
        fetchScans(apiBaseUrl, { limit: 6, offset: 0 }),
      ]);
      setAnalytics(stats);
      setRecentScans(scansResp.items);
    } catch (err) {
      setAnalytics(null);
      setRecentScans([]);
      setLoadError(
        err instanceof ApiClientError
          ? err.message
          : 'Unable to reach FastAPI backend for SQLite telemetry.'
      );
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadDashboardData();
  }, [loadDashboardData]);

  const maxDailyTotal = Math.max(
    1,
    ...(analytics?.daily_activity.map((b) => b.total) ?? [1])
  );

  const readyModelCount = health
    ? health.execution_mode.toLowerCase().includes('lightweight')
      ? Number(health.models.ai_generated_detector.status === 'ready')
      : [
          health.models.ai_generated_detector.status === 'ready',
          health.models.univfd_detector?.status === 'ready',
          health.models.face_manipulation_detector.status === 'ready',
          health.models.dinov2_extractor.status === 'ready',
        ].filter(Boolean).length
    : 0;
  const totalCatalogModels = health?.models.univfd_detector ? 4 : 3;

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-6 space-y-6">
      {/* Intro & Actions */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">
            Forensic Operations Overview
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Live SQLite investigation counts, hardware compute telemetry, and verified checkpoint readiness.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => void loadDashboardData()}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white transition-colors cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
            <span>Refresh Data</span>
          </button>
          <button
            type="button"
            onClick={onStartNewInvestigation}
            className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 transition-colors cursor-pointer"
          >
            <ScanSearch className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Start Investigation</span>
          </button>
        </div>
      </div>

      {loadError && (
        <div className="rounded-xl border border-amber-900/60 bg-amber-950/20 px-4 py-3 text-xs text-amber-200 flex flex-wrap items-center justify-between gap-3">
          <span>{loadError}</span>
          <button
            type="button"
            onClick={() => onNavigate('system')}
            className="font-semibold text-cyan-300 hover:text-cyan-200 cursor-pointer"
          >
            Check System Settings
          </button>
        </div>
      )}

      {/* Real SQLite KPI Row */}
      <section aria-label="Investigation Summary Metrics" className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
          <div className="text-xs text-slate-400">Total Images Analyzed</div>
          <div className="mt-2 font-mono text-3xl font-semibold tabular-nums text-slate-100">
            {analytics ? analytics.total_scans : '—'}
          </div>
          <div className="mt-1.5 text-xs text-slate-500">
            Persisted SQLite scan records
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
          <div className="text-xs text-slate-400">AI-Generated Flagged</div>
          <div className="mt-2 font-mono text-3xl font-semibold tabular-nums text-rose-400">
            {analytics ? analytics.ai_flagged_count : '—'}
          </div>
          <div className="mt-1.5 text-xs text-slate-500 tabular-nums">
            {analytics ? `${analytics.ai_evaluated_count} evaluated by Imager-1-Tiny` : 'Awaiting backend'}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
          <div className="text-xs text-slate-400">Face Manipulation Flagged</div>
          <div className="mt-2 font-mono text-3xl font-semibold tabular-nums text-amber-400">
            {analytics ? analytics.face_flagged_count : '—'}
          </div>
          <div className="mt-1.5 text-xs text-slate-500 tabular-nums">
            {analytics ? `${analytics.face_evaluated_count} face crops scored · ${analytics.face_no_face_count} no-face` : 'Awaiting backend'}
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
          <div className="text-xs text-slate-400">Mean Inference Latency</div>
          <div className="mt-2 font-mono text-3xl font-semibold tabular-nums text-cyan-400">
            {analytics?.avg_inference_ms !== null && analytics?.avg_inference_ms !== undefined
              ? `${Math.round(analytics.avg_inference_ms)} ms`
              : '—'}
          </div>
          <div className="mt-1.5 text-xs text-slate-500 tabular-nums">
            {analytics?.max_vram_peak_mb !== null && analytics?.max_vram_peak_mb !== undefined
              ? `Peak CUDA VRAM: ${analytics.max_vram_peak_mb} MB`
              : 'Measured across recorded runs'}
          </div>
        </div>
      </section>

      {/* Activity Chart + System/GPU/Model Readiness */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[1.5fr_1fr]">
        {/* Real Daily Activity Chart */}
        <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 flex flex-col justify-between">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">Analysis Activity</h3>
              <p className="mt-0.5 text-xs text-slate-400">
                Daily scan volume and flagged detections computed directly from SQLite timestamps.
              </p>
            </div>
            <button
              type="button"
              onClick={() => onNavigate('analytics')}
              className="flex items-center gap-1 text-xs font-medium text-cyan-400 hover:text-cyan-300 cursor-pointer whitespace-nowrap"
            >
              <span>Full Analytics</span>
              <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
            </button>
          </div>

          {analytics && analytics.daily_activity.length > 0 ? (
            <div className="mt-6">
              <div className="flex h-44 items-end gap-3 border-b border-slate-800 pb-2">
                {analytics.daily_activity.map((bucket) => {
                  const totalPct = Math.max(8, Math.round((bucket.total / maxDailyTotal) * 100));
                  const aiPct = Math.round((bucket.ai_flagged / bucket.total) * 100);
                  return (
                    <div
                      key={bucket.date}
                      title={`${bucket.date}: ${bucket.total} scans (${bucket.ai_flagged} AI flagged, ${bucket.face_flagged} face flagged)`}
                      className="flex flex-1 flex-col items-center justify-end h-full min-w-[28px]"
                    >
                      <span className="mb-1.5 font-mono text-[11px] tabular-nums text-slate-300">
                        {bucket.total}
                      </span>
                      <div
                        style={{ height: `${totalPct}%` }}
                        className="relative w-full max-w-[36px] overflow-hidden rounded-t bg-cyan-500/25 border border-cyan-500/40"
                      >
                        {aiPct > 0 && (
                          <div
                            style={{ height: `${aiPct}%` }}
                            className="absolute inset-x-0 bottom-0 bg-rose-500/75"
                          />
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-2 flex items-center justify-between gap-2 overflow-x-auto font-mono text-[11px] tabular-nums text-slate-500">
                {analytics.daily_activity.map((bucket) => (
                  <span key={bucket.date} className="flex-1 text-center min-w-[28px]">
                    {bucket.date.slice(5)}
                  </span>
                ))}
              </div>
              <div className="mt-4 flex items-center gap-5 text-xs text-slate-400">
                <span className="flex items-center gap-2">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm bg-cyan-500/40 border border-cyan-500/60" />
                  <span>Total Analyzed</span>
                </span>
                <span className="flex items-center gap-2">
                  <span className="inline-block h-2.5 w-2.5 rounded-sm bg-rose-500/75" />
                  <span>AI-Generated Signal Share</span>
                </span>
              </div>
            </div>
          ) : (
            <div className="mt-6 flex h-44 flex-col items-center justify-center rounded-lg border border-dashed border-slate-800 bg-[#07090E] p-6 text-center">
              <p className="text-xs font-medium text-slate-300">
                No analysis activity recorded in SQLite yet
              </p>
              <p className="mt-1 text-xs text-slate-500">
                Run an image investigation to populate daily detection activity charts.
              </p>
            </div>
          )}
        </section>

        {/* GPU, Backend Health & Model Readiness */}
        <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-4">
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-semibold text-slate-100">
              Hardware &amp; Model Readiness
            </h3>
            <button
              type="button"
              onClick={() => onNavigate('models')}
              className="text-xs font-medium text-cyan-400 hover:text-cyan-300 cursor-pointer"
            >
              Manage Models
            </button>
          </div>

          <div className="space-y-3 border-t border-slate-800 pt-3 text-xs">
            <div className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-slate-400">
                <Server className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
                <span>Backend Health</span>
              </span>
              <span
                className={`font-mono font-medium ${
                  health?.status === 'ok'
                    ? 'text-emerald-400'
                    : health
                    ? 'text-amber-400'
                    : 'text-rose-400'
                }`}
              >
                {health ? `${health.status.toUpperCase()} (${readyModelCount}/${totalCatalogModels} ready)` : 'Disconnected'}
              </span>
            </div>

            <div className="flex items-center justify-between gap-3">
              <span className="flex items-center gap-2 text-slate-400">
                <Cpu className="h-3.5 w-3.5 text-slate-500" aria-hidden="true" />
                <span>Compute Device</span>
              </span>
              <span className="font-mono text-slate-200 truncate max-w-[200px]">
                {health?.device_name || (healthError ? 'Backend Offline' : 'Checking...')}
              </span>
            </div>

            <div className="flex items-center justify-between gap-3">
              <span className="text-slate-400">VRAM Telemetry</span>
              <span className="font-mono tabular-nums text-slate-200">
                {health?.vram_measured?.total_mb
                  ? `${health.vram_measured.allocated_mb?.toFixed(0) ?? 0} / ${health.vram_measured.total_mb.toFixed(0)} MB`
                  : 'CPU Fallback Active'}
              </span>
            </div>
          </div>

          {health && (
            <div className="divide-y divide-slate-800/80 border-t border-slate-800 pt-2 text-xs">
              {[
                { label: 'AI-Generated · Imager-1-Tiny (Primary)', comp: health.models.ai_generated_detector },
                ...(health.models.univfd_detector
                  ? [{ label: 'AI Comparison · UnivFD CLIP', comp: health.models.univfd_detector }]
                  : []),
                { label: 'Face Manip. · EffNet-B0 FF++', comp: health.models.face_manipulation_detector },
                { label: 'Saliency · DINOv2 ViT-S/14', comp: health.models.dinov2_extractor },
              ].map((row) => (
                <div key={row.label} className="flex items-center justify-between gap-3 py-2.5">
                  <span className="text-slate-300 truncate">{row.label}</span>
                  <span
                    className={`font-mono shrink-0 ${
                      row.comp.status === 'ready'
                        ? 'text-emerald-400'
                        : row.comp.status === 'error'
                        ? 'text-rose-400'
                        : 'text-amber-400'
                    }`}
                  >
                    {row.comp.status}
                  </span>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>

      {/* Recent Investigations Table */}
      <section className="overflow-hidden rounded-xl border border-slate-800 bg-[#0B0E17]">
        <div className="flex items-center justify-between gap-4 border-b border-slate-800 px-5 py-3.5">
          <h3 className="text-sm font-semibold text-slate-100">Recent Investigations</h3>
          <button
            type="button"
            onClick={() => onNavigate('history')}
            className="flex items-center gap-1 text-xs font-medium text-cyan-400 hover:text-cyan-300 cursor-pointer"
          >
            <span>View All History</span>
            <ArrowRight className="h-3.5 w-3.5" aria-hidden="true" />
          </button>
        </div>

        {recentScans.length === 0 ? (
          <div className="p-8 text-center text-xs text-slate-400">
            {loading
              ? 'Loading recent investigations...'
              : 'No archived investigations in SQLite yet. Analyze an image to record the first investigation.'}
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 bg-[#07090E] text-slate-400">
                  <th className="px-4 py-3 font-medium">Timestamp (UTC)</th>
                  <th className="px-4 py-3 font-medium">Filename</th>
                  <th className="px-4 py-3 font-medium text-right">Dimensions</th>
                  <th className="px-4 py-3 font-medium">AI-Generated Signal</th>
                  <th className="px-4 py-3 font-medium">Face Manipulation</th>
                  <th className="px-4 py-3 font-medium text-right">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {recentScans.map((row) => (
                  <tr key={row.scan_id} className="h-11 hover:bg-[#111827]/70 transition-colors">
                    <td className="px-4 py-2 font-mono tabular-nums text-slate-400 whitespace-nowrap">
                      {row.created_at.slice(0, 19).replace('T', ' ')}
                    </td>
                    <td className="px-4 py-2 font-medium text-slate-100 max-w-[220px] truncate">
                      {row.filename}
                    </td>
                    <td className="px-4 py-2 text-right font-mono tabular-nums text-slate-300 whitespace-nowrap">
                      {row.width}×{row.height}
                    </td>
                    <td className="px-4 py-2 font-mono tabular-nums whitespace-nowrap">
                      {row.ai_generated_status === 'evaluated' && row.ai_generated_score !== null ? (
                        <span className={row.ai_generated_score >= 0.5 ? 'text-rose-300' : 'text-emerald-300'}>
                          {(row.ai_generated_score * 100).toFixed(1)}% · {row.ai_generated_label || 'Evaluated'}
                        </span>
                      ) : (
                        <span className="text-slate-400">{row.ai_generated_status}</span>
                      )}
                    </td>
                    <td className="px-4 py-2 font-mono tabular-nums whitespace-nowrap">
                      {row.face_manipulation_status === 'evaluated' && row.face_manipulation_score !== null ? (
                        <span className={row.face_manipulation_score >= 0.5 ? 'text-rose-300' : 'text-emerald-300'}>
                          {(row.face_manipulation_score * 100).toFixed(1)}% · {row.face_manipulation_label || 'Evaluated'}
                        </span>
                      ) : (
                        <span className="text-slate-400">
                          {row.face_manipulation_status === 'no_face_detected'
                            ? 'No Face Detected'
                            : row.face_manipulation_status}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2 text-right whitespace-nowrap">
                      <button
                        type="button"
                        onClick={() => void onInspectScan(row.scan_id)}
                        className="inline-flex items-center gap-1 rounded border border-slate-700 px-2.5 py-1 text-xs text-slate-200 hover:border-cyan-500 hover:text-cyan-300 cursor-pointer"
                      >
                        <Eye className="h-3 w-3" aria-hidden="true" />
                        <span>Inspect</span>
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
