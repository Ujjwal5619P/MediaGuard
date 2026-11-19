import { useCallback, useEffect, useState } from 'react';
import {
  Cpu,
  DownloadCloud,
  RefreshCw,
  Server,
  ShieldCheck,
  Trash2,
} from 'lucide-react';
import {
  DEFAULT_API_BASE_URL,
  fetchModelsStatus,
  purgeExpiredScans,
  triggerModelsSetup,
} from '../services/api';
import type {
  HealthResponse,
  ModelComponentReadiness,
  ModelsStatusResponse,
  PurgeExpiredResponse,
} from '../types/forensics';
import { formatBytes } from '../utils/fileInspector';

interface SystemViewProps {
  apiBaseUrl: string;
  health: HealthResponse | null;
  healthError: string | null;
  onUpdateApiBaseUrl: (url: string) => void;
  onRefreshHealth: () => Promise<void>;
}

const API_ENDPOINTS = [
  { method: 'GET', path: '/api/v1/health', summary: 'Liveness, CUDA verification, measured VRAM & 4-model readiness' },
  { method: 'GET', path: '/api/v1/models/status', summary: 'Persistent checkpoint manifest, SHA-256 verification & telemetry' },
  { method: 'POST', path: '/api/v1/models/setup', summary: 'Download missing checkpoints, verify SHA-256 & load models' },
  { method: 'POST', path: '/api/v1/features/dinov2', summary: 'Extract real 384-d DINOv2 ViT-S/14 image embeddings' },
  { method: 'POST', path: '/api/v1/analyze/image', summary: 'Run Imager-1-Tiny primary + optional local comparison/face/features' },
  { method: 'GET', path: '/api/v1/analytics', summary: 'Real aggregate detection & performance metrics from SQLite' },
  { method: 'GET', path: '/api/v1/evaluation/inspect-folder', summary: 'Count local dataset/ai_generated/ and dataset/real/ images' },
  { method: 'POST', path: '/api/v1/evaluation/run', summary: 'Run offline confusion matrix & metric evaluation without tuning' },
  { method: 'GET', path: '/api/v1/evaluation/runs', summary: 'List archived SQLite dataset evaluation benchmark runs' },
  { method: 'GET', path: '/api/v1/evaluation/runs/{id}/export', summary: 'Download dataset evaluation report as CSV or JSON' },
  { method: 'GET', path: '/api/v1/artifacts/{filename}', summary: 'Serve generated Grad-CAM or DINOv2 patch attention PNGs' },
  { method: 'GET', path: '/api/v1/scans', summary: 'Paginated SQLite scan ledger with search & status filtering' },
  { method: 'GET', path: '/api/v1/scans/{scan_id}', summary: 'Retrieve full archived investigation metadata & indicators' },
  { method: 'DELETE', path: '/api/v1/scans/{scan_id}', summary: 'Delete scan metadata record and any linked artifact' },
  { method: 'DELETE', path: '/api/v1/scans/{scan_id}/artifact', summary: 'Purge attribution artifact only while preserving scan record' },
  { method: 'POST', path: '/api/v1/scans/purge-expired', summary: 'Delete expired records and attribution artifacts per TTL policy' },
  { method: 'GET', path: '/api/v1/scans/export', summary: 'Download filtered scan ledger as CSV or JSON' },
];

export function SystemView({
  apiBaseUrl,
  health,
  healthError,
  onUpdateApiBaseUrl,
  onRefreshHealth,
}: SystemViewProps) {
  const [draftUrl, setDraftUrl] = useState(apiBaseUrl);
  const [probing, setProbing] = useState(false);
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const [settingUpModels, setSettingUpModels] = useState(false);
  const [purging, setPurging] = useState(false);
  const [purgeResult, setPurgeResult] = useState<PurgeExpiredResponse | null>(null);
  const [purgeError, setPurgeError] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    try {
      const data = await fetchModelsStatus(apiBaseUrl);
      setModelsStatus(data);
      setModelsError(null);
    } catch (err) {
      setModelsStatus(null);
      setModelsError(
        err instanceof Error ? err.message : 'Unable to fetch model manifest.'
      );
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const handleSave = async () => {
    onUpdateApiBaseUrl(draftUrl);
    setProbing(true);
    try {
      await onRefreshHealth();
      await loadModels();
    } finally {
      setProbing(false);
    }
  };

  const handleReset = () => {
    setDraftUrl(DEFAULT_API_BASE_URL);
    onUpdateApiBaseUrl(DEFAULT_API_BASE_URL);
  };

  const handleSetupModels = async () => {
    setSettingUpModels(true);
    setModelsError(null);
    try {
      const updated = await triggerModelsSetup(apiBaseUrl);
      setModelsStatus(updated);
      await onRefreshHealth();
    } catch (err) {
      setModelsError(
        err instanceof Error ? err.message : 'Model download/setup failed.'
      );
    } finally {
      setSettingUpModels(false);
    }
  };

  const handlePurgeExpired = async () => {
    setPurging(true);
    setPurgeError(null);
    try {
      const res = await purgeExpiredScans(apiBaseUrl);
      setPurgeResult(res);
    } catch (err) {
      setPurgeError(
        err instanceof Error ? err.message : 'Unable to execute retention purge.'
      );
    } finally {
      setPurging(false);
    }
  };

  const renderComponentRow = (label: string, comp: ModelComponentReadiness) => (
    <tr key={label} className="h-11">
      <td className="px-4 py-2 font-medium text-slate-200">{label}</td>
      <td className="px-4 py-2 font-mono text-slate-300">{comp.model_id}</td>
      <td className="px-4 py-2 font-mono">
        <span
          className={
            comp.status === 'ready'
              ? 'text-emerald-400'
              : comp.status === 'error'
              ? 'text-rose-400'
              : 'text-amber-400'
          }
        >
          {comp.status}
        </span>
      </td>
      <td className="px-4 py-2 font-mono text-slate-300">
        {comp.checkpoint_verified ? 'Verified' : 'Unverified / Absent'}
      </td>
      <td className="px-4 py-2 text-slate-400">{comp.reason}</td>
    </tr>
  );

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-6 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">
            System Settings &amp; Infrastructure Diagnostics
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Backend API connectivity, CUDA GPU telemetry, storage retention policies, model verification, and application diagnostics.
          </p>
        </div>
        <div className="font-mono text-xs text-slate-400">
          Application Version: <span className="text-slate-100">v{health?.version || '1.2.0'}</span>
        </div>
      </div>

      {/* API Connectivity Configuration */}
      <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Server className="h-4 w-4 text-cyan-400" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-slate-100">API Connectivity Configuration</h3>
          </div>
          <div className="font-mono text-xs">
            Status:{' '}
            <span
              className={
                health?.status === 'ok'
                  ? 'text-emerald-400'
                  : health
                  ? 'text-amber-400'
                  : 'text-rose-400'
              }
            >
              {health ? `${health.status.toUpperCase()} (${health.service})` : 'Disconnected'}
            </span>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <input
            type="url"
            value={draftUrl}
            onChange={(e) => setDraftUrl(e.target.value)}
            aria-label="FastAPI Base URL"
            placeholder="http://127.0.0.1:8000"
            className="flex-1 min-w-[260px] rounded-lg border border-slate-800 bg-[#07090E] px-3 py-2 font-mono text-xs text-slate-100 focus:border-cyan-500 focus:outline-none"
          />
          <button
            type="button"
            disabled={probing}
            onClick={() => void handleSave()}
            className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 transition-colors cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${probing ? 'animate-spin' : ''}`} aria-hidden="true" />
            <span>Save &amp; Probe /api/v1/health</span>
          </button>
          <button
            type="button"
            onClick={handleReset}
            className="rounded-lg border border-slate-800 bg-[#111827] px-3 py-2 text-xs font-medium text-slate-300 hover:text-white cursor-pointer"
          >
            Reset Default
          </button>
        </div>

        {healthError && !health && (
          <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 p-3 text-xs text-amber-200">
            {healthError}
          </div>
        )}
      </section>

      {/* CUDA Status & GPU Telemetry */}
      {health && (
        <>
          <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
              <div className="flex items-center gap-1.5 text-xs text-slate-400">
                <Cpu className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
                <span>CUDA Status</span>
              </div>
              <div className="mt-1 font-mono text-sm font-semibold text-slate-100">
                {health.cuda_available ? 'CUDA Enabled' : 'CPU Fallback Active'}
              </div>
              <div className="mt-1 text-xs text-slate-500">{health.execution_mode}</div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
              <div className="text-xs text-slate-400">GPU / Compute Device</div>
              <div className="mt-1 font-mono text-sm font-semibold text-slate-100">
                {health.device_name}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
              <div className="text-xs text-slate-400">Measured VRAM (Alloc / Peak / Total)</div>
              <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-slate-100">
                {health.vram_measured?.allocated_mb !== null && health.vram_measured?.allocated_mb !== undefined
                  ? `${health.vram_measured.allocated_mb.toFixed(1)} / ${(health.vram_measured.peak_mb ?? 0).toFixed(1)} / ${(health.vram_measured.total_mb ?? 0).toFixed(0)} MB`
                  : 'N/A (CPU Mode)'}
              </div>
            </div>
            <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
              <div className="text-xs text-slate-400">Concurrency &amp; Offload</div>
              <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-slate-100">
                Sequential ({health.max_concurrent_gpu_workers} worker · Offload {health.low_vram_offload ? 'ON' : 'OFF'})
              </div>
            </div>
          </section>

          <section className="overflow-hidden rounded-xl border border-slate-800 bg-[#0B0E17]">
            <div className="border-b border-slate-800 px-4 py-3">
              <h3 className="text-sm font-semibold text-slate-100">
                Pipeline Readiness Matrix
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-left text-xs">
                <thead>
                  <tr className="border-b border-slate-800 bg-[#07090E] text-slate-400">
                    <th className="px-4 py-2.5 font-medium">Component</th>
                    <th className="px-4 py-2.5 font-medium">Identifier</th>
                    <th className="px-4 py-2.5 font-medium">Status</th>
                    <th className="px-4 py-2.5 font-medium">Checkpoint</th>
                    <th className="px-4 py-2.5 font-medium">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/80">
                  {renderComponentRow('Pipeline A (Primary): Imager-1-Tiny · CPU', health.models.ai_generated_detector)}
                  {health.models.univfd_detector &&
                    renderComponentRow('Pipeline A (Comparison): UnivFD', health.models.univfd_detector)}
                  {renderComponentRow('Pipeline B: Face Manip.', health.models.face_manipulation_detector)}
                  {renderComponentRow('Feature Extractor (DINOv2)', health.models.dinov2_extractor)}
                  {renderComponentRow('Calibration Profile', health.models.calibration_profile)}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}

      {/* Model Download Summary + Storage & Retention Settings */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-semibold text-slate-100">
                Model Download Status
              </h3>
              <p className="mt-0.5 font-mono text-[11px] text-slate-400">
                {modelsStatus
                  ? `Verified: ${String(modelsStatus.all_verified)} · Loaded: ${String(modelsStatus.all_loaded)}`
                  : 'Connect FastAPI to inspect checkpoint files'}
              </p>
            </div>
            <button
              type="button"
              disabled={settingUpModels || !health}
              onClick={() => void handleSetupModels()}
              className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500 cursor-pointer"
            >
              <DownloadCloud className="h-3.5 w-3.5" aria-hidden="true" />
              <span>{settingUpModels ? 'Verifying...' : 'Download & Verify Models'}</span>
            </button>
          </div>

          {modelsError && (
            <div className="rounded-lg border border-amber-900/60 bg-amber-950/20 px-3 py-2 text-xs text-amber-200">
              {modelsError}
            </div>
          )}

          {modelsStatus && (
            <div className="divide-y divide-slate-800/80 border-t border-slate-800 pt-2 text-xs">
              {modelsStatus.models.map((m) => (
                <div key={m.key} className="flex items-center justify-between gap-3 py-2.5">
                  <div className="min-w-0">
                    <div className="font-medium text-slate-200 truncate">{m.name}</div>
                    <div className="font-mono text-[11px] text-slate-500 truncate">{m.repo_id}</div>
                  </div>
                  <div className="font-mono tabular-nums text-right shrink-0">
                    <span className={m.verified && m.loaded_in_memory ? 'text-emerald-400' : 'text-amber-400'}>
                      {m.status}
                    </span>
                    <div className="text-[11px] text-slate-500">{m.size_mb.toFixed(1)} MB</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* Storage & Retention Policy + Manual Purge */}
        <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <ShieldCheck className="h-4 w-4 text-emerald-400" aria-hidden="true" />
              <h3 className="text-sm font-semibold text-slate-100">
                Storage &amp; Retention Settings
              </h3>
            </div>
            <button
              type="button"
              disabled={purging || !health}
              onClick={() => void handlePurgeExpired()}
              className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-[#111827] px-3 py-1.5 text-xs font-medium text-amber-300 hover:border-amber-500 disabled:cursor-not-allowed disabled:opacity-50 cursor-pointer"
            >
              <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
              <span>{purging ? 'Purging...' : 'Purge Expired Records'}</span>
            </button>
          </div>

          {health ? (
            <dl className="grid grid-cols-1 gap-3 border-t border-slate-800 pt-3 text-xs sm:grid-cols-2">
              <div>
                <dt className="text-slate-400">Original Upload Handling</dt>
                <dd className="mt-1 font-mono text-slate-200">
                  {health.retention_policy.delete_original_after_analysis
                    ? 'Deleted immediately after analysis'
                    : 'Retained'}
                </dd>
              </div>
              <div>
                <dt className="text-slate-400">Split Retention TTL</dt>
                <dd className="mt-1 font-mono tabular-nums text-slate-200">
                  Records: {health.retention_policy.record_retention_days}d · Artifacts: {health.retention_policy.artifact_retention_days}d
                </dd>
              </div>
              <div>
                <dt className="text-slate-400">Upload Security Bounds</dt>
                <dd className="mt-1 font-mono tabular-nums text-slate-200">
                  Max {formatBytes(health.retention_policy.max_upload_bytes)} · {health.retention_policy.max_dimension}px
                </dd>
              </div>
              <div>
                <dt className="text-slate-400">Training Opt-Out</dt>
                <dd className="mt-1 font-mono text-slate-200">
                  {health.retention_policy.never_train_without_consent
                    ? 'Enforced (never trained)'
                    : 'Configured'}
                </dd>
              </div>
            </dl>
          ) : (
            <p className="text-xs text-slate-400">
              Connect the FastAPI backend to inspect live retention settings.
            </p>
          )}

          {purgeResult && (
            <div className="rounded-lg border border-emerald-900/60 bg-emerald-950/20 px-3 py-2 font-mono text-xs text-emerald-200">
              Purged {purgeResult.expired_records_deleted} expired records and {purgeResult.expired_artifacts_purged} expired attribution artifacts.
            </div>
          )}
          {purgeError && (
            <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 px-3 py-2 text-xs text-rose-200">
              {purgeError}
            </div>
          )}
        </section>
      </div>

      {/* Normalized REST API Reference & Local Launch Instructions */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="overflow-hidden rounded-xl border border-slate-800 bg-[#0B0E17]">
          <div className="border-b border-slate-800 px-4 py-3">
            <h3 className="text-sm font-semibold text-slate-100">
              Normalized API v1 Contract
            </h3>
          </div>
          <div className="divide-y divide-slate-800/80 text-xs">
            {API_ENDPOINTS.map((ep) => (
              <div key={`${ep.method}-${ep.path}`} className="flex items-start justify-between gap-4 px-4 py-2.5">
                <div className="font-mono text-slate-200 whitespace-nowrap">
                  <span className="text-cyan-400">{ep.method}</span> {ep.path}
                </div>
                <div className="text-right text-slate-400">{ep.summary}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-3">
          <h3 className="text-sm font-semibold text-slate-100">
            Windows Conda Launch Commands (mediaguard · RTX 2050)
          </h3>
          <p className="text-xs leading-relaxed text-slate-400">
            Uses the existing <span className="font-mono text-slate-200">mediaguard</span> Conda environment. The primary Imager-1-Tiny detector runs on CPU; optional cached models stay available in normal local mode. No package upgrade is needed when the environment already has the documented dependencies.
          </p>
          <pre className="overflow-x-auto rounded-lg border border-slate-800 bg-[#07090E] p-3 font-mono text-[11px] leading-relaxed text-slate-300">
{`cd backend
conda activate mediaguard
python setup_models.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1`}
          </pre>
        </section>
      </div>
    </div>
  );
}
