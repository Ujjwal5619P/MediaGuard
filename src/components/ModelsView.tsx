import { useCallback, useEffect, useState } from 'react';
import {
  CheckCircle2,
  Cpu,
  DownloadCloud,
  ExternalLink,
  HardDrive,
  RefreshCw,
  XCircle,
} from 'lucide-react';
import {
  ApiClientError,
  fetchModelsStatus,
  triggerModelsSetup,
} from '../services/api';
import type { HealthResponse, ModelsStatusResponse } from '../types/forensics';

interface ModelsViewProps {
  apiBaseUrl: string;
  health: HealthResponse | null;
  onRefreshHealth: () => Promise<void>;
}

export function ModelsView({ apiBaseUrl, health, onRefreshHealth }: ModelsViewProps) {
  const [modelsStatus, setModelsStatus] = useState<ModelsStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [settingUp, setSettingUp] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadModels = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchModelsStatus(apiBaseUrl);
      setModelsStatus(data);
      setError(null);
    } catch (err) {
      setModelsStatus(null);
      setError(
        err instanceof ApiClientError
          ? err.message
          : 'Unable to fetch AI model manifest from FastAPI.'
      );
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl]);

  useEffect(() => {
    void loadModels();
  }, [loadModels]);

  const handleSetup = async () => {
    setSettingUp(true);
    setError(null);
    try {
      const updated = await triggerModelsSetup(apiBaseUrl);
      setModelsStatus(updated);
      await onRefreshHealth();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Checkpoint setup failed.');
    } finally {
      setSettingUp(false);
    }
  };

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-6 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">
            AI Models &amp; Verified Checkpoint Registry
          </h2>
          <p className="mt-1 text-xs text-slate-400">
            Persistent local Hugging Face checkpoints, pinned SHA-256 manifests, licenses, execution devices, and measured runtime telemetry.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            disabled={loading}
            onClick={() => void loadModels()}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
            <span>Refresh Manifest</span>
          </button>
          <button
            type="button"
            disabled={settingUp || !health}
            onClick={() => void handleSetup()}
            className="flex items-center gap-1.5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500 cursor-pointer"
          >
            <DownloadCloud className="h-3.5 w-3.5" aria-hidden="true" />
            <span>
              {settingUp || modelsStatus?.download_in_progress
                ? 'Downloading & Verifying...'
                : 'Download & Verify Checkpoints'}
            </span>
          </button>
        </div>
      </div>

      {/* Hardware Telemetry Strip */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
          <div className="text-xs text-slate-400">Active Compute Device</div>
          <div className="mt-1 font-mono text-sm font-semibold text-slate-100">
            {health?.device_name || 'Backend Offline'}
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
          <div className="text-xs text-slate-400">VRAM Allocated / Peak / Total</div>
          <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-slate-100">
            {health?.vram_measured?.total_mb
              ? `${health.vram_measured.allocated_mb?.toFixed(1) ?? 0} / ${health.vram_measured.peak_mb?.toFixed(1) ?? 0} / ${health.vram_measured.total_mb.toFixed(0)} MB`
              : 'N/A (CPU Mode)'}
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
          <div className="text-xs text-slate-400">Execution Worker Policy</div>
          <div className="mt-1 font-mono text-sm font-semibold tabular-nums text-slate-100">
            {health
              ? `1 GPU Worker · Offload ${health.low_vram_offload ? 'ON' : 'OFF'}`
              : '—'}
          </div>
        </div>
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-4">
          <div className="text-xs text-slate-400">Offline Readiness</div>
          <div className="mt-1 font-mono text-sm font-semibold text-slate-100">
            {modelsStatus
              ? modelsStatus.all_verified && modelsStatus.all_loaded
                ? 'Offline Ready (All Loaded)'
                : modelsStatus.all_verified
                ? 'Verified (Loading / Idle)'
                : 'Incomplete Checkpoints'
              : '—'}
          </div>
        </div>
      </section>

      {error && (
        <div className="rounded-xl border border-rose-900/60 bg-rose-950/20 p-4 text-xs text-rose-200">
          {error}
        </div>
      )}

      {modelsStatus && (
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] px-4 py-3 text-xs font-mono text-slate-400 flex flex-wrap items-center justify-between gap-2">
          <span className="flex items-center gap-2">
            <HardDrive className="h-3.5 w-3.5 text-cyan-400" aria-hidden="true" />
            <span>Storage Directory: {modelsStatus.models_dir}</span>
          </span>
          <span>Offline CLI: python setup_models.py</span>
        </div>
      )}

      {/* Individual Model Cards */}
      {modelsStatus && (
        <div className="space-y-4">
          {modelsStatus.models.map((m) => {
            const roleLabel =
              m.key === 'ai_generated_detector'
                ? 'Primary AI-Generated Detector'
                : m.key === 'univfd_detector'
                ? 'Secondary Comparison Detector'
                : m.key === 'face_manipulation_detector'
                ? 'Facial Manipulation Classifier'
                : 'Self-Supervised Feature Extractor';

            return (
            <section
              key={m.key}
              className="rounded-xl border border-slate-800 bg-[#0B0E17] p-6 space-y-4"
            >
              <div className="flex flex-wrap items-start justify-between gap-4">
                <div className="space-y-1">
                  <div className="font-mono text-[11px] uppercase tracking-wider text-cyan-400">
                    {roleLabel}
                  </div>
                  <div className="flex items-center gap-2.5">
                    <Cpu className="h-4 w-4 text-cyan-400 shrink-0" aria-hidden="true" />
                    <h3 className="text-base font-semibold text-slate-100">{m.name}</h3>
                  </div>
                  <p className="font-mono text-xs text-slate-400">
                    {m.repo_id} · Revision {m.revision.slice(0, 12)}
                  </p>
                </div>

                <div className="flex items-center gap-4 font-mono text-xs tabular-nums">
                  <span className="flex items-center gap-1.5">
                    {m.verified && m.loaded_in_memory ? (
                      <CheckCircle2 className="h-4 w-4 text-emerald-400" aria-hidden="true" />
                    ) : m.status === 'error' ? (
                      <XCircle className="h-4 w-4 text-rose-400" aria-hidden="true" />
                    ) : null}
                    <span
                      className={
                        m.loaded_in_memory
                          ? 'text-emerald-400 font-semibold'
                          : m.status === 'error'
                          ? 'text-rose-400 font-semibold'
                          : 'text-amber-400 font-semibold'
                      }
                    >
                      {m.loaded_in_memory ? 'Loaded & Ready' : m.status}
                    </span>
                  </span>
                  <span className="text-slate-600" aria-hidden="true">·</span>
                  <span className="text-slate-300">{m.size_mb.toFixed(2)} MB</span>
                </div>
              </div>

              <dl className="grid grid-cols-1 gap-4 border-t border-slate-800 pt-4 text-xs md:grid-cols-2 lg:grid-cols-4">
                <div>
                  <dt className="text-slate-500">Architecture</dt>
                  <dd className="mt-1 font-mono text-slate-200 leading-relaxed">{m.architecture}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">Intended Task</dt>
                  <dd className="mt-1 text-slate-200 leading-relaxed">{m.intended_task}</dd>
                </div>
                <div>
                  <dt className="text-slate-500">License &amp; Provenance</dt>
                  <dd className="mt-1 text-slate-200 leading-relaxed">
                    <div>{m.license}</div>
                    <a
                      href={m.source_url}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-1 inline-flex items-center gap-1 font-mono text-[11px] text-cyan-400 hover:text-cyan-300"
                    >
                      <span>{m.source_url.replace(/^https?:\/\//, '')}</span>
                      <ExternalLink className="h-3 w-3" aria-hidden="true" />
                    </a>
                  </dd>
                </div>
                <div>
                  <dt className="text-slate-500">Runtime Telemetry</dt>
                  <dd className="mt-1 font-mono tabular-nums text-slate-200 space-y-1">
                    <div>Device: {m.active_device || 'cpu'}</div>
                    <div>
                      Last Latency:{' '}
                      {m.last_inference_ms !== null ? `${m.last_inference_ms} ms` : 'Not run yet'}
                    </div>
                    <div>
                      Peak VRAM:{' '}
                      {m.last_vram_peak_mb !== null ? `${m.last_vram_peak_mb} MB` : 'N/A (CPU)'}
                    </div>
                  </dd>
                </div>
              </dl>

              <div className="grid grid-cols-1 gap-4 border-t border-slate-800/80 pt-3 text-xs md:grid-cols-2">
                <div>
                  <span className="text-slate-500">Preprocessing: </span>
                  <span className="font-mono text-slate-300">{m.preprocessing}</span>
                </div>
                <div>
                  <span className="text-slate-500">Primary SHA-256: </span>
                  <span className="font-mono text-slate-300 break-all">
                    {m.actual_sha256 || `Expected ${m.expected_sha256}`}
                  </span>
                </div>
              </div>

              {m.error_message && (
                <div className="rounded-lg border border-rose-900/60 bg-rose-950/20 px-3 py-2 text-xs text-rose-200">
                  {m.error_message}
                </div>
              )}
            </section>
            );
          })}
        </div>
      )}
    </div>
  );
}
