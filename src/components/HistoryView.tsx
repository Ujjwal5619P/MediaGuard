import { useCallback, useEffect, useState } from 'react';
import {
  ArrowUpRight,
  Download,
  Eye,
  FileSearch,
  RefreshCw,
  Search,
  ShieldOff,
  Trash2,
  X,
} from 'lucide-react';
import {
  ApiClientError,
  deleteScanArtifact,
  deleteScanRecord,
  downloadScansExport,
  fetchScanById,
  fetchScans,
} from '../services/api';
import type {
  AnalyzeImageResponse,
  AppView,
  ScanSummary,
  TaskStatus,
} from '../types/forensics';
import { formatBytes } from '../utils/fileInspector';

interface HistoryViewProps {
  apiBaseUrl: string;
  onInspectScan: (scanId: string) => Promise<void>;
  onNavigate: (view: AppView) => void;
}

const FILTER_OPTIONS: Array<{ value: TaskStatus | 'all'; label: string }> = [
  { value: 'all', label: 'All Scans' },
  { value: 'evaluated', label: 'Evaluated' },
  { value: 'no_face_detected', label: 'No Face' },
  { value: 'unavailable', label: 'Unavailable' },
];

export function HistoryView({ apiBaseUrl, onInspectScan, onNavigate }: HistoryViewProps) {
  const [items, setItems] = useState<ScanSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<TaskStatus | 'all'>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [detailModal, setDetailModal] = useState<AnalyzeImageResponse | null>(null);

  const loadScans = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await fetchScans(apiBaseUrl, {
        limit: 50,
        offset: 0,
        search,
        status: statusFilter,
      });
      setItems(resp.items);
      setTotal(resp.total);
    } catch (err) {
      setItems([]);
      setTotal(0);
      if (err instanceof ApiClientError) {
        setError(err.message);
      } else {
        setError('Unable to load SQLite scan history.');
      }
    } finally {
      setLoading(false);
    }
  }, [apiBaseUrl, search, statusFilter]);

  useEffect(() => {
    void loadScans();
  }, [loadScans]);

  const handleOpenDetails = async (scanId: string) => {
    setBusyId(scanId);
    try {
      const record = await fetchScanById(apiBaseUrl, scanId);
      setDetailModal(record);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to fetch archived investigation details.');
    } finally {
      setBusyId(null);
    }
  };

  const handleDeleteRecord = async (scanId: string) => {
    setBusyId(scanId);
    try {
      await deleteScanRecord(apiBaseUrl, scanId);
      if (detailModal?.scan_id === scanId) {
        setDetailModal(null);
      }
      await loadScans();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete scan record.');
    } finally {
      setBusyId(null);
    }
  };

  const handlePurgeArtifact = async (scanId: string) => {
    setBusyId(scanId);
    try {
      await deleteScanArtifact(apiBaseUrl, scanId);
      if (detailModal?.scan_id === scanId) {
        const refreshed = await fetchScanById(apiBaseUrl, scanId);
        setDetailModal(refreshed);
      }
      await loadScans();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to purge attribution artifact.');
    } finally {
      setBusyId(null);
    }
  };

  const handleExport = async (fmt: 'csv' | 'json') => {
    try {
      await downloadScansExport(apiBaseUrl, fmt, statusFilter);
    } catch (err) {
      setError(err instanceof Error ? err.message : `Failed to export ${fmt.toUpperCase()}.`);
    }
  };

  const formatScoreOrStatus = (
    status: TaskStatus,
    score: number | null,
    label: string | null
  ) => {
    if (status === 'evaluated' && score !== null) {
      const pct = `${(score * 100).toFixed(1)}%`;
      return label ? `${pct} · ${label}` : pct;
    }
    if (status === 'no_face_detected') return 'No Face Detected';
    if (status === 'inconclusive') return 'Inconclusive';
    if (status === 'error') return 'Error';
    return 'Unavailable';
  };

  return (
    <div className="mx-auto max-w-[1440px] px-6 py-6 space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold text-slate-100">SQLite Investigation History</h2>
          <p className="mt-1 text-xs text-slate-400">
            Archived forensic investigation records, model versions, execution metrics, and artifact retention controls. Originals are automatically deleted after analysis.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            onClick={() => void handleExport('csv')}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Export CSV</span>
          </button>
          <button
            type="button"
            onClick={() => void handleExport('json')}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Export JSON</span>
          </button>
          <button
            type="button"
            onClick={() => void loadScans()}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-2 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Search & Segmented Filters */}
      <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-[#0B0E17] p-3">
        <div className="relative flex-1 min-w-[240px] max-w-md">
          <Search
            className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500"
            aria-hidden="true"
          />
          <input
            type="search"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter by filename or SHA-256 prefix..."
            aria-label="Filter investigation history"
            className="w-full rounded-lg border border-slate-800 bg-[#07090E] py-2 pl-9 pr-3 text-xs text-slate-100 placeholder:text-slate-500 focus:border-cyan-500 focus:outline-none"
          />
        </div>

        <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-[#07090E] p-1">
          {FILTER_OPTIONS.map((opt) => (
            <button
              key={opt.value}
              type="button"
              onClick={() => setStatusFilter(opt.value)}
              className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                statusFilter === opt.value
                  ? 'bg-slate-800 text-slate-100'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Table / States */}
      <div className="overflow-hidden rounded-xl border border-slate-800 bg-[#0B0E17]">
        {error ? (
          <div className="p-8 text-center">
            <h3 className="text-sm font-semibold text-amber-300">
              FastAPI History Store Unreachable
            </h3>
            <p className="mt-2 text-xs text-slate-400">{error}</p>
            <div className="mt-4 flex items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => void loadScans()}
                className="rounded-lg border border-slate-700 bg-[#111827] px-4 py-2 text-xs font-medium text-slate-200 hover:text-white cursor-pointer"
              >
                Retry Connection
              </button>
              <button
                type="button"
                onClick={() => onNavigate('system')}
                className="rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer"
              >
                System Settings
              </button>
            </div>
          </div>
        ) : loading ? (
          <div className="divide-y divide-slate-800/80">
            {Array.from({ length: 5 }).map((_, idx) => (
              <div key={idx} className="flex h-12 items-center justify-between px-4 animate-pulse">
                <div className="h-3 w-36 rounded bg-slate-800" />
                <div className="h-3 w-48 rounded bg-slate-800" />
                <div className="h-3 w-28 rounded bg-slate-800" />
                <div className="h-3 w-24 rounded bg-slate-800" />
              </div>
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="p-10 text-center">
            <h3 className="text-sm font-semibold text-slate-200">No Investigation Records Found</h3>
            <p className="mt-1.5 text-xs text-slate-400">
              Run an image investigation to persist forensic records and model scores in SQLite.
            </p>
            <button
              type="button"
              onClick={() => onNavigate('investigation')}
              className="mt-4 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer"
            >
              Open New Investigation
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-slate-800 bg-[#07090E] text-slate-400">
                  <th className="px-4 py-3 font-medium">Timestamp (UTC)</th>
                  <th className="px-4 py-3 font-medium">File &amp; SHA-256</th>
                  <th className="px-4 py-3 font-medium">AI-Generated Signal</th>
                  <th className="px-4 py-3 font-medium">Face Manipulation</th>
                  <th className="px-4 py-3 font-medium">Models &amp; Device</th>
                  <th className="px-4 py-3 font-medium">Retention</th>
                  <th className="px-4 py-3 font-medium text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/80">
                {items.map((row) => (
                  <tr key={row.scan_id} className="hover:bg-[#111827]/70 transition-colors">
                    <td className="px-4 py-3 font-mono tabular-nums text-slate-300 whitespace-nowrap">
                      {row.created_at.slice(0, 19).replace('T', ' ')}
                    </td>
                    <td className="px-4 py-3">
                      <div className="max-w-[200px] truncate font-medium text-slate-100" title={row.filename}>
                        {row.filename}
                      </div>
                      <div className="font-mono text-[11px] text-slate-500 tabular-nums">
                        {row.sha256.slice(0, 14)}… · {row.width}×{row.height} · {formatBytes(row.byte_size)}
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums whitespace-nowrap">
                      <div
                        className={
                          row.ai_generated_status === 'evaluated' && (row.ai_generated_score ?? 0) >= 0.5
                            ? 'text-rose-300'
                            : row.ai_generated_status === 'evaluated'
                            ? 'text-emerald-300'
                            : 'text-slate-400'
                        }
                      >
                        {formatScoreOrStatus(
                          row.ai_generated_status,
                          row.ai_generated_score,
                          row.ai_generated_label
                        )}
                      </div>
                      {row.univfd_status && (
                        <div className="mt-0.5 text-[11px] text-slate-400">
                          UnivFD:{' '}
                          {row.univfd_status === 'evaluated' && row.univfd_score !== null && row.univfd_score !== undefined
                            ? `${(row.univfd_score * 100).toFixed(1)}%`
                            : row.univfd_status}
                        </div>
                      )}
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums whitespace-nowrap">
                      <span
                        className={
                          row.face_manipulation_status === 'evaluated' && (row.face_manipulation_score ?? 0) >= 0.5
                            ? 'text-rose-300'
                            : row.face_manipulation_status === 'evaluated'
                            ? 'text-emerald-300'
                            : 'text-slate-400'
                        }
                      >
                        {formatScoreOrStatus(
                          row.face_manipulation_status,
                          row.face_manipulation_score,
                          row.face_manipulation_label
                        )}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono text-[11px] text-slate-400 whitespace-nowrap">
                      <div>{row.ai_model_id || 'Imagers/Imager-1-tiny'}</div>
                      <div className="tabular-nums text-slate-500">
                        {row.execution_device || 'cpu'}
                        {row.inference_ms !== null && row.inference_ms !== undefined
                          ? ` · ${row.inference_ms} ms`
                          : ''}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-slate-400 whitespace-nowrap">
                      <span>{row.original_deleted ? 'Orig. Deleted' : 'Orig. Kept'}</span>
                      <span aria-hidden="true" className="mx-1.5 text-slate-600">·</span>
                      <span>{row.has_artifact ? 'Artifact Stored' : 'No Artifact'}</span>
                    </td>
                    <td className="px-4 py-3 text-right whitespace-nowrap">
                      <div className="inline-flex items-center gap-1.5">
                        <button
                          type="button"
                          disabled={busyId === row.scan_id}
                          onClick={() => void handleOpenDetails(row.scan_id)}
                          title="View full investigation record details"
                          className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-cyan-500 hover:text-cyan-300 cursor-pointer"
                        >
                          <FileSearch className="h-3 w-3" aria-hidden="true" />
                          <span>Details</span>
                        </button>
                        <button
                          type="button"
                          disabled={busyId === row.scan_id}
                          onClick={() => void onInspectScan(row.scan_id)}
                          title="Load record into Investigation Workspace"
                          className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 text-xs text-slate-200 hover:border-cyan-500 hover:text-cyan-300 cursor-pointer"
                        >
                          <Eye className="h-3 w-3" aria-hidden="true" />
                          <span>Workspace</span>
                        </button>
                        {row.has_artifact && (
                          <button
                            type="button"
                            disabled={busyId === row.scan_id}
                            onClick={() => void handlePurgeArtifact(row.scan_id)}
                            title="Delete experimental attribution artifact only"
                            className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 text-xs text-amber-300 hover:border-amber-500 cursor-pointer"
                          >
                            <ShieldOff className="h-3 w-3" aria-hidden="true" />
                            <span>Purge</span>
                          </button>
                        )}
                        <button
                          type="button"
                          disabled={busyId === row.scan_id}
                          onClick={() => void handleDeleteRecord(row.scan_id)}
                          title="Delete SQLite scan record"
                          className="inline-flex items-center gap-1 rounded border border-slate-700 px-2 py-1 text-xs text-rose-300 hover:border-rose-500 cursor-pointer"
                        >
                          <Trash2 className="h-3 w-3" aria-hidden="true" />
                          <span>Delete</span>
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {!error && !loading && (
        <div className="text-xs font-mono tabular-nums text-slate-400">
          Showing {items.length} of {total} persisted SQLite records
        </div>
      )}

      {/* Archived Investigation Details Modal */}
      {detailModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-label="Investigation Details"
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
        >
          <div className="w-full max-w-2xl rounded-xl border border-slate-700 bg-[#0B0E17] p-6 space-y-4 max-h-[88vh] overflow-y-auto">
            <div className="flex items-start justify-between gap-4 border-b border-slate-800 pb-3">
              <div>
                <h3 className="text-base font-semibold text-slate-100">
                  Investigation Record Details
                </h3>
                <p className="mt-0.5 font-mono text-xs text-slate-400">{detailModal.scan_id}</p>
              </div>
              <button
                type="button"
                onClick={() => setDetailModal(null)}
                aria-label="Close investigation details"
                className="rounded-lg border border-slate-800 p-1.5 text-slate-400 hover:text-white cursor-pointer"
              >
                <X className="h-4 w-4" aria-hidden="true" />
              </button>
            </div>

            <dl className="grid grid-cols-1 gap-3 text-xs sm:grid-cols-2">
              <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                <dt className="text-slate-400">File Metadata</dt>
                <dd className="mt-1 font-mono text-slate-200 space-y-1">
                  <div className="truncate">{detailModal.file_metadata.filename}</div>
                  <div>
                    {detailModal.file_metadata.detected_mime} · {detailModal.file_metadata.width}×{detailModal.file_metadata.height} px · {formatBytes(detailModal.file_metadata.byte_size)}
                  </div>
                  <div className="break-all text-[11px] text-slate-400">
                    SHA-256: {detailModal.sha256}
                  </div>
                </dd>
              </div>

              <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                <dt className="text-slate-400">Execution &amp; Privacy</dt>
                <dd className="mt-1 font-mono text-slate-200 space-y-1 tabular-nums">
                  <div>Device: {detailModal.execution_device}</div>
                  <div>
                    Latency: {detailModal.inference_ms} ms
                    {detailModal.vram_peak_mb !== null ? ` · VRAM ${detailModal.vram_peak_mb} MB` : ''}
                  </div>
                  <div className="text-emerald-300">
                    {detailModal.original_deleted ? 'Original Deleted' : 'Original Retained'}
                  </div>
                </dd>
              </div>

              <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                <dt className="text-slate-400">Primary AI-Generated Indicator (Imager-1-Tiny)</dt>
                <dd className="mt-1 font-mono text-slate-200 space-y-1 tabular-nums">
                  <div>Model: {detailModal.ai_generated_indicator.model_name || detailModal.ai_generated_indicator.model_id}</div>
                  <div>
                    Score:{' '}
                    {detailModal.ai_generated_indicator.score !== null
                      ? `${(detailModal.ai_generated_indicator.score * 100).toFixed(2)}%`
                      : detailModal.ai_generated_indicator.status}{' '}
                    ({detailModal.ai_generated_indicator.prediction_label || '—'})
                  </div>
                  <div className="text-[11px] text-slate-400">
                    {detailModal.ai_generated_indicator.calibrated ? 'Calibrated' : 'Uncalibrated raw output'}
                  </div>
                </dd>
              </div>

              {detailModal.univfd_indicator && (
                <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                  <dt className="text-slate-400">Secondary Comparison (UnivFD · CLIP ViT-L/14)</dt>
                  <dd className="mt-1 font-mono text-slate-200 space-y-1 tabular-nums">
                    <div>Model: {detailModal.univfd_indicator.model_name || detailModal.univfd_indicator.model_id}</div>
                    <div>
                      Score:{' '}
                      {detailModal.univfd_indicator.score !== null
                        ? `${(detailModal.univfd_indicator.score * 100).toFixed(2)}%`
                        : detailModal.univfd_indicator.status}{' '}
                      ({detailModal.univfd_indicator.prediction_label || '—'})
                    </div>
                    <div className="text-[11px] text-slate-400">
                      Comparison score only (uncalibrated)
                    </div>
                  </dd>
                </div>
              )}

              <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3">
                <dt className="text-slate-400">Face Manipulation Indicator</dt>
                <dd className="mt-1 font-mono text-slate-200 space-y-1 tabular-nums">
                  <div>Model: {detailModal.face_manipulation_indicator.model_name || detailModal.face_manipulation_indicator.model_id}</div>
                  <div>
                    Score:{' '}
                    {detailModal.face_manipulation_indicator.score !== null
                      ? `${(detailModal.face_manipulation_indicator.score * 100).toFixed(2)}%`
                      : detailModal.face_manipulation_indicator.status}
                  </div>
                  <div className="text-[11px] text-slate-400">
                    Faces Detected: {detailModal.face_manipulation_indicator.faces_detected ?? 0}
                  </div>
                </dd>
              </div>
            </dl>

            <div className="flex flex-wrap items-center justify-end gap-2 border-t border-slate-800 pt-3">
              <button
                type="button"
                onClick={() => setDetailModal(null)}
                className="rounded-lg border border-slate-700 bg-[#111827] px-3.5 py-1.5 text-xs font-medium text-slate-200 hover:text-white cursor-pointer"
              >
                Close
              </button>
              <button
                type="button"
                onClick={() => {
                  const id = detailModal.scan_id;
                  setDetailModal(null);
                  void onInspectScan(id);
                }}
                className="inline-flex items-center gap-1.5 rounded-lg bg-cyan-500 px-3.5 py-1.5 text-xs font-semibold text-slate-950 hover:bg-cyan-400 cursor-pointer"
              >
                <span>Open in Investigation Workspace</span>
                <ArrowUpRight className="h-3.5 w-3.5" aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
