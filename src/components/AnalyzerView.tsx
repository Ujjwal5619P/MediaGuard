import { useRef, useState, type ChangeEvent, type DragEvent } from 'react';
import {
  AlertCircle,
  Check,
  Copy,
  Download,
  FileImage,
  Info,
  Printer,
  ShieldCheck,
  Trash2,
  Upload,
} from 'lucide-react';
import { analyzeImage, ApiClientError, probeQueuedModality } from '../services/api';
import type {
  AnalyzeImageResponse,
  AppView,
  ClientFileInspection,
  HealthResponse,
  ModalityTab,
  QueuedModalityResponse,
} from '../types/forensics';
import { formatBytes, inspectFileClientSide } from '../utils/fileInspector';
import { ForensicCanvas } from './ForensicCanvas';
import { IndicatorCard } from './IndicatorCard';
import { ModalityBar } from './ModalityBar';

const MAX_CLIENT_BYTES = 15 * 1024 * 1024; // 15 MB

interface AnalyzerViewProps {
  apiBaseUrl: string;
  health: HealthResponse | null;
  healthError: string | null;
  activeModality: ModalityTab;
  onSelectModality: (modality: ModalityTab) => void;
  activeFile: File | null;
  clientInspection: ClientFileInspection | null;
  analysisResult: AnalyzeImageResponse | null;
  onFileInspected: (file: File | null, inspection: ClientFileInspection | null) => void;
  onAnalysisComplete: (result: AnalyzeImageResponse | null) => void;
  onNavigate: (view: AppView) => void;
}

export function AnalyzerView({
  apiBaseUrl,
  health,
  healthError,
  activeModality,
  onSelectModality,
  activeFile,
  clientInspection,
  analysisResult,
  onFileInspected,
  onAnalysisComplete,
  onNavigate,
}: AnalyzerViewProps) {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [includeAttribution, setIncludeAttribution] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [copiedHash, setCopiedHash] = useState(false);
  const [queuedProbeResult, setQueuedProbeResult] = useState<QueuedModalityResponse | null>(null);
  const [queuedProbeError, setQueuedProbeError] = useState<string | null>(null);

  const handleSelectedFile = async (file: File) => {
    setFileError(null);
    setAnalysisError(null);
    onAnalysisComplete(null);

    if (file.size > MAX_CLIENT_BYTES) {
      setFileError(`File exceeds 15 MB limit (${formatBytes(file.size)}).`);
      return;
    }

    try {
      const inspection = await inspectFileClientSide(file);
      if (!inspection.signatureValid) {
        URL.revokeObjectURL(inspection.previewUrl);
        setFileError(
          'Unsupported or malformed image signature. Only valid JPEG, PNG, and WebP files are accepted.'
        );
        return;
      }
      if (clientInspection?.previewUrl) {
        URL.revokeObjectURL(clientInspection.previewUrl);
      }
      onFileInspected(file, inspection);
    } catch (err) {
      setFileError(
        err instanceof Error ? err.message : 'Unable to inspect selected image file.'
      );
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      void handleSelectedFile(file);
    }
    e.target.value = '';
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void handleSelectedFile(file);
    }
  };

  const handleClear = () => {
    if (clientInspection?.previewUrl) {
      URL.revokeObjectURL(clientInspection.previewUrl);
    }
    setFileError(null);
    setAnalysisError(null);
    onFileInspected(null, null);
    onAnalysisComplete(null);
  };

  const handleRunAnalysis = async () => {
    if (!activeFile) return;
    setIsAnalyzing(true);
    setAnalysisError(null);

    try {
      const response = await analyzeImage(apiBaseUrl, activeFile, includeAttribution);
      onAnalysisComplete(response);
    } catch (err) {
      if (err instanceof ApiClientError) {
        setAnalysisError(`${err.code}: ${err.message}`);
      } else {
        setAnalysisError('Unexpected error communicating with the FastAPI backend.');
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleCopyHash = async (sha: string) => {
    try {
      await navigator.clipboard.writeText(sha);
      setCopiedHash(true);
      setTimeout(() => setCopiedHash(false), 1800);
    } catch {
      // Ignore clipboard failures
    }
  };

  const handleExportJsonReport = () => {
    if (!analysisResult && !clientInspection) return;
    const payload = analysisResult ?? {
      note: 'Client-side deterministic inspection only (backend analysis not executed)',
      deterministic_inspection: {
        filename: clientInspection?.filename,
        detected_mime: clientInspection?.detectedSignatureMime,
        byte_size: clientInspection?.byteSize,
        width: clientInspection?.width,
        height: clientInspection?.height,
        sha256: clientInspection?.sha256,
        exif: clientInspection?.exif,
      },
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `mediaguard-report-${analysisResult?.scan_id || 'local'}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  const handleProbeQueued = async (modality: 'video' | 'audio') => {
    setQueuedProbeError(null);
    setQueuedProbeResult(null);
    try {
      const resp = await probeQueuedModality(apiBaseUrl, modality);
      setQueuedProbeResult(resp);
    } catch (err) {
      setQueuedProbeError(
        err instanceof Error ? err.message : 'Unable to reach backend capability endpoint.'
      );
    }
  };

  const displayFilename = analysisResult?.file_metadata.filename ?? clientInspection?.filename ?? null;
  const displayWidth = analysisResult?.file_metadata.width ?? clientInspection?.width ?? 0;
  const displayHeight = analysisResult?.file_metadata.height ?? clientInspection?.height ?? 0;
  const displayMime =
    analysisResult?.file_metadata.detected_mime ?? clientInspection?.detectedSignatureMime ?? null;
  const displayBytes = analysisResult?.file_metadata.byte_size ?? clientInspection?.byteSize ?? null;
  const displaySha = analysisResult?.sha256 ?? clientInspection?.sha256 ?? null;
  const displayExif = analysisResult?.file_metadata.exif_summary ?? clientInspection?.exif ?? null;

  return (
    <>
      <ModalityBar activeModality={activeModality} onSelectModality={onSelectModality} />

      {activeModality !== 'image' ? (
        <div className="mx-auto max-w-[1440px] px-6 py-10">
          <div className="max-w-2xl rounded-xl border border-slate-800 bg-[#0B0E17] p-8">
            <div className="text-xs font-medium text-amber-400">
              Queued Modality · Post-v1 Milestone
            </div>
            <h2 className="mt-2 text-2xl font-semibold text-slate-100">
              {activeModality === 'video' ? 'Video Forensics Pipeline' : 'Audio Forensics Pipeline'}
            </h2>
            <p className="mt-3 text-sm leading-relaxed text-slate-300">
              Image investigation is the active production pipeline in v1. Video temporal consistency and synthetic voice detection remain queued for post-v1 releases.
            </p>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <button
                type="button"
                onClick={() => void handleProbeQueued(activeModality)}
                className="rounded-lg border border-slate-700 bg-[#111827] px-4 py-2 text-xs font-medium text-slate-200 hover:border-slate-600 hover:text-white transition-colors cursor-pointer"
              >
                Query Backend Capability (HTTP 501)
              </button>
            </div>
            {queuedProbeResult && (
              <div className="mt-4 rounded-lg border border-slate-800 bg-[#111827] p-4 text-xs text-slate-300 font-mono">
                {queuedProbeResult.modality.toUpperCase()} · {queuedProbeResult.status} · {queuedProbeResult.milestone} — {queuedProbeResult.message}
              </div>
            )}
            {queuedProbeError && (
              <div className="mt-4 rounded-lg border border-amber-900/60 bg-amber-950/20 p-4 text-xs text-amber-200">
                {queuedProbeError}
              </div>
            )}
          </div>
        </div>
      ) : (
        <div className="mx-auto max-w-[1440px] px-6 py-6 space-y-6">
          {/* Primary Forensic Stage: Large Central Image Viewport + Right Evidence Inspector */}
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1fr_360px]">
            <div className="flex flex-col gap-4">
              {analysisResult && !clientInspection && (
                <div className="rounded-xl border border-slate-800 bg-[#0B0E17] px-4 py-3 text-xs text-slate-300">
                  Viewing archived SQLite investigation <span className="font-mono text-slate-100">{analysisResult.scan_id}</span>. Original uploaded image was deleted after initial analysis ({analysisResult.original_deleted ? 'verified deleted' : 'retained'}).
                </div>
              )}

              <ForensicCanvas
                imageUrl={clientInspection?.previewUrl ?? null}
                filename={displayFilename}
                width={displayWidth}
                height={displayHeight}
                attribution={analysisResult?.experimental_attribution ?? null}
                faceBoxes={analysisResult?.face_manipulation_indicator.face_boxes ?? null}
                apiBaseUrl={apiBaseUrl}
                onDropFile={(file) => void handleSelectedFile(file)}
                onRequestFileSelect={() => fileInputRef.current?.click()}
              />
            </div>

            {/* Right Evidence & Controls Column */}
            <aside className="no-print flex flex-col gap-5">
              {/* Upload & Run Controls */}
              <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5 space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-sm font-semibold text-slate-100">Evidence Upload &amp; Run</h2>
                  {(clientInspection || analysisResult) && (
                    <button
                      type="button"
                      onClick={handleClear}
                      className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-200 transition-colors cursor-pointer"
                    >
                      <Trash2 className="h-3.5 w-3.5" aria-hidden="true" />
                      <span>Clear</span>
                    </button>
                  )}
                </div>

                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setIsDragging(true);
                  }}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={handleDrop}
                  className={`flex flex-col items-center justify-center rounded-lg border border-dashed px-4 py-5 text-center transition-colors ${
                    isDragging
                      ? 'border-cyan-400 bg-cyan-950/20'
                      : 'border-slate-700 bg-[#07090E] hover:border-slate-600'
                  }`}
                >
                  <Upload className="h-5 w-5 text-slate-400" aria-hidden="true" />
                  <p className="mt-2 text-xs font-medium text-slate-200">
                    Drop PNG, JPEG, or WebP
                  </p>
                  <p className="mt-0.5 text-xs text-slate-500">Max 15 MB · 8192px · 36 MP</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/png,image/jpeg,image/webp"
                    onChange={handleFileInputChange}
                    className="sr-only"
                    aria-label="Choose image file to inspect"
                  />
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="mt-3 rounded-lg border border-slate-700 bg-[#111827] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-600 hover:text-white transition-colors cursor-pointer"
                  >
                    Select Image File
                  </button>
                </div>

                {fileError && (
                  <div
                    role="alert"
                    className="flex items-start gap-2 rounded-lg border border-rose-900/60 bg-rose-950/30 p-3 text-xs text-rose-200"
                  >
                    <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-400" aria-hidden="true" />
                    <span>{fileError}</span>
                  </div>
                )}

                <label className="flex items-start gap-2.5 text-xs text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={includeAttribution}
                    onChange={(e) => setIncludeAttribution(e.target.checked)}
                    className="mt-0.5 accent-cyan-500"
                  />
                  <span>
                    Generate attribution overlays (Face Grad-CAM &amp; DINOv2 patch attention)
                  </span>
                </label>

                <button
                  type="button"
                  disabled={!activeFile || isAnalyzing}
                  onClick={() => void handleRunAnalysis()}
                  className="w-full rounded-lg bg-cyan-500 px-4 py-2.5 text-xs font-semibold text-slate-950 transition-colors hover:bg-cyan-400 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500 cursor-pointer"
                >
                  {isAnalyzing ? 'Running Pretrained Models...' : 'Run Forensic Analysis'}
                </button>

                {analysisError && (
                  <div
                    role="alert"
                    className="rounded-lg border border-rose-900/60 bg-rose-950/30 p-3 text-xs leading-relaxed text-rose-200"
                  >
                    {analysisError}
                  </div>
                )}

                <div className="flex items-start gap-2 border-t border-slate-800 pt-3 text-xs leading-relaxed text-slate-400">
                  <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" aria-hidden="true" />
                  <span>
                    Originals are automatically deleted immediately after inference. Never used for training.
                  </span>
                </div>
              </div>

              {/* File Metadata, SHA-256 & EXIF */}
              <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
                <div className="flex items-center justify-between gap-2">
                  <h2 className="text-sm font-semibold text-slate-100">
                    File Integrity &amp; EXIF Summary
                  </h2>
                  <FileImage className="h-4 w-4 text-slate-400" aria-hidden="true" />
                </div>

                {displayFilename ? (
                  <dl className="mt-3 space-y-2.5 text-xs">
                    <div className="flex items-center justify-between gap-3">
                      <dt className="text-slate-400">Filename</dt>
                      <dd className="truncate font-mono text-slate-200 max-w-[190px]" title={displayFilename}>
                        {displayFilename}
                      </dd>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <dt className="text-slate-400">Verified MIME</dt>
                      <dd className="font-mono text-slate-200">{displayMime || '—'}</dd>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <dt className="text-slate-400">Dimensions</dt>
                      <dd className="font-mono tabular-nums text-slate-200">
                        {displayWidth > 0 ? `${displayWidth} × ${displayHeight} px` : '—'}
                      </dd>
                    </div>
                    <div className="flex items-center justify-between gap-3">
                      <dt className="text-slate-400">Byte Size</dt>
                      <dd className="font-mono tabular-nums text-slate-200">
                        {displayBytes !== null ? formatBytes(displayBytes) : '—'}
                      </dd>
                    </div>
                    <div className="border-t border-slate-800 pt-2.5">
                      <div className="flex items-center justify-between gap-2">
                        <dt className="text-slate-400">SHA-256 Digest</dt>
                        {displaySha && (
                          <button
                            type="button"
                            onClick={() => void handleCopyHash(displaySha)}
                            className="flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 cursor-pointer"
                          >
                            {copiedHash ? (
                              <>
                                <Check className="h-3 w-3" aria-hidden="true" />
                                <span>Copied</span>
                              </>
                            ) : (
                              <>
                                <Copy className="h-3 w-3" aria-hidden="true" />
                                <span>Copy</span>
                              </>
                            )}
                          </button>
                        )}
                      </div>
                      <dd className="mt-1 break-all font-mono text-[11px] text-slate-300">
                        {displaySha || '—'}
                      </dd>
                    </div>

                    <div className="border-t border-slate-800 pt-2.5 space-y-1">
                      <div className="text-slate-400">EXIF Metadata</div>
                      {displayExif?.has_exif ? (
                        <div className="space-y-1 font-mono text-[11px] text-slate-300">
                          <div>Camera: {[displayExif.camera_make, displayExif.camera_model].filter(Boolean).join(' ') || 'Not specified'}</div>
                          <div>Software: {displayExif.software || 'Not specified'}</div>
                          <div>Captured: {displayExif.datetime_original || 'Not specified'}</div>
                        </div>
                      ) : (
                        <div className="text-slate-500">No EXIF header metadata detected.</div>
                      )}
                    </div>
                  </dl>
                ) : (
                  <p className="mt-2.5 text-xs leading-relaxed text-slate-400">
                    Load an image to inspect cryptographic SHA-256 hash, magic-byte MIME signature, pixel dimensions, and EXIF headers.
                  </p>
                )}

                <div className="mt-4 flex items-center justify-between border-t border-slate-800 pt-3 text-xs">
                  <span
                    className={
                      health?.status === 'ok'
                        ? 'text-emerald-400 font-medium'
                        : health
                        ? 'text-amber-400 font-medium'
                        : 'text-rose-400 font-medium'
                    }
                  >
                    {health?.status === 'ok'
                      ? `Models Ready (${health.execution_mode})`
                      : health
                      ? 'Backend Degraded'
                      : 'Backend Offline'}
                  </span>
                  <button
                    type="button"
                    onClick={() => onNavigate('models')}
                    className="text-cyan-400 hover:text-cyan-300 font-medium cursor-pointer"
                  >
                    AI Models
                  </button>
                </div>
              </div>
            </aside>
          </div>

          {/* Task-Separated Detection Results */}
          <section aria-label="Task-Separated Detection Results" className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="text-base font-semibold text-slate-100">
                  Independent Pretrained Detection Results
                </h2>
                <p className="text-xs text-slate-400">
                  Primary AI-generated detection uses Imager-1-Tiny (ResNet-50, 224×224, CPU); UnivFD CLIP ViT-L/14 reports a separate secondary comparison score; face manipulation runs independently on detected faces.
                </p>
              </div>

              <div className="no-print flex items-center gap-2">
                <button
                  type="button"
                  disabled={!analysisResult && !clientInspection}
                  onClick={handleExportJsonReport}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white disabled:cursor-not-allowed disabled:opacity-40 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Download className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>Export JSON Report</span>
                </button>
                <button
                  type="button"
                  disabled={!analysisResult && !clientInspection}
                  onClick={() => window.print()}
                  className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#0B0E17] px-3 py-1.5 text-xs font-medium text-slate-200 hover:border-slate-700 hover:text-white disabled:cursor-not-allowed disabled:opacity-40 transition-colors whitespace-nowrap shrink-0 cursor-pointer"
                >
                  <Printer className="h-3.5 w-3.5" aria-hidden="true" />
                  <span>Print Summary</span>
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 items-stretch gap-4 md:grid-cols-2">
              <IndicatorCard
                roleLabel="Primary AI-Generated Detector"
                title="Imager-1-Tiny"
                scopeDescription="Evaluates whole-image synthetic origin via Imagers/Imager-1-tiny (ResNet-50 with ImageNet normalization and a 0.385 decision threshold)."
                indicator={analysisResult?.ai_generated_indicator ?? null}
                offlineMessage={
                  !analysisResult && healthError && !health
                    ? 'Backend disconnected. Start FastAPI on /api/v1/health to run CPU image inference.'
                    : null
                }
              />

              <IndicatorCard
                roleLabel="Task-Separated Facial Forensics"
                title="Face Manipulation Detection"
                scopeDescription="Detects frontal faces via OpenCV and evaluates face swaps/alterations via EfficientNet-B0 FaceForensics++ C23 + Grad-CAM."
                indicator={analysisResult?.face_manipulation_indicator ?? null}
                offlineMessage={
                  !analysisResult && healthError && !health
                    ? 'Backend disconnected. Start FastAPI on /api/v1/health to execute face manipulation detection.'
                    : null
                }
              />
            </div>

            {/* Secondary Comparison Detector + DINOv2 Feature Summary */}
            <div className="grid grid-cols-1 items-stretch gap-4 md:grid-cols-2">
              <IndicatorCard
                roleLabel="Secondary Comparison Detector (Non-Primary Reference)"
                title="UniversalFakeDetect (UnivFD · CLIP ViT-L/14)"
                scopeDescription="Independent secondary comparison score from openai/clip-vit-large-patch14 + fc_weights.pth linear head. Never fused or blended with the primary detector."
                indicator={analysisResult?.univfd_indicator ?? null}
                offlineMessage={
                  !analysisResult && healthError && !health
                    ? 'Backend disconnected. Start FastAPI to compare secondary UnivFD score.'
                    : null
                }
              />

              <div className="flex flex-col justify-between rounded-xl border border-slate-800 bg-[#0B0E17] p-6 text-xs">
                <div className="space-y-3">
                  <div className="font-mono text-[11px] uppercase tracking-wider text-slate-400">
                    Self-Supervised Feature &amp; Runtime Telemetry
                  </div>
                  <h3 className="text-base font-semibold text-slate-100">
                    DINOv2 ViT-S/14 &amp; Pipeline Execution
                  </h3>
                  {analysisResult ? (
                    <div className="space-y-3 border-t border-slate-800 pt-4">
                      <div className="font-mono text-slate-300 tabular-nums leading-relaxed">
                        {analysisResult.dinov2_features.status === 'evaluated'
                          ? `Status: Evaluated · ${analysisResult.dinov2_features.embedding_dim}-d CLS embedding · L2 norm ${analysisResult.dinov2_features.l2_norm} · ${analysisResult.dinov2_features.processing_ms} ms (non-diagnostic)`
                          : analysisResult.dinov2_features.reason}
                      </div>
                      {analysisResult.dinov2_features.embedding_preview && (
                        <div className="rounded-lg border border-slate-800 bg-[#07090E] p-3 font-mono text-[11px] text-slate-400 tabular-nums">
                          CLS Preview [0..7]: [{analysisResult.dinov2_features.embedding_preview.join(', ')}]
                        </div>
                      )}
                    </div>
                  ) : (
                    <p className="border-t border-slate-800 pt-4 text-slate-400 leading-relaxed">
                      Extracts 384-dimensional DINOv2 ViT-S/14 visual representations and experimental CLS patch-attention overlays alongside measured CUDA/CPU execution telemetry.
                    </p>
                  )}
                </div>

                {analysisResult && (
                  <dl className="mt-5 grid grid-cols-2 gap-3 border-t border-slate-800/80 pt-4 font-mono tabular-nums">
                    <div>
                      <dt className="text-slate-500">Execution Device</dt>
                      <dd className="mt-1 text-slate-200">{analysisResult.execution_device}</dd>
                    </div>
                    <div className="text-right">
                      <dt className="text-slate-500">Detector Pipeline Latency / Peak VRAM</dt>
                      <dd className="mt-1 text-slate-200">
                        {analysisResult.inference_ms} ms
                        {analysisResult.vram_peak_mb !== null
                          ? ` · ${analysisResult.vram_peak_mb} MB`
                          : ''}
                      </dd>
                    </div>
                  </dl>
                )}
              </div>
            </div>

            <div className="flex items-start gap-2.5 rounded-xl border border-slate-800 bg-[#0B0E17] px-4 py-3 text-xs text-slate-400">
              <Info className="mt-0.5 h-4 w-4 shrink-0 text-cyan-400" aria-hidden="true" />
              <p className="leading-relaxed">
                <span className="font-medium text-slate-200">Methodological Notice:</span> MediaGuard reports uncalibrated raw model probabilities from Imager-1-Tiny (Primary), UnivFD (Secondary Comparison), and EfficientNet-B0 FF++ (Face Manipulation), plus experimental saliency overlays. Outputs are investigative indicators, never definitive proof of synthetic origin or manipulation.
              </p>
            </div>
          </section>
        </div>
      )}
    </>
  );
}
