import type {
  AnalyticsResponse,
  AnalyzeImageResponse,
  DatasetEvaluationDetail,
  DatasetEvaluationRequest,
  DatasetEvaluationSummary,
  DatasetInspectResponse,
  DeleteArtifactResponse,
  DeleteScanResponse,
  HealthResponse,
  ModelsStatusResponse,
  PurgeExpiredResponse,
  QueuedModalityResponse,
  ScanListResponse,
  TaskStatus,
} from '../types/forensics';

const STORAGE_KEY = 'mediaguard_api_base_url';

export const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/+$/, '') || 'http://127.0.0.1:8000';

export function getSavedApiBaseUrl(): string {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && saved.trim()) {
      return saved.trim().replace(/\/+$/, '');
    }
  } catch {
    // Ignore storage errors
  }
  return DEFAULT_API_BASE_URL;
}

export function saveApiBaseUrl(url: string): string {
  const normalized = url.trim().replace(/\/+$/, '') || DEFAULT_API_BASE_URL;
  try {
    localStorage.setItem(STORAGE_KEY, normalized);
  } catch {
    // Ignore storage errors
  }
  return normalized;
}

export class ApiClientError extends Error {
  statusCode: number | null;
  code: string;

  constructor(message: string, statusCode: number | null = null, code = 'ERR_API') {
    super(message);
    this.name = 'ApiClientError';
    this.statusCode = statusCode;
    this.code = code;
  }
}

async function parseErrorResponse(res: Response): Promise<ApiClientError> {
  try {
    const data = await res.json();
    const detail =
      typeof data?.detail === 'string'
        ? data.detail
        : data?.detail?.message || data?.message || `HTTP ${res.status}`;
    const code = data?.detail?.code || `HTTP_${res.status}`;
    return new ApiClientError(detail, res.status, code);
  } catch {
    return new ApiClientError(`Backend returned HTTP ${res.status}`, res.status, `HTTP_${res.status}`);
  }
}

async function requestJson<T>(baseUrl: string, path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${baseUrl}${path}`, init);
  } catch {
    throw new ApiClientError(
      `Unable to reach FastAPI backend at ${baseUrl}. Ensure the local Windows/Conda server is running.`,
      null,
      'ERR_BACKEND_UNREACHABLE'
    );
  }
  if (!res.ok) {
    throw await parseErrorResponse(res);
  }
  return (await res.json()) as T;
}

export async function fetchHealth(baseUrl: string): Promise<HealthResponse> {
  return requestJson<HealthResponse>(baseUrl, '/api/v1/health');
}

export async function fetchModelsStatus(baseUrl: string): Promise<ModelsStatusResponse> {
  return requestJson<ModelsStatusResponse>(baseUrl, '/api/v1/models/status');
}

export async function triggerModelsSetup(baseUrl: string): Promise<ModelsStatusResponse> {
  return requestJson<ModelsStatusResponse>(baseUrl, '/api/v1/models/setup', {
    method: 'POST',
  });
}

export async function analyzeImage(
  baseUrl: string,
  file: File,
  includeAttribution: boolean
): Promise<AnalyzeImageResponse> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('include_attribution', String(includeAttribution));

  return requestJson<AnalyzeImageResponse>(baseUrl, '/api/v1/analyze/image', {
    method: 'POST',
    body: formData,
  });
}

export async function fetchAnalytics(baseUrl: string): Promise<AnalyticsResponse> {
  return requestJson<AnalyticsResponse>(baseUrl, '/api/v1/analytics');
}

export async function fetchScans(
  baseUrl: string,
  params: {
    limit?: number;
    offset?: number;
    search?: string;
    status?: TaskStatus | 'all';
  }
): Promise<ScanListResponse> {
  const query = new URLSearchParams();
  if (params.limit !== undefined) query.set('limit', String(params.limit));
  if (params.offset !== undefined) query.set('offset', String(params.offset));
  if (params.search && params.search.trim()) query.set('search', params.search.trim());
  if (params.status && params.status !== 'all') query.set('status', params.status);

  const qs = query.toString();
  return requestJson<ScanListResponse>(baseUrl, `/api/v1/scans${qs ? `?${qs}` : ''}`);
}

export async function fetchScanById(baseUrl: string, scanId: string): Promise<AnalyzeImageResponse> {
  return requestJson<AnalyzeImageResponse>(baseUrl, `/api/v1/scans/${encodeURIComponent(scanId)}`);
}

export async function deleteScanRecord(baseUrl: string, scanId: string): Promise<DeleteScanResponse> {
  return requestJson<DeleteScanResponse>(baseUrl, `/api/v1/scans/${encodeURIComponent(scanId)}`, {
    method: 'DELETE',
  });
}

export async function deleteScanArtifact(
  baseUrl: string,
  scanId: string
): Promise<DeleteArtifactResponse> {
  return requestJson<DeleteArtifactResponse>(
    baseUrl,
    `/api/v1/scans/${encodeURIComponent(scanId)}/artifact`,
    {
      method: 'DELETE',
    }
  );
}

export async function purgeExpiredScans(baseUrl: string): Promise<PurgeExpiredResponse> {
  return requestJson<PurgeExpiredResponse>(baseUrl, '/api/v1/scans/purge-expired', {
    method: 'POST',
  });
}

export async function downloadScansExport(
  baseUrl: string,
  format: 'csv' | 'json',
  status?: TaskStatus | 'all'
): Promise<void> {
  const query = new URLSearchParams({ format });
  if (status && status !== 'all') {
    query.set('status', status);
  }
  let res: Response;
  try {
    res = await fetch(`${baseUrl}/api/v1/scans/export?${query.toString()}`);
  } catch {
    throw new ApiClientError(
      `Unable to reach FastAPI backend at ${baseUrl} for export.`,
      null,
      'ERR_BACKEND_UNREACHABLE'
    );
  }
  if (!res.ok) {
    throw await parseErrorResponse(res);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `mediaguard-scans.${format}`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function inspectEvaluationFolder(
  baseUrl: string,
  datasetDir?: string
): Promise<DatasetInspectResponse> {
  const query = new URLSearchParams();
  if (datasetDir && datasetDir.trim()) {
    query.set('dataset_dir', datasetDir.trim());
  }
  const qs = query.toString();
  return requestJson<DatasetInspectResponse>(
    baseUrl,
    `/api/v1/evaluation/inspect-folder${qs ? `?${qs}` : ''}`
  );
}

export async function runDatasetEvaluation(
  baseUrl: string,
  payload: DatasetEvaluationRequest
): Promise<DatasetEvaluationDetail> {
  return requestJson<DatasetEvaluationDetail>(baseUrl, '/api/v1/evaluation/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
}

export async function fetchEvaluationRuns(
  baseUrl: string,
  limit = 20
): Promise<DatasetEvaluationSummary[]> {
  return requestJson<DatasetEvaluationSummary[]>(
    baseUrl,
    `/api/v1/evaluation/runs?limit=${limit}`
  );
}

export async function fetchEvaluationRun(
  baseUrl: string,
  evaluationId: string
): Promise<DatasetEvaluationDetail> {
  return requestJson<DatasetEvaluationDetail>(
    baseUrl,
    `/api/v1/evaluation/runs/${encodeURIComponent(evaluationId)}`
  );
}

export async function downloadEvaluationExport(
  baseUrl: string,
  evaluationId: string,
  format: 'csv' | 'json'
): Promise<void> {
  let res: Response;
  try {
    res = await fetch(
      `${baseUrl}/api/v1/evaluation/runs/${encodeURIComponent(evaluationId)}/export?format=${format}`
    );
  } catch {
    throw new ApiClientError(
      `Unable to reach FastAPI backend at ${baseUrl} for evaluation export.`,
      null,
      'ERR_BACKEND_UNREACHABLE'
    );
  }
  if (!res.ok) {
    throw await parseErrorResponse(res);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `mediaguard-eval-${evaluationId}.${format}`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function probeQueuedModality(
  baseUrl: string,
  modality: 'video' | 'audio'
): Promise<QueuedModalityResponse> {
  let res: Response;
  try {
    res = await fetch(`${baseUrl}/api/v1/analyze/${modality}`, { method: 'POST' });
  } catch {
    throw new ApiClientError(
      `Backend offline at ${baseUrl}. ${modality} analysis remains queued for a future milestone.`,
      null,
      'ERR_BACKEND_UNREACHABLE'
    );
  }
  const data = await res.json();
  if (res.status === 501 && data?.detail) {
    return data.detail as QueuedModalityResponse;
  }
  if (!res.ok) {
    throw await parseErrorResponse(res);
  }
  return data as QueuedModalityResponse;
}
