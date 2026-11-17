export type AppView =
  | 'dashboard'
  | 'investigation'
  | 'history'
  | 'models'
  | 'analytics'
  | 'methodology'
  | 'system';

export type ModalityTab = 'image' | 'video' | 'audio';

export type TaskStatus =
  | 'unavailable'
  | 'loading'
  | 'no_face_detected'
  | 'inconclusive'
  | 'evaluated'
  | 'error';

export type ModelComponentStatus =
  | 'ready'
  | 'loading'
  | 'downloading'
  | 'unavailable'
  | 'not_trained'
  | 'error';

export type DownloadLifecycleStatus =
  | 'not_downloaded'
  | 'downloading'
  | 'verified'
  | 'loaded'
  | 'error';

export interface MeasuredVram {
  allocated_mb: number | null;
  peak_mb: number | null;
  free_mb: number | null;
  total_mb: number | null;
}

export interface ModelComponentReadiness {
  status: ModelComponentStatus;
  model_id: string;
  role: string;
  checkpoint_verified: boolean;
  license: string;
  source: string;
  reason: string;
}

export interface ModelManifestEntry {
  key: string;
  name: string;
  repo_id: string;
  revision: string;
  source_url: string;
  license: string;
  architecture: string;
  training_dataset: string;
  intended_task: string;
  preprocessing: string;
  limitations: string;
  status: DownloadLifecycleStatus;
  verified: boolean;
  loaded_in_memory: boolean;
  size_mb: number;
  expected_sha256: string;
  actual_sha256: string | null;
  local_dir: string;
  active_device: string;
  last_inference_ms: number | null;
  last_vram_peak_mb: number | null;
  error_message: string | null;
  updated_at: string;
}

export interface ModelsStatusResponse {
  models_dir: string;
  manifest_path: string;
  all_verified: boolean;
  all_loaded: boolean;
  download_in_progress: boolean;
  models: ModelManifestEntry[];
}

export interface RetentionPolicySummary {
  delete_original_after_analysis: boolean;
  never_train_without_consent: boolean;
  record_retention_days: number;
  artifact_retention_days: number;
  max_upload_bytes: number;
  max_image_pixels: number;
  max_dimension: number;
}

export interface HealthResponse {
  status: 'ok' | 'degraded';
  service: string;
  milestone: string;
  version?: string;
  timestamp_utc: string;
  cuda_available: boolean;
  device_name: string;
  execution_mode: string;
  vram_measured: MeasuredVram | null;
  sequential_gpu_lock: boolean;
  max_concurrent_gpu_workers: number;
  low_vram_offload: boolean;
  models: {
    dinov2_extractor: ModelComponentReadiness;
    ai_generated_detector: ModelComponentReadiness;
    univfd_detector?: ModelComponentReadiness | null;
    face_manipulation_detector: ModelComponentReadiness;
    calibration_profile: ModelComponentReadiness;
  };
  retention_policy: RetentionPolicySummary;
}

export interface ExifSummary {
  camera_make: string | null;
  camera_model: string | null;
  software: string | null;
  datetime_original: string | null;
  has_exif: boolean;
}

export interface FileMetadata {
  filename: string;
  detected_mime: string;
  declared_mime: string | null;
  byte_size: number;
  width: number;
  height: number;
  exif_summary: ExifSummary;
}

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface Dinov2FeatureSummary {
  status: 'evaluated' | 'unavailable' | 'loading' | 'error';
  model_id: string;
  embedding_dim: number | null;
  l2_norm: number | null;
  embedding_preview: number[] | null;
  processing_ms: number | null;
  execution_device: string | null;
  reason: string;
}

export interface TaskIndicator {
  task: 'ai_generated_image_detection' | 'face_manipulation_detection';
  status: TaskStatus;
  score: number | null;
  decision_threshold: number;
  prediction_label: string | null;
  calibrated: boolean;
  model_id: string;
  model_name?: string;
  architecture?: string;
  model_source?: string;
  model_version: string;
  execution_device?: string | null;
  processing_ms: number | null;
  faces_detected: number | null;
  face_boxes: FaceBox[] | null;
  limitations: string;
  reason: string;
}

export interface ExperimentalAttribution {
  available: boolean;
  gradcam_available: boolean;
  gradcam_url: string | null;
  patch_attention_available: boolean;
  patch_attention_url: string | null;
  method: string | null;
  artifact_url: string | null;
  disclaimer: string;
  reason: string;
}

export interface AnalyzeImageResponse {
  scan_id: string;
  created_at: string;
  sha256: string;
  original_deleted: boolean;
  inference_ms: number;
  execution_device: string;
  vram_peak_mb: number | null;
  file_metadata: FileMetadata;
  dinov2_features: Dinov2FeatureSummary;
  ai_generated_indicator: TaskIndicator;
  univfd_indicator?: TaskIndicator | null;
  face_manipulation_indicator: TaskIndicator;
  experimental_attribution: ExperimentalAttribution;
  record_expires_at: string;
  artifact_expires_at: string | null;
}

export interface ScanSummary {
  scan_id: string;
  created_at: string;
  sha256: string;
  filename: string;
  detected_mime: string;
  byte_size: number;
  width: number;
  height: number;
  ai_generated_status: TaskStatus;
  ai_generated_score: number | null;
  ai_generated_label: string | null;
  ai_model_id?: string | null;
  univfd_status?: TaskStatus | null;
  univfd_score?: number | null;
  univfd_label?: string | null;
  univfd_model_id?: string | null;
  face_manipulation_status: TaskStatus;
  face_manipulation_score: number | null;
  face_manipulation_label: string | null;
  face_model_id?: string | null;
  execution_device?: string | null;
  inference_ms?: number | null;
  has_artifact: boolean;
  original_deleted: boolean;
  record_expires_at: string;
  artifact_expires_at: string | null;
}

export interface ScanListResponse {
  items: ScanSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface DailyActivityBucket {
  date: string;
  total: number;
  ai_flagged: number;
  face_flagged: number;
}

export interface AnalyticsResponse {
  total_scans: number;
  ai_evaluated_count: number;
  ai_flagged_count: number;
  ai_natural_count: number;
  univfd_evaluated_count?: number;
  univfd_flagged_count?: number;
  face_evaluated_count: number;
  face_flagged_count: number;
  face_no_face_count: number;
  artifacts_retained_count: number;
  originals_deleted_count: number;
  avg_inference_ms: number | null;
  min_inference_ms: number | null;
  max_inference_ms: number | null;
  avg_vram_peak_mb: number | null;
  max_vram_peak_mb: number | null;
  ai_score_buckets: number[];
  daily_activity: DailyActivityBucket[];
  device_breakdown: Record<string, number>;
  mime_breakdown: Record<string, number>;
}

export type GroundTruthLabel = 'AI_GENERATED' | 'REAL';
export type DatasetSplitRole = 'validation' | 'evaluation' | 'final_test';

export interface ConfusionMatrixCounts {
  tp: number;
  fp: number;
  tn: number;
  fn: number;
}

export interface DetectorEvaluationMetrics {
  model_id: string;
  model_name: string;
  evaluated_count: number;
  error_count: number;
  accuracy: number | null;
  precision: number | null;
  recall: number | null;
  f1_score: number | null;
  false_positive_rate: number | null;
  false_negative_rate: number | null;
  confusion_matrix: ConfusionMatrixCounts;
  mean_inference_ms: number | null;
  total_inference_ms: number;
  peak_vram_mb: number | null;
}

export interface EvaluationImageResult {
  filename: string;
  relative_path: string;
  ground_truth: GroundTruthLabel;
  smaug_status: TaskStatus;
  smaug_score: number | null;
  smaug_prediction: GroundTruthLabel | null;
  smaug_correct: boolean | null;
  smaug_inference_ms: number | null;
  univfd_status: TaskStatus | null;
  univfd_score: number | null;
  univfd_prediction: GroundTruthLabel | null;
  univfd_correct: boolean | null;
  univfd_inference_ms: number | null;
  execution_device: string;
  error_message: string | null;
}

export interface DatasetInspectResponse {
  dataset_dir: string;
  ai_generated_dir: string;
  real_dir: string;
  ai_generated_exists: boolean;
  real_exists: boolean;
  ai_generated_count: number;
  real_count: number;
  total_supported_count: number;
  unsupported_or_skipped: string[];
}

export interface DatasetEvaluationRequest {
  dataset_dir?: string | null;
  split_role: DatasetSplitRole;
  decision_threshold: number;
  include_univfd: boolean;
  max_images?: number | null;
}

export interface DatasetEvaluationSummary {
  evaluation_id: string;
  created_at: string;
  dataset_dir: string;
  split_role: DatasetSplitRole;
  decision_threshold: number;
  threshold_auto_tuned: boolean;
  threshold_policy_note: string;
  models_trained_or_finetuned: boolean;
  include_univfd: boolean;
  total_images: number;
  ai_generated_count: number;
  real_count: number;
  skipped_count: number;
  execution_device: string;
  total_duration_ms: number;
  smaug_metrics: DetectorEvaluationMetrics;
  univfd_metrics: DetectorEvaluationMetrics | null;
}

export interface DatasetEvaluationDetail extends DatasetEvaluationSummary {
  items: EvaluationImageResult[];
}

export interface DeleteScanResponse {
  deleted: boolean;
  scan_id: string;
  artifact_deleted: boolean;
}

export interface DeleteArtifactResponse {
  artifact_deleted: boolean;
  scan_id: string;
  record_preserved: boolean;
}

export interface PurgeExpiredResponse {
  expired_records_deleted: number;
  expired_artifacts_purged: number;
}

export interface QueuedModalityResponse {
  modality: 'video' | 'audio';
  status: 'queued';
  milestone: string;
  message: string;
}

export interface ClientFileInspection {
  filename: string;
  declaredMime: string;
  detectedSignatureMime: string | null;
  signatureValid: boolean;
  byteSize: number;
  width: number;
  height: number;
  sha256: string;
  exif: ExifSummary;
  previewUrl: string;
}
