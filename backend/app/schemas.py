from typing import List, Literal, Optional

from pydantic import BaseModel, Field

TaskStatus = Literal[
    "unavailable",
    "loading",
    "no_face_detected",
    "inconclusive",
    "evaluated",
    "error",
]
ModelComponentStatus = Literal[
    "ready",
    "loading",
    "downloading",
    "unavailable",
    "not_trained",
    "error",
]
DownloadLifecycleStatus = Literal[
    "not_downloaded",
    "downloading",
    "verified",
    "loaded",
    "error",
]
GroundTruthLabel = Literal["AI_GENERATED", "REAL"]
DatasetSplitRole = Literal["validation", "evaluation", "final_test"]


class MeasuredVram(BaseModel):
    allocated_mb: Optional[float] = None
    peak_mb: Optional[float] = None
    free_mb: Optional[float] = None
    total_mb: Optional[float] = None


class ModelComponentReadiness(BaseModel):
    status: ModelComponentStatus
    model_id: str
    role: str
    checkpoint_verified: bool
    license: str
    source: str
    reason: str


class ModelReadinessMap(BaseModel):
    dinov2_extractor: ModelComponentReadiness
    ai_generated_detector: ModelComponentReadiness
    univfd_detector: Optional[ModelComponentReadiness] = None
    face_manipulation_detector: ModelComponentReadiness
    calibration_profile: ModelComponentReadiness


class ModelManifestEntry(BaseModel):
    key: str
    name: str
    repo_id: str
    revision: str
    source_url: str
    license: str
    architecture: str
    training_dataset: str
    intended_task: str
    preprocessing: str
    limitations: str
    status: DownloadLifecycleStatus
    verified: bool
    loaded_in_memory: bool
    size_mb: float
    expected_sha256: str
    actual_sha256: Optional[str] = None
    local_dir: str
    active_device: str = "cpu"
    last_inference_ms: Optional[int] = None
    last_vram_peak_mb: Optional[float] = None
    error_message: Optional[str] = None
    updated_at: str


class ModelsStatusResponse(BaseModel):
    models_dir: str
    manifest_path: str
    all_verified: bool
    all_loaded: bool
    download_in_progress: bool
    models: List[ModelManifestEntry]


class RetentionPolicySummary(BaseModel):
    delete_original_after_analysis: bool
    never_train_without_consent: bool
    record_retention_days: int
    artifact_retention_days: int
    max_upload_bytes: int
    max_image_pixels: int
    max_dimension: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    milestone: str
    version: str = "1.3.0"
    timestamp_utc: str
    cuda_available: bool
    device_name: str
    execution_mode: str
    vram_measured: Optional[MeasuredVram] = None
    sequential_gpu_lock: bool
    max_concurrent_gpu_workers: int = 1
    low_vram_offload: bool = True
    models: ModelReadinessMap
    retention_policy: RetentionPolicySummary


class ExifSummary(BaseModel):
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[str] = None
    has_exif: bool = False


class FileMetadata(BaseModel):
    filename: str
    detected_mime: str
    declared_mime: Optional[str] = None
    byte_size: int
    width: int
    height: int
    exif_summary: ExifSummary


class FaceBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class Dinov2FeatureSummary(BaseModel):
    status: Literal["evaluated", "unavailable", "loading", "error"]
    model_id: str
    embedding_dim: Optional[int] = None
    l2_norm: Optional[float] = None
    embedding_preview: Optional[List[float]] = None
    processing_ms: Optional[int] = None
    execution_device: Optional[str] = None
    reason: str


class TaskIndicator(BaseModel):
    task: Literal["ai_generated_image_detection", "face_manipulation_detection"]
    status: TaskStatus
    score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    decision_threshold: float = 0.385
    prediction_label: Optional[str] = None
    calibrated: bool = False
    model_id: str
    model_name: str = ""
    architecture: str = ""
    model_source: str = ""
    model_version: str
    execution_device: Optional[str] = None
    processing_ms: Optional[int] = None
    faces_detected: Optional[int] = None
    face_boxes: Optional[List[FaceBox]] = None
    limitations: str = ""
    reason: str


class ExperimentalAttribution(BaseModel):
    available: bool = False
    gradcam_available: bool = False
    gradcam_url: Optional[str] = None
    patch_attention_available: bool = False
    patch_attention_url: Optional[str] = None
    method: Optional[str] = None
    artifact_url: Optional[str] = None
    disclaimer: str = (
        "Experimental model saliency aid only (Grad-CAM / ViT patch attention); not definitive proof of manipulation."
    )
    reason: str


class AnalyzeImageResponse(BaseModel):
    scan_id: str
    created_at: str
    sha256: str
    original_deleted: bool
    inference_ms: int
    execution_device: str
    vram_peak_mb: Optional[float] = None
    file_metadata: FileMetadata
    dinov2_features: Dinov2FeatureSummary
    ai_generated_indicator: TaskIndicator
    univfd_indicator: Optional[TaskIndicator] = None
    face_manipulation_indicator: TaskIndicator
    experimental_attribution: ExperimentalAttribution
    record_expires_at: str
    artifact_expires_at: Optional[str] = None


class ScanSummary(BaseModel):
    scan_id: str
    created_at: str
    sha256: str
    filename: str
    detected_mime: str
    byte_size: int
    width: int
    height: int
    ai_generated_status: TaskStatus
    ai_generated_score: Optional[float] = None
    ai_generated_label: Optional[str] = None
    ai_model_id: Optional[str] = None
    univfd_status: Optional[TaskStatus] = None
    univfd_score: Optional[float] = None
    univfd_label: Optional[str] = None
    univfd_model_id: Optional[str] = None
    face_manipulation_status: TaskStatus
    face_manipulation_score: Optional[float] = None
    face_manipulation_label: Optional[str] = None
    face_model_id: Optional[str] = None
    execution_device: Optional[str] = None
    inference_ms: Optional[int] = None
    has_artifact: bool
    original_deleted: bool
    record_expires_at: str
    artifact_expires_at: Optional[str] = None


class ScanListResponse(BaseModel):
    items: list[ScanSummary]
    total: int
    limit: int
    offset: int


class DailyActivityBucket(BaseModel):
    date: str
    total: int
    ai_flagged: int
    face_flagged: int


class AnalyticsResponse(BaseModel):
    total_scans: int
    ai_evaluated_count: int
    ai_flagged_count: int
    ai_natural_count: int
    univfd_evaluated_count: int = 0
    univfd_flagged_count: int = 0
    face_evaluated_count: int
    face_flagged_count: int
    face_no_face_count: int
    artifacts_retained_count: int
    originals_deleted_count: int
    avg_inference_ms: Optional[float] = None
    min_inference_ms: Optional[int] = None
    max_inference_ms: Optional[int] = None
    avg_vram_peak_mb: Optional[float] = None
    max_vram_peak_mb: Optional[float] = None
    ai_score_buckets: List[int]
    daily_activity: List[DailyActivityBucket]
    device_breakdown: dict[str, int]
    mime_breakdown: dict[str, int]


class DeleteScanResponse(BaseModel):
    deleted: bool
    scan_id: str
    artifact_deleted: bool


class DeleteArtifactResponse(BaseModel):
    artifact_deleted: bool
    scan_id: str
    record_preserved: bool = True


class PurgeExpiredResponse(BaseModel):
    expired_records_deleted: int
    expired_artifacts_purged: int


class QueuedModalityResponse(BaseModel):
    modality: Literal["video", "audio"]
    status: Literal["queued"] = "queued"
    milestone: str
    message: str


# --- Dataset Evaluation Schemas ---


class ConfusionMatrixCounts(BaseModel):
    tp: int
    fp: int
    tn: int
    fn: int


class DetectorEvaluationMetrics(BaseModel):
    model_id: str
    model_name: str
    evaluated_count: int
    error_count: int
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    false_positive_rate: Optional[float] = None
    false_negative_rate: Optional[float] = None
    confusion_matrix: ConfusionMatrixCounts
    mean_inference_ms: Optional[float] = None
    total_inference_ms: int
    peak_vram_mb: Optional[float] = None


class EvaluationImageResult(BaseModel):
    filename: str
    relative_path: str
    ground_truth: GroundTruthLabel
    smaug_status: TaskStatus
    smaug_score: Optional[float] = None
    smaug_prediction: Optional[GroundTruthLabel] = None
    smaug_correct: Optional[bool] = None
    smaug_inference_ms: Optional[int] = None
    univfd_status: Optional[TaskStatus] = None
    univfd_score: Optional[float] = None
    univfd_prediction: Optional[GroundTruthLabel] = None
    univfd_correct: Optional[bool] = None
    univfd_inference_ms: Optional[int] = None
    execution_device: str = "cpu"
    error_message: Optional[str] = None


class DatasetInspectResponse(BaseModel):
    dataset_dir: str
    ai_generated_dir: str
    real_dir: str
    ai_generated_exists: bool
    real_exists: bool
    ai_generated_count: int
    real_count: int
    total_supported_count: int
    unsupported_or_skipped: List[str]


class DatasetEvaluationRequest(BaseModel):
    dataset_dir: Optional[str] = None
    split_role: DatasetSplitRole = "evaluation"
    decision_threshold: float = Field(default=0.385, gt=0.0, lt=1.0)
    include_univfd: bool = True
    max_images: Optional[int] = Field(default=None, ge=1, le=5000)


class DatasetEvaluationSummary(BaseModel):
    evaluation_id: str
    created_at: str
    dataset_dir: str
    split_role: DatasetSplitRole
    decision_threshold: float
    threshold_auto_tuned: bool = False
    threshold_policy_note: str
    models_trained_or_finetuned: bool = False
    include_univfd: bool
    total_images: int
    ai_generated_count: int
    real_count: int
    skipped_count: int
    execution_device: str
    total_duration_ms: int
    smaug_metrics: DetectorEvaluationMetrics
    univfd_metrics: Optional[DetectorEvaluationMetrics] = None


class DatasetEvaluationDetail(DatasetEvaluationSummary):
    items: List[EvaluationImageResult]
