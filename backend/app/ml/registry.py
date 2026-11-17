import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from app.ml.inference import InferenceEngine
from app.ml.model_manager import ModelManager
from app.ml.specs import MODEL_CATALOG
from app.schemas import (
    MeasuredVram,
    ModelComponentReadiness,
    ModelComponentStatus,
    ModelReadinessMap,
)


class ModelRegistry:
    """
    Coordinates ModelManager (persistent downloads & SHA-256 verification) and
    InferenceEngine (sequential GPU / CPU fallback execution).
    """

    def __init__(
        self,
        models_dir: Path,
        artifacts_dir: Path,
        low_vram_offload: bool = True,
        decision_threshold: float = 0.50,
    ) -> None:
        self.gpu_semaphore = asyncio.Semaphore(1)
        self.manager = ModelManager(models_dir=models_dir)
        self.engine = InferenceEngine(
            model_manager=self.manager,
            artifacts_dir=artifacts_dir,
            low_vram_offload=low_vram_offload,
            decision_threshold=decision_threshold,
        )
        self.calibration_evaluated = False

    def load_cached_models(self) -> None:
        self.engine.load_verified_models()

    def setup_and_load_all(self) -> None:
        self.manager.download_all_models()
        self.engine.load_verified_models()

    def probe_hardware(self) -> Tuple[bool, str, str, Optional[MeasuredVram]]:
        return self.engine.probe_hardware()

    def _component_readiness(self, key: str, role: str) -> ModelComponentReadiness:
        spec = MODEL_CATALOG[key]
        entry = self.manager.get_entry(key)

        status: ModelComponentStatus
        if entry.loaded_in_memory:
            status = "ready"
        elif entry.status == "downloading":
            status = "downloading"
        elif entry.status == "verified":
            status = "loading" if self.manager.download_in_progress else "ready"
        elif entry.status == "error":
            status = "error"
        else:
            status = "unavailable"

        reason = entry.error_message or (
            f"Verified and ready ({entry.repo_id})."
            if status == "ready"
            else "Weights not downloaded yet. Run `python setup_models.py` or trigger setup in System view."
        )

        return ModelComponentReadiness(
            status=status,
            model_id=spec.repo_id,
            role=role,
            checkpoint_verified=entry.verified,
            license=spec.license,
            source=spec.source_url,
            reason=reason,
        )

    def get_readiness_map(self) -> ModelReadinessMap:
        return ModelReadinessMap(
            dinov2_extractor=self._component_readiness(
                "dinov2_extractor",
                "Self-supervised 384-d feature extractor & experimental patch attention",
            ),
            ai_generated_detector=self._component_readiness(
                "ai_generated_detector",
                "Primary AI-generated image detector (abacusai/Smaug-AI-Image-Detector · three 528x528 crops)",
            ),
            univfd_detector=self._component_readiness(
                "univfd_detector",
                "Secondary comparison AI-generated detector (UnivFD · CLIP ViT-L/14 + fc_weights.pth)",
            ),
            face_manipulation_detector=self._component_readiness(
                "face_manipulation_detector",
                "Face-swap & manipulation classifier on OpenCV face crops + Grad-CAM",
            ),
            calibration_profile=ModelComponentReadiness(
                status="ready" if self.calibration_evaluated else "unavailable",
                model_id="post_hoc_calibration",
                role="Empirical probability calibration (ECE / reliability curves)",
                checkpoint_verified=self.calibration_evaluated,
                license="N/A",
                source="MediaGuard Evaluation Suite",
                reason="Raw model probabilities are reported uncalibrated; formal ECE calibration has not been applied.",
            ),
        )

    @staticmethod
    def utc_now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
