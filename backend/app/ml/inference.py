import gc
import importlib.util
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.ml.model_manager import ModelManager
from app.ml.specs import MODEL_CATALOG
from app.schemas import (
    Dinov2FeatureSummary,
    ExperimentalAttribution,
    FaceBox,
    MeasuredVram,
    TaskIndicator,
)


class InferenceEngine:
    """
    Hardware-optimized sequential PyTorch inference engine for RTX 2050 (4 GB VRAM)
    and CPU fallback.

    - Pipeline A (Primary AI-generated detector): Official `abacusai/Smaug-AI-Image-Detector`
      via `SmaugImageDetect.from_pretrained(...)` scoring three 528x528 crops.
      Runs in bfloat16 when supported on CUDA or float32 (never fp16, per official repo).
    - Pipeline A2 (Secondary comparison AI-generated detector): UniversalFakeDetect (UnivFD)
      using `openai/clip-vit-large-patch14` vision backbone + `fc_weights.pth` linear head.
    - Pipeline B (Face manipulation detection): OpenCV Haar face detection +
      `Xicor9/efficientnet-b0-ffpp-c23` + CNN Grad-CAM.
    - Feature / Saliency extractor: `facebook/dinov2-small` (384-d CLS + patch attention).
    """

    def __init__(
        self,
        model_manager: ModelManager,
        artifacts_dir: Path,
        low_vram_offload: bool = True,
        decision_threshold: float = 0.50,
    ) -> None:
        self.manager = model_manager
        self.artifacts_dir = artifacts_dir
        self.low_vram_offload = low_vram_offload
        self.decision_threshold = decision_threshold

        self._dinov2_processor: Any = None
        self._dinov2_model: Any = None
        self._smaug_detector: Any = None
        self._univfd_processor: Any = None
        self._univfd_model: Any = None
        self._face_model: Any = None
        self._face_cascade_primary: Any = None
        self._face_cascade_alt: Any = None

    def probe_hardware(self) -> Tuple[bool, str, str, Optional[MeasuredVram]]:
        try:
            import torch
        except ImportError:
            return False, "CPU (PyTorch unavailable)", "cpu", None

        if not torch.cuda.is_available():
            return False, "CPU (CUDA not detected)", "cpu", None

        try:
            idx = 0
            _ = torch.empty((1,), device="cuda:0")
            name = torch.cuda.get_device_name(idx)
            allocated = torch.cuda.memory_allocated(idx) / (1024 * 1024)
            peak = torch.cuda.max_memory_allocated(idx) / (1024 * 1024)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            vram = MeasuredVram(
                allocated_mb=round(allocated, 2),
                peak_mb=round(peak, 2),
                free_mb=round(free_b / (1024 * 1024), 2),
                total_mb=round(total_b / (1024 * 1024), 2),
            )
            return True, name, f"cuda:0 ({name})", vram
        except Exception as exc:
            return False, f"CPU (CUDA verification failed: {exc})", "cpu", None

    def _init_face_cascades(self) -> None:
        if self._face_cascade_primary is None:
            primary_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            alt_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_alt2.xml"
            self._face_cascade_primary = cv2.CascadeClassifier(str(primary_path))
            self._face_cascade_alt = cv2.CascadeClassifier(str(alt_path))

    def load_verified_models(self) -> None:
        self.manager.verify_existing_models()
        self._load_dinov2_if_verified()
        self._load_smaug_if_verified()
        self._load_univfd_if_verified()
        self._load_face_detector_if_verified()

    def unload_models(self) -> None:
        self._dinov2_processor = None
        self._dinov2_model = None
        self._smaug_detector = None
        self._univfd_processor = None
        self._univfd_model = None
        self._face_model = None
        for key in (
            "dinov2_extractor",
            "ai_generated_detector",
            "univfd_detector",
            "face_manipulation_detector",
        ):
            self.manager.mark_loaded(key, False)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _load_dinov2_if_verified(self) -> bool:
        if self._dinov2_model is not None:
            return True
        entry = self.manager.get_entry("dinov2_extractor")
        if not entry.verified:
            return False
        try:
            from transformers import AutoImageProcessor, Dinov2Model

            mdir = str(self.manager.get_model_dir("dinov2_extractor"))
            self._dinov2_processor = AutoImageProcessor.from_pretrained(
                mdir, local_files_only=True
            )
            self._dinov2_model = Dinov2Model.from_pretrained(
                mdir,
                local_files_only=True,
                attn_implementation="eager",
            )
            self._dinov2_model.eval()
            self.manager.mark_loaded("dinov2_extractor", True)
            return True
        except Exception as exc:
            self._dinov2_model = None
            self.manager.mark_loaded("dinov2_extractor", False, f"Load failed: {exc}")
            return False

    def _load_smaug_if_verified(self) -> bool:
        if self._smaug_detector is not None:
            return True
        entry = self.manager.get_entry("ai_generated_detector")
        if not entry.verified:
            return False
        try:
            mdir = self.manager.get_model_dir("ai_generated_detector")
            helper_path = mdir / "smaug_image_detect.py"
            if not helper_path.exists():
                raise RuntimeError("Official smaug_image_detect.py missing from model directory.")

            mod_name = "mediaguard_smaug_image_detect"
            spec_obj = importlib.util.spec_from_file_location(mod_name, str(helper_path))
            if spec_obj is None or spec_obj.loader is None:
                raise RuntimeError("Unable to load official smaug_image_detect module spec.")
            module = importlib.util.module_from_spec(spec_obj)
            sys.modules[mod_name] = module
            spec_obj.loader.exec_module(module)

            SmaugImageDetect = getattr(module, "SmaugImageDetect")
            detector = SmaugImageDetect.from_pretrained(str(mdir), device="cpu")
            detector.st.eval()
            self._smaug_detector = detector
            self.manager.mark_loaded("ai_generated_detector", True)
            return True
        except Exception as exc:
            self._smaug_detector = None
            self.manager.mark_loaded(
                "ai_generated_detector", False, f"Smaug load failed: {exc}"
            )
            return False

    def _load_univfd_if_verified(self) -> bool:
        if self._univfd_model is not None:
            return True
        entry = self.manager.get_entry("univfd_detector")
        if not entry.verified:
            return False
        try:
            import torch
            import torch.nn as nn
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            mdir = self.manager.get_model_dir("univfd_detector")
            head_path = mdir / "fc_weights.pth"

            processor = CLIPImageProcessor.from_pretrained(
                str(mdir), local_files_only=True
            )
            vision_model = CLIPVisionModelWithProjection.from_pretrained(
                str(mdir),
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            vision_model.eval()

            state = torch.load(head_path, map_location="cpu", weights_only=True)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise RuntimeError(f"Unsupported UnivFD checkpoint format: {type(state)!r}")

            cleaned = {}
            for key, value in state.items():
                norm_k = (
                    key.replace("module.", "")
                    .replace("model.", "")
                    .replace("fc.", "")
                    .replace("head.", "")
                )
                if norm_k in {"weight", "bias"}:
                    cleaned[norm_k] = value

            embed_dim = int(getattr(vision_model.config, "projection_dim", 768))
            head = nn.Linear(embed_dim, 1)
            if cleaned:
                head.load_state_dict(cleaned, strict=True)
            else:
                head.load_state_dict(state, strict=False)
            head.eval()

            class UnivFDCompositeModule(nn.Module):
                def __init__(self, vision: nn.Module, linear_head: nn.Module) -> None:
                    super().__init__()
                    self.vision = vision
                    self.head = linear_head

                def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
                    out = self.vision(pixel_values=pixel_values)
                    feats = out.image_embeds.float()
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    return self.head(feats).flatten()

            composite = UnivFDCompositeModule(vision_model, head)
            composite.eval()

            self._univfd_processor = processor
            self._univfd_model = composite
            self.manager.mark_loaded("univfd_detector", True)
            return True
        except Exception as exc:
            self._univfd_model = None
            self.manager.mark_loaded("univfd_detector", False, f"Load failed: {exc}")
            return False

    def _load_face_detector_if_verified(self) -> bool:
        if self._face_model is not None:
            return True
        entry = self.manager.get_entry("face_manipulation_detector")
        if not entry.verified:
            return False
        try:
            import torch
            import torch.nn as nn
            from torchvision import models

            spec = MODEL_CATALOG["face_manipulation_detector"]
            weight_path = (
                self.manager.get_model_dir("face_manipulation_detector")
                / spec.primary_weight_file
            )
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
            state_dict = torch.load(
                weight_path, map_location="cpu", weights_only=True
            )
            model.load_state_dict(state_dict)
            model.eval()
            self._face_model = model
            self._init_face_cascades()
            self.manager.mark_loaded("face_manipulation_detector", True)
            return True
        except Exception as exc:
            self._face_model = None
            self.manager.mark_loaded(
                "face_manipulation_detector", False, f"Load failed: {exc}"
            )
            return False

    @contextmanager
    def _activate_model_device(self, model: Any) -> Iterator[Tuple[str, bool]]:
        import torch

        cuda_ok = torch.cuda.is_available()
        device = "cuda:0" if cuda_ok else "cpu"
        use_amp = cuda_ok
        if cuda_ok:
            try:
                torch.cuda.reset_peak_memory_stats(0)
                model.to(device)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                device = "cpu"
                use_amp = False
                model.to("cpu")
        try:
            yield device, use_amp
        finally:
            if cuda_ok and self.low_vram_offload:
                model.to("cpu")
                torch.cuda.empty_cache()
                gc.collect()

    @staticmethod
    def _measure_peak_vram_mb() -> Optional[float]:
        try:
            import torch

            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
        except Exception:
            pass
        return None

    @staticmethod
    def _save_heatmap_overlay(
        base_rgb: Image.Image,
        heatmap_01: np.ndarray,
        out_path: Path,
        alpha: float = 0.48,
    ) -> None:
        w, h = base_rgb.size
        resized = cv2.resize(heatmap_01.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        resized = np.clip(resized, 0.0, 1.0)
        uint8_map = (resized * 255).astype(np.uint8)
        colored_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_INFERNO)
        colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
        base_np = np.asarray(base_rgb, dtype=np.float32)
        blended = (1.0 - alpha) * base_np + alpha * colored_rgb.astype(np.float32)
        blended_uint8 = np.clip(blended, 0, 255).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(blended_uint8).save(out_path, format="PNG")

    def run_dinov2(
        self,
        pil_rgb: Image.Image,
        scan_id: str,
        include_attribution: bool = False,
    ) -> Tuple[Dinov2FeatureSummary, Optional[Path], Optional[float], str]:
        spec = MODEL_CATALOG["dinov2_extractor"]
        if not self._load_dinov2_if_verified():
            entry = self.manager.get_entry("dinov2_extractor")
            return (
                Dinov2FeatureSummary(
                    status="unavailable",
                    model_id=spec.repo_id,
                    reason=entry.error_message
                    or "DINOv2 weights not downloaded/verified. Run `python setup_models.py`.",
                ),
                None,
                None,
                "unavailable",
            )

        import torch

        started = time.perf_counter()
        peak_vram: Optional[float] = None
        exec_device = "cpu"
        patch_path: Optional[Path] = None

        def _forward(device: str, use_amp: bool):
            inputs = self._dinov2_processor(images=pil_rgb, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.inference_mode():
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        outputs = self._dinov2_model(
                            **inputs, output_attentions=include_attribution
                        )
                else:
                    outputs = self._dinov2_model(
                        **inputs, output_attentions=include_attribution
                    )
            return outputs

        try:
            with self._activate_model_device(self._dinov2_model) as (device, use_amp):
                exec_device = device
                try:
                    outputs = _forward(device, use_amp)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self._dinov2_model.to("cpu")
                    exec_device = "cpu (CUDA OOM fallback)"
                    outputs = _forward("cpu", False)

                cls_vec = outputs.last_hidden_state[0, 0].detach().float().cpu().numpy()
                l2_norm = float(np.linalg.norm(cls_vec))

                if include_attribution and getattr(outputs, "attentions", None):
                    last_attn = outputs.attentions[-1][0, :, 0, 1:].detach().float().cpu()
                    mean_attn = last_attn.mean(dim=0).numpy()
                    grid = int(round(np.sqrt(mean_attn.shape[0])))
                    if grid * grid == mean_attn.shape[0]:
                        attn_map = mean_attn.reshape(grid, grid)
                        lo, hi = float(attn_map.min()), float(attn_map.max())
                        norm_map = (attn_map - lo) / (hi - lo + 1e-8)
                        patch_path = self.artifacts_dir / f"{scan_id}_dinov2_patch.png"
                        self._save_heatmap_overlay(pil_rgb, norm_map, patch_path)

                peak_vram = self._measure_peak_vram_mb()

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.manager.record_runtime_metrics("dinov2_extractor", exec_device, elapsed_ms, peak_vram)
            return (
                Dinov2FeatureSummary(
                    status="evaluated",
                    model_id=spec.repo_id,
                    embedding_dim=int(cls_vec.shape[0]),
                    l2_norm=round(l2_norm, 6),
                    embedding_preview=[round(float(x), 5) for x in cls_vec[:8]],
                    processing_ms=elapsed_ms,
                    execution_device=exec_device,
                    reason="384-d self-supervised visual feature vector extracted (non-diagnostic).",
                ),
                patch_path,
                peak_vram,
                exec_device,
            )
        except Exception as exc:
            return (
                Dinov2FeatureSummary(
                    status="error",
                    model_id=spec.repo_id,
                    reason=f"DINOv2 feature extraction failed: {str(exc)[:180]}",
                ),
                None,
                peak_vram,
                exec_device,
            )

    def run_ai_generated_detection(
        self, pil_rgb: Image.Image
    ) -> Tuple[TaskIndicator, Optional[float], str]:
        """
        Primary AI-Generated Image Detector: Official `abacusai/Smaug-AI-Image-Detector`
        invoked via `SmaugImageDetect.from_pretrained(...)` with 3x 528x528 crops.
        """
        spec = MODEL_CATALOG["ai_generated_detector"]
        if not self._load_smaug_if_verified():
            entry = self.manager.get_entry("ai_generated_detector")
            return (
                TaskIndicator(
                    task="ai_generated_image_detection",
                    status="unavailable",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    prediction_label=None,
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device="unavailable",
                    limitations=spec.limitations,
                    reason=entry.error_message
                    or "Smaug AI Image Detector weights not downloaded/verified. Run `python setup_models.py`.",
                ),
                None,
                "unavailable",
            )

        import torch

        started = time.perf_counter()
        peak_vram: Optional[float] = None
        exec_device = "cpu"
        cuda_ok = torch.cuda.is_available()
        st = self._smaug_detector.st

        try:
            if cuda_ok:
                use_bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
                target_dtype = torch.bfloat16 if use_bf16 else torch.float32
                try:
                    torch.cuda.reset_peak_memory_stats(0)
                    st.to(device="cuda:0", dtype=target_dtype)
                    exec_device = "cuda:0 (bf16)" if use_bf16 else "cuda:0 (fp32)"
                    with torch.inference_mode():
                        ai_score = float(self._smaug_detector(pil_rgb, crops=3))
                    peak_vram = self._measure_peak_vram_mb()
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    st.to(device="cpu", dtype=torch.float32)
                    exec_device = "cpu (CUDA OOM fallback)"
                    with torch.inference_mode():
                        ai_score = float(self._smaug_detector(pil_rgb, crops=3))
                finally:
                    if self.low_vram_offload:
                        st.to(device="cpu", dtype=torch.float32)
                        torch.cuda.empty_cache()
                        gc.collect()
            else:
                exec_device = "cpu"
                st.to(device="cpu", dtype=torch.float32)
                with torch.inference_mode():
                    ai_score = float(self._smaug_detector(pil_rgb, crops=3))

            ai_score = max(0.0, min(1.0, ai_score))
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.manager.record_runtime_metrics(
                "ai_generated_detector", exec_device, elapsed_ms, peak_vram
            )
            label = (
                "AI-Generated Signal"
                if ai_score >= self.decision_threshold
                else "Human / Natural Signal"
            )
            return (
                TaskIndicator(
                    task="ai_generated_image_detection",
                    status="evaluated",
                    score=round(ai_score, 4),
                    decision_threshold=self.decision_threshold,
                    prediction_label=label,
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device=exec_device,
                    processing_ms=elapsed_ms,
                    limitations=spec.limitations,
                    reason=(
                        f"Smaug official 3x 528x528 crop inference returned raw P(ai)={ai_score:.4f} "
                        f"vs threshold {self.decision_threshold:.2f} on {exec_device}."
                    ),
                ),
                peak_vram,
                exec_device,
            )
        except Exception as exc:
            return (
                TaskIndicator(
                    task="ai_generated_image_detection",
                    status="error",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device=exec_device,
                    limitations=spec.limitations,
                    reason=f"Smaug inference failed: {str(exc)[:180]}",
                ),
                peak_vram,
                exec_device,
            )

    def run_univfd_detection(
        self, pil_rgb: Image.Image
    ) -> Tuple[TaskIndicator, Optional[float], str]:
        """Secondary comparison detector: UniversalFakeDetect (CLIP ViT-L/14 + fc_weights.pth)."""
        spec = MODEL_CATALOG["univfd_detector"]
        if not self._load_univfd_if_verified():
            entry = self.manager.get_entry("univfd_detector")
            return (
                TaskIndicator(
                    task="ai_generated_image_detection",
                    status="unavailable",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    prediction_label=None,
                    calibrated=False,
                    model_id="univfd-clip-vit-l14",
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device="unavailable",
                    limitations=spec.limitations,
                    reason=entry.error_message
                    or "UnivFD comparison weights not downloaded/verified.",
                ),
                None,
                "unavailable",
            )

        import torch

        started = time.perf_counter()
        peak_vram: Optional[float] = None
        exec_device = "cpu"

        def _forward(device: str, use_amp: bool):
            inputs = self._univfd_processor(images=pil_rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.inference_mode():
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = self._univfd_model(pixel_values)
                else:
                    logits = self._univfd_model(pixel_values)
            return logits.detach().float().cpu()

        try:
            with self._activate_model_device(self._univfd_model) as (device, use_amp):
                exec_device = device
                try:
                    logits = _forward(device, use_amp)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    self._univfd_model.to("cpu")
                    exec_device = "cpu (CUDA OOM fallback)"
                    logits = _forward("cpu", False)

                raw_logit = float(logits[0].item())
                ai_score = float(torch.sigmoid(logits[0]).item())
                peak_vram = self._measure_peak_vram_mb()

            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.manager.record_runtime_metrics("univfd_detector", exec_device, elapsed_ms, peak_vram)
            label = (
                "AI-Generated Signal"
                if ai_score >= self.decision_threshold
                else "Human / Natural Signal"
            )
            return (
                TaskIndicator(
                    task="ai_generated_image_detection",
                    status="evaluated",
                    score=round(ai_score, 4),
                    decision_threshold=self.decision_threshold,
                    prediction_label=label,
                    calibrated=False,
                    model_id="univfd-clip-vit-l14",
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device=exec_device,
                    processing_ms=elapsed_ms,
                    limitations=spec.limitations,
                    reason=(
                        f"UnivFD comparison raw sigmoid P(ai)={ai_score:.4f} (logit={raw_logit:+.4f}) "
                        f"vs threshold {self.decision_threshold:.2f} on {exec_device}."
                    ),
                ),
                peak_vram,
                exec_device,
            )
        except Exception as exc:
            return (
                TaskIndicator(
                    task="ai_generated_image_detection",
                    status="error",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    calibrated=False,
                    model_id="univfd-clip-vit-l14",
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device=exec_device,
                    limitations=spec.limitations,
                    reason=f"UnivFD inference failed: {str(exc)[:180]}",
                ),
                peak_vram,
                exec_device,
            )

    def _detect_faces(self, pil_rgb: Image.Image) -> List[Tuple[int, int, int, int]]:
        self._init_face_cascades()
        rgb_np = np.asarray(pil_rgb)
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        min_dim = min(pil_rgb.width, pil_rgb.height)
        min_face = max(32, int(min_dim * 0.06))

        faces = self._face_cascade_primary.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
        )
        if len(faces) == 0:
            faces = self._face_cascade_alt.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_face, min_face)
            )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def run_face_manipulation_detection(
        self,
        pil_rgb: Image.Image,
        scan_id: str,
        include_attribution: bool = False,
    ) -> Tuple[TaskIndicator, Optional[Path], Optional[float], str]:
        spec = MODEL_CATALOG["face_manipulation_detector"]
        if not self._load_face_detector_if_verified():
            entry = self.manager.get_entry("face_manipulation_detector")
            return (
                TaskIndicator(
                    task="face_manipulation_detection",
                    status="unavailable",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    prediction_label=None,
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device="unavailable",
                    limitations=spec.limitations,
                    reason=entry.error_message
                    or "Face manipulation checkpoint not downloaded/verified. Run `python setup_models.py`.",
                ),
                None,
                None,
                "unavailable",
            )

        started = time.perf_counter()
        faces = self._detect_faces(pil_rgb)
        if not faces:
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.manager.record_runtime_metrics("face_manipulation_detector", "cpu (face detector)", elapsed_ms, None)
            return (
                TaskIndicator(
                    task="face_manipulation_detection",
                    status="no_face_detected",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    prediction_label="No Face Detected",
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device="cpu (face detector)",
                    processing_ms=elapsed_ms,
                    faces_detected=0,
                    face_boxes=[],
                    limitations=spec.limitations,
                    reason="No detectable frontal face found; face manipulation classifier skipped.",
                ),
                None,
                None,
                "cpu (face detector)",
            )

        import torch
        from torchvision import transforms

        preprocess = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        img_w, img_h = pil_rgb.size
        face_boxes_schema: List[FaceBox] = []
        crops_with_coords: List[Tuple[torch.Tensor, Tuple[int, int, int, int]]] = []

        for (x, y, w, h) in faces:
            face_boxes_schema.append(FaceBox(x=x, y=y, width=w, height=h))
            margin_x = int(w * 0.25)
            margin_y = int(h * 0.25)
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(img_w, x + w + margin_x)
            y2 = min(img_h, y + h + margin_y)
            crop = pil_rgb.crop((x1, y1, x2, y2))
            tensor = preprocess(crop).unsqueeze(0)
            crops_with_coords.append((tensor, (x1, y1, x2, y2)))

        peak_vram: Optional[float] = None
        exec_device = "cpu"
        gradcam_path: Optional[Path] = None

        try:
            with self._activate_model_device(self._face_model) as (device, use_amp):
                exec_device = device
                scores: List[float] = []
                cam_canvas = np.zeros((img_h, img_w), dtype=np.float32)

                for tensor, (x1, y1, x2, y2) in crops_with_coords:
                    def _run_face(dev: str, amp: bool):
                        x_in = tensor.to(dev)
                        if include_attribution:
                            captured: dict[str, Any] = {}

                            def _hook(_m: Any, _inp: Any, out: Any) -> None:
                                captured["acts"] = out

                            handle = self._face_model.features[-1].register_forward_hook(_hook)
                            try:
                                with torch.enable_grad():
                                    logits = self._face_model(x_in)
                                    acts = captured["acts"]
                                    fake_logit = logits[0, 1]
                                    grads = torch.autograd.grad(
                                        fake_logit, acts, retain_graph=False
                                    )[0]
                                    weights = grads.mean(dim=(2, 3), keepdim=True)
                                    cam = torch.relu((weights * acts).sum(dim=1))[0]
                                    cam_np = cam.detach().float().cpu().numpy()
                            finally:
                                handle.remove()
                            probs = torch.softmax(logits.detach().float(), dim=1)
                            return float(probs[0, 1].item()), cam_np
                        else:
                            with torch.inference_mode():
                                if amp:
                                    with torch.amp.autocast("cuda", dtype=torch.float16):
                                        logits = self._face_model(x_in)
                                else:
                                    logits = self._face_model(x_in)
                            probs = torch.softmax(logits.detach().float(), dim=1)
                            return float(probs[0, 1].item()), None

                    try:
                        fake_prob, cam_np = _run_face(device, use_amp)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        self._face_model.to("cpu")
                        exec_device = "cpu (CUDA OOM fallback)"
                        fake_prob, cam_np = _run_face("cpu", False)

                    scores.append(fake_prob)
                    if cam_np is not None:
                        lo, hi = float(cam_np.min()), float(cam_np.max())
                        norm_cam = (cam_np - lo) / (hi - lo + 1e-8)
                        resized_cam = cv2.resize(
                            norm_cam, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC
                        )
                        cam_canvas[y1:y2, x1:x2] = np.maximum(
                            cam_canvas[y1:y2, x1:x2], resized_cam
                        )

                peak_vram = self._measure_peak_vram_mb()

            if include_attribution and cam_canvas.max() > 0:
                gradcam_path = self.artifacts_dir / f"{scan_id}_gradcam.png"
                self._save_heatmap_overlay(pil_rgb, cam_canvas, gradcam_path)

            max_fake = max(scores)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self.manager.record_runtime_metrics("face_manipulation_detector", exec_device, elapsed_ms, peak_vram)
            label = (
                "Face Manipulation Signal"
                if max_fake >= self.decision_threshold
                else "No Manipulation Signal"
            )
            return (
                TaskIndicator(
                    task="face_manipulation_detection",
                    status="evaluated",
                    score=round(max_fake, 4),
                    decision_threshold=self.decision_threshold,
                    prediction_label=label,
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device=exec_device,
                    processing_ms=elapsed_ms,
                    faces_detected=len(faces),
                    face_boxes=face_boxes_schema,
                    limitations=spec.limitations,
                    reason=(
                        f"Evaluated {len(faces)} detected face crop(s); max raw P(fake)={max_fake:.4f} "
                        f"vs threshold {self.decision_threshold:.2f} on {exec_device}."
                    ),
                ),
                gradcam_path,
                peak_vram,
                exec_device,
            )
        except Exception as exc:
            return (
                TaskIndicator(
                    task="face_manipulation_detection",
                    status="error",
                    score=None,
                    decision_threshold=self.decision_threshold,
                    calibrated=False,
                    model_id=spec.repo_id,
                    model_name=spec.name,
                    architecture=spec.architecture,
                    model_source=spec.source_url,
                    model_version=spec.revision[:10],
                    execution_device=exec_device,
                    faces_detected=len(faces),
                    face_boxes=face_boxes_schema,
                    limitations=spec.limitations,
                    reason=f"Face manipulation inference failed: {str(exc)[:180]}",
                ),
                None,
                peak_vram,
                exec_device,
            )

    def analyze_image_pipeline(
        self,
        image_bytes: bytes,
        scan_id: str,
        include_attribution: bool,
    ) -> Tuple[
        Dinov2FeatureSummary,
        TaskIndicator,
        TaskIndicator,
        TaskIndicator,
        ExperimentalAttribution,
        Optional[Path],
        Optional[Path],
        Optional[float],
        str,
    ]:
        import io

        with Image.open(io.BytesIO(image_bytes)) as img:
            pil_rgb = ImageOps.exif_transpose(img).convert("RGB")

        dinov2_summary, patch_path, vram_dino, dev_dino = self.run_dinov2(
            pil_rgb, scan_id=scan_id, include_attribution=include_attribution
        )
        ai_indicator, vram_smaug, dev_smaug = self.run_ai_generated_detection(pil_rgb)
        univfd_indicator, vram_univfd, dev_univfd = self.run_univfd_detection(pil_rgb)
        face_indicator, gradcam_path, vram_face, dev_face = (
            self.run_face_manipulation_detection(
                pil_rgb, scan_id=scan_id, include_attribution=include_attribution
            )
        )

        vram_values = [v for v in (vram_dino, vram_smaug, vram_univfd, vram_face) if v is not None]
        peak_vram = max(vram_values) if vram_values else None
        devices = [d for d in (dev_smaug, dev_univfd, dev_face, dev_dino) if d != "unavailable"]
        execution_device = devices[0] if devices else "unavailable"

        gradcam_url = f"/api/v1/artifacts/{gradcam_path.name}" if gradcam_path else None
        patch_url = f"/api/v1/artifacts/{patch_path.name}" if patch_path else None
        any_attr = bool(gradcam_url or patch_url)

        if include_attribution and any_attr:
            parts = []
            if gradcam_url:
                parts.append("EfficientNet-B0 Grad-CAM")
            if patch_url:
                parts.append("DINOv2 CLS Patch Attention")
            attr_reason = f"Generated experimental overlays: {', '.join(parts)}."
        elif include_attribution:
            attr_reason = (
                "No attribution overlay generated (models unavailable or no face detected for Grad-CAM)."
            )
        else:
            attr_reason = "Experimental attribution was not requested for this scan."

        attribution = ExperimentalAttribution(
            available=any_attr,
            gradcam_available=bool(gradcam_url),
            gradcam_url=gradcam_url,
            patch_attention_available=bool(patch_url),
            patch_attention_url=patch_url,
            method="gradcam+dinov2_attention"
            if (gradcam_url and patch_url)
            else ("gradcam" if gradcam_url else ("dinov2_attention" if patch_url else None)),
            artifact_url=gradcam_url or patch_url,
            reason=attr_reason,
        )

        return (
            dinov2_summary,
            ai_indicator,
            univfd_indicator,
            face_indicator,
            attribution,
            gradcam_path,
            patch_path,
            peak_vram,
            execution_device,
        )
