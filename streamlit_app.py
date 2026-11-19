"""Standalone, CPU-only Streamlit demo using MediaGuard's real Imager-1-Tiny path."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.ml.specs import MODEL_CATALOG  # noqa: E402


MODEL_KEY = "ai_generated_detector"
MODEL_SPEC = MODEL_CATALOG[MODEL_KEY]
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MAX_IMAGE_DIMENSION = 8192
DEFAULT_MODELS_DIR = Path.home() / ".cache" / "mediaguard-streamlit" / "models"
MODELS_DIR = Path(
    os.environ.get("MEDIAGUARD_STREAMLIT_MODELS_DIR", str(DEFAULT_MODELS_DIR))
).expanduser()
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf-cache"))
os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch-cache"))
ARTIFACTS_DIR = Path(tempfile.gettempdir()) / "mediaguard-streamlit" / "artifacts"
LOCAL_CHECKPOINT = (
    APP_DIR / "backend" / "models" / "imager_1_tiny" / MODEL_SPEC.primary_weight_file
)
CACHED_CHECKPOINT = MODELS_DIR / "imager_1_tiny" / MODEL_SPEC.primary_weight_file


st.set_page_config(
    page_title="MediaGuard | Image Check",
    page_icon="🛡️",
    layout="centered",
)


@st.cache_resource(show_spinner=False, max_entries=1)
def load_primary_engine() -> tuple[Any, threading.Lock]:
    """Verify/cache and load only Imager-1-Tiny, then keep it in CPU memory."""
    from app.ml.inference import InferenceEngine
    from app.ml.model_manager import ModelManager

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Reuse an already-present ignored local checkpoint without letting the
    # backend ModelManager touch or migrate the rest of backend/models/.
    if not CACHED_CHECKPOINT.is_file() and LOCAL_CHECKPOINT.is_file():
        CACHED_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LOCAL_CHECKPOINT, CACHED_CHECKPOINT)

    manager = ModelManager(models_dir=MODELS_DIR)
    primary = manager.ensure_model_downloaded(MODEL_KEY, force_verify=True)
    if not primary.verified:
        raise RuntimeError(
            primary.error_message
            or "Imager-1-Tiny weights are missing or failed SHA-256 verification."
        )

    engine = InferenceEngine(
        model_manager=manager,
        artifacts_dir=ARTIFACTS_DIR,
        decision_threshold=0.385,
    )
    engine.lightweight_mode = True
    if not engine._load_primary_detector_if_verified():
        detail = manager.get_entry(MODEL_KEY).error_message
        raise RuntimeError(detail or "Imager-1-Tiny could not be loaded on CPU.")

    return engine, threading.Lock()


def read_uploaded_image(uploaded_file: Any) -> tuple[Image.Image, dict[str, Any]]:
    raw = uploaded_file.getvalue()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ValueError("This demo accepts image files up to 15 MiB.")
    if not raw:
        raise ValueError("The selected file is empty.")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            detected_format = source.format
            if detected_format not in {"PNG", "JPEG"}:
                raise ValueError("Please upload a PNG or JPEG image.")
            original_width, original_height = source.size
            if (
                original_width > MAX_IMAGE_DIMENSION
                or original_height > MAX_IMAGE_DIMENSION
            ):
                raise ValueError("Image dimensions must be 8192 pixels or smaller.")
            if original_width * original_height > MAX_IMAGE_PIXELS:
                raise ValueError(
                    "This demo accepts images up to 12 megapixels to limit memory use."
                )
            source.verify()

        with Image.open(io.BytesIO(raw)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise ValueError("The selected file is not a valid readable PNG or JPEG image.") from exc

    width, height = image.size
    metadata = {
        "Filename": uploaded_file.name,
        "Format": "PNG" if detected_format == "PNG" else "JPEG",
        "File size": f"{len(raw):,} bytes",
        "Dimensions": f"{width:,} × {height:,} px",
        "SHA-256": hashlib.sha256(raw).hexdigest(),
    }
    return image, metadata


def main() -> None:
    st.title("🛡️ MediaGuard")
    st.subheader("AI-generated image check")
    st.write(
        "Inspect an image with MediaGuard’s Imager-1-Tiny detector. "
        "The model runs locally in this app process on CPU."
    )

    st.markdown(
        f"**Detector:** [{MODEL_SPEC.name}]({MODEL_SPEC.source_url})  "
        f"\n**Runtime:** CPU · FP32 · fixed threshold 0.385"
    )

    if st.session_state.get("mediaguard_primary_loaded"):
        st.caption("Model status: loaded and ready · CPU inference · Optional detectors are not loaded.")
    else:
        st.caption("Model status: not loaded · it will load only when you analyze an image.")

    if CACHED_CHECKPOINT.is_file():
        st.caption(
            "Checkpoint file is present; SHA-256 and checkpoint metadata are verified "
            "when analysis first loads the detector."
        )
    elif LOCAL_CHECKPOINT.is_file():
        st.caption("A local checkpoint is available and will be copied to the app cache on first analysis.")
    else:
        st.info(
            "The Imager-1-Tiny checkpoint is not cached yet. On the first analysis, "
            "the app downloads the pinned public checkpoint (94,371,929 bytes, about 90 MiB) "
            "from Hugging Face and verifies its SHA-256."
        )

    uploaded = st.file_uploader(
        "Choose a PNG or JPEG image",
        type=["png", "jpg", "jpeg"],
        max_upload_size=15,
        help="Maximum 15 MiB and 12 megapixels for this CPU demo.",
    )
    if uploaded is None:
        st.caption("No image selected. The detector stays unloaded until you analyze an image.")
        return

    try:
        image, metadata = read_uploaded_image(uploaded)
    except ValueError as exc:
        st.error(str(exc))
        return

    st.image(image, caption=uploaded.name, use_container_width=True)
    with st.expander("Image details", expanded=True):
        for label, value in metadata.items():
            st.write(f"**{label}:** {value}")

    if not st.button("Analyze image", type="primary", use_container_width=True):
        return

    try:
        with st.spinner("Verifying weights and running Imager-1-Tiny on CPU…"):
            engine, inference_lock = load_primary_engine()
            with inference_lock:
                indicator, vram_peak_mb, device = engine.run_ai_generated_detection(image)
    except Exception as exc:
        st.error(f"Imager-1-Tiny could not be loaded or run: {exc}")
        st.caption(
            "Check network access to the public Hugging Face model repository and "
            "available app memory, then retry."
        )
        return

    if indicator.status != "evaluated" or indicator.score is None:
        st.error(indicator.reason or "The detector did not return an evaluated result.")
        return

    st.session_state["mediaguard_primary_loaded"] = True
    ai_score = float(indicator.score)
    selected_class_score = ai_score if ai_score >= indicator.decision_threshold else 1.0 - ai_score

    st.success(indicator.prediction_label or "Analysis complete")
    result_col, score_col = st.columns(2)
    result_col.metric("Prediction", indicator.prediction_label or "Evaluated")
    score_col.metric("Selected-class score", f"{selected_class_score:.1%}")

    st.metric("Raw P(AI), uncalibrated", f"{ai_score:.1%}")
    st.progress(min(1.0, max(0.0, ai_score)))
    st.caption(
        f"Decision threshold: {indicator.decision_threshold:.3f}. "
        "Scores are raw, uncalibrated model outputs—not certainty or proof."
    )

    with st.expander("Analysis details", expanded=True):
        st.write(f"**Model:** {indicator.model_name} ({indicator.model_id})")
        st.write(f"**Architecture:** {indicator.architecture}")
        st.write(f"**Model revision:** {indicator.model_version}")
        st.write(f"**Actual execution device:** {indicator.execution_device or device}")
        st.write(f"**Inference time:** {indicator.processing_ms} ms")
        if vram_peak_mb is not None:
            st.write(f"**Measured peak VRAM:** {vram_peak_mb:.2f} MB")
        st.write(f"**Detector note:** {indicator.reason}")
        st.write(f"**Limitations:** {indicator.limitations}")

if __name__ == "__main__":
    main()
