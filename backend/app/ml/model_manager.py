import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, Optional

from app.ml.specs import MODEL_CATALOG, ModelSpec
from app.schemas import DownloadLifecycleStatus, ModelManifestEntry, ModelsStatusResponse

logger = logging.getLogger(__name__)


def compute_file_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ModelManager:
    """
    Manages persistent checkpoint downloads, SHA-256 verification, provenance
    manifest tracking, runtime telemetry, and offline error reporting in `models_dir`.
    """

    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.manifest_path = models_dir / "manifest.json"
        self._lock = Lock()
        self.download_in_progress = False
        self._state: Dict[str, dict] = {}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_legacy_univfd_dir()
        self._load_or_init_manifest()

    def _migrate_legacy_univfd_dir(self) -> None:
        legacy_ai_dir = self.models_dir / "ai_generated_detector"
        univfd_dir = self.models_dir / "univfd_detector"
        if (legacy_ai_dir / "fc_weights.pth").exists() and not (
            legacy_ai_dir / "smaug_image_detect.py"
        ).exists():
            if not univfd_dir.exists():
                shutil.move(str(legacy_ai_dir), str(univfd_dir))
            else:
                shutil.rmtree(legacy_ai_dir, ignore_errors=True)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _model_dir(self, spec: ModelSpec) -> Path:
        # Keep the historical Smaug checkpoint directory intact. The new
        # lightweight primary has its own directory and never overwrites it.
        if spec.key == "ai_generated_detector":
            return self.models_dir / "imager_1_tiny"
        return self.models_dir / spec.key

    def _load_or_init_manifest(self) -> None:
        persisted: Dict[str, dict] = {}
        if self.manifest_path.exists():
            try:
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                persisted = raw.get("models", {})
            except Exception:
                persisted = {}

        for key in MODEL_CATALOG:
            prev = persisted.get(key, {})
            self._state[key] = {
                "status": prev.get("status", "not_downloaded"),
                "verified": bool(prev.get("verified", False)),
                "loaded_in_memory": False,
                "actual_sha256": prev.get("actual_sha256"),
                "active_device": prev.get("active_device", "cpu"),
                "last_inference_ms": prev.get("last_inference_ms"),
                "last_vram_peak_mb": prev.get("last_vram_peak_mb"),
                "error_message": prev.get("error_message"),
                "updated_at": prev.get("updated_at", self._now_iso()),
            }

    def _save_manifest(self) -> None:
        payload = {
            "updated_at": self._now_iso(),
            "models": self._state,
        }
        tmp_path = self.manifest_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp_path, self.manifest_path)

    def _verify_spec_on_disk(self, spec: ModelSpec) -> tuple[bool, Optional[str], Optional[str]]:
        mdir = self._model_dir(spec)
        primary_sha: Optional[str] = None

        for fspec in spec.files:
            fpath = mdir / fspec.filename
            if not fpath.exists() or not fpath.is_file():
                return False, None, f"Missing file: {fspec.filename}"
            if fspec.filename.endswith(".json"):
                try:
                    json.loads(fpath.read_text(encoding="utf-8"))
                except Exception as exc:
                    return False, None, f"Corrupt JSON file {fspec.filename}: {exc}"
            if fspec.expected_sha256:
                actual = compute_file_sha256(fpath)
                if actual != fspec.expected_sha256:
                    return (
                        False,
                        actual,
                        f"SHA-256 mismatch on {fspec.filename} (expected {fspec.expected_sha256[:12]}..., got {actual[:12]}...).",
                    )
                if fspec.filename == spec.primary_weight_file:
                    primary_sha = actual

        if primary_sha is None:
            primary_path = mdir / spec.primary_weight_file
            if primary_path.exists():
                primary_sha = compute_file_sha256(primary_path)

        return True, primary_sha, None

    def verify_existing_models(self, keys: Optional[list[str]] = None) -> None:
        with self._lock:
            selected_keys = keys if keys is not None else list(MODEL_CATALOG)
            for key in selected_keys:
                spec = MODEL_CATALOG[key]
                ok, sha, err = self._verify_spec_on_disk(spec)
                entry = self._state[key]
                if ok:
                    entry["verified"] = True
                    entry["actual_sha256"] = sha
                    entry["error_message"] = None
                    if entry["status"] not in {"loaded"}:
                        entry["status"] = "verified"
                else:
                    entry["verified"] = False
                    entry["loaded_in_memory"] = False
                    mdir = self._model_dir(spec)
                    if mdir.exists() and any(mdir.iterdir()):
                        entry["status"] = "error" if sha else "not_downloaded"
                        entry["error_message"] = err
                    else:
                        entry["status"] = "not_downloaded"
                        entry["error_message"] = None
                    entry["actual_sha256"] = sha
                entry["updated_at"] = self._now_iso()
            self._save_manifest()

    def mark_loaded(self, key: str, loaded: bool, error_message: Optional[str] = None) -> None:
        with self._lock:
            entry = self._state[key]
            entry["loaded_in_memory"] = loaded
            if error_message:
                entry["status"] = "error"
                entry["error_message"] = error_message
            elif loaded:
                entry["status"] = "loaded"
                entry["error_message"] = None
            elif entry["verified"]:
                entry["status"] = "verified"
            entry["updated_at"] = self._now_iso()
            self._save_manifest()

    def mark_error(self, key: str, error_message: str) -> None:
        """Persist a failed model lifecycle state, including worker-level errors."""
        with self._lock:
            entry = self._state[key]
            entry["loaded_in_memory"] = False
            entry["status"] = "error"
            entry["error_message"] = error_message[:500]
            entry["updated_at"] = self._now_iso()
            self._save_manifest()

    def record_runtime_metrics(
        self,
        key: str,
        device: str,
        inference_ms: Optional[int],
        vram_peak_mb: Optional[float],
    ) -> None:
        with self._lock:
            entry = self._state[key]
            entry["active_device"] = device
            if inference_ms is not None:
                entry["last_inference_ms"] = inference_ms
            if vram_peak_mb is not None:
                entry["last_vram_peak_mb"] = vram_peak_mb
            entry["updated_at"] = self._now_iso()
            self._save_manifest()

    def ensure_model_downloaded(self, key: str, force_verify: bool = True) -> ModelManifestEntry:
        spec = MODEL_CATALOG[key]
        logger.info("Checking model checkpoint key=%s path=%s", key, self._model_dir(spec))
        if force_verify:
            ok, sha, _ = self._verify_spec_on_disk(spec)
            if ok:
                with self._lock:
                    self._state[key]["verified"] = True
                    self._state[key]["actual_sha256"] = sha
                    self._state[key]["error_message"] = None
                    if self._state[key]["status"] != "loaded":
                        self._state[key]["status"] = "verified"
                    self._state[key]["updated_at"] = self._now_iso()
                    self._save_manifest()
                logger.info("Model checkpoint already verified key=%s sha256=%s", key, sha)
                return self.get_entry(key)

        logger.info("Starting model download key=%s repo=%s revision=%s", key, spec.repo_id, spec.revision)
        with self._lock:
            self._state[key]["status"] = "downloading"
            self._state[key]["error_message"] = None
            self._state[key]["updated_at"] = self._now_iso()
            self._save_manifest()

        mdir = self._model_dir(spec)
        mdir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            logger.exception("Cannot download model key=%s: huggingface_hub is unavailable", key)
            with self._lock:
                self._state[key]["verified"] = False
                self._state[key]["status"] = "error"
                self._state[key]["error_message"] = (
                    f"huggingface_hub is not installed: {exc}"
                )
                self._state[key]["updated_at"] = self._now_iso()
                self._save_manifest()
            return self.get_entry(key)

        try:
            for fspec in spec.files:
                repo_id = fspec.repo_id or spec.repo_id
                revision = fspec.revision or spec.revision
                remote_name = fspec.remote_filename or fspec.filename
                target_path = mdir / fspec.filename
                target_path.parent.mkdir(parents=True, exist_ok=True)

                if remote_name == fspec.filename:
                    hf_hub_download(
                        repo_id=repo_id,
                        revision=revision,
                        filename=remote_name,
                        local_dir=str(mdir),
                    )
                else:
                    cached = hf_hub_download(
                        repo_id=repo_id,
                        revision=revision,
                        filename=remote_name,
                    )
                    shutil.copy2(cached, target_path)

            ok, sha, err = self._verify_spec_on_disk(spec)
            with self._lock:
                if ok:
                    self._state[key]["verified"] = True
                    self._state[key]["actual_sha256"] = sha
                    self._state[key]["status"] = "verified"
                    self._state[key]["error_message"] = None
                    logger.info("Model download and SHA-256 verification succeeded key=%s sha256=%s", key, sha)
                else:
                    primary_path = mdir / spec.primary_weight_file
                    if primary_path.exists() and sha:
                        primary_path.unlink()
                    self._state[key]["verified"] = False
                    self._state[key]["actual_sha256"] = sha
                    self._state[key]["status"] = "error"
                    self._state[key]["error_message"] = err or "Verification failed."
                    logger.error("Downloaded model failed verification key=%s reason=%s", key, err)
                self._state[key]["updated_at"] = self._now_iso()
                self._save_manifest()
        except Exception as exc:
            logger.exception("Model download failed key=%s repo=%s", key, spec.repo_id)
            msg = str(exc)
            lower = msg.lower()
            if any(
                tok in lower
                for tok in ("connection", "offline", "nameresolutionerror", "timeout", "max retries")
            ):
                friendly = (
                    f"Network unavailable while downloading {spec.repo_id}. "
                    "Connect to the internet and run `python setup_models.py` to cache weights locally."
                )
            else:
                friendly = f"Download/verification failed for {spec.repo_id}: {msg[:240]}"
            with self._lock:
                self._state[key]["verified"] = False
                self._state[key]["status"] = "error"
                self._state[key]["error_message"] = friendly
                self._state[key]["updated_at"] = self._now_iso()
                self._save_manifest()

        return self.get_entry(key)

    def download_all_models(self, keys: Optional[list[str]] = None) -> ModelsStatusResponse:
        self.download_in_progress = True
        try:
            # The default setup path downloads only the primary CPU detector.
            # Other model checkpoints remain supported and can be requested
            # explicitly without being downloaded as a side effect.
            selected_keys = keys if keys is not None else ["ai_generated_detector"]
            for key in selected_keys:
                if key not in MODEL_CATALOG:
                    raise KeyError(f"Unknown model key: {key}")
                self.ensure_model_downloaded(key, force_verify=True)
        finally:
            self.download_in_progress = False
        return self.get_status()

    def get_model_dir(self, key: str) -> Path:
        return self._model_dir(MODEL_CATALOG[key])

    def get_entry(self, key: str) -> ModelManifestEntry:
        spec = MODEL_CATALOG[key]
        st = self._state[key]
        total_mb = round(sum(f.size_bytes for f in spec.files) / (1024 * 1024), 2)
        expected_sha = next(
            (f.expected_sha256 for f in spec.files if f.filename == spec.primary_weight_file),
            "",
        )
        status_val: DownloadLifecycleStatus = st["status"]
        return ModelManifestEntry(
            key=spec.key,
            name=spec.name,
            repo_id=spec.repo_id,
            revision=spec.revision,
            source_url=spec.source_url,
            license=spec.license,
            architecture=spec.architecture,
            training_dataset=spec.training_dataset,
            intended_task=spec.intended_task,
            preprocessing=spec.preprocessing,
            limitations=spec.limitations,
            status=status_val,
            verified=bool(st["verified"]),
            loaded_in_memory=bool(st["loaded_in_memory"]),
            size_mb=total_mb,
            expected_sha256=expected_sha,
            actual_sha256=st["actual_sha256"],
            local_dir=str(self._model_dir(spec)),
            active_device=st.get("active_device", "cpu"),
            last_inference_ms=st.get("last_inference_ms"),
            last_vram_peak_mb=st.get("last_vram_peak_mb"),
            error_message=st["error_message"],
            updated_at=st["updated_at"],
        )

    def get_status(self) -> ModelsStatusResponse:
        entries = [self.get_entry(k) for k in MODEL_CATALOG]
        return ModelsStatusResponse(
            models_dir=str(self.models_dir),
            manifest_path=str(self.manifest_path),
            all_verified=all(e.verified for e in entries),
            all_loaded=all(e.loaded_in_memory for e in entries),
            download_in_progress=self.download_in_progress,
            models=entries,
        )
