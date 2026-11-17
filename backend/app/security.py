import hashlib
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Deque, Iterator, Optional, Tuple

from fastapi import HTTPException, Request, UploadFile
from PIL import ExifTags, Image, UnidentifiedImageError

from app.config import Settings
from app.schemas import ExifSummary

ALLOWED_SIGNATURE_MIMES = {"image/jpeg", "image/png", "image/webp"}
EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def detect_magic_byte_mime(header: bytes) -> Optional[str]:
    if len(header) < 12:
        return None
    if header.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return "image/webp"
    return None


def sanitize_filename(filename: Optional[str]) -> str:
    base = Path(filename or "upload.bin").name
    cleaned = "".join(ch for ch in base if ch.isalnum() or ch in {".", "-", "_"})
    return cleaned[:120] or "upload.bin"


def extract_exif_summary(img: Image.Image) -> ExifSummary:
    try:
        exif_data = img.getexif()
        if not exif_data:
            return ExifSummary(has_exif=False)

        tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif_data.items()}

        def _clean(val: object) -> Optional[str]:
            if val is None:
                return None
            text = str(val).replace("\x00", "").strip()
            return text[:120] if text else None

        return ExifSummary(
            camera_make=_clean(tag_map.get("Make")),
            camera_model=_clean(tag_map.get("Model")),
            software=_clean(tag_map.get("Software")),
            datetime_original=_clean(
                tag_map.get("DateTimeOriginal") or tag_map.get("DateTime")
            ),
            has_exif=True,
        )
    except Exception:
        return ExifSummary(has_exif=False)


@contextmanager
def safe_temp_file(temp_dir: Path, suffix: str) -> Iterator[Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    candidate = temp_dir / f"mg_{uuid.uuid4().hex}{suffix}"
    try:
        yield candidate
    finally:
        try:
            if candidate.exists():
                candidate.unlink()
        except OSError:
            pass


async def read_validated_upload(
    upload: UploadFile, settings: Settings
) -> Tuple[bytes, str, str]:
    # Stream read with hard upper bound to prevent memory exhaustion
    chunks: list[bytes] = []
    total = 0
    chunk_size = 256 * 1024

    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "code": "ERR_FILE_TOO_LARGE",
                    "message": f"Uploaded file exceeds maximum allowed size of {settings.max_upload_bytes} bytes.",
                },
            )
        chunks.append(chunk)

    raw_bytes = b"".join(chunks)
    if len(raw_bytes) < 12:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ERR_INVALID_IMAGE_HEADER",
                "message": "Uploaded file is empty or too small to contain a valid image signature.",
            },
        )

    detected_mime = detect_magic_byte_mime(raw_bytes[:16])
    if detected_mime not in ALLOWED_SIGNATURE_MIMES:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "ERR_UNSUPPORTED_SIGNATURE",
                "message": "Invalid file signature. Only verified JPEG, PNG, and WebP images are accepted.",
            },
        )

    declared_mime = (upload.content_type or "").lower().strip()
    if declared_mime and declared_mime != "application/octet-stream":
        normalized_declared = "image/jpeg" if declared_mime == "image/jpg" else declared_mime
        if normalized_declared != detected_mime:
            raise HTTPException(
                status_code=400,
                detail={
                    "code": "ERR_MIME_SIGNATURE_MISMATCH",
                    "message": "Declared MIME type does not match actual image file signature.",
                },
            )

    safe_name = sanitize_filename(upload.filename)
    return raw_bytes, detected_mime, safe_name


def inspect_image_safely(
    raw_bytes: bytes,
    detected_mime: str,
    settings: Settings,
) -> Tuple[int, int, str, ExifSummary, bool]:
    """
    Writes validated bytes to an isolated randomized temporary file, verifies integrity
    against decompression bombs and malformed streams, extracts dimensions/EXIF, and
    guarantees deletion of the temporary file before returning.
    """
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    sha256_hex = hashlib.sha256(raw_bytes).hexdigest()
    suffix = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(detected_mime, ".img")

    temp_path_ref: Optional[Path] = None
    with safe_temp_file(settings.temp_dir, suffix) as temp_path:
        temp_path_ref = temp_path
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        fd = os.open(temp_path, flags, 0o600)
        try:
            view = memoryview(raw_bytes)
            written = 0
            while written < len(raw_bytes):
                written += os.write(fd, view[written:])
        finally:
            os.close(fd)

        try:
            with Image.open(temp_path) as img:
                img.verify()

            with Image.open(temp_path) as img:
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "ERR_MALFORMED_IMAGE",
                            "message": "Image has non-positive dimensions.",
                        },
                    )
                if width > settings.max_dimension or height > settings.max_dimension:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "ERR_DIMENSION_LIMIT_EXCEEDED",
                            "message": f"Image dimensions {width}x{height} exceed maximum allowed {settings.max_dimension}px.",
                        },
                    )
                if width * height > settings.max_image_pixels:
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "code": "ERR_PIXEL_LIMIT_EXCEEDED",
                            "message": "Image pixel count exceeds decompression safety limit.",
                        },
                    )
                img.load()
                exif_summary = extract_exif_summary(img)
        except Image.DecompressionBombError:
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "ERR_DECOMPRESSION_BOMB",
                    "message": "Image rejected by decompression-bomb protection.",
                },
            )
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "ERR_MALFORMED_IMAGE",
                    "message": "Image stream is malformed, truncated, or unreadable.",
                },
            )

    original_deleted = settings.delete_original_after_analysis and (
        temp_path_ref is None or not temp_path_ref.exists()
    )
    return width, height, sha256_hex, exif_summary, original_deleted


class SlidingWindowRateLimiter:
    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(self.window_seconds - (now - bucket[0])))
                raise HTTPException(
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                    detail={
                        "code": "ERR_RATE_LIMITED",
                        "message": "Rate limit exceeded. Please wait before submitting another request.",
                    },
                )
            bucket.append(now)

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def get_client_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"
