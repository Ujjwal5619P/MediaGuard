import csv
import io
import json
import sqlite3
from collections import defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple

from app.schemas import (
    AnalyticsResponse,
    AnalyzeImageResponse,
    DailyActivityBucket,
    DatasetEvaluationDetail,
    DatasetEvaluationSummary,
    Dinov2FeatureSummary,
    ExifSummary,
    ExperimentalAttribution,
    FileMetadata,
    ScanListResponse,
    ScanSummary,
    TaskIndicator,
    TaskStatus,
)


class ScanRepository:
    def __init__(self, db_path: Path, artifacts_dir: Path) -> None:
        self.db_path = db_path
        self.artifacts_dir = artifacts_dir

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def initialize(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    scan_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    detected_mime TEXT NOT NULL,
                    declared_mime TEXT,
                    byte_size INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    exif_json TEXT NOT NULL,
                    ai_generated_status TEXT NOT NULL,
                    ai_generated_score REAL,
                    ai_generated_label TEXT,
                    ai_generated_json TEXT NOT NULL,
                    univfd_status TEXT,
                    univfd_score REAL,
                    univfd_label TEXT,
                    univfd_json TEXT,
                    face_manipulation_status TEXT NOT NULL,
                    face_manipulation_score REAL,
                    face_manipulation_label TEXT,
                    face_manipulation_json TEXT NOT NULL,
                    dinov2_json TEXT,
                    attribution_json TEXT NOT NULL,
                    execution_device TEXT DEFAULT 'cpu',
                    vram_peak_mb REAL,
                    inference_ms INTEGER NOT NULL,
                    original_deleted INTEGER NOT NULL,
                    artifact_path TEXT,
                    gradcam_path TEXT,
                    patch_attention_path TEXT,
                    record_expires_at TEXT NOT NULL,
                    artifact_expires_at TEXT
                );
                """
            )
            existing_cols = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(scans);").fetchall()
            }
            migrations = {
                "ai_generated_label": "TEXT",
                "univfd_status": "TEXT",
                "univfd_score": "REAL",
                "univfd_label": "TEXT",
                "univfd_json": "TEXT",
                "face_manipulation_label": "TEXT",
                "dinov2_json": "TEXT",
                "execution_device": "TEXT DEFAULT 'cpu'",
                "vram_peak_mb": "REAL",
                "gradcam_path": "TEXT",
                "patch_attention_path": "TEXT",
            }
            for col, col_type in migrations.items():
                if col not in existing_cols:
                    conn.execute(f"ALTER TABLE scans ADD COLUMN {col} {col_type};")

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans(created_at DESC);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scans_sha256 ON scans(sha256);"
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    evaluation_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    dataset_dir TEXT NOT NULL,
                    split_role TEXT NOT NULL,
                    decision_threshold REAL NOT NULL,
                    include_univfd INTEGER NOT NULL,
                    detail_json TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_eval_created_at ON evaluation_runs(created_at DESC);"
            )
            conn.commit()

    def insert_scan(
        self,
        record: AnalyzeImageResponse,
        gradcam_path: Optional[str] = None,
        patch_attention_path: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> None:
        primary_artifact = artifact_path or gradcam_path or patch_attention_path
        u_ind = record.univfd_indicator
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO scans (
                    scan_id, created_at, sha256, filename, detected_mime, declared_mime,
                    byte_size, width, height, exif_json,
                    ai_generated_status, ai_generated_score, ai_generated_label, ai_generated_json,
                    univfd_status, univfd_score, univfd_label, univfd_json,
                    face_manipulation_status, face_manipulation_score, face_manipulation_label, face_manipulation_json,
                    dinov2_json, attribution_json, execution_device, vram_peak_mb,
                    inference_ms, original_deleted, artifact_path, gradcam_path, patch_attention_path,
                    record_expires_at, artifact_expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    record.scan_id,
                    record.created_at,
                    record.sha256,
                    record.file_metadata.filename,
                    record.file_metadata.detected_mime,
                    record.file_metadata.declared_mime,
                    record.file_metadata.byte_size,
                    record.file_metadata.width,
                    record.file_metadata.height,
                    record.file_metadata.exif_summary.model_dump_json(),
                    record.ai_generated_indicator.status,
                    record.ai_generated_indicator.score,
                    record.ai_generated_indicator.prediction_label,
                    record.ai_generated_indicator.model_dump_json(),
                    u_ind.status if u_ind else None,
                    u_ind.score if u_ind else None,
                    u_ind.prediction_label if u_ind else None,
                    u_ind.model_dump_json() if u_ind else None,
                    record.face_manipulation_indicator.status,
                    record.face_manipulation_indicator.score,
                    record.face_manipulation_indicator.prediction_label,
                    record.face_manipulation_indicator.model_dump_json(),
                    record.dinov2_features.model_dump_json(),
                    record.experimental_attribution.model_dump_json(),
                    record.execution_device,
                    record.vram_peak_mb,
                    record.inference_ms,
                    1 if record.original_deleted else 0,
                    primary_artifact,
                    gradcam_path,
                    patch_attention_path,
                    record.record_expires_at,
                    record.artifact_expires_at,
                ),
            )
            conn.commit()

    @staticmethod
    def _extract_model_id(raw_json: Optional[str]) -> Optional[str]:
        if not raw_json:
            return None
        try:
            data = json.loads(raw_json)
            return data.get("model_name") or data.get("model_id")
        except Exception:
            return None

    def _row_to_summary(self, row: sqlite3.Row) -> ScanSummary:
        has_any_art = bool(
            row["artifact_path"] or row["gradcam_path"] or row["patch_attention_path"]
        )
        return ScanSummary(
            scan_id=row["scan_id"],
            created_at=row["created_at"],
            sha256=row["sha256"],
            filename=row["filename"],
            detected_mime=row["detected_mime"],
            byte_size=row["byte_size"],
            width=row["width"],
            height=row["height"],
            ai_generated_status=row["ai_generated_status"],
            ai_generated_score=row["ai_generated_score"],
            ai_generated_label=row["ai_generated_label"],
            ai_model_id=self._extract_model_id(row["ai_generated_json"]),
            univfd_status=row["univfd_status"],
            univfd_score=row["univfd_score"],
            univfd_label=row["univfd_label"],
            univfd_model_id=self._extract_model_id(row["univfd_json"]),
            face_manipulation_status=row["face_manipulation_status"],
            face_manipulation_score=row["face_manipulation_score"],
            face_manipulation_label=row["face_manipulation_label"],
            face_model_id=self._extract_model_id(row["face_manipulation_json"]),
            execution_device=row["execution_device"],
            inference_ms=row["inference_ms"],
            has_artifact=has_any_art,
            original_deleted=bool(row["original_deleted"]),
            record_expires_at=row["record_expires_at"],
            artifact_expires_at=row["artifact_expires_at"],
        )

    def _row_to_full(self, row: sqlite3.Row) -> AnalyzeImageResponse:
        exif = ExifSummary.model_validate_json(row["exif_json"])
        ai_ind = TaskIndicator.model_validate_json(row["ai_generated_json"])
        univfd_ind = (
            TaskIndicator.model_validate_json(row["univfd_json"])
            if row["univfd_json"]
            else None
        )
        face_ind = TaskIndicator.model_validate_json(row["face_manipulation_json"])
        attribution = ExperimentalAttribution.model_validate_json(row["attribution_json"])

        if row["dinov2_json"]:
            dinov2 = Dinov2FeatureSummary.model_validate_json(row["dinov2_json"])
        else:
            dinov2 = Dinov2FeatureSummary(
                status="unavailable",
                model_id="facebook/dinov2-small",
                reason="Archived Milestone 1 scan without DINOv2 feature extraction.",
            )

        if not (row["artifact_path"] or row["gradcam_path"] or row["patch_attention_path"]):
            attribution.available = False
            attribution.gradcam_available = False
            attribution.gradcam_url = None
            attribution.patch_attention_available = False
            attribution.patch_attention_url = None
            attribution.artifact_url = None

        return AnalyzeImageResponse(
            scan_id=row["scan_id"],
            created_at=row["created_at"],
            sha256=row["sha256"],
            original_deleted=bool(row["original_deleted"]),
            inference_ms=row["inference_ms"],
            execution_device=row["execution_device"] or "cpu",
            vram_peak_mb=row["vram_peak_mb"],
            file_metadata=FileMetadata(
                filename=row["filename"],
                detected_mime=row["detected_mime"],
                declared_mime=row["declared_mime"],
                byte_size=row["byte_size"],
                width=row["width"],
                height=row["height"],
                exif_summary=exif,
            ),
            dinov2_features=dinov2,
            ai_generated_indicator=ai_ind,
            univfd_indicator=univfd_ind,
            face_manipulation_indicator=face_ind,
            experimental_attribution=attribution,
            record_expires_at=row["record_expires_at"],
            artifact_expires_at=row["artifact_expires_at"],
        )

    def list_scans(
        self,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
        status: Optional[TaskStatus] = None,
    ) -> ScanListResponse:
        clauses: list[str] = []
        params: list[Any] = []

        if search and search.strip():
            pattern = f"%{search.strip()}%"
            clauses.append("(filename LIKE ? OR sha256 LIKE ?)")
            params.extend([pattern, pattern])

        if status:
            clauses.append(
                "(ai_generated_status = ? OR face_manipulation_status = ?)"
            )
            params.extend([status, status])

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        with closing(self._connect()) as conn:
            total = conn.execute(
                f"SELECT COUNT(*) FROM scans {where_sql};", params
            ).fetchone()[0]
            rows = conn.execute(
                f"""
                SELECT * FROM scans
                {where_sql}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?;
                """,
                [*params, limit, offset],
            ).fetchall()

        return ScanListResponse(
            items=[self._row_to_summary(r) for r in rows],
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_analytics(self) -> AnalyticsResponse:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT created_at, detected_mime, ai_generated_status, ai_generated_score,
                       univfd_status, univfd_score,
                       face_manipulation_status, face_manipulation_score,
                       execution_device, vram_peak_mb, inference_ms,
                       original_deleted, artifact_path, gradcam_path, patch_attention_path
                FROM scans
                ORDER BY created_at ASC;
                """
            ).fetchall()

        total_scans = len(rows)
        ai_evaluated = 0
        ai_flagged = 0
        ai_natural = 0
        univfd_evaluated = 0
        univfd_flagged = 0
        face_evaluated = 0
        face_flagged = 0
        face_no_face = 0
        artifacts_retained = 0
        originals_deleted = 0

        latencies: list[int] = []
        vrams: list[float] = []
        bins = [0, 0, 0, 0, 0]
        device_breakdown: dict[str, int] = defaultdict(int)
        mime_breakdown: dict[str, int] = defaultdict(int)
        daily_map: dict[str, dict[str, int]] = {}

        for r in rows:
            day = str(r["created_at"])[:10]
            if day not in daily_map:
                daily_map[day] = {"total": 0, "ai_flagged": 0, "face_flagged": 0}
            daily_map[day]["total"] += 1

            if r["ai_generated_status"] == "evaluated":
                ai_evaluated += 1
                s = r["ai_generated_score"]
                if s is not None:
                    if s >= 0.5:
                        ai_flagged += 1
                        daily_map[day]["ai_flagged"] += 1
                    else:
                        ai_natural += 1
                    idx = min(4, max(0, int(s * 5)))
                    bins[idx] += 1

            if r["univfd_status"] == "evaluated":
                univfd_evaluated += 1
                us = r["univfd_score"]
                if us is not None and us >= 0.5:
                    univfd_flagged += 1

            if r["face_manipulation_status"] == "evaluated":
                face_evaluated += 1
                fs = r["face_manipulation_score"]
                if fs is not None and fs >= 0.5:
                    face_flagged += 1
                    daily_map[day]["face_flagged"] += 1
            elif r["face_manipulation_status"] == "no_face_detected":
                face_no_face += 1

            if r["artifact_path"] or r["gradcam_path"] or r["patch_attention_path"]:
                artifacts_retained += 1
            if r["original_deleted"]:
                originals_deleted += 1

            if r["inference_ms"] is not None:
                latencies.append(int(r["inference_ms"]))
            if r["vram_peak_mb"] is not None:
                vrams.append(float(r["vram_peak_mb"]))

            dev = r["execution_device"] or "unknown"
            device_breakdown[dev] += 1
            mime = r["detected_mime"] or "unknown"
            mime_breakdown[mime] += 1

        daily_activity = [
            DailyActivityBucket(
                date=d,
                total=vals["total"],
                ai_flagged=vals["ai_flagged"],
                face_flagged=vals["face_flagged"],
            )
            for d, vals in sorted(daily_map.items())[-14:]
        ]

        return AnalyticsResponse(
            total_scans=total_scans,
            ai_evaluated_count=ai_evaluated,
            ai_flagged_count=ai_flagged,
            ai_natural_count=ai_natural,
            univfd_evaluated_count=univfd_evaluated,
            univfd_flagged_count=univfd_flagged,
            face_evaluated_count=face_evaluated,
            face_flagged_count=face_flagged,
            face_no_face_count=face_no_face,
            artifacts_retained_count=artifacts_retained,
            originals_deleted_count=originals_deleted,
            avg_inference_ms=round(sum(latencies) / len(latencies), 1) if latencies else None,
            min_inference_ms=min(latencies) if latencies else None,
            max_inference_ms=max(latencies) if latencies else None,
            avg_vram_peak_mb=round(sum(vrams) / len(vrams), 2) if vrams else None,
            max_vram_peak_mb=round(max(vrams), 2) if vrams else None,
            ai_score_buckets=bins,
            daily_activity=daily_activity,
            device_breakdown=dict(device_breakdown),
            mime_breakdown=dict(mime_breakdown),
        )

    def get_scan(self, scan_id: str) -> Optional[AnalyzeImageResponse]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT * FROM scans WHERE scan_id = ?;", (scan_id,)
            ).fetchone()
        if not row:
            return None
        return self._row_to_full(row)

    @staticmethod
    def _unlink_paths(row: sqlite3.Row) -> bool:
        deleted_any = False
        for col in ("artifact_path", "gradcam_path", "patch_attention_path"):
            raw = row[col]
            if raw:
                p = Path(raw)
                if p.exists():
                    p.unlink()
                deleted_any = True
        return deleted_any

    def delete_artifact(self, scan_id: str) -> Tuple[bool, bool]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT artifact_path, gradcam_path, patch_attention_path FROM scans WHERE scan_id = ?;",
                (scan_id,),
            ).fetchone()
            if not row:
                return False, False

            artifact_deleted = self._unlink_paths(row)

            attribution = ExperimentalAttribution(
                available=False,
                gradcam_available=False,
                gradcam_url=None,
                patch_attention_available=False,
                patch_attention_url=None,
                method=None,
                artifact_url=None,
                reason="Experimental attribution artifact was purged under retention policy.",
            )
            conn.execute(
                """
                UPDATE scans
                SET artifact_path = NULL,
                    gradcam_path = NULL,
                    patch_attention_path = NULL,
                    artifact_expires_at = NULL,
                    attribution_json = ?
                WHERE scan_id = ?;
                """,
                (attribution.model_dump_json(), scan_id),
            )
            conn.commit()
            return True, artifact_deleted

    def delete_scan(self, scan_id: str) -> Tuple[bool, bool]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT artifact_path, gradcam_path, patch_attention_path FROM scans WHERE scan_id = ?;",
                (scan_id,),
            ).fetchone()
            if not row:
                return False, False

            artifact_deleted = self._unlink_paths(row)
            conn.execute("DELETE FROM scans WHERE scan_id = ?;", (scan_id,))
            conn.commit()
            return True, artifact_deleted

    def purge_expired(self, now_utc: Optional[datetime] = None) -> Tuple[int, int]:
        now_iso = (now_utc or datetime.now(timezone.utc)).isoformat()
        expired_artifacts = 0
        expired_records = 0

        expired_attr_json = ExperimentalAttribution(
            available=False,
            gradcam_available=False,
            gradcam_url=None,
            patch_attention_available=False,
            patch_attention_url=None,
            method=None,
            artifact_url=None,
            reason="Experimental attribution artifact expired under retention policy.",
        ).model_dump_json()

        with closing(self._connect()) as conn:
            art_rows = conn.execute(
                """
                SELECT scan_id, artifact_path, gradcam_path, patch_attention_path FROM scans
                WHERE (artifact_path IS NOT NULL OR gradcam_path IS NOT NULL OR patch_attention_path IS NOT NULL)
                  AND artifact_expires_at IS NOT NULL
                  AND artifact_expires_at <= ?;
                """,
                (now_iso,),
            ).fetchall()

            for row in art_rows:
                self._unlink_paths(row)
                conn.execute(
                    """
                    UPDATE scans
                    SET artifact_path = NULL,
                        gradcam_path = NULL,
                        patch_attention_path = NULL,
                        artifact_expires_at = NULL,
                        attribution_json = ?
                    WHERE scan_id = ?;
                    """,
                    (expired_attr_json, row["scan_id"]),
                )
                expired_artifacts += 1

            rec_rows = conn.execute(
                "SELECT scan_id, artifact_path, gradcam_path, patch_attention_path FROM scans WHERE record_expires_at <= ?;",
                (now_iso,),
            ).fetchall()
            for row in rec_rows:
                self._unlink_paths(row)
                conn.execute("DELETE FROM scans WHERE scan_id = ?;", (row["scan_id"],))
                expired_records += 1

            conn.commit()

        return expired_records, expired_artifacts

    def export_scans(
        self,
        fmt: Literal["csv", "json"],
        status: Optional[TaskStatus] = None,
    ) -> str:
        resp = self.list_scans(limit=1000, offset=0, status=status)
        if fmt == "json":
            return json.dumps([item.model_dump() for item in resp.items], indent=2)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "scan_id",
                "created_at",
                "sha256",
                "filename",
                "detected_mime",
                "byte_size",
                "width",
                "height",
                "ai_generated_status",
                "ai_generated_score",
                "ai_generated_label",
                "ai_model_id",
                "univfd_status",
                "univfd_score",
                "univfd_label",
                "face_manipulation_status",
                "face_manipulation_score",
                "face_manipulation_label",
                "execution_device",
                "inference_ms",
                "has_artifact",
                "original_deleted",
                "record_expires_at",
                "artifact_expires_at",
            ]
        )
        for item in resp.items:
            writer.writerow(
                [
                    item.scan_id,
                    item.created_at,
                    item.sha256,
                    item.filename,
                    item.detected_mime,
                    item.byte_size,
                    item.width,
                    item.height,
                    item.ai_generated_status,
                    item.ai_generated_score if item.ai_generated_score is not None else "",
                    item.ai_generated_label or "",
                    item.ai_model_id or "",
                    item.univfd_status or "",
                    item.univfd_score if item.univfd_score is not None else "",
                    item.univfd_label or "",
                    item.face_manipulation_status,
                    item.face_manipulation_score if item.face_manipulation_score is not None else "",
                    item.face_manipulation_label or "",
                    item.execution_device or "",
                    item.inference_ms if item.inference_ms is not None else "",
                    item.has_artifact,
                    item.original_deleted,
                    item.record_expires_at,
                    item.artifact_expires_at or "",
                ]
            )
        return buf.getvalue()

    # --- Dataset Evaluation Persistence & Export ---

    def insert_evaluation(self, detail: DatasetEvaluationDetail) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO evaluation_runs (
                    evaluation_id, created_at, dataset_dir, split_role,
                    decision_threshold, include_univfd, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    detail.evaluation_id,
                    detail.created_at,
                    detail.dataset_dir,
                    detail.split_role,
                    detail.decision_threshold,
                    1 if detail.include_univfd else 0,
                    detail.model_dump_json(),
                ),
            )
            conn.commit()

    def list_evaluations(self, limit: int = 20) -> List[DatasetEvaluationSummary]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT detail_json FROM evaluation_runs ORDER BY created_at DESC LIMIT ?;",
                (limit,),
            ).fetchall()
        summaries: List[DatasetEvaluationSummary] = []
        for row in rows:
            detail = DatasetEvaluationDetail.model_validate_json(row["detail_json"])
            data = detail.model_dump(exclude={"items"})
            summaries.append(DatasetEvaluationSummary(**data))
        return summaries

    def get_evaluation(self, evaluation_id: str) -> Optional[DatasetEvaluationDetail]:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT detail_json FROM evaluation_runs WHERE evaluation_id = ?;",
                (evaluation_id,),
            ).fetchone()
        if not row:
            return None
        return DatasetEvaluationDetail.model_validate_json(row["detail_json"])

    def export_evaluation(
        self, evaluation_id: str, fmt: Literal["csv", "json"]
    ) -> Optional[str]:
        detail = self.get_evaluation(evaluation_id)
        if not detail:
            return None
        if fmt == "json":
            return detail.model_dump_json(indent=2)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            [
                "evaluation_id",
                "split_role",
                "decision_threshold",
                "filename",
                "relative_path",
                "ground_truth",
                "smaug_status",
                "smaug_score",
                "smaug_prediction",
                "smaug_correct",
                "smaug_inference_ms",
                "univfd_status",
                "univfd_score",
                "univfd_prediction",
                "univfd_correct",
                "univfd_inference_ms",
                "execution_device",
                "error_message",
            ]
        )
        for item in detail.items:
            writer.writerow(
                [
                    detail.evaluation_id,
                    detail.split_role,
                    detail.decision_threshold,
                    item.filename,
                    item.relative_path,
                    item.ground_truth,
                    item.smaug_status,
                    item.smaug_score if item.smaug_score is not None else "",
                    item.smaug_prediction or "",
                    item.smaug_correct if item.smaug_correct is not None else "",
                    item.smaug_inference_ms if item.smaug_inference_ms is not None else "",
                    item.univfd_status or "",
                    item.univfd_score if item.univfd_score is not None else "",
                    item.univfd_prediction or "",
                    item.univfd_correct if item.univfd_correct is not None else "",
                    item.univfd_inference_ms if item.univfd_inference_ms is not None else "",
                    item.execution_device,
                    item.error_message or "",
                ]
            )
        return buf.getvalue()


def compute_expiry_iso(days: int, base_time: Optional[datetime] = None) -> str:
    dt = (base_time or datetime.now(timezone.utc)) + timedelta(days=days)
    return dt.isoformat()
