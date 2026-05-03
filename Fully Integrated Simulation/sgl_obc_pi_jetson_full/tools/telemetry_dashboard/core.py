from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import os
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class MetricSpec:
    group: str
    label: str
    candidates: Tuple[str, ...]


GROUP_ORDER = [
    "Power/EPS",
    "ADCS",
    "Thermal",
    "Propulsion",
    "Payload",
    "COMMS",
    "Jetson/Processing",
    "Other",
]

METRIC_SPECS: Sequence[MetricSpec] = [
    MetricSpec("Power/EPS", "Source Power (W)", ("source_w",)),
    MetricSpec("Power/EPS", "Noncompute Load (W)", ("noncompute_w",)),
    MetricSpec("Power/EPS", "Compute Budget (W)", ("compute_budget_w",)),
    MetricSpec("Power/EPS", "Total Bus Load (W)", ("total_bus_load_w",)),
    MetricSpec("Power/EPS", "Scheduler Mode", ("scheduler_mode",)),
    MetricSpec("ADCS", "ADCS Power (W)", ("adcs_power_w",)),
    MetricSpec("ADCS", "Wheel Power (W)", ("wheel_power_w",)),
    MetricSpec("ADCS", "Truth Pointing Error (deg)", ("truth_pointing_err_deg",)),
    MetricSpec("ADCS", "Estimated Pointing Error (deg)", ("est_pointing_err_deg",)),
    MetricSpec("ADCS", "Tracker Confidence", ("tracker_conf",)),
    MetricSpec("ADCS", "Tracker Valid", ("tracker_valid",)),
    MetricSpec("ADCS", "Tracked Stars", ("tracked_stars",)),
    MetricSpec("Thermal", "Thermal Power (W)", ("thermal_power_w",)),
    MetricSpec("Thermal", "Heater Active", ("heater_active",)),
    MetricSpec("Thermal", "Thermal Temp (C)", ("thermal_temp_c",)),
    MetricSpec("Propulsion", "Propulsion Power (W)", ("propulsion_power_w",)),
    MetricSpec("Propulsion", "Propulsion Active", ("propulsion_active",)),
    MetricSpec("Propulsion", "Propulsion Thrust (N)", ("propulsion_thrust_n",)),
    MetricSpec("Payload", "Payload Power (W)", ("payload_power_w",)),
    MetricSpec("Payload", "Payload Active", ("payload_active",)),
    MetricSpec("Payload", "Dataset Ready", ("dataset_ready",)),
    MetricSpec("Payload", "Dataset Count", ("dataset_count",)),
    MetricSpec("Payload", "Acquisition Stage", ("acquisition_stage",)),
    MetricSpec("Payload", "Camera Frame Ready", ("camera_frame_ready",)),
    MetricSpec("Payload", "Alignment Valid", ("alignment_valid",)),
    MetricSpec("Payload", "Alignment Score", ("alignment_score",)),
    MetricSpec("Payload", "Blur Score", ("blur_score",)),
    MetricSpec("Payload", "Brightness Mean", ("brightness_mean",)),
    MetricSpec("Payload", "Contrast Score", ("contrast_score",)),
    MetricSpec("COMMS", "COMMS Power (W)", ("comms_power_w",)),
    MetricSpec("COMMS", "Downlink Queue (bits)", ("comms_backlog_bits", "downlink_queue", "downlink_queue_bits", "downlink_backlog_bits")),
    MetricSpec("Jetson/Processing", "Jetson Power (W)", ("jetson_power_w",)),
    MetricSpec("Jetson/Processing", "ROI Count", ("roi_count",)),
    MetricSpec("Jetson/Processing", "Processing Queue", ("processing_queue_depth", "processing_queue")),
]

STATUS_CANDIDATES: Sequence[Tuple[str, str]] = [
    ("adcs_mode", "adcs_mode"),
    ("thermal_mode", "thermal_mode"),
    ("propulsion_mode", "propulsion_mode"),
    ("payload_mode", "payload_mode"),
    ("camera_mode", "camera_mode"),
    ("raw_capture_path", "raw_capture_path"),
    ("rectified_image_path", "rectified_image_path"),
    ("dataset_id", "dataset_id"),
    ("comms_mode", "comms_mode"),
    ("jetson_mode", "jetson_mode"),
    ("jetson_job_type", "jetson_job_type"),
]


@dataclass
class MetricPlan:
    grouped_metrics: Dict[str, List[Tuple[str, str]]]
    status_fields: List[Tuple[str, str]]


@dataclass(frozen=True)
class EventRecord:
    cycle: int
    event_type: str
    severity: str
    message: str
    value: str


@dataclass(frozen=True)
class ImagePreviewPaths:
    raw_capture: str | None
    rectified: str | None
    ring_preview: str | None
    coarse: str | None
    refined: str | None


@dataclass(frozen=True)
class RunSummary:
    run_dir: str
    run_name: str
    run_status: str
    completion_reason: str
    requested_stages: Tuple[int, ...]
    completed_stages: Tuple[int, ...]
    missing_required_outputs: Tuple[str, ...]
    missing_optional_outputs: Tuple[str, ...]



def _resolve(candidates: Iterable[str], fields: set[str]) -> str | None:
    for c in candidates:
        if c in fields:
            return c
    return None



def detect_metric_plan(fieldnames: Sequence[str]) -> MetricPlan:
    fields = set(fieldnames)
    grouped: Dict[str, List[Tuple[str, str]]] = {g: [] for g in GROUP_ORDER}
    used = {"cycle"}

    for spec in METRIC_SPECS:
        col = _resolve(spec.candidates, fields)
        if col is None:
            continue
        grouped[spec.group].append((col, spec.label))
        used.add(col)

    status_fields: List[Tuple[str, str]] = []
    for col, label in STATUS_CANDIDATES:
        if col in fields:
            status_fields.append((col, label))
            used.add(col)

    # Gracefully surface new appended numeric-like columns.
    for col in fieldnames:
        if col in used:
            continue
        grouped["Other"].append((col, col))

    grouped = {k: v for k, v in grouped.items() if v}
    return MetricPlan(grouped_metrics=grouped, status_fields=status_fields)


def detect_default_event_path(telemetry_path: str) -> str:
    root = os.path.dirname(os.path.abspath(telemetry_path))
    return os.path.join(root, "events.csv")


def detect_default_manifest_path(telemetry_path: str) -> str:
    root = os.path.dirname(os.path.abspath(telemetry_path))
    return os.path.join(root, "products_manifest.csv")


def _clean(v: object) -> str:
    return str(v if v is not None else "").strip().strip('"')


def resolve_image_path(path_value: object, telemetry_path: str) -> str | None:
    p = _clean(path_value)
    if not p:
        return None
    if os.path.isabs(p):
        return p if os.path.exists(p) else None

    telemetry_abs = os.path.abspath(telemetry_path)
    telemetry_dir = os.path.dirname(telemetry_abs)
    out_root = os.path.dirname(telemetry_dir)
    candidates = [
        os.path.abspath(os.path.join(os.getcwd(), p)),
        os.path.abspath(os.path.join(telemetry_dir, p)),
        os.path.abspath(os.path.join(out_root, p)),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _latest_manifest_products(manifest_path: str, telemetry_path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not manifest_path or not os.path.exists(manifest_path):
        return out
    try:
        with open(manifest_path, "r", newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return out

    for r in reversed(rows):
        kind = _clean(r.get("kind", "")).lower()
        p = resolve_image_path(r.get("path", ""), telemetry_path)
        if not p:
            continue
        if kind in ("coarse", "recon_base", "recon_upscaled") and "coarse" not in out:
            out["coarse"] = p
        elif kind in ("refined", "recon_refined") and "refined" not in out:
            out["refined"] = p
        if "coarse" in out and "refined" in out:
            break
    return out


def _detect_ring_preview(row: Mapping[str, object], telemetry_path: str) -> str | None:
    dataset_id = _clean(row.get("dataset_id", ""))
    if not dataset_id:
        return None
    telemetry_abs = os.path.abspath(telemetry_path)
    out_root = os.path.dirname(os.path.dirname(telemetry_abs))
    base = os.path.join(out_root, "datasets", dataset_id)
    candidates = [
        os.path.join(base, "ring_preview.ppm"),
        os.path.join(base, "ring_unwrapped_preview.ppm"),
        os.path.join(base, "ring_detect_overlay.ppm"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def _latest_nonempty_path(
    telemetry_rows: Sequence[Mapping[str, object]] | None,
    field: str,
    telemetry_path: str,
) -> str | None:
    if not telemetry_rows:
        return None
    for r in reversed(telemetry_rows):
        resolved = resolve_image_path(r.get(field, ""), telemetry_path)
        if resolved:
            return resolved
    return None


def discover_image_preview_paths(
    telemetry_tail_row: Mapping[str, object] | None,
    telemetry_path: str,
    manifest_path: str | None = None,
    telemetry_rows: Sequence[Mapping[str, object]] | None = None,
) -> ImagePreviewPaths:
    row = telemetry_tail_row or {}
    resolved_raw = resolve_image_path(row.get("raw_capture_path", ""), telemetry_path) or _latest_nonempty_path(
        telemetry_rows, "raw_capture_path", telemetry_path
    )
    resolved_rect = resolve_image_path(row.get("rectified_image_path", ""), telemetry_path) or _latest_nonempty_path(
        telemetry_rows, "rectified_image_path", telemetry_path
    )
    resolved_ring = _detect_ring_preview(row, telemetry_path)
    products = _latest_manifest_products(
        manifest_path or detect_default_manifest_path(telemetry_path),
        telemetry_path,
    )
    return ImagePreviewPaths(
        raw_capture=resolved_raw,
        rectified=resolved_rect,
        ring_preview=resolved_ring,
        coarse=products.get("coarse"),
        refined=products.get("refined"),
    )


def load_event_records(event_path: str, max_events: int = 500) -> List[EventRecord]:
    if not event_path or not os.path.exists(event_path):
        return []
    try:
        with open(event_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return []

    out: List[EventRecord] = []
    for r in rows[-max_events:]:
        try:
            cyc = int(float(str(r.get("cycle", "0")).strip().strip('"')))
        except Exception:
            cyc = 0
        out.append(
            EventRecord(
                cycle=cyc,
                event_type=str(r.get("event_type", "")).strip().strip('"'),
                severity=str(r.get("severity", "")).strip().strip('"'),
                message=str(r.get("message", "")).strip().strip('"'),
                value=str(r.get("value", "")).strip().strip('"'),
            )
        )
    return out


def resolve_run_paths(run_or_telemetry: str) -> Dict[str, str]:
    p = os.path.abspath(run_or_telemetry)
    if os.path.isdir(p):
        csv_dir = os.path.join(p, "csv")
        if os.path.isdir(csv_dir):
            return {
                "run_dir": p,
                "telemetry": os.path.join(csv_dir, "telemetry_cycles.csv"),
                "events": os.path.join(csv_dir, "events.csv"),
                "manifest": os.path.join(csv_dir, "products_manifest.csv"),
                "downlink": os.path.join(csv_dir, "downlink_queue.csv"),
                "quality": os.path.join(csv_dir, "reconstruction_quality.csv"),
                "stage_timings": os.path.join(csv_dir, "progressive_stage_timings.csv"),
                "run_metadata": os.path.join(p, "run_metadata.json"),
            }
        ms_dir = os.path.join(p, "mission_store")
        if os.path.isdir(ms_dir):
            return {
                "run_dir": p,
                "telemetry": os.path.join(ms_dir, "telemetry_cycles.csv"),
                "events": os.path.join(ms_dir, "events.csv"),
                "manifest": os.path.join(ms_dir, "products_manifest.csv"),
                "downlink": os.path.join(ms_dir, "downlink_queue.csv"),
                "quality": os.path.join(ms_dir, "reconstruction_quality.csv"),
                "stage_timings": os.path.join(ms_dir, "progressive_stage_timings.csv"),
                "run_metadata": os.path.join(p, "run_metadata.json"),
            }
    root = os.path.dirname(p)
    run_dir = os.path.dirname(root)
    return {
        "run_dir": run_dir,
        "telemetry": p,
        "events": os.path.join(root, "events.csv"),
        "manifest": os.path.join(root, "products_manifest.csv"),
        "downlink": os.path.join(root, "downlink_queue.csv"),
        "quality": os.path.join(root, "reconstruction_quality.csv"),
        "stage_timings": os.path.join(root, "progressive_stage_timings.csv"),
        "run_metadata": os.path.join(run_dir, "run_metadata.json"),
    }


def load_run_summary(run_metadata_path: str, run_dir: str) -> RunSummary:
    meta: Dict[str, object] = {}
    if run_metadata_path and os.path.exists(run_metadata_path):
        try:
            with open(run_metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    def _as_int_tuple(v: object) -> Tuple[int, ...]:
        if not isinstance(v, list):
            return tuple()
        out: List[int] = []
        for x in v:
            try:
                out.append(int(x))
            except Exception:
                continue
        return tuple(out)

    def _as_str_tuple(v: object) -> Tuple[str, ...]:
        if not isinstance(v, list):
            return tuple()
        return tuple(str(x) for x in v)

    return RunSummary(
        run_dir=run_dir,
        run_name=os.path.basename(run_dir.rstrip("/")) if run_dir else "",
        run_status=str(meta.get("run_completion_status", "running")),
        completion_reason=str(meta.get("completion_reason", "")),
        requested_stages=_as_int_tuple(meta.get("requested_stages")),
        completed_stages=_as_int_tuple(meta.get("completed_stages")),
        missing_required_outputs=_as_str_tuple(meta.get("missing_required_outputs")),
        missing_optional_outputs=_as_str_tuple(meta.get("missing_optional_outputs")),
    )


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def filter_events(events: Sequence[EventRecord], token: str) -> List[EventRecord]:
    t = (token or "").strip().lower()
    if not t or t == "all":
        return list(events)
    out: List[EventRecord] = []
    for e in events:
        hay = f"{e.event_type} {e.severity} {e.message}".lower()
        if t in ("warning/error", "warn", "error"):
            if e.severity.lower() in ("warn", "error"):
                out.append(e)
        elif t in hay:
            out.append(e)
    return out
