#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ImageRGB:
    w: int
    h: int
    rgb: bytearray  # packed RGB


@dataclass
class SupportStats:
    ok: bool
    x0: int = 0
    y0: int = 0
    x1: int = -1
    y1: int = -1
    mean_luma: float = 0.0
    mean_sat: float = 0.0
    sat_frac: float = 0.0
    support_frac: float = 0.0
    margin_px: int = 0
    margin_frac: float = 0.0

    @property
    def aspect(self) -> float:
        if not self.ok:
            return 0.0
        return (self.x1 - self.x0 + 1) / max(1.0, (self.y1 - self.y0 + 1))


def read_ppm(path: Path) -> Optional[ImageRGB]:
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data.startswith(b"P6"):
        return None
    i = 2
    n = len(data)

    def skip_ws_comments(idx: int) -> int:
        while idx < n:
            c = data[idx]
            if c == 35:  # '#'
                while idx < n and data[idx] not in (10, 13):
                    idx += 1
                continue
            if c in (9, 10, 11, 12, 13, 32):
                idx += 1
                continue
            break
        return idx

    def read_int(idx: int) -> Tuple[int, int]:
        idx = skip_ws_comments(idx)
        j = idx
        while j < n and 48 <= data[j] <= 57:
            j += 1
        if j == idx:
            return -1, idx
        return int(data[idx:j]), j

    w, i = read_int(i)
    h, i = read_int(i)
    mv, i = read_int(i)
    if w <= 0 or h <= 0 or mv != 255:
        return None
    i = skip_ws_comments(i)
    if i >= n:
        return None
    raw = data[i:]
    need = w * h * 3
    if len(raw) < need:
        return None
    return ImageRGB(w=w, h=h, rgb=bytearray(raw[:need]))


def read_image(path: Path) -> Optional[ImageRGB]:
    if not path.exists():
        return None
    if path.suffix.lower() == ".ppm":
        return read_ppm(path)
    # Optional dependency path: use ImageMagick identify/convert if available.
    try:
        out = subprocess.check_output(
            ["convert", str(path), "ppm:-"],
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    tmp = Path("/tmp/_sgl_validate_tmp.ppm")
    tmp.write_bytes(out)
    img = read_ppm(tmp)
    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass
    return img


def pix_rgb(img: ImageRGB, x: int, y: int) -> Tuple[float, float, float]:
    i = 3 * (y * img.w + x)
    return img.rgb[i] / 255.0, img.rgb[i + 1] / 255.0, img.rgb[i + 2] / 255.0


def luma(r: float, g: float, b: float) -> float:
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def sat(r: float, g: float, b: float) -> float:
    return max(r, g, b) - min(r, g, b)


def border_bg(img: ImageRGB) -> Tuple[float, float, float, float, float]:
    border = max(2, min(img.w, img.h) // 24)
    sr = sg = sb = sl = ss = 0.0
    n = 0.0
    for y in range(img.h):
        for x in range(img.w):
            if x >= border and y >= border and x + border < img.w and y + border < img.h:
                continue
            r, g, b = pix_rgb(img, x, y)
            sr += r
            sg += g
            sb += b
            sl += luma(r, g, b)
            ss += sat(r, g, b)
            n += 1.0
    if n <= 0.0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return sr / n, sg / n, sb / n, sl / n, ss / n


def support_stats(img: ImageRGB, bg_value: int = 0) -> SupportStats:
    br, bg, bb, bl, bs = border_bg(img)
    pix = img.w * img.h
    fg = bytearray(pix)
    vis = bytearray(pix)
    bgv = bg_value / 255.0
    for y in range(img.h):
        row = y * img.w
        for x in range(img.w):
            i = row + x
            r, g, b = pix_rgb(img, x, y)
            cd = math.sqrt((r - br) ** 2 + (g - bg) ** 2 + (b - bb) ** 2)
            s = sat(r, g, b)
            ll = luma(r, g, b)
            is_bg = abs(r - bgv) <= (1.0 / 255.0) and abs(g - bgv) <= (1.0 / 255.0) and abs(b - bgv) <= (1.0 / 255.0)
            keep = (not is_bg) and (((cd > max(0.012, bs + 0.003)) and s > max(0.002, bs * 0.08)) or (abs(ll - bl) > 0.010))
            fg[i] = 1 if keep else 0

    best_area = 0
    bx0 = by0 = 0
    bx1 = by1 = -1
    for y in range(img.h):
        for x in range(img.w):
            start = y * img.w + x
            if fg[start] == 0 or vis[start] != 0:
                continue
            q = deque([start])
            vis[start] = 1
            cx0 = cx1 = x
            cy0 = cy1 = y
            area = 0
            while q:
                idx = q.pop()
                area += 1
                px = idx % img.w
                py = idx // img.w
                cx0 = min(cx0, px)
                cy0 = min(cy0, py)
                cx1 = max(cx1, px)
                cy1 = max(cy1, py)
                for nx, ny in ((px - 1, py), (px + 1, py), (px, py - 1), (px, py + 1)):
                    if nx < 0 or ny < 0 or nx >= img.w or ny >= img.h:
                        continue
                    ni = ny * img.w + nx
                    if fg[ni] == 0 or vis[ni] != 0:
                        continue
                    vis[ni] = 1
                    q.append(ni)
            if area > best_area:
                best_area = area
                bx0, by0, bx1, by1 = cx0, cy0, cx1, cy1

    if best_area <= 0 or bx1 < bx0 or by1 < by0:
        return SupportStats(ok=False)

    sl = ss = n = sat_hot = 0.0
    for y in range(by0, by1 + 1):
        row = y * img.w
        for x in range(bx0, bx1 + 1):
            i = row + x
            if fg[i] == 0:
                continue
            r, g, b = pix_rgb(img, x, y)
            s = sat(r, g, b)
            sl += luma(r, g, b)
            ss += s
            sat_hot += 1.0 if s >= 0.10 else 0.0
            n += 1.0

    if n <= 0.0:
        return SupportStats(ok=False)
    margin_px = min(bx0, by0, img.w - 1 - bx1, img.h - 1 - by1)
    return SupportStats(
        ok=True,
        x0=bx0,
        y0=by0,
        x1=bx1,
        y1=by1,
        mean_luma=sl / n,
        mean_sat=ss / n,
        sat_frac=sat_hot / n,
        support_frac=n / float(pix),
        margin_px=margin_px,
        margin_frac=margin_px / float(min(img.w, img.h)),
    )


def find_path(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def overlay_red_bbox(overlay_img: Optional[ImageRGB]) -> Optional[Tuple[int, int, int, int]]:
    if overlay_img is None:
        return None
    xs: List[int] = []
    ys: List[int] = []
    for y in range(overlay_img.h):
        for x in range(overlay_img.w):
            i = 3 * (y * overlay_img.w + x)
            r = overlay_img.rgb[i]
            g = overlay_img.rgb[i + 1]
            b = overlay_img.rgb[i + 2]
            # detector bbox is drawn in red-ish pixels (255,48,48) in reconstruction.cpp
            if r >= 230 and g <= 90 and b <= 90:
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def parse_run_metadata(bundle: Path) -> Dict[str, str]:
    md_path = bundle / "run_metadata.json"
    if not md_path.exists():
        return {}
    try:
        return json.loads(md_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def latest_bundle_for(case_name: str, outputs_root: Path) -> Optional[Path]:
    matches = sorted(outputs_root.glob(f"*_{case_name}"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def summarize_case(case_name: str, source_out: Path, bundle: Optional[Path], source_image: Path) -> Dict[str, object]:
    raw_path = find_path([
        source_out / "datasets" / "dataset_0" / "raw_capture_from_file.ppm",
        source_out / "datasets" / "dataset_0" / "rectified_input.ppm",
    ])
    pre_path = find_path([
        source_out / "datasets" / "dataset_0" / "preconditioned_source.ppm",
        source_out / "datasets" / "dataset_1" / "preconditioned_source.ppm",
    ])
    src_overlay_path = find_path([
        source_out / "datasets" / "dataset_0" / "source_overlay.ppm",
        source_out / "datasets" / "dataset_1" / "source_overlay.ppm",
    ])
    final_path = find_path([
        source_out / "products" / "dataset_0_s4_refined_2048.ppm",
        source_out / "products" / "dataset_0_s3_refined_1024.ppm",
    ])
    sheet_path = find_path([
        source_out / "products" / "reconstruction_contact_sheet.ppm",
        bundle / "images" / "products" / "reconstruction_contact_sheet.png" if bundle else Path("__missing__"),
    ])

    raw_img = read_image(raw_path) if raw_path else None
    pre_img = read_image(pre_path) if pre_path else None
    final_img = read_image(final_path) if final_path else None
    src_overlay_img = read_image(src_overlay_path) if src_overlay_path else None

    raw_stats = support_stats(raw_img) if raw_img else SupportStats(ok=False)
    pre_stats = support_stats(pre_img) if pre_img else SupportStats(ok=False)
    final_stats = support_stats(final_img) if final_img else SupportStats(ok=False)
    raw_overlay_bbox = overlay_red_bbox(src_overlay_img)
    if raw_overlay_bbox is not None:
        x0, y0, x1, y1 = raw_overlay_bbox
        raw_overlay_aspect = (x1 - x0 + 1) / max(1.0, float(y1 - y0 + 1))
    else:
        raw_overlay_aspect = raw_stats.aspect

    manifest = source_out / "mission_store" / "products_manifest.csv"
    stages_done = set()
    if manifest.exists():
        with manifest.open("r", encoding="utf-8", newline="") as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                status = (r.get("status", "") or "").strip().lower()
                if not status or "fail" in status or "error" in status:
                    continue
                kind = r.get("kind", "")
                n = r.get("out_n", "")
                if kind.startswith("recon_") and n:
                    stages_done.add(f"{kind}:{n}")
    expected = {
        "recon_base:128",
        "recon_upscaled:256", "recon_refined:256",
        "recon_upscaled:512", "recon_refined:512",
        "recon_upscaled:1024", "recon_refined:1024",
        "recon_upscaled:2048", "recon_refined:2048",
    }
    missing = sorted(expected - stages_done)
    complete = len(missing) == 0
    if pre_stats.ok and raw_overlay_aspect > 0.0:
        pre_raw_aspect_delta_pct = 100.0 * abs(pre_stats.aspect - raw_overlay_aspect) / max(0.1, raw_overlay_aspect)
    else:
        pre_raw_aspect_delta_pct = float("nan")
    if final_stats.ok and raw_overlay_aspect > 0.0:
        final_raw_aspect_delta_pct = 100.0 * abs(final_stats.aspect - raw_overlay_aspect) / max(0.1, raw_overlay_aspect)
    else:
        final_raw_aspect_delta_pct = float("nan")

    notes: List[str] = []
    if not complete:
        notes.append("missing_required_outputs=" + ";".join(missing))
    if case_name.startswith("saturn"):
        if not math.isnan(pre_raw_aspect_delta_pct) and pre_raw_aspect_delta_pct > 15.0:
            notes.append("saturn_pre_aspect_delta_gt_15pct")
        if pre_stats.ok and pre_stats.mean_luma > raw_stats.mean_luma + 0.12:
            notes.append("saturn_luma_washout")
        if pre_stats.ok and pre_stats.mean_sat < raw_stats.mean_sat * 0.70:
            notes.append("saturn_saturation_drop")
    if not notes:
        notes.append("PASS")

    return {
        "case_name": case_name,
        "source_out": str(source_out),
        "bundle_path": str(bundle) if bundle else "",
        "source_image": str(source_image),
        "run_completion_status": "complete" if complete else "partial",
        "missing_required_outputs": "|".join(missing),
        "raw_path": str(raw_path) if raw_path else "",
        "preconditioned_path": str(pre_path) if pre_path else "",
        "final_refined_path": str(final_path) if final_path else "",
        "contact_sheet_path": str(sheet_path) if sheet_path else "",
        "raw_w": raw_img.w if raw_img else 0,
        "raw_h": raw_img.h if raw_img else 0,
        "pre_w": pre_img.w if pre_img else 0,
        "pre_h": pre_img.h if pre_img else 0,
        "final_w": final_img.w if final_img else 0,
        "final_h": final_img.h if final_img else 0,
        "raw_aspect": raw_overlay_aspect,
        "pre_aspect": pre_stats.aspect,
        "final_aspect": final_stats.aspect,
        "pre_raw_aspect_delta_pct": pre_raw_aspect_delta_pct,
        "final_raw_aspect_delta_pct": final_raw_aspect_delta_pct,
        "raw_margin_px": raw_stats.margin_px,
        "raw_margin_frac": raw_stats.margin_frac,
        "pre_margin_px": pre_stats.margin_px,
        "pre_margin_frac": pre_stats.margin_frac,
        "final_margin_px": final_stats.margin_px,
        "final_margin_frac": final_stats.margin_frac,
        "raw_luma": raw_stats.mean_luma,
        "pre_luma": pre_stats.mean_luma,
        "final_luma": final_stats.mean_luma,
        "raw_sat": raw_stats.mean_sat,
        "pre_sat": pre_stats.mean_sat,
        "final_sat": final_stats.mean_sat,
        "raw_sat_frac": raw_stats.sat_frac,
        "pre_sat_frac": pre_stats.sat_frac,
        "final_sat_frac": final_stats.sat_frac,
        "raw_support_frac": raw_stats.support_frac,
        "pre_support_frac": pre_stats.support_frac,
        "final_support_frac": final_stats.support_frac,
        "notes": ";".join(notes),
    }


def write_reports(rows: List[Dict[str, object]], out_csv: Path, out_md: Path) -> None:
    fields = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = ["# Reconstruction Validation Report", ""]
    for r in rows:
        lines.append(f"## {r['case_name']}")
        lines.append(f"- run_completion_status: `{r['run_completion_status']}`")
        lines.append(f"- bundle_path: `{r['bundle_path']}`")
        lines.append(f"- source_out: `{r['source_out']}`")
        lines.append(f"- source_image: `{r['source_image']}`")
        lines.append(f"- raw/pre/final aspect: `{r['raw_aspect']:.4f}` / `{r['pre_aspect']:.4f}` / `{r['final_aspect']:.4f}`")
        lines.append(f"- pre_raw_aspect_delta_pct: `{r['pre_raw_aspect_delta_pct']:.2f}`")
        lines.append(f"- final_raw_aspect_delta_pct: `{r['final_raw_aspect_delta_pct']:.2f}`")
        lines.append(f"- raw/pre/final luma: `{r['raw_luma']:.4f}` / `{r['pre_luma']:.4f}` / `{r['final_luma']:.4f}`")
        lines.append(f"- raw/pre/final sat: `{r['raw_sat']:.4f}` / `{r['pre_sat']:.4f}` / `{r['final_sat']:.4f}`")
        lines.append(f"- raw/pre/final margin_frac: `{r['raw_margin_frac']:.4f}` / `{r['pre_margin_frac']:.4f}` / `{r['final_margin_frac']:.4f}`")
        lines.append(f"- final_refined_path: `{r['final_refined_path']}`")
        lines.append(f"- contact_sheet_path: `{r['contact_sheet_path']}`")
        lines.append(f"- notes: `{r['notes']}`")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate latest Earth/Mars/Saturn reconstruction outputs deterministically.")
    ap.add_argument("--repo-root", default=".", help="Path to sgl_obc_pi_jetson_full root")
    ap.add_argument("--out-csv", default="validation_jpg_2048_report.csv")
    ap.add_argument("--out-md", default="validation_jpg_2048_report.md")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    outputs_root = repo / "outputs"
    cases = [
        ("earth_jpg_2048_force_complete", repo / "out_jpg_earth_2048_force_complete", repo.parent / "bluemarble.jpg"),
        ("mars_jpg_2048_force_complete", repo / "out_jpg_mars_2048_force_complete", repo.parent / "mars.jpg"),
        ("saturn_jpg_2048_force_complete", repo / "out_jpg_saturn_2048_force_complete", repo.parent / "saturn.jpg"),
    ]
    rows: List[Dict[str, object]] = []
    for case_name, out_root, src in cases:
        bundle = latest_bundle_for(case_name, outputs_root)
        rows.append(summarize_case(case_name, out_root, bundle, src))

    out_csv = (repo / args.out_csv).resolve()
    out_md = (repo / args.out_md).resolve()
    write_reports(rows, out_csv, out_md)
    print(str(out_csv))
    print(str(out_md))
    for r in rows:
        print(f"{r['case_name']}: status={r['run_completion_status']} pre_raw_aspect_delta_pct={r['pre_raw_aspect_delta_pct']:.2f} notes={r['notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
