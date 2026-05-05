#!/usr/bin/env python3
import argparse
import csv
import io
import json
import os
import sys
import time
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from core import (
    GROUP_ORDER,
    detect_metric_plan,
    discover_image_preview_paths,
    filter_events,
    load_csv_rows,
    load_event_records,
    load_run_summary,
    resolve_run_paths,
)


def as_float(v, default=0.0):
    try:
        s = str(v).strip().strip('"')
        return float(s)
    except Exception:
        return default


def as_int(v, default=0):
    try:
        return int(as_float(v, float(default)))
    except Exception:
        return default


PIPELINE_ORDER = [
    "raw_capture",
    "rectified",
    "ring_preview",
    "base_128",
    "upscaled_256",
    "refined_256",
    "upscaled_512",
    "refined_512",
    "upscaled_1024",
    "refined_1024",
    "upscaled_2048",
    "refined_2048",
    "contact_sheet",
    "original_source",
]

PANEL_ORDER = [
    "Power/EPS",
    "Subsystem Power Summary",
    "ADCS",
    "Thermal/Propulsion",
    "Payload/Image",
    "Scheduler/Jetson",
    "COMMS/Downlink",
    "Other/Debug",
]


class CsvTailReader:
    """Incremental CSV reader for append-only telemetry files."""

    def __init__(self, path):
        self.path = path
        self.mtime_ns = -1
        self.size = 0
        self.offset = 0
        self.header = ""
        self.remainder = ""
        self.fieldnames = None
        self.initialized = False

    def _stat(self):
        try:
            st = os.stat(self.path)
            return st.st_mtime_ns, st.st_size
        except OSError:
            return -1, 0

    def _full_reload(self):
        with open(self.path, "r", newline="", encoding="utf-8") as f:
            txt = f.read()
        self.offset = len(txt.encode("utf-8"))
        self.remainder = ""
        lines = txt.splitlines(True)
        if not lines:
            self.header = ""
            self.fieldnames = None
            return [], []
        self.header = lines[0].rstrip("\n")
        body = "".join(lines[1:])
        rows = []
        if body.strip():
            rows = list(csv.DictReader(io.StringIO(self.header + "\n" + body)))
        self.fieldnames = rows[0].keys() if rows else next(csv.reader(io.StringIO(self.header)), [])
        return list(self.fieldnames), rows

    def read_changes(self):
        mtime_ns, size = self._stat()
        if mtime_ns < 0 or size <= 0:
            return False, None, []

        if (not self.initialized) or size < self.size or mtime_ns < self.mtime_ns:
            fields, rows = self._full_reload()
            self.initialized = True
            self.size = size
            self.mtime_ns = mtime_ns
            return True, fields, rows

        if size == self.size and mtime_ns == self.mtime_ns:
            return False, None, []

        # append-only fast path
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()

        self.size = size
        self.mtime_ns = mtime_ns

        if not chunk:
            return False, None, []

        text = self.remainder + chunk
        if "\n" not in text:
            self.remainder = text
            return False, None, []

        if text.endswith("\n"):
            complete = text
            self.remainder = ""
        else:
            last_nl = text.rfind("\n")
            complete = text[: last_nl + 1]
            self.remainder = text[last_nl + 1 :]

        if not self.header:
            # if file started from empty and then got content
            lines = complete.splitlines(True)
            if not lines:
                return False, None, []
            self.header = lines[0].rstrip("\n")
            complete = "".join(lines[1:])
            if not complete.strip():
                return False, next(csv.reader(io.StringIO(self.header)), []), []

        rows = list(csv.DictReader(io.StringIO(self.header + "\n" + complete))) if complete.strip() else []
        fields = list(rows[0].keys()) if rows else (list(self.fieldnames) if self.fieldnames else next(csv.reader(io.StringIO(self.header)), []))
        if fields:
            self.fieldnames = fields
        return True, fields, rows


class TelemetryDashboard(QtWidgets.QWidget):
    def __init__(self, paths, refresh_ms=200, max_points=3000, max_events=400, config_path=None):
        super().__init__()
        self.paths = paths
        self.telemetry_path = paths["telemetry"]
        self.event_path = paths["events"]
        self.manifest_path = paths["manifest"]
        self.downlink_path = paths["downlink"]
        self.quality_path = paths["quality"]
        self.stage_timings_path = paths["stage_timings"]
        self.run_metadata_path = paths.get("run_metadata", "")
        self.config_path = config_path
        self.run_summary = load_run_summary(self.run_metadata_path, paths.get("run_dir", ""))
        self.is_review_mode = os.path.isdir(os.path.join(paths.get("run_dir", ""), "csv"))
        self.is_live_mode = not self.is_review_mode

        self.refresh_ms = refresh_ms
        self.max_points = max_points
        self.max_events = max_events

        self.plan = None
        self.telemetry_rows = deque(maxlen=max_points)
        self.cycles = deque(maxlen=max_points)
        self.buffers = {}
        self.curves = {}
        self.checks = {}
        self.last_checked = {}

        self.events = []
        self.event_markers = []
        self.last_preview_key = "refined_512"
        self.preview_paths = {}
        self.current_preview_path = None
        self.current_preview_mtime_ns = -1

        self.status_tail = {}
        self.cached_manifest_rows = []
        self.metrics_dirty = False
        self.curve_meta = {}
        self.curve_panel = {}
        self.panel_plots = {}

        self.file_sig = {}
        self.tail_reader = CsvTailReader(self.telemetry_path)
        self.effective_cfg = self._load_effective_config()
        self.playback_enabled = self._cfg_bool("live_playback_buffer_enabled", default=self.is_live_mode)
        self.playback_cycle_period_ms = self._cfg_int("live_playback_cycle_period_ms", 180)
        self.playback_lag_cycles = self._cfg_int("live_playback_lag_cycles", 5)
        self.playback_catchup_multiplier = self._cfg_float("live_playback_catchup_multiplier", 2.0)
        self.latest_data_cycle = -1
        self.display_cycle = -1
        self.last_telemetry_update_ts = time.monotonic()
        self.last_playback_step_ts = time.monotonic()

        self._build_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._refresh_all)
        self.timer.start(self.refresh_ms)
        self._refresh_all(force=True)

    def _load_effective_config(self):
        run_dir = self.paths.get("run_dir", "")
        p = os.path.join(run_dir, "config", "effective_config.json") if run_dir else ""
        if p and os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _cfg_bool(self, key, default):
        v = self.effective_cfg.get(key, default)
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        return bool(default)

    def _cfg_int(self, key, default):
        try:
            return int(self.effective_cfg.get(key, default))
        except Exception:
            return int(default)

    def _cfg_float(self, key, default):
        try:
            return float(self.effective_cfg.get(key, default))
        except Exception:
            return float(default)

    def _file_changed(self, path):
        try:
            st = os.stat(path)
            sig = (st.st_mtime_ns, st.st_size)
        except OSError:
            sig = None
        prev = self.file_sig.get(path)
        if prev == sig:
            return False
        self.file_sig[path] = sig
        return True

    def _build_ui(self):
        self.setWindowTitle("SGL Dashboard (Read-only)")
        self.resize(1820, 1040)
        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        root.addWidget(self.tabs)

        self._build_overview_tab()
        self._build_metrics_tab()
        self._build_events_tab()
        self._build_pipeline_tab()
        self._build_quality_tab()
        self._build_downlink_tab()

    def _build_overview_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.overview_text = QtWidgets.QTextEdit()
        self.overview_text.setReadOnly(True)
        layout.addWidget(self.overview_text)
        self.tabs.addTab(tab, "Overview")

    def _build_metrics_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QScrollArea()
        left.setWidgetResizable(True)
        left.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.controls_widget = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_widget)
        left.setWidget(self.controls_widget)

        self.group_selector = QtWidgets.QComboBox()
        self.group_selector.addItem("Demo Default", "__demo_default__")
        self.group_selector.addItem("Power Summary", "__power_summary__")
        self.group_selector.addItem("Single Group: Power/EPS", "Power/EPS")
        self.group_selector.addItem("Single Group: ADCS", "ADCS")
        self.group_selector.addItem("Single Group: Thermal", "Thermal")
        self.group_selector.addItem("Single Group: Propulsion", "Propulsion")
        self.group_selector.addItem("Single Group: Payload", "Payload")
        self.group_selector.addItem("Single Group: COMMS/Downlink", "COMMS")
        self.group_selector.addItem("Single Group: Scheduler/Jetson", "Jetson/Processing")
        self.group_selector.addItem("All Groups: Stacked", "__all_stacked__")
        self.group_selector.addItem("Other/Debug", "Other")
        self.group_selector.currentIndexChanged.connect(self._apply_group_filter)
        self.controls_layout.addWidget(self.group_selector)
        self.group_hint = QtWidgets.QLabel("")
        self.group_hint.setWordWrap(True)
        self.group_hint.setStyleSheet("QLabel { color: #caa45a; }")
        self.controls_layout.addWidget(self.group_hint)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_check_visible = QtWidgets.QPushButton("Check Visible")
        self.btn_uncheck_visible = QtWidgets.QPushButton("Uncheck Visible")
        self.btn_reset_view = QtWidgets.QPushButton("Reset Defaults")
        self.btn_check_visible.clicked.connect(lambda: self._set_visible_group_checks(True))
        self.btn_uncheck_visible.clicked.connect(lambda: self._set_visible_group_checks(False))
        self.btn_reset_view.clicked.connect(self._reset_default_view)
        btn_row.addWidget(self.btn_check_visible)
        btn_row.addWidget(self.btn_uncheck_visible)
        btn_row.addWidget(self.btn_reset_view)
        self.controls_layout.addLayout(btn_row)

        self.show_markers_cb = QtWidgets.QCheckBox("Show Event Markers")
        self.show_markers_cb.setChecked(False)
        self.show_markers_cb.stateChanged.connect(self._on_marker_ui_changed)
        self.controls_layout.addWidget(self.show_markers_cb)
        marker_row = QtWidgets.QHBoxLayout()
        marker_row.addWidget(QtWidgets.QLabel("Marker Filter:"))
        self.marker_filter = QtWidgets.QComboBox()
        self.marker_filter.addItem("Warnings Only", "warnings")
        self.marker_filter.addItem("Scheduler/Jetson", "scheduler_jetson")
        self.marker_filter.addItem("Payload/Image", "payload_image")
        self.marker_filter.addItem("ADCS/Thermal/Propulsion", "adcs_thermal_prop")
        self.marker_filter.addItem("All", "all")
        self.marker_filter.currentIndexChanged.connect(self._on_marker_ui_changed)
        marker_row.addWidget(self.marker_filter, 1)
        self.controls_layout.addLayout(marker_row)
        self.marker_top_only_cb = QtWidgets.QCheckBox("Markers On Top Panel Only")
        self.marker_top_only_cb.setChecked(True)
        self.marker_top_only_cb.stateChanged.connect(self._on_marker_ui_changed)
        self.controls_layout.addWidget(self.marker_top_only_cb)

        self.plot_scroll = QtWidgets.QScrollArea()
        self.plot_scroll.setWidgetResizable(True)
        self.plot_scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plot_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.plot_host = QtWidgets.QWidget()
        self.plot_host.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.MinimumExpanding)
        self.plot_host_layout = QtWidgets.QVBoxLayout(self.plot_host)
        self.plot_scroll.setWidget(self.plot_host)
        for panel_name in PANEL_ORDER:
            box = QtWidgets.QGroupBox(panel_name)
            bl = QtWidgets.QVBoxLayout(box)
            pw = pg.PlotWidget(title=panel_name)
            pw.showGrid(x=True, y=True, alpha=0.25)
            pw.setLabel("left", "Value")
            pw.setLabel("bottom", "Cycle")
            pw.addLegend()
            pw.setMinimumHeight(220)
            bl.addWidget(pw)
            self.plot_host_layout.addWidget(box)
            self.panel_plots[panel_name] = (box, pw)
        self.plot_host_layout.addStretch(1)
        layout.addWidget(left, 0)
        layout.addWidget(self.plot_scroll, 1)
        self._on_marker_ui_changed()
        self.tabs.addTab(tab, "Metrics/Subsystem")

    def _build_events_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Filter:"))
        self.events_filter = QtWidgets.QComboBox()
        for x in ["All", "adcs", "eps", "thermal", "propulsion", "comms", "payload", "scheduler", "jetson", "Warning/Error"]:
            self.events_filter.addItem(x)
        self.events_filter.currentTextChanged.connect(self._refresh_events_table)
        row.addWidget(self.events_filter)
        self.events_search = QtWidgets.QLineEdit()
        self.events_search.setPlaceholderText("Search event text...")
        self.events_search.textChanged.connect(self._refresh_events_table)
        row.addWidget(self.events_search, 1)
        layout.addLayout(row)

        self.events_table = QtWidgets.QTableWidget(0, 5)
        self.events_table.setHorizontalHeaderLabels(["Cycle", "Type", "Severity", "Message", "Value"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.events_table)

        self.tabs.addTab(tab, "Events")

    def _build_pipeline_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(tab)

        left = QtWidgets.QVBoxLayout()
        self.pipeline_selector = QtWidgets.QComboBox()
        self.pipeline_selector.currentTextChanged.connect(self._update_pipeline_image)
        left.addWidget(self.pipeline_selector)

        self.pipeline_path = QtWidgets.QLabel("No image selected")
        self.pipeline_path.setWordWrap(True)
        left.addWidget(self.pipeline_path)

        self.pipeline_list = QtWidgets.QListWidget()
        left.addWidget(self.pipeline_list)

        layout.addLayout(left, 0)

        self.pipeline_image = QtWidgets.QLabel("No image")
        self.pipeline_image.setAlignment(QtCore.Qt.AlignCenter)
        self.pipeline_image.setStyleSheet("QLabel { background:#111; color:#ddd; border:1px solid #444; }")
        self.pipeline_image.setMinimumSize(780, 780)
        layout.addWidget(self.pipeline_image, 1)

        self.tabs.addTab(tab, "Image Pipeline")

    def _build_quality_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.quality_table = QtWidgets.QTableWidget(0, 8)
        self.quality_table.setHorizontalHeaderLabels(["Stage", "N", "Kind", "NMAE", "MSE", "Obs Used", "Obs Added", "Status"])
        self.quality_table.horizontalHeader().setStretchLastSection(True)
        self.quality_table.verticalHeader().setVisible(False)
        layout.addWidget(self.quality_table)

        self.timing_table = QtWidgets.QTableWidget(0, 7)
        self.timing_table.setHorizontalHeaderLabels(["Stage", "N", "ROI", "Base ms", "Upscale ms", "Refine ms", "Total ms"])
        self.timing_table.horizontalHeader().setStretchLastSection(True)
        self.timing_table.verticalHeader().setVisible(False)
        layout.addWidget(self.timing_table)

        self.tabs.addTab(tab, "Quality/Profile")

    def _build_downlink_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        self.downlink_summary = QtWidgets.QLabel("Downlink summary unavailable")
        layout.addWidget(self.downlink_summary)

        self.manifest_table = QtWidgets.QTableWidget(0, 5)
        self.manifest_table.setHorizontalHeaderLabels(["Cycle", "Dataset", "Kind", "N", "Status"])
        self.manifest_table.horizontalHeader().setStretchLastSection(True)
        self.manifest_table.verticalHeader().setVisible(False)
        layout.addWidget(self.manifest_table)

        self.downlink_table = QtWidgets.QTableWidget(0, 4)
        self.downlink_table.setHorizontalHeaderLabels(["Cycle", "Dataset", "Path", "Status"])
        self.downlink_table.horizontalHeader().setStretchLastSection(True)
        self.downlink_table.verticalHeader().setVisible(False)
        layout.addWidget(self.downlink_table)

        self.tabs.addTab(tab, "Downlink")

    def _rebuild_metric_controls(self, fieldnames):
        for k, cb in self.checks.items():
            self.last_checked[k] = cb.isChecked()

        self.plan = detect_metric_plan(fieldnames)

        while self.controls_layout.count() > 6:
            item = self.controls_layout.takeAt(6)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for _, pw in self.panel_plots.values():
            pw.clear()

        self.buffers = {}
        self.curves = {}
        self.checks = {}
        self.curve_meta = {}
        self.curve_panel = {}

        colors = [
            "#0B3C5D", "#328CC1", "#D9B310", "#1D2731", "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02",
            "#A6761D", "#666666", "#5E548E", "#264653", "#2A9D8F", "#E76F51", "#6A994E", "#386641", "#3A86FF", "#8338EC",
            "#FB5607", "#FF006E", "#277DA1", "#43AA8B",
        ]

        self.key_to_group = {}
        ci = 0
        for group in GROUP_ORDER:
            metrics = self.plan.grouped_metrics.get(group, []) if self.plan else []
            if not metrics:
                continue
            box = QtWidgets.QGroupBox(group)
            if group == "Other":
                box.setCheckable(True)
                box.setChecked(False)
            box_l = QtWidgets.QVBoxLayout(box)
            for key, label in metrics:
                cb = QtWidgets.QCheckBox(label)
                cb.setChecked(self.last_checked.get(key, False))
                cb.stateChanged.connect(self._refresh_visibility)
                self.checks[key] = cb
                self.key_to_group[key] = group
                box_l.addWidget(cb)
                panel = self._panel_for_metric(group, key)
                pen = pg.mkPen(colors[ci % len(colors)], width=2)
                pw = self.panel_plots[panel][1]
                self.curves[key] = pw.plot([], [], pen=pen, name=f"{group}: {label}")
                self.buffers[key] = deque(maxlen=self.max_points)
                self.curve_meta[key] = {"label": label, "group": group}
                self.curve_panel[key] = panel
                ci += 1
            self.controls_layout.addWidget(box)

        self.controls_layout.addStretch(1)
        self._reset_default_view()
        self._update_plot_host_min_height()

    def _panel_for_metric(self, group, key):
        if key in ("source_w", "noncompute_w", "compute_budget_w", "total_bus_load_w"):
            return "Power/EPS"
        if key.endswith("_power_w"):
            return "Subsystem Power Summary"
        if group == "ADCS":
            return "ADCS"
        if group in ("Thermal", "Propulsion"):
            return "Thermal/Propulsion"
        if group == "Payload":
            return "Payload/Image"
        if group == "Jetson/Processing":
            return "Scheduler/Jetson"
        if group == "COMMS":
            return "COMMS/Downlink"
        return "Other/Debug"

    def _default_metric_keys(self):
        preferred = {
            "source_w", "compute_budget_w", "total_bus_load_w", "scheduler_mode",
            "thermal_temp_c", "heater_active", "truth_pointing_err_deg",
            "payload_active", "dataset_ready", "jetson_mode", "processing_queue", "roi_count", "comms_backlog_bits",
        }
        return preferred

    def _update_plot_host_min_height(self):
        visible = [box for (box, _) in self.panel_plots.values() if box.isVisible()]
        if not visible:
            self.plot_host.setMinimumHeight(480)
            return
        spacing = self.plot_host_layout.spacing()
        total = 24
        for box in visible:
            total += max(box.sizeHint().height(), 260) + spacing
        self.plot_host.setMinimumHeight(total + 24)

    def _reset_default_view(self):
        default_keys = self._default_metric_keys()
        for key, cb in self.checks.items():
            cb.setChecked(key in default_keys)
        self.group_selector.setCurrentIndex(0)
        self._apply_group_filter(force_redraw=True)

    def _set_visible_group_checks(self, checked):
        active = self.group_selector.currentData()
        for key, cb in self.checks.items():
            grp = self.key_to_group.get(key, "")
            if active in ("__all_stacked__", "__demo_default__", "__power_summary__") or grp == active:
                cb.setChecked(checked)
        self._apply_group_filter(force_redraw=True)

    def _apply_group_filter(self, force_redraw=False):
        active = self.group_selector.currentData()
        if active == "__all_stacked__":
            self.group_hint.setText("All Groups uses stacked panels with independent axes.")
        else:
            self.group_hint.setText("")
        # panel visibility
        visible_panels = set()
        if active == "__all_stacked__":
            visible_panels = set(PANEL_ORDER)
        elif active == "__power_summary__":
            visible_panels = {"Power/EPS", "Subsystem Power Summary"}
        elif active == "__demo_default__":
            visible_panels = {"Power/EPS", "Subsystem Power Summary", "ADCS", "Thermal/Propulsion", "Payload/Image", "Scheduler/Jetson", "COMMS/Downlink"}
        elif active in ("Power/EPS", "ADCS", "Thermal", "Propulsion", "Payload", "COMMS", "Jetson/Processing", "Other"):
            mapped = {
                "Power/EPS": {"Power/EPS"},
                "ADCS": {"ADCS"},
                "Thermal": {"Thermal/Propulsion"},
                "Propulsion": {"Thermal/Propulsion"},
                "Payload": {"Payload/Image"},
                "COMMS": {"COMMS/Downlink"},
                "Jetson/Processing": {"Scheduler/Jetson"},
                "Other": {"Other/Debug"},
            }
            visible_panels = mapped.get(active, {"Power/EPS"})
        for panel_name, (box, _) in self.panel_plots.items():
            box.setVisible(panel_name in visible_panels)
        for key, curve in self.curves.items():
            grp = self.key_to_group.get(key, "")
            cb = self.checks.get(key)
            panel = self.curve_panel.get(key, "Other/Debug")
            panel_ok = panel in visible_panels
            if active == "__all_stacked__":
                want_group = True
            elif active == "__power_summary__":
                want_group = panel in ("Power/EPS", "Subsystem Power Summary")
            else:
                want_group = (active in ("__demo_default__",) or grp == active)
            curve.setVisible(bool(cb and cb.isChecked() and want_group and panel_ok))
        if force_redraw:
            self.metrics_dirty = True
            self._redraw_metrics_from_cache()
        self._update_plot_host_min_height()
        self._render_event_markers()

    def _refresh_visibility(self):
        self._apply_group_filter(force_redraw=True)

    def _redraw_metrics_from_cache(self):
        if not self.telemetry_rows:
            return
        self.cycles.clear()
        for k in self.buffers:
            self.buffers[k].clear()

        rows = self._rows_for_display()
        if not rows:
            rows = [self.telemetry_rows[-1]]

        for r in rows:
            self.cycles.append(as_int(r.get("cycle", 0)))
            for key in self.buffers:
                self.buffers[key].append(as_float(r.get(key, 0.0)))

        x = list(self.cycles)
        for key, curve in self.curves.items():
            vals = list(self.buffers.get(key, []))
            label = self.curve_meta.get(key, {}).get("label", key)
            scaled_vals, scaled_label = self._scale_series_for_display(key, vals, label)
            group = self.curve_meta.get(key, {}).get("group", "Metric")
            curve.setData(x, scaled_vals, name=f"{group}: {scaled_label}")

        for _, pw in self.panel_plots.values():
            pw.update()
        self.metrics_dirty = False

    def _rows_for_display(self):
        rows = list(self.telemetry_rows)
        if not self.playback_enabled or self.display_cycle < 0:
            return rows
        shown = [r for r in rows if as_int(r.get("cycle", 0)) <= self.display_cycle]
        return shown

    def _advance_playback_cycle(self):
        if not self.telemetry_rows:
            return False
        latest = as_int(self.telemetry_rows[-1].get("cycle", 0))
        self.latest_data_cycle = latest
        prev = self.display_cycle
        now = time.monotonic()

        if not self.playback_enabled:
            self.display_cycle = latest
            self.last_playback_step_ts = now
            return self.display_cycle != prev

        if self.display_cycle < 0:
            self.display_cycle = max(0, latest - max(0, self.playback_lag_cycles))
            self.last_playback_step_ts = now
            return self.display_cycle != prev

        dt_ms = max(0.0, (now - self.last_playback_step_ts) * 1000.0)
        if dt_ms <= 0.0:
            return False
        self.last_playback_step_ts = now

        idle_s = now - self.last_telemetry_update_ts
        idle_threshold_s = max(1.5, (self.playback_cycle_period_ms * max(1, self.playback_lag_cycles)) / 1000.0)
        if idle_s >= idle_threshold_s:
            target = latest
        else:
            target = max(0, latest - max(0, self.playback_lag_cycles))

        if self.display_cycle >= target:
            self.display_cycle = target
            return self.display_cycle != prev

        cycles_per_tick = dt_ms / max(1.0, float(self.playback_cycle_period_ms))
        if target - self.display_cycle > max(1, self.playback_lag_cycles):
            cycles_per_tick *= max(1.0, self.playback_catchup_multiplier)
        step = max(1, int(cycles_per_tick))
        self.display_cycle = min(target, self.display_cycle + step)
        return self.display_cycle != prev

    def _scale_series_for_display(self, key, values, label):
        if not values:
            return values, label
        vmax = max(abs(v) for v in values)
        kl = key.lower()
        ll = label.lower()
        if ("bits" in kl or "bits" in ll or "queue" in kl or "backlog" in kl) and vmax >= 1e6:
            return [v / 1e6 for v in values], label.replace("(bits)", "(Mbits)") if "(bits)" in label else f"{label} (Mbits)"
        if ("bytes" in kl or "bytes" in ll) and vmax >= 1e6:
            return [v / 1e6 for v in values], f"{label} (MB)"
        if ("pixel" in kl or "annulus" in kl) and vmax >= 1e5:
            return [v / 1e4 for v in values], f"{label} (x10k)"
        if vmax >= 1e6 and not kl.endswith("_w"):
            return [v / 1e6 for v in values], f"{label} (x1e6)"
        if vmax >= 1e3 and not kl.endswith("_w"):
            return [v / 1e3 for v in values], f"{label} (x1e3)"
        return values, label

    def _refresh_overview(self):
        t = self.status_tail or {}
        last_event = self.events[-1] if self.events else None
        mode = "packaged review" if self.is_review_mode else "live working output"

        sim_cycles = self.effective_cfg.get("sim_cycles", "n/a")
        pmax = self.effective_cfg.get("progressive_max_N", "n/a")

        lines = [
            f"Run: {self.run_summary.run_name or 'n/a'}",
            f"Mode: {mode}",
            f"Run status: {self.run_summary.run_status} ({self.run_summary.completion_reason})",
            f"Cycle: {t.get('cycle', 'n/a')} / max cycles: {sim_cycles}",
            f"Playback: {'buffered' if self.playback_enabled else 'direct'} | latest={self.latest_data_cycle if self.latest_data_cycle >= 0 else 'n/a'} displayed={self.display_cycle if self.display_cycle >= 0 else 'n/a'} lag={(self.latest_data_cycle - self.display_cycle) if (self.latest_data_cycle >= 0 and self.display_cycle >= 0) else 'n/a'}",
            f"Scheduler mode: {t.get('scheduler_mode', 'n/a')}",
            f"Compute budget (W): {t.get('compute_budget_w', 'n/a')}",
            f"Source/Bus load (W): {t.get('source_w', 'n/a')} / {t.get('total_bus_load_w', 'n/a')}",
            f"Jetson: mode={t.get('jetson_mode', 'n/a')} job={t.get('jetson_job_type', 'n/a')}",
            f"Payload: mode={t.get('payload_mode', 'n/a')} dataset={t.get('dataset_id', 'n/a')}",
            f"Downlink backlog: {t.get('comms_backlog_bits', 'n/a')}",
            f"Requested stages: {list(self.run_summary.requested_stages)}",
            f"Completed stages: {list(self.run_summary.completed_stages)}",
            f"Missing required: {list(self.run_summary.missing_required_outputs)}",
            f"Missing optional: {list(self.run_summary.missing_optional_outputs)}",
            f"progressive_max_N: {pmax}",
        ]
        if str(sim_cycles) == "80":
            lines.append("Note: sim_cycles=80 means last cycle index 79 is expected.")
        if last_event:
            lines.append(f"Latest event: [cycle {last_event.cycle}] {last_event.event_type} ({last_event.severity}) {last_event.message}")
        self.overview_text.setPlainText("\n".join(lines))

    def _refresh_events_table(self):
        filtered = filter_events(self.events, self.events_filter.currentText())
        q = self.events_search.text().strip().lower()
        if q:
            filtered = [e for e in filtered if q in f"{e.event_type} {e.severity} {e.message} {e.value}".lower()]

        self.events_table.setRowCount(len(filtered))
        for i, e in enumerate(filtered):
            self.events_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(e.cycle)))
            self.events_table.setItem(i, 1, QtWidgets.QTableWidgetItem(e.event_type))
            self.events_table.setItem(i, 2, QtWidgets.QTableWidgetItem(e.severity))
            self.events_table.setItem(i, 3, QtWidgets.QTableWidgetItem(e.message))
            self.events_table.setItem(i, 4, QtWidgets.QTableWidgetItem(e.value))

    def _render_event_markers(self):
        for m in self.event_markers:
            for _, pw in self.panel_plots.values():
                pw.removeItem(m)
        self.event_markers = []

        if not self.show_markers_cb.isChecked():
            return

        target_panels = self._marker_target_panels()
        if not target_panels:
            return

        for e in self.events:
            if not self._event_matches_marker_filter(e):
                continue
            sev = e.severity.lower()
            color = (90, 160, 255, 65)
            if sev == "warn":
                color = (255, 170, 70, 95)
            elif sev == "error":
                color = (255, 80, 80, 120)
            for pw in target_panels:
                line = pg.InfiniteLine(pos=e.cycle, angle=90, movable=False, pen=pg.mkPen(color, width=1))
                pw.addItem(line)
                self.event_markers.append(line)

    def _marker_target_panels(self):
        visible = [self.panel_plots[p][1] for p in PANEL_ORDER if self.panel_plots[p][0].isVisible()]
        if not visible:
            visible = [pw for (_, pw) in self.panel_plots.values()]
        if self.marker_top_only_cb.isChecked():
            return visible[:1]
        return visible

    def _event_matches_marker_filter(self, event):
        mode = self.marker_filter.currentData() if hasattr(self, "marker_filter") else "warnings"
        text = f"{event.event_type} {event.message}".lower()
        sev = event.severity.lower()
        if mode == "all":
            return True
        if mode == "warnings":
            return sev in ("warn", "error")
        if mode == "scheduler_jetson":
            keys = ("scheduler", "jetson", "throttle", "suspend", "job_", "job ")
            return any(k in text for k in keys)
        if mode == "payload_image":
            keys = ("payload", "dataset", "camera", "ring", "recon", "acquisition", "alignment")
            return any(k in text for k in keys)
        if mode == "adcs_thermal_prop":
            keys = ("adcs", "tracker", "pointing", "wheel", "thermal", "heater", "propulsion", "burn")
            return any(k in text for k in keys)
        return True

    def _on_marker_ui_changed(self):
        enabled = self.show_markers_cb.isChecked()
        self.marker_filter.setEnabled(enabled)
        self.marker_top_only_cb.setEnabled(enabled)
        self._render_event_markers()

    def _refresh_pipeline(self, telemetry_changed):
        if not self.telemetry_rows:
            return

        row = self.telemetry_rows[-1]
        previews = discover_image_preview_paths(row, self.telemetry_path, self.manifest_path, list(self.telemetry_rows))

        # stage products from manifest
        stage_paths = {}
        for r in self.cached_manifest_rows:
            kind = str(r.get("kind", "")).strip().lower()
            out_n = str(r.get("out_n", "")).strip()
            p = str(r.get("path", "")).strip().strip('"')
            if not p:
                continue
            if not os.path.isabs(p):
                # manifest path is relative to repo root or out dir
                cands = [
                    os.path.abspath(os.path.join(os.getcwd(), p)),
                    os.path.abspath(os.path.join(os.path.dirname(self.telemetry_path), p)),
                    os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(self.telemetry_path)), p)),
                ]
                rp = ""
                for c in cands:
                    if os.path.exists(c):
                        rp = c
                        break
                if not rp:
                    continue
                p = rp
            elif not os.path.exists(p):
                continue

            if kind == "recon_base" and out_n == "128":
                stage_paths["base_128"] = p
            elif kind == "recon_upscaled":
                stage_paths[f"upscaled_{out_n}"] = p
            elif kind == "recon_refined":
                stage_paths[f"refined_{out_n}"] = p

        run_dir = self.paths.get("run_dir", "")
        cs = os.path.join(run_dir, "images", "products", "reconstruction_contact_sheet.png")
        if os.path.exists(cs):
            stage_paths["contact_sheet"] = cs

        # original source heuristic
        raw = previews.raw_capture
        if raw:
            stage_paths["original_source"] = raw

        new_map = {
            "raw_capture": previews.raw_capture,
            "rectified": previews.rectified,
            "ring_preview": previews.ring_preview,
            **stage_paths,
        }

        keys_old = [self.pipeline_selector.itemText(i) for i in range(self.pipeline_selector.count())]
        keys_new = [k for k in PIPELINE_ORDER if k in new_map]
        if not keys_new:
            keys_new = PIPELINE_ORDER

        combo_open = self.pipeline_selector.view().isVisible() or self.pipeline_selector.hasFocus()
        if (keys_old != keys_new) and not combo_open:
            selected = self.pipeline_selector.currentText() or self.last_preview_key
            self.pipeline_selector.blockSignals(True)
            self.pipeline_selector.clear()
            for k in keys_new:
                self.pipeline_selector.addItem(k)
            self.pipeline_selector.blockSignals(False)
            idx = self.pipeline_selector.findText(selected)
            if idx < 0:
                idx = self.pipeline_selector.findText(self.last_preview_key)
            if idx < 0:
                idx = 0
            if self.pipeline_selector.count() > 0:
                self.pipeline_selector.setCurrentIndex(idx)

        self.preview_paths = new_map

        self.pipeline_list.clear()
        for k in PIPELINE_ORDER:
            p = self.preview_paths.get(k)
            status = "OK" if p and os.path.exists(p) else "MISSING / NOT COMPLETED"
            self.pipeline_list.addItem(f"{k}: {status}")

        if telemetry_changed or self._file_changed(self.manifest_path):
            self._update_pipeline_image()

    def _update_pipeline_image(self):
        key = self.pipeline_selector.currentText() or self.last_preview_key
        self.last_preview_key = key
        path = self.preview_paths.get(key)

        if not path or not os.path.exists(path):
            self.current_preview_path = None
            self.current_preview_mtime_ns = -1
            self.pipeline_image.setPixmap(QtGui.QPixmap())
            self.pipeline_image.setText("MISSING / NOT COMPLETED")
            self.pipeline_path.setText(f"{key}: missing")
            return

        try:
            mtime_ns = os.stat(path).st_mtime_ns
        except OSError:
            mtime_ns = -1

        if path == self.current_preview_path and mtime_ns == self.current_preview_mtime_ns:
            self.pipeline_path.setText(path)
            return

        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self.current_preview_path = None
            self.current_preview_mtime_ns = -1
            self.pipeline_image.setPixmap(QtGui.QPixmap())
            self.pipeline_image.setText("Unable to decode image")
            self.pipeline_path.setText(path)
            return

        self.current_preview_path = path
        self.current_preview_mtime_ns = mtime_ns
        self.pipeline_image.setText("")
        self.pipeline_image.setPixmap(pix.scaled(self.pipeline_image.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
        self.pipeline_path.setText(path)

    def _refresh_quality(self):
        if not self._file_changed(self.quality_path) and not self._file_changed(self.stage_timings_path):
            return

        qrows = load_csv_rows(self.quality_path)
        self.quality_table.setRowCount(len(qrows))
        for i, r in enumerate(qrows):
            vals = [
                r.get("stage_index", ""),
                r.get("out_n", ""),
                r.get("output_kind", ""),
                r.get("nmae", ""),
                r.get("mse", ""),
                r.get("observations_used", ""),
                r.get("observations_added", r.get("observations_added_this_stage", "")),
                "ok",
            ]
            for j, v in enumerate(vals):
                self.quality_table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))

        trows = load_csv_rows(self.stage_timings_path)
        self.timing_table.setRowCount(len(trows))
        for i, r in enumerate(trows):
            vals = [
                r.get("stage_index", ""),
                r.get("out_n", ""),
                r.get("roi_count", ""),
                r.get("base_runtime_ms", ""),
                r.get("upscale_runtime_ms", ""),
                r.get("refine_runtime_ms", ""),
                r.get("total_stage_runtime_ms", ""),
            ]
            for j, v in enumerate(vals):
                self.timing_table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))

    def _refresh_downlink(self):
        changed_manifest = self._file_changed(self.manifest_path)
        changed_downlink = self._file_changed(self.downlink_path)
        if not changed_manifest and not changed_downlink:
            return

        if changed_manifest:
            self.cached_manifest_rows = load_csv_rows(self.manifest_path)
            self.manifest_table.setRowCount(len(self.cached_manifest_rows))
            for i, r in enumerate(self.cached_manifest_rows):
                vals = [r.get("cycle", ""), r.get("dataset_id", ""), r.get("kind", ""), r.get("out_n", ""), r.get("status", "")]
                for j, v in enumerate(vals):
                    self.manifest_table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))

        drows = load_csv_rows(self.downlink_path) if changed_downlink else None
        if drows is not None:
            self.downlink_table.setRowCount(len(drows))
            for i, r in enumerate(drows):
                vals = [r.get("cycle", ""), r.get("dataset_id", ""), r.get("path", ""), r.get("status", "")]
                for j, v in enumerate(vals):
                    self.downlink_table.setItem(i, j, QtWidgets.QTableWidgetItem(str(v)))

            self.downlink_summary.setText(
                f"products_manifest rows={len(self.cached_manifest_rows)} | downlink_queue rows={len(drows)}"
            )

    def _refresh_telemetry(self):
        changed, fields, rows = self.tail_reader.read_changes()
        if not changed:
            return False

        if self.plan is None and fields:
            self._rebuild_metric_controls(list(fields))

        if rows:
            for r in rows:
                self.telemetry_rows.append(r)
            self.last_telemetry_update_ts = time.monotonic()
        if not self.telemetry_rows:
            return True
        self.status_tail = self.telemetry_rows[-1]
        self.metrics_dirty = True
        return True

    def _refresh_events(self):
        if not self._file_changed(self.event_path):
            return
        self.events = load_event_records(self.event_path, max_events=self.max_events)
        self._refresh_events_table()
        self._render_event_markers()

    def _refresh_all(self, force=False):
        telemetry_changed = self._refresh_telemetry() if (force or self._file_changed(self.telemetry_path)) else False
        playback_changed = self._advance_playback_cycle()
        if playback_changed and self.telemetry_rows:
            self.metrics_dirty = True
        if self.metrics_dirty and self.telemetry_rows:
            self._redraw_metrics_from_cache()

        self._refresh_events()
        self._refresh_downlink()
        self._refresh_pipeline(telemetry_changed=telemetry_changed)
        self._refresh_quality()
        self._refresh_overview()

    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    parser = argparse.ArgumentParser(description="Read-only telemetry dashboard for SGL simulation")
    parser.add_argument("run_dir", nargs="?", default=None, help="Run directory (e.g. outputs/latest) for packaged review mode")
    parser.add_argument("--run-dir", dest="run_dir_flag", default=None, help="Run directory (e.g. outputs/latest) for packaged review mode")
    parser.add_argument("--telemetry", default=None, help="Telemetry CSV path")
    parser.add_argument("--events", default=None, help="Events CSV path")
    parser.add_argument("--manifest", default=None, help="Manifest CSV path")
    parser.add_argument("--refresh-ms", type=int, default=None, help="Refresh interval ms")
    parser.add_argument("--config-path", default=None, help="Optional config JSON path (used for live playback knobs)")
    args = parser.parse_args()

    run_dir = args.run_dir_flag or args.run_dir
    if run_dir:
        paths = resolve_run_paths(run_dir)
    else:
        telem = args.telemetry or "out/mission_store/telemetry_cycles.csv"
        paths = resolve_run_paths(telem)

    if args.events:
        paths["events"] = args.events
    if args.manifest:
        paths["manifest"] = args.manifest

    if not os.path.exists(paths["telemetry"]):
        raise SystemExit(f"Telemetry not found: {paths['telemetry']}")

    # Slower default for packaged review mode to reduce idle polling.
    if args.refresh_ms is None:
        if os.path.isdir(os.path.join(paths.get("run_dir", ""), "csv")):
            refresh_ms = 1000
        else:
            refresh_ms = 200
    else:
        refresh_ms = args.refresh_ms

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = TelemetryDashboard(paths, refresh_ms=refresh_ms, config_path=args.config_path)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
