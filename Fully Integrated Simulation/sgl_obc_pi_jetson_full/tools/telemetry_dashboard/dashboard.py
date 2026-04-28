#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from collections import deque

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from core import (
    GROUP_ORDER,
    detect_default_event_path,
    detect_default_manifest_path,
    detect_metric_plan,
    discover_image_preview_paths,
    load_event_records,
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


class TelemetryDashboard(QtWidgets.QWidget):
    def __init__(
        self,
        csv_path,
        event_path=None,
        manifest_path=None,
        refresh_ms=200,
        max_points=3000,
        max_events=300,
    ):
        super().__init__()
        self.csv_path = csv_path
        self.event_path = event_path or detect_default_event_path(csv_path)
        self.manifest_path = manifest_path or detect_default_manifest_path(csv_path)
        self.refresh_ms = refresh_ms
        self.max_points = max_points
        self.max_events = max_events
        self.last_size = 0
        self.last_fieldnames = None
        self.plan = None

        self.last_event_size = -1
        self.events = []
        self.event_markers = []

        self.cycles = deque(maxlen=max_points)
        self.buffers = {}
        self.curves = {}
        self.checks = {}
        self.last_checked = {}
        self.preview_paths = {}
        self.last_preview_choice = "refined"

        self._build_shell_ui()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(self.refresh_ms)

    def _build_shell_ui(self):
        self.setWindowTitle("SGL Telemetry Dashboard (Read-only)")
        self.resize(1700, 960)

        main = QtWidgets.QHBoxLayout(self)

        self.left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(self.left)

        self.summary_label = QtWidgets.QLabel("Waiting for telemetry...")
        left_layout.addWidget(self.summary_label)

        self.show_markers_cb = QtWidgets.QCheckBox("Show Event Markers")
        self.show_markers_cb.setChecked(True)
        self.show_markers_cb.stateChanged.connect(self._render_event_markers)
        left_layout.addWidget(self.show_markers_cb)

        self.status_table = QtWidgets.QTableWidget(0, 2)
        self.status_table.setHorizontalHeaderLabels(["Field", "Value"])
        self.status_table.horizontalHeader().setStretchLastSection(True)
        self.status_table.verticalHeader().setVisible(False)
        self.status_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.status_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.status_table.setMaximumHeight(220)
        left_layout.addWidget(self.status_table)

        self.events_table = QtWidgets.QTableWidget(0, 4)
        self.events_table.setHorizontalHeaderLabels(["Cycle", "Type", "Severity", "Message"])
        self.events_table.horizontalHeader().setStretchLastSection(True)
        self.events_table.verticalHeader().setVisible(False)
        self.events_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.events_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.events_table.setMaximumHeight(260)
        left_layout.addWidget(self.events_table)

        preview_box = QtWidgets.QGroupBox("Image Preview (Read-only)")
        preview_layout = QtWidgets.QVBoxLayout(preview_box)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("View:"))
        self.preview_selector = QtWidgets.QComboBox()
        self.preview_selector.currentTextChanged.connect(self._update_preview_image)
        top_row.addWidget(self.preview_selector, 1)
        preview_layout.addLayout(top_row)

        self.preview_path_label = QtWidgets.QLabel("No image selected.")
        self.preview_path_label.setWordWrap(True)
        preview_layout.addWidget(self.preview_path_label)

        self.preview_image_label = QtWidgets.QLabel("No preview image available yet.")
        self.preview_image_label.setMinimumSize(420, 260)
        self.preview_image_label.setAlignment(QtCore.Qt.AlignCenter)
        self.preview_image_label.setStyleSheet("QLabel { background-color: #111; color: #ddd; border: 1px solid #444; }")
        preview_layout.addWidget(self.preview_image_label)

        left_layout.addWidget(preview_box)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.controls_widget = QtWidgets.QWidget()
        self.controls_layout = QtWidgets.QVBoxLayout(self.controls_widget)
        self.controls_layout.addStretch(1)
        self.scroll.setWidget(self.controls_widget)
        left_layout.addWidget(self.scroll)

        self.plot = pg.PlotWidget(title="SGL Telemetry")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("left", "Value")
        self.plot.setLabel("bottom", "Cycle")
        self.plot.addLegend()

        main.addWidget(self.left, 0)
        main.addWidget(self.plot, 1)

    def _rebuild_metric_controls(self, fieldnames):
        for k, cb in self.checks.items():
            self.last_checked[k] = cb.isChecked()

        self.plan = detect_metric_plan(fieldnames)
        self.last_fieldnames = tuple(fieldnames)

        while self.controls_layout.count() > 0:
            item = self.controls_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        for c in self.curves.values():
            self.plot.removeItem(c)

        self.buffers = {}
        self.curves = {}
        self.checks = {}

        colors = [
            "#0B3C5D", "#328CC1", "#D9B310", "#1D2731", "#1B9E77", "#D95F02", "#7570B3", "#E7298A", "#66A61E", "#E6AB02",
            "#A6761D", "#666666", "#5E548E", "#264653", "#2A9D8F", "#E76F51", "#6A994E", "#386641", "#3A86FF", "#8338EC",
            "#FB5607", "#FF006E", "#6D597A", "#4361EE", "#4D908E", "#277DA1", "#F94144", "#F8961E", "#43AA8B", "#577590",
        ]

        color_i = 0
        for group in GROUP_ORDER:
            metrics = self.plan.grouped_metrics.get(group, []) if self.plan else []
            if not metrics:
                continue
            box = QtWidgets.QGroupBox(group)
            box_l = QtWidgets.QVBoxLayout(box)
            for key, label in metrics:
                cb = QtWidgets.QCheckBox(label)
                cb.setChecked(self.last_checked.get(key, color_i < 6))
                cb.stateChanged.connect(self._refresh_visibility)
                self.checks[key] = cb
                box_l.addWidget(cb)

                pen = pg.mkPen(colors[color_i % len(colors)], width=2)
                self.curves[key] = self.plot.plot([], [], pen=pen, name=f"{group}: {label}")
                self.buffers[key] = deque(maxlen=self.max_points)
                color_i += 1
            self.controls_layout.addWidget(box)

        self.controls_layout.addStretch(1)
        self._refresh_visibility()
        self._render_event_markers()

    def _refresh_visibility(self):
        for key, curve in self.curves.items():
            cb = self.checks.get(key)
            curve.setVisible(bool(cb and cb.isChecked()))

    def _set_preview_choices(self, mapping):
        current = self.preview_selector.currentData()
        self.preview_selector.blockSignals(True)
        self.preview_selector.clear()
        for key, path in mapping.items():
            if path:
                self.preview_selector.addItem(key.replace("_", " ").title(), key)
        self.preview_selector.blockSignals(False)

        if self.preview_selector.count() == 0:
            self.preview_image_label.setText("No preview image available yet.")
            self.preview_path_label.setText("No image path discovered from telemetry/manifest.")
            return

        preferred = self.last_preview_choice
        chosen_index = 0
        for i in range(self.preview_selector.count()):
            data = self.preview_selector.itemData(i)
            if data == current:
                chosen_index = i
                break
            if data == preferred:
                chosen_index = i
        self.preview_selector.setCurrentIndex(chosen_index)
        self._update_preview_image()

    def _update_preview_image(self):
        key = self.preview_selector.currentData()
        if not key:
            return
        self.last_preview_choice = str(key)
        path = self.preview_paths.get(str(key))
        if not path or not os.path.exists(path):
            self.preview_image_label.setPixmap(QtGui.QPixmap())
            self.preview_image_label.setText("Selected preview image is missing.")
            self.preview_path_label.setText(f"Missing: {path or 'n/a'}")
            return
        pix = QtGui.QPixmap(path)
        if pix.isNull():
            self.preview_image_label.setPixmap(QtGui.QPixmap())
            self.preview_image_label.setText("Unable to decode image format for preview.")
            self.preview_path_label.setText(path)
            return
        scaled = pix.scaled(
            self.preview_image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.preview_image_label.setText("")
        self.preview_image_label.setPixmap(scaled)
        self.preview_path_label.setText(path)

    def _update_status_table(self, tail):
        if self.plan is None:
            return
        rows = self.plan.status_fields
        self.status_table.setRowCount(len(rows))
        for i, (key, label) in enumerate(rows):
            self.status_table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
            self.status_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(tail.get(key, "n/a")).strip('"')))

        summary_keys = ["jetson_mode", "jetson_job_type", "scheduler_mode", "adcs_mode", "payload_mode", "comms_mode"]
        parts = []
        for k in summary_keys:
            if k in tail:
                parts.append(f"{k}={str(tail.get(k, 'n/a')).strip('"')}")
        self.summary_label.setText(" | ".join(parts) if parts else "Telemetry loaded")

    def _update_events(self):
        if not self.event_path or not os.path.exists(self.event_path):
            self.events = []
            self.events_table.setRowCount(0)
            self._render_event_markers()
            return

        try:
            sz = os.path.getsize(self.event_path)
        except Exception:
            return

        if sz == self.last_event_size:
            return
        self.last_event_size = sz

        self.events = load_event_records(self.event_path, max_events=self.max_events)
        self.events_table.setRowCount(len(self.events))
        for i, e in enumerate(self.events):
            self.events_table.setItem(i, 0, QtWidgets.QTableWidgetItem(str(e.cycle)))
            self.events_table.setItem(i, 1, QtWidgets.QTableWidgetItem(e.event_type))
            self.events_table.setItem(i, 2, QtWidgets.QTableWidgetItem(e.severity))
            msg = e.message if not e.value else f"{e.message} [{e.value}]"
            self.events_table.setItem(i, 3, QtWidgets.QTableWidgetItem(msg))

        self._render_event_markers()

    def _render_event_markers(self):
        for m in self.event_markers:
            self.plot.removeItem(m)
        self.event_markers = []

        if not self.show_markers_cb.isChecked():
            return

        if not self.events:
            return

        color_by_severity = {
            "info": (90, 160, 255, 110),
            "warn": (255, 170, 70, 130),
            "error": (255, 80, 80, 150),
        }
        for e in self.events:
            color = color_by_severity.get(e.severity.lower(), (180, 180, 180, 100))
            line = pg.InfiniteLine(pos=e.cycle, angle=90, movable=False, pen=pg.mkPen(color, width=1))
            self.plot.addItem(line)
            self.event_markers.append(line)

    def update_data(self):
        if not os.path.exists(self.csv_path):
            return
        sz = os.path.getsize(self.csv_path)
        if sz == 0 or sz == self.last_size:
            self._update_events()
            return
        self.last_size = sz

        try:
            with open(self.csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
        except Exception:
            return

        if not rows or not fieldnames:
            return

        if self.last_fieldnames != tuple(fieldnames):
            self._rebuild_metric_controls(fieldnames)

        self.cycles.clear()
        for k in self.buffers:
            self.buffers[k].clear()

        for r in rows[-self.max_points :]:
            self.cycles.append(as_int(r.get("cycle", 0)))
            for key in self.buffers:
                self.buffers[key].append(as_float(r.get(key, 0.0)))

        tail = rows[-1]
        self._update_status_table(tail)

        x = list(self.cycles)
        for key, curve in self.curves.items():
            curve.setData(x, list(self.buffers.get(key, [])))

        self._update_events()
        previews = discover_image_preview_paths(
            tail,
            telemetry_path=self.csv_path,
            manifest_path=self.manifest_path,
            telemetry_rows=rows,
        )
        self.preview_paths = {
            "raw_capture": previews.raw_capture,
            "rectified": previews.rectified,
            "ring_preview": previews.ring_preview,
            "coarse": previews.coarse,
            "refined": previews.refined,
        }
        self._set_preview_choices(self.preview_paths)


def main():
    parser = argparse.ArgumentParser(description="Read-only telemetry dashboard for SGL simulation")
    parser.add_argument(
        "--telemetry",
        default="out/mission_store/telemetry_cycles.csv",
        help="Path to telemetry CSV written by sgl_pi_flight",
    )
    parser.add_argument(
        "--events",
        default=None,
        help="Optional events CSV path. Defaults to events.csv next to telemetry file.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional products manifest path. Defaults to products_manifest.csv next to telemetry file.",
    )
    parser.add_argument("--refresh-ms", type=int, default=200, help="Refresh interval in milliseconds")
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = TelemetryDashboard(args.telemetry, event_path=args.events, manifest_path=args.manifest, refresh_ms=args.refresh_ms)
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
