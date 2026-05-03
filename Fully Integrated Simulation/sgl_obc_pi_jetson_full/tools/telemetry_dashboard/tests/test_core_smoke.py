import csv
import io
import os
import tempfile
import unittest

from core import (
    detect_default_event_path,
    detect_default_manifest_path,
    detect_metric_plan,
    discover_image_preview_paths,
    filter_events,
    load_csv_rows,
    load_event_records,
    load_run_summary,
    resolve_run_paths,
)


CSV_TEXT = """cycle,source_w,noncompute_w,compute_budget_w,total_bus_load_w,scheduler_mode,adcs_power_w,wheel_power_w,truth_pointing_err_deg,est_pointing_err_deg,tracker_conf,tracker_valid,tracked_stars,adcs_mode,thermal_power_w,thermal_mode,heater_active,thermal_temp_c,propulsion_power_w,propulsion_mode,propulsion_active,propulsion_thrust_n,payload_power_w,payload_mode,payload_active,dataset_ready,dataset_id,dataset_count,acquisition_stage,comms_power_w,comms_mode,comms_backlog_bits,jetson_power_w,jetson_mode,jetson_job_type,roi_count,processing_queue\n0,120,40,50,60,0,15,5,2.1,1.7,0.8,1,18,CORRECTING,2,PASSIVE,0,18.0,1,IDLE,0,0.0,10,ACQUIRE,1,0,,0,1,4,READY,1000,5,ACTIVE,coarse,8,3\n"""


class TestCoreSmoke(unittest.TestCase):
    def test_detect_metric_groups_and_status(self):
        reader = csv.DictReader(io.StringIO(CSV_TEXT))
        rows = list(reader)
        self.assertTrue(rows)
        plan = detect_metric_plan(reader.fieldnames or [])

        self.assertIn("Power/EPS", plan.grouped_metrics)
        self.assertIn("ADCS", plan.grouped_metrics)
        self.assertIn("Thermal", plan.grouped_metrics)
        self.assertIn("Propulsion", plan.grouped_metrics)
        self.assertIn("Payload", plan.grouped_metrics)
        self.assertIn("COMMS", plan.grouped_metrics)
        self.assertIn("Jetson/Processing", plan.grouped_metrics)

        adcs_keys = {k for k, _ in plan.grouped_metrics["ADCS"]}
        self.assertIn("wheel_power_w", adcs_keys)
        self.assertIn("truth_pointing_err_deg", adcs_keys)
        self.assertIn("est_pointing_err_deg", adcs_keys)

        status_keys = {k for k, _ in plan.status_fields}
        self.assertIn("adcs_mode", status_keys)
        self.assertIn("jetson_mode", status_keys)
        self.assertIn("dataset_id", status_keys)

    def test_event_parser_present_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            telemetry = os.path.join(td, "telemetry_cycles.csv")
            with open(telemetry, "w") as f:
                f.write("cycle,a\n0,1\n")
            events = detect_default_event_path(telemetry)
            self.assertEqual(events, os.path.join(td, "events.csv"))
            # missing should be graceful
            self.assertEqual(load_event_records(events), [])

            with open(events, "w") as f:
                f.write("cycle,event_type,severity,message,value\n")
                f.write("3,\"scheduler_mode_changed\",\"info\",\"Scheduler mode changed\",\"1\"\n")
                f.write("4,\"payload_dataset_ready\",\"info\",\"Payload dataset ready\",\"dataset_0\"\n")
            parsed = load_event_records(events)
            self.assertEqual(len(parsed), 2)
            self.assertEqual(parsed[0].cycle, 3)
            self.assertEqual(parsed[1].event_type, "payload_dataset_ready")

    def test_image_preview_discovery(self):
        with tempfile.TemporaryDirectory() as td:
            out_root = os.path.join(td, "out_demo")
            ms = os.path.join(out_root, "mission_store")
            ds = os.path.join(out_root, "datasets", "dataset_0")
            prod = os.path.join(out_root, "products")
            os.makedirs(ms, exist_ok=True)
            os.makedirs(ds, exist_ok=True)
            os.makedirs(prod, exist_ok=True)

            telemetry = os.path.join(ms, "telemetry_cycles.csv")
            manifest = detect_default_manifest_path(telemetry)
            self.assertEqual(manifest, os.path.join(ms, "products_manifest.csv"))

            raw = os.path.join(ds, "raw_capture.ppm")
            rect = os.path.join(ds, "rectified_input.ppm")
            ring = os.path.join(ds, "ring_preview.ppm")
            coarse = os.path.join(prod, "dataset_0_s0_coarse_128.ppm")
            refined = os.path.join(prod, "dataset_0_s0_refined_128.ppm")
            for p in (raw, rect, ring, coarse, refined):
                with open(p, "wb") as f:
                    f.write(b"P6\n1 1\n255\n\x00\x00\x00")

            with open(manifest, "w") as f:
                f.write("cycle,dataset_id,stage,kind,out_n,path,bytes,roi_count,roi_score_mean,status\n")
                f.write(f"2,\"dataset_0\",0,\"coarse\",128,\"{coarse}\",10,2,0.4,\"ok\"\n")
                f.write(f"2,\"dataset_0\",0,\"refined\",128,\"{refined}\",10,2,0.4,\"ok\"\n")

            row = {
                "dataset_id": "dataset_0",
                "raw_capture_path": raw,
                "rectified_image_path": rect,
            }
            previews = discover_image_preview_paths(row, telemetry_path=telemetry, manifest_path=manifest)
            self.assertEqual(previews.raw_capture, raw)
            self.assertEqual(previews.rectified, rect)
            self.assertEqual(previews.ring_preview, ring)
            self.assertEqual(previews.coarse, coarse)
            self.assertEqual(previews.refined, refined)

            # latest available should still resolve if tail row is empty.
            tail = {"dataset_id": "dataset_0", "raw_capture_path": "", "rectified_image_path": ""}
            previews2 = discover_image_preview_paths(
                tail,
                telemetry_path=telemetry,
                manifest_path=manifest,
                telemetry_rows=[row, tail],
            )
            self.assertEqual(previews2.raw_capture, raw)
            self.assertEqual(previews2.rectified, rect)

    def test_run_dir_resolution_and_summary(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = os.path.join(td, "outputs", "demo_run")
            csv_dir = os.path.join(run_dir, "csv")
            os.makedirs(csv_dir, exist_ok=True)
            with open(os.path.join(csv_dir, "telemetry_cycles.csv"), "w") as f:
                f.write("cycle,a\n0,1\n")
            with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
                f.write(
                    '{"run_completion_status":"partial","completion_reason":"missing required outputs",'
                    '"requested_stages":[128,256,512],"completed_stages":[128,256],'
                    '"missing_required_outputs":["512:refined"],"missing_optional_outputs":[]}'
                )

            paths = resolve_run_paths(run_dir)
            self.assertTrue(paths["telemetry"].endswith("csv/telemetry_cycles.csv"))
            rows = load_csv_rows(paths["telemetry"])
            self.assertEqual(len(rows), 1)
            summary = load_run_summary(paths["run_metadata"], run_dir)
            self.assertEqual(summary.run_status, "partial")
            self.assertEqual(summary.completed_stages, (128, 256))
            self.assertEqual(summary.missing_required_outputs, ("512:refined",))

    def test_event_filter(self):
        events = [
            type("E", (), {"cycle": 1, "event_type": "adcs_correction_started", "severity": "info", "message": "ADCS correction started", "value": ""})(),
            type("E", (), {"cycle": 2, "event_type": "compute_budget_low", "severity": "warn", "message": "budget low", "value": ""})(),
        ]
        out = filter_events(events, "adcs")
        self.assertEqual(len(out), 1)
        out2 = filter_events(events, "Warning/Error")
        self.assertEqual(len(out2), 1)


if __name__ == "__main__":
    unittest.main()
