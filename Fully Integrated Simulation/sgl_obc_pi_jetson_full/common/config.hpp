#pragma once
#include <string>
namespace sgl {
struct Config {
  double power_cap_W=25.0, nominal_fraction=0.75, reserve_margin_W=20.0;
  int lowres_N=256, highres_N=1024, coarse_groups_x=4, coarse_groups_y=4, roi_count=8, tile_px_x=64, tile_px_y=64;
  int progressive_base_N=128, progressive_max_N=1024, progressive_scale=2, progressive_max_stages=4, progressive_roi_growth=2;
  double ring_radius=0.38, ring_sigma=0.04, pi_idle_W=4.0, pi_active_W=8.0, jetson_idle_W=5.0, jetson_coarse_W=10.0, jetson_refine_W=15.0;
  bool source_preconditioning_enabled=true;
  int source_canvas_N=2048;
  std::string source_object_type="auto";
  double disk_fill_fraction=0.62;
  double extended_fill_fraction=0.85;
  double source_object_padding_fraction=0.10;
  double minimum_source_margin_fraction=0.08;
  int source_background_value=0;
  int ring_sensor_N=4096;
  double ring_radius_fraction=0.40;
  int ring_radial_width_px=64;
  std::string ring_processing_mode="annulus_unwrapped";
  int ring_angular_samples=8192;
  int ring_radial_samples=96;
  bool store_full_ring_frames=false;
  int store_ring_preview_every=16;
  bool store_all_full_ring_frames_debug=false;
  int outputs_keep_last_cases=5;
  bool outputs_retention_enabled=true;
  int outputs_keep_lightweight_runs=10;
  int outputs_keep_full_runs=3;
  double outputs_max_total_gb=0.0;
  bool outputs_prune_raw_ppm=true;
  bool outputs_prune_ring_frames=true;
  bool outputs_prune_annulus_dumps=true;
  bool outputs_preserve_marked_runs=true;
  bool outputs_retention_include_out_profile=false;
  bool outputs_retention_include_working_outs=false;
  double min_free_disk_gb_before_run=0.0;
  double warn_free_disk_gb=25.0;
  double fail_if_disk_below_gb=10.0;
  std::string reconstruction_mode="sgl_annulus";
  std::string payload_input_mode="synthetic_image"; double payload_fusion_alpha=0.4;
  std::string jetson_transport="tcp";
  bool require_adcs_stable_for_jetson=true;
  std::string jetson_backend="cpu"; bool jetson_allow_cpu_fallback=true;
  int connect_timeout_ms=3000, job_ack_timeout_ms=2000, job_result_timeout_ms=12000;
  std::string host="127.0.0.1"; int port=5500, sim_cycles=80; double dt_s=1.0; std::string source_image="bluemarble.ppm", out_dir="out", jetson_scratch_dir="out/jetson_scratch";
  bool profiling_mode=false;
  bool profiling_force_full_compute=false;
  std::string profile_name="default";
  int observation_count_stage0=1, observation_count_stage1=1, observation_count_stage2=1, observation_count_stage3=1;
};
bool load_config_json(const std::string& path, Config& C, std::string& err);
} // namespace sgl
