#include "config.hpp"

#include <cctype>
#include <fstream>
#include <sstream>

namespace sgl {

static bool parse_number(const std::string& s, const std::string& key, double& out) {
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  pos = s.find(':', pos);
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < s.size() && std::isspace((unsigned char)s[pos])) pos++;
  size_t end = pos;
  while (end < s.size() &&
         (std::isdigit((unsigned char)s[end]) || s[end] == '.' || s[end] == '-' || s[end] == 'e' || s[end] == 'E' || s[end] == '+'))
    end++;
  out = std::stod(s.substr(pos, end - pos));
  return true;
}

static bool parse_int(const std::string& s, const std::string& key, int& out) {
  double d = 0;
  if (!parse_number(s, key, d)) return false;
  out = (int)d;
  return true;
}

static bool parse_bool(const std::string& s, const std::string& key, bool& out) {
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  pos = s.find(':', pos);
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < s.size() && std::isspace((unsigned char)s[pos])) pos++;
  if (s.compare(pos, 4, "true") == 0) {
    out = true;
    return true;
  }
  if (s.compare(pos, 5, "false") == 0) {
    out = false;
    return true;
  }
  return false;
}

static bool parse_string(const std::string& s, const std::string& key, std::string& out) {
  auto pos = s.find("\"" + key + "\"");
  if (pos == std::string::npos) return false;
  pos = s.find(':', pos);
  if (pos == std::string::npos) return false;
  pos++;
  while (pos < s.size() && std::isspace((unsigned char)s[pos])) pos++;
  if (pos >= s.size() || s[pos] != '"') return false;
  pos++;
  size_t end = s.find('"', pos);
  if (end == std::string::npos) return false;
  out = s.substr(pos, end - pos);
  return true;
}

bool load_config_json(const std::string& path, Config& C, std::string& err) {
  std::ifstream f(path);
  if (!f) {
    err = "Could not open config: " + path;
    return false;
  }
  std::ostringstream ss;
  ss << f.rdbuf();
  std::string s = ss.str();

  parse_number(s, "power_cap_W", C.power_cap_W);
  parse_number(s, "nominal_fraction", C.nominal_fraction);
  parse_number(s, "reserve_margin_W", C.reserve_margin_W);
  parse_int(s, "lowres_N", C.lowres_N);
  parse_int(s, "highres_N", C.highres_N);
  parse_int(s, "coarse_groups_x", C.coarse_groups_x);
  parse_int(s, "coarse_groups_y", C.coarse_groups_y);
  parse_int(s, "roi_count", C.roi_count);
  parse_int(s, "tile_px_x", C.tile_px_x);
  parse_int(s, "tile_px_y", C.tile_px_y);
  parse_int(s, "progressive_base_N", C.progressive_base_N);
  parse_int(s, "progressive_max_N", C.progressive_max_N);
  parse_int(s, "progressive_scale", C.progressive_scale);
  parse_int(s, "progressive_max_stages", C.progressive_max_stages);
  parse_int(s, "progressive_roi_growth", C.progressive_roi_growth);
  parse_number(s, "ring_radius", C.ring_radius);
  parse_number(s, "ring_sigma", C.ring_sigma);
  parse_number(s, "pi_idle_W", C.pi_idle_W);
  parse_number(s, "pi_active_W", C.pi_active_W);
  parse_number(s, "jetson_idle_W", C.jetson_idle_W);
  parse_number(s, "jetson_coarse_W", C.jetson_coarse_W);
  parse_number(s, "jetson_refine_W", C.jetson_refine_W);

  parse_bool(s, "source_preconditioning_enabled", C.source_preconditioning_enabled);
  parse_int(s, "source_canvas_N", C.source_canvas_N);
  parse_string(s, "source_object_type", C.source_object_type);
  parse_number(s, "disk_fill_fraction", C.disk_fill_fraction);
  parse_number(s, "extended_fill_fraction", C.extended_fill_fraction);
  parse_number(s, "source_object_padding_fraction", C.source_object_padding_fraction);
  parse_number(s, "minimum_source_margin_fraction", C.minimum_source_margin_fraction);
  parse_int(s, "source_background_value", C.source_background_value);
  parse_string(s, "source_brightness_normalization_mode", C.source_brightness_normalization_mode);
  parse_number(s, "source_brightness_target_luma", C.source_brightness_target_luma);
  parse_number(s, "source_brightness_gain_min_disk", C.source_brightness_gain_min_disk);
  parse_number(s, "source_brightness_gain_max_disk", C.source_brightness_gain_max_disk);
  parse_number(s, "source_brightness_gain_min_extended", C.source_brightness_gain_min_extended);
  parse_number(s, "source_brightness_gain_max_extended", C.source_brightness_gain_max_extended);
  parse_number(s, "source_disk_photo_center_x", C.source_disk_photo_center_x);
  parse_number(s, "source_disk_photo_center_y", C.source_disk_photo_center_y);
  parse_number(s, "source_disk_photo_radius_px", C.source_disk_photo_radius_px);
  parse_number(s, "source_disk_photo_crop_half_px", C.source_disk_photo_crop_half_px);
  parse_number(s, "disk_photo_center_x", C.source_disk_photo_center_x);
  parse_number(s, "disk_photo_center_y", C.source_disk_photo_center_y);
  parse_number(s, "disk_photo_radius_px", C.source_disk_photo_radius_px);
  parse_number(s, "disk_photo_crop_half_px", C.source_disk_photo_crop_half_px);
  parse_int(s, "ring_sensor_N", C.ring_sensor_N);
  parse_number(s, "ring_radius_fraction", C.ring_radius_fraction);
  parse_int(s, "ring_radial_width_px", C.ring_radial_width_px);
  parse_string(s, "ring_processing_mode", C.ring_processing_mode);
  parse_int(s, "ring_angular_samples", C.ring_angular_samples);
  parse_int(s, "ring_radial_samples", C.ring_radial_samples);
  parse_bool(s, "store_full_ring_frames", C.store_full_ring_frames);
  parse_int(s, "store_ring_preview_every", C.store_ring_preview_every);
  parse_bool(s, "store_all_full_ring_frames_debug", C.store_all_full_ring_frames_debug);
  parse_int(s, "outputs_keep_last_cases", C.outputs_keep_last_cases);
  parse_bool(s, "outputs_retention_enabled", C.outputs_retention_enabled);
  parse_int(s, "outputs_keep_lightweight_runs", C.outputs_keep_lightweight_runs);
  parse_int(s, "outputs_keep_full_runs", C.outputs_keep_full_runs);
  parse_number(s, "outputs_max_total_gb", C.outputs_max_total_gb);
  parse_bool(s, "outputs_prune_raw_ppm", C.outputs_prune_raw_ppm);
  parse_bool(s, "outputs_prune_ring_frames", C.outputs_prune_ring_frames);
  parse_bool(s, "outputs_prune_annulus_dumps", C.outputs_prune_annulus_dumps);
  parse_bool(s, "outputs_preserve_marked_runs", C.outputs_preserve_marked_runs);
  parse_bool(s, "outputs_retention_include_out_profile", C.outputs_retention_include_out_profile);
  parse_bool(s, "outputs_retention_include_working_outs", C.outputs_retention_include_working_outs);
  parse_number(s, "min_free_disk_gb_before_run", C.min_free_disk_gb_before_run);
  parse_number(s, "warn_free_disk_gb", C.warn_free_disk_gb);
  parse_number(s, "fail_if_disk_below_gb", C.fail_if_disk_below_gb);
  parse_string(s, "reconstruction_mode", C.reconstruction_mode);

  parse_string(s, "payload_input_mode", C.payload_input_mode);
  parse_number(s, "payload_fusion_alpha", C.payload_fusion_alpha);
  parse_string(s, "jetson_transport", C.jetson_transport);
  parse_bool(s, "require_adcs_stable_for_jetson", C.require_adcs_stable_for_jetson);
  parse_string(s, "jetson_backend", C.jetson_backend);
  parse_bool(s, "jetson_allow_cpu_fallback", C.jetson_allow_cpu_fallback);
  parse_int(s, "connect_timeout_ms", C.connect_timeout_ms);
  parse_int(s, "job_ack_timeout_ms", C.job_ack_timeout_ms);
  parse_int(s, "job_result_timeout_ms", C.job_result_timeout_ms);
  parse_string(s, "host", C.host);
  parse_int(s, "port", C.port);
  parse_int(s, "sim_cycles", C.sim_cycles);
  parse_number(s, "dt_s", C.dt_s);
  parse_string(s, "source_image", C.source_image);
  parse_string(s, "out_dir", C.out_dir);
  parse_string(s, "jetson_scratch_dir", C.jetson_scratch_dir);
  parse_bool(s, "profiling_mode", C.profiling_mode);
  parse_bool(s, "profiling_force_full_compute", C.profiling_force_full_compute);
  parse_string(s, "profile_name", C.profile_name);
  parse_int(s, "observation_count_stage0", C.observation_count_stage0);
  parse_int(s, "observation_count_stage1", C.observation_count_stage1);
  parse_int(s, "observation_count_stage2", C.observation_count_stage2);
  parse_int(s, "observation_count_stage3", C.observation_count_stage3);

  return true;
}

} // namespace sgl
