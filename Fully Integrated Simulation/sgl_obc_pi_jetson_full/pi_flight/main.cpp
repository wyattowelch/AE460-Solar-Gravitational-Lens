#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "../common/config.hpp"
#include "../common/image_io.hpp"
#include "../common/logger.hpp"
#include "../common/net.hpp"
#include "../common/protocol.hpp"
#include "../common/scheduler.hpp"
#include "../jetson_processing/processor.hpp"
#include "sgl/eps_module.hpp"
#include "subsystems.hpp"

namespace fs = std::filesystem;
using sgl::LogLevel;
using sgl::Logger;

namespace {
bool write_bytes(const std::string& path, const std::vector<uint8_t>& bytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return false;
  f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  return static_cast<bool>(f);
}

std::string csv_escape(const std::string& s) {
  std::string out = "\"";
  for (char c : s) out += (c == '"') ? "\"\"" : std::string(1, c);
  out += "\"";
  return out;
}

struct MissionStore {
  std::string manifest_csv;
  std::string downlink_csv;
  std::string telemetry_csv;
  std::string events_csv;
  std::string stage_timings_csv;
  std::string quality_csv;
  bool initialized = false;

  void init(const std::string& out_dir) {
    const fs::path root = fs::path(out_dir) / "mission_store";
    fs::create_directories(root);
    manifest_csv = (root / "products_manifest.csv").string();
    downlink_csv = (root / "downlink_queue.csv").string();
    telemetry_csv = (root / "telemetry_cycles.csv").string();
    events_csv = (root / "events.csv").string();
    stage_timings_csv = (root / "progressive_stage_timings.csv").string();
    quality_csv = (root / "reconstruction_quality.csv").string();
    {
      std::ofstream f(manifest_csv, std::ios::trunc);
      f << "cycle,dataset_id,stage,kind,out_n,path,bytes,roi_count,roi_score_mean,status\n";
    }
    {
      std::ofstream f(downlink_csv, std::ios::trunc);
      f << "cycle,dataset_id,priority,bits,kind,path,status\n";
    }
    {
      std::ofstream f(telemetry_csv, std::ios::trunc);
      f << "cycle,source_w,reserve_w,total_bus_load_w,noncompute_w,compute_budget_w,jetson_allow_w,scheduler_mode,adcs_mode,adcs_power_w,wheel_power_w,comms_power_w,thermal_power_w,propulsion_power_w,payload_power_w,pi_power_w,jetson_power_w,jetson_mode,jetson_job_type,truth_pointing_err_deg,est_pointing_err_deg,tracker_conf,tracker_valid,tracked_stars,comms_mode,comms_backlog_bits,payload_mode,payload_active,dataset_ready,dataset_id,dataset_count,acquisition_stage,active_stage,active_stage_n,roi_count,processing_queue,thermal_mode,heater_active,thermal_temp_c,propulsion_mode,propulsion_active,propulsion_thrust_n,camera_mode,camera_frame_ready,alignment_valid,alignment_score,blur_score,brightness_mean,contrast_score,raw_capture_path,rectified_image_path,preconditioned_source_path,source_object_type_detected,source_bbox,source_fill_fraction_used,source_margin_fraction,source_clipping_guard_triggered,preconditioning_method,alignment_method,ring_sensor_N,ring_radius_px,ring_radial_width_px,active_annulus_pixels,active_pixel_fraction,ring_angular_samples,ring_radial_samples,ring_processing_mode\n";
    }
    {
      std::ofstream f(events_csv, std::ios::trunc);
      f << "cycle,event_type,severity,message,value\n";
    }
    {
      std::ofstream f(stage_timings_csv, std::ios::trunc);
      f << "profile_name,stage_index,out_n,observations_used,new_observations_added,roi_count,base_runtime_ms,upscale_runtime_ms,refine_runtime_ms,roi_selection_ms,total_stage_runtime_ms,base_path,upscaled_path,refined_path\n";
    }
    {
      std::ofstream f(quality_csv, std::ios::trunc);
      f << "dataset_id,stage_index,output_n,output_kind,nmae,mse,observations_used,observations_added_this_stage\n";
    }
    initialized = true;
  }

  void append_manifest(int cycle, const std::string& dataset_id, int stage, const std::string& kind, int out_n, const std::string& path, size_t bytes, size_t roi_count, double roi_score_mean, const std::string& status) const {
    std::ofstream f(manifest_csv, std::ios::app);
    f << cycle << "," << csv_escape(dataset_id) << "," << stage << "," << csv_escape(kind) << "," << out_n << "," << csv_escape(path) << "," << bytes << "," << roi_count << "," << roi_score_mean << "," << csv_escape(status) << "\n";
  }

  void enqueue_downlink(int cycle, const std::string& dataset_id, int priority, size_t bits, const std::string& kind, const std::string& path) const {
    std::ofstream f(downlink_csv, std::ios::app);
    f << cycle << "," << csv_escape(dataset_id) << "," << priority << "," << bits << "," << csv_escape(kind) << "," << csv_escape(path) << "," << csv_escape("QUEUED") << "\n";
  }

  void append_telemetry(int cycle,double source_w,double reserve_w,double total_bus_load_w,double noncompute_w,double compute_budget,double jetson_allow,int scheduler_mode,const std::string& adcs_mode,double adcs_power_w,double wheel_power_w,double comms_power_w,double thermal_power_w,double propulsion_power_w,double payload_power_w,double pi_power_w,double jetson_power_w,const std::string& jetson_mode,const std::string& jetson_job_type,double truth_pointing_err,double est_pointing_err,double tracker_conf,bool tracker_valid,uint32_t tracked_stars,const std::string& comms_mode,size_t backlog_bits,const std::string& payload_mode,bool payload_active,bool dataset_ready,const std::string& dataset_id,int dataset_count,int acquisition_stage,int active_stage,int active_stage_n,int roi_count,int processing_queue,const std::string& thermal_mode,bool heater_active,double thermal_temp_c,const std::string& propulsion_mode,bool propulsion_active,double propulsion_thrust_n,const std::string& camera_mode,bool camera_frame_ready,bool alignment_valid,double alignment_score,double blur_score,double brightness_mean,double contrast_score,const std::string& raw_capture_path,const std::string& rectified_image_path,const std::string& preconditioned_source_path,const std::string& source_object_type_detected,const std::string& source_bbox,double source_fill_fraction_used,double source_margin_fraction,bool source_clipping_guard_triggered,const std::string& preconditioning_method,const std::string& alignment_method,int ring_sensor_N,int ring_radius_px,int ring_radial_width_px,long long active_annulus_pixels,double active_pixel_fraction,int ring_angular_samples,int ring_radial_samples,const std::string& ring_processing_mode) const {
    std::ofstream f(telemetry_csv, std::ios::app);
    f << cycle << "," << source_w << "," << reserve_w << "," << total_bus_load_w << "," << noncompute_w << "," << compute_budget << "," << jetson_allow << "," << scheduler_mode << "," << csv_escape(adcs_mode) << "," << adcs_power_w << "," << wheel_power_w << "," << comms_power_w << "," << thermal_power_w << "," << propulsion_power_w << "," << payload_power_w << "," << pi_power_w << "," << jetson_power_w << "," << csv_escape(jetson_mode) << "," << csv_escape(jetson_job_type) << "," << truth_pointing_err << "," << est_pointing_err << "," << tracker_conf << "," << (tracker_valid ? 1 : 0) << "," << tracked_stars << "," << csv_escape(comms_mode) << "," << backlog_bits << "," << csv_escape(payload_mode) << "," << (payload_active ? 1 : 0) << "," << (dataset_ready ? 1 : 0) << "," << csv_escape(dataset_id) << "," << dataset_count << "," << acquisition_stage << "," << active_stage << "," << active_stage_n << "," << roi_count << "," << processing_queue << "," << csv_escape(thermal_mode) << "," << (heater_active ? 1 : 0) << "," << thermal_temp_c << "," << csv_escape(propulsion_mode) << "," << (propulsion_active ? 1 : 0) << "," << propulsion_thrust_n << "," << csv_escape(camera_mode) << "," << (camera_frame_ready ? 1 : 0) << "," << (alignment_valid ? 1 : 0) << "," << alignment_score << "," << blur_score << "," << brightness_mean << "," << contrast_score << "," << csv_escape(raw_capture_path) << "," << csv_escape(rectified_image_path) << "," << csv_escape(preconditioned_source_path) << "," << csv_escape(source_object_type_detected) << "," << csv_escape(source_bbox) << "," << source_fill_fraction_used << "," << source_margin_fraction << "," << (source_clipping_guard_triggered ? 1 : 0) << "," << csv_escape(preconditioning_method) << "," << csv_escape(alignment_method) << "," << ring_sensor_N << "," << ring_radius_px << "," << ring_radial_width_px << "," << active_annulus_pixels << "," << active_pixel_fraction << "," << ring_angular_samples << "," << ring_radial_samples << "," << csv_escape(ring_processing_mode) << "\n";
  }

  void append_event(int cycle, const std::string& event_type, const std::string& severity, const std::string& message, const std::string& value = "") const {
    std::ofstream f(events_csv, std::ios::app);
    f << cycle << "," << csv_escape(event_type) << "," << csv_escape(severity) << "," << csv_escape(message) << "," << csv_escape(value) << "\n";
  }

  void append_stage_timing(const std::string& profile_name,int stage_index,int out_n,int observations_used,int new_observations_added,int roi_count,double base_runtime_ms,double upscale_runtime_ms,double refine_runtime_ms,double roi_selection_ms,double total_stage_runtime_ms,const std::string& base_path,const std::string& upscaled_path,const std::string& refined_path) const {
    std::ofstream f(stage_timings_csv, std::ios::app);
    f << csv_escape(profile_name) << "," << stage_index << "," << out_n << "," << observations_used << "," << new_observations_added << "," << roi_count << "," << base_runtime_ms << "," << upscale_runtime_ms << "," << refine_runtime_ms << "," << roi_selection_ms << "," << total_stage_runtime_ms << "," << csv_escape(base_path) << "," << csv_escape(upscaled_path) << "," << csv_escape(refined_path) << "\n";
  }

  void append_quality(const std::string& dataset_id, int stage_index, int reconstruction_N, const std::string& output_kind, double nmae, double mse, int observations_used, int observations_added) const {
    std::ofstream f(quality_csv, std::ios::app);
    f << csv_escape(dataset_id) << "," << stage_index << "," << reconstruction_N << "," << csv_escape(output_kind) << "," << nmae << "," << mse << "," << observations_used << "," << observations_added << "\n";
  }
};

struct StageState {
  int stage_index = 0;
  int out_n = 128;
  int roi_count = 8;
  bool coarse_done = false;
  bool upscaled_done = false;
  bool refine_done = false;
  int retries = 0;
  int observations_used = 1;
  int new_observations_added = 0;
  double coarse_runtime_ms = 0.0;
  double upscale_runtime_ms = 0.0;
  double refine_runtime_ms = 0.0;
  double roi_selection_ms = 0.0;
  bool timing_recorded = false;
  std::string base_path;
  std::string upscaled_path;
  std::string coarse_path;
  std::string refined_path;
  std::vector<uint8_t> upscaled_image_bytes;
  std::vector<uint8_t> coarse_image_bytes;
  std::vector<uint8_t> refined_image_bytes;
  std::vector<sgl::proto::RegionOfInterest> rois;
};

struct ProcessingState {
  std::string dataset_id;
  std::string dataset_csv;
  std::string preconditioned_source_path;
  std::string ring_preview_path;
  unsigned src_w = 0;
  unsigned src_h = 0;
  bool sgl_annulus_payload = false;
  std::vector<StageState> stages;
  int active_stage = 0;
};

struct FdirState {
  int unstable_cycles = 0;
  int jetson_failures = 0;
  int cooldown_cycles = 0;
  sgl::SchedulerMode mode = sgl::SchedulerMode::Nominal;
};

struct JobResult {
  bool success = false;
  std::string status;
  std::vector<sgl::proto::RegionOfInterest> rois;
  std::vector<uint8_t> image;
  double reconstruction_ms = 0.0;
  double roi_selection_ms = 0.0;
};

double roi_mean_score(const std::vector<sgl::proto::RegionOfInterest>& rois) {
  if (rois.empty()) return 0.0;
  double sum = 0.0;
  for (const auto& r : rois) sum += r.score;
  return sum / static_cast<double>(rois.size());
}

bool decode_ppm_bytes(const std::vector<uint8_t>& bytes, sgl::ImageRGBA& img) {
  if (bytes.size() < 16) return false;
  std::string s(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (s.rfind("P6\n", 0) != 0) return false;
  size_t p = 3;
  auto next_tok = [&](std::string& tok) -> bool {
    while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
    if (p >= s.size()) return false;
    size_t b = p;
    while (p < s.size() && !std::isspace(static_cast<unsigned char>(s[p]))) p++;
    tok = s.substr(b, p - b);
    return !tok.empty();
  };
  std::string tw, th, tm;
  if (!next_tok(tw) || !next_tok(th) || !next_tok(tm)) return false;
  unsigned w = static_cast<unsigned>(std::stoul(tw));
  unsigned h = static_cast<unsigned>(std::stoul(th));
  unsigned mv = static_cast<unsigned>(std::stoul(tm));
  if (mv != 255) return false;
  while (p < s.size() && std::isspace(static_cast<unsigned char>(s[p]))) p++;
  const size_t need = static_cast<size_t>(w) * h * 3;
  if (p + need > bytes.size()) return false;
  img.w = w;
  img.h = h;
  img.rgba.assign(4ull * w * h, 255);
  for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
    img.rgba[4 * i + 0] = bytes[p + 3 * i + 0];
    img.rgba[4 * i + 1] = bytes[p + 3 * i + 1];
    img.rgba[4 * i + 2] = bytes[p + 3 * i + 2];
  }
  return true;
}

sgl::ImageRGBA upscale_nearest(const sgl::ImageRGBA& in, unsigned out_n) {
  sgl::ImageRGBA out;
  out.w = out_n;
  out.h = out_n;
  out.rgba.assign(4ull * out_n * out_n, 255);
  if (in.w == 0 || in.h == 0) return out;
  for (unsigned y = 0; y < out_n; ++y) {
    const unsigned sy = std::min(in.h - 1, static_cast<unsigned>((static_cast<uint64_t>(y) * in.h) / out_n));
    for (unsigned x = 0; x < out_n; ++x) {
      const unsigned sx = std::min(in.w - 1, static_cast<unsigned>((static_cast<uint64_t>(x) * in.w) / out_n));
      const size_t si = 4ull * (static_cast<size_t>(sy) * in.w + sx);
      const size_t di = 4ull * (static_cast<size_t>(y) * out_n + x);
      out.rgba[di + 0] = in.rgba[si + 0];
      out.rgba[di + 1] = in.rgba[si + 1];
      out.rgba[di + 2] = in.rgba[si + 2];
    }
  }
  return out;
}

std::vector<uint8_t> compose_incremental_refine(const std::vector<uint8_t>& prev_refined, const std::vector<uint8_t>& stage_coarse, const std::vector<uint8_t>& stage_refined, unsigned out_n) {
  sgl::ImageRGBA prev{}, coarse{}, refined{};
  if (!decode_ppm_bytes(stage_refined, refined)) return stage_refined;
  if (!decode_ppm_bytes(stage_coarse, coarse)) return stage_refined;
  if (!decode_ppm_bytes(prev_refined, prev)) return stage_refined;
  auto up = upscale_nearest(prev, out_n);
  sgl::ImageRGBA out = refined;
  const size_t pix = static_cast<size_t>(out.w) * out.h;
  for (size_t i = 0; i < pix; ++i) {
    for (int k = 0; k < 3; ++k) {
      int delta = static_cast<int>(refined.rgba[4 * i + k]) - static_cast<int>(coarse.rgba[4 * i + k]);
      int v = static_cast<int>(up.rgba[4 * i + k]) + delta;
      out.rgba[4 * i + k] = static_cast<uint8_t>(std::clamp(v, 0, 255));
    }
  }
  return sgl::ppm_bytes(out);
}

double compute_nmae(const sgl::ImageRGBA& ref, const sgl::ImageRGBA& rec) {
  if (ref.w == 0 || ref.h == 0 || rec.w != ref.w || rec.h != ref.h) return 1.0;
  double sum = 0.0;
  const size_t n = static_cast<size_t>(ref.w) * ref.h;
  for (size_t i = 0; i < n; ++i) {
    const int dr = std::abs((int)ref.rgba[4*i+0] - (int)rec.rgba[4*i+0]);
    const int dg = std::abs((int)ref.rgba[4*i+1] - (int)rec.rgba[4*i+1]);
    const int db = std::abs((int)ref.rgba[4*i+2] - (int)rec.rgba[4*i+2]);
    sum += (dr + dg + db) / (3.0 * 255.0);
  }
  return sum / std::max<size_t>(1, n);
}

double compute_mse(const sgl::ImageRGBA& ref, const sgl::ImageRGBA& rec) {
  if (ref.w == 0 || ref.h == 0 || rec.w != ref.w || rec.h != ref.h) return 1.0;
  double sum = 0.0;
  const size_t n = static_cast<size_t>(ref.w) * ref.h;
  for (size_t i = 0; i < n; ++i) {
    const double dr = (double)ref.rgba[4*i+0] - (double)rec.rgba[4*i+0];
    const double dg = (double)ref.rgba[4*i+1] - (double)rec.rgba[4*i+1];
    const double db = (double)ref.rgba[4*i+2] - (double)rec.rgba[4*i+2];
    sum += (dr*dr + dg*dg + db*db) / (3.0 * 255.0 * 255.0);
  }
  return sum / std::max<size_t>(1, n);
}

std::vector<uint8_t> blend_refined_with_upscaled(const std::vector<uint8_t>& upscaled_bytes, const std::vector<uint8_t>& refined_bytes) {
  sgl::ImageRGBA up{}, refined{};
  if (!decode_ppm_bytes(refined_bytes, refined)) return refined_bytes;
  if (!decode_ppm_bytes(upscaled_bytes, up) || up.w != refined.w || up.h != refined.h) return refined_bytes;
  sgl::ImageRGBA out = refined;
  const size_t pix = static_cast<size_t>(out.w) * out.h;
  for (size_t i = 0; i < pix; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double u = static_cast<double>(up.rgba[4 * i + k]);
      const double r = static_cast<double>(refined.rgba[4 * i + k]);
      const double detail = r - u;
      const double v = u + 0.75 * detail;
      out.rgba[4 * i + k] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0, 255.0)));
    }
  }
  return sgl::ppm_bytes(out);
}

void append_quality_for_image(const MissionStore& store, const ProcessingState& state, const StageState& st, const std::string& output_kind, const std::vector<uint8_t>& image_bytes) {
  if (state.preconditioned_source_path.empty() || image_bytes.empty()) return;
  sgl::ImageRGBA ref{}, rec{};
  std::string qerr;
  if (sgl::read_ppm(state.preconditioned_source_path, ref, qerr) && decode_ppm_bytes(image_bytes, rec)) {
    auto refN = upscale_nearest(ref, static_cast<unsigned>(st.out_n));
    const double nmae = compute_nmae(refN, rec);
    const double mse = compute_mse(refN, rec);
    store.append_quality(state.dataset_id, st.stage_index, st.out_n, output_kind, nmae, mse, st.observations_used, st.new_observations_added);
  }
}

struct ContactTile {
  std::string label;
  sgl::ImageRGBA image;
};

const std::unordered_map<char, std::array<uint8_t, 7>>& tiny_font_5x7() {
  static const std::unordered_map<char, std::array<uint8_t, 7>> kFont{
      {' ', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}},
      {'?', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04}},
      {'0', {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E}},
      {'1', {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E}},
      {'2', {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F}},
      {'3', {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E}},
      {'4', {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02}},
      {'5', {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E}},
      {'6', {0x07, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E}},
      {'7', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08}},
      {'8', {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E}},
      {'9', {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x1C}},
      {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
      {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
      {'C', {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E}},
      {'D', {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E}},
      {'E', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F}},
      {'F', {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10}},
      {'G', {0x0E, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0E}},
      {'H', {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}},
      {'I', {0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E}},
      {'J', {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E}},
      {'K', {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11}},
      {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}},
      {'M', {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11}},
      {'N', {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11}},
      {'O', {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
      {'P', {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10}},
      {'Q', {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D}},
      {'R', {0x1E, 0x11, 0x11, 0x1E, 0x12, 0x11, 0x11}},
      {'S', {0x0E, 0x11, 0x10, 0x0E, 0x01, 0x11, 0x0E}},
      {'T', {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04}},
      {'U', {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E}},
      {'V', {0x11, 0x11, 0x11, 0x11, 0x0A, 0x0A, 0x04}},
      {'W', {0x11, 0x11, 0x11, 0x15, 0x15, 0x1B, 0x11}},
      {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}},
      {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
      {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
      {'_', {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1F}},
      {'-', {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00}},
      {'/', {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10}},
  };
  return kFont;
}

void draw_rect_alpha(sgl::ImageRGBA& img, unsigned x0, unsigned y0, unsigned x1, unsigned y1, uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
  if (img.w == 0 || img.h == 0) return;
  x0 = std::min(x0, img.w);
  y0 = std::min(y0, img.h);
  x1 = std::min(x1, img.w);
  y1 = std::min(y1, img.h);
  if (x0 >= x1 || y0 >= y1) return;
  for (unsigned y = y0; y < y1; ++y) {
    for (unsigned x = x0; x < x1; ++x) {
      const size_t i = 4ull * (static_cast<size_t>(y) * img.w + x);
      const uint16_t inv = static_cast<uint16_t>(255 - a);
      img.rgba[i + 0] = static_cast<uint8_t>((a * r + inv * img.rgba[i + 0]) / 255);
      img.rgba[i + 1] = static_cast<uint8_t>((a * g + inv * img.rgba[i + 1]) / 255);
      img.rgba[i + 2] = static_cast<uint8_t>((a * b + inv * img.rgba[i + 2]) / 255);
      img.rgba[i + 3] = 255;
    }
  }
}

void draw_text_5x7(sgl::ImageRGBA& img, unsigned x0, unsigned y0, const std::string& text, unsigned scale, uint8_t r, uint8_t g, uint8_t b) {
  if (img.w == 0 || img.h == 0 || scale == 0) return;
  const auto& font = tiny_font_5x7();
  static std::unordered_set<char> warned_unknown_chars;
  unsigned pen_x = x0;
  const unsigned glyph_w = 5 * scale;
  const unsigned glyph_h = 7 * scale;
  const unsigned adv = glyph_w + scale;
  for (char c0 : text) {
    char c = static_cast<char>(std::toupper(static_cast<unsigned char>(c0)));
    auto it = font.find(c);
    if (it == font.end()) {
      if (warned_unknown_chars.insert(c).second) {
        std::cerr << "[contact_sheet] unsupported glyph '" << c
                  << "' (0x" << std::hex << (static_cast<int>(static_cast<unsigned char>(c)))
                  << std::dec << "), rendering '?'" << std::endl;
      }
      it = font.find('?');
    }
    const auto& rows = it->second;
    for (unsigned gy = 0; gy < 7; ++gy) {
      for (unsigned gx = 0; gx < 5; ++gx) {
        if ((rows[gy] & (1u << (4 - gx))) == 0) continue;
        for (unsigned sy = 0; sy < scale; ++sy) {
          for (unsigned sx = 0; sx < scale; ++sx) {
            const unsigned x = pen_x + gx * scale + sx;
            const unsigned y = y0 + gy * scale + sy;
            if (x >= img.w || y >= img.h) continue;
            const size_t i = 4ull * (static_cast<size_t>(y) * img.w + x);
            img.rgba[i + 0] = r;
            img.rgba[i + 1] = g;
            img.rgba[i + 2] = b;
            img.rgba[i + 3] = 255;
          }
        }
      }
    }
    pen_x += adv;
    if (pen_x + glyph_w >= img.w || y0 + glyph_h >= img.h) break;
  }
}

void blit_fit_center_nearest(const sgl::ImageRGBA& src, sgl::ImageRGBA& dst, unsigned tile_x, unsigned tile_y, unsigned tile_w, unsigned tile_h) {
  if (src.w == 0 || src.h == 0 || dst.w == 0 || dst.h == 0) return;
  const double scale = std::min(static_cast<double>(tile_w) / std::max(1u, src.w), static_cast<double>(tile_h) / std::max(1u, src.h));
  const unsigned out_w = std::max(1u, std::min(tile_w, static_cast<unsigned>(std::lround(src.w * scale))));
  const unsigned out_h = std::max(1u, std::min(tile_h, static_cast<unsigned>(std::lround(src.h * scale))));
  const unsigned ox = tile_x + (tile_w - out_w) / 2;
  const unsigned oy = tile_y + (tile_h - out_h) / 2;
  for (unsigned y = 0; y < out_h; ++y) {
    const unsigned sy = std::min(src.h - 1, static_cast<unsigned>((static_cast<uint64_t>(y) * src.h) / out_h));
    for (unsigned x = 0; x < out_w; ++x) {
      const unsigned sx = std::min(src.w - 1, static_cast<unsigned>((static_cast<uint64_t>(x) * src.w) / out_w));
      const size_t si = 4ull * (static_cast<size_t>(sy) * src.w + sx);
      const size_t di = 4ull * (static_cast<size_t>(oy + y) * dst.w + (ox + x));
      dst.rgba[di + 0] = src.rgba[si + 0];
      dst.rgba[di + 1] = src.rgba[si + 1];
      dst.rgba[di + 2] = src.rgba[si + 2];
      dst.rgba[di + 3] = 255;
    }
  }
}

sgl::ImageRGBA make_contact_sheet_grid(const std::vector<ContactTile>& tiles, unsigned cols, unsigned rows, unsigned tile_w, unsigned tile_h, unsigned label_scale = 8) {
  sgl::ImageRGBA out;
  out.w = cols * tile_w;
  out.h = rows * tile_h;
  out.rgba.assign(4ull * out.w * out.h, 255);
  for (size_t i = 0; i < static_cast<size_t>(out.w) * out.h; ++i) {
    out.rgba[4 * i + 0] = 16;
    out.rgba[4 * i + 1] = 16;
    out.rgba[4 * i + 2] = 16;
    out.rgba[4 * i + 3] = 255;
  }

  const size_t max_tiles = std::min(tiles.size(), static_cast<size_t>(cols) * rows);
  for (size_t idx = 0; idx < max_tiles; ++idx) {
    const unsigned cx = static_cast<unsigned>(idx % cols);
    const unsigned cy = static_cast<unsigned>(idx / cols);
    const unsigned tx = cx * tile_w;
    const unsigned ty = cy * tile_h;
    const auto& t = tiles[idx];
    const bool has_image = (t.image.w > 0 && t.image.h > 0 && !t.image.rgba.empty());
    if (has_image) {
      blit_fit_center_nearest(t.image, out, tx, ty, tile_w, tile_h);
    } else {
      draw_rect_alpha(out, tx + 32u, ty + 32u, tx + tile_w - 32u, ty + tile_h - 32u, 96, 16, 16, 220);
      draw_text_5x7(out, tx + 48u, ty + 64u, "MISSING", std::max(1u, label_scale), 255, 220, 220);
      draw_text_5x7(out, tx + 48u, ty + 64u + 10u * label_scale, "NOT COMPLETED", std::max(1u, label_scale), 255, 220, 220);
    }

    draw_rect_alpha(out, tx, ty, tx + tile_w, ty + (label_scale * 9u), 0, 0, 0, 150);
    std::string label = t.label;
    if (!has_image) label += " [MISSING]";
    draw_text_5x7(out, tx + 24u, ty + 16u, label, label_scale, 255, 255, 255);

    // subtle tile border
    draw_rect_alpha(out, tx, ty, tx + tile_w, ty + 2u, 220, 220, 220, 255);
    draw_rect_alpha(out, tx, ty + tile_h - 2u, tx + tile_w, ty + tile_h, 220, 220, 220, 255);
    draw_rect_alpha(out, tx, ty, tx + 2u, ty + tile_h, 220, 220, 220, 255);
    draw_rect_alpha(out, tx + tile_w - 2u, ty, tx + tile_w, ty + tile_h, 220, 220, 220, 255);
  }
  return out;
}

sgl::ImageRGBA make_contact_label_test_image() {
  sgl::ImageRGBA out;
  out.w = 2048;
  out.h = 640;
  out.rgba.assign(4ull * out.w * out.h, 255);
  for (size_t i = 0; i < static_cast<size_t>(out.w) * out.h; ++i) {
    out.rgba[4 * i + 0] = 20;
    out.rgba[4 * i + 1] = 20;
    out.rgba[4 * i + 2] = 20;
    out.rgba[4 * i + 3] = 255;
  }
  const std::vector<std::string> labels{
      "PRECONDITIONED SOURCE",
      "RING PREVIEW",
      "2048 UPSCALED",
      "2048 REFINED",
      "ORIGINAL SOURCE",
  };
  unsigned y = 24;
  for (const auto& label : labels) {
    draw_text_5x7(out, 24, y, label, 8, 255, 255, 255);
    y += 120;
  }
  return out;
}

std::vector<StageState> build_stages(const sgl::Config& C) {
  std::vector<StageState> stages;
  int base = std::max(16, C.progressive_base_N);
  int mx = std::max(base, C.progressive_max_N);
  int scale = std::max(2, C.progressive_scale);
  int max_stages = std::max(1, C.progressive_max_stages);
  int n = base;
  const int obs_cfg[] = {
      std::max(1, C.observation_count_stage0),
      std::max(1, C.observation_count_stage1),
      std::max(1, C.observation_count_stage2),
      std::max(1, C.observation_count_stage3)};
  int prev_obs = 0;
  for (int i = 0; i < max_stages && n <= mx; ++i) {
    const int obs = obs_cfg[std::min(i, 3)];
    StageState st;
    st.stage_index = i;
    st.out_n = n;
    st.roi_count = std::max(1, C.roi_count + i * std::max(0, C.progressive_roi_growth));
    st.observations_used = obs;
    st.new_observations_added = std::max(0, obs - prev_obs);
    prev_obs = obs;
    stages.push_back(st);
    if (n == mx) break;
    long long next = static_cast<long long>(n) * scale;
    n = static_cast<int>(std::min<long long>(mx, next));
  }
  return stages;
}

bool all_done(const ProcessingState& s) {
  if (s.dataset_csv.empty() || s.stages.empty()) return false;
  for (const auto& st : s.stages) {
    if (s.sgl_annulus_payload && st.stage_index == 0) {
      if (!st.coarse_done) return false;
    } else if (s.sgl_annulus_payload) {
      if (!st.upscaled_done || !st.coarse_done || !st.refine_done) return false;
    } else {
      if (!st.coarse_done || !st.refine_done) return false;
    }
  }
  return true;
}

JobResult send_job(sgl::net::TcpSocket& sock, const sgl::proto::HeaderMap& headers, const std::vector<uint8_t>& payload_b, int ack_timeout_ms, int result_timeout_ms) {
  JobResult out;
  if (!sock.send_frame(sgl::proto::encode_header_block(headers), payload_b)) {
    out.status = "send_failed";
    return out;
  }
  sock.set_recv_timeout_ms(ack_timeout_ms);
  std::string hdr_ack;
  std::vector<uint8_t> payload_ack;
  if (!sock.recv_frame(hdr_ack, payload_ack)) {
    out.status = "ack_timeout";
    return out;
  }
  sgl::proto::HeaderMap ack_h;
  sgl::proto::decode_header_block(hdr_ack, ack_h);
  std::string ack_type;
  sgl::proto::get_string(ack_h, "msg_type", ack_type);
  if (ack_type != "JobAccepted") {
    out.status = "job_rejected";
    return out;
  }
  sock.set_recv_timeout_ms(result_timeout_ms);
  std::string hdr_done;
  std::vector<uint8_t> payload_done;
  if (!sock.recv_frame(hdr_done, payload_done)) {
    out.status = "result_timeout";
    return out;
  }
  sgl::proto::HeaderMap done_h;
  sgl::proto::decode_header_block(hdr_done, done_h);
  std::string done_type;
  sgl::proto::get_string(done_h, "msg_type", done_type);
  sgl::proto::get_string(done_h, "status", out.status);
  std::string recon_ms_s, roi_ms_s;
  if (sgl::proto::get_string(done_h, "reconstruction_ms", recon_ms_s)) out.reconstruction_ms = std::stod(recon_ms_s);
  if (sgl::proto::get_string(done_h, "roi_selection_ms", roi_ms_s)) out.roi_selection_ms = std::stod(roi_ms_s);
  std::string rois_s;
  if (sgl::proto::get_string(done_h, "rois", rois_s)) out.rois = sgl::proto::decode_rois(rois_s);
  out.image = std::move(payload_done);
  out.success = (done_type == "JobComplete");
  if (out.status.empty()) out.status = out.success ? "ok" : "failed";
  return out;
}

JobResult run_job_local(const sgl::proto::HeaderMap& headers, const std::vector<uint8_t>& payload_b, const sgl::Config& C) {
  JobResult out;
  std::string type_s;
  sgl::proto::get_string(headers, "msg_type", type_s);
  auto type = sgl::proto::msg_type_from_string(type_s);
  int outW = C.highres_N, outH = C.highres_N, gx = C.coarse_groups_x, gy = C.coarse_groups_y, roi_count = C.roi_count;
  int prior_roi_growth = std::max(0, C.progressive_roi_growth);
  int observation_count = 1;
  sgl::proto::get_int(headers, "out_w", outW);
  sgl::proto::get_int(headers, "out_h", outH);
  sgl::proto::get_int(headers, "coarse_groups_x", gx);
  sgl::proto::get_int(headers, "coarse_groups_y", gy);
  sgl::proto::get_int(headers, "roi_count", roi_count);
  sgl::proto::get_int(headers, "prior_roi_growth", prior_roi_growth);
  sgl::proto::get_int(headers, "observation_count", observation_count);
  std::string dataset_csv(payload_b.begin(), payload_b.end());
  sgl::ProcessResult result;
  if (type == sgl::proto::MsgType::ProcessCoarse) {
    std::string prior_rois_s;
    std::vector<sgl::proto::RegionOfInterest> prior_rois;
    if (sgl::proto::get_string(headers, "prior_rois", prior_rois_s)) prior_rois = sgl::proto::decode_rois(prior_rois_s);
    result = sgl::process_coarse_job(dataset_csv, (unsigned)outW, (unsigned)outH, gx, gy, roi_count, prior_rois, prior_roi_growth, observation_count, C.jetson_scratch_dir, C.jetson_backend, C.jetson_allow_cpu_fallback);
  } else if (type == sgl::proto::MsgType::RefineRois) {
    std::string rois_s;
    sgl::proto::get_string(headers, "rois", rois_s);
    result = sgl::process_refine_job(dataset_csv, (unsigned)outW, (unsigned)outH, gx, gy, sgl::proto::decode_rois(rois_s), observation_count, C.jetson_scratch_dir, C.jetson_backend, C.jetson_allow_cpu_fallback);
  } else {
    result.status = "unsupported";
  }
  out.success = result.success;
  out.status = result.status.empty() ? (result.success ? "ok" : "failed") : result.status;
  out.rois = std::move(result.rois);
  out.image = std::move(result.image_ppm);
  out.reconstruction_ms = result.reconstruction_ms;
  out.roi_selection_ms = result.roi_selection_ms;
  return out;
}
}  // namespace

int main(int argc, char** argv) {
  std::string cfgPath = "config/config.json";
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--config" && i + 1 < argc) cfgPath = argv[++i];
  }

  sgl::Config C;
  std::string err;
  if (!sgl::load_config_json(cfgPath, C, err)) {
    std::cerr << err << "\n";
    return 1;
  }

  if (C.profiling_mode) {
    omp_set_dynamic(0);
    const int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    omp_set_num_threads(hw_threads);
  }

  fs::create_directories(fs::path(C.out_dir) / "logs");
  fs::create_directories(fs::path(C.out_dir) / "products");
  fs::create_directories(fs::path(C.out_dir) / "datasets");
  Logger log;
  log.open((fs::path(C.out_dir) / "logs" / "sgl_pi_flight.log").string());
  log.log(LogLevel::INFO, "Pi flight software starting");

  MissionStore store;
  store.init(C.out_dir);

  sgl::ADCSSim adcs;
  sgl::CommsSim comms;
  sgl::ThermalSim thermal;
  sgl::PropulsionSim propulsion;
  sgl::PayloadSim payload;
  sgl::SourcePreconditioningConfig pre_cfg{};
  pre_cfg.enabled = C.source_preconditioning_enabled;
  pre_cfg.canvas_N = C.source_canvas_N;
  pre_cfg.object_type = C.source_object_type;
  pre_cfg.disk_fill_fraction = C.disk_fill_fraction;
  pre_cfg.extended_fill_fraction = C.extended_fill_fraction;
  pre_cfg.object_padding_fraction = C.source_object_padding_fraction;
  pre_cfg.minimum_source_margin_fraction = C.minimum_source_margin_fraction;
  pre_cfg.background_value = C.source_background_value;
  pre_cfg.brightness_normalization_mode = C.source_brightness_normalization_mode;
  pre_cfg.brightness_target_luma = C.source_brightness_target_luma;
  pre_cfg.brightness_gain_min_disk = C.source_brightness_gain_min_disk;
  pre_cfg.brightness_gain_max_disk = C.source_brightness_gain_max_disk;
  pre_cfg.brightness_gain_min_extended = C.source_brightness_gain_min_extended;
  pre_cfg.brightness_gain_max_extended = C.source_brightness_gain_max_extended;
  pre_cfg.disk_photo_center_x = C.source_disk_photo_center_x;
  pre_cfg.disk_photo_center_y = C.source_disk_photo_center_y;
  pre_cfg.disk_photo_radius_px = C.source_disk_photo_radius_px;
  pre_cfg.disk_photo_crop_half_px = C.source_disk_photo_crop_half_px;
  sgl::SglObservationConfig obs_cfg{};
  obs_cfg.ring_sensor_N = C.ring_sensor_N;
  obs_cfg.ring_radius_fraction = C.ring_radius_fraction;
  obs_cfg.ring_radial_width_px = C.ring_radial_width_px;
  obs_cfg.ring_processing_mode = C.ring_processing_mode;
  obs_cfg.ring_angular_samples = C.ring_angular_samples;
  obs_cfg.ring_radial_samples = C.ring_radial_samples;
  obs_cfg.store_full_ring_frames = C.store_full_ring_frames;
  obs_cfg.store_ring_preview_every = C.store_ring_preview_every;
  obs_cfg.store_all_full_ring_frames_debug = C.store_all_full_ring_frames_debug;
  obs_cfg.observation_count = std::max({1, C.observation_count_stage0, C.observation_count_stage1, C.observation_count_stage2, C.observation_count_stage3});
  // Keep payload ring synthesis aligned with mission output fidelity.
  // Using lowres_N here caps reconstruction detail even when progressive stages
  // run up to much higher resolutions.
  payload.configure(C.source_image, C.tile_px_x, C.tile_px_y, C.highres_N, C.ring_radius, C.ring_sigma, (fs::path(C.out_dir) / "datasets").string(), C.payload_input_mode, C.payload_fusion_alpha, pre_cfg, obs_cfg, C.reconstruction_mode);
  sgl::eps::EpsModel eps_model;
  eps_model.reset();
  FdirState fdir;

  const bool local_jetson = (C.jetson_transport == "local");
  sgl::net::TcpSocket sock;
  if (!local_jetson) {
    sock = sgl::net::connect_to(C.host, static_cast<uint16_t>(C.port), C.connect_timeout_ms);
    if (!sock.valid()) {
      log.log(LogLevel::ERROR, "connect failed to %s:%d", C.host.c_str(), C.port);
      store.append_event(0, "jetson_unavailable", "error", "Jetson TCP connection failed", C.host + ":" + std::to_string(C.port));
      return 1;
    }
    sock.set_send_timeout_ms(C.job_ack_timeout_ms);
    sock.set_recv_timeout_ms(C.job_ack_timeout_ms);
    sgl::proto::HeaderMap hello{{"msg_type", "Hello"}, {"node", "pi_flight"}};
    if (!sock.send_frame(sgl::proto::encode_header_block(hello), {})) {
      log.log(LogLevel::ERROR, "Hello send failed");
      store.append_event(0, "jetson_unavailable", "error", "Jetson hello send failed");
      return 1;
    }
    {
      std::string hdr;
      std::vector<uint8_t> p;
      if (!sock.recv_frame(hdr, p)) {
        log.log(LogLevel::ERROR, "Hello ack timeout");
        store.append_event(0, "jetson_unavailable", "error", "Jetson hello ack timeout");
        return 1;
      }
    }
  } else {
    log.log(LogLevel::INFO, "Pi flight running with local Jetson transport");
  }

  ProcessingState state;
  std::string prev_adcs_mode = adcs.mode_string();
  bool prev_tracker_valid = adcs.tracker_valid();
  bool prev_heater_active = thermal.heater_active();
  bool prev_propulsion_active = propulsion.active();
  bool prev_dataset_ready = payload.dataset_ready();
  bool prev_downlink_active = comms.downlink_active();
  auto prev_scheduler_mode = fdir.mode;
  bool prev_budget_low = false;
  std::string last_contact_sheet_dataset;
  const double budget_low_threshold = 40.0;
  const double budget_recover_threshold = 50.0;

  const bool infinite_run = (C.sim_cycles <= 0);
  for (int cycle = 0; infinite_run || cycle < C.sim_cycles; ++cycle) {
    const double dt = C.dt_s;
    adcs.sense(dt); adcs.decide(dt); adcs.act(dt);
    comms.sense(dt); comms.decide(dt); comms.act(dt);
    thermal.sense(dt); thermal.decide(dt); thermal.act(dt);
    propulsion.sense(dt); propulsion.decide(dt); propulsion.act(dt);
    payload.sense(dt); payload.decide(dt); payload.act(dt);
    for (const auto& ev : payload.drain_events()) {
      store.append_event(cycle, ev.type, ev.severity, ev.message, ev.value);
    }

    if (state.dataset_csv.empty() && payload.has_dataset()) {
      auto ds = payload.pop_dataset();
      state.dataset_id = ds.dataset_id;
      state.dataset_csv = ds.csv;
      state.preconditioned_source_path = payload.preconditioned_source_path();
      state.ring_preview_path = ds.ring_preview_path;
      state.src_w = ds.src_w;
      state.src_h = ds.src_h;
      state.sgl_annulus_payload = state.dataset_csv.rfind("format,sgl_annulus_v2", 0) == 0;
      state.stages = build_stages(C);
      state.active_stage = 0;
      log.log(LogLevel::INFO, "captured %s (%ux%u) stages=%d", state.dataset_id.c_str(), state.src_w, state.src_h, static_cast<int>(state.stages.size()));
      store.append_event(cycle, "ring_generation_timing", "info", "Ring/dataset generation runtime (ms)", std::to_string(payload.last_ring_generation_ms()));
      if (payload.ring_sensor_N() > 0) {
        store.append_event(cycle, "ring_sensor_config", "info", "Ring sensor configuration", "N=" + std::to_string(payload.ring_sensor_N()) + ",radius_px=" + std::to_string(payload.ring_radius_px()) + ",width_px=" + std::to_string(payload.ring_radial_width_px()));
        store.append_event(cycle, "ring_annulus_stats", "info", "Ring annulus active-pixel stats", "active_pixels=" + std::to_string(payload.active_annulus_pixels()) + ",active_fraction=" + std::to_string(payload.active_pixel_fraction()));
      }
    }

    double adcs_power_w = adcs.current_power_w();
    double comms_power_w = comms.current_power_w();
    double thermal_power_w = thermal.current_power_w();
    double propulsion_power_w = propulsion.current_power_w();
    double payload_power_w = payload.current_power_w();
    double noncompute_w = adcs_power_w + comms_power_w + thermal_power_w + propulsion_power_w + payload_power_w;
    bool pending_jobs = !state.dataset_csv.empty() && !all_done(state);
    double pi_draw = pending_jobs ? C.pi_active_W : C.pi_idle_W;
    double jetson_power_w = C.jetson_idle_W;
    sgl::eps::EpsInput eps_in;
    eps_in.dt_s = dt;
    eps_in.noncompute_load_w = noncompute_w;
    eps_in.reserve_w = C.reserve_margin_W;
    eps_in.safe_fraction = C.nominal_fraction;
    eps_in.pi_load_w = pi_draw;
    eps_in.jetson_load_w = jetson_power_w;
    auto eps_tel = eps_model.step(eps_in);
    double source_w = eps_tel.source_w;
    double compute_budget = eps_tel.compute_budget_w;
    double jetson_allow = std::max(0.0, compute_budget - pi_draw);
    std::string jetson_mode = "IDLE";
    std::string jetson_job_type = "none";
    bool adcs_stable = adcs.stable();
    bool stable = (!C.require_adcs_stable_for_jetson || adcs_stable) && !comms.downlink_active();
    const bool profiling_force = C.profiling_mode && C.profiling_force_full_compute;

    if (C.require_adcs_stable_for_jetson && !adcs_stable) fdir.unstable_cycles++;
    else fdir.unstable_cycles = 0;
    if (fdir.cooldown_cycles > 0) fdir.cooldown_cycles--;
    fdir.mode = sgl::decide_scheduler_mode(jetson_allow, comms.backlog_bits(), fdir.cooldown_cycles, fdir.unstable_cycles, C.jetson_refine_W);
    if (profiling_force) {
      stable = true;
      fdir.mode = sgl::SchedulerMode::Nominal;
      jetson_allow = std::max(jetson_allow, C.jetson_refine_W + C.jetson_coarse_W + 1.0);
    }
    if (fdir.mode != prev_scheduler_mode) {
      store.append_event(cycle, "scheduler_mode_changed", "info", "Scheduler mode changed", std::to_string(static_cast<int>(fdir.mode)));
      if (fdir.mode == sgl::SchedulerMode::Suspended) store.append_event(cycle, "fdir_safe_mode", "warn", "Scheduler suspended by FDIR/safety conditions");
      prev_scheduler_mode = fdir.mode;
    }

    int active_stage_n = (state.active_stage < static_cast<int>(state.stages.size())) ? state.stages[state.active_stage].out_n : -1;
    int roi_count = (state.active_stage < static_cast<int>(state.stages.size())) ? state.stages[state.active_stage].roi_count : 0;
    int processing_queue = 0;
    if (!state.stages.empty() && state.active_stage < static_cast<int>(state.stages.size())) processing_queue = static_cast<int>(state.stages.size()) - state.active_stage;

    if (!state.dataset_csv.empty() && state.active_stage < static_cast<int>(state.stages.size()) && (profiling_force || fdir.mode != sgl::SchedulerMode::Suspended)) {
      auto& st = state.stages[state.active_stage];
      std::vector<uint8_t> payload_b(state.dataset_csv.begin(), state.dataset_csv.end());
      const bool sgl_annulus_payload = state.dataset_csv.rfind("format,sgl_annulus_v2", 0) == 0;

      if (sgl_annulus_payload && st.stage_index > 0 && !st.upscaled_done && !state.stages[st.stage_index - 1].refined_image_bytes.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        sgl::ImageRGBA prev{};
        if (decode_ppm_bytes(state.stages[st.stage_index - 1].refined_image_bytes, prev)) {
          auto up = upscale_nearest(prev, static_cast<unsigned>(st.out_n));
          st.upscaled_image_bytes = sgl::ppm_bytes(up);
          st.upscaled_path = (fs::path(C.out_dir) / "products" / (state.dataset_id + "_s" + std::to_string(st.stage_index) + "_upscaled_" + std::to_string(st.out_n) + ".ppm")).string();
          write_bytes(st.upscaled_path, st.upscaled_image_bytes);
          const auto t1 = std::chrono::steady_clock::now();
          st.upscale_runtime_ms = std::chrono::duration<double,std::milli>(t1 - t0).count();
          st.upscaled_done = true;
          comms.enqueue_bits(st.upscaled_image_bytes.size() * 8ull);
          store.append_manifest(cycle, state.dataset_id, st.stage_index, "recon_upscaled", st.out_n, st.upscaled_path, st.upscaled_image_bytes.size(), st.rois.size(), roi_mean_score(st.rois), "upscaled previous reconstruction");
          store.enqueue_downlink(cycle, state.dataset_id, 2, st.upscaled_image_bytes.size() * 8ull, "recon_upscaled", st.upscaled_path);
          append_quality_for_image(store, state, st, "upscaled", st.upscaled_image_bytes);
          store.append_event(cycle, "reconstruction_upscaled", "info", "Progressive upscaled product written", st.upscaled_path);
        }
      }

      if (!st.coarse_done && stable && (profiling_force || jetson_allow >= C.jetson_coarse_W)) {
        jetson_power_w = C.jetson_coarse_W;
        jetson_mode = "ACTIVE";
        jetson_job_type = "coarse";
        store.append_event(cycle, "jetson_coarse_started", "info", "Jetson coarse job started", state.dataset_id + "_s" + std::to_string(st.stage_index));
        sgl::proto::HeaderMap h{
            {"msg_type", "ProcessCoarse"},
            {"job_id", state.dataset_id + "_s" + std::to_string(st.stage_index) + "_coarse"},
            {"out_w", std::to_string(st.out_n)},
            {"out_h", std::to_string(st.out_n)},
            {"coarse_groups_x", std::to_string(C.coarse_groups_x)},
            {"coarse_groups_y", std::to_string(C.coarse_groups_y)},
            {"roi_count", std::to_string(st.roi_count)},
            {"observation_count", std::to_string(st.observations_used)},
            {"prior_roi_growth", std::to_string(std::max(0, C.progressive_roi_growth))},
            {"prior_rois", (st.stage_index > 0) ? sgl::proto::encode_rois(state.stages[st.stage_index - 1].rois) : std::string{}}};
        JobResult jr = local_jetson ? run_job_local(h, payload_b, C) : send_job(sock, h, payload_b, C.job_ack_timeout_ms, C.job_result_timeout_ms);
        if (jr.success) {
          st.rois = jr.rois;
          st.coarse_done = true;
          st.coarse_image_bytes = jr.image;
          st.coarse_runtime_ms = jr.reconstruction_ms;
          st.roi_selection_ms = jr.roi_selection_ms;
          if (sgl_annulus_payload && st.stage_index == 0) {
            std::string out_path = (fs::path(C.out_dir) / "products" / (state.dataset_id + "_s0_base_" + std::to_string(st.out_n) + ".ppm")).string();
            write_bytes(out_path, jr.image);
            st.base_path = out_path;
            st.refined_path = out_path;
            st.refined_image_bytes = jr.image;
            comms.enqueue_bits(jr.image.size() * 8ull);
            store.append_manifest(cycle, state.dataset_id, st.stage_index, "recon_base", st.out_n, out_path, jr.image.size(), st.rois.size(), roi_mean_score(st.rois), jr.status);
            store.enqueue_downlink(cycle, state.dataset_id, 1, jr.image.size() * 8ull, "recon_base", out_path);
            append_quality_for_image(store, state, st, "base", jr.image);
          } else if (!sgl_annulus_payload) {
            std::string out_path = (fs::path(C.out_dir) / "products" / (state.dataset_id + "_s" + std::to_string(st.stage_index) + "_coarse_" + std::to_string(st.out_n) + ".ppm")).string();
            write_bytes(out_path, jr.image);
            st.coarse_path = out_path;
            comms.enqueue_bits(jr.image.size() * 8ull);
            store.append_manifest(cycle, state.dataset_id, st.stage_index, "coarse", st.out_n, out_path, jr.image.size(), st.rois.size(), roi_mean_score(st.rois), jr.status);
            store.enqueue_downlink(cycle, state.dataset_id, 2, jr.image.size() * 8ull, "coarse", out_path);
          }
          store.append_event(cycle, "jetson_coarse_completed", "info", "Jetson coarse job completed", jr.status);
        } else {
          st.retries++;
          fdir.jetson_failures++;
          log.log(LogLevel::WARN, "coarse failed stage=%d status=%s retries=%d", st.stage_index, jr.status.c_str(), st.retries);
          store.append_event(cycle, "jetson_coarse_failed", "warn", "Jetson coarse job failed", jr.status);
          if (jr.status == "send_failed" || jr.status == "ack_timeout" || jr.status == "result_timeout") {
            store.append_event(cycle, "jetson_unavailable", "warn", "Jetson transport unavailable during coarse job", jr.status);
          }
        }
      }

      bool allow_refine = profiling_force || (fdir.mode == sgl::SchedulerMode::Nominal) || (fdir.mode == sgl::SchedulerMode::Throttled && st.stage_index == 0);
      if ((!sgl_annulus_payload || st.stage_index > 0) && st.coarse_done && !st.refine_done && allow_refine && stable && (profiling_force || jetson_allow >= C.jetson_refine_W)) {
        jetson_power_w = C.jetson_refine_W;
        jetson_mode = "ACTIVE";
        jetson_job_type = "refine";
        store.append_event(cycle, "jetson_refine_started", "info", "Jetson refine job started", state.dataset_id + "_s" + std::to_string(st.stage_index));
        sgl::proto::HeaderMap h{
            {"msg_type", "RefineRois"},
            {"job_id", state.dataset_id + "_s" + std::to_string(st.stage_index) + "_refine"},
            {"out_w", std::to_string(st.out_n)},
            {"out_h", std::to_string(st.out_n)},
            {"coarse_groups_x", std::to_string(C.coarse_groups_x)},
            {"coarse_groups_y", std::to_string(C.coarse_groups_y)},
            {"observation_count", std::to_string(st.observations_used)},
            {"rois", sgl::proto::encode_rois(st.rois)}};
        JobResult jr = local_jetson ? run_job_local(h, payload_b, C) : send_job(sock, h, payload_b, C.job_ack_timeout_ms, C.job_result_timeout_ms);
        if (jr.success) {
          std::string out_path = (fs::path(C.out_dir) / "products" / (state.dataset_id + "_s" + std::to_string(st.stage_index) + "_refined_" + std::to_string(st.out_n) + ".ppm")).string();
          std::vector<uint8_t> final_img = jr.image;
          if (sgl_annulus_payload && !st.upscaled_image_bytes.empty()) {
            final_img = blend_refined_with_upscaled(st.upscaled_image_bytes, jr.image);
          } else if (!sgl_annulus_payload &&
              st.stage_index > 0 &&
              !state.stages[st.stage_index - 1].refined_image_bytes.empty() &&
              !st.coarse_image_bytes.empty()) {
            final_img = compose_incremental_refine(state.stages[st.stage_index - 1].refined_image_bytes, st.coarse_image_bytes, jr.image, static_cast<unsigned>(st.out_n));
          }
          write_bytes(out_path, final_img);
          st.refine_done = true;
          st.refine_runtime_ms = jr.reconstruction_ms;
          st.refined_path = out_path;
          st.refined_image_bytes = final_img;
          comms.enqueue_bits(final_img.size() * 8ull);
          store.append_manifest(cycle, state.dataset_id, st.stage_index, sgl_annulus_payload ? "recon_refined" : "refined", st.out_n, out_path, final_img.size(), st.rois.size(), roi_mean_score(st.rois), jr.status);
          int prio = (st.stage_index == 0) ? 1 : 3;
          store.enqueue_downlink(cycle, state.dataset_id, prio, final_img.size() * 8ull, sgl_annulus_payload ? "recon_refined" : "refined", out_path);
          store.append_event(cycle, "jetson_refine_completed", "info", "Jetson refine job completed", jr.status);
          append_quality_for_image(store, state, st, sgl_annulus_payload ? "refined" : "refined", final_img);
        } else {
          st.retries++;
          fdir.jetson_failures++;
          log.log(LogLevel::WARN, "refine failed stage=%d status=%s retries=%d", st.stage_index, jr.status.c_str(), st.retries);
          store.append_event(cycle, "jetson_refine_failed", "warn", "Jetson refine job failed", jr.status);
          if (jr.status == "send_failed" || jr.status == "ack_timeout" || jr.status == "result_timeout") {
            store.append_event(cycle, "jetson_unavailable", "warn", "Jetson transport unavailable during refine job", jr.status);
          }
        }
      }

      if (st.retries >= 3) {
        fdir.cooldown_cycles = 8;
        st.retries = 0;
        log.log(LogLevel::ERROR, "FDIR entered Jetson cooldown due to repeated failures");
        store.append_event(cycle, "fdir_warning", "error", "FDIR entered Jetson cooldown due to repeated failures");
      }

      const bool stage_complete = sgl_annulus_payload ? ((st.stage_index == 0 && st.coarse_done) || (st.stage_index > 0 && st.upscaled_done && st.coarse_done && st.refine_done)) : (st.coarse_done && st.refine_done);
      if (stage_complete && !st.timing_recorded) {
        store.append_stage_timing(C.profile_name, st.stage_index, st.out_n, st.observations_used, st.new_observations_added, static_cast<int>(st.rois.size()), st.coarse_runtime_ms, st.upscale_runtime_ms, st.refine_runtime_ms, st.roi_selection_ms, st.coarse_runtime_ms + st.upscale_runtime_ms + st.refine_runtime_ms, st.base_path, st.upscaled_path, st.refined_path);
        st.timing_recorded = true;
        if (state.active_stage + 1 < static_cast<int>(state.stages.size())) state.active_stage++;
      }
    }

    if (!state.dataset_id.empty() && all_done(state) && state.dataset_id != last_contact_sheet_dataset) {
      std::vector<ContactTile> tiles;
      std::string e;
      auto load_ppm = [&](const std::string& p) -> sgl::ImageRGBA {
        sgl::ImageRGBA out{};
        if (!p.empty()) {
          std::string ee;
          sgl::read_image_auto(p, out, ee);
        }
        return out;
      };
      auto find_stage_path = [&](int out_n, const std::string& kind) -> std::string {
        for (const auto& st : state.stages) {
          if (st.out_n != out_n) continue;
          if (kind == "base") return st.base_path;
          if (kind == "upscaled") return st.upscaled_path;
          if (kind == "refined") return st.refined_path;
        }
        return "";
      };

      std::string original_source_path = C.source_image;
      if (!original_source_path.empty() && !fs::exists(original_source_path)) {
        fs::path p = fs::path(C.out_dir) / original_source_path;
        if (fs::exists(p)) original_source_path = p.string();
      }
      if (!original_source_path.empty() && !fs::exists(original_source_path)) {
        fs::path p = fs::path(state.preconditioned_source_path).parent_path() / "raw_capture_from_file.ppm";
        if (fs::exists(p)) original_source_path = p.string();
      }

      tiles.push_back(ContactTile{"PRECONDITIONED SOURCE", load_ppm(state.preconditioned_source_path)});
      tiles.push_back(ContactTile{"RING PREVIEW", load_ppm(state.ring_preview_path)});
      tiles.push_back(ContactTile{"128 BASE", load_ppm(find_stage_path(128, "base"))});
      tiles.push_back(ContactTile{"256 UPSCALED", load_ppm(find_stage_path(256, "upscaled"))});
      tiles.push_back(ContactTile{"256 REFINED", load_ppm(find_stage_path(256, "refined"))});
      tiles.push_back(ContactTile{"512 UPSCALED", load_ppm(find_stage_path(512, "upscaled"))});
      tiles.push_back(ContactTile{"512 REFINED", load_ppm(find_stage_path(512, "refined"))});
      tiles.push_back(ContactTile{"1024 UPSCALED", load_ppm(find_stage_path(1024, "upscaled"))});
      tiles.push_back(ContactTile{"1024 REFINED", load_ppm(find_stage_path(1024, "refined"))});
      tiles.push_back(ContactTile{"2048 UPSCALED", load_ppm(find_stage_path(2048, "upscaled"))});
      tiles.push_back(ContactTile{"2048 REFINED", load_ppm(find_stage_path(2048, "refined"))});
      tiles.push_back(ContactTile{"ORIGINAL SOURCE", load_ppm(original_source_path)});

      auto sheet = make_contact_sheet_grid(tiles, 3, 4, 2048, 2048, 8);
      const std::string sheet_path = (fs::path(C.out_dir) / "products" / "reconstruction_contact_sheet.ppm").string();
      if (sgl::write_ppm(sheet_path, sheet)) {
        // Optional lightweight preview for quick local inspection.
        auto preview = make_contact_sheet_grid(tiles, 3, 4, 512, 512, 2);
        const std::string preview_path = (fs::path(C.out_dir) / "products" / "reconstruction_contact_sheet_preview.ppm").string();
        sgl::write_ppm(preview_path, preview);
        const std::string label_test_path = (fs::path(C.out_dir) / "products" / "contact_label_test.ppm").string();
        sgl::write_ppm(label_test_path, make_contact_label_test_image());
        store.append_event(cycle, "reconstruction_contact_sheet_ready", "info", "Reconstruction contact sheet generated", sheet_path);
      }
      last_contact_sheet_dataset = state.dataset_id;
    }

    if (fdir.mode == sgl::SchedulerMode::Suspended) jetson_mode = "SUSPENDED";
    else if (!stable || jetson_allow < C.jetson_coarse_W) jetson_mode = "THROTTLED";

    eps_in.jetson_load_w = jetson_power_w;
    auto eps_bus = eps_model.evaluate(eps_in);
    double total_bus_load_w = eps_bus.total_bus_load_w;
    const std::string adcs_mode = adcs.mode_string();
    if (adcs_mode != prev_adcs_mode) {
      if (adcs_mode == "CORRECTING") store.append_event(cycle, "adcs_correction_started", "info", "ADCS correction started");
      if (prev_adcs_mode == "CORRECTING" && adcs_mode != "CORRECTING") store.append_event(cycle, "adcs_correction_stopped", "info", "ADCS correction stopped");
      prev_adcs_mode = adcs_mode;
    }
    if (adcs.tracker_valid() != prev_tracker_valid) {
      store.append_event(cycle, adcs.tracker_valid() ? "tracker_recovered" : "tracker_degraded", adcs.tracker_valid() ? "info" : "warn", adcs.tracker_valid() ? "Tracker recovered" : "Tracker degraded");
      prev_tracker_valid = adcs.tracker_valid();
    }
    if (thermal.heater_active() != prev_heater_active) {
      store.append_event(cycle, thermal.heater_active() ? "heater_activated" : "heater_deactivated", "info", thermal.heater_active() ? "Thermal heater activated" : "Thermal heater deactivated");
      prev_heater_active = thermal.heater_active();
    }
    if (propulsion.active() != prev_propulsion_active) {
      store.append_event(cycle, propulsion.active() ? "propulsion_burn_started" : "propulsion_burn_stopped", "info", propulsion.active() ? "Propulsion burn started" : "Propulsion burn stopped");
      prev_propulsion_active = propulsion.active();
    }
    if (payload.dataset_ready() && !prev_dataset_ready) {
      store.append_event(cycle, "payload_dataset_ready", "info", "Payload dataset ready", payload.last_dataset_id());
    }
    prev_dataset_ready = payload.dataset_ready();
    if (comms.downlink_active() != prev_downlink_active) {
      store.append_event(cycle, comms.downlink_active() ? "downlink_active" : "downlink_inactive", "info", comms.downlink_active() ? "Downlink became active" : "Downlink became inactive");
      prev_downlink_active = comms.downlink_active();
    }
    bool budget_low = compute_budget < budget_low_threshold;
    if (budget_low && !prev_budget_low) store.append_event(cycle, "compute_budget_low", "warn", "Compute budget dropped below threshold", std::to_string(compute_budget));
    if (!budget_low && prev_budget_low && compute_budget > budget_recover_threshold) store.append_event(cycle, "compute_budget_recovered", "info", "Compute budget recovered", std::to_string(compute_budget));
    prev_budget_low = budget_low;
    log.log(LogLevel::INFO, "cycle=%d source=%.1f bus=%.1f noncompute=%.1f compute=%.1f jetson_allow=%.1f mode=%d cooldown=%d adcs=%s err_truth=%.3f err_est=%.3f conf=%.2f valid=%d stars=%u comms=%s backlog=%zu jetson_mode=%s job=%s", cycle, source_w, total_bus_load_w, noncompute_w, compute_budget, jetson_allow, static_cast<int>(fdir.mode), fdir.cooldown_cycles, adcs_mode.c_str(), adcs.truth_pointing_error_deg(), adcs.est_pointing_error_deg(), adcs.tracker_confidence(), adcs.tracker_valid() ? 1 : 0, adcs.tracked_stars(), comms.mode_string().c_str(), comms.backlog_bits(), jetson_mode.c_str(), jetson_job_type.c_str());
    store.append_telemetry(cycle, source_w, C.reserve_margin_W, total_bus_load_w, noncompute_w, compute_budget, jetson_allow, static_cast<int>(fdir.mode), adcs_mode, adcs_power_w, adcs.wheel_power_w(), comms_power_w, thermal_power_w, propulsion_power_w, payload_power_w, pi_draw, jetson_power_w, jetson_mode, jetson_job_type, adcs.truth_pointing_error_deg(), adcs.est_pointing_error_deg(), adcs.tracker_confidence(), adcs.tracker_valid(), adcs.tracked_stars(), comms.mode_string(), comms.backlog_bits(), payload.mode_string(), payload.active(), payload.dataset_ready(), payload.last_dataset_id(), payload.dataset_count(), payload.acquisition_stage(), state.active_stage, active_stage_n, roi_count, processing_queue, thermal.mode_string(), thermal.heater_active(), thermal.temperature_c(), propulsion.mode_string(), propulsion.active(), propulsion.thrust_n(), payload.camera_mode(), payload.camera_frame_ready(), payload.alignment_valid(), payload.alignment_score(), payload.blur_score(), payload.brightness_mean(), payload.contrast_score(), payload.raw_capture_path(), payload.rectified_image_path(), payload.preconditioned_source_path(), payload.source_object_type_detected(), payload.source_bbox_string(), payload.source_fill_fraction_used(), payload.source_margin_fraction(), payload.source_clipping_guard_triggered(), payload.preconditioning_method(), payload.alignment_method(), payload.ring_sensor_N(), payload.ring_radius_px(), payload.ring_radial_width_px(), payload.active_annulus_pixels(), payload.active_pixel_fraction(), payload.ring_angular_samples(), payload.ring_radial_samples(), payload.ring_processing_mode());

    if (!C.profiling_mode) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }

  if (!local_jetson) {
    sgl::proto::HeaderMap bye{{"msg_type", "Shutdown"}};
    sock.set_send_timeout_ms(C.job_ack_timeout_ms);
    sock.set_recv_timeout_ms(C.job_ack_timeout_ms);
    if (sock.send_frame(sgl::proto::encode_header_block(bye), {})) {
      std::string hdr;
      std::vector<uint8_t> payload;
      sock.recv_frame(hdr, payload);
    }
  }
  log.log(LogLevel::INFO, "Pi flight software exiting");
  return 0;
}
