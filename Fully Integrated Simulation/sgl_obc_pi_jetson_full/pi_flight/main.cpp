#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include "../common/config.hpp"
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
  bool initialized = false;

  void init(const std::string& out_dir) {
    const fs::path root = fs::path(out_dir) / "mission_store";
    fs::create_directories(root);
    manifest_csv = (root / "products_manifest.csv").string();
    downlink_csv = (root / "downlink_queue.csv").string();
    telemetry_csv = (root / "telemetry_cycles.csv").string();
    events_csv = (root / "events.csv").string();
    stage_timings_csv = (root / "progressive_stage_timings.csv").string();
    if (!fs::exists(manifest_csv)) {
      std::ofstream f(manifest_csv);
      f << "cycle,dataset_id,stage,kind,out_n,path,bytes,roi_count,roi_score_mean,status\n";
    }
    if (!fs::exists(downlink_csv)) {
      std::ofstream f(downlink_csv);
      f << "cycle,dataset_id,priority,bits,kind,path,status\n";
    }
    if (!fs::exists(telemetry_csv)) {
      std::ofstream f(telemetry_csv);
      f << "cycle,source_w,reserve_w,total_bus_load_w,noncompute_w,compute_budget_w,jetson_allow_w,scheduler_mode,adcs_mode,adcs_power_w,wheel_power_w,comms_power_w,thermal_power_w,propulsion_power_w,payload_power_w,pi_power_w,jetson_power_w,jetson_mode,jetson_job_type,truth_pointing_err_deg,est_pointing_err_deg,tracker_conf,tracker_valid,tracked_stars,comms_mode,comms_backlog_bits,payload_mode,payload_active,dataset_ready,dataset_id,dataset_count,acquisition_stage,active_stage,active_stage_n,roi_count,processing_queue,thermal_mode,heater_active,thermal_temp_c,propulsion_mode,propulsion_active,propulsion_thrust_n,camera_mode,camera_frame_ready,alignment_valid,alignment_score,blur_score,brightness_mean,contrast_score,raw_capture_path,rectified_image_path\n";
    }
    if (!fs::exists(events_csv)) {
      std::ofstream f(events_csv);
      f << "cycle,event_type,severity,message,value\n";
    }
    if (!fs::exists(stage_timings_csv)) {
      std::ofstream f(stage_timings_csv);
      f << "profile_name,stage_index,out_n,observations_used,new_observations_added,roi_count,coarse_runtime_ms,refine_runtime_ms,roi_selection_ms,total_stage_runtime_ms,coarse_path,refined_path\n";
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

  void append_telemetry(int cycle,double source_w,double reserve_w,double total_bus_load_w,double noncompute_w,double compute_budget,double jetson_allow,int scheduler_mode,const std::string& adcs_mode,double adcs_power_w,double wheel_power_w,double comms_power_w,double thermal_power_w,double propulsion_power_w,double payload_power_w,double pi_power_w,double jetson_power_w,const std::string& jetson_mode,const std::string& jetson_job_type,double truth_pointing_err,double est_pointing_err,double tracker_conf,bool tracker_valid,uint32_t tracked_stars,const std::string& comms_mode,size_t backlog_bits,const std::string& payload_mode,bool payload_active,bool dataset_ready,const std::string& dataset_id,int dataset_count,int acquisition_stage,int active_stage,int active_stage_n,int roi_count,int processing_queue,const std::string& thermal_mode,bool heater_active,double thermal_temp_c,const std::string& propulsion_mode,bool propulsion_active,double propulsion_thrust_n,const std::string& camera_mode,bool camera_frame_ready,bool alignment_valid,double alignment_score,double blur_score,double brightness_mean,double contrast_score,const std::string& raw_capture_path,const std::string& rectified_image_path) const {
    std::ofstream f(telemetry_csv, std::ios::app);
    f << cycle << "," << source_w << "," << reserve_w << "," << total_bus_load_w << "," << noncompute_w << "," << compute_budget << "," << jetson_allow << "," << scheduler_mode << "," << csv_escape(adcs_mode) << "," << adcs_power_w << "," << wheel_power_w << "," << comms_power_w << "," << thermal_power_w << "," << propulsion_power_w << "," << payload_power_w << "," << pi_power_w << "," << jetson_power_w << "," << csv_escape(jetson_mode) << "," << csv_escape(jetson_job_type) << "," << truth_pointing_err << "," << est_pointing_err << "," << tracker_conf << "," << (tracker_valid ? 1 : 0) << "," << tracked_stars << "," << csv_escape(comms_mode) << "," << backlog_bits << "," << csv_escape(payload_mode) << "," << (payload_active ? 1 : 0) << "," << (dataset_ready ? 1 : 0) << "," << csv_escape(dataset_id) << "," << dataset_count << "," << acquisition_stage << "," << active_stage << "," << active_stage_n << "," << roi_count << "," << processing_queue << "," << csv_escape(thermal_mode) << "," << (heater_active ? 1 : 0) << "," << thermal_temp_c << "," << csv_escape(propulsion_mode) << "," << (propulsion_active ? 1 : 0) << "," << propulsion_thrust_n << "," << csv_escape(camera_mode) << "," << (camera_frame_ready ? 1 : 0) << "," << (alignment_valid ? 1 : 0) << "," << alignment_score << "," << blur_score << "," << brightness_mean << "," << contrast_score << "," << csv_escape(raw_capture_path) << "," << csv_escape(rectified_image_path) << "\n";
  }

  void append_event(int cycle, const std::string& event_type, const std::string& severity, const std::string& message, const std::string& value = "") const {
    std::ofstream f(events_csv, std::ios::app);
    f << cycle << "," << csv_escape(event_type) << "," << csv_escape(severity) << "," << csv_escape(message) << "," << csv_escape(value) << "\n";
  }

  void append_stage_timing(const std::string& profile_name,int stage_index,int out_n,int observations_used,int new_observations_added,int roi_count,double coarse_runtime_ms,double refine_runtime_ms,double roi_selection_ms,double total_stage_runtime_ms,const std::string& coarse_path,const std::string& refined_path) const {
    std::ofstream f(stage_timings_csv, std::ios::app);
    f << csv_escape(profile_name) << "," << stage_index << "," << out_n << "," << observations_used << "," << new_observations_added << "," << roi_count << "," << coarse_runtime_ms << "," << refine_runtime_ms << "," << roi_selection_ms << "," << total_stage_runtime_ms << "," << csv_escape(coarse_path) << "," << csv_escape(refined_path) << "\n";
  }
};

struct StageState {
  int stage_index = 0;
  int out_n = 128;
  int roi_count = 8;
  bool coarse_done = false;
  bool refine_done = false;
  int retries = 0;
  int observations_used = 1;
  int new_observations_added = 0;
  double coarse_runtime_ms = 0.0;
  double refine_runtime_ms = 0.0;
  double roi_selection_ms = 0.0;
  bool timing_recorded = false;
  std::string coarse_path;
  std::string refined_path;
  std::vector<sgl::proto::RegionOfInterest> rois;
};

struct ProcessingState {
  std::string dataset_id;
  std::string dataset_csv;
  unsigned src_w = 0;
  unsigned src_h = 0;
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
    if (!st.coarse_done || !st.refine_done) return false;
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
  payload.configure(C.source_image, C.tile_px_x, C.tile_px_y, C.lowres_N, C.ring_radius, C.ring_sigma, (fs::path(C.out_dir) / "datasets").string(), C.payload_input_mode, C.payload_fusion_alpha);
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
  const double budget_low_threshold = 40.0;
  const double budget_recover_threshold = 50.0;

  for (int cycle = 0; cycle < C.sim_cycles; ++cycle) {
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
      state.src_w = ds.src_w;
      state.src_h = ds.src_h;
      state.stages = build_stages(C);
      state.active_stage = 0;
      log.log(LogLevel::INFO, "captured %s (%ux%u) stages=%d", state.dataset_id.c_str(), state.src_w, state.src_h, static_cast<int>(state.stages.size()));
      store.append_event(cycle, "ring_generation_timing", "info", "Ring/dataset generation runtime (ms)", std::to_string(payload.last_ring_generation_ms()));
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

    if (C.require_adcs_stable_for_jetson && !adcs_stable) fdir.unstable_cycles++;
    else fdir.unstable_cycles = 0;
    if (fdir.cooldown_cycles > 0) fdir.cooldown_cycles--;
    fdir.mode = sgl::decide_scheduler_mode(jetson_allow, comms.backlog_bits(), fdir.cooldown_cycles, fdir.unstable_cycles, C.jetson_refine_W);
    if (fdir.mode != prev_scheduler_mode) {
      store.append_event(cycle, "scheduler_mode_changed", "info", "Scheduler mode changed", std::to_string(static_cast<int>(fdir.mode)));
      if (fdir.mode == sgl::SchedulerMode::Suspended) store.append_event(cycle, "fdir_safe_mode", "warn", "Scheduler suspended by FDIR/safety conditions");
      prev_scheduler_mode = fdir.mode;
    }

    int active_stage_n = (state.active_stage < static_cast<int>(state.stages.size())) ? state.stages[state.active_stage].out_n : -1;
    int roi_count = (state.active_stage < static_cast<int>(state.stages.size())) ? state.stages[state.active_stage].roi_count : 0;
    int processing_queue = 0;
    if (!state.stages.empty() && state.active_stage < static_cast<int>(state.stages.size())) processing_queue = static_cast<int>(state.stages.size()) - state.active_stage;

    if (!state.dataset_csv.empty() && state.active_stage < static_cast<int>(state.stages.size()) && fdir.mode != sgl::SchedulerMode::Suspended) {
      auto& st = state.stages[state.active_stage];
      std::vector<uint8_t> payload_b(state.dataset_csv.begin(), state.dataset_csv.end());

      if (!st.coarse_done && stable && jetson_allow >= C.jetson_coarse_W) {
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
          std::string out_path = (fs::path(C.out_dir) / "products" / (state.dataset_id + "_s" + std::to_string(st.stage_index) + "_coarse_" + std::to_string(st.out_n) + ".ppm")).string();
          write_bytes(out_path, jr.image);
          st.coarse_done = true;
          st.coarse_runtime_ms = jr.reconstruction_ms;
          st.roi_selection_ms = jr.roi_selection_ms;
          st.coarse_path = out_path;
          comms.enqueue_bits(jr.image.size() * 8ull);
          store.append_manifest(cycle, state.dataset_id, st.stage_index, "coarse", st.out_n, out_path, jr.image.size(), st.rois.size(), roi_mean_score(st.rois), jr.status);
          store.enqueue_downlink(cycle, state.dataset_id, 2, jr.image.size() * 8ull, "coarse", out_path);
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

      bool allow_refine = (fdir.mode == sgl::SchedulerMode::Nominal) || (fdir.mode == sgl::SchedulerMode::Throttled && st.stage_index == 0);
      if (st.coarse_done && !st.refine_done && allow_refine && stable && jetson_allow >= C.jetson_refine_W) {
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
          write_bytes(out_path, jr.image);
          st.refine_done = true;
          st.refine_runtime_ms = jr.reconstruction_ms;
          st.refined_path = out_path;
          comms.enqueue_bits(jr.image.size() * 8ull);
          store.append_manifest(cycle, state.dataset_id, st.stage_index, "refined", st.out_n, out_path, jr.image.size(), st.rois.size(), roi_mean_score(st.rois), jr.status);
          int prio = (st.stage_index == 0) ? 1 : 3;
          store.enqueue_downlink(cycle, state.dataset_id, prio, jr.image.size() * 8ull, "refined", out_path);
          store.append_event(cycle, "jetson_refine_completed", "info", "Jetson refine job completed", jr.status);
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

      if (st.coarse_done && st.refine_done && !st.timing_recorded) {
        store.append_stage_timing(C.profile_name, st.stage_index, st.out_n, st.observations_used, st.new_observations_added, static_cast<int>(st.rois.size()), st.coarse_runtime_ms, st.refine_runtime_ms, st.roi_selection_ms, st.coarse_runtime_ms + st.refine_runtime_ms, st.coarse_path, st.refined_path);
        st.timing_recorded = true;
        if (state.active_stage + 1 < static_cast<int>(state.stages.size())) state.active_stage++;
      }
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
    store.append_telemetry(cycle, source_w, C.reserve_margin_W, total_bus_load_w, noncompute_w, compute_budget, jetson_allow, static_cast<int>(fdir.mode), adcs_mode, adcs_power_w, adcs.wheel_power_w(), comms_power_w, thermal_power_w, propulsion_power_w, payload_power_w, pi_draw, jetson_power_w, jetson_mode, jetson_job_type, adcs.truth_pointing_error_deg(), adcs.est_pointing_error_deg(), adcs.tracker_confidence(), adcs.tracker_valid(), adcs.tracked_stars(), comms.mode_string(), comms.backlog_bits(), payload.mode_string(), payload.active(), payload.dataset_ready(), payload.last_dataset_id(), payload.dataset_count(), payload.acquisition_stage(), state.active_stage, active_stage_n, roi_count, processing_queue, thermal.mode_string(), thermal.heater_active(), thermal.temperature_c(), propulsion.mode_string(), propulsion.active(), propulsion.thrust_n(), payload.camera_mode(), payload.camera_frame_ready(), payload.alignment_valid(), payload.alignment_score(), payload.blur_score(), payload.brightness_mean(), payload.contrast_score(), payload.raw_capture_path(), payload.rectified_image_path());

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
