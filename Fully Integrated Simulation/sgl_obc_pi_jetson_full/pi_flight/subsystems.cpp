#include "subsystems.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <utility>
#include "../common/reconstruction.hpp"
namespace fs = std::filesystem;
namespace sgl {
namespace {
struct CaptureQuality {
  double blur_score = 0.0;
  double brightness_mean = 0.0;
  double contrast_score = 0.0;
};

static inline double clamp01(double v) { return std::clamp(v, 0.0, 1.0); }

static inline double luminance(const uint8_t* p) {
  return (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) / 255.0;
}

CaptureQuality compute_quality(const ImageRGBA& img) {
  CaptureQuality q{};
  if (img.w == 0 || img.h == 0 || img.rgba.empty()) return q;
  const size_t n = static_cast<size_t>(img.w) * img.h;
  std::vector<double> lum(n, 0.0);
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i) {
    lum[i] = luminance(&img.rgba[4 * i]);
    mean += lum[i];
  }
  mean /= std::max<size_t>(1, n);
  double var = 0.0;
  for (double v : lum) {
    const double d = v - mean;
    var += d * d;
  }
  var /= std::max<size_t>(1, n);

  double lap_mean = 0.0;
  double lap_sq_mean = 0.0;
  size_t lap_n = 0;
  for (unsigned y = 1; y + 1 < img.h; ++y) {
    for (unsigned x = 1; x + 1 < img.w; ++x) {
      const double c = lum[static_cast<size_t>(y) * img.w + x];
      const double l = lum[static_cast<size_t>(y) * img.w + (x - 1)];
      const double r = lum[static_cast<size_t>(y) * img.w + (x + 1)];
      const double u = lum[static_cast<size_t>(y - 1) * img.w + x];
      const double d = lum[static_cast<size_t>(y + 1) * img.w + x];
      const double lap = 4.0 * c - l - r - u - d;
      lap_mean += lap;
      lap_sq_mean += lap * lap;
      lap_n++;
    }
  }
  lap_mean /= std::max<size_t>(1, lap_n);
  lap_sq_mean /= std::max<size_t>(1, lap_n);
  q.blur_score = std::max(0.0, lap_sq_mean - lap_mean * lap_mean);
  q.brightness_mean = mean;
  q.contrast_score = std::sqrt(std::max(0.0, var));
  return q;
}

ImageRGBA center_crop_square(const ImageRGBA& in) {
  ImageRGBA out;
  const unsigned s = std::max(1u, std::min(in.w, in.h));
  const unsigned ox = (in.w > s) ? (in.w - s) / 2 : 0;
  const unsigned oy = (in.h > s) ? (in.h - s) / 2 : 0;
  out.w = s;
  out.h = s;
  out.rgba.assign(4ull * s * s, 255);
  for (unsigned y = 0; y < s; ++y) {
    for (unsigned x = 0; x < s; ++x) {
      const size_t si = 4ull * ((static_cast<size_t>(oy + y) * in.w) + (ox + x));
      const size_t di = 4ull * (static_cast<size_t>(y) * s + x);
      out.rgba[di] = in.rgba[si];
      out.rgba[di + 1] = in.rgba[si + 1];
      out.rgba[di + 2] = in.rgba[si + 2];
      out.rgba[di + 3] = 255;
    }
  }
  return out;
}

bool detect_content_bbox(const ImageRGBA& in, unsigned& x0, unsigned& y0, unsigned& x1, unsigned& y1, double& confidence) {
  confidence = 0.0;
  if (in.w == 0 || in.h == 0 || in.rgba.empty()) return false;
  std::vector<double> lum(static_cast<size_t>(in.w) * in.h, 0.0);
  double lmin = 1.0;
  double lmax = 0.0;
  for (size_t i = 0; i < lum.size(); ++i) {
    lum[i] = luminance(&in.rgba[4 * i]);
    lmin = std::min(lmin, lum[i]);
    lmax = std::max(lmax, lum[i]);
  }
  const double range = std::max(1e-5, lmax - lmin);
  const double bg = 0.5 * (lmax + lmin);
  const double thr = std::min(0.18, 0.28 * range + 0.03);
  unsigned bx0 = in.w, by0 = in.h, bx1 = 0, by1 = 0;
  size_t cnt = 0;
  for (unsigned y = 0; y < in.h; ++y) {
    for (unsigned x = 0; x < in.w; ++x) {
      const double d = std::fabs(lum[static_cast<size_t>(y) * in.w + x] - bg);
      if (d < thr) continue;
      bx0 = std::min(bx0, x);
      by0 = std::min(by0, y);
      bx1 = std::max(bx1, x);
      by1 = std::max(by1, y);
      cnt++;
    }
  }
  if (cnt < static_cast<size_t>(in.w * in.h) / 250) return false;
  const unsigned w = (bx1 >= bx0) ? (bx1 - bx0 + 1) : 0;
  const unsigned h = (by1 >= by0) ? (by1 - by0 + 1) : 0;
  if (w < in.w / 8 || h < in.h / 8) return false;
  x0 = bx0;
  y0 = by0;
  x1 = bx1;
  y1 = by1;
  confidence = std::clamp(static_cast<double>(cnt) / static_cast<double>(in.w * in.h) * 6.0, 0.0, 1.0);
  return true;
}

ImageRGBA crop_bbox_square(const ImageRGBA& in, unsigned x0, unsigned y0, unsigned x1, unsigned y1) {
  if (x1 < x0 || y1 < y0 || in.w == 0 || in.h == 0) return center_crop_square(in);
  const unsigned bw = x1 - x0 + 1;
  const unsigned bh = y1 - y0 + 1;
  const unsigned side = std::max(1u, std::max(bw, bh));
  const double cx = 0.5 * (x0 + x1);
  const double cy = 0.5 * (y0 + y1);
  int sx0 = static_cast<int>(std::lround(cx - 0.5 * side));
  int sy0 = static_cast<int>(std::lround(cy - 0.5 * side));
  sx0 = std::clamp(sx0, 0, std::max(0, static_cast<int>(in.w) - static_cast<int>(side)));
  sy0 = std::clamp(sy0, 0, std::max(0, static_cast<int>(in.h) - static_cast<int>(side)));
  const unsigned ux0 = static_cast<unsigned>(std::max(0, sx0));
  const unsigned uy0 = static_cast<unsigned>(std::max(0, sy0));
  ImageRGBA out;
  out.w = side;
  out.h = side;
  out.rgba.assign(4ull * side * side, 255);
  for (unsigned y = 0; y < side; ++y) {
    for (unsigned x = 0; x < side; ++x) {
      const unsigned sx = std::min(in.w - 1, ux0 + x);
      const unsigned sy = std::min(in.h - 1, uy0 + y);
      const size_t si = 4ull * (static_cast<size_t>(sy) * in.w + sx);
      const size_t di = 4ull * (static_cast<size_t>(y) * side + x);
      out.rgba[di] = in.rgba[si];
      out.rgba[di + 1] = in.rgba[si + 1];
      out.rgba[di + 2] = in.rgba[si + 2];
      out.rgba[di + 3] = 255;
    }
  }
  return out;
}

void draw_bbox_overlay(ImageRGBA& img, unsigned x0, unsigned y0, unsigned x1, unsigned y1, uint8_t r, uint8_t g, uint8_t b) {
  if (img.w == 0 || img.h == 0) return;
  x0 = std::min(x0, img.w - 1);
  x1 = std::min(x1, img.w - 1);
  y0 = std::min(y0, img.h - 1);
  y1 = std::min(y1, img.h - 1);
  for (unsigned x = x0; x <= x1; ++x) {
    size_t i0 = 4ull * (static_cast<size_t>(y0) * img.w + x);
    size_t i1 = 4ull * (static_cast<size_t>(y1) * img.w + x);
    img.rgba[i0] = r; img.rgba[i0 + 1] = g; img.rgba[i0 + 2] = b;
    img.rgba[i1] = r; img.rgba[i1 + 1] = g; img.rgba[i1 + 2] = b;
  }
  for (unsigned y = y0; y <= y1; ++y) {
    size_t i0 = 4ull * (static_cast<size_t>(y) * img.w + x0);
    size_t i1 = 4ull * (static_cast<size_t>(y) * img.w + x1);
    img.rgba[i0] = r; img.rgba[i0 + 1] = g; img.rgba[i0 + 2] = b;
    img.rgba[i1] = r; img.rgba[i1 + 1] = g; img.rgba[i1 + 2] = b;
  }
}

bool detect_corner_markers(const ImageRGBA& in, std::array<std::pair<double, double>, 4>& pts, double& score) {
  score = 0.0;
  if (in.w < 32 || in.h < 32) return false;
  const unsigned mx = std::max(8u, in.w / 5);
  const unsigned my = std::max(8u, in.h / 5);
  struct Box { unsigned x0, x1, y0, y1; };
  const std::array<Box, 4> boxes{
      Box{0, mx, 0, my},
      Box{in.w - mx, in.w, 0, my},
      Box{in.w - mx, in.w, in.h - my, in.h},
      Box{0, mx, in.h - my, in.h},
  };

  double total_score = 0.0;
  for (size_t bi = 0; bi < boxes.size(); ++bi) {
    const auto& b = boxes[bi];
    double min_l = 1.0;
    double mean_l = 0.0;
    size_t area = 0;
    for (unsigned y = b.y0; y < b.y1; ++y) {
      for (unsigned x = b.x0; x < b.x1; ++x) {
        const double l = luminance(&in.rgba[4ull * (static_cast<size_t>(y) * in.w + x)]);
        min_l = std::min(min_l, l);
        mean_l += l;
        area++;
      }
    }
    if (area == 0) return false;
    mean_l /= static_cast<double>(area);
    const double thr = std::min(0.35, min_l + 0.10);
    double wx = 0.0, wy = 0.0, ws = 0.0;
    size_t cnt = 0;
    for (unsigned y = b.y0; y < b.y1; ++y) {
      for (unsigned x = b.x0; x < b.x1; ++x) {
        const double l = luminance(&in.rgba[4ull * (static_cast<size_t>(y) * in.w + x)]);
        if (l > thr) continue;
        const double w = std::max(0.01, thr - l);
        wx += w * x;
        wy += w * y;
        ws += w;
        cnt++;
      }
    }
    const double frac = static_cast<double>(cnt) / static_cast<double>(area);
    if (ws <= 1e-9 || frac < 0.001) return false;
    pts[bi] = {wx / ws, wy / ws};
    const double darkness = clamp01((mean_l - min_l) / 0.4);
    const double coverage = clamp01(frac / 0.03);
    total_score += 0.5 * darkness + 0.5 * coverage;
  }
  score = total_score / 4.0;
  return score > 0.15;
}

bool solve_8x8(double A[8][8], double b[8], double x[8]) {
  for (int i = 0; i < 8; ++i) {
    int piv = i;
    for (int r = i + 1; r < 8; ++r) {
      if (std::fabs(A[r][i]) > std::fabs(A[piv][i])) piv = r;
    }
    if (std::fabs(A[piv][i]) < 1e-10) return false;
    if (piv != i) {
      for (int c = i; c < 8; ++c) std::swap(A[i][c], A[piv][c]);
      std::swap(b[i], b[piv]);
    }
    const double inv = 1.0 / A[i][i];
    for (int c = i; c < 8; ++c) A[i][c] *= inv;
    b[i] *= inv;
    for (int r = 0; r < 8; ++r) {
      if (r == i) continue;
      const double f = A[r][i];
      if (std::fabs(f) < 1e-12) continue;
      for (int c = i; c < 8; ++c) A[r][c] -= f * A[i][c];
      b[r] -= f * b[i];
    }
  }
  for (int i = 0; i < 8; ++i) x[i] = b[i];
  return true;
}

bool homography_from_4pt(
    const std::array<std::pair<double, double>, 4>& from,
    const std::array<std::pair<double, double>, 4>& to,
    std::array<double, 9>& H) {
  double A[8][8]{};
  double b[8]{};
  for (int i = 0; i < 4; ++i) {
    const double x = from[i].first;
    const double y = from[i].second;
    const double u = to[i].first;
    const double v = to[i].second;
    const int r0 = 2 * i;
    const int r1 = r0 + 1;
    A[r0][0] = x; A[r0][1] = y; A[r0][2] = 1.0;
    A[r0][6] = -u * x; A[r0][7] = -u * y;
    b[r0] = u;
    A[r1][3] = x; A[r1][4] = y; A[r1][5] = 1.0;
    A[r1][6] = -v * x; A[r1][7] = -v * y;
    b[r1] = v;
  }
  double x[8]{};
  if (!solve_8x8(A, b, x)) return false;
  H = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0};
  return true;
}

ImageRGBA warp_with_inverse_homography(const ImageRGBA& src, const std::array<double, 9>& invH, unsigned out_w, unsigned out_h) {
  ImageRGBA out;
  out.w = out_w;
  out.h = out_h;
  out.rgba.assign(4ull * out_w * out_h, 255);
  for (unsigned y = 0; y < out_h; ++y) {
    for (unsigned x = 0; x < out_w; ++x) {
      const double den = invH[6] * x + invH[7] * y + invH[8];
      if (std::fabs(den) < 1e-12) continue;
      const double sx = (invH[0] * x + invH[1] * y + invH[2]) / den;
      const double sy = (invH[3] * x + invH[4] * y + invH[5]) / den;
      const int ix = static_cast<int>(std::lround(sx));
      const int iy = static_cast<int>(std::lround(sy));
      if (ix < 0 || iy < 0 || ix >= static_cast<int>(src.w) || iy >= static_cast<int>(src.h)) continue;
      const size_t si = 4ull * (static_cast<size_t>(iy) * src.w + static_cast<unsigned>(ix));
      const size_t di = 4ull * (static_cast<size_t>(y) * out_w + x);
      out.rgba[di] = src.rgba[si];
      out.rgba[di + 1] = src.rgba[si + 1];
      out.rgba[di + 2] = src.rgba[si + 2];
      out.rgba[di + 3] = 255;
    }
  }
  return out;
}

void normalize_lighting(ImageRGBA& img) {
  if (img.w == 0 || img.h == 0 || img.rgba.empty()) return;
  auto q = compute_quality(img);
  const double m = q.brightness_mean;
  const double c = std::max(0.02, q.contrast_score);
  const double target_m = 0.50;
  const double target_c = 0.22;
  const double gain = target_c / c;
  for (size_t i = 0; i < img.rgba.size() / 4; ++i) {
    for (int k = 0; k < 3; ++k) {
      const double v = img.rgba[4 * i + k] / 255.0;
      const double nv = (v - m) * gain + target_m;
      img.rgba[4 * i + k] = static_cast<uint8_t>(std::lround(255.0 * clamp01(nv)));
    }
  }
}

bool run_shell_command(const std::string& cmd) {
  const int rc = std::system(cmd.c_str());
  return rc == 0;
}

bool try_capture_frame_to_jpg(const std::string& out_jpg, std::string& why) {
  why.clear();
  const std::string rpicam_cmd =
      "sh -lc 'command -v rpicam-jpeg >/dev/null 2>&1 && rpicam-jpeg -n -o \"" + out_jpg + "\" >/dev/null 2>&1'";
  if (run_shell_command(rpicam_cmd)) return true;
  const std::string libcamera_cmd =
      "sh -lc 'command -v libcamera-jpeg >/dev/null 2>&1 && libcamera-jpeg -n -o \"" + out_jpg + "\" >/dev/null 2>&1'";
  if (run_shell_command(libcamera_cmd)) return true;
  const std::string ffmpeg_cmd =
      "sh -lc 'command -v ffmpeg >/dev/null 2>&1 && ffmpeg -y -f video4linux2 -i /dev/video0 -vframes 1 \"" + out_jpg + "\" >/dev/null 2>&1'";
  if (run_shell_command(ffmpeg_cmd)) return true;
  const std::string fswebcam_cmd =
      "sh -lc 'command -v fswebcam >/dev/null 2>&1 && fswebcam --no-banner -r 1280x720 \"" + out_jpg + "\" >/dev/null 2>&1'";
  if (run_shell_command(fswebcam_cmd)) return true;
  why = "No camera capture command succeeded (rpicam-jpeg/libcamera-jpeg/ffmpeg/fswebcam).";
  return false;
}
}  // namespace

ADCSSim::ADCSSim() {
  StarTrackerConfig cfg{};
  cfg.update_hz = 2.0;
  cfg.dropout_probability = 0.01;
  cfg.false_star_probability = 0.02;
  adcs_.set_tracker_config(cfg);
  adcs_.set_controller_config(AdcsControllerConfig{0.08, 0.45, 0.12});
  adcs_.reset(42, 84);
  cmd_.desired_q_bi = Quaternion{};
  cmd_.desired_omega_b_rad_s = {0.0, 0.0, 0.0};
}

void ADCSSim::sense(double dt_s) {
  t_s_ += dt_s;
  disturbance_omega_ = {0.0, 0.0004, 0.0};
  if (t_s_ > 20.0 && t_s_ < 45.0) disturbance_omega_ = {0.0003, 0.0010, -0.0002};
  if (t_s_ > 65.0) disturbance_omega_ = {-0.00025, 0.00025, 0.0007};
}

void ADCSSim::decide(double) {
  correcting_ = est_pointing_error_deg_ > 0.03 || !tracker_valid_ || wheel_saturated_;
}

void ADCSSim::act(double dt_s) {
  AdcsSystemStepInput in{};
  in.dt_s = dt_s;
  in.disturbance_omega_rad_s = disturbance_omega_;
  in.command = cmd_;
  const auto& t = adcs_.step(in);
  wheel_saturated_ = t.wheel_saturated;
  est_pointing_error_deg_ = t.est_pointing_error_deg;
  truth_pointing_error_deg_ = t.truth_pointing_error_deg;
  tracker_confidence_ = t.tracker_confidence;
  tracked_stars_ = t.tracked_stars;
  tracker_valid_ = t.tracker_valid;
  power_w_ = t.total_power_w;
  wheel_power_w_ = t.wheel.power_w;
}

std::string ADCSSim::mode_string() const {
  if (wheel_saturated_) return "WHEEL_SAT";
  if (!tracker_valid_ && tracked_stars_ == 0) return "INIT";
  if (!tracker_valid_) return "TRACKER_DEGRADED";
  return correcting_ ? "CORRECTING" : "HOLD";
}

bool ADCSSim::stable() const {
  return est_pointing_error_deg_ < 0.02 && tracker_valid_ && tracker_confidence_ > 0.55 && !wheel_saturated_;
}

void CommsSim::enqueue_bits(std::size_t bits){ pending_enqueue_bits_ += bits; }
void CommsSim::sense(double dt_s){ pending_dt_s_ = dt_s; }
void CommsSim::decide(double){}
void CommsSim::act(double){
  comms::CommsInput in;
  in.dt_s = pending_dt_s_;
  in.enqueue_bits = pending_enqueue_bits_;
  auto t = model_.step(in);
  pending_enqueue_bits_ = 0;
  tx_active_ = t.tx_active;
  window_open_ = t.window_open;
  backlog_bits_ = t.backlog_bits;
  power_w_ = t.power_w;
  mode_ = t.mode;
}
std::string CommsSim::mode_string() const { return mode_; }
void ThermalSim::sense(double dt_s){ pending_dt_s_ = dt_s; }
void ThermalSim::decide(double){}
void ThermalSim::act(double){
  thermal::ThermalInput in;
  in.dt_s = pending_dt_s_;
  auto t = model_.step(in);
  temperature_c_ = t.temperature_c;
  heater_on_ = t.heater_on;
  low_temp_warn_ = t.low_temp_warning;
  high_temp_warn_ = t.high_temp_warning;
  heater_power_w_ = t.heater_power_w;
  power_w_ = t.power_w;
  mode_ = t.mode;
}
std::string ThermalSim::mode_string() const { return mode_; }
void PropulsionSim::sense(double dt_s){ pending_dt_s_ = dt_s; }
void PropulsionSim::decide(double){}
void PropulsionSim::act(double){
  propulsion::PropulsionInput in;
  in.dt_s = pending_dt_s_;
  auto t = model_.step(in);
  active_ = t.active;
  burn_event_ = t.burn_event;
  power_w_ = t.power_w;
  thrust_n_ = t.thrust_n;
  remaining_propellant_kg_ = t.remaining_propellant_kg;
  mode_ = t.mode;
}
std::string PropulsionSim::mode_string() const { return mode_; }

void PayloadSim::configure(const std::string& source_ppm,int tile_px_x,int tile_px_y,int ring_N,double ring_radius,double ring_sigma,const std::string& out_dir,const std::string& input_mode,double fusion_alpha,const SourcePreconditioningConfig& pre_cfg,const SglObservationConfig& obs_cfg,const std::string& reconstruction_mode){
  source_ppm_=source_ppm;
  tile_px_x_=tile_px_x;
  tile_px_y_=tile_px_y;
  ring_N_=ring_N;
  ring_radius_=ring_radius;
  ring_sigma_=ring_sigma;
  out_dir_=out_dir;
  input_mode_=input_mode;
  if (input_mode_ == "synthetic_truth") input_mode_ = "synthetic_image";
  fusion_alpha_=std::clamp(fusion_alpha,0.01,1.0);
  pre_cfg_ = pre_cfg;
  obs_cfg_ = obs_cfg;
  reconstruction_mode_ = reconstruction_mode;
  camera_mode_ = input_mode_;
}
void PayloadSim::sense(double dt_s){ t_s_ += dt_s; pending_dt_s_ = dt_s; }
void PayloadSim::decide(double){}
void PayloadSim::act(double){
  payload::PayloadInput in;
  in.dt_s = pending_dt_s_;
  auto t = model_.step(in);
  active_ = t.active;
  dataset_ready_ = t.dataset_ready;
  dataset_count_ = t.dataset_counter;
  acquisition_stage_ = t.acquisition_stage;
  synthetic_signal_score_ = t.synthetic_signal_score;
  mode_ = t.mode;
  power_w_ = t.power_w;
  if (!t.dataset_id.empty()) last_dataset_id_ = t.dataset_id;
  camera_mode_ = input_mode_;
  camera_frame_ready_ = false;
  alignment_valid_ = false;
  alignment_score_ = 0.0;
  blur_score_ = 0.0;
  brightness_mean_ = 0.0;
  contrast_score_ = 0.0;
  raw_capture_path_.clear();
  rectified_image_path_.clear();
  preconditioned_source_path_.clear();
  source_object_type_detected_ = "unknown";
  source_bbox_string_.clear();
  source_fill_fraction_used_ = 0.0;
  source_margin_fraction_before_ = 0.0;
  source_margin_fraction_ = 0.0;
  source_truncation_suspected_ = false;
  used_raw_fallback_for_preconditioning_ = false;
  source_clipping_guard_triggered_ = false;
  detected_planet_center_x_ = 0.0;
  detected_planet_center_y_ = 0.0;
  detected_planet_radius_px_ = 0.0;
  preconditioning_method_ = "none";
  alignment_method_ = "none";
  ring_summary_ = SglObservationSummary{};

  if (dataset_ready_) {
    fs::create_directories(out_dir_);
    std::string csv, ring_path, err;
    unsigned sw=0, sh=0;
    std::string dataset_id = t.dataset_id.empty() ? ("dataset_"+std::to_string(std::max(0, dataset_count_-1))) : t.dataset_id;
    std::string ddir=(fs::path(out_dir_)/dataset_id).string();
    fs::create_directories(ddir);
    pending_events_.push_back(PayloadEvent{"camera_capture_started", "info", "Camera/demo capture started", input_mode_});
    bool ok=false;
    last_ring_generation_ms_ = 0.0;
    bool capture_ok = true;
    ImageRGBA capture{};
    if (input_mode_ == "synthetic_image") {
      const auto t_rg0 = std::chrono::steady_clock::now();
      ok = generate_payload_dataset(source_ppm_,tile_px_x_,tile_px_y_,ring_N_,ring_radius_,ring_sigma_,ddir,csv,ring_path,sw,sh,err);
      const auto t_rg1 = std::chrono::steady_clock::now();
      last_ring_generation_ms_ = std::chrono::duration<double,std::milli>(t_rg1-t_rg0).count();
      if (ok) pending_events_.push_back(PayloadEvent{"payload_capture_accepted", "info", "Synthetic payload capture accepted", dataset_id});
    } else {
      if (input_mode_ == "pi_camera_demo") {
        raw_capture_path_ = (fs::path(ddir) / "raw_capture.jpg").string();
        std::string why;
        capture_ok = try_capture_frame_to_jpg(raw_capture_path_, why);
        if (capture_ok) {
          pending_events_.push_back(PayloadEvent{"camera_capture_completed", "info", "Camera capture completed", raw_capture_path_});
        } else {
          pending_events_.push_back(PayloadEvent{"camera_capture_failed", "warn", "Camera capture failed", why});
          if (fs::exists(source_ppm_)) {
            raw_capture_path_ = (fs::path(ddir) / "raw_capture_fallback.ppm").string();
            std::string e2;
            ImageRGBA fallback{};
            if (read_image_auto(source_ppm_, fallback, e2) && write_ppm(raw_capture_path_, fallback)) {
              capture_ok = true;
              pending_events_.push_back(PayloadEvent{"camera_capture_completed", "info", "Fallback image used for camera demo", raw_capture_path_});
            }
          }
        }
      } else {
        std::string e2;
        ImageRGBA file_img{};
        capture_ok = read_image_auto(source_ppm_, file_img, e2);
        raw_capture_path_ = (fs::path(ddir) / "raw_capture_from_file.ppm").string();
        if (capture_ok && write_ppm(raw_capture_path_, file_img)) {
          pending_events_.push_back(PayloadEvent{"camera_capture_completed", "info", "Image-file capture ready", raw_capture_path_});
        } else {
          capture_ok = false;
          pending_events_.push_back(PayloadEvent{"camera_capture_failed", "warn", "Image file missing or unreadable", source_ppm_});
        }
      }

      if (capture_ok) {
        std::string e3;
        if (!read_image_auto(raw_capture_path_, capture, e3)) {
          capture_ok = false;
          pending_events_.push_back(PayloadEvent{"camera_capture_failed", "warn", "Failed to read captured image", e3});
        }
      }

      if (capture_ok) {
        camera_frame_ready_ = true;
        std::array<std::pair<double, double>, 4> src_pts{};
        const bool markers = detect_corner_markers(capture, src_pts, alignment_score_);
        ImageRGBA rectified{};
        ImageRGBA overlay = capture;
        if (markers) {
          const unsigned out_s = std::max(256u, std::min(capture.w, capture.h));
          const std::array<std::pair<double, double>, 4> dst_pts{
              std::pair<double, double>{0.0, 0.0},
              std::pair<double, double>{static_cast<double>(out_s - 1), 0.0},
              std::pair<double, double>{static_cast<double>(out_s - 1), static_cast<double>(out_s - 1)},
              std::pair<double, double>{0.0, static_cast<double>(out_s - 1)},
          };
          std::array<double, 9> invH{};
          if (homography_from_4pt(dst_pts, src_pts, invH)) {
            rectified = warp_with_inverse_homography(capture, invH, out_s, out_s);
            alignment_valid_ = true;
            alignment_method_ = "marker_homography";
            for (const auto& p : src_pts) {
              const int px = std::clamp(static_cast<int>(std::lround(p.first)), 1, static_cast<int>(overlay.w) - 2);
              const int py = std::clamp(static_cast<int>(std::lround(p.second)), 1, static_cast<int>(overlay.h) - 2);
              for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                  const size_t i = 4ull * (static_cast<size_t>(py + dy) * overlay.w + static_cast<size_t>(px + dx));
                  overlay.rgba[i] = 64;
                  overlay.rgba[i + 1] = 255;
                  overlay.rgba[i + 2] = 64;
                }
              }
            }
            pending_events_.push_back(PayloadEvent{"payload_alignment_succeeded", "info", "Marker alignment succeeded", std::to_string(alignment_score_)});
          } else {
            unsigned x0 = 0, y0 = 0, x1 = capture.w > 0 ? capture.w - 1 : 0, y1 = capture.h > 0 ? capture.h - 1 : 0;
            double bbox_conf = 0.0;
            if (detect_content_bbox(capture, x0, y0, x1, y1, bbox_conf)) {
              rectified = crop_bbox_square(capture, x0, y0, x1, y1);
              draw_bbox_overlay(overlay, x0, y0, x1, y1, 255, 160, 64);
              alignment_score_ = std::max(alignment_score_, 0.35 * bbox_conf);
            } else {
              rectified = center_crop_square(capture);
            }
            alignment_valid_ = false;
            alignment_method_ = "homography_failed_bbox_or_center";
            pending_events_.push_back(PayloadEvent{"payload_alignment_failed", "warn", "Homography solve failed; using center-crop fallback", ""});
          }
        } else {
          unsigned x0 = 0, y0 = 0, x1 = capture.w > 0 ? capture.w - 1 : 0, y1 = capture.h > 0 ? capture.h - 1 : 0;
          double bbox_conf = 0.0;
          if (detect_content_bbox(capture, x0, y0, x1, y1, bbox_conf)) {
            rectified = crop_bbox_square(capture, x0, y0, x1, y1);
            draw_bbox_overlay(overlay, x0, y0, x1, y1, 255, 160, 64);
            alignment_score_ = 0.25 + 0.45 * bbox_conf;
            alignment_method_ = "content_bbox_fallback";
            pending_events_.push_back(PayloadEvent{"payload_alignment_failed", "warn", "Markers not detected; content-bbox fallback", std::to_string(alignment_score_)});
          } else {
            rectified = center_crop_square(capture);
            alignment_score_ = 0.0;
            alignment_method_ = "center_crop_fallback";
            pending_events_.push_back(PayloadEvent{"payload_alignment_failed", "warn", "Markers not detected; using center-crop fallback", ""});
          }
          alignment_valid_ = false;
        }

        normalize_lighting(rectified);
        auto q = compute_quality(rectified);
        blur_score_ = q.blur_score;
        brightness_mean_ = q.brightness_mean;
        contrast_score_ = q.contrast_score;
        rectified_image_path_ = (fs::path(ddir) / "rectified_input.ppm").string();
        write_ppm(rectified_image_path_, rectified);
        write_ppm((fs::path(ddir) / "alignment_overlay.ppm").string(), overlay);

        const bool quality_ok =
            (blur_score_ > 0.00025) &&
            (brightness_mean_ > 0.07 && brightness_mean_ < 0.93) &&
            (contrast_score_ > 0.045);
        if (!quality_ok) {
          pending_events_.push_back(PayloadEvent{"payload_capture_rejected", "warn", "Payload capture rejected by quality gates", "blur=" + std::to_string(blur_score_) + ",brightness=" + std::to_string(brightness_mean_) + ",contrast=" + std::to_string(contrast_score_)});
          ok = false;
        } else {
          pending_events_.push_back(PayloadEvent{"payload_capture_accepted", "info", "Payload capture accepted", dataset_id});
          const auto t_pre0 = std::chrono::steady_clock::now();
          SourcePreconditioningResult chosen{};
          std::string chosen_err;
          SourcePreconditioningResult pre_rect{};
          std::string pre_rect_err;
          SourcePreconditioningResult pre_raw{};
          std::string pre_raw_err;
          bool rect_ok = false;
          bool raw_ok = false;

          const bool have_raw = !raw_capture_path_.empty() && raw_capture_path_ != rectified_image_path_;
          const bool prefer_raw_first = (pre_cfg_.object_type == "disk_photo") && have_raw;

          if (prefer_raw_first) {
            raw_ok = precondition_source_image(raw_capture_path_, ddir, pre_cfg_, pre_raw, pre_raw_err);
            if (raw_ok) {
              chosen = pre_raw;
              chosen.used_raw_fallback_for_preconditioning = true;
              chosen_err.clear();
              ok = true;
            } else {
              rect_ok = precondition_source_image(rectified_image_path_, ddir, pre_cfg_, pre_rect, pre_rect_err);
              chosen = pre_rect;
              chosen_err = pre_rect_err;
              ok = rect_ok;
            }
          } else {
            rect_ok = precondition_source_image(rectified_image_path_, ddir, pre_cfg_, pre_rect, pre_rect_err);
            chosen = pre_rect;
            chosen_err = pre_rect_err;
            ok = rect_ok;
            if ((!rect_ok || pre_rect.source_truncation_suspected) && have_raw) {
              raw_ok = precondition_source_image(raw_capture_path_, ddir, pre_cfg_, pre_raw, pre_raw_err);
              if (raw_ok) {
                const bool prefer_raw =
                    !rect_ok ||
                    (!pre_raw.source_truncation_suspected && pre_rect.source_truncation_suspected) ||
                    (pre_raw.source_margin_fraction_before > pre_rect.source_margin_fraction_before + 0.01);
                if (prefer_raw) {
                  chosen = pre_raw;
                  chosen.used_raw_fallback_for_preconditioning = true;
                  chosen_err.clear();
                  ok = true;
                }
              } else if (!rect_ok) {
                chosen_err = pre_raw_err;
              }
            }
          }
          const auto t_pre1 = std::chrono::steady_clock::now();
          if (!ok) {
            err = chosen_err;
            pending_events_.push_back(PayloadEvent{"payload_capture_rejected", "warn", "Source preconditioning failed", chosen_err});
          } else {
            preconditioned_source_path_ = chosen.preconditioned_source_path;
            source_object_type_detected_ = chosen.object_type_detected;
            source_bbox_string_ = std::to_string(chosen.bbox_x0) + ":" + std::to_string(chosen.bbox_y0) + ":" + std::to_string(chosen.bbox_x1) + ":" + std::to_string(chosen.bbox_y1);
            source_fill_fraction_used_ = chosen.fill_fraction_used;
            source_margin_fraction_before_ = chosen.source_margin_fraction_before;
            source_margin_fraction_ = chosen.margin_fraction;
            source_truncation_suspected_ = chosen.source_truncation_suspected;
            used_raw_fallback_for_preconditioning_ = chosen.used_raw_fallback_for_preconditioning;
            source_clipping_guard_triggered_ = chosen.clipping_guard_triggered;
            detected_planet_center_x_ = chosen.detected_planet_center_x;
            detected_planet_center_y_ = chosen.detected_planet_center_y;
            detected_planet_radius_px_ = chosen.detected_planet_radius_px;
            preconditioning_method_ = chosen.method;
            pending_events_.push_back(PayloadEvent{"source_preconditioned", "info", "Source preconditioning completed", chosen.object_type_detected});
            pending_events_.push_back(PayloadEvent{"source_bbox", "info", "Source bbox detected", source_bbox_string_});
            pending_events_.push_back(PayloadEvent{"source_fill_fraction", "info", "Source fill fraction selected", std::to_string(source_fill_fraction_used_)});
            pending_events_.push_back(PayloadEvent{"source_margin_fraction_before", "info", "Source margin fraction before preconditioning", std::to_string(source_margin_fraction_before_)});
            pending_events_.push_back(PayloadEvent{"source_margin_fraction", "info", "Source margin fraction after preconditioning", std::to_string(source_margin_fraction_)});
            pending_events_.push_back(PayloadEvent{"source_detected_planet_center", "info", "Detected planet center x:y", std::to_string(detected_planet_center_x_) + ":" + std::to_string(detected_planet_center_y_)});
            pending_events_.push_back(PayloadEvent{"source_detected_planet_radius_px", "info", "Detected planet radius px", std::to_string(detected_planet_radius_px_)});
            if (source_truncation_suspected_) {
              pending_events_.push_back(PayloadEvent{"source_truncation_suspected", "warn", "Source appears truncated before canonicalization", std::to_string(source_margin_fraction_before_)});
            }
            if (used_raw_fallback_for_preconditioning_) {
              pending_events_.push_back(PayloadEvent{"used_raw_fallback_for_preconditioning", "warn", "Used raw capture fallback for preconditioning", raw_capture_path_});
            }
            if (source_clipping_guard_triggered_) {
              pending_events_.push_back(PayloadEvent{"source_clipping_guard_triggered", "warn", "Source clipping guard triggered", std::to_string(source_margin_fraction_)});
            }

            if (reconstruction_mode_ == "legacy_tiles") {
              const auto t_rg0 = std::chrono::steady_clock::now();
              ok = generate_payload_dataset(preconditioned_source_path_, tile_px_x_, tile_px_y_, ring_N_, ring_radius_, ring_sigma_, ddir, csv, ring_path, sw, sh, err);
              const auto t_rg1 = std::chrono::steady_clock::now();
              last_ring_generation_ms_ = std::chrono::duration<double,std::milli>(t_rg1-t_rg0).count();
              pending_events_.push_back(PayloadEvent{"ring_observation_skipped", "info", "Legacy tile reconstruction mode enabled", ""});
            } else {
              const auto t_rg0 = std::chrono::steady_clock::now();
              ok = generate_sgl_observation_dataset(preconditioned_source_path_, ddir, obs_cfg_, ring_summary_, csv, sw, sh, err);
              const auto t_rg1 = std::chrono::steady_clock::now();
              last_ring_generation_ms_ = std::chrono::duration<double,std::milli>(t_rg1-t_rg0).count();
              ring_path = ring_summary_.ring_preview_path;
              pending_events_.push_back(PayloadEvent{"ring_observation_skipped", "info", "Image-file/full-image path uses synthetic SGL observations", ""});
              pending_events_.push_back(PayloadEvent{"annulus_generated", "info", "Annulus/unwrapped observation sequence generated", ring_summary_.annulus_bin_path});
              pending_events_.push_back(PayloadEvent{"preconditioning_timing", "info", "Preconditioning runtime (ms)", std::to_string(std::chrono::duration<double,std::milli>(t_pre1-t_pre0).count())});
            }
          }
        }
      }
    }
    if (ok) {
      std::vector<TileStat> tiles; int tx=0,ty=0;
      if(tiles_from_csv(csv,tiles,tx,ty)){
        if(!have_fused_ || tx!=fused_tx_ || ty!=fused_ty_){ fused_tiles_=tiles; fused_tx_=tx; fused_ty_=ty; have_fused_=true; }
        else { double a=fusion_alpha_; for(size_t i=0;i<fused_tiles_.size()&&i<tiles.size();++i){ fused_tiles_[i].r=(1.0-a)*fused_tiles_[i].r + a*tiles[i].r; fused_tiles_[i].g=(1.0-a)*fused_tiles_[i].g + a*tiles[i].g; fused_tiles_[i].b=(1.0-a)*fused_tiles_[i].b + a*tiles[i].b; fused_tiles_[i].l=(1.0-a)*fused_tiles_[i].l + a*tiles[i].l; } }
        csv=tiles_to_csv(fused_tiles_,fused_tx_,fused_ty_);
      }
      datasets_.push_back(DatasetRecord{dataset_id,csv,ring_path,sw,sh});
      last_dataset_id_ = dataset_id;
    } else {
      if (!err.empty()) pending_events_.push_back(PayloadEvent{"payload_capture_rejected", "warn", "Payload dataset generation failed", err});
    }
  }
}
std::string PayloadSim::mode_string() const { return mode_; }
DatasetRecord PayloadSim::pop_dataset(){ auto r=datasets_.front(); datasets_.pop_front(); return r; }
std::vector<PayloadEvent> PayloadSim::drain_events(){ auto out = pending_events_; pending_events_.clear(); return out; }
double SourcePowerModel::update(double dt_s){ t_s_ += dt_s; available_w_ = 120.0 + 8.0*std::sin(0.05*t_s_) - 4.0*std::sin(0.2*t_s_); if(available_w_<90.0) available_w_=90.0; return available_w_; }
double PowerPolicy::compute_budget_w(double source_w,double noncompute_w) const { double left=source_w-noncompute_w-reserve_w_; if(left<0.0) left=0.0; return left*derate_fraction_; }
} // namespace sgl
