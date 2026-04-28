#include "sgl/comms_module.hpp"
#include <cmath>

namespace sgl::comms {
void CommsModel::reset(std::size_t initial_backlog_bits) {
  t_s_ = 0.0;
  backlog_bits_ = initial_backlog_bits;
}

CommsTelemetry CommsModel::step(const CommsInput& in) {
  t_s_ += in.dt_s;
  backlog_bits_ += in.enqueue_bits;

  const int period_s = 30;
  const int on_s = 8;
  const int m = static_cast<int>(std::fmod(t_s_, static_cast<double>(period_s)));
  const bool window_open = (m < on_s);
  const bool tx_active = window_open && backlog_bits_ > 0;

  double power_w = 4.0;
  if (tx_active) {
    const std::size_t sent = static_cast<std::size_t>(bitrate_bps_ * in.dt_s);
    backlog_bits_ = (sent >= backlog_bits_) ? 0 : (backlog_bits_ - sent);
    power_w = 15.0;
  }

  CommsTelemetry t;
  t.tx_active = tx_active;
  t.window_open = window_open;
  t.backlog_bits = backlog_bits_;
  t.power_w = power_w;
  t.mode = tx_active ? "DOWNLINK" : (window_open ? "READY" : "STANDBY");
  return t;
}
}  // namespace sgl::comms
