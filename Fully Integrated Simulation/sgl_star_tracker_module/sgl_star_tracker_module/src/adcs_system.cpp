#include "sgl/adcs_system.hpp"

namespace sgl {

AdcsSystem::AdcsSystem(const StarTrackerConfig& cfg, const AdcsControllerConfig& ccfg)
    : tracker_(cfg), gyro_(cfg), controller_(ccfg) {
  reset();
}

void AdcsSystem::reset(uint64_t tracker_seed, uint64_t gyro_seed) {
  t_s_ = 0.0;
  last_torque_cmd_nm_ = {};
  tracker_.reset(tracker_seed);
  gyro_.reset(gyro_seed);
  filter_.reset();
  wheels_.reset();
  tel_ = {};
  tel_.truth.q_bi = quat_from_axis_angle({0.0, 1.0, 0.0}, 0.03);
  tel_.truth.omega_b_rad_s = {0.0, 0.0004, 0.0};
}

void AdcsSystem::set_tracker_config(const StarTrackerConfig& cfg) {
  tracker_.set_config(cfg);
  gyro_.set_config(cfg);
}

void AdcsSystem::set_controller_config(const AdcsControllerConfig& cfg) {
  controller_.set_config(cfg);
}

const AdcsSystemTelemetry& AdcsSystem::step(const AdcsSystemStepInput& in) {
  t_s_ += in.dt_s;

  // Keep closed-loop coupling stable for long mission runs: damp prior rates and inject
  // both disturbance and control response as bounded body-rate contributions.
  const double rate_damping = 0.92;
  const double control_rate_gain = 0.22;
  tel_.truth.omega_b_rad_s = rate_damping * tel_.truth.omega_b_rad_s + in.disturbance_omega_rad_s + control_rate_gain * last_torque_cmd_nm_;
  tel_.truth.q_bi = quat_integrate_body_rate(tel_.truth.q_bi, tel_.truth.omega_b_rad_s, in.dt_s);
  tel_.truth.t_s = t_s_;

  tracker_.step(tel_.truth, in.dt_s);
  if (tracker_.has_new_measurement()) {
    tel_.tracker = tracker_.latest();
    tel_.have_tracker = true;
  }

  auto gm = gyro_.measure(tel_.truth, in.dt_s);
  const StarTrackerMeasurement* tm = tel_.have_tracker ? &tel_.tracker : nullptr;
  tel_.estimate = filter_.update(gm, tm, in.dt_s);

  tel_.torque_cmd_nm = controller_.compute_body_torque(tel_.estimate, in.command);
  tel_.wheel = wheels_.step(tel_.torque_cmd_nm, in.dt_s);
  tel_.wheel_saturated = tel_.wheel.saturated;
  last_torque_cmd_nm_ = tel_.torque_cmd_nm;

  tel_.tracker_power_w = tracker_.current_power_w();
  tel_.total_power_w = tel_.tracker_power_w + tel_.wheel.power_w;
  tel_.truth_pointing_error_deg = quat_angular_error_rad(in.command.desired_q_bi, tel_.truth.q_bi) * (180.0 / 3.14159265358979323846);
  tel_.est_pointing_error_deg = quat_angular_error_rad(in.command.desired_q_bi, tel_.estimate.q_bi) * (180.0 / 3.14159265358979323846);
  tel_.tracked_stars = tel_.have_tracker ? tel_.tracker.tracked_stars : 0;
  tel_.tracker_confidence = tel_.have_tracker ? tel_.tracker.confidence : 0.0;
  tel_.tracker_valid = tel_.have_tracker && tel_.tracker.valid;

  return tel_;
}

}  // namespace sgl
