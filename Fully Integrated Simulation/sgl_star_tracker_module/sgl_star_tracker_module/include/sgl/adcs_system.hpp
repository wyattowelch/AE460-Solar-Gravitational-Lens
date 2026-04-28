#pragma once
#include "sgl/adcs_controller.hpp"
#include "sgl/attitude_filter.hpp"
#include "sgl/gyro_sim.hpp"
#include "sgl/reaction_wheel_array.hpp"
#include "sgl/star_tracker_sim.hpp"

namespace sgl {

struct AdcsSystemStepInput {
  double dt_s{1.0};
  Vec3 disturbance_omega_rad_s{};
  AdcsCommand command{};
};

struct AdcsSystemTelemetry {
  TruthState truth{};
  FusedAttitudeState estimate{};
  StarTrackerMeasurement tracker{};
  bool have_tracker{false};
  ReactionWheelTelemetry wheel{};
  Vec3 torque_cmd_nm{};
  double tracker_power_w{0.0};
  double total_power_w{0.0};
  double truth_pointing_error_deg{0.0};
  double est_pointing_error_deg{0.0};
  uint32_t tracked_stars{0};
  double tracker_confidence{0.0};
  bool tracker_valid{false};
  bool wheel_saturated{false};
};

class AdcsSystem {
 public:
  explicit AdcsSystem(const StarTrackerConfig& cfg = {}, const AdcsControllerConfig& ccfg = {});
  void reset(uint64_t tracker_seed = 42, uint64_t gyro_seed = 84);
  void set_tracker_config(const StarTrackerConfig& cfg);
  void set_controller_config(const AdcsControllerConfig& cfg);
  const AdcsSystemTelemetry& step(const AdcsSystemStepInput& in);
  const AdcsSystemTelemetry& telemetry() const { return tel_; }

 private:
  StarTrackerSim tracker_{};
  GyroSim gyro_{};
  AttitudeFilter filter_{};
  AdcsController controller_{};
  ReactionWheelArray wheels_{};

  double t_s_{0.0};
  Vec3 last_torque_cmd_nm_{};
  AdcsSystemTelemetry tel_{};
};

}  // namespace sgl
