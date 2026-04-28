#include "sgl/propulsion_module.hpp"
namespace sgl::propulsion {
void PropulsionModel::reset() {
  t_s_ = 0.0;
  remaining_propellant_kg_ = 8.0;
}
PropulsionTelemetry PropulsionModel::step(const PropulsionInput& in) {
  t_s_ += in.dt_s;
  const int cyc = static_cast<int>(t_s_);
  const bool firing = ((cyc % 120) >= 0 && (cyc % 120) < 4);
  const bool burn_event = firing;
  if (burn_event) {
    remaining_propellant_kg_ -= 0.0025 * in.dt_s;
    if (remaining_propellant_kg_ < 0.0) remaining_propellant_kg_ = 0.0;
  }
  PropulsionTelemetry t;
  t.active = firing;
  t.burn_event = burn_event;
  t.power_w = firing ? 15.0 : 1.0;
  t.thrust_n = firing ? 0.12 : 0.0;
  t.remaining_propellant_kg = remaining_propellant_kg_;
  t.mode = firing ? "CORRECTION_BURN" : "IDLE";
  return t;
}
}  // namespace sgl::propulsion
