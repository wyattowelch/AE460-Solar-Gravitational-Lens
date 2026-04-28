#include "sgl/thermal_module.hpp"
namespace sgl::thermal {
void ThermalModel::reset(double temp_c) { temperature_c_ = temp_c; }
ThermalTelemetry ThermalModel::step(const ThermalInput& in) {
  temperature_c_ -= 0.03 * in.dt_s;
  const bool heater_on = temperature_c_ < 16.5;
  if (heater_on) temperature_c_ += 0.18 * in.dt_s;
  ThermalTelemetry t;
  t.temperature_c = temperature_c_;
  t.heater_on = heater_on;
  t.heater_power_w = heater_on ? 10.0 : 0.0;
  t.power_w = heater_on ? 12.0 : 2.0;
  t.low_temp_warning = (temperature_c_ < 15.5);
  t.high_temp_warning = (temperature_c_ > 35.0);
  if (t.low_temp_warning) t.mode = "LOW_TEMP";
  else if (t.high_temp_warning) t.mode = "HIGH_TEMP";
  else t.mode = heater_on ? "HEATING" : "PASSIVE";
  return t;
}
}  // namespace sgl::thermal
