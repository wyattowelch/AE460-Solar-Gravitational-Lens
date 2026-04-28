#pragma once
#include <string>

namespace sgl::thermal {
struct ThermalInput { double dt_s{1.0}; };
struct ThermalTelemetry {
  double temperature_c{18.0};
  bool heater_on{false};
  bool low_temp_warning{false};
  bool high_temp_warning{false};
  double heater_power_w{0.0};
  double power_w{2.0};
  std::string mode{"PASSIVE"};
};
class ThermalModel {
 public:
  void reset(double temp_c = 18.0);
  ThermalTelemetry step(const ThermalInput& in);
 private:
  double temperature_c_{18.0};
};
}  // namespace sgl::thermal
