#pragma once
#include <string>

namespace sgl::propulsion {
struct PropulsionInput { double dt_s{1.0}; };
struct PropulsionTelemetry {
  bool active{false};
  bool burn_event{false};
  double power_w{1.0};
  double thrust_n{0.0};
  double remaining_propellant_kg{8.0};
  std::string mode{"IDLE"};
};
class PropulsionModel {
 public:
  void reset();
  PropulsionTelemetry step(const PropulsionInput& in);
 private:
  double t_s_{0.0};
  double remaining_propellant_kg_{8.0};
};
}  // namespace sgl::propulsion
