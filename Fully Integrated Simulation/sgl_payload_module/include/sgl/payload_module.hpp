#pragma once
#include <string>

namespace sgl::payload {
struct PayloadInput { double dt_s{1.0}; };
struct PayloadTelemetry {
  bool active{false};
  bool dataset_ready{false};
  int dataset_counter{0};
  std::string dataset_id{};
  int acquisition_stage{0};
  double synthetic_signal_score{0.0};
  double power_w{6.0};
  std::string mode{"IDLE"};
};
class PayloadModel {
 public:
  void reset();
  PayloadTelemetry step(const PayloadInput& in);
 private:
  int acquire_countdown_{0};
  int acquisition_progress_{0};
  int dataset_counter_{0};
};
}  // namespace sgl::payload
