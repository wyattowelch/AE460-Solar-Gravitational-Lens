#pragma once

namespace sgl::eps {
struct EpsInput {
  double dt_s{1.0};
  double noncompute_load_w{0.0};
  double reserve_w{20.0};
  double safe_fraction{0.75};
  double pi_load_w{0.0};
  double jetson_load_w{0.0};
};

struct EpsTelemetry {
  double source_w{0.0};
  double noncompute_load_w{0.0};
  double reserve_w{0.0};
  double compute_budget_w{0.0};
  double total_bus_load_w{0.0};
  double bus_margin_w{0.0};
  bool low_power_state{false};
  double power_w{0.5};
};

class EpsModel {
 public:
  void reset();
  EpsTelemetry evaluate(const EpsInput& in) const;
  EpsTelemetry step(const EpsInput& in);
  double source_power_w() const { return source_w_; }

 private:
  double t_s_{0.0};
  double source_w_{120.0};
};
}  // namespace sgl::eps
