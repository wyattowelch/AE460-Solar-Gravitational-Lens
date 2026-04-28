#include "sgl/eps_module.hpp"
#include <cmath>

namespace sgl::eps {
void EpsModel::reset() {
  t_s_ = 0.0;
  source_w_ = 120.0;
}

EpsTelemetry EpsModel::evaluate(const EpsInput& in) const {
  EpsTelemetry t;
  t.source_w = source_w_;
  t.noncompute_load_w = in.noncompute_load_w;
  t.reserve_w = in.reserve_w;
  double left = t.source_w - in.noncompute_load_w - in.reserve_w;
  if (left < 0.0) left = 0.0;
  t.compute_budget_w = left * in.safe_fraction;
  t.total_bus_load_w = in.noncompute_load_w + in.pi_load_w + in.jetson_load_w;
  t.bus_margin_w = t.source_w - t.total_bus_load_w - in.reserve_w;
  t.low_power_state = (t.bus_margin_w < 0.0);
  t.power_w = 0.5;
  return t;
}

EpsTelemetry EpsModel::step(const EpsInput& in) {
  t_s_ += in.dt_s;
  source_w_ = 120.0 + 8.0 * std::sin(0.05 * t_s_) - 4.0 * std::sin(0.2 * t_s_);
  if (source_w_ < 90.0) source_w_ = 90.0;
  return evaluate(in);
}
}  // namespace sgl::eps
