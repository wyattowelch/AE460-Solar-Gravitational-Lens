#include <iostream>
#include "sgl/eps_module.hpp"

int main() {
  sgl::eps::EpsModel model;
  model.reset();
  for (int i = 0; i < 12; ++i) {
    sgl::eps::EpsInput in;
    in.dt_s = 1.0;
    in.noncompute_load_w = (i < 6) ? 30.0 : 45.0;
    in.reserve_w = 20.0;
    in.safe_fraction = 0.75;
    in.pi_load_w = 6.0;
    in.jetson_load_w = (i % 3 == 0) ? 15.0 : 5.0;
    auto t = model.step(in);
    std::cout << "cycle=" << i << " source_w=" << t.source_w << " noncompute_w=" << t.noncompute_load_w << " compute_budget_w=" << t.compute_budget_w << " total_bus_w=" << t.total_bus_load_w << " low_power=" << (t.low_power_state ? 1 : 0) << "\n";
  }
  return 0;
}
