#include <cassert>
#include "sgl/eps_module.hpp"

int main() {
  sgl::eps::EpsModel m;
  m.reset();
  auto a = m.step({1.0, 20.0, 20.0, 0.75, 4.0, 5.0});
  auto b = m.evaluate({0.0, 50.0, 20.0, 0.75, 10.0, 15.0});
  assert(b.compute_budget_w < a.compute_budget_w);
  assert(b.total_bus_load_w > a.total_bus_load_w);
  return 0;
}
