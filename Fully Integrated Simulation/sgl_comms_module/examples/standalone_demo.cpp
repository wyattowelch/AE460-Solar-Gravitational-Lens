#include <iostream>
#include "sgl/comms_module.hpp"

int main() {
  sgl::comms::CommsModel model;
  model.reset();
  for (int i = 0; i < 20; ++i) {
    sgl::comms::CommsInput in;
    in.dt_s = 1.0;
    if (i == 0) in.enqueue_bits = 120000;
    auto t = model.step(in);
    std::cout << "cycle=" << i << " mode=" << t.mode << " backlog=" << t.backlog_bits << " power_w=" << t.power_w << "\n";
  }
  return 0;
}
