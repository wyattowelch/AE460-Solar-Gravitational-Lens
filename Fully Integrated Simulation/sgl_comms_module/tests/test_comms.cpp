#include <cassert>
#include "sgl/comms_module.hpp"

int main() {
  sgl::comms::CommsModel model;
  model.reset();

  sgl::comms::CommsInput in;
  in.dt_s = 1.0;
  in.enqueue_bits = 80000;
  auto t0 = model.step(in);
  assert(t0.backlog_bits <= 80000);

  bool saw_downlink = false;
  bool saw_standby_or_ready = false;
  for (int i = 0; i < 50; ++i) {
    sgl::comms::CommsInput tick;
    tick.dt_s = 1.0;
    auto t = model.step(tick);
    if (t.mode == "DOWNLINK") saw_downlink = true;
    if (t.mode == "STANDBY" || t.mode == "READY") saw_standby_or_ready = true;
  }
  assert(saw_downlink);
  assert(saw_standby_or_ready);
  return 0;
}
