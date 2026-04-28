#pragma once
#include <cstddef>
#include <string>

namespace sgl::comms {
struct CommsInput {
  double dt_s{1.0};
  std::size_t enqueue_bits{0};
};

struct CommsTelemetry {
  bool tx_active{false};
  bool window_open{false};
  std::size_t backlog_bits{0};
  double power_w{0.0};
  std::string mode{"STANDBY"};
};

class CommsModel {
 public:
  void reset(std::size_t initial_backlog_bits = 0);
  CommsTelemetry step(const CommsInput& in);

 private:
  double t_s_{0.0};
  std::size_t backlog_bits_{0};
  std::size_t bitrate_bps_{8000};
};
}  // namespace sgl::comms
