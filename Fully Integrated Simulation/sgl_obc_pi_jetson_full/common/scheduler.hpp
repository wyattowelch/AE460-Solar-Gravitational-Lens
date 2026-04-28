#pragma once
#include <cstddef>

namespace sgl {
enum class SchedulerMode { Nominal = 0, Throttled = 1, Suspended = 2 };

inline SchedulerMode decide_scheduler_mode(double jetson_allow_w, std::size_t comms_backlog_bits, int cooldown_cycles, int unstable_cycles, double jetson_refine_w) {
  if (cooldown_cycles > 0 || unstable_cycles > 4) return SchedulerMode::Suspended;
  if (jetson_allow_w < jetson_refine_w || comms_backlog_bits > 6000000ull) return SchedulerMode::Throttled;
  return SchedulerMode::Nominal;
}
}  // namespace sgl

