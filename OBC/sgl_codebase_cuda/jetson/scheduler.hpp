#pragma once
#include "../common/config.hpp"
#include "../common/logger.hpp"
#include "power_manager.hpp"

// Scheduler chooses processing mode based on power headroom and progression:
// - always do low-res pass early
// - refine gradually when power permits
class Scheduler {
  const sgl::Config cfg_;
  PowerManager& pm_;
  sgl::Logger& log_;
  int refine_done_ = 0;
public:
  Scheduler(const sgl::Config& C, PowerManager& pm, sgl::Logger& log)
  : cfg_(C), pm_(pm), log_(log) {}

  void tick(int step);
};
