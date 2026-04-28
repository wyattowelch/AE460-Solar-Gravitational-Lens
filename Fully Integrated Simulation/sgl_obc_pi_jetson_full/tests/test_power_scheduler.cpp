#include <iostream>

#include "../common/scheduler.hpp"
#include "sgl/eps_module.hpp"
#include "sgl/payload_module.hpp"
#include "sgl/propulsion_module.hpp"
#include "sgl/thermal_module.hpp"

int main() {
  sgl::eps::EpsModel eps;
  eps.reset();
  sgl::eps::EpsInput seed;
  seed.dt_s = 1.0;
  seed.noncompute_load_w = 20.0;
  seed.reserve_w = 20.0;
  seed.safe_fraction = 0.75;
  seed.pi_load_w = 4.0;
  seed.jetson_load_w = 5.0;
  (void)eps.step(seed);

  auto low = eps.evaluate(sgl::eps::EpsInput{0.0, 10.0, 20.0, 0.75, 4.0, 5.0});
  auto high = eps.evaluate(sgl::eps::EpsInput{0.0, 40.0, 20.0, 0.75, 4.0, 5.0});
  const double b0 = low.compute_budget_w;
  const double b1 = high.compute_budget_w;
  if (!(b1 < b0)) {
    std::cerr << "EPS compute budget did not drop with higher subsystem load\n";
    return 2;
  }
  if (!(high.total_bus_load_w > low.total_bus_load_w)) {
    std::cerr << "EPS total bus load did not reflect higher loads\n";
    return 7;
  }

  sgl::thermal::ThermalModel thermal;
  thermal.reset(16.0);
  auto warm = thermal.step({1.0});
  auto passive = thermal.step({1.0});
  auto with_heater = eps.evaluate(sgl::eps::EpsInput{0.0, 25.0 + warm.power_w, 20.0, 0.75, 4.0, 5.0});
  auto without_heater = eps.evaluate(sgl::eps::EpsInput{0.0, 25.0 + passive.power_w, 20.0, 0.75, 4.0, 5.0});
  if (!(with_heater.compute_budget_w <= without_heater.compute_budget_w)) {
    std::cerr << "thermal heater load did not reduce/limit compute budget\n";
    return 8;
  }

  sgl::propulsion::PropulsionModel prop;
  prop.reset();
  auto p_active = prop.step({1.0});
  auto p_idle = prop.step({5.0});
  auto with_burn = eps.evaluate(sgl::eps::EpsInput{0.0, 25.0 + p_active.power_w, 20.0, 0.75, 4.0, 5.0});
  auto without_burn = eps.evaluate(sgl::eps::EpsInput{0.0, 25.0 + p_idle.power_w, 20.0, 0.75, 4.0, 5.0});
  if (!(with_burn.compute_budget_w < without_burn.compute_budget_w)) {
    std::cerr << "propulsion active load did not reduce compute budget\n";
    return 9;
  }

  sgl::payload::PayloadModel payload;
  payload.reset();
  auto pay_active = payload.step({1.0});
  // Advance into idle/ready cadence after acquisition.
  auto pay_idle = payload.step({1.0});
  for (int i = 0; i < 5; ++i) pay_idle = payload.step({1.0});
  auto with_payload_active = eps.evaluate(sgl::eps::EpsInput{0.0, 25.0 + pay_active.power_w, 20.0, 0.75, 4.0, 5.0});
  auto with_payload_idle = eps.evaluate(sgl::eps::EpsInput{0.0, 25.0 + pay_idle.power_w, 20.0, 0.75, 4.0, 5.0});
  if (!(with_payload_active.compute_budget_w <= with_payload_idle.compute_budget_w)) {
    std::cerr << "payload active load did not reduce/limit compute budget\n";
    return 10;
  }

  const auto nominal = sgl::decide_scheduler_mode(40.0, 1000, 0, 0, 15.0);
  const auto throttled = sgl::decide_scheduler_mode(8.0, 1000, 0, 0, 15.0);
  const auto suspended_cd = sgl::decide_scheduler_mode(40.0, 1000, 2, 0, 15.0);
  const auto suspended_unstable = sgl::decide_scheduler_mode(40.0, 1000, 0, 5, 15.0);

  if (nominal != sgl::SchedulerMode::Nominal) return 3;
  if (throttled != sgl::SchedulerMode::Throttled) return 4;
  if (suspended_cd != sgl::SchedulerMode::Suspended) return 5;
  if (suspended_unstable != sgl::SchedulerMode::Suspended) return 6;

  return 0;
}
