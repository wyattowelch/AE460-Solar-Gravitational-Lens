#include <algorithm>
#include <cassert>
#include <vector>

#include "sgl/adcs_system.hpp"

int main() {
  sgl::StarTrackerConfig cfg{};
  cfg.dropout_probability = 0.01;
  cfg.false_star_probability = 0.02;
  cfg.update_hz = 2.0;
  sgl::AdcsSystem adcs(cfg, sgl::AdcsControllerConfig{0.08, 0.45, 0.12});
  adcs.reset();

  std::vector<double> truth_err;
  std::vector<double> est_err;
  std::vector<double> power;
  for (int i = 0; i < 120; ++i) {
    sgl::AdcsSystemStepInput in;
    in.dt_s = 1.0;
    in.command.desired_q_bi = sgl::Quaternion{};
    in.command.desired_omega_b_rad_s = {0.0, 0.0, 0.0};
    in.disturbance_omega_rad_s = {0.0, 0.0004, 0.0};
    if (i > 20 && i < 45) in.disturbance_omega_rad_s = {0.0003, 0.0010, -0.0002};
    if (i > 65) in.disturbance_omega_rad_s = {-0.00025, 0.00025, 0.0007};
    const auto& t = adcs.step(in);
    truth_err.push_back(t.truth_pointing_error_deg);
    est_err.push_back(t.est_pointing_error_deg);
    power.push_back(t.total_power_w);
  }

  auto [min_t, max_t] = std::minmax_element(truth_err.begin(), truth_err.end());
  auto [min_e, max_e] = std::minmax_element(est_err.begin(), est_err.end());
  assert(((*max_t - *min_t) > 0.01));
  assert(((*max_e - *min_e) > 0.01));

  bool sep = false;
  for (size_t i = 0; i < truth_err.size(); ++i) {
    if (std::abs(truth_err[i] - est_err[i]) > 1e-6) { sep = true; break; }
  }
  assert(sep);

  assert(*std::max_element(power.begin(), power.end()) > *std::min_element(power.begin(), power.end()));
  return 0;
}
