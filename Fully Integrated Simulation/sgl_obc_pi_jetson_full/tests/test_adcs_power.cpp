#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../pi_flight/subsystems.hpp"

int main() {
  sgl::ADCSSim adcs;
  const double dt = 1.0;
  std::vector<double> truth_err;
  std::vector<double> est_err;
  std::vector<double> p_corr;
  std::vector<double> p_other;

  for (int i = 0; i < 160; ++i) {
    adcs.sense(dt);
    adcs.decide(dt);
    adcs.act(dt);
    truth_err.push_back(adcs.truth_pointing_error_deg());
    est_err.push_back(adcs.est_pointing_error_deg());
    const std::string mode = adcs.mode_string();
    if (mode == "CORRECTING") p_corr.push_back(adcs.current_power_w());
    else p_other.push_back(adcs.current_power_w());
  }

  const auto [min_truth_it, max_truth_it] = std::minmax_element(truth_err.begin(), truth_err.end());
  const auto [min_est_it, max_est_it] = std::minmax_element(est_err.begin(), est_err.end());

  if (truth_err.empty() || est_err.empty()) return 2;

  if ((*max_truth_it - *min_truth_it) < 0.01) {
    std::cerr << "truth_pointing_err_deg did not change enough\n";
    return 3;
  }
  if ((*max_est_it - *min_est_it) < 0.01) {
    std::cerr << "est_pointing_err_deg did not change enough\n";
    return 4;
  }

  bool any_sep = false;
  for (size_t i = 0; i < truth_err.size() && i < est_err.size(); ++i) {
    if (std::fabs(truth_err[i] - est_err[i]) > 1e-6) {
      any_sep = true;
      break;
    }
  }
  if (!any_sep) {
    std::cerr << "truth and estimated errors are not logged separately\n";
    return 5;
  }

  const double max_truth = *max_truth_it;
  if (max_truth > 15.0) {
    std::cerr << "truth pointing error unbounded: " << max_truth << " deg\n";
    return 6;
  }

  if (p_corr.empty() || p_other.empty()) {
    std::cerr << "insufficient correcting/non-correcting samples for power comparison\n";
    return 7;
  }
  double mean_corr = 0.0;
  for (double v : p_corr) mean_corr += v;
  mean_corr /= static_cast<double>(p_corr.size());
  double mean_other = 0.0;
  for (double v : p_other) mean_other += v;
  mean_other /= static_cast<double>(p_other.size());

  if (mean_corr < mean_other) {
    std::cerr << "ADCS correcting power lower than non-correcting mean\n";
    return 8;
  }

  return 0;
}
