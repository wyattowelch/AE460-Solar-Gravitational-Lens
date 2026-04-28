#pragma once
#include <random>
#include <vector>
#include "sgl/star_tracker_types.hpp"

namespace sgl {

class StarTrackerSim {
public:
    explicit StarTrackerSim(const StarTrackerConfig& cfg = {});

    void set_config(const StarTrackerConfig& cfg);
    const StarTrackerConfig& config() const;

    void reset(uint64_t seed = 42);
    void step(const TruthState& truth, double dt_s);

    bool has_new_measurement() const;
    StarTrackerMeasurement latest() const;

    double current_power_w() const;
    std::string mode_string() const;

private:
    StarTrackerMeasurement synthesize_measurement(const TruthState& truth);
    Quaternion noisy_attitude(const Quaternion& q_true, double sigma_arcsec);
    double gaussian(double sigma);
    double uniform();

    StarTrackerConfig cfg_{};
    std::mt19937_64 rng_{};
    double t_s_{0.0};
    double next_update_s_{0.0};
    bool has_new_{false};
    double power_w_{0.0};
    StarTrackerMeasurement latest_{};
};

} // namespace sgl
