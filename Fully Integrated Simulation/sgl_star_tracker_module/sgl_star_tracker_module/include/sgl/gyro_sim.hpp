#pragma once
#include <random>
#include "sgl/star_tracker_types.hpp"

namespace sgl {

class GyroSim {
public:
    explicit GyroSim(const StarTrackerConfig& cfg = {});
    void set_config(const StarTrackerConfig& cfg);
    void reset(uint64_t seed = 84);
    GyroMeasurement measure(const TruthState& truth, double dt_s);

private:
    double gaussian(double sigma);
    StarTrackerConfig cfg_{};
    std::mt19937_64 rng_{};
    Vec3 bias_rad_s_{};
};

} // namespace sgl
