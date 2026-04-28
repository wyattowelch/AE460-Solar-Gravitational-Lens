#include "sgl/gyro_sim.hpp"

namespace sgl {

GyroSim::GyroSim(const StarTrackerConfig& cfg): cfg_(cfg) { reset(); }
void GyroSim::set_config(const StarTrackerConfig& cfg){ cfg_ = cfg; }
void GyroSim::reset(uint64_t seed){ rng_.seed(seed); bias_rad_s_ = {}; }

double GyroSim::gaussian(double sigma){
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng_);
}

GyroMeasurement GyroSim::measure(const TruthState& truth, double dt_s){
    bias_rad_s_.x += gaussian(cfg_.gyro_bias_walk_rad_s_per_s * dt_s);
    bias_rad_s_.y += gaussian(cfg_.gyro_bias_walk_rad_s_per_s * dt_s);
    bias_rad_s_.z += gaussian(cfg_.gyro_bias_walk_rad_s_per_s * dt_s);

    GyroMeasurement m{};
    m.t_s = truth.t_s;
    m.omega_b_rad_s = {
        truth.omega_b_rad_s.x + bias_rad_s_.x + gaussian(cfg_.gyro_noise_rad_s_1sigma),
        truth.omega_b_rad_s.y + bias_rad_s_.y + gaussian(cfg_.gyro_noise_rad_s_1sigma),
        truth.omega_b_rad_s.z + bias_rad_s_.z + gaussian(cfg_.gyro_noise_rad_s_1sigma)
    };
    return m;
}

} // namespace sgl
