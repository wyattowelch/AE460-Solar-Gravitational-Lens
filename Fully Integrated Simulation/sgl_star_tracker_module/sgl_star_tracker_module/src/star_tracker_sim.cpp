#include "sgl/star_tracker_sim.hpp"
#include <algorithm>
#include <cmath>

namespace sgl {

static double arcsec_to_rad(double arcsec){ return arcsec * (M_PI / (180.0 * 3600.0)); }

StarTrackerSim::StarTrackerSim(const StarTrackerConfig& cfg): cfg_(cfg) { reset(); }
void StarTrackerSim::set_config(const StarTrackerConfig& cfg){ cfg_ = cfg; }
const StarTrackerConfig& StarTrackerSim::config() const { return cfg_; }
void StarTrackerSim::reset(uint64_t seed){
    rng_.seed(seed);
    t_s_ = 0.0;
    next_update_s_ = 0.0;
    has_new_ = false;
    latest_ = {};
    power_w_ = cfg_.nominal_power_w;
}

double StarTrackerSim::gaussian(double sigma){
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng_);
}

double StarTrackerSim::uniform(){
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng_);
}

Quaternion StarTrackerSim::noisy_attitude(const Quaternion& q_true, double sigma_arcsec){
    Vec3 axis{gaussian(1.0), gaussian(1.0), gaussian(1.0)};
    axis = normalized(axis);
    double ang = gaussian(arcsec_to_rad(sigma_arcsec));
    return quat_normalized(quat_multiply(q_true, quat_from_axis_angle(axis, ang)));
}

StarTrackerMeasurement StarTrackerSim::synthesize_measurement(const TruthState& truth){
    StarTrackerMeasurement m{};
    m.t_s = truth.t_s + cfg_.latency_s;
    m.mode = "TRACKING";
    m.valid = true;
    m.dropout = false;
    m.false_star_event = false;

    if(uniform() < cfg_.dropout_probability){
        m.valid = false;
        m.dropout = true;
        m.confidence = 0.0;
        m.tracked_stars = 0;
        m.q_bi = latest_.q_bi;
        m.omega_b_rad_s = truth.omega_b_rad_s;
        m.mode = "DROPOUT";
        return m;
    }

    std::uniform_int_distribution<uint32_t> stars_dist(cfg_.min_star_count, cfg_.max_star_count);
    m.tracked_stars = stars_dist(rng_);
    m.false_star_event = (uniform() < cfg_.false_star_probability);

    double sigma_att = cfg_.attitude_noise_arcsec_1sigma;
    if(m.false_star_event) sigma_att *= 3.0;
    if(m.tracked_stars < cfg_.min_star_count + 2) sigma_att *= 1.5;

    m.q_bi = noisy_attitude(truth.q_bi, sigma_att);
    m.omega_b_rad_s = {
        truth.omega_b_rad_s.x + gaussian(cfg_.gyro_noise_rad_s_1sigma),
        truth.omega_b_rad_s.y + gaussian(cfg_.gyro_noise_rad_s_1sigma),
        truth.omega_b_rad_s.z + gaussian(cfg_.gyro_noise_rad_s_1sigma)
    };

    double err_rad = quat_angular_error_rad(truth.q_bi, m.q_bi);
    double sigma_ref = arcsec_to_rad(std::max(1.0, sigma_att));
    double conf = std::exp(-0.5 * (err_rad*err_rad)/(sigma_ref*sigma_ref));
    if(m.false_star_event) conf *= 0.65;
    conf *= std::min(1.0, static_cast<double>(m.tracked_stars) / static_cast<double>(cfg_.max_star_count));
    m.confidence = std::clamp(conf, 0.0, 1.0);
    if(m.confidence < cfg_.valid_confidence_floor){
        m.valid = false;
        m.mode = "DEGRADED";
    }

    double var = sigma_att * sigma_att;
    double var_rad = arcsec_to_rad(std::sqrt(var));
    double cov = var_rad * var_rad;
    m.covariance = {cov,0,0, 0,cov,0, 0,0,cov};
    return m;
}

void StarTrackerSim::step(const TruthState& truth, double dt_s){
    t_s_ += dt_s;
    has_new_ = false;
    double period = 1.0 / std::max(1.0e-6, cfg_.update_hz);
    power_w_ = cfg_.nominal_power_w;
    if(t_s_ + 1e-12 >= next_update_s_){
        latest_ = synthesize_measurement(truth);
        has_new_ = true;
        next_update_s_ += period;
        power_w_ = latest_.valid ? cfg_.active_power_w : cfg_.degraded_power_w;
    }
}

bool StarTrackerSim::has_new_measurement() const { return has_new_; }
StarTrackerMeasurement StarTrackerSim::latest() const { return latest_; }
double StarTrackerSim::current_power_w() const { return power_w_; }
std::string StarTrackerSim::mode_string() const { return latest_.mode; }

} // namespace sgl
