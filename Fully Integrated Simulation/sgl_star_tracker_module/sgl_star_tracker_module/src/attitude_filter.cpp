#include "sgl/attitude_filter.hpp"
#include "sgl/quaternion.hpp"
#include <algorithm>

namespace sgl {

void AttitudeFilter::reset(const Quaternion& q0){
    q_est_ = quat_normalized(q0);
    omega_est_ = {};
    t_s_ = 0.0;
}

void AttitudeFilter::set_blend_gain(double gain){ blend_gain_ = std::clamp(gain, 0.0, 1.0); }

FusedAttitudeState AttitudeFilter::update(const GyroMeasurement& gyro,
                                          const StarTrackerMeasurement* tracker,
                                          double dt_s){
    q_est_ = quat_integrate_body_rate(q_est_, gyro.omega_b_rad_s, dt_s);
    omega_est_ = gyro.omega_b_rad_s;
    if(tracker && tracker->valid){
        double gain = blend_gain_ * tracker->confidence;
        q_est_ = quat_slerp(q_est_, tracker->q_bi, gain);
    }
    t_s_ += dt_s;
    FusedAttitudeState out{};
    out.t_s = tracker ? tracker->t_s : t_s_;
    out.q_bi = q_est_;
    out.omega_b_rad_s = omega_est_;
    out.confidence = (tracker && tracker->valid) ? tracker->confidence : 0.2;
    out.tracker_valid = tracker ? tracker->valid : false;
    return out;
}

} // namespace sgl
