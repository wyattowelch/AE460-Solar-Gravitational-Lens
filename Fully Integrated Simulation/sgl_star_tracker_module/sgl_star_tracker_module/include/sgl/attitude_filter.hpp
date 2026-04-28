#pragma once
#include "sgl/star_tracker_types.hpp"

namespace sgl {

class AttitudeFilter {
public:
    void reset(const Quaternion& q0 = Quaternion{});
    void set_blend_gain(double gain);
    FusedAttitudeState update(const GyroMeasurement& gyro,
                              const StarTrackerMeasurement* tracker,
                              double dt_s);

private:
    Quaternion q_est_{};
    Vec3 omega_est_{};
    double blend_gain_{0.08};
    double t_s_{0.0};
};

} // namespace sgl
