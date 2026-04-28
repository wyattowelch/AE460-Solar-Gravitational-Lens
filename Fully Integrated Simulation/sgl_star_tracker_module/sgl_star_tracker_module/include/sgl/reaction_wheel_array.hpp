#pragma once
#include "sgl/star_tracker_types.hpp"

namespace sgl {

class ReactionWheelArray {
public:
    ReactionWheelArray();
    void reset();
    ReactionWheelTelemetry step(const Vec3& body_torque_cmd_nm, double dt_s);
    double current_power_w() const;

private:
    std::array<Vec3, 4> wheel_axes_{};
    std::array<double, 4> speed_rad_s_{};
    std::array<double, 4> torque_cmd_nm_{};
    double power_w_{0.0};
    bool saturated_{false};
};

} // namespace sgl
