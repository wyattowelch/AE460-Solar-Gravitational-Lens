#include "sgl/adcs_controller.hpp"
#include <algorithm>

namespace sgl {

AdcsController::AdcsController(const AdcsControllerConfig& cfg): cfg_(cfg) {}
void AdcsController::set_config(const AdcsControllerConfig& cfg){ cfg_ = cfg; }

Vec3 AdcsController::compute_body_torque(const FusedAttitudeState& state,
                                         const AdcsCommand& command) const {
    Vec3 att_err = quat_error_body_axis(command.desired_q_bi, state.q_bi);
    Vec3 rate_err = command.desired_omega_b_rad_s - state.omega_b_rad_s;
    Vec3 torque{
        cfg_.kp * att_err.x + cfg_.kd * rate_err.x,
        cfg_.kp * att_err.y + cfg_.kd * rate_err.y,
        cfg_.kp * att_err.z + cfg_.kd * rate_err.z
    };
    auto clamp_one = [&](double v){ return std::clamp(v, -cfg_.max_body_torque_nm, cfg_.max_body_torque_nm); };
    torque.x = clamp_one(torque.x);
    torque.y = clamp_one(torque.y);
    torque.z = clamp_one(torque.z);
    return torque;
}

} // namespace sgl
