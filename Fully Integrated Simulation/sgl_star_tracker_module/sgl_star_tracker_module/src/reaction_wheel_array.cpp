#include "sgl/reaction_wheel_array.hpp"
#include <algorithm>
#include <cmath>

namespace sgl {

ReactionWheelArray::ReactionWheelArray(){
    double c = 1.0 / std::sqrt(3.0);
    wheel_axes_ = {Vec3{ c, c, c}, Vec3{-c,-c, c}, Vec3{-c, c,-c}, Vec3{ c,-c,-c}};
    reset();
}

void ReactionWheelArray::reset(){
    speed_rad_s_.fill(0.0);
    torque_cmd_nm_.fill(0.0);
    power_w_ = 2.0;
    saturated_ = false;
}

ReactionWheelTelemetry ReactionWheelArray::step(const Vec3& body_torque_cmd_nm, double dt_s){
    constexpr double max_wheel_torque_nm = 0.06;
    constexpr double max_speed_rad_s = 650.0;
    constexpr double wheel_inertia = 2.0e-4;
    saturated_ = false;

    for(size_t i=0;i<4;i++){
        double cmd = 0.5 * dot(wheel_axes_[i], body_torque_cmd_nm);
        cmd = std::clamp(cmd, -max_wheel_torque_nm, max_wheel_torque_nm);
        torque_cmd_nm_[i] = cmd;
        speed_rad_s_[i] += (cmd / wheel_inertia) * dt_s;
        if(std::abs(speed_rad_s_[i]) > max_speed_rad_s){
            speed_rad_s_[i] = std::clamp(speed_rad_s_[i], -max_speed_rad_s, max_speed_rad_s);
            saturated_ = true;
        }
    }

    double torque_sum = 0.0;
    for(double x : torque_cmd_nm_) torque_sum += std::abs(x);
    power_w_ = 4.0 + 35.0 * torque_sum + 0.002 * (std::abs(speed_rad_s_[0]) + std::abs(speed_rad_s_[1]) + std::abs(speed_rad_s_[2]) + std::abs(speed_rad_s_[3]));

    ReactionWheelTelemetry out{};
    out.wheel_speed_rad_s = speed_rad_s_;
    out.wheel_torque_cmd_nm = torque_cmd_nm_;
    out.power_w = power_w_;
    out.saturated = saturated_;
    return out;
}

double ReactionWheelArray::current_power_w() const { return power_w_; }

} // namespace sgl
