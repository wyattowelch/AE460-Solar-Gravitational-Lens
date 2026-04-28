#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include "sgl/quaternion.hpp"

namespace sgl {

struct TruthState {
    double t_s{0.0};
    Quaternion q_bi{};          // body -> inertial
    Vec3 omega_b_rad_s{};       // body rates
};

struct StarTrackerConfig {
    double update_hz{1.0};
    double fov_deg{12.0};
    double centroid_noise_arcsec_1sigma{1.5};
    double attitude_noise_arcsec_1sigma{6.0};
    double gyro_noise_rad_s_1sigma{1.0e-5};
    double gyro_bias_walk_rad_s_per_s{1.0e-7};
    double dropout_probability{0.005};
    double false_star_probability{0.01};
    uint32_t min_star_count{5};
    uint32_t max_star_count{25};
    double valid_confidence_floor{0.45};
    double nominal_power_w{10.0};
    double active_power_w{12.0};
    double degraded_power_w{9.0};
    double latency_s{0.05};
};

struct StarTrackerMeasurement {
    double t_s{0.0};
    Quaternion q_bi{};
    Vec3 omega_b_rad_s{};
    std::array<double, 9> covariance{}; // 3x3 flattened
    double confidence{1.0};
    uint32_t tracked_stars{0};
    bool valid{true};
    bool dropout{false};
    bool false_star_event{false};
    std::string mode{"TRACKING"};
};

struct GyroMeasurement {
    double t_s{0.0};
    Vec3 omega_b_rad_s{};
};

struct FusedAttitudeState {
    double t_s{0.0};
    Quaternion q_bi{};
    Vec3 omega_b_rad_s{};
    double confidence{1.0};
    bool tracker_valid{true};
};

struct ReactionWheelTelemetry {
    std::array<double, 4> wheel_speed_rad_s{};
    std::array<double, 4> wheel_torque_cmd_nm{};
    double power_w{0.0};
    bool saturated{false};
};

struct AdcsCommand {
    Quaternion desired_q_bi{};
    Vec3 desired_omega_b_rad_s{};
};

struct AdcsControllerConfig {
    double kp{0.04};
    double kd{0.30};
    double max_body_torque_nm{0.15};
};

} // namespace sgl
