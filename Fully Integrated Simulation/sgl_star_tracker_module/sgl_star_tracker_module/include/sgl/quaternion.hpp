#pragma once
#include <array>
#include <cmath>
#include <string>

namespace sgl {

struct Vec3 {
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct Quaternion {
    double w{1.0};
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

Vec3 operator+(const Vec3& a, const Vec3& b);
Vec3 operator-(const Vec3& a, const Vec3& b);
Vec3 operator*(double s, const Vec3& v);
Vec3 operator*(const Vec3& v, double s);
Vec3 operator/(const Vec3& v, double s);
double dot(const Vec3& a, const Vec3& b);
Vec3 cross(const Vec3& a, const Vec3& b);
double norm(const Vec3& v);
Vec3 normalized(const Vec3& v);

Quaternion quat_normalized(const Quaternion& q);
Quaternion quat_conjugate(const Quaternion& q);
Quaternion quat_multiply(const Quaternion& a, const Quaternion& b);
Vec3 quat_rotate(const Quaternion& q, const Vec3& v);
Quaternion quat_from_axis_angle(const Vec3& axis, double angle_rad);
Quaternion quat_from_omega_dt(const Vec3& omega_rad_s, double dt_s);
Quaternion quat_slerp(const Quaternion& a, const Quaternion& b, double t);
Quaternion quat_integrate_body_rate(const Quaternion& q_bi, const Vec3& omega_body_rad_s, double dt_s);
double quat_angular_error_rad(const Quaternion& q_ref, const Quaternion& q_meas);
Vec3 quat_error_body_axis(const Quaternion& q_ref, const Quaternion& q_body_to_inertial);
std::string to_string(const Quaternion& q);

} // namespace sgl
