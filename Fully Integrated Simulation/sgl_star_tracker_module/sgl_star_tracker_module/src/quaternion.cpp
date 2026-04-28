#include "sgl/quaternion.hpp"
#include <algorithm>
#include <sstream>

namespace sgl {

Vec3 operator+(const Vec3& a, const Vec3& b){ return {a.x+b.x,a.y+b.y,a.z+b.z}; }
Vec3 operator-(const Vec3& a, const Vec3& b){ return {a.x-b.x,a.y-b.y,a.z-b.z}; }
Vec3 operator*(double s, const Vec3& v){ return {s*v.x,s*v.y,s*v.z}; }
Vec3 operator*(const Vec3& v, double s){ return s*v; }
Vec3 operator/(const Vec3& v, double s){ return {v.x/s,v.y/s,v.z/s}; }
double dot(const Vec3& a, const Vec3& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }
Vec3 cross(const Vec3& a, const Vec3& b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
double norm(const Vec3& v){ return std::sqrt(dot(v,v)); }
Vec3 normalized(const Vec3& v){ double n=norm(v); return (n>1e-15)? v/n : Vec3{}; }

Quaternion quat_normalized(const Quaternion& q){
    double n = std::sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if(n <= 1e-15) return {};
    return {q.w/n,q.x/n,q.y/n,q.z/n};
}
Quaternion quat_conjugate(const Quaternion& q){ return {q.w,-q.x,-q.y,-q.z}; }
Quaternion quat_multiply(const Quaternion& a, const Quaternion& b){
    return {
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}
Vec3 quat_rotate(const Quaternion& q_in, const Vec3& v){
    Quaternion q = quat_normalized(q_in);
    Quaternion p{0.0,v.x,v.y,v.z};
    Quaternion r = quat_multiply(quat_multiply(q,p), quat_conjugate(q));
    return {r.x,r.y,r.z};
}
Quaternion quat_from_axis_angle(const Vec3& axis, double angle_rad){
    Vec3 a = normalized(axis);
    double s = std::sin(0.5*angle_rad);
    return quat_normalized({std::cos(0.5*angle_rad), a.x*s, a.y*s, a.z*s});
}
Quaternion quat_from_omega_dt(const Vec3& omega_rad_s, double dt_s){
    double mag = norm(omega_rad_s);
    if(mag < 1e-15) return {};
    return quat_from_axis_angle(omega_rad_s/mag, mag*dt_s);
}
Quaternion quat_slerp(const Quaternion& a_in, const Quaternion& b_in, double t){
    Quaternion a=quat_normalized(a_in), b=quat_normalized(b_in);
    double cosTheta = a.w*b.w + a.x*b.x + a.y*b.y + a.z*b.z;
    if(cosTheta < 0.0){ b={-b.w,-b.x,-b.y,-b.z}; cosTheta=-cosTheta; }
    if(cosTheta > 0.9995){
        Quaternion out{a.w+t*(b.w-a.w), a.x+t*(b.x-a.x), a.y+t*(b.y-a.y), a.z+t*(b.z-a.z)};
        return quat_normalized(out);
    }
    double theta = std::acos(std::clamp(cosTheta,-1.0,1.0));
    double s1 = std::sin((1.0-t)*theta)/std::sin(theta);
    double s2 = std::sin(t*theta)/std::sin(theta);
    return quat_normalized({s1*a.w+s2*b.w, s1*a.x+s2*b.x, s1*a.y+s2*b.y, s1*a.z+s2*b.z});
}
Quaternion quat_integrate_body_rate(const Quaternion& q_bi, const Vec3& omega_body_rad_s, double dt_s){
    Quaternion dq = quat_from_omega_dt(omega_body_rad_s, dt_s);
    return quat_normalized(quat_multiply(q_bi, dq));
}
double quat_angular_error_rad(const Quaternion& q_ref, const Quaternion& q_meas){
    Quaternion dq = quat_multiply(quat_conjugate(quat_normalized(q_ref)), quat_normalized(q_meas));
    dq = quat_normalized(dq);
    double ang = 2.0*std::acos(std::clamp(std::abs(dq.w),0.0,1.0));
    return ang;
}
Vec3 quat_error_body_axis(const Quaternion& q_ref, const Quaternion& q_body_to_inertial){
    Quaternion dq = quat_multiply(quat_conjugate(quat_normalized(q_body_to_inertial)), quat_normalized(q_ref));
    dq = quat_normalized(dq);
    double sign = (dq.w >= 0.0) ? 1.0 : -1.0;
    return {2.0*sign*dq.x, 2.0*sign*dq.y, 2.0*sign*dq.z};
}
std::string to_string(const Quaternion& q){
    std::ostringstream oss;
    oss << "["<<q.w<<", "<<q.x<<", "<<q.y<<", "<<q.z<<"]";
    return oss.str();
}

} // namespace sgl
