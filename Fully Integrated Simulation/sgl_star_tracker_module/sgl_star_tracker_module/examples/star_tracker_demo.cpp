#include <fstream>
#include <iomanip>
#include <iostream>
#include "sgl/star_tracker_sim.hpp"
#include "sgl/gyro_sim.hpp"
#include "sgl/attitude_filter.hpp"
#include "sgl/adcs_controller.hpp"
#include "sgl/reaction_wheel_array.hpp"
#include "sgl/quaternion.hpp"

using namespace sgl;

int main(){
    StarTrackerConfig cfg{};
    cfg.update_hz = 2.0;
    cfg.nominal_power_w = 10.0;
    cfg.active_power_w = 12.0;
    cfg.degraded_power_w = 9.0;

    StarTrackerSim tracker(cfg);
    GyroSim gyro(cfg);
    AttitudeFilter filter;
    filter.reset();
    AdcsController controller({0.08, 0.45, 0.12});
    ReactionWheelArray wheels;

    AdcsCommand cmd{};
    cmd.desired_q_bi = Quaternion{};
    cmd.desired_omega_b_rad_s = {0.0,0.0,0.0};

    TruthState truth{};
    truth.q_bi = quat_from_axis_angle({0,1,0}, 0.02);
    truth.omega_b_rad_s = {0.0, 0.0005, 0.0};

    std::ofstream csv("star_tracker_demo.csv");
    csv << "t,truth_err_deg,tracker_valid,tracker_conf,tracker_stars,tracker_power_w,wheel_power_w,total_adcs_power_w\n";

    double dt = 0.05;
    double t_end = 60.0;
    StarTrackerMeasurement last_tracker{};
    bool have_tracker = false;

    for(double t=0.0; t<=t_end; t+=dt){
        truth.t_s = t;
        if(t > 8.0 && t < 20.0) truth.omega_b_rad_s = {0.0003, 0.0008, -0.0002};
        else if(t > 35.0) truth.omega_b_rad_s = {-0.0002, 0.0002, 0.0006};
        else truth.omega_b_rad_s = {0.0,0.0005,0.0};
        truth.q_bi = quat_integrate_body_rate(truth.q_bi, truth.omega_b_rad_s, dt);

        tracker.step(truth, dt);
        if(tracker.has_new_measurement()){
            last_tracker = tracker.latest();
            have_tracker = true;
        }

        GyroMeasurement gm = gyro.measure(truth, dt);
        const StarTrackerMeasurement* tm = have_tracker ? &last_tracker : nullptr;
        auto fused = filter.update(gm, tm, dt);
        Vec3 torque_cmd = controller.compute_body_torque(fused, cmd);
        auto wheel_tm = wheels.step(torque_cmd, dt);

        double truth_err_deg = quat_angular_error_rad(cmd.desired_q_bi, truth.q_bi) * 180.0 / M_PI;
        double total_adcs_power = tracker.current_power_w() + wheel_tm.power_w;
        csv << std::fixed << std::setprecision(6)
            << t << ','
            << truth_err_deg << ','
            << (have_tracker && last_tracker.valid ? 1 : 0) << ','
            << (have_tracker ? last_tracker.confidence : 0.0) << ','
            << (have_tracker ? last_tracker.tracked_stars : 0) << ','
            << tracker.current_power_w() << ','
            << wheel_tm.power_w << ','
            << total_adcs_power << '\n';
    }

    std::cout << "Wrote star_tracker_demo.csv\n";
    return 0;
}
