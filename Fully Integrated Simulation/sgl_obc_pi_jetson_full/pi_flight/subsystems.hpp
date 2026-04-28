#pragma once
#include <cstdint>
#include <deque>
#include <string>
#include <vector>
#include "sgl/comms_module.hpp"
#include "sgl/payload_module.hpp"
#include "sgl/propulsion_module.hpp"
#include "sgl/thermal_module.hpp"
#include "../common/reconstruction.hpp"
#include "sgl/adcs_controller.hpp"
#include "sgl/adcs_system.hpp"
#include "sgl/attitude_filter.hpp"
#include "sgl/gyro_sim.hpp"
#include "sgl/reaction_wheel_array.hpp"
#include "sgl/star_tracker_sim.hpp"
#include "../common/subsystem.hpp"
namespace sgl {
struct DatasetRecord { std::string dataset_id, csv, ring_preview_path; unsigned src_w=0, src_h=0; };
struct PayloadEvent { std::string type, severity, message, value; };
class ADCSSim : public ISubsystem {
 public:
  ADCSSim();
  void sense(double) override;
  void decide(double) override;
  void act(double) override;
  double current_power_w() const override { return power_w_; }
  std::string mode_string() const override;
  bool healthy() const override { return healthy_; }
  bool stable() const;
  double pointing_error_deg() const { return est_pointing_error_deg_; }
  double est_pointing_error_deg() const { return est_pointing_error_deg_; }
  double truth_pointing_error_deg() const { return truth_pointing_error_deg_; }
  double tracker_confidence() const { return tracker_confidence_; }
  bool tracker_valid() const { return tracker_valid_; }
  uint32_t tracked_stars() const { return tracked_stars_; }
  double wheel_power_w() const { return wheel_power_w_; }
 private:
  AdcsSystem adcs_{};
  AdcsCommand cmd_{};
  bool healthy_=true;
  bool correcting_=false;
  bool wheel_saturated_=false;
  double power_w_=0.0;
  double t_s_=0.0;
  double est_pointing_error_deg_=0.0;
  double truth_pointing_error_deg_=0.0;
  double tracker_confidence_=0.0;
  uint32_t tracked_stars_=0;
  bool tracker_valid_{false};
  double wheel_power_w_=0.0;
  Vec3 disturbance_omega_{};
};
class CommsSim : public ISubsystem { public: void sense(double) override; void decide(double) override; void act(double) override; double current_power_w() const override { return power_w_; } std::string mode_string() const override; bool healthy() const override { return healthy_; } void enqueue_bits(std::size_t bits); std::size_t backlog_bits() const { return backlog_bits_; } bool downlink_active() const { return tx_active_; } private: bool healthy_=true, tx_active_=false, window_open_=false; double power_w_=0.0, pending_dt_s_=1.0; std::size_t backlog_bits_=0, pending_enqueue_bits_=0; std::string mode_="STANDBY"; comms::CommsModel model_{}; };
class ThermalSim : public ISubsystem { public: void sense(double) override; void decide(double) override; void act(double) override; double current_power_w() const override { return power_w_; } std::string mode_string() const override; bool healthy() const override { return healthy_; } bool heater_active() const { return heater_on_; } double temperature_c() const { return temperature_c_; } private: bool healthy_=true, heater_on_=false; bool low_temp_warn_=false, high_temp_warn_=false; double temperature_c_=18.0, power_w_=0.0, heater_power_w_=0.0, pending_dt_s_=1.0; std::string mode_="PASSIVE"; thermal::ThermalModel model_{}; };
class PropulsionSim : public ISubsystem { public: void sense(double) override; void decide(double) override; void act(double) override; double current_power_w() const override { return power_w_; } std::string mode_string() const override; bool healthy() const override { return healthy_; } bool active() const { return active_; } double thrust_n() const { return thrust_n_; } bool burn_event() const { return burn_event_; } double remaining_propellant_kg() const { return remaining_propellant_kg_; } private: bool healthy_=true, active_=false, burn_event_=false; double power_w_=0.0, pending_dt_s_=1.0, thrust_n_=0.0, remaining_propellant_kg_=8.0; std::string mode_="IDLE"; propulsion::PropulsionModel model_{}; };
class PayloadSim : public ISubsystem { public: void configure(const std::string&,int,int,int,double,double,const std::string&,const std::string&,double); void sense(double) override; void decide(double) override; void act(double) override; double current_power_w() const override { return power_w_; } std::string mode_string() const override; bool healthy() const override { return healthy_; } bool has_dataset() const { return !datasets_.empty(); } DatasetRecord pop_dataset(); bool active() const { return active_; } bool dataset_ready() const { return dataset_ready_; } int dataset_count() const { return dataset_count_; } std::string last_dataset_id() const { return last_dataset_id_; } int acquisition_stage() const { return acquisition_stage_; } std::string camera_mode() const { return camera_mode_; } bool camera_frame_ready() const { return camera_frame_ready_; } bool alignment_valid() const { return alignment_valid_; } double alignment_score() const { return alignment_score_; } double blur_score() const { return blur_score_; } double brightness_mean() const { return brightness_mean_; } double contrast_score() const { return contrast_score_; } double last_ring_generation_ms() const { return last_ring_generation_ms_; } const std::string& raw_capture_path() const { return raw_capture_path_; } const std::string& rectified_image_path() const { return rectified_image_path_; } std::vector<PayloadEvent> drain_events(); private: bool healthy_=true, active_=false, dataset_ready_=false; double power_w_=0.0, t_s_=0.0, pending_dt_s_=1.0, synthetic_signal_score_=0.0; int dataset_count_=0, acquisition_stage_=0; std::string last_dataset_id_=""; std::string mode_="IDLE"; payload::PayloadModel model_{}; std::string source_ppm_; int tile_px_x_=64,tile_px_y_=64,ring_N_=512; double ring_radius_=0.38, ring_sigma_=0.04; std::string out_dir_="out/datasets"; std::string input_mode_="synthetic_image"; std::string camera_mode_="synthetic_image"; bool camera_frame_ready_=false; bool alignment_valid_=false; double alignment_score_=0.0, blur_score_=0.0, brightness_mean_=0.0, contrast_score_=0.0; double last_ring_generation_ms_=0.0; std::string raw_capture_path_="", rectified_image_path_=""; double fusion_alpha_=0.4; bool have_fused_=false; std::vector<TileStat> fused_tiles_; int fused_tx_=0,fused_ty_=0; std::deque<DatasetRecord> datasets_; std::vector<PayloadEvent> pending_events_; };
class SourcePowerModel { public: double update(double dt_s); private: double t_s_=0.0; double available_w_=120.0; };
class PowerPolicy { public: PowerPolicy(double reserve_w,double derate_fraction): reserve_w_(reserve_w), derate_fraction_(derate_fraction){} double compute_budget_w(double source_w,double noncompute_w) const; private: double reserve_w_=20.0, derate_fraction_=0.75; };
} // namespace sgl
