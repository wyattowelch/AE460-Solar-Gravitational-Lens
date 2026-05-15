#pragma once
#include <string>
#include <vector>
#include "image_io.hpp"
#include "protocol.hpp"
namespace sgl {
struct TileStat { double r=0,g=0,b=0,l=0; };
struct SourcePreconditioningConfig {
  bool enabled = true;
  int canvas_N = 2048;
  std::string object_type = "auto";
  double disk_fill_fraction = 0.62;
  double extended_fill_fraction = 0.85;
  double object_padding_fraction = 0.10;
  double minimum_source_margin_fraction = 0.08;
  int background_value = 0;
  std::string brightness_normalization_mode = "auto";
  double brightness_target_luma = 0.48;
  double brightness_gain_min_disk = 0.85;
  double brightness_gain_max_disk = 1.10;
  double brightness_gain_min_extended = 0.90;
  double brightness_gain_max_extended = 1.05;
  double disk_photo_center_x = -1.0;
  double disk_photo_center_y = -1.0;
  double disk_photo_radius_px = -1.0;
  double disk_photo_crop_half_px = -1.0;
};
struct SourcePreconditioningResult {
  bool ok = false;
  std::string object_type_detected = "unknown";
  int bbox_x0 = 0, bbox_y0 = 0, bbox_x1 = 0, bbox_y1 = 0;
  double fill_fraction_used = 0.0;
  double source_margin_fraction_before = 0.0;
  double margin_fraction = 0.0;
  bool source_truncation_suspected = false;
  bool used_raw_fallback_for_preconditioning = false;
  bool clipping_guard_triggered = false;
  double detected_planet_center_x = 0.0;
  double detected_planet_center_y = 0.0;
  double detected_planet_radius_px = 0.0;
  double detected_planet_radius_x_px = 0.0;
  double detected_planet_radius_y_px = 0.0;
  double mask_coverage_fraction = 0.0;
  double detected_bbox_aspect = 0.0;
  double output_support_aspect = 0.0;
  double output_circularity_score = 0.0;
  double fit_center_x = 0.0;
  double fit_center_y = 0.0;
  double fit_radius_x_px = 0.0;
  double fit_radius_y_px = 0.0;
  double tone_gain_used = 1.0;
  std::string method = "none";
  std::string preconditioned_source_path;
  std::string source_mask_path;
  std::string source_texture_mask_path;
  std::string source_overlay_path;
};
struct SglObservationConfig {
  int ring_sensor_N = 4096;
  double ring_radius_fraction = 0.40;
  int ring_radial_width_px = 64;
  int ring_angular_samples = 8192;
  int ring_radial_samples = 96;
  int observation_count = 96;
  bool store_full_ring_frames = false;
  int store_ring_preview_every = 16;
  bool store_all_full_ring_frames_debug = false;
  std::string ring_processing_mode = "annulus_unwrapped";
};
struct SglObservationSummary {
  int ring_sensor_N = 0;
  int ring_radius_px = 0;
  int ring_radial_width_px = 0;
  long long active_annulus_pixels = 0;
  double active_pixel_fraction = 0.0;
  int ring_angular_samples = 0;
  int ring_radial_samples = 0;
  std::string ring_processing_mode = "annulus_unwrapped";
  std::string annulus_bin_path;
  std::string observations_csv_path;
  std::string ring_preview_path;
};
std::vector<TileStat> compute_tiles(const ImageRGBA& src,int tile_px_x,int tile_px_y,int& tx,int& ty);
ImageRGBA render_ring_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int N,double ring_radius,double ring_sigma);
ImageRGBA reconstruct_coarse_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int coarse_groups_x,int coarse_groups_y,unsigned outW,unsigned outH,std::vector<proto::RegionOfInterest>* rois=nullptr,int max_rois=0,const std::vector<proto::RegionOfInterest>* prior_rois=nullptr,int prior_roi_growth=0,double* roi_selection_ms=nullptr);
ImageRGBA refine_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int coarse_groups_x,int coarse_groups_y,const std::vector<proto::RegionOfInterest>& rois,unsigned outW,unsigned outH);
std::string tiles_to_csv(const std::vector<TileStat>& tiles,int tx,int ty);
bool tiles_from_csv(const std::string& csv,std::vector<TileStat>& tiles,int& tx,int& ty);
std::vector<uint8_t> ppm_bytes(const ImageRGBA& img);
bool generate_payload_dataset(const std::string& source_ppm,int tile_px_x,int tile_px_y,int ring_N,double ring_radius,double ring_sigma,const std::string& out_dir,std::string& dataset_csv,std::string& ring_preview_path,unsigned& srcW,unsigned& srcH,std::string& err);
bool generate_payload_dataset_from_ring_observation(const std::string& ring_observation_ppm,int tile_px_x,int tile_px_y,const std::string& out_dir,std::string& dataset_csv,std::string& ring_preview_path,unsigned& srcW,unsigned& srcH,std::string& err);
bool precondition_source_image(const std::string& input_ppm,const std::string& out_dir,const SourcePreconditioningConfig& cfg,SourcePreconditioningResult& out,std::string& err);
bool generate_sgl_observation_dataset(const std::string& preconditioned_source_ppm,const std::string& out_dir,const SglObservationConfig& cfg,SglObservationSummary& summary,std::string& dataset_descriptor,unsigned& srcW,unsigned& srcH,std::string& err);
bool is_sgl_dataset_descriptor(const std::string& descriptor);
}
