#pragma once
#include <string>
#include <vector>
#include "image_io.hpp"
#include "protocol.hpp"
namespace sgl {
struct TileStat { double r=0,g=0,b=0,l=0; };
std::vector<TileStat> compute_tiles(const ImageRGBA& src,int tile_px_x,int tile_px_y,int& tx,int& ty);
ImageRGBA render_ring_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int N,double ring_radius,double ring_sigma);
ImageRGBA reconstruct_coarse_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int coarse_groups_x,int coarse_groups_y,unsigned outW,unsigned outH,std::vector<proto::RegionOfInterest>* rois=nullptr,int max_rois=0,const std::vector<proto::RegionOfInterest>* prior_rois=nullptr,int prior_roi_growth=0,double* roi_selection_ms=nullptr);
ImageRGBA refine_from_tiles(const std::vector<TileStat>& tiles,int tx,int ty,int coarse_groups_x,int coarse_groups_y,const std::vector<proto::RegionOfInterest>& rois,unsigned outW,unsigned outH);
std::string tiles_to_csv(const std::vector<TileStat>& tiles,int tx,int ty);
bool tiles_from_csv(const std::string& csv,std::vector<TileStat>& tiles,int& tx,int& ty);
std::vector<uint8_t> ppm_bytes(const ImageRGBA& img);
bool generate_payload_dataset(const std::string& source_ppm,int tile_px_x,int tile_px_y,int ring_N,double ring_radius,double ring_sigma,const std::string& out_dir,std::string& dataset_csv,std::string& ring_preview_path,unsigned& srcW,unsigned& srcH,std::string& err);
bool generate_payload_dataset_from_ring_observation(const std::string& ring_observation_ppm,int tile_px_x,int tile_px_y,const std::string& out_dir,std::string& dataset_csv,std::string& ring_preview_path,unsigned& srcW,unsigned& srcH,std::string& err);
}
